import os
import time
import argparse
import csv
import json
import subprocess
import psutil
import numpy as np
import faiss
import utils
import multiprocessing

# ==========================================
# SCRIPT DESCRIPTION
# ==========================================
# This script: 
# 1. Loads pre-computed query embeddings and ground truth indices.
# 2. Connects to each vector DB backend (FAISS, Chroma, Milvus, Pinecone).
# 3. Executes ANN searches with varying 'ef' parameters.
# 4. Measures latency, memory usage, and calculates Recall@K.
# 5. Exports results to CSV and detailed JSON comparison files.
# ==========================================


# --- Configuration ---
# Paths are derived from the utils.BASE constant.
# GT_DIR: Contains .npy files with the "correct" nearest neighbors (calculated via brute force).
# RES_DIR: Destination for benchmark CSVs and detailed JSON exports.
GT_DIR = os.path.join(utils.BASE, "ground_truth")
RES_DIR = os.path.join(utils.BASE, "results_benchmark")
INDICES_DIR = os.path.join(utils.BASE, "indices")
os.makedirs(RES_DIR, exist_ok=True)

# The 'ef' parameter (Expansion Factor) controls the trade-off between search speed and accuracy 
# in HNSW indices. 
# Higher EF = deeper search queue = higher recall but higher latency.
EF_VALUES = [100, 200, 300]

# --- RAM Monitor ---
class MemoryMonitor:
    """
    Monitors RAM usage. Handles the distinction between:
    1. In-process libraries (FAISS, Chroma): Uses psutil to check current process RSS.
    2. Containerized services (Milvus): Uses `docker stats` to check the separate container.
    """
    def __init__(self, backend):
        self.backend = backend.lower()
        self.process = psutil.Process(os.getpid())
        self.base_mem = 0
        
    def start(self):
        """Captures the baseline memory usage before the heavy operation starts."""
        if self.backend == "milvus": self.base_mem = self._get_docker_mem()
        else: self.base_mem = self.process.memory_info().rss / (1024 * 1024)

    def measure(self):
        """Returns the memory increase (Delta) in MB since start() was called."""
        if self.backend == "milvus": return self._get_docker_mem()
        elif self.backend == "pinecone": return 0.0
        else:
            current = self.process.memory_info().rss / (1024 * 1024)
            return max(0, current - self.base_mem)

    def _get_docker_mem(self):
        """Parses `docker stats` output to find memory usage of 'milvus-standalone'."""
        try:
            cmd = ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", "milvus-standalone"]
            out = subprocess.check_output(cmd).decode("utf-8").strip()
            val_str = out.split("/")[0].strip()
            if "GiB" in val_str: return float(val_str.replace("GiB", "")) * 1024
            elif "MiB" in val_str: return float(val_str.replace("MiB", ""))
            elif "KiB" in val_str: return float(val_str.replace("KiB", "")) / 1024
            elif "B" in val_str: return float(val_str.replace("B", "")) / (1024*1024)
            return 0.0
        except Exception: return 0.0

# --- Helpers ---
def load_ground_truth(dataset):
    """
    Loads computed ground truth data for evaluation.
    
    Returns:
        queries (np.array): The vector embeddings used as search queries.
        gt_indices (np.array): A 2D array where each row contains the True Nearest Neighbor IDs 
                               (sorted by distance) for the corresponding query.
    """
    name_map = {"ml20m_movie": "ml20m", "fiqa_corpus": "fiqa"}
    prefix = name_map.get(dataset, dataset)
    return np.load(os.path.join(GT_DIR, f"{prefix}_queries.npy")), \
           np.load(os.path.join(GT_DIR, f"{prefix}_gt_indices.npy"))

def calc_recall(results, ground_truth, k):
    """
    Calculates Recall@K.
    Formula: Intersection(Retrieved_Set_at_K, True_Set_at_K) / K
    
    Args:
        results (list of lists): The IDs returned by the Vector DB.
        ground_truth (numpy array): The true IDs from the brute-force search.
        k (int): The cut-off rank (e.g., top 10, top 100).
    """
    hits = 0
    total = len(results) * k 
    for i in range(len(results)):
        hits += len(set(results[i][:k]).intersection(set(ground_truth[i][:k])))
    return hits / total

def export_detailed_comparison(dataset, backend, ef_val, results, ground_truth, k, raw_ids, titles):
    """
    Exports a qualitative JSON report. Useful for the thesis to show specific examples 
    of what the DB returned vs what it should have returned (e.g., Movie Titles).
    """
    export_data = []
    # Only export first 5 queries to keep file size manageable
    for i in range(min(5, len(results))):
        gt_row = ground_truth[i]
        ann_row = results[i]
        
        gt_set = set(gt_row[:k])
        hits = 0
        ann_items = []
        for idx in ann_row:
            if idx == -1: 
                ann_items.append("Error")
                continue
            # Map the integer index back to the real dataset ID (e.g., IMDB ID)
            real_id = str(raw_ids[idx]) if raw_ids is not None else str(idx)
            # Fetch the human-readable title
            ann_items.append(titles.get(real_id, f"ID:{real_id}"))
            if idx in gt_set: hits += 1

        gt_items = [titles.get(str(raw_ids[idx]) if raw_ids is not None else str(idx), f"ID:{idx}") for idx in gt_row[:k]]
        
        export_data.append({
            "metrics": {"recall": hits/k, "ef": ef_val}, 
            "ground_truth": gt_items, 
            "ann_results": ann_items
        })
    
    filename = f"comp_{dataset}_{backend}_ef{ef_val}.json"
    with open(os.path.join(RES_DIR, filename), "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=4, ensure_ascii=False)

# --- Workers ---
def _chroma_worker(collection_name, queries, k, ef, id_map, queue):
    try:
        import chromadb
        from chromadb.config import Settings
        
        persist_dir = os.path.join(INDICES_DIR, "chroma", "db_files")
        monitor = MemoryMonitor("chroma")
        monitor.start()
        
        client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
        coll = client.get_collection(collection_name)
        coll.peek() 
        mem_usage = monitor.measure()
        
        BATCH = 256 # Batch processing to prevent memory overload
        all_ids = []
        t0 = time.time()
        for start in range(0, len(queries), BATCH):
            batch = queries[start : start + BATCH].tolist()
            res = coll.query(query_embeddings=batch, n_results=k)

            # Chroma returns IDs as strings (often "doc:ID"). 
            # We map these back to the integer indices used in Ground Truth.
            for id_list in res["ids"]:
                row = [id_map.get(str(doc_id).split(":")[-1], -1) for doc_id in id_list]
                all_ids.append(row)
        lat = (time.time() - t0) / len(queries) * 1000.0
        queue.put({"latency_ms": lat, "indices": all_ids, "memory_mb": mem_usage})
    except Exception as e:
        queue.put({"error": str(e)})

def run_chroma(base_name, queries, k, ef_values, id_map):
    """Orchestrates ChromaDB benchmarking across different EF configurations."""
    results_list = []
    for ef in ef_values:
        # Note: Chroma HNSW params are set at index creation. 
        # This script assumes you have pre-created collections named 'dataset_ef100', etc.
        target_collection = f"{base_name}_ef{ef}"
        print(f"[Chroma] Loading Collection: {target_collection}...")
        
        queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=_chroma_worker, args=(target_collection, queries, k, ef, id_map, queue))
        p.start()
        res = queue.get()
        p.join()
        
        if "error" in res:
            print(f"Error: {res['error']}")
        else:
            results_list.append({"ef": ef, "latency_ms": res["latency_ms"], "indices": res["indices"], "memory_mb": res["memory_mb"]})
    return results_list

def run_faiss(name, queries, k, ef_values, monitor):
    """
    Benchmarks FAISS.
    FAISS allows dynamic modification of HNSW search parameters (efSearch) at runtime without reloading.
    """
    index_path = os.path.join(INDICES_DIR, "faiss", f"{name}.index")
    monitor.start()
    index = faiss.read_index(index_path)
    mem_usage = monitor.measure()
    res = []
    for ef in ef_values:
        print(f"[FAISS] Setting efSearch={ef}...")
        index.hnsw.efSearch = ef
        t0 = time.time()
        D, I = index.search(queries, k)
        lat = (time.time() - t0)/len(queries)*1000
        res.append({"ef": ef, "latency_ms": lat, "indices": I, "memory_mb": mem_usage})
    return res

def run_milvus(name, queries, k, ef_values, id_map, monitor):
    """Benchmarks Milvus (Dockerized)."""
    from pymilvus import connections, Collection
    import dotenv
    dotenv.load_dotenv(os.path.join(utils.BASE, ".env"))
    host = os.getenv("MILVUS_HOST", "localhost")
    port = os.getenv("MILVUS_PORT", "19530")

    connections.connect("default", host=host, port=port)
    coll = Collection(name); monitor.start(); coll.load(); mem = monitor.measure()
    res = []
    for ef in ef_values:
        print(f"[Milvus] Setting search params ef={ef}...")
        t0 = time.time()
        all_ids = []
        for i in range(0, len(queries), 1000):
            s = coll.search(queries[i:i+1000], "vec", {"metric_type":"COSINE", "params":{"ef": ef}}, k, output_fields=["id"])
            for h in s: all_ids.append([id_map.get(str(x.id).split(":")[-1], -1) for x in h])
        lat = (time.time()-t0)/len(queries)*1000
        res.append({"ef": ef, "latency_ms": lat, "indices": all_ids, "memory_mb": mem})
    coll.release()
    return res

def run_pinecone(dataset, queries, k, id_map):
    """
    Benchmarks Pinecone (Serverless).
    Note: Pinecone does not expose 'ef' tuning to the user. It is auto-managed.
    Therefore, we only return one result row.
    """
    from pinecone import Pinecone
    from dotenv import load_dotenv
    load_dotenv(os.path.join(utils.BASE, ".env"))
    
    # 1. Select Host based on Dataset
    if "ml20m" in dataset:
        host = os.getenv("PINECONE_HOST_MOVIELENS")
        print(f"[Pinecone] Connecting to MovieLens Host...")
    else:
        host = os.getenv("PINECONE_HOST_FIQA")
        print(f"[Pinecone] Connecting to FiQA Host...")

    if not host:
        print("Error: Pinecone Host not found in .env")
        return []

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    idx = pc.Index(host=host)
    
    t0 = time.time()
    all_ids = []
    for i, q in enumerate(queries):
        m = idx.query(vector=q.tolist(), top_k=k)['matches']
        all_ids.append([id_map.get(x['id'].split(":")[-1], -1) for x in m])
        if i % 200 == 0 and i > 0: print(f"   Processed {i} queries...")
        
    lat = (time.time()-t0)/len(queries)*1000
    
    # Pinecone doesn't have variable 'ef', so we return one row
    return [{"ef": "Serverless", "latency_ms": lat, "indices": all_ids, "memory_mb": 0.0}]

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()
    
    queries, gt = load_ground_truth(args.dataset)
    _, raw_ids, id_map = utils.load_embeddings_and_ids(args.dataset, "mini")
    titles = utils.load_movie_titles() if args.export and "ml20m" in args.dataset else {}
    monitor = MemoryMonitor(args.backend)
    
    base = f"{args.dataset}_{args.backend}"
    
    if args.backend == "chroma":
        results = run_chroma(base, queries, 100, EF_VALUES, id_map)
    elif args.backend == "faiss":
        results = run_faiss(base, queries, 100, EF_VALUES, monitor)
    elif args.backend == "milvus":
        results = run_milvus(base, queries, 100, EF_VALUES, id_map, monitor)
    elif args.backend == "pinecone":
        results = run_pinecone(args.dataset, queries, 100, id_map)

    # Save CSV
    csv_name = f"benchmark_{args.dataset}_{args.backend}.csv"
    with open(os.path.join(RES_DIR, csv_name), "w") as f:
        w = csv.writer(f)
        w.writerow(["Config/ef", "Latency_ms", "Memory_MB", "Recall@10", "Recall@50", "Recall@100"])
        
        print("\nConfig/ef | Latency | Mem | Rec@10 | Rec@50 | Rec@100\n" + "-"*60)
        
        for r in results:
            rec10 = calc_recall(r['indices'], gt, 10)
            rec50 = calc_recall(r['indices'], gt, 50) 
            rec100 = calc_recall(r['indices'], gt, 100)
            
            w.writerow([r['ef'], round(r['latency_ms'],2), round(r['memory_mb'],1), round(rec10,4), round(rec50,4), round(rec100,4)])
            
            print(f"{r['ef']:<9} | {r['latency_ms']:.2f}    | {r['memory_mb']:.0f}  | {rec10:.4f} | {rec50:.4f} | {rec100:.4f}")
            
            if args.export:
                export_detailed_comparison(args.dataset, args.backend, r['ef'], r['indices'], gt, 100, raw_ids, titles)
                
    print(f"\nSaved results to {os.path.join(RES_DIR, csv_name)}")