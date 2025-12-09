import os
import argparse
import time
import numpy as np
import utils

# ==========================================
# SCRIPT DESCRIPTION
# ==========================================
# This script: 

# 1. Loads pre-computed embeddings (numpy arrays) and IDs.
# 2. Normalizes HNSW parameters (M, efConstruction) across backends for fair comparison.
# 3. Dispatches data to the specific backend (FAISS, Chroma, Milvus, Pinecone).
# ==========================================

# --- Configuration ---
# Standard HNSW parameters for fair comparison
# M: The number of bi-directional links created for every new element during construction.
# efConstruction: The size of the dynamic list for the nearest neighbors candidate list during construction.
# NOTE: Keeping these constant across DBs is crucial for a fair "apples-to-apples" benchmark.
DEFAULT_EF_CONSTRUCTION = 128 
DEFAULT_M = 16

# Target search ef values to create variants for Chroma which does not allow dynamic ef setting
# Specific configuration for Chroma. Since Chroma (and the underlying HNSW lib it uses) 
# binds the search configuration at the time of Index creation in this specific setup,
# we cannot dynamically change 'ef_search' during the query phase easily.
# SOLUTION: We create 3 separate collections, pre-configured for these specific search depths.
TARGET_SEARCH_EFS = [100, 200, 300]

def make_ids(prefix_ids, tag):
    """
    Formats raw integer IDs into unique string IDs.
    Example: 123 -> "doc:123" or "movie:123"
    This ensures ID uniqueness if multiple datasets are merged later.
    """
    return [f"{tag}:{i}" for i in prefix_ids]

def index_faiss(vecs, out_path):
    """
    Builds a local FAISS index and saves it to disk (.index file).
    Type: HNSWFlat (HNSW index with full vectors stored in the bottom layer).
    """
    import faiss
    d = vecs.shape[1]
    index = faiss.IndexHNSWFlat(d, DEFAULT_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = DEFAULT_EF_CONSTRUCTION
    
    print(f"[FAISS] Building HNSW (d={d}, M={DEFAULT_M}, efConstruction={DEFAULT_EF_CONSTRUCTION})...")
    t0 = time.time()
    index.add(vecs.astype(np.float32))
    dt = (time.time() - t0)
    
    faiss.write_index(index, out_path)
    print(f"[FAISS] Built and saved to {out_path} in {dt:.2f} sec")

def index_chroma_variants(base_name, ids, vecs):
    """
    LOGIC: Creates multiple Chroma collections for the SAME datat 
    to compare Recall vs Speed at different 'ef_search' levels (100, 200, 300).
    
    Chroma's HNSW implementation often locks the search configuration in the metadata
    upon creation. To benchmark effectively, we pre-build indices for every target configuration.
    """
    from chromadb.config import Settings
    from chromadb import PersistentClient
    
    persist_dir = os.path.join(utils.INDICES_DIR, "chroma", "db_files") 
    os.makedirs(persist_dir, exist_ok=True)
    client = PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
    
    BATCH = 5000 # Upsert batch size to avoid memory overflows
    n = len(ids)

    print(f"\n>>> Generating {len(TARGET_SEARCH_EFS)} Chroma variants: {TARGET_SEARCH_EFS}")

    for ef in TARGET_SEARCH_EFS:
        # Name e.g. ml20m_movie_chroma_ef200
        col_name = f"{base_name}_ef{ef}"
        
        try: client.delete_collection(col_name)
        except Exception: pass

        # Critical: Inject HNSW params into metadata. 
        # Chroma reads "hnsw:*" keys to configure the underlying index.
        metadata = {
            "hnsw:space": "cosine",
            "hnsw:construction_ef": DEFAULT_EF_CONSTRUCTION,
            "hnsw:M": DEFAULT_M,
            "hnsw:search_ef": ef
        }
        
        print(f"   [Chroma] Building '{col_name}'...")
        coll = client.create_collection(name=col_name, metadata=metadata)
        
        t0 = time.time()
        for start in range(0, n, BATCH):
            end = min(start + BATCH, n)
            coll.upsert(
                ids=[str(x) for x in ids[start:end]], 
                embeddings=vecs[start:end].tolist()
            )
        print(f"   -> Done in {(time.time() - t0):.2f} sec")

def index_milvus(name, ids, vecs):
    """
    Connects to a running Milvus instance (Docker), defines schema, inserts data, 
    and builds the index explicitly.
    """
    from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
    host = os.getenv("MILVUS_HOST", "127.0.0.1")
    port = os.getenv("MILVUS_PORT", "19530")
    connections.connect("default", host=host, port=port)
    
    if utility.has_collection(name): utility.drop_collection(name)

    # 1. Define Schema
    dim = vecs.shape[1]
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=128, is_primary=True, auto_id=False),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    coll = Collection(name, CollectionSchema(fields, description=f"{name} collection"))

    # 2. Insert Data
    B = 2048
    t0 = time.time()
    for i in range(0, len(ids), B):
        coll.insert([ids[i:i+B], vecs[i:i+B].astype(np.float32)])
    coll.flush()
    
    # 3. Build Index
    index_params = {
        "index_type": "HNSW", 
        "metric_type": "COSINE", 
        "params": {"M": DEFAULT_M, "efConstruction": DEFAULT_EF_CONSTRUCTION}
    }
    t1 = time.time()
    coll.create_index(field_name="vec", index_params=index_params)
    coll.load() 
    print(f"[Milvus] Ready. Insert: {(time.time()-t0):.2f}s, Index: {(time.time()-t1):.2f}s")

def index_pinecone(name, ids, vecs):
    """
    Uploads data to Pinecone Serverless (Cloud).
    Requires PINECONE_API_KEY in .env file.
    """
    from pinecone import Pinecone, ServerlessSpec
    from dotenv import load_dotenv
    load_dotenv(os.path.join(utils.BASE, ".env"))
    
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # Pinecone names must be lower-case and alphanumeric/hyphens
    safe_name = name.replace("_", "-").lower()
    
    if safe_name in [i.name for i in pc.list_indexes()]: 
        pc.delete_index(safe_name); time.sleep(5) 

    d = vecs.shape[1]
    print(f"[Pinecone] Creating index '{safe_name}'...")
    pc.create_index(name=safe_name, dimension=d, metric="cosine", 
                    spec=ServerlessSpec(cloud="aws", region=os.getenv("PINECONE_REGION", "us-east-1")))
    
    while not pc.describe_index(safe_name).status.ready: time.sleep(1)
    index = pc.Index(safe_name)
    
    BATCH = 1000 # Upsert in small batches (Pinecone has a request size limit)
    n = len(ids)
    for i in range(0, n, BATCH):
        batch = [{"id": ids[j], "values": vecs[j].tolist()} for j in range(i, min(i+BATCH, n))]
        index.upsert(vectors=batch)
    print("[Pinecone] Ready.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["fiqa_corpus", "ml20m_movie"])
    ap.add_argument("--backend", required=True, choices=["faiss", "chroma", "milvus", "pinecone"])
    ap.add_argument("--model", default="mini")
    ap.add_argument("--name", default=None)
    
    args = ap.parse_args()

    vecs, raw_ids, _ = utils.load_embeddings_and_ids(args.dataset, args.model)
    tag = "doc" if "fiqa" in args.dataset else "movie"
    full_ids = make_ids(raw_ids, tag)

    if args.name:
        name = args.name
    else:
        name = f"{args.dataset}_{args.model}_{args.backend}" if args.model != "mini" else f"{args.dataset}_{args.backend}"
        
    print(f"--- Indexing {name} ---")

    if args.backend == "faiss":
        os.makedirs(os.path.join(utils.INDICES_DIR, "faiss"), exist_ok=True)
        out = os.path.join(utils.INDICES_DIR, "faiss", f"{name}.index")
        index_faiss(vecs, out)
    elif args.backend == "chroma":
        index_chroma_variants(name, full_ids, vecs) # 3 variants of the chroma index
    elif args.backend == "milvus":
        index_milvus(name, full_ids, vecs)
    elif args.backend == "pinecone":
        index_pinecone(name, full_ids, vecs)