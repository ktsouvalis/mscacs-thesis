import os, json, csv, argparse
import numpy as np
from sentence_transformers import SentenceTransformer

# ==========================================
# SCRIPT DESCRIPTION
# ==========================================
# This script generates dense vector embeddings for two specific datasets:
# 1. Beir FiQA (Financial Q&A)
# 2. MovieLens 20M (Movies)
#
# It supports three models:
# - 'mini':   all-MiniLM-L6-v2 (384 dimensions) - Fast, local
# - 'mpnet':  all-mpnet-base-v2 (768 dimensions) - High accuracy, local
# - 'openai': text-embedding-3-small (1536 dimensions) - API-based
#
# OUTPUT FORMAT:
# The script saves data as .npy (NumPy) files for efficiency:
# - *_emb.npy: Float32 matrix of shape (N, Dimensions).
# - *_ids.npy: Object array of shape (N,) containing the string IDs.
# ==========================================

# --- Configuration ---
BASE = os.path.dirname(os.path.dirname(__file__)) if "__file__" in globals() else os.getcwd()
DATA = os.path.join(BASE, "data")
EMB = os.path.join(BASE, "embeddings")
FIQA_DIR = os.path.join(DATA, "beir_fiqa")
ML20M_DIR = os.path.join(DATA, "movielens", "ml-20m")
DOTENV_PATH= os.path.join(BASE, ".env")

# Try importing OpenAI
try:
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv(DOTENV_PATH)
except ImportError:
    OpenAI = None

os.makedirs(EMB, exist_ok=True)

def get_embeddings(texts, model_name, model_alias):
    """
    Generates normalized embeddings for a list of text strings.

    Args:
        texts (list): List of strings to embed.
        model_name (str): The specific HuggingFace or OpenAI model name.
        model_alias (str): 'mini', 'mpnet', or 'openai'.

    Returns:
        np.array: A float32 numpy array of shape (num_texts, embedding_dim).
                  The vectors are normalized (L2 norm = 1).
    """
    print(f"   Generating embeddings for {len(texts)} texts using {model_alias}...")
    
    # --- OpenAI (Native 1536d) ---
    if model_alias == "openai":
        if OpenAI is None: raise ImportError("Please install openai")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        BATCH_SIZE = 100
        all_embs = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            batch_clean = [t.replace("\n", " ") for t in batch]
            try:
                # By default, dimensions=1536 for text-embedding-3-small
                resp = client.embeddings.create(input=batch_clean, model=model_name)
                all_embs.extend([d.embedding for d in resp.data])
            except Exception as e:
                print(f"   Error: {e}")
                raise e
            if i % 1000 == 0 and i > 0: print(f"   Progress: {i}/{len(texts)}...")
            
        emb_array = np.array(all_embs, dtype="float32")
        # Ensure normalization
        norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
        return emb_array / norms

    # --- Local Models (Native 384d or 768d) ---
    else:
        model = SentenceTransformer(model_name, device='cpu')
        return model.encode(texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)

def get_filenames(dataset_prefix, model_alias):
    """
    Standardizes output filenames.
    - 'mini' uses generic names (e.g., fiqa_corpus_emb.npy) because it is 
      the baseline for our Database Benchmark.
    - 'mpnet'/'openai' use specific names (e.g., fiqa_corpus_mpnet_emb.npy) 
      because they are used for the Model Quality Benchmark.
    """
    if model_alias == "mini":
        return f"{dataset_prefix}_emb.npy", f"{dataset_prefix}_ids.npy"
    else:
        return f"{dataset_prefix}_{model_alias}_emb.npy", f"{dataset_prefix}_ids.npy"

def process_fiqa(model_name, model_alias):
    """
    Processes the FiQA dataset (Financial Question Answering).
    
    Data Structure:
    - Corpus: A collection of financial documents/snippets.
    - Queries: Questions asking for financial information.
    
    Logic:
    Only runs for 'mini' model to save time/cost, unless 'all' was specified elsewhere.
    Combines 'title' and 'text' of the corpus to ensure the embedding captures full context.
    """
    # Only needed for DB Benchmark (MiniLM)
    if model_alias != "mini": return

    corpus_path = os.path.join(FIQA_DIR, "fiqa", "corpus.jsonl")
    queries_path = os.path.join(FIQA_DIR, "fiqa", "queries.jsonl")
    if not os.path.exists(corpus_path): 
        print(f"[FiQA] Missing file: {corpus_path}"); return

    print(f"--- Processing FiQA ({model_alias}) ---")
    
    # Load
    c_ids, c_texts = [], []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            c_ids.append(obj.get("_id"))
            c_texts.append((obj.get("title") or "") + " " + (obj.get("text") or ""))

    q_ids, q_texts = [], []
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            q_ids.append(obj.get("_id"))
            q_texts.append(obj.get("text") or "")

    # Generate embeddings
    c_embs = get_embeddings(c_texts, model_name, model_alias)
    q_embs = get_embeddings(q_texts, model_name, model_alias)

    print(f"   [Verify] FiQA Corpus Shape: {c_embs.shape}") 
    print(f"   [Verify] FiQA Query Shape:  {q_embs.shape}")

    # Save embeddings
    c_name, c_id_name = get_filenames("fiqa_corpus", model_alias)
    q_name = "fiqa_queries_emb.npy" if model_alias == "mini" else f"fiqa_queries_{model_alias}_emb.npy"
    np.save(os.path.join(EMB, c_name), c_embs)
    np.save(os.path.join(EMB, c_id_name), np.array(c_ids, dtype=object))
    np.save(os.path.join(EMB, q_name), q_embs)
    print(f"[FiQA] Saved.")

def process_ml20m(model_name, model_alias):
    """
    Processes the MovieLens 20M dataset.
    
    Data Structure:
    - Input: CSV file with columns 'movieId', 'title', 'genres'.
    
    Logic:
    We create a sentence string:
    "Toy Story. Genres: Animation, Children's, Comedy"
    This allows the vector model to understand the movie conceptually.
    """
    csv_path = os.path.join(ML20M_DIR, "movies.csv")
    if not os.path.exists(csv_path): 
        print(f"[ML20M] Missing file: {csv_path}"); return

    print(f"--- Processing MovieLens 20M ({model_alias}) ---")
    
    ids, texts = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get("title", "").strip()
            genres = row.get("genres", "").replace("|", ", ")
            ids.append(row["movieId"])
            texts.append(f"{title}. Genres: {genres}")

    embs = get_embeddings(texts, model_name, model_alias)

    print(f"   [Verify] ML20M Shape: {embs.shape}")
    if model_alias == "mini" and embs.shape[1] != 384: print("WARNING: Expected 384d!")
    if model_alias == "mpnet" and embs.shape[1] != 768: print("WARNING: Expected 768d!")
    if model_alias == "openai" and embs.shape[1] != 1536: print("WARNING: Expected 1536d!")

    # Save embeddings
    emb_name, id_name = get_filenames("ml20m_movie", model_alias)
    np.save(os.path.join(EMB, emb_name), embs)
    np.save(os.path.join(EMB, id_name), np.array(ids, dtype=object))
    print(f"[ML20M] Saved to {emb_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["mini", "mpnet", "openai"], default="mini",)
    parser.add_argument("--dataset", required=True, choices=["ml20m", "fiqa", "all"])
    args = parser.parse_args()

    if args.model == "mini":
        hf_name = "sentence-transformers/all-MiniLM-L6-v2" # Native 384
    elif args.model == "mpnet":
        hf_name = "sentence-transformers/all-mpnet-base-v2" # Native 768
    elif args.model == "openai":
        hf_name = "text-embedding-3-small" # Native 1536

    if args.dataset in ["fiqa", "all"]: process_fiqa(hf_name, args.model)
    if args.dataset in ["ml20m", "all"]: process_ml20m(hf_name, args.model)