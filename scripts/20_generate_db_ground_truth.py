import os
import numpy as np
import faiss
import utils

# ==========================================
# SCRIPT DESCRIPTION
# ==========================================
# This script: 

# 1. Loads dataset embeddings (corpus and queries).
# 2. Uses Faiss IndexFlatIP to perform an exact brute-force search
#    to find the top-k nearest neighbors for each query.
# 3. Saves the ground truth indices for later evaluation of approximate methods in the following format:
#    - {dataset}_queries.npy: The actual query vectors used.
#    - {dataset}_gt_indices.npy: The indices of the top-k nearest neighbors for each query.
# ==========================================

# --- Configuration ---
GT_DIR = os.path.join(utils.BASE, "ground_truth")
os.makedirs(GT_DIR, exist_ok=True)

def generate_ground_truth(dataset_name, corpus_file, queries_file=None, k=100, num_queries=1000):
    """
    Performs an exact Brute Force search to generate the 'Ground Truth' for evaluation.
    
    Logic:
        1. Loads all database vectors (Corpus).
        2. Loads or selects specific query vectors.
        3. Uses Faiss IndexFlatIP to calculate exact Inner Product similarity 
           between every query and every corpus vector.
        4. Saves the indices of the top-k closest vectors. This becomes the 
           benchmark to test other algorithms against.

    Args:
        dataset_name (str): Label for the dataset (e.g., 'ml20m', 'fiqa').
        corpus_file (str): Filename of the embeddings to search *in* (Database).
        queries_file (str, optional): Filename of embeddings to search *with*. 
                                      If None, samples from the corpus (e.g., item-to-item recs).
        k (int): How many top neighbors to save (usually 100).
        num_queries (int): How many queries to process (to save time/space).
    """
    print(f"\n--- Generating Ground Truth for: {dataset_name} ---")
    
    # Initialize random number generator for reproducible sampling
    rng = np.random.default_rng(42)

    # ---------------------------------------------------------
    # 1. Load Corpus Vectors (The Database)
    # ---------------------------------------------------------
    # Data Structure: 2D Numpy Array
    # Shape: (N, D) where N = number of items, D = vector dimension (e.g. 384, 768)
    # Type: float32 is required by Faiss.
    corpus_path = os.path.join(utils.EMB_DIR, corpus_file)
    if not os.path.exists(corpus_path):
        print(f"Error: Corpus file not found at {corpus_path}")
        return
    corpus_vecs = np.load(corpus_path).astype('float32')
    print(f"Loaded corpus shape: {corpus_vecs.shape}")

    # ---------------------------------------------------------
    # 2. Prepare Queries (The Search Terms)
    # ---------------------------------------------------------
    # Logic: 
    # - If a specific query file exists (e.g., User Questions for FiQA), load it.
    # - If no query file (e.g., MovieLens), randomly pick existing movies 
    #   to act as queries (finding similar movies to a specific movie).
    if queries_file:
        q_path = os.path.join(utils.EMB_DIR, queries_file)
        if not os.path.exists(q_path):
            print(f"Error: Queries file not found")
            return
        query_vecs = np.load(q_path).astype('float32')
        
        # Sub-sample queries if the file is too large to process quickly
        if query_vecs.shape[0] > num_queries:
            # replace=False for unique selection
            indices = rng.choice(query_vecs.shape[0], num_queries, replace=False)
            query_vecs = query_vecs[indices]
    else:
        # MovieLens case: Randomly pick items as queries
        total_items = corpus_vecs.shape[0]
        actual_num_queries = min(num_queries, total_items)
        indices = rng.choice(total_items, actual_num_queries, replace=False)
        query_vecs = corpus_vecs[indices]

    print(f"Using {query_vecs.shape[0]} queries.")

    # ---------------------------------------------------------
    # 3. Brute Force Search (The "Truth" Calculation)
    # ---------------------------------------------------------
    # Logic: 
    # IndexFlatIP calculates the Inner Product. 
    # NOTE: If vectors are normalized, Inner Product == Cosine Similarity.
    # This index does NOT use clustering or compression. It compares 
    # every query against every database vector. It is slow but 100% accurate.
    d = corpus_vecs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(corpus_vecs)
    
    print(f"Running Brute Force search (k={k})...")
    # D: Distances (Similarity Scores) - Shape: (num_queries, k)
    # I: Indices ( The IDs of the nearest neighbors) - Shape: (num_queries, k)
    D, I = index.search(query_vecs, k)

    # ---------------------------------------------------------
    # 4. Save Results
    # ---------------------------------------------------------
    # We save the actual query vectors used and the 'I' (indices) array.
    # 'I' is the ground truth. E.g., for Query 0, I[0] contains the IDs 
    # of the true top-100 most similar items.
    np.save(os.path.join(GT_DIR, f"{dataset_name}_queries.npy"), query_vecs)
    np.save(os.path.join(GT_DIR, f"{dataset_name}_gt_indices.npy"), I)
    
    print(f"Done! Saved in {GT_DIR}")

if __name__ == "__main__":
    # --- MovieLens 20M (Item-to-Item Similarity) ---
    # Logic: Finding movies similar to other movies.
    generate_ground_truth(
        dataset_name="ml20m",
        corpus_file="ml20m_movie_emb.npy",
        k=100,
        num_queries=27278 # All movies
    )

    # --- BEIR / FiQA (Question-to-Answer Similarity) ---
    # Logic: Finding financial answers similar to user questions.
    generate_ground_truth(
        dataset_name="fiqa",
        corpus_file="fiqa_corpus_emb.npy",
        queries_file="fiqa_queries_emb.npy",
        k=100,
        num_queries=1000
    )