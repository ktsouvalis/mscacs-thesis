import os, argparse, csv, json
import numpy as np
import pandas as pd
import faiss
from collections import defaultdict
import utils

# ==========================================
# SCRIPT DESCRIPTION
# ==========================================
# This script: 
# 1. Loads pre-computed embeddings and user-item interaction data.
# 2. Builds user profiles based on historical interactions.
# 3. Uses a FAISS index to retrieve top-K recommendations for each user.
# 4. Evaluates recommendation quality using nDCG and Precision metrics.
# 5. Exports results to CSV and optionally detailed JSON reports.
# ==========================================

# --- Configuration ---
RESULTS = os.path.join(utils.BASE, "results_model_benchmark")
os.makedirs(RESULTS, exist_ok=True)

def get_faiss_index_path(model_alias):
    """
    Resolves the file path for the FAISS index based on the model name.
    """
    if model_alias == "mini":
        idx_path = os.path.join(utils.INDICES_DIR, "faiss", "ml20m_movie_faiss.index") 
        if not os.path.exists(idx_path): # Fallback
             idx_path = os.path.join(utils.INDICES_DIR, "faiss", f"ml20m_movie_{model_alias}_faiss.index")
    else:
        idx_path = os.path.join(utils.INDICES_DIR, "faiss", f"ml20m_movie_{model_alias}_faiss.index")
    return idx_path

# ---------------- Recommendation Logic ----------------
def load_ratings_split(train_ratio=0.8, seed=42):
    """
    Loads ratings, filters for active users, and performs a chronological Train/Test split.
    
    Logic:
    1. Selects 1000 users who have rated at least 20 movies.
    2. Sorts their ratings by timestamp (past -> future).
    3. The first 80% of interactions become 'Train' (History).
    4. The last 20% become 'Test' (Future/Ground Truth).

    Returns:
        train (list of tuples): [(userId, movieId, rating), ...]
        test (list of tuples):  [(userId, movieId, rating), ...]
    """
    path = os.path.join(utils.DATA_DIR, "ratings.csv")
    print(f"Loading ratings from {os.path.basename(path)}...")
    
    # Load only necessary columns to save memory
    df = pd.read_csv(path, usecols=['userId', 'movieId', 'rating', 'timestamp'])
    
    # Filter: Keep only users with >= 20 interactions
    user_counts = df['userId'].value_counts()
    active_users = user_counts[user_counts >= 20].index.tolist()
    
    # Randomly sample 1000 of these active users for the benchmark
    rng = np.random.default_rng(seed)
    selected_users = rng.choice(active_users, 1000, replace=False)
    
    df_small = df[df['userId'].isin(selected_users)].copy()
    # CRITICAL: Sort by timestamp to ensure we predict future behavior based on past interactions
    df_small.sort_values('timestamp', inplace=True)
    
    train, test = [], []
    grouped = df_small.groupby('userId')
    for uid, group in grouped:
        # Create a list of (movieId, rating) tuples for this user
        interactions = list(zip(group['movieId'], group['rating']))
        # Calculate split index
        split_point = int(len(interactions) * train_ratio)
        # Append data as flat tuples: (User, Movie, Rating)
        for m, r in interactions[:split_point]: train.append((uid, m, r))
        for m, r in interactions[split_point:]: test.append((uid, m, r))
            
    return train, test

def build_user_profiles(train, id_map, vecs, min_pos=3.5):
    """
    Creates a single vector representation for each user by averaging the vectors 
    of movies they liked in the training set.

    Args:
        train: List of (uid, mid, rating) tuples.
        id_map: Dictionary mapping 'movieId' string -> Embedding Matrix Index (int).
        vecs: The numpy matrix of movie embeddings.
        min_pos: Minimum rating (e.g., 3.5) to consider a movie "liked".

    Returns:
        profiles (dict): {userId: numpy_array_vector}
        train_history_set (dict): {userId: set(movieIds)} -> Used to exclude seen movies.
        user_history_list (dict): {userId: list(movieIds)} -> Used for display/export.
    """
    print("Building User Profiles...")
    user_vecs_accum = defaultdict(list)
    train_history_set = defaultdict(set)
    user_history_list = defaultdict(list)
    
    valid_mids = set(id_map.keys())
    
    for uid, mid, rating in train:
        if rating >= min_pos:
            s_mid = str(mid)
            train_history_set[uid].add(s_mid)
            user_history_list[uid].append(s_mid)
            
            # Retrieve the vector for this movie and add it to the user's list
            if s_mid in valid_mids:
                idx = id_map[s_mid]
                user_vecs_accum[uid].append(vecs[idx])

    profiles = {}
    for uid, vector_list in user_vecs_accum.items():
        if not vector_list: continue

        # The user is the "center point" of all movies they liked.
        avg = np.mean(vector_list, axis=0)
        # Normalize to unit length (L2 norm) for cosine similarity compatibility
        norm = np.linalg.norm(avg)
        if norm > 0: avg = avg / norm
        profiles[uid] = avg.astype(np.float32)
            
    return profiles, train_history_set, user_history_list

# ---------------- Main ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["mini", "mpnet", "openai"])
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()

    print(f"--- Model Benchmark (Retrieval): {args.model.upper()} ---")

    # 1. Load Data
    # vecs: Matrix of shape (N_movies, Embedding_Dim)
    # raw_ids: List of movie IDs corresponding to the matrix rows
    # id_map: Dict mapping MovieID (str) -> Matrix Row Index (int)
    vecs, raw_ids, id_map = utils.load_embeddings_and_ids("ml20m_movie", args.model)
    # Create reverse map: Matrix Row Index -> MovieID string (needed after FAISS search)
    idx_to_mid = {i: str(mid) for i, mid in enumerate(raw_ids)}
    
    movie_titles = utils.load_movie_titles() if args.export else {}

    # 2. Build Profiles
    train, test = load_ratings_split()
    user_profiles, train_history_set, user_history_list = build_user_profiles(train, id_map, vecs)
    
    # Prepare Ground Truth (Testing Phase)
    # This identifies what the user *actually* watched/liked in the test period
    test_ground_truth = defaultdict(set)
    test_ground_truth_list = defaultdict(list)
    for uid, mid, r in test:
        if uid in user_profiles and r >= 3.5: 
            test_ground_truth[uid].add(str(mid))
            test_ground_truth_list[uid].append(str(mid))

    # Filter users: Only evaluate users who actually have positive items in the test set        
    eval_users = [u for u in user_profiles if len(test_ground_truth[u]) > 0]
    print(f"Evaluating {len(eval_users)} users...")

    # 3. Search
    idx_path = get_faiss_index_path(args.model)
    print(f"Loading Index: {os.path.basename(idx_path)}")
    index = faiss.read_index(idx_path)
    # Stack all user vectors into a single matrix for batch searching
    query_matrix = np.vstack([user_profiles[u] for u in eval_users])
    SEARCH_K = 100 
    D, I = index.search(query_matrix, SEARCH_K)

    # 4. Metrics
    metrics = {"NDCG": [], "Precision": []}
    detailed_export = []

    for i, user_id in enumerate(eval_users):
        indices = I[i]
        # Convert FAISS indices back to Movie IDs
        retrieved_mids = [idx_to_mid[idx] for idx in indices]
        
        # Filter training history
        # Remove movies the user has already seen in the Training set.
        history = train_history_set[user_id]
        final_recs = [mid for mid in retrieved_mids if mid not in history][:args.topk]
        
        truth_set = test_ground_truth[user_id]
        
        # For nDCG we count relevance based on presence in truth set
        # We create a relevance list where each recommended item is 1 if in truth set, else 0
        relevance_scores = [1 if m in truth_set else 0 for m in final_recs]
        ndcg = utils.calculate_ndcg(relevance_scores, args.topk)
        prec = utils.precision_at_k(final_recs, truth_set, args.topk)
        
        metrics["NDCG"].append(ndcg)
        metrics["Precision"].append(prec)

        if args.export and i < 50:
            def get_title(mid): return movie_titles.get(mid, f"ID:{mid}")
            detailed_export.append({
                "user_id": int(user_id),
                "metrics": {"ndcg": round(ndcg, 4), "precision": round(prec, 4)},
                "history_last_5": [get_title(m) for m in user_history_list[user_id][-5:]],
                "recommendations": [get_title(m) for m in final_recs],
                "hits": [get_title(m) for m in final_recs if m in truth_set]
            })

    # 5. Output
    avg_ndcg = np.mean(metrics["NDCG"])
    avg_prec = np.mean(metrics["Precision"])
    
    print(f"\nResults for {args.model} @ Top-{args.topk}:")
    print(f"   NDCG:      {avg_ndcg:.4f}")
    print(f"   Precision: {avg_prec:.4f}")
    
    csv_path = os.path.join(RESULTS, f"benchmark_{args.model}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        w.writerow([f"NDCG@{args.topk}", avg_ndcg])
        w.writerow([f"Precision@{args.topk}", avg_prec])
    print(f"Saved metrics to {csv_path}")

    if args.export:
        json_path = os.path.join(RESULTS, f"details_{args.model}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(detailed_export, f, indent=4, ensure_ascii=False)