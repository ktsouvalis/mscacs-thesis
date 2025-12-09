import os, argparse, json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import utils 

# ==========================================
# SCRIPT DESCRIPTION
# ==========================================
# This script:
# 1. Loads pre-computed embeddings and user-item interaction data.
# 2. Builds user profiles based on historical interactions.
# 3. Uses cosine similarity to re-rank candidate (inside the ground truth) items for each user.
# 4. Evaluates re-ranking quality using nDCG and Precision metrics.
# 5. Exports results to CSV and optionally detailed JSON reports.
# ==========================================

# --- Configuration ---
RESULTS = os.path.join(utils.BASE, "results_rerank")
os.makedirs(RESULTS, exist_ok=True)

# ---------------- Helpers ----------------
def load_ratings_split(train_ratio=0.8, seed=42):
    """
    Loads ratings from the dataset, filters for active users, and splits 
    the interactions for each user into a Training set (History) and a Test set (Candidates).

    The split is temporal: older interactions go to 'train', newer ones to 'test'.
    
    Returns:
        dict: A dictionary where keys are 'userId's (int) and values are dictionaries:
              {
                  "train": list of (movieId, rating) tuples (User History),
                  "test": list of (movieId, rating) tuples (Candidate set for re-ranking)
              }
    """
    path = os.path.join(utils.DATA_DIR, "ratings.csv")
    print("Loading ratings...")
    df = pd.read_csv(path, usecols=['userId', 'movieId', 'rating', 'timestamp'])
    
    # Only consider users with at least 20 ratings
    user_counts = df['userId'].value_counts()
    active_users = user_counts[user_counts >= 20].index.tolist()
    
    # Consistent Random Sampling
    rng = np.random.default_rng(seed)
    selected_users = rng.choice(active_users, 1000, replace=False)
    
    df = df[df['userId'].isin(selected_users)].copy()
    # CRITICAL: Sort by timestamp to ensure a temporal split (History vs Future)
    df.sort_values('timestamp', inplace=True)
    
    user_data = {}
    grouped = df.groupby('userId')
    
    for uid, group in grouped:
        interactions = list(zip(group['movieId'], group['rating']))
        split = int(len(interactions) * train_ratio)
        user_data[uid] = {
            "train": interactions[:split], 
            "test": interactions[split:]   
        }
    return user_data

# ---------------- Main ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["mini", "mpnet", "openai"])
    parser.add_argument("--export", action="store_true", help="Export JSON with details")
    parser.add_argument("--profile_threshold", type=float, default=3.5)
    parser.add_argument("--topk", type=int, default=10, help="K for metric evaluation")
    args = parser.parse_args()

    # 1. Load Data
    # vecs: NumPy array of movie embeddings (vectors)
    # raw_ids: List of raw movie IDs corresponding to the indices of 'vecs'
    # id_map: Dictionary mapping movie IDs (str) to their index in 'vecs' for fast lookup
    vecs, raw_ids, id_map = utils.load_embeddings_and_ids("ml20m_movie", args.model)
    
    # user_data: Structure loaded in Step 1 {uid: {"train": [(mid, rating), ...], "test": [...]}}
    user_data = load_ratings_split()
    movie_titles = utils.load_movie_titles() if args.export else {}
    
    # Store metrics
    metrics_all = {"NDCG_Full": [], "NDCG_TopK": [], "Precision_TopK": []}
    detailed_export = []
    
    print(f"--- Running Re-Ranking Benchmark ({args.model.upper()}) ---")
    
    for i, (uid, data) in enumerate(user_data.items()):
        # 2. Build User Profile (Mean Pooling from the Train set)
        history_vecs = []
        
        for mid, rate in data['train']:
            # Use only movies the user liked (>threshold) for the profile
            if rate >= args.profile_threshold and str(mid) in id_map:
                idx = id_map[str(mid)]
                history_vecs.append(vecs[idx])
        
        if not history_vecs: continue

        # CRITICAL: Create the User Vector using MEAN POOLING (averaging the vectors of liked movies)
        user_vec = np.mean(history_vecs, axis=0).reshape(1, -1)
        
        # 3. Prepare Candidates (Test set)
        # The candidates are the movies the user interacted with AFTER the training period.
        candidate_vecs = []
        candidate_ratings = []
        candidate_ids = []
        
        for mid, rate in data['test']:
            if str(mid) in id_map:
                idx = id_map[str(mid)]
                candidate_vecs.append(vecs[idx])
                candidate_ratings.append(rate)
                candidate_ids.append(str(mid))
                
        if not candidate_vecs: continue
        
        # 4. Model Scoring (Cosine Similarity)
        # Calculate the relevance score by finding the cosine similarity between the 
        # User Profile vector and all Candidate Movie vectors.
        sim_scores = cosine_similarity(user_vec, np.array(candidate_vecs))[0]
        
        # 5. Rank
        # Pair scores with ratings and ids, then sort by score
        by_model = sorted(zip(sim_scores, candidate_ratings, candidate_ids), key=lambda x: x[0], reverse=True)
        
        # For Export: We want to see what the ideal ranking would be
        if args.export:
            by_rating = sorted(zip(candidate_ratings, candidate_ids), key=lambda x: x[0], reverse=True)
            ideal_rank_map = {mid: r+1 for r, (_, mid) in enumerate(by_rating)}

        # 6. Evaluate Metrics
        ranked_ratings_only = [r for s, r, m in by_model] # List of ratings in the order predicted by the model
        ranked_ids = [m for s, r, m in by_model]
        
        # NDCG (Works with ratings, e.g., 5.0, 4.0)
        ndcg_full = utils.calculate_ndcg(ranked_ratings_only, k=None)
        ndcg_topk = utils.calculate_ndcg(ranked_ratings_only, k=args.topk)
        
        # Precision (We need binary relevance, so a set with ratings >= threshold)
        true_positives_set = {mid for mid, rate in zip(candidate_ids, candidate_ratings) if rate >= args.profile_threshold}
        prec_topk = utils.precision_at_k(ranked_ids, true_positives_set, args.topk)
        
        metrics_all["NDCG_Full"].append(ndcg_full)
        metrics_all["NDCG_TopK"].append(ndcg_topk)
        metrics_all["Precision_TopK"].append(prec_topk)

        # 7. Export Logic
        if args.export and i < 50:
            def get_title(mid): return movie_titles.get(mid, f"ID:{mid}")
            
            comparison_list = []
            for rank, (pred_score, real_rating, mid) in enumerate(by_model[:10]):
                comparison_list.append({
                    "title": get_title(mid),
                    "user_rating": float(real_rating),
                    "model_score": float(pred_score),
                    "model_rank": rank + 1,
                    "ideal_rank": ideal_rank_map.get(mid, -1)
                })

            detailed_export.append({
                "user_id": int(uid),
                "metrics": {
                    "ndcg_full": round(ndcg_full, 4),
                    f"ndcg_@{args.topk}": round(ndcg_topk, 4),
                    f"precision_@{args.topk}": round(prec_topk, 4)
                },
                "ranking_comparison_top10": comparison_list
            })

    # Averages
    # Calculate average metrics across all 1000 users to evaluate global model performance.
    avg_ndcg_full = np.mean(metrics_all["NDCG_Full"])
    avg_ndcg_topk = np.mean(metrics_all["NDCG_TopK"])
    avg_prec_topk = np.mean(metrics_all["Precision_TopK"])
    
    print(f"\nResults for {args.model}:")
    print(f"   NDCG (Full List):     {avg_ndcg_full:.4f}")
    print(f"   NDCG @ Top-{args.topk}:       {avg_ndcg_topk:.4f}")
    print(f"   Precision @ Top-{args.topk}:  {avg_prec_topk:.4f}")
    
    # Save Summary CSV: Useful for comparing different models (e.g., mini vs mpnet) side-by-side.
    csv_path = os.path.join(RESULTS, f"rerank_{args.model}.csv")
    with open(csv_path, "w") as f:
        f.write(f"metric,value\n")
        f.write(f"ndcg_full,{avg_ndcg_full}\n")
        f.write(f"ndcg_{args.topk},{avg_ndcg_topk}\n")
        f.write(f"precision_{args.topk},{avg_prec_topk}\n")
    print(f"Saved metrics to {csv_path}")

    # Save Detailed JSON: Contains specific ranking comparisons for the first 50 users.
    # This is nice for qualitative analysis (e.g., "Why did the model recommend X?").
    if args.export:
        json_path = os.path.join(RESULTS, f"rerank_details_{args.model}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(detailed_export, f, indent=4, ensure_ascii=False)
        print(f"Saved details to {json_path}")