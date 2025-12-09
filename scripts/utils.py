import os
import pandas as pd
import numpy as np

# --- Global Configurations ---
BASE = os.path.dirname(os.path.dirname(__file__)) if "__file__" in globals() else os.getcwd()
DATA_DIR = os.path.join(BASE, "data", "movielens", "ml-20m")
EMB_DIR = os.path.join(BASE, "embeddings")
INDICES_DIR = os.path.join(BASE, "indices")

def load_movie_titles():
    """Φορτώνει τους τίτλους ταινιών. Χρήσιμο για logs και exports."""
    path = os.path.join(DATA_DIR, "movies.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    # Mapping: ID -> "Title [Genres]"
    return dict(zip(df['movieId'].astype(str), df['title'] + " [" + df['genres'] + "]"))

def load_embeddings_and_ids(dataset, model_alias):
    """Φορτώνει Embeddings και IDs δυναμικά με βάση το μοντέλο."""
    # IDs path (πάντα το ίδιο pattern)
    ids_path = os.path.join(EMB_DIR, f"{dataset}_ids.npy")
    
    # Embeddings path
    if model_alias == "mini":
        emb_path = os.path.join(EMB_DIR, f"{dataset}_emb.npy")
    else:
        emb_path = os.path.join(EMB_DIR, f"{dataset}_{model_alias}_emb.npy")

    if not os.path.exists(ids_path) or not os.path.exists(emb_path):
        raise FileNotFoundError(f"Missing files for {dataset}/{model_alias}.\nChecked: {emb_path}")

    print(f"[Utils] Loading vectors: {os.path.basename(emb_path)}")
    vecs = np.load(emb_path)
    raw_ids = np.load(ids_path, allow_pickle=True)
    
    # Επιστροφή Map {str_id: index} και των raw vectors
    id_map = {str(mid): i for i, mid in enumerate(raw_ids)}
    return vecs, raw_ids, id_map

# --- Metrics ---

def calculate_ndcg(ranked_scores, k=None):
    """Υπολογίζει το Normalized Discounted Cumulative Gain."""
    if not ranked_scores: return 0.0
    
    limit = len(ranked_scores)
    if k is not None:
        limit = min(k, limit)
    
    dcg = 0.0
    for i in range(limit):
        rel = ranked_scores[i]
        # Χρήση 2^rel - 1 αν τα scores είναι integer ratings, αλλιώς απλό rel αν είναι binary
        # Εδώ υποθέτουμε ότι το input είναι ratings
        dcg += (2**rel - 1) / np.log2(i + 2)
        
    ideal_ratings = sorted(ranked_scores, reverse=True)
    idcg = 0.0
    for i in range(limit):
        rel = ideal_ratings[i]
        idcg += (2**rel - 1) / np.log2(i + 2)
        
    return dcg / idcg if idcg > 0 else 0.0

def precision_at_k(retrieved_ids, true_set, k):
    """Υπολογίζει το Precision @ K (για Retrieval tasks)."""
    if not retrieved_ids: return 0.0
    relevant = 0
    for mid in retrieved_ids[:k]:
        if mid in true_set:
            relevant += 1
    return relevant / k