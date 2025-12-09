import os, zipfile, urllib.request

BASE = os.path.dirname(os.path.dirname(__file__)) if "__file__" in globals() else os.getcwd()
DATA = os.path.join(BASE, "data")
FIQA_DIR = os.path.join(DATA, "beir_fiqa")
ML_DIR = os.path.join(DATA, "movielens")

os.makedirs(FIQA_DIR, exist_ok=True)
os.makedirs(ML_DIR, exist_ok=True)

# --- BEIR FiQA (corpus.jsonl, queries.jsonl, qrels/test.tsv) ---
fiqa_url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"
fiqa_zip = os.path.join(FIQA_DIR, "fiqa.zip")
if not os.path.exists(os.path.join(FIQA_DIR, "corpus.jsonl")):
    print("Downloading FiQA...")
    urllib.request.urlretrieve(fiqa_url, fiqa_zip)
    with zipfile.ZipFile(fiqa_zip, 'r') as z:
        z.extractall(FIQA_DIR)
    print("FiQA ready.")
else:
    print("FiQA already present.")

# --- MovieLens 20M ---
ml_20m_url = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
ml_20m_zip = os.path.join(ML_DIR, "ml-20m.zip")
if not os.path.exists(os.path.join(ML_DIR, "ml-20m")):
    print("Downloading MovieLens 20M...")
    urllib.request.urlretrieve(ml_20m_url, ml_20m_zip)
    with zipfile.ZipFile(ml_20m_zip, 'r') as z:
        z.extractall(ML_DIR)
    print("MovieLens 20M ready.")
else:
    print("MovieLens 20M already present.")