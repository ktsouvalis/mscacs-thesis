# Performance Evaluation of Vector Databases in Recommender Systems

[![Python](https://img.shields.io/badge/Python-3.13.9-blue?logo=python&logoColor=white)](https://www.python.org/) 
[![FAISS](https://img.shields.io/badge/Library-FAISS-yellow)](https://github.com/facebookresearch/faiss) [![Milvus](https://img.shields.io/badge/Docker-Milvus-2496ED?logo=docker&logoColor=white)](https://milvus.io/) [![Chroma](https://img.shields.io/badge/Vector_DB-Chroma-orange)](https://www.trychroma.com/) [![Pinecone](https://img.shields.io/badge/Vector_DB-Pinecone-red)](https://www.pinecone.io/)
[![MiniLM](https://img.shields.io/badge/SBERT-all--MiniLM--L6--v2-ffcc00?logo=huggingface&logoColor=black)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) [![MPNet](https://img.shields.io/badge/SBERT-all--mpnet--base--v2-ffcc00?logo=huggingface&logoColor=black)](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) [![OpenAI Embeddings](https://img.shields.io/badge/OpenAI-text--embedding--3--small-412991?logo=openai&logoColor=white)](https://platform.openai.com/docs/models/text-embedding-3-small)

This repository contains the source code and experimental framework for my Master's Thesis at the **University of West Attica**. It implements a benchmark of Vector Databases (VDBMS) and Embedding Models, evaluating **Recall**, **Precision**, **nDCG** and **Latency**.

---

## 🛠️ Setup & Installation

### 1. Install Dependencies
This project was developed using **Python 3.13.9**. To ensure reproducibility, install the pinned versions:

```bash
pip install -r requirements.txt
```

### 2. Configuration (.env)
Create a `.env` file in the root directory to configure API keys for OpenAI and Pinecone:

```env
# OpenAI embeddings (used by --model openai)
OPENAI_API_KEY=your_openai_key_here

# Pinecone access (used by 30_build_indexes.py and 40_db_benchmark.py)
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_INDEX_FIQA=your_fiqa_index_name_here
PINECONE_INDEX_MOVIELENS=your_movielens_index_name_here

# Milvus configuration
MILVUS_HOST=your_milvus_host_here
MILVUS_PORT=your_milvus_port_here
```

### 3. Infrastructure Setup

#### 🐳 Milvus (Local Docker)
Run Milvus locally as a standalone container. 
Instructions can be found [here](https://milvus.io/docs/install_standalone-docker.md).

#### ☁️ Pinecone (Cloud)
Ensure you have created the necessary indexes in your Pinecone console (Serverless or Pod-based) and added the credentials to the `.env` file.

---

## 🚀 Workflow

### Step 0: Download Datasets
Download and prepare the BEIR FiQA and MovieLens 20M datasets.
```bash
python scripts/00_get_data.py
```

---

### 🧪 Phase 1: Vector Database Benchmark
*Goal: Study Infrastructure Behavior (FAISS, Chroma, Milvus, Pinecone) using `all-mini-lm-l6-v2`.*

**1. Generate Embeddings**
Create 384D embeddings for both datasets using SentenceTransformers.
```bash
python scripts/10_make_embeddings.py --model mini --dataset all
```

**2. Generate Ground Truth**
Calculate exact k-NN (Brute Force) to serve as the baseline for Recall calculations.
```bash
python scripts/20_generate_db_ground_truth.py
```

**3. Build Indexes**
Populate all vector stores with the generated data.
```bash
# FAISS
python scripts/30_build_indexes.py --dataset fiqa_corpus --backend faiss --model mini
python scripts/30_build_indexes.py --dataset ml20m_movie --backend faiss --model mini

# Chroma
python scripts/30_build_indexes.py --dataset fiqa_corpus --backend chroma --model mini
python scripts/30_build_indexes.py --dataset ml20m_movie --backend chroma --model mini

# Milvus
python scripts/30_build_indexes.py --dataset fiqa_corpus --backend milvus --model mini
python scripts/30_build_indexes.py --dataset ml20m_movie --backend milvus --model mini

# Pinecone
python scripts/30_build_indexes.py --dataset fiqa_corpus --backend pinecone --model mini
python scripts/30_build_indexes.py --dataset ml20m_movie --backend pinecone --model mini
```

**4. Run Benchmarks**
Execute queries and measure latency/recall. Use `--export` to save qualitative results (JSON).

```bash
# FAISS
python scripts/40_db_benchmark.py --dataset ml20m_movie --backend faiss --export
python scripts/40_db_benchmark.py --dataset fiqa_corpus --backend faiss --export

# Chroma
python scripts/40_db_benchmark.py --dataset ml20m_movie --backend chroma --export
python scripts/40_db_benchmark.py --dataset fiqa_corpus --backend chroma --export

# Milvus
python scripts/40_db_benchmark.py --dataset ml20m_movie --backend milvus --export
python scripts/40_db_benchmark.py --dataset fiqa_corpus --backend milvus --export

# Pinecone
python scripts/40_db_benchmark.py --dataset ml20m_movie --backend pinecone --export
python scripts/40_db_benchmark.py --dataset fiqa_corpus --backend pinecone --export
```

**5. (Optional)Visualize Results**
Generate plots comparing the backends (Recall vs Latency).
```bash
python scripts/41_plot_db_results.py
```

**6. (Optional)Extract summary csv**
```bash
python scripts/42_extract_summary.py
```

---

### 🧠 Phase 2: Embedding Model Benchmark
*Goal: Study models' behavior and their semantic representations (all-MiniLM-L6-v2, all-mpnet-base-v2 και text-embedding-3-small) using FAISS as the backend.*

**1. Create Embeddings for Advanced Models**
```bash
python scripts/10_make_embeddings.py --model mpnet --dataset ml20m
python scripts/10_make_embeddings.py --model openai --dataset ml20m
# (MiniLM was already created in Phase 1)
```

**2. Build Indexes (FAISS)**
```bash
python scripts/30_build_indexes.py --dataset ml20m_movie --backend faiss --model mpnet
python scripts/30_build_indexes.py --dataset ml20m_movie --backend faiss --model openai
```

**3. Full Retrieval Benchmark**
See how well the model retrieves relevant items from the whole corpus.
```bash
python scripts/50_model_benchmark.py --model mini --export
python scripts/50_model_benchmark.py --model mpnet --export
python scripts/50_model_benchmark.py --model openai --export
```

**4. Re-ranking Benchmark**
Study the model's ability to rank a candidate list.
```bash
python scripts/60_rerank_model_benchmark.py --model mini --export
python scripts/60_rerank_model_benchmark.py --model mpnet --export
python scripts/60_rerank_model_benchmark.py --model openai --export
```

---

## Disclaimer

This work represents an exploratory study conducted within the scope of an MSc thesis. While every effort has been made to ensure accuracy, the findings should be viewed as observations specific to the testing environment rather than definitive conclusions. Any errors or oversights are my own, and I welcome constructive feedback.