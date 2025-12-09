import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import utils 

# --- Configuration ---
RESULTS_DIR = os.path.join(utils.BASE, "results_benchmark")
PLOTS_DIR = os.path.join(utils.BASE, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")

def parse_filename(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    parts = name.split('_')
    if len(parts) < 3: return "unknown", "unknown"
    backend = parts[-1]
    dataset = "_".join(parts[1:-1])
    return dataset, backend

# Βασικός χάρτης σχημάτων
BASE_MARKERS = {"FAISS": "o", "CHROMA": "v", "MILVUS": "s", "PINECONE": "D"}

def check_log_scale(df):
    x_min, x_max = df["Latency_ms"].min(), df["Latency_ms"].max()
    if x_max == 0: return False
    return (x_max / max(x_min, 1e-9)) > 20

def format_axis(ax, use_log, metric_name):
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    if use_log:
        ax.set_xscale("log")
        ax.set_xlabel("Latency (ms) - Log Scale", fontsize=12, fontweight='bold')
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    else:
        ax.set_xlabel("Latency (ms)", fontsize=12, fontweight='bold')

def plot_recall_latency_scatter(dataset, combined_df, metrics, use_log):
    if "Config/ef" in combined_df.columns:
        combined_df = combined_df.rename(columns={"Config/ef": "ef"})

    # 1. Legend Label Logic
    combined_df['Legend_Label'] = combined_df.apply(
        lambda row: f"{row['Backend']} (ef={row['ef']})", axis=1
    )
    
    # 2. Marker Map Logic
    # Map a marker per backend, regardless of ef
    unique_labels = combined_df['Legend_Label'].unique()
    full_marker_map = {}
    for label in unique_labels:
        for backend, marker in BASE_MARKERS.items():
            if label.startswith(backend):
                full_marker_map[label] = marker
                break
        if label not in full_marker_map:
            full_marker_map[label] = "o"

    # Sort by Backend and Latency for consistent plotting
    combined_df = combined_df.sort_values(by=['Backend', 'Latency_ms'])

    for metric in metrics:
        plt.figure(figsize=(14, 10))
        
        # Scatter Plot
        sns.scatterplot(
            data=combined_df, 
            x="Latency_ms", 
            y=metric,
            hue="Legend_Label",
            style="Legend_Label",
            markers=full_marker_map,
            palette="tab20", 
            s=250, 
            alpha=0.9,
            edgecolor="black"
        )
        
        ax = plt.gca()
        format_axis(ax, use_log, metric)
        plt.title(f"{dataset}: {metric} Trade-off", fontsize=16, fontweight='bold')
        
        plt.legend(
            bbox_to_anchor=(1.02, 1), 
            loc='upper left', 
            borderaxespad=0, 
            title="Configuration", 
            fontsize=10,
            labelspacing=1.2
        )
        
        plt.grid(True, linestyle='--', alpha=0.5)
        
        clean_metric = metric.replace('@', '').lower()
        out = os.path.join(PLOTS_DIR, f"41_scatter_{dataset}_{clean_metric}.png")
        
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   -> Saved Scatter Plot: {out}")

def main():
    # Only consider benchmark result CSVs to avoid SUMMARY and other aux files
    csv_files = glob.glob(os.path.join(RESULTS_DIR, "benchmark_*.csv"))
    if not csv_files: print("No CSV files found."); return

    datasets_data = {}
    for file in csv_files:
        dataset, backend = parse_filename(file)
        try:
            df = pd.read_csv(file)
            df["Backend"] = backend.upper()
            datasets_data.setdefault(dataset, []).append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    print("-" * 30)
    for dataset, df_list in datasets_data.items():
        print(f"Processing {dataset}...")
        combined_df = pd.concat(df_list, ignore_index=True)
        
        metrics = [c for c in ["Recall@10", "Recall@50", "Recall@100"] if c in combined_df.columns]
        if metrics:
            use_log = check_log_scale(combined_df)
            plot_recall_latency_scatter(dataset, combined_df, metrics, use_log)
            
    print("\nDone. Check 'plots/' folder.")

if __name__ == "__main__":
    main()