import os
import glob
import pandas as pd
import utils

# --- Configuration ---
RESULTS_DIR = os.path.join(utils.BASE, "results_benchmark")

def parse_filename(filename):
    """
    Exports dataset and backend from the benchmark CSV filename.
    """
    basename = os.path.splitext(os.path.basename(filename))[0]
    parts = basename.split('_')
    if len(parts) < 3: return "unknown", "unknown"
    
    backend = parts[-1]
    dataset = "_".join(parts[1:-1])
    
    return dataset, backend

def main():
    print(f"Scanning for CSV files in: {RESULTS_DIR} ...")
    
    csv_files = glob.glob(os.path.join(RESULTS_DIR, "benchmark_*.csv"))
    if not csv_files:
        print("No benchmark CSV files found!")
        return

    all_data = []

    for file in csv_files:
        try:
            dataset, backend = parse_filename(file)
            df = pd.read_csv(file)
            df.insert(0, "Backend", backend.upper())
            df["Dataset"] = dataset
            all_data.append(df)
        except Exception as e:
            print(f"   [Error] Could not read {file}: {e}")

    if not all_data:
        print("No valid data found.")
        return

    full_df = pd.concat(all_data, ignore_index=True)
    unique_datasets = full_df["Dataset"].unique()
    
    print("-" * 50)
    for ds in unique_datasets:
        subset = full_df[full_df["Dataset"] == ds].copy()
        subset.drop(columns=["Dataset"], inplace=True)

        sort_cols = ["Backend"]
        if "Config/ef" in subset.columns:
            sort_cols.append("Config/ef")
        elif "ef" in subset.columns:
            sort_cols.append("ef")
            
        subset = subset.sort_values(by=sort_cols)
        
        output_filename = f"SUMMARY_{ds}.csv"
        output_path = os.path.join(RESULTS_DIR, output_filename)
        
        subset.to_csv(output_path, index=False)
        print(f"-> Created: {output_filename} ({len(subset)} rows)")
        print(f"   Backends included: {', '.join(subset['Backend'].unique())}")

    print("-" * 50)
    print("Done.")

if __name__ == "__main__":
    main()