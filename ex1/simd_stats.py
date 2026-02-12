#!/usr/bin/env python3
"""
Script for the polynomial multiplication experiments.
Reads from the raw runs csv file and exports statistical data to a stats csv file, ready for plotting.
"""
import os
import sys
import pandas as pd
import numpy as np
import glob

# -------------------- PATHS --------------------
ROOT = os.path.abspath(os.path.dirname(__file__))

data_root = os.path.join(ROOT, "data")
runs_dir = os.path.join(data_root, "runs")
stats_dir = os.path.join(data_root, "stats")
# plots_dir = os.path.join(data_root, "plots")
os.makedirs(stats_dir, exist_ok=True)
# os.makedirs(plots_dir, exist_ok=True)

RUNS_CSV_NAME_PREFIX = "simd_runs"


def get_latest_csv(directory):
    # Pattern to match the prefix and any suffix
    pattern = os.path.join(directory, f"{RUNS_CSV_NAME_PREFIX}*.csv")
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    # Alphabetical max works because YYYYMMDD_HHMMSS is sortable
    return max(files)


def load_runs_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing file: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"valid", "degree", "scalar_time", "avx2_time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)}. Found: {list(df.columns)}")

    df = df.sort_values("degree").reset_index(drop=True)
    df["valid"] = df["valid"].astype(int)
    df["degree"] = df["degree"].astype(int)
    for c in ["scalar_time", "avx2_time"]:
        df[c] = df[c].astype(float)
    return df


def compute_stats(raw_df: pd.DataFrame) -> pd.DataFrame:
    # 1. Filter: Keep only valid runs
    #    If valid is 0, the times might be garbage or timeouts.
    df = raw_df[raw_df["valid"] == 1].copy()

    if df.empty:
        print("[WARNING] No valid rows found in input data!")
        return pd.DataFrame(columns=["degree", "scalar_time_min", "avx2_time_min", "speedup"])

    # 2. Aggregation: Calculate min, mean, and std for both columns
    #    This creates a hierarchical column index (MultiIndex)
    stats = df.groupby("degree")[["scalar_time", "avx2_time"]].agg(["min", "mean", "std"])

    # 3. Flatten Columns: Convert ('scalar_time', 'min') to 'scalar_time_min'
    stats.columns = [f"{col}_{stat}" for col, stat in stats.columns]
    stats = stats.reset_index()

    # 4. Compute Speedup (using the minimums)
    #    Using numpy where to handle potential division by zero safely
    stats["speedup"] = np.where(
        stats["avx2_time_min"] > 0, 
        stats["scalar_time_min"] / stats["avx2_time_min"], 
        0.0
    )

    # Optional: Fill NaN std deviations with 0.0 (happens if repeats=1)
    stats.fillna(0.0, inplace=True)

    # Optional: Sort by degree just to be safe
    return stats.sort_values("degree").reset_index(drop=True)


def export_stats(stats_df: pd.DataFrame, input_csv_path: str, output_dir: str):
    """
    Exports the computed stats to a CSV file.
    Filename format: simd_stats_YYYYMMDD_HHMMSS.csv (derived from input).
    """
    # 1. Generate new filename based on input
    base_name = os.path.splitext(os.path.basename(input_csv_path))[0]
    
    # If input is 'simd_results_2026...', this becomes 'simd_stats_2026...'
    if "runs" in base_name:
        new_base = base_name.replace("runs", "stats")
    else:
        # Fallback if input didn't follow the naming convention
        new_base = f"simd_stats_{base_name}"

    out_path = os.path.join(output_dir, f"{new_base}.csv")

    # 2. formatting for the CSV
    # Create a copy so we don't modify the df used for plotting in memory
    export_df = stats_df.copy()

    # 3. Save
    # float_format="%.6f" ensures times (like 0.000015) don't turn into 1.5e-05
    export_df.to_csv(out_path, index=False, float_format="%.6f")
    
    print(f"[INFO] Stats exported: {os.path.relpath(out_path)}")


def main():
    # Determine the csv_path
    if len(sys.argv) >= 2:
        runs_csv_path = sys.argv[1]
    else:
        print(f"[INFO] No runs file provided. Searching in: {os.path.relpath(runs_dir)}")
        runs_csv_path = get_latest_csv(runs_dir)

    # Validation
    if not runs_csv_path or not os.path.exists(runs_csv_path):
        # specific check to avoid crashing if csv_path is None
        bad_path = os.path.relpath(runs_csv_path) if runs_csv_path else "None"
        print(f"[ERROR] No valid CSV found. (Tried: {bad_path})")
        sys.exit(1)

    print(f"[INFO] Using file: {os.path.relpath(runs_csv_path)}")
    
    # Load file to dataframe
    runs_df = load_runs_csv(runs_csv_path) 
    stats_df = compute_stats(runs_df)
    export_stats(stats_df, runs_csv_path, stats_dir)


if __name__ == "__main__":
    main()
