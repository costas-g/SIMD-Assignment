#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import glob
# import argparse

# -------------------- PATHS --------------------
ROOT = os.path.abspath(os.path.dirname(__file__))

data_root = os.path.join(ROOT, "data")
runs_root = os.path.join(data_root, "runs")
stats_dir = os.path.join(data_root, "stats")
plots_dir = os.path.join(data_root, "plots")
os.makedirs(stats_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

CSV_NAME_PREFIX = "simd_runs"
# STATS_CSV = os.path.join(runs_root, "simd_results_stats.csv")


def get_latest_run(directory):
    # Pattern to match the prefix and any suffix
    pattern = os.path.join(directory, f"{CSV_NAME_PREFIX}*.csv")
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    # Alphabetical max works because YYYYMMDD_HHMMSS is sortable
    return max(files)


def load_csv(csv_path: str) -> pd.DataFrame:
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

    # 3. Flatten Columns: Convert ('scalar_time', 'min') -> 'scalar_time_min'
    stats.columns = [f"{col}_{stat}" for col, stat in stats.columns]
    stats = stats.reset_index()

    # 4. Compute Speedup (using the minimums, as requested)
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
    if "results" in base_name:
        new_base = base_name.replace("results", "stats")
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


def format_k(x, pos):
    """Converts 1024 -> 1K, 2048 -> 2K, etc."""
    if x >= 1000:
        return f"{int(x/1000)}K"
    return str(int(x))


def plot_times_v_deg(df: pd.DataFrame, out_path: str):
    x = df["degree"].values + 1

    fig, ax = plt.subplots()
    # --- Min Values ---
    # Using the same colors to group Scalar vs AVX, but different style
    ax.plot(x, df["scalar_time_min"].values, "^-", label="Scalar min", color="tab:blue")
    ax.plot(x, df["avx2_time_min"].values, "^-", label="AVX2 min", color="tab:orange")

    # --- Mean Values (Dashed Lines) ---
    # Note: Ensure these column names match your dataframe (e.g., 'scalar_time_mean' vs 'scalar_mean')
    ax.errorbar(x, df["scalar_time_mean"].values, yerr=df["scalar_time_std"].values,
                fmt="o--", capsize=3, label="Scalar mean", color="tab:blue", alpha=0.7)
    ax.errorbar(x, df["avx2_time_mean"].values, yerr=df["avx2_time_std"].values,
                fmt="o--", capsize=3, label="AVX2 mean", color="tab:orange", alpha=0.7)

    # --- AXIS FORMATTING ---
    ax.set_xlabel("Polynomial degree (N)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Scalar vs AVX2 Runtime")

    # 1. Set Log Scale Base 2
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")

    # 2. Force Ticks at Every Data Point
    #    This ensures 256, 512, 1024... all get a tick mark
    ax.set_xticks(x)

    # 3. Apply Custom "K" Formatting
    ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(format_k))

    # Grid and Legend
    # Grid and Legend
    ax.grid(True, which="major", linestyle="-", alpha=0.8)
    ax.grid(True, which="minor", linestyle=":", alpha=0.4)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_speedup_v_deg(df: pd.DataFrame, out_path: str):
    x = df["degree"].values
    y = df["speedup"].values

    fig, ax = plt.subplots()
    ax.plot(x, y, "o--")
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("Speedup (scalar_min / avx2_min)")
    ax.set_title("Speedup vs degree")
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def main():
    # 1. Determine the csv_path
    if len(sys.argv) >= 2:
        csv_path = sys.argv[1]
    else:
        print(f"[INFO] No file provided. Searching in: {os.path.relpath(runs_root)}")
        csv_path = get_latest_run(runs_root)

    # 2. Validation
    if not csv_path or not os.path.exists(csv_path):
        # specific check to avoid crashing if csv_path is None
        bad_path = os.path.relpath(csv_path) if csv_path else "None"
        print(f"[ERROR] No valid CSV found. (Tried: {bad_path})")
        sys.exit(1)

    print(f"[INFO] Using file: {os.path.relpath(csv_path)}")
    
    # Load file to dataframe
    # df = load_stats(csv_path) # Pass the actual path here
    raw_df = load_csv(csv_path) 
    stats_df = compute_stats(raw_df)
    export_stats(stats_df, csv_path, stats_dir)

    # Use input filename to name the plots
    raw_base = os.path.splitext(os.path.basename(csv_path))[0]
    base_name = raw_base.replace("results", "plot")
    out1 = os.path.join(plots_dir, f"{base_name}_times.svg")
    out2 = os.path.join(plots_dir, f"{base_name}_speedup.svg")

    plot_times_v_deg(stats_df, out1)
    plot_speedup_v_deg(stats_df, out2)

    print(f"[INFO] Saved: {os.path.relpath(out1)}")
    print(f"[INFO] Saved: {os.path.relpath(out2)}")


if __name__ == "__main__":
    main()
