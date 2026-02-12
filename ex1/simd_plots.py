#!/usr/bin/env python3
"""
Script for the polynomial multiplication experiments.
Reads from the stats csv file and exports the relevant plots as svg files.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import glob

# -------------------- PATHS --------------------
ROOT = os.path.abspath(os.path.dirname(__file__))

data_root = os.path.join(ROOT, "data")
# runs_dir = os.path.join(data_root, "runs")
stats_dir = os.path.join(data_root, "stats")
plots_dir = os.path.join(data_root, "plots")
# os.makedirs(stats_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

STATS_CSV_NAME_PREFIX = "simd_stats"


def get_latest_csv(directory):
    # Pattern to match the prefix and any suffix
    pattern = os.path.join(directory, f"{STATS_CSV_NAME_PREFIX}*.csv")
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    # Alphabetical max works because YYYYMMDD_HHMMSS is sortable
    return max(files)


def format_k(x, pos):
    """Converts 1024 -> 1K, 2048 -> 2K, etc."""
    if x >= 1000:
        return f"{int(x/1000)}K"
    return str(int(x))


def plot_times_v_deg(df: pd.DataFrame, out_path: str):
    x = df["degree"].values

    fig, ax = plt.subplots()
    # --- Min and Mean (dashed) Values ---
    # Using the same colors to group Scalar vs AVX2, but different style
    # Scalar
    h1, = ax.plot(x, df["scalar_time_min"].values, "s-", markersize=8, label="Scalar Minimum Time", color="tab:blue")
    h2  = ax.errorbar(x, df["scalar_time_mean"].values, yerr=df["scalar_time_std"].values, fmt="^:", markersize=8, capsize=8, label="Scalar Mean Time", color="tab:blue", alpha=0.7)
    # AVX2
    h3, = ax.plot(x, df["avx2_time_min"].values, "s-", markersize=8, label="avx2 Minimum Time", color="tab:orange")
    h4  = ax.errorbar(x, df["avx2_time_mean"].values, yerr=df["avx2_time_std"].values, fmt="^:", markersize=8, capsize=8, label="avx2 Mean Time", color="tab:orange", alpha=0.7)

    # --- AXIS FORMATTING ---
    ax.set_xlabel("Polynomial Degree")
    ax.set_ylabel("Compute Time (s)")
    ax.set_title("Scalar and SIMD Compute Time vs. Polynomial Degree")

    # Set Log Scale Base 2
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")

    # Define ticks at exact powers of 2 (128, 256, 512, 1024...)
    # We generate these based on the range of the data
    low_pow = int(np.floor(np.log2(x.min() + 1)))
    high_pow = int(np.ceil(np.log2(x.max() + 1)))
    exact_powers = [2**i for i in range(low_pow, high_pow + 1)]
    
    # Set the ticks at the exact powers
    ax.set_xticks(exact_powers)

    # Apply Custom "K" Formatting
    ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(format_k))

    # Grid and Legend
    ax.grid(True, which="major", linestyle="-", alpha=0.8)
    ax.grid(True, which="minor", linestyle=":", alpha=0.4)
    ax.legend(handles=[h1, h2, h3, h4])

    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_speedup_v_deg(df: pd.DataFrame, out_path: str):
    x = df["degree"].values
    y = df["speedup"].values

    fig, ax = plt.subplots()
    ax.plot(x, y, "s--", markersize=10, label="Speedup")

    # --- AXIS FORMATTING ---
    ax.set_xlabel("Polynomial Degree")
    ax.set_ylabel("Speedup")
    ax.set_title("SIMD Speedup vs. Polynomial Degree")

    # Set Log Scale Base 2
    ax.set_xscale("log", base=2)

    # Define ticks at exact powers of 2 (128, 256, 512, 1024...)
    # We generate these based on the range of our data
    low_pow = int(np.floor(np.log2(x.min() + 1)))
    high_pow = int(np.ceil(np.log2(x.max() + 1)))
    exact_powers = [2**i for i in range(low_pow, high_pow + 1)]
    
    # Set the ticks at the exact powers
    ax.set_xticks(exact_powers)

    # Apply Custom "K" Formatting
    ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(format_k))

    # Grid and Legend
    ax.grid(True, which="major", linestyle="-", alpha=0.8)
    ax.grid(True, which="minor", linestyle=":", alpha=0.4)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def main():
    # Determine the stats csv_path
    if len(sys.argv) >= 2:
        stats_csv_path = sys.argv[1]
    else:
        print(f"[INFO] No stats file provided. Searching in: {os.path.relpath(stats_dir)}")
        stats_csv_path = get_latest_csv(stats_dir)

    # Validation
    if not stats_csv_path or not os.path.exists(stats_csv_path):
        # specific check to avoid crashing if csv_path is None
        bad_path = os.path.relpath(stats_csv_path) if stats_csv_path else "None"
        print(f"[ERROR] No valid CSV found. (Tried: {bad_path})")
        sys.exit(1)

    print(f"[INFO] Using file: {os.path.relpath(stats_csv_path)}")
    
    # Load stats file to dataframe
    stats_df = pd.read_csv(stats_csv_path)

    # Use input filename to name the plots
    raw_base = os.path.splitext(os.path.basename(stats_csv_path))[0]
    base_name = raw_base.replace("stats", "plot")
    out1 = os.path.join(plots_dir, f"{base_name}_times.svg")
    out2 = os.path.join(plots_dir, f"{base_name}_speedup.svg")

    plot_times_v_deg(stats_df, out1)
    plot_speedup_v_deg(stats_df, out2)

    print(f"[INFO] Saved: {os.path.relpath(out1)}")
    print(f"[INFO] Saved: {os.path.relpath(out2)}")


if __name__ == "__main__":
    main()
