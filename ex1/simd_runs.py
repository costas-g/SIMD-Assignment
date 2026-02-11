#!/usr/bin/env python3
"""
Benchmark driver for the polynomial multiplication experiments.

Usage:
    py runs.py           # run experiments, build stats, write stats CSV
"""
import os
import re
import subprocess
import numpy as np
import pandas as pd
import argparse
from datetime import datetime

# ---------------- CONFIG ----------------
ROOT = os.path.abspath(os.path.dirname(__file__))
BIN_DIR = os.path.join(ROOT, "bin")
EXE_PATH = os.path.join(BIN_DIR, "main")

data_root = os.path.join(ROOT, "data")
os.makedirs(data_root, exist_ok=True)

runs_root = os.path.join(data_root, "runs")
os.makedirs(runs_root, exist_ok=True)

# Generate the timestamp string
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# Construct the unique filename
filename = f"simd_runs_{timestamp}.csv"
# FINAL output CSV (stats) adjust name 
STATS_CSV = os.path.join(runs_root, filename)

parser = argparse.ArgumentParser(description="Run polynomial multiplication (serial vs parallel)")
parser.add_argument("-s", "--skip-experiments", action="store_true",
                    help="Skip running experiments and use the temporary raw CSV")
parser.add_argument("--keep-raw", action="store_true",
                    help="Keep the temporary raw CSV (otherwise it is deleted at the end).")
args = parser.parse_args()


# DEGREE_VALUES = [10 ** p for p in range(3, 6)]  # 1e3..1e5
MIN_EXP = 7  # 2**7  = 128
MAX_EXP = 17 # 2**17 = 131072
DEGREE_VALUES = [2 ** p - 1 for p in range(MIN_EXP, MAX_EXP+1)] # from 127 to 131071
REPEATS = [min(30, 3 * 2**i) for i in range(len(DEGREE_VALUES))][::-1]

# CVS Header line
# HEADER_RE = re.compile(r"^valid,deg,scalar_time,avx2_time\s*$")
DATA_RE = re.compile(
    r"^(?P<valid>[01])"
    r",(?P<deg>-?\d+)"
    r",(?P<scalar_time>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r",(?P<avx2_time>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"\s*$"
)


# ---------- RUNNING THE PROGRAM ----------
def run_single(degree: int):
    
    cmd = [  EXE_PATH, str(degree)]
    
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Program failed for degree={degree}")
        print("[CMD]", e.cmd)
        print("---- STDOUT ----")
        print(e.stdout)
        print("---- STDERR ----")
        print(e.stderr)
        raise  # or return None to skip
        
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")

    last_match = None
    for line in output.splitlines():
        m = DATA_RE.match(line.strip())
        if m:
            last_match = m

    if last_match is None:
        raise RuntimeError("No CSV data line found in output")

    valid = int(last_match["valid"])
    deg = int(last_match["deg"])
    scalar_time = float(last_match["scalar_time"])
    avx2_time = float(last_match["avx2_time"])

    return valid, deg, scalar_time, avx2_time

def run_and_get_row(degree):
    """
    Run run_single() one time, and return one dict row for the CSV.
    """
    try:
        valid, deg, scalar_time, avx2_time = run_single(degree)
    except RuntimeError as e:
        print("[ERROR]", e)

    speedup = scalar_time / avx2_time if avx2_time > 0 else float("inf")
    return {
        "valid"       : valid,
        "degree"      : deg,
        "scalar_time" : f"{scalar_time:.6f}",
        "avx2_time"   : f"{avx2_time:.6f}",
        "speedup"     : f"{speedup:.3f}",
    }


def do_runs_and_write_csv():
    """
    For each (degree):
      run repeats -> compute stats -> append ONE row to STATS_CSV
    """
    # Fresh run: delete stats file if it exists
    if os.path.exists(STATS_CSV):
        os.remove(STATS_CSV)

    file_exists = False

    # for degree in DEGREE_VALUES:
    for degree, my_repeats in zip(DEGREE_VALUES, REPEATS):
        
        print(f"[INFO] Degree {degree:>6}, Runs {my_repeats:>2}")
        print(f"[INFO] Appending to file...", end=" ", flush=True)

        for it in range(my_repeats):
            row = run_and_get_row(degree)
            if row is None:
                print(f"[WARN] No successful samples for run={it+1} of degree={degree}. Skipping.")
                continue

            pd.DataFrame([row]).to_csv(
                STATS_CSV,
                mode="a",
                header=not file_exists,
                index=False
            )

            file_exists = True
            print(f"|", end="", flush=True)

        print("") # newline

    print(f"[INFO] Appended all run results -> {STATS_CSV}")


# ---------- MAIN ----------
def main():
    print(f"[INFO] Executable: {EXE_PATH}")
    print(f"[INFO] Degrees: {DEGREE_VALUES}")
    print(f"[INFO] REPEATS: {REPEATS}")
    print(f"[INFO] Stats CSV: {STATS_CSV}")

    if args.skip_experiments:
        if not os.path.exists(STATS_CSV):
            raise FileNotFoundError(f"--skip-experiments but stats CSV not found: {STATS_CSV}")
        stats_df = pd.read_csv(STATS_CSV)
        print(f"[INFO] Loaded stats CSV (rows={len(stats_df)})")
        return

    do_runs_and_write_csv()

    stats_df = pd.read_csv(STATS_CSV)
    print(f"[INFO] Done. Final stats rows: {len(stats_df)}")
    # TODO: plots/tables using stats_df

if __name__ == "__main__":
    main()
