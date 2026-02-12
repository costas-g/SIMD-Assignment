#!/usr/bin/env python3
"""
Benchmark driver for the polynomial multiplication experiments.
Runs experiments and writes the raw results to a runs .csv file.
"""
import os
import re
import subprocess
import pandas as pd
import argparse
from datetime import datetime

# ---------------- CONFIG ----------------
ROOT = os.path.abspath(os.path.dirname(__file__))
BIN_DIR = os.path.join(ROOT, "bin")
EXE_PATH = os.path.join(BIN_DIR, "main")

data_root = os.path.join(ROOT, "data")
os.makedirs(data_root, exist_ok=True)

runs_dir = os.path.join(data_root, "runs")
os.makedirs(runs_dir, exist_ok=True)

# Generate the timestamp string
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# Construct the unique filename
filename = f"simd_runs_{timestamp}.csv"
# FINAL output CSV 
RUNS_CSV_PATH = os.path.join(runs_dir, filename)

# parser = argparse.ArgumentParser(description="Run polynomial multiplication (Scalar vs SIMD)")
# parser.add_argument("-s", "--skip-experiments", action="store_true", help="Skip running experiments and use the temporary raw CSV")
# parser.add_argument("--keep-raw", action="store_true", help="Keep the temporary raw CSV (otherwise it is deleted at the end).")
# args = parser.parse_args()

BASE        = 2
MIN_EXP     = 7     # 2**7  = 128
MAX_EXP     = 17    # 2**17 = 131072
MIN_REPEATS = 3     # number of repeated runs at the slowest experiment
MAX_REPEATS = 30    # max number of repeated runs

DEGREE_VALUES = [BASE ** p - 1 for p in range(MIN_EXP, MAX_EXP+1)]                                  # from 127 to 131071
REPEATS = [min(MAX_REPEATS, MIN_REPEATS * (BASE*BASE)**i) for i in range(len(DEGREE_VALUES))][::-1] # 30 max repeats, down to 3 repeats at the slowest experiment

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
    if os.path.exists(RUNS_CSV_PATH):
        os.remove(RUNS_CSV_PATH)

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
                RUNS_CSV_PATH,
                mode="a",
                header=not file_exists,
                index=False
            )

            file_exists = True
            print(f"|", end="", flush=True) # show live progress

        print("") # newline

    print(f"[INFO] Appended all run results -> {RUNS_CSV_PATH}")


# ---------- MAIN ----------
def main():
    print(f"[INFO] Executable: {EXE_PATH}")
    print(f"[INFO] Degrees: {DEGREE_VALUES}")
    print(f"[INFO] REPEATS: {REPEATS}")
    print(f"[INFO] Runs CSV: {os.path.relpath(RUNS_CSV_PATH)}")

    # if args.skip_experiments:
    #     if not os.path.exists(RUNS_CSV_PATH):
    #         raise FileNotFoundError(f"--skip-experiments but stats CSV not found: {os.path.relpath(RUNS_CSV_PATH)}")
    #     runs_df = pd.read_csv(RUNS_CSV_PATH)
    #     print(f"[INFO] Loaded stats CSV (rows={len(runs_df)})")
    #     return

    do_runs_and_write_csv()

    runs_df = pd.read_csv(RUNS_CSV_PATH)
    print(f"[INFO] Done. Final stats rows: {len(runs_df)}")
    print(f"[INFO] Results exported: {os.path.relpath(RUNS_CSV_PATH)}")
    # TODO: stats/plots using runs_df

if __name__ == "__main__":
    main()
