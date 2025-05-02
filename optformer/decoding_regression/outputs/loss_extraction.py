#!/usr/bin/env python3
import re
import sys
import os
import argparse
from statistics import mean, stdev

def extract_val_losses(log_text):
    # Matches "val_loss: 1.2345" or "val_loss=1.2345"
    pattern = re.compile(r"val_loss\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)")
    return [float(m) for m in pattern.findall(log_text)]

def extract_dataset_name(log_text, logfile_path):
    # Try to find a line like "Dataset: <name>"
    m = re.search(r"^Dataset:\s*(\S+)", log_text, re.MULTILINE)
    if m:
        return m.group(1)
    # Fallback to filename without extension
    return os.path.splitext(os.path.basename(logfile_path))[0]

def main():
    parser = argparse.ArgumentParser(
        description="Print dataset name, all val_loss entries, and compute mean ± std from a Keras log file"
    )
    parser.add_argument("logfile", help="path to training output log (e.g. yacht_output.txt)")
    args = parser.parse_args()

    try:
        with open(args.logfile, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: file not found: {args.logfile}", file=sys.stderr)
        sys.exit(1)

    # Extract and print dataset name
    dataset = extract_dataset_name(text, args.logfile)
    print(f"Dataset: {dataset}\n{'='*40}")

    vals = extract_val_losses(text)
    if not vals:
        print(f"No val_loss entries found in {args.logfile}", file=sys.stderr)
        sys.exit(1)

    # Compute and print stats
    μ = mean(vals)
    σ = stdev(vals) if len(vals) > 1 else 0.0

    print(f"\nFound {len(vals)} entries.")
    print(f"Mean NLL (val_loss): {μ:.4f}")
    print(f"Std  NLL (val_loss): {σ:.4f}")

if __name__ == "__main__":
    main()
