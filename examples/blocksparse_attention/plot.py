#!/usr/bin/env python3
# filepath: /home/v-yizhaogao/tilelang/examples/blocksparse_attention/results/plot.py

import argparse
import os
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot speedup graphs for different sequence lengths"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/home/v-yizhaogao/tilelang/examples/blocksparse_attention/results/sparse_gqa_decode_varlen_mask.txt",
        help="Path to the input txt file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./mask",
        help="Output directory to save the plots",
    )
    return parser.parse_args()

def read_data(input_file):
    # Data structure: { seqlen: { batch: (dense_time, tilelang_time, triton_time) } }
    data = {}
    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            parts = [item.strip() for item in line.split(",")]
            entry = {}
            for part in parts:
                if "=" in part:
                    key, val = part.split("=", 1)
                    entry[key.strip()] = val.strip()
            try:
                batch = int(entry.get("batch"))
                seqlen = int(entry.get("max_cache_seqlen"))
                dense_time = float(entry.get("dense_time").replace("ms", ""))
                tilelang_time = float(entry.get("tilelang_time").replace("ms", ""))
                triton_time = float(entry.get("triton_time").replace("ms", ""))
            except Exception as e:
                print("Error parsing line:", line)
                continue

            if seqlen not in data:
                data[seqlen] = {}
            data[seqlen][batch] = (dense_time, tilelang_time, triton_time)
    return data

def create_and_save_plots(data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for seqlen in sorted(data.keys()):
        batches = sorted(data[seqlen].keys())
        tilelang_speedups = []
        triton_speedups = []
        for batch in batches:
            dense, tilelang, triton = data[seqlen][batch]
            # Compute speedup relative to dense_time
            tilelang_speedups.append(dense / tilelang)
            triton_speedups.append(dense / triton)

        plt.figure(figsize=(6, 4))
        # Use range(len(batches)) to ensure equal spacing
        x = range(len(batches))
        plt.plot(x, tilelang_speedups, marker="o", label="Tilelang Speedup")
        plt.plot(x, triton_speedups, marker="s", label="Triton Speedup")
        plt.axhline(y=1.0, color="r", linestyle="--", label="fa2 Baseline")

        plt.title(f"Max Cache Seqlen = {seqlen}")
        plt.xlabel("Batch Size")
        plt.ylabel("Speedup (normalized over dense)")
        # Set x-axis ticks evenly spaced and label them with batch values (8, 16, 32)
        plt.xticks(x, [str(b) for b in batches])
        # plt.grid(True)
        plt.legend()
        plt.tight_layout()

        output_path = os.path.join(output_dir, f"plot_seqlen_{seqlen}.png")
        plt.savefig(output_path)
        print(f"Saved plot for seqlen {seqlen} to {output_path}")
        plt.close()

if __name__ == "__main__":
    args = parse_args()
    data = read_data(args.input)
    create_and_save_plots(data, args.output)