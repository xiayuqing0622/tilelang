#!/usr/bin/env python3

import subprocess
import sys

sequences = [8192, 16384, 32768, 65536]
batches = [1, 2, 4, 8, 16, 32]

for seq in sequences:
    for batch in batches:
        print(f"Running auto_run.py with max_cache_seqlen={seq} and batch={batch}...")
        cmd = [
            sys.executable, "example_tilelang_sparse_gqa_decode_varlen_indice_lib.py",
            "--max_cache_seqlen", str(seq),
            "--batch", str(batch)
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Command failed with return code {result.returncode}")
        print("-" * 50)