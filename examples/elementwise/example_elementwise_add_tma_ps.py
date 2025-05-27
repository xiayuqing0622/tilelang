# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import argparse
import itertools
import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import AutoTuner
import math

def cdiv(a, b):
    return math.ceil(a / b)


total_sm = 22

num_block_m = cdiv(65536, 128)
num_block_n = cdiv(1024, 256)
total_tiles = num_block_m * num_block_n

first_wave_tiles = total_tiles % total_sm
sm_patition_factor = total_tiles // total_sm


def ref_program(x, y):
    return x + y

def ps_elementwise_add(M, N, block_M, block_N, in_dtype, out_dtype, threads):
    @T.macro
    def compute_last_wave(
        pid: T.int32,
        A_buf: T.Tensor,
        A_buf_shared: T.SharedBuffer,
        B_buf: T.Tensor,
        B_buf_shared: T.SharedBuffer,
        C: T.Tensor,
        C_local: T.LocalBuffer,
        C_shared: T.SharedBuffer,
    ):
        if pid < first_wave_tiles:
            tile_id = pid + sm_patition_factor * total_sm
            pid_m = tile_id // T.ceildiv(N, block_N)
            pid_n = tile_id % T.ceildiv(N, block_N)
            T.copy(A_buf[pid_m * block_M, pid_n * block_N], A_buf_shared)
            T.copy(B_buf[pid_m * block_M, pid_n * block_N], B_buf_shared)
            
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = A_buf_shared[i, j] + B_buf_shared[i, j]  
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[pid_m * block_M, pid_n * block_N])


    @T.macro
    def compute_full_tiles(
        pid: T.int32,
        A_buf: T.Tensor,
        A_shared: T.SharedBuffer,
        B_buf: T.Tensor,
        B_shared: T.SharedBuffer,
        C: T.Tensor,
        C_local: T.LocalBuffer,
        C_shared: T.SharedBuffer,
    ):
        for p in T.serial(sm_patition_factor):
            tile_id = pid + p * total_sm
            pid_m = tile_id // T.ceildiv(N, block_N)
            pid_n = tile_id % T.ceildiv(N, block_N)


            T.copy(A_buf[pid_m * block_M, pid_n * block_N], A_shared)
            T.copy(B_buf[pid_m * block_M, pid_n * block_N], B_shared)
            
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = A_shared[i, j] + B_shared[i, j]
            
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[pid_m * block_M, pid_n * block_N])

    @T.prim_func
    def ps_main(A: T.Tensor((M, N), in_dtype), B: T.Tensor((M, N), in_dtype), C: T.Tensor((M, N), out_dtype)):
        with T.Kernel(total_sm, threads=threads) as pid:
            A_shared = T.alloc_shared((block_M, block_N), in_dtype)
            B_shared = T.alloc_shared((block_M, block_N), in_dtype)
            # A_shared_full_tiles = T.alloc_shared((block_M, block_N), in_dtype)
            # B_shared_full_tiles = T.alloc_shared((block_M, block_N), in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), out_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)
            if sm_patition_factor > 0:
                compute_full_tiles(pid, A, A_shared, B, B_shared, C, C_local, C_shared)
            compute_last_wave(pid, A, A_shared, B, B_shared, C, C_local, C_shared)
    return ps_main

def elementwise_add(M, N, block_M, block_N, in_dtype, out_dtype, threads):

    @T.prim_func
    def elem_add(A: T.Tensor((M, N), in_dtype), B: T.Tensor((M, N), in_dtype), C: T.Tensor((M, N),
                                                                                       out_dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), in_dtype)
            B_shared = T.alloc_shared((block_M, block_N), in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), out_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)

            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(B[by * block_M, bx * block_N], B_shared)
            
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = A_shared[i, j] + B_shared[i, j]
            
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return elem_add






def get_configs(M, N):
    block_M = [64, 128, 256]
    block_N = [64, 128, 256]
    threads = [64, 128, 256]
    configs = list(itertools.product(block_M, block_N, threads))
    return [{"block_M": bm, "block_N": bn, "threads": th} for bm, bn, th in configs]


def get_best_config(M, N):

    def kernel(block_M=None, block_N=None, threads=None):
        return elementwise_add(M, N, block_M, block_N, "float16", "float16", threads)

    autotuner = AutoTuner.from_kernel(
        kernel=kernel, configs=get_configs(M, N)).set_compile_args(
            out_idx=[-1],
            supply_type=tilelang.TensorSupplyType.Auto,
            ref_prog=ref_program,
            skip_check=False,
            target="cuda",
        )
    return autotuner.run(warmup=3, rep=20)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--use_autotune", action="store_true", default=False)
    args, _ = parser.parse_known_args()
    M, N = args.m, args.n

    a = torch.randn(M, N, dtype=torch.float16, device="cuda")
    b = torch.randn(M, N, dtype=torch.float16, device="cuda")

    if args.use_autotune:
        result = get_best_config(M, N)
        kernel = result.kernel
    else:
        # Default config
        config = {"block_M": 128, "block_N": 256, "threads": 128}
        kernel = tilelang.compile(
            elementwise_add(M, N, **config, in_dtype="float16", out_dtype="float16"), out_idx=-1)
        # print(kernel.get_kernel_source())
        kernel_ps = tilelang.compile(ps_elementwise_add(M, N, **config, in_dtype="float16", out_dtype="float16"), out_idx=-1)
    out = kernel(a, b)
    out_ps = kernel_ps(a, b)
    out_ref = ref_program(a, b)
    print(out_ps.shape, out_ref.shape)
    torch.testing.assert_close(out, ref_program(a, b), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(out_ps, out_ref, rtol=1e-2, atol=1e-2)

if __name__ == "__main__":
    main()
