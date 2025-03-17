import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
from einops import rearrange, einsum
import argparse
import itertools
from flash_attn_interface import flash_attn_with_kvcache, flash_attn_func
from einops import repeat
import time

def get_configs():
    block_N = [128]
    block_H = [64, 128, 192]
    num_split = [2, 4, 8]
    num_stages = [1, 2, 3]
    threads = [128, 256, 384]
    _configs = list(itertools.product(block_N, block_H, num_split, num_stages, threads))

    configs = [{
        'block_N': c[0],
        'block_H': c[1],
        'num_split': c[2],
        'num_stages': c[3],
        'threads': c[4]
    } for c in _configs]
    return configs


def flashattn(batch, heads, groups, seqlen_kv, dim, selected_blocks, tune=False):
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    shape_q = [batch, heads, dim]
    shape_k = [batch, seqlen_kv, groups, dim]
    shape_v = [batch, seqlen_kv, groups, dim]
    shape_o = [batch, heads, dim]
    dtype = "float16"
    accum_dtype = "float"
    kv_group_num = heads // groups


    def kernel_func(block_N, block_H, num_split, num_stages, threads):
        part_shape = [batch, heads, num_split, dim]
        valid_block_H = min(block_H, kv_group_num)

        @T.macro
        def flash_attn_split(
                Q: T.Buffer(shape_q, dtype),
                K: T.Buffer(shape_k, dtype),
                V: T.Buffer(shape_v, dtype),
                BlockIndices: T.Buffer([batch, groups, selected_blocks], "int32"),
                glse: T.Buffer([batch, heads, num_split], dtype),
                Output_partial: T.Buffer(part_shape, dtype),
        ):
            with T.Kernel(
                    batch, heads // valid_block_H, num_split, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_H, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([valid_block_H, dim], dtype)
                acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
                acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_H], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
                scores_scale = T.alloc_fragment([block_H], accum_dtype)
                scores_sum = T.alloc_fragment([block_H], accum_dtype)
                logsum = T.alloc_fragment([block_H], accum_dtype)

                bid = bx
                hid = by
                sid = bz
                cur_kv_head = hid // (kv_group_num // valid_block_H)

                T.copy(Q[bid, hid * valid_block_H:hid * valid_block_H + block_H, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                #T.ceildiv((seqlen_kv // num_split), block_N)
                loop_range = T.ceildiv(selected_blocks, num_split)
                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    i_s = BlockIndices[bid, hid, sid * loop_range + k] * block_N
                    if i_s >= 0:
                        # T.copy(
                        #     K[bid, (seqlen_kv // num_split) * sid +
                        #       k * block_N:(seqlen_kv // num_split) * sid + (k + 1) * block_N,
                        #       cur_kv_head, :], K_shared)
                        T.copy(
                            K[bid, i_s: i_s + block_N,
                            cur_kv_head, :], K_shared)
                        # T.copy(
                        #     mask[bid, (seqlen_kv // num_split) * sid +
                        #          k * block_N:(seqlen_kv // num_split) * sid + (k + 1) * block_N,
                        #          cur_kv_head], mask_local)
                        T.clear(acc_s)
                        T.gemm(
                            Q_shared,
                            K_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow)
                        # for i, j in T.Parallel(block_H, block_N):
                        #     acc_s[i, j] = T.if_then_else(mask_local[j] != 0, acc_s[i, j],
                        #                                  -T.infinity(accum_dtype))
                        T.copy(scores_max, scores_max_prev)
                        T.fill(scores_max, -T.infinity(accum_dtype))
                        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                        for i in T.Parallel(block_H):
                            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                        for i, j in T.Parallel(block_H, block_N):
                            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                        T.reduce_sum(acc_s, scores_sum, dim=1)
                        for i in T.Parallel(block_H):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                        T.copy(acc_s, acc_s_cast)
                        for i, j in T.Parallel(block_H, dim):
                            acc_o[i, j] *= scores_scale[i]
                        # T.copy(
                        #     V[bid, (seqlen_kv // num_split) * sid +
                        #       k * block_N:(seqlen_kv // num_split) * sid + (k + 1) * block_N,
                        #       cur_kv_head, :], V_shared)
                        T.copy(
                            V[bid, i_s:i_s + block_N,
                            cur_kv_head, :], V_shared)
                        T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] /= logsum[i]
                for i in T.Parallel(block_H):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale

                T.copy(logsum[:valid_block_H],
                       glse[bid, hid * valid_block_H:(hid + 1) * valid_block_H, sid])
                T.copy(acc_o[:valid_block_H, :], O_shared)
                T.copy(O_shared, Output_partial[bid, hid * valid_block_H:(hid + 1) * valid_block_H,
                                                sid, :])

        @T.macro
        def combine(
                glse: T.Buffer([batch, heads, num_split], dtype),
                Output_partial: T.Buffer(part_shape, dtype),
                Output: T.Buffer(shape_o, dtype),
        ):
            with T.Kernel(heads, batch, threads=128) as (by, bz):
                po_local = T.alloc_fragment([dim], dtype)
                o_accum_local = T.alloc_fragment([dim], accum_dtype)
                lse_local = T.alloc_fragment([num_split, 128], dtype)
                lse_local_split = T.alloc_local([1], accum_dtype)
                lse_logsum_local = T.alloc_local([1], accum_dtype)
                lse_max_local = T.alloc_fragment([128], accum_dtype)
                scale_local = T.alloc_local([1], accum_dtype)

                T.annotate_layout({
                    lse_logsum_local:
                        T.Fragment(lse_logsum_local.shape, forward_thread_fn=lambda i: i),
                    lse_max_local:
                        T.Fragment(lse_max_local.shape, forward_thread_fn=lambda i: i),
                    lse_local:
                        T.Fragment(lse_local.shape, forward_thread_fn=lambda i, j: j),
                })

                T.clear(lse_logsum_local)
                T.clear(o_accum_local)
                for k in T.Parallel(num_split):
                    lse_local[k, 0] = glse[bz, by, k]
                T.reduce_max(lse_local, lse_max_local, dim=0, clear=True)
                for k in T.Pipelined(num_split, num_stages=1):
                    lse_local_split[0] = glse[bz, by, k]
                    lse_logsum_local[0] += T.exp2(lse_local_split[0] - lse_max_local[0])
                lse_logsum_local[0] = T.log2(lse_logsum_local[0]) + lse_max_local[0]
                for k in T.serial(num_split):
                    for i in T.Parallel(dim):
                        po_local[i] = Output_partial[bz, by, k, i]
                    lse_local_split[0] = glse[bz, by, k]
                    scale_local[0] = T.exp2(lse_local_split[0] - lse_logsum_local[0])
                    for i in T.Parallel(dim):
                        o_accum_local[i] += po_local[i] * scale_local[0]
                for i in T.Parallel(dim):
                    Output[bz, by, i] = o_accum_local[i]

        @T.prim_func
        def main(
                Q: T.Buffer(shape_q, dtype),
                K: T.Buffer(shape_k, dtype),
                V: T.Buffer(shape_v, dtype),
                BlockIndices: T.Buffer([batch, groups, selected_blocks], "int32"),
                glse: T.Buffer([batch, heads, num_split], dtype),
                Output_partial: T.Buffer(part_shape, dtype),
                Output: T.Buffer(shape_o, dtype),
        ):
            flash_attn_split(Q, K, V, BlockIndices, glse, Output_partial)
            combine(glse, Output_partial, Output)

        return main

    if tune:

        @autotune(
            configs=get_configs(),
            keys=["block_N", "block_H", "num_split", "num_stages", "threads"],
            warmup=10,
            rep=10)
        @jit(
            out_idx=[6],
            supply_type=tilelang.TensorSupplyType.Auto,
            ref_prog=ref_program,
            max_mismatched_ratio=0.05,
            profiler="auto")
        def kernel(block_N=None, block_H=None, num_split=None, num_stages=None, threads=None):
            return kernel_func(block_N, block_H, num_split, num_stages, threads)

        return kernel()
    else:

        def kernel(block_N, block_H, num_split, num_stages, threads):
            return kernel_func(block_N, block_H, num_split, num_stages, threads)

        return kernel



def ref_program(query, key, value,  BlockIndices, glse, Output_partial):
    query = query.unsqueeze(1)
    output = flash_attn_with_kvcache(query, key, value)
    output = output.squeeze(1)
    return output



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--heads', type=int, default=32, help='heads')
    parser.add_argument('--groups', type=int, default=8, help='groups')
    parser.add_argument('--kv_seqlen', type=int, default=8192, help='kv sequence length')
    parser.add_argument('--dim', type=int, default=128, help='dim')
    parser.add_argument('--tune', action='store_true', help='tune configs')
    args = parser.parse_args()

    batch, heads, groups, kv_seqlen, dim = args.batch, args.heads, args.groups, args.kv_seqlen, args.dim
    qk_flops = 2 * batch * heads * kv_seqlen * dim
    pv_flops = 2 * batch * heads * kv_seqlen * dim
    total_flops = qk_flops + pv_flops

    selected_blocks = 114
    block_size = 32
    dtype = torch.float16

    program = flashattn(
        batch, heads, groups, kv_seqlen, dim, selected_blocks=selected_blocks, tune=args.tune)(
                block_N=block_size, block_H=64, num_split=6, num_stages=2, threads=128)
    kernel = tilelang.compile(program, out_idx=None, execution_backend="ctypes")
    Q = torch.randn((batch, heads, dim), dtype=dtype, device='cuda').requires_grad_(True)
    K = torch.randn((batch, kv_seqlen, groups, dim), dtype=dtype, device='cuda').requires_grad_(True)
    V = torch.randn((batch, kv_seqlen, groups, dim), dtype=dtype, device='cuda').requires_grad_(True)
    glse = torch.zeros((batch, heads, 1), dtype=dtype, device='cuda')
    Output_partial = torch.zeros((batch, heads, 1, dim), dtype=dtype, device='cuda')
    out = torch.zeros((batch, heads, dim), dtype=dtype, device='cuda')
    block_indices_t = torch.full((batch, groups, selected_blocks), kv_seqlen, dtype=torch.long, device='cuda')
    for b in range(batch):
            for h in range(groups):
                i_i = torch.randperm(max(1, (1// block_size)))[:selected_blocks]
                block_indices_t[b, h, :len(i_i)] = i_i
    block_indices_t = block_indices_t.sort(-1)[0]
    block_counts = torch.randint(1, selected_blocks + 1, (batch, 1, groups), device='cuda')

    block_indices = block_indices_t.to(torch.int32)
 
    #warmp up
    for i in range(10):
        out = torch.zeros((batch, heads, dim), dtype=dtype, device='cuda')
        kernel(Q, K, V, block_indices, glse, Output_partial, out)
        # out = kernel(Q, K, V, block_indices, glse, Output_partial)
    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)
    # start.record()
    torch.cuda.synchronize()
    start = time.time()
    for i in range(100):
        out = torch.empty((batch, heads, dim), dtype=dtype, device='cuda')
        kernel(Q, K, V, block_indices, glse, Output_partial,out)
        # out = kernel(Q, K, V, block_indices, glse, Output_partial)
    # end.record()
    torch.cuda.synchronize()
    # print("sparse time: ",start.elapsed_time(end) / 100)
    print("sparse time: ", (time.time() - start) / 100*1000)

    # warmp up
    for i in range(10):
        ref = ref_program(Q, K, V, block_indices, None, None)
    # start.record()
    torch.cuda.synchronize()
    start = time.time()
    for i in range(100):
        ref = ref_program(Q, K, V, block_indices, None, None)
    # end.record()
    torch.cuda.synchronize()
    # print("dense time: ",start.elapsed_time(end) / 100)
    print("dense time: ", (time.time() - start) / 100*1000)



