"""
Author: Yizhao Gao
"""

import torch

import triton
import triton.language as tl

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
    ],
    key=['gqa_group_size', 'BLOCK_H', 'BLOCK_N', 'BLOCK_D', 'BLOCK_V', 'BLOCK_Q', 'is_causal'],
)
@triton.jit
def _fwd_kernel_varlen(
    Q, K, V, Out, L,
    sm_scale,
    cu_seqlens_q,
    cu_seqlens_k,
    block_mask_ptr,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_ot, stride_oh, stride_od,
    stride_lt, stride_lh,
    stride_bmask_z, stride_bmask_h, stride_bmask_m, stride_bmask_n,
    gqa_group_size: tl.constexpr,
    BLOCK_H: tl.constexpr, 
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    is_causal: tl.constexpr = True,
):

    start_t = tl.program_id(0).to(tl.int64) * BLOCK_Q
    off_h_for_kv = tl.program_id(1).to(tl.int64)
    off_h_q = off_h_for_kv * gqa_group_size
    off_z = tl.program_id(2).to(tl.int64)

    cu_q_start = tl.load(cu_seqlens_q + off_z).to(tl.int64)
    cu_q_end = tl.load(cu_seqlens_q + off_z + 1).to(tl.int64)
    seqlen_q = cu_q_end - cu_q_start

    cu_k_start = tl.load(cu_seqlens_k + off_z).to(tl.int64)
    cu_k_end = tl.load(cu_seqlens_k + off_z + 1).to(tl.int64)
    seqlen_k = cu_k_end - cu_k_start

    if start_t >= seqlen_q:
        return
    
    offs_m = tl.arange(0, BLOCK_H) ## head 
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    offs_v = tl.arange(0, BLOCK_V)
    offs_q = tl.arange(0, BLOCK_Q)
    valid_q_mask = start_t + offs_q < seqlen_q


    Q += (cu_q_start + start_t + offs_q) * stride_qt + off_h_q * stride_qh
    K += cu_k_start * stride_kt + off_h_for_kv * stride_kh
    V += cu_k_start * stride_vt + off_h_for_kv * stride_vh
    Out += (cu_q_start + start_t + offs_q) * stride_ot + off_h_q * stride_oh

    q = tl.load(Q[:, None, None] + offs_m[:, None] * stride_qh + offs_d[None, :] * stride_qd,
                mask=(offs_m[:, None] < gqa_group_size) & valid_q_mask[:, None, None]) ## padding to min 16
    q = tl.reshape(q, [BLOCK_Q * BLOCK_H, BLOCK_D])

    block_mask_ptr += off_z * stride_bmask_z + off_h_for_kv * stride_bmask_h + (start_t + offs_q) * stride_bmask_m


    k_block_start = 0
    k_block_end = tl.cdiv(start_t + BLOCK_Q + seqlen_k - seqlen_q, BLOCK_N)

    m_i = tl.full([BLOCK_Q * BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_Q * BLOCK_H], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q * BLOCK_H, BLOCK_V], dtype=tl.float32)

    k_ptrs = K + offs_n[None, :] * stride_kt + offs_d[:, None] * stride_kd
    v_ptrs = V + offs_n[:, None] * stride_vt + offs_v[None, :] * stride_vd


    for k_block_col_idx in range(k_block_start, k_block_end):
        mask_val = tl.load(block_mask_ptr + k_block_col_idx * stride_bmask_n, mask=valid_q_mask)
        if tl.sum(mask_val.to(tl.float32)) > 0:
            start_n = k_block_col_idx * BLOCK_N
            k = tl.load(k_ptrs + start_n * stride_kt, mask=offs_n[None, :] + start_n < seqlen_k)
            qk = tl.dot(q, k)
            qk *= sm_scale
            if is_causal:
                causal_mask = start_t + offs_q[:, None, None] + seqlen_k - seqlen_q >= (start_n + offs_n[None, None, :])
                causal_mask = causal_mask & mask_val[:, None, None]
                qk = tl.reshape(qk, [BLOCK_Q, BLOCK_H, BLOCK_N])
                qk = tl.where(causal_mask, qk, -1.0e6)
                qk = tl.reshape(qk, [BLOCK_Q * BLOCK_H, BLOCK_N])
            else:
                qk = tl.reshape(qk, [BLOCK_Q, BLOCK_H, BLOCK_N])
                qk = tl.where(mask_val[:, None, None], qk, -1.0e6)
                qk = tl.reshape(qk, [BLOCK_Q * BLOCK_H, BLOCK_N])

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
            p = tl.exp(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            
            v = tl.load(v_ptrs + start_n * stride_vt, mask=offs_n[:, None] + start_n < seqlen_k)
            p = p.to(v.type.element_ty)

            acc += tl.dot(p, v)
            m_i = m_ij

    m_i += tl.math.log(l_i)
    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip
    acc = acc.to(Out.dtype.element_ty) 

    l_ptrs = L + (off_h_q + offs_m) * stride_lh + (cu_q_start + start_t + offs_q[:, None]) * stride_lt
    m_i = tl.reshape(m_i, [BLOCK_Q, BLOCK_H])
    tl.store(l_ptrs, m_i, mask=(offs_m < gqa_group_size) & valid_q_mask[:, None])

    O_ptrs = Out[:, None, None] + offs_m[:, None] * stride_oh + offs_v[None, :] * stride_od
    acc = tl.reshape(acc, [BLOCK_Q, BLOCK_H, BLOCK_V])
    tl.store(O_ptrs, acc, mask=(offs_m[:, None] < gqa_group_size) & valid_q_mask[:, None, None])



def blocksparse_flash_attn_varlen_fwd(
    q, k, v, # (#tokens, n_heads, key_dim or head_dim)
    cu_seqlens_q,
    cu_seqlens_k,
    sm_scale,
    block_mask,
    block_size=64,
    is_causal=True,
):
    # split q to blocks
    _, n_heads, key_dim = q.shape
    _, n_kv_heads, head_dim = v.shape

    batch = cu_seqlens_k.size(0) - 1

    assert q.dim() == k.dim() == v.dim() == 3
    assert q.size(2) == k.size(2)
    assert k.size(1) == v.size(1)
    assert cu_seqlens_k.dim() == 1
    assert cu_seqlens_k.size(0) == cu_seqlens_q.size(0)
    assert key_dim in {64, 128, 256}
    assert head_dim in {64, 128, 256}
    assert triton.next_power_of_2(block_size) == block_size, "block size must be power of 2"

    k_lens = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).cpu()
    q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).cpu()
    max_seqlen = q_lens.max()

    gqa_group_size = n_heads // n_kv_heads
    
    out = torch.empty((q.shape[0], n_heads, head_dim), device=q.device, dtype=q.dtype)
    cu_seqlens_q = cu_seqlens_q.contiguous()
    cu_seqlens_k = cu_seqlens_k.contiguous()

    if is_hip():
        extra_kern_args = {"waves_per_eu": 1}
    else:
        extra_kern_args = {}

    # block_h = gqa_group_size if gqa_group_size > 16 else 16
    # block_q = 1
    block_h = gqa_group_size
    block_q = max(64 // block_h, 1)
    
    grid = lambda META: (triton.cdiv(max_seqlen, block_q), n_kv_heads, batch)

    L = torch.empty((q.shape[0], n_heads), device=q.device, dtype=torch.float32)

    with torch.cuda.device(q.device.index): 
        _fwd_kernel_varlen[grid](
            q, k, v, out, L,
            sm_scale,
            cu_seqlens_q,
            cu_seqlens_k,
            block_mask,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *out.stride(),
            *L.stride(),
            *block_mask.stride(),
            gqa_group_size,
            BLOCK_H = block_h,
            BLOCK_N = block_size,
            BLOCK_D = key_dim,
            BLOCK_V = head_dim,
            BLOCK_Q = block_q,
            is_causal = is_causal,
            **extra_kern_args
        )

    return out, L

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
    ],
    key=['BLOCK_M', 'BLOCK_DMODEL', 'N_CTX_Q', 'H'],
)
@triton.jit
def _bwd_preprocess_use_o(
    Out,
    DO,
    Delta,
    stride_om, stride_oh, stride_ok,
    stride_dom, stride_doh, stride_dok,
    stride_deltam, stride_deltah, 
    cu_seqlens_q,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    N_CTX_Q: tl.constexpr,
    H: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_bh = tl.program_id(1).to(tl.int64)

    # Compute batch and head indices
    off_z = pid_bh // H
    off_h = pid_bh % H


    q_start = tl.load(cu_seqlens_q + off_z).to(tl.int64)
    q_end = tl.load(cu_seqlens_q + off_z + 1).to(tl.int64)

    # Compute actual sequence lengths
    N_CTX_Q = q_end - q_start

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, BLOCK_DMODEL)

    mask_m = off_m < N_CTX_Q

    if pid_m * BLOCK_M >= N_CTX_Q:
        return


    o_offset = Out + off_h * stride_oh + q_start * stride_om
    do_offset = DO + off_h * stride_oh + q_start * stride_om

    out_ptrs = o_offset + off_m[:, None] * stride_om + off_d[None, :] * stride_ok
    do_ptrs = do_offset + off_m[:, None] * stride_dom + off_d[None, :] * stride_dok

    # load
    o = tl.load(out_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

    delta = tl.sum(o * do, axis=1)

    delta_offset = Delta + off_h * stride_deltah + q_start * stride_deltam
    delta_ptrs = delta_offset + off_m * stride_deltam
    tl.store(delta_ptrs, delta, mask=mask_m)

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
    ],
    key=['gqa_group_size', 'BLOCK_H', 'BLOCK_N', 'BLOCK_Q', 'BLOCK_K', 'BLOCK_DMODEL', 'is_causal'],
)
@triton.jit
def _bwd_dkdv(
    Q, K, V, sm_scale,
    block_mask_ptr,
    DO, DK, DV,
    L, D,
    stride_qm, stride_qh, stride_qd,
    stride_kn, stride_kh, stride_kd,
    stride_vn, stride_vh, stride_vd,
    stride_om, stride_oh, stride_od,
    stride_lm, stride_lh,
    stride_deltam, stride_deltah, 
    stride_bmask_z, stride_bmask_h, stride_bmask_m, stride_bmask_n,
    cu_seqlens_q,  
    cu_seqlens_k,
    num_kv_heads: tl.constexpr,
    gqa_group_size: tl.constexpr,
    BLOCK_H: tl.constexpr, 
    BLOCK_N: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    is_causal: tl.constexpr = True,
):
    start_n = tl.program_id(0).to(tl.int64)
    off_hz = tl.program_id(1).to(tl.int64)
    off_z = off_hz // num_kv_heads
    off_h_for_kv = off_hz % num_kv_heads
    off_h_q = off_h_for_kv * gqa_group_size

    q_start = tl.load(cu_seqlens_q + off_z).to(tl.int64)
    q_end = tl.load(cu_seqlens_q + off_z + 1).to(tl.int64)
    k_start = tl.load(cu_seqlens_k + off_z).to(tl.int64)
    k_end = tl.load(cu_seqlens_k + off_z + 1).to(tl.int64)

    # Compute actual sequence lengths
    seqlen_k = k_end - k_start
    seqlen_q = q_end - q_start

    if start_n * BLOCK_N >= seqlen_k:
        return


    q_offset = Q + off_h_q * stride_qh + q_start * stride_qm
    k_offset = K + off_h_for_kv * stride_kh + k_start * stride_kn
    v_offset = V + off_h_for_kv * stride_vh + k_start * stride_vn
    do_offset = DO + off_h_q * stride_oh + q_start * stride_om
    l_offset = L + off_h_q * stride_lh + q_start * stride_lm
    d_offset = D + off_h_q * stride_deltah + q_start * stride_deltam


    dk_offset = DK + off_h_for_kv * stride_kh + k_start * stride_kn
    dv_offset = DV + off_h_for_kv * stride_vh + k_start * stride_vn

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_h = tl.arange(0, BLOCK_H)

    mask_n = offs_n < seqlen_k
    mask_h = offs_h < gqa_group_size
    kv_mask = mask_n[:, None]

    # initialize pointers to value-like data
    k_ptrs = k_offset + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kd)
    v_ptrs = v_offset + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)

    # initialize dv and dk
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)
    # k and v stay in SRAM throughout
    k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
    v = tl.load(v_ptrs, mask=kv_mask, other=0.0)
    # loop over rows

    block_mask_ptr += off_z * stride_bmask_z + off_h_for_kv * stride_bmask_h + start_n * stride_bmask_n
    start_q = max(start_n * BLOCK_N - (seqlen_k - seqlen_q), 0)

    for start_m in range(start_q, seqlen_q, BLOCK_Q):
        block_q_ptr = start_m + tl.arange(0, BLOCK_Q)
        valid_q = block_q_ptr < seqlen_q
        mask_val = tl.load(block_mask_ptr + block_q_ptr * stride_bmask_m, mask=valid_q, other=0)
        if tl.sum(mask_val.to(tl.float32)) > 0:
            q_ptrs = q_offset + (block_q_ptr[:, None, None] * stride_qm + offs_h[:, None] * stride_qh + offs_k[None, :] * stride_qd)
            do_ptrs = do_offset + (block_q_ptr[:, None, None] * stride_om + offs_h[:, None] * stride_oh + offs_d[None, :] * stride_od)

            q = tl.load(q_ptrs, mask=mask_h[:, None] & valid_q[:, None, None], other=0.0)
            do = tl.load(do_ptrs, mask=mask_h[:, None] & valid_q[:, None, None], other=0.0)
            
            q = tl.reshape(q, [BLOCK_Q * BLOCK_H, BLOCK_K])
            do = tl.reshape(do, [BLOCK_Q * BLOCK_H, BLOCK_DMODEL])

            qk = tl.zeros([BLOCK_Q * BLOCK_H, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, tl.trans(k))
            
            if is_causal:
                causal_mask = block_q_ptr[:, None, None] + seqlen_k - seqlen_q >= (offs_n[None, :])
                causal_mask = causal_mask & mask_val[:, None, None]
                qk = tl.reshape(qk, [BLOCK_Q, BLOCK_H, BLOCK_N])
                qk += tl.where(causal_mask, 0, -1.0e6)
                qk = tl.reshape(qk, [BLOCK_Q * BLOCK_H, BLOCK_N])
            else:
                qk = tl.reshape(qk, [BLOCK_Q, BLOCK_H, BLOCK_N])
                qk += tl.where(mask_val[:, None, None], 0, -1.0e6)
                qk = tl.reshape(qk, [BLOCK_Q * BLOCK_H, BLOCK_N])
                
            l_i = tl.load(l_offset + block_q_ptr[:, None] * stride_lm + offs_h * stride_lh, mask=mask_h & valid_q[:, None])
            l_i = tl.reshape(l_i, [BLOCK_Q * BLOCK_H])
            p = tl.exp(qk * sm_scale - l_i[:, None])
            
            p_mask = mask_n[None, :]
            p = tl.where(p_mask, p, 0.0)

            dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)
            dp = tl.dot(do, tl.trans(v))


            d_ptrs = d_offset + block_q_ptr[:, None] * stride_deltam + offs_h * stride_deltah
            Di = tl.load(d_ptrs, mask=mask_h & valid_q[:, None])
            Di = tl.reshape(Di, [BLOCK_Q * BLOCK_H])
            ds = (p * (dp - Di[:, None])) * sm_scale
            ds = tl.where(p_mask, ds, 0.0).to(Q.dtype.element_ty)

            dk += tl.dot(tl.trans(ds), q)

    # write-back
    dk_ptrs = dk_offset + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kd)
    dv_ptrs = dv_offset + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
    tl.store(dk_ptrs, dk.to(K.dtype.element_ty), mask=kv_mask)
    tl.store(dv_ptrs, dv.to(V.dtype.element_ty), mask=kv_mask)



@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
    ],
    key=['gqa_group_size', 'BLOCK_H', 'BLOCK_N', 'BLOCK_Q', 'BLOCK_K', 'BLOCK_DMODEL', 'is_causal'],
)
@triton.jit
def _bwd_dq(
    Q, K, V, sm_scale,
    block_mask_ptr,
    DO, DQ,
    L, D,
    stride_qm, stride_qh, stride_qd,
    stride_kn, stride_kh, stride_kd,
    stride_vn, stride_vh, stride_vd,
    stride_om, stride_oh, stride_od,
    stride_lm, stride_lh,
    stride_deltam, stride_deltah, 
    stride_bmask_z, stride_bmask_h, stride_bmask_m, stride_bmask_n,
    cu_seqlens_q,  
    cu_seqlens_k,
    num_kv_heads: tl.constexpr,
    gqa_group_size: tl.constexpr,
    BLOCK_H: tl.constexpr, 
    BLOCK_N: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    is_causal: tl.constexpr = True,
):


    start_m = tl.program_id(0).to(tl.int64) * BLOCK_Q
    off_hz = tl.program_id(1).to(tl.int64)
    off_z = off_hz // num_kv_heads
    off_h_for_kv = off_hz % num_kv_heads
    off_h_q = off_h_for_kv * gqa_group_size

    q_start = tl.load(cu_seqlens_q + off_z).to(tl.int64)
    q_end = tl.load(cu_seqlens_q + off_z + 1).to(tl.int64)
    k_start = tl.load(cu_seqlens_k + off_z).to(tl.int64)
    k_end = tl.load(cu_seqlens_k + off_z + 1).to(tl.int64)
    seqlen_k = k_end - k_start

    seqlen_q = q_end - q_start
    offs_h = tl.arange(0, BLOCK_H)  
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_k = tl.arange(0, BLOCK_K)
    offs_q = tl.arange(0, BLOCK_Q)
    valid_q_mask = start_m + offs_q < seqlen_q

    if start_m >= seqlen_q:
        return

    mask_h = offs_h < gqa_group_size
    q_offset = Q + (off_h_q + offs_h[:, None]) * stride_qh + (q_start + start_m + offs_q[:, None, None]) * stride_qm + offs_k[None, :] * stride_qd
    do_offset = DO + (off_h_q + offs_h[:, None]) * stride_oh + (q_start + start_m + offs_q[:, None, None]) * stride_om + offs_d[None, :] * stride_od
    l_offset = L + (off_h_q + offs_h) * stride_lh + (q_start + start_m + offs_q[:, None]) * stride_lm 
    d_offset = D + (off_h_q + offs_h) * stride_deltah + (q_start + start_m + offs_q[:, None]) * stride_deltam 

    do = tl.load(do_offset, mask=mask_h[:, None] & valid_q_mask[:, None, None], other=0.0)
    l_i = tl.load(l_offset, mask=mask_h & valid_q_mask[:, None], other=0.0)
    Di = tl.load(d_offset, mask=mask_h & valid_q_mask[:, None], other=0.0)
    q = tl.load(q_offset, mask=mask_h[:, None] & valid_q_mask[:, None, None], other=0.0)
    do = tl.reshape(do, [BLOCK_Q * BLOCK_H, BLOCK_DMODEL])
    l_i = tl.reshape(l_i, [BLOCK_Q * BLOCK_H])
    Di = tl.reshape(Di, [BLOCK_Q * BLOCK_H])
    q = tl.reshape(q, [BLOCK_Q * BLOCK_H, BLOCK_K])
    dq = tl.zeros([BLOCK_Q * BLOCK_H, BLOCK_K], dtype=tl.float32)  

    block_mask_ptr += off_z * stride_bmask_z + off_h_for_kv * stride_bmask_h + (start_m + offs_q) * stride_bmask_m

    start_l = 0
    end_l = tl.cdiv(start_m + BLOCK_Q + seqlen_k - seqlen_q, BLOCK_N)

    for col_idx in range(start_l, end_l):
        mask_val = tl.load(block_mask_ptr + col_idx * stride_bmask_n, mask=valid_q_mask, other=0)
        if tl.sum(mask_val.to(tl.float32)) > 0:
            start_n = col_idx * BLOCK_N
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < seqlen_k

            k_ptrs = K + off_h_for_kv * stride_kh + (k_start + offs_n[:, None]) * stride_kn + offs_k[None, :] * stride_kd
            v_ptrs = V + off_h_for_kv * stride_vh + (k_start + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vd
            k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

            qk = tl.zeros([BLOCK_Q * BLOCK_H, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, tl.trans(k))
            
            if is_causal:
                qk = tl.reshape(qk, [BLOCK_Q, BLOCK_H, BLOCK_N])
                causal_mask = start_m + offs_q[:, None, None] + seqlen_k - seqlen_q >= (offs_n[None, None, :])
                causal_mask = causal_mask & mask_val[:, None, None]
                qk += tl.where(causal_mask, 0, -1.0e6)
                qk = tl.reshape(qk, [BLOCK_Q * BLOCK_H, BLOCK_N])
            else:
                qk = tl.reshape(qk, [BLOCK_Q, BLOCK_H, BLOCK_N])
                qk += tl.where(mask_val[:, None, None], 0, -1.0e6)
                qk = tl.reshape(qk, [BLOCK_Q * BLOCK_H, BLOCK_N])
                
                
            p = tl.exp(qk * sm_scale - l_i[:, None])
            
            p_mask = mask_n[None, :]
            p = tl.where(p_mask, p, 0.0)

            dp = tl.dot(do, v.T).to(tl.float32)

            ds = (p * (dp - Di[:, None])) * sm_scale
            ds = tl.where(p_mask, ds, 0.0).to(Q.dtype.element_ty)

            dq += tl.dot(ds, k)  

    dq_offset = DQ + (off_h_q + offs_h[:, None]) * stride_qh + (q_start + start_m + offs_q[:, None, None]) * stride_qm + offs_k[None, :] * stride_qd
    dq = tl.reshape(dq, [BLOCK_Q, BLOCK_H, BLOCK_K])
    tl.store(dq_offset, dq, mask=mask_h[:, None] & valid_q_mask[:, None, None])


def blocksparse_flash_attn_varlen_bwd(
    do, q, k, v, o, L, 
    cu_seqlens_q,
    cu_seqlens_k,
    sm_scale,
    block_mask,
    block_size=64,
    is_causal=True,
):
    do = do.contiguous()
    L = L.contiguous()
    assert q.dim() == k.dim() == v.dim() == 3

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    o = o.contiguous()

    cu_seqlens_q = cu_seqlens_q.contiguous()
    cu_seqlens_k = cu_seqlens_k.contiguous()
    q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).cpu()
    k_lens = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).cpu()
    max_seqlen_q = q_lens.max()
    max_seqlen_k = k_lens.max()

    num_block = triton.cdiv(max_seqlen_k, block_size)
    num_q_heads = q.size(1)
    num_kv_heads = k.size(1)
    gqa_group_size = num_q_heads // num_kv_heads
    key_dim, head_dim = q.size(-1), v.size(-1)


    dq = torch.zeros_like(q)
    dk = torch.empty_like(k) 
    dv = torch.empty_like(v)

    delta = torch.empty_like(L)

    batch = cu_seqlens_k.size(0) - 1
    batch_q_head_size = batch * num_q_heads
    batch_kv_head_size = batch * num_kv_heads

    if is_hip():
        extra_kern_args = {"waves_per_eu": 1}
    else:
        extra_kern_args = {}

    block_mask = block_mask.contiguous()
    
    block_h = gqa_group_size
    block_q = max(64 // block_h, 1)

    with torch.cuda.device(q.device.index): 
        _bwd_preprocess_use_o[(triton.cdiv(max_seqlen_q, block_size), batch_q_head_size)](
            o, do, delta,
            *o.stride(),
            *do.stride(),
            *delta.stride(),
            cu_seqlens_q,
            BLOCK_M = block_size,
            BLOCK_DMODEL = head_dim,
            N_CTX_Q = q.size(0),
            H = num_q_heads,
        )


        _bwd_dkdv[(num_block, batch_kv_head_size)](
            q, k, v, sm_scale,
            block_mask,
            do, dk, dv,
            L, delta,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *do.stride(),
            *L.stride(),
            *delta.stride(),
            *block_mask.stride(),
            cu_seqlens_q,
            cu_seqlens_k,
            num_kv_heads,
            gqa_group_size,
            BLOCK_H = block_h,
            BLOCK_N = block_size,
            BLOCK_Q = block_q,
            BLOCK_K = key_dim,
            BLOCK_DMODEL = head_dim,
            is_causal = is_causal,
            **extra_kern_args
        )

        _bwd_dq[(triton.cdiv(max_seqlen_q, block_q), batch_kv_head_size)](
            q, k, v, sm_scale,
            block_mask,
            do, dq,
            L, delta,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *do.stride(),
            *L.stride(),
            *delta.stride(),
            *block_mask.stride(),
            cu_seqlens_q,
            cu_seqlens_k,
            num_kv_heads,
            gqa_group_size,
            BLOCK_H = block_h,
            BLOCK_N = block_size,
            BLOCK_Q = block_q,
            BLOCK_K = key_dim,
            BLOCK_DMODEL = head_dim,
            is_causal = is_causal,
            **extra_kern_args
        )

    return dq, dk, dv

class _block_sparse_attn_varlen(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens_q, cu_seqlens_k, dense_block_mask, sm_scale=None, block_size=64, is_causal=True):
        sm_scale = sm_scale if sm_scale is not None else 1.0 / q.shape[-1] ** 0.5
        out, L = blocksparse_flash_attn_varlen_fwd(q, k, v, cu_seqlens_q, cu_seqlens_k, sm_scale, dense_block_mask, block_size=block_size, is_causal=is_causal)
        ctx.save_for_backward(out, L, q, k, v, cu_seqlens_q, cu_seqlens_k, dense_block_mask)
        ctx.sm_scale = sm_scale
        ctx.block_size = block_size
        ctx.is_causal = is_causal
        return out

    @staticmethod
    def backward(ctx, do):
        out, L, q, k, v, cu_seqlens_q, cu_seqlens_k, dense_block_mask = ctx.saved_tensors
        dq, dk, dv = blocksparse_flash_attn_varlen_bwd(
            do, 
            q, 
            k, 
            v, 
            out, 
            L, 
            cu_seqlens_q, 
            cu_seqlens_k, 
            ctx.sm_scale, 
            dense_block_mask,
            ctx.block_size,
            ctx.is_causal
        )
        return dq, dk, dv, None, None, None, None, None, None


def improved_block_sparse_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, sparse_mask, sm_scale=None, block_size=64):
    return  _block_sparse_attn_varlen.apply(q, k, v, cu_seqlens_q, cu_seqlens_k, sparse_mask, sm_scale, block_size)

def main():
    from torch.nn import functional as F
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    from sparse_mask import fused_sparse_mask
    from sparse_attention import block_sparse_attn_varlen_func
    from dense_sparse_attention import dense_sparse_attn_varlen_func
    import time
    torch.cuda.manual_seed(0)
    bsz, n_head, n_seq, key_dim = 1, 4, 4096, 256
    n_kv_seq = 7653
    head_dim = 128
    gqa_size = 8
    block_size = 32
    sparse_ratio = 0.1
    is_causal = True
    dtype = torch.bfloat16
    xq = torch.randn((bsz, n_seq, n_head * gqa_size, key_dim), device='cuda', dtype=dtype)
    xk = torch.randn((bsz, n_kv_seq, n_head, key_dim), device='cuda', dtype=dtype)
    xv = torch.randn((bsz, n_kv_seq, n_head, head_dim), device='cuda', dtype=dtype)
    xq.requires_grad = True
    xk.requires_grad = True
    xv.requires_grad = True
    xq_triton = xq.detach().clone()
    xk_triton = xk.detach().clone()
    xv_triton = xv.detach().clone()
    xq_triton.requires_grad = True
    xk_triton.requires_grad = True
    xv_triton.requires_grad = True
    cu_seqlens_q = torch.tensor([0, 1077, 2048, 3456, 4096], device='cuda', dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, 1344, 2688, 4432, 7653], device='cuda', dtype=torch.int32)

    assert ((cu_seqlens_q[1:] - cu_seqlens_q[:-1]) <= (cu_seqlens_k[1:] - cu_seqlens_k[:-1])).all().item(), "cu_seqlens_q must be less than or equal to cu_seqlens_k"
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(100):
        sparse_mask = fused_sparse_mask(xq_triton[0], xk_triton[0], cu_seqlens_q, cu_seqlens_k, sparse_ratio=sparse_ratio, block_size=block_size, local_block_num=2)   
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"Sparse Mask Time taken: {end_time - start_time} seconds")
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(100):
        xq_triton.grad, xk_triton.grad, xv_triton.grad = None, None, None
        triton_output = improved_block_sparse_attn_varlen_func(xq_triton[0], xk_triton[0], xv_triton[0], cu_seqlens_q, cu_seqlens_k, sparse_mask, block_size=block_size)
        triton_output.sum().backward()
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"Triton Time taken: {end_time - start_time} seconds")
    
    naive_mask = torch.zeros((bsz, n_head, n_seq, n_kv_seq), device=xq.device, dtype=torch.bool)
    _, _, n_head, max_block_k = sparse_mask.shape
    for i in range(len(cu_seqlens_q) - 1):
        start_q, end_q = cu_seqlens_q[i], cu_seqlens_q[i + 1]
        start_k, end_k = cu_seqlens_k[i], cu_seqlens_k[i + 1]
        block_mask = sparse_mask[i, :, :end_q - start_q].repeat_interleave(block_size, dim=-1)
        naive_mask[:, :, start_q:end_q, start_k:end_k] = torch.tril(block_mask[:, :, :end_k - start_k], diagonal=(end_k - start_k) - (end_q - start_q))
    
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(100):
        xq.grad, xk.grad, xv.grad = None, None, None
        output = F.scaled_dot_product_attention(xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2), attn_mask=naive_mask.repeat_interleave(gqa_size, dim=1), enable_gqa=True).transpose(1, 2)[0]
        output.sum().backward()
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"Torch SDPA Time taken: {end_time - start_time} seconds")
    print(output.shape, triton_output.shape)
    print((output - triton_output).abs().max(), (output - triton_output).abs().mean())
    print((xq.grad - xq_triton.grad).abs().max(), (xq.grad - xq_triton.grad).abs().mean())
    print((xk.grad - xk_triton.grad).abs().max(), (xk.grad - xk_triton.grad).abs().mean())
    print((xv.grad - xv_triton.grad).abs().max(), (xv.grad - xv_triton.grad).abs().mean())
    
if __name__ == "__main__":
    main()