import math

import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,
                    K, V,
                    stride_kb, stride_kh, stride_kn, stride_kk,
                    stride_vb, stride_vh, stride_vn, stride_vk,
                    off_b, off_h, N, start_m, qk_scale,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    HEAD_DIM: tl.constexpr, D: tl.constexpr, STAGE: tl.constexpr,
                    EVEN_N: tl.constexpr):
    """Flash Attention 内层：K/V 分块 + 在线 softmax（m_i/l_i 递推）。
    STAGE=1 下三角外整块(免 mask 快路径)；STAGE=2 对角带(causal mask)；STAGE=3 全量(non-causal)。
    [B,H,N,D] fp16 输入，QK^T 与 PV 全部 fp32 累加。
    """
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    d_mask = offs_d < D

    # K/V 的 batch/head 基址：stride_kn 是 n 行方向元素跨度，stride_kk 是 head_dim 方向元素跨度
    k_base = K + off_b * stride_kb + off_h * stride_kh
    v_base = V + off_b * stride_vb + off_h * stride_vh

    # 循环区间：causal 拆成 [0, start_m*BLOCK_M) 与对角带；non-causal 覆盖全序列
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, tl.minimum((start_m + 1) * BLOCK_M, N)
    else:
        lo, hi = 0, N

    k_ptrs = k_base + (lo + offs_n)[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = v_base + (lo + offs_n)[:, None] * stride_vn + offs_d[None, :] * stride_vk

    for start_n in range(lo, hi, BLOCK_N):
        if STAGE == 1 or EVEN_N:
            # n 方向整体在界内（STAGE=1 恒成立；EVEN_N 表示 N 是 BLOCK_N 的整数倍）
            k = tl.load(k_ptrs, mask=d_mask[None, :], other=0.0)
            v = tl.load(v_ptrs, mask=d_mask[None, :], other=0.0)
        else:
            n_mask = (start_n + offs_n) < N  # 非整除尾部（如 N=1000）
            k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        # QK^T：Tensor Core，fp32 累加，再在 fp32 上乘 scale（保持精度）
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        if STAGE == 2:
            # causal mask：只保留 n <= m
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], qk, float("-inf"))
        # 在线 softmax：m_i/l_i 递推（exp2 底数，qk_scale 已含 log2(e)）
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        # PV：P 降 fp16 走 Tensor Core，累加器保持 fp32
        acc = tl.dot(p.to(tl.float16), v, acc)
        m_i = m_new
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
    return acc, l_i, m_i


@triton.jit
def _attn_fwd(Q, K, V, Out,
              stride_qb, stride_qh, stride_qm, stride_qk,
              stride_kb, stride_kh, stride_kn, stride_kk,
              stride_vb, stride_vh, stride_vn, stride_vk,
              stride_ob, stride_oh, stride_om, stride_on,
              H, N, sm_scale,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
              HEAD_DIM: tl.constexpr, D: tl.constexpr, IS_CAUSAL: tl.constexpr,
              EVEN_M: tl.constexpr, EVEN_N: tl.constexpr):
    """Flash Attention 前向 (prefill MHA)：Q/K/V [B,H,N,D] fp16 -> O [B,H,N,D] fp16。
    grid = (cdiv(N, BLOCK_M), B*H)，每个 program 处理 BLOCK_M 行 query。
    """
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // H
    off_h = off_hb % H

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    d_mask = offs_d < D

    # Q tile 一次载入：[BLOCK_M, HEAD_DIM]
    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh \
        + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    if EVEN_M:
        q = tl.load(q_ptrs, mask=d_mask[None, :], other=0.0)
    else:
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < N) & d_mask[None, :], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    qk_scale = sm_scale * 1.4426950408889634

    if IS_CAUSAL:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K, V,
                                        stride_kb, stride_kh, stride_kn, stride_kk,
                                        stride_vb, stride_vh, stride_vn, stride_vk,
                                        off_b, off_h, N, start_m, qk_scale,
                                        BLOCK_M, BLOCK_N, HEAD_DIM, D, 1, EVEN_N)
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K, V,
                                        stride_kb, stride_kh, stride_kn, stride_kk,
                                        stride_vb, stride_vh, stride_vn, stride_vk,
                                        off_b, off_h, N, start_m, qk_scale,
                                        BLOCK_M, BLOCK_N, HEAD_DIM, D, 2, EVEN_N)
    else:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K, V,
                                        stride_kb, stride_kh, stride_kn, stride_kk,
                                        stride_vb, stride_vh, stride_vn, stride_vk,
                                        off_b, off_h, N, start_m, qk_scale,
                                        BLOCK_M, BLOCK_N, HEAD_DIM, D, 3, EVEN_N)

    acc = acc / l_i[:, None]  # 按有效长度归一化
    out_ptrs = Out + off_b * stride_ob + off_h * stride_oh \
        + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    if EVEN_M:
        tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=d_mask[None, :])
    else:
        tl.store(out_ptrs, acc.to(Out.dtype.element_ty),
                 mask=(offs_m[:, None] < N) & d_mask[None, :])


def kernel(K, Q, V, causal):
    """keyword-wrapper-v1：输出由 wrapper 内部分配并返回。"""
    assert Q.dim() == 4 and K.shape == Q.shape and V.shape == Q.shape
    B, H, N, D = Q.shape
    out = torch.empty((B, H, N, D), dtype=Q.dtype, device=Q.device)
    HEAD_DIM = triton.next_power_of_2(D)
    BLOCK_M = 128
    BLOCK_N = 128
    grid = (triton.cdiv(N, BLOCK_M), B * H)
    _attn_fwd[grid](
        Q, K, V, out,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        H, N, 1.0 / math.sqrt(D),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM, D=D,
        IS_CAUSAL=bool(causal),
        EVEN_M=(N % BLOCK_M == 0), EVEN_N=(N % BLOCK_N == 0),
        num_warps=8, num_stages=2,
    )
    return out