"""Fused residual-add + LayerNorm.

Semantics (frozen operator "builtin-fused-layernorm-residual-fp16-cuda-v1"):
    s   = R + X                      (accumulated in fp32)
    mu  = mean(s, dim=-1)            (fp32)
    var = biased_var(s, dim=-1)      (fp32)
    out = (s - mu) / sqrt(var + eps) (fp16 output, eps = 1e-5)

The operator body is a hand-written Triton kernel. Only tensor
infrastructure (torch.empty, .shape/.stride) is used from torch.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_ln_residual_kernel(
    R_ptr, X_ptr, O_ptr,
    N,
    stride_r0, stride_x0,
    eps,
    BLOCK: tl.constexpr,
    NBLK: tl.constexpr,
    EXACT: tl.constexpr,
    R_INNER: tl.constexpr,
    X_INNER: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, NBLK)[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]

    if R_INNER == 1:
        r_addr = R_ptr + row * stride_r0 + cols
    else:
        r_addr = R_ptr + row * stride_r0 + cols * R_INNER
    if X_INNER == 1:
        x_addr = X_ptr + row * stride_x0 + cols
    else:
        x_addr = X_ptr + row * stride_x0 + cols * X_INNER

    if EXACT:
        r = tl.load(r_addr).to(tl.float32)
        x = tl.load(x_addr).to(tl.float32)
    else:
        mask = cols < N
        r = tl.load(r_addr, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_addr, mask=mask, other=0.0).to(tl.float32)

    s = r + x
    mean = tl.sum(s) / N
    # 补齐的 lane 必须**排除在中心化之后**：s 在 pad 区是 0，直接 s-mean 会得到
    # -mean，把它们累进方差就会把 var 抬大（N=33 时甚至整块判否）。
    diff = tl.where(cols < N, s - mean, 0.0)
    var = tl.sum(diff * diff) / N
    out = diff / tl.sqrt(var + eps)

    o_addr = O_ptr + row * N + cols
    if EXACT:
        tl.store(o_addr, out.to(tl.float16))
    else:
        tl.store(o_addr, out.to(tl.float16), mask=mask)


def kernel(R, X):
    M, N = R.shape
    out = torch.empty((M, N), dtype=torch.float16, device=R.device)
    if N <= 1024:
        BLOCK = max(1, triton.next_power_of_2(N))
        NBLK = 1
    else:
        BLOCK = 1024
        # tl.arange() 的 bound 必须是 2 的幂：N=6144 时 cdiv=6，直接传会编译失败。
        NBLK = triton.next_power_of_2(triton.cdiv(N, BLOCK))
    EXACT = (N % BLOCK) == 0 and NBLK == triton.cdiv(N, BLOCK)
    num_warps = 8 if N >= 4096 else 4
    _fused_ln_residual_kernel[(M,)](
        R, X, out,
        N,
        R.stride(0), X.stride(0),
        1e-5,
        BLOCK=BLOCK, NBLK=NBLK, EXACT=EXACT,
        R_INNER=R.stride(1), X_INNER=X.stride(1),
        num_warps=num_warps,
    )
    return out
