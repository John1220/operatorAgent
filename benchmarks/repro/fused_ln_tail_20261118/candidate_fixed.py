"""Fused residual add + LayerNorm (fp16 in/out, fp32 accumulation).

Semantics (per row, over the last dimension N):
    s    = R + X                      (fp32)
    mu   = mean(s)
    var  = mean((s - mu) ** 2)        (biased, like torch F.layer_norm)
    out  = (s - mu) / sqrt(var + eps) (fp16)
"""
import torch
import triton
import triton.language as tl

_EPS = 1e-5


@triton.jit
def _fused_ln_residual_kernel(
    r_ptr, x_ptr, o_ptr,
    r_stride, x_stride, o_stride,
    r_inner, x_inner, o_inner,
    n, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    mask = idx < n

    # 内层 stride 必须乘进去：strided 输入的行内步长不是 1（ABI 明确要求
    # 「read inputs and outputs independently」）。
    r = tl.load(r_ptr + row * r_stride + idx * r_inner,
                mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row * x_stride + idx * x_inner,
                mask=mask, other=0.0).to(tl.float32)
    s = r + x

    mean = tl.sum(s) / n
    # 补齐 lane 的 s 是 0，直接 s-mean 会得到 -mean；必须 mask 掉再算方差，
    # 否则 var 被抬高（N=33 时误差 9.96e-02，first_bad@0）。
    d = tl.where(mask, s - mean, 0.0)
    var = tl.sum(d * d) / n
    rstd = 1.0 / tl.sqrt(var + eps)
    y = d * rstd

    tl.store(o_ptr + row * o_stride + idx * o_inner, y.to(tl.float16), mask=mask)


def kernel(R, X):
    out = torch.empty(R.shape, dtype=R.dtype, device=R.device)
    n = R.shape[-1]
    rows = R.numel() // n
    BLOCK = triton.next_power_of_2(n)
    if BLOCK <= 128:
        num_warps = 1
    elif BLOCK <= 1024:
        num_warps = 4
    else:
        num_warps = 16
    _fused_ln_residual_kernel[(rows,)](
        R, X, out,
        R.stride(-2), X.stride(-2), out.stride(-2),
        R.stride(-1), X.stride(-1), out.stride(-1),
        n, _EPS,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return out
