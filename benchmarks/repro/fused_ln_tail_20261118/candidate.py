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
    n, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    mask = idx < n

    r = tl.load(r_ptr + row * r_stride + idx, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row * x_stride + idx, mask=mask, other=0.0).to(tl.float32)
    s = r + x

    mean = tl.sum(s) / n
    d = s - mean
    var = tl.sum(d * d) / n
    rstd = 1.0 / tl.sqrt(var + eps)
    y = d * rstd

    tl.store(o_ptr + row * o_stride + idx, y.to(tl.float16), mask=mask)


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
        n, _EPS,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return out
