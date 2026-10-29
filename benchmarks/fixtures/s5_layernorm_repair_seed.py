import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_kernel(X, Y, stride_xm, stride_xn, stride_ym, stride_yn,
                      n_cols, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    x = tl.load(X + row * stride_xm + cols * stride_xn,
                mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / n_cols
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0)
    y = xc / tl.sqrt(var + eps)
    tl.store(Y + row * stride_ym + cols * stride_yn, y, mask=mask)


def kernel(X, eps=1e-5):
    M, N = X.shape
    Y = torch.empty_like(X)
    _layernorm_kernel[(M,)](
        X, Y, X.stride(0), X.stride(1), Y.stride(0), Y.stride(1),
        N, float(eps), BLOCK_N=triton.next_power_of_2(max(N, 16)), num_warps=1)
    return Y
