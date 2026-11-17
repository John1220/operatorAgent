import torch
import triton
import triton.language as tl


@triton.jit
def _fused_ln_residual_kernel(x_ptr, r_ptr, y_ptr, N, eps,
                              sxm, sxn, srm, srn, sym, syn, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row * sxm + cols * sxn, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + row * srm + cols * srn, mask=mask, other=0.0).to(tl.float32)
    s = x + r
    mean = tl.sum(s, axis=0) / N
    centred = tl.where(mask, s - mean, 0.0)
    var = tl.sum(centred * centred, axis=0) / N
    y = centred * tl.rsqrt(var + eps)
    tl.store(y_ptr + row * sym + cols * syn, y.to(y_ptr.dtype.element_ty), mask=mask)


def kernel(X, R):
    M, N = X.shape
    Y = torch.empty_like(X)
    _fused_ln_residual_kernel[(M,)](
        X, R, Y, N, 1e-5,
        X.stride(0), X.stride(1), R.stride(0), R.stride(1),
        Y.stride(0), Y.stride(1),
        BLOCK=triton.next_power_of_2(N), num_warps=8)
    return Y
