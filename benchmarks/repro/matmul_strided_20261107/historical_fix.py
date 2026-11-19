import triton
import triton.language as tl
import torch


@triton.jit
def _matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                   sa_m, sa_k, sb_k, sb_n,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   BLOCK_K: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + rows[:, None] * sa_m + ks[None, :] * sa_k
    b_ptrs = b_ptr + ks[:, None] * sb_k + cols[None, :] * sb_n
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(rows[:, None] < M) & (ks[None, :] < remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(cols[None, :] < N) & (ks[:, None] < remaining), other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_K * sa_k
        b_ptrs += BLOCK_K * sb_k
    c_ptrs = c_ptr + rows[:, None] * N + cols[None, :]
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty),
             mask=(rows[:, None] < M) & (cols[None, :] < N))


def kernel(A, B):
    M, K = A.shape
    _, N = B.shape
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))
    _matmul_kernel[grid](A, B, C, M, N, K,
                         A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                         BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_stages=3)
    return C
