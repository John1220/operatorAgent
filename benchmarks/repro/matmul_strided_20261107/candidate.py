# S3 reconstructed regression fixture; this is not the original 2026-10-19 kernel.
# Pattern source, read without extracting or changing the archive:
# output/plugin_completion/20261026-s0/history-matmul4096-autonomous.tar.gz
#   declaration.json: precision=fp16
#   candidate/code.py: fp16 pointer loads, FP32 accumulator, final output cast.
# The retained H07 source uses 128x128x64. This reduced fixture uses 256x256x64
# to reproduce the documented old 384 KiB proxy calculation. Its actual shared
# memory is unknown without a trusted layout contract or native compiler receipt.
# The old 192 KiB half-width proxy is not a measured resource receipt either.
import triton
import triton.language as tl
import torch


@triton.jit
def _matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   BLOCK_K: tl.constexpr):
    """FP16 storage, FP32 accumulation; pointer loads preserve input dtype."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + rows[:, None] * K + ks[None, :]
    b_ptrs = b_ptr + ks[:, None] * N + cols[None, :]
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=(rows[:, None] < M)
                    & (ks[None, :] < K - k * BLOCK_K), other=0.0)
        b = tl.load(b_ptrs, mask=(cols[None, :] < N)
                    & (ks[:, None] < K - k * BLOCK_K), other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N
    c_ptrs = c_ptr + rows[:, None] * N + cols[None, :]
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty),
             mask=(rows[:, None] < M) & (cols[None, :] < N))


def kernel(A, B):
    """A and B are FP16 tensors; FP32 accumulation does not widen their storage."""
    M, K = A.shape
    _, N = B.shape
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    grid = (triton.cdiv(M, 256), triton.cdiv(N, 256))
    _matmul_kernel[grid](A, B, C, M, N, K,
                         BLOCK_M=256, BLOCK_N=256, BLOCK_K=64,
                         num_stages=3)
    return C
