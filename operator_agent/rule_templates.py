"""规则模板生成器 - Coder 全部失败时的最后回退
单一职责: 按算子类型 + DSL 返回保底可运行代码; 无覆盖算子返回空串"""

from .schemas import TaskSchema, UnifiedSketch


_MATMUL_NUMPY = """import numpy as np

def kernel(A, B):
    return (A @ B).astype(A.dtype)
"""

_SOFTMAX_NUMPY = """import numpy as np

def kernel(X):
    mx = np.max(X, axis=-1, keepdims=True)
    ex = np.exp(X - mx)
    return (ex / np.sum(ex, axis=-1, keepdims=True)).astype(X.dtype)
"""

_MATMUL_TRITON = """import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + (offs_k + k_start)[None, :] * stride_ak)
        b = tl.load(b_ptr + (offs_k + k_start)[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        acc += tl.dot(a, b)
    c = acc.to(tl.float32)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c)

def kernel(A, B):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return C
"""

_SOFTMAX_TRITON = """import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel(x_ptr, y_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    x = tl.load(x_ptr + pid * n_cols + offs, mask=mask, other=-float('inf'))
    x_max = tl.max(x, axis=0)
    x_exp = tl.exp(x - x_max)
    x_sum = tl.sum(x_exp, axis=0)
    y = x_exp / x_sum
    tl.store(y_ptr + pid * n_cols + offs, y, mask=mask)

def kernel(X):
    n_rows, n_cols = X.shape
    Y = torch.empty_like(X)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    softmax_kernel[(n_rows,)](X, Y, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return Y
"""


class RuleTemplateCoder:
    """规则模板回退: 按算子族 + DSL 返回保底实现"""

    def generate(self, task: TaskSchema, sketch: UnifiedSketch = None) -> str:
        kt = (task.source.kernel_type or "").lower()
        dsl = (task.target.dsl or "").lower()
        is_triton = dsl == "triton"

        if "matmul" in kt or "gemm" in kt:
            return _MATMUL_TRITON if is_triton else _MATMUL_NUMPY
        if "softmax" in kt:
            return _SOFTMAX_TRITON if is_triton else _SOFTMAX_NUMPY
        # 无覆盖算子: 返回空, 上层决定后续行为
        return ""
