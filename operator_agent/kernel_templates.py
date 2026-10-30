"""算子模板的单一真源 —— 同一份 kernel 源码只有一个定义处。

为什么需要它（不是"该有个 utils"）：
  SOFTMAX_TRITON 曾被逐字抄在两处 —— 语料种子 kernel_db._TRITON_SOFTMAX_CODE
  与回退实现 rule_templates._SOFTMAX_TRITON（675 字符逐字相同）。两份都在跑，
  没有任何断言让它们同源；改一处、另一处静默留在旧版。

⚠️ 这里只放"有多个读者"的模板。语料的 matmul / attention / layernorm 种子
（kernel_db.py）与下面的 MATMUL_TRITON 是刻意不同的实现，不要合并：
  - 语料种子是给 LLM 看的正样本：带边界 mask、BLOCK=128、tl.dot(a, b, acc)，
    要对不能整除的 shape 也正确；
  - MATMUL_TRITON 是 Coder 全部失败时的保底：无 mask、BLOCK=64、
    acc += tl.dot(a, b)，只要能在对齐 shape 上跑起来。
  两者承担的角色不同，合并任何一个都会改掉另一个的语义。

"单一真源"在这里是被机械强制的：tests/unit/test_template_single_source.py
会扫 operator_agent/ 里有没有第二份逐字相同的副本，并把这条守卫本身反向标定。
"""

import ast as _ast


MATMUL_NUMPY = """import numpy as np

def kernel(A, B):
    return (A @ B).astype(A.dtype)
"""


SOFTMAX_NUMPY = """import numpy as np

def kernel(X):
    mx = np.max(X, axis=-1, keepdims=True)
    ex = np.exp(X - mx)
    return (ex / np.sum(ex, axis=-1, keepdims=True)).astype(X.dtype)
"""


MATMUL_TRITON = """import triton
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


SOFTMAX_TRITON = """import triton
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



def jit_kernel_source(source: str) -> str:
    """返回 source 里 @triton.jit 修饰的那个函数的原文（保留原格式）。

    存在的理由是让"取自某模板"这句话变成可派生而不是手抄：
    benchmarks/measure_heartbeat_real.py 曾手抄一份 matmul 核（语义相同、签名被
    重新折行 ⇒ 逐字比较抓不到），注释写着"取自 rule_templates.py"。模板一改，
    那个量具就会静静地量另一个核，而注释继续撒谎。

    取不到时抛 ValueError（不返回空串 ——「没有」与「没取到」本仓分得很开）。
    """
    tree = _ast.parse(source)
    for node in _ast.walk(tree):
        if not isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            continue
        is_jit = any(
            (isinstance(d, _ast.Attribute) and d.attr == "jit")
            or (isinstance(d, _ast.Name) and d.id == "jit")
            for d in node.decorator_list
        )
        if not is_jit:
            continue
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        src_lines = source.splitlines()
        return chr(10).join(src_lines[start - 1:node.end_lineno])
    raise ValueError("source 里没有 @triton.jit 修饰的函数")
