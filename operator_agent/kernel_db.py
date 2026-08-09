"""
Kernel Database & RAG 检索
Simple in-memory database with embedding-like retrieval for kernel examples.
Supports Python and Triton DSL seed data.
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from .schemas import UnifiedSketch


def _tokenize(text: str) -> set:
    """将文本切分为小写词元集合 (用于相关性评分)"""
    return set(re.findall(r"[a-z0-9_]+", (text or "").lower()))


@dataclass
class KernelRecord:
    """数据库中的 Kernel 记录"""
    kernel_id: str
    algorithm: str
    hardware: str
    dsl: str              # python | triton | cuda_c
    code: str
    sketch: Optional[dict] = None
    performance: float = 0.0  # GFLOPS
    tags: list = field(default_factory=list)
    compile_info: dict = field(default_factory=dict)  # 编译元信息


# ─── Triton 种子代码 ──────────────────────────────────────────

_TRITON_MATMUL_CODE = """import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        mask_a = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        mask_b = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = acc.to(tl.float32)
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=mask_c)

def kernel(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    matmul_kernel[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1), BLOCK_M=128, BLOCK_N=128, BLOCK_K=32)
    return C
"""

_TRITON_SOFTMAX_CODE = """import triton
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

_TRITON_ATTENTION_CODE = """import triton
import triton.language as tl
import torch

@triton.jit
def attention_kernel(q_ptr, k_ptr, v_ptr, o_ptr, seq_len, d_k, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, d_k)
    q = tl.load(q_ptr + offs_m[:, None] * d_k + offs_d[None, :])
    acc = tl.zeros((BLOCK_M, d_k), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for n_start in range(0, seq_len, BLOCK_N):
        n = n_start + offs_n
        mask_n = n < seq_len
        k = tl.load(k_ptr + n[:, None] * d_k + offs_d[None, :], mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptr + n[:, None] * d_k + offs_d[None, :], mask=mask_n[:, None], other=0.0)
        scores = tl.dot(q, k.T) / tl.sqrt(tl.cast(d_k, tl.float32))
        scores = tl.where(mask_n[None, :], scores, -float('inf'))
        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(mask_n[None, :], p, 0.0)
        l_new = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v)
        m_i = m_new
        l_i = l_new
    o = acc / l_i[:, None]
    tl.store(o_ptr + offs_m[:, None] * d_k + offs_d[None, :], o)

def kernel(Q, K, V):
    seq_len, d_k = Q.shape
    O = torch.empty((seq_len, d_k), device=Q.device, dtype=torch.float32)
    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(seq_len, BLOCK_M),)
    attention_kernel[grid](Q, K, V, O, seq_len, d_k, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return O
"""

_TRITON_LAYERNORM_CODE = """import triton
import triton.language as tl
import torch

@triton.jit
def layernorm_kernel(X_ptr, Y_ptr, W_ptr, B_ptr, M, N, stride_x, stride_y, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x_ptrs = X_ptr + row * stride_x + offs
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = x_centered * inv_std
    w = tl.load(W_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x_norm * w + b
    y_ptrs = Y_ptr + row * stride_y + offs
    tl.store(y_ptrs, y.to(tl.float32), mask=mask)

def kernel(X: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    M, N = X.shape
    Y = torch.empty_like(X)
    BLOCK_SIZE = triton.next_power_of_2(N)
    layernorm_kernel[(M,)](X, Y, weight, bias, M, N, X.stride(0), Y.stride(0), eps, BLOCK_SIZE=BLOCK_SIZE)
    return Y
"""


class KernelDatabase:
    """
    算子数据库 - 存储 & 检索历史成功案例
    模拟 RAG (Retrieval-Augmented Generation) 检索能力
    支持 Python 和 Triton DSL 的混合检索
    """

    def __init__(self):
        self._records: dict[str, KernelRecord] = {}
        self._init_seed_data()

    def _init_seed_data(self):
        """预置种子数据，模拟历史成功案例"""

        # ─── Python 种子数据 (Mock GPU) ───
        py_matmul = KernelRecord(
            kernel_id="seed_py_001",
            algorithm="matmul",
            hardware="mock_gpu",
            dsl="python",
            code="# tiled matmul kernel (numpy)",
            sketch={
                "algorithm": "matmul",
                "tiling": {"block_m": 64, "block_n": 64, "block_k": 32},
            },
            performance=8.5,
            tags=["matmul", "standard", "tiled"],
        )
        self._records[py_matmul.kernel_id] = py_matmul

        py_softmax = KernelRecord(
            kernel_id="seed_py_002",
            algorithm="softmax",
            hardware="mock_gpu",
            dsl="python",
            code="# online softmax kernel (numpy)",
            sketch={
                "algorithm": "softmax",
                "tiling": {"block_m": 128, "block_n": 128, "block_k": 1},
            },
            performance=5.2,
            tags=["softmax", "reduction", "online"],
        )
        self._records[py_softmax.kernel_id] = py_softmax

        # ─── Triton 种子数据 (NVIDIA GPU) ───
        triton_matmul = KernelRecord(
            kernel_id="seed_triton_001",
            algorithm="matmul",
            hardware="nvidia_gpu",
            dsl="triton",
            code=_TRITON_MATMUL_CODE,
            sketch={
                "algorithm": "basic_matmul",
                "tiling": {"block_m": 128, "block_n": 128, "block_k": 32},
                "target_dsl": "triton",
                "l2_cache_grouping": False,
                "pointer_advancement": True,
                "three_arg_dot": False,
                "modulo_bounds": False,
            },
            performance=85.0,
            tags=["matmul", "triton", "basic", "tensor_core"],
            compile_info={
                "num_warps": 4,
                "num_stages": 2,
            },
        )
        self._records[triton_matmul.kernel_id] = triton_matmul

        triton_softmax = KernelRecord(
            kernel_id="seed_triton_002",
            algorithm="softmax",
            hardware="nvidia_gpu",
            dsl="triton",
            code=_TRITON_SOFTMAX_CODE,
            sketch={
                "algorithm": "online_softmax",
                "tiling": {"block_m": 128, "block_n": 128, "block_k": 1},
                "target_dsl": "triton",
            },
            performance=30.0,
            tags=["softmax", "triton", "reduction", "online"],
            compile_info={
                "num_warps": 4,
                "num_stages": 1,
            },
        )
        self._records[triton_softmax.kernel_id] = triton_softmax

        triton_attention = KernelRecord(
            kernel_id="seed_triton_003",
            algorithm="attention",
            hardware="nvidia_gpu",
            dsl="triton",
            code=_TRITON_ATTENTION_CODE,
            sketch={
                "algorithm": "flash_attention",
                "tiling": {"block_m": 64, "block_n": 64, "block_k": 64},
                "target_dsl": "triton",
            },
            performance=60.0,
            tags=["attention", "triton", "flash", "online_softmax", "a100"],
            compile_info={
                "num_warps": 8,
                "num_stages": 3,
            },
        )
        self._records[triton_attention.kernel_id] = triton_attention

        triton_layernorm = KernelRecord(
            kernel_id="seed_triton_004",
            algorithm="layernorm",
            hardware="nvidia_gpu",
            dsl="triton",
            code=_TRITON_LAYERNORM_CODE,
            sketch={
                "algorithm": "two_pass_layernorm",
                "tiling": {"block_size": 1024},
                "target_dsl": "triton",
                "fused_compute": False,
            },
            performance=35.0,
            tags=["layernorm", "triton", "normalization", "reduce"],
            compile_info={
                "num_warps": 4,
                "num_stages": 1,
            },
        )
        self._records[triton_layernorm.kernel_id] = triton_layernorm

    def search(
        self,
        algorithm: str = "",
        hardware: str = "",
        dsl: str = "",
        tags: list = None,
        limit: int = 5,
    ) -> list[KernelRecord]:
        """
        评分式检索 - 模拟多层级 RAG
        - hardware: 硬过滤 (mock_gpu 记录视为通用)
        - dsl: 精确匹配加权, python 作为通用回退
        - algorithm/tags: 词元重叠 + 子串相关性评分 (提供了查询但零相关则剔除)
        - performance: 轻微加分, 同分时优先历史高性能案例
        返回按评分降序的前 limit 条
        """
        candidates = list(self._records.values())

        if hardware:
            candidates = [r for r in candidates if r.hardware == hardware or r.hardware == "mock_gpu"]

        query_tokens = _tokenize(algorithm + " " + " ".join(tags or []))
        scored: list[tuple[float, KernelRecord]] = []
        for r in candidates:
            score = 0.0
            if dsl:
                if r.dsl == dsl:
                    score += 3.0
                elif r.dsl == "python":
                    score += 0.5
                else:
                    continue
            relevance = 0.0
            record_tokens = _tokenize(
                " ".join([r.algorithm, " ".join(r.tags), str(r.sketch or {})])
            )
            relevance += 2.0 * len(query_tokens & record_tokens)
            if algorithm and algorithm.lower() in r.algorithm.lower():
                relevance += 2.0
            if tags and any(t in r.tags for t in tags):
                relevance += 2.0
            if (algorithm or tags) and relevance == 0:
                continue
            score += relevance + min(r.performance, 100.0) / 100.0
            scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:limit]]

    def add(self, record: KernelRecord):
        """添加新记录"""
        self._records[record.kernel_id] = record

    def get(self, kernel_id: str) -> Optional[KernelRecord]:
        return self._records.get(kernel_id)

    def list_all(self) -> list[KernelRecord]:
        return list(self._records.values())

    def list_by_dsl(self, dsl: str) -> list[KernelRecord]:
        """按 DSL 过滤记录"""
        return [r for r in self._records.values() if r.dsl == dsl]

    def list_triton(self) -> list[KernelRecord]:
        """列出所有 Triton 记录"""
        return self.list_by_dsl("triton")


# ─── 失败案例库 (负样本沉淀 + Reflexion 结构化教训) ──────────────────

@dataclass
class FailureRecord:
    """失败案例记录: 验证失败 -> 修复 -> 教训 (Reflexion 结构化经验)

    原字段 (kernel_type/hardware/dsl/level/error_summary/fix_summary/fixed) 保留向后兼容;
    新增的 lessons/error_keywords/applicable_hardware/root_cause 是 Reflexion 自我反思
    产出的结构化条目, 用于 FixerAgent 在 prompt 中直接注入 "同类型错误如何修复"。
    """
    failure_id: str
    kernel_type: str
    hardware: str
    dsl: str
    level: str                          # L1 / L2 / L3 / L1+L2
    error_summary: str                  # 截断后的错误摘要 (搜索时做词元匹配)
    fix_summary: str = ""               # 修复说明 (修复成功时填写)
    fixed: bool = False                 # 该失败最终是否被修复
    created_at: float = 0.0
    # --- Reflexion 新字段 (结构化自我反思) ---
    root_cause: str = ""                # 一句话根因, 如 "K mask 与 pointer arithmetic 混用"
    lessons: list = field(default_factory=list)  # list[str]: 每条是独立可复用的教训
    error_keywords: list = field(default_factory=list)  # 用于相似度检索的关键词
    applicable_hardware: list = field(default_factory=list)  # 适用硬件 (空=全部适用)

    def __post_init__(self):
        if not self.created_at:
            import time
            self.created_at = time.time()


class FailureDatabase:
    """
    失败案例库 - 沉淀 错误类型 -> 修复手段 的负样本经验
    供 Fixer Agent 检索历史修复案例，避免重复踩坑

    支持两种检索:
      - search_by_context(kernel_type, dsl, hardware)  — 基础上下文检索 (原行为)
      - search_by_error(error_text, kernel_type, dsl)   — 基于 error_summary 词元相似度 (Reflexion 新增)
    """

    def __init__(self):
        self._records: dict[str, FailureRecord] = {}

    def add(self, record: FailureRecord):
        self._records[record.failure_id] = record

    def search(self, kernel_type: str = "", dsl: str = "", limit: int = 3) -> list[FailureRecord]:
        """老接口等价于 search_by_context(kernel_type=kernel_type, dsl=dsl)"""
        return self.search_by_context(kernel_type=kernel_type, dsl=dsl, limit=limit)

    def search_by_context(self, kernel_type: str = "", dsl: str = "",
                          hardware: str = "", limit: int = 3) -> list[FailureRecord]:
        """
        检索相关失败案例: kernel_type 子串匹配 + dsl 精确匹配 + hardware 词元提示
        排序: 已修复案例优先 (含修复手段, 参考价值高), 其次按 applicable_hardware 命中,
             最后按时间倒序
        """
        matches = []
        for r in self._records.values():
            if kernel_type and kernel_type.lower() not in r.kernel_type.lower() \
                    and r.kernel_type.lower() not in kernel_type.lower():
                continue
            if dsl and r.dsl != dsl:
                continue
            matches.append(r)

        hw_tok = (hardware or "").lower()
        def _sort_key(r):
            hw_match = hw_tok and (hw_tok in r.hardware.lower() or
                                    any(hw_tok in h.lower() for h in r.applicable_hardware))
            return (not r.fixed, not hw_match, -r.created_at)
        matches.sort(key=_sort_key)
        return matches[:limit]

    def search_by_error(self, error_text: str, kernel_type: str = "",
                        dsl: str = "", limit: int = 3) -> list[FailureRecord]:
        """
        Reflexion 新增: 基于错误摘要词元相似度检索。
        用于 Fixer 拿到当前错误日志后, 找到 "曾经相似错误 + 对应 lessons" 的历史案例。
        匹配: kernel_type/dsl 先过滤, 再按 error_text 与 r.error_summary + r.error_keywords
             的词元 Jaccard 相似度排序。
        """
        candidates = self.search_by_context(kernel_type=kernel_type, dsl=dsl,
                                             limit=max(50, limit * 5))
        if not error_text:
            return candidates[:limit]

        error_tokens = _tokenize(error_text)
        if not error_tokens:
            return candidates[:limit]

        scored = []
        for r in candidates:
            r_tokens = _tokenize(r.error_summary) | set(k.lower() for k in r.error_keywords)
            if not r_tokens:
                continue
            jaccard = len(error_tokens & r_tokens) / len(error_tokens | r_tokens)
            scored.append((jaccard, r))
        scored.sort(key=lambda t: (t[0], not t[1].fixed, -t[1].created_at), reverse=True)
        return [r for _, r in scored if _ > 0][:limit] or candidates[:limit]

    def list_all(self) -> list[FailureRecord]:
        return list(self._records.values())

    def __len__(self) -> int:
        return len(self._records)
