"""
Kernel Database & RAG 检索
结构化键检索的历史案例库 (原注释自称 "embedding-like" 是误导: 这里从来没有 embedding,
`search()` 走的是硬过滤 + 词元重叠评分 —— 与外部调研的结论一致, 见 docs/research/43)。

持久化: 一条 JSONL 事件日志, 启动时重放 (缺口 2 之前是「零文件 I/O」, 语料永远是
_init_seed_data() 里那 6 条字面量)。
"""

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional
from .schemas import UnifiedSketch


def _tokenize(text: str) -> set:
    """将文本切分为小写词元集合 (用于相关性评分)"""
    return set(re.findall(r"[a-z0-9_]+", (text or "").lower()))


def _perf_weight(record: "KernelRecord") -> float:
    """历史高性能案例的排序加分 (沿用旧口径的量级: 最多加 1.0)

    有可信加速比 (score) 时用 score, 没有时才回退 gflops。回退不是偷懒: 6 条手写 seed
    本来就没有同机基线可测, 而**旧排序只看 gflops 正是缺口 4 记过的那件事** —— 对
    memory-bound 算子, gflops 高的核可能恰恰是错的示范。score 上限截在 3.0×,
    因为加速比再高也不该盖过相关性本身。
    """
    if record.score > 0:
        return min(record.score, 3.0) / 3.0
    return min(record.performance, 100.0) / 100.0


# `OPERATOR_AGENT_KERNEL_DB=off` 与另外两条 env 一样用于「跑测试不得改动仓库状态」
_DISABLED = ("off", "none", "0", "false")


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
    # ---- 下面几条服务于「这条经验值不值得被再次取用」(docs/research/43 第四节) ----
    # score: 可信的同机加速比。**不是** gflops —— 缺口 4 之前判据固定看 gflops,
    # 对 memory-bound 算子整个失真 (doc 00 缺口 4 的原始证据)。它的读者是 search():
    # 历史高性能案例的排序权重。
    score: float = 0.0
    # 入库那一刻的测量有效性档位。没这个字段就没法区分「核真的快」和
    # 「数字描述的是 launch 开销」(缺口 1), 而持久化会把后者一起固化成经验。
    # 读者: 门 (measurement.persist_decision) 与 summary() 里的分档计数。
    measurement_validity: str = ""
    retrieved_count: int = 0
    created_at: float = 0.0
    # ⚠️ 刻意**没有** `is_buggy` 与 `shape_key`：前者在这条路径上没有写者 (只入库
    # all_passed 的核)，后者的读者 (按 shape 聚合可达上界, 见 doc 42 第六节) 还不存在。
    # 空字段就是下一份「算了没人读」。

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "KernelRecord":
        # 忽略未知键: 老日志里没新字段, 新日志多出的字段也不该让旧版本读不了。
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})



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
    算子数据库 - 存储 & 检索历史成功案例 (结构化键, 无 embedding)

    持久化是一行一条 JSON 的**追加式事件日志** (`kernel` / `retrieved`), 启动时重放。
    为什么是这个形态而不是向量库 / SQLite: 见 docs/research/43 第二、四节 ——
    检索键是结构化的 (`algorithm`/`hardware`/`dsl` + 词元评分), 语料量级是几百条,
    而三个用 embedding 的外部系统都依赖外部 embedding 服务, 与本项目的依赖约束冲突。
    """

    def __init__(self, jsonl_path: Optional[str] = None, persist_retrievals: bool = True):
        self._records: dict[str, KernelRecord] = {}
        self._jsonl = self._resolve_jsonl(jsonl_path)
        self._persist_retrievals = persist_retrievals
        self._init_seed_data()
        self.loaded_from_disk = 0
        self.disk_skipped_lines = 0
        self.retrievals_this_run = 0
        if self._jsonl is not None:
            self.replay()

    @staticmethod
    def _resolve_jsonl(jsonl_path: Optional[str]) -> Optional[Path]:
        """显式参数 > OPERATOR_AGENT_KERNEL_DB > <repo>/output/kernel_db.jsonl

        `output/` 已在 .gitignore 里 —— 语料是**这台机器**测出来的东西, 不进仓库
        (换一张卡, 里面的 score 就不成立了, 见 doc 42 的达成率口径)。
        值为 off/none/0/false 时整个关闭 (单元测试用它来保持仓库无副作用)。
        """
        p = jsonl_path or os.environ.get("OPERATOR_AGENT_KERNEL_DB", "").strip()
        if not p:
            p = str(Path(__file__).resolve().parent.parent / "output" / "kernel_db.jsonl")
        if p.lower() in _DISABLED:
            return None
        return Path(p)

    # ---- 事件日志 ----

    def _append_event(self, event: dict) -> None:
        if self._jsonl is None:
            return
        try:
            self._jsonl.parent.mkdir(parents=True, exist_ok=True)
            with self._jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError:
            # 写不进日志不该让一次真跑失败: 语料是副产品, 产物才是主品。
            pass

    def replay(self) -> int:
        """重放事件日志, 返回从磁盘读回的记录数 (不含 seed)

        坏行**跳过并计数**, 不整体放弃: 追加式日志被 Ctrl-C 截断最后一行是常态。
        """
        if self._jsonl is None or not self._jsonl.exists():
            self.disk_skipped_lines = 0
            return 0
        loaded = 0
        skipped = 0
        with self._jsonl.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                kind = ev.get("event")
                if kind == "kernel":
                    try:
                        rec = KernelRecord.from_dict(ev["record"])
                    except (KeyError, TypeError):
                        skipped += 1
                        continue
                    self._records[rec.kernel_id] = rec
                    loaded += 1
                elif kind == "retrieved":
                    for kid in ev.get("ids", []):
                        r = self._records.get(kid)
                        if r is not None:
                            r.retrieved_count += 1
        self.disk_skipped_lines = skipped
        self.loaded_from_disk = loaded
        return loaded

    def summary(self) -> str:
        """给人在每次运行的日志里读的一句 —— 没有这一句, 「载入了历史」就是看不见的"""
        if self._jsonl is None:
            return f"{len(self._records)} 条记录 (持久化关闭, 只有 seed)"
        n_seed = sum(1 for k in self._records if k.startswith("seed_"))
        mine = len(self._records) - n_seed
        used = sum(1 for r in self._records.values() if r.retrieved_count)
        # 分档计数是 `measurement_validity` 这个字段的读者: 语料里有多少是「当年测的时候
        # 就不干净」的, 直接决定这批 score 能不能拿去做判断 (doc 43 第四节的入库门)。
        tiers = {}
        for r in self._records.values():
            if r.kernel_id.startswith("seed_"):
                continue
            tiers[r.measurement_validity or "未标"] = tiers.get(r.measurement_validity or "未标", 0) + 1
        tier_text = " ".join(f"{k}={v}" for k, v in sorted(tiers.items())) or "无"
        return (f"{len(self._records)} 条记录 = seed {n_seed} + 历史 {mine} "
                f"(从 {self._jsonl.name} 重放 {self.loaded_from_disk} 条"
                f"{f', 跳过坏行 {self.disk_skipped_lines}' if self.disk_skipped_lines else ''}"
                f"); 曾被检索 {used} 条; 历史部分测量档位 {tier_text}")

    def note_retrievals(self, ids: list) -> None:
        if not ids:
            return
        # 本次运行取用了多少条经验 —— 没有这个数, 「RAG 命中 N 个案例」就永远只是
        # 「语料里有 N 条」的另一种说法 (缺口 2 的原始失效方式)。与 D2 同一条纪律:
        # 先观测、看得见, 再谈用它定退役规则。
        self.retrievals_this_run += len(ids)
        if not self._persist_retrievals:
            return
        self._append_event({"event": "retrieved", "ids": list(ids), "at": time.time()})
        for kid in ids:
            r = self._records.get(kid)
            if r is not None:
                r.retrieved_count += 1

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
            score += relevance + _perf_weight(r)
            scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        hits = [r for _, r in scored[:limit]]
        # 检索计数是「这条经验有没有被用」的唯一证据链, 也是将来定退役规则的输入
        # (docs/research/43 第四节: 本轮**不设**阈值, 只攒数)。
        self.note_retrievals([r.kernel_id for r in hits])
        return hits

    def add(self, record: KernelRecord):
        """添加新记录 (只进内存)"""
        self._records[record.kernel_id] = record

    def promote(self, record: KernelRecord) -> bool:
        """入库 + 落盘。同 id 已存在且新分数不高时**不覆盖**。

        替换规则借自 ShinkaEvolve 的 crowding (`archive_selection_strategy="crowding"`:
        更相似的存档成员被更优者替换, 见 docs/research/43 第二节)。这里的键更简单,
        所以只保留「同一个核的更好测量才替换」这一种情况。
        """
        old = self._records.get(record.kernel_id)
        if old is not None and record.score <= old.score:
            return False
        record.created_at = record.created_at or time.time()
        self._records[record.kernel_id] = record
        self._append_event({"event": "kernel", "record": record.to_dict(),
                            "at": record.created_at})
        return True

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
