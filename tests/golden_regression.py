"""
Golden 回归 Harness (#3 号 Harness)

对 OperatorAgent 最典型的几个"典型算子任务"建立一套基准(baseline) 产物目录，
后续任意 Agent 代码变更 (prompt/配方/schema/路由/LLM 策略) 后，运行本脚本能快速判断：
  * 是否有 case 从 PASS 变成 FAIL (功能回归)
  * 性能 (GFLOPS / 速度比) 是否下降超过容忍阈值 (性能回归)

用法:
  # 1) 初次建立 baseline (写入 tests/golden_registry.json)
  python tests/golden_regression.py baseline

  # 2) 对比当前代码 vs baseline (CI 里用这个)
  #    exit code: 0 = 没有要处理的事; 1 = 有回归要修代码; 2 = 期望过期了, 要人签认
  python tests/golden_regression.py compare
  python tests/golden_regression.py compare --tolerance 0.15        # 性能**双向** ±15% (默认 0.2 = ±20%)
  python tests/golden_regression.py compare --case matmul_mock_py   # 只跑单个 case

  # 3) 列出所有注册的 golden case
  python tests/golden_regression.py list

设计原则:
  * 默认全部跑 MockGPU，不依赖真 GPU/SSH/LLM（LLM 用 MockLLM 注入"已知能通过的代码"作为输出），
    这样 CI 在本地/无网环境也能跑；需要真 GPU case 可 enable_gpu=True 单独加。
  * baseline 存相对 repo 根的 artifact_dir + 验证通过/GFLOPS/延迟（旧格式仍可读）。⚠️ 那个目录在 `output/`
    下、是 gitignore 的，所以它**不是**「权威代码副本」—— 旧版本地绝对路径曾让整份
    registry 只能活在这台机上 (见 doc 54)。
  * compare 的判据是**正交事实各自上报**（`FACTS`），状态由事实推出；「一致地失败」
    「修好了」「性能数上漂」「没有性能读数」都不许再写成 PASS。失败时不改 baseline
    (避免 silent 覆盖, 需要显式 `baseline` 更新)。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tempfile
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Optional

# 允许直接 `python tests/golden_regression.py` 运行
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# Golden 是独立运行的 harness；显式打开仅限 harness 的隔离开关，避免
# `jsonl_path="off"` 在生产默认值收紧后落到仓库持久化路径。
from operator_agent.app._harness_controls import enable_for_harness
enable_for_harness()

from operator_agent import (
    TaskSchema, SourceSpec, TargetSpec, ConstraintSpec,
    TaskType, TaskComplexity, Precision, DSLEnum,
    Conductor, MockGPU, KernelDatabase,
)
from operator_agent.core.kernel_db import FailureDatabase


REGISTRY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_registry.json")


def portable(p: str) -> str:
    """把 repo 内的绝对路径收成**相对 repo 根**的形式。

    registry 要能被提交，前提是它不含机器本地路径：旧版写的是
    `/Users/.../output/...` 这种绝对串，那既让整份期望文件只能活在一台机上
    （它因此被 .gitignore 排除、从未进过版本库），又把用户名写进了一个本该可分享的
    产物里。留着这个目录本身没意义（`output/` 是 gitignore 的），它只是**出处指针**。
    """
    if not p:
        return ""
    ap = os.path.abspath(p)
    if ap == REPO_ROOT or ap.startswith(REPO_ROOT + os.sep):
        return os.path.relpath(ap, REPO_ROOT)
    return ap
DEFAULT_TOLERANCE = 0.20  # baseline GFLOPS 的 ±20% 以内算 OK (⚠️ 见下面的 FACTS: 旧实现只罚下行)

# ── 正交事实清单 (#73, 吸纳 DSH `docs/defensive-patterns.md:9` + `docs/testing.md:27-29`) ──
# 规则原文: "A result can be several things at once ... Surface each independent fact on
# its own; never nest one flag's report inside another's branch, or a caller reads a
# cut-short run as a clean success."
#
# 本仓的失效形态不是截断，是**一个 `PASS` 盖住四件不同的事**（实测 4 例里只有 1 例是活的
# 仪器，报表却写 `PASS 4 / FAIL 0`）：一致地坏、修好了没人更新期望、性能数漂了 +492%、
# 压根没有性能读数 —— 以前全叫 PASS。所以下面每个事实**单独一个名字、单独计数**，
# 汇总行不许只报 PASS 数。
FACTS = (
    "functional_regression",   # baseline 过 → 当前不过: 唯一确定是回归的一族
    "perf_regression",         # 掉出容忍窗口的下行: 也是回归
    "perf_drift_up",           # 上行超出容忍窗口: 不是回归, 但那个期望值必须被人重新签认
    "baseline_was_failing",    # 两边都失败 ⇒ 这一例检测不了任何东西 (死仪器)
    "fixed_since_baseline",    # baseline 坏 → 当前好: 期望过期了
    "no_perf_reference",       # 没有 GFLOPS/延迟或不判性能
)

#: 状态词。`REVIEW` 与 `FAIL` 分开是因为它们要求的动作不同：
#: FAIL = 代码坏了；REVIEW = **期望**过期了，得有人决定是刷新还是修 case。
#: 混成一个词就回到了本片要消除的那个失效形态。
STATUSES = ("PASS", "REVIEW", "FAIL", "SKIP")


# ─── Mock 代码库: 保证 MockGPU 能稳定通过 L1/L2, 性能仍是本机 CPU 实测，不是伪造的 GPU 数字 ──

PYTHON_VALID_MATMUL = """\
import numpy as np

def kernel(A, B):
    '''Python 参考矩阵乘法 (MockGPU L2 100% 对比用)'''
    return np.matmul(A, B)
"""

PYTHON_VALID_SOFTMAX = """\
import numpy as np

def kernel(X, axis=-1):
    '''数值稳定 softmax'''
    m = np.max(X, axis=axis, keepdims=True)
    e = np.exp(X - m)
    return e / np.sum(e, axis=axis, keepdims=True)
"""

PYTHON_VALID_LAYERNORM = """\
import numpy as np

def kernel(X, eps=1e-5):
    mu = np.mean(X, axis=-1, keepdims=True)
    var = np.var(X, axis=-1, keepdims=True)
    return (X - mu) / np.sqrt(var + eps)
"""

PYTHON_VALID_RELU = """\
import numpy as np

def kernel(X):
    return np.maximum(X, 0)
"""


@dataclass
class GoldenCase:
    case_id: str
    description: str
    build_task: Callable[[], TaskSchema]          # 工厂函数, 每次调用返回一个全新 TaskSchema
    build_inputs: Callable[[], dict]              # 工厂函数, 每次返回新的随机/固定 test_inputs
    expected_code: Optional[str] = None           # 若给, 则用固定代码的 MockLLM (保证稳定)
    enable_gpu: bool = False                      # True = 需要真 GPU, 默认 CI 里跳过
    min_speedup: float = 0.5                      # 这个 case 的 L3 最低通过线


@dataclass
class BaselineRecord:
    case_id: str
    description: str
    timestamp: float
    git_hash: str = ""                            # 若在 git repo 内, 可记录 commit
    artifact_dir: str = ""
    all_passed: bool = False
    l1_passed: bool = False
    l2_passed: bool = False
    l3_passed: bool = False
    gflops: float = 0.0
    latency_ms: Optional[float] = None            # 内存算子没有 GFLOPS，比较实测平均延迟
    speedup_vs_baseline: float = 0.0
    llm_total_tokens: int = 0
    llm_cost_usd: float = 0.0


@dataclass
class CompareReport:
    case_id: str
    status: str                   # "PASS" | "REVIEW" | "FAIL" | "SKIP" (见 STATUSES)
    reason: str = ""
    facts: tuple = ()             # 命中的独立事实 (FACTS 里的词), 可为多只
    baseline_gflops: float = 0.0
    current_gflops: float = 0.0
    delta_ratio: float = 0.0      # 吞吐收益：GFLOPS cur/bl-1 或延迟 bl/cur-1
    metric: str = "gflops"
    baseline_latency_ms: Optional[float] = None
    current_latency_ms: Optional[float] = None
    baseline_all_passed: Optional[bool] = None
    current_all_passed: Optional[bool] = None


# ───────────────────────── 4 个内置典型 golden case ─────────────────────────

def _case_matmul_mock_python() -> GoldenCase:
    def build_task():
        return TaskSchema(
            task_type=TaskType.OPERATOR_GENERATION,
            source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.MEDIUM),
            target=TargetSpec(hardware="mock_gpu", dsl=DSLEnum.PYTHON.value, compute_units=64, warp_size=32),
            constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=0.5),
        )
    def build_inputs():
        import numpy as np
        rng = np.random.RandomState(42)
        M, N, K = 256, 128, 64
        A = rng.randn(M, K).astype(np.float32)
        B = rng.randn(K, N).astype(np.float32)
        return {"A": A, "B": B}
    return GoldenCase(
        case_id="matmul_mock_py",
        description="MockGPU FP32 matmul (256x64x128), Python DSL",
        build_task=build_task,
        build_inputs=build_inputs,
        expected_code=PYTHON_VALID_MATMUL,
        min_speedup=0.5,
    )


def _case_softmax_mock_python() -> GoldenCase:
    def build_task():
        return TaskSchema(
            task_type=TaskType.OPERATOR_GENERATION,
            source=SourceSpec(kernel_type="softmax", complexity=TaskComplexity.LOW),
            target=TargetSpec(hardware="mock_gpu", dsl=DSLEnum.PYTHON.value),
            constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=0.5),
        )
    def build_inputs():
        import numpy as np
        rng = np.random.RandomState(17)
        X = rng.randn(8, 64).astype(np.float32)
        return {"X": X}
    return GoldenCase(
        case_id="softmax_mock_py",
        description="MockGPU FP32 softmax (8x64), Python DSL",
        build_task=build_task,
        build_inputs=build_inputs,
        expected_code=PYTHON_VALID_SOFTMAX,
    )


def _case_layernorm_mock_python() -> GoldenCase:
    def build_task():
        return TaskSchema(
            task_type=TaskType.OPERATOR_GENERATION,
            source=SourceSpec(kernel_type="layernorm", complexity=TaskComplexity.LOW),
            target=TargetSpec(hardware="mock_gpu", dsl=DSLEnum.PYTHON.value),
            constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=0.5),
        )
    def build_inputs():
        import numpy as np
        rng = np.random.RandomState(9)
        X = rng.randn(4, 16).astype(np.float32)
        return {"X": X}
    return GoldenCase(
        case_id="layernorm_mock_py",
        description="MockGPU FP32 layernorm (4x16), Python DSL",
        build_task=build_task,
        build_inputs=build_inputs,
        expected_code=PYTHON_VALID_LAYERNORM,
    )


def _case_relu_mock_python() -> GoldenCase:
    def build_task():
        return TaskSchema(
            task_type=TaskType.OPERATOR_GENERATION,
            source=SourceSpec(kernel_type="relu", complexity=TaskComplexity.LOW),
            target=TargetSpec(hardware="mock_gpu", dsl=DSLEnum.PYTHON.value),
            constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=0.5),
        )
    def build_inputs():
        import numpy as np
        rng = np.random.RandomState(3)
        X = rng.randn(16, 16).astype(np.float32)
        return {"X": X}
    return GoldenCase(
        case_id="relu_mock_py",
        description="MockGPU FP32 ReLU elementwise (16x16), Python DSL",
        build_task=build_task,
        build_inputs=build_inputs,
        expected_code=PYTHON_VALID_RELU,
    )


DEFAULT_CASES: list[GoldenCase] = [
    _case_matmul_mock_python(),
    _case_softmax_mock_python(),
    _case_layernorm_mock_python(),
    _case_relu_mock_python(),
]


# ───────────────────────── MockLLM: 注入固定代码, 保证可重复性 ───────────────

class _FixedCodeMockLLM:
    """Mock LLM: Designer 返回合法 sketch, Coder 每次返回用户指定的 expected_code。

    注意: conductor 内部会按 `caller_tag` 区分 Designer 和 Coder, 这里根据 prompt
    内容粗略判断是 "要 sketch 的 JSON" 还是 "要代码"。
    """

    def __init__(self, code: str, sketch_overrides: Optional[dict] = None):
        self._code = code
        self._sketch_overrides = sketch_overrides or {}
        # 成本账本兼容: 提供 cost_tracker 空属性, conductor 读 summary 不会炸
        class _EmptyTracker:
            total_calls = 0
            def summary(self):
                return {
                    "total_calls": 0, "total_prompt_tokens": 0,
                    "total_completion_tokens": 0, "total_tokens": 0,
                    "total_cost_usd": 0.0,
                    "breakdown_by_caller": {}, "n_estimated_usage": 0,
                }

            def watermark(self):
                return 0

            def summary_since(self, n):
                return self.summary()
            def pretty(self):
                return "LLM 账单摘要: 0 次调用, $0.0000"
        self.cost_tracker = _EmptyTracker()

    def chat(self, messages, temperature=None, max_tokens=None, raw=False, caller_tag="Unknown"):
        # Coder prompt 里包含 "代码" / "kernel" / "Triton" 关键词 → 吐代码
        last = (messages[-1].get("content") or "") if messages else ""
        if "代码" in last or "kernel" in last or "Triton" in last or "算子" in last:
            return self._code
        # 其它情况 (比如 Fixer) 默认吐和 Coder 一样的代码
        return self._code

    def chat_json(self, messages, temperature=None, max_tokens=None, caller_tag="Unknown"):
        # Designer 的 sketch JSON
        sketch = {
            "algorithm": "matmul",
            "tiling": {"block_m": 64, "block_n": 64, "block_k": 32},
            "memory_plan": {},
            "pipeline": [],
            "optimization_hints": [],
        }
        sketch.update(self._sketch_overrides)
        return dict(sketch)


# ───────────────────────── GoldenRegressionHarness 核心 ──────────────────────

class GoldenRegressionHarness:
    def __init__(self, registry_path: str = REGISTRY_PATH, cases: Optional[list[GoldenCase]] = None):
        self.registry_path = registry_path
        self.cases = cases or DEFAULT_CASES

    # --- 公共 API ---

    def list_cases(self) -> list[dict]:
        return [
            {
                "case_id": c.case_id,
                "description": c.description,
                "enable_gpu": c.enable_gpu,
                "dsl": c.build_task().target.dsl,
                "kernel_type": c.build_task().source.kernel_type,
            }
            for c in self.cases
        ]

    def load_registry(self) -> dict[str, BaselineRecord]:
        if not os.path.exists(self.registry_path):
            return {}
        with open(self.registry_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        out = {}
        for cid, d in (raw.get("baselines") or {}).items():
            out[cid] = BaselineRecord(**d)
        return out

    def save_registry(self, baselines: dict[str, BaselineRecord]) -> None:
        payload = {
            "schema_version": 1,
            "updated_at": time.time(),
            "baselines": {cid: asdict(rec) for cid, rec in baselines.items()},
        }
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def run_case(self, case: GoldenCase) -> BaselineRecord:
        """跑单个 golden case, 返回 BaselineRecord (不写 registry)"""
        if case.enable_gpu:
            # TODO: 目前只做 MockGPU 的无依赖 golden；真 GPU 留后续扩展
            raise RuntimeError(f"case {case.case_id} 需要真 GPU, 本 harness 暂未提供远端 backend 注入")

        task = case.build_task()
        task.constraints.min_speedup = case.min_speedup
        inputs = case.build_inputs()

        # 算法名和 case 对齐, 防止 Designer sketch 输出 matmul 导致 mismatch
        algo = task.source.kernel_type or "matmul"
        sketch_overrides = {"algorithm": algo}
        llm = _FixedCodeMockLLM(code=case.expected_code or "", sketch_overrides=sketch_overrides)

        conductor = Conductor(
            backend=MockGPU(),
            kernel_db=KernelDatabase(jsonl_path="off"),
            failure_db=FailureDatabase(jsonl_path="off"),
            llm=llm,
            max_optimize_rounds=1,   # golden 回归只跑 1 轮, 测"当前 Agent 首过能力"
        )
        artifact = conductor.execute(task=task, test_inputs=inputs, target_gflops=1e-6)

        l1_passed = l2_passed = l3_passed = False
        gflops = 0.0
        latency_ms = None
        speedup = 0.0
        for r in artifact.verification_results:
            from operator_agent.core.schemas import VerificationLevel as VL
            if r.level == VL.L1_COMPILE:
                l1_passed = r.passed
            elif r.level == VL.L2_FUNCTION:
                l2_passed = r.passed
            elif r.level == VL.L3_PERFORMANCE:
                l3_passed = r.passed
                gflops = float(r.metrics.get("gflops") or 0.0)
                latency_ms = r.metrics.get("avg_time_ms")
                speedup = float(r.metrics.get("speedup_vs_baseline") or 0.0)

        cost = (artifact.metadata or {}).get("llm_cost", {}) or {}
        return BaselineRecord(
            case_id=case.case_id,
            description=case.description,
            timestamp=time.time(),
            git_hash=_git_hash() or "",
            artifact_dir=portable(artifact.saved_dir or ""),
            all_passed=artifact.all_passed,
            l1_passed=l1_passed,
            l2_passed=l2_passed,
            l3_passed=l3_passed,
            gflops=gflops,
            latency_ms=latency_ms,
            speedup_vs_baseline=speedup,
            llm_total_tokens=int(cost.get("total_tokens") or 0),
            llm_cost_usd=float(cost.get("total_cost_usd") or 0.0),
        )

    def baseline(self, only_case: Optional[str] = None) -> dict[str, BaselineRecord]:
        """运行每个 case 并保存为 baseline。若 only_case 指定则只更新一个。"""
        existing = self.load_registry()
        new_baselines: dict[str, BaselineRecord] = dict(existing)
        for case in self.cases:
            if only_case and case.case_id != only_case:
                continue
            if case.enable_gpu:
                print(f"[Golden] ⏭️  跳过 GPU case {case.case_id} ({case.description})")
                continue
            print(f"\n========== baseline {case.case_id}: {case.description} ==========")
            try:
                rec = self.run_case(case)
            except Exception as e:
                print(f"[Golden] ❌ {case.case_id} 执行失败: {e}")
                raise
            print(f"[Golden] PASS={rec.all_passed} (L1={rec.l1_passed} L2={rec.l2_passed} L3={rec.l3_passed}) "
                  f"GFLOPS={rec.gflops:.3f} speedup={rec.speedup_vs_baseline:.2f}x dir={rec.artifact_dir}")
            new_baselines[case.case_id] = rec
        self.save_registry(new_baselines)
        print(f"\n[Golden] baseline 已写入 {self.registry_path} ({len(new_baselines)} 个 case)")
        return new_baselines

    def compare(self, tolerance: float = DEFAULT_TOLERANCE, only_case: Optional[str] = None,
                strict_l3: bool = True) -> list[CompareReport]:
        """对比当前代码 vs registry 里的 baseline。返回 report list, 任何 FAIL 则 sys.exit(1)。

        strict_l3=False 时只关心"验证全通过/不通过"和 L2 正确性, 不判 L3 性能差异 (宽松模式)。
        """
        baselines = self.load_registry()
        if not baselines:
            print(f"[Golden] ⚠️  registry 为空 ({self.registry_path}), 请先跑 `baseline`")
            sys.exit(2)

        reports: list[CompareReport] = []
        for case in self.cases:
            if only_case and case.case_id != only_case:
                continue
            bl = baselines.get(case.case_id)
            if bl is None:
                reports.append(CompareReport(
                    case_id=case.case_id, status="SKIP",
                    reason=f"没有 baseline, 跳过 (先运行 baseline 建立)",
                ))
                print(f"[Golden] ⏭️  {case.case_id}: 无 baseline, 跳过")
                continue
            if case.enable_gpu:
                reports.append(CompareReport(
                    case_id=case.case_id, status="SKIP", reason="GPU case, 本 harness 暂不执行",
                ))
                continue

            print(f"\n========== compare {case.case_id}: {case.description} ==========")
            try:
                cur = self.run_case(case)
            except Exception as e:
                reports.append(CompareReport(
                    case_id=case.case_id, status="FAIL",
                    reason=f"执行异常: {e}", baseline_all_passed=bl.all_passed, current_all_passed=None,
                ))
                print(f"[Golden] ❌ {case.case_id} 执行异常: {e}")
                continue

            status, facts, reason, dr = _compare_one(bl, cur, tolerance=tolerance, strict_l3=strict_l3)
            print(f"[Golden] {status}  {reason}")
            reports.append(CompareReport(
                case_id=case.case_id, status=status, reason=reason, facts=facts,
                baseline_gflops=bl.gflops, current_gflops=cur.gflops, delta_ratio=dr,
                metric=("latency_ms" if bl.gflops <= 0 and bl.latency_ms else "gflops"),
                baseline_latency_ms=bl.latency_ms, current_latency_ms=cur.latency_ms,
                baseline_all_passed=bl.all_passed, current_all_passed=cur.all_passed,
            ))

        self._print_report(reports)
        return reports

    # --- 内部辅助 ---

    @staticmethod
    def _print_report(reports: list[CompareReport]) -> None:
        w = max(16, max((len(r.case_id) for r in reports), default=10))
        print(f"\n{'='*(w + 56)}")
        print(f"  {'Case':<{w}}  {'Status':<6}  {'Metric':<10}  {'Baseline':>9}  {'Current':>9}  {'Δ%':>7}  Reason")
        print(f"{'-'*(w + 56)}")
        for r in reports:
            if r.metric == "latency_ms":
                bl = f"{r.baseline_latency_ms:.6f}" if r.baseline_latency_ms else "-"
                cu = f"{r.current_latency_ms:.6f}" if r.current_latency_ms else "-"
            else:
                bl = f"{r.baseline_gflops:.2f}" if r.baseline_gflops else "-"
                cu = f"{r.current_gflops:.2f}" if r.current_gflops else "-"
            dr = f"{r.delta_ratio*100:+.1f}%" if r.delta_ratio else "-"
            print(f"  {r.case_id:<{w}}  {r.status:<6}  {r.metric:<10}  {bl:>9}  {cu:>9}  {dr:>7}  {r.reason}")
        print(f"{'='*(w + 56)}")
        n = {s: sum(1 for r in reports if r.status == s) for s in STATUSES}
        print(f"汇总: PASS {n['PASS']} / REVIEW {n['REVIEW']} / FAIL {n['FAIL']} "
              f"/ SKIP {n['SKIP']} (共 {len(reports)} case)")
        # 独立事实**各自一个数**，不折叠进状态里（DSH `defensive-patterns.md:9`）。
        # 一个 case 可以同时命中多个 ⇒ 这些数之和可以大于 case 数 —— 而这正是它比
        # 「PASS 4」更有用的原因：以前那个数把死仪器和漂了 5 倍的期望都说成好消息。
        hits = {f: sum(1 for r in reports if f in r.facts) for f in FACTS}
        print("独立事实: " + (", ".join(f"{f}={c}" for f, c in hits.items() if c) or "无"))
        dead = [r.case_id for r in reports if "baseline_was_failing" in r.facts]
        if dead:
            print(f"⚠️ 死仪器 (baseline 本身就是 FAIL, 检测不了任何回归): {', '.join(dead)}")


def _compare_one(bl: BaselineRecord, cur: BaselineRecord, tolerance: float,
                 strict_l3: bool) -> tuple[str, tuple, str, float]:
    """比较单个 baseline vs 当前结果, 返回 (status, facts, reason, delta_ratio)。

    `facts` 是**独立事实的集合**，不是一个状态词（DSH `defensive-patterns.md:9`）；
    状态由 facts 推出：命中回归类事实 ⇒ FAIL，只有「期望过期」类事实 ⇒ REVIEW，空 ⇒ PASS。
    分开是因为这两者要求的动作不同 —— 一个是**代码坏了**，一个是**期望要有人签认**。

    旧版把下面这四件事都返回成 PASS：两边都失败（死仪器）、baseline 坏而现在好了、
    性能数漂了 +492%（容忍窗写成 ±20% 却只罚下行）、以及本例压根没有性能读数
    （而且 reason 文案还谎称 `strict_l3=False`，实际那一维是开着的）。
    """
    facts: list = []
    notes: list = []
    dr = 0.0

    # 功能轴
    if bl.all_passed and not cur.all_passed:
        facts.append("functional_regression")
        notes.append(f"功能回归: baseline all_passed=True → 当前 False "
                     f"(L1={cur.l1_passed} L2={cur.l2_passed} L3={cur.l3_passed})")
    elif not bl.all_passed and not cur.all_passed:
        facts.append("baseline_was_failing")
        notes.append("两边都失败: 这一例检测不了任何回归, 是**死仪器**而不是「无回归」")
    elif not bl.all_passed and cur.all_passed:
        facts.append("fixed_since_baseline")
        notes.append("baseline 是 FAIL 而当前 PASS ⇒ 期望已过期, 需要有人刷新或说明为什么留着")

    # 性能轴: 只有两边都通过时「加速比」这件事才有对象
    if bl.all_passed and cur.all_passed:
        if not strict_l3:
            facts.append("no_perf_reference")
            notes.append("lenient 模式: 本例不判性能")
        else:
            # 吞吐算子保留 GFLOPS 比较；内存算子不能伪造 FLOP 数，只能比较
            # 同种统计量的实测延迟。旧 registry 没存延迟 ⇒ 仍是 REVIEW，不代填。
            metric_note = None
            if bl.gflops > 0:
                dr = (cur.gflops - bl.gflops) / bl.gflops
                metric_note = f"{bl.gflops:.2f}→{cur.gflops:.2f} GFLOPS"
            elif (bl.latency_ms is not None and bl.latency_ms > 0
                  and cur.latency_ms is not None and cur.latency_ms > 0):
                # 延迟越小越好；转成吞吐方向的收益，保留原有 ±20% 判据。
                dr = bl.latency_ms / cur.latency_ms - 1.0
                metric_note = f"{bl.latency_ms:.6f}→{cur.latency_ms:.6f} ms"
            else:
                facts.append("no_perf_reference")
                notes.append(f"baseline GFLOPS={bl.gflops} 或延迟不是可用分母 ⇒ 性能轴无从比较 "
                             f"(当前 GFLOPS={cur.gflops}, latency_ms={cur.latency_ms})")
            if metric_note is not None:
                if dr < -tolerance:
                    facts.append("perf_regression")
                    notes.append(f"性能回归: {metric_note} ({dr*100:+.1f}%), "
                                 f"超出容忍 -{tolerance*100:.0f}%")
                elif dr > tolerance:
                    facts.append("perf_drift_up")
                    notes.append(f"性能上漂: {metric_note} ({dr*100:+.1f}%): "
                                 f"不是回归, 但这个期望值已不代表任何被测物, 要人签认后刷新")

    if {"functional_regression", "perf_regression"} & set(facts):
        status = "FAIL"
    elif facts:
        status = "REVIEW"
    else:
        status = "PASS"
    return status, tuple(facts), "; ".join(notes) or "功能与性能都在容忍窗口内", dr


def _git_hash() -> Optional[str]:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


# ───────────────────────── CLI 入口 ──────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Golden Regression Harness for OperatorAgent")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="列出所有注册的 golden case")

    p_bl = sub.add_parser("baseline", help="跑全部 case, 写入/更新 baseline registry")
    p_bl.add_argument("--case", help="只跑指定 case_id")

    p_cmp = sub.add_parser("compare", help="跑全部 case 对比 baseline: 回归 exit=1, 期望待签认 exit=2")
    p_cmp.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                       help=f"性能回归容忍率 (默认 {DEFAULT_TOLERANCE:.0%})")
    p_cmp.add_argument("--case", help="只跑指定 case_id")
    p_cmp.add_argument("--lenient", action="store_true",
                       help="宽松模式: 只判功能回归, 不判 L3 性能回归")

    args = ap.parse_args(argv)
    # standalone CLI 不经过 tests/conftest.py，必须自己隔离 mock 产物。
    # 保留用户显式给的落点；不自动清理，失败现场和 baseline 出处可复查。
    if args.cmd == "compare" and (not os.environ.get("OPERATOR_AGENT_OUTPUT_DIR")
                               or not os.environ.get("OPERATOR_AGENT_LOG_DIR")):
        root = tempfile.mkdtemp(prefix="operator_agent_golden_")
        for name, dirname in (("OPERATOR_AGENT_OUTPUT_DIR", "output"),
                              ("OPERATOR_AGENT_LOG_DIR", "logs")):
            if not os.environ.get(name):
                os.environ[name] = os.path.join(root, dirname)
        print(f"[Golden] 临时运行目录: {root}")
    harness = GoldenRegressionHarness()

    if args.cmd == "list":
        cases = harness.list_cases()
        print(f"共注册 {len(cases)} 个 golden case:")
        for c in cases:
            gpu = " [GPU]" if c["enable_gpu"] else ""
            print(f"  - {c['case_id']:<20} ({c['kernel_type']:>10} / {c['dsl']:>7}){gpu}  {c['description']}")
        return 0

    if args.cmd == "baseline":
        harness.baseline(only_case=getattr(args, "case", None))
        return 0

    if args.cmd == "compare":
        reports = harness.compare(
            tolerance=args.tolerance,
            only_case=args.case,
            strict_l3=not args.lenient,
        )
        codes = {r.status for r in reports}
        # 两个退出码分开，是因为它们要求的**动作**不同：1 = 代码坏了要修，
        # 2 = 期望过期了要有人签认（刷新 baseline 或说明为什么留着）。
        # 合成一个码就等于把本片刚拆开的两件事又叠回去。
        if "FAIL" in codes:
            return 1
        if "REVIEW" in codes:
            return 2
        return 0

    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
