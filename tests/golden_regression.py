"""
Golden 回归 Harness (#3 号 Harness)

对 OperatorAgent 最典型的几个"典型算子任务"建立一套基准(baseline) 产物目录，
后续任意 Agent 代码变更 (prompt/配方/schema/路由/LLM 策略) 后，运行本脚本能快速判断：
  * 是否有 case 从 PASS 变成 FAIL (功能回归)
  * 性能 (GFLOPS / 速度比) 是否下降超过容忍阈值 (性能回归)

用法:
  # 1) 初次建立 baseline (写入 tests/golden_registry.json)
  python tests/golden_regression.py baseline

  # 2) 对比当前代码 vs baseline (CI 里用这个, 失败时 exit code != 0)
  python tests/golden_regression.py compare
  python tests/golden_regression.py compare --tolerance 0.15        # 性能 15% 以内可容忍 (默认 0.2 = 20%)
  python tests/golden_regression.py compare --case matmul_mock_py   # 只跑单个 case

  # 3) 列出所有注册的 golden case
  python tests/golden_regression.py list

设计原则:
  * 默认全部跑 MockGPU，不依赖真 GPU/SSH/LLM（LLM 用 MockLLM 注入"已知能通过的代码"作为输出），
    这样 CI 在本地/无网环境也能跑；需要真 GPU case 可 enable_gpu=True 单独加。
  * baseline 只存 artifact_dir 路径 + 验证通过/GFLOPS，不存代码副本 (artifact_dir 本身就是权威副本)。
  * compare 失败时把 "FAILED" 的 case 列出来, 不改 baseline (避免 silent 覆盖, 需要显式 `baseline` 更新)。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Optional

# 允许直接 `python tests/golden_regression.py` 运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from operator_agent import (
    TaskSchema, SourceSpec, TargetSpec, ConstraintSpec,
    TaskType, TaskComplexity, Precision, DSLEnum,
    Conductor, MockGPU, KernelDatabase,
)
from operator_agent.kernel_db import FailureDatabase


REGISTRY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_registry.json")
DEFAULT_TOLERANCE = 0.20  # baseline GFLOPS 的 ±20% 以内算 OK


# ─── Mock 代码库: 保证 MockGPU 能稳定通过 L1/L2, 且给出稳定的 "伪 GFLOPS" 数字 ──

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
    speedup_vs_baseline: float = 0.0
    llm_total_tokens: int = 0
    llm_cost_usd: float = 0.0


@dataclass
class CompareReport:
    case_id: str
    status: str                   # "PASS" | "FAIL" | "SKIP"
    reason: str = ""
    baseline_gflops: float = 0.0
    current_gflops: float = 0.0
    delta_ratio: float = 0.0      # (current - baseline) / baseline
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
            source=SourceSpec(kernel_type="elementwise", complexity=TaskComplexity.LOW),
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
            kernel_db=KernelDatabase(),
            failure_db=FailureDatabase(),
            llm=llm,
            max_optimize_rounds=1,   # golden 回归只跑 1 轮, 测"当前 Agent 首过能力"
        )
        artifact = conductor.execute(task=task, test_inputs=inputs, target_gflops=1e-6)

        l1_passed = l2_passed = l3_passed = False
        gflops = 0.0
        speedup = 0.0
        for r in artifact.verification_results:
            from operator_agent.schemas import VerificationLevel as VL
            if r.level == VL.L1_COMPILE:
                l1_passed = r.passed
            elif r.level == VL.L2_FUNCTION:
                l2_passed = r.passed
            elif r.level == VL.L3_PERFORMANCE:
                l3_passed = r.passed
                gflops = float(r.metrics.get("gflops") or 0.0)
                speedup = float(r.metrics.get("speedup_vs_baseline") or 0.0)

        cost = (artifact.metadata or {}).get("llm_cost", {}) or {}
        return BaselineRecord(
            case_id=case.case_id,
            description=case.description,
            timestamp=time.time(),
            git_hash=_git_hash() or "",
            artifact_dir=artifact.saved_dir or "",
            all_passed=artifact.all_passed,
            l1_passed=l1_passed,
            l2_passed=l2_passed,
            l3_passed=l3_passed,
            gflops=gflops,
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

            status, reason, dr = _compare_one(bl, cur, tolerance=tolerance, strict_l3=strict_l3)
            print(f"[Golden] {status}  {reason}")
            reports.append(CompareReport(
                case_id=case.case_id, status=status, reason=reason,
                baseline_gflops=bl.gflops, current_gflops=cur.gflops, delta_ratio=dr,
                baseline_all_passed=bl.all_passed, current_all_passed=cur.all_passed,
            ))

        self._print_report(reports)
        failed = [r for r in reports if r.status == "FAIL"]
        return reports

    # --- 内部辅助 ---

    @staticmethod
    def _print_report(reports: list[CompareReport]) -> None:
        w = max(16, max((len(r.case_id) for r in reports), default=10))
        print(f"\n{'='*(w + 56)}")
        print(f"  {'Case':<{w}}  {'Status':<6}  {'BL GFLOPS':>9}  {'Cur GFLOPS':>9}  {'Δ%':>7}  Reason")
        print(f"{'-'*(w + 56)}")
        for r in reports:
            bl = f"{r.baseline_gflops:.2f}" if r.baseline_gflops else "-"
            cu = f"{r.current_gflops:.2f}" if r.current_gflops else "-"
            dr = f"{r.delta_ratio*100:+.1f}%" if r.delta_ratio else "-"
            print(f"  {r.case_id:<{w}}  {r.status:<6}  {bl:>9}  {cu:>9}  {dr:>7}  {r.reason}")
        print(f"{'='*(w + 56)}")
        n_pass = sum(1 for r in reports if r.status == "PASS")
        n_fail = sum(1 for r in reports if r.status == "FAIL")
        n_skip = sum(1 for r in reports if r.status == "SKIP")
        print(f"汇总: PASS {n_pass} / FAIL {n_fail} / SKIP {n_skip} (共 {len(reports)} case)")


def _compare_one(bl: BaselineRecord, cur: BaselineRecord, tolerance: float,
                 strict_l3: bool) -> tuple[str, str, float]:
    """比较单个 baseline vs 当前结果, 返回 (status, reason, delta_ratio)"""
    # 功能回归: baseline PASS 时当前也必须 PASS
    if bl.all_passed and not cur.all_passed:
        return ("FAIL",
                f"功能回归: baseline all_passed=True → 当前 all_passed=False (L1={cur.l1_passed} L2={cur.l2_passed} L3={cur.l3_passed})",
                0.0)
    # baseline 是 FAIL 的 (比如故意存的反例), 允许当前变 PASS; 但 FAIL→FAIL 且同样失败类型, 算 PASS
    if not bl.all_passed and not cur.all_passed:
        return ("PASS", "baseline 本就是 FAIL, 当前也 FAIL (无功能回归)", 0.0)
    if not bl.all_passed and cur.all_passed:
        return ("PASS", "baseline 是 FAIL → 当前 PASS (修复了, 建议更新 baseline)", 0.0)

    # L3 性能回归
    if strict_l3 and bl.gflops > 0:
        delta = cur.gflops - bl.gflops
        dr = delta / bl.gflops
        # 容忍窗口: 允许 baseline * (1 - tolerance) ~ +inf
        if cur.gflops < bl.gflops * (1.0 - tolerance):
            return (
                "FAIL",
                f"性能回归: 当前 GFLOPS={cur.gflops:.2f} 比 baseline {bl.gflops:.2f} 低 {dr*100:.1f}% (容忍阈值 -{tolerance*100:.0f}%)",
                dr,
            )
        dr = (cur.gflops - bl.gflops) / bl.gflops
        return ("PASS",
                f"功能一致, GFLOPS {bl.gflops:.2f}→{cur.gflops:.2f} ({dr*100:+.1f}%)",
                dr)

    dr = 0.0
    if bl.gflops > 0:
        dr = (cur.gflops - bl.gflops) / bl.gflops
    return ("PASS", f"功能一致 (strict_l3=False, 不判性能)", dr)


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

    p_cmp = sub.add_parser("compare", help="跑全部 case, 对比 baseline, FAIL 则 exit=1")
    p_cmp.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                       help=f"性能回归容忍率 (默认 {DEFAULT_TOLERANCE:.0%})")
    p_cmp.add_argument("--case", help="只跑指定 case_id")
    p_cmp.add_argument("--lenient", action="store_true",
                       help="宽松模式: 只判功能回归, 不判 L3 性能回归")

    args = ap.parse_args(argv)
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
        failed = [r for r in reports if r.status == "FAIL"]
        return 1 if failed else 0

    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
