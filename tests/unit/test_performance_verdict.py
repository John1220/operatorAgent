"""§16 收口: 性能面的**成功标准**是什么（切片 J / CHECKLOG I95）

设计 §16 的反模式原话是「**用单形状的加速比作为成功标准**」; §3.4 说性能 = `cases` + 权重 +
聚合成目标函数; §6.2 条款 3 说逐 shape 的硬回退**优先于**聚合。本片之前这三句话都没有判据面:
L3 只有 `passed = speedup >= min_required`（主 case 那一个数），契约里声明了权重与口径也只被
发布成**数据**（I93 登记过这条边界: 「目标函数是数据不是门」）。

这一片把成功标准收进一处（`eligibility.performance_verdict`），并规定: 契约声明了目标函数
（权重或口径有其一）⇒ **判目标函数**，未确立 ⇒ **不通过**（fail-closed: 声明了就是承诺按它
评，「评不出来」不是「没有判据」而是「没有证据说达标」）; 没声明 ⇒ 退回主 case 的口径，并
把「§16 没对它收口」写进产物与报告（单形状成功标准仍然可能发生，但它必须是看得见的）。

四组, 各挡一类失效:

A. 纯函数: 「声明/没声明/只声明了一半」三种形状的判决 + 未确立一律不通过 + 逐 case 回退越限
   优先于聚合 + **目标函数不回头要求主 case 也达标**（拿单形状当标准正是 §16 的反模式）。
B. 门（`Verifier._verify_performance` + 确定性载荷）: 主 case 2.0x ≥ 门槛 1.5、另一条 0.5x —
   权重压在那条慢的 case 上 ⇒ 聚合 0.53 < 1.5 ⇒ **L3 不通过**（今天会过: 这正是反模式）;
   权重反过来 ⇒ 聚合 1.87 ≥ 1.5 ⇒ 通过（另一条自己 < 门槛也不挡）—— 判的是**分布**。
C. 三态与来源: 声明了目标函数却没量全 ⇒ 不通过; 契约冻不住 ⇒ 读任务里的原始声明并登记;
   没声明的契约 ⇒ `basis="primary_case"`。
D. 结构: 判据 / 目标函数 / 账本 / 回退各一处生产; 评测侧不再有「主 case 那一个数就是标准」;
   报告只读那一格、不重算。

运行: python3 -m unittest tests.test_performance_verdict -v
"""

import contextlib
import io
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from operator_agent.core.backend_caps import Cap
from operator_agent.decide.eligibility import (PERF_BASIS_OBJECTIVE, PERF_BASIS_PRIMARY_CASE,
                                        performance_verdict)
from operator_agent.core.schemas import (ConstraintSpec, Precision, SourceSpec, TargetSpec,
                                    TaskComplexity, TaskSchema, TaskType)
from operator_agent.decide.verifier import Verifier
from tests.backend_fakes import DeclaredFakeBackend

CASE_A = "A=16x16|B=16x16"
CASE_B = "A=32x32|B=32x32"
BASELINE_MS = 0.400
#: 逐 case 的候选时间: 16x16 比基线快 2 倍, 32x32 慢一倍 (0.5x)。**确定性**载荷,
#: 所以判决的算术可以手算复核 (不是「跑出来是多少就是多少」)。
CANDIDATE_MS = {16: 0.200, 32: 0.800}
MIN_SPEEDUP = 1.5

CODE = "def kernel(A, B):\n    return A @ B\n"


def _inputs(n):
    import numpy as np
    return {"A": np.ones((n, n), dtype=np.float32),
            "B": np.ones((n, n), dtype=np.float32)}


def _expected_weighted_geomean(pairs):
    total = sum(w for _, w in pairs)
    return math.exp(sum(w * math.log(v) for v, w in pairs) / total)


class TestTheVerdictIsPure(unittest.TestCase):
    """A 组: 判据面（`eligibility.performance_verdict`）"""

    def test_no_declaration_is_the_primary_criterion(self):
        passed, basis, why, value = performance_verdict(MIN_SPEEDUP, primary_speedup=2.0)
        self.assertIs(True, passed)
        self.assertEqual(PERF_BASIS_PRIMARY_CASE, basis)
        self.assertEqual(2.0, value)
        self.assertIn("§16 未收口", why, "退回单形状必须**说出来**")

    def test_no_declaration_and_no_reading_does_not_pass(self):
        passed, basis, _why, value = performance_verdict(MIN_SPEEDUP)
        self.assertIs(False, passed)
        self.assertEqual(PERF_BASIS_PRIMARY_CASE, basis)
        self.assertIsNone(value)

    def test_a_primary_case_below_the_floor_fails(self):
        self.assertIs(False, performance_verdict(MIN_SPEEDUP, primary_speedup=1.2)[0])

    def test_weights_alone_are_a_declaration(self):
        passed, basis, why, _ = performance_verdict(
            MIN_SPEEDUP, case_weights=((CASE_A, 1.0),), case_speedups={CASE_A: 2.0})
        self.assertIs(False, passed, "只声明权重、没声明口径 ⇒ 目标函数未确立 ⇒ 不通过")
        self.assertEqual(PERF_BASIS_OBJECTIVE, basis)
        self.assertIn("目标函数未确立", why)

    def test_aggregate_alone_is_a_declaration(self):
        passed, basis, _why, value = performance_verdict(
            MIN_SPEEDUP, aggregate="weighted_geomean", case_speedups={CASE_A: 2.0})
        self.assertIs(False, passed)
        self.assertEqual(PERF_BASIS_OBJECTIVE, basis)
        self.assertIsNone(value)

    def test_an_established_objective_is_the_criterion(self):
        weights = ((CASE_A, 3.0), (CASE_B, 1.0))
        want = _expected_weighted_geomean(((2.0, 3.0), (1.0, 1.0)))
        passed, basis, why, value = performance_verdict(
            MIN_SPEEDUP, aggregate="weighted_geomean", case_weights=weights,
            case_speedups={CASE_A: 2.0, CASE_B: 1.0})
        self.assertAlmostEqual(want, value, places=9)
        self.assertIs(True, passed)
        self.assertEqual(PERF_BASIS_OBJECTIVE, basis)
        self.assertIn("weighted_geomean", why)

    def test_the_objective_does_not_also_require_the_primary_case(self):
        """主 case 只是分布里的一条 —— 拿它当标准正是 §16 的反模式"""
        passed, basis, _why, _v = performance_verdict(
            MIN_SPEEDUP, aggregate="weighted_geomean", case_weights=((CASE_A, 1.0),),
            case_speedups={CASE_A: 3.0}, primary_speedup=0.5)
        self.assertIs(True, passed)
        self.assertEqual(PERF_BASIS_OBJECTIVE, basis)

    def test_an_objective_below_the_floor_fails(self):
        self.assertIs(False, performance_verdict(
            MIN_SPEEDUP, aggregate="mean", case_weights=((CASE_A, 1.0),),
            case_speedups={CASE_A: 1.0}, primary_speedup=9.0)[0])
        # 主 case 9 倍也救不了「聚合不够」—— 判的是分布

    def test_an_unestablished_objective_fails_closed(self):
        passed, basis, why, _ = performance_verdict(
            MIN_SPEEDUP, aggregate="mean", case_weights=((CASE_A, 1.0), (CASE_B, 1.0)),
            case_speedups={CASE_A: 2.0})
        self.assertIs(False, passed, "量不全 ⇒ 没有证据说达标")
        self.assertEqual(PERF_BASIS_OBJECTIVE, basis)
        self.assertIn("未确立", why)

    def test_a_non_positive_case_blocks_the_geomean(self):
        passed, _basis, why, value = performance_verdict(
            MIN_SPEEDUP, aggregate="weighted_geomean", case_weights=((CASE_A, 1.0),),
            case_speedups={CASE_A: 0.0})
        self.assertIs(False, passed)
        self.assertIsNone(value)
        self.assertIn("未确立", why)

    def test_a_regression_over_limit_beats_the_aggregate(self):
        weights = ((CASE_A, 1.0),)
        ok, _b, _w, value = performance_verdict(
            MIN_SPEEDUP, aggregate="mean", case_weights=weights,
            case_speedups={CASE_A: 3.0})
        self.assertIs(True, ok)
        passed, basis, why, _v = performance_verdict(
            MIN_SPEEDUP, aggregate="mean", case_weights=weights,
            case_speedups={CASE_A: 3.0}, max_regression=((CASE_A, 0.1),),
            regression_over_limit_value=True)
        self.assertIs(False, passed, "§6.2 条款 3: 逐 shape 硬回退优先于聚合")
        self.assertEqual(PERF_BASIS_OBJECTIVE, basis)
        self.assertIn("条款 3", why)
        self.assertIsNotNone(value)

    def test_zero_floor_passes_any_established_objective(self):
        self.assertIs(True, performance_verdict(
            0.0, aggregate="mean", case_weights=((CASE_A, 1.0),),
            case_speedups={CASE_A: 0.5})[0])


class TestThroughTheGate(unittest.TestCase):
    """B/C 组: 真过 L3 门（确定性载荷: 16x16 = 2.0x, 32x32 = 0.5x）"""

    def _payload(self, inputs):
        n = int(inputs["A"].shape[0])
        ms = CANDIDATE_MS[n]
        return {
            "success": True, "device": "cuda", "gflops": 100.0,
            "candidate": {"min_time_ms": ms, "median_time_ms": ms, "avg_time_ms": ms,
                          "max_time_ms": ms, "runs": 30},
            "baselines": {"pytorch_eager": {"median_time_ms": BASELINE_MS,
                                            "min_time_ms": BASELINE_MS}},
            "baseline_selected": "pytorch_eager", "paired": {},
        }

    def _l3(self, task, sets):
        backend = DeclaredFakeBackend(
            caps={Cap.REAL_DEVICE},
            benchmark_with_baselines=lambda *a, **kw: self._payload(a[1]),
            get_hardware_info=lambda: {"name": "NVIDIA H20", "device": "cuda",
                                       "compute_capability": "9.0"})
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = Verifier(backend)._verify_performance(CODE, sets, 0.0, task)
        return r, r.metrics, buf.getvalue()

    def _task(self, weights=None, aggregate="", max_regression=None, min_speedup=MIN_SPEEDUP):
        return TaskSchema(
            task_type=TaskType.OPERATOR_GENERATION,
            source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.MEDIUM),
            target=TargetSpec(hardware="nvidia_gpu", dsl="python"),
            constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=min_speedup,
                                       max_error=1e-2, case_weights=dict(weights or {}),
                                       aggregate=aggregate,
                                       max_regression=dict(max_regression or {})))

    def test_without_a_declaration_the_primary_case_is_the_criterion(self):
        r, m, out = self._l3(self._task(), [_inputs(16), _inputs(32)])
        v = m["performance_verdict"]
        self.assertIs(True, v["passed"])
        self.assertEqual(PERF_BASIS_PRIMARY_CASE, v["basis"])
        self.assertIs(True, r.passed)
        self.assertIn("§16 未收口", out, "退回单形状必须在日志里看得见")

    def test_the_objective_is_the_gate_when_declared(self):
        # 主 case 2.0x 够门槛, 但权重压在 0.5x 那条上 ⇒ 聚合 0.53 ⇒ **不通过**。
        # 本片之前这一次会过 —— 那正是 §16 点名的「用单形状的加速比作为成功标准」。
        task = self._task(weights={CASE_A: 0.05, CASE_B: 1.0},
                          aggregate="weighted_geomean")
        r, m, _out = self._l3(task, [_inputs(16), _inputs(32)])
        v = m["performance_verdict"]
        self.assertEqual(PERF_BASIS_OBJECTIVE, v["basis"])
        self.assertIs(False, v["passed"])
        self.assertIs(False, r.passed, "门必须真的挡住")
        want = _expected_weighted_geomean(((2.0, 0.05), (0.5, 1.0)))
        self.assertAlmostEqual(want, v["value"], places=6)
        self.assertGreater(m["speedup_vs_baseline"], MIN_SPEEDUP,
                           "主 case 那条读数确实够门槛 —— 挡住它的是目标函数")

    def test_a_distribution_beats_the_slow_case_when_the_weights_say_so(self):
        task = self._task(weights={CASE_A: 1.0, CASE_B: 0.05},
                          aggregate="weighted_geomean")
        r, m, _out = self._l3(task, [_inputs(16), _inputs(32)])
        self.assertIs(True, m["performance_verdict"]["passed"])
        self.assertIs(True, r.passed)
        self.assertEqual(0.5, m["cases"][CASE_B]["speedup_vs_baseline"])
        # 32x32 自己 < 门槛, 但权重说它只占 5% ⇒ 判的是分布, 不是任何单条。

    def test_an_objective_that_cannot_be_established_fails_the_gate(self):
        task = self._task(weights={CASE_A: 1.0, CASE_B: 1.0}, aggregate="mean")
        r, m, _out = self._l3(task, [_inputs(16)])
        v = m["performance_verdict"]
        self.assertIs(False, v["passed"], "声明了目标函数却只量到一条 ⇒ 没有证据说达标")
        self.assertIs(False, r.passed)
        self.assertIsNone(v["value"])

    def test_a_regression_over_the_declared_limit_fails_the_gate(self):
        task = self._task(weights={CASE_A: 1.0, CASE_B: 1.0}, aggregate="worst_case",
                          max_regression={CASE_B: 0.1})
        r, m, _out = self._l3(task, [_inputs(16), _inputs(32)])
        v = m["performance_verdict"]
        self.assertIs(True, v["regression_over_limit"], "0.5x = 回退 50% > 上限 10%")
        self.assertIs(False, v["passed"], "§6.2 条款 3: 回退越限优先于聚合")
        self.assertIs(False, r.passed)

    def test_the_verdict_is_published_even_when_earlier_gates_refuse(self):
        r, m, _out = self._l3(self._task(), [_inputs(16), _inputs(32)])
        self.assertIn("performance_verdict", m)
        self.assertEqual(2, m["performance_verdict"]["cases"])

    def test_an_unfreezable_contract_does_not_silently_weaken_the_criterion(self):
        # 权重指向一条**没在 cases 里声明**的 case ⇒ I94 的 lint 拒绝冻结。
        # 那时判据面读任务里的原始声明: 「声明了目标函数」这件事不许因为冻不住就消失。
        task = self._task(weights={"A=99x99|B=99x99": 1.0}, aggregate="mean")
        task.constraints.cases = ({"name": "x", "seed": 0, "tensors": [
            {"name": "A", "shape": [16, 16], "dtype": "fp32"},
            {"name": "B", "shape": [16, 16], "dtype": "fp32"}]},)
        r, m, _out = self._l3(task, [_inputs(16)])
        v = m["performance_verdict"]
        self.assertIs(False, v["contract_frozen"], "这一份契约冻不住")
        self.assertEqual(PERF_BASIS_OBJECTIVE, v["basis"],
                         "声明了目标函数就是声明了 —— 不许退回单形状")
        self.assertIs(False, v["passed"])
        self.assertIs(False, r.passed)
        self.assertIn("契约不可判定", v["why"])

    def test_the_verdict_names_its_criteria_source(self):
        _r, m, _out = self._l3(self._task(weights={CASE_A: 1.0}, aggregate="mean"),
                               [_inputs(16)])
        self.assertIs(True, m["performance_verdict"]["contract_frozen"],
                      "判据面读的是冻结后的契约, 不是任务原始字段")

class TestTheAccountIsThreeState(unittest.TestCase):
    """账本收集（I95 搬到判据层）: 「量过但没得出」不许被当成 0"""

    def test_a_ledger_entry_without_a_number_stays_none(self):
        from operator_agent.decide.eligibility import case_speedups
        out = case_speedups({"cases": {CASE_A: {"speedup_vs_baseline": None}}})
        self.assertIn(CASE_A, out)
        self.assertIsNone(out[CASE_A], "量过但没得出 ≠ 0")

    def test_the_hint_is_only_used_when_there_is_no_ledger(self):
        from operator_agent.decide.eligibility import case_speedups
        self.assertEqual({CASE_A: 2.0}, case_speedups(
            {"cases": {CASE_A: {"speedup_vs_baseline": 2.0}}},
            case_key_hint=CASE_B, speedup_hint=9.0))
        self.assertEqual({CASE_B: 9.0}, case_speedups({}, case_key_hint=CASE_B,
                                                     speedup_hint=9.0))
        self.assertEqual({}, case_speedups({}))


class TestTheReportSaysWhichStandardWasUsed(unittest.TestCase):
    """报告那一行（只读 `performance_verdict`，不重算）—— 三态"""

    def _lines(self, **m):
        from operator_agent.core.report import _performance_verdict_lines
        return "\n".join(_performance_verdict_lines(m))

    def test_the_objective_basis_is_printed(self):
        txt = self._lines(performance_verdict={"basis": "objective", "passed": True,
                                              "value": 1.8721, "min_speedup": 1.5,
                                              "why": "目标函数 weighted_geomean ...",
                                              "cases": 2, "regression_over_limit": False})
        self.assertIn("**目标函数**", txt)
        self.assertIn("1.872", txt)
        self.assertIn("⇒ 过", txt)
        self.assertIn("逐 case 硬回退判据: 否", txt)

    def test_the_primary_case_basis_prints_the_open_debt(self):
        txt = self._lines(performance_verdict={"basis": "primary_case", "passed": True,
                                              "min_speedup": 1.5, "why": "退回主 case ..."})
        self.assertIn("**主 case 加速比**", txt)
        self.assertIn("§16 未对它收口", txt)

    def test_an_old_payload_without_the_key_prints_nothing(self):
        self.assertEqual("", self._lines())
        self.assertEqual("", self._lines(performance_verdict=None))
    def test_a_partial_payload_is_never_printed_as_a_pass(self):
        txt = self._lines(performance_verdict={"basis": "objective"})
        self.assertIn("未确立", txt, "没说过的结论不许印成「过」")
        self.assertNotIn("⇒ 过", txt)

class TestOneProducerPerFact(unittest.TestCase):
    """D 组: 结构 —— 判据 / 账本 / 回退 / 目标函数各一处, 门里不再有「主 case 就是标准」"""

    def _src(self, rel):
        return (Path(__file__).resolve().parents[2] / rel).read_text(encoding="utf-8")

    def test_the_verdict_has_one_definition(self):
        for rel in ("operator_agent/decide/verifier.py", "operator_agent/app/conductor.py",
                    "operator_agent/core/report.py"):
            self.assertNotIn("def performance_verdict(", self._src(rel), rel)
        self.assertEqual(1, self._src("operator_agent/decide/eligibility.py")
                         .count("def performance_verdict("))

    def test_the_gate_applies_the_verdict_in_one_method(self):
        ve = self._src("operator_agent/decide/verifier.py")
        self.assertEqual(1, ve.count("def _apply_performance_verdict("))
        self.assertEqual(3, ve.count("self._apply_performance_verdict("),
                         "远端、本机、有声明目标却缺基线的拒绝出口")

    def test_the_primary_speedup_is_no_longer_the_success_criterion(self):
        ve = self._src("operator_agent/decide/verifier.py")
        self.assertNotIn("passed = speedup >= min_required", ve,
                         "§16 的反模式就是这一行")
        self.assertEqual(2, ve.count("passed = True\n                if _win_scope")
                         + ve.count("passed = True\n            base_ms"),
                         "两条路都只立前置条件（判决在返回前那一处算）")

    def test_the_report_only_reads_the_published_verdict(self):
        rp = self._src("operator_agent/core/report.py")
        self.assertIn('_get', rp)
        for forbidden in ("aggregate_speedup(", "performance_verdict(", "case_speedups("):
            self.assertNotIn(forbidden, rp, forbidden)
        self.assertIn('m.get("performance_verdict")', rp)

    def test_the_verdict_publishes_the_registered_fields(self):
        ve = self._src("operator_agent/decide/verifier.py")
        for field in ("\"basis\"", "\"passed\"", "\"value\"", "\"why\"",
                      "\"min_speedup\"", "\"regression_over_limit\"", "\"contract_frozen\""):
            self.assertIn(field, ve, field)

    def test_the_regression_judgement_stays_in_one_place(self):
        ve = self._src("operator_agent/decide/verifier.py")
        self.assertIn("regression_over_limit(limits, speedups)", ve)
        self.assertNotIn("1 - speedup >", ve, "越限的算术只许在判据层写一遍")


if __name__ == "__main__":
    unittest.main(verbosity=2)
