"""高精度参考那把尺 (ρ = d_opt / d_ref) —— KernelOPT 行 372-389 落进 L2，**只观测不判定**

外部一手 (arXiv:2609.30059v1 行 1348-1351，逐字；`‖·‖∞` 的竖线在 HTML 抽取里丢了):
    "In the normal case (dref >= 1e-8), the error ratio rho = dopt/dref passes if rho <= 10;
     in practice, correct TF32 kernels yield rho in [1,3] while algorithmically incorrect
     kernels yield rho > 100."

它与我方 AND 那条门的**根本差别**不是「阈值松紧」而是「比的是什么」: AND 比「候选 vs 参考」，
条带是绝对 atol ⇒ 参考幅值一小就看不见任何东西。`benchmarks/probes/probe_l2_sampling_gate.py` 臂 1/2
拍到过这个现场 —— 全零核在 scale=1e-3 / 1e-6 的输入上被判 **PASS**。这把尺的分母是**参考自己
到 float64 真值的距离**，条带因此自动随幅值缩放。

本机实测 (`2b6a390`+ 本片，numpy float32 参考 vs float64 参考，K=64，零机时):

    scale   候选            L2 判   oracle 反不反对
    1.0     correct         PASS     不反对 (ρ=1)
    1.0     all-zeros       FAIL     反对 (ρ=3.8e6)
    1e-3    correct         PASS     不反对
    1e-3    all-zeros       **PASS** ← 盲区   反对
    1e-3    scaled 0.97     **PASS** ← 盲区   反对
    1e-6    all-zeros       **PASS** ← 盲区   反对
⇒ 三处 AND 看不见的错误答案，这把尺都看见了，而它对正确核**没有**误报。

⚠️ 但它今天**不参与判定**，只上报 + 在 message 里念一句。理由不是胆小：ρ≤10 与 1e-4 这两个阈
是他们 TF32/Triton 语料上的数，我方在 H20 上的 ρ 分布还没测过一次 —— 拿抄来的阈去翻转判决，
就是「把没标定的数当自己的标定」（与 #89「自报只作参考」同形）。定阈要机时，见 #82。

一处本片自己撞出来的缺陷，值得留在测试里：读数是在**每组**那里算的，而聚合处手写的是一份**新的**
metrics ⇒ 第一版把「通过」那一侧的 oracle 键全丢了（值算出来了，死在搬运路上；本仓第三次：
doc 33 的 `runs`/`max_time_ms`、doc 37 的 `llm_cost`、doc 48 的 `candidate_signature`）。
`test_passing_side_carries_the_oracle` 就是那条搬运的证人。

变异标定 (每条只改一处，跑完立刻逐字节复原并断言 sha256 一致。2026-09-30 实跑，基线 49 passed，
**12 条全部拦住**；`tolerance.py` 与 `verifier.py` 两个文件分别核对):
    T1 分支判据 `>=` 改成 `>`                          → 2 条红
    T2 ρ 阈改成严格小于 (`<=` → `<`)                    → 1 条红
    T3 幅值为零时不兜底 (会除零)                         → 1 条红
    T4 三条判据折成一个布尔 (`meets_ratio` 里 OR 上绝对兜底) → 2 条红
    T5 反向: 阈设成 0，逢人便反对                        → 3 条红  ← I27 那条要求的反向变异
    V1 聚合处不再搬 oracle (`**oracle_agg` 删掉)          → 3 条红  ← 「死在搬运路上」的证人
    V2 `any` 改成 `all`                                 → 1 条红
    V3 最差一组改成最好一组 (`max` → `min`)              → 1 条红
    V4 「反对」读成「不反对」(`is False` → `is True`)      → 2 条红
    V5 缺 ρ 也写一个 None 进去                           → 1 条红 (第一版**逃逸**，见下)
    V6b oracle 与参考是同一份 (d_ref 写死 0)              → 2 条红 (第一版是 V6，**逃逸**，见下)
    V7 int8 也去算 float64 参考                          → 1 条红

2026-09-30 第二批 (`_oracle_note` 那处修复, 见 `TestNoteReadsBothShapes`; 基线 22 passed，
**8 条全部拦住**、每轮逐字节复原):
    N1 退回「只读聚合那份」= 当初那行 `（None 支）`本身        → 3 条红  ← 本片真正的牙齿
    N2 反过来一律当逐组那份 (聚合的「最差一组」口径丢了)       → 1 条红
    N3 去掉「最差一组」那半句                                  → 1 条红
    N4 分支名缺失时直接印 None (空证人)                        → 1 条红
    N5 先看幅值再看 ρ (ratio 支被印成幅值倍数)                 → 2 条红
    N6 两把数都没有时沉默收尾 (印一句什么都没支持的话)          → 1 条红
    N7 门拆掉: 没测到 (absent / error) 也逢人便反对            → 1 条红  ← I27 要求的反向变异
    N8 门拆掉: 那把尺不反对时也印反对                          → 2 条红
N1 的红名单里有**唯一一条走真代码**的那条 (`test_real_per_set_path_names_...`)，另外两条是
合成形状 —— 合成那两条在 N1 下也红，所以这条修复有两个证人而不是一个。登记 I29。

两条一开始逃逸的变异，学到的不是「用例写松了」而是两条别的规矩：
  · **V5**：我当时断言的是 `assertNotIn("oracle_rho", 聚合 metrics)` —— 而聚合里本来就只有
    `oracle_rho_max` 没有 `oracle_rho`。那是一条**永真断言**，杀不掉任何东西 (I25 那一族，
    只是这次的「空气」是我自己造的键名)。改打在每组的读数上之后它变 KILLED。
  · **V6**：本想「把 oracle 退回 float32 ⇒ 分母归零」，结果它**什么都没改变**、12 条里唯一全绿的
    就是它。原因是 float32 累加的结果与 float64 累加后再舍入到 float32 仍然不同 —— 这件事本身就是
    关于那把尺的一条事实：**它的分母量的是累加顺序的差，不是存储精度**。换成 V6b (直接把 d_ref
    写死 0) 才有证人。⇒ 变异没杀掉东西时，先怀疑变异，再怀疑用例。
"""
from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from operator_agent import MockGPU, Verifier
from operator_agent.core import tolerance
from operator_agent.core.schemas import (ConstraintSpec, Precision, SourceSpec, TargetSpec,
                                    TaskComplexity, TaskSchema, TaskType)

CORRECT = "import numpy as np\n\ndef kernel(A=None, B=None, **kw):\n    return A @ B\n"
ZEROS = "import numpy as np\n\ndef kernel(A=None, B=None, **kw):\n    return np.zeros_like(A @ B)\n"


def _task(precision=Precision.FP32, max_error=5e-1, reference_source=None):
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.LOW,
                          reference_source=reference_source),
        target=TargetSpec(hardware="mock_gpu", dsl="python"),
        constraints=ConstraintSpec(precision=precision, min_speedup=0.0, max_error=max_error),
    )


def _inputs(scale=1.0, n=32):
    rng = np.random.default_rng(7)
    return {"A": (rng.standard_normal((n, n)) * scale).astype(np.float32),
            "B": (rng.standard_normal((n, n)) * scale).astype(np.float32)}


def _l2(verifier, code, scale, **task_kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return verifier._verify_function(code, _inputs(scale), None, _task(**task_kw))


class TestErrorRatioRule(unittest.TestCase):
    """`tolerance.error_ratio` 的规则本身 (纯标量函数)"""

    def test_ratio_branch_divides_and_compares(self):
        r = tolerance.error_ratio(d_opt=3e-5, d_ref=1e-5, ref_amplitude=28.0)
        self.assertEqual("ratio", r["branch"])
        self.assertAlmostEqual(3.0, r["rho"], places=10)
        self.assertTrue(r["meets_ratio"])

    def test_threshold_is_inclusive_at_ten(self):
        """边界两侧各取一个样本: 只测一侧时 `<` 与 `<=` 互换会绿一半

        ⚠️ 这里刻意用 1e-7 / 1e-8 而不是 2e-7 / 2e-8 —— 后者在二进制浮点下商是
        10.000000000000002, 「恰好在阈上」这条根本测不到 (测到的是二进制表示)。
        """
        self.assertTrue(tolerance.error_ratio(1e-7, 1e-8, 1.0)["meets_ratio"],
                        "ρ 恰为 10 应当算 meets (原文是 rho <= 10)")
        self.assertFalse(tolerance.error_ratio(1.1e-7, 1e-8, 1.0)["meets_ratio"])

    def test_exact_denominator_does_not_divide(self):
        """d_ref < 1e-8 ⇒ **不返回 ρ**。分母不可信时不许替它判 (#64 付过学费)"""
        r = tolerance.error_ratio(d_opt=1e-11, d_ref=1e-11, ref_amplitude=2.8e-5)
        self.assertEqual("exact", r["branch"])
        self.assertIsNone(r["rho"])
        self.assertTrue(r["meets_scale_relative"],
                        "退化分支看的是「误差是参考幅值的多少倍」，correct 应当 meets")

    def test_denominator_exactly_at_the_boundary_takes_the_ratio_branch(self):
        r = tolerance.error_ratio(1e-7, tolerance.ORACLE_EXACT_DENOM, 1.0)
        self.assertEqual("ratio", r["branch"], "原文是 dref >= 1e-8，写成 > 会把边界推进取舍噪声的一支")

    def test_zero_amplitude_does_not_divide_by_zero(self):
        r = tolerance.error_ratio(d_opt=1e-9, d_ref=1e-9, ref_amplitude=0.0)
        self.assertEqual("exact", r["branch"])
        self.assertTrue(np.isfinite(r["scale_relative_error"]))

    def test_three_verdicts_are_not_folded_into_one_boolean(self):
        """原文是三条各自成立的规则，折成一个 `passed` 就把两种事压成一层

        这个样本是专门挑的: ρ 不过 (ratio 支反对) 而绝对兜底过 —— 一个布尔说不清这种局面。
        """
        r = tolerance.error_ratio(d_opt=2e-4, d_ref=1e-5, ref_amplitude=1.0)
        self.assertFalse(r["meets_ratio"])
        self.assertTrue(r["meets_abs_bound"], "d_opt=2e-4 在 max(5e-3, 1e-3·1) 之内")
        self.assertFalse(r["meets_scale_relative"])

    def test_nan_candidate_object_without_raising(self):
        """候选里带 NaN 时不许抛异常 —— 观测性判据没有权利炸掉主流程"""
        r = tolerance.error_ratio(float("nan"), 1e-5, 1.0)
        self.assertFalse(r["meets_ratio"], "NaN 与任何数比较都是 False ⇒ 不许读成通过")


class TestOracleThroughVerifier(unittest.TestCase):
    """过真实的 MockGPU 路径：同一份参考算两遍 (float32 / float64)"""

    def setUp(self):
        self.v = Verifier(MockGPU())

    def test_correct_kernel_is_not_objected_to(self):
        r = _l2(self.v, CORRECT, 1.0)
        self.assertTrue(r.passed, r.message)
        self.assertEqual("measured", r.metrics["ref_oracle"])
        self.assertFalse(r.metrics["oracle_objects"],
                         "对正确核误报 ⇒ 这把尺会变成噪音，下一条用例的对照全靠它")
        self.assertAlmostEqual(1.0, r.metrics["oracle_rho_max"], places=3)
        self.assertNotIn("高精度参考那把尺反对", r.message)

    def test_blind_band_all_zero_kernel_is_objected_to(self):
        """**这一条是整片的目的**: AND 门在 scale=1e-3 上放行了全零核，那把尺看见了

        `assertIsTrue(r.passed)` 不是笔误 —— 它钉的是「今天 L2 仍然判通过」这个事实，
        这样「把 oracle 接成判定」那一天这条会红，逼着那次改动重写它（而不是悄悄改绿）。
        """
        r = _l2(self.v, ZEROS, 1e-3, max_error=1e-1)
        self.assertTrue(r.passed, "前提变了：L2 现在拦住了全零核，这片的存在理由要重审")
        self.assertTrue(r.metrics["oracle_objects"])
        self.assertEqual("exact", r.metrics["oracle_branches"])
        self.assertIn("高精度参考那把尺反对", r.message)
        # ρ 不许被造出来。⚠️ 断言必须打在**每组的读数**上: 聚合那一份本来就只有
        # `oracle_rho_max` 而没有 `oracle_rho`, 在聚合上 assertNotIn("oracle_rho") 是一条
        # 永真断言 —— 第一版就是这么写的, 于是「把 None 也写进去」那条变异逃逸了。
        task, xs = _task(max_error=1e-1), _inputs(1e-3)
        ref32 = self.v._generate_reference(task, xs)
        self.assertIsNotNone(ref32, "构造前提: 内建参考在这个任务上算得出来")
        per_set = self.v._oracle_reading(task, xs, np.zeros_like(ref32), ref32)
        self.assertEqual("exact", per_set["oracle_branch"],
                         f"构造前提: 这个幅值下 d_ref 应当小于 {tolerance.ORACLE_EXACT_DENOM}")
        self.assertNotIn("oracle_rho", per_set)
        self.assertIn("oracle_scale_relative", per_set)
        self.assertTrue(per_set["oracle_objects"])

    def test_passing_side_carries_the_oracle(self):
        """搬运工证人：多组输入全部通过时，聚合 metrics 里必须有 oracle 那几个键

        删掉聚合处的 `**oracle_agg` 时这一条会红 —— 而那正是第一版的形状。
        """
        with contextlib.redirect_stdout(io.StringIO()):
            r = self.v._verify_function(CORRECT, [_inputs(1.0), _inputs(8.0)], None, _task())
        self.assertTrue(r.passed, r.message)
        self.assertEqual(2, r.metrics["oracle_sets_total"])
        self.assertEqual(2, r.metrics["oracle_sets_measured"])
        self.assertIn("oracle_objects", r.metrics)
        self.assertIn("oracle_rho_max", r.metrics)

    def test_absent_oracle_is_reported_as_absent_not_zero(self):
        """外挂 reference_source 与 int8 任务：没有 float64 那份参考 ⇒ 只写 absent

        这里**不许**出现 `oracle_objects` —— 缺证据时写 False 会被读成「那把尺不反对」，
        那是把「没测」读成「测过且通过」(doc 49 §八 那一族)。
        """
        int8 = self.v._oracle_reading(_task(precision=Precision.INT8), _inputs(),
                                      np.zeros((2, 2)), np.zeros((2, 2)))
        self.assertEqual({"ref_oracle": "absent"}, int8)
        ext = self.v._oracle_reading(
            _task(reference_source="def reference(**inputs):\n    return inputs['A'] @ inputs['B']\n"),
            _inputs(), np.zeros((32, 32)), np.zeros((32, 32)))
        self.assertEqual({"ref_oracle": "absent"}, ext)

    def test_broken_reference_reports_type_not_message(self):
        class _Boom(Verifier):
            def _generate_reference(self, task, inputs, high_precision=False):
                if high_precision:
                    raise RuntimeError("远程错误原文，含路径 /private/somewhere/secret")
                return np.zeros((32, 32), dtype=np.float32)

        out = _Boom(MockGPU())._oracle_reading(_task(), _inputs(),
                                               np.zeros((32, 32)), np.zeros((32, 32)))
        self.assertEqual("error:RuntimeError", out["ref_oracle"])
        self.assertNotIn("secret", str(out), "错误原文不许跟着读数进仓库/进 metrics")

    def test_shape_mismatch_is_not_silently_compared(self):
        class _WrongShape(Verifier):
            def _generate_reference(self, task, inputs, high_precision=False):
                return np.zeros((3, 3), dtype=np.float64) if high_precision else None

        out = _WrongShape(MockGPU())._oracle_reading(
            _task(), _inputs(), np.zeros((32, 32), dtype=np.float32),
            np.zeros((32, 32), dtype=np.float32))
        self.assertEqual({"ref_oracle": "shape_mismatch"}, out)


class TestAggregateOraclesShape(unittest.TestCase):
    def _o(self, **kw):
        # 刻意**不**放 oracle_rho: 那是只有 ratio 支才有的键, 放在夹具底座上会让
        # 「没有 ρ 时不许发布这个键」那条用例假绿 (第一版就是这样, 是它自己红出来的)
        base = {"ref_oracle": "measured", "oracle_branch": "ratio",
                "oracle_scale_relative": 1e-9, "oracle_objects": False}
        base.update(kw)
        return base

    def test_no_readings_produce_nothing(self):
        self.assertEqual({}, Verifier._aggregate_oracles([]))
        self.assertEqual({"ref_oracle": "absent"},
                         Verifier._aggregate_oracles([{"ref_oracle": "absent"}]))

    def test_one_objecting_set_objects_for_the_suite(self):
        """any() 的方向: 一组反对就说明这把尺与本级判决不一致，不能被别组平均掉

        两组**都**带 ρ (1.0 与 400.0) 而不是只给一组: 只有一组有值时 `max` 与 `min` 读出来
        同一个数, 「取最差一组」这条断言就杀不掉任何变异。
        """
        agg = Verifier._aggregate_oracles([self._o(oracle_rho=1.0),
                                           self._o(oracle_objects=True, oracle_rho=400.0)])
        self.assertTrue(agg["oracle_objects"])
        self.assertEqual(400.0, agg["oracle_rho_max"], "取最差一组而不是平均/最好")
        self.assertEqual(1e-9, agg["oracle_scale_relative_max"])

    def test_partially_measured_is_visible_and_rho_absent_when_no_set_has_it(self):
        agg = Verifier._aggregate_oracles([self._o(oracle_branch="exact"),
                                           {"ref_oracle": "absent"}])
        self.assertEqual("partially_measured", agg["ref_oracle"])
        self.assertEqual((1, 2), (agg["oracle_sets_measured"], agg["oracle_sets_total"]))
        self.assertNotIn("oracle_rho_max", agg, "没有任何一组有 ρ 时不许发布这个键")


class TestNoteReadsBothShapes(unittest.TestCase):
    """`_oracle_note` 有两种输入形状: 逐组那份的键名没有 `_max`/`_es` 后缀

    这一组是补的账 —— 第一版这个 formatter 只读聚合那一份的键名, 于是**逐组那一路印出来的
    是「（None 支）」而且幅值那半句整个消失**: 一句话印了, 却什么都没支持。2026-09-30 做
    #94 的探针 (`参考自己产 NaN` 那一组读数) 时才看见那行 `None 支`, 而它没被任何用例挡住 ——
    因为原有那条只核了聚合路径 (`_verify_function` 会重算 message 并丢掉逐组那句)。
    今天逐组那句只有直接调 `_verify_single_set` 的探针/单测看得见, 但 #94 正要在同一个函数上
    动参考侧的检查, 留一个「读不到就不说话」的 formatter 等于把这条错句子交接下去。
    """

    def setUp(self):
        self.v = Verifier(MockGPU())

    def test_real_per_set_path_names_the_branch_and_the_magnitude(self):
        """**这一条才是那行 `None 支` 的证人**: 走真代码, 不发一分钱的数据"""
        task, xs = _task(max_error=1e-1), _inputs(1e-3)
        ref = self.v._generate_reference(task, xs)
        self.assertIsNotNone(ref, "构造前提: 内建参考在这个任务上算得出来")
        with contextlib.redirect_stdout(io.StringIO()):
            r = self.v._verify_single_set(ZEROS, xs, ref, task)
        self.assertTrue(r.passed, "前提: AND 那条在这个幅值上仍放行全零核")
        self.assertIn("exact 支", r.message, r.message)
        self.assertNotIn("None 支", r.message)
        self.assertIn("误差是参考幅值的", r.message)
        self.assertNotIn("最差一组", r.message,
                         "逐组那一路只有这一组, 加「最差一组」会读成还有更差的组")

    def test_aggregate_shape_keeps_the_worst_set_qualifier(self):
        note = self.v._oracle_note({
            "ref_oracle": "measured", "oracle_objects": True,
            "oracle_branches": "ratio", "oracle_rho_max": 400.0,
            "oracle_scale_relative_max": 1e-9})
        self.assertIn("ratio 支", note)
        self.assertIn("最差一组ρ=400", note)
        self.assertNotIn("None", note)

    def test_per_set_shape_reports_rho_without_the_aggregate_wording(self):
        note = self.v._oracle_note({
            "ref_oracle": "measured", "oracle_objects": True,
            "oracle_branch": "ratio", "oracle_rho": 38.0,
            "oracle_scale_relative": 1e-9})
        self.assertIn("ratio 支", note)
        self.assertIn("ρ=38", note)
        self.assertNotIn("最差一组", note)

    def test_no_magnitude_available_says_so_instead_of_trailing_off(self):
        """两把数都拿不到时那句得自己承认, 不是印一句像结论的话 (缺证据不当证据)"""
        note = self.v._oracle_note({"ref_oracle": "measured", "oracle_objects": True,
                                    "oracle_branch": "exact"})
        self.assertIn("exact 支", note)
        self.assertIn("没给出可比的数", note)

    def test_branch_name_missing_does_not_print_a_bare_none(self):
        note = self.v._oracle_note({"ref_oracle": "measured", "oracle_objects": True,
                                    "oracle_rho": 12.0})
        self.assertIn("分支未知", note)
        self.assertNotIn("None", note)

    def test_absent_or_unobjecting_readings_print_nothing(self):
        """成对的那一侧: 这两类不许印句子, 否则「反对」与「没测」在日志里长得一样"""
        for m in ({"ref_oracle": "absent", "oracle_objects": True},
                  {"ref_oracle": "error:ZeroDivisionError", "oracle_objects": True},
                  {"ref_oracle": "measured", "oracle_objects": False, "oracle_rho": 3.0}):
            self.assertEqual("", self.v._oracle_note(m), m)


if __name__ == "__main__":
    unittest.main()
