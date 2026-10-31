"""参考自己产 NaN/Inf 时，L2 的坏点统计与分母 (#94 第 1 片)

外部一手 (arXiv:2609.33074v1 行 143-148，KernelZero 给 proposal 的三段验证，逐字)：
    "(3) Numerical Stability. ... Any sample that produces invalid values such as NaN or
     Inf is filtered out."

他们做在**任务侧**：一个自己会产 NaN 的 module 根本不进训练池，于是它的失败不会被算成
候选的失败。我方原来只在**候选侧**查有限性 (`verifier.py` 的 `has_nonfinite` 看的是
`actual_f`)，参考那份没人查。后果不是「错核被放行」那么轻，而是**分母在撒谎**。

改之前的实测读数 (2026-09-30 控制机 numpy + MockGPU，`scale=1.0` 的 16×16、把参考第 0 行
全置 NaN/Inf ⇒ 256 个元素里 16 个非有限)：

    参考含 NaN   passed=True  bad_elements=0  matched_ratio=1.0  max_error=nan  compared=256
    参考含 Inf   passed=True  bad_elements=0  matched_ratio=1.0  max_error=inf  compared=256

两个方向的失效不一样，这是读代码时才看清的，也是下面每个方向各有一条用例的理由：
  · NaN ⇒ `abs_error` 是 NaN ⇒ `NaN > atol` **为假**；
  · Inf ⇒ `abs_error` 是 inf、过第一道，但 `rel_error = inf/inf = NaN` ⇒ 第二道为假。
于是 AND 形式「两条都超才算坏点」在这里变成**任一侧出非有限值就一票否决坏点**，而那 16 个
元素照样留在 `compared_elements` 里。

本片做的是**测量诚实性**：比得上的元素显式成一侧掩码，`compared_elements`/`coverage`/
`max_error`/`matched_ratio` 全部只数那些，并新上报 `reference_nonfinite`。
⚠️ 仍**不改判决**：一个正确的候选在坏参考下今天照样 `passed=True` (它比参考能比的那 240 个
元素确实对上了)。把「本级没有对象」接成判定，需要给失败一个**归属**（`fixer.py:99` 那套按
「哪一级报的失败」分支的机制今天会把坏参考当成坏候选，发出一份指向错误对象的处方）⇒
第 2 片，与远程那一路 (`remote_scripts/run_kernel.py` 同一个形状) 一起，见 doc 58 §4.2。

变异标定 (每条只改一处，跑完立刻逐字节复原并断言 sha256 一致。2026-09-30 实跑，
基线 13 passed，**11/12 拦下，1 条逃逸** —— 逃逸那条的理由写在它下面，不是漏了用例：
R 系列打在 AND 那条，S 系列打在采样那条，A 系列打在聚合搬运)：
    R1 「比得上」的掩码不生效 (`& comparable` 去掉)            → **0 红 (逃逸)**
    R2 分母退回 `expected.size` (改 `evaluate` 的第二个实参)     → 2 红
    R3 `reference_nonfinite` 不上报 (整键删掉)                  → 5 红
    R4 全非有限时读成「候选 0/0 个坏点 = 通过」 (`tolerance.evaluate`) → 1 红
    R5 `max_error` 退回不筛可比元素 (NaN 又出来)                → 1 红
    R6 `coverage` 写死 1.0 (分母与覆盖率不同源)                 → 1 红
    S1 采样那条路的窗口掩码去掉                                 → 2 红
    S2 候选自己产 NaN 被当参考的错 (归属混了)                    → 1 红
    S3 采样那条路的非有限计数不报 (删键)                         → 2 红
    A1 聚合处不再搬 `reference_nonfinite` (死在搬运路上)         → 3 红
    A2 一组都没报时发布 0 (把没检查读成查过且干净)               → 1 红
    A3 只写进 metrics、message 不喊 (第 1 片那条 M4 的续)        → 1 红

R1 为什么杀不掉 (文件头第 17-21 行那两个方向已经把话说完了)：AND 那条要置坏点必须
`abs_error > atol` 与 `rel_error > rtol` **同时为真**，而参考非有限时这两个数里必有一个是
NaN —— NaN 与任何数比较都是假 ⇒ 那些元素在**加掩码之前就已经不是坏点**。所以 `& comparable`
对分子是恒等的，只对分母 (R2) 和上报 (R3/R5/R6) 有牙齿。这不是「一条任何变异都杀不掉的用例」
那一族，而是**一行任何用例都撞不红的代码**：处置写在 `verifier.py` 那句注释里 (判据形状一改
它立刻变成承重墙，而那一步由 `tests/unit/test_l2_tolerance.py` 的行为用例把守 —— 所以这片不为它
补假用例)。

R2/R3 各 2 与 5 条红、A1 3 条 ⇒ 这三处是这片真正的承重墙；R4 只有 1 条 (全 NaN 那条用例)
是唯一打在 `tolerance.py` 空分母分支上的证人，与 #87 片 1 的 M2 同一族。
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
from operator_agent.execute.remote_executor import RemoteExecutionResult
from operator_agent.core.schemas import (ConstraintSpec, Precision, SourceSpec, TargetSpec,
                                    TaskComplexity, TaskSchema, TaskType,
                                    VerificationLevel, VerificationResult)
from tests.backend_fakes import DeclaredFakeBackend

CORRECT = "import numpy as np\n\ndef kernel(A, B):\n    return A @ B\n"
SCALED = "import numpy as np\n\ndef kernel(A, B):\n    return (A @ B) * 0.97\n"
ZEROS = "import numpy as np\n\ndef kernel(A, B):\n    return np.zeros_like(A @ B)\n"

#: 16×16 的 matmul ⇒ 256 个输出元素；下面把第 0 行 (16 个) 弄成非有限
N_TOTAL = 256
N_REF_BAD = 16
N_COMPARED = N_TOTAL - N_REF_BAD


def _task(max_error=1e-4, precision=Precision.FP32):
    return TaskSchema(task_id="ref-nonfinite", task_type=TaskType.OPERATOR_GENERATION,
                      source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.LOW),
                      target=TargetSpec(hardware="mock_gpu", dsl="python"),
                      constraints=ConstraintSpec(precision=precision, min_speedup=0.0,
                                                 max_error=max_error))


def _inputs(n=16, seed=3):
    rng = np.random.default_rng(seed)
    return {"A": rng.standard_normal((n, n)).astype(np.float32),
            "B": rng.standard_normal((n, n)).astype(np.float32)}


def _truth(n=16):
    """那份**干净**的参考 (与 `_inputs()` 同一份 seed, 所以 `CORRECT` 逐元素相等)"""
    rng = np.random.default_rng(3)
    return rng.standard_normal((n, n)).astype(np.float32) @ rng.standard_normal((n, n)).astype(np.float32)


def _ref_with(kind, n=16):
    """把参考的第 0 行 (16 个元素) 换成 NaN/Inf —— 候选在那一行上无从比对"""
    return np.where(np.indices(_truth(n).shape)[0] == 0, np.asarray(kind, np.float32), _truth(n))


def _l2(verifier, code, expected):
    with contextlib.redirect_stdout(io.StringIO()):
        return verifier._verify_single_set(code, _inputs(), expected, _task())


class TestAndGateDenominator(unittest.TestCase):
    """AND 那条 (全量比对): 分母只数比得上的元素"""

    def setUp(self):
        self.v = Verifier(MockGPU())

    def test_clean_reference_publishes_the_full_denominator_and_zero(self):
        """正向对照 (两侧取样的另一侧): 参考干净时一切照旧满, 且 `reference_nonfinite=0`

        这里 0 是**真读数** (查过了, 没有), 与「没查所以不写这个键」不同 —— 后者见下面
        `test_aggregate_does_not_publish_zero_when_nobody_reported`。
        """
        r = _l2(self.v, CORRECT, _truth())
        self.assertTrue(r.passed, r.message)
        self.assertEqual(N_TOTAL, r.metrics["compared_elements"])
        self.assertEqual(0, r.metrics["reference_nonfinite"])
        self.assertEqual(1.0, r.metrics["coverage"])
        self.assertNotIn("非有限值", r.message)

    def test_nan_in_reference_is_excluded_from_both_numerator_and_denominator(self):
        m = _l2(self.v, CORRECT, _ref_with(np.nan)).metrics
        self.assertEqual(N_REF_BAD, m["reference_nonfinite"])
        self.assertEqual(N_COMPARED, m["compared_elements"],
                         "分母还在数那些 NaN 元素 ⇒ 「比了 256 个」是一句谎话")
        self.assertAlmostEqual(N_COMPARED / N_TOTAL, m["coverage"], places=6)
        # R5 的证人: 参考的 NaN 以前会把 max_error 污染成 nan
        self.assertTrue(np.isfinite(m["max_error"]), f"max_error={m['max_error']!r}")
        self.assertEqual(0.0, m["max_error"])

    def test_inf_in_reference_fails_the_same_way_not_the_same_reason(self):
        """两个方向各一条: NaN 死在绝对项, Inf 死在相对项 (`inf/inf = NaN`)

        只在一种方向上取样, 「把两个都当坏点」和「都不当坏点」都能绿一半。
        """
        m = _l2(self.v, CORRECT, _ref_with(np.inf)).metrics
        self.assertEqual(N_REF_BAD, m["reference_nonfinite"])
        self.assertEqual(N_COMPARED, m["compared_elements"])
        self.assertTrue(np.isfinite(m["max_rel_error"]),
                        f"max_rel_error={m['max_rel_error']!r}")

    def test_honest_denominator_changes_the_reported_ratio_not_the_verdict(self):
        """稀释的真实代价: 12 个坏点在 240 个可比元素上是 5%, 在 256 个上是 4.7%

        这张容差表里 rho=1.0 ⇒ 两种分母都判不过, 所以本片**没有**改任何判决 (下面那条
        成对用例把这件事钉死)。但 `matched_ratio` 是一个会被读、会被写进语料的数,
        它必须说「在能比的那些里错了多少」。
        """
        # 这一份候选把第 0、1 行都清零: 第 0 行本来就无从比对 (参考是 NaN), 第 1 行是
        # **16 个真坏点** —— 于是坏点数与参考坏点数恰好相等, 正是稀释最容易看不见的情形
        code = ("import numpy as np\n\ndef kernel(A, B):\n    return "
                "np.where(np.indices(A.shape)[0] <= 1, 0.0, A @ B)\n")
        r = _l2(self.v, code, _ref_with(np.nan))
        self.assertFalse(r.passed, "第 1 行是可比元素且被清零 ⇒ 该判不过")
        self.assertEqual(16, r.metrics["bad_elements"])
        self.assertAlmostEqual(1.0 - 16.0 / N_COMPARED, r.metrics["matched_ratio"], places=9,
                               msg="分母还在用 256 ⇒ 坏点比例被参考自己的 NaN 稀释")
        self.assertNotAlmostEqual(1.0 - 16.0 / N_TOTAL, r.metrics["matched_ratio"], places=9,
                                  msg="这一半是旧口径的数, 出现在这里就说明分母没换过来")

    def test_candidate_that_only_matches_the_reference_still_passes_today(self):
        """⚠️ 这条钉的是「今天仍然放行」这个事实, 不是「放行是对的」

        判决侧要等第 2 片 (失败归属) —— 把它写成断言而不是注释, 是照 `test_oracle_error_ratio`
        里 `assertIsTrue(r.passed)` 那条规矩: 接成判定那一天这条会红, 逼着那次改动重写它。
        """
        r = _l2(self.v, CORRECT, _ref_with(np.nan))
        self.assertTrue(r.passed)
        self.assertIn("240/256 元素可比", r.message)
        self.assertIn("参考另有 16 个非有限值", r.message)

    def test_reference_with_no_comparable_element_refuses_without_blaming_the_candidate(self):
        """分母为空: 这不是「候选 0 个坏点」, 也不是「候选全错」, 而是**本级没有对象**

        `tolerance.evaluate` 在 `total<=0` 时给 (False, 0.0) —— 那是本仓那条「空分母不许读成
        完美」(#53/#81) 的镜像, 本片让它成为**说得出理由**的拒绝。
        """
        r = _l2(self.v, ZEROS, np.full((16, 16), np.nan, np.float32))
        self.assertFalse(r.passed)
        self.assertEqual(0, r.metrics["compared_elements"])
        self.assertEqual(N_TOTAL, r.metrics["reference_nonfinite"])
        self.assertIn("本级判不出", r.message)
        self.assertIn("不是候选错", r.message)
        self.assertNotIn("同时超 atol", r.message, "不许把没有对象读成候选超容差")
        self.assertNotIn("最大误差", r.message, "一个都没比上时那一句会印 0.00e+00 当证据")


class TestSamplingGateDenominator(unittest.TestCase):
    """采样那条路 (`Cap.FULL_OUTPUT_TENSOR` 不在): 同一个洞, 但归属必须分得开"""

    @staticmethod
    def _verifier(sample, shape=(16, 16)):
        def execute(kernel_code, inputs):
            return RemoteExecutionResult(success=True, output_shape=list(shape),
                                        output_sample=list(sample))

        # 刻意**不**声明 FULL_OUTPUT_TENSOR: 走的就是「执行侧只回样本」那一路
        return Verifier(DeclaredFakeBackend(caps=(), execute=execute))

    def test_nan_inside_the_sample_window_is_excluded_not_a_nan_max_error(self):
        """候选那份是**干净的**真值, 坏的是参考 —— 于是那 16 个元素该被剔出分母而不是算成候选的错"""
        sample = _truth().ravel().tolist()
        truth = _ref_with(np.nan)
        v = self._verifier(sample)
        with contextlib.redirect_stdout(io.StringIO()):
            r = v._verify_single_set(CORRECT, _inputs(), truth, _task())
        self.assertTrue(r.passed, r.message)
        self.assertEqual(16, r.metrics["reference_nonfinite"])
        self.assertEqual(N_COMPARED, r.metrics["compared_elements"])
        self.assertTrue(np.isfinite(r.metrics["max_error"]))
        self.assertIn("16 个元素参考非有限已剔除", r.message)

    def test_all_nonfinite_window_refuses_and_blames_the_reference(self):
        """窗口内参考全是非有限: 候选给的是干净的 0, 所以归属只能是参考

        代码里这一支排在「候选含非有限值」**之前** —— 两边都坏时先说参考, 因为参考一坏就没有
        任何对象可比, 那句才是当务之急。
        """
        v = self._verifier([0.0] * N_TOTAL)
        with contextlib.redirect_stdout(io.StringIO()):
            r = v._verify_single_set(CORRECT, _inputs(),
                                     np.full((16, 16), np.nan, np.float32), _task())
        self.assertFalse(r.passed)
        self.assertEqual(0, r.metrics["compared_elements"])
        self.assertEqual(N_TOTAL, r.metrics["reference_nonfinite"])
        self.assertIn("不是候选错", r.message)

    def test_candidate_producing_nan_is_still_the_candidate_fault(self):
        """成对的那一侧: 归属不能混 —— 参考坏说参考, 候选坏说候选

        只测参考那一侧的话, 「把两侧都当参考的错」这条变异能绿 (S2)。
        """
        v = self._verifier([np.nan] * 8 + [0.0] * (N_TOTAL - 8))
        with contextlib.redirect_stdout(io.StringIO()):
            r = v._verify_single_set(CORRECT, _inputs(), np.zeros((16, 16), np.float32), _task())
        self.assertFalse(r.passed)
        self.assertIn("候选输出含 8 个非有限值", r.message)
        self.assertNotIn("不是候选错", r.message)


class TestAggregateCarriesIt(unittest.TestCase):
    """聚合处的搬运: 「通过」那一侧是唯一会被读的, 这条死过三次 (doc 33/37/48)"""

    def setUp(self):
        self.v = Verifier(MockGPU())

    def _scripted(self, per_set_metrics):
        results = [VerificationResult(level=VerificationLevel.L2_FUNCTION, passed=True,
                                      message="脚本发放", metrics=m)
                   for m in per_set_metrics]
        orig = self.v._verify_single_set
        self.v._verify_single_set = lambda *a, **k: results.pop(0)
        try:
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                r = self.v._verify_function(CORRECT, [_inputs() for _ in per_set_metrics],
                                            None, _task())
        finally:
            self.v._verify_single_set = orig
        return r, buf.getvalue()

    @staticmethod
    def _m(**kw):
        base = {"max_error": 1e-7, "matched_ratio": 1.0, "bad_elements": 0,
                "compared_elements": 240, "coverage": 240 / 256, "reference_nonfinite": 16}
        base.update(kw)
        return base

    def test_aggregate_sums_the_reference_counts(self):
        r, log = self._scripted([self._m(), self._m(reference_nonfinite=32)])
        self.assertTrue(r.passed)
        self.assertEqual(48, r.metrics["reference_nonfinite"])
        self.assertEqual(480, r.metrics["compared_elements"])
        self.assertIn("参考另有 48 个非有限值", r.message)
        self.assertIn("覆盖率=93.8%", r.message, "coverage 与这个数是同一件事的两面, 得同源")

    def test_aggregate_does_not_publish_zero_when_nobody_reported(self):
        """远程那一路今天的形状: 载荷里没这个键 ⇒ 聚合**也不许**凭空写一个 0

        写 0 会被读成「查过了, 参考是干净的」, 而那正是本片在别处修的错觉 (#53/#81 那一族)。
        """
        r, _ = self._scripted([{"max_error": 1e-7, "matched_ratio": 1.0, "bad_elements": 0,
                                "compared_elements": 256, "coverage": 1.0},
                               {"max_error": 1e-7, "matched_ratio": 1.0, "bad_elements": 0,
                                "compared_elements": 256, "coverage": 1.0}])
        self.assertNotIn("reference_nonfinite", r.metrics)
        self.assertNotIn("非有限值", r.message)

    def test_a_clean_run_publishes_the_zero_because_it_was_actually_checked(self):
        r, _ = self._scripted([self._m(reference_nonfinite=0), self._m(reference_nonfinite=0)])
        self.assertEqual(0, r.metrics["reference_nonfinite"])
        self.assertNotIn("非有限值", r.message, "0 不需要喊一句")

    def test_real_multiset_mockgpu_run_is_clean_end_to_end(self):
        """不发放的正向对照: 真内建参考、真 MockGPU、两组输入 ⇒ 满分母 + 零非有限

        这一条也是「R1~R6 那些变异不许把主路径判坏」的守卫 (I27 那一族的反向证人)。
        """
        with contextlib.redirect_stdout(io.StringIO()):
            r = self.v._verify_function(CORRECT, [_inputs(), _inputs(seed=9)], None,
                                        _task(max_error=1e-1))
        self.assertTrue(r.passed, r.message)
        self.assertEqual(N_TOTAL * 2, r.metrics["compared_elements"],
                         "两个输入集各 256 个元素, 都比上了")
        self.assertEqual(0, r.metrics["reference_nonfinite"])
        self.assertEqual(1.0, r.metrics["coverage"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
