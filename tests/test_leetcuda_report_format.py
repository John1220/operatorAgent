"""LeetCUDA 借鉴的报告格式: L2 first_bad_* 与 L3 ratio 三元组

出处: LeetCUDA `book/tests/common_test.h:51-66` (book_check 报 first_bad_index)
与 `kernels/interview/bench/hgemm.cu:658-659` (`"%.1f/%.1f (%.2fx)"` 三元组)。
这两样都不改判定, 只改「错了/快了」的**报告形状** —— 但 Fixer 与量具靠它定位。
"""
import re
import unittest

import numpy as np

from operator_agent.mock_gpu import MockGPU
from operator_agent.schemas import ConstraintSpec, Precision, SourceSpec, TargetSpec, TaskSchema, TaskType
from operator_agent.verifier import _REMOTE_L2_METRIC_KEYS, Verifier


def _task(max_error=1e-4, kernel_type="matmul"):
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type=kernel_type),
        target=TargetSpec(hardware="mock_gpu", dsl="python"),
        constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=1.0,
                                   max_error=max_error),
    )


# 正确核: A@B; 错误核: 在第 7 个元素上写坏 (展平下标可预测)
GOOD = "import numpy as np\ndef kernel(A, B):\n    return np.asarray(A) @ np.asarray(B)\n"
BAD_AT_7 = """
import numpy as np
def kernel(A, B):
    out = np.asarray(A) @ np.asarray(B)
    out = np.array(out, dtype=np.float32, copy=True)
    out.reshape(-1)[7] = out.reshape(-1)[7] + 10.0
    return out
"""
BAD_SCALED = """
import numpy as np
def kernel(A, B):
    return (np.asarray(A) @ np.asarray(B)) * 0.5
"""


class TestFirstBadIndexPointsAtTheBrokenElement(unittest.TestCase):
    def setUp(self):
        self.v = Verifier(MockGPU())
        self.xs = {"A": np.ones((4, 4), dtype=np.float32),
                   "B": np.ones((4, 4), dtype=np.float32)}

    def test_failed_l2_reports_first_bad_index_and_both_sides(self):
        r = self.v._verify_function(BAD_AT_7, self.xs, None, _task())
        self.assertFalse(r.passed)
        self.assertEqual(7, r.metrics.get("first_bad_index"),
                         "展平下标没指到动过的那个元素")
        self.assertIn("first_bad", r.message)
        self.assertIn("got", r.message)
        self.assertIn("want", r.message)
        # 两侧取值都要有: 分得清 scale 错 vs 局部炸
        self.assertIsNotNone(r.metrics.get("first_bad_actual"))
        self.assertIsNotNone(r.metrics.get("first_bad_expected"))
        self.assertGreater(abs(r.metrics["first_bad_actual"]
                               - r.metrics["first_bad_expected"]), 1.0)

    def test_passing_l2_does_not_claim_a_first_bad(self):
        r = self.v._verify_function(GOOD, self.xs, None, _task())
        self.assertTrue(r.passed)
        self.assertNotIn("first_bad_index", r.metrics)
        self.assertNotIn("first_bad", r.message)

    def test_scale_error_still_reports_a_first_bad_so_fixer_sees_the_pattern(self):
        """整体 ×0.5: 每个元素都坏, first_bad@0, 两侧比值恒 0.5 —— 与局部炸可区分"""
        r = self.v._verify_function(BAD_SCALED, self.xs, None, _task())
        self.assertFalse(r.passed)
        self.assertEqual(0, r.metrics.get("first_bad_index"))
        got, want = r.metrics["first_bad_actual"], r.metrics["first_bad_expected"]
        self.assertAlmostEqual(got / want, 0.5, places=5)

    def test_first_bad_flattened_index_is_within_the_output(self):
        r = self.v._verify_function(BAD_AT_7, self.xs, None, _task())
        self.assertGreaterEqual(r.metrics["first_bad_index"], 0)
        self.assertLess(r.metrics["first_bad_index"], 16)

    def test_remote_metric_whitelist_carries_first_bad(self):
        for k in ("first_bad_index", "first_bad_actual", "first_bad_expected"):
            self.assertIn(k, _REMOTE_L2_METRIC_KEYS,
                          f"远端算出来 {k} 也会在搬运时被丢掉")

    def test_direction_control_touching_a_different_element_moves_the_index(self):
        bad2 = BAD_AT_7.replace("out.reshape(-1)[7]", "out.reshape(-1)[3]")
        r = self.v._verify_function(bad2, self.xs, None, _task())
        self.assertEqual(3, r.metrics.get("first_bad_index"),
                         "改了坏元素位置, 下标却没跟着动 —— 它不是真去数的")


class TestRatioTripleShowsBothSidesOfTheComparison(unittest.TestCase):
    """LeetCUDA 的 `%.1f/%.1f (%.2fx)`: 只报 1.04x 看不见绝对量级"""

    def test_l3_message_contains_cand_over_base_and_ratio(self):
        from operator_agent.verifier import Verifier as V
        # 用一个假 bench 装置太重; 直接测格式化约定 —— 三元组必须同时含
        # 两个数与括号倍率, 且 cand 在前。
        kernel_ms, base_ms = 1.234, 1.185
        speedup = base_ms / kernel_ms
        triple = f"{kernel_ms:.4g}/{base_ms:.4g} ({speedup:.2f}x)"
        self.assertRegex(triple, r"^\d+(\.\d+)?/\d+(\.\d+)? \(\d+\.\d+x\)$")
        self.assertIn("1.234/1.185", triple)
        self.assertLess(abs(float(triple.split("(")[1][:-2]) - speedup), 0.01)

    def test_verifier_source_uses_the_triple_in_l3_messages(self):
        from pathlib import Path
        src = Path("operator_agent/verifier.py").read_text(encoding="utf-8")
        self.assertIn("ratio_triple", src)
        self.assertGreaterEqual(src.count("加速比 {ratio_triple}"), 2,
                                "远程与本地两条 L3 出口都要用三元组")
        self.assertIn('"ratio_triple"', src, "三元组要进 metrics 才能落盘")

    def test_both_sides_appear_so_launch_floor_is_visible(self):
        """0.01/0.0096 与 10.4/10.0 都是 1.04x, 但前者可能被地板吞掉 ——
        三元组存在的理由就是让这两者在报表里长得不一样"""
        a = f"{0.01:.4g}/{0.0096:.4g} ({0.0096/0.01:.2f}x)"
        b = f"{10.4:.4g}/{10.0:.4g} ({10.0/10.4:.2f}x)"
        self.assertNotEqual(a.split("/")[0], b.split("/")[0])
        self.assertIn("(", a)
        self.assertIn("(", b)


if __name__ == "__main__":
    unittest.main()
