"""标定心跳的离线标定 —— 尺子的自检, 全绿才允许往上盖 Stage 3/4

设计 §10.5-1 / §13.5 / §4.3 第 2 条 / §14 表第 8 行:
  * 一对已知参照 (好核 + 故意慢核) 都在带内 → 尺子可信
  * 任一偏出 → RULER_BROKEN: 停止基于数字决策、该窗口观测失效、不持久化

方向检查 (注入的是读数方向, 不是 GPU):
  1. 诚实 bench → ok
  2. 读数整体偏慢 (放大) / 偏快 (缩小) → RULER_BROKEN  (绝对锚点两臂)
  3. 慢核分不出来 (与好核同速) / 反而更快 → RULER_BROKEN  (分离度)
  4. 心跳失败后, 同窗观测可被标为不可用; 跨窗不受牵连

纯 Python 参考核, 不需要 GPU/torch。
运行: python3 -B -m unittest tests.test_calibration -v
"""
import math
import unittest

from operator_agent.calibration import (
    HeartbeatResult, RULER_BROKEN, default_bench_fn, default_inputs, heartbeat,
    heartbeat_due, known_good_kernel, known_slow_kernel, ruler_broken_window,
)


def _fixed_bench(good_cost, slow_cost):
    """按核名返回固定读数的假 bench —— 方向检查只关心数字, 不关心真跑。"""
    def bench(code, inputs):
        name = getattr(code, "__name__", str(code))
        return slow_cost if "slow" in name else good_cost
    return bench


def _scale_bench(factor, inner=default_bench_fn):
    """把诚实读数整体缩放 factor 倍 —— 模拟「尺子被缩放」的偏慢/偏快。"""
    def bench(code, inputs):
        return inner(code, inputs) * factor
    return bench


class TestHonestPairPasses(unittest.TestCase):
    def test_default_reference_pair_with_honest_bench(self):
        """默认纯 Python 参照对 + 诚实计时 → ok, 且慢核显著更慢 (≥2×)。"""
        result = heartbeat(default_bench_fn)
        self.assertTrue(result.ok, result.reasons)
        self.assertGreaterEqual(result.known_slow_ratio, 2.0, result.reasons)
        self.assertFalse(ruler_broken_window(result))

    def test_fixed_readings_inside_bands_pass(self):
        result = heartbeat(_fixed_bench(1.0, 4.0), expected_good=1.0)
        self.assertTrue(result.ok, result.reasons)
        self.assertAlmostEqual(result.known_good_ratio, 1.0)
        self.assertAlmostEqual(result.known_slow_ratio, 4.0)

    def test_result_carries_epoch_and_window(self):
        result = heartbeat(_fixed_bench(1.0, 4.0), epoch=7, window_id="w-7")
        self.assertEqual(7, result.epoch)
        self.assertEqual("w-7", result.window_id)
        self.assertIsInstance(result, HeartbeatResult)


class TestAbsoluteBiasBreaksRuler(unittest.TestCase):
    """方向检查两臂: 读数整体偏慢 / 偏快, 都必须宣布 RULER_BROKEN。"""

    def test_biased_slow_readings_break_ruler(self):
        result = heartbeat(_fixed_bench(10.0, 40.0), expected_good=1.0)
        self.assertFalse(result.ok)
        self.assertTrue(any(RULER_BROKEN in r for r in result.reasons),
                        result.reasons)
        self.assertGreater(result.known_good_ratio, 2.0)

    def test_biased_fast_readings_break_ruler(self):
        result = heartbeat(_fixed_bench(0.1, 0.4), expected_good=1.0)
        self.assertFalse(result.ok)
        self.assertTrue(any(RULER_BROKEN in r for r in result.reasons),
                        result.reasons)
        self.assertLess(result.known_good_ratio, 0.5)

    def test_scaled_honest_bench_is_caught_with_anchor(self):
        """真实计时路径的两臂: 把诚实 bench 整体放大 / 缩小, 有绝对锚点就必须红。"""
        for factor, label in ((10.0, "偏慢"), (0.1, "偏快")):
            with self.subTest(factor=factor, label=label):
                result = heartbeat(_scale_bench(factor), expected_good=1.0)
                self.assertFalse(result.ok, f"{label} 未被拦: {result.reasons}")
                self.assertTrue(any(RULER_BROKEN in r for r in result.reasons))


class TestSeparationBreaksRuler(unittest.TestCase):
    def test_indistinguishable_pair_breaks_ruler(self):
        """慢核与好核同速 = 尺子分不出快慢 → RULER_BROKEN。"""
        result = heartbeat(_fixed_bench(1.0, 1.0), expected_good=1.0)
        self.assertFalse(result.ok)
        self.assertTrue(any(RULER_BROKEN in r for r in result.reasons),
                        result.reasons)
        self.assertLess(result.known_slow_ratio, 2.0)

    def test_slow_kernel_reading_faster_breaks_ruler(self):
        """慢核反而更快 —— 方向反了, 比「分不出」更严重。"""
        result = heartbeat(_fixed_bench(1.0, 0.5), expected_good=1.0)
        self.assertFalse(result.ok)
        self.assertTrue(any(RULER_BROKEN in r for r in result.reasons),
                        result.reasons)

    def test_no_absolute_anchor_still_checks_separation(self):
        """无绝对锚点时只做分离度, 但分离度失效照样红。"""
        result = heartbeat(_fixed_bench(1.0, 1.0))
        self.assertFalse(result.ok)
        self.assertTrue(math.isnan(result.known_good_ratio) or
                        result.known_good_ratio == 1.0)

    def test_max_slow_ratio_upper_band(self):
        result = heartbeat(_fixed_bench(1.0, 100.0), expected_good=1.0,
                           max_slow_ratio=50.0)
        self.assertFalse(result.ok)


class TestBrokenWindowInvalidatesObservations(unittest.TestCase):
    def test_failed_heartbeat_marks_same_window_unusable(self):
        """心跳失败后, 同窗观测可被标为不可用 (设计 §4.3 第 2 条)。"""
        result = heartbeat(_fixed_bench(10.0, 40.0), expected_good=1.0,
                           window_id="w-3")
        self.assertFalse(result.ok)
        self.assertTrue(ruler_broken_window(result, "w-3"))
        self.assertTrue(ruler_broken_window(result))  # 无窗口 id → 失败关闭

    def test_other_windows_stay_usable(self):
        """跨窗不受牵连 —— 一张卡/一扇窗坏, 不代表别的窗坏 (§13.5)。"""
        result = heartbeat(_fixed_bench(10.0, 40.0), expected_good=1.0,
                           window_id="w-3")
        self.assertFalse(result.ok)
        self.assertFalse(ruler_broken_window(result, "w-4"))

    def test_passing_heartbeat_never_invalidates(self):
        result = heartbeat(_fixed_bench(1.0, 4.0), expected_good=1.0,
                           window_id="w-1")
        self.assertTrue(result.ok)
        self.assertFalse(ruler_broken_window(result, "w-1"))
        self.assertFalse(ruler_broken_window(result))

    def test_none_result_is_not_a_broken_window(self):
        """没跑过心跳 ≠ 尺子坏了: 另一条资格路径的事, 不在这里判。"""
        self.assertFalse(ruler_broken_window(None, "w-1"))


class TestHeartbeatDue(unittest.TestCase):
    def test_fires_on_interval_multiples_only(self):
        self.assertFalse(heartbeat_due(0, 5))
        self.assertFalse(heartbeat_due(3, 5))
        self.assertTrue(heartbeat_due(5, 5))
        self.assertFalse(heartbeat_due(6, 5))
        self.assertTrue(heartbeat_due(10, 5))

    def test_interval_must_be_positive(self):
        with self.assertRaises(ValueError):
            heartbeat_due(5, 0)
        with self.assertRaises(ValueError):
            heartbeat_due(5, -1)


class TestDefaultPairIsRunnable(unittest.TestCase):
    def test_default_kernels_agree_numerically(self):
        """慢核只是多算几遍, 结果必须与好核一致 —— 参照物自己不能是错核。"""
        A, B = default_inputs(n=8)
        self.assertEqual(known_good_kernel(A, B), known_slow_kernel(A, B, repeats=3))

    def test_default_inputs_shape(self):
        A, B = default_inputs(n=8)
        self.assertEqual(8, len(A))
        self.assertEqual(8, len(B[0]))


if __name__ == "__main__":
    unittest.main()
