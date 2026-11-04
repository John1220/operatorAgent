# -*- coding: utf-8 -*-
"""RTX 3080（GA102 20 GB）那条 `_KNOWN_HARDWARE` 条目的守卫。

它 2026-11-01 在一台 gpushare 容器上实测出来的，加它的**唯一理由**是让
`launch_floor_us` 有值 —— 在那之前本卡 `detect_hardware_caps()` 拿不到地板，
`launch_dominance()` 返回 `None`，于是一个 4 µs 的核和一个 2.5 ms 的核在
「launch 支配」这条轴上**都没有读数**，小形状永远不会被判 uninformative。

## 这个字段记哪个口径，是被数据定下来的，不是选的

同一张卡有两个地板读数，差 **25 倍**：

* **不冲刷 median = 76.8 µs**（`measure_launch_floor.bench()`，每轮把流跑干 ⇒ 量的是
  CPU 派发延迟；H20 那行记的 20.24 也是这个口径，它自己的注释写着冲刷口径只有 4.64 µs）；
* **L2 冲刷 min = 3.07 µs**（与 runner `_timed_runs` 同口径 —— 生产计时前一定冲 256 MB）。

2026-11-01 由 relu 那一格反推：`relu 16384×6144` 跑到 **700.9 GB/s**，而本卡实测带宽上界
是 **706.4 GB/s**（99.2%）。若 574.4 µs 里真有 76.8 µs 是 launch 开销，搬运期就得跑
**809 GB/s > 上界** —— 物理上不可能。开销的上界只有 **4.5 µs**。

所以字段记 **3.07 µs**。用 76.8 会把 574 µs 的核判成 13% 支配 ⇒ `degraded`，
那是口径造成的**假降级**。
"""
import unittest

from operator_agent.core.hardware_caps import detect_hardware_caps
from operator_agent.decide.measurement import Validity, assess, launch_dominance

HW = {"name": "NVIDIA GeForce RTX 3080", "compute_capability": "8.6", "device": "cuda"}


class TestTheMeasuredFields(unittest.TestCase):

    def test_detection_returns_what_was_measured(self):
        caps = detect_hardware_caps(HW)
        self.assertEqual("RTX 3080", caps.device_name)
        self.assertEqual(68, caps.sm_count)
        self.assertEqual(100, caps.shared_mem_per_sm_kb)
        self.assertEqual(5120, caps.l2_cache_size_kb)

    def test_the_floor_is_no_longer_unknown(self):
        """加这条之前 `launch_dominance()` 恒为 None —— 那才是小形状被放行的原因。"""
        caps = detect_hardware_caps(HW)
        self.assertIsNone(launch_dominance(None, 0.0132))
        self.assertIsNotNone(launch_dominance(caps.launch_floor_us, 0.0132))

    def test_the_floor_is_the_flush_convention(self):
        self.assertEqual(3.07, detect_hardware_caps(HW).launch_floor_us)


class TestItDoesNotInventDenominators(unittest.TestCase):
    """实测的是「地板与几何」，不是「峰值」。"""

    def test_every_peak_field_stays_unknown(self):
        caps = detect_hardware_caps(HW)
        for field in ("fp32_tflops", "tf32_tflops", "fp16_tflops", "bf16_tflops",
                      "global_mem_bandwidth_gbs"):
            self.assertEqual(0.0, getattr(caps, field),
                             "%s 被填了：那是 roofline 分母，只能从 peak_probe 的实测进" % field)


class TestTheTiersThatCameOutOfRealCells(unittest.TestCase):
    """四个真格子的 median 直接喂进来，钉住档位。基线是 relu 那格的带宽反推。"""

    def setUp(self):
        self.floor = detect_hardware_caps(HW).launch_floor_us

    def test_the_three_measured_shapes_are_valid(self):
        for name, ms in (("matmul 4096^3", 2.516), ("relu 16384x6144", 0.5744),
                         ("softmax 4096^2", 0.3740)):
            got = assess(median_time_ms=ms, launch_floor_us=self.floor).validity
            self.assertIs(Validity.VALID, got, name)

    def test_the_negative_control_still_goes_uninformative(self):
        """小 relu（1024×768）在**两个口径下都**该是 uninformative —— 负对照不能因为换口径失效。"""
        for floor in (3.07, 76.8):
            got = assess(median_time_ms=0.0045, launch_floor_us=floor).validity
            self.assertIs(Validity.UNINFORMATIVE, got, "floor=%r" % floor)

    def test_the_no_flush_ruler_would_fake_a_degradation(self):
        """把 76.8 记进字段会得到什么 —— 留一条用例，免得有人「为了和 H20 一致」改回去。"""
        self.assertIs(Validity.DEGRADED,
                      assess(median_time_ms=0.5744, launch_floor_us=76.8).validity)


if __name__ == "__main__":
    unittest.main()