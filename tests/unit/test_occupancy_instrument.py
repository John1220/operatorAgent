"""§10.3 条件 2 的第二把量具：占用率上限（`occupancy_bound_evidence`）

纪律与 `smem_bound_evidence` 同：**说得出出处才算证据**；缺任一侧 ⇒ 空（三态）。
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from operator_agent.decide.escalation import (  # noqa: E402
    OCCUPANCY_FLOOR_CTAS, occupancy_bound_evidence, smem_bound_evidence)

FULL = dict(shared_mem_per_sm_kb=228, max_threads_per_sm=2048)


class TestTheInstrumentOnlySpeaksWithEvidence(unittest.TestCase):
    def test_a_thread_limited_two_cta_tile_is_cited_evidence(self):
        ev = occupancy_bound_evidence(64 * 1024, 32, **FULL)   # 32 warps = 1024 线程
        self.assertEqual(1, len(ev))
        self.assertIn("occupancy-bound", ev[0])
        self.assertIn("TritonCompileConfig.num_warps", ev[0], "必须指出出处")
        self.assertIn("HardwareCaps.max_threads_per_sm", ev[0])
        self.assertIn("1024", ev[0])

    def test_a_wide_tile_is_not_evidence(self):
        """8 个 CTA 的余地 ⇒ 谈不上占用率硬限制"""
        self.assertEqual((), occupancy_bound_evidence(8 * 1024, 4, **FULL))

    def test_a_missing_side_is_silence_not_a_default(self):
        """三态: 没测到 ≠ 没上限 ≠ 有硬限制 —— 缺任一输入一律不产出证据"""
        for kwargs in (dict(FULL, max_threads_per_sm=None),
                       dict(FULL, shared_mem_per_sm_kb=None),   # smem 未知 ⇒ 无权声称瓶颈是线程
                       dict(max_threads_per_sm=2048)):
            for warps in (32, None, 0, True):
                with self.subTest(kwargs=kwargs, warps=warps):
                    self.assertEqual((), occupancy_bound_evidence(64 * 1024, warps, **kwargs))

    def test_smem_being_the_tighter_side_is_left_to_the_smem_instrument(self):
        """同一件事只许一个认领者: smem 侧更紧 ⇒ 那是 smem-bound, 本条不报"""
        tile = 256 * 1024        # 已贴上界 ⇒ smem-bound 会认领
        self.assertEqual(1, len(smem_bound_evidence(tile, 228 * 1024)))
        self.assertEqual((), occupancy_bound_evidence(tile, 64, **FULL),
                         "更紧的是 smem 侧 ⇒ 不许再报一条占用率")

    def test_a_bool_is_not_a_number(self):
        """`True` 是 1, 但「没说」不许被读成 1 —— 三态的第 2 条边界"""
        self.assertEqual((), occupancy_bound_evidence(64 * 1024, 4,
                                                      shared_mem_per_sm_kb=228,
                                                      max_threads_per_sm=True))
        self.assertEqual((), occupancy_bound_evidence(64 * 1024, True,
                                                      shared_mem_per_sm_kb=228,
                                                      max_threads_per_sm=2048))

    def test_the_threshold_is_named_and_used(self):
        self.assertEqual(2, OCCUPANCY_FLOOR_CTAS)
        self.assertTrue(occupancy_bound_evidence(64 * 1024, 32, **FULL))
        self.assertFalse(occupancy_bound_evidence(64 * 1024, 20, **FULL))  # 3 个 CTA

    def test_both_instruments_do_not_double_report_the_same_thing(self):
        tile = 256 * 1024        # 贴上界: smem 侧更紧
        self.assertEqual(1, len(smem_bound_evidence(tile, 228 * 1024)))
        self.assertEqual((), occupancy_bound_evidence(tile, 32, **FULL),
                         "smem 已经贴到上界时, 占用率侧不许再报一条")


class TestTheInstrumentHasAProducerOnTheRealPath(unittest.TestCase):
    """量具的输入必须有人**发布** —— 否则它是一条永远不产出证据的判据

    §10.3 条件 2 的另一半输入 (`num_warps`) 仍缺发布者, 这一点写在边界里, 不用
    「反正恒为 0」把它盖过去。
    """

    def test_the_caps_publish_carries_the_thread_bound(self):
        from operator_agent import MockGPU
        from operator_agent.decide.verifier import Verifier

        metrics = Verifier(MockGPU())._detect_theoretical_peak()
        self.assertIn("max_threads_per_sm", metrics,
                      "没有发布者 ⇒ 占用率量具永远拿不到输入")
        value = metrics["max_threads_per_sm"]
        self.assertIsInstance(value, int)
        self.assertFalse(isinstance(value, bool))
        self.assertGreaterEqual(value, 0, "0 = 不知道 (三态), 不是编造的默认值")

    def test_it_is_the_same_number_as_the_caps_field(self):
        """与 `shared_mem_per_sm_kb` 同一条真源纪律: 不许在这里另抄一个数字"""
        from operator_agent import MockGPU
        from operator_agent.core.hardware_caps import detect_hardware_caps
        from operator_agent.decide.verifier import Verifier

        v = Verifier(MockGPU())
        metrics = v._detect_theoretical_peak()
        caps = detect_hardware_caps(v.backend.get_hardware_info())
        self.assertEqual(int(caps.max_threads_per_sm or 0),
                         metrics["max_threads_per_sm"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
