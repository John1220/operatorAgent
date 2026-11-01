# -*- coding: utf-8 -*-
"""采集器与门之间的字段契约。

这两个文件分开写，最容易出的事故不是崩，而是**采集器填了 `per_sm`、门读 `per_sm_kb`**：
于是五门全 unknown，谁都没notice。所以契约本身要有读者，而且采集器**不许给没测的东西编值**。
"""
import unittest

from benchmarks.probes.collect_unknown_card_probe import (
    FILLED_SECTIONS, MEASURED_BY_ELSEWHERE, assemble)
from benchmarks.probes.unknown_card_gates import assess

PROPS_4070 = {"name": "NVIDIA GeForce RTX 4070", "cc": "8.9",
              "total_memory_bytes": 12 * 2 ** 30,
              "per_block_optin_bytes": 99 * 1024, "per_sm_kb": 100}
UUID = "GPU-4070-mine"


class TestContractWithGates(unittest.TestCase):

    def test_filled_sections_are_recognised_by_the_gates(self):
        payload = assemble(PROPS_4070, UUID, "exclusive", 0)
        result = assess(payload)
        self.assertEqual(result["device"]["uuid"], UUID)
        self.assertEqual(result["gates"]["G2_shared_memory"]["verdict"], "pass",
                         "采集器填的共享内存键必须被门读到")
        self.assertEqual(result["gates"]["G2_shared_memory"]["reading_condition"],
                         "本卡 %s" % UUID)

    def test_unmeasured_sections_stay_unknown_never_green(self):
        payload = assemble(PROPS_4070, UUID, "exclusive", 0)
        result = assess(payload)
        self.assertFalse(result["all_green"])
        for gate in ("G1_launch_floor", "G3_peak_denominators", "G5_shape_inheritance"):
            self.assertEqual(result["gates"][gate]["verdict"], "unknown", gate)
        # G4 只有形态观测、没有阈值 → 仍然 unknown，不因给了 sharing_pattern 就放行
        self.assertEqual(result["gates"]["G4_noise_bands"]["verdict"], "unknown")

    def test_the_collector_does_not_invent_the_parts_it_cannot_measure(self):
        payload = assemble(PROPS_4070, UUID, "exclusive", 0)
        for section in ("launch_floor", "peak", "shapes"):
            self.assertNotIn(section, payload, "没测的段不许出现在产物里")
        self.assertEqual(sorted(payload), sorted(list(FILLED_SECTIONS) + ["collected_from",
                                                                          "still_missing"]))

    def test_still_missing_matches_what_the_gates_report_as_unknown(self):
        """采集器说的「还缺」与门判的 unknown 必须是同一批东西。"""
        payload = assemble(PROPS_4070, UUID, "exclusive", 0)
        result = assess(payload)
        unknown = {k for k, v in result["gates"].items() if v["verdict"] == "unknown"}
        self.assertTrue(payload["still_missing"])
        for section in MEASURED_BY_ELSEWHERE:
            head = section.split(".")[0]
            if head in ("launch_floor", "peak", "shapes"):
                self.assertIn({"launch_floor": "G1_launch_floor", "peak": "G3_peak_denominators",
                               "shapes": "G5_shape_inheritance"}[head], unknown)


class TestRefusals(unittest.TestCase):

    def test_missing_sharing_is_refused_not_defaulted(self):
        for bad in (None, "", "unknown"):
            with self.assertRaises(SystemExit) as caught:
                assemble(PROPS_4070, UUID, bad, 0)
            self.assertIn("exclusive", str(caught.exception))

    def test_missing_uuid_is_refused(self):
        with self.assertRaises(SystemExit):
            assemble(PROPS_4070, "", "exclusive", 0)

    def test_absent_smem_limit_is_refused(self):
        props = dict(PROPS_4070, per_block_optin_bytes=None)
        with self.assertRaises(SystemExit) as caught:
            assemble(props, UUID, "exclusive", 0)
        self.assertIn("共享内存", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
