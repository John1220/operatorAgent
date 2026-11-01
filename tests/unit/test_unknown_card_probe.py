# -*- coding: utf-8 -*-
"""采集器与门之间的字段契约。

这两个文件分开写，最容易出的事故不是崩，而是**采集器填了 `per_sm`、门读 `per_sm_kb`**：
于是五门全 unknown，谁都没notice。所以契约本身要有读者，而且采集器**不许给没测的东西编值**。
"""
import json
import os
import re
import tempfile
import unittest
from pathlib import Path

from benchmarks.probes.collect_unknown_card_probe import (
    MEASURED_BY_ELSEWHERE, PROBE_PREFIX, assemble, load_probes)
from benchmarks.probes.unknown_card_gates import MIN_FLOOR_SAMPLES, assess

# 没有实测输入时，采集器**只许**产出这三段 + 两个元数据键
UNPROBED = ["collected_from", "device", "noise", "shared_memory", "still_missing"]
PROPS_4070 = {"name": "NVIDIA GeForce RTX 4070", "cc": "8.9",
              "total_memory_bytes": 12 * 2 ** 30,
              "per_block_optin_bytes": 99 * 1024, "per_sm_kb": 100}
UUID = "GPU-4070-mine"
FLOOR_PROBE = {"uuid": UUID, "launch_floor": {"median_us": 11.4, "samples": 200},
               "measured_by": "measure_launch_floor.py"}
PEAK_PROBE = {"uuid": UUID, "peak_by_dtype": {"fp16": 94.2, "bf16": 94.0},
              "mem_bw_gbs": 504.0, "measured_by": "calibrate_achievable_peak.py"}


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
        self.assertEqual(sorted(payload), UNPROBED)

    def test_still_missing_without_probes_lists_every_gap(self):
        payload = assemble(PROPS_4070, UUID, "exclusive", 0)
        self.assertEqual(payload["still_missing"],
                         ["launch_floor", "noise.band/dispersion", "peak", "shapes"])

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

    def test_absent_smem_limits_are_refused(self):
        """两个上限全没读到才拒；只暴露一个是 torch 版本的差别，不是硬件的秘密。"""
        props = dict(PROPS_4070, per_block_optin_bytes=None, per_sm_kb=None)
        with self.assertRaises(SystemExit) as caught:
            assemble(props, UUID, "exclusive", 0)
        self.assertIn("共享内存", str(caught.exception))

    def test_one_smem_limit_is_enough_and_the_gate_still_passes(self):
        """老 build 只给 opt-in：采集器照装，门据来源戳放行（缺的那键留在表里是 None）。"""
        props = dict(PROPS_4070, per_sm_kb=None)
        payload = assemble(props, UUID, "exclusive", 0)
        self.assertIsNone(payload["shared_memory"]["per_sm_kb"])
        self.assertEqual(assess(payload)["gates"]["G2_shared_memory"]["verdict"], "pass")


class TestProbeMerging(unittest.TestCase):
    """实测段的来源核对：本卡的才盖戳，来历不明或别卡的直接拒。"""

    def test_matching_probe_fills_floor_and_peak(self):
        payload = assemble(PROPS_4070, UUID, "exclusive", 0, [FLOOR_PROBE, PEAK_PROBE])
        result = assess(payload)
        self.assertEqual(result["gates"]["G1_launch_floor"]["verdict"], "pass")
        self.assertEqual(result["gates"]["G3_peak_denominators"]["verdict"], "pass")
        self.assertEqual(payload["launch_floor"]["source_uuid"], UUID)
        self.assertEqual(payload["peak"]["source_uuid"], UUID)
        self.assertEqual(sorted(payload), sorted(UNPROBED + ["launch_floor", "peak"]))
        self.assertEqual(payload["still_missing"], ["noise.band/dispersion", "shapes"])

    def test_a_peak_probe_without_bandwidth_does_not_open_g3(self):
        """只测了算力没测带宽 ⇒ G3 仍 unknown，而不是拿半截数判 pass。"""
        payload = assemble(PROPS_4070, UUID, "exclusive", 0,
                           [{"uuid": UUID, "peak_by_dtype": {"fp16": 94.2},
                             "measured_by": "calibrate_achievable_peak.py"}])
        self.assertEqual(assess(payload)["gates"]["G3_peak_denominators"]["verdict"], "unknown")
        self.assertIn("peak", payload["still_missing"])

    def test_floor_probe_below_the_sample_floor_is_fail_not_pass(self):
        payload = assemble(PROPS_4070, UUID, "exclusive", 0,
                           [{"uuid": UUID, "launch_floor": {"median_us": 11.4, "samples": 2}}])
        self.assertEqual(assess(payload)["gates"]["G1_launch_floor"]["verdict"], "fail")

    def test_probe_without_uuid_is_refused(self):
        bad = dict(FLOOR_PROBE, uuid=None)
        with self.assertRaises(SystemExit) as caught:
            assemble(PROPS_4070, UUID, "exclusive", 0, [bad])
        self.assertIn("uuid", str(caught.exception))

    def test_probe_from_another_card_is_refused(self):
        bad = dict(FLOOR_PROBE, uuid="GPU-someone-elses")
        with self.assertRaises(SystemExit) as caught:
            assemble(PROPS_4070, UUID, "exclusive", 0, [bad])
        self.assertIn("不是本机", str(caught.exception))


class TestProbeLoading(unittest.TestCase):
    """`--from-probe` 读的是脚本尾巴上的 `PROBE_JSON` 行，不是手抄的数字。"""

    def test_loads_the_last_probe_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "floor.txt")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("人类可读的一行\n")
                fh.write("%s %s\n" % (PROBE_PREFIX, json.dumps(FLOOR_PROBE, ensure_ascii=False)))
                fh.write("%s {\"uuid\": \"second\"}\n" % PROBE_PREFIX)
            self.assertEqual(load_probes([path]), [{"uuid": "second"}])

    def test_refuses_a_file_without_the_machine_readable_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "peak.txt")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("=== 计算峰值阶梯 ===\n只有人类可读的表\n")
            with self.assertRaises(SystemExit) as caught:
                load_probes([path])
            self.assertIn("PROBE_JSON", str(caught.exception))


class TestEmitterWireFormat(unittest.TestCase):
    """产出 `PROBE_JSON` 行的两个脚本，必须**单独跑在远端机器上**（scp 单文件过去），
    所以它们没法 import 这里的 schema。键名一旦改了就没人拦：门会一直 unknown，
    而 `--from-probe` 照样把文件读进去、不报任何错。这一类静态文本核对是唯一便宜的拦法。
    """

    FLOOR = Path(__file__).resolve().parents[2] / "benchmarks/measurement/measure_launch_floor.py"
    PEAK = Path(__file__).resolve().parents[2] / "benchmarks/calibration/calibrate_achievable_peak.py"

    def _text(self, path):
        self.assertTrue(path.is_file(), "%s 不在了" % path.name)
        return path.read_text(encoding="utf-8")

    def test_floor_emitter_carries_the_keys_the_collector_needs(self):
        text = self._text(self.FLOOR)
        for token in ("PROBE_JSON ", '"uuid"', '"launch_floor"', '"median_us"',
                      '"samples"', '"measured_by"'):
            self.assertIn(token, text, "%s 里少了 %s" % (self.FLOOR.name, token))

    def test_peak_emitter_carries_the_keys_the_collector_needs(self):
        text = self._text(self.PEAK)
        for token in ("PROBE_JSON ", '"uuid"', '"peak_by_dtype"', '"mem_bw_gbs"',
                      '"measured_by"'):
            self.assertIn(token, text, "%s 里少了 %s" % (self.PEAK.name, token))

    def test_floor_sample_count_satisfies_the_gate_it_feeds(self):
        """地板脚本的重复次数必须 ≥ 门要求的下限，否则盖了戳也过不了 G1。"""
        text = self._text(self.FLOOR)
        match = re.search(r"^REP\s*=\s*(\d+)", text, re.MULTILINE)
        self.assertIsNotNone(match, "%s 里找不到 `REP = <整数>`：样本数没落成常数" % self.FLOOR.name)
        self.assertGreaterEqual(int(match.group(1)), MIN_FLOOR_SAMPLES)


if __name__ == "__main__":
    unittest.main()
