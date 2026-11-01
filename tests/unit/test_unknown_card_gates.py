# -*- coding: utf-8 -*-
"""未知卡迁移门（G1–G5）的用例。

这条探针的全部价值在于**它不许在无读数时绿**，所以每组正例都配一条反例：
无读数 ⇒ unknown、继承别的卡的值 ⇒ fail、样本不够 ⇒ fail。
「五门全 pass」只在所有来源都是本卡 UUID 时才允许出现。
"""
import unittest

from benchmarks.probes.unknown_card_gates import GATES, INHERITED, assess, main


def good_info():
    """一张卡把五门全部答上，且每条读数的来源都是它自己。"""
    return {"device": {"uuid": "GPU-mine", "name": "SomeCard", "cc": "8.9",
                       "total_memory_bytes": 16 * 2 ** 30},
            "launch_floor": {"median_us": 11.3, "samples": 9, "source_uuid": "GPU-mine"},
            "shared_memory": {"per_block_optin_bytes": 101376, "per_sm_kb": 100,
                              "source_uuid": "GPU-mine"},
            "peak": {"mem_bw_gbs": 500.0, "peak_by_dtype": {"fp16": 90.0},
                     "trusted": True, "source_uuid": "GPU-mine"},
            "noise": {"band": {"valid": 0.002, "degraded": 0.006, "uninformative": 0.03},
                      "contamination_util_pct": 3.0, "dispersion_degraded": 0.08,
                      "peers_observed": 0, "sharing_pattern": "独占", "source_uuid": "GPU-mine"},
            "shapes": [{"label": "layernorm 4096x2048", "bytes": 32 * 2 ** 20,
                        "validated_on_uuid": "GPU-mine"}]}


class TestNeverGreenWithoutReadings(unittest.TestCase):

    def test_no_readings_at_all_leaves_every_gate_unknown(self):
        result = assess({"device": {"uuid": "GPU-mine"}})
        self.assertEqual(sorted(result["gates"]), sorted(GATES))
        for name, gate in result["gates"].items():
            self.assertEqual(gate["verdict"], "unknown", name)
        self.assertFalse(result["all_green"])

    def test_main_without_input_exits_nonzero(self):
        self.assertEqual(main([]), 2, "没给 --from-json 绝不能返回 0")

    def test_fully_measured_card_is_the_only_green_case(self):
        result = assess(good_info())
        self.assertTrue(result["all_green"], result["gates"])
        self.assertEqual({g["verdict"] for g in result["gates"].values()}, {"pass"})


class TestInheritedValuesAreRefused(unittest.TestCase):

    def _refuse(self, mutator):
        info = good_info()
        mutator(info)
        result = assess(info)
        return result

    def test_launch_floor_from_another_card_is_fail_not_pass(self):
        result = self._refuse(lambda i: i["launch_floor"].update(source_uuid="H20-archive"))
        self.assertEqual(result["gates"]["G1_launch_floor"]["verdict"], "fail")
        self.assertFalse(result["all_green"])

    def test_h20_launch_constant_cannot_be_inherited(self):
        # 数值恰好是 H20 那条 18.69µs，且来源不是本卡 ⇒ fail（继承不是标定）
        def mut(i):
            i["launch_floor"].update(median_us=INHERITED["launch_floor_us_h20"],
                                     source_uuid="H20-archive")
        self.assertEqual(self._refuse(mut)["gates"]["G1_launch_floor"]["verdict"], "fail")

    def test_floor_with_too_few_samples_is_fail(self):
        result = self._refuse(lambda i: i["launch_floor"].update(samples=2))
        self.assertEqual(result["gates"]["G1_launch_floor"]["verdict"], "fail")

    def test_untrusted_peak_closes_the_stop_exit(self):
        result = self._refuse(lambda i: i["peak"].update(trusted=False))
        gate = result["gates"]["G3_peak_denominators"]
        self.assertEqual(gate["verdict"], "fail")
        self.assertIn("判停", gate["why"])

    def test_noise_band_copied_from_the_tree_never_passes(self):
        """搬仓内 H20 的数又没给来源 ⇒ unknown 或 fail，唯独不许 pass。"""
        def mut(i):
            i["noise"] = {"band": INHERITED["noise_band"],
                          "contamination_util_pct": INHERITED["contamination_util_pct"],
                          "dispersion_degraded": INHERITED["dispersion_degraded"]}
        verdict = self._refuse(mut)["gates"]["G4_noise_bands"]["verdict"]
        self.assertNotEqual(verdict, "pass", "继承不是标定")

    def test_shared_memory_from_another_card_is_fail(self):
        result = self._refuse(lambda i: i["shared_memory"].update(source_uuid="H20-archive"))
        self.assertEqual(result["gates"]["G2_shared_memory"]["verdict"], "fail")

    def test_noise_thresholds_without_provenance_are_unknown(self):
        """光写一句「独占」不构成标定 —— 来源缺失时不得给 pass。"""
        result = self._refuse(lambda i: i["noise"].pop("source_uuid"))
        self.assertEqual(result["gates"]["G4_noise_bands"]["verdict"], "unknown")

    def test_noise_thresholds_from_another_card_are_fail(self):
        result = self._refuse(lambda i: i["noise"].update(source_uuid="H20-archive"))
        self.assertEqual(result["gates"]["G4_noise_bands"]["verdict"], "fail")

    def test_own_source_but_copied_h20_numbers_are_fail(self):
        """声称本卡测的，值却与仓内 H20 标定逐位相同 ⇒ 抄的，不是测的。"""
        def mut(i):
            i["noise"] = {"band": INHERITED["noise_band"],
                          "contamination_util_pct": INHERITED["contamination_util_pct"],
                          "dispersion_degraded": INHERITED["dispersion_degraded"],
                          "peers_observed": 0, "sharing_pattern": "独占",
                          "source_uuid": "GPU-mine"}
        self.assertEqual(self._refuse(mut)["gates"]["G4_noise_bands"]["verdict"], "fail")

    def test_shape_beyond_the_pool_is_fail(self):
        def mut(i):
            i["shapes"] = [{"label": "matmul 8192^3 fp16 输入", "bytes": 20 * 2 ** 30,
                            "validated_on_uuid": "GPU-mine"}]
        self.assertEqual(self._refuse(mut)["gates"]["G5_shape_inheritance"]["verdict"], "fail")

    def test_shape_validated_only_elsewhere_is_fail(self):
        def mut(i):
            i["shapes"] = [{"label": "layernorm 16384x6144", "bytes": 200 * 2 ** 20,
                            "validated_on_uuid": "GPU-h20-96g"}]
        gate = self._refuse(mut)["gates"]["G5_shape_inheritance"]
        self.assertEqual(gate["verdict"], "fail")
        self.assertIn("别的卡", gate["why"])


class TestSelfTestAndExitCode(unittest.TestCase):

    def test_selftest_passes(self):
        self.assertEqual(main(["--selftest"]), 0)

    def test_main_with_a_good_file_exits_zero(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "probe.json"
            path.write_text(json.dumps(good_info()), encoding="utf-8")
            self.assertEqual(main(["--from-json", str(path)]), 0)

    def test_main_with_an_inherited_file_exits_two(self):
        import json
        import tempfile
        from pathlib import Path
        info = good_info()
        info["launch_floor"]["source_uuid"] = "H20-archive"
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "probe.json"
            path.write_text(json.dumps(info), encoding="utf-8")
            self.assertEqual(main(["--from-json", str(path)]), 2)

    def test_missing_file_argument_is_refused(self):
        self.assertEqual(main(["--from-json"]), 2)


if __name__ == "__main__":
    unittest.main()
