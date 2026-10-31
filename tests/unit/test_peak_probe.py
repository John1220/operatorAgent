"""§13.5 的 `probe_peak`: 参考下界标定 —— 判据、存放处、按精度读取

设计说「`probe_peak` 必须实测 …探测失败 → 天花板出口永久关闭」。这一族用例钉三件事:

1. **判据**（`peak_probe.probe_peak`）: 七条拒绝理由各自单独可红, 任何一条也不过就不发布。
2. **存放处**（`hardware_caps`）: 按 (设备, 精度) 分组、两个出口都合并、返回的是拷贝。
3. **读取**（`verifier._roofline_analysis`）: 按**这一轮分母的那个精度键**取, 取不到就
   当作未实测 —— 不许借用别的精度 (tf32 与 fp32 差近一倍, 借用会把还能优化的核提前判成到顶)。

运行: python3 -m unittest tests.test_peak_probe -v
"""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend_fakes import DeclaredFakeBackend
from operator_agent.core import hardware_caps as hc
from operator_agent.core.backend_caps import Cap
from operator_agent.decide.peak_probe import (
    PEAK_PRECISIONS, PeakSample, precision_key_of, probe_peak,
    sample_from_l3_metrics,
)
from operator_agent.decide.verifier import Verifier
from tests.unit.test_search_belief_gate import _inputs, _task

H20_INFO = {"name": "NVIDIA H20", "device": "cuda", "compute_capability": "9.0"}


def _sample(**kw):
    base = dict(tflops=68.0, precision="tf32",
                method="L3 基线臂 (同机/同形状/同精度的已知好参考核): flops / baseline_ms",
                real_device=True, validity="valid",
                arithmetic_intensity=683.0, ridge_point=18.5,
                rel_dispersion=0.004, spec_tflops=74.0)
    base.update(kw)
    return PeakSample(**base)


class TestTheJudgement(unittest.TestCase):

    def test_a_good_sample_is_published(self):
        d = probe_peak(_sample())
        self.assertTrue(d.measured)
        self.assertEqual((), d.refusals)
        self.assertEqual("tf32", d.profile.precision)
        self.assertAlmostEqual(68.0, d.profile.tflops)
        self.assertEqual({"measured", "profile", "refusals", "detail"},
                         set(d.as_dict()))

    def test_a_simulation_reading_can_never_be_a_measured_peak(self):
        """仿真读数可以驱动搜索, 但不能当参考下界 —— 这一条挡的是「MockGPU 上也算实测」"""
        d = probe_peak(_sample(real_device=False))
        self.assertFalse(d.measured)
        self.assertTrue(any("REAL_DEVICE" in r for r in d.refusals), d.refusals)

    def test_the_precision_key_must_be_recognizable(self):
        for bad in (None, "", "fp64", "tf32_tensor_core"):
            d = probe_peak(_sample(precision=bad))
            self.assertFalse(d.measured, bad)
            self.assertTrue(any("精度键" in r for r in d.refusals), d.refusals)

    def test_a_sample_without_a_method_is_refused(self):
        """分不出「实测」与「抄表」的数不许进分母: 出处是必填的"""
        d = probe_peak(_sample(method=""))
        self.assertFalse(d.measured)
        self.assertTrue(any("测量方法" in r for r in d.refusals), d.refusals)

    def test_a_non_positive_or_non_finite_value_is_refused(self):
        for bad in (None, 0.0, -1.0, float("nan"), float("inf"), True):
            d = probe_peak(_sample(tflops=bad))
            self.assertFalse(d.measured, bad)
            self.assertTrue(any("有限正数" in r for r in d.refusals), d.refusals)

    def test_the_probe_itself_must_be_valid(self):
        """探针自己也得过尺子的档位: 拿 degraded 的读数当分母 = 把结论建在不可信的测量上"""
        for bad in ("degraded", "uninformative", None):
            d = probe_peak(_sample(validity=bad))
            self.assertFalse(d.measured, bad)
            self.assertTrue(any("valid" in r for r in d.refusals), d.refusals)

    def test_an_unknown_dispersion_is_refused(self):
        d = probe_peak(_sample(rel_dispersion=None))
        self.assertFalse(d.measured)
        self.assertTrue(any("离散度未知" in r for r in d.refusals), d.refusals)

    def test_a_memory_bound_sample_measures_bandwidth_not_flops(self):
        """AI < 脊点 ⇒ 这个样本量的是带宽。用它当分母会把『访存受限的快』误读成算力到顶"""
        d = probe_peak(_sample(arithmetic_intensity=10.67, ridge_point=18.5))
        self.assertFalse(d.measured)
        self.assertTrue(any("没打到算力峰值" in r for r in d.refusals), d.refusals)

    def test_an_unknown_ridge_point_is_refused(self):
        """未知设备没有脊点 ⇒ 分不出量的是算力还是带宽 ⇒ 不发布 (缺证据不是证据)"""
        for kw in (dict(ridge_point=0.0), dict(ridge_point=None),
                   dict(arithmetic_intensity=None)):
            d = probe_peak(_sample(**kw))
            self.assertFalse(d.measured, kw)
            self.assertTrue(any("脊点未知" in r for r in d.refusals), d.refusals)

    def test_a_value_above_the_spec_is_not_the_same_quantity(self):
        """实测量高出规格 5% 以上 ⇒ 采样买到的不是同一个量 (稀疏/混精最常见)"""
        d = probe_peak(_sample(tflops=80.0, spec_tflops=74.0))
        self.assertFalse(d.measured)
        self.assertTrue(any("高于规格" in r for r in d.refusals), d.refusals)
        # 5% 以内照发 (规格不是上界, 只是「同一个量」的对照)
        self.assertTrue(probe_peak(_sample(tflops=77.0, spec_tflops=74.0)).measured)
        # 没有规格可对照时不做这一条判断 (但仍然要过其余六条)
        self.assertTrue(probe_peak(_sample(tflops=500.0, spec_tflops=None)).measured)

    def test_the_precision_key_mapping_is_shared(self):
        self.assertEqual("tf32", precision_key_of("tf32_tensor_core"))
        self.assertEqual("fp32", precision_key_of("fp32_cuda_core"))
        for bad in (None, "", "fp8_tensor_core", "unknown"):
            self.assertIsNone(precision_key_of(bad), bad)
        self.assertEqual(4, len(PEAK_PRECISIONS))


class TestTheSampleAssembly(unittest.TestCase):

    ROOFLINE = {"compute_used": "tf32_tensor_core", "flops": 2 * 4096 ** 3,
                "arithmetic_intensity": 683.0, "ridge_point": 18.5}

    def test_the_baseline_arm_value_is_flops_over_its_own_time(self):
        # 2.0 ms 是刻意的: 4096³ 的 flops 除以它 ≈ 68.7 TFLOPS, 落在 H20 tf32 规格
        # (74) 的 5% 带内 —— 0.5 ms 会得到 274.9 TFLOPS, 那条「不许多报一个量」会把
        # 整个用例判成拒绝 (实测抓到过一次, 见下一条用例)。
        metrics = {"measurement_validity": "valid", "baseline_time_ms": 2.0,
                   "dispersion_iqr_over_median": 0.004, "gflops": 100.0,
                   "roofline_analysis": dict(self.ROOFLINE)}
        s = sample_from_l3_metrics(metrics, real_device=True, spec_tflops=74.0,
                                   method="m", flops=self.ROOFLINE["flops"],
                                   baseline_ms=2.0)
        self.assertAlmostEqual(2 * 4096 ** 3 / 2.0e-3 / 1e12, s.tflops, places=6)
        self.assertTrue(probe_peak(s).measured)

    def test_a_missing_numerator_or_baseline_refuses_instead_of_guessing(self):
        metrics = {"measurement_validity": "valid",
                   "roofline_analysis": dict(self.ROOFLINE)}
        for flops, ms in ((None, 2.0), (2 * 4096 ** 3, None), (None, None)):
            s = sample_from_l3_metrics(metrics, real_device=True, spec_tflops=74.0,
                                       method="m", flops=flops, baseline_ms=ms)
            self.assertIsNone(s.tflops, (flops, ms))
            self.assertFalse(probe_peak(s).measured)


class TestTheStore(unittest.TestCase):

    def setUp(self):
        hc.forget_measured_peaks()
        self.addCleanup(hc.forget_measured_peaks)

    def test_the_store_is_merged_into_caps_on_both_paths(self):
        """命中表 (H20) 与没命中表的卡都要合并 —— 只在末尾合并会让前者永远读不到"""
        self.assertIsNone(hc.detect_hardware_caps(H20_INFO).reference_lower_bound_tflops)
        hc.remember_reference_lower_bound(hc.device_key_of(H20_INFO), "tf32", 68.5, "L3 基线臂")
        self.assertEqual({"tf32": 68.5},
                         hc.detect_hardware_caps(H20_INFO).reference_lower_bound_tflops)
        # 没命中表的卡: 实测值照样要读得到 (它最需要实测)
        new_card = {"name": "Some New GPU", "device": "cuda", "compute_capability": "12.0"}
        hc.remember_reference_lower_bound(hc.device_key_of(new_card), "fp16", 900.0, "L3 基线臂")
        self.assertEqual({"fp16": 900.0},
                         hc.detect_hardware_caps(new_card).reference_lower_bound_tflops)

    def test_the_store_is_keyed_by_device_and_precision(self):
        hc.remember_reference_lower_bound("nvidia h20", "tf32", 68.5, "m")
        hc.remember_reference_lower_bound("nvidia h20", "fp16", 140.0, "m")
        hc.remember_reference_lower_bound("other card", "tf32", 10.0, "m")
        self.assertEqual({"tf32": 68.5, "fp16": 140.0}, hc.measured_peaks_for("nvidia h20"))
        self.assertEqual({"tf32": 10.0}, hc.measured_peaks_for("other card"))
        self.assertEqual({}, hc.measured_peaks_for("never seen"))

    def test_a_record_comes_back_as_a_copy(self):
        hc.remember_reference_lower_bound("nvidia h20", "tf32", 68.5, "m", {"baseline_ms": 0.5})
        rec = hc.measured_peak_record("nvidia h20", "tf32")
        rec["evidence"]["baseline_ms"] = 999.0
        rec["tflops"] = 1.0
        self.assertEqual(0.5, hc.measured_peak_record("nvidia h20", "tf32")["evidence"]["baseline_ms"])
        self.assertEqual(68.5, hc.measured_peak_record("nvidia h20", "tf32")["tflops"])

    def test_a_refusal_never_reaches_the_store(self):
        """判据与存放处是两件事: 存放处不做判断, 但**只有判过的样本**才被交给它"""
        d = probe_peak(_sample(real_device=False))
        self.assertFalse(d.measured)
        self.assertIsNone(d.profile)
        self.assertEqual({}, hc.measured_peaks_for(hc.device_key_of(H20_INFO)))


class TestTheReaderInTheVerifier(unittest.TestCase):
    """收敛出口读的是**这一轮分母那个精度**的实测值"""

    def setUp(self):
        hc.forget_measured_peaks()
        self.addCleanup(hc.forget_measured_peaks)

    def _verifier(self):
        fake = DeclaredFakeBackend(caps=(Cap.REAL_DEVICE,),
                                   get_hardware_info=lambda: dict(H20_INFO))
        return Verifier(fake)

    def _roofline(self, v):
        return v._roofline_analysis(task=_task(min_speedup=5.0), bench_inputs=_inputs(),
                                    actual_gflops=1000.0, actual_bandwidth_gbs=100.0)

    def test_the_denominator_switches_to_the_measured_value(self):
        v = self._verifier()
        r0 = self._roofline(v)
        self.assertEqual("tf32", r0["peak_key"], "H20 + cuda + matmul ⇒ tf32 分母")
        self.assertIs(r0["reference_lower_bound"], False, "没实测过 ⇒ 出口的位是关的")
        self.assertAlmostEqual(74000.0, r0["hw_peak_gflops"], places=1)

        hc.remember_reference_lower_bound(hc.device_key_of(H20_INFO), "tf32", 60.0,
                                 "L3 基线臂", {"baseline_ms": 0.5})
        r1 = self._roofline(v)
        self.assertIs(r1["reference_lower_bound"], True)
        self.assertAlmostEqual(60000.0, r1["hw_peak_gflops"], places=1,
                               msg="分母要换成实测值, 而不是继续用表")
        self.assertIn("L3 基线臂", r1["reference_lower_bound_source"],
                      "分母的出处要跟着分母走")
        self.assertEqual({"baseline_ms": 0.5}, r1["reference_lower_bound_evidence"])

    def test_another_precisions_measurement_is_never_borrowed(self):
        """存了 fp16 的实测值, 而这一轮的分母是 tf32 ⇒ 仍然算未实测 (差近一倍)"""
        hc.remember_reference_lower_bound(hc.device_key_of(H20_INFO), "fp16", 300.0, "L3 基线臂")
        r = self._roofline(self._verifier())
        self.assertEqual("tf32", r["peak_key"])
        self.assertIs(r["reference_lower_bound"], False)
        self.assertAlmostEqual(74000.0, r["hw_peak_gflops"], places=1)
        self.assertEqual({}, r["reference_lower_bound_evidence"])

    def test_the_whole_dict_is_published_for_transparency(self):
        """读者要能看出「这张卡哪些精度实测过」, 而不是只有一个真/假位"""
        hc.remember_reference_lower_bound(hc.device_key_of(H20_INFO), "fp16", 300.0, "m")
        peak = self._verifier()._detect_theoretical_peak()
        self.assertEqual({"fp16": 300.0}, peak["reference_lower_bound_tflops"])
        self.assertIs(peak["reference_lower_bound"], True,
                      "标量那个位回答的是「任何一个精度实测过没有」(透明用)")


class TestThePublisherInTheVerifier(unittest.TestCase):
    """`_publish_peak_probe`: 拿同一次 L3 的基线臂做标定, 判过才发布"""

    #: 只有 roofline 自己产出的那几格; 设备身份**不从这里读** ——
    #: 它取 `_detect_theoretical_peak` 发布的 name/cc (与认卡、与存放处同一个真源)。
    ROOFLINE = {"compute_used": "tf32_tensor_core", "flops": 2 * 4096 ** 3,
                "arithmetic_intensity": 683.0, "ridge_point": 18.5}

    def setUp(self):
        hc.forget_measured_peaks()
        self.addCleanup(hc.forget_measured_peaks)

    def _metrics(self, **kw):
        m = {"measurement_validity": "valid", "baseline_time_ms": 2.0,
             "dispersion_iqr_over_median": 0.004, "gflops": 1000.0,
             "roofline_analysis": dict(self.ROOFLINE)}
        m.update(kw)
        return m

    def _verifier(self, caps=(Cap.REAL_DEVICE,)):
        fake = DeclaredFakeBackend(caps=caps, get_hardware_info=lambda: dict(H20_INFO))
        return Verifier(fake)

    def test_a_valid_compute_bound_baseline_publishes_the_peak(self):
        v = self._verifier()
        v._detect_theoretical_peak()
        metrics = self._metrics()
        v._publish_peak_probe(metrics)
        probe = metrics["peak_probe"]
        self.assertTrue(probe["measured"], probe["refusals"])
        self.assertEqual("tf32", probe["profile"]["precision"])
        self.assertEqual({"tf32": probe["profile"]["tflops"]},
                         hc.measured_peaks_for(hc.device_key_of(H20_INFO)))
        # 分母语义: 基线臂的 flops / 它自己的时间
        self.assertAlmostEqual(2 * 4096 ** 3 / 2.0e-3 / 1e12,
                               probe["profile"]["tflops"], places=6)

    def test_a_simulated_device_publishes_nothing_and_says_why(self):
        v = self._verifier(caps=())
        v._detect_theoretical_peak()
        metrics = self._metrics()
        v._publish_peak_probe(metrics)
        self.assertFalse(metrics["peak_probe"]["measured"])
        self.assertEqual({}, hc.measured_peaks_for(hc.device_key_of(H20_INFO)))
        self.assertTrue(any("REAL_DEVICE" in r
                            for r in metrics["peak_probe"]["refusals"]))

    def test_an_untrustworthy_reading_publishes_nothing(self):
        """标定自己也得站得住: 档位不是 valid 时不许拿它当分母"""
        v = self._verifier()
        v._detect_theoretical_peak()
        for bad in ("degraded", "uninformative"):
            metrics = self._metrics(measurement_validity=bad)
            v._publish_peak_probe(metrics)
            self.assertFalse(metrics["peak_probe"]["measured"], bad)
            self.assertEqual({}, hc.measured_peaks_for(hc.device_key_of(H20_INFO)), bad)

    def test_a_memory_bound_baseline_publishes_nothing(self):
        v = self._verifier()
        v._detect_theoretical_peak()
        metrics = self._metrics(roofline_analysis=dict(self.ROOFLINE,
                                                       arithmetic_intensity=2.0,
                                                       ridge_point=18.5))
        v._publish_peak_probe(metrics)
        self.assertFalse(metrics["peak_probe"]["measured"])
        self.assertTrue(any("没打到算力峰值" in r
                            for r in metrics["peak_probe"]["refusals"]))

    def test_the_baseline_value_is_sane(self):
        """反向核对: 发布出去的值必须是个正有限数 (判据里那条拒绝的另一面)"""
        v = self._verifier()
        v._detect_theoretical_peak()
        metrics = self._metrics()
        v._publish_peak_probe(metrics)
        got = metrics["peak_probe"]["profile"]["tflops"]
        self.assertTrue(math.isfinite(got) and got > 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)