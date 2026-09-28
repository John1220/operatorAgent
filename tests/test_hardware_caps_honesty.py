"""硬件能力表的诚实性守卫 (doc 49 §3.3 的 S8 / S9 / S10)。

三个缺陷同型: 失败路径上返回一个**看起来像样的编造值**, 而不是报错。

  S9  查表命中后返回的是模块级共享实例, 紧接着就地写 `sm_count` ⇒ 第一次探测把
      自己的 SM 数写进全局表, 之后所有调用方 (不同任务、不同设备) 都读到污染值。
  S8  认不出的目标兜底到 A100 (只要含 "nvidia" 字样) 或 mock ⇒ 「不知道这台卡是
      什么」被下游读成 156 TF32 / 2039 GB/s / 108 SM, 或读成 2 个 SM 的模拟器。
      同一模块里还并存两套匹配策略 (键名长者优先 vs 字典插入顺序), 于是 "h200"
      命中 "h20" —— 表里根本没有 h200 条目, 拿到的却是 H20 的规格。
  S10 判不了的 `requires_caps` 默认通过 ⇒ 配方里一个拼错的字段名让该配方在**任何**
      硬件上都入选, 而 `_caps_failure_note` 只注解失败的要求, 静默通过不留痕。

每条断言都做过变异标定: 把修复改回旧写法, 对应用例必须变红。
"""

import copy
import dataclasses
import unittest
from unittest import mock

from operator_agent.hardware_caps import (
    HardwareCaps,
    UnknownHardware,
    _KNOWN_HARDWARE,
    detect_hardware_caps,
    get_caps_for_target,
    unknown_caps,
)
from operator_agent.optimization_planner import (
    CapsRequirementUnparsable,
    OptimizationPlanner,
    Recipe,
)

_HAS_FIELDS = tuple(f.name for f in dataclasses.fields(HardwareCaps)
                    if f.name.startswith("has_"))
_NUM_FIELDS = ("fp32_tflops", "tf32_tflops", "fp16_tflops", "bf16_tflops", "int8_tops",
               "global_mem_bandwidth_gbs", "l2_cache_size_kb", "shared_mem_per_sm_kb",
               "sm_count")

#: 判不了的条件, 每种写法各占一条分支 (用例数按分支算, 不按函数算)
UNPARSABLE_REQS = (
    "has_tma=1",                # 布尔写成了数字
    "has_tma = yes",            # 布尔写成了别的字面量
    "vendor == nvidia",         # 字符串比较, 数值分支只认整数
    "has_tma or has_pdl",       # 布尔表达式
    "sm_count != 0",            # 不支持的运算符
    "shared_mem 足够大",         # 过了关键词过滤, 但没有可判定的形状
    "no_such_thing",            # 裸词, 不是 has_ 前缀
)


def _table_snapshot() -> dict:
    return {k: v.to_dict() for k, v in _KNOWN_HARDWARE.items()}


class TableGuardedCase(unittest.TestCase):
    """每个用例前后各做一次全表快照。

    被测的缺陷本身就是「污染全局表」: 不还原的话, 一条变红的用例会把后面的用例
    一起带红, 变异标定时就分不清是哪条守卫在响。
    """

    def setUp(self):
        self._pristine = {k: copy.deepcopy(v) for k, v in _KNOWN_HARDWARE.items()}
        self._snapshot = _table_snapshot()

    def tearDown(self):
        polluted = self._snapshot != _table_snapshot()
        _KNOWN_HARDWARE.clear()
        _KNOWN_HARDWARE.update({k: copy.deepcopy(v) for k, v in self._pristine.items()})
        self.assertFalse(polluted,
                         "_KNOWN_HARDWARE 被就地改动了 (查表函数必须返回副本)")


class TestKnownTableIsNeverMutatedInPlace(TableGuardedCase):
    """S9: 查表命中返回的必须是副本, 探测值只写在副本上。"""

    def test_detect_writes_the_probe_onto_a_copy(self):
        caps = detect_hardware_caps({"name": "NVIDIA H20", "multiprocessor_count": 132,
                                     "compute_capability": "9.0", "device": "cuda"})
        self.assertEqual(132, caps.sm_count)
        self.assertEqual(78, _KNOWN_HARDWARE["h20"].sm_count,
                         "探测到的 SM 数被写进了全局表")

    def test_second_caller_is_not_poisoned_by_the_first(self):
        """旧写法的真实症状: 第二个调用方读到第一次探测的数。"""
        detect_hardware_caps({"name": "NVIDIA H20", "multiprocessor_count": 132,
                              "device": "cuda"})
        again = detect_hardware_caps({"name": "NVIDIA H20", "device": "cuda"})
        self.assertEqual(78, again.sm_count)

    def test_detect_result_is_not_the_shared_instance(self):
        for key in _KNOWN_HARDWARE:
            caps = detect_hardware_caps({"name": key, "device": "cuda"})
            self.assertIsNot(_KNOWN_HARDWARE[key], caps,
                             f"detect 把表里的 {key} 直接交出去了")

    def test_get_caps_result_is_not_the_shared_instance(self):
        for key in _KNOWN_HARDWARE:
            caps = get_caps_for_target(key)
            self.assertIsNot(_KNOWN_HARDWARE[key], caps,
                             f"get_caps 把表里的 {key} 直接交出去了")
            caps.sm_count = -1
            self.assertEqual(self._snapshot[key]["sm_count"], _KNOWN_HARDWARE[key].sm_count)

    def test_mutable_dsl_set_is_not_shared_either(self):
        """浅拷贝 (dataclasses.replace) 会漏掉这一个字段。"""
        for key in _KNOWN_HARDWARE:
            caps = get_caps_for_target(key)
            self.assertIsNot(_KNOWN_HARDWARE[key].supported_dsls, caps.supported_dsls)
            caps.supported_dsls.add("opencl")
            self.assertNotIn("opencl", _KNOWN_HARDWARE[key].supported_dsls,
                             f"{key} 的 supported_dsls 是共享对象")

    def test_a_round_of_detects_leaves_the_table_pristine(self):
        for probe in ({"name": "NVIDIA H20", "multiprocessor_count": 132, "device": "cuda"},
                      {"name": "NVIDIA A100-SXM4-40GB", "multiprocessor_count": 108},
                      {"name": "MockGPU", "compute_units": 64},
                      {"name": "NVIDIA H100", "multiprocessor_count": 132, "device": "cuda"}):
            detect_hardware_caps(probe)
        self.assertEqual(self._snapshot, _table_snapshot())


class TestUnrecognizedTargetIsNotFabricated(TableGuardedCase):
    """S8: 认不出就抛, 不许兜到某个具体型号的规格上。"""

    def test_unknown_nvidia_target_is_not_read_as_a100(self):
        # 旧写法: `if "nvidia" in th: return _KNOWN_HARDWARE["a100"]`
        with self.assertRaises(UnknownHardware) as ctx:
            get_caps_for_target("nvidia_gpu")
        msg = str(ctx.exception)
        self.assertIn("nvidia_gpu", msg, "异常消息里要带上收到的字符串")
        for key in ("a100", "h20"):
            self.assertIn(key, msg, "异常消息里要带上已知键名, 否则没人知道该补哪条")

    def test_unknown_accelerator_is_not_read_as_mock(self):
        # 旧写法: 兜底 `return _KNOWN_HARDWARE["mock"]` ⇒ 2 个 SM / fp32 0.1 TFLOPS
        for target in ("kunlunxin_r3", "iluvatar", "cambricon_mlu370", "ascend", "npu"):
            with self.assertRaises(UnknownHardware, msg=target):
                get_caps_for_target(target)

    def test_unknown_hardware_is_a_keyerror(self):
        """继承 KeyError: 它本来就是查表未命中, 老的 except KeyError 仍然接得住。"""
        self.assertTrue(issubclass(UnknownHardware, KeyError))
        with self.assertRaises(KeyError):
            get_caps_for_target("nvidia_gpu")

    def test_h200_does_not_resolve_to_h20(self):
        with self.assertRaises(UnknownHardware):
            get_caps_for_target("h200")
        caps = detect_hardware_caps({"name": "NVIDIA H200", "multiprocessor_count": 132,
                                     "compute_capability": "9.0", "device": "cuda"})
        # 拿 H20 的能力当 H200 用, 后果是 roofline 分母偏小 6.7 倍 ⇒ 会提前判停
        self.assertNotEqual("H20", caps.device_name)
        self.assertEqual(0.0, caps.global_mem_bandwidth_gbs)
        self.assertEqual(0.0, caps.fp16_tflops)
        self.assertIsNone(caps.launch_floor_us, "H20 的实测地板被当成了 H200 的")
        self.assertEqual(132, caps.sm_count, "探测到的 SM 数仍应保留")

    def test_recognized_targets_still_resolve(self):
        expected = {
            "NVIDIA H20": "H20",
            "NVIDIA H20 96GB": "H20",
            "nvidia a100-sxm4-40gb": "A100",
            "H100 80GB HBM3": "H100",
            "NVIDIA GeForce RTX 3090": "RTX 3090",
            "RTX3090": "RTX 3090",
            "MI250X": "MI250",
            "V100S-PCIe": "V100",
            "mock_gpu": "MockGPU/CPU",
            "MockGPU": "MockGPU/CPU",
        }
        for target, name in expected.items():
            self.assertEqual(name, get_caps_for_target(target).device_name, target)

    def test_every_table_entry_is_reachable_by_its_own_key(self):
        """反向完备性: 键名本身必须能匹配上自己。

        挡的是「往表里加了一条永远查不出来的记录」—— 例如键名带大写、带空格,
        或尾部紧跟数字 (表里 "ascend910b" 认不出设备名 "Ascend 910B", 那是键名的
        写法问题, 不是匹配策略问题)。
        """
        for key, entry in _KNOWN_HARDWARE.items():
            self.assertEqual(entry.device_name, get_caps_for_target(key).device_name,
                             f"键 {key!r} 匹配不上自己")

    def test_longer_key_wins_over_the_shorter_one_it_contains(self):
        """「键名长者优先」必须是策略而不是碰巧: 短键先插进表里也不许抢先命中。

        现有条目里没有任何一个键是另一个键的子串, 所以只看这张表这条永远绿 ——
        必须临时造一对重叠键才钉得住 (变异: 去掉 sorted(key=len, reverse=True))。
        """
        marker = HardwareCaps(vendor="nvidia", device_name="MARKER-LONGER-KEY")
        with mock.patch.dict(_KNOWN_HARDWARE, {"h20-3e": marker}):
            self.assertEqual("MARKER-LONGER-KEY",
                             get_caps_for_target("nvidia h20-3e").device_name)
            self.assertEqual("MARKER-LONGER-KEY",
                             detect_hardware_caps({"name": "nvidia h20-3e"}).device_name)

    def test_detect_and_get_caps_share_one_matching_strategy(self):
        """差分守卫: 同一个串在两个入口必须得到同一个答案。

        旧代码一个按键名长度降序、一个按字典插入顺序。两套策略并存时, 修好其中
        一处不会让另一处跟着好 —— 这正是 S8/S9 里 h200 那个症状的成因。
        """
        samples = ("nvidia h20", "h200", "nvidia h200", "a100", "nvidia a100-sxm4-40gb",
                   "mock_gpu", "mockgpu", "rtx 3090", "mi250x", "v100s", "h100 80gb hbm3",
                   "ascend910b", "ascend 910b", "nvidia_gpu", "kunlunxin_r3", "l40s", "")
        for sample in samples:
            try:
                from_target = get_caps_for_target(sample)
                recognized = True
            except UnknownHardware:
                from_target, recognized = None, False
            detected = detect_hardware_caps({"name": sample})
            if recognized:
                self.assertEqual(from_target.device_name, detected.device_name,
                                 f"{sample!r}: 两个入口认出了不同的硬件")
            else:
                # 未命中表时 detect 走推断路径, device_name 就是它收到的串
                self.assertEqual(sample.lower() or "unknown", detected.device_name,
                                 f"{sample!r}: get_caps 认不出, detect 却认成了 "
                                 f"{detected.device_name!r}")


class TestUnknownCapsIsExplicitlyUnknown(TableGuardedCase):
    """「认不出」要有自己的值, 而且这个值不许看起来像任何一台真机。"""

    def test_unknown_caps_claims_no_capability(self):
        caps = unknown_caps("nvidia_gpu")
        self.assertEqual("unknown", caps.vendor)
        self.assertEqual("nvidia_gpu", caps.device_name)
        for f in _HAS_FIELDS:
            self.assertFalse(getattr(caps, f), f"未知设备不该宣称 {f}")
        for f in _NUM_FIELDS:
            self.assertEqual(0, getattr(caps, f), f"{f} 必须是 0, 不能是某台真机的规格")
        self.assertIsNone(caps.launch_floor_us)
        self.assertEqual(set(), caps.supported_dsls)

    def test_unknown_caps_is_not_confusable_with_the_old_fallbacks(self):
        """凡是旧兜底 (a100 / mock) 有值的字段, 未知设备一律是 0。"""
        caps = unknown_caps("nvidia_gpu")
        for key in ("a100", "mock"):
            entry = _KNOWN_HARDWARE[key]
            for f in _NUM_FIELDS:
                if getattr(entry, f):
                    self.assertEqual(0, getattr(caps, f),
                                     f"未知设备在 {f} 上带着 {key} 条目的值")

    def test_each_unknown_caps_call_is_independent(self):
        a, b = unknown_caps("x"), unknown_caps("y")
        a.sm_count = 108
        a.supported_dsls.add("triton")
        self.assertEqual(0, b.sm_count)
        self.assertEqual(set(), b.supported_dsls)


class TestPlannerNeverReadsUnknownAsA100(TableGuardedCase):
    """S8 的调用方: planner 拿不到硬件信息时不许凭空得到一台 A100。"""

    def test_plan_without_hardware_info_gets_explicitly_unknown_caps(self):
        caps = OptimizationPlanner().plan(
            kernel_type="matmul", dsl="triton", hardware={}).hw_caps
        self.assertEqual("unknown", caps.vendor)
        for f in _HAS_FIELDS:
            self.assertFalse(getattr(caps, f), f"没探测过任何硬件, 却宣称 {f}")
        for f in _NUM_FIELDS:
            self.assertEqual(0, getattr(caps, f), f"{f} 凭空有了值")

    def test_capability_recipes_are_not_selected_without_a_probe(self):
        plan = OptimizationPlanner().plan(kernel_type="matmul", dsl="triton", hardware={})
        rejected = dict((r.id, why) for r, why in plan.rejected_recipes)
        for rid in ("R-MATMUL-03", "R-MATMUL-13", "R-UNIV-04"):
            self.assertNotIn(rid, plan.recipe_ids, "没探测过硬件却选出了这条配方")
            self.assertIn("missing hardware capability", rejected[rid])

    def test_plan_with_probed_hardware_still_gets_the_real_entry(self):
        plan = OptimizationPlanner().plan(
            kernel_type="matmul", dsl="triton",
            hardware={"name": "NVIDIA H20", "multiprocessor_count": 78,
                      "compute_capability": "9.0", "device": "cuda"})
        self.assertEqual("H20", plan.hw_caps.device_name)
        self.assertEqual(4000.0, plan.hw_caps.global_mem_bandwidth_gbs)
        self.assertIn("R-MATMUL-13", plan.recipe_ids)


class TestUnparsableRequirementIsNotAPass(TableGuardedCase):
    """S10: 判不了 ≠ 满足。"""

    def test_unparsable_requirement_raises(self):
        caps = _KNOWN_HARDWARE["h20"]
        for req in UNPARSABLE_REQS:
            with self.assertRaises(CapsRequirementUnparsable, msg=req):
                OptimizationPlanner._check_caps_requirement(req, caps)

    def test_unparsable_requirement_is_refused_by_the_aggregate_check(self):
        caps = _KNOWN_HARDWARE["h20"]
        for req in UNPARSABLE_REQS:
            ok, reason = OptimizationPlanner()._check_all_caps([req], caps)
            self.assertFalse(ok, req)
            self.assertIn("无法判定", reason, f"{req!r} 被拒了但没说清是配方写错")

    def test_caps_failure_note_covers_the_unparsable_case(self):
        caps = _KNOWN_HARDWARE["h20"]
        for req in UNPARSABLE_REQS:
            self.assertIn("无法判定",
                          OptimizationPlanner._caps_failure_note(req, caps), req)

    def test_legal_requirement_verdicts_are_unchanged(self):
        """合法要求串的行为必须一字不改 (含五个比较运算符与两种「配方写错」)。"""
        h20, a100 = _KNOWN_HARDWARE["h20"], _KNOWN_HARDWARE["a100"]
        cases = [
            ("has_tma", h20, True), ("has_tma", a100, False),
            ("has_tma=True", h20, True), ("has_tma=True", a100, False),
            ("has_tma=False", a100, True), ("has_tma=FALSE", a100, True),
            ("l2_cache_size_kb > 0", h20, True),
            ("l2_cache_size_kb >= 61440", h20, True),
            ("l2_cache_size_kb >= 61441", h20, False),
            ("sm_count <= 78", h20, True),
            ("sm_count < 78", h20, False),
            ("vector_width_bytes == 16", h20, True),
            ("shared_mem_per_sm_kb >= 64", a100, True),
            ("compute_capability >= 90", h20, False),   # 字符串字段: 判不满足, 不抛
            ("no_such_field >= 1", h20, False),         # 不存在的字段: 判不满足, 不抛
        ]
        for req, caps, expected in cases:
            self.assertEqual(expected,
                             OptimizationPlanner._check_caps_requirement(req, caps),
                             f"{req!r} 的判定变了")

    def test_note_still_separates_recipe_typo_from_weak_hardware(self):
        note = OptimizationPlanner._caps_failure_note("compute_capability >= 90",
                                                     _KNOWN_HARDWARE["h20"])
        self.assertIn("非数值", note)
        self.assertIn("无 no_such_field 字段",
                      OptimizationPlanner._caps_failure_note("no_such_field >= 1",
                                                             _KNOWN_HARDWARE["h20"]))
        # 硬件真的不够时不该扣「配方写错」的帽子
        self.assertEqual("", OptimizationPlanner._caps_failure_note(
            "has_tma=True", _KNOWN_HARDWARE["a100"]))


class TestTypoedRecipeCannotBeSelected(TableGuardedCase):
    """端到端: 写错 requires_caps 的配方要进拒绝名单并留痕, 不许静默入选。"""

    @staticmethod
    def _plan_with(req: str):
        recipe = Recipe(id="T-TYPO-01", name="拼错能力要求的配方", applies_to=["matmul"],
                        applies_when="always", priority=1, requires_caps=[req],
                        generic_template="# 占位模板: 没有模板的配方会被另一道门拒掉")
        recipe._conditions = OptimizationPlanner._parse_conditions("always")
        planner = OptimizationPlanner()
        planner._all_recipes.append(recipe)
        plan = planner.plan(kernel_type="matmul", dsl="triton",
                            hw_caps=_KNOWN_HARDWARE["h20"])
        return plan, dict((r.id, why) for r, why in plan.rejected_recipes)

    def test_typoed_recipe_is_rejected_not_selected(self):
        plan, rejected = self._plan_with("has_tma=1")
        self.assertNotIn("T-TYPO-01", plan.recipe_ids, "判不了的条件被当成了通过")
        self.assertIn("T-TYPO-01", rejected)
        self.assertIn("无法判定", rejected["T-TYPO-01"])

    def test_correctly_spelled_recipe_is_still_selected(self):
        plan, _ = self._plan_with("has_tma=True")
        self.assertIn("T-TYPO-01", plan.recipe_ids)

    def test_plan_survives_every_unparsable_shape(self):
        """判不了不许把整轮 plan() 打断 —— 那是 compute_capability 那次事故的教训。"""
        for req in UNPARSABLE_REQS:
            self._plan_with(req)


if __name__ == "__main__":
    unittest.main()
