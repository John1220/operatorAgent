"""资格位推导的方向检查 (docs/design/autonomous-operator-agent-design.md §4.3)

钉住的不是每条分支的措辞, 而是**方向**:

1. DEGRADED 不许 `can_persist` / `can_count_progress` —— 一次噪声读数永远不能
   永久拉黑配方, 也永远不能算作收敛证据 (§4.3 第一条设计意图)。
2. REAL_DEVICE 缺失不许 `can_verdict` —— 「数不存在」与「数不可信」是两条不同的
   边界, 前者整组拒绝 (§4.3 第二条)。
3. 8 条规则的优先级: 上位的硬否决 (契约哈希 / 心跳 / 出处 / lib fallback / 超峰值)
   必须压过下位的 DEGRADED 与 VALID, 不许「降级抵消硬否决」。
4. `real_device=None` 与 `False` 不是一回事: 缺键 ≠ 检查过、不合格 (老路径不追究)。

纯本地, 不 import measurement, 不需要 GPU / LLM。

运行: python3 -m pytest tests/test_eligibility.py -q
"""

import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operator_agent.eligibility import (
    ALL_FALSE, ALL_TRUE, DEGRADED, PEAK_SLACK, Eligibility,
    derive_eligibility, exceeds_measured_peak,
)


def _bits(e: Eligibility) -> tuple:
    """字段顺序与数据类一致, 便于断言位型"""
    return (e.can_verdict, e.can_tune, e.can_persist, e.can_report, e.can_count_progress)


ALL_FALSE_BITS = (False, False, False, False, False)
# 设计 §4.3 规则 6/7 的位型: 判决/调参/报数开, 持久化/计进度关
DEGRADED_BITS = (True, True, False, True, False)
ALL_TRUE_BITS = (True, True, True, True, True)


class TestRuleDirections(unittest.TestCase):
    """8 条规则各自把位型推向哪个方向"""

    def test_default_is_valid_all_true(self):
        """规则 8: 没有任何坏信号 → 全 True"""
        self.assertEqual(_bits(derive_eligibility()), ALL_TRUE_BITS)

    def test_contract_hash_mismatch_refuses_everything(self):
        """规则 1: 尺子被改写, 任何数字无资格 (调用方已拒评, 资格也拒)"""
        got = derive_eligibility(contract_hash_ok=False)
        self.assertEqual(_bits(got), ALL_FALSE_BITS)
        # 即使其余信号全好也不许翻案
        got = derive_eligibility(contract_hash_ok=False, validity="valid")
        self.assertEqual(_bits(got), ALL_FALSE_BITS)

    def test_heartbeat_failure_window_refuses_everything(self):
        """规则 2: 心跳失败窗口内的观测全部失效"""
        self.assertEqual(_bits(derive_eligibility(heartbeat_ok=False)), ALL_FALSE_BITS)

    def test_real_device_missing_refuses_can_verdict(self):
        """规则 3: 契约要真卡却没真卡 → can_verdict 必须 False (数没有指称对象)"""
        got = derive_eligibility(requires_real_device=True, real_device=False)
        self.assertFalse(got.can_verdict, "REAL_DEVICE 缺失还判对错 = 对不存在的对象下结论")
        self.assertEqual(_bits(got), ALL_FALSE_BITS)

    def test_l3_pass_is_refused_implies_all_false(self):
        """评估器已认定「通过」无对象 → 资格同向整组拒绝"""
        self.assertEqual(_bits(derive_eligibility(l3_pass_is_refused=True)), ALL_FALSE_BITS)

    def test_lib_fallback_refuses_everything(self):
        """规则 4: 静默回退到库 = 在测别的东西"""
        self.assertEqual(_bits(derive_eligibility(lib_fallback=True)), ALL_FALSE_BITS)

    def test_peak_exceeded_is_uninformative_all_false(self):
        """规则 5: 超物理上界说明测量错了, 不是说明很棒"""
        self.assertEqual(_bits(derive_eligibility(peak_exceeded=True)), ALL_FALSE_BITS)

    def test_cv_exceeded_is_degraded_without_persist(self):
        """规则 6: 信噪比不够 → DEGRADED, 不许写长期记忆"""
        got = derive_eligibility(cv_exceeded=True)
        self.assertEqual(_bits(got), DEGRADED_BITS)
        self.assertFalse(got.can_persist, "DEGRADED 不许 can_persist")
        self.assertFalse(got.can_count_progress, "DEGRADED 不许 can_count_progress")

    def test_launch_dominance_exceeded_is_degraded_without_persist(self):
        """规则 6 的另一支: launch 下限占比超阈值同样是 DEGRADED"""
        got = derive_eligibility(launch_dominance_exceeded=True)
        self.assertEqual(_bits(got), DEGRADED_BITS)
        self.assertFalse(got.can_persist)

    def test_partition_drift_is_degraded_without_persist(self):
        """规则 7: 环境漂移 → DEGRADED (跨漂移比较作废)"""
        got = derive_eligibility(partition_drift=True)
        self.assertEqual(_bits(got), DEGRADED_BITS)
        self.assertFalse(got.can_persist)
        self.assertFalse(got.can_count_progress)


class TestPriority(unittest.TestCase):
    """优先级: 上位硬否决压过下位降级, 降级压过 VALID"""

    def test_hard_refusal_beats_degraded_signals(self):
        """规则 1~5 的全 False 不许被规则 6/7 的 DEGRADED 位型覆盖"""
        for kwargs in (
            {"contract_hash_ok": False, "cv_exceeded": True},
            {"heartbeat_ok": False, "partition_drift": True},
            {"lib_fallback": True, "launch_dominance_exceeded": True},
            {"peak_exceeded": True, "cv_exceeded": True, "partition_drift": True},
            {"requires_real_device": True, "real_device": False, "cv_exceeded": True},
        ):
            with self.subTest(**kwargs):
                self.assertEqual(_bits(derive_eligibility(**kwargs)), ALL_FALSE_BITS,
                                 f"{kwargs} 只许是全 False, 不许降级成 DEGRADED")

    def test_degraded_beats_valid(self):
        """坏信号把 VALID 拉下来, 不许「validity 说 valid 就全开」"""
        got = derive_eligibility(cv_exceeded=True, validity="valid")
        self.assertEqual(_bits(got), DEGRADED_BITS)

    def test_validity_uninformative_refuses_everything(self):
        """桥接: 评估器已判 UNINFORMATIVE → 与规则 5 同向全 False"""
        self.assertEqual(_bits(derive_eligibility(validity="uninformative")), ALL_FALSE_BITS)

    def test_validity_degraded_matches_rule_six_bits(self):
        """桥接: 评估器的 DEGRADED (污染等没单列成布尔的信号) 落到同一位型"""
        got = derive_eligibility(validity="degraded")
        self.assertEqual(_bits(got), DEGRADED_BITS)
        self.assertFalse(got.can_persist)

    def test_validity_accepts_enum_like_value(self):
        """调用方传 measurement.Validity 成员或裸字符串都行 (不 import measurement)"""
        class _V:
            value = "degraded"
        self.assertEqual(_bits(derive_eligibility(validity=_V())), DEGRADED_BITS)
        self.assertEqual(_bits(derive_eligibility(validity="VALID")), ALL_TRUE_BITS)


class TestRealDeviceNoneIsNotFalse(unittest.TestCase):
    """`real_device=None` (没做出处检查) 与 `False` (检查过、不合格) 必须分开

    否则一份诚实的本地 mock 运行会因为载荷老实写了 device="cpu" 而被判死 ——
    与 MeasurementAssessment 里 `is False` 的写法同一边界。
    """

    def test_none_does_not_trigger_rule_three(self):
        got = derive_eligibility(requires_real_device=True, real_device=None)
        self.assertEqual(_bits(got), ALL_TRUE_BITS)

    def test_false_does_trigger_rule_three(self):
        got = derive_eligibility(requires_real_device=True, real_device=False)
        self.assertEqual(_bits(got), ALL_FALSE_BITS)

    def test_real_device_false_without_requirement_is_not_fatal(self):
        """契约没要真卡时, 真卡缺席不是「数没有指称对象」"""
        got = derive_eligibility(requires_real_device=False, real_device=False)
        self.assertEqual(_bits(got), ALL_TRUE_BITS)


class TestEligibilityShape(unittest.TestCase):
    """数据类本身的约束"""

    def test_frozen(self):
        e = derive_eligibility()
        with self.assertRaises(FrozenInstanceError):
            e.can_persist = False

    def test_as_dict_matches_design_json(self):
        """§4.1 观测记录里的 eligibility 字段名"""
        self.assertEqual(
            ALL_TRUE.as_dict(),
            {"can_verdict": True, "can_tune": True, "can_persist": True,
             "can_report": True, "can_count_progress": True},
        )

    def test_presets_are_distinct(self):
        """三套预设位型互不相同 —— 若有人把 DEGRADED 写成 ALL_TRUE, 上面全部测不到"""
        self.assertNotEqual(_bits(ALL_FALSE), _bits(DEGRADED))
        self.assertNotEqual(_bits(DEGRADED), _bits(ALL_TRUE))
        self.assertNotEqual(_bits(ALL_FALSE), _bits(ALL_TRUE))

    def test_degraded_never_carries_persist(self):
        """方向检查的最小形式: DEGRADED 这个名字存在的全部意义"""
        self.assertFalse(DEGRADED.can_persist)
        self.assertFalse(DEGRADED.can_count_progress)
        self.assertTrue(DEGRADED.can_verdict)
        self.assertTrue(DEGRADED.can_tune)
        self.assertTrue(DEGRADED.can_report)


class TestPeakHelper(unittest.TestCase):
    """exceeds_measured_peak: 阈值常数与缺证据不触发"""

    def test_slack_constant_matches_design(self):
        self.assertEqual(PEAK_SLACK, 1.05)

    def test_triggers_above_slack(self):
        self.assertTrue(exceeds_measured_peak(106.0, 100.0))
        self.assertFalse(exceeds_measured_peak(105.0, 100.0))   # 边界不触发 (严格大于)
        self.assertFalse(exceeds_measured_peak(104.0, 100.0))

    def test_missing_evidence_does_not_trigger(self):
        """缺峰值 / 缺读数时不猜「超了」—— 缺证据 ≠ 超上界"""
        self.assertFalse(exceeds_measured_peak(None, 100.0))
        self.assertFalse(exceeds_measured_peak(999.0, None))
        self.assertFalse(exceeds_measured_peak(999.0, 0.0))


if __name__ == "__main__":
    unittest.main()
