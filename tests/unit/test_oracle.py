"""Oracle 唯一收口点的离线标定 (设计 §13.4 / spine-review P0-2)

三条硬性语义必须在 evaluate **内部**而不是调用方:
  1. 契约哈希**记录携带** —— 不是「调用方同时给两侧」的恒真比较
  2. 心跳失败窗口 → 全资格 False (但数仍记录)
  3. 资格决定后果: DEGRADED 不许 can_persist
外加: 入库 = can_persist **且** 两次确认。
"""
import unittest

from operator_agent.decide.calibration import HeartbeatResult
from operator_agent.core.confirm import ConfirmRecord
from operator_agent.core.contract import BaselineDef, DTypeSemantics, ShapeDist, freeze
from operator_agent.decide.oracle import admit_to_corpus, evaluate


def _contract():
    return freeze(
        kernel_type="matmul",
        dtype_semantics=DTypeSemantics(dtype="fp16"),
        baseline=BaselineDef(name="cuBLAS_fp16"),
        shapes=ShapeDist(entries=(((256, 256, 256), 1.0),)),
        max_error=1e-2,
        min_speedup=1.0,
    )


def _measure(value=1.2, validity="valid", **kw):
    d = {"value": value, "validity": validity}
    d.update(kw)
    return d


def _eval(c=None, measure=None, **kw):
    c = c or _contract()
    measure = measure or (lambda: _measure())
    return evaluate(c, measure, record_hash=c.hash, **kw)


class TestRecordCarriedHash(unittest.TestCase):
    """spine-review P0-2: 比较两侧不许都由调用方随手给"""

    def test_missing_record_hash_refuses_without_measuring(self):
        c = _contract()
        called = []
        obs = evaluate(c, lambda: called.append(1) or _measure())
        self.assertEqual([], called, "没带哈希却仍调用了 measure()")
        self.assertTrue(obs.refused)
        self.assertIn("记录未携带", obs.refused)

    def test_wrong_record_hash_refuses_without_measuring(self):
        c = _contract()
        called = []
        obs = evaluate(c, lambda: called.append(1) or _measure(),
                       record_hash="0" * 64)
        self.assertEqual([], called)
        self.assertTrue(obs.refused)
        self.assertFalse(obs.eligibility.can_persist)

    def test_matching_hash_does_measure(self):
        c = _contract()
        obs = evaluate(c, lambda: _measure(1.3), record_hash=c.hash)
        self.assertEqual(1.3, obs.value)
        self.assertTrue(obs.eligibility.can_persist)

    def test_tampered_contract_and_original_hash_refuses(self):
        """记录带着旧哈希、契约被改过 → 拒评 (这才是「改不了尺子」)"""
        c1 = _contract()
        c2 = freeze(
            kernel_type=c1.kernel_type, dtype_semantics=c1.dtype_semantics,
            baseline=c1.baseline, shapes=c1.shapes,
            max_error=1e-1, min_speedup=c1.min_speedup,
        )
        obs = evaluate(c2, lambda: _measure(2.0), record_hash=c1.hash)
        self.assertTrue(obs.refused)
        self.assertEqual("mismatch", obs.hash_check,
                         "不匹配要被机器可读地记下来, 而不是只留一句文案")

    def test_fresh_observation_measures_without_claiming_verification(self):
        """新观测 (没有旧记录可校): 必须能测, 但结论要写明「没校验过」

        spine-review P0-2 的要害不是「拒评一切」, 而是**不许把没得校说成校验过**。
        生产路径上每个 L3 读数都是新观测 —— 若这里也拒评, 整条闭环直接死。
        """
        c = _contract()
        obs = evaluate(c, lambda: _measure(1.3), fresh=True)
        self.assertEqual(1.3, obs.value, "新观测被误拒: 没得校不等于不能测")
        self.assertFalse(obs.refused)
        self.assertEqual("fresh_unverified", obs.hash_check)
        # 溯源戳仍在: 这条读数是在哪个契约下取得的 (供日后复用比对)
        self.assertEqual(c.hash, obs.provenance["contract_hash"])
        self.assertTrue(obs.eligibility.can_persist,
                        "新观测不该因为「没有旧记录」这个结构性事实而被降格")

    def test_verified_and_fresh_are_distinguishable(self):
        """「校验过」与「没得校」必须在结论上可区分 —— 拆掉恒真的全部意义在这"""
        c = _contract()
        verified = evaluate(c, lambda: _measure(1.3), record_hash=c.hash)
        fresh = evaluate(c, lambda: _measure(1.3), fresh=True)
        self.assertEqual("verified", verified.hash_check)
        self.assertEqual("fresh_unverified", fresh.hash_check)

    def test_fresh_with_record_hash_refuses_as_ambiguous(self):
        """两种形态同时给出 = 说不清是新观测还是复用旧记录 → 拒评, 不许挑一个信"""
        c = _contract()
        called = []
        obs = evaluate(c, lambda: called.append(1) or _measure(),
                       record_hash=c.hash, fresh=True)
        self.assertEqual([], called, "说不清形态却仍然测了")
        self.assertTrue(obs.refused)
        self.assertIn("同时给出", obs.refused)
        self.assertEqual("missing", obs.hash_check)


class TestHeartbeatWindowInvalidates(unittest.TestCase):
    def test_ruler_broken_forces_all_false_even_if_value_looks_good(self):
        hb = HeartbeatResult(ok=False, epoch=3, reasons=["RULER_BROKEN"],
                             known_good_ratio=2.0, known_slow_ratio=1.0,
                             window_id="w1")
        obs = _eval(measure=lambda: _measure(2.0, "valid"), heartbeat=hb)
        self.assertFalse(obs.eligibility.can_verdict)
        self.assertFalse(obs.eligibility.can_persist)
        self.assertTrue(any("RULER_BROKEN" in r for r in obs.reasons))
        self.assertEqual(2.0, obs.value, "数仍记录 —— 只是不让它造成后果")

    def test_good_heartbeat_keeps_eligibility(self):
        hb = HeartbeatResult(ok=True, epoch=3, reasons=[],
                             known_good_ratio=1.0, known_slow_ratio=3.0,
                             window_id="w1")
        obs = _eval(measure=lambda: _measure(1.5), heartbeat=hb)
        self.assertTrue(obs.eligibility.can_persist)


class TestHeartbeatDueButMissing(unittest.TestCase):
    """spine-review P1-5: 「没做心跳」不许读成「心跳通过」

    判据是**到期**(由拿着测量计数的那一层明说), 不是「永远要求」: 短跑/离线路径上
    心跳本来就不到期, 一律要求会把每一次读数都降级 —— 与打开
    `require_double_confirm` 就把入库率归零是同一个陷阱。
    """

    def test_due_but_missing_breaks_the_ruler(self):
        obs = _eval(heartbeat=None, heartbeat_expected=True)
        self.assertTrue(obs.ruler_broken)
        self.assertFalse(obs.eligibility.can_verdict,
                         "到期没跑心跳却仍然判决了 —— 「忘了跑」成了静默放行通道")
        self.assertFalse(obs.eligibility.can_persist)
        self.assertEqual("RULER_BROKEN",
                         obs.provenance["heartbeat_reasons"][0].split(":")[0])
        self.assertIsNotNone(obs.value, "数仍要记录 —— 只是不让它造成后果")

    def test_not_due_and_missing_is_not_a_ruler_claim(self):
        """没到期就没得说; 但也**不许**说成「检查过、是好的」"""
        obs = _eval()
        self.assertFalse(obs.ruler_broken)
        self.assertTrue(obs.eligibility.can_verdict)
        self.assertFalse(obs.provenance["heartbeat_checked"])
        self.assertFalse(obs.provenance["heartbeat_expected"])

    def test_checked_and_passed_says_so(self):
        hb = HeartbeatResult(ok=True, epoch=1, reasons=[], known_good_ratio=1.0,
                             known_slow_ratio=3.0, window_id="hb-1")
        obs = _eval(heartbeat=hb, heartbeat_expected=True)
        self.assertFalse(obs.ruler_broken)
        self.assertTrue(obs.eligibility.can_verdict)
        self.assertTrue(obs.provenance["heartbeat_checked"])
        self.assertEqual("hb-1", obs.provenance["heartbeat_window_id"])

    def test_a_broken_heartbeat_only_taints_its_own_window(self):
        """§13.5: 一张卡/一扇窗的尺子坏了, 不牵连别的窗口"""
        hb = HeartbeatResult(ok=False, epoch=1, reasons=["RULER_BROKEN: 分离度不足"],
                             known_good_ratio=1.0, known_slow_ratio=1.0,
                             window_id="hb-1")
        same = _eval(heartbeat=hb, candidate_window="hb-1")
        other = _eval(heartbeat=hb, candidate_window="hb-2")
        self.assertTrue(same.ruler_broken)
        self.assertFalse(other.ruler_broken,
                         "跨窗仍被牵连 —— 窗口判据 (ruler_broken_window) 没有被派发")
        self.assertTrue(other.eligibility.can_verdict)

    def test_a_passing_heartbeat_from_another_window_backs_nothing(self):
        """§13.5 的另一面: 别窗的绿也不许借 —— 本次读数没被检查过

        「检查过」若按「收到过一张心跳」记, 递一张陈旧心跳就能把没被测过的窗口洗成
        已检查 (报表上显示为已检查)。
        """
        hb = HeartbeatResult(ok=True, epoch=1, reasons=[], known_good_ratio=1.0,
                             known_slow_ratio=3.0, window_id="hb-1")
        obs = _eval(heartbeat=hb, candidate_window="hb-2")
        self.assertFalse(obs.ruler_broken)
        self.assertTrue(obs.eligibility.can_verdict)
        self.assertFalse(obs.provenance["heartbeat_checked"],
                         "别窗的心跳被记成了「本次检查过」")
        self.assertEqual("hb-1", obs.provenance["heartbeat_window_id"])

    def test_another_windows_heartbeat_does_not_satisfy_a_due_check(self):
        """到期了却只递来别窗的读数 → 本窗仍算「到期没跑」, 按坏处理"""
        hb = HeartbeatResult(ok=True, epoch=1, reasons=[], known_good_ratio=1.0,
                             known_slow_ratio=3.0, window_id="hb-1")
        obs = _eval(heartbeat=hb, heartbeat_expected=True, candidate_window="hb-2")
        self.assertTrue(obs.ruler_broken, "别窗的心跳冒充了本窗的到期检查")
        self.assertFalse(obs.eligibility.can_verdict)
        self.assertIn("别窗", " ".join(obs.provenance["heartbeat_reasons"]))


class TestEligibilityDrivesConsequences(unittest.TestCase):
    def test_degraded_cannot_persist_or_count_progress(self):
        obs = _eval(measure=lambda: _measure(1.2, "degraded", cv_exceeded=True))
        self.assertTrue(obs.eligibility.can_verdict)
        self.assertTrue(obs.eligibility.can_tune)
        self.assertTrue(obs.eligibility.can_report)
        self.assertFalse(obs.eligibility.can_persist)
        self.assertFalse(obs.eligibility.can_count_progress)

    def test_real_device_missing_is_all_false(self):
        obs = _eval(measure=lambda: _measure(1.2, "valid", real_device=False),
                    require_real_device=True)
        self.assertFalse(obs.eligibility.can_verdict)
        self.assertFalse(obs.eligibility.can_persist)

    def test_lib_fallback_is_all_false(self):
        obs = _eval(measure=lambda: _measure(5.0, "valid", lib_fallback=True))
        self.assertFalse(obs.eligibility.can_persist)

    def test_peak_exceeded_is_uninformative(self):
        obs = _eval(measure=lambda: _measure(200.0, "valid"), peak=100.0)
        self.assertFalse(obs.eligibility.can_verdict)
        self.assertEqual("uninformative", obs.validity)


class TestAdmitNeedsBothEligibilityAndConfirmation(unittest.TestCase):
    """入库判据的输入是**生产路径真正持有的东西**: 序列化的 metrics + 重测记录

    WHY (2026-10-01, spine-review ①): 判据原先要一个 `Observation` 对象, 而生产路径
    (`VerificationResult.metrics`) 手里只有 JSON 化的读数 ⇒ 调用方只能自己再写一遍三道门,
    于是同一个决定有两个实现、两种理由文本, 用例测的是没有生产调用者的那一个。
    现在这里的输入形状与 `Conductor._save_to_db` 传进去的完全一致 —— 包括
    `eligibility` 与 `observation` 两个键都由 verifier 写入这一事实。
    """

    def _good_obs(self, **kw):
        return _eval(measure=lambda: _measure(1.3, "valid", **kw))

    def _metrics(self, obs, **extra):
        """verifier 写进 metrics 的那几格 (production shape)"""
        m = {
            "speedup_vs_baseline": obs.value,
            "measurement_validity": obs.validity,
            "eligibility": obs.eligibility.as_dict(),
            "observation": obs.as_dict(),
            "measurement_can_gate": True,
            "backend_real_device": True,
        }
        m.update(extra)
        return m

    def _remeasure(self, **kw):
        """盖了契约戳的重测记录 —— 没戳的现在会被判「无从证明是同一条判据」"""
        kw.setdefault("contract_hash", self._good_obs().provenance["contract_hash"])
        return ConfirmRecord(**kw)

    def test_pass_with_independent_remeasure(self):
        ok, _ = admit_to_corpus(
            self._metrics(self._good_obs()),
            self._remeasure(ok=True, speedup=1.25, validity="valid"))
        self.assertTrue(ok)

    def test_refused_without_remeasure(self):
        ok, msg = admit_to_corpus(self._metrics(self._good_obs()), None)
        self.assertFalse(ok)
        self.assertIn("两次确认", msg)

    def test_the_not_required_mode_says_so_in_the_reason(self):
        """默认关的那道门必须在**理由里**留下痕迹

        否则「入库了」会被读成「两次确认通过了」—— 而实际是一次读数。"""
        ok, msg = admit_to_corpus(self._metrics(self._good_obs()), None,
                                  requires_remeasure=False)
        self.assertTrue(ok)
        self.assertIn("未要求", msg)
        self.assertIn("不是「两次确认已通过」", msg)

    def test_refused_when_eligibility_blocks_even_if_two_passes(self):
        obs = _eval(measure=lambda: _measure(1.3, "degraded", cv_exceeded=True))
        ok, msg = admit_to_corpus(
            self._metrics(obs),
            ConfirmRecord(ok=True, speedup=1.2, validity="valid",
                          contract_hash=obs.provenance["contract_hash"]))
        self.assertFalse(ok, "can_persist=False 却因为两次确认过了就入库")
        self.assertIn("can_persist", msg)

    def test_the_first_gate_fires_before_the_others(self):
        """门 1 (读数够不够格) 短路 —— 它拒了就不该走到确认那一步"""
        obs = self._good_obs()
        ok, msg = admit_to_corpus(
            self._metrics(obs, measurement_can_gate=False),
            self._remeasure(ok=True, speedup=1.25, validity="valid"))
        self.assertFalse(ok)
        self.assertIn("can_gate=False", msg)

    def test_a_legacy_reading_without_eligibility_is_not_re_judged(self):
        """缺键 ≠ 判过且不合格: 老产物没经过收口点, 不做追溯性重判

        (与 `persist_decision` 对 `backend_real_device` 的 `is False` 同一条纪律。)
        """
        m = {"speedup_vs_baseline": 1.3, "measurement_validity": "valid",
             "measurement_can_gate": True, "backend_real_device": True}
        ok, _ = admit_to_corpus(m, None, requires_remeasure=False)
        self.assertTrue(ok, "老产物被新门追溯性拒了")

    def test_a_present_but_incomplete_eligibility_mask_refuses(self):
        """有键而掩码缺一位 ⇒ 没有「能入库」的凭据 (fail-closed)"""
        m = {"speedup_vs_baseline": 1.3, "measurement_validity": "valid",
             "measurement_can_gate": True, "backend_real_device": True,
             "eligibility": {"can_verdict": True, "can_tune": True}}
        ok, msg = admit_to_corpus(m, None, requires_remeasure=False)
        self.assertFalse(ok)
        self.assertIn("can_persist", msg)

    def test_direction_faster_second_pass_is_better(self):
        obs = self._good_obs()
        m = self._metrics(obs)
        good = admit_to_corpus(m, self._remeasure(ok=True, speedup=1.4, validity="valid"))
        bad = admit_to_corpus(m, self._remeasure(ok=True, speedup=0.5, validity="valid"))
        self.assertTrue(good[0])
        self.assertFalse(bad[0])

    def test_a_remeasure_without_the_contract_stamp_is_refused(self):
        """判据同一性是两次确认的**前提**: 说不清就用不了 (spine-review P0-2 的残留)"""
        ok, msg = admit_to_corpus(
            self._metrics(self._good_obs()),
            ConfirmRecord(ok=True, speedup=1.25, validity="valid"))
        self.assertFalse(ok, "没带契约哈希的重测被当成了有效的第二次确认")
        self.assertIn("契约哈希", msg)

    def test_a_remeasure_under_another_contract_is_refused(self):
        """两次测量之间判据被改写 → 那不是「同一条断言量了两遍」"""
        ok, msg = admit_to_corpus(
            self._metrics(self._good_obs()),
            ConfirmRecord(ok=True, speedup=1.25, validity="valid",
                          contract_hash="f" * 64))
        self.assertFalse(ok)
        self.assertIn("另一条契约", msg)


if __name__ == "__main__":
    unittest.main()
