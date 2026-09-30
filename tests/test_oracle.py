"""Oracle 唯一收口点的离线标定 (设计 §13.4)

三条硬性语义必须在 evaluate **内部**而不是调用方:
  1. 契约哈希不匹配 → 拒评, **不调用 measure()**
  2. 心跳失败窗口 → 全资格 False (但数仍记录)
  3. 资格决定后果: DEGRADED 不许 can_persist
外加: 入库 = can_persist **且** 两次确认。
"""
import unittest

from operator_agent.contract import BaselineDef, DTypeSemantics, ShapeDist, freeze
from operator_agent.confirm import ConfirmRecord
from operator_agent.oracle import admit_to_corpus, evaluate


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


class TestRefuseOnContractHashMismatch(unittest.TestCase):
    def test_measure_is_not_even_called(self):
        c = _contract()
        called = []

        def m():
            called.append(1)
            return _measure()

        obs = evaluate(c, "0" * 64, m)
        self.assertEqual([], called, "哈希不匹配却仍调用了 measure() —— 拒评是空的")
        self.assertTrue(obs.refused)
        self.assertFalse(obs.eligibility.can_verdict)
        self.assertFalse(obs.eligibility.can_persist)

    def test_matching_hash_does_measure(self):
        c = _contract()
        obs = evaluate(c, c.hash, lambda: _measure(1.3))
        self.assertEqual(1.3, obs.value)
        self.assertTrue(obs.eligibility.can_persist)


class TestHeartbeatWindowInvalidates(unittest.TestCase):
    def test_ruler_broken_forces_all_false_even_if_value_looks_good(self):
        from operator_agent.calibration import HeartbeatResult
        hb = HeartbeatResult(ok=False, epoch=3, reasons=["RULER_BROKEN"],
                             known_good_ratio=2.0, known_slow_ratio=1.0, window_id="w1")
        obs = evaluate(_contract(), _contract().hash,
                       lambda: _measure(2.0, "valid"), heartbeat=hb)
        self.assertFalse(obs.eligibility.can_verdict)
        self.assertFalse(obs.eligibility.can_persist)
        self.assertTrue(any("RULER_BROKEN" in r for r in obs.reasons))
        self.assertEqual(2.0, obs.value, "数仍记录 —— 只是不让它造成后果")

    def test_good_heartbeat_keeps_eligibility(self):
        from operator_agent.calibration import HeartbeatResult
        hb = HeartbeatResult(ok=True, epoch=3, reasons=[],
                             known_good_ratio=1.0, known_slow_ratio=3.0, window_id="w1")
        obs = evaluate(_contract(), _contract().hash,
                       lambda: _measure(1.5), heartbeat=hb)
        self.assertTrue(obs.eligibility.can_persist)


class TestEligibilityDrivesConsequences(unittest.TestCase):
    def test_degraded_cannot_persist_or_count_progress(self):
        obs = evaluate(_contract(), _contract().hash,
                       lambda: _measure(1.2, "degraded", cv_exceeded=True))
        self.assertTrue(obs.eligibility.can_verdict)
        self.assertTrue(obs.eligibility.can_tune)
        self.assertTrue(obs.eligibility.can_report)
        self.assertFalse(obs.eligibility.can_persist)
        self.assertFalse(obs.eligibility.can_count_progress)

    def test_real_device_missing_is_all_false(self):
        obs = evaluate(_contract(), _contract().hash,
                       lambda: _measure(1.2, "valid", real_device=False),
                       require_real_device=True)
        self.assertFalse(obs.eligibility.can_verdict)
        self.assertFalse(obs.eligibility.can_persist)

    def test_lib_fallback_is_all_false(self):
        obs = evaluate(_contract(), _contract().hash,
                       lambda: _measure(5.0, "valid", lib_fallback=True))
        self.assertFalse(obs.eligibility.can_persist)

    def test_peak_exceeded_is_uninformative(self):
        obs = evaluate(_contract(), _contract().hash,
                       lambda: _measure(200.0, "valid"), peak=100.0)
        self.assertFalse(obs.eligibility.can_verdict)
        self.assertEqual("uninformative", obs.validity)


class TestAdmitNeedsBothEligibilityAndConfirmation(unittest.TestCase):
    def _good_obs(self):
        return evaluate(_contract(), _contract().hash, lambda: _measure(1.3, "valid"))

    def test_pass_with_independent_remeasure(self):
        ok, _ = admit_to_corpus(
            self._good_obs(), 1.3, "valid",
            ConfirmRecord(ok=True, speedup=1.25, validity="valid"))
        self.assertTrue(ok)

    def test_refused_without_remeasure(self):
        ok, msg = admit_to_corpus(self._good_obs(), 1.3, "valid", None)
        self.assertFalse(ok)
        self.assertIn("两次确认", msg)

    def test_refused_when_eligibility_blocks_even_if_two_passes(self):
        obs = evaluate(_contract(), _contract().hash,
                       lambda: _measure(1.3, "degraded", cv_exceeded=True))
        ok, msg = admit_to_corpus(
            obs, 1.3, "degraded",
            ConfirmRecord(ok=True, speedup=1.2, validity="valid"))
        self.assertFalse(ok)
        self.assertIn("can_persist", msg)

    def test_direction_faster_second_pass_is_better(self):
        obs = self._good_obs()
        good = admit_to_corpus(obs, 1.2, "valid",
                               ConfirmRecord(ok=True, speedup=1.4, validity="valid"))
        bad = admit_to_corpus(obs, 1.2, "valid",
                              ConfirmRecord(ok=True, speedup=0.5, validity="valid"))
        self.assertTrue(good[0])
        self.assertFalse(bad[0])


if __name__ == "__main__":
    unittest.main()
