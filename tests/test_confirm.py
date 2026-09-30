"""重测确认 + 「无法确立」报告的离线标定

设计 §10.5-4 / §17: 一次侥幸读数不能变成永久真理; 从不说测不准的系统不可信。
"""
import unittest

from operator_agent.confirm import (ConfirmRecord, UnableToEstablish,
                                    confirm_persist)


class TestTwoConfirmationGate(unittest.TestCase):
    def test_pass_needs_a_valid_first_and_independent_remeasure(self):
        ok, msg = confirm_persist(1.2, "valid",
                                 ConfirmRecord(ok=True, speedup=1.15, validity="valid"))
        self.assertTrue(ok, msg)

    def test_no_remeasure_is_refused_not_implicitly_passed(self):
        ok, msg = confirm_persist(1.2, "valid", None)
        self.assertFalse(ok)
        self.assertIn("两次确认", msg)

    def test_dependent_remeasure_does_not_count(self):
        ok, msg = confirm_persist(1.2, "valid",
                                 ConfirmRecord(ok=True, speedup=1.15,
                                               validity="valid", independent=False))
        self.assertFalse(ok)
        self.assertIn("独立", msg)

    def test_remeasure_dropping_below_threshold_refuses(self):
        ok, msg = confirm_persist(1.5, "valid",
                                 ConfirmRecord(ok=True, speedup=0.9, validity="valid"))
        self.assertFalse(ok)
        self.assertIn("门槛", msg)

    def test_uninformative_first_never_persists(self):
        ok, msg = confirm_persist(1.2, "uninformative",
                                 ConfirmRecord(ok=True, speedup=1.1, validity="valid"))
        self.assertFalse(ok)
        self.assertIn("uninformative", msg)

    def test_uninformative_remeasure_cannot_confirm(self):
        ok, msg = confirm_persist(1.2, "valid",
                                 ConfirmRecord(ok=True, speedup=1.1,
                                               validity="uninformative"))
        self.assertFalse(ok)
        self.assertIn("uninformative", msg)

    def test_direction_slower_remeasure_is_worse_than_faster(self):
        worse = confirm_persist(1.2, "valid",
                                ConfirmRecord(ok=True, speedup=0.5, validity="valid"))
        better = confirm_persist(1.2, "valid",
                                 ConfirmRecord(ok=True, speedup=1.3, validity="valid"))
        self.assertFalse(worse[0])
        self.assertTrue(better[0])

    def test_degraded_remeasure_still_works_if_above_gate(self):
        """DEGRADED 是「数不够做永久决定」, 但重测只验证「没掉回门槛」——
        不需要 VALID 才能确认。与 can_persist 只在 VALID 放开是两件事:
        那是「首次」的资格, 这是「第二次」的确认。"""
        ok, _ = confirm_persist(1.2, "valid",
                                ConfirmRecord(ok=True, speedup=1.05, validity="degraded"))
        self.assertTrue(ok)


class TestUnableToEstablishIsFirstClass(unittest.TestCase):
    def test_metrics_extend_picks_up_gate_failures(self):
        u = UnableToEstablish()
        u.extend_metrics({
            "measurement_can_gate": False,
            "measurement_validity": "uninformative",
            "measurement_validity_reasons": ["launch 地板支配 95%"],
        })
        self.assertEqual(1, len(u.items))
        self.assertIn("launch", u.items[0].reason)

    def test_ruler_broken_and_contract_mismatch_are_separate_claims(self):
        u = UnableToEstablish()
        u.extend_metrics({"ruler_broken": True, "contract_hash_mismatch": True})
        kinds = {i.kind for i in u.items}
        self.assertEqual({"ruler", "contract"}, kinds)

    def test_render_mentions_not_a_negative_claim(self):
        u = UnableToEstablish()
        u.add("某加速比测不准", "cv 超阈值", kind="measurement")
        text = u.render()
        self.assertIn("我无法确立的事", text)
        self.assertIn("不许", text)

    def test_empty_still_prints_the_section(self):
        """空章节也要在 —— 「没有无法确立的事」本身是一个断言, 不是省略"""
        self.assertIn("无", UnableToEstablish().render())

    def test_contaminated_window_is_recorded(self):
        u = UnableToEstablish()
        u.extend_metrics({"benchmark_contaminated": True})
        self.assertTrue(any("抢占" in i.claim or "抢占" in i.reason for i in u.items))


if __name__ == "__main__":
    unittest.main()
