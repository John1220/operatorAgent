"""`operator_agent.repair_loop` 的守卫：判 → 处方 → 修 → 再判 的控制流

这一层不产生新判据（全部来自 `Verifier` / `failure_route` / `Fixer`），所以守卫只钉
**控制流**，用替身把每条分支都走到，不调 LLM、不碰 GPU：

1. 一次就过 ⇒ **不调 Fixer**（别对着通过的结果开处方）；
2. 失败 + 可行动处方 ⇒ 调一次 Fixer，用第二份代码再判；
3. `hold`（这一轮没判出对象）⇒ **不调 Fixer**，停并说明；
4. 模型只提问不给代码 ⇒ 停，把提问原文带出来；
5. 轮数用尽 ⇒ 停，不假装成功。
"""

import unittest

from operator_agent.generate.repair_loop import repair
from operator_agent.core.schemas import (VerificationLevel, VerificationResult)


class FakeVerifier:
    """按调用次序返回预设结果。"""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def verify(self, code, task, inputs, expected_output=None, target_gflops=0.0,
               environment_drift=None, stop_after=None, heartbeat=None,
               heartbeat_expected=False):
        self.calls.append(code)
        return self.outcomes[min(len(self.calls) - 1, len(self.outcomes) - 1)]


class FakeFixer:
    def __init__(self, code="fixed-code", summary="修好了"):
        self.code = code
        self.summary = summary
        self.calls = []

    def run_with_action(self, code, task, failed_results, **kwargs):
        self.calls.append({"code": code, "failed": len(failed_results), **kwargs})

        class Outcome:
            pass
        out = Outcome()
        out.code = self.code
        out.summary = self.summary
        out.envelope = None
        return out


def ok():
    return [VerificationResult(VerificationLevel.L2_FUNCTION, True, "通过")]


def bad(message="数值不对"):
    return [VerificationResult(VerificationLevel.L2_FUNCTION, False, message)]


def hold():
    # 参考侧无对象这一族：classify 会判 hold / silent
    return [VerificationResult(VerificationLevel.L3_PERFORMANCE, False,
                               "reference_uninformative: 参考侧没有可比对象")]


class TestRepairControlFlow(unittest.TestCase):

    def test_a_passing_kernel_never_calls_the_fixer(self):
        verifier, fixer = FakeVerifier([ok()]), FakeFixer()
        out = repair("code", None, {}, verifier, fixer, max_rounds=3)
        self.assertTrue(out["passed"])
        self.assertEqual(fixer.calls, [])
        self.assertEqual([r["action"] for r in out["rounds"]], ["done"])

    def test_a_failure_is_fixed_once_then_reverified(self):
        verifier = FakeVerifier([bad(), ok()])
        fixer = FakeFixer(code="fixed")
        out = repair("broken", None, {}, verifier, fixer, max_rounds=3)
        self.assertTrue(out["passed"])
        self.assertEqual(len(fixer.calls), 1)
        self.assertEqual(fixer.calls[0]["code"], "broken")
        self.assertEqual([r["action"] for r in out["rounds"]], ["fixed", "done"])
        self.assertEqual(verifier.calls, ["broken", "fixed"])
        self.assertTrue(out["rounds"][0]["route"]["fix_not_rewrite"])

    def test_hold_does_not_trigger_a_fix(self):
        verifier, fixer = FakeVerifier([hold()]), FakeFixer()
        out = repair("code", None, {}, verifier, fixer, max_rounds=3)
        self.assertFalse(out["passed"])
        self.assertEqual(fixer.calls, [], "hold 是「没判出对象」，不该开处方")
        self.assertEqual(out["rounds"][0]["action"], "stop")
        self.assertIn("hold", out["rounds"][0]["why"])

    def test_an_answer_without_code_stops_and_says_why(self):
        verifier = FakeVerifier([bad()])
        fixer = FakeFixer(code="")
        fixer.summary = "请给我 profiling 数据"
        out = repair("code", None, {}, verifier, fixer, max_rounds=3)
        self.assertFalse(out["passed"])
        self.assertEqual(out["rounds"][0]["action"], "ask")
        self.assertIn("请给我 profiling 数据", out["rounds"][0]["why"])
        self.assertEqual(len(verifier.calls), 1, "只提问时不该再判一次")

    def test_budget_exhaustion_is_reported_not_hidden(self):
        verifier = FakeVerifier([bad()])
        fixer = FakeFixer(code="still-bad")
        out = repair("code", None, {}, verifier, fixer, max_rounds=2)
        self.assertFalse(out["passed"])
        self.assertEqual([r["action"] for r in out["rounds"]],
                         ["fixed", "budget_exhausted"])
        self.assertEqual(len(fixer.calls), 1)

    def test_max_rounds_must_be_positive(self):
        with self.assertRaises(ValueError):
            repair("code", None, {}, FakeVerifier([ok()]), FakeFixer(), max_rounds=0)


if __name__ == "__main__":
    unittest.main()
