"""设计 §14-9 对照实验比较器的守卫 (五态判据, 逐态成对)

这个比较器要回答的是设计里唯一**无法从 §14 八条验收里得到**的问题: 资格审计到底有
没有收益。它的失效方式有两种, 都很难看:

  1. **从「门没被需要过」推出「门没用」** —— 两臂一样好, 于是宣布「审计没收益」, 而
     实际是这次实验压根没打到点上 (没有一次本该被拦的读数)。这与 `measurement.assess()`
     里「地板未知 ⇒ 记未做检查」、报告里「空清单必须带依据」是同一条纪律。
  2. **把缺键当成 0** —— 老产物的血缘没有 `permanent_effects`/`eligibility`, 数不出脏账。
     当成 0 就会把「没记」读成「干净」。

所以五态各有自己的用例, 而且都成对 (好/坏、有/无)。另加一条政策守卫: 这个脚本**不许**
长出一个「关掉资格审计」的命令行开关 —— 设计 §10.5-5 说 agent 不能编辑自己的尺子,
一个一键关安全门的开关就是那把尺子。

运行: python3 -m unittest tests.test_ablation_eligibility -v
"""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _load():
    """按路径加载 benchmarks/ablation_eligibility.py (benchmarks 不是包)"""
    path = ROOT / "benchmarks" / "ablation_eligibility.py"
    spec = importlib.util.spec_from_file_location("ablation_eligibility", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ABL = _load()


def _r(n, speedup, *, l3=True, validity="valid", can_tune=True, can_persist=True,
       best_updated=False, disabled=(), effects=True):
    rec = {"round": n, "levels_passed": {"L1": True, "L2": True, "L3": l3},
           "measurement_validity": validity if l3 else None,
           "speedup": speedup if l3 else None,
           "eligibility": ({"can_tune": can_tune, "can_persist": can_persist,
                            "can_verdict": True, "can_report": True,
                            "can_count_progress": can_persist} if l3 else None)}
    if effects:
        rec["permanent_effects"] = {"best_updated": best_updated,
                                    "recipes_disabled": list(disabled),
                                    "stagnation_counted": False}
    return rec


class TestSummarize(unittest.TestCase):
    def test_it_counts_a_dirty_permanent_effect(self):
        """没资格却造成了永久后果 ⇒ 脏账 (这是审计在防的那件事)"""
        s = ABL.summarize([_r(1, 1.1), _r(2, 0.9, validity="degraded", can_tune=False,
                                         can_persist=False, best_updated=True,
                                         disabled=("R-1",))])
        self.assertEqual(1, len(s["dirty"]))
        self.assertEqual(2, s["dirty"][0]["round"])
        self.assertIn("覆盖 best", s["dirty"][0]["effects"])
        self.assertIn("can_tune", s["dirty"][0]["off"])
        self.assertEqual(2, s["measurements"])

    def test_an_entitled_permanent_effect_is_not_dirty(self):
        """反向: 有资格的读数改配置/覆盖 best 是正常工作, 不许被算成脏账"""
        s = ABL.summarize([_r(1, 1.1), _r(2, 1.3, best_updated=True, disabled=("R-1",))])
        self.assertEqual([], s["dirty"])

    def test_missing_effects_are_unknown_not_clean(self):
        """缺键 ≠ 干净: 老产物数不出脏账, 要单独记, 不许当 0"""
        s = ABL.summarize([_r(1, 1.1, effects=False), _r(2, 1.2, best_updated=True)])
        self.assertEqual(1, s["unknown_effects"])
        self.assertEqual([], s["dirty"])

    def test_a_round_without_eligibility_is_unknown_not_dirty(self):
        """有后果但资格位缺 (更老的产物) 同样进 unknown"""
        rec = _r(1, 1.2, best_updated=True)
        rec["eligibility"] = None
        s = ABL.summarize([rec])
        self.assertEqual(1, s["unknown_effects"])
        self.assertEqual([], s["dirty"])

    def test_best_falls_back_to_gflops_with_a_named_caveat(self):
        hist = [{"round": 1, "levels_passed": {"L3": True}, "speedup": None,
                 "gflops": 42.0, "permanent_effects": {"best_updated": True,
                                                       "recipes_disabled": []},
                 "eligibility": {"can_tune": True, "can_persist": True}}]
        s = ABL.summarize(hist)
        self.assertEqual(42.0, s["best"])
        self.assertIn("gflops", s["best_from"])


class TestVerdicts(unittest.TestCase):
    """五态, 逐态成对"""

    A = [_r(1, 1.10, best_updated=True), _r(2, 1.30, best_updated=True)]

    def _v(self, b):
        return ABL.judge(ABL.summarize(self.A), ABL.summarize(b))

    def test_no_benefit_is_the_direction_that_must_be_reported(self):
        """B 一样好 + B 有脏账 ⇒ 「没有收益」, 且理由里必须写「要写进报告」"""
        verdict, why = self._v([_r(1, 1.10, best_updated=True),
                                _r(2, 1.28, validity="uninformative", can_tune=False,
                                   can_persist=False, best_updated=True)])
        self.assertEqual("没有收益", verdict)
        self.assertTrue(any("必须写进报告" in w for w in why), why)

    def test_a_real_loss_is_a_benefit(self):
        verdict, _ = self._v([_r(1, 1.10, best_updated=True),
                              _r(2, 0.90, validity="degraded", can_tune=False,
                                 can_persist=False, best_updated=True)])
        self.assertEqual("有收益", verdict)

    def test_a_borderline_difference_is_not_a_conclusion(self):
        """带外沿: 不下结论 —— 从 3.85% 的差里读不出「审计有用」"""
        verdict, why = self._v([_r(1, 1.10, best_updated=True),
                                _r(2, 1.25, validity="degraded", can_tune=False,
                                   can_persist=False, best_updated=True)])
        self.assertEqual("证据不足", verdict)
        self.assertTrue(any("不下结论" in w for w in why))

    def test_an_unexercised_gate_is_not_evidence_of_uselessness(self):
        """最重要的一态: 门一次都没被需要过 ⇒ 「没打到点上」, 不是「没有收益」"""
        verdict, why = self._v([_r(1, 1.10, best_updated=True),
                                _r(2, 1.32, best_updated=True)])
        self.assertEqual("没打到点上", verdict)
        self.assertTrue(any("推不出" in w for w in why), why)

    def test_unusable_lineage_says_so_instead_of_guessing(self):
        verdict, _ = self._v([_r(1, 0.0, l3=False)])
        self.assertEqual("无法比较", verdict)

    def test_small_samples_are_flagged_in_the_reasons(self):
        _, why = self._v([_r(1, 1.10, best_updated=True), _r(2, 1.28,
                                                             best_updated=True)])
        self.assertTrue(any("轮 (< 5)" in w or "只有" in w for w in why),
                        f"两轮样本必须自我声明: {why}")

    def test_the_verdict_vocabulary_is_exactly_the_five_states(self):
        self.assertEqual({"无法比较", "没打到点上", "没有收益", "证据不足", "有收益"},
                         set(ABL.VERDICTS))


class TestLoadHistory(unittest.TestCase):
    def test_it_reads_a_result_json_from_a_directory(self):
        with tempfile.TemporaryDirectory() as d:
            hist = [_r(1, 1.2, best_updated=True)]
            (Path(d) / "result.json").write_text(json.dumps({"iteration_history": hist}),
                                                 encoding="utf-8")
            got, note = ABL.load_history(d)
            self.assertEqual(1, len(got))
            self.assertIn("result.json", note)

    def test_it_falls_back_to_the_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "execution_snapshot.json").write_text(
                json.dumps({"iteration_history": [_r(1, 1.1)]}), encoding="utf-8")
            got, note = ABL.load_history(d)
            self.assertEqual(1, len(got))
            self.assertIn("execution_snapshot", note)

    def test_a_directory_without_lineage_says_why(self):
        with tempfile.TemporaryDirectory() as d:
            got, note = ABL.load_history(d)
            self.assertIsNone(got)
            self.assertIn("没有 result.json", note)

    def test_an_empty_lineage_is_not_a_zero_length_comparison(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "result.json").write_text(json.dumps({"iteration_history": []}),
                                                 encoding="utf-8")
            got, note = ABL.load_history(d)
            self.assertIsNone(got)
            self.assertIn("iteration_history", note)


class TestPolicy(unittest.TestCase):
    """政策守卫: 这个脚本不许长出「关掉资格审计」的开关"""

    def test_there_is_no_switch_to_turn_the_audit_off(self):
        src = (ROOT / "benchmarks" / "ablation_eligibility.py").read_text(encoding="utf-8")
        for forbidden in ("--disable", "--no-eligibility", "--bypass",
                          "OPERATOR_AGENT_DISABLE"):
            self.assertNotIn(forbidden, src,
                             f"{forbidden}: 一键关掉安全门就是「agent 编辑自己的尺子」"
                             f"(设计 §10.5-5) —— B 组只能在 harness 侧 monkeypatch")
        self.assertIn("不许做成产品开关", src, "这条理由本身要留在文件里")

    def test_the_band_is_taken_from_the_repo_not_invented(self):
        from operator_agent.measurement import NOISE_BAND, Validity
        self.assertEqual(float(NOISE_BAND[Validity.UNINFORMATIVE]), ABL.BAND)


if __name__ == "__main__":
    unittest.main(verbosity=2)