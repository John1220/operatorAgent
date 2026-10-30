"""LeetCUDA pilot 评分器的离线标定

三件事:
  1. 题目集结构 (≥20 题, pid 唯一, 参考可调用, split 按族切开)
  2. 评分器非空转 (oracle 全过 / 坏核全挂)
  3. 难度曲线聚合方向正确
"""
import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from benchmarks.leetcuda_bench.evaluate import (aggregate, broken_code, format_curve,  # noqa: E402
                                                oracle_code, run_suite)
from benchmarks.leetcuda_bench.problems import PROBLEMS, make_inputs, splits  # noqa: E402


class TestProblemSetShape(unittest.TestCase):
    def test_at_least_twenty_unique_pids(self):
        self.assertGreaterEqual(len(PROBLEMS), 20)
        self.assertEqual(len(PROBLEMS), len({p.pid for p in PROBLEMS}))

    def test_every_reference_is_callable_and_accepts_its_inputs(self):
        for p in PROBLEMS:
            xs = make_inputs(p, seed=0)
            out = p.reference(**xs)
            self.assertIsNotNone(out, p.pid)

    def test_splits_do_not_mix_families(self):
        """**同一个 reference 函数**不许跨 split —— 公式一样只换 shape 就是泄漏。

        家族可以跨 (relu 与 hardshrink 公式不同); 向量打包变体 (pilot 未收录)
        若加入必须与语义题同侧。这条守卫的是「同公式的两题一半 train 一半 test」。
        """
        ref_split = {}
        for p in PROBLEMS:
            ref_split.setdefault(p.reference, set()).add((p.pid, p.split))
        mixed = {s for s in ref_split.values() if len({x[1] for x in s}) > 1}
        self.assertEqual(set(), mixed,
                         f"同一个参考函数跨了 split: {mixed}")

    def test_both_splits_are_nonempty(self):
        s = splits()
        self.assertGreaterEqual(len(s["dev"]), 10)
        self.assertGreaterEqual(len(s["test"]), 5)

    def test_input_spec_keys_match_reference_params(self):
        """reference(**inputs) 按关键字调用 —— 键名对不上就是 TypeError"""
        import inspect
        for p in PROBLEMS:
            sig = inspect.signature(p.reference)
            params = {n for n in sig.parameters
                      if sig.parameters[n].default is inspect.Parameter.empty}
            missing = set(p.input_spec) - params
            self.assertEqual(set(), missing,
                             f"{p.pid} input_spec 有 {missing} 但 reference 不收")


class TestScorerIsNotVacuous(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oracle_runs = run_suite({p.pid: oracle_code(p) for p in PROBLEMS})

    def test_oracle_passes_every_problem(self):
        bad = [r for r in self.oracle_runs if not r["passed"]]
        self.assertEqual([], bad, f"oracle 挂了: {bad}")

    def test_broken_kernel_fails_every_problem(self):
        runs = run_suite({p.pid: broken_code(p) for p in PROBLEMS})
        self.assertEqual(0, sum(1 for r in runs if r["passed"]),
                         "×0.5 坏核居然过了 —— 评分器空转")

    def test_broken_kernel_reports_first_bad(self):
        runs = run_suite({p.pid: broken_code(p) for p in PROBLEMS})
        with_fb = [r for r in runs if r.get("first_bad_index") is not None]
        self.assertGreaterEqual(len(with_fb), 15,
                                "坏核大面积没报 first_bad —— LeetCUDA 那条借鉴失效了")


class TestDifficultyCurve(unittest.TestCase):
    def _runs(self, specs):
        return [{"pid": pid, "difficulty": d, "family": f, "split": s,
                 "passed": ok, "message": "" if ok else "x", "first_bad_index": None}
                for pid, d, f, s, ok in specs]

    def test_curve_moves_with_difficulty(self):
        runs = self._runs([
            ("E1", "easy", "a", "dev", True),
            ("E2", "easy", "a", "dev", True),
            ("M1", "medium", "b", "dev", True),
            ("M2", "medium", "b", "test", False),
            ("H1", "hard", "c", "test", False),
        ])
        agg = aggregate(runs)
        by = {r["key"]: r["pct"] for r in agg["by_difficulty"]}
        self.assertEqual(100.0, by["easy"])
        self.assertEqual(50.0, by["medium"])
        self.assertEqual(0.0, by["hard"])
        self.assertGreater(by["easy"], by["medium"])
        self.assertGreater(by["medium"], by["hard"])

    def test_split_rows_are_separate(self):
        runs = self._runs([
            ("E1", "easy", "a", "dev", True),
            ("T1", "easy", "a", "test", False),
        ])
        by = {r["key"]: r["pct"] for r in aggregate(runs)["by_split"]}
        self.assertEqual(100.0, by["dev"])
        self.assertEqual(0.0, by["test"])

    def test_format_curve_mentions_holdout_warning(self):
        runs = self._runs([("E1", "easy", "a", "test", True)])
        text = format_curve(aggregate(runs))
        self.assertIn("holdout", text)
        self.assertIn("难度", text)


class TestCliSelfTestExitsClean(unittest.TestCase):
    def test_self_test_subprocess_zero(self):
        r = subprocess.run(
            [sys.executable, "-B", "-m", "benchmarks.leetcuda_bench", "--self-test"],
            cwd=str(REPO), capture_output=True, text=True,
            env={k: v for k, v in __import__("os").environ.items()
                 if not k.startswith("OPERATOR_AGENT_")},
        )
        self.assertEqual(0, r.returncode, r.stdout[-800:] + r.stderr[-400:])
        self.assertIn("oracle 全过", r.stdout)

    def test_fixture_subprocess_zero(self):
        r = subprocess.run(
            [sys.executable, "-B", "-m", "benchmarks.leetcuda_bench", "--check-fixture"],
            cwd=str(REPO), capture_output=True, text=True,
            env={k: v for k, v in __import__("os").environ.items()
                 if not k.startswith("OPERATOR_AGENT_")},
        )
        self.assertEqual(0, r.returncode, r.stdout[-800:] + r.stderr[-400:])
        self.assertIn("坏核全挂", r.stdout)


if __name__ == "__main__":
    unittest.main()
