"""§10.2 三道预算门在**生产链路**上的守卫

账本本身 (`tests/test_budget.py`) 只证「算得对」; 这里证「接上了」—— 判据算了却没人读,
正是这个仓最常见的失效方式 (CHECKLOG I46/I47/I49/I50/I52 同一族)。

要钉住四件事:
  1. **oracle 次数用尽 ⇒ 停, 且停因自成一条词汇** (`budget_exhausted`), 不是 `rounds_exhausted`;
  2. **预留不可挪用**: 循环被预算挡住时 `confirm` 一格仍原封不动; 而真有一条结论要进
     长期记忆时, 补测花的就是那一格 (`require_double_confirm`);
  3. **墙钟到点 ⇒ 一开机时都不再花** (fail-closed, 不是「再跑完这一轮」);
  4. **token 超了 ⇒ 退化成纯数值搜索** (不问 LLM 要结构改动), 而且反向: 没超就照常问。

运行: python3 -m unittest tests.test_budget_gate -v
"""

import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operator_agent.budget import BudgetPlan
from operator_agent.conductor import Conductor
from operator_agent.kernel_db import KernelDatabase
from operator_agent.schemas import (
    TaskSchema, SourceSpec, TargetSpec, ConstraintSpec,
    TaskType, TaskComplexity, Precision,
)
from tests.test_measurement_validity import (
    LaunchDominatedGPU, NET_MS_MATMUL_4096,
)
from tests.test_search_belief_gate import GOOD_KERNEL, _inputs


class _Tracker:
    """只有 `summary_since` 一个方法的账本桩 (口径与真 `CostTracker` 一致: 本次运行)"""

    def __init__(self, tokens):
        self.tokens = tokens
        self.total_calls = 1

    def summary_since(self, n):
        return {"total_tokens": self.tokens}

    def watermark(self):
        return 0

    def pretty(self):
        return "桩账单"


class _TaggingLlm:
    """记下每次 chat_json 的 caller_tag —— 「有没有问 LLM 要结构改动」看这个"""

    def __init__(self, tokens=None):
        self.tags = []
        if tokens is not None:
            self.cost_tracker = _Tracker(tokens)

    def chat(self, messages, **kw):
        return GOOD_KERNEL

    def chat_json(self, messages, **kw):
        self.tags.append(kw.get("caller_tag", "?"))
        return {"algorithm": "matmul",
                "tiling": {"block_m": 64, "block_n": 64, "block_k": 32},
                "memory_plan": {}, "pipeline": [], "optimization_hints": []}


def _task(min_speedup=5.0):
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(framework="pytorch", kernel_type="matmul", fused_ops=[],
                          complexity=TaskComplexity.MEDIUM),
        target=TargetSpec(hardware="mock_gpu", dsl="python", compute_units=78,
                          warp_size=32),
        constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=min_speedup,
                                   max_error=1e-5),
    )


class TestTheOracleBudgetStopsTheLoop(unittest.TestCase):

    maxDiff = 600

    def _run(self, budget, max_rounds, min_speedup=5.0, llm=None,
             require_double_confirm=False):
        # 心跳周期调大: 这里量的是预算, 不让一次标定把标定那一格搅进来
        c = Conductor(backend=LaunchDominatedGPU(NET_MS_MATMUL_4096),
                      kernel_db=KernelDatabase(jsonl_path=None),
                      llm=llm or _TaggingLlm(), max_optimize_rounds=max_rounds,
                      heartbeat_interval=1000, budget=budget,
                      require_double_confirm=require_double_confirm)
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            art = c.execute(_task(min_speedup), _inputs(), None, target_gflops=0.0)
        return art, buf.getvalue()

    def test_measurements_stop_at_the_total_minus_the_reserve(self):
        """total=3 ⇒ 标定 1 / 探索 1 / 精修 0 / **确认 1** ⇒ 非 confirm 只能花 2 次

        为什么故意把预算压到 3 次: 停滞策略 (`max_stagnant_rounds=2`) 会在第 3 轮收尾,
        而预算门在第 3 轮**开头** —— 两条出口挨得这么近才验得出「先撞上的是预算那条」。
        更小的预算就不必跟停滞比快了, 但那样也验不到「两道门谁先响」。
        """
        art, out = self._run(BudgetPlan(oracle_total=3), max_rounds=6)
        st = art.metadata["budget"]
        self.assertEqual("budget_exhausted", art.metadata["stop"]["reason"],
                         f"停因不对: {art.metadata.get('stop')}")
        self.assertEqual(2, st["spent_all"], f"花超了: {st}")
        self.assertEqual(2, st["spent"]["explore"])
        self.assertEqual(0, st["spent"]["confirm"], "预留被挪用了")
        self.assertEqual(1, st["refusals"]["explore"], "被挡下来的那次要留痕")
        self.assertIn("真值测量预算用尽", out)
        # 停因自成一条词汇: 不许与「轮次练够了」混成一个数
        self.assertNotEqual("rounds_exhausted", art.metadata["stop"]["reason"])

    def test_the_reserve_is_spent_by_the_final_confirmation(self):
        """反向: 真有一条结论要进长期记忆时, 补测花的正是预留那一格"""
        art, _ = self._run(BudgetPlan(oracle_total=5), max_rounds=6, min_speedup=0.001,
                           require_double_confirm=True)
        st = art.metadata["budget"]
        self.assertTrue(art.all_passed, "前提: 这一跑要让结论有资格进长期记忆")
        self.assertEqual(1, st["spent"]["confirm"],
                         f"补测没花预留那一格: {st}")
        # 第 1 轮就过了 L3 ⇒ 探索只花 1 次; 补测那一次花的是**另一格** ——
        # 两笔账分得开, 就是「预留不与其他格混同」的可观测后果。
        self.assertEqual(1, st["spent"]["explore"], f"它不该从探索那一格出钱: {st}")
        self.assertEqual("passed", art.metadata["stop"]["reason"])

    def test_wall_clock_stops_before_spending_anything(self):
        """墙钟到点 ⇒ 一开机时都不再花 (fail-closed, 不是「再跑完这一轮」)"""
        art, out = self._run(BudgetPlan(wall_clock_s=1e-9), max_rounds=3)
        st = art.metadata["budget"]
        self.assertEqual("wall_clock_exhausted", art.metadata["stop"]["reason"])
        self.assertEqual(0, st["spent_all"], f"到点了还花了机时: {st}")
        self.assertIn("墙钟预算用尽", out)
        self.assertNotIn("轮次 2/3", out, "到点了还在开新的一轮")

    def test_no_budget_means_no_stop_and_a_plan_that_says_so(self):
        """默认全关: 不拦也不假装拦过 —— 账本仍落盘, `plan` 里明写三个 None"""
        art, _ = self._run(BudgetPlan(), max_rounds=2)
        st = art.metadata["budget"]
        self.assertEqual({"oracle_total": None, "wall_clock_s": None, "llm_tokens": None},
                         {k: st["plan"][k] for k in ("oracle_total", "wall_clock_s",
                                                     "llm_tokens")})
        self.assertNotIn(art.metadata["stop"]["reason"],
                         ("budget_exhausted", "wall_clock_exhausted"))
        self.assertGreaterEqual(st["spent_all"], 1)


class TestTheRulerIsNotStarvedByTheBudget(unittest.TestCase):
    """§10.2 的 5% 是「预检标定」那一格 —— 但它**不许变成心跳的闸门**

    让预算饿死标定, 等于把「尺子有没有坏」变成一件可以因为缺钱而跳过的事
    (§10.5-5: agent 不能编辑自己的尺子, 同一族的病)。所以这一格只记账、超支报警。
    """

    def test_heartbeats_keep_running_after_the_calibration_share_is_used_up(self):
        c = Conductor(backend=LaunchDominatedGPU(NET_MS_MATMUL_4096),
                      kernel_db=KernelDatabase(jsonl_path=None), llm=_TaggingLlm(),
                      max_optimize_rounds=3, heartbeat_interval=1,
                      budget=BudgetPlan(oracle_total=10))       # 标定份额 = 1
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            art = c.execute(_task(), _inputs(), None, target_gflops=0.0)
        out = buf.getvalue()
        st = art.metadata["budget"]
        self.assertGreaterEqual(st["spent"]["calibration"], 2,
                                f"标定被预算挡住了: {st}")
        self.assertGreaterEqual(st["calibration_over_plan"], 1, "超支必须留痕 (只报警不拦)")
        self.assertIn("[Heartbeat]", out)


class TestTheTokenBudgetDegradesToNumericSearch(unittest.TestCase):

    def _run(self, budget, tokens, max_rounds=2):
        llm = _TaggingLlm(tokens=tokens)
        c = Conductor(backend=LaunchDominatedGPU(NET_MS_MATMUL_4096),
                      kernel_db=KernelDatabase(jsonl_path=None), llm=llm,
                      max_optimize_rounds=max_rounds, heartbeat_interval=1000,
                      budget=budget)
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            art = c.execute(_task(), _inputs(), None, target_gflops=0.0)
        return llm, art, buf.getvalue()

    def test_over_budget_never_asks_the_llm_for_a_structural_change(self):
        llm, art, out = self._run(BudgetPlan(llm_tokens=100), tokens=150)
        self.assertNotIn("Search", llm.tags,
                         f"token 超了还在问 LLM 要结构改动: {llm.tags}")
        self.assertIn("退化成纯数值搜索", out)
        st = art.metadata["budget"]
        self.assertEqual(150, st["llm_tokens_used"])
        self.assertGreaterEqual(st["llm_degraded_rounds"], 1)

    def test_under_budget_still_asks_the_llm(self):
        """反向: 门不是「永远不问 LLM」"""
        llm, art, out = self._run(BudgetPlan(llm_tokens=10 ** 6), tokens=150)
        self.assertIn("Search", llm.tags, f"没超预算却不问 LLM 了: {llm.tags}")
        self.assertNotIn("退化成纯数值搜索", out)
        self.assertEqual(0, art.metadata["budget"]["llm_degraded_rounds"])

    def test_without_a_token_budget_nothing_changes(self):
        llm, art, _ = self._run(BudgetPlan(), tokens=10 ** 9)
        self.assertIn("Search", llm.tags)
        self.assertEqual(0, art.metadata["budget"]["llm_degraded_rounds"])


if __name__ == "__main__":
    unittest.main(verbosity=2)