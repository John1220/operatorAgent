"""血缘里到底有没有「这一轮卡在哪一级」

`iteration_history` 原来只记 `all_passed`。而 `all_verification_results` 每轮被覆盖
(`conductor.py` 里 `all_verification_results = verify_results`), 所以出了循环只剩**最后
一轮**的分级结果 —— 「第 2 轮卡在 L1」与「第 2 轮卡在 L3」在产物里无法区分。

这一组测试钉的是补上的两个键: `levels_passed` 与 `measurement_validity`。它们不是装饰:
成功率量具 (`benchmarks/harness_pass_rate.py`) 的 rounds-to-pass 与「有多少轮的读数有资格
驱动闭环」两项, 只能从血缘里算, 因为真机跑一轮很贵、不可能为了统计再跑一遍。

用 MockGPU + 假 LLM, 不碰网络也不碰 GPU。
"""
import contextlib
import io
import unittest

import numpy as np

from operator_agent.conductor import Conductor
from operator_agent.kernel_db import KernelDatabase
from operator_agent.llm_client import CostTracker
from operator_agent.measurement import Validity
from operator_agent.mock_gpu import MockGPU
from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec, TargetSpec,
                                    TaskComplexity, TaskSchema, TaskType)

# 正确的 matmul (对角加权): MockGPU 的参考实现能对上
CORRECT = '''import numpy as np

def kernel(A, B):
    n = A.shape[0]
    out = np.zeros((n, n), dtype=A.dtype)
    for i in range(n):
        out[i, i] = np.sum(A[i] * B[i])
    return out + B
'''

# 能编译、能跑, 但结果是错的 —— 于是这一轮必然卡在 L2 而不是 L1
WRONG_RESULT = CORRECT.replace("out + B", "out - B")


class _FakeLLM:
    """Designer 走 chat_json, Coder 走 chat; 所有轮次都返回同一份代码"""

    def __init__(self, code):
        self.code = code
        self._sketch = {"algorithm": "matmul",
                        "tiling": {"block_m": 64, "block_n": 64, "block_k": 32},
                        "memory_plan": {}, "pipeline": [], "optimization_hints": []}
        # 真账本: 每次 chat/chat_json 记一笔, 于是「一个 client 跑多次运行」的累计问题
        # 能在本地被复现出来 (TestCostScopeIsPerRun)。
        self.cost_tracker = CostTracker()

    def chat(self, messages, temperature=None, max_tokens=None, raw=False,
             caller_tag="Unknown"):
        self.cost_tracker.record(model="deepseek-chat", prompt_tokens=1000,
                                 completion_tokens=500, usage_source="api",
                                 caller_tag=caller_tag)
        return f"```python\n{self.code}```"

    def chat_json(self, messages, temperature=None, max_tokens=None, caller_tag="Unknown"):
        self.cost_tracker.record(model="deepseek-chat", prompt_tokens=700,
                                 completion_tokens=200, usage_source="estimated",
                                 caller_tag=caller_tag)
        return dict(self._sketch)

    class _Tracker:
        total_calls = 0

        def summary(self):
            return {}

        def watermark(self):
            return 0

        def summary_since(self, n):
            return self.summary()

        def pretty(self):
            return "LLM: fake"

    cost_tracker = _Tracker()


def _task() -> TaskSchema:
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.LOW),
        target=TargetSpec(hardware="mock_gpu", dsl="python"),
        constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=0.5,
                                   max_error=1e-3))


class TestLineageCarriesLevelOutcomes(unittest.TestCase):
    def _run(self, code, rounds=2):
        llm = _FakeLLM(code)
        c = Conductor(backend=MockGPU(), kernel_db=KernelDatabase(), llm=llm,
                      max_optimize_rounds=rounds)
        inputs = {"A": np.ones((8, 8), dtype=np.float32),
                  "B": np.ones((8, 8), dtype=np.float32)}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            art = c.execute(_task(), inputs, target_gflops=0.0)
        return art.iteration_history

    def test_passing_round_records_all_levels(self):
        hist = self._run(CORRECT)
        self.assertTrue(hist, "一轮都没记, 这组测试会空转")
        for h in hist:
            self.assertIn("levels_passed", h,
                          "分级结果没进血缘 —— 出了循环就只剩最后一轮的分级")
            self.assertTrue(h["levels_passed"].get("L1"),
                            f"第 {h['round']} 轮的 L1 应为通过: {h['levels_passed']}")

    def test_failing_round_records_which_level_it_got_stuck_at(self):
        """方向检查: 结果错的核必须被记成「卡在 L2」, 而不是笼统的「没过」。

        这一条是补这个键的全部理由。只断 `all_passed is False` 的话, 删掉
        `levels_passed` 这一行不会有任何用例变红 —— 那正是本项目反复栽的
        「算了没人读」的反面: 记了但没人能读出区别。
        """
        hist = self._run(WRONG_RESULT)
        self.assertTrue(hist, "一轮都没记, 这组测试会空转")
        stuck = [h for h in hist if not h["all_passed"]]
        self.assertTrue(stuck, "结果错的核居然全过了 —— 假 LLM 或 MockGPU 的参考实现变了, "
                               "这组测试的前提已经不成立")
        for h in stuck:
            lv = h["levels_passed"]
            self.assertIs(True, lv.get("L1"), f"L1 本该通过: {lv}")
            self.assertIs(False, lv.get("L2"), f"该卡在 L2 而实际是: {lv}")
            self.assertNotIn("L3", lv,
                             f"L2 没过就不该有 L3 的读数: {lv}")

    def test_validity_key_is_present_and_never_a_mystery_string(self):
        """`measurement_validity` 缺失与为 None 是两件事, 不许混。

        量具要统计「有多少轮的读数有资格驱动闭环」。键**不存在**会被读成
        「这一轮没有 L3」, 而 None 的含义是「有 L3 但没报档位」—— 两者混在一起,
        一份全 None 的产物会被读成「全程盲跑」, 那是一条很响的错误结论。
        """
        hist = self._run(CORRECT)
        allowed = {v.value for v in Validity} | {None}
        for h in hist:
            self.assertIn("measurement_validity", h,
                          "档位没进血缘 —— 无法离线统计有多少轮在盲跑")
            self.assertIn(h["measurement_validity"], allowed,
                          f"出现了三档之外的值: {h['measurement_validity']!r}")

    def test_per_round_speedup_is_recorded(self):
        """逐轮加速比必须在血缘里: `benchmark_data` 只留最后一轮。

        缺了它, 「这几轮到底有没有在变快」就只能看最后一个数 —— 而优化循环的意义
        恰恰在于**轨迹**。这一条与 `levels_passed` 是同一类搬运缺陷的两个面。
        """
        hist = self._run(CORRECT)
        for h in hist:
            self.assertIn("speedup", h, "逐轮加速比没进血缘, 只剩最后一轮")
            sp = h["speedup"]
            self.assertTrue(sp is None or isinstance(sp, (int, float)),
                            f"加速比既不是数也不是 None: {sp!r}")


class TestCostScopeIsPerRun(unittest.TestCase):
    """一个 client 跑两次运行时, 落盘的成本必须是**本次**那份。

    `CostTracker` 挂在 client 上累加、`reset()` 全仓没人调, 所以后一次运行的 `summary()`
    含着前一次的量。这不是假想: `harness_pass_rate.run_one` 的 docstring 记过一次真事故
    —— 六次运行的 token 被当成六份独立样本再求和, 总量高估 **3.25 倍**
    (15638→34457→78801→101156→143717→166345 那条单调曲线其实是累计值)。
    这条用例用**同一个 llm 对象**跑两次, 把那个失效形态钉住。
    """

    def _run(self, llm):
        c = Conductor(backend=MockGPU(), kernel_db=KernelDatabase(), llm=llm,
                      max_optimize_rounds=1)
        inputs = {"A": np.ones((8, 8), dtype=np.float32),
                  "B": np.ones((8, 8), dtype=np.float32)}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return c.execute(_task(), inputs, target_gflops=0.0)

    def test_the_second_run_does_not_bill_the_first_one(self):
        llm = _FakeLLM(CORRECT)
        first = self._run(llm).metadata["llm_cost"]
        second = self._run(llm).metadata["llm_cost"]
        self.assertEqual("this_run", first["scope"])
        self.assertGreater(first["total_calls"], 0, "一次调用都没记的话, 这组断言是空转")
        self.assertEqual(first["total_calls"], second["total_calls"],
                         f"第二次运行账单涨了 ⇒ 落的是累计值: {first} → {second}")
        self.assertEqual(first["total_cost_usd"], second["total_cost_usd"])
        # 绝对锚点: 两次之和等于账本总量 (口径不同源就会在这里露出来)
        self.assertEqual(llm.cost_tracker.total_calls,
                         first["total_calls"] + second["total_calls"])
        self.assertEqual(second["process_total_calls"], llm.cost_tracker.total_calls,
                         "进程累计位必须与账本一致 —— 它是「这是本次值」的旁证")

    def test_the_breakdown_is_also_scoped_to_this_run(self):
        llm = _FakeLLM(CORRECT)
        first = self._run(llm).metadata["llm_cost"]
        second = self._run(llm).metadata["llm_cost"]
        self.assertEqual({k: v["calls"] for k, v in first["breakdown_by_caller"].items()},
                         {k: v["calls"] for k, v in second["breakdown_by_caller"].items()},
                         "按 caller 的分桶也带着上一次的量")
        self.assertEqual(sum(v["calls"] for v in second["breakdown_by_caller"].values()),
                         second["total_calls"])
if __name__ == "__main__":
    unittest.main()
