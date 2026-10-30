"""§21.3 的生产者侧守卫: 那八项**真的进了 LLM 的 prompt**, 且只有一个拼装点

`tests/unit/test_agent_runtime.py::TestProposalContext` 钉的是「渲染器只认得这八项」;
这里问的是接线之后的三件事 —— 少了这三条, §21.3 就是「规范已写、代码零命中」里那类
「能力有、没有生产调用者」的东西 (状态文档 §21 行 ④ 记过这一笔):

1. **真的到了 Fixer 的 messages 里** —— 走真 `Conductor.execute`, 把送进 LLM 的
   prompt 逐字抓下来 (替身 LLM 是唯一证人: 断言「Conductor 拼好了」不等于「模型看到了」);
2. **只有那八段, 且模板变量不许漏出去** —— 未替换的 `{{structured_evidence}}` 出现在
   prompt 里是最容易被忽略的失效形态 (模板加了占位符、调用方忘了传);
3. **开关是干净的两臂** —— 关掉时 prompt 里既没有那一段也没有占位符, 产物侧
   (`fix_evidence`) 能区分两臂。对照实验的前提就是这个可区分性。

外加两条单源守卫: 契约摘要派生自 `freeze_from_task` (同一份契约), profiler 那一格
只喂 `verifier.COMPILE_FACT_KEYS` (与 L3 写进 metrics 的是同一份名单)。

运行: python3 -m unittest tests.test_structured_evidence -v
"""

import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from operator_agent import agent_runtime as ar
from operator_agent.contract import freeze_from_task
from operator_agent.conductor import Conductor
from operator_agent.failure_route import classify
from operator_agent.kernel_db import KernelDatabase
from operator_agent.mock_gpu import MockGPU
from operator_agent.schemas import (
    ConstraintSpec, Precision, SourceSpec, TargetSpec, TaskComplexity, TaskSchema,
    TaskType, VerificationLevel, VerificationResult,
)
from operator_agent.verifier import COMPILE_FACT_KEYS

#: L1 过、L2 挂 (结果错) ⇒ 走「数值修复」那一格, Fixer 被调用
WRONG_NUMBERS = "import numpy as np\n\ndef kernel(A, B):\n    return A @ B + 1.0\n"
GOOD_KERNEL = "import numpy as np\n\ndef kernel(A, B):\n    return A @ B\n"
#: 修复产物: **仍然错, 但与前一份不同** —— 于是「第二次修复」那一轮手里有真正的
#: 「上一份 vs 这一份」可对比 (一字不改的产物与「没有父代」在 diff 上长得一样)
WRONG_V2 = "import numpy as np\n\ndef kernel(A, B):\n    return A @ B + 2.0\n"
SKETCH = {"algorithm": "matmul",
          "tiling": {"block_m": 64, "block_n": 64, "block_k": 32},
          "memory_plan": {}, "pipeline": [], "optimization_hints": []}

#: §21.3 的八段 (设计原文的八项, 顺序即 `ProposalContext.render()` 的顺序)
EIGHT = ("当前契约摘要", "父候选 diff", "最近一次失败分类", "首个错误与误差形态",
         "相关 profiler 指标", "可用工具列表", "剩余预算", "允许修改的文件")


class _CapturingLlm:
    """把送进 LLM 的每一段 prompt 逐字记下来 —— 断言「模型看到了什么」只能靠它"""

    def __init__(self, generate=WRONG_NUMBERS, fix=GOOD_KERNEL):
        self.generate, self.fix = generate, fix
        self.sent = []          # [(caller_tag, prompt)]

    def chat(self, messages, **kw):
        tag = kw.get("caller_tag", "?")
        self.sent.append((tag, "\n".join(str(m.get("content", "")) for m in messages)))
        return self.fix if tag.startswith("Fixer") else self.generate

    def chat_json(self, messages, **kw):
        return dict(SKETCH)

    def prompts(self, tag):
        return [p for t, p in self.sent if t == tag]


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


def _inputs():
    rng = np.random.default_rng(0)
    return {"A": rng.standard_normal((64, 64)).astype(np.float32),
            "B": rng.standard_normal((64, 64)).astype(np.float32)}


def _run(**kw):
    llm = kw.pop("llm", None) or _CapturingLlm()
    c = Conductor(backend=MockGPU(), kernel_db=KernelDatabase(jsonl_path=None),
                  llm=llm, max_optimize_rounds=kw.pop("rounds", 2),
                  heartbeat_interval=1000, **kw)
    with contextlib.redirect_stdout(io.StringIO()):
        art = c.execute(_task(), _inputs())
    return c, art, llm


class TestTheEvidenceReachesThePrompt(unittest.TestCase):

    maxDiff = 4000

    def test_the_eight_items_are_in_the_fixers_prompt(self):
        """八段逐段名字都在 Fixer 真的收到的那段 prompt 里 (不是「Conductor 拼好了」)"""
        _, _, llm = _run()
        prompts = llm.prompts("Fixer")
        self.assertTrue(prompts, "Fixer 一次都没被调用 —— 前提变了, 这条守卫测不到东西")
        prompt = prompts[0]
        self.assertIn("## 结构化证据", prompt)
        for item in EIGHT:
            self.assertIn(f"### {item}", prompt, f"§21.3 的「{item}」没进 prompt")

    def test_the_round_facts_are_in_it_not_boilerplate(self):
        """那段文字必须带**这一轮**的事实: 否则它就是一段每次都一样的装饰"""
        _, art, llm = _run()
        prompt = llm.prompts("Fixer")[0]
        self.assertIn("最近一次失败分类", prompt)
        self.assertIn("numeric", prompt, "失败分类没进 prompt (§8 路由是修法的一半)")
        # 首个错误那一段取的是**失败结果**的原文 (取错一条就是空段, 而标题还在)
        self.assertIn("形态=whole_tensor", prompt, "首个错误的原文没进 prompt")
        # 契约摘要派生自 `freeze_from_task` ⇒ 短哈希与那一份契约一致
        digest = ar.contract_digest(freeze_from_task(_task()))
        self.assertIn(digest.splitlines()[-1], prompt,
                      "契约摘要不是从 `freeze_from_task` 那一份派生的")
        # 可写路径: §21.2 的边界, 模型得知道它能改哪儿
        self.assertIn("candidate", prompt)
        # 留痕: 产物侧知道这一轮注入了, 且知道喂了哪些 profiler 键。
        # ⚠️ 记在**第 1 轮**: Fixer 是在「发现失败的那一轮」结束时被调用的
        # (产物进下一轮), 而第 2 轮那一格记的是「这一轮的候选是修出来的」。
        rounds = {h["round"]: h for h in art.iteration_history}
        rec = rounds[1]["fix_evidence"]
        self.assertTrue(rec["injected"])
        self.assertGreater(rec["chars"], 0)
        self.assertTrue(set(rec["profiler_facts"]) <= set(COMPILE_FACT_KEYS))

    def test_no_placeholder_ever_leaks_into_a_prompt(self):
        """模板里加了 `{{structured_evidence}}`, 调用方忘了传 —— 这是最容易被忽略的形态"""
        for switch in (True, False):
            _, _, llm = _run(structured_evidence=switch)
            for tag, prompt in llm.sent:
                self.assertNotIn("{{structured_evidence}}", prompt,
                                 f"未替换的模板变量漏进了 {tag} 的 prompt (开关={switch})")

    def test_the_switch_off_is_a_real_off(self):
        """关掉的那一臂: prompt 里没有那一段; 产物侧 `fix_evidence` 是 None"""
        _, art, llm = _run(structured_evidence=False)
        prompt = llm.prompts("Fixer")[0]
        self.assertNotIn("结构化证据", prompt)
        self.assertNotIn("契约短哈希", prompt)
        rounds = {h["round"]: h for h in art.iteration_history}
        self.assertFalse(rounds[1]["fix_evidence"]["injected"],
                         "关掉了却在产物里记着注入过 —— 两臂从产物上分不开")

    def test_the_two_arms_differ_on_the_prompt_itself(self):
        """对照实验的前提: 两臂的 prompt **确实不同**, 且差别就在那一段"""
        _, _, on = _run(structured_evidence=True)
        _, _, off = _run(structured_evidence=False)
        p_on, p_off = on.prompts("Fixer")[0], off.prompts("Fixer")[0]
        self.assertNotEqual(p_on, p_off)
        self.assertGreater(len(p_on), len(p_off))
        self.assertIn("结构化证据", p_on)
        self.assertNotIn("结构化证据", p_off)

    def test_a_round_without_a_fix_records_no_evidence(self):
        """没有失败就没有修复提示 ⇒ 那一格必须是「没注入」(不是「注入了空串」)"""
        _, art, llm = _run(rounds=1, llm=_CapturingLlm(generate=GOOD_KERNEL,
                                                       fix=GOOD_KERNEL))
        self.assertFalse(llm.prompts("Fixer"), "这一轮不该有 Fixer")
        self.assertFalse(art.iteration_history[0]["fix_evidence"]["injected"])
        self.assertEqual({}, {k: v for k, v in art.iteration_history[0]["fix_evidence"].items()
                              if k != "injected"})


class TestTheDigestIsDerived(unittest.TestCase):
    """契约摘要与 profiler 名单都**派生**自既有真源, 不另写一份口径"""

    def test_the_digest_says_it_cannot_when_there_is_no_contract(self):
        self.assertEqual("", ar.contract_digest(None))

    def test_the_digest_carries_the_contract_facts(self):
        c = freeze_from_task(_task(min_speedup=2.5))
        text = ar.contract_digest(c)
        self.assertIn("matmul", text)
        self.assertIn("2.5", text)
        self.assertIn("1e-05", text)
        self.assertIn((c.hash or c.compute_hash())[:12], text)
        # 同一个对象算两遍必须一样 (派生而不是现编)
        self.assertEqual(text, ar.contract_digest(c))

    def test_the_profiler_whitelist_has_one_source(self):
        """名单只有 `verifier.COMPILE_FACT_KEYS` 一份: 两处各写一遍就会各自漂"""
        src = (Path(__file__).resolve().parents[2] / "operator_agent"
               / "verifier.py").read_text(encoding="utf-8")
        self.assertIn("for k in COMPILE_FACT_KEYS if k in bench", src,
                      "L3 的 metrics 组装不再读那个常量 —— profiler 名单分成两份了")
        rt_src = (Path(__file__).resolve().parents[2] / "operator_agent"
                  / "agent_runtime.py").read_text(encoding="utf-8")
        self.assertNotIn("n_spills\"", rt_src,
                         "agent_runtime 里又抄了一份 profiler 键名 (应当只读常量)")
        cond = (Path(__file__).resolve().parents[2] / "operator_agent"
                / "conductor.py").read_text(encoding="utf-8")
        self.assertIn("for k in COMPILE_FACT_KEYS", cond,
                      "采集那一侧没读常量")


class TestTheDiffBaselineIsThePreviousCandidate(unittest.TestCase):
    """「父候选 diff」的基线是**本轮的上一份**, 不是「正在被修的那一份」

    钉这一条的理由: 后者与 `current_code` 常常是同一份 ⇒ diff 恒空, 而屏幕上仍然
    印着「父候选 diff」这一节 —— 读者会以为「这一轮没改动」, 而真相是「基线取错了」。
    这正是本仓反复出现的失效形态 (字段在、值恒空、没人发现)。
    """

    def test_the_diff_shows_what_this_round_changed(self):
        # 第 2 轮末尾那次修复才有父代 (第 1 轮是首次提议 ⇒ 没有可比对象)
        _, _, llm = _run(rounds=3, llm=_CapturingLlm(generate=WRONG_NUMBERS,
                                                     fix=WRONG_V2))
        prompts = llm.prompts("Fixer")
        self.assertGreaterEqual(len(prompts), 2, "第二次修复没有被调用 —— 前提变了")
        prompt = prompts[1]
        self.assertIn("### 父候选 diff", prompt)
        self.assertIn("--- a/candidate/code.py", prompt,
                      "修复提示里没有 diff —— 基线取成了「正在被修的那一份」")
        self.assertIn("+    return A @ B + 2.0", prompt, "diff 里看不出这一轮的改动")

    def test_the_record_says_how_big_the_diff_was(self):
        _, art, _ = _run(rounds=3, llm=_CapturingLlm(generate=WRONG_NUMBERS,
                                                     fix=WRONG_V2))
        rec = {h["round"]: h for h in art.iteration_history}[2]["fix_evidence"]
        self.assertTrue(rec["has_parent_diff"])
        self.assertTrue(any(rec["parent_diff_lines"]), f"行数差是 0: {rec}")


class TestTheContextShowsTheShapeWhenThereIsOne(unittest.TestCase):
    """误差形态 (§8.2) 是「修 vs 重写」的直接依据 —— 有形态时必须出现在上下文里"""

    def test_the_shape_travels_with_the_route(self):
        # 形态读的是**产物里记的那一份** (`metrics["error_shape"]`), 不是现算的
        route = classify([VerificationResult(
            level=VerificationLevel.L2_FUNCTION, passed=False,
            message="功能验证失败: 最大误差=1.0",
            metrics={"error_shape": "whole_tensor"})])
        rt = ar.AgentRuntime()
        ctx = rt.context_for(route=route)
        self.assertEqual("whole_tensor", ctx.error_shape)
        self.assertIn("whole_tensor", ctx.render())


class TestTheProfilerFactsComeFromTheWhitelist(unittest.TestCase):
    """profiler 那一格只喂 `COMPILE_FACT_KEYS` 里**这一轮真的报了**的键

    直接调生产者 (`Conductor._fix_evidence`) 的理由: 让 MockGPU 的一轮里报出
    `n_regs` 得造一个假后端, 而这里要钉的是**筛选**这件事本身 (白名单之外的一律不喂)。
    """

    def _evidence(self, metrics):
        c = Conductor(backend=MockGPU(), llm=None, max_optimize_rounds=1,
                      heartbeat_interval=1000)
        res = VerificationResult(level=VerificationLevel.L2_FUNCTION, passed=False,
                                 message="功能验证失败", error_log="第一处失败: 细节",
                                 metrics=metrics)
        return c._fix_evidence(task=_task(), route=None, verify_results=[res],
                               current_code=GOOD_KERNEL, parent_code="", gate_ctx=None)

    def test_only_compile_facts_are_fed(self):
        block, rec = self._evidence({"n_regs": 42, "num_stages": 3,
                                     "max_error": 1.0, "other_metric": 7})
        self.assertEqual(["n_regs", "num_stages"], rec["profiler_facts"])
        self.assertIn("- n_regs: 42", block)
        self.assertIn("- num_stages: 3", block)
        self.assertNotIn("other_metric", block, "白名单之外的指标被喂进去了")
        self.assertNotIn("max_error: 1.0", block)

    def test_a_round_without_compile_facts_says_so(self):
        block, rec = self._evidence({})
        self.assertEqual([], rec["profiler_facts"])
        self.assertIn("(没有相关指标", block,
                      "没有读数时那一节应明说「没有」, 不许留空让人以为是 0")

    def test_the_first_error_is_taken_from_the_failed_result(self):
        _, rec = self._evidence({"n_regs": 1})
        self.assertTrue(rec["injected"])
        block, _ = self._evidence({"n_regs": 1})
        self.assertIn("第一处失败: 细节", block)


if __name__ == "__main__":
    unittest.main(verbosity=2)