"""§21 Agent Runtime 在**生产链路**上的守卫（判据算了却没人读 = 本仓最贵的缺陷）

单元判据在 `tests/test_agent_runtime.py`; 这里只问两件事:

1. **门真的在路上**: 一次 `Conductor.execute` 里, 一份违反 §21.4 的候选**没有**被
   送去编译/执行 (后端收到的代码里逐字看不出它), 而本轮以一条**定义好的 L1 失败**
   (`compile_status == "agent_action_refused"`) 结束 —— 于是 §8 的路由与 Fixer 照旧
   工作, 拒绝不是静默丢弃;
2. **裁决真进了产物**: 每一轮的 `iteration_history[i]["agent_action"]` 里有动作、
   有裁决、**有当时看到的上下文** —— 「拒了」与「上下文装配错了」必须分得开。

外加三件只有走真链路才看得见的装配事实: 动作种类跟着循环在做什么变、声明的 token
预算与真正花掉的那个上限是同一个数、被拒的候选不再触发远端依赖分析 (SSH 往返)。

运行: python3 -m unittest tests.test_agent_runtime_wiring -v
"""

import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from operator_agent.agent_runtime import candidate_path
from operator_agent.coder_pipeline import CoderPipeline
from operator_agent.conductor import Conductor
from operator_agent.failure_route import classify
from operator_agent.fixer import FixerAgent
from operator_agent.kernel_db import KernelDatabase
from operator_agent.mock_gpu import MockGPU
from operator_agent.schemas import (
    TaskSchema, SourceSpec, TargetSpec, ConstraintSpec, VerificationLevel,
    VerificationResult, TaskType, TaskComplexity, Precision,
)
from tests.backend_fakes import DeclaredFakeBackend

#: L1 过、L2 挂 (结果是错的) —— 用它把循环推进「数值修复」那一格
BAD_RESULT = "import numpy as np\n\ndef kernel(A, B):\n    return A @ B + 1.0\n"
#: Fixer 的产物: 比父代**多一个外部依赖** ⇒ §21.4 第 4 条 (零 GPU 秒判死)
FIXER_ADDS_DEP = ("import numpy as np\nimport torch\n\n"
                  "def kernel(A, B):\n    return torch.matmul(A, B)\n")
GOOD_KERNEL = "import numpy as np\n\ndef kernel(A, B):\n    return A @ B\n"
#: 语法错 (L1 真的会失败)。只在单路生成下用 —— 流水线的编译剪枝会把它换掉
BROKEN_COMPILE = "import numpy as np\n\ndef kernel(A, B):\n    return A @\n"

SKETCH = {"algorithm": "matmul",
          "tiling": {"block_m": 64, "block_n": 64, "block_k": 32},
          "memory_plan": {}, "pipeline": [], "optimization_hints": []}


class _CountingGPU(MockGPU):
    """记下**每一份**被送进编译/执行的源码 —— 判据是内容级的, 不是次数级的

    次数会被生成侧的编译剪枝搅进来 (`CoderPipeline` 自己会编译几个候选), 而
    「这份源码有没有到过后端」不会被搅。
    """

    def __init__(self):
        super().__init__()
        self.seen = []

    def compile(self, kernel_code: str) -> bool:
        self.seen.append(kernel_code)
        return super().compile(kernel_code)

    def execute(self, kernel_code: str, inputs):
        self.seen.append(kernel_code)
        return super().execute(kernel_code, inputs)

    def benchmark(self, kernel_code: str, inputs, warmup=3, repeat=10):
        self.seen.append(kernel_code)
        return super().benchmark(kernel_code, inputs, warmup, repeat)


class _ScriptedLlm:
    """按 `caller_tag` 分派: 生成侧给「结果错」的码, 修复侧给「夹带依赖」的码

    分派必须按 tag 而不是按调用次序: 生成侧的采样次数 (`CoderPipeline` 那几个
    候选) 与本用例要测的东西无关, 数调用次序会让用例随采样配置一起悄悄变形。
    """

    def __init__(self, generate=BAD_RESULT, fix=FIXER_ADDS_DEP):
        self.generate, self.fix = generate, fix
        self.tags = []

    def chat(self, messages, **kw):
        tag = kw.get("caller_tag", "?")
        self.tags.append(tag)
        return self.fix if tag.startswith("Fixer") else self.generate

    def chat_json(self, messages, **kw):
        self.tags.append(kw.get("caller_tag", "?"))
        return dict(SKETCH)


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


class TestTheGateIsOnTheRealLoop(unittest.TestCase):

    maxDiff = 2000

    def _run(self, llm=None, backend=None, rounds=2, **kw):
        backend = backend or _CountingGPU()
        llm = llm or _ScriptedLlm()
        c = Conductor(backend=backend, kernel_db=KernelDatabase(jsonl_path=None),
                      llm=llm, max_optimize_rounds=rounds,
                      heartbeat_interval=1000, **kw)
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            art = c.execute(_task(), _inputs())
        return c, art, buf.getvalue(), backend

    @staticmethod
    def _actions(art):
        """每一轮的 §21 裁决 (**产物里读出来的**, 不是内存里猜的)"""
        return [(h["round"], h["agent_action"]) for h in art.iteration_history
                if h.get("agent_action") is not None]

    def test_a_candidate_that_breaks_a21_rule_never_reaches_the_backend(self):
        """§21.4 第 4 条: 夹带新依赖的修复候选 —— 后端一个字符都不该看到它"""
        _, art, out, backend = self._run()
        self.assertIn("动作被拒", out, "门没在这条路上")
        self.assertFalse(any(FIXER_ADDS_DEP == code for code in backend.seen),
                         "被拒的候选还是被送去编译/执行了 —— 门在路的旁边, 不在路上")
        rounds = dict(self._actions(art))
        self.assertEqual(2, len(rounds), f"断言的轮次形状变了: {sorted(rounds)}")
        refused = rounds[2]
        self.assertFalse(refused["accepted"])
        violation = [o for o in refused["precompile"]["outcomes"]
                     if o["state"] == "violation"]
        self.assertEqual(["new_dependency"], [o["rule"] for o in violation])
        self.assertIn("torch", violation[0]["detail"])
        # 这一轮以一条**定义好的 L1 失败**结束 (不是静默丢弃, 也不是停止)
        l1 = [r for r in art.verification_results if r.level == VerificationLevel.L1_COMPILE]
        self.assertTrue(l1, "连 L1 结果都没有 —— 拒绝没有变成一条结果")
        self.assertEqual("agent_action_refused", l1[-1].metrics.get("compile_status"))
        self.assertEqual("l1_l2_failed", (art.metadata.get("stop") or {}).get("reason"))

    def test_the_same_violating_candidate_is_refused_every_time(self):
        """I78: 同一个违规候选**重复提交**也必须每轮都被拒

        真机一手抓到的洞 —— 修复产物一直夹带 `import torch` 时:
          * 老口径: 「新增依赖」对着 **patch 的父代**判, 而第 3 轮的父代正是第 2 轮那份
            **被拒的**候选 ⇒ `torch` 不再算「新增」, 门放行, 那份源码真的上了远端编译
            (真机日志: 第 3 轮 `L1 PASSED: 编译成功`, 只有 L2 的反作弊静态审计拦住它);
          * 新口径: 基线 = 上一次**过门**的候选 ⇒ 每一轮都判得出「新增了 torch」。
        这一条用 rounds=3 才测得到: 第 2 轮的重复提交发生在第 3 轮。
        """
        _, art, _, backend = self._run(rounds=3)
        rounds = dict(self._actions(art))
        self.assertGreaterEqual(len(rounds), 3, f"轮次形状变了: {sorted(rounds)}")
        for n in (2, 3):
            self.assertFalse(rounds[n]["accepted"], f"第 {n} 轮那份违规候选被放行了")
            violation = [o for o in rounds[n]["precompile"]["outcomes"]
                         if o["state"] == "violation"]
            self.assertEqual(["new_dependency"], [o["rule"] for o in violation])
        self.assertFalse(any(FIXER_ADDS_DEP == code for code in backend.seen),
                         "重复提交的违规候选还是被送去编译/执行了 —— 门被自己走过的路绕开")

    def test_the_refusal_reaches_the_fixer_as_the_repair_prescription(self):
        """拒绝理由要**逐字**进修复提示: 否则 Fixer 会照着「编译失败」修一个不存在的问题"""
        _, _, out, _ = self._run()
        self.assertIn("Fixer", out)
        self.assertIn("新增了外部依赖", out, "拒绝理由没有出现在本轮的修复路径上")

    def test_every_round_records_the_action_the_verdict_and_the_context(self):
        _, art, _, _ = self._run(llm=_ScriptedLlm(generate=GOOD_KERNEL,
                                                  fix=GOOD_KERNEL), rounds=1)
        actions = self._actions(art)
        self.assertEqual([1], [n for n, _ in actions])
        verdict = actions[0][1]
        self.assertEqual({"action", "accepted", "context", "schema_problems",
                          "permission_problems", "budget_problems", "precompile",
                          "structural_licensed", "budget_deferred_note",
                          "fallback_available"}, set(verdict))
        self.assertTrue(verdict["accepted"], verdict)
        action = verdict["action"]
        self.assertEqual("propose_source_patch", action["kind"])
        self.assertIsNone(action["parent_candidate"], "首轮没有父代, 不许编一个")
        self.assertEqual(candidate_path(), action["patch"]["path"])
        # 上下文也要留痕: 「拒了」与「当时的输入装错了」必须分得开
        ctx = verdict["context"]
        self.assertEqual(["candidate"], ctx["authorized_roots"])
        self.assertEqual([["A", "B"]], ctx["input_key_sets"])
        self.assertTrue(ctx["spend_committed"], "候选这个位置已经花过钱了")
        self.assertIsNone(ctx["tokens_left"], "没设上限被写成了别的数")
        # 覆盖率 = 「真被查过」+「有对象但没量具」这两端。首轮没有父代 ⇒ 两条以父代为
        # 基线的判据是 `not_applicable`（**这一份候选没有这个对象**），它们**不进**覆盖表：
        # 把「没有对象」并进 uncovered 会让「这一层查不了」多报两条 (I90)。
        pre = verdict["precompile"]

        def _state(rule):
            return next(o["state"] for o in pre["outcomes"] if o["rule"] == rule)

        self.assertEqual(["abi", "protected_paths", "write_scope"],
                         pre["coverage"]["covered_rules"])
        self.assertEqual(["capability"], pre["coverage"]["uncovered_rules"])
        for rule in ("new_dependency", "duplicate_of_parent"):
            self.assertEqual("not_applicable", _state(rule), rule)
            self.assertNotIn(rule, pre["coverage"]["covered_rules"], rule)
            self.assertNotIn(rule, pre["coverage"]["uncovered_rules"], rule)
        self.assertEqual("unknown", _state("capability"),
                         "源码级「幻觉 API」是**有对象、没量具**, 不是「没有对象」")

    def test_a_clean_run_is_never_refused_and_says_so(self):
        """反向: 门不是「一律拒」—— 正常候选全过, 且计数是 0 而不是缺省"""
        c, art, out, _ = self._run(llm=_ScriptedLlm(generate=GOOD_KERNEL,
                                                    fix=GOOD_KERNEL), rounds=2)
        self.assertNotIn("动作被拒", out)
        self.assertEqual({}, c.agent_runtime.rejected)
        self.assertGreaterEqual(c.agent_runtime.admitted, 2)
        self.assertTrue(all(v["accepted"] for _, v in self._actions(art)))

    def test_the_fix_round_names_the_parent_it_repairs(self):
        """§21.1 的「可回滚」: 修复轮必须点名父代, 且父代哈希 = 上一轮那份源码"""
        _, art, _, _ = self._run()
        rounds = dict(self._actions(art))
        first, second = rounds[1]["action"], rounds[2]["action"]
        self.assertEqual("repair_correctness", second["kind"])
        self.assertEqual("src:" + first["patch"]["source_hash"], second["parent_candidate"])
        self.assertTrue(second["patch"]["added_lines"] or second["patch"]["removed_lines"],
                        "修复轮与父代的行数差是 0 —— diff 基线取错了")
        self.assertEqual("numeric", rounds[2]["context"]["route_kind"])
        self.assertTrue(rounds[2]["context"]["fix_not_rewrite"],
                        "§19.3「修 > 重写」没有传到位")

    def test_a_refused_candidate_skips_the_remote_dependency_probe(self):
        """被拒 = 没进编译 ⇒ 为它跑一次远端依赖分析 (SSH 往返) 是纯粹的白花

        这条用例**自带活性证明**: 同一轮里第 1 份候选真的编译失败, 探针确实被调用了
        一次 (带的就是那份坏码)。否则「探针没看到被拒的码」也可能只是因为整条路没走到。
        单路生成 (`use_coder_pipeline=False`) 是必要的: 走流水线时编译剪枝会把这份坏码
        换掉, L1 就永远不会真的失败。轮数必须**大于**被拒的那一轮: 依赖分析那一段在
        `round_num < max_optimize_rounds` 里面 —— 用 `rounds=2` 时那一轮根本走不到它,
        于是这条守卫会在放开跳过之后**照样绿** (变异标定的 M25 抓到过这一版)。
        """
        calls = []
        backend = DeclaredFakeBackend(
            caps=[], analyze_dependencies=lambda code: (calls.append(code), [])[1])
        _, art, _, _ = self._run(backend=backend, rounds=3,
                                 llm=_ScriptedLlm(generate=BROKEN_COMPILE,
                                                  fix=FIXER_ADDS_DEP),
                                 use_coder_pipeline=False)
        # 比较前先 strip: Coder 抽码时会去掉首尾空行, 而这条断言要问的是「哪一份源码」
        self.assertIn(BROKEN_COMPILE.strip(), [c.strip() for c in calls],
                      "探针一次都没被调用 —— 这条守卫测不到东西 (前提变了)")
        self.assertFalse(any(FIXER_ADDS_DEP.strip() == code.strip() for code in calls),
                         "被拒的候选触发了远端依赖分析")
        self.assertFalse([h for h in art.iteration_history
                          if h["round"] == 2][0]["agent_action"]["accepted"])

    def test_the_runtime_is_reset_between_runs_on_one_conductor(self):
        """一个 Conductor 跑两个任务: 记账量的是**这一次运行**, 不是「这个进程到现在」"""
        c = Conductor(backend=_CountingGPU(), kernel_db=KernelDatabase(jsonl_path=None),
                      llm=_ScriptedLlm(), max_optimize_rounds=2, heartbeat_interval=1000)
        with contextlib.redirect_stdout(io.StringIO()):
            c.execute(_task(), _inputs())
            self.assertEqual({"repair_correctness": 1}, c.agent_runtime.rejected)
            c.execute(_task(), _inputs())
        self.assertEqual({"repair_correctness": 1}, c.agent_runtime.rejected,
                         "第二次运行把上一次的记账累加进来了 (两个任务的现场混在一起)")

    def test_the_gate_does_not_fire_when_the_route_is_not_a_candidate_round(self):
        """门只在有候选可裁的时候说话 —— 没有失败驱动的首轮不是「修复」"""
        _, art, _, _ = self._run(llm=_ScriptedLlm(generate=GOOD_KERNEL), rounds=1)
        verdict = self._actions(art)[0][1]
        self.assertEqual("", verdict["context"]["route_kind"])
        self.assertEqual((), tuple(verdict["permission_problems"]))


class TestTheActionKindFollowsTheLoop(unittest.TestCase):
    """装配事实: 循环在做什么 → §21.1 的哪一种动作 (写错种类 = 权限表判错对象)"""

    def test_compile_and_numeric_repairs_are_told_apart(self):
        compile_route = classify([VerificationResult(
            level=VerificationLevel.L1_COMPILE, passed=False,
            message="SyntaxError: 少了括号")])
        numeric_route = classify([VerificationResult(
            level=VerificationLevel.L2_FUNCTION, passed=False,
            message="功能验证失败: 最大误差=1.0")])
        self.assertEqual("compile", compile_route.kind)
        self.assertEqual("numeric", numeric_route.kind)
        self.assertEqual("repair_compile",
                         Conductor._round_action_kind("fix", compile_route))
        self.assertEqual("repair_correctness",
                         Conductor._round_action_kind("fix", numeric_route))
        self.assertEqual("propose_template",
                         Conductor._round_action_kind("template", None))
        self.assertEqual("propose_source_patch",
                         Conductor._round_action_kind("generate", None))
        self.assertEqual("propose_source_patch",
                         Conductor._round_action_kind("optimize", None))
        # `fix` 而没有路由: 只能是「就地修」那一侧的一种, 不许造第三种词
        self.assertEqual("repair_compile", Conductor._round_action_kind("fix", None))

    def test_the_declared_token_budget_is_the_limit_the_call_will_use(self):
        """声明与实际必须同源: 「声明 1000、实际花 8192」会让这道门看起来永远能过"""
        self.assertEqual(CoderPipeline.GEN_MAX_TOKENS,
                         Conductor._llm_tokens_budget_claim("generate"))
        self.assertEqual(CoderPipeline.GEN_MAX_TOKENS,
                         Conductor._llm_tokens_budget_claim("optimize"))
        self.assertEqual(FixerAgent.FIX_MAX_TOKENS,
                         Conductor._llm_tokens_budget_claim("fix"))
        src = (Path(__file__).resolve().parents[1] / "operator_agent"
               / "fixer.py").read_text(encoding="utf-8")
        self.assertIn("max_tokens=self.FIX_MAX_TOKENS", src,
                      "Fixer 那次调用的 max_tokens 不再读常量了 —— 声明与实际又分家了")
        self.assertEqual(1, Conductor._llm_tokens_budget_claim("template"),
                         "模板轮不花 token, 声明必须是个正整数下限")


class TestTheRefusalResult(unittest.TestCase):
    """拒绝的**后果**是定义好的: 一条 L1 失败 + 两条给读者用的字"""

    def test_it_is_an_l1_failure_that_says_it_never_compiled(self):
        from operator_agent import agent_runtime as ar
        rt = ar.AgentRuntime()
        action = ar.AgentAction("r1-fix", ar.KIND_REPAIR_COMPILE, "修",
                                parent_candidate="src:x",
                                patch=ar.Patch("candidate/code.py", "def kernel(A): pass",
                                               "def kernel(A, B): pass"),
                                token_budget=64)
        verdict = rt.admit(action, ar.ActionContext(route_kind="compile",
                                                    input_key_sets=(("A", "B"),)))
        self.assertFalse(verdict.accepted)
        result = Conductor._refusal_result(verdict)
        self.assertEqual(VerificationLevel.L1_COMPILE, result.level)
        self.assertFalse(result.passed)
        self.assertTrue(result.metrics["agent_action_refused"])
        self.assertEqual("agent_action_refused", result.metrics["compile_status"])
        self.assertIn("没有", result.error_log)
        self.assertIn("不修就会被同样拦下", result.error_log)
        self.assertIn("ABI", result.message)
        # Fixer 的处方是**这段字**: 它必须能照着修
        self.assertIn(verdict.problems[0], result.error_log)


if __name__ == "__main__":
    unittest.main(verbosity=2)