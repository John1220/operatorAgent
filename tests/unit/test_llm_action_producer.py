"""§21.1 **生产者侧**的守卫: 动作信封真的从 LLM 交上来了, 而且判据一个都没被绕开

接线前 `AgentAction` 是 harness 把候选包出来的 —— `kind` / `rationale` /
`expected_bottleneck` / `requested_tools` 全由 harness 填, 于是七个动作种类里
`request_profiler` / `request_new_shape` / `explain_failure` **没有生产者**。
这一组问五件事:

1. **读得出来** —— 信封三态 (`parsed` / `absent` / `malformed`) 分得开, 且信封那一段
   在**代码提取之前**就被切掉 (它里面写什么都不可能被当成候选);
2. **判据没被绕开** —— 模型写的种类/rationale 走的是**同一套** §21.1/§21.2/§21.4,
   越权、表外词汇、不点误差形态一律被拒 (零 GPU 秒);
3. **数字与路径仍是 harness 的** —— `patch.path` 与 `token_budget` 不许模型自己填
   (自己填预算 = 让被判的人填判据里的数);
4. **询问类动作真的有响应** —— 三个 ask 种类都有生产者, 答/拒说清是哪一种,
   上界是「响应一次后仍无补丁就停」(不设上界就是一条白烧 token 的环);
5. **单源** —— 解析只有一处; 模板里那七个种类就是 `ACTION_KINDS` (闭环词汇表不许漂)。

运行: python3 -m unittest tests.test_llm_action_producer -v
"""

import contextlib
import io
import json
import re
import ast
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from operator_agent import agent_runtime as ar
from operator_agent.agent_runtime import (
    ACTION_KINDS, ENVELOPE_ABSENT, ENVELOPE_FENCE, ENVELOPE_MALFORMED,
    ENVELOPE_PARSED, SIDE_EFFECT_FREE_KINDS, TOOL_PROFILER,
    AgentAction, AgentRuntime, ask_unavailable_reason, parse_action_envelope,
)
from operator_agent.conductor import Conductor
from operator_agent.fixer import FixerAgent
from operator_agent.kernel_db import KernelDatabase
from operator_agent.mock_gpu import MockGPU
from operator_agent.schemas import (
    ConstraintSpec, Precision, SourceSpec, TargetSpec, TaskComplexity, TaskSchema,
    TaskType, VerificationLevel, VerificationResult,
)

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "operator_agent" / "prompts" / "fixer_repair.md"

#: L1 过、L2 挂 (结果错) ⇒ 走「数值修复」那一格, 真机读数: 形态 = `whole_tensor`
WRONG_NUMBERS = "import numpy as np\n\ndef kernel(A, B):\n    return A @ B + 1.0\n"
GOOD_KERNEL = "import numpy as np\n\ndef kernel(A, B):\n    return A @ B\n"
SHAPE = "whole_tensor"
SKETCH = {"algorithm": "matmul",
          "tiling": {"block_m": 64, "block_n": 64, "block_k": 32},
          "memory_plan": {}, "pipeline": [], "optimization_hints": []}


def _envelope(**payload) -> str:
    """一段信封 + 一份代码 (修 patch 类动作的常规形状)"""
    body = json.dumps(payload, ensure_ascii=False)
    return f"```{ENVELOPE_FENCE}\n{body}\n```\n\n```python\n{GOOD_KERNEL}\n```\n"


def _ask(kind: str, rationale: str = "") -> str:
    """只交信封、不交代码 (不改候选的那三类动作)"""
    return "```" + ENVELOPE_FENCE + "\n" + json.dumps(
        {"kind": kind, "rationale": rationale}, ensure_ascii=False) + "\n```\n"


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


class _CountingGpu(MockGPU):
    """记下每一份到过后端的源码 —— 判据是**内容级**的, 不是次数级的"""

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


class _EnvelopeLlm:
    """Fixer 的每一次调用按**脚本**发回复 (生成侧恒给「结果错」的码)

    按 `caller_tag` 分派而不是按调用次序: 生成侧那几个采样候选与这里要测的东西无关。
    `fix_replies` 用完后重复最后一条 —— 于是「第二次还是提问」这种用例很好写。
    """

    def __init__(self, fix_replies, generate=WRONG_NUMBERS):
        self.fix_replies = list(fix_replies)
        self.generate = generate
        self.sent = []
        self.fix_calls = 0

    def chat(self, messages, **kw):
        tag = kw.get("caller_tag", "?")
        prompt = "\n".join(str(m.get("content", "")) for m in messages)
        self.sent.append((tag, prompt))
        if tag.startswith("Fixer"):
            i = min(self.fix_calls, len(self.fix_replies) - 1)
            self.fix_calls += 1
            return self.fix_replies[i]
        return self.generate

    def chat_json(self, messages, **kw):
        return dict(SKETCH)

    def prompts(self, tag="Fixer"):
        return [p for t, p in self.sent if t.startswith(tag)]


def _run(fix_replies, *, rounds=3, llm=None, backend=None, **kw):
    llm = llm or _EnvelopeLlm(fix_replies)
    backend = backend or _CountingGpu()
    c = Conductor(backend=backend, kernel_db=KernelDatabase(jsonl_path=None),
                  llm=llm, max_optimize_rounds=rounds, heartbeat_interval=1000, **kw)
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        art = c.execute(_task(), _inputs())
    return c, art, llm, backend, buf.getvalue()


def _round(art, n):
    return next(h for h in art.iteration_history if h.get("round") == n)


# ─── 1. 读得出来 (三态 + 切段) ──────────────────────────────────

class TestTheEnvelopeIsReadTriState(unittest.TestCase):

    def test_no_envelope_is_absent_not_an_error(self):
        """接线前模型就是这么回话的 ⇒ `absent` 是**正常**档, 不是失败"""
        env, body = parse_action_envelope("```python\nimport numpy as np\n```")
        self.assertEqual(ENVELOPE_ABSENT, env.status)
        self.assertTrue(env.reason, "absent 也要说清为什么 (产物里只有状态没有原因是无用的)")
        self.assertFalse(env.usable)
        self.assertIn("import numpy", body, "没有信封时正文不许被动过")

    def test_malformed_is_separate_from_absent(self):
        """「读不出来」与「读出来了但判据不过」是两件事; 压成一个布尔就分不开了"""
        cases = {
            "JSON 错": "```agent-action\n{kind: x}\n```",
            "不是对象": "```agent-action\n[1, 2]\n```",
            "没有 kind": '```agent-action\n{"rationale": "x"}\n```',
            "类型不对": '```agent-action\n{"kind": "explain_failure", "requested_tools": "profiler"}\n```',
            "两段": _ask("explain_failure") + _ask("explain_failure"),
        }
        for why, text in cases.items():
            env, _ = parse_action_envelope(text)
            self.assertEqual(ENVELOPE_MALFORMED, env.status, why)
            self.assertTrue(env.reason, why)
            self.assertFalse(env.usable, why)

    def test_an_unknown_kind_is_parsed_and_left_to_the_judge(self):
        """模型说了它想干什么, 只是那件事不被允许 ⇒ 让判据去拒, 并留下拒绝理由"""
        env, _ = parse_action_envelope(_ask("teleport"))
        self.assertEqual(ENVELOPE_PARSED, env.status)
        self.assertEqual("teleport", env.kind)
        self.assertFalse(env.kind_known, "表外的种类必须认出来是表外的")
        problems = ar.schema_problems(AgentAction(
            action_id="a", kind=env.kind, rationale="x", token_budget=1))
        self.assertTrue(any("闭合词汇表" in p for p in problems),
                        f"表外种类没被判据拦下: {problems}")

    def test_the_envelope_is_cut_out_before_code_is_extracted(self):
        """信封那一段在代码提取**之前**就被切掉 —— 它里面写什么都不可能当候选"""
        reply = _envelope(kind="repair_correctness",
                          rationale="先 import os 再说话 (这句在信封里, 不在候选里)")
        env, body = parse_action_envelope(reply)
        self.assertEqual(ENVELOPE_PARSED, env.status)
        self.assertNotIn(ENVELOPE_FENCE, body)
        self.assertNotIn("先 import os 再说话", body, "信封的正文没被切干净")
        self.assertEqual(GOOD_KERNEL.strip(), FixerAgent._extract_code(body).strip())

    def test_the_idempotent_shape_of_the_record(self):
        env, _ = parse_action_envelope(_envelope(kind="repair_compile", rationale="r"))
        d = env.as_dict()
        self.assertEqual({"status", "reason", "kind", "rationale", "expected_bottleneck",
                          "requested_tools", "declared_token_budget", "unknown_keys"},
                         set(d))
        self.assertNotIn("raw", d, "原文不进血缘: 它已经在 prompt 侧留过痕了")


# ─── 2/3. 模型写意图, harness 写数字与路径 ───────────────────────

class TestTheModelAuthoredActionReachesTheGate(unittest.TestCase):

    maxDiff = 3000

    def test_the_kind_and_the_rationale_come_from_the_model(self):
        _, art, _, backend, _ = _run([_envelope(
            kind="repair_correctness", rationale=f"误差形态={SHAPE}: 累加器 dtype 错了")])
        record = _round(art, 2)
        action = record["agent_action"]["action"]
        self.assertEqual("repair_correctness", action["kind"])
        self.assertIn("累加器 dtype 错了", action["rationale"], "理由不是模型写的那份")
        self.assertEqual([], action["requested_tools"])
        self.assertEqual("llm", record["action_envelope"]["source"])
        self.assertTrue(record["action_envelope"]["used_for_action"])
        self.assertTrue(record["agent_action"]["accepted"],
                        {"schema": record["agent_action"]["schema_problems"],
                         "permission": record["agent_action"]["permission_problems"],
                         "budget": record["agent_action"]["budget_problems"]})
        # 后端收到的那份与常量可能只差首尾空白 ⇒ 逐字比会把「到了」读成「没到」
        self.assertIn(GOOD_KERNEL.strip(), [c.strip() for c in backend.seen],
                      "过了门的候选必须真的被送去验证")

    def test_the_harness_still_owns_the_path_and_the_budget(self):
        """模型自己填 `path` 与预算 ⇒ 路径仍是布局给的, 预算仍是 harness 那个上限"""
        _, art, _, _, _ = _run([_envelope(
            kind="repair_correctness",
            rationale=f"误差形态={SHAPE}",
            path="contract/frozen.json", token_budget=1)])
        record = _round(art, 2)
        env_rec = record["action_envelope"]
        self.assertIn("path", env_rec["unknown_keys"],
                      "信封里那个越权的 path 键被静默丢了 —— 它必须留下痕迹")
        self.assertEqual(1, env_rec["declared_token_budget"],
                         "模型声明的预算要留档 (但**不进判据**)")
        patch = record["agent_action"]["action"]["patch"]
        self.assertTrue(patch["path"].endswith("candidate/code.py"),
                        f"写哪个文件不由模型说了算: {patch['path']}")
        self.assertEqual(FixerAgent.FIX_MAX_TOKENS,
                         record["agent_action"]["action"]["token_budget"],
                         "预算那一格被模型写的数顶掉了 ⇒ 预算门形同虚设")

    def test_a_model_requested_tool_this_backend_lacks_is_refused(self):
        """模型点了一个本后端没有的能力 ⇒ 判据在**零 GPU 秒**上拒 (它说了不算)

        这是判据第一次遇到一个真正不可靠的生产者: `verify_numerics` 在注册表里有实现物,
        但它要的后端能力 `remote_verify` 不在 MockGPU 声明的能力集合里 —— 两件事分得开
        (「没有实现物」与「这台机器不支持」) 正是 `ask_unavailable_reason` 与 §21.4
        第 7 条同源那一条要守的。
        """
        _, art, _, backend, _ = _run([_envelope(
            kind="repair_correctness", rationale=f"误差形态={SHAPE}: 想先验证一遍",
            requested_tools=["verify_numerics"])])
        record = _round(art, 2)
        self.assertFalse(record["agent_action"]["accepted"])
        violation = [o for o in record["agent_action"]["precompile"]["outcomes"]
                     if o["state"] == "violation"]
        self.assertEqual(["capability"], [o["rule"] for o in violation])
        self.assertIn("remote_verify", violation[0]["detail"])
        self.assertNotIn(GOOD_KERNEL.strip(), [c.strip() for c in backend.seen],
                         "被拒的候选还是上了后端")

    def test_a_model_kind_outside_the_route_is_refused(self):
        """数值失败那一轮不许提交「结构性变更」(§21.2 的表) —— 模型说了不算"""
        _, art, _, backend, out = _run([_envelope(
            kind="propose_source_patch", rationale=f"误差形态={SHAPE}")])
        record = _round(art, 2)
        self.assertFalse(record["agent_action"]["accepted"])
        self.assertTrue(any("不被允许" in p
                            for p in record["agent_action"]["permission_problems"]),
                        record["agent_action"]["permission_problems"])
        self.assertNotIn(GOOD_KERNEL.strip(), [c.strip() for c in backend.seen],
                         "被拒的候选还是上了后端")
        self.assertIn("动作被拒", out)

    def test_a_rationale_without_the_error_shape_is_refused(self):
        """§21.2「必须带误差形态」从**构造保证**变成了对真实输入的判据

        原先理由由 harness 拼装 ⇒ 那条判据永远不会红。现在模型自己写理由, 不点名形态就
        被拒 (零 GPU 秒), 而拒绝理由会作为下一轮的错误信息回到模型眼前。
        """
        _, art, llm, backend, _ = _run([_envelope(kind="repair_correctness",
                                                  rationale="我把索引改了一下")])
        record = _round(art, 2)
        self.assertFalse(record["agent_action"]["accepted"])
        self.assertTrue(any("误差形态" in p
                            for p in record["agent_action"]["permission_problems"]),
                        record["agent_action"]["permission_problems"])
        self.assertNotIn(GOOD_KERNEL.strip(), [c.strip() for c in backend.seen])
        # 「公平」那一半: 这个 token 必须在 prompt 里给过模型 (否则就是猜谜)
        self.assertIn(SHAPE, llm.prompts("Fixer")[0],
                      "误差形态没进 prompt ⇒ 判据变成了猜谜")
        self.assertEqual("我把索引改了一下",
                         record["agent_action"]["action"]["rationale"],
                         "harness 把误差形态补进了模型的理由 ⇒ 这条判据又变成永不失败")

    def test_a_plain_reply_falls_back_to_the_harness_action(self):
        """交不出信封 ⇒ 接线前的行为一字不差 (动作由 harness 按轮次与路由拼)"""
        _, art, _, _, _ = _run(["```python\n" + GOOD_KERNEL + "\n```"])
        record = _round(art, 2)
        self.assertEqual(ENVELOPE_ABSENT, record["action_envelope"]["status"])
        self.assertFalse(record["action_envelope"]["used_for_action"])
        self.assertEqual("harness", record["action_envelope"]["source"])
        self.assertEqual("repair_correctness", record["agent_action"]["action"]["kind"],
                         "harness 拼的那一档改了 ⇒ 这不是「退回接线前」")
        self.assertIn("第 2 轮 fix", record["agent_action"]["action"]["rationale"])

    def test_the_envelope_does_not_leak_into_a_later_round(self):
        """信封只在「修复轮 → 下一轮」这一格有效 (与 `pending_fix` 同生同灭)"""
        # 第 2 轮修复交信封 + 好代码 ⇒ 第 2 轮通过; 第 3 轮由 Coder 生成 ⇒ 没有信封
        _, art, _, _, _ = _run([_envelope(kind="repair_correctness",
                                          rationale=f"误差形态={SHAPE}")], rounds=4)
        self.assertEqual("llm", _round(art, 2)["action_envelope"]["source"])
        for n in (1, 3):
            self.assertEqual("harness", _round(art, n)["action_envelope"]["source"],
                             f"第 {n} 轮不该带着上一轮的信封")

    def test_the_parser_is_the_only_way_in(self):
        """解析只有一处: 别的模块自己 `re.search` 一遍就是第二个真源"""
        users = []
        for p in sorted((ROOT / "operator_agent").glob("*.py")):
            if "parse_action_envelope" in p.read_text(encoding="utf-8"):
                users.append(p.name)
        self.assertEqual(["agent_runtime.py", "fixer.py"], users,
                         f"信封解析/定义出现在了 {users} —— 单源被破坏")
        calls = (ROOT / "operator_agent" / "fixer.py").read_text(encoding="utf-8")
        self.assertEqual(1, calls.count("parse_action_envelope("),
                         "Fixer 里解析了不止一次")

    def test_the_pending_envelope_is_armed_once_and_read_once(self):
        """防腐 (AST): 那扇门**读一处、写四处**, 多一处就要说清「谁清它」

        读 = 唯一消费点 (与 `current_code = pending_fix` 同一个 if 块); 写 = 初值 /
        消费时清空 / 修复轮武装 / 提前 break 时清空。写多一处不是错, 但它必须是有意的 ——
        本仓 I77/I78 两个缺陷的形状正是「一个写进去、没人负责清的留痕」。
        """
        tree = ast.parse((ROOT / "operator_agent" / "conductor.py").read_text(encoding="utf-8"))
        loads, stores = 0, 0
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute)
                    and node.attr == "_pending_action_envelope"
                    and isinstance(node.value, ast.Name) and node.value.id == "self"):
                if isinstance(node.ctx, ast.Load):
                    loads += 1
                elif isinstance(node.ctx, ast.Store):
                    stores += 1
        self.assertEqual(1, loads, f"信封的读点有 {loads} 处 (要恰好 1: 与 pending_fix 同格)")
        self.assertEqual(4, stores,
                         f"信封的写点有 {stores} 处 (要恰好 4: 初值/消费清空/武装/break 清空)"
                         " —— 加了一处就要顺带说清它在哪一格清")


    def test_a_resume_loses_the_intent_not_the_candidate(self):
        """续跑只在快照里存了 `pending_fix`(**代码**), 没存信封(**意图**)

        这不是遗漏而是取舍: 快照的读者是「进程没了以后把循环接上」, 而动作信封只对
        **紧接着的那一轮**成立 (它就是那一轮候选的来意)。取舍要留痕: 续跑之后那一轮
        的血缘里 `action_envelope.status` 必须是 `absent` —— 也就是「harness 按轮次与
        路由拼了动作」, 而**不是**凭空造一份模型理由。哪天有人真把信封写进快照, 这条
        守卫会红, 那时要把 `_restore_state_from_snapshot` 一起接上。
        """
        tree = ast.parse((ROOT / "operator_agent" / "conductor.py").read_text(encoding="utf-8"))
        def body_src(name):
            node = next(n for n in ast.walk(tree)
                        if isinstance(n, ast.FunctionDef) and n.name == name)
            return ast.dump(node)
        self.assertIn("'pending_fix'", body_src("_build_snapshot_dict"),
                      "快照不再存候选了 ⇒ 续跑接不上, 这条守卫的前提变了")
        for name in ("_build_snapshot_dict", "_restore_state_from_snapshot"):
            dumped = body_src(name)
            self.assertNotIn("action_envelope", dumped, f"{name} 动了信封的存取")
            self.assertNotIn("_pending_action_envelope", dumped,
                             f"{name} 动了信封属性的存取 ⇒ 要同时接上另一半, 别只做一半")


# ─── 4. 询问类动作: 真的有生产者, 也真的有响应 ──────────────────

class TestTheAskKindsHaveRealProducers(unittest.TestCase):

    #: 今天每个 ask 种类落在哪一支 —— 工具落地那天这张表会红, 逼人就着新事实改
    EXPECTED = {
        "explain_failure": ("accepted_explanation", "你的解释已记入本轮失败记录"),
        "request_profiler": ("refused_no_gauge", "这条请求**答不出来**"),
        "request_new_shape": ("refused_no_gauge", "这条请求**答不出来**"),
    }

    def test_every_ask_kind_is_answered_and_then_a_patch_is_asked_for(self):
        """三个 ask 种类逐个走一遍: 响应一次 ⇒ 要点补丁 ⇒ 拿到补丁

        断言的是**那一句答复**逐字进了第二份 prompt, 而不是「第二份 prompt 里有动作信封
        那段」—— 后者永远为真 (静态协议段两臂都在), 是一条空转的断言。
        """
        self.assertEqual(set(SIDE_EFFECT_FREE_KINDS), set(self.EXPECTED),
                         "ask 种类增删了 ⇒ 这张分支表要跟着改 (否则新种类没有响应)")
        for kind, (answer_kind, fragment) in self.EXPECTED.items():
            with self.subTest(kind=kind):
                # 第一份回复是询问、第二份是补丁 —— 询问发生在**第 1 轮**的修复步
                # (于是它记在第 1 轮的血缘里), 补丁在第 2 轮被判据过。
                _, art, llm, _, out = _run([_ask(kind, "我需要先看一眼"),
                                            "```python\n" + GOOD_KERNEL + "\n```"])
                record = _round(art, 1)
                fix = record["fix_action"]
                self.assertEqual(kind, fix["ask_field"])
                self.assertEqual(1, fix["ask_turns"], fix)
                self.assertEqual("patch_after_ask", fix["ask_outcome"], fix)
                self.assertEqual(answer_kind, fix["ask_answer_kind"], fix)
                self.assertGreaterEqual(len(llm.prompts("Fixer")), 2)
                self.assertIn(fragment, llm.prompts("Fixer")[1],
                              "那句答复没到模型眼前 ⇒ 响应等于没发生")
                self.assertNotIn(fragment, llm.prompts("Fixer")[0],
                                 "第一份 prompt 里就有了答复 ⇒ 答复被拼进了别处")
                self.assertIn(f"响应动作信封 {kind}", out)
                # 补丁真的被要到了: 循环走到了第 2 轮 (没有在第 1 轮就停)
                self.assertGreaterEqual(max(h.get("round", 0)
                                            for h in art.iteration_history), 2)

    def test_the_profiler_ask_is_refused_for_lack_of_a_gauge(self):
        """今天 profiler 那一档注册表里是 `None` ⇒ 答不出来, 而且**不许**说成「观测是空」"""
        reason = ask_unavailable_reason("request_profiler")
        self.assertTrue(reason, "profiler 今天没有实现物 —— 这条请求答不出来")
        _, art, llm, _, _ = _run([_ask("request_profiler"),
                                  "```python\n" + GOOD_KERNEL + "\n```"])
        fix = _round(art, 1)["fix_action"]
        self.assertEqual("refused_no_gauge", fix["ask_answer_kind"], fix)
        self.assertEqual(reason, fix["ask_refusal"])
        self.assertIn("答不出来", llm.prompts("Fixer")[1])
        self.assertIn(TOOL_PROFILER, llm.prompts("Fixer")[1])

    def test_an_explanation_is_recorded_not_just_acknowledged(self):
        _, art, _, _, _ = _run([_ask("explain_failure", "根因是 benchmark 侧输入分布"),
                                "```python\n" + GOOD_KERNEL + "\n```"])
        fix = _round(art, 1)["fix_action"]
        self.assertEqual("accepted_explanation", fix["ask_answer_kind"])
        self.assertIn("benchmark 侧输入分布", fix["explanation"],
                      "模型写下的解释必须留档 —— 只说「收到了」等于把它丢了")

    def test_asking_twice_stops_with_a_reason(self):
        """上界: 响应一次后仍是提问 ⇒ 停, 并把这一档记进出口词汇表"""
        _, art, llm, backend, out = _run([_ask("explain_failure", "我还是不改")])
        stop = (art.metadata or {}).get("stop") or {}
        self.assertEqual("ask_turns_exhausted", stop.get("reason"), stop)
        fix = _round(art, 1)["fix_action"]
        self.assertEqual(2, fix["ask_turns"])
        self.assertEqual(2, llm.fix_calls, "第二次之外还接着问 ⇒ 环没有上界")
        self.assertNotIn(GOOD_KERNEL.strip(), [c.strip() for c in backend.seen],
                         "没有候选却被送上了后端")

    def test_an_ask_that_comes_with_code_loses_the_code(self):
        """「借请求之名改代码」: 声明自己不改候选时, harness **不收**那份代码"""
        class _Llm:
            def chat(self, messages, **kw):
                return _envelope(kind="explain_failure", rationale="顺手也改了")

            def chat_json(self, messages, **kw):
                return dict(SKETCH)

        f = FixerAgent(llm=_Llm())
        out = f.run_with_action(WRONG_NUMBERS, _task(),
                                [VerificationResult(level=VerificationLevel.L2_FUNCTION,
                                                    passed=False, message="功能验证失败")])
        self.assertTrue(out.asked, out.as_dict())
        self.assertEqual("", out.code, "声明不改候选时不许把代码交出去")
        # `run` 是同一个东西的两格投影 (既有调用方不必知道信封)
        pair = f.run(WRONG_NUMBERS, _task(),
                     [VerificationResult(level=VerificationLevel.L2_FUNCTION,
                                         passed=False, message="功能验证失败")])
        self.assertEqual(("", out.summary), pair)

    def test_the_capability_judgement_has_one_source(self):
        """`ask_unavailable_reason` 与 §21.4 第 7 条必须**同源** (各写一份必然漂)

        漂的方向很具体: profiler 哪天落地了, 一处改了、另一处没改 ⇒ 「请求了却没有」与
        「请求了就有」同时成立。
        """
        rt = AgentRuntime()
        for kind in SIDE_EFFECT_FREE_KINDS:
            action = AgentAction(action_id="a", kind=kind, rationale="r", token_budget=1)
            verdict = rt.admit(action, ar.ActionContext(language="python"))
            details = [o.detail for o in (verdict.precompile.outcomes
                                          if verdict.precompile else ())]
            flagged = any("没有实现物" in d for d in details)
            self.assertEqual(
                bool(ask_unavailable_reason(kind)), flagged,
                f"{kind}: 两处判据不一致 (答不出的原因={ask_unavailable_reason(kind)!r}, "
                f"判据说={flagged})")


# ─── 5. 模板与词汇表不漂 ───────────────────────────────────────

class TestThePromptAndTheVocabularyDoNotDrift(unittest.TestCase):

    def test_the_template_teaches_the_same_fence_tag(self):
        text = TEMPLATE.read_text(encoding="utf-8")
        self.assertIn(f"```{ENVELOPE_FENCE}", text,
                      "模板教的围栏标签与解析器认的不是同一个 ⇒ 信封永远读不出来")

    #: 信封段里合法出现的非动作名 (JSON 键与围栏标签本身)。多一个就要问它是什么。
    ENVELOPE_WORDS = frozenset({"agent-action", "kind", "rationale",
                                "expected_bottleneck", "requested_tools", "path"})

    def test_the_template_lists_exactly_the_closed_vocabulary(self):
        """模板教的种类就是 `ACTION_KINDS`: 一边加了另一边没加 ⇒ 那个种类谁也交不出来

        反方向也要钉: 模板里出现一个**不在** ACTION_KINDS 里的动作名 (改名/漏删) ⇒
        模型会照着一个判据不认的名字去交信封。
        """
        text = TEMPLATE.read_text(encoding="utf-8")
        section = text.split("## 动作信封", 1)[1].split("## 输出要求", 1)[0]
        for kind in ACTION_KINDS:
            self.assertIn(kind, section, f"模板没告诉模型有 {kind} 这个种类")
        listed = set(re.findall(r"`([a-z_]+)`", section))
        extra = listed - set(ACTION_KINDS) - self.ENVELOPE_WORDS
        self.assertFalse(extra, f"模板里出现了词汇表外的名字: {sorted(extra)}")

    def test_no_placeholder_leaks_with_the_new_sections(self):
        """模板新加的 `{{ask_answer}}` / `{{error_shape}}` 也必须在每个调用点都传上"""
        _, _, llm, _, _ = _run([_ask("explain_failure", "x")])
        for tag, prompt in llm.sent:
            self.assertNotIn("{{", prompt, f"未替换的模板变量漏进了 {tag} 的 prompt")


if __name__ == "__main__":
    unittest.main()