"""调用契约由 harness 广播给 Coder 的守卫 (#76③ 第 1 片 / 架构缺口 6)

性质只有一条: **模型写签名时看的输入名, 必须是调用方实测要传的那些**, 而不是它自己
从 `kernel_type` 猜的 (CUDA-Harness arXiv:2609.00058v1 §4.1.2 把这条做成了编译期广播)。

为什么值得单独守: 这一处之前是反着的 —— `planner.py:54,84` 按 kernel_type 拼一个
`def kernel(A, B)` 放进计划 JSON, 那份 JSON 整体注入 coder 的 prompt ⇒ layernorm 这类
任务里 prompt 同时带着一条**错的**契约, 而正确的形状要等远端抛 `TypeError` 才知道
(`interface_contract.py` 在 L1 前拦的那条)。所以这里三面一起测:

1. 广播的**内容** (三种形态: 单组 / 多组同名 / 多组不同名 / 无输入信息);
2. 广播与**判据同源** —— 照广播写的核必须过 `check_kernel_interface` (L1 那一面),
   少一个名字就必须拦下;
3. 广播真的**进了两条 coder 路径的 prompt** (legacy `_llm_generate_code` 与
   `CoderPipeline.run`), 且四个生成模板都渲染它 —— #67 那条守卫按调用点取**并集**,
   所以「只在 triton 模板里加占位符、cuda_c 那条走不到」在它眼里是绿的。

变异标定 (每条只改一处, 逐轮复原并做逐字节自检; 教训见 doc 49 §八 第 22 条):
  M1 广播恒走 `_UNKNOWN` (不看 test_inputs)          → 8 条红
  M2 多组输入只取第一组                                → 多组不同名 + 组数
  M3 多组同名时不报组数                                → 「共 2 组」断言
  M4 只在一个模板里放 `{{calling_convention}}`          → 模板覆盖那条 (#67 的并集语义看不见它)
  M5 `CoderAgent.run` 不往 pipeline 传 test_inputs     → 入口转发那条
  M6 `_sample_one` 渲染时不传 calling_convention        → pipeline 功能 + 装配契约
  M7 conductor 那处 `coder.run` 不传 test_inputs        → 链路首环的 AST 用例
  M8 去掉「以本节为准」那句裁决                          → layernorm 覆盖用例
实测: 8/8 拦住, 0 逃逸。**M5 第一次跑是绿的** —— 当时的用例从 `_llm_generate_code` 和
`CoderPipeline.run` 进去, 测的是被调方, 漏的正是中间那层转发; 补了
`test_coder_agent_run_forwards_inputs_on_both_branches` 之后才红 (doc 49 §八 第 24 条)。

运行: python3 -B -m unittest tests.test_calling_convention_broadcast -v
"""

import ast
import io
import os
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from operator_agent.agents import CoderAgent                      # noqa: E402
from operator_agent.coder_pipeline import CoderPipeline           # noqa: E402
from operator_agent.interface_contract import (                   # noqa: E402
    broadcast_calling_convention, check_kernel_interface, input_key_sets)
from operator_agent.schemas import (                              # noqa: E402
    TaskSchema, SourceSpec, TargetSpec, ConstraintSpec, UnifiedSketch,
    TaskType, TaskComplexity, Precision)

PKG = ROOT / "operator_agent"
PROMPTS = PKG / "prompts"
HEADER = "## 调用契约 (由 harness 广播, 不是你的推测)"
#: 生成阶段的四个模板 (coder_repair/fixer_* 不在这一片里)
GENERATION_TEMPLATES = ("coder_from_plan", "coder_generate",
                        "coder_generate_triton", "coder_generate_cuda_c")

_NAMED_LINE = re.compile(r"会传入的输入名[:：](.+)")


def _broadcast_names(text: str) -> set:
    """从广播文本里取出「本次会传入的输入名」列出的那些反引号名。

    只在含「会传入的输入名」的行上取 —— `_UNKNOWN` 那一段里也带反引号
    (`kernel(**输入)`), 全局取会把它当成一个输入名。
    """
    out = set()
    for line in text.splitlines():
        for m in _NAMED_LINE.finditer(line):
            out |= {t.strip("`") for t in re.findall(r"`[^`]+`", m.group(1))}
    return out


class _RecordingLLM:
    """记录每次 chat 的完整 prompt; chat_json 交回一份可预测的计划"""

    def __init__(self, reply, plan):
        self.prompts = []
        self._reply = reply
        self._plan = plan

    def chat(self, messages, temperature=None, max_tokens=None, raw=False, caller_tag="Unknown"):
        self.prompts.append(messages[-1]["content"])
        return self._reply

    def chat_json(self, messages, temperature=None, max_tokens=None, caller_tag="Unknown"):
        return dict(self._plan)


class _EmptyDB:
    def search(self, **kwargs):
        return []


def _task(dsl="python", kernel_type="layernorm") -> TaskSchema:
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type=kernel_type, complexity=TaskComplexity.LOW),
        target=TargetSpec(hardware="mock_gpu", dsl=dsl, compute_units=64, warp_size=32),
        constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=0.5, max_error=1e-5),
    )


def _sketch(dsl="python", algorithm="layernorm") -> UnifiedSketch:
    return UnifiedSketch(algorithm=algorithm, version=1, target_dsl=dsl)


#: Planner 按 kernel_type 猜出来的那份签名 —— 与真实输入名不同, 是这一片要压制的东西
GUESS_PLAN = {"algorithm": "layernorm", "kernel_signature": "def kernel(A, B)",
              "jit_signature": "", "steps": ["alloc", "launch"],
              "tiling": {"block_m": 64, "block_n": 64, "block_k": 32},
              "risks": [], "notes": ""}
LAYERNORM_INPUTS = {"X": None, "weight": None, "bias": None}
MATMUL_CODE = "import numpy as np\n\ndef kernel(A, B):\n    return A @ B\n"


# ─── 1. 广播的内容 ───────────────────────────────────────────

class TestBroadcastShapes(unittest.TestCase):

    def test_single_input_set_lists_every_name(self):
        text = broadcast_calling_convention(LAYERNORM_INPUTS)
        self.assertIn(HEADER, text)
        self.assertEqual({"X", "weight", "bias"}, _broadcast_names(text))
        self.assertIn("全关键字", text)

    def test_multiple_sets_with_different_names_list_each_group(self):
        text = broadcast_calling_convention([{"A": 1, "B": 2}, {"A": 1, "causal": 2}])
        lines = [ln for ln in text.splitlines() if "会传入的输入名" in ln]
        self.assertEqual(2, len(lines), f"两组输入各列一行, 实际 {len(lines)} 行: {text}")
        self.assertEqual({"A", "B"}, _broadcast_names(lines[0]))
        self.assertEqual({"A", "causal"}, _broadcast_names(lines[1]))
        self.assertIn("不同", text)

    def test_multiple_sets_with_same_names_collapse_but_count_the_sets(self):
        text = broadcast_calling_convention([{"A": 1, "B": 2}, {"A": 1, "B": 2}])
        self.assertEqual({"A", "B"}, _broadcast_names(text))
        self.assertIn("共 2 组", text, "名字相同也要说清有几组, 否则读起来像只有一组")

    def test_unknown_inputs_still_broadcast_a_non_empty_contract(self):
        """契约永不为空: 没有 test_inputs 时不能整段消失, 那是把「没说」变成「没这块」。
        `{}` 也算未知 —— 摊出来是一组空名字, 照字面渲染会印出「会传入的输入名: (什么都没有)」,
        把一个「不知道」说成一个假的空集。"""
        for unknown in (None, {}, [], "nonsense"):
            text = broadcast_calling_convention(unknown)
            self.assertIn(HEADER, text)
            self.assertEqual(set(), _broadcast_names(text), f"{unknown} 不该列出输入名")
            self.assertIn("拿到输入名", text, f"{unknown} 应走「harness 不知道」这一支")


# ─── 2. 广播与判据同源 ───────────────────────────────────────

class TestBroadcastAndGateAgree(unittest.TestCase):
    """照广播写的核必须过 L1 契约门 —— 广播不是文案, 它必须与**拦截用的那份口径**一致"""

    def _kernel_source(self, names):
        args = ", ".join(sorted(names))
        return f"import numpy as np\n\ndef kernel({args}):\n    return 0\n"

    def test_kernel_written_to_the_broadcast_passes_the_gate(self):
        for inputs in (LAYERNORM_INPUTS, [{"A": 1, "B": 2}, {"A": 1, "B": 2}]):
            text = broadcast_calling_convention(inputs)
            names = _broadcast_names(text)
            self.assertTrue(names, f"广播没列出名字: {text}")
            for keys in input_key_sets(inputs):
                self.assertIsNone(check_kernel_interface(self._kernel_source(names), keys).problem,
                                  f"照广播写的核被 L1 契约门拦下了: {sorted(names)} vs {keys}")

    def test_dropping_one_broadcasted_name_is_caught(self):
        """反空转: 少写一个广播里的名字必须被拦 —— 否则上面那条恒真"""
        names = sorted(_broadcast_names(broadcast_calling_convention(LAYERNORM_INPUTS)))
        self.assertEqual(3, len(names), names)
        short = self._kernel_source(names[:2])
        self.assertIsNotNone(check_kernel_interface(short, names).problem,
                             "照广播少写一个参数却没被 L1 拦住 ⇒ 广播与判据不同源")

    def test_the_planners_guess_would_have_failed(self):
        """缺陷存在性的证据: Planner 那句猜测对 layernorm 就是错的契约"""
        msg = check_kernel_interface("def kernel(A, B):\n    return A @ B\n",
                                     list(LAYERNORM_INPUTS)).problem
        self.assertIsNotNone(msg, "如果这句变成 None, 那「模型在猜错的契约」这个前提没了, "
                                  "本文件的用例需要重读")


# ─── 3. 广播进了两条 coder 路径 ──────────────────────────────

class TestBothCoderPathsRenderIt(unittest.TestCase):

    def test_legacy_generate_prompt_contains_the_broadcast(self):
        llm = _RecordingLLM(MATMUL_CODE, GUESS_PLAN)
        CoderAgent(kernel_db=_EmptyDB(), llm=llm)._llm_generate_code(
            _sketch(), _task(), test_inputs=LAYERNORM_INPUTS)
        self.assertIn(HEADER, llm.prompts[0])
        self.assertEqual({"X", "weight", "bias"}, _broadcast_names(llm.prompts[0]))

    def test_legacy_path_without_inputs_still_carries_a_contract(self):
        llm = _RecordingLLM(MATMUL_CODE, GUESS_PLAN)
        CoderAgent(kernel_db=_EmptyDB(), llm=llm)._llm_generate_code(_sketch(), _task())
        self.assertIn(HEADER, llm.prompts[0])

    def test_pipeline_prompt_contains_the_broadcast_for_every_candidate(self):
        llm = _RecordingLLM(MATMUL_CODE, GUESS_PLAN)
        pipe = CoderPipeline(llm=llm, kernel_db=_EmptyDB())
        pipe.run(_sketch(), _task(), test_inputs=LAYERNORM_INPUTS)
        self.assertGreaterEqual(len(llm.prompts), 2, "并行采样应有多个候选的 prompt")
        for i, prompt in enumerate(llm.prompts):
            self.assertIn(HEADER, prompt, f"第 {i+1} 个候选的 prompt 里没有契约")
            self.assertEqual({"X", "weight", "bias"}, _broadcast_names(prompt))

    def test_coder_agent_run_forwards_inputs_on_both_branches(self):
        """走 `CoderAgent.run` —— 那才是 conductor 调的那一层。

        上面两条各测一条分支的内部函数 (`_llm_generate_code` / `CoderPipeline.run`),
        于是「Agent.run 忘了往下传」这一处**曾经逃过变异标定** (M5 全绿):
        测的是被调方, 漏的是转发方。这里两条分支都从入口进一次。
        """
        llm = _RecordingLLM(MATMUL_CODE, GUESS_PLAN)
        CoderAgent(kernel_db=_EmptyDB(), llm=llm).run(
            _sketch(), _task(), test_inputs=LAYERNORM_INPUTS)
        self.assertIn(HEADER, llm.prompts[-1], "legacy 分支: Agent.run → _llm_generate_code")

        llm2 = _RecordingLLM(MATMUL_CODE, GUESS_PLAN)
        pipe = CoderPipeline(llm=llm2, kernel_db=_EmptyDB())
        CoderAgent(kernel_db=_EmptyDB(), llm=llm2, pipeline=pipe).run(
            _sketch(), _task(), test_inputs=LAYERNORM_INPUTS)
        self.assertIn(HEADER, llm2.prompts[-1], "pipeline 分支: Agent.run → Pipeline.run")
        self.assertEqual({"X", "weight", "bias"}, _broadcast_names(llm2.prompts[-1]))

    def test_broadcast_outranks_the_planners_guess(self):
        """plan 里那句 `def kernel(A, B)` 仍在 (它是 Planner 的输出, 本片不动它),
        所以广播必须当场裁决「以本节为准」, 否则 prompt 里是两条并列的契约。"""
        llm = _RecordingLLM(MATMUL_CODE, GUESS_PLAN)
        CoderPipeline(llm=llm, kernel_db=_EmptyDB()).run(
            _sketch(), _task(), test_inputs=LAYERNORM_INPUTS)
        prompt = llm.prompts[-1]
        self.assertIn("def kernel(A, B)", prompt, "前提: 猜测值确实还在这份 prompt 里")
        self.assertIn("以本节为准", prompt)


# ─── 4. 装配契约: 四个模板与链路首环 ─────────────────────────

class TestAssemblyContract(unittest.TestCase):

    def test_every_generation_template_renders_the_slot(self):
        """#67 那条守卫按调用点取并集 ⇒ 「只在一个模板里放占位符」在它眼里是绿的,
        而另外几条 DSL 路径上这个值就被静默丢掉了。逐个模板钉一遍。"""
        for name in GENERATION_TEMPLATES:
            with io.open(PROMPTS / (name + ".md"), encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("{{calling_convention}}", text,
                          f"{name}.md 不渲染契约 ⇒ 走这条 DSL 时值算了没人读")

    def test_conductor_hands_test_inputs_to_the_coder(self):
        """链路首环: conductor 调 Coder 时必须带 test_inputs（含统一记录包装器）。
        结构断言 (AST 上的调用点) 而不是跑一次 conductor —— conductor 的其余部分要 SSH。"""
        src = (PKG / "conductor.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        funcs = [n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        calls = []
        for node in ast.walk(tree):
            direct = (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "run"
                    and isinstance(node.func.value, ast.Attribute)
                    and node.func.value.attr == "coder")
            recorded = (isinstance(node, ast.Call)
                        and ast.unparse(node.func) == 'self._call_generator'
                        and [ast.unparse(arg) for arg in node.args[:2]] ==
                            ['self.coder', 'self.coder.run'])
            if direct or recorded:
                owner = next((f for f in funcs
                              if any(a is node for a in ast.walk(f))), None)
                calls.append((owner.name if owner else "<module>",
                              {kw.arg for kw in node.keywords}))
        self.assertEqual(1, len(calls), f"conductor 里的 coder.run 调用点数变了: {calls}")
        fname, kwargs = calls[0]
        self.assertIn("test_inputs", kwargs,
                      f"conductor.py 的 {fname}() 没把 test_inputs 交给 coder ⇒ "
                      f"广播在生产路径上恒为「没有拿到输入名」")


if __name__ == "__main__":
    unittest.main(verbosity=2)
