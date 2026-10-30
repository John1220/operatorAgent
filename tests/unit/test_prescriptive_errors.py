"""C1 处方式错误反馈 + 外部归因的守卫 (docs/research/20 第三节 C1; 依据 SPARC arXiv:2607.09803)

三类守卫, 对应这次吸纳的三个可失败点:
1. **处方表没有死条目**: 每条规则都要被某个真实产出方的文本命中, 且命中的是它自己
   (规则表按特异性排序, 顺序错了会被前面的宽规则吃掉)
2. **两种模式不能混**: 修复 prompt 带归因段与处方; Reflexion prompt 不带 —— 那里只是
   回看失败, 带上处方会让沉淀的教训变成抄模板, 而且 failure_id 的哈希取的是
   `_format_failures` 输出的前 200 字符, 开头换成固定归因段会让不同错误撞进同一签名
3. **`syntax_error_report` 必须带源码与插入符**: `str(SyntaxError)` 只有
   `invalid syntax (<f>, line N)` 一行, Fixer 拿到行号却看不到那一行长什么样
"""

import re
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from operator_agent.fixer import (
    FixerAgent, prescribe, _ATTRIBUTION, _PRESCRIPTIONS,
)
from operator_agent.verifier import Verifier, syntax_error_report, failure_pair
from operator_agent.mock_gpu import MockGPU
from operator_agent.schemas import VerificationResult, VerificationLevel

import tests.unit.test_fixer as tfx
from tests.backend_fakes import fake_backend


# ─── 1. 处方表的产出方样本 ───────────────────────────────────
#
# 预检样本全部现调 `_validate_code` 取真实 issue 原文, 不手写字符串 —— 这样规则表和
# `fixer.py::_validate_code` 的产出绑死: 改了 issue 措辞而忘了改签名, 下面的
# test_every_rule_is_reachable 就会报死规则。

def _issues(code, dsl):
    ok, issues = FixerAgent._validate_code(code, dsl)
    assert not ok, f"样本本应预检失败: dsl={dsl} {code[:40]!r}"
    return issues

TRUNC = _issues("import numpy as np\n\ndef kernel(A, B", "python")
EMPTY = _issues("", "python")
INDENT = _issues("import numpy as np\n\n\ndef kernel(A, B):\nreturn A @ B", "python")
SYNTAX = _issues("import numpy as np\n\n\ndef kernel(A, B:\n    return A @ B", "python")
NO_JIT = _issues("import triton\nimport triton.language as tl\n\n\ndef kernel(A):\n"
                 "    return tl.load(A)", "triton")
CROSS_DSL = _issues("import triton\nimport triton.language as tl\n\n@triton.jit\n"
                    "def kernel(A):\n    return tl.load(A)", "python")

# 以下来自 Verifier / anticheat / 远程执行侧的 VerificationResult 文本
ANTICHEAT = ["反作弊: 检测到禁用的原生算子回退调用 (2 处)\n"
             "  - torch.softmax() @ line 31"]
MISSING_API = ["AttributeError: module 'triton.language' has no attribute "
               "'load_cache_modifier'"]
NUMERIC = ["功能验证失败: max_abs_error=3.2e-2 超出容差 1e-3"]

PRODUCER_SAMPLES = [
    ("截断", TRUNC), ("空代码", EMPTY), ("缩进", INDENT), ("语法", SYNTAX),
    ("结构缺失", NO_JIT), ("跨硬件禁令", CROSS_DSL),
    ("反作弊", ANTICHEAT), ("缺失API", MISSING_API), ("数值", NUMERIC),
]

UNKNOWN = ["kernel 在 step 7 之后没有返回任何内容, 远程容器直接退出 code=137"]


class TestPrescriptionTableIsLive(unittest.TestCase):
    """每条处方规则都必须有真实产出方命中 —— 死规则等于没写"""

    def test_every_rule_is_reachable(self):
        winners = {}
        for label, texts in PRODUCER_SAMPLES:
            for t in texts:
                got = prescribe(t)
                if got is None:
                    continue
                idx = [i for i, r in enumerate(_PRESCRIPTIONS) if r[1] == got[0]][0]
                winners.setdefault(idx, []).append(label)
        dead = [i for i in range(len(_PRESCRIPTIONS)) if i not in winners]
        self.assertEqual(
            [], dead,
            f"处方表第 {dead} 条从未被任何真实产出方命中 (死规则): "
            f"要么删掉它, 要么补上命中它的产出方; 已命中的: {winners}")

    def test_every_entry_has_all_three_fields(self):
        for i, (pat, req, act) in enumerate(_PRESCRIPTIONS):
            self.assertTrue(pat.strip(), f"第 {i} 条缺签名")
            self.assertTrue(req.strip(), f"第 {i} 条缺「要求」")
            self.assertTrue(act.strip(), f"第 {i} 条缺「该改哪里」")
            self.assertIsNotNone(re.compile(pat, re.IGNORECASE), pat)

    def test_specific_rules_win_over_generic_syntax(self):
        """顺序守卫: 缩进/截断/空代码不能被更宽的规则吃掉"""
        for text, want in (
            ("语法错误: expected an indented block (第 4 行)", "某个 def/if/for 之后没有缩进块"),
            ("括号 ( 不配对 (疑似输出截断)", "上一版在某个结构闭合之前就被截断了"),
            ("代码为空", "上一版里没有可编译的代码"),
        ):
            fx = prescribe(text)
            self.assertIsNotNone(fx, text)
            self.assertTrue(fx[0].startswith(want),
                            f"{text!r} 被错误的规则吃掉: {fx[0]!r}")

    def test_unrecognized_returns_none(self):
        for t in UNKNOWN + ["", "正常输出"]:
            self.assertIsNone(prescribe(t), t)


# ─── 2. _format_failures 的两种模式 ──────────────────────────

def _vr(level=VerificationLevel.L2_FUNCTION, message="失败", log=""):
    return VerificationResult(level=level, passed=False, message=message, error_log=log)


class TestFormatFailuresPrescriptive(unittest.TestCase):

    def setUp(self):
        self.fixer = FixerAgent(llm=None)

    def test_header_and_prescription_present(self):
        out = self.fixer._format_failures([_vr(message="功能验证失败", log=NUMERIC[0])])
        self.assertTrue(out.startswith(_ATTRIBUTION))
        self.assertIn("Wait", out)
        self.assertIn("要求:", out)
        self.assertIn("该改哪里:", out)
        self.assertIn("执行侧原话:", out)

    def test_attribution_blames_the_toolchain_not_the_model(self):
        """这条是**措辞锁**而不是行为锁: 外部归因本身就是吸纳来的干预, 改措辞=改干预.

        SPARC 的结论是"归因到外部来源时模型能修, 归因到它自己的输出时修不了", 所以
        这里锁三个成分: 外部来源被点名 (编译器/GPU/检查器) + 明确与上一版作者脱钩。
        """
        for noun in ("编译器", "GPU", "数值检查器"):
            self.assertIn(noun, _ATTRIBUTION, f"归因段没有点名外部来源 {noun}")
        self.assertIn("与上一版是谁写的无关", _ATTRIBUTION)
        for blame in ("你上一版写错", "你写的代码", "是你的问题"):
            self.assertNotIn(blame, _ATTRIBUTION)

    def test_unknown_gets_explicit_no_prescription(self):
        out = self.fixer._format_failures([_vr(message=UNKNOWN[0], log=UNKNOWN[0])])
        self.assertIn("无现成处方", out)
        # 没有处方时不许编造建议
        self.assertNotIn("   该改哪里:", out)

    def test_signature_in_truncated_middle_still_matched(self):
        """签名落在 _truncate_error 省略掉的中间段时, 匹配仍要用全文"""
        log = ("编译输出开头\n" + "a" * 900
               + "\ntorch.matmul() 验证执行期间禁止调用原生算子\n" + "b" * 3000)
        out = self.fixer._format_failures([_vr(message="L1 失败", log=log)])
        self.assertIn("候选核里出现了被屏蔽的高层算子", out)
        self.assertIn("省略", out)
        self.assertNotIn("禁止调用原生算子\nbbb", out)  # 注入的日志本身仍被截断

    def test_reflexion_mode_has_no_attribution_or_prescription(self):
        cases = [_vr(message=NUMERIC[0], log="traceback"),
                 _vr(level=VerificationLevel.L1_COMPILE, message="编译失败",
                     log="语法错误: unexpected EOF")]
        out = self.fixer._format_failures(cases, prescriptive=False)
        self.assertNotIn("Wait", out)
        self.assertNotIn("要求:", out)
        self.assertNotIn("该改哪里", out)
        self.assertIn("错误日志:", out)

    def test_reflexion_prefix_still_distinguishes_different_errors(self):
        """failure_id 哈希吃前 200 字符: 非处方模式下不同错误前缀必须不同"""
        a = self.fixer._format_failures([_vr(message="A 类失败: 括号不配对")],
                                        prescriptive=False)[:200]
        b = self.fixer._format_failures([_vr(message="B 类失败: 数值超容差")],
                                        prescriptive=False)[:200]
        self.assertNotEqual(a, b)

    def test_empty_result_list(self):
        self.assertEqual(self.fixer._format_failures([]), _ATTRIBUTION)
        self.assertEqual(self.fixer._format_failures([], prescriptive=False),
                         "无详细错误信息")


class TestPrecheckRetryFeedbackIsPrescriptive(unittest.TestCase):
    """预检重试路径也必须带处方, 否则表里"截断/空/结构/跨硬件"四条永远没人读"""

    def test_second_prompt_carries_prescription(self):
        calls = []

        def handler(prompt):
            calls.append(prompt)
            return tfx.BROKEN_KERNEL if len(calls) == 1 else tfx.FENCED_VALID_KERNEL

        llm = tfx.MockLLM(chat_handler=handler)
        fixer = FixerAgent(llm=llm)
        fixed, _ = fixer.run(tfx.BROKEN_KERNEL, tfx._make_task(),
                             [_vr(level=VerificationLevel.L1_COMPILE,
                                  message="编译失败", log="invalid syntax")])
        self.assertEqual(fixed, tfx.VALID_KERNEL.strip())
        self.assertGreaterEqual(len(calls), 2)
        self.assertNotIn("上次修复未通过预检", calls[0])
        # 必须断言在反馈段**之内**: base prompt 里本来就带处方, 整段 prompt 断言会被
        # 它满足 (变异测试 M2 就是这么漏过去的)
        _, marker, tail = calls[1].partition("上次修复未通过预检")
        self.assertTrue(marker, "第二次 prompt 没有预检反馈段")
        self.assertIn("疑似输出截断", tail)
        self.assertIn("该改哪里:", tail)
        self.assertIn("结构闭合之前就被截断", tail)   # 命中的是截断那条, 不是别的处方
        # 与 fixer_repair.md 的输出要求一致: 允许一行根因, 两处措辞不能互相矛盾
        self.assertIn("根因:", calls[1])


# ─── 3. syntax_error_report ─────────────────────────────────

class TestRootCausePreambleStillExtractable(unittest.TestCase):
    """输出要求新允许的那行 `根因:` 不许把代码提取搞坏

    `fixer_repair.md` 第 5 条现在是「代码块之前只允许一行 `根因:`」, 而 `Wait` 标记
    (来自 SPARC 那条 89.3%) 只在模型真的复述原因时才起作用 —— 所以提取器必须扛得住
    带前言的回复, 否则这条吸纳会在自己制造的格式上翻车。
    """

    def test_fenced_reply_with_preamble(self):
        raw = "根因: 上一版在 `def kernel(A, B:` 处截断, 括号没闭合。\n" \
              "```python\n" + tfx.VALID_KERNEL + "```\n希望有帮助"
        out = FixerAgent._extract_code(raw)
        self.assertEqual(tfx.VALID_KERNEL.strip(), out)
        compile(out, "<t>", "exec")

    def test_unfenced_reply_with_preamble(self):
        raw = "根因: 累加器用了 fp16。\n" + tfx.VALID_KERNEL
        out = FixerAgent._extract_code(raw)
        self.assertTrue(out.startswith("import numpy"), out[:40])
        compile(out, "<t>", "exec")


class TestSyntaxErrorReport(unittest.TestCase):

    def _err(self, src):
        try:
            compile(src, "<l1_precheck>", "exec")
        except SyntaxError as e:
            return e
        self.fail("样本应当是 SyntaxError")

    def test_shows_source_line_and_caret(self):
        src = "import numpy as np\n\n\ndef kernel(A, B:\n    return A @ B\n"
        e = self._err(src)
        report = syntax_error_report(src, e)
        self.assertIn("def kernel(A, B:", report)
        self.assertIn("  >>>", report)
        self.assertIn("^", report)
        # 报的是哪一行随解释器版本变 (3.9 指向 return 行, 3.10+ 指向未闭合的 def 行),
        # 所以断言"标注的行 == e.lineno", 而不是钉死一个行号
        self.assertIn(f"位置: 第 {e.lineno} 行", report)
        marked = [ln for ln in report.splitlines() if ln.lstrip().startswith(">>>")]
        self.assertEqual(1, len(marked))
        self.assertIn(f"{e.lineno} |", marked[0])

    def test_beyond_str_of_syntaxerror(self):
        src = "def kernel(A, B:\n    pass\n"
        e = self._err(src)
        self.assertNotIn("def kernel", str(e))   # 先确认 str(e) 确实没有源码
        report = syntax_error_report(src, e)
        self.assertNotEqual(report, str(e))
        self.assertIn("编译器看到的源码", report)

    def test_end_lineno_span_note(self):
        e = SyntaxError("was never closed", ("<f>", 2, 9, "x = (1,"))
        e.end_lineno = 40
        out = syntax_error_report("x = 1\nx = (1,\ny = 2\nz = 3\n", e)
        self.assertIn("一直延伸到第 40 行", out)
        self.assertIn("少了闭合符号", out)

    def test_missing_end_lineno_does_not_crash(self):
        """没有 end_lineno 的错误对象 (老解释器/手搓) 必须走 getattr 而不是属性直取"""
        err = types.SimpleNamespace(msg="invalid syntax", lineno=1, offset=1, text="x ==")
        self.assertFalse(hasattr(err, "end_lineno"))
        out = syntax_error_report("x ==\n", err)
        self.assertIn("invalid syntax", out)
        self.assertNotIn("延伸", out)

    def test_l1_precheck_uses_it(self):
        """端到端: Verifier L1 预检失败时 error_log 里要能看到源码窗口"""
        v = Verifier(MockGPU())
        src = "import numpy as np\n\n\ndef kernel(A, B:\n    return A @ B\n"
        r = v._verify_compile(src, "python")
        self.assertFalse(r.passed)
        self.assertIn("  >>>", r.error_log)   # 标出出错行的那一行 (不是说明文字里的 >>>)
        self.assertIn("def kernel(A, B:", r.error_log)
        self.assertIn("编译器看到的源码", r.error_log)


class TestErrorTextAppearsOnce(unittest.TestCase):
    """远程错误的全文只能出现一遍: message 留首行, 全文归 error_log

    真实形状: 远程 Triton `CompilationError` 会把整个核的源码回显进错误里 (实测一次
    就上千字符)。此前 `message=f"编译失败: {e}"` 与 `error_log=str(e)` 存的是同一份
    全文, 于是 Fixer prompt 里同一份源码回显出现两遍 —— 而 `_truncate_error` 只压得住
    error_log 那一份, message 那一份是**无上限**的。
    """

    MARKER = "<<<唯一出现在最后一行的标记>>>"

    def _long_error(self):
        return "Remote verification error: CompilationError: at 1:0:\n" + \
               ("    src line\n" * 300) + self.MARKER

    def _backend(self, **patch):
        # 声明式假件: 替换了哪个可选方法就等于声明了那项能力。旧的
        # `setattr(RemoteExecutor(mode="local"), ...)` 在能力模型下会静默失效 ——
        # 门面声明的能力派生自内层 LocalExecutor (空集), 挂上去的方法没人会调。
        return Verifier(fake_backend(**patch))

    def test_l1_remote_compile_error_not_duplicated(self):
        err = self._long_error()
        v = self._backend(compile=lambda *_: (_ for _ in ()).throw(RuntimeError(err)))
        r = v._verify_compile("def kernel(A):\n    return A\n", "triton")
        self.assertFalse(r.passed)
        self.assertEqual(1, len(r.message.splitlines()), "message 必须压成单行现象")
        out = FixerAgent(llm=None)._format_failures([r])
        self.assertEqual(1, out.count(self.MARKER), "全文在 prompt 里出现了不止一遍")

    def test_l2_remote_execution_error_not_duplicated(self):
        err = self._long_error()
        v = self._backend(verify_function=lambda *a, **kw: {
            "success": False, "error": err})
        r = v._verify_function("def kernel(A):\n    return A\n",
                              {"A": np.zeros((4, 4), dtype=np.float32)},
                              np.zeros((4, 4), dtype=np.float32), tfx._make_task())
        self.assertFalse(r.passed)
        self.assertEqual(1, len(r.message.splitlines()))
        out = FixerAgent(llm=None)._format_failures([r])
        self.assertEqual(1, out.count(self.MARKER))

    def test_failure_pair_shape(self):
        head, full = failure_pair("第一行\n第二行")
        self.assertEqual("第一行", head)
        self.assertEqual("第一行\n第二行", full)
        self.assertEqual(("无错误详情", ""), failure_pair(""))
        self.assertEqual(("无错误详情", ""), failure_pair(None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
