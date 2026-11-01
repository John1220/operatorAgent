"""「本级判不出」从归因接到判决 (#94 第 2 片)

外部一手 (arXiv:2609.33074v1, 行 143-148, KernelZero 的 Numerical Stability 那一段):
一个自己会产 NaN/Inf 的实现**根本不进池子**, 于是它的失败不会被算成候选的失败。第 1 片把
这件事落成了 L2 的**归因** —— 采样窗口与全量比对都显式算「比得上几个元素」, 一个都没有时
那句 message 写「本级判不出, 这不是候选错; 该查 reference_source / 内建参考」。

第 1 片之后归因是对的, 判决一侧原样不变, 于是那句话在下游被当成一条候选失败处理了三件事:
  ① Fixer 拿到的是 `_L2_RETREAT` —— 「correctness 优先于性能, 撤销可疑的激进改动, 回到最
     保守的正确写法」。这一轮没有不一致可修, 照它改就是把唯一一份还没被否证过的实现换掉。
  ② 归因前言 `_ATTRIBUTION` 断言「下面每一条都是…实测到的现象」, 而这一轮的现象是**没有
     发生任何一次判定**; 处方兜底那句 («只依据上面的执行侧原话定位根因») 更直接 —— 它要求
     模型在自己代码里找一个并不存在的根因。
  ③ 两条写库路径 (`_record_failure` 的兜底与循环内的 Reflexion) 把这份指向错误对象的教训
     沉淀下来, 而同任务的后续轮次会把它当「同类错误的修复经验」检索回 prompt。

那三件事各有一个读者, 所以本文件的用例也分四层: 分类器认得出 (词表与门文案是两份字面量)
→ 取向分得到 (分流条件) → 错误段不再给处方 → 语料不写 + 挡掉的量看得见。另有两条结构守卫
钉住「两个写库点问的是同一个分类器」与「反思那次调用真在那道守卫后面」。

变异标定 (每条只改一处, 跑完立刻逐字节复原并断言 sha256 一致。2026-09-30 实跑,
**行为分母 12/12 + 结构分母 2/2 全拦下, 0 逃逸**):
    M1  词表里那条 marker 删掉                    → 8 条红
    M2  门改措辞 (词表与文案脱钩)                  → 11 条红
    M3  分流条件恒假                              → 2 条红
    M4  分流条件恒真 (一切失败都算判不出)           → 16 条红
    M5  第四路整条不存在                           → 2 条红
    M6  hold 轮仍然印归因前言                      → 2 条红
    M7  hold 轮仍然好处方                          → 1 条红
    M8  无 hold 的轮次也印 hold 注解               → 1 条红
    M9  `_record_failure` 那道门恒假               → 2 条红
    M10 循环里那次 Reflexion 不再被挡              → 2 条红
    M11 结尾那行账把轮数写死成 0                   → 1 条红
    M12 排除集合退回只挡 infra                     → 1 条红
    M13 [结构] 只改桶名的值 (改名)                  → 1 条红
    M14 [结构] 写库侧换成自己正则看文案             → 1 条红

⚠️ 这张表**第一版有两条假绿, 都是同一个形状** (CHECKLOG I32 的第 3、4 次):
  · `assertIn("参考侧判不出", log)` 是永真断言 —— 结尾那行账**每次都印**这四个字, 印的还是
    「0 轮」。第一次跑反向对照 (`assertNotIn` 那半) 就红了, 才看见它俩其实在抢同一个串。
    修法是把断言打在**紧邻的轮数**上 (`参考侧判不出 1 轮` / `0 轮`), M11 因此才有牙齿。
  · M6/M7 与 M12/M13 各自的红名单**逐条相同** —— 标定报的是用例名, 两条断言挤在同一个方法
    里时看不出是谁杀的。拆成两个方法之后 M6 变 2 条红、M7 仍是 1 条, 归属才分得开。

跑: python3 -B -m benchmarks.calibration.mutation_calibrate --spec benchmarks/mut_specs/reference_uninformative_route.py

⚠️ 两件不在这一片的事写在 doc 58 §4.2, 这里不假装做全了:
  · 控制流仍然照常调 Fixer —— 「省掉这一轮的 LLM 调用」要带 A/B 才能改, 另片;
  · 远程那一路 (`remote_scripts/run_kernel.py`) 还没有第 1 片那个掩码, 所以它今天产不出
    「本级判不出」这句话, 这条分流在远程路径上是**空转的** (不是失效, 是无输入)。

运行: python3 -m pytest tests/unit/test_reference_uninformative_route.py -q
"""
from __future__ import annotations

import ast
import contextlib
import io
import os
import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from operator_agent.generate.agent_runtime import ENVELOPE_ABSENT, ActionEnvelope  # noqa: E402
from operator_agent.app.conductor import Conductor  # noqa: E402
from operator_agent.generate.fixer import (  # noqa: E402
    FixOutcome, FixerAgent, _ATTRIBUTION, _HOLD_NOTE, _L1_INPLACE, _L1_INTERFACE,
    _L2_RETREAT, _REFERENCE_HOLD, failure_class, stage_directive,
)
from operator_agent.core.kernel_db import (  # noqa: E402
    FAILURE_CLASS_INTERFACE, FAILURE_CLASS_REFERENCE, FailureDatabase, FailureRecord,
    KernelDatabase, failure_class_of, failure_signature, is_promotable_failure,
)
from operator_agent.execute.mock_gpu import MockGPU  # noqa: E402
from operator_agent.core.schemas import (  # noqa: E402
    ConstraintSpec, Precision, SourceSpec, TargetSpec, TaskComplexity, TaskSchema,
    TaskType, VerificationLevel, VerificationResult,
)
from operator_agent.decide.verifier import Verifier  # noqa: E402
from tests.unit import test_reference_nonfinite as refn  # noqa: E402
from tests.unit import test_verified_parent_retreat as vpr  # noqa: E402

L1 = VerificationLevel.L1_COMPILE
L2 = VerificationLevel.L2_FUNCTION
L3 = VerificationLevel.L3_PERFORMANCE

#: 一条**真的**超容差失败: 反向对照用它, 用来证明那些「不许出现」不是永真断言
_NUM_MESSAGE = ("功能验证失败: 60/240 个可比元素同时超 atol=1.0e-04 与 rtol=1.0e-04, "
                "最大误差=3.1e-02")

BROKEN = "import numpy as np\n\ndef kernel(A, B:\n    return A @ B\n"
FIXED = "import numpy as np\n\ndef kernel(A, B):\n    return A @ B\n"


def _real_hold_result():
    """**由产这句话的那道门自己**造一条「本级判不出」的结果

    抄一份 message 字面量当夹具的话, 门改了措辞这个文件还会全绿, 而分流已经哑了 ——
    那正是 `test_fixer_stage_branch.py` 的 M19 用 7 条红换来的教训。
    """
    return refn._l2(Verifier(MockGPU()), refn.ZEROS,
                    np.full((16, 16), np.nan, np.float32))


def _res(level, msg):
    return VerificationResult(level=level, passed=False, message=msg, error_log=msg)


def _task(min_speedup=0.0):
    return TaskSchema(task_id="t-ref-hold", task_type=TaskType.OPERATOR_GENERATION,
                      source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.LOW),
                      target=TargetSpec(hardware="mock_gpu", dsl="python"),
                      constraints=ConstraintSpec(precision=Precision.FP32,
                                                 min_speedup=min_speedup, max_error=1e-4))


class _LLM:
    def __init__(self):
        self.prompts: list[str] = []

    def chat(self, messages, **kw):
        self.prompts.append(messages[-1]["content"])
        return f"```python\n{FIXED}```"

    def chat_json(self, messages, **kw):
        self.prompts.append(messages[-1]["content"])
        return {}


def _prompt_for(results):
    llm = _LLM()
    FixerAgent(llm=llm).run(BROKEN, _task(), results)
    return llm.prompts[0]


class TestTheClassifierCarriesTheClass(unittest.TestCase):
    """第 1 层: 词表认得出那道门产的那句话 (两份字面量, 改一处就红)"""

    def setUp(self):
        self.res = _real_hold_result()

    def test_the_gate_still_owns_the_producer_side(self):
        """夹具前提: 这道门今天确实还在产那句话, 否则下面全部用例都在空转"""
        self.assertFalse(self.res.passed)
        self.assertEqual(L2, self.res.level)
        self.assertIn("本级判不出", self.res.message)

    def test_the_real_message_classifies_as_reference(self):
        self.assertEqual(FAILURE_CLASS_REFERENCE, failure_class([self.res]),
                         f"门原文: {self.res.message}")
        self.assertEqual(FAILURE_CLASS_REFERENCE,
                         failure_class_of(self.res.level.value, self.res.message))

    def test_the_marker_has_exactly_the_files_we_expect(self):
        """「本级判不出」这个串在 `operator_agent/` 里出现在哪几个文件, 必须是登记过的四个

        四种角色各有各的理由: `decide/verifier.py` 产它 (两处 L2 分支), `core/kernel_db.py` 认它
        (词表里那条 marker), `app/conductor.py` 与 `generate/fixer.py` 在注释与日志文案里引用它。
        多一个文件 ⇒ 有人新写了一句相同措辞, 不相干的失败会被领进这条分流;
        少一个 (只剩词表自己) ⇒ 门改了措辞, 词表与判决一起哑掉。
        两边都该红, 所以数**文件集合**而不是数出现次数。
        """
        holders = []
        pkg = _ROOT / "operator_agent"
        # 2026-10-31: 包分五层 ⇒ 递归枚举，标签用包内相对路径
        for path in sorted(pkg.rglob("*.py")):
            if "本级判不出" in path.read_text(encoding="utf-8"):
                holders.append(path.relative_to(pkg).as_posix())
        self.assertEqual(["app/conductor.py", "core/kernel_db.py", "decide/verifier.py", "generate/fixer.py"], holders,
                         "产它的只有 verifier, 认它的只有词表, 另两处是引用 —— 见本条 docstring")

    def test_a_numeric_failure_still_classifies_as_numeric(self):
        """反向对照: 这条新支路不许把所有 L2 都静音"""
        self.assertEqual("numeric_tolerance", failure_class_of("L2", _NUM_MESSAGE))
        self.assertNotEqual(FAILURE_CLASS_REFERENCE, failure_class([_res(L2, _NUM_MESSAGE)]))


class TestTheFourthOrientation(unittest.TestCase):
    """第 2 层: 取向分得到这一类, 且归类排在级别之前"""

    def test_reference_class_wins_over_the_level_route(self):
        self.assertEqual(_REFERENCE_HOLD, stage_directive({"L2"}, FAILURE_CLASS_REFERENCE))
        self.assertEqual(_REFERENCE_HOLD, stage_directive({"L1", "L2"}, FAILURE_CLASS_REFERENCE))

    def test_reachability_through_run_with_the_producer_text(self):
        prompt = _prompt_for([_real_hold_result()])
        self.assertIn("本轮不改你的核", prompt)
        self.assertNotIn("correctness 优先于性能", prompt,
                         "退让取向出现在这里说明分流没生效")

    def test_an_unknown_class_still_falls_back_to_the_level_route(self):
        """不认识的归类不许拿到第四路 (含 `unknown` 与真异常名), 取向永不为空"""
        self.assertEqual(_L1_INPLACE, stage_directive({"L1"}, "TypeError"))
        self.assertEqual(_L2_RETREAT, stage_directive({"L2"}, "TypeError"))
        self.assertEqual(_L2_RETREAT, stage_directive({"L2"}, ""))
        self.assertTrue(stage_directive(set()).strip(), "无级别时取向也不能是空串")

    def test_the_interface_route_is_not_shadowed(self):
        """第四路插在第三条之前, 不许把第三路挤掉: 两条支路各认各的类名"""
        self.assertEqual(_L1_INTERFACE, stage_directive({"L1"}, FAILURE_CLASS_INTERFACE))


class TestTheErrorSectionStopsPrescribing(unittest.TestCase):
    """第 3 层: 归因前言与处方都不给, 但错误段不能空着"""

    def test_no_attribution_for_a_hold_round(self):
        prompt = _prompt_for([_real_hold_result()])
        self.assertNotIn("实测到的现象", prompt,
                         "归因前言断言「每一条都是实测到的现象」, 而这一轮没有发生判定")
        # 门原文本身必须还在 —— 抑制的是我们的解读, 不是证据
        self.assertIn("本级判不出", prompt)

    def test_no_prescription_for_a_hold_round(self):
        """与上面那条分开放: 变异标定报的是**红名单里的用例名**, 两条断言挤在同一个方法里时
        M6 与 M7 的红看起来是同一条 (与 #98 那两张「逐条相同」的红名单同一个问题)
        """
        prompt = _prompt_for([_real_hold_result()])
        self.assertNotIn("该改哪里", prompt)
        self.assertIn("本轮不给处方", prompt)

    def test_a_numeric_round_still_gets_both(self):
        """反向对照: 上面那三条不许是「对所有输入都成立」的永真断言"""
        prompt = _prompt_for([_res(L2, _NUM_MESSAGE)])
        self.assertIn("实测到的现象", prompt)
        self.assertIn("该改哪里", prompt)
        self.assertNotIn(_HOLD_NOTE.strip(), prompt)
        self.assertNotIn("本轮不改你的核", prompt)

    def test_the_reflexion_side_is_not_touched(self):
        """`prescriptive=False` 那条路 (Reflexion 回看) 本来就不给处方, 也不许被塞进 hold 注解"""
        text = FixerAgent(llm=_LLM())._format_failures([_real_hold_result()],
                                                       prescriptive=False)
        self.assertIn("本级判不出", text)
        self.assertNotIn("本轮不给处方", text)
        self.assertNotIn("实测到的现象", text)

    def test_the_empty_case_is_unchanged(self):
        """没有失败结果时那句归因前言照旧 —— 这条防的是「hold 判断写在了 return 之前」"""
        f = FixerAgent(llm=_LLM())
        self.assertIn(_ATTRIBUTION, f._format_failures([]))


class TestTheCorpusIsNotWritten(unittest.TestCase):
    """第 4 层: 两条写库路径都不写, 而挡掉的量在结尾那行看得见"""

    def _rec(self, level: str, msg: str) -> FailureRecord:
        return FailureRecord(failure_id=f"f-{level}", kernel_type="matmul",
                             hardware="mock_gpu", dsl="python", level=level,
                             error_summary=msg, signature=failure_signature(level, msg))

    def test_the_bucket_name_is_pinned_in_the_signature(self):
        """整串钉成期望值, 与 `test_failure_memory.py:113` 对 `interface_mismatch` 的写法一致

        桶名会被写进已入库记录 id (`cx-{task_id}:{signature}`), 所以「只改常量的值」不是一次
        自由重构 —— 它让历史记录与新归类分家 (M13 的证人)。M13 与 M12 各红各的方法, 标定
        报出来的红名单才分得开是谁杀的。
        """
        msg = _real_hold_result().message
        self.assertEqual("L2|reference_uninformative|-",
                         failure_signature("L2", msg), msg)

    def test_promotability_excludes_the_reference_class(self):
        rec = self._rec("L2", _real_hold_result().message)
        self.assertEqual(FAILURE_CLASS_REFERENCE, rec.signature.split("|")[1],
                         "排除应当来自新的那一类, 而不是数值容差那条老规则顺手挡掉")
        self.assertFalse(is_promotable_failure(rec),
                         "重放回来的历史坏参考记录仍然不许注入 prompt")

    def test_an_interface_record_is_still_promotable(self):
        """逐键判定不连坐: 新类进 `_NOT_KERNEL_CLASSES` 不许顺手挡掉别的类"""
        self.assertTrue(is_promotable_failure(
            self._rec("L1", "kernel() 签名与之对不上 —— 真跑起来必然抛 TypeError")))

    def _conductor(self):
        db = FailureDatabase(jsonl_path=None)
        c = Conductor(backend=MockGPU(), kernel_db=KernelDatabase(jsonl_path=None),
                      llm=None, max_optimize_rounds=3)
        c.failure_db = db
        return c, db

    def test_record_failure_skips_the_write_and_says_so(self):
        c, db = self._conductor()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            c._record_failure(_task(), [_real_hold_result()], "fake fix", fixed=False)
        self.assertEqual({}, db._records, "「本级判不出」不该进负样本库")
        self.assertIn("未入负样本库", buf.getvalue())
        self.assertEqual(1, c._reference_hold_rounds)

    def test_record_failure_still_writes_a_real_candidate_failure(self):
        c, db = self._conductor()
        with contextlib.redirect_stdout(io.StringIO()):
            c._record_failure(_task(), [_res(L2, _NUM_MESSAGE)], "fix", fixed=False)
        self.assertEqual(1, len(db._records))
        self.assertEqual(0, c._reference_hold_rounds)

    def test_the_hold_round_in_the_loop_writes_nothing(self):
        """整条环路: 坏参考那一轮既不产 Reflexion, 也不由兜底入库, 结尾那行报出轮数"""
        fixer, log, db = _run_hold_loop([_real_hold_result()])
        self.assertTrue(fixer.bases,
                        "Fixer 没被调用 ⇒ 控制流已经改了, 这条用例在空转 (见文件头那条限制)")
        self.assertEqual(0, fixer.reflexion_calls,
                         "判不出那一轮不该让模型编一份候选侧根因再去沉淀")
        self.assertNotIn("[Reflexion] 已入库", log)
        self.assertEqual({}, db._records)
        self.assertIn("不沉淀教训、不进负样本库", log)
        # 打在**紧邻的轮数**上, 不是打在「参考侧判不出」这个短语上 —— 结尾那行账每次都印它,
        # 印的还是 0 轮。断言只查短语时它对任何输入都成立 (CHECKLOG I32 的第三次, 这次是
        # 下面那条反向对照替我撞出来的: 它原本写着 assertNotIn, 于是第一跑就红)。
        self.assertIn("参考侧判不出 1 轮", log)

    def test_a_numeric_round_in_the_loop_still_writes(self):
        """反向对照: 上面那三条的「0 次 / 空库」不能是因为环路根本没走到写库那一步"""
        fixer, log, db = _run_hold_loop([_res(L2, _NUM_MESSAGE)])
        self.assertTrue(fixer.bases)
        self.assertGreaterEqual(fixer.reflexion_calls, 1)
        self.assertEqual(1, len(db._records), f"日志: {log[-400:]}")
        self.assertNotIn("不沉淀教训、不进负样本库", log)
        self.assertIn("参考侧判不出 0 轮", log)


class TestTheWiringIsStructural(unittest.TestCase):
    """结构分母: 两个写库点都必须问同一个分类器, 而不是各自看一遍文案"""

    SRC = _ROOT / "operator_agent" / "app" / "conductor.py"

    def _func_src(self, name: str) -> str:
        text = self.SRC.read_text(encoding="utf-8")
        tree = ast.parse(text)
        fn = next((n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == name), None)
        self.assertIsNotNone(fn, f"conductor 里已没有 `{name}` ⇒ 本条守卫的前提已变")
        return ast.get_source_segment(text, fn)

    def test_record_failure_asks_the_shared_classifier(self):
        src = self._func_src("_record_failure")
        called = {n.func.id for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertIn("failure_class", called,
                      "入库前那道判断不许自己正则看文案 —— 要与 Fixer 用同一个桶")
        self.assertIn("FAILURE_CLASS_REFERENCE", src)

    def test_the_loop_guard_sits_before_the_reflexion_call(self):
        """判据是**顺序**而不是存在: 要的正是 `produce_reflexion()` 被那道守卫包住"""
        text = self.SRC.read_text(encoding="utf-8")
        hold = text.index("reference_hold = failure_class(")
        gate = text.index("if not reference_hold:")
        call = text.index("self.fixer.produce_reflexion()")
        self.assertLess(hold, gate, "标志必须在用之前算出来")
        self.assertLess(gate, call, "反思那次调用不在守卫之后")
        self.assertLess(call - gate, 200, "那次调用离守卫太远, 大概不在它的块里")


class _CountingFixer:
    """假 Fixer: 记被调用了几次, 并在被问反思时**真的**返回一条记录

    返回非 None 是必要的 —— 否则「有没有被问反思」与「反思有没有入库」分不开,
    那句 `failure_db.add(reflexion)` 在假件返回 None 时永远走不到。
    """

    def __init__(self):
        self.bases: list[str] = []
        self.reflexion_calls = 0

    def run_with_action(self, code, task, failed_results, **kw):
        """生产入口 (§21.1 生产者侧接线后 Conductor 走的是带信封的那一个)。

        ⚠️ 只有 `run` 的假件会让「被调用了几次」恒为 0 —— 那时本组的断言会在
        「控制流已经改了」与「例在空转」之间说不清 (见文件头那条限制)。信封给
        `absent`: 这一组只观测反思有没有被问。
        """
        self.bases.append(code)
        return FixOutcome(code=FIXED, summary="fake fix",
                          envelope=ActionEnvelope(status=ENVELOPE_ABSENT,
                                                  reason="假件不交信封"))

    def produce_reflexion(self):
        self.reflexion_calls += 1
        return FailureRecord(failure_id="f-reflexion", kernel_type="matmul",
                             hardware="mock_gpu", dsl="python", level="L2",
                             error_summary="fake", fix_summary="fake", lessons=["x"])


class _ScriptedVerifier:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def verify(self, **kwargs):
        r = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return r


def _run_hold_loop(failing_results, rounds=3):
    """跑真实主循环, 但 Verifier/Coder/Fixer 是假件 ⇒ 观测面只剩「这一轮沉淀了什么」

    第 1 轮发放「L1/L2 都过、只输在 L3」—— 那才是让循环继续的那一档 (min_speedup=99
    是 vpr 那份驱动现场撞出来的默认值: 抬 `target_gflops` 没用, L3 读的是加速比门槛)。
    之后每一轮都发放传进来的那份失败。
    """
    c = Conductor(backend=MockGPU(), kernel_db=KernelDatabase(jsonl_path=None),
                  llm=vpr._DesignerOnlyLLM(), max_optimize_rounds=rounds)
    db = FailureDatabase(jsonl_path=None)
    c.failure_db = db
    c.coder = vpr._QueueCoder([FIXED])
    fixer = _CountingFixer()
    c.fixer = fixer
    c.verifier = _ScriptedVerifier([[vpr._r(L1, True), vpr._r(L2, True), vpr._r(L3, False)],
                                   failing_results])
    inputs = {"A": np.ones((8, 8), dtype=np.float32), "B": np.ones((8, 8), dtype=np.float32)}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        c.execute(_task(99.0), inputs, target_gflops=0.0)
    return fixer, buf.getvalue(), db


if __name__ == "__main__":
    unittest.main()
