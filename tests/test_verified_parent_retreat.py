"""修订起点必须是「已验证的那一版」—— AReaL-TIK 行 214 落进 Conductor 的第一片

外部一手 (arXiv:2609.35140v1, 逐字):
    "If an offspring fails, revisions must start from a retained **verified** parent,
     not that offspring."

我方的先决事实 (改之前): `conductor.py` 里 Fixer 唯一的入口拿的是 `current_code`,
而 `current_code` 就是**刚刚失败的那一版** —— 于是「修 → 又失败 → 再修」会在一版没通过
正确性验证的代码上继续叠。CUDA-Harness 那句是同一个意思的另一半 (doc 55 §3.3):
"When optimization is built on a kernel that looks high-performance but is functionally
flawed, subsequent refinement typically remains trapped in the same faulty pattern."

这里的 **verified 取「正确性已验证」= L1 与 L2 都到场且都通过**, 不含 L3 的性能门:
一个「正确但不够快」的核正是要留在手里当起点的那份。两个细节都不是多余的:
`Verifier.verify()` 是 fail-fast, L1 挂时列表里根本没有 L2 那行 —— 数空集会得到 True,
那是把「没测过」读成「测过且通过」(doc 49 §八 那一族)。

续跑的那一半: 已验证祖先**不新开快照字段**, 而是从血缘最后一条重建 (`levels_passed`
每轮都存, 而快照的 `current_code` 就是那一轮那份)。重建不出来时如实返回 None ⇒ 退回
改之前的形状, 而不是猜一个祖先。这条「续跑只能重建一步」是有意接受的限制, 记在 doc 58 §三。

变异标定 (每条只改一处, 跑完立刻逐字节复原并断言。2026-09-30 实跑, 7/7 拦住；
表还没进 `benchmarks/mut_specs/` —— 迁一次要重跑全套, 等它真被重跑时再迁):
    M1 `correctness_verified` 改成 `all(r.passed for r in results)` (数空集)  → 4 条红
    M2 「L1 通过即可」(去掉 L2 那一项)                                       → 1 条红
    M3 Fixer 入口拿回 `current_code` (即本片改之前的样子)                     → 3 条红
    M4 退回条件去掉 `verified_parent != current_code` 那半句                  → 1 条红
    M5 `verified_parent_from` 只看 L1 那一格                                 → 1 条红
    M6 退回起点时把错误证据一起丢掉 (`failed_results = []`)                    → 1 条红
    M7 主循环里不再更新 `verified_parent` (只留续跑那条重建)                    → 2 条红

⚠️ **M4 第一版是逃逸的** (全绿)。那不是用例写松了, 是我先写了一个当时观察不到的条件:
`verified_parent == current_code` 只在「同一份文本两次验证给不同答案」时才可能, 而那需要
非确定的验证。补了 `_ScriptedVerifier` (发放预写好的分级结果) 造出那个现场之后, 同一条变异
变成 KILLED —— 它的观察面是那句 print: 起点没变却报「退回」, 读日志的人会以为祖先与这一版
是两份东西。⇒ 与 CHECKLOG I19/I23 同族: **只在条件成立的那一侧取样, 测的是代码在不在,
不是条件对不对**; 写一个「今天构造不出反例」的条件时, 要么把反例造出来, 要么把它删掉。

夹具自己的一条纠正: 一开始那份「能过 L1+L2 的核」是抄 `tests/test_iteration_lineage.py` 的
`CORRECT` —— 第一次跑就红了: 那份核过 L1 但**过不了 L2** (`got 9, want 8`), 而那个文件的
用例只断言过 L1。也就是说「它是我已有的正确夹具」这件事并不构成证据。
"""
from __future__ import annotations

import ast
import contextlib
import io
import unittest
from pathlib import Path

import numpy as np

from operator_agent.agent_runtime import ENVELOPE_ABSENT, ActionEnvelope
from operator_agent.conductor import (Conductor, correctness_verified,
                                      verified_parent_from)
from operator_agent.fixer import FixOutcome
from operator_agent.kernel_db import KernelDatabase
from operator_agent.mock_gpu import MockGPU
from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec, TargetSpec,
                                    TaskComplexity, TaskSchema, TaskType,
                                    VerificationLevel, VerificationResult)

# 一份**在 MockGPU 上真能过 L1+L2** 的核 (参考实现就是矩阵乘), 带一个只在这份里出现的标记。
# ⚠️ 这里原本抄的是 `tests/test_iteration_lineage.py` 那份 `CORRECT` —— 抄过来第一次跑就红了:
# 那份核过 L1 但**过不了 L2** (`got 9, want 8`), 而那个文件的用例只断言过 L1。
# 也就是说「它是我们已有的正确夹具」这件事并不构成证据, 抄夹具也要跑一次。
ANCESTOR = '''import numpy as np

def kernel(A, B):
    # ANCESTOR-MARKER
    return A @ B
'''

# 语法就错 —— 于是这一轮**必然**卡在 L1 (L2 那行根本不会出现, 正好是 fail-fast 那一侧)
OFFSPRING = '''import numpy as np

# OFFSPRING-MARKER
def kernel(A, B):
    return (A @ B
'''

# 假 Fixer 交回的修法: 同样编不过。它存在的意义是让「坏的后代」与「下一轮的起点」
# 是两个不同的字符串 —— 否则「一直在修同一份代码」这种空转看起来和退回到祖先一模一样
BROKEN_FIX = '''import numpy as np

# FIX-MARKER
def kernel(A, B):
    return (A @ B
'''

L1 = VerificationLevel.L1_COMPILE
L2 = VerificationLevel.L2_FUNCTION
L3 = VerificationLevel.L3_PERFORMANCE


def _r(level: VerificationLevel, passed: bool) -> VerificationResult:
    return VerificationResult(level=level, passed=passed, message="", error_log=None,
                              metrics={})


def _task(min_speedup: float = 0.5) -> TaskSchema:
    # `min_speedup` 是 L3 真正读的那个门槛 (不是 `target_gflops`: 第一版我抬它,
    # 结果 L3 照样 1.67x 通过、第 1 轮就 break, Fixer 一次都没被叫到)
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.LOW),
        target=TargetSpec(hardware="mock_gpu", dsl="python"),
        constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=min_speedup,
                                   max_error=1e-3))


class TestTheVerifiedPredicate(unittest.TestCase):
    """谓词本身: 两级都要到场且都通过。缺一级不是「通过」, 是「不知道」"""

    def test_both_levels_present_and_passed(self):
        self.assertTrue(correctness_verified([_r(L1, True), _r(L2, True)]))

    def test_fail_fast_shape_is_not_verified(self):
        # L1 挂时 Verifier 直接 return ⇒ 列表里只有 L1 这一行。`all(...)` 在这种
        # 「只剩通过的那些」的列表上会给 True, 而那正是把没测过读成测过
        self.assertFalse(correctness_verified([_r(L1, False)]))
        self.assertFalse(correctness_verified([]), "空集不是「都通过」")
        self.assertFalse(correctness_verified([_r(L1, True)]),
                         "只有 L1 通过 = L2 还没跑, 不是 L2 通过")

    def test_l3_outcome_does_not_change_correctness(self):
        # 「正确但不够快」的那一版是这条规则里唯一值得留下的起点
        self.assertTrue(correctness_verified([_r(L1, True), _r(L2, True), _r(L3, False)]))


class TestResumeRebuildsTheAncestor(unittest.TestCase):
    """续跑: 从血缘最后一条重建, 重建不出来就 None, 不猜祖先"""

    def _hist(self, levels: dict) -> list:
        return [{"round": 3, "levels_passed": levels}]

    def test_last_verified_round_rebuilds_to_the_snapshotted_code(self):
        hist = self._hist({L1.value: True, L2.value: True})
        self.assertEqual(ANCESTOR, verified_parent_from(hist, ANCESTOR))

    def test_a_last_round_that_broke_returns_none(self):
        for levels in ({L1.value: True, L2.value: False},
                       {L1.value: False},              # fail-fast: L2 那一格根本没出现
                       {},
                       {L1.value: None, L2.value: True}):
            self.assertIsNone(verified_parent_from(self._hist(levels), ANCESTOR),
                              f"这份血缘最后一条并不支持「已验证」: {levels}")

    def test_empty_inputs_return_none_rather_than_the_code(self):
        self.assertIsNone(verified_parent_from([], ANCESTOR))
        self.assertIsNone(verified_parent_from(self._hist({L1.value: True}), ""))


class _QueueCoder:
    """按轮次发代码的假 Coder —— 让「第 N 轮交出哪一版」是测试写死的, 不是模型决定的"""

    def __init__(self, codes):
        self._codes = list(codes)
        self.calls = 0

    def run(self, sketch, task, optimization_recipes=None, test_inputs=None, **kw):
        code = self._codes[min(self.calls, len(self._codes) - 1)]
        self.calls += 1
        return code


class _RecordingFixer:
    def __init__(self, fixed=BROKEN_FIX):
        self.fixed = fixed
        self.bases: list[str] = []
        self.n_failed_results: list[int] = []

    def run_with_action(self, code, task, failed_results, **kw):
        """⚠️ 假件的入口必须是**生产入口** `run_with_action`: §21.1 生产者侧接线后
        Conductor 走的是带动作信封的那一个, 而只有一个 `run` 的假件会让 Fixer
        「一次都没被调用」—— 这一组用例就会在空转里全绿 (`fixer.bases == []`)。
        信封给 `absent`（与「模型没交信封」同一档）: 这组用例只观测**起点**这一件事。
        """
        self.bases.append(code)
        self.n_failed_results.append(len(failed_results))
        return FixOutcome(code=self.fixed, summary="fake fix",
                          envelope=ActionEnvelope(status=ENVELOPE_ABSENT,
                                                  reason="假件不交信封"))

    def produce_reflexion(self):
        return None


class _DesignerOnlyLLM:
    """只服务 Designer 的 chat_json; 真 LLM 调用一次都不该发生 (coder/fixer 都被换了)"""

    class _Tracker:
        total_calls = 0

        def summary(self):
            return {}

        def watermark(self):
            return 0

        def summary_since(self, n):
            return self.summary()

        def pretty(self):
            return "LLM: designer-only"

    cost_tracker = _Tracker()

    def __init__(self):
        self.chat_calls = 0
        self._sketch = {"algorithm": "matmul",
                        "tiling": {"block_m": 64, "block_n": 64, "block_k": 32},
                        "memory_plan": {}, "pipeline": [], "optimization_hints": []}

    def chat(self, messages, temperature=None, max_tokens=None, raw=False,
             caller_tag="Unknown"):
        self.chat_calls += 1
        return "```python\n" + ANCESTOR + "```"

    def chat_json(self, messages, temperature=None, max_tokens=None, caller_tag="Unknown"):
        return dict(self._sketch)


def _run_loop(codes, rounds=4, min_speedup: float = 99.0):
    """跑真实的主循环, 但把 Coder/Fixer 换成假件 ⇒ 观测面只剩「起点是谁」

    两样默认值都是现场撞出来的, 不是随手挑的:
    * `min_speedup=99.0` 让「正确但不够快」这一档真的停在 L3 (抬 `target_gflops` 没用,
      L3 读的是加速比门槛), 于是第 1 轮不会 break、循环才走到第 2 轮那份坏代码;
    * `rounds=4` 是因为最后一轮进不了 Fixer (那道 `round_num < max_optimize_rounds` 的门),
      要拿到**两次**修复调用的现场得留一轮。
    """
    fixer = _RecordingFixer()
    c = Conductor(backend=MockGPU(), kernel_db=KernelDatabase(jsonl_path=None),
                  llm=_DesignerOnlyLLM(), max_optimize_rounds=rounds)
    c.coder = _QueueCoder(codes)
    c.fixer = fixer
    inputs = {"A": np.ones((8, 8), dtype=np.float32),
              "B": np.ones((8, 8), dtype=np.float32)}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        # target_gflops 抬到不可能达到 ⇒ L1/L2 能过的那一版会停在 L3, 循环才会继续
        art = c.execute(_task(min_speedup), inputs, target_gflops=0.0)
    return fixer, art, buf.getvalue()


class TestTheLoopHandsTheAncestorToTheFixer(unittest.TestCase):
    def test_a_broken_revision_is_repaired_from_the_verified_one(self):
        """本片存在的理由: 第一轮过了 L1/L2 (只输在性能), 第二轮交出语法都错的版本
        —— 那么修的那一起点必须是第一轮那份, 不是第二轮这份坏的"""
        fixer, art, log = _run_loop([ANCESTOR, OFFSPRING, OFFSPRING, OFFSPRING])
        self.assertTrue(fixer.bases,
                        "Fixer 根本没被调用过 ⇒ 这三条环路用例在空转 (前提已变)")
        self.assertEqual(ANCESTOR, fixer.bases[0],
                         f"修的是坏的那一版本身: {fixer.bases[0][:60]!r}")
        self.assertNotIn("OFFSPRING-MARKER", fixer.bases[0])
        self.assertIn("修订起点退回", log, "退回了却没说 ⇒ 读产物的人分不清这两种形状")

    def test_the_same_text_never_claims_a_retreat(self):
        """`verified_parent != current_code` 那半句唯一的现场: 同一份文本这次没过

        起点是同一份字符串时「退回祖先」是一句谎话 (退到的就是它自己), 所以那半句不是装饰 ——
        而这一条就是它的证人: 删掉那半句时这一条会红 (M4，第一版**逃逸**，见文件头那张表)。
        """
        fixer, log = _run_loop_with_verifier(
            [ANCESTOR, ANCESTOR, ANCESTOR],
            [[_r(L1, True), _r(L2, True), _r(L3, False)], [_r(L1, False)]],
        )
        self.assertTrue(fixer.bases, "Fixer 没被叫到 ⇒ 这条在空转")
        self.assertEqual(ANCESTOR, fixer.bases[0])
        self.assertNotIn("修订起点退回", log,
                         "起点没变却报「退回」—— 读日志的人会以为祖先与这一版是两份东西")

    def test_the_error_evidence_still_comes_from_the_broken_version(self):
        """起点退回祖先, **错误信息不许跟着退回** —— 那是「上一次改动坏在哪」的唯一证据。

        这一条是 M6 的证人: 把两者一起换成祖先那份时, 失败结果列表会跟着变成祖先的
        (祖先过了 L1/L2 ⇒ 只剩 L3 那一条失败), 于是这里的计数对不上。
        """
        fixer, _, _ = _run_loop([ANCESTOR, OFFSPRING, OFFSPRING, OFFSPRING])
        self.assertEqual(1, fixer.n_failed_results[0],
                         f"应当只带「这一版」的那一条 L1 失败: {fixer.n_failed_results}")

    def test_repeated_broken_revisions_keep_starting_from_the_same_ancestor(self):
        """连坏两轮 ⇒ 两轮的起点必须是**同一份**祖先, 而不是第二轮拿第一轮的修法当起点。

        这才是那句规则挡的东西: 修环在坏的后代上叠层。
        """
        fixer, _, _ = _run_loop([ANCESTOR, OFFSPRING, OFFSPRING, OFFSPRING])
        self.assertGreaterEqual(len(fixer.bases), 2,
                                "只修了一轮 ⇒ 这条「叠层」判据没被执行到, 它在空转")
        self.assertEqual({ANCESTOR}, set(fixer.bases))

    def test_without_any_verified_ancestor_it_still_fixes_the_failing_version(self):
        """反向对照 (只在条件成立那侧取样测的是代码在不在, 不是条件对不对):
        第一轮就编不过 ⇒ 本次运行没有任何已验证版本 ⇒ 行为与改之前完全相同。"""
        fixer, _, log = _run_loop([OFFSPRING, OFFSPRING, OFFSPRING, OFFSPRING],
                                  min_speedup=0.5)
        self.assertTrue(fixer.bases, "Fixer 没被调用, 这条对照在空转")
        self.assertEqual(OFFSPRING, fixer.bases[0])
        self.assertNotIn("修订起点退回", log)


class _ScriptedVerifier:
    """按调用顺序发放**预先写好的**分级结果

    要造的现场是「同一份文本两次验证给不同答案」—— 真机上这是真的 (共享卡、容差边界、
    L2 输入由 seed 派生), 而它恰好是那条 `verified_parent != current_code` 唯一能生效的场合。
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def verify(self, **kwargs):
        r = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return r


def _run_loop_with_verifier(codes, responses, rounds=3):
    fixer = _RecordingFixer()
    c = Conductor(backend=MockGPU(), kernel_db=KernelDatabase(jsonl_path=None),
                  llm=_DesignerOnlyLLM(), max_optimize_rounds=rounds)
    c.coder = _QueueCoder(codes)
    c.fixer = fixer
    c.verifier = _ScriptedVerifier(responses)
    inputs = {"A": np.ones((8, 8), dtype=np.float32),
              "B": np.ones((8, 8), dtype=np.float32)}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        c.execute(_task(99.0), inputs, target_gflops=0.0)
    return fixer, buf.getvalue()


class TestTheGuardIsWiredInTheRightOrder(unittest.TestCase):
    """结构性分母: 行为侧全绿也拦不住「以后有人把起点改回去」, 所以钉住现场

    判据是**顺序**而不是存在 (doc 50 §7.8 那条同样的教训): 「函数里存在某个 if」随便
    一个 `if na:` 就能满足, 要的正是「那次退回赋值发生在 fixer.run 之前」。
    """

    SRC = Path("operator_agent/conductor.py")

    def _holder(self):
        """主循环那一处调用 (起点规则管的就是它)

        接线后 `self.fixer.run_with_action(...)` 有**两处**: 主循环里那次修复, 与
        `_answer_ask_and_retry` 里「响应询问后再要一次补丁」的那次。判据是「哪一处的
        **起点**是 `repair_base`」⇒ 取**同时**有 `repair_base = ...` 赋值的那一个函数,
        并要求它唯一 (两处都赋值会让人分不清哪一处是起点)。
        """
        tree = ast.parse(self.SRC.read_text(encoding="utf-8"))
        holders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            call = self._fixer_call(node)
            if call is None:
                continue
            assigns = [n for n in ast.walk(node)
                       if isinstance(n, ast.Assign)
                       and any(getattr(t, "id", None) == "repair_base" for t in n.targets)]
            if assigns:
                holders.append((node, call))
        if len(holders) != 1:
            self.fail(f"带 `repair_base` 赋值的修复调用点有 {len(holders)} 处 (要恰好 1) "
                      "—— 这条守卫的前提已变: 生产入口换名 / 多了一条不退回起点的修复路")
        return holders[0]

    @staticmethod
    def _fixer_call(fn):
        for sub in ast.walk(fn):
            f = getattr(sub, "func", None)
            if (isinstance(f, ast.Attribute) and f.attr == "run_with_action"
                    and isinstance(f.value, ast.Attribute) and f.value.attr == "fixer"):
                return sub
            if (isinstance(sub, ast.Call) and ast.unparse(sub.func) == 'self._call_generator'
                    and [ast.unparse(arg) for arg in sub.args[:2]] ==
                        ['self.fixer', 'self.fixer.run_with_action']):
                # Normalize the receipt wrapper; keep location/order and the
                # original check of the actual repair-base argument.
                call = ast.Call(func=sub.args[1], args=sub.args[2:], keywords=sub.keywords)
                return ast.copy_location(call, sub)
        return None

    def test_fixer_is_handed_the_retreat_variable_not_the_failing_code(self):
        _, call = self._holder()
        arg = call.args[0]
        self.assertIsInstance(arg, ast.Name,
                              "第一个实参不再是变量名 —— 这条守卫需要重读 (它钉的是起点)")
        self.assertEqual("repair_base", arg.id,
                         "Fixer 又直接拿 `current_code` 了 ⇒ 那片规则没了")

    def test_the_ask_retry_path_also_starts_from_the_retreat_variable(self):
        """第二个调用点 (响应询问后再要补丁) **也**必须拿 `repair_base`

        少这一条时: 那条路若直接拿 `current_code`, 于是「你问了一次 profiler」就足以让
        修复起点从已验证祖先漂回那一版坏代码 —— 而这正是这一组用例挡的漂移, 只不过
        它换了一扇门进来。
        """
        tree = ast.parse(self.SRC.read_text(encoding="utf-8"))
        retry = next((n for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef)
                      and n.name == "_answer_ask_and_retry"), None)
        self.assertIsNotNone(retry, "`_answer_ask_and_retry` 不在了 —— 重读这条守卫")
        call = self._fixer_call(retry)
        self.assertIsNotNone(call, "询问重试那条路不再调用 Fixer ⇒ 这条守卫的前提已变")
        kwargs = {k.arg for k in call.keywords}
        positional = [a for a in call.args]
        self.assertTrue(
            "repair_base" in kwargs
            or (positional and isinstance(positional[0], ast.Name)
                and positional[0].id == "repair_base"),
            f"询问重试那条路没把起点传下去 (第一个实参={positional[:1]}, 关键字={sorted(kwargs)})")

    def test_the_retreat_happens_before_the_fixer_is_called(self):
        fn, call = self._holder()
        assigns = [n.lineno for n in ast.walk(fn)
                   if isinstance(n, ast.Assign)
                   and any(getattr(t, "id", None) == "repair_base" for t in n.targets)]
        self.assertTrue(assigns, "函数里根本没有 `repair_base = ...` 那次赋值")
        self.assertLess(max(assigns), call.lineno,
                        "退回发生在 fixer.run 之后 ⇒ 它决定不了起点")


if __name__ == "__main__":
    unittest.main()
