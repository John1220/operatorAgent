"""§10.3 的三条件升级门: 「什么时候才许让 LLM 做结构性改动」

判据本体 (`operator_agent/escalation.py`) 只证算得对; 这里同时证**它接在生产链路上**:
门关 ⇒ 不问 LLM 要结构改动 (只做参数搜索), 门开 ⇒ 问, 且升级买到 K 轮。

最要紧的一条边界: 门开需要**三条件同时成立**, 而条件 2 今天只有 `smem-bound` 有量具
(`spill` / `bank conflict` / 占用率上限都还没有) —— 所以「门关着」是**预期的**默认状态,
不是接线坏了。用例把两端都钉住, 免得日后有人把「门一直关着」读成「门没接上」。

运行: python3 -m unittest tests.test_escalation -v
"""

import contextlib
import io
import os
import shutil
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operator_agent.conductor import MAX_STAGNANT_ROUNDS, Conductor
from operator_agent.escalation import (
    COND_DIRECTION, COND_HARD_LIMIT, COND_PLATEAU, EscalationInputs, decide,
    plateaued, smem_bound_evidence, untried_directions,
)
from operator_agent.kernel_db import KernelDatabase
from tests.test_measurement_validity import (
    LaunchDominatedGPU, NET_MS_LAYERNORM_SMALL, NET_MS_MATMUL_4096,
)
from tests.test_search_belief_gate import _LoopLlm, _inputs, _task


class TestThePlateauJudgement(unittest.TestCase):
    """条件 1: K 轮 valid 且 SAME/AMBIGUOUS"""

    def test_k_same_rounds_is_a_plateau(self):
        self.assertTrue(plateaued(["SAME", "SAME"], 2))
        self.assertTrue(plateaued(["SAME", "AMBIGUOUS"], 2))
        self.assertTrue(plateaued(["FAST", "SAME", "SAME"], 2), "只看尾部 K 轮")

    def test_fewer_than_k_valid_rounds_is_not_a_plateau(self):
        """轮数不够 ⇒ 「还没跑够」, 不是「平台化」——缺轮次与平台化是两件事"""
        self.assertFalse(plateaued(["SAME"], 2))
        self.assertFalse(plateaued([], 2))

    def test_an_improvement_breaks_the_plateau(self):
        self.assertFalse(plateaued(["SAME", "FAST"], 2))
        self.assertFalse(plateaued(["SAME", "SAME", "FAST"], 2))

    def test_a_regression_is_not_a_plateau(self):
        """退步是「还在动」(而且它有自己的出路: 禁用配方), 不是「调不动了」"""
        self.assertFalse(plateaued(["SAME", "SLOW"], 2))

    def test_k_must_be_positive(self):
        with self.assertRaises(ValueError):
            EscalationInputs(verdicts=("SAME",), k_valid_rounds=0)


class TestTheHardLimitEvidence(unittest.TestCase):
    """条件 2: 只认能指出出处的证据; 今天有量具的只有 smem-bound"""

    def test_a_tile_at_the_ceiling_is_evidence(self):
        ev = smem_bound_evidence(245760, 233472)
        self.assertEqual(1, len(ev))
        self.assertIn("smem-bound", ev[0])
        self.assertIn("240KiB", ev[0])
        self.assertIn("228KiB", ev[0])

    def test_a_tile_with_room_is_not_evidence(self):
        """「离上界还有 20% 余量」不构成硬限制 —— 那正是设计说的「感觉调不动了」"""
        self.assertEqual((), smem_bound_evidence(131072, 233472))

    def test_missing_meters_produce_no_evidence(self):
        """缺估算 / 缺上界 ⇒ 空证据 (fail-closed), 不是「大概超了吧」"""
        self.assertEqual((), smem_bound_evidence(None, 233472))
        self.assertEqual((), smem_bound_evidence(245760, None))
        self.assertEqual((), smem_bound_evidence(245760, 0))


class TestTheStructuralDirection(unittest.TestCase):
    """条件 3: planner 选中但没试过的配方"""

    def test_applied_and_disabled_recipes_are_not_directions(self):
        self.assertEqual(("R-A",), untried_directions(
            ["R-A", "R-B", "R-C"], ["R-B"], ["R-C"]))
        self.assertEqual(("R-A", "R-B"), untried_directions(["R-A", "R-B"], [], []))

    def test_order_follows_the_planner_and_duplicates_collapse(self):
        self.assertEqual(("R-B", "R-A"), untried_directions(
            ["R-B", "R-A", "R-B"], [], []))

    def test_nothing_selected_means_no_direction(self):
        self.assertEqual((), untried_directions([], [], []))


class TestTheGateItself(unittest.TestCase):

    def _inputs(self, **kw):
        base = dict(verdicts=("SAME", "SAME"), k_valid_rounds=2,
                    hard_limits=("smem-bound: …",), untried_directions=("R-A",))
        base.update(kw)
        return EscalationInputs(**base)

    def test_all_three_conditions_allow_the_escalation(self):
        d = decide(self._inputs())
        self.assertTrue(d.allowed)
        self.assertEqual((), d.refusals)
        self.assertEqual({COND_PLATEAU: True, COND_HARD_LIMIT: True,
                          COND_DIRECTION: True}, d.as_dict()["conditions"])

    def test_each_missing_condition_is_named_in_the_refusal(self):
        for kw, expect in (
                (dict(verdicts=("FAST", "SAME")), COND_PLATEAU),
                (dict(hard_limits=()), COND_HARD_LIMIT),
                (dict(untried_directions=()), COND_DIRECTION)):
            d = decide(self._inputs(**kw))
            self.assertFalse(d.allowed, kw)
            self.assertIn(expect, d.refusals)
            self.assertIn(expect, d.refusal_note())

    def test_no_evidence_at_all_means_closed(self):
        """三条件全缺 ⇒ 三条理由都要在 (日志与产物里要能分出缺哪条)"""
        d = decide(EscalationInputs(verdicts=(), k_valid_rounds=MAX_STAGNANT_ROUNDS))
        self.assertFalse(d.allowed)
        self.assertEqual(3, len(d.refusals))


#: 一套「贴到上界」的核: BLOCK_M=192/BN=192/BK=64, float32 ⇒
#: (192*64 + 64*192 + 192*192) * 4 = 245760 B = 240 KiB; H20 的每 SM 上界是 228 KiB
#: ⇒ est ≥ limit 且 est ≤ limit×1.2 (1.2 是 L1 预检的宽松系数) ⇒ **L1 放行且构成硬限制**。
_TIGHT_TILE_KERNEL = ("import numpy as np\n"
                      "BLOCK_M = 192\nBLOCK_N = 192\nBLOCK_K = 64\n"
                      "dtype = np.float32\n\n"
                      "def kernel(A, B):\n    return A @ B\n")


class _TightTileLlm(_LoopLlm):
    """出「贴到上界」的那份码 —— 条件 2 的证据按**被测量的代码**算"""

    def chat(self, messages, **kw):
        return _TIGHT_TILE_KERNEL


class TestTheGateInTheProductionLoop(unittest.TestCase):

    maxDiff = 600

    def setUp(self):
        """把**环境指纹**的落点隔离出来 —— 不然这条用例量的是别的用例干过什么

        实测 (整套跑时本条红、单独跑绿): 环境指纹是**跨运行**比对的, 前面的用例
        (别的后端/别的硬件名) 已经把指纹写进了共享的输出目录, 于是本轮读数被判成
        「环境漂移 ⇒ degraded」; 而升级门**只对可信读数**做判决 (判决块要求
        `measurement_ok`) ⇒ 裁决日志一条都没有。而同一类里后一条用例却绿 —— 因为
        那条之前指纹已经被本类的第一次运行对上了。把指纹落到本用例自己的目录里,
        「读数可不可信」就不再是「前一条用例是谁」的函数。
        """
        self._out = tempfile.mkdtemp(prefix="escalation_out_")
        self._env = unittest.mock.patch.dict(
            os.environ, {"OPERATOR_AGENT_OUTPUT_DIR": self._out})
        self._env.start()

    def tearDown(self):
        self._env.stop()
        shutil.rmtree(self._out, ignore_errors=True)

    def _run(self, llm, max_rounds=5):
        # 语料必须**显式**关掉: `jsonl_path=None` 不是「无库」, 而是「用默认库」
        # (`OPERATOR_AGENT_KERNEL_DB` 或 `<repo>/output/kernel_db.jsonl`) —— 在整套
        # 用例里那个库会被前面的用例写进记录, 本用例于是走 RAG 命中那条路、不再有
        # 「L3 没过 ⇒ 优化分支」这些轮, 裁决日志就空了 (实测: 单独跑绿、整套跑红)。
        c = Conductor(backend=LaunchDominatedGPU(NET_MS_MATMUL_4096),
                      kernel_db=KernelDatabase(jsonl_path="off"), llm=llm,
                      max_optimize_rounds=max_rounds, heartbeat_interval=1000)
        tags = []
        orig = llm.chat_json

        def _spy(messages, **kw):
            tags.append(kw.get("caller_tag", "?"))
            return orig(messages, **kw)

        llm.chat_json = _spy
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            art = c.execute(_task(min_speedup=5.0), _inputs(), None, target_gflops=0.0)
        return art, buf.getvalue(), tags

    def test_without_hard_limit_evidence_the_llm_is_never_asked_again(self):
        """默认假核没有 BLOCK_* ⇒ 条件 2 无量具 ⇒ 门关 ⇒ 只做参数搜索"""
        art, out, tags = self._run(_LoopLlm())
        self.assertNotIn("Search", tags, f"门关着却问了 LLM: {tags}")
        self.assertIn("结构升级门未通过", out)
        esc = art.metadata["escalation"]
        self.assertTrue(esc, "裁决必须落盘 (「没升级」与「没接门」要分得开)")
        self.assertTrue(all(not e["allowed"] for e in esc))
        self.assertTrue(all(COND_HARD_LIMIT in e["refusals"] for e in esc),
                        f"缺的应该是硬限制证据那一支: {esc[-1]}")

    def test_with_hard_limit_evidence_and_a_plateau_the_llm_is_asked(self):
        """三条件成立 ⇒ 问 LLM, 且升级买到 K 轮 (停滞计数清零 ⇒ 跑得比上限更久)"""
        art, out, tags = self._run(_TightTileLlm())
        self.assertIn("Search", tags, f"三条件都成立却没问 LLM: {tags}")
        self.assertIn("结构升级门通过", out)
        allowed = [e for e in art.metadata["escalation"] if e["allowed"]]
        self.assertTrue(allowed, f"没有一轮通过过门: {art.metadata['escalation']}")
        self.assertTrue(all(COND_DIRECTION in e["conditions"] for e in allowed))
        # 买到 K 轮的可观测后果: 循环没有在第 K 轮就因停滞结束
        rounds = len(art.iteration_history)
        self.assertGreater(rounds, MAX_STAGNANT_ROUNDS + 1,
                           f"升级没有换来额外轮次 (轮数={rounds})")
        self.assertNotEqual("stagnant", art.metadata["stop"]["reason"])

    def test_an_untrustworthy_round_still_denies_the_escalation(self):
        """读数不可信的轮次, 门必须**关着** —— 没有判决就没有平台化证据 (fail-closed)

        这一格是**缺省裁决**的读者: 判决块 (要求 `measurement_ok`) 整段跳过时, 裁决不会
        被重算, 而优化分支照样要发布一份 —— 「不问 LLM」必须由缺省给出, 不能拿 `None`
        交给下游各自解释 (那必然有人解释成放行)。
        """
        c = Conductor(backend=LaunchDominatedGPU(NET_MS_LAYERNORM_SMALL),
                      kernel_db=KernelDatabase(jsonl_path="off"), llm=_LoopLlm(),
                      max_optimize_rounds=3, heartbeat_interval=1000)
        with contextlib.redirect_stdout(io.StringIO()):
            art = c.execute(_task(min_speedup=5.0), _inputs(), None, target_gflops=0.0)
        esc = art.metadata["escalation"]
        self.assertTrue(esc, "优化分支发布过裁决, 产物里就该有——"
                             f"stop={art.metadata['stop']} "
                             f"actions={[r.get('action') for r in art.iteration_history]}")
        self.assertTrue(all(not e["allowed"] for e in esc),
                        f"读数不可信却放行了结构升级: {esc}")
        self.assertIn(COND_PLATEAU,
                      [r for e in esc for r in e["refusals"]],
                      f"缺的应该是「没有 K 轮 valid 平台」这一条: {esc[-1]['refusals']}")

    def test_the_gate_is_recorded_even_when_it_denies(self):
        """判据算了没人读的镜像: 门开着/关着都要在产物里留下**判过什么**"""
        art, _, _ = self._run(_LoopLlm(), max_rounds=5)
        esc = art.metadata["escalation"]
        self.assertTrue(esc, "裁决日志是空的 —— 门算过没算过必须看得出来: "
                             f"stop={art.metadata['stop']} "
                             f"actions={[r.get('action') for r in art.iteration_history]}")
        last = esc[-1]
        self.assertEqual({"allowed", "conditions", "refusals", "detail", "round"},
                         set(last))
        self.assertIn("valid_rounds_seen", last["detail"])
        self.assertEqual(MAX_STAGNANT_ROUNDS, last["detail"]["k_valid_rounds"],
                         "K 必须与停滞上限是同一个数")


if __name__ == "__main__":
    unittest.main(verbosity=2)