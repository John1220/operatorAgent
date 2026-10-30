"""§12 的「模板产出候选」那一半：参数空间 → 候选队列 → 落进主循环（Stage 3 第二片）

CHECKLOG I61 明写了控制器侧还差的正是这一片：`instantiate_points` 建得出源码，但没人
消费它。这里把它接上：`select_template_candidate` 剪枝 → bandit 排序 → **按顺位取第一个
建得出来的点**，Conductor 在轮首取一次，本轮**跳过 Coder/Search**，血缘里落模板身份。

四条判据各自的失效：

| 若没有这道判据 | 会读成什么 |
|---|---|
| 排序走**已有的** bandit（臂 = 点的去重键） | 「这个点上次变快了没有」随调用方各自解释；两套臂身份 |
| 顺位取第一个**建得出来**的点 | 某个点上实例化器一直失败 ⇒ 这一轮直接没得跑（而它本该跳过它试下一个） |
| 全部点建不出来 ⇒ 停机并说明 | 悄悄退回代码型生成路径（§12 点名的失效） |
| 模板轮跳过 Coder/Search | 模板点与 LLM 结构混进同一个候选队列，身份口径分叉 |

运行: python3 -m unittest tests.test_template_queue -v
"""

import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from operator_agent.backend_caps import Backend, Instantiation
from operator_agent.bandit import Bandit
from operator_agent.candidates import (ParamSpace, TemplateCandidate,
                                      dedup_key, point_label,
                                      select_template_candidate, template_arm,
                                      template_points, template_resource_binding)
from operator_agent.conductor import Conductor
from operator_agent.kernel_db import KernelDatabase
from operator_agent.mock_gpu import MockGPU
from operator_agent.resource_check import MemCaps, SharedMemoryEvidence, caps_from_metrics
from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec, TargetSpec,
                                    TaskComplexity, TaskSchema, TaskType)
from tests.unit.test_conductor_stagnation import SLOW_MATMUL

# Full template parameters participate in resource binding; shapes alone do not
# establish physical shared storage. Explicit synthetic receipts are below.
_SPACE = ParamSpace(template_id="mm_tile", backend="fake", instantiator_ver="tv-1",
                    axes={"BLOCK_M": (64, 128), "BLOCK_N": (64, 128), "BLOCK_K": (64,)})


def _resource_receipts(space=_SPACE):
    """Synthetic native receipts, explicitly supplied by the test harness."""
    receipts = {}
    for point, byte_count in zip(space.points(), (49152, 81920, 81920, 131072)):
        binding = template_resource_binding(point)
        receipts[binding] = SharedMemoryEvidence(
            binding=binding, kind="native_compiler", producer="queue-test-native-harness-v1",
            native_bytes=byte_count)
    return receipts


class _TemplateBackend(MockGPU):
    """一个真的声明了参数空间的后端假件 (MockGPU + 实例化器)

    实例化器交出来的**是可跑的 Python 核** (`SLOW_MATMUL`): 模板后端在链路上要能真的
    走完 L1/L2/L3 —— 这也是「模板候选不是源码字符串, 但它会把源码交给后端」的实证。
    """

    def __init__(self, space=None, source=SLOW_MATMUL, fail_on=()):
        super().__init__()
        self._space = space or _SPACE
        self._source = source
        self._fail_on = set(fail_on)
        self.instantiate_calls = []

    def param_space(self, contract=None):
        return self._space

    def instantiate(self, template_id, params, contract=None):
        self.instantiate_calls.append(dict(params))
        if params.get("BLOCK_M") in self._fail_on:
            raise RuntimeError(f"实例化器不支持 BLOCK_M={params.get('BLOCK_M')}")
        return Instantiation(source=self._source, instantiator_ver="tv-1",
                             evidence=f"{template_id} {sorted(params.items())}")


class _RefusingLLM:
    """一旦被调用就炸 —— 用来证明模板轮**没有**走 LLM/Coder/Search"""

    def __init__(self):
        self.chat_calls = 0

    def chat(self, messages, temperature=None, max_tokens=None, raw=False,
             caller_tag="Unknown"):
        self.chat_calls += 1
        raise AssertionError("模板轮不该调 LLM/Coder")

    def chat_json(self, messages, temperature=None, max_tokens=None, caller_tag="Unknown"):
        self.chat_calls += 1
        raise AssertionError("模板轮不该调 LLM/Coder")


class _DesignerOnlyLLM(_RefusingLLM):
    """只允许 Designer 那一次 (它产出 sketch, 循环结构仍然需要)"""

    def chat_json(self, messages, temperature=None, max_tokens=None, caller_tag="Unknown"):
        return {"algorithm": "matmul",
                "tiling": {"block_m": 64, "block_n": 64, "block_k": 32},
                "memory_plan": {}, "pipeline": [], "optimization_hints": []}


class _KernelLLM(_DesignerOnlyLLM):
    """代码型后端那条路要真的过 Coder ⇒ 允许 `chat` 并交出一份能过 L1/L2 的核"""

    def chat(self, messages, temperature=None, max_tokens=None, raw=False,
             caller_tag="Unknown"):
        self.chat_calls += 1
        return "```python\n" + SLOW_MATMUL + "```"


class _BoomSearch:
    """一被调用就炸 —— 用来证明模板轮**没有**走 LLM/邻域搜索那条路"""

    def run(self, *a, **kw):
        raise AssertionError("模板轮不该走 SearchAgent")


def _task(min_speedup=1.0):
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(framework="pytorch", kernel_type="matmul",
                          fused_ops=[], complexity=TaskComplexity.MEDIUM),
        target=TargetSpec(hardware="mock_gpu", dsl="python",
                          compute_units=64, warp_size=32),
        constraints=ConstraintSpec(precision=Precision.FP32,
                                   min_speedup=min_speedup, max_error=1e-5),
    )


def _inputs():
    rng = np.random.default_rng(0)
    return {"A": rng.standard_normal((64, 64)).astype(np.float32),
            "B": rng.standard_normal((64, 64)).astype(np.float32)}


class TestTheRanking(unittest.TestCase):

    def test_the_arm_is_the_point_identity(self):
        """臂 = 去重键 (同一个点只有一个臂), 与候选身份同源"""
        point = _SPACE.points()[0]
        self.assertEqual(dedup_key(point), template_arm(point))
        self.assertNotEqual(template_arm(point), template_arm(_SPACE.points()[1]))

    def test_no_order_keeps_the_enumeration_order(self):
        """调用方不给顺序 ⇒ 枚举序 (选择器是**上面**那一层的事, 这里不自己排)"""
        backend = _TemplateBackend()
        sel = select_template_candidate(_SPACE, backend, caps=MemCaps(228))
        self.assertEqual([point_label(c) for c in _SPACE.points()],
                         [point_label(c) for c in sel.ranked])

    def test_a_caller_order_is_honoured_and_must_be_complete(self):
        """调用方给的顺序说了算 (排序权威在 `selection`); 漏点当场炸, 不悄悄少候选"""
        backend = _TemplateBackend()
        points = _SPACE.points()
        wanted = [points[2], points[0], points[3], points[1]]
        sel = select_template_candidate(_SPACE, backend, caps=MemCaps(228), order=wanted)
        self.assertEqual([point_label(c) for c in wanted],
                         [point_label(c) for c in sel.ranked])
        self.assertEqual(point_label(points[2]), point_label(sel.chosen.candidate))
        with self.assertRaises(ValueError) as ctx:
            select_template_candidate(_SPACE, backend, caps=MemCaps(228),
                                      order=wanted[:2])
        self.assertIn("漏了", str(ctx.exception))

    def test_the_choice_is_the_first_point_that_builds(self):
        """排在第一的点建不出来 ⇒ 跳过它试下一个, 并**逐条留痕**"""
        backend = _TemplateBackend(fail_on=(64,))
        sel = select_template_candidate(_SPACE, backend, caps=MemCaps(228))
        self.assertIsNotNone(sel.chosen)
        self.assertEqual(128, sel.chosen.candidate.params["BLOCK_M"],
                         "BLOCK_M=64 那两个点建不出来, 顺位应该走到 128")
        self.assertEqual(2, len(sel.refusals))
        self.assertIn("RuntimeError", sel.refusals[0][1])
        self.assertIn("前 2 个点建不出来", sel.note())

    def test_nothing_buildable_is_not_an_empty_space(self):
        backend = _TemplateBackend(fail_on=(64, 128))
        sel = select_template_candidate(_SPACE, backend, caps=MemCaps(228))
        self.assertIsNone(sel.chosen)
        self.assertEqual(4, len(sel.ranked), "空间非空: 4 个点都在, 只是都建不出来")
        self.assertEqual(4, len(sel.refusals))
        self.assertIn("一个都没建成", sel.note())
        self.assertIn("拒绝 4 条", sel.note())

    def test_the_prune_note_is_carried_and_unknown_limit_is_said(self):
        backend = _TemplateBackend()
        receipts = _resource_receipts()
        no_limit = select_template_candidate(_SPACE, backend, caps=MemCaps(0),
                                             resource_evidence=receipts)
        self.assertTrue(no_limit.no_limit)
        self.assertIn("未生效", no_limit.pruned_note)
        unknown = select_template_candidate(_SPACE, backend, caps=MemCaps(60))
        self.assertEqual(4, len(unknown.ranked), "只给 caps 和形状不能确定资源超限")
        self.assertIn("未知", unknown.pruned_note)
        # Receipts establish 49152 B vs 81920/131072 B, not a source formula.
        tight = select_template_candidate(_SPACE, backend, caps=MemCaps(60),
                                          resource_evidence=receipts)
        self.assertIn("摘掉", tight.pruned_note)
        self.assertFalse(tight.no_limit)
        self.assertEqual(1, len(tight.ranked))
        # 1 KiB: 四个点全超 ⇒ 一个都不摘 (剪枝不是「让这一轮没得跑」), 理由仍留着
        starved = select_template_candidate(_SPACE, backend, caps=MemCaps(1),
                                            resource_evidence=receipts)
        self.assertIn("全超上界", starved.pruned_note)
        self.assertEqual(4, len(starved.ranked))

    def test_all_pruned_falls_back_to_keeping_them(self):
        """剪枝全摘光时一个都不摘 (它不是「让这一轮没得跑」)"""
        backend = _TemplateBackend()
        sel = select_template_candidate(_SPACE, backend, caps=MemCaps(1),
                                        resource_evidence=_resource_receipts())
        self.assertEqual(4, len(sel.ranked), "全摘光 ⇒ 全部保留待远端判定")
        self.assertIn("全超上界", sel.pruned_note)

    def test_a_junk_limit_is_unknown_not_one_kilobyte(self):
        """脏上界值 ⇒ **上界未知** (不剪), 不是「1 KB」(那会把点全判成放不下)

        需要检查带实际字段的脏值；不能只测取不到属性的对象。
        """
        self.assertEqual(MemCaps(0), caps_from_metrics({"shared_mem_per_sm_kb": "abc"}))
        self.assertEqual(MemCaps(0), caps_from_metrics("junk"))
        self.assertEqual(MemCaps(0), caps_from_metrics(None))
        backend = _TemplateBackend()
        junk = select_template_candidate(
            _SPACE, backend, caps=caps_from_metrics({"shared_mem_per_sm_kb": "abc"}),
            resource_evidence=_resource_receipts())
        self.assertTrue(junk.no_limit)
        self.assertIn("未生效", junk.pruned_note)
        self.assertEqual(4, len(junk.ranked))

    def test_the_selection_is_recorded_as_data(self):
        backend = _TemplateBackend()
        sel = select_template_candidate(_SPACE, backend, caps=MemCaps(228))
        d = sel.as_dict()
        self.assertEqual(point_label(sel.chosen.candidate), d["chosen"])
        self.assertEqual("template", d["chosen_lineage"]["kind"])
        self.assertIn("params", d["chosen_lineage"])
        self.assertEqual(4, len(d["ranked"]))

    def test_template_points_forwards_only_explicit_harness_resource_receipts(self):
        unknown = template_points(_SPACE, MemCaps(60))
        self.assertEqual(4, len(unknown.kept))
        self.assertEqual(4, unknown.resource_unknown)
        known = template_points(_SPACE, MemCaps(60), resource_evidence=_resource_receipts())
        self.assertEqual([_SPACE.points()[0]], known.kept)
        self.assertEqual(3, len(known.dropped))
        self.assertEqual(0, known.resource_unknown)

    def test_old_version_or_parameter_receipts_cannot_prune_a_new_template_space(self):
        old_receipts = _resource_receipts()
        variants = (
            ParamSpace(template_id="mm_tile", backend="fake", instantiator_ver="tv-2",
                       axes=_SPACE.axes),
            ParamSpace(template_id="mm_tile", backend="fake", instantiator_ver="tv-1",
                       axes={"BLOCK_M": (64, 128), "BLOCK_N": (64, 128), "BLOCK_K": (128,)}),
        )
        for space in variants:
            with self.subTest(version=space.instantiator_ver, axes=space.axes):
                res = template_points(space, MemCaps(1), resource_evidence=old_receipts)
                self.assertEqual(space.points(), res.kept)
                self.assertEqual([], res.dropped)
                self.assertEqual(4, res.resource_unknown)
                self.assertFalse(res.kept_all_to_avoid_starvation)
                self.assertIn("未知", res.as_note())


class TestTheLoopConsumesTheQueue(unittest.TestCase):

    def _run(self, backend, llm=None, rounds=2, min_speedup=1.0):
        """⚠️ 要看「这一轮的臂 / 搜索入口」必须 `rounds >= 2`: 那一段整体挂在
        `round_num < max_optimize_rounds` 之下 —— 只跑一轮时它整条都不进 (实测撞出来的:
        臂是 None 不是 bug, 是那句门)。停滞出口 (`MAX_STAGNANT_ROUNDS=2`) 也是在那之后的
        事, 所以第 2 轮照样能看到第 1 轮的臂。"""
        c = Conductor(backend=backend, kernel_db=KernelDatabase(jsonl_path=None),
                      llm=llm or _DesignerOnlyLLM(), max_optimize_rounds=rounds)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            art = c.execute(_task(min_speedup), _inputs(), target_gflops=0.0)
        return c, art, buf.getvalue()

    def test_a_template_round_skips_the_coder_and_records_its_identity(self):
        backend = _TemplateBackend()
        c, art, out = self._run(backend, rounds=1)
        self.assertIn("模板候选 (跳过 Coder)", out)
        self.assertIn("模板选择:", out)
        history = art.iteration_history
        self.assertTrue(history, "至少要有一轮血缘")
        sel = history[0]["template_selection"]
        self.assertIsNotNone(sel, "模板轮必须把「选了哪个点」落进血缘")
        self.assertEqual("template", sel["chosen_lineage"]["kind"])
        self.assertEqual("mm_tile", sel["chosen_lineage"]["template_id"])
        self.assertEqual("template", history[0]["action"])
        self.assertEqual(1, len(backend.instantiate_calls),
                         "只实例化顺位第一那个点 (惰性), 不是把空间全建一遍")
        # 交出去的源码真的走了验证链
        levels = history[0]["levels_passed"]
        self.assertIn("L1", levels)

    def test_the_template_source_is_what_got_verified(self):
        """实例化器的源码就是被测的那份 —— 不是 Coder 又生成了一份"""
        backend = _TemplateBackend(source=SLOW_MATMUL)
        c, art, out = self._run(backend, rounds=1)
        l1_ok = art.iteration_history[0]["levels_passed"].get("L1")
        self.assertTrue(l1_ok, "交出去的模板源码必须能过 L1")

    def test_a_code_backend_records_no_template_selection(self):
        """代码型后端 (今天全部) 这一格必须是 None, 而不是空 dict"""
        c, art, out = self._run(MockGPU(), llm=_KernelLLM(), rounds=1)
        self.assertIsNone(art.iteration_history[0]["template_selection"])
        self.assertNotIn("模板选择:", out)

    def test_no_buildable_point_stops_the_run_with_a_named_reason(self):
        backend = _TemplateBackend(fail_on=(64, 128))
        c, art, out = self._run(backend, rounds=2)
        stop = (art.metadata or {}).get("stop") or {}
        self.assertEqual("template_no_candidate", stop.get("reason"))
        self.assertIn("不退回代码型", stop.get("detail", ""))
        self.assertEqual([], art.iteration_history,
                         "一个候选都没产生 ⇒ 没有血缘条目 (不许记一条空的)")

    def test_the_instantiator_is_asked_for_the_points_in_ranked_order(self):
        """第二轮会再取一次 (bandit 吃过回报 ⇒ 顺位跟着证据走)"""
        backend = _TemplateBackend()
        c, art, out = self._run(backend, rounds=2, min_speedup=99.0)
        self.assertEqual(2, len(backend.instantiate_calls))

    def test_the_llm_is_not_asked_to_generate_a_kernel(self):
        llm = _DesignerOnlyLLM()
        backend = _TemplateBackend()
        c, art, out = self._run(backend, llm=llm, rounds=1)
        self.assertEqual(0, llm.chat_calls, "模板轮不许调 LLM 生成核")

    def test_the_search_path_is_not_taken_on_a_template_round(self):
        """L3 失败那一轮也不走 SearchAgent —— 否则模板点与 LLM 结构混进一个队列

        观察面是**调用本身**: 让 SearchAgent 一被调用就抛, 于是「没走这条路」是
        一个真的断言, 而不是从 stdout 里读一行字。
        """
        backend = _TemplateBackend()
        c = Conductor(backend=backend, kernel_db=KernelDatabase(jsonl_path=None),
                      llm=_DesignerOnlyLLM(), max_optimize_rounds=2)
        c.search = _BoomSearch()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            art = c.execute(_task(99.0), _inputs(), target_gflops=0.0)
        self.assertIn("模板型后端", buf.getvalue())
        self.assertEqual("template", art.iteration_history[0]["action"])

    def test_the_bandit_arm_is_the_point_identity(self):
        """回报记到**那个点**头上 (不是 sketch 的 tiling 三元组)

        臂只在 L3 没过那一轮才产生 (进了搜索/回报) ⇒ 门槛抬到 99 让这一轮的 L3 挂。
        """
        backend = _TemplateBackend()
        c, art, out = self._run(backend, rounds=2, min_speedup=99.0)
        arms = list(((art.metadata or {}).get("bandit") or {}).get("arms") or {})
        self.assertEqual(1, len(arms), "第 1 轮进了搜索 ⇒ 恰好一个臂")
        self.assertIn("mm_tile", arms[0], "臂 = 那个模板点的去重键")
        # 血缘里存的是**臂对象本身** (元组, JSON 化时变成数组); bandit 那一格是它的
        # 字符串形式 —— 两者说的是同一个臂, 不是两个口径。
        lineage_arm = art.iteration_history[0]["bandit_arm"]
        self.assertIsInstance(lineage_arm, tuple)
        self.assertEqual(str(lineage_arm), arms[0])

    def test_a_code_backend_keeps_the_tiling_arm(self):
        """代码型轮仍然是 tiling 三元组 —— 臂口径不因这一片而变"""
        c, art, out = self._run(MockGPU(), llm=_KernelLLM(), rounds=2, min_speedup=99.0)
        self.assertEqual("64x64x32", art.iteration_history[0]["bandit_arm"],
                         "sketch 的 tiling 三元组 (Designer 假件给的是 64/64/32)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
