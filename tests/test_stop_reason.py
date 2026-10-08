"""「循环为什么停」记成了数据，并且词汇表与写者被差集钉住 (#71 的第一步)。

doc 53 §7.1 从 DSH 借的是**形状**而不是 hook 系统：`agent/turn-stopping` 是
「自然停止边界上一个被 await 的串行通知」(`agent-lifecycle.md:59-60`、
`architecture.md:84`、`interception-extension-points.md:21`)，要继续得显式 `agent.steer()`。
本仓今天的对应物是 `conductor.py` 优化循环里的 14 处终止出口 (`return _ROUND_STOP`,
散在 _run_round 与它拆出来的 _round_enter_or_stop / _round_close_out 里) + 1 处走满轮次，
其中 5 处的「为什么」只活在 stdout 的一行 print 上 —— 产物里
「没练够」「练到顶了」「修坏了」「压根没验过」全部同形 (`all_passed=False` + 一个轮次数)。
rounds-to-pass 那类统计 (#48) 要的正是这个区分，所以这一步先记成数据、不加任何新阈值。

⚠️ 这一片**没有**引入 `budget` 出口 —— 那是 #71 剩下的那一半，需要先有忙时/成本的分布。
词汇表里刻意不含它，并且有一条用例证明「现在写 `budget` 会报错」：
一个没有写者的枚举值就是「算了没人读」的又一个副本。

变异标定（9 条，0 逃逸；期望在跑之前按「这条变异让哪条断言的前提消失」推）。
`P/F/R/V/X/B/S/N` = 本文件 9 条用例，`C` = `tests/test_cost_persistence.py`：
  M1 passed 出口的 reason 换成 converged → P,V          （F **不该**红：`converged` 在词汇表里）
  M2 词汇表多加一个没写者的词 `budget`   → V,X
  M3 删掉走满轮次那一档                 → R,V
  M4 passed 那处整行删                  → P,V,B          （落到的 reason 是 `rounds_exhausted`，
                                                         仍在词汇表里 ⇒ F 同样不该红）
  M5 不落盘 stop                        → S
  M6 整包倒 metadata 代替单键           → S,**C**        （N 打不红：那次运行的 metadata 只有
                                                         `stop` 一个键，pop 后两种写法不可区分
                                                         —— 见 CHECKLOG I8）
  M7 词汇校验被拆（任何字符串都收）     → X
  M8 无条件写 stop 键（缺值时写出 null）→ N
  M9 绕过 setter 直写词汇外的 reason    → P,F,V,B        （F 唯一的活路：其余静态守卫看不见它）

运行: python3 -B -m unittest tests.test_stop_reason -v
"""

import ast
import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from operator_agent.artifact_store import ArtifactStore  # noqa: E402
from operator_agent.conductor import Conductor  # noqa: E402
from operator_agent.kernel_db import KernelDatabase  # noqa: E402
from operator_agent.mock_gpu import MockGPU  # noqa: E402
from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec, TargetSpec,  # noqa: E402
                                    TaskComplexity, TaskSchema, TaskType)

# 与 MockGPU 的 matmul 参考实现真对上的核 (试过 `A @ B` 出 9 vs 参考 8 的那种「样例核」
# 只会让所有用例都走 l1_l2_failed, 于是 passed/rounds_exhausted 两侧都测不到)
GOOD = "import numpy as np\n\ndef kernel(A, B):\n    return A @ B\n"
BAD = "import numpy as np\n\ndef kernel(A, B):\n    return (A @ B) + 7.0\n"


class _FakeLLM:
    """Designer 走 chat_json, Coder/Fixer/Search 走 chat。带一个真账本以便与 #70 的用例同形。"""

    def __init__(self, code):
        self.code = code
        self._sketch = {"algorithm": "matmul",
                        "tiling": {"block_m": 64, "block_n": 64, "block_k": 32},
                        "memory_plan": {}, "pipeline": [], "optimization_hints": []}

    def chat(self, messages, temperature=None, max_tokens=None, raw=False, caller_tag="Unknown"):
        return f"```python\n{self.code}```"

    def chat_json(self, messages, temperature=None, max_tokens=None, caller_tag="Unknown"):
        return dict(self._sketch)


def _task() -> TaskSchema:
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.LOW),
        target=TargetSpec(hardware="mock_gpu", dsl="python"),
        constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=0.5, max_error=1e-3))


def _run(code, rounds, target_gflops=0.0):
    c = Conductor(backend=MockGPU(), kernel_db=KernelDatabase(), llm=_FakeLLM(code),
                  max_optimize_rounds=rounds)
    inputs = {"A": np.ones((8, 8), dtype=np.float32), "B": np.ones((8, 8), dtype=np.float32)}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        art = c.execute(_task(), inputs, target_gflops=target_gflops)
    return art, buf.getvalue()


class TestStopReasonIsRecorded(unittest.TestCase):
    def test_a_passing_run_reports_passed(self):
        art, _ = _run(GOOD, 1)
        self.assertTrue(art.all_passed, "样例核没过 L1/L2/L3 的话, 这一组用例全在空转")
        stop = art.metadata["stop"]
        self.assertEqual("passed", stop["reason"])
        self.assertEqual(1, stop["round"])

    def test_a_failed_run_reports_which_failure_ended_it(self):
        art, log = _run(BAD, 1)
        self.assertFalse(art.all_passed)
        self.assertEqual("l1_l2_failed", art.metadata["stop"]["reason"],
                         f"结果错 + 轮次用完, 却报成别的: {art.metadata['stop']}")

    def test_the_reason_is_not_a_free_form_string(self):
        for code, rounds in ((GOOD, 1), (BAD, 1)):
            art, _ = _run(code, rounds)
            self.assertIn(art.metadata["stop"]["reason"], Conductor.STOP_REASONS)

    def test_running_out_of_rounds_says_so_rather_than_looking_like_a_failure(self):
        """「没练够」这一档。这一片的全部理由就在这一条上。

        L3 的门槛抬到 5.0x ⇒ 每轮都过不了 L3 但也没触发提前出口, 循环走到末尾。
        以前这种运行与「修坏了」「已到上界」在产物里完全同形 (都是 `all_passed=False`
        加一个轮次数), 而 rounds-to-pass 那类统计要的正是区分它们。
        """
        task = _task()
        task.constraints.min_speedup = 5.0
        c = Conductor(backend=MockGPU(), kernel_db=KernelDatabase(), llm=_FakeLLM(GOOD),
                      max_optimize_rounds=2)
        inputs = {"A": np.ones((8, 8), dtype=np.float32), "B": np.ones((8, 8), dtype=np.float32)}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            art = c.execute(task, inputs, target_gflops=0.0)
        self.assertFalse(art.all_passed, "样例核过了的话这一档走不到, 用例是空转")
        stop = art.metadata["stop"]
        self.assertEqual("rounds_exhausted", stop["reason"],
                         f"L3 不达标且没触发提前出口, 却报成: {stop}")
        self.assertEqual(2, stop["round"])
        self.assertNotIn(stop["reason"], ("passed", "converged", "l1_l2_failed"))

    def test_the_vocabulary_has_no_word_without_a_writer(self):
        """差集等式: 词汇表 == 循环里真的写出来的那些 reason。

        子集式断言对「加了个没人写的词」是盲的 (doc 51 §五 那条通则)，而这里两侧都要钉:
        写者用了词汇外的词会抛 (`test_an_unknown_reason_raises_instead_of_being_stored`)，
        词汇表里多一个没写者的词由这条抓。
        """
        src = io.open(ROOT / "operator_agent" / "conductor.py", encoding="utf-8").read()
        written = set()
        for node in ast.walk(ast.parse(src)):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "_record_stop" and len(node.args) >= 3):
                third = node.args[2]
                if isinstance(third, ast.Constant) and isinstance(third.value, str):
                    written.add(third.value)
        self.assertEqual(set(Conductor.STOP_REASONS), written,
                         f"只有写者没有词: {sorted(written - set(Conductor.STOP_REASONS))} / "
                         f"只有词没有写者: {sorted(set(Conductor.STOP_REASONS) - written)}")

    def test_an_unknown_reason_raises_instead_of_being_stored(self):
        """`budget` 还不该存在 —— #71 加出口时同时加写者, 现在写了就该红。"""
        art, _ = _run(GOOD, 1)
        with self.assertRaises(ValueError):
            Conductor._record_stop(Conductor(backend=MockGPU(), kernel_db=KernelDatabase(),
                                            llm=None), art, 1, "budget", "还没这一档")

    #: 轮次的组成部分 —— 出口可能住在其中任何一个里。Stage3 从 _run_round 里搬出了
    #: 轮首两道门与收尾两段, 出口跟着走。少列一个方法, 它里面的出口就会漏计。
    _ROUND_PARTS = ("_run_round", "_round_enter_or_stop", "_round_close_out")

    def test_every_exit_from_a_round_has_a_reason_attached(self):
        """防腐: 轮次里每个「终止优化循环」的出口, 紧邻上方必须是 `_record_stop(...)`。

        新增出口而忘了记 reason 是最自然的走法 (出口已经在那儿了)，而那时产物会静默地
        缺这一档 —— 与 doc 49 §7.7 那条「差分守卫要配绝对锚点」同族, 这里用**计数**钉住。

        Stage2 把循环体搬进 _run_round 之后, 出口的写法从 `break` 变成了
        `return _ROUND_STOP`; Stage3 又把轮首门与收尾搬成方法, 出口跟着散到三个
        方法里。判据要盯的东西一个字没变 (每个终止出口都得带 reason), 所以这里跟着走 ——
        是换位置, 不是放宽。

        多出来的一类出口是**委派**式的:

            if self._round_enter_or_stop(state, round_num) is _ROUND_STOP:
                return _ROUND_STOP

        它的 reason 记在被委派的那个方法里, 所以判据把这种出口单独数, 并要求它委派到的
        方法里**确实**有一个带 reason 的实质出口 —— 否则委派就成了藏出口的通道。
        """
        tree = ast.parse(io.open(ROOT / "operator_agent" / "conductor.py",
                                 encoding="utf-8").read())
        conductor = next(n for n in ast.walk(tree)
                         if isinstance(n, ast.ClassDef) and n.name == "Conductor")
        parts = [next(n for n in conductor.body
                      if isinstance(n, ast.FunctionDef) and n.name == name)
                 for name in self._ROUND_PARTS]
        calls = {n.lineno for fn in parts for n in ast.walk(fn)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "_record_stop"}
        exits, delegated = [], []
        for fn in parts:
            parents = {ch: n for n in ast.walk(fn) for ch in ast.iter_child_nodes(n)}
            for node in ast.walk(fn):
                if not (isinstance(node, ast.Return) and isinstance(node.value, ast.Name)
                        and node.value.id == "_ROUND_STOP"):
                    continue
                parent = parents.get(node)
                targets = ([x.func.attr for x in ast.walk(parent.test)
                            if isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute)
                            and x.func.attr in self._ROUND_PARTS]
                           if isinstance(parent, ast.If) else [])
                if targets:
                    delegated.append((node.lineno, targets[0], fn.name))
                else:
                    exits.append((node.lineno, fn.name))
        exits.sort()          # ast.walk 是 BFS 不是行序, 下面是按行序钉的
        reasoned = [e for e in exits if any(0 < e[0] - line <= 3 for line in calls)]
        self.assertEqual(len(exits), len(reasoned),
                         f"轮次里 {len(exits)} 个实质出口, 只有 {len(reasoned)} 个带着 reason")
        for lineno, target, holder in delegated:
            self.assertTrue([e for e in exits if e[1] == target],
                            f"{holder}:L{lineno} 的委派出口把 reason 推给了 {target}, "
                            f"而 {target} 里没有带 reason 的实质出口 —— 委派成了藏出口的通道")
        # 绝对锚点 (doc 49 §7.7): 「每个出口都带 reason」是差分判据, 它拦不住「把一个出口
        # 改成 continue」—— 少一个出口, 两边一起少, 等式照样成立。实测: 把第一个出口
        # (wall_clock_exhausted) 改成 return _ROUND_CONTINUE, 全量 2570 条全绿。
        # 所以个数与对应关系都写死。
        self.assertEqual(14, len(exits),
                         "终止出口的个数变了 —— 少一个就是某个出口被改成了「继续」")
        self.assertEqual(1, len(delegated),
                         "委派式出口的个数变了 —— 进出口清单的话, 这里也要改")
        calls_by_line = {}
        for fn in parts:
            for n in ast.walk(fn):
                if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                        and n.func.attr == "_record_stop"):
                    calls_by_line[n.lineno] = n
        reasons = []
        for lineno, _holder in exits:
            near = sorted((c for line, c in calls_by_line.items() if 0 < lineno - line <= 3),
                          key=lambda c: c.lineno)
            reasons.append(ast.unparse(near[-1].args[2]) if near else None)
        self.assertEqual(
            ["'wall_clock_exhausted'", "'budget_exhausted'", "'budget_exhausted'",
             "'search_error'", "'budget_exhausted'", "'template_no_candidate'",
             "'transport_unknown_hold'", "'passed'", "'stagnant'", "'converged'", "'infra_retry'", "'fix_error'",
             "'ask_turns_exhausted'", "'l1_l2_failed'"],
            reasons,
            "出口与 reason 的对应关系漂了 —— 这份清单按全轮的三个方法、按行序钉")


class TestStopReasonReachesResultJson(unittest.TestCase):
    def test_the_stop_block_crosses_the_persistence_boundary(self):
        art, _ = _run(GOOD, 1)
        stop = art.metadata["stop"]
        d = Path(tempfile.mkdtemp(prefix="stop-"))
        try:
            saved = ArtifactStore(output_dir=str(d)).save(art, _task())
            with io.open(saved / "result.json", encoding="utf-8") as fh:
                result = json.load(fh)
            self.assertEqual(stop, result["stop"],
                             "reason 只在内存里的话, 量具与事后复盘都读不到")
            self.assertNotIn("metadata", result, "整包 metadata 不许跟着落盘")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_an_artifact_without_a_stop_block_writes_no_key(self):
        """缺证据不许当否定证据: 没有出口记录时不造一个 `{"reason": null}`。

        「整包 metadata 不许跟着落盘」那一半不在这里 —— 它已经有读者
        （`tests/test_cost_persistence.py:122 test_the_rest_of_metadata_is_not_dumped`），
        这条只管「缺值时不写键」这一件事。
        """
        art, _ = _run(GOOD, 1)
        art.metadata.pop("stop")
        d = Path(tempfile.mkdtemp(prefix="stop-"))
        try:
            saved = ArtifactStore(output_dir=str(d)).save(art, _task())
            with io.open(saved / "result.json", encoding="utf-8") as fh:
                self.assertNotIn("stop", json.load(fh))
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
