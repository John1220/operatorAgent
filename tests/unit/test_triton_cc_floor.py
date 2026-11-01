# -*- coding: utf-8 -*-
"""CC 下限预检：一台编不出码的卡，不许变成一张候选能力表。

起因是三次选卡提问（TITAN X 5.2 / P4 6.1 / 市场 20 档）。判据本身在仓里：Triton 的
CUDA 后端有 CC 下限，而失败路由的 `_INFRA_MARKERS` 只认 oom 那一族 ⇒ 架构不支持产生的
编译失败会被记成**候选失败**（不算迭代、不学习、重试也不救），跑完一整轮交出的是
「这个 agent 写的核全错」，而真因是这台机器不能编译。见 docs/research/80 §2 判据 1。

两道保障分开钉：
  * 纯函数：判词与拒绝的边界（含「解析不出来 ≠ 低于下限」这条反向对照）。
  * 接线：预检真的在 `Conductor.execute` 里，而且**在任何 LLM 调用之前**——最后那组用例
    用一个「被调就炸」的 LLM 假件来证明一次钱都没花。
"""
import ast
import unittest
from pathlib import Path

import numpy as np

from operator_agent.core.hardware_caps import (
    TRITON_MIN_CC_MAJOR, TritonUnsupported, require_triton_support, triton_cc_verdict)
from operator_agent.core.kernel_db import KernelDatabase
from operator_agent.core.schemas import (ConstraintSpec, Precision, SourceSpec, TargetSpec,
                                         TaskComplexity, TaskSchema, TaskType)
from operator_agent.app.conductor import Conductor

_ROOT = Path(__file__).resolve().parents[2]
_CONDUCTOR_SOURCE = (_ROOT / "operator_agent/app/conductor.py").read_text(encoding="utf-8")

P4 = {"name": "Tesla P4", "compute_capability": "6.1", "device": "cuda"}
BELOW = ("5.2", "6.0", "6.1", "3.7")
SUNSET = ("7.0", "7.2", "7.5")
OK = ("8.0", "8.6", "8.9", "9.0", "10.0", "12.0")
UNPARSEABLE = (None, "", "unknown", "gfx90a", "  ", "x.y")


def _calls_to(source, func_name, callee):
    out = []
    for fn in [n for n in ast.walk(ast.parse(source))
               if isinstance(n, ast.FunctionDef) and n.name == func_name]:
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id == callee:
                out.append(node.lineno)
    return out


class TestVerdict(unittest.TestCase):

    def test_below_the_floor_is_named_below(self):
        for cc in BELOW:
            self.assertEqual("below", triton_cc_verdict({"compute_capability": cc})[0], cc)

    def test_the_floor_itself_and_above_are_not_below(self):
        for cc in SUNSET:
            self.assertEqual("sunset", triton_cc_verdict({"compute_capability": cc})[0], cc)
            self.assertEqual(("sunset", cc), require_triton_support({"compute_capability": cc}),
                             "%s 只该被提示，不该被拦下" % cc)
        for cc in OK:
            self.assertEqual("ok", triton_cc_verdict({"compute_capability": cc})[0], cc)
            self.assertEqual(("ok", cc), require_triton_support({"compute_capability": cc}), cc)

    def test_unparseable_is_unknown_not_zero(self):
        """**这是本模块最重要的一条**：悄悄当 0 就把 CPU 回退和模拟器一起判死了。"""
        for cc in UNPARSEABLE:
            self.assertEqual("unknown", triton_cc_verdict({"compute_capability": cc})[0],
                             repr(cc))
            self.assertEqual("unknown", require_triton_support({"compute_capability": cc})[0],
                             repr(cc))

    def test_real_backends_without_cc_are_not_refused_here(self):
        """反向对照：MockGPU 与 LocalExecutor 的字典里没有 cc。

        拦它们不是这道门的活（`smem_budget()` 与采集器管「没探测到」）；混在一起会让
        无 GPU 的开发路径整条打回。
        """
        from operator_agent.execute.mock_gpu import MockGPU
        from operator_agent.execute.remote_executor import LocalExecutor
        for info in (MockGPU().get_hardware_info(), LocalExecutor().get_hardware_info()):
            self.assertEqual("unknown", require_triton_support(info)[0])


class TestRefusal(unittest.TestCase):

    def test_p4_is_refused_before_anything_else(self):
        with self.assertRaises(TritonUnsupported) as ctx:
            require_triton_support(P4)
        msg = str(ctx.exception)
        self.assertIn("6.1", msg, "要回显读到的 cc")
        self.assertIn("Tesla P4", msg, "要回显设备名")
        self.assertIn(str(TRITON_MIN_CC_MAJOR), msg, "要说清下限在哪")
        self.assertIn("候选失败", msg, "要说清**为什么**不许跑到编译期")
        self.assertIn("docs/research/80", msg, "要指到判据出处")

    def test_the_backend_label_is_carried_into_the_message(self):
        with self.assertRaises(TritonUnsupported) as ctx:
            require_triton_support(P4, context="ssh:gpu-3")
        self.assertIn("ssh:gpu-3", str(ctx.exception))

    def test_at_or_above_the_floor_returns_the_verdict(self):
        self.assertEqual(("ok", "8.6"), require_triton_support({"compute_capability": "8.6"}))


class _BoomLlm:
    """LLM 假件：只实现真接口的那两个方法，**被调就炸**。

    刻意不写 `__getattr__`：`Conductor` 会 `hasattr(llm, "cost_tracker")`，一个来者不拒的
    假件会让那次探测拿到函数、然后在别处炸出与本题无关的红。
    """

    def __init__(self):
        self.calls = 0

    def _hit(self, method):
        self.calls += 1
        raise AssertionError(f"CC 下限预检之前就发了 LLM 调用: {method}")

    def chat(self, messages, **kw):
        self._hit("chat")

    def chat_json(self, messages, **kw):
        self._hit("chat_json")


def _task():
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(framework="pytorch", kernel_type="matmul", fused_ops=[],
                          complexity=TaskComplexity.MEDIUM),
        target=TargetSpec(hardware="tesla_p4", dsl="python", compute_units=128,
                          warp_size=32),
        constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=1.0,
                                   max_error=1e-5))


class TestItIsActuallyWired(unittest.TestCase):
    """抽了函数不等于接上了。三条从不同角度钉：AST、端到端、活性。"""

    def test_conductor_execute_calls_the_guard(self):
        """预检落在 `_execute_inner`（`execute()` 只是它的包装），守卫按真实落点算。"""
        self.assertEqual(1, len(_calls_to(_CONDUCTOR_SOURCE, "_execute_inner",
                                          "require_triton_support")),
                         "Conductor 的运行体不再调这道门，上面所有用例守的都是死代码")

    def test_the_ast_parser_sees_calls_at_all(self):
        self.assertEqual(1, len(_calls_to(_CONDUCTOR_SOURCE, "_execute_inner", "fingerprint_of")),
                         "_calls_to 没在解析函数体，上一条在空转")
        self.assertEqual([], _calls_to(_CONDUCTOR_SOURCE, "no_such_method", "require_triton_support"))

    def test_run_stops_before_the_first_llm_call(self):
        """真跑一遍 `execute()`：一台 P4 ⇒ 抛 `TritonUnsupported`，且 LLM 一次都没被摸到。"""
        import contextlib
        import io

        from backend_fakes import DeclaredFakeBackend

        llm = _BoomLlm()
        backend = DeclaredFakeBackend(get_hardware_info=lambda: dict(P4))
        c = Conductor(backend=backend, kernel_db=KernelDatabase(jsonl_path=None), llm=llm,
                      max_optimize_rounds=1, heartbeat_interval=1000)
        rng = np.random.default_rng(0)
        inputs = {"A": rng.standard_normal((32, 32)).astype(np.float32),
                  "B": rng.standard_normal((32, 32)).astype(np.float32)}
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(TritonUnsupported) as ctx:
                c.execute(_task(), inputs)
        self.assertEqual(0, llm.calls, "预检之前就发了 LLM 调用 ⇒ 这道门不省钱")
        self.assertIn("6.1", str(ctx.exception), "拒绝理由要带着读到的 cc")
