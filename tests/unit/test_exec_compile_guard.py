"""§22.2 第一项「编译超时」：编译走受控子进程之后的**接线**与**传递链**

三件事，每一件都有行为证人：

 ① **对外契约逐字不变**：`MockGPU.compile` 的 `SyntaxError` 原样重抛 / 其余包成
    `RuntimeError("Compile error: …")`；`LocalExecutor.compile` 的类型原样透传。
    编译方式换了（进程内 `exec` → 受控子进程），契约不许跟着换。
 ② **阶段级上限真的传到了执行器**（调用方 → 后端 → `exec_guard`）。只写一个构造参数、
    没有消费端，就是本仓记过的「声明 + 无生产调用点」那一类 —— 所以这里钉的是
    **读数**（把墙钟压到 400ms，看悬挂候选多久被拦下），不是钉字段等于多少。
 ③ `Verifier` 把「执行器拦下」这一档发布成 `compile_status = compile_<status>`：
    编译超时与「候选自己语法错」不许在产物里长得一样（I65/I11 那一族）。

⚠️ 不许读成：编译期被隔离了（同一个 uid、网络没关、无容器）；也不表示编译产物与真机一致。
    本文件只证「编译期那一步在受控子进程里跑，且上限真的到了那里」。
"""

import sys
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from operator_agent.execute import mock_gpu as mock_gpu_module  # noqa: E402
from operator_agent.app.conductor import Conductor  # noqa: E402
from operator_agent.execute.exec_guard import (  # noqa: E402
    STATUS_OK, GuardRefusal, GuardResult, Limits,
)
from operator_agent.core.kernel_db import failure_class_of  # noqa: E402
from operator_agent.execute.mock_gpu import MockGPU, TRITON_COMPILE_IMPORTS  # noqa: E402
from operator_agent.execute.remote_executor import LocalExecutor  # noqa: E402
from operator_agent.decide.verifier import Verifier  # noqa: E402

HANG = "while True:\n    pass\n"
OK_CODE = "def kernel(**kwargs):\n    return 1\n"
TRITON_SRC = ("import triton\nimport triton.language as tl\n"
              "@triton.jit\ndef _k():\n    pass\ndef kernel(x):\n    return x\n")


class TestTheCompileContractIsUnchanged(unittest.TestCase):
    """换的是执行方式，不是契约 —— 每一行都是搬走之前的行为"""

    def setUp(self):
        self.gpu = MockGPU()

    def test_a_valid_candidate_still_compiles(self):
        self.assertIs(True, self.gpu.compile(OK_CODE))
        self.assertIs(True, LocalExecutor().compile(OK_CODE))

    def test_a_missing_kernel_is_a_syntax_error(self):
        for backend, message in ((self.gpu, "Missing 'kernel' function in code"),
                                (LocalExecutor(), "Missing 'kernel' function")):
            with self.assertRaises(SyntaxError) as caught:
                backend.compile("value = 1\n")
            self.assertEqual(message, str(caught.exception))

    def test_a_raw_syntax_error_is_reraised_as_is(self):
        with self.assertRaises(SyntaxError) as caught:
            self.gpu.compile("def kernel(:\n    pass\n")
        self.assertIn("invalid syntax", str(caught.exception))
        # 行号/列号也要还在（跨进程重建时最容易只剩 str()）
        self.assertEqual(1, caught.exception.lineno)

    def test_other_exceptions_are_wrapped_by_mock_gpu_only(self):
        with self.assertRaises(RuntimeError) as caught:
            self.gpu.compile("undefined_thing_xyz\n")
        self.assertIn("Compile error:", str(caught.exception))
        # 反直觉但**是原行为**：`TypeError("'kernel' is not callable")` 也被包成 RuntimeError
        with self.assertRaises(RuntimeError) as caught:
            self.gpu.compile("kernel = 3\n")
        self.assertIn("'kernel' is not callable", str(caught.exception))
        # `LocalExecutor` 不做包装：候选抛什么类型就是什么类型
        with self.assertRaises(NameError):
            LocalExecutor().compile("undefined_thing_xyz\n")

    def test_the_triton_static_branch_never_execs(self):
        """无真 Triton 环境那条路**一次 exec 都不做**（搬走前后都如此）

        pin 住它是为了防「顺手统一」：这条路过去对模块级 NameError / 死循环是**通过**的，
        一旦被塞进 `run_compile`，那些候选会从过门变成不过门 —— 行为变更必须是有意的。
        """
        with mock.patch.object(mock_gpu_module, "_has_cuda", return_value=False):
            self.gpu._triton_available = True
            decorated = TRITON_SRC.replace(
                "def kernel(x):", "undefined_at_module_level(A)\ndef kernel(x):")
            self.assertIs(True, self.gpu.compile(decorated), "静态分支不该 exec")
            with self.assertRaises(SyntaxError) as caught:
                self.gpu.compile("import triton\n@triton.jit\nx = 1\n")
            self.assertEqual("Triton code must define a 'kernel' wrapper function",
                             str(caught.exception))

    def test_the_real_triton_branch_goes_through_the_guard_with_preloads(self):
        """真 Triton 那条支路必须走 `run_compile`，且把那三个模块名预置进去"""
        fake = GuardResult(status=STATUS_OK,
                           metrics={"has_kernel": True, "kernel_callable": True})
        with mock.patch.object(mock_gpu_module, "run_compile", return_value=fake) as called:
            with mock.patch.object(mock_gpu_module, "_has_cuda", return_value=True):
                self.gpu._triton_available = True
                self.assertIs(True, self.gpu.compile(TRITON_SRC))
        self.assertEqual(1, called.call_count)
        self.assertEqual(TRITON_COMPILE_IMPORTS, called.call_args.kwargs.get("imports"))


class TestTheLimitsChainReachesTheGuard(unittest.TestCase):
    """阶段级上限的**消费端**：把墙钟压到 400ms，看悬挂候选多久被拦下

    这条链是「调用方 → 后端 → `exec_guard`」。钉行为而不是钉字段：`limits_for` 的两个
    默认值不同（编译那一档要装得下子进程自己的 import），所以「传进去了」必须能看见。
    """

    def test_mock_gpu_passes_its_compile_limits(self):
        gpu = MockGPU(compile_limits=Limits(wall_ms=400))
        started = time.time()
        with self.assertRaises(GuardRefusal) as caught:
            gpu.compile(HANG)
        wall = time.time() - started
        self.assertEqual("timeout", caught.exception.status)
        self.assertEqual("wall", caught.exception.kind)
        self.assertLess(wall, 10.0, "上限没传到执行器：它按默认值跑（或干脆没超时）")

    def test_local_executor_passes_its_compile_limits(self):
        local = LocalExecutor(compile_limits=Limits(wall_ms=400))
        started = time.time()
        with self.assertRaises(GuardRefusal) as caught:
            local.compile(HANG)
        self.assertEqual("timeout", caught.exception.status)
        self.assertLess(time.time() - started, 10.0)

    def test_the_conductor_hands_its_limits_to_the_backend_it_builds(self):
        compile_limits, exec_limits = Limits(wall_ms=1234), Limits(wall_ms=5678)
        conductor = Conductor(compile_limits=compile_limits, execution_limits=exec_limits)
        self.assertIs(compile_limits, conductor.backend.compile_limits)
        self.assertIs(exec_limits, conductor.backend.exec_limits)

    def test_an_injected_backend_keeps_its_own_policy(self):
        """注入后端时策略属于那个后端：Conductor 再塞一份就会出现两个地方改同一个数"""
        injected = MockGPU()
        conductor = Conductor(backend=injected, compile_limits=Limits(wall_ms=1234))
        self.assertIsNone(injected.compile_limits)
        self.assertIs(injected, conductor.backend)

    def test_no_limits_means_the_module_default(self):
        """`None` 不是「没有上限」：默认值在 `exec_guard` 里只有一处"""
        from operator_agent.execute.exec_guard import (COMPILE_LIMITS, MODE_COMPILE,
                                               MODE_RUN, limits_for)
        self.assertIsNone(MockGPU().compile_limits)
        self.assertEqual(MODE_RUN, "run")
        self.assertEqual(MODE_COMPILE, "compile")
        self.assertEqual(COMPILE_LIMITS.wall_ms, limits_for(MODE_COMPILE).wall_ms)


class TestTheVerifierPublishesTheGuardVerdict(unittest.TestCase):
    """`compile_status` 是「编译失败」的**分类**：执行器拦下 ≠ 候选自己编译失败"""

    def test_a_compile_timeout_is_published_as_its_own_status(self):
        verifier = Verifier(MockGPU(compile_limits=Limits(wall_ms=400)))
        result = verifier._verify_compile(HANG, "python", [["A"]])
        self.assertFalse(result.passed)
        metrics = result.metrics or {}
        self.assertEqual("compile_timeout", metrics.get("compile_status"))
        self.assertEqual("wall", metrics.get("exec_guard_kind"))
        self.assertIn("被 §22.2 拦下", result.error_log)

    def test_the_failure_class_is_not_unknown(self):
        """签名落成 `L1|unknown|-` 就等于「说不清是什么错」—— 它必须有自己的桶"""
        text = "编译失败: 编译期被 §22.2 拦下 (timeout/wall): 墙钟 400ms 到点，进程组已被杀"
        self.assertEqual("exec_guard_refused", failure_class_of("L1", text))

    def test_the_corpus_honesty_probe_reads_it_as_local(self):
        """`benchmarks/verify/verify_corpus_honesty.py::origin_of` 的默认支是 **remote**

        漏认这一族 ⇒ 一次本地编译超时会被那条探针读成「远端报的」，结论方向正好相反。
        """
        import importlib.util
        path = (Path(__file__).resolve().parents[2] / "benchmarks"
                / "verify" / "verify_corpus_honesty.py")
        spec = importlib.util.spec_from_file_location("_verify_corpus_honesty", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class _Result:
            metrics = {}

        local = _Result()
        local.metrics = {"compile_status": "compile_timeout"}
        self.assertEqual("local", module.origin_of(local))
        remote = _Result()
        remote.metrics = {"compile_status": "success"}
        remote.message = "远端编译失败"
        self.assertEqual("remote", module.origin_of(remote))


if __name__ == "__main__":
    unittest.main(verbosity=2)
