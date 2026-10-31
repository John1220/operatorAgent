"""执行侧那一格:「没听到回答」不许和「听到了失败」住在同一格 (#102 第 1 片, **只观测**)

一手依据 (doc 60 §三~§四, Limbo `2609.29095v1` 行 220-232，逐字)：

    "s ∈ {absent, committed, late, partial, duplicated}"
    "A timeout is observation-equivalent under absent, committed (response lost) and late
     (request still in flight). A recovery policy therefore cannot condition on the outcome
     state directly."

**一次超时在那三种世界里对客户端完全同形**。我方此前两条路都没走：
`SSHExecutor.execute` 的 `except Exception as e: return RemoteExecutionResult(success=False,
error=str(e))` 把 `subprocess.TimeoutExpired` 收成一个失败，`Verifier._verify_single_set` 再收成
`message="执行失败: …"` —— 于是「这台卡可能已经跑过一遍」与「跑完了但输出不对」在判决层同形，
而 Fixer 会对着前者给候选开处方。

这片**不改判决**：那些轮今天照样判 L2 失败。它做的是一件事 —— 让「未知」在血缘里第一次有名字，
于是 #102 第 2/3 片（幂等清点、异常原文过落盘边界）有一句可以离线问的话。

三条判据形状，每条对着本仓已经犯过的错：
  · **两格不许折叠**：`answered` / `outcome_unknown` 是两个值，而「这一级压根没走到远程执行」
    是**第三种**情况（血缘里那格是 `None`）。`TestThroughARealRun` 里第 2 轮就是它 ——
    把 None 折成 `answered` 是「拿缺证据当通过」(#53 那一族)。
  · **保守方向**：只把「确定没听到回答」那一族异常判成 unknown（见 `_NO_RESPONSE_EXCEPTIONS`）。
    误报一次 unknown 只是多问一句；漏报一次是把「这张卡可能已经跑过」擦掉。所以既有
    `RuntimeError` → `answered` 那一条，也专门有 **rc=1 → `answered`** 那条反向对照：
    「一切非零 rc 都算传输层丢了」这种写法能杀掉本文件其余整组的用例。
  · **有报才发布**：远端载荷里没有这一格时，`metrics` 里也不许出现（与 `aggregated_over`
    同一纪律）。这一片不能反过来造一个「answered」出来。

⚠️ 一处**已登记的漏报**（不是本片引入的，也不在这片修）：`_ssh_exec` 在 `rc != 0` 但 stdout
非空时**把 rc 丢掉**并返回 stdout（`remote_executor.py` 里那段 "# [_ssh_exec rc=… diagnostics]"）。
那条路上一次真正的传输失败会走 JSON 解析、抛 `ValueError`、被判成 `answered`。要杀掉它得先把
rc 穿过那个 return，而那要改 `_ssh_exec` 的返回类型 —— 与本片的「只观测不改判决」不是一件事。

⚠️ 写报表那组用例时的学费：`format_report` 自己也有一道**分母门**（运行数或轮数 < 4 时印
「分母不足」并直接 return），所以合成批次必须凑够 4 次运行 —— 少了它，那三条用例红的原因是
那道门而不是本栏，第一次跑就是这样红的。与 I25 那一族同形：断言落在一段**根本没被执行到**的
输出上时，"它为什么红" 与 "它为什么绿" 都读不出来。

变异标定（2026-09-30 实跑，基线 33 passed，**10/10 全部拦住、无逃逸**；表在
`benchmarks/mut_specs/transport_outcome_cell.py`，每条只改一处、跑完立刻逐字节复原并断言 sha256
一致。括号里是**实测**红数，不是写表时预计的）：

    T1  分类器不看远端 rc                          → 2 红   ← 我以为 3, `absent_rc` 那条对它是绿的
    T2  把「答了失败」折进「没答话」                 → 5 红
    T3  `_ssh_exec` 不把 rc 挂到异常上              → 1 红
    T4  四个出口里漏一个带格 (verify_function)      → 1 红
    T5  verifier 远程那路缺格也填 answered           → 1 红  ← 造证据那一侧
    T6  verifier 本地那路换回直接取属性              → 1 红  ← 本片撞出的**真崩溃**
    T7  `_round_transport` 取第一格而不是 L2         → 2 红
    T8  血缘那一格缺值时兜成 answered                → 5 红
    T9  报表把「这一轮没报」并进 answered             → 3 红
    T10 报表删掉「另有 N 轮是空的」那一支             → 2 红

T2 与 T8、T9 与 T10 各是一对**方向相反**的退化（折叠两格 / 把缺证据当通过 / 把没检查当检查过），
四侧都有证人 —— 只测一侧的判据在上一片那条 M8/M10 的教训里已经付过学费了。

运行: PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest tests/unit/test_transport_outcome_cell.py -q
"""
from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.drivers.harness_pass_rate import (aggregate, format_report,  # noqa: E402
                                          record_rounds)
from operator_agent.execute import remote_executor as rx  # noqa: E402
from operator_agent.app.conductor import Conductor, _round_transport  # noqa: E402
from operator_agent.core.kernel_db import KernelDatabase  # noqa: E402
from operator_agent.decide.measurement import (  # noqa: E402
    SSH_TRANSPORT_EXIT_CODE, TRANSPORT_ANSWERED, TRANSPORT_KEY, TRANSPORT_STATES,
    TRANSPORT_UNKNOWN, transport_outcome, transport_outcome_of_result)
from operator_agent.execute.mock_gpu import MockGPU  # noqa: E402
from operator_agent.execute.remote_executor import RemoteExecutionResult, SSHExecutor  # noqa: E402
from operator_agent.core.schemas import (  # noqa: E402
    ConstraintSpec, Precision, SourceSpec, TargetSpec, TaskComplexity, TaskSchema,
    TaskType, VerificationLevel, VerificationResult)
from operator_agent.decide.verifier import Verifier  # noqa: E402
from tests.backend_fakes import DeclaredFakeBackend  # noqa: E402


def _result(level=VerificationLevel.L2_FUNCTION, passed=False, metrics=None):
    return VerificationResult(level=level, passed=passed, message="m",
                              metrics=dict(metrics or {}))


class TestTheClassifier(unittest.TestCase):
    """判据本身: 哪些异常属于「没听到回答」, 哪些属于「答了, 答的是失败」"""

    def test_timeout_is_not_an_answer(self):
        self.assertEqual(TRANSPORT_UNKNOWN,
                         transport_outcome(subprocess.TimeoutExpired("ssh", 5)))

    def test_the_no_response_family_is_one_cell(self):
        """逐个点名那一族 —— 少列一个就是「漏报」, 而漏报的方向是擦掉「可能已经跑过」"""
        import http.client
        import socket
        import urllib.error
        cases = [socket.timeout("t"), TimeoutError("t"),
                 ConnectionError("c"), ConnectionResetError("c"),
                 ConnectionAbortedError("c"), BrokenPipeError("c"),
                 http.client.RemoteDisconnected("c"),
                 http.client.IncompleteRead(b""),
                 urllib.error.URLError("u")]
        for exc in cases:
            self.assertEqual(TRANSPORT_UNKNOWN, transport_outcome(exc), type(exc).__name__)

    def test_a_subclass_is_caught_by_its_mro_not_by_its_own_name(self):
        """分类器沿 MRO 找类名, 所以传输层的子类不用逐个登记。

        这条同时钉住实现方式: 写成 `type(exc).__name__ in _NO_RESPONSE_EXCEPTIONS`
        只对**恰好同名**的异常有效, 下面这个子类就会读成 answered —— 而那正是一次漏报。
        """
        class _DownstreamOfConnectionReset(ConnectionResetError):
            pass

        self.assertEqual(TRANSPORT_UNKNOWN, transport_outcome(_DownstreamOfConnectionReset()))

    def test_a_refusal_from_the_far_end_is_an_answer(self):
        """「答了, 答的是失败」不是「没答话」 —— T2 的证人, 也是本片存在理由的反面"""
        for exc in (RuntimeError("远端说误差 3e-2 超标"), ValueError("bad json"),
                    PermissionError("no such kernel"), TypeError("kernel() missing arg")):
            self.assertEqual(TRANSPORT_ANSWERED, transport_outcome(exc), type(exc).__name__)

    def test_ssh_exit_255_is_unknown_even_when_the_exception_type_says_otherwise(self):
        """rc=255 是 ssh **自己**失败的保留码, 与异常类型无关: 远端命令可能根本没跑"""
        self.assertEqual(SSH_TRANSPORT_EXIT_CODE, 255)     # 常量与它唯一的读者一处定义
        self.assertEqual(TRANSPORT_UNKNOWN,
                         transport_outcome(RuntimeError("SSH exec failed"), remote_rc=255))

    def test_a_nonzero_return_code_from_the_far_end_is_not_a_transport_loss(self):
        """**反向对照**: 远端跑完了并返回 1/127 是「听到了回答」

        没有这条, 「凡是 rc!=0 都算 unknown」这种写法能杀掉本文件其余用例的大半。
        """
        for rc in (1, 2, 126, 127):
            self.assertEqual(TRANSPORT_ANSWERED,
                             transport_outcome(RuntimeError("SSH exec failed"), remote_rc=rc),
                             f"rc={rc}")

    def test_absent_rc_does_not_become_unknown(self):
        """`ssh_rc` 没挂上 (本地执行 / 别的传输层) 时按 answered 读, 不猜成 unknown"""
        self.assertEqual(TRANSPORT_ANSWERED,
                         transport_outcome(RuntimeError("x"), remote_rc=None))

    def test_the_vocabulary_has_exactly_two_cells(self):
        """产它的与读它的共用这两个字符串; 第三个值 (None) 的含义是「没报」, 不在词表里"""
        self.assertEqual({"answered", "outcome_unknown"}, set(TRANSPORT_STATES))
        self.assertEqual(TRANSPORT_KEY, "transport_outcome")


class TestHeardVsNoAnswerThroughResults(unittest.TestCase):
    """从**执行结果对象**那一侧读: 手里有结果就说明听到了回答"""

    def test_a_result_in_hand_is_an_answer_even_without_the_cell(self):
        """MockGPU 的 `ExecutionResult` 与测试假件都不带这个字段。

        直接 `result.transport_outcome` 会把一条「候选算错了」的判决路径换成 `AttributeError`
        —— 而这条路径今天**零证人** (T6 就是这么撞出来的)。有 result 在手 = 传输层给回了东西,
        它落在 Limbo 那三种等价世界之外: 「没答话」那一侧从来不表现为一份结果, 只表现为异常。
        """
        bare = SimpleNamespace(success=False, error="boom")
        self.assertEqual(TRANSPORT_ANSWERED, transport_outcome_of_result(bare))

    def test_a_result_that_reports_unknown_is_not_overridden(self):
        r = RemoteExecutionResult(success=False, error="t", transport_outcome=TRANSPORT_UNKNOWN)
        self.assertEqual(TRANSPORT_UNKNOWN, transport_outcome_of_result(r))

    def test_the_dataclass_field_name_is_the_key_the_reader_uses(self):
        """字段名与 `TRANSPORT_KEY` 必须**同源**: 改了字段名之后 getattr 永远读不到,
        每一次远端超时都会被安静地读成 answered —— 而这一片的全部意义就是那一格不丢。"""
        names = {f.name for f in dataclasses.fields(RemoteExecutionResult)}
        self.assertIn(TRANSPORT_KEY, names)
        self.assertEqual(TRANSPORT_ANSWERED,
                         RemoteExecutionResult().transport_outcome,
                         "默认值选不报警那一侧: 能构造出这个对象就说明手里有一份远端返回")


class TestExecutorExits(unittest.TestCase):
    """四个失败出口都要带这一格 —— 少一个不是「少一栏」, 是那一格的读数在那个场合永远是空的"""

    def _ssh(self):
        ex = SSHExecutor(host="unit-test-host", user="unit-test-user")
        ex._deployed = True          # 跳过 deploy(): 本片只问「失败之后带没带那一格」
        return ex

    def test_execute_over_a_timeout_is_unknown(self):
        ex = self._ssh()
        ex._ssh_exec_json = mock.Mock(side_effect=subprocess.TimeoutExpired("ssh", 1))
        out = ex.execute("def kernel(A): return A", {"A": np.ones((4, 4), np.float32)})
        self.assertFalse(out.success)
        self.assertEqual(TRANSPORT_UNKNOWN, out.transport_outcome)

    def test_verify_function_over_a_reset_connection_is_unknown(self):
        ex = self._ssh()
        ex._ssh_exec_json = mock.Mock(side_effect=ConnectionResetError("reset by peer"))
        out = ex.verify_function("def kernel(A): return A", {"A": np.ones((4, 4), np.float32)},
                                 "matmul", 1e-3)
        self.assertFalse(out["success"])
        self.assertEqual(TRANSPORT_UNKNOWN, out[TRANSPORT_KEY])

    def test_benchmark_over_a_refused_run_is_answered(self):
        """远端答了「编译失败」= 听到了回答。这一格不能因为「反正是失败」就一律写 unknown"""
        ex = self._ssh()
        ex._ssh_exec_json = mock.Mock(side_effect=RuntimeError("compile error on remote"))
        out = ex.benchmark("def kernel(A): return A", {"A": np.ones((4, 4), np.float32)})
        self.assertEqual(TRANSPORT_ANSWERED, out[TRANSPORT_KEY])

    def test_baseline_benchmark_carries_the_cell_too(self):
        """第四个出口: 它是 L3 那条, 与 L2 的三路各有各的 except"""
        ex = self._ssh()
        ex._ssh_exec_json = mock.Mock(side_effect=subprocess.TimeoutExpired("ssh", 1))
        out = ex.benchmark_with_baselines("def kernel(A): return A",
                                          {"A": np.ones((4, 4), np.float32)}, "matmul")
        self.assertEqual(TRANSPORT_UNKNOWN, out[TRANSPORT_KEY])

    def _proc(self, rc):
        return SimpleNamespace(returncode=rc, stdout="", stderr="Connection to host closed.")

    def test_ssh_exit_255_reaches_the_verdict_as_unknown(self):
        """整条路: subprocess 返回 255 → `_ssh_exec` 把 rc 挂在异常上 → execute 判成 unknown

        这条是 T3 的证人 (删掉 `err.ssh_rc = proc.returncode` 之后 255 与 1 就同形了),
        也是「异常类型不换」那句注释的证人: `_ssh_exec` 抛的仍是 `RuntimeError`, 因为文件里
        有三处 `except RuntimeError` 靠这个类型走探测分支。
        """
        ex = self._ssh()
        with mock.patch.object(rx.subprocess, "run", return_value=self._proc(255)):
            with self.assertRaises(RuntimeError) as caught:
                ex._ssh_exec("true")
            self.assertEqual(255, caught.exception.ssh_rc)
            out = ex.execute("def kernel(A): return A", {"A": np.ones((4, 4), np.float32)})
        self.assertEqual(TRANSPORT_UNKNOWN, out.transport_outcome, f"execute error: {out.error}")

    def test_a_real_remote_failure_over_the_same_transport_stays_answered(self):
        """**反向对照**: 同一条 ssh 路上 rc=1 是「远端答了失败」, 不许和 255 挤成一格"""
        ex = self._ssh()
        with mock.patch.object(rx.subprocess, "run", return_value=self._proc(1)):
            out = ex.execute("def kernel(A): return A", {"A": np.ones((4, 4), np.float32)})
        self.assertEqual(TRANSPORT_ANSWERED, out.transport_outcome)


class TestVerifierPublishes(unittest.TestCase):
    """发布那一侧: 载荷报了才带出去; 载荷没报时**不许**造一个 answered"""

    @staticmethod
    def _task():
        return TaskSchema(
            task_id="transport-cell", task_type=TaskType.OPERATOR_GENERATION,
            source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.LOW),
            target=TargetSpec(hardware="mock_gpu", dsl="python"),
            constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=0.0,
                                       max_error=1e-3))

    @staticmethod
    def _inputs():
        return {"A": np.ones((8, 8), np.float32), "B": np.ones((8, 8), np.float32)}

    def _remote(self, payload):
        backend = DeclaredFakeBackend(caps=(), verify_function=lambda *a, **kw: payload)
        v = Verifier(backend)
        with contextlib.redirect_stdout(io.StringIO()):
            return v._verify_single_set("def kernel(A, B): return A @ B",
                                        self._inputs(), None, self._task())

    def test_a_remote_unknown_is_published_into_metrics(self):
        r = self._remote({"success": False, "error": "timed out", TRANSPORT_KEY: TRANSPORT_UNKNOWN})
        self.assertFalse(r.passed)
        self.assertEqual(TRANSPORT_UNKNOWN, r.metrics[TRANSPORT_KEY])

    def test_a_remote_that_says_nothing_about_transport_is_not_answered_on_its_behalf(self):
        """T5 的证人: 缺的键不填默认值。

        远端脚本今天**不报**这一格 —— 载荷里那一格只能来自执行侧的 except。写成
        `remote_verify.get(KEY, ANSWERED)` 之后「远端答了失败」与「这一级没区分」在血缘里
        就同形了, 而这正是本片要拆开的那一格。
        """
        r = self._remote({"success": False, "error": "no such file"})
        self.assertNotIn(TRANSPORT_KEY, r.metrics, r.metrics)

    def test_the_numeric_branch_is_untouched_by_this_slice(self):
        """判决没被这一片动过: 误差超标那条支路的 message/passed 与从前一样

        这一条也是「只观测」那句承诺的证人 —— 把那一格接进这条支路 (或据此改判决) 都会让它红。
        """
        r = self._remote({"success": False, "max_error": 3e-2, "error": "too far"})
        self.assertFalse(r.passed)
        self.assertIn("远程功能验证失败", r.message)
        self.assertNotIn(TRANSPORT_KEY, r.metrics)

    def test_a_local_failure_does_not_raise_and_is_published_as_answered(self):
        """本机执行 (MockGPU 没有 REMOTE_VERIFY) 失败那一路 —— T6 撞出来的那个崩溃

        `transport_outcome` 直接取属性时, 一份 `ExecutionResult` 会抛 `AttributeError`,
        于是「候选算错了」这个判决变成一个 harness 崩溃。
        """
        v = Verifier(MockGPU())
        bad = "def kernel(A, B):\n    raise RuntimeError('boom in kernel')\n"
        with contextlib.redirect_stdout(io.StringIO()):
            r = v._verify_single_set(bad, self._inputs(), np.ones((8, 8), np.float32),
                                     self._task())
        self.assertFalse(r.passed)
        self.assertEqual(TRANSPORT_ANSWERED, r.metrics[TRANSPORT_KEY])
        self.assertIn("boom in kernel", r.error_log)


class TestRoundCell(unittest.TestCase):
    """`_round_transport`: 一轮里那一格取的是 L2 那格, 取不到就是「没报」"""

    def test_the_l2_cell_is_returned_verbatim(self):
        self.assertEqual(TRANSPORT_UNKNOWN,
                         _round_transport([_result(metrics={TRANSPORT_KEY: TRANSPORT_UNKNOWN})]))

    def test_a_round_that_reported_nothing_is_none_not_answered(self):
        self.assertIsNone(_round_transport([_result(metrics={})]))

    def test_a_round_that_never_reached_l2_is_none(self):
        self.assertIsNone(_round_transport([_result(level=VerificationLevel.L1_COMPILE)]))
        self.assertIsNone(_round_transport([]))

    def test_only_the_l2_row_counts(self):
        """逐键判定不连坐: L1 那格被塞了值而 L2 没报 ⇒ 读出来仍是 None

        「取第一格」这种写法 (T7) 会在这里红, 而在真实循环里它读到的通常是 L1 ——
        于是报表那一栏印的其实是「编译那一步有没有听到回答」。
        """
        rows = [_result(level=VerificationLevel.L1_COMPILE,
                        metrics={TRANSPORT_KEY: TRANSPORT_UNKNOWN}), _result(metrics={})]
        self.assertIsNone(_round_transport(rows))


class TestThroughARealRun(unittest.TestCase):
    """真跑 Conductor + MockGPU: 注入的格子要穿过落盘边界, 而**判决不许被它动过**

    为什么不直接调 `_round_transport` 就完事 (P7 那一课的原文: 「变异第一版 0 红是真逃逸 ——
    用例都直接调 helper, 没有一条从真实循环里查过『血缘有没有这一格』」): 这一片的全部价值在
    「conductor 真的把它写进了 result.json」, 那一格在 helper 层面绿而在线上没有, 是本仓
    「算了没人读」那一族的正面版本。
    """

    PYTHON_MATMUL = "import numpy as np\n\ndef kernel(A, B):\n    return A @ B\n"
    SKETCH = {"algorithm": "matmul",
              "tiling": {"block_m": 32, "block_n": 32, "block_k": 16},
              "memory_plan": {}, "pipeline": [], "optimization_hints": []}

    class _Llm:
        def chat(self, messages, **kw):
            return TestThroughARealRun.PYTHON_MATMUL

        def chat_json(self, messages, **kw):
            return dict(TestThroughARealRun.SKETCH)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _run(self, inject_unknown_on_first_call=False):
        c = Conductor(backend=MockGPU(), kernel_db=KernelDatabase(jsonl_path=None),
                      llm=TestThroughARealRun._Llm(), max_optimize_rounds=2)
        c.store.output_dir = Path(self.tmp.name)
        calls = []
        orig = c.verifier._verify_function

        def wrapped(*a, **kw):
            res = orig(*a, **kw)
            calls.append(1)
            # 「第 1 轮那次远程执行没听到回答」—— 现场只能由执行侧的 except 产生, 这里注入的是
            # **同一个键同一个值**, 于是链路 (verifier metrics → _round_transport → 血缘) 全真
            if inject_unknown_on_first_call and len(calls) == 1:
                res.metrics[TRANSPORT_KEY] = TRANSPORT_UNKNOWN
            return res

        c.verifier._verify_function = wrapped
        inputs = {"A": np.ones((24, 16), np.float32), "B": np.ones((16, 24), np.float32)}
        task = TaskSchema(
            task_id="transport-case", task_type=TaskType.OPERATOR_GENERATION,
            source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.MEDIUM),
            target=TargetSpec(hardware="mock_gpu", dsl="python"),
            constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=5.0,
                                       max_error=1e-3))
        with contextlib.redirect_stdout(io.StringIO()):
            art = c.execute(task, inputs, None, target_gflops=1.0)
        saved = Path(art.saved_dir)
        return json.loads((saved / "result.json").read_text(encoding="utf-8"))["iteration_history"]

    def test_the_cell_survives_the_persistence_boundary(self):
        history = self._run(inject_unknown_on_first_call=True)
        self.assertGreaterEqual(len(history), 2, "构造前提: 这确实是跨轮的两次验证")
        self.assertEqual(TRANSPORT_UNKNOWN, history[0][TRANSPORT_KEY])

    def test_a_round_that_reported_nothing_stays_empty_in_the_saved_lineage(self):
        """第 2 轮没人报 → 落盘是 `null`, 不是 `answered` (T8 的证人, 过的是 JSON 那道边界)"""
        history = self._run(inject_unknown_on_first_call=True)
        self.assertIn(TRANSPORT_KEY, history[1])
        self.assertIsNone(history[1][TRANSPORT_KEY])

    def test_the_verdict_is_unchanged(self):
        """**本片只观测**: 那一格写成 unknown 之后, L2 该过的照样过

        这一条是「先观测后判定」那句承诺的证人。哪天有人拿它去判 L2 失败 (第 2 片之后的事),
        这里会红, 而那一次改判决需要单独的标定与真机前后对照。
        """
        for history in (self._run(), self._run(inject_unknown_on_first_call=True)):
            for rec in history[:2]:
                self.assertIs(True, rec["levels_passed"][VerificationLevel.L2_FUNCTION.value], rec)

    def test_a_plain_local_run_does_not_fabricate_the_cell(self):
        """正向对照 (不发放): 主路径上 MockGPU 从没走过远程那路 except ⇒ 每一轮这一格都该是空的

        少了这条, 「conductor 每轮无条件写 answered」也能让上面三条全绿 —— 而那样这一栏
        读起来就像「每次都听到了回答」。
        """
        history = self._run()
        self.assertTrue(all(r[TRANSPORT_KEY] is None for r in history), history)


class TestTheReportHasAReader(unittest.TestCase):
    """第二个读者: 端到端报表。三格并排、分母各带各的"""

    @staticmethod
    def _run(case, cells):
        """cells: 每轮血缘里那一格的值 (None = 这一轮没报, 与真产物同形)"""
        rounds = []
        for i, cell in enumerate(cells, 1):
            rounds.append({"round": i, "all_passed": False,
                           "levels_passed": {VerificationLevel.L1_COMPILE.value: True,
                                             VerificationLevel.L2_FUNCTION.value: True},
                           TRANSPORT_KEY: cell})
        return {"case": case, "ok": True, "rounds": rounds, "final_all_passed": False,
                "rounds_to_pass": None, "passed_via": None, "admitted": False,
                "admit_reason": ""}

    def test_the_three_cells_are_counted_separately(self):
        agg = aggregate([self._run("a", [TRANSPORT_UNKNOWN, TRANSPORT_ANSWERED, None])])
        self.assertEqual({TRANSPORT_UNKNOWN: 1, TRANSPORT_ANSWERED: 1}, agg["transport_counts"])
        self.assertEqual(2, agg["transport_reported"], "「没报」不进分母, 也不并进 answered (T9)")
        self.assertEqual(1, agg["rounds"] - agg["transport_reported"])

    def test_the_column_is_printed_with_each_denominator(self):
        # ⚠️ `format_report` 有自己的**分母门**: 运行数或轮数不足 4 时它直接印「分母不足」并
        # 在这里返回, 后面的每一栏都不会出现。合成批次必须凑够 4 次运行, 否则用例测的是那道门
        # 而不是本栏 (这一点是第一次跑就红出来的, 不是先验知道的)。
        text = format_report(aggregate([
            self._run("a", [TRANSPORT_UNKNOWN, TRANSPORT_ANSWERED]),
            self._run("b", [TRANSPORT_UNKNOWN, None]),
            self._run("c", [TRANSPORT_ANSWERED, None]),
            self._run("d", [TRANSPORT_ANSWERED, TRANSPORT_ANSWERED])]))
        self.assertIn("【执行侧「没听到回答」】", text)
        self.assertIn("outcome_unknown 2 · answered 4 · 报了这一格的轮 6/8", text)
        self.assertIn("另有 2 轮这一格是空的", text)
        self.assertIn("只观测不改判决", text)

    def test_an_all_silent_batch_is_not_reported_as_all_answered(self):
        """**反向对照**: 没有原料时那一栏要说出「空的」, 而不是印一个像结论的 0

        T10 与「把 `_silent` 那支删掉」都由这条杀掉: 只剩 `answered 0` 时读者会以为
        「查过了, 每次都听到了回答」—— 与 #53 的 `speedup=None` 印成「没有加速」同一族。
        """
        agg = aggregate([self._run(f"old-{i}", [None]) for i in range(1, 5)])
        self.assertEqual({}, agg["transport_counts"])
        self.assertEqual(0, agg["transport_reported"])
        text = format_report(agg)
        self.assertIn("另有 4 轮这一格是空的", text)
        self.assertIn("那不是「听到回答了」", text)

    def test_record_rounds_does_not_drop_the_cell(self):
        """落盘搬运那一步也不许把它当白名单外的键丢掉（doc 48 那批血缘永久没了的老病）"""
        out = record_rounds([{TRANSPORT_KEY: TRANSPORT_UNKNOWN, "round": 1}])
        self.assertEqual(TRANSPORT_UNKNOWN, out[0][TRANSPORT_KEY])


if __name__ == "__main__":
    unittest.main()
