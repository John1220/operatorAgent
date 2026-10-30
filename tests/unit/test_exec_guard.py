"""§22.2 的最小一块：**本地候选执行搬进受控子进程** —— 墙钟 / rlimit / 进程组 / 产物上限

设计原文（`docs/design/autonomous-operator-agent-design.md:1552-1564`）列了九项隔离，理由
是「候选代码在评测进程里运行时，它与 harness 共享地址空间、文件系统与网络」。
`operator_agent/exec_guard.py` 落地的是**本地这一半里能实测站得住的那几项**，其余
（网络默认关闭、独立 uid / 容器、文件系统白名单、GPU watchdog、设备租约回收）逐条登记为未落。

这一组用例钉六件事：
  ① **真的会超时被杀**，而且**杀的是整个进程组**（含孙子进程）——「超时」不许变成「留一堆孤儿」；
  ② **单文件大小 / 进程数**两条 rlimit 真的拦得住，且失败原因**带读数**；
  ③ **内存那一档如实报**：本机（macOS）`RLIMIT_AS` 内核 EINVAL，设不上就报 `not_enforced`
     ＋原始原因，**不许**报成「没有内存上限」也不许报成「拦住了」（本仓 I65/I11 那一族）；
     平台真设得上时（Linux）则必须真的拦住 —— 同一条用例两边都钉；
  ④ **产物字节数封顶**：超了是 `limit_exceeded`，**不是**截断后当通过；
  ⑤ **正常候选照旧**：返回值 / 耗时语义与搬走之前逐字一致，反作弊盾仍在子进程里生效
     （`lib_fallback` 证据跨进程带回来）；
  ⑥ **父进程不被污染**：候选改模块属性 / 环境变量 / cwd，harness 这边一个字节都不变
     —— 这是原先进程内 `exec` 做不到的。

⚠️ 本文件**不**证明、也不许被读成（与 `exec_guard` 模块 docstring 同一份边界）：
  · 「候选被隔离了」—— 同一个 uid、网络没关、没有容器；挡的是**跑飞/吃满/失控**；
  · 「内存跑飞拦得住」—— 本机设不上；
  · 「远端也被管住了」—— 远端 runner 仍是同 uid 落 `/tmp` 后 `exec_module`；
  · CUDA 执行路径（`MockGPU._execute_triton_cuda`）**没有**接进来。

跑: python3 -m pytest tests/unit/test_exec_guard.py -q
"""

import errno
import os
import signal
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from operator_agent import exec_guard as eg  # noqa: E402
from operator_agent.exec_guard import (  # noqa: E402
    COMPILE_LIMITS, ENFORCED, EXECUTOR_AUTO, EXECUTOR_FORK, EXECUTOR_SUBPROCESS,
    ITEMS, KIND_ARTIFACT, KIND_CPU, KIND_FILE_SIZE, KIND_MEMORY, KIND_PROCESSES,
    KIND_WALL, MODE_COMPILE, MODE_RUN, NOT_ENFORCED, STATUS_CRASHED,
    STATUS_LIMIT_EXCEEDED, STATUS_OK, STATUS_TIMEOUT, STATUSES, GuardRefusal,
    Limits, candidate_error, classify_death, classify_exception, limits_for,
    refusal_from, run_candidate, run_compile,
)

try:
    import torch
    HAS_TORCH = True
except ImportError:                       # pragma: no cover
    HAS_TORCH = False

A = np.arange(6, dtype=np.float64).reshape(2, 3)
B = np.arange(12, dtype=np.float64).reshape(3, 4)
PRODUCT = A @ B

NORMAL = """import numpy as np

def kernel(A, B):
    return A @ B
"""


def _sleeping_grandchild(marker: str, delay: float) -> str:
    """一段「起个孙子进程，它过一会儿写一个文件」的候选 —— 用来钉「孤儿」这件事。"""
    inner = ("import time, pathlib; time.sleep(%r); pathlib.Path(%r).write_text('survived')"
             % (delay, marker))
    return ("import subprocess, sys, time\n"
            "def kernel(A, B):\n"
            "    subprocess.Popen([sys.executable, '-c', %r])\n"
            "    time.sleep(30)\n"
            "    return A @ B\n" % inner)


class TestTheNormalPathIsUnchanged(unittest.TestCase):
    """⑤ 搬走执行方式**不许**改语义（这是接线的底线）"""

    def test_a_normal_candidate_still_works(self):
        r = run_candidate(NORMAL, {"A": A, "B": B})
        self.assertEqual(r.status, STATUS_OK, r.detail)
        self.assertIsNone(r.kind)
        self.assertEqual(r.exit_code, 0)
        self.assertIsNone(r.signal)
        np.testing.assert_allclose(np.asarray(r.output), PRODUCT)
        self.assertGreaterEqual(r.elapsed_ms, 0.0)
        self.assertGreater(r.artifact_bytes, 0)

    def test_mock_gpu_returns_the_same_shape_as_before(self):
        """接线点回归：`MockGPU.execute` 的产物形状与搬走之前一致"""
        from operator_agent.mock_gpu import MockGPU
        res = MockGPU().execute(NORMAL, {"A": A, "B": B})
        self.assertTrue(res.success, res.error)
        np.testing.assert_allclose(res.output, PRODUCT)
        self.assertEqual(res.language, "python")
        self.assertEqual(res.device, "cpu")
        self.assertGreater(res.execution_time_ms, 0.0)
        self.assertIn("exec_guard_executor", res.metrics)

    def test_the_candidate_cwd_is_the_scratch_dir(self):
        here = tempfile.mkdtemp(prefix="exec-guard-test-")
        self.addCleanup(lambda: __import__("shutil").rmtree(here, ignore_errors=True))
        r = run_candidate("def kernel(A, B):\n"
                          "    open('wrote-here.txt', 'w').write('x')\n"
                          "    return A @ B\n",
                          {"A": A, "B": B}, scratch_dir=here)
        self.assertEqual(r.status, STATUS_OK)
        self.assertTrue(os.path.exists(os.path.join(here, "wrote-here.txt")),
                        "候选的 cwd 不是它自己的 scratch 目录")

    def test_the_candidate_prints_come_back_in_a_bounded_log(self):
        r = run_candidate("def kernel(A, B):\n"
                          "    print('候选自己的一行输出')\n"
                          "    return A @ B\n", {"A": A, "B": B})
        self.assertIn("候选自己的一行输出", r.log)

    def test_a_print_bomb_is_capped_in_the_log(self):
        r = run_candidate("def kernel(A, B):\n"
                          "    for _ in range(20000):\n"
                          "        print('x' * 100)\n"
                          "    return A @ B\n",
                          {"A": A, "B": B}, limits=Limits(log_bytes=4096))
        self.assertEqual(r.status, STATUS_OK)
        self.assertLess(len(r.log), 4096 + 512, "日志没有被截住")
        self.assertIn("日志被截到", r.log)

    def test_a_temp_scratch_dir_is_removed_but_an_explicit_one_is_kept(self):
        auto = run_candidate(NORMAL, {"A": A, "B": B})
        self.assertFalse(os.path.exists(auto.scratch_dir), "默认的临时 scratch 目录没删")
        kept = tempfile.mkdtemp(prefix="exec-guard-test-")
        self.addCleanup(lambda: __import__("shutil").rmtree(kept, ignore_errors=True))
        r = run_candidate(NORMAL, {"A": A, "B": B}, scratch_dir=kept)
        self.assertTrue(os.path.exists(kept), "显式传进来的 scratch 目录不该被删")

    @unittest.skipUnless(HAS_TORCH, "torch 未安装")
    def test_a_dynamic_cheat_is_still_caught_by_the_shield_in_the_child(self):
        """盾现在装在**子进程**里 ⇒ 命中的证据必须跨进程带回来（否则规则 4 又会恒 False）"""
        from operator_agent.mock_gpu import MockGPU
        res = MockGPU().execute("""import torch

def kernel(A, B):
    fn = getattr(torch, "mat" + "mul")
    return fn(A, B)
""", {"A": A, "B": B})
        self.assertFalse(res.success)
        self.assertIs(res.metrics.get("lib_fallback"), True,
                      f"跨进程丢了盾的证据: {res.metrics}")
        self.assertEqual(res.metrics.get("exec_guard_status"), "cheat")
        self.assertIn("反作弊", res.error)


class TestTheWallClockTimeout(unittest.TestCase):
    """① 超时 = 杀掉整个进程组，不留孤儿"""

    def test_a_dead_loop_is_killed_by_the_wall_clock(self):
        r = run_candidate("def kernel(A, B):\n    while True:\n        pass\n",
                          {"A": A, "B": B}, limits=Limits(wall_ms=400))
        self.assertEqual(r.status, STATUS_TIMEOUT)
        self.assertEqual(r.kind, KIND_WALL)
        self.assertIn("400", r.detail)                      # 读数进 detail
        self.assertGreaterEqual(r.wall_ms, 380.0)
        self.assertIn(r.signal, (signal.SIGTERM, signal.SIGKILL))

    def test_the_whole_process_group_is_killed_including_a_grandchild(self):
        """两个场景一次等：墙钟被杀 / 正常返回，都不许把孙子进程留下"""
        work = tempfile.mkdtemp(prefix="exec-guard-orphan-")
        self.addCleanup(lambda: __import__("shutil").rmtree(work, ignore_errors=True))
        on_timeout = os.path.join(work, "survived-timeout.txt")
        on_clean = os.path.join(work, "survived-clean.txt")

        r1 = run_candidate(_sleeping_grandchild(on_timeout, 1.0), {"A": A, "B": B},
                           limits=Limits(wall_ms=300, processes=None), scratch_dir=work)
        self.assertEqual(r1.status, STATUS_TIMEOUT, r1.detail)
        r2 = run_candidate("import subprocess, sys, time\n"
                           "def kernel(A, B):\n"
                           "    subprocess.Popen([sys.executable, '-c', %r])\n"
                           "    return A @ B\n" % (
                               "import time, pathlib; time.sleep(1.0); "
                               "pathlib.Path(%r).write_text('survived')" % on_clean),
                           {"A": A, "B": B}, limits=Limits(processes=None))
        self.assertEqual(r2.status, STATUS_OK, r2.detail)

        time.sleep(2.5)                     # 孙进程若活着，这段时间足够它留下痕迹
        # 余量要留够（2026-10-06 实测教训）：孙进程是**新起一个解释器**再 sleep 1.0s，
        # 于是它落盘的时刻 = 解释器启动（本机 50~300ms，机器忙时更久）+ 1.0s。
        # 原来这里等 1.3s ⇒ 余量只有 0~0.25s：机器一忙，痕迹就晚于检查出现，
        # **变异表 M2 会被读成「逃逸」**（串行标定时真发生过一次），而那个逃逸是量具的
        # 余量不够，不是判据被绕过。2.5s 把余量拉到 ~1.2s 以上。
        self.assertFalse(os.path.exists(on_timeout),
                         "墙钟超时没把孙进程一起杀掉（留了孤儿）")
        self.assertFalse(os.path.exists(on_clean),
                         "候选正常返回后，它起的后台子进程被留下了（进程树清理只做了超时那一路）")

    def test_a_timeout_does_not_claim_the_limits_it_could_not_read(self):
        """被杀的子进程没机会回报 ⇒ 读不到的项报 not_enforced，不许默认它们设上了"""
        r = run_candidate("def kernel(A, B):\n    while True:\n        pass\n",
                          {"A": A, "B": B}, limits=Limits(wall_ms=300, memory_bytes=None))
        self.assertEqual(r.status, STATUS_TIMEOUT)
        self.assertEqual(r.enforcement_of("wall").state, ENFORCED)
        self.assertEqual(r.enforcement_of("artifact").state, ENFORCED)
        for item in ("cpu", "file_size", "processes"):
            self.assertEqual(r.enforcement_of(item).state, NOT_ENFORCED,
                             f"{item} 读不到却报成了 enforced")
            self.assertTrue(r.enforcement_of(item).reason)

    def test_the_cpu_limit_is_reported_as_its_own_kind(self):
        r = run_candidate("def kernel(A, B):\n    while True:\n        pass\n",
                          {"A": A, "B": B}, limits=Limits(wall_ms=6000, cpu_s=1))
        self.assertEqual(r.status, STATUS_LIMIT_EXCEEDED)
        self.assertEqual(r.kind, KIND_CPU)
        cpu = r.enforcement_of("cpu")
        self.assertEqual(cpu.state, ENFORCED)
        self.assertIn("SIGXCPU", cpu.reading)
        self.assertIn("SIGXCPU", r.detail)


class TestRlimitsAreReportedHonestly(unittest.TestCase):
    """②③④⑥ 拦得住的说拦得住，设不上的说设不上"""

    def test_a_huge_file_write_fails_with_a_reading(self):
        work = tempfile.mkdtemp(prefix="exec-guard-fsize-")
        self.addCleanup(lambda: __import__("shutil").rmtree(work, ignore_errors=True))
        r = run_candidate("def kernel(A, B):\n"
                          "    with open('big.bin', 'wb') as f:\n"
                          "        for _ in range(20):\n"
                          "            f.write(b'x' * 1048576)\n"
                          "            f.flush()\n"
                          "    return A @ B\n",
                          {"A": A, "B": B},
                          # 产物是经一个文件回传的 ⇒ 单文件上限要装得下产物预算
                          # （`Limits` 的构造期不变量拦着这种组合）
                          limits=Limits(file_size_bytes=1024 * 1024,
                                        artifact_bytes=256 * 1024),
                          scratch_dir=work)
        self.assertEqual(r.status, STATUS_LIMIT_EXCEEDED)
        self.assertEqual(r.kind, KIND_FILE_SIZE)
        self.assertIn("File too large", r.detail)          # 原始读数
        self.assertIn("maxrss", r.detail)                  # 失败原因**带读数**
        self.assertEqual(r.enforcement_of("file_size").state, ENFORCED)
        size = os.path.getsize(os.path.join(work, "big.bin"))
        self.assertLessEqual(size, 1024 * 1024,
                             f"文件没有被上限截住: {size} 字节")

    def test_the_process_limit_stops_a_mild_fork_bomb(self):
        r = run_candidate("import subprocess\n"
                          "def kernel(A, B):\n"
                          "    for _ in range(200):\n"
                          "        subprocess.Popen(['sleep', '5'])\n"
                          "    return A @ B\n",
                          {"A": A, "B": B}, limits=Limits(processes=4, wall_ms=5000))
        self.assertEqual(r.status, STATUS_LIMIT_EXCEEDED)
        self.assertEqual(r.kind, KIND_PROCESSES)
        self.assertIn("Resource temporarily unavailable", r.detail)
        self.assertIn("maxrss", r.detail)                  # 失败原因**带读数**
        self.assertEqual(r.enforcement_of("processes").state, ENFORCED)

    def test_the_memory_item_is_honest_on_every_platform(self):
        """平台设得上 ⇒ 必须拦住；设不上 ⇒ 如实报 `not_enforced` + 原始原因。

        本机（macOS）走的是后一条：`RLIMIT_AS` 内核直接 EINVAL（实测见模块 docstring）。
        两种平台都钉，所以这条用例不是「跳过就算过」。
        """
        r = run_candidate("import numpy as np\n"
                          "def kernel(A, B):\n"
                          "    x = np.zeros((512, 1024, 128), dtype=np.float64)\n"
                          "    x[:] = 1.0\n"
                          "    return A @ B\n",
                          {"A": A, "B": B},
                          limits=Limits(memory_bytes=64 * 1024 * 1024, wall_ms=8000))
        item = r.enforcement_of("memory")
        if item.state == ENFORCED:
            self.assertEqual(r.status, STATUS_LIMIT_EXCEEDED, r.detail)
            self.assertEqual(r.kind, KIND_MEMORY)
            self.assertIn("MB", r.detail)                  # 失败原因带读数
        else:
            self.assertIn("memory", r.not_enforced)
            self.assertTrue(item.reason, "设不上必须带原始原因，不能只是沉默")
            self.assertIn("Error", item.reason)
            self.assertNotEqual(r.status, STATUS_LIMIT_EXCEEDED,
                                "设不上却报成了「拦住了」")
            self.assertFalse(r.fully_enforced)

    def test_an_item_that_was_not_requested_is_not_reported_as_enforced(self):
        """`None` = **不请求**，不是「请求 0」，更不能被读成「已设上」"""
        r = run_candidate(NORMAL, {"A": A, "B": B},
                          limits=Limits(cpu_s=None, memory_bytes=None,
                                        file_size_bytes=None, processes=None))
        self.assertEqual(r.status, STATUS_OK)
        for item in ("cpu", "memory", "file_size", "processes"):
            e = r.enforcement_of(item)
            self.assertEqual(e.state, NOT_ENFORCED, f"{item} 没请求却报成了 enforced")
            self.assertIn("没有请求", e.reason)

    def test_every_enforcement_record_is_a_known_pair(self):
        r = run_candidate(NORMAL, {"A": A, "B": B})
        self.assertEqual(tuple(e.item for e in r.enforcement), ITEMS)
        for e in r.enforcement:
            self.assertIn(e.state, (ENFORCED, NOT_ENFORCED))

    def test_the_two_budgets_cannot_contradict_each_other(self):
        """产物是经一个文件回传的 ⇒ 单文件上限装不下产物预算时，**构造期**就失败

        2026-10-04 实测教训: 这两个数不一致时，子进程写到上限被截断，父侧只读到一句
        `UnpicklingError: pickle data was truncated` —— 看起来像「候选坏了」，
        实际是「两个上限打架」。可判定的组合就该在这里失败。
        """
        with self.assertRaises(ValueError) as ctx:
            Limits(file_size_bytes=1024 * 1024, artifact_bytes=8 * 1024 * 1024)
        self.assertIn("artifact_bytes", str(ctx.exception))
        Limits(file_size_bytes=8 * 1024 * 1024, artifact_bytes=8 * 1024 * 1024)  # 相等可以
        Limits(artifact_bytes=None, file_size_bytes=1024)                       # 不请求也可以

    def test_the_limits_cannot_be_raised_back_from_inside(self):
        """软硬上限一起降 ⇒ 上限在子进程里**不可逆**：候选自己 `setrlimit` 也抬不回去

        这条是「上限不是自愿的」那一半。**另一半必须一起读**：同一 uid 下候选仍能读 harness
        的文件、联网、改自己的 scratch —— 不可逆的是**上限**，不是整个隔离。
        """
        code = ("import resource\n"
                "def kernel(A, B):\n"
                "    try:\n"
                "        resource.setrlimit(resource.RLIMIT_FSIZE,\n"
                "                           (resource.RLIM_INFINITY, resource.RLIM_INFINITY))\n"
                "        return 'raised'\n"
                "    except (ValueError, OSError):\n"
                "        return 'refused'\n")
        r = run_candidate(code, {"A": A, "B": B},
                          limits=Limits(file_size_bytes=1024 * 1024,
                                        artifact_bytes=64 * 1024))
        self.assertEqual(r.status, STATUS_OK, r.detail)
        self.assertEqual(r.output, "refused",
                         "候选把单文件上限抬回去了 —— 硬上限没有跟着一起降")

    def test_the_exception_to_kind_mapping_is_pinned(self):
        """本机内存那档走不到，但**判据**仍然钉得住（唯一真源在父侧）"""
        self.assertEqual(classify_exception(MemoryError()), (STATUS_LIMIT_EXCEEDED, KIND_MEMORY))
        self.assertEqual(classify_exception(OSError(errno.ENOMEM, "Cannot allocate memory")),
                         (STATUS_LIMIT_EXCEEDED, KIND_MEMORY))
        self.assertEqual(classify_exception(BlockingIOError(errno.EAGAIN, "Resource unavailable")),
                         (STATUS_LIMIT_EXCEEDED, KIND_PROCESSES))
        self.assertEqual(classify_exception(OSError(errno.EFBIG, "File too large")),
                         (STATUS_LIMIT_EXCEEDED, KIND_FILE_SIZE))
        self.assertEqual(classify_exception(ValueError("别的错")), (STATUS_CRASHED, None))

    def test_classify_death_maps_signals_without_guessing(self):
        self.assertEqual(classify_death(None, signal.SIGXCPU, True),
                         (STATUS_TIMEOUT, KIND_WALL, "墙钟到点，进程组已被杀"))
        self.assertEqual(classify_death(None, signal.SIGXCPU, False),
                         (STATUS_LIMIT_EXCEEDED, KIND_CPU, "撞上 CPU 时间上限 (SIGXCPU)"))
        self.assertEqual(classify_death(None, signal.SIGXFSZ, False)[:2],
                         (STATUS_LIMIT_EXCEEDED, KIND_FILE_SIZE))
        self.assertEqual(classify_death(None, signal.SIGKILL, False)[:2],
                         (STATUS_CRASHED, None))
        self.assertIn("不是墙钟超时", classify_death(None, signal.SIGKILL, False)[2])
        self.assertEqual(classify_death(3, None, False)[:2], (STATUS_CRASHED, None))
        self.assertEqual(classify_death(None, None, False)[0], STATUS_CRASHED)


class TestTheArtifactCap(unittest.TestCase):
    """④ 产物是证据：超了要**失败**，截断的证据不是证据"""

    def test_a_too_large_artifact_fails_instead_of_being_truncated(self):
        r = run_candidate("import numpy as np\n"
                          "def kernel(A, B):\n"
                          "    return np.zeros((1024, 1024, 2))\n",
                          {"A": A, "B": B}, limits=Limits(artifact_bytes=256 * 1024))
        self.assertEqual(r.status, STATUS_LIMIT_EXCEEDED)
        self.assertEqual(r.kind, KIND_ARTIFACT)
        self.assertIsNone(r.output, "超限的产物不许被截断后当通过")
        self.assertIn(">", r.detail)
        self.assertGreater(r.max_rss_mb or 0.0, 0.0)       # 读数也带回来了

    def test_the_cap_does_not_hold_a_legitimate_output(self):
        big = np.zeros((512, 512), dtype=np.float64)       # 2 MiB
        r = run_candidate("import numpy as np\n"
                          "def kernel(A):\n"
                          "    return np.zeros((512, 512))\n",
                          {"A": big}, limits=Limits(artifact_bytes=8 * 1024 * 1024))
        self.assertEqual(r.status, STATUS_OK, r.detail)
        self.assertEqual(np.asarray(r.output).shape, (512, 512))


class TestTheChildIsItsOwnProcess(unittest.TestCase):
    """子进程的形状：五档状态分得开、后端是个读数、死法不靠猜"""

    def test_the_status_vocabulary_is_closed(self):
        r = run_candidate(NORMAL, {"A": A, "B": B})
        self.assertIn(r.status, STATUSES)
        self.assertIn(r.executor, (EXECUTOR_FORK, EXECUTOR_SUBPROCESS))
        self.assertEqual(r.as_dict()["executor"], r.executor)

    def test_the_candidate_is_marked_from_the_candidate_source(self):
        r = run_candidate("def kernel(A, B):\n    return A @ missing_name\n",
                          {"A": A, "B": B})
        self.assertEqual(r.status, STATUS_CRASHED)
        self.assertIn("missing_name", r.error)
        self.assertIn("<candidate>:2", r.detail)           # 指到候选自己那一帧

    def test_a_child_that_exits_hard_is_reported_as_crashed(self):
        r = run_candidate("import os\ndef kernel(A, B):\n    os._exit(3)\n",
                          {"A": A, "B": B})
        self.assertEqual(r.status, STATUS_CRASHED)
        self.assertEqual(r.exit_code, 3)
        self.assertIn("rc=3", r.detail)

    def test_a_child_killed_by_sigkill_is_not_reported_as_a_timeout(self):
        r = run_candidate("import os, signal\n"
                          "def kernel(A, B):\n"
                          "    os.kill(os.getpid(), signal.SIGKILL)\n",
                          {"A": A, "B": B}, limits=Limits(wall_ms=5000))
        self.assertEqual(r.status, STATUS_CRASHED)
        self.assertEqual(r.signal, signal.SIGKILL)
        self.assertNotEqual(r.status, STATUS_TIMEOUT)

    def test_a_candidate_that_calls_sys_exit_does_not_kill_the_harness(self):
        """`SystemExit` 是 BaseException —— 进程内 `exec` 时它会**穿出** `execute()`"""
        r = run_candidate("import sys\ndef kernel(A, B):\n    sys.exit(0)\n",
                          {"A": A, "B": B})
        self.assertEqual(r.status, STATUS_CRASHED)
        self.assertIn("SystemExit", r.detail)

    def test_explicit_subprocess_backend_is_honored_and_agrees(self):
        r = run_candidate(NORMAL, {"A": A, "B": B}, executor=EXECUTOR_SUBPROCESS)
        self.assertEqual(r.executor, EXECUTOR_SUBPROCESS)
        self.assertEqual(r.status, STATUS_OK, r.detail)
        np.testing.assert_allclose(np.asarray(r.output), PRODUCT)

    def test_auto_picks_subprocess_when_another_thread_is_running(self):
        """`auto` 的判据是可观察的一条：父进程此刻是不是只有主线程"""
        release = threading.Event()
        t = threading.Thread(target=release.wait, daemon=True)
        t.start()
        try:
            self.assertEqual(eg._pick_executor(EXECUTOR_AUTO), EXECUTOR_SUBPROCESS,
                             "多线程父进程里还选 fork —— fork 的死锁风险没人挡")
        finally:
            release.set()
            t.join(timeout=5)
        if hasattr(os, "fork"):
            self.assertEqual(eg._pick_executor(EXECUTOR_AUTO), EXECUTOR_FORK)
        self.assertEqual(eg._pick_executor(EXECUTOR_SUBPROCESS), EXECUTOR_SUBPROCESS)
        self.assertEqual(eg._pick_executor(EXECUTOR_FORK), EXECUTOR_FORK)

    def test_a_fork_failure_falls_back_instead_of_failing_the_candidate(self):
        """fork 自己起不来 ≠ 候选失败：退回新解释器，保证一模一样，只是慢"""
        if not hasattr(os, "fork"):
            self.skipTest("这个平台没有 os.fork")
        with mock.patch.object(eg.os, "fork",
                               side_effect=OSError(errno.EAGAIN, "no more processes")):
            r = run_candidate(NORMAL, {"A": A, "B": B})
        self.assertEqual(r.status, STATUS_OK, r.detail)
        self.assertEqual(r.executor, EXECUTOR_SUBPROCESS)
        np.testing.assert_allclose(np.asarray(r.output), PRODUCT)


class TestTheParentIsNotPolluted(unittest.TestCase):
    """⑥ 各自一份地址空间 ⇒ 候选改不动 harness —— 这是原先进程内 `exec` 做不到的"""

    def test_module_state_environment_and_cwd_do_not_leak(self):
        marker = "EXEC_GUARD_LEAK_MARKER"
        cwd_before = os.getcwd()
        zeros_before = np.zeros
        r = run_candidate("import os, numpy as np\n"
                          "def kernel(A, B):\n"
                          "    os.environ[%r] = '1'\n"
                          "    np.zeros = None\n"
                          "    os.chdir('/')\n"
                          "    return A @ B\n" % marker, {"A": A, "B": B})
        self.assertEqual(r.status, STATUS_OK, r.detail)
        self.assertNotIn(marker, os.environ, "候选改的环境变量漏进了 harness")
        self.assertIs(np.zeros, zeros_before, "候选改的模块属性漏进了 harness")
        self.assertEqual(os.getcwd(), cwd_before, "候选 chdir 漏进了 harness")

    def test_the_shield_does_not_stay_installed_in_the_parent(self):
        if not HAS_TORCH:
            self.skipTest("torch 未安装")
        before = torch.matmul
        r = run_candidate("import torch\ndef kernel(A, B):\n    return A @ B\n",
                          {"A": A, "B": B})
        self.assertEqual(r.status, STATUS_OK, r.detail)
        self.assertIs(torch.matmul, before, "子进程里的盾把父进程的 torch 也改了")


class TestTheSingleDoorRatchet(unittest.TestCase):
    """结构性守卫：`operator_agent/` 里**只有一扇门**做本地进程控制

    与 §22.1 的「唯一写入口」同一条纪律：本地候选执行的控制点必须只有一个地方，
    否则「执行方式被换回进程内 `exec`」这种改动不会有任何行为证人。
    """

    def test_only_exec_guard_controls_local_processes(self):
        root = Path(__file__).resolve().parents[2] / "operator_agent"
        hits = {}
        for path in sorted(root.glob("*.py")):
            if path.name == "exec_guard.py":
                continue
            text = path.read_text("utf-8")
            for needle in ("resource.setrlimit", "os.fork(", "os.killpg",
                           "start_new_session=True"):
                if needle in text:
                    hits.setdefault(path.name, []).append(needle)
        self.assertEqual({}, hits, f"本地进程控制出现了第二扇门: {hits}")

    def test_the_guard_entries_are_imported_by_name(self):
        """入口**名字**也要钉住：接线换成别的实现（或本地又写一个）就没人挡了

        （主 agent 提醒的一类：假件入口/替身入口没跟上生产入口，会让一批用例在空转里全绿。）

        2026-10-06 改成 AST 判据：编译那一档接上同一扇门之后，这一行从「一个名字」变成
        「一组名字」，字面子串判据只会随排版漂 —— 判据该看的是「这两个入口来自
        `exec_guard`」，不是那一行怎么写。
        """
        import ast
        path = Path(__file__).resolve().parents[2] / "operator_agent" / "mock_gpu.py"
        tree = ast.parse(path.read_text("utf-8"))
        imported = {alias.name: node.module
                    for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                    for alias in node.names}
        for name in ("run_candidate", "run_compile"):
            self.assertIn(name, imported,
                          f"受控执行入口 {name} 不再来自 exec_guard "
                          f"—— 复核「本地执行只有这一扇门」")
            self.assertEqual("exec_guard", imported[name])

    def test_the_guard_is_reachable_from_mock_gpu_only_through_exec_guard(self):
        """接线点的结构：`_execute_python` 里不许再出现进程内 `exec`"""
        import ast
        path = Path(__file__).resolve().parents[2] / "operator_agent" / "mock_gpu.py"
        tree = ast.parse(path.read_text("utf-8"))
        func = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "_execute_python")
        self.assertIn("run_candidate", ast.dump(func), "本地执行没有走受控入口")
        self.assertNotIn("exec", [n.id for n in ast.walk(func)
                                  if isinstance(n, ast.Name)] or [],
                         "_execute_python 里还留着进程内 exec")


class TestCompileMode(unittest.TestCase):
    """§22.2 第一项「编译超时」：编译也走同一个受控子进程（`run_compile`）

    编译期原先在 harness 进程里 `exec` —— 没有墙钟、没有 rlimit、没有进程组。这一组钉五件：
      ① 只 `exec`、**不调用** kernel（编译期本来就不该跑核）；
      ② 候选自己抛的异常**带类型**回来（`exc_type`），`candidate_error` 按原类型重建；
      ③ 墙钟真的管用（顶层死循环被整组杀掉，不是「等它跑完」）；
      ④ `has_kernel` / `kernel_callable` 由**子进程**回报（判据与文案留在调用方，只有一处）；
      ⑤ `imports` 预置的模块真的进了子进程命名空间（Triton 那条路的等价物）。

    ⚠️ 不许读成：编译期被隔离了（同一个 uid、网络没关）；也**不**表示编译产物与真机一致
    （这条只证「在受控子进程里 exec」这一个事实）。
    """

    def test_compile_mode_does_not_call_the_kernel(self):
        code = ("def kernel(**kwargs):\n"
                "    raise RuntimeError('编译期不该跑核')\n")
        guard = run_compile(code)
        self.assertEqual(STATUS_OK, guard.status, guard.detail)
        self.assertIs(True, guard.metrics.get("has_kernel"))
        self.assertIs(True, guard.metrics.get("kernel_callable"))
        # 同一份代码在**运行**模式下必须炸 —— 否则上面那条断言就是空转（没跑核也可能是因为没接线）
        ran = run_candidate(code, {})
        self.assertEqual(STATUS_CRASHED, ran.status)
        self.assertIn("编译期不该跑核", ran.error or "")

    def test_the_candidate_exception_type_survives_the_boundary(self):
        guard = run_compile("def kernel(:\n    pass\n")
        self.assertEqual(STATUS_CRASHED, guard.status)
        self.assertEqual("SyntaxError", guard.exc_type)
        rebuilt = candidate_error(guard)
        self.assertIsInstance(rebuilt, SyntaxError)
        self.assertIn("invalid syntax", str(rebuilt))
        # **字段也要一起活下来**, 不只是 `str()`: `msg/lineno/offset/text/filename` 是
        # `SyntaxError` 契约的一部分（`verifier.syntax_error_report()` 读行号与列号）。
        # 拿进程内 exec 的原始异常做基准 —— 逐字段比对，不是「看着差不多」。
        try:
            exec("def kernel(:\n    pass\n", {})       # noqa: S102 — 基准, 不执行候选
        except SyntaxError as raw:
            self.assertEqual(str(raw), str(rebuilt))
            self.assertEqual(
                (raw.msg, raw.lineno, raw.offset, raw.text, raw.filename),
                (rebuilt.msg, rebuilt.lineno, rebuilt.offset, rebuilt.text,
                 rebuilt.filename))

        guard = run_compile("raise ZeroDivisionError('顶层就炸')\n")
        self.assertEqual("ZeroDivisionError", guard.exc_type)
        rebuilt = candidate_error(guard)
        self.assertIsInstance(rebuilt, ZeroDivisionError)
        self.assertEqual("顶层就炸", str(rebuilt))

        # 拿不到的类型名 ⇒ 退化成 RuntimeError，但**保留原名**（不许让读者以为是候选抛的）
        from operator_agent.exec_guard import GuardResult
        odd = GuardResult(status=STATUS_CRASHED, exc_type="MyPrivateError", error="x")
        fallback = candidate_error(odd)
        self.assertIsInstance(fallback, RuntimeError)
        self.assertIn("MyPrivateError", str(fallback))

    def test_the_wall_clock_kills_a_hanging_compile(self):
        limits = Limits(wall_ms=400)
        started = time.time()
        guard = run_compile("while True:\n    pass\n", limits=limits)
        wall = time.time() - started
        self.assertEqual(STATUS_TIMEOUT, guard.status, guard.detail)
        self.assertEqual(KIND_WALL, guard.kind)
        self.assertLess(wall, 10.0, "墙钟没管用：它跑完了（或跑了很久）")
        # 被拦下的这一档**不是**候选的编译错误 —— 它在异常里也必须分得开
        refusal = refusal_from(guard, where="编译期")
        self.assertIsInstance(refusal, GuardRefusal)
        self.assertEqual(STATUS_TIMEOUT, refusal.status)
        self.assertEqual(KIND_WALL, refusal.kind)
        self.assertIn("编译期", str(refusal))

    def test_missing_kernel_is_reported_not_invented(self):
        guard = run_compile("value = 1\n")
        self.assertEqual(STATUS_OK, guard.status, guard.detail)
        self.assertIs(False, guard.metrics.get("has_kernel"))
        self.assertIs(False, guard.metrics.get("kernel_callable"))

    def test_preloaded_modules_land_in_the_child_namespace(self):
        code = "total = tl.arange(0, 4).sum()\ndef kernel(**kwargs):\n    return 1\n"
        without = run_compile(code)
        self.assertEqual(STATUS_CRASHED, without.status, "没有预置却跑通了 ⇒ 这条用例没在证什么")
        self.assertEqual("NameError", without.exc_type)
        with_imports = run_compile(code, imports={"tl": "numpy"})
        self.assertEqual(STATUS_OK, with_imports.status, with_imports.detail)
        self.assertIs(True, with_imports.metrics.get("has_kernel"))

    def test_one_default_per_mode(self):
        self.assertEqual(Limits().wall_ms, limits_for(MODE_RUN).wall_ms)
        self.assertEqual(COMPILE_LIMITS.wall_ms, limits_for(MODE_COMPILE).wall_ms)
        self.assertNotEqual(limits_for(MODE_RUN).wall_ms, limits_for(MODE_COMPILE).wall_ms,
                            "两个阶段用了同一个默认值 ⇒ 编译那一档会被运行那一档的墙钟掐掉")
        mine = Limits(wall_ms=1234)
        self.assertIs(mine, limits_for(MODE_RUN, mine))
        self.assertIs(mine, limits_for(MODE_COMPILE, mine))
        with self.assertRaises(ValueError):
            limits_for("benchmark")


class TestTheInProcessExecInventory(unittest.TestCase):
    """**只减不增**的账本：`operator_agent/` 里还剩几处进程内 `exec`，分别是谁

    为什么要有这本账而不是一句「都搬完了」：搬了一半的形态最危险 —— 「候选执行已进沙箱」
    这句话在**编译期**与 `LocalExecutor` 那条降级路终点上今天都不成立，而它们在文件树里
    看不出差别。这份清单是**登记**（谁还没搬、属于哪一类），不是许可：加一处新的会红。
    """

    #: `文件 → 含进程内 `exec` 的函数名（排序）`。改小这里的数**没有**代价，
    #: 改大必须说清它为什么不能走 `exec_guard`。
    EXPECTED = {
        # 候选代码，**仍是 §22.2 的洞**（登记在状态文档 §22 行）
        "mock_gpu.py": ["_benchmark_python", "_benchmark_triton_cuda",
                        "_execute_triton_cuda"],
        "reference.py": ["load_reference_source"],
        "remote_executor.py": ["benchmark", "execute"],
        # 不是候选代码：harness 自己造的参照核 / 子进程程序入口
        "calibration.py": ["_from_source"],
        "exec_guard.py": ["_fork_child"],
    }

    def test_the_inventory_is_exactly_this(self):
        import ast
        root = Path(__file__).resolve().parents[2] / "operator_agent"
        found = {}
        for path in sorted(root.glob("*.py")):
            tree = ast.parse(path.read_text("utf-8"))
            funcs = [n for n in ast.walk(tree)
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == "exec"):
                    continue
                owners = [f.name for f in funcs
                          if f.lineno <= node.lineno <= (f.end_lineno or f.lineno)]
                owner = max(owners, key=len) if owners else "<module>"
                found.setdefault(path.name, []).append(owner)
        found = {k: sorted(v) for k, v in found.items()}
        self.assertEqual(self.EXPECTED, found,
                         "进程内 exec 的账本变了：少了一处 ⇒ 更新这份清单；"
                         "多了一处 ⇒ §22.2 多了一个洞，先补受控执行再改这里")

    def test_the_two_compile_paths_have_no_in_process_exec(self):
        import ast
        root = Path(__file__).resolve().parents[2] / "operator_agent"
        targets = (("mock_gpu.py", "MockGPU", ("compile", "_compile_in_guard")),
                   ("remote_executor.py", "LocalExecutor", ("compile",)))
        for filename, class_name, funcs in targets:
            tree = ast.parse((root / filename).read_text("utf-8"))
            # 按**类**定位: `remote_executor.py` 里有两个 `compile`（SSH 那个把源码发远端,
            # 本地 `exec` 的只有 `LocalExecutor` 这一个）—— 按函数名找会找错那一个。
            cls = next(n for n in ast.walk(tree)
                       if isinstance(n, ast.ClassDef) and n.name == class_name)
            for func_name in funcs:
                func = next(n for n in cls.body
                            if isinstance(n, ast.FunctionDef) and n.name == func_name)
                self.assertNotIn("exec", [n.id for n in ast.walk(func)
                                          if isinstance(n, ast.Name)],
                                 f"{filename}::{class_name}.{func_name} 里还留着进程内 exec")
                self.assertIn("run_compile", ast.dump(func),
                              f"{filename}::{class_name}.{func_name} 没有走受控编译入口")


if __name__ == "__main__":
    unittest.main(verbosity=2)