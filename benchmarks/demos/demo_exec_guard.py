"""§22.2 最小一块的本地零依赖演示 —— 亲眼看一下「候选跑飞了会怎样」

不需要 GPU、不需要 LLM、不联网，全部投影落在临时目录里:

    python3 benchmarks/demos/demo_exec_guard.py

四个场景回答的是设计 §22.2 那句话（`docs/design/autonomous-operator-agent-design.md:1552`）
「候选代码在评测进程里运行时，它与 harness 共享地址空间、文件系统与网络」在代码里
到底被收掉了哪几格：

  1. 正常运行 —— 候选照旧跑通，而且**父进程不被污染**（候选改模块属性/环境变量/cwd
     都影响不到 harness：各自一份地址空间）
  2. 死循环 ⇒ 墙钟到点，**整个进程组被杀**（含候选自己起的孙子进程），不留孤儿
  3. 吃满：单文件大小真的拦得住 / 进程数真的拦得住 / 产物字节数超了是**失败**不是截断
  4. **哪一项没设上，如实报** —— 本机（macOS）`RLIMIT_AS` 内核 EINVAL，内存那一档
     只能报 `not_enforced` ＋原始原因；「设不上」不许被读成「没有内存上限」

可执行文档: `tests/unit/test_exec_guard.py`(32 条) 与 `tests/unit/test_demo_exec_guard.py`；
变异表 7 行（只标定、不改源）:

    python3 -B -m benchmarks.mutation_calibrate --spec benchmarks/mut_specs/exec_guard.py

⚠️ 边界 (与 CHECKLOG 素材、模块 docstring 同一句话): 这只证明**本地这一半**里能实测站住的
那几项。这**不是**敌意候选的沙箱 —— 同一个 uid、网络**没关**、没有容器/cgroup、
没有文件系统白名单、没有 GPU watchdog、没有设备租约回收；远端 runner 侧一个字都没说；
CUDA 执行路径没有接进来。挡的是**跑飞/吃满/失控**，不是敌意攻击。
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from operator_agent import exec_guard as eg  # noqa: E402

A = np.arange(6, dtype=np.float64).reshape(2, 3)
B = np.arange(12, dtype=np.float64).reshape(3, 4)

NORMAL = """import numpy as np

def kernel(A, B):
    return A @ B
"""


def _line(title: str) -> None:
    print("\n" + "=" * 68 + f"\n{title}\n" + "=" * 68)


def _show(tag: str, r: eg.GuardResult) -> None:
    print(f"  [{tag}] status={r.status} kind={r.kind} executor={r.executor} "
          f"wall={r.wall_ms:.0f}ms elapsed={r.elapsed_ms:.2f}ms")
    print(f"        fully_enforced={r.fully_enforced} "
          f"没设上的项={list(r.not_enforced)} maxrss={r.max_rss_mb}")
    if r.detail:
        print(f"        detail: {r.detail[:96]}")


def scene_1_normal_and_no_leak() -> None:
    _line("1. 正常运行 + 父进程不被污染（各自一份地址空间）")
    r = eg.run_candidate(NORMAL, {"A": A, "B": B})
    _show("正常候选", r)
    same = bool(np.allclose(np.asarray(r.output), A @ B))
    print(f"        产物与 A @ B 一致: {same}")
    marker = "DEMO_EXEC_GUARD_LEAK"
    cwd_before = os.getcwd()
    r2 = eg.run_candidate(
        "import os, numpy as np\n"
        "def kernel(A, B):\n"
        "    os.environ[%r] = '1'\n"
        "    np.zeros = None\n"
        "    os.chdir('/')\n"
        "    return A @ B\n" % marker, {"A": A, "B": B})
    print(f"  候选在子进程里改环境变量/模块属性/chdir 之后: status={r2.status}")
    print(f"        harness 侧环境变量被改? {marker in os.environ}"
          f" | numpy.zeros 被改? {np.zeros is None}"
          f" | cwd 被改? {os.getcwd() != cwd_before}")
    if marker in os.environ or np.zeros is None or os.getcwd() != cwd_before:
        print("  !! 候选的状态漏进了 harness —— 这不是「共享地址空间」该有的样子")


def scene_2_timeout_kills_the_group() -> None:
    _line("2. 死循环 ⇒ 墙钟到点，整个进程组被杀（含孙子进程）")
    t0 = time.monotonic()
    r = eg.run_candidate("def kernel(A, B):\n    while True:\n        pass\n",
                         {"A": A, "B": B}, limits=eg.Limits(wall_ms=400))
    _show("死循环", r)
    print(f"        墙钟读数 {r.wall_ms:.0f}ms（请求 400ms），比死循环自己停下来早得多")
    work = tempfile.mkdtemp(prefix="demo-exec-guard-orphan-")
    marker = os.path.join(work, "grandchild-survived.txt")
    inner = ("import time, pathlib; time.sleep(1.0); "
             "pathlib.Path(%r).write_text('survived')" % marker)
    r2 = eg.run_candidate(
        "import subprocess, sys, time\n"
        "def kernel(A, B):\n"
        "    subprocess.Popen([sys.executable, '-c', %r])\n"
        "    time.sleep(30)\n"
        "    return A @ B\n" % inner,
        {"A": A, "B": B}, limits=eg.Limits(wall_ms=300, processes=None), scratch_dir=work)
    _show("死循环 + 孙子进程", r2)
    time.sleep(1.3)
    survived = os.path.exists(marker)
    print(f"        孙进程留下的痕迹存在? {survived}  ⇒ {'留了孤儿' if survived else '连孙子一起收了'}")
    if survived:
        print("  !! 墙钟超时留下了孤儿进程 —— 「超时」不许变成「留一堆孤儿」")
    shutil.rmtree(work, ignore_errors=True)
    print(f"        （本场景共 {time.monotonic() - t0:.1f}s）")


def scene_3_limits_that_really_bite() -> None:
    _line("3. 吃满：单文件大小 / 进程数 / 产物字节数")
    work = tempfile.mkdtemp(prefix="demo-exec-guard-fsize-")
    r = eg.run_candidate("def kernel(A, B):\n"
                         "    with open('big.bin', 'wb') as f:\n"
                         "        for _ in range(20):\n"
                         "            f.write(b'x' * 1048576)\n"
                         "            f.flush()\n"
                         "    return A @ B\n",
                         {"A": A, "B": B},
                         # 产物经一个 pickle 文件回传 ⇒ 单文件上限也要装得下产物预算
                         limits=eg.Limits(file_size_bytes=1024 * 1024,
                                          artifact_bytes=256 * 1024),
                         scratch_dir=work)
    _show("想写 20MB 的候选", r)
    size = os.path.getsize(os.path.join(work, "big.bin")) if os.path.exists(
        os.path.join(work, "big.bin")) else 0
    print(f"        它写出来的文件停在 {size} 字节（上限 1048576）")
    shutil.rmtree(work, ignore_errors=True)

    r2 = eg.run_candidate("import subprocess\n"
                          "def kernel(A, B):\n"
                          "    for _ in range(200):\n"
                          "        subprocess.Popen(['sleep', '5'])\n"
                          "    return A @ B\n",
                          {"A": A, "B": B}, limits=eg.Limits(processes=4))
    _show("想连起 200 个进程的候选", r2)
    print("        ⚠️ RLIMIT_NPROC 是 **per-UID** 的：设 4 之后，该用户已有的进程数早就超了，"
          "所以它一个也起不来")

    r3 = eg.run_candidate("import numpy as np\n"
                          "def kernel(A, B):\n"
                          "    return np.zeros((1024, 1024, 2))\n",
                          {"A": A, "B": B}, limits=eg.Limits(artifact_bytes=256 * 1024))
    _show("产物 16MB 的候选（上限 256KB）", r3)
    print(f"        产物被读回来了吗? {r3.output is not None} ⇒ "
          f"{'截断后当通过了' if r3.output is not None else '失败（不是截断后当通过）'}")

    # CPU 时间：墙钟给宽（6s），CPU 上限 1s ⇒ 死的是 CPU 那一档，不是墙钟
    r4 = eg.run_candidate("def kernel(A, B):\n    while True:\n        pass\n",
                          {"A": A, "B": B}, limits=eg.Limits(wall_ms=6000, cpu_s=1))
    _show("死循环但墙钟给到 6s、CPU 只给 1s", r4)
    print(f"        cpu 那一档的读数: {r4.enforcement_of('cpu').reading}")
    if r4.kind != eg.KIND_CPU:
        print("  !! CPU 上限到点却没有被认成 CPU 那一档")


def scene_4_what_is_not_enforced() -> None:
    _line("4. 哪一项没设上，如实报（三态纪律：设不上 ≠ 没有超限）")
    r = eg.run_candidate("import numpy as np\n"
                         "def kernel(A, B):\n"
                         "    x = np.zeros((512, 1024, 128), dtype=np.float64)\n"
                         "    x[:] = 1.0\n"
                         "    return A @ B\n",
                         {"A": A, "B": B},
                         limits=eg.Limits(memory_bytes=64 * 1024 * 1024, wall_ms=8000))
    for e in r.enforcement:
        print(f"        {e.item:<11} {e.state:<13} requested={e.requested}"
              f" {('| ' + e.reason[:52]) if e.reason else ''}")
    mem = r.enforcement_of("memory")
    print(f"  内存那一档: state={mem.state} reason={mem.reason!r}")
    print(f"  候选申请了 512MB 而 status={r.status} ⇒ 「设不上」没有被读成「拦住了」，"
          f"也没有被读成「没有内存上限」")
    print(f"  fully_enforced={r.fully_enforced} —— 这是一个**读数**，不是装饰")
    print("  对照（本机 libc 直调，绕开 Python 包装，所以能分清「内核不给」与「Python 不给」）:"
          " AS/RSS(5) 与 DATA(2) → errno=22 EINVAL；STACK(3) → EPERM；CPU(0)/FSIZE(1)/NPROC(7) → rc=0")


def main() -> None:
    print("§22.2 最小一块：本地候选执行的墙钟 / rlimit / 进程组 / 产物上限")
    print("（零 GPU / 零 LLM / 不联网；所有投影落在临时目录里，仓库一个字节都不动）")
    scene_1_normal_and_no_leak()
    scene_2_timeout_kills_the_group()
    scene_3_limits_that_really_bite()
    scene_4_what_is_not_enforced()
    _line("边界（不许读成已经做到的事）")
    print("  · 不是敌意候选的沙箱：**同一个 uid**、网络**没关**、没有容器/cgroup")
    print("  · 没有文件系统白名单（候选仍可读 harness 的文件、写自己的 scratch）")
    print("  · 内存那一档本机设不上（macOS RLIMIT_AS 内核 EINVAL）⇒ 本地内存跑飞仍拦不住")
    print("  · 没有 GPU watchdog、没有设备租约回收（真机上没有可靠的卡级看门狗量具）")
    print("  · 远端 runner 侧零覆盖（仍是同 uid 落 /tmp 后 exec_module）")
    print("  · CUDA 执行路径 MockGPU._execute_triton_cuda **没有**接进来（要真卡才能验）")


if __name__ == "__main__":
    main()