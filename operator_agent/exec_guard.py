"""§22.2（最小一块）：把**本地**候选执行搬进受控子进程 —— 墙钟超时 + rlimit + 进程组清理

设计原文（`docs/design/autonomous-operator-agent-design.md:1552-1564`）列了九项：
`编译超时 · 运行超时 · GPU watchdog · CPU / memory quota · 文件系统白名单 · 网络默认关闭 ·
进程树清理 · artifact 大小限制 · 设备租约超时与回收`，理由是「候选代码在评测进程里运行时，
它与 harness 共享地址空间、文件系统与网络。没有隔离，『候选不能修改 harness』就只能靠
候选自觉」。
本模块落地的是**本地这一半里能实测站得住的那几项**，其余逐条登记为未落（见文末）。

## 形状

候选代码不再在 harness 进程里 `exec`，而是：

```
父 (harness)                          子 (受控, 自己一个新的会话/进程组)
  ├─ 写 job.pkl (code / inputs / limits)   ─→ 读 job.pkl
  ├─ Popen(python -c <子程序>,             ─→ ① 自己给自己设 rlimit（逐项记录设上没设上）
  │        start_new_session=True,          ─→ ② 装 shield_torch（与今天同一把盾）
  │        stdout/stderr → child.log)       ─→ ③ ns = {}; exec(code, ns); kernel(**inputs)
  ├─ wait(timeout=wall_ms)                  ─→ ④ 把 产物/耗时/异常/命中/逐项强制读数 写成 result.pkl
  └─ 分类：ok / timeout / limit_exceeded / crashed / cheat
```

**关键决定：rlimit 由子进程自己设**（子程序第一件事就是 `resource.setrlimit`）。
不用 `preexec_fn`：它在多线程父进程里不是线程安全的（Python 官方文档明说不该用），
而 harness 里确实有网络客户端线程。子进程自己设 = 同一件事，但没有那个隐患。

## 实测（本机 macOS / darwin，python 3.9.12，2026-10-04，原始读数）

先实测再写结论（本仓硬要求）。逐项 **libc `setrlimit` 直调** 的读数（绕开 Python 包装，
所以能分清「内核不给」与「Python 前置检查不给」）：

| 项 | 本机读数 | 结论 |
|---|---|---|
| `RLIMIT_CPU` (0) | `setrlimit(1,1)` → **rc=0**；死循环子进程 → `rc=-24` = **SIGXCPU**，wall 1.04s | ✅ **真管用** |
| `RLIMIT_FSIZE` (1) | `setrlimit(1MB,1MB)` → **rc=0**；写第 2MB → `OSError [Errno 27] File too large`，文件停在 1048576 字节（**没有** SIGXFSZ，Darwin 返回 EFBIG） | ✅ **真管用** |
| `RLIMIT_NPROC` (7) | `setrlimit(8,8)` → **rc=0**；随后 `subprocess.Popen` 60 次**全部** `BlockingIOError [Errno 35]` | ⚠️ 管用，但**是 per-UID 的**：该 UID 当时已有 >8 个进程，所以「设 8」= 立刻一个也起不来。它不是「本进程最多 8 个子进程」 |
| `RLIMIT_AS` (5，Darwin 上就是 `RLIMIT_RSS`) | `setrlimit(256MB, inf)` → **rc=-1 errno=22 EINVAL**；(1GB/inf) EINVAL；(100GB/inf) EINVAL；只有「降到 2^63-2」才 rc=0（等于没降）。Python 包装成 `ValueError: current limit exceeds maximum limit` | ❌ **设不上** ⇒ 必须报 `not_enforced` |
| `RLIMIT_DATA` (2) | `(256MB,256MB)` 与 `(256MB,inf)` → **rc=-1 errno=22** | ❌ 设不上 |
| `RLIMIT_STACK` (3) | 降上限 → **rc=-1 errno=1 EPERM**（root-only） | ❌ 设不上 |
| 父侧看**活着**的子进程内存 | `ps` 在本沙箱里被挡（`PermissionError: Operation not permitted`）；`getrusage(RUSAGE_CHILDREN).ru_maxrss` 在子进程活着时**恒 0**（轮询 10 次全 0.0MB），子进程退出后才跳到 675.2MB | ❌ 做不了活体看门狗 |
| `start_new_session=True` + `os.killpg` | 子进程 pgid == pid ✓；killpg 后 `os.killpg(pgid, 0)` → `ProcessLookupError`；孙进程（2s 后写文件的 `sleep`）**没留下痕迹** | ✅ **杀得干净** |

⇒ **本机（macOS）上「内存上限」这一项是 `not_enforced`**。Linux 上 `RLIMIT_AS` 通常可用，
但**我没有在 Linux 上实测过，所以这里不声称**：`memory_bytes=None` 表示「调用方不请求」，
请求了但设不上就如实报（`Limits.memory_bytes` 的说明与 `GuardResult.not_enforced` 是读者）。

## 可证 / 不可证

**可以声称（有守卫 `tests/unit/test_exec_guard.py` + 演示 `benchmarks/demos/demo_exec_guard.py`）**：
1. 墙钟超时会**杀掉整个进程组**（含孙子进程），不留孤儿，并给出 `timeout` + 用了多少 ms；
2. `SIGXCPU`（CPU 时间）/ EFBIG（单文件大小）/ `EAGAIN`+`BlockingIOError`（进程数）/
   `MemoryError`（内存）四档在**返回值里分得开**，且带读数；
3. **设不上的项如实报**：每一项都有 `enforced` / `not_enforced` + 原始原因文本，
   `GuardResult.fully_enforced` 是显式读数 —— 「设不上」不许被读成「没有超限」（本仓 I65/I11 那一族）；
4. 读回来的产物**字节数封顶**，超了是 `limit_exceeded`（**失败**，不是截断后当通过）；
5. 正常候选照旧跑通：`ns = {}`、`ns["kernel"]`、返回值、耗时语义与今天**逐字一致**
   （接线是换执行方式，不是换语义）；
6. **候选改不动 harness**：模块属性、环境变量、`cwd`、`torch` 的补丁都只活在子进程里
   （各自一份地址空间）—— 这是原先「进程内 `exec`」**做不到**的一条（那时候选可以顺手把
   `numpy.zeros` 改成别的，harness 下一轮就带着那个改动跑）；
7. **上限在子进程里不可逆**：软硬上限一起降，候选自己调 `setrlimit` 也抬不回去。

⚠️ 一条 2026-10-04 实测出来的**必须写下来的账**：产物是经一个 pickle 文件回传的，而那个
文件与候选写的文件受**同一个** `RLIMIT_FSIZE`（上限是 per-process 的，没有「只豁免传输文件」
这回事）。所以 `artifact_bytes <= file_size_bytes` 是**物理事实**，由 `Limits.__post_init__`
在构造期拦住。第一版没写这条不变量、又把 FSIZE 默认成 64MB，于是 `benchmarks/leetcuda_*`
那些**合法**返回 64MB 张量的 fixture 被截断成一句 `UnpicklingError: pickle data was truncated`
—— 看起来像「候选坏了」，实际是「两个上限打架」。

**不许声称（本模块一个字都没管）**：
* **不是敌意候选的沙箱**：同一个 uid、同一份文件系统 —— 候选仍可读 harness 的文件、
  写自己的 scratch、改 `chmod`（§22.1 的边界同一条）。挡的是**跑飞/吃满/失控**；
* **网络没关**（`网络默认关闭` 未落）：候选照样能联网；
* **文件系统白名单未落**：只有单文件大小上限与「cwd 指向临时 scratch」，没有路径白名单；
* **没有独立 uid / 容器 / cgroup**（那才是 §22.2 的「隔离」本体）；
* **没有 GPU watchdog**（真机上没有可靠的卡级看门狗量具）与**设备租约回收**；
* **内存上限在本机设不上**（上面那张表），Linux 上的行为未实测；
* **CUDA 执行路径（`MockGPU._execute_triton_cuda`）没有接进来**：它需要真卡才能验，
  且子进程里重新初始化 CUDA 会改掉计时语义与 triton 缓存 —— 那是它自己的一片（欠账）；
* **远端侧零覆盖**：远端 runner 仍是同 uid 落 `/tmp` 后 `exec_module`。
"""

from __future__ import annotations

import os
import pickle
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

# ─── 闭合词汇 ①：结果状态（三态纪律：这五档不许压成布尔）────────────
#: 跑完了、拿到产物
STATUS_OK = "ok"
#: 墙钟到点，进程组已被杀
STATUS_TIMEOUT = "timeout"
#: 撞上某一档资源上限（`kind` 说是哪一档），或者产物超上限
STATUS_LIMIT_EXCEEDED = "limit_exceeded"
#: 候选自己抛异常/子进程异常退出（含 `kernel` 缺失、语法错）
STATUS_CRASHED = "crashed"
#: 运行时盾拦到原生算子回退（lib_fallback 生产者；`metadata` 里那句文案是同一个真源）
STATUS_CHEAT = "cheat"
STATUSES: Tuple[str, ...] = (STATUS_OK, STATUS_TIMEOUT, STATUS_LIMIT_EXCEEDED,
                             STATUS_CRASHED, STATUS_CHEAT)

# ─── 闭合词汇 ②：撞的是哪一档 ────────────────────────────────
KIND_CPU = "cpu"                  # RLIMIT_CPU → SIGXCPU
KIND_MEMORY = "memory"            # RLIMIT_AS → MemoryError（本机设不上，见 docstring）
KIND_FILE_SIZE = "file_size"      # RLIMIT_FSIZE → EFBIG / SIGXFSZ
KIND_PROCESSES = "processes"      # RLIMIT_NPROC → EAGAIN / BlockingIOError
KIND_WALL = "wall"                # 父侧墙钟
KIND_ARTIFACT = "artifact"        # 产物字节数上限（父侧判）
KINDS: Tuple[str, ...] = (KIND_CPU, KIND_MEMORY, KIND_FILE_SIZE, KIND_PROCESSES,
                          KIND_WALL, KIND_ARTIFACT)

# ─── 闭合词汇 ③：逐项强制的状态（**这一档是「不许读成通过」的那一档**）──
ENFORCED = "enforced"
NOT_ENFORCED = "not_enforced"
ENFORCEMENT_STATES: Tuple[str, ...] = (ENFORCED, NOT_ENFORCED)

#: 强制项的名字。每一个都要有读者：`GuardResult.enforcement` / `not_enforced` /
#: `fully_enforced` / 演示脚本 / 变异表。
ITEM_WALL = "wall"
ITEM_CPU = "cpu"
ITEM_MEMORY = "memory"
ITEM_FILE_SIZE = "file_size"
ITEM_PROCESSES = "processes"
ITEM_ARTIFACT = "artifact"
ITEMS: Tuple[str, ...] = (ITEM_WALL, ITEM_CPU, ITEM_MEMORY, ITEM_FILE_SIZE,
                          ITEM_PROCESSES, ITEM_ARTIFACT)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── 闭合词汇 ④：用哪个执行后端 ────────────────────────────────
#: 每次 `os.fork()`：子进程继承父进程**已经 import 好的** torch，所以每次执行的开销是
#: 毫秒级（本机实测 4~8ms）。代价：多线程父进程里 fork 有死锁风险（见 `run_candidate`）。
EXECUTOR_FORK = "fork"
#: 每次起一个**全新解释器**：`import torch` 每次都要重付（本机实测 ≈860ms），但完全没有
#: fork 安全性问题。
EXECUTOR_SUBPROCESS = "subprocess"
#: 模块按一条**可观察**的规则自己选：父进程此刻只有主线程 ⇒ `fork`；否则 ⇒ `subprocess`。
EXECUTOR_AUTO = "auto"
EXECUTORS: Tuple[str, ...] = (EXECUTOR_FORK, EXECUTOR_SUBPROCESS, EXECUTOR_AUTO)

# ─── 闭合词汇 ⑤：这一次受控执行做的是「跑」还是「编」───────────────
#: 跑候选：`exec(code)` 之后调 `kernel(**inputs)`（`run_candidate` 的语义，逐字未变）。
MODE_RUN = "run"
#: 只做编译期那一步：`exec(code)`，**不调用** `kernel`（`run_compile`）。
#:
#: WHY（§22.2 的第一项「编译超时」）: 编译期原先在 harness 进程里 `exec` —— 没有墙钟、
#: 没有 rlimit、没有进程组。候选只要在模块顶层写一个死循环或一次巨型分配，就能在
#: **编译**这一步把评测进程拖死，而那时 §22.2 的运行期那一半（`_execute_python`）还
#: 根本没轮到。把编译也搬进同一个受控子进程，这一项才有落点。
MODE_COMPILE = "compile"
MODES: Tuple[str, ...] = (MODE_RUN, MODE_COMPILE)


def start_background_worker(argv, *, cwd, log_path, env=None) -> subprocess.Popen:
    """Launch a trusted harness supervisor with a lifetime independent of MCP.

    This is process lifecycle plumbing, not candidate execution: it applies no
    candidate timeout, rlimits, shielding or automatic exit cleanup. Generated
    kernels must still use ``run_candidate``/``run_compile``. A long-running
    trusted worker owns its status files and the caller receives its real PID.

    Input is closed, output appends to the caller-owned log, unrelated file
    descriptors are closed, and a separate session prevents the tool's session
    shutdown from taking the worker with it. No shell or pre-exec callback is
    accepted. Closing the parent's log descriptor does not close the child's.
    """
    if (not isinstance(argv, (list, tuple)) or not argv
            or any(not isinstance(part, str) or '\x00' in part for part in argv)
            or not argv[0]):
        raise ValueError('background worker requires an explicit argument vector')
    with open(os.fspath(log_path), 'ab') as log:
        return subprocess.Popen(
            list(argv), cwd=os.fspath(cwd), env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True, shell=False,
        )


def start_task_worker(argv, *, cwd, log_path=None):
    """Launch the trusted task worker; candidate execution still uses guards."""
    if (not isinstance(argv, (list, tuple)) or not argv
            or any(type(part) is not str or not part or '\x00' in part for part in argv)):
        raise ValueError('task worker requires an explicit argument vector')
    if log_path is not None:
        with open(os.fspath(log_path), 'ab') as log:
            return subprocess.Popen(list(argv), cwd=os.fspath(cwd), stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True, shell=False)
    return subprocess.Popen(list(argv), cwd=os.fspath(cwd), stdin=subprocess.DEVNULL,
        start_new_session=True, close_fds=True, shell=False)


class OwnedGroups(set):
    """Observed group leaders plus start identities, so stale PGIDs aren't killed."""
    def __init__(self, values=()):
        super().__init__(values)
        self.identities = dict(getattr(values, 'identities', {}))
        self.discovery_complete = getattr(values, 'discovery_complete', True)

    def update(self, values):
        super().update(values)
        self.identities.update(getattr(values, 'identities', {}))
        self.discovery_complete &= getattr(values, 'discovery_complete', True)


def _process_rows():
    listing = subprocess.run(['ps', '-axo', 'pid=,ppid=,pgid=,lstart='],
        capture_output=True, text=True, timeout=.5, check=True)
    return [(int(parts[0]), int(parts[1]), int(parts[2]), ' '.join(parts[3:]))
            for parts in (row.split() for row in listing.stdout.splitlines()) if parts]


def process_identity(pid):
    """Best available OS start identity; unavailable stays unknown."""
    try:
        return next((start for actual, _, _, start in _process_rows() if actual == pid), None)
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def task_process_groups(worker_pid):
    """Snapshot groups whose leaders are in this worker's current tree."""
    descendants = {worker_pid}
    groups = OwnedGroups([worker_pid])
    try:
        rows = _process_rows()
        while True:
            added = {pid for pid, parent, _, _ in rows if parent in descendants} - descendants
            if not added:
                break
            descendants.update(added)
        groups.update(group for pid, _, group, _ in rows
                      if pid in descendants and group in descendants)
        groups.identities = {pid: start for pid, _, group, start in rows
                             if pid == group and group in groups}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        groups.discovery_complete = False
        return groups, False
    return groups, True


def stop_task_worker(process, observed_groups=(), *, discover=None, graceful_s=0):
    """Clean known task groups on every exit; expose discovery uncertainty."""
    # S4's trusted worker may first close its own persistent SSH session. Only
    # the held Popen is signalled; the existing group cleanup remains the fallback.
    if graceful_s > 0 and process.poll() is None:
        try:
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=min(float(graceful_s), 15.0))
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
        except OSError:
            pass
    live = process.poll() is None
    current, discovered = ((discover or task_process_groups)(process.pid)
                           if live else (OwnedGroups(), True))
    groups = OwnedGroups(observed_groups)
    groups.update(current)
    discovered &= groups.discovery_complete
    try:
        identities = {pid: start for pid, _, group, start in _process_rows() if pid == group}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        identities, discovered = {}, False
    skipped_reused = 0
    # Candidate and HTTP guards create their own sessions. Kill those owned
    # groups as well, so a blocked guard cannot outlive its local supervisor.
    for group in groups:
        if group != os.getpgrp():
            expected = groups.identities.get(group)
            actual = identities.get(group)
            if expected and actual and expected != actual:
                skipped_reused += 1
                continue
            if not (expected and expected == actual) and not (live and group == process.pid):
                if actual is not None:  # no identity proof for a still-present group
                    discovered = False
                continue
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                discovered = False
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        discovered = False
    return dict(status='observed_groups_terminated' if discovered else 'cleanup_unknown',
        groups_observed=len(groups), worker_reaped=process.poll() is not None,
        stale_groups_skipped=skipped_reused, all_descendants_proven=False)


class GuardRefusal(RuntimeError):
    """候选被**受控执行器**拦下（超时 / 撞上限 / 盾命中 / 起不来）—— 不是它自己编译错了。

    为什么要一个类型而不是一句话: 上层要能把「编译期被 §22.2 拦下」与「候选编译失败」
    在产物里分开（`Verifier` 发布的 `compile_status`）。靠消息字符串匹配迟早会漂，而
    「撞了上限」被读成「候选语法错」正是本仓 I65/I11 那一族的形状。
    """

    def __init__(self, message: str, *, status: str, kind: Optional[str] = None,
                 detail: str = "", not_enforced: Tuple[str, ...] = (),
                 enforcement: Tuple["Enforcement", ...] = (),
                 executor: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.kind = kind
        self.detail = detail
        self.not_enforced = tuple(not_enforced)
        self.enforcement = tuple(enforcement)
        self.executor = executor

    def as_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "kind": self.kind, "detail": self.detail,
                "not_enforced": list(self.not_enforced),
                "fully_enforced": not self.not_enforced,
                "executor": self.executor}


@dataclass(frozen=True)
class Limits:
    """这一次执行请求的上限。**默认值是「有限但宽松」**：够日常候选跑完，又能兜住跑飞。

    `memory_bytes=None` = **不请求**这一项（不是「请求 0」）。为什么默认不请求：
    本机（macOS）上 `RLIMIT_AS` 内核直接 EINVAL（见模块 docstring 的实测表），请求了也只是
    多一条 `not_enforced`。要试的平台可以显式传一个值 —— 设上了就 `enforced`，设不上就
    `not_enforced` + 原始原因，两档都看得见。
    """

    wall_ms: int = 5000
    cpu_s: int = 10
    memory_bytes: Optional[int] = None
    #: ⚠️ 这一项**同时**管两件事：①候选自己写的每个文件；②**回传产物**的那个 pickle 文件。
    #: 上限是 per-process 的，没有「只豁免传输文件」这回事 —— 所以它必须大于
    #: `artifact_bytes`，这一条由构造期不变量保证（见 `__post_init__`）。
    #: 默认 512MB 的来由：实测（2026-10-04）合法的产物有 64MB 级的（`benchmarks/leetcuda_*`
    #: 的 fixture 直接返回 `1024x1024xfp64`），64MB 的默认值会把**合法**产物截断成
    #: 一句 "pickle data was truncated"。
    file_size_bytes: int = 512 * 1024 * 1024
    processes: int = 32
    #: 读回来的产物（pickle 载荷）字节上限。超了**失败**，不截断后当通过。
    #:
    #: 为什么默认给到 256MB 而不是一个「看起来更严」的小数：实测（2026-10-04 全仓跑）
    #: 8MB 会把**合法**的大产物判成失败 —— `benchmarks/leetcuda_*.py` 的 fixture 直接返回
    #: 64MB 的张量（`1024x1024xfp64`，pickle 后 67MB），于是 L2 报「产物超上限」而不是
    #: 那个核真正的对错，三条用例因此红。这一格的作用是**给失控收口**，不是给内存做预算；
    #: 真正需要小上限的调用方显式传（`tests/unit/test_exec_guard.py` 就是这么钉的）。
    artifact_bytes: int = 256 * 1024 * 1024
    #: 父侧保留的候选 stdout/stderr 尾巴（字节）。只是日志，不是判据。
    log_bytes: int = 8192

    def __post_init__(self) -> None:
        """**构造期**就把「产物装不进那个文件」这种组合拦住。

        WHY（2026-10-04 实测）: 产物是经一个 pickle 文件回传的，而那个文件与候选写的文件
        受同一个 `RLIMIT_FSIZE`。两者不一致时，子进程写到上限被截断，父侧读回来只有
        `UnpicklingError` —— 看起来像「候选坏了」，实际是「两个上限打架」。
        这种组合是**可判定**的，所以在这里失败，而不是留到运行时去猜。
        """
        # `None` = 调用方不请求这一项 —— 不参与比较（`None > int` 会 TypeError，
        # 而「没请求」本来就不是「不一致」）
        if (self.artifact_bytes is not None and self.file_size_bytes is not None
                and self.artifact_bytes > self.file_size_bytes):
            raise ValueError(
                f"artifact_bytes ({self.artifact_bytes}) 不能大于 file_size_bytes "
                f"({self.file_size_bytes})：产物是经一个 pickle 文件回传的，那个文件与候选"
                f"写的文件受同一个单文件上限（上限是 per-process 的，没法只豁免它）。"
                f"要么抬 file_size_bytes，要么降 artifact_bytes。")

    def as_dict(self) -> Dict[str, Any]:
        return {"wall_ms": self.wall_ms, "cpu_s": self.cpu_s,
                "memory_bytes": self.memory_bytes,
                "file_size_bytes": self.file_size_bytes,
                "processes": self.processes, "artifact_bytes": self.artifact_bytes}

    def rlimit_requests(self) -> Dict[str, Optional[int]]:
        """子进程要设的那四项 —— 键名就是 `ITEMS` 里的名字。

        单独一个方法而不是让子程序去读 `as_dict()`：`as_dict` 是给**读者**看的产物
        （键名 `cpu_s` / `file_size_bytes`），子进程的键名必须与 `Enforcement.item` 同源，
        否则「设上了没有」会记到两个不同的名字上（本模块第一版就这么错过一次：
        `cpu` 项被读成 `None` ⇒ 报「调用方没有请求」，而调用方明明请求了 10s）。
        """
        return {ITEM_CPU: self.cpu_s, ITEM_MEMORY: self.memory_bytes,
                ITEM_FILE_SIZE: self.file_size_bytes, ITEM_PROCESSES: self.processes}


#: **编译模式**的默认上限。与 `Limits` 的运行默认值不同，理由是这一档的墙钟**包含**
#: 子进程自己的启动与 import：`subprocess` 后端下一次 `import torch` / `import triton`
#: 本机实测 ≈860ms（triton+torch 一起更久），而 `fork` 只有在父进程单线程时才用得上
#: （见 `_pick_executor`）。用运行那一档的 5000ms 会把「子进程还没 import 完」记成
#: 「候选编译超时」—— 那正是本仓不许的形状（把「不知道/不是它」写成结论）。
#:
#: ⚠️ **这个数是暂定的**：真机上「一次 Triton / CUDA 编译要多久」我没有实测过，
#: 所以取一个宽松但有界的值（30s / 60s CPU），并且**允许调用方逐次覆盖**（切片 B 的
#: 任务/阶段级传递链就是为它准备的开口）。真机读数落地之前，不许把这几个数读成
#: 「编译需要这么久」。
COMPILE_LIMITS = Limits(wall_ms=30000, cpu_s=60)


def limits_for(mode: str, limits: Optional[Limits] = None) -> Limits:
    """这一次执行真正要用的上限：调用方给了就用它的，没给就**按模式**取默认值。

    为什么要有这一个函数（而不是各调用点写 `limits or Limits()`）：默认值必须只有一处，
    否则「编译那一档」与「运行那一档」迟早被同一句 `or Limits()` 抹平 —— 那时子进程自己
    的 import 开销会被记成「候选编译超时」，而这正是本仓不许的形状。传递链是
    **调用方 → 后端 → 本模块**，每一段都只传 `None` 或一个具体值（不接配置文件 /
    环境变量 / 命令行，§3.5）。
    """
    if limits is not None:
        return limits
    if mode == MODE_COMPILE:
        return COMPILE_LIMITS
    if mode == MODE_RUN:
        return Limits()
    raise ValueError(f"mode 必须是 {MODES} 之一: {mode!r}")


@dataclass(frozen=True)
class Enforcement:
    """一项上限「设上了没有」的留痕。`reason` 是原始文本，不做二次概括。

    为什么要有这个对象而不是一个布尔：本仓的 I65/I11 那一族就是「设不上」被读成了
    「没有超限」。「设不上」必须是一条**能被读到**的结论。
    """

    item: str
    state: str
    requested: Optional[int] = None
    reason: str = ""
    reading: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"item": self.item, "state": self.state, "requested": self.requested,
                "reason": self.reason, "reading": self.reading}


@dataclass(frozen=True)
class GuardResult:
    """一次受控执行的完整产物。**五项状态 + 逐项强制读数**，都留在里面。"""

    status: str
    kind: Optional[str] = None
    detail: str = ""
    output: Any = None
    error: Optional[str] = None
    #: 候选抛出来的异常**类型名**（`SyntaxError` / `RuntimeError` / …）；没有异常时为空串。
    #: 为什么要它: 编译那条路今天按异常类型分两支（`SyntaxError` 原样重抛，其余包成
    #: `RuntimeError("Compile error: …")`），而类型名在子进程那一侧就得记下来 ——
    #: 靠 `detail` 的 `"SyntaxError: …"` 前缀去认就是字符串匹配。
    exc_type: str = ""
    #: `SyntaxError` 的 `msg` / `filename` / `lineno` / `offset` / `text`。
    #: 跨进程重建时**只保住 `str()` 是不够的**（那会静默丢掉行号与列号，
    #: `verifier.syntax_error_report()` 读的正是这几个）；空 dict = 不是语法错。
    syntax_fields: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    wall_ms: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    log: str = ""
    enforcement: Tuple[Enforcement, ...] = ()
    exit_code: Optional[int] = None
    signal: Optional[int] = None
    artifact_bytes: int = 0
    scratch_dir: str = ""
    #: 子进程自己报的 maxrss（MB）；被杀/起不来时为 `None`（**不是 0**）
    max_rss_mb: Optional[float] = None
    #: 这一次真的用了哪个后端（`fork` / `subprocess`）—— 「哪个后端」也要是个读数，
    #: 不能让读者去猜（两个后端的**保证**一样，开销差两个数量级）
    executor: str = ""

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    @property
    def not_enforced(self) -> Tuple[str, ...]:
        """今天**没设上**的那几项。空元组 = 六项全部设上了（不含 `not_enforced` 的沉默）"""
        return tuple(e.item for e in self.enforcement if e.state == NOT_ENFORCED)

    @property
    def fully_enforced(self) -> bool:
        """显式读数：`True` 才表示「这一跑，六项上限都真的设上了」。

        ⚠️ 它**不**表示候选被隔离了（同 uid / 网络未关 / 无容器，见模块 docstring）。
        """
        return not self.not_enforced

    def enforcement_of(self, item: str) -> Optional[Enforcement]:
        return next((e for e in self.enforcement if e.item == item), None)

    def as_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "kind": self.kind, "detail": self.detail,
                "error": self.error, "exc_type": self.exc_type,
                "syntax_fields": dict(self.syntax_fields),
                "elapsed_ms": round(self.elapsed_ms, 3),
                "wall_ms": round(self.wall_ms, 3), "metrics": dict(self.metrics),
                "enforcement": [e.as_dict() for e in self.enforcement],
                "not_enforced": list(self.not_enforced),
                "fully_enforced": self.fully_enforced,
                "exit_code": self.exit_code, "signal": self.signal,
                "artifact_bytes": self.artifact_bytes,
                "max_rss_mb": self.max_rss_mb,
                "executor": self.executor}


# ─── 子程序源码 ──────────────────────────────────────────────
# 作为 `python3 -c` 的源串传给子进程（不落盘成第二个模块：本片只允许一个新模块）。
# ⚠️ 它**必须**与 `_execute_python` 今天的语义逐字等价：`ns = {}`、`ns["kernel"]`、
#    返回值原样、耗时只量 `kernel(**inputs)` 这一次调用。
_CHILD_SOURCE = r'''
import os, pickle, resource, sys, time

item_map = [
    ("cpu",         getattr(resource, "RLIMIT_CPU", None),        None),
    ("memory",      getattr(resource, "RLIMIT_AS", None),         None),
    ("file_size",   getattr(resource, "RLIMIT_FSIZE", None),      None),
    ("processes",   getattr(resource, "RLIMIT_NPROC", None),      None),
]

def _set(item, which, value):
    """设一项上限。**设不上就如实记 not_enforced**，绝不假装设上了。"""
    if which is None:
        return {"item": item, "state": "not_enforced", "requested": value,
                "reason": "这个平台没有这一项 (resource 模块里不存在)", "reading": ""}
    if value is None:
        return {"item": item, "state": "not_enforced", "requested": None,
                "reason": "调用方没有请求这一项 (None 不等于 0)", "reading": ""}
    try:
        cur_soft, cur_hard = resource.getrlimit(which)
        hard = value if cur_hard == resource.RLIM_INFINITY else min(cur_hard, value)
        # 软硬上限**一起降**（不是只降 soft）：这样这些上限在子进程里是**不可逆**的 ——
        # 候选自己调 `setrlimit` 也抬不回去（它能把硬上限再降，抬不高）。代价是**父侧也**
        # 抬不回去，所以「产物要装进一个文件」这件事必须靠 `Limits` 的不变量留出余量
        # （见 `Limits.__post_init__` 与 2026-10-04 那次 LeetCUDA 实测）。
        resource.setrlimit(which, (value, max(value, hard)))
        now = resource.getrlimit(which)
        return {"item": item, "state": "enforced", "requested": value,
                "reason": "", "reading": "%s soft=%d hard=%d" % (item, now[0], now[1])}
    except Exception as exc:
        return {"item": item, "state": "not_enforced", "requested": value,
                "reason": "%s: %s" % (type(exc).__name__, exc), "reading": ""}

def main():
    # Python 层的 stdout/stderr 也要指向同一个 fd: 测试环境里 (pytest 的捕获对象)
    # 它们**不是** fd 1 的包装 —— 只 dup2 的话候选的 print 会漏到别处去, 日志就是空的。
    # 放在这里 (而不是只在 fork 那一侧) 是因为两个后端的子侧行为只能有一个定义处。
    try:
        sys.stdout = os.fdopen(os.dup(1), "w", buffering=1)
        sys.stderr = sys.stdout
    except Exception:                    # noqa: BLE001
        pass
    with open(sys.argv[1], "rb") as fh:
        job = pickle.load(fh)
    limits = job["limits"]
    enforcement = []
    requests = job["rlimits"]          # 生产者: Limits.rlimit_requests()（键名与 Enforcement.item 同源）
    for item, which, _ in item_map:
        enforcement.append(_set(item, which, requests.get(item)))
    enforcement.append({"item": "wall", "state": "enforced",
                        "requested": limits.get("wall_ms"),
                        "reason": "", "reading": "父侧墙钟 %s ms" % limits.get("wall_ms")})
    enforcement.append({"item": "artifact", "state": "enforced",
                        "requested": limits.get("artifact_bytes"),
                        "reason": "", "reading": "父侧产物字节上限"})

    result = {"status": "ok", "kind": None, "detail": "", "output": None,
              "error": None, "elapsed_ms": 0.0, "metrics": {},
              "enforcement": enforcement, "hit": None, "maxrss_mb": None}

    mode = job.get("mode", "run")          # "run" / "compile"（见父侧 MODE_* 闭合词汇）

    def _exec_only():
        ns = {}
        # `imports` 只服务**编译模式**里需要预置模块的那条路（Triton: triton / tl / torch）。
        # run 模式永远是空 dict ⇒ `exec(code, {})` 与搬进子进程之前逐字等价。
        for _name, _mod in (job.get("imports") or {}).items():
            ns[_name] = __import__(_mod, fromlist=["*"])
        exec(job["code"], ns)                      # 与今天逐字等价: 空 ns (run 模式)
        return ns

    def _exec_and_call():
        if "input_protocol" in job:
            from operator_agent.input_fidelity import decode_inputs, describe_inputs, PROTOCOL
            runtime_inputs = decode_inputs(job["input_protocol"])
            # Read the reconstructed tensors BEFORE exec(candidate), not the
            # declarations and not values that the candidate may have mutated.
            result["metrics"]["input_fidelity"] = {
                "protocol": PROTOCOL, "observed": describe_inputs(runtime_inputs)}
        else:
            runtime_inputs = job["inputs"]
        ns = _exec_only()
        kernel = ns["kernel"]
        t0 = time.perf_counter()
        out = kernel(**runtime_inputs)
        return out, (time.perf_counter() - t0) * 1000.0

    def _compile_only():
        # 只做编译期那一步: exec 之后**不调用** kernel。
        t0 = time.perf_counter()
        ns = _exec_only()
        return ns, (time.perf_counter() - t0) * 1000.0

    try:
        def _dispatch():
            if mode == "compile":
                return _compile_only()
            return _exec_and_call()
        if job["shield"]:
            from operator_agent.anticheat import shield_torch
            with shield_torch():
                got, elapsed = _dispatch()
        else:
            got, elapsed = _dispatch()
        if mode == "compile":
            # 「有没有 kernel / 可不可调」只能由**子进程**回报 —— 它手里才有 exec 之后的 ns。
            # 判据本身（该抛 SyntaxError 还是 TypeError）与文案留在调用方，只有一处。
            result["metrics"] = {"has_kernel": "kernel" in got,
                                 "kernel_callable": callable(got.get("kernel"))}
        else:
            result["output"] = got
        result["elapsed_ms"] = elapsed
    except BaseException as exc:                    # noqa: BLE001 —— 候选抛什么都要结构化回来
        import traceback
        # `error` 用 **str(exc)**: 今天 `MockGPU.execute` 写进 `result.error` 的就是 `str(e)`,
        # 所以这一格必须逐字等价 —— 类型前缀只进 detail, 不进 error。
        bare = str(exc)
        text = "%s: %s" % (type(exc).__name__, bare)
        from operator_agent.anticheat import lib_fallback_from_error, lib_fallback_metrics
        hit = lib_fallback_from_error(text)
        frames = traceback.extract_tb(exc.__traceback__)
        # 指到**候选自己**那一帧 (动态调用栈的最后一帧常常在 stdlib 里, 对 Fixer 没用)
        cand = [f for f in frames if f.filename == "<string>"]
        last = (cand or frames or [None])[-1]
        where = ("" if last is None else
                 " @ %s:%d in %s" % ("<candidate>" if last.filename == "<string>"
                                     else os.path.basename(last.filename),
                                     last.lineno, last.name))
        if hit:
            result["status"] = "cheat"
            result["hit"] = hit
        else:
            # 「异常 → 哪一档」只有一处定义 (父侧 `classify_exception`): 子程序 import 它,
            # 不自己再写一份 —— 否则两个后端 / 父子两侧迟早各漂一份
            from operator_agent.exec_guard import classify_exception
            status, kind = classify_exception(exc)
            result["status"] = status
            result["kind"] = kind
        result["error"] = bare
        # 类型名要单独记一格: 编译那条路按它分两支 (`SyntaxError` 原样重抛 / 其余包成
        # `RuntimeError("Compile error: …")`), 而 `detail` 里的 `"SyntaxError: …"` 前缀
        # 是**给人读的**, 不该被上层的判据拿去做字符串匹配。
        result["exc_type"] = type(exc).__name__
        if isinstance(exc, SyntaxError):
            # `SyntaxError` 的**其它字段**也要带回来: `str()` 之外, `msg` / `lineno` /
            # `offset` / `text` / `filename` 是它的契约的一部分 ——
            # `verifier.syntax_error_report()` 读的就是这四个, 而跨进程重建时它们会
            # **静默丢光**（`str()` 逐字相同 ⇒ 不会红, 但读者拿到的是 None）。
            result["syntax_fields"] = {
                "msg": getattr(exc, "msg", None),
                "filename": getattr(exc, "filename", None),
                "lineno": getattr(exc, "lineno", None),
                "offset": getattr(exc, "offset", None),
                "text": getattr(exc, "text", None),
            }
        try:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
            result["maxrss_mb"] = rss
            where += " (子进程 maxrss ≈ %.1f MB)" % rss      # 失败原因带读数
        except Exception:                # noqa: BLE001 —— 读数拿不到不该盖住失败本身
            pass
        result["detail"] = text + where
        # 盾的命中证据由**子进程**这一侧生产 (唯一生产者: anticheat.lib_fallback_metrics)。
        # 先取本进程的命中 (盾就装在这个进程里, 它是权威), 取不到再用错误文本认领。
        metrics = lib_fallback_metrics()
        if not metrics and hit:
            metrics = lib_fallback_metrics(remote_error=bare)
        if metrics:
            # **合并**而不是覆盖: 编译模式那两个读数 (`has_kernel` / `kernel_callable`) 是
            # 这一趟唯一的产物, 被盾的命中证据顶掉就等于编译结果没回来。
            result["metrics"] = {**(result.get("metrics") or {}), **metrics}
    if result["status"] == "ok":
        from operator_agent.anticheat import lib_fallback_metrics
        result["metrics"] = {**(result.get("metrics") or {}), **lib_fallback_metrics()}
    if result["maxrss_mb"] is None:
        try:
            result["maxrss_mb"] = resource.getrusage(
                resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
        except Exception:                # noqa: BLE001
            pass

    # ⚠️ 接下来写的是**传输文件**（结果 pickle），它和候选写的文件受**同一个** FSIZE 预算
    # —— 上限是 per-process 的，没法只豁免这一个文件。所以「产物装得下」这件事由
    # `Limits.__post_init__` 的不变量保证（`artifact_bytes <= file_size_bytes`），而不是在
    # 这里抬上限：`_set` 把软硬上限一起降了，进程内**抬不回去**
    # （2026-10-04 实测教训：LeetCUDA 的 64MB 产物正好撞上 64MB 的 FSIZE，父侧只读到一句
    # "pickle data was truncated"，把「上限打架」伪装成了「候选坏了」）。
    blob = pickle.dumps(result, protocol=4)
    cap = limits.get("artifact_bytes")
    if cap is not None and len(blob) > cap:
        # **超了要失败, 不许截断后当通过**（产物那一段是证据, 截断的证据不是证据）
        result = {"status": "limit_exceeded", "kind": "artifact", "detail":
                  "产物 %d 字节 > 上限 %d 字节" % (len(blob), cap),
                  "output": None, "error": None, "elapsed_ms": result["elapsed_ms"],
                  "exc_type": result.get("exc_type", ""),
                  "syntax_fields": result.get("syntax_fields", {}),
                  "metrics": result["metrics"], "enforcement": enforcement,
                  "hit": None, "artifact_bytes": len(blob),
                  "maxrss_mb": result["maxrss_mb"]}
        blob = pickle.dumps(result, protocol=4)
    with open(job["result_path"], "wb") as fh:
        fh.write(blob)

main()
'''


def _signal_name(signum: Optional[int]) -> str:
    if not signum:
        return ""
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"signal {signum}"


def classify_exception(exc: BaseException) -> Tuple[str, Optional[str]]:
    """候选抛出来的异常 → `(status, kind)`。**全模块唯一真源**：子程序 `import` 它。

    为什么放在父侧: ①两个后端（fork / subprocess）行为只能有一个定义处；②这样这一层
    是**可单测的**（例如本机 macOS 上 `RLIMIT_AS` 设不上、内存那一档走不到，但
    `MemoryError → (limit_exceeded, memory)` 这条判据仍然钉得住）。

    只认 `errno` 与异常类型：`EFBIG`(27) 是 Darwin 在 `RLIMIT_FSIZE` 到顶时的返回值
    （实测读数见模块 docstring，Linux 上通常是 `SIGXFSZ`）；`EAGAIN`(11/35) 与
    `BlockingIOError` 是 `RLIMIT_NPROC` 到顶；`ENOMEM`(12) 与 `MemoryError` 是内存。
    认不出来的**不算超限**（`crashed`）—— 「不知道」不许被读成「撞了上限」。
    """
    name = type(exc).__name__
    errno = getattr(exc, "errno", None)
    if name == "MemoryError" or errno == 12:
        return STATUS_LIMIT_EXCEEDED, KIND_MEMORY
    if name == "BlockingIOError" or errno in (11, 35):
        return STATUS_LIMIT_EXCEEDED, KIND_PROCESSES
    if errno in (27, 28):
        return STATUS_LIMIT_EXCEEDED, KIND_FILE_SIZE
    return STATUS_CRASHED, None


def candidate_error(guard: GuardResult) -> BaseException:
    """把子进程带回来的候选异常**按原类型**重建（调用方决定抛还是包）。

    为什么需要: 搬进子进程之后，候选抛的对象过了 pickle 就不再是原来那个类型。而
    **类型是契约的一部分** —— `MockGPU.compile` 靠 `SyntaxError` 分「语法错」与「其它
    编译错」两支，`kernel_db.failure_signature` 也按异常文本归类。类型名由子进程记在
    `GuardResult.exc_type`（不是从 `detail` 的前缀里猜）。

    拿不到那个类型（候选自定义的异常类）时退化成 `RuntimeError`，但**保留原名**：
    写成裸 `RuntimeError("…")` 会让读者以为候选抛的就是 RuntimeError。

    `SyntaxError` 额外把它自己的字段一起重建（`msg` / `filename` / `lineno` / `offset` /
    `text`）：`str()` 之外这几个也是契约 —— `verifier.syntax_error_report()` 读行号与列号，
    而「`str()` 一样所以没关系」正是**静默丢契约**的形状（改完不红，读者拿到 None）。
    """
    import builtins
    name = guard.exc_type or ""
    cls = getattr(builtins, name, None)
    message = guard.error or guard.detail
    if isinstance(cls, type) and issubclass(cls, SyntaxError) and guard.syntax_fields:
        fields = guard.syntax_fields
        msg = fields.get("msg") or message
        filename = fields.get("filename")
        if filename:
            return cls(msg, (filename, fields.get("lineno"), fields.get("offset"),
                             fields.get("text")))
        return cls(msg)
    if isinstance(cls, type) and issubclass(cls, BaseException):
        return cls(message)
    return RuntimeError(f"{name or '(没有类型名)'}: {message}")


def refusal_from(guard: GuardResult, *, where: str = "") -> GuardRefusal:
    """把「执行器拦下了候选」这一档做成异常对象（各后端只决定**在哪一层**抛它）。

    单独一个构造函数而不是每个后端各拼一句：这句话是**产物的一部分**（它会经 `Verifier`
    进 `error_log`，再进 Fixer 的处方），两个后端各写一遍就会漂成两种说法。
    """
    stage = f"{where} " if where else ""
    return GuardRefusal(
        f"{stage}被 §22.2 拦下 ({guard.status}"
        + (f"/{guard.kind}" if guard.kind else "") + f"): {guard.detail}",
        status=guard.status, kind=guard.kind, detail=guard.detail,
        not_enforced=guard.not_enforced, enforcement=guard.enforcement,
        executor=guard.executor)


def classify_death(exit_code: Optional[int], signum: Optional[int],
                   timed_out: bool) -> Tuple[str, Optional[str], str]:
    """纯函数：**没有子进程自报结果**时，从「怎么死的」判它是哪一档。

    有自报结果时（子进程写了一句话回来）不走这里 —— 那种情况子进程知道得更多。
    单独抽出来是为了可单测：本机杀进程的读数不该只活在集成用例里。
    """
    if timed_out:
        return STATUS_TIMEOUT, KIND_WALL, "墙钟到点，进程组已被杀"
    if signum == getattr(signal, "SIGXCPU", None):
        return STATUS_LIMIT_EXCEEDED, KIND_CPU, "撞上 CPU 时间上限 (SIGXCPU)"
    if signum == getattr(signal, "SIGXFSZ", None):
        return STATUS_LIMIT_EXCEEDED, KIND_FILE_SIZE, "撞上单文件大小上限 (SIGXFSZ)"
    if signum == getattr(signal, "SIGKILL", None):
        return STATUS_CRASHED, None, "被 SIGKILL 杀掉，且不是墙钟超时 (谁杀的不知道)"
    if signum:
        return STATUS_CRASHED, None, f"被 {_signal_name(signum)} 杀掉"
    if exit_code not in (0, None):
        return STATUS_CRASHED, None, f"异常退出 rc={exit_code}，且没有留下产物文件"
    return STATUS_CRASHED, None, "子进程没有任何产物文件（可能启动就失败了）"


class _Child:
    """一次执行的子进程句柄 —— `fork` 与 `subprocess` 两个后端在这里收成同一个形状。

    为什么要有这层：两个后端的**父侧逻辑**（等 / 杀 / 分类 / 读产物）必须一模一样，
    否则「同一条守卫在两个后端下行为不同」就没人钉得住了。
    """

    def __init__(self, pid: int, backend: str,
                 proc: Optional[subprocess.Popen] = None) -> None:
        self.pid = pid
        self.backend = backend
        self._proc = proc
        self.returncode: Optional[int] = None
        #: 进程组 id **按构造**就是子进程 pid（fork 那侧自己 `setsid()`；
        #: subprocess 那侧 `start_new_session=True` 也在 exec 前 setsid）。
        #: 为什么要记账而不是每次 `os.getpgid(pid)`：子进程被收割之后
        #: `getpgid` 就查不到了，而「它正常返回、但留了后台子进程」正是要靠这个 pgid
        #: 去收尾的（不然收尾那一步会静默失效 —— 第一版就是这么错的）。
        self.pgid = pid
        self._group_verified: Optional[bool] = None

    def wait(self, timeout_s: float) -> bool:
        """等它退出。`True` = 已退出（`returncode` 有效）；`False` = 到点还没退。

        幂等：已经收过的孩子再等一次直接返回 `True` —— `_kill_process_group` 里那次
        `wait` 可能已经把它收掉了，父侧流程不该因为「收过了」而炸
        （`os.waitpid` 对已收的孩子抛 `ChildProcessError`）。
        """
        if self.returncode is not None:
            return True
        if self._proc is not None:
            try:
                self._proc.wait(timeout=max(timeout_s, 0.0))
            except subprocess.TimeoutExpired:
                return False
            self.returncode = self._proc.returncode
            return True
        deadline = time.monotonic() + max(timeout_s, 0.0)
        while True:
            try:
                pid, status = os.waitpid(self.pid, os.WNOHANG)
            except ChildProcessError:
                return True
            if pid:
                self.returncode = os.waitstatus_to_exitcode(status)
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.005)

    @property
    def exit_code(self) -> Optional[int]:
        if self.returncode is None or self.returncode < 0:
            return None
        return self.returncode

    @property
    def signum(self) -> Optional[int]:
        return -self.returncode if (self.returncode or 0) < 0 else None

    def killpg(self, sig: int) -> bool:
        """杀**整个进程组**。返回是否真的发出去了。

        ⚠️ 一个必须防住的事故：如果子进程还没来得及 `setsid()`，`os.getpgid(pid)` 拿到的是
        **父进程自己的组**，照着它 `killpg` 会把 harness 一起杀掉。所以：子进程还活着时
        **核对一次** `pgid == pid`（它是组长）并记住结论；核对不过就只杀它自己 —— 那个
        窗口里子进程还没跑任何候选代码，不可能有孙子进程，退化成单杀也不漏孤儿。
        子进程已经被收割时 `getpgid` 查不到，这时用记账的 `self.pgid`（按构造等于 pid），
        因为「组长死了但组还在」正是要收尾的那一种。
        """
        if self._group_verified is None:
            try:
                self._group_verified = (os.getpgid(self.pid) == self.pid)
            except ProcessLookupError:
                self._group_verified = True      # 组长已经没了 ⇒ 用记账的 pgid
        try:
            if self._group_verified:
                os.killpg(self.pgid, sig)
            else:
                os.kill(self.pid, sig)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return False


def _pick_executor(requested: str) -> str:
    """`auto` 的判据只有一条、且**可观察**：父进程此刻是不是只有主线程。

    为什么这条判据：`fork` 的代价（毫秒级）远低于 `subprocess`（每次重付 `import torch`，
    本机实测 ≈860ms），而 `fork` 在多线程进程里可能死锁 —— 子进程只继承**调用线程**，
    若另一个线程正握着子进程要用的锁（比如 import 锁、malloc 锁），子进程就卡在那里。
    所以：单线程 ⇒ fork；有别的线程 ⇒ 老老实实起新解释器。

    ⚠️ 这是**启发式**，不是证明：C 扩展自己起的线程（不注册到 `threading`）它看不见。
    要绝对避免 fork 的调用方可以显式传 `executor="subprocess"`。
    """
    if requested != EXECUTOR_AUTO:
        return requested
    if not hasattr(os, "fork"):
        return EXECUTOR_SUBPROCESS
    import threading
    return EXECUTOR_FORK if threading.active_count() == 1 else EXECUTOR_SUBPROCESS


def _fork_child(job_path: str, workdir: str, log_path: str) -> None:
    """`fork` 出来的子进程这一侧。**不返回**（成功 `os._exit(0)`，失败 `os._exit(70)`）。

    跑的是**同一份** `_CHILD_SOURCE` —— 两个后端的子侧行为只能有一个定义处，
    否则「同一条守卫在两个后端下行为不同」就没人钉得住。
    """
    import traceback

    try:
        os.setsid()                      # 自己一个会话/进程组 ⇒ killpg 能连孙子一起收
    except OSError:
        pass
    try:
        os.chdir(workdir)
        # 用内建 `open` 而不是 `os.open(..., O_CREAT)`：§22.1 有一条结构守卫
        # （`tests/unit/test_run_layout.py::TestOnlyOneDoorInTheHarness`）钉着「`operator_agent/`
        # 里打开写窗的能力只有 `run_layout.py` 一处」。日志文件不是那条纪律的对象，
        # 但**不要**为了自己方便去撞别人的不变量 —— 这里就按同样的方式开。
        with open(log_path, "ab") as log_fh:
            os.dup2(log_fh.fileno(), 1)
            os.dup2(log_fh.fileno(), 2)
    except Exception:                    # noqa: BLE001 —— 日志接不上不该阻断执行
        pass
    try:
        sys.argv = ["exec_guard-child", job_path]
        exec(compile(_CHILD_SOURCE, "<exec_guard_child>", "exec"),
             {"__name__": "__main__", "__file__": "<exec_guard_child>"})
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except BaseException:                # noqa: BLE001
        try:
            sys.stderr.write("exec_guard: 子进程在写产物之前就炸了\n")
            traceback.print_exc()
            sys.stderr.flush()
        except Exception:                # noqa: BLE001
            pass
        os._exit(70)


def _read_bounded(path: str, cap: int) -> Tuple[bytes, int]:
    """读一个文件，最多读 `cap` 字节；返回 (内容, 文件实际大小)"""
    size = os.path.getsize(path) if os.path.exists(path) else 0
    with open(path, "rb") as fh:
        return fh.read(cap), size


def run_candidate(
    code: str,
    inputs: Dict[str, Any],
    *,
    limits: Optional[Limits] = None,
    scratch_dir: Optional[str] = None,
    shield: bool = True,
    keep_scratch: bool = False,
    python: Optional[str] = None,
    grace_ms: int = 200,
    executor: str = EXECUTOR_AUTO,
    preserve_inputs: bool = False,
) -> GuardResult:
    """在受控子进程里跑一段候选代码，返回 `GuardResult`。

    Args:
        code:   候选源码（`kernel(**inputs)` 是入口，与今天同一个约定）
        inputs: 传给 `kernel` 的关键字参数（需要可 pickle —— 子进程要拿到它）
        preserve_inputs: use the bounded NumPy storage/view protocol and return
            pre-candidate input readback. MockGPU Python always requests this;
            the default retains compatibility for generic guard callers.
        limits: `Limits`；`None` = 用默认（有限但宽松）
        scratch_dir: 子进程的 cwd 与临时文件落点。`None` = 每次新建临时目录并在结束时删掉。
            传了就**不删**（调用方要自己看候选写了什么时用得上）
        shield: 装不装运行时反作弊盾（默认装，与今天一致）
        keep_scratch: `scratch_dir=None` 时是否保留那个临时目录（诊断用）
        python: 子进程解释器（`subprocess` 后端用；默认 `sys.executable`）
        grace_ms: 墙钟到点后先 SIGTERM 给多少毫秒再 SIGKILL
        executor: `fork` / `subprocess` / `auto`（默认）。两个后端的**保证完全一样**
            （rlimit、进程组、墙钟、产物上限都在子进程这一侧），差别只有开销与 fork 的
            线程安全隐患；`auto` 的判据见 `_pick_executor`。实际用了哪个进
            `GuardResult.executor`。

    Returns:
        `GuardResult`：`status` 五档之一；`enforcement` 是**逐项**「设上了没有」；
        `not_enforced` / `fully_enforced` 是显式读数（不许读成「没超限」）。
    """
    return _run_guarded(code, inputs, mode=MODE_RUN, imports={}, limits=limits,
                        scratch_dir=scratch_dir, shield=shield,
                        keep_scratch=keep_scratch, python=python,
                        grace_ms=grace_ms, executor=executor,
                        preserve_inputs=preserve_inputs)


def run_compile(
    code: str,
    *,
    limits: Optional[Limits] = None,
    imports: Optional[Dict[str, str]] = None,
    scratch_dir: Optional[str] = None,
    shield: bool = False,
    keep_scratch: bool = False,
    python: Optional[str] = None,
    grace_ms: int = 200,
    executor: str = EXECUTOR_AUTO,
) -> GuardResult:
    """**编译模式**的受控执行：`exec(code)` 之后不调用 `kernel`（§22.2 的「编译超时」）。

    与 `run_candidate` 共用同一套子进程机制（墙钟 + rlimit + 进程组 + 产物上限 + 逐项
    强制读数），差别只有两处，且两处都是**语义**差别而不是实现差别：

    * 入口只 `exec`，不调 `kernel(**inputs)` —— 编译期本来就不该跑核；
    * `imports` 是给 Triton 那条路预置模块的（搬进子进程之前是
      `local_ns = {"triton": …, "tl": …, "torch": …}`），run 模式永远是空 dict。

    `shield` 默认 **False**：编译期原先（harness 进程内 `exec`）没有装盾，这里保持同一
    语义；要看编译期有没有库回退的调用方显式传 `True`。

    Returns:
        `GuardResult`。`status == "ok"` 只表示「这份源码能被 exec、且是否定义了 `kernel`
        已经报出来」，**不表示**它能跑或算得对；`metrics["has_kernel"]` /
        `metrics["kernel_callable"]` 是子进程回报的两格。编译**失败**时返回
        `status=STATUS_CRASHED` + `exc_type`（不是抛异常）—— 由调用方决定重抛什么类型。
    """
    return _run_guarded(code, {}, mode=MODE_COMPILE, imports=dict(imports or {}),
                        limits=limits, scratch_dir=scratch_dir, shield=shield,
                        keep_scratch=keep_scratch, python=python,
                        grace_ms=grace_ms, executor=executor)


def _run_guarded(
    code: str,
    inputs: Dict[str, Any],
    *,
    mode: str,
    imports: Dict[str, str],
    limits: Optional[Limits],
    scratch_dir: Optional[str],
    shield: bool,
    keep_scratch: bool,
    python: Optional[str],
    grace_ms: int,
    executor: str,
    preserve_inputs: bool = False,
) -> GuardResult:
    """两种模式**共用**的那一段：起子进程 / 等或杀 / 读产物 / 分类 / 逐项读数。

    单独一层而不是让两个公开入口各写一遍：两个后端（fork / subprocess）与两种模式
    （run / compile）的**保证必须逐字相同**，而「同一件事写两遍」正是它们开始漂的起点。
    """
    if mode not in MODES:
        raise ValueError(f"mode 必须是 {MODES} 之一: {mode!r}")
    limits = limits_for(mode, limits)
    # Encode before creating scratch files; invalid input leaves no leaked workdir.
    packet = None
    if preserve_inputs:
        from .input_fidelity import encode_inputs
        packet = encode_inputs(inputs)
    created = scratch_dir is None
    workdir = scratch_dir or tempfile.mkdtemp(prefix="exec-guard-")
    os.makedirs(workdir, exist_ok=True)
    job_path = os.path.join(workdir, "job.pkl")
    result_path = os.path.join(workdir, "result.pkl")
    log_path = os.path.join(workdir, "child.log")

    job = {"code": code, "inputs": inputs, "shield": bool(shield),
           "mode": mode, "imports": dict(imports or {}),
           "result_path": result_path, "limits": limits.as_dict(),
           "rlimits": limits.rlimit_requests()}
    if packet is not None:
        job.pop("inputs")
        job["input_protocol"] = packet
    with open(job_path, "wb") as fh:
        pickle.dump(job, fh, protocol=4)

    backend = _pick_executor(executor)
    started = time.monotonic()
    child = None
    if backend == EXECUTOR_FORK:
        # fork 前把自己的缓冲刷掉: 子进程会继承这份缓冲, 不刷就会**重复**打印一遍
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            pid = os.fork()
        except OSError:
            # fork 自己起不来（比如进程数上限已到）**不能**被读成「候选失败」:
            # 退回新解释器后端, 保证一模一样, 只是慢。用哪个后端仍进 `executor`。
            backend = EXECUTOR_SUBPROCESS
        else:
            if pid == 0:
                _fork_child(job_path, workdir, log_path)     # 不返回
            child = _Child(pid, backend)
    if child is None:
        env = dict(os.environ)
        env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
        log_fh = open(log_path, "wb")
        proc = subprocess.Popen(
            [python or sys.executable, "-c", _CHILD_SOURCE, job_path],
            cwd=workdir, env=env, stdin=subprocess.DEVNULL,
            stdout=log_fh, stderr=subprocess.STDOUT,
            start_new_session=True,      # 自己一个会话/进程组 ⇒ killpg 能连孙子一起收
        )
        log_fh.close()
        child = _Child(proc.pid, backend, proc=proc)

    timed_out = False
    if not child.wait(max(limits.wall_ms, 1) / 1000.0):
        timed_out = True
        _kill_process_group(child, grace_ms=grace_ms)
        if not child.wait(2.0):          # 进程组都杀了还不退: 只能报出来
            timed_out = True
    else:
        # **正常结束也要清一次**: 候选可能自己起了后台子进程然后正常返回 —— 只杀直接
        # 子进程的话它们会被 reparent 成孤儿（§22.2 的「进程树清理」不是只管超时那一路）。
        # 子进程已经收掉了, 所以这里杀的是同一个进程组里剩下的孙子进程;
        # 组早没了就是 ProcessLookupError, 静默返回。
        child.killpg(signal.SIGKILL)
    wall_ms = (time.monotonic() - started) * 1000.0
    exit_code = child.exit_code
    signum = child.signum

    log_bytes, log_size = _read_bounded(log_path, limits.log_bytes)
    log = log_bytes.decode("utf-8", "replace")
    if log_size > limits.log_bytes:
        log += f"\n...（日志被截到 {limits.log_bytes} 字节，原文件 {log_size} 字节）"

    report = None
    artifact_bytes = 0
    if os.path.exists(result_path):
        blob, artifact_bytes = _read_bounded(result_path, limits.artifact_bytes + 1)
        if artifact_bytes <= limits.artifact_bytes:
            try:
                report = pickle.loads(blob)
            except Exception as exc:                 # noqa: BLE001
                # 一个**必须说清楚**的死法: 传输文件正好停在单文件上限上 ⇒ 它不是「候选坏了」,
                # 而是「两个上限打架」（产物装不进那个文件）。判据是「大小恰好等于上限」,
                # 不是猜 —— 猜的话就等于把「不知道」写成「撞了上限」。
                if limits.artifact_bytes >= limits.file_size_bytes:
                    report = {"status": STATUS_LIMIT_EXCEEDED, "kind": KIND_FILE_SIZE,
                              "detail": f"产物文件被单文件上限截断: 文件 {artifact_bytes} 字节,"
                                        f" file_size_bytes 上限 {limits.file_size_bytes} ——"
                                        f" 产物预算 {limits.artifact_bytes} 装不进那个文件"
                                        f"（`Limits` 的构造期不变量本应拦住这种组合）"}
                else:
                    report = {"status": STATUS_CRASHED,
                              "detail": f"产物文件读不回来: {type(exc).__name__}: {exc}"
                                        f"（文件 {artifact_bytes} 字节；"
                                        f"artifact_bytes={limits.artifact_bytes} "
                                        f"file_size_bytes={limits.file_size_bytes}）"}
        else:
            report = {"status": STATUS_LIMIT_EXCEEDED, "kind": KIND_ARTIFACT,
                      "detail": f"产物文件 {artifact_bytes} 字节 > 上限 "
                                f"{limits.artifact_bytes} 字节（父侧复核）"}

    enforcement = _enforcement_of(report, limits, signum)
    #: 三个**分支各自决定**的字段不进 base：`output` / `elapsed_ms` / `metrics`
    base = dict(wall_ms=wall_ms, log=log, enforcement=enforcement,
                exit_code=exit_code, signal=signum,
                artifact_bytes=artifact_bytes, scratch_dir=workdir,
                executor=backend,
                max_rss_mb=(report or {}).get("maxrss_mb"))

    if timed_out:
        res = GuardResult(status=STATUS_TIMEOUT, kind=KIND_WALL, output=None,
                          elapsed_ms=0.0, metrics={},
                          detail=f"墙钟 {limits.wall_ms}ms 到点，进程组已被杀"
                                 f"（实际 wall {wall_ms:.1f}ms）",
                          error=f"exec_guard: 墙钟超时 ({limits.wall_ms}ms)", **base)
    elif report is None:
        status, kind, detail = classify_death(exit_code, signum, timed_out)
        res = GuardResult(status=status, kind=kind, detail=detail, output=None,
                          elapsed_ms=0.0, metrics={},
                          error=f"exec_guard: {detail}", **base)
    else:
        status = str(report.get("status") or STATUS_CRASHED)
        if status not in STATUSES:
            status = STATUS_CRASHED
        res = GuardResult(status=status, kind=report.get("kind"),
                          detail=str(report.get("detail") or ""),
                          output=report.get("output"),
                          error=report.get("error"),
                          exc_type=str(report.get("exc_type") or ""),
                          syntax_fields=dict(report.get("syntax_fields") or {}),
                          elapsed_ms=float(report.get("elapsed_ms") or 0.0),
                          metrics=dict(report.get("metrics") or {}), **base)

    if created and not keep_scratch:
        shutil.rmtree(workdir, ignore_errors=True)
    return res


def _enforcement_of(report: Optional[dict], limits: Limits,
                    signum: Optional[int] = None) -> Tuple[Enforcement, ...]:
    """逐项强制读数。子进程没回来（被杀/起不来）时，**从父侧的口径补一份**。

    为什么父侧也要能给一份：被杀的子进程**没有机会**报告它设上了什么。
    那时「墙钟」与「产物上限」是父侧真正在执行的（`enforced`）；CPU / 内存 / 文件大小 /
    进程数四项默认**无法确认** ⇒ 报 `not_enforced` + 原因「子进程没能回报」，
    而不是默认它们设上了（本仓 I65/I11 那一族）。

    一个例外：**死于 `SIGXCPU` / `SIGXFSZ` 本身就是那两项生效的证据** —— 内核不会凭空
    发这两个信号。这时把对应项记成 `enforced`，理由写清「它拦住了」而不是「设上了」：
    值是子进程报回来的，而它已经死了，所以 `reading` 里没有具体数字，只有这个事实。
    """
    if report and report.get("enforcement"):
        return _ordered(Enforcement(**e) for e in report["enforcement"])
    out = [Enforcement(item=ITEM_WALL, state=ENFORCED, requested=limits.wall_ms,
                       reading="父侧墙钟（父侧执行，无需子进程回报）"),
           Enforcement(item=ITEM_ARTIFACT, state=ENFORCED, requested=limits.artifact_bytes,
                       reading="父侧产物字节上限（父侧执行）")]
    fired = {getattr(signal, "SIGXCPU", None): (ITEM_CPU, limits.cpu_s),
             getattr(signal, "SIGXFSZ", None): (ITEM_FILE_SIZE, limits.file_size_bytes)}
    out.append(Enforcement(
        item=ITEM_CPU, state=ENFORCED if signum == getattr(signal, "SIGXCPU", None)
        else NOT_ENFORCED, requested=limits.cpu_s,
        reason="" if signum == getattr(signal, "SIGXCPU", None)
        else "子进程没能回报（被杀/起不来），这一项无法确认",
        reading="子进程死于 SIGXCPU ⇒ 这一项真的生效了（数字没回报，它已经死了）"
        if signum == getattr(signal, "SIGXCPU", None) else ""))
    out.append(Enforcement(item=ITEM_MEMORY, state=NOT_ENFORCED,
                           requested=limits.memory_bytes,
                           reason=("调用方没有请求这一项 (None 不等于 0)"
                                   if limits.memory_bytes is None else
                                   "子进程没能回报（被杀/起不来），这一项无法确认")))
    out.append(Enforcement(
        item=ITEM_FILE_SIZE, state=ENFORCED if signum == getattr(signal, "SIGXFSZ", None)
        else NOT_ENFORCED, requested=limits.file_size_bytes,
        reason="" if signum == getattr(signal, "SIGXFSZ", None)
        else "子进程没能回报（被杀/起不来），这一项无法确认",
        reading="子进程死于 SIGXFSZ ⇒ 这一项真的生效了（数字没回报，它已经死了）"
        if signum == getattr(signal, "SIGXFSZ", None) else ""))
    out.append(Enforcement(item=ITEM_PROCESSES, state=NOT_ENFORCED,
                           requested=limits.processes,
                           reason="子进程没能回报（被杀/起不来），这一项无法确认"))
    del fired                          # 只为上面那张对照表存在, 别留个没人读的局部
    return _ordered(out)


def _ordered(records) -> Tuple[Enforcement, ...]:
    """把逐项读数按 `ITEMS` 的**唯一顺序**排好。

    顺序也要有唯一出处: 子进程报的顺序是「先 rlimit 四项再 wall/artifact」, 父侧补的顺序
    不同, 读者逐个对位就会错位 —— 排序由 `ITEMS` 定义, 两边都过这一道。
    """
    by_item = {e.item: e for e in records}
    return tuple(by_item[i] for i in ITEMS if i in by_item)


def _kill_process_group(child: _Child, grace_ms: int = 200) -> None:
    """杀掉**整个进程组**：先 SIGTERM 给 `grace_ms`，再 SIGKILL。

    为什么必须杀进程组而不是只杀直接子进程：候选可以自己起子进程（编译、多进程、
    甚至 fork 炸弹的温和版），只杀直接子进程会把它们全留下 —— 「超时」变成
    「留一堆孤儿」是本仓明确不许的形状（§22.2 的「进程树清理」）。

    SIGTERM 先行的理由：正常候选（比如卡在很长的循环里）有机会自己收拾退场；
    `grace_ms` 之后一律 SIGKILL —— **不**接受「它还在跑所以不算超时」。
    """
    for sig, wait_s in ((signal.SIGTERM, grace_ms / 1000.0),
                        (signal.SIGKILL, 0.0)):
        if not child.killpg(sig):
            return
        if wait_s and child.wait(wait_s):
            return
