"""跨进程的同卡占用声明 (CHECKLOG I68 的修复半片)

为什么需要它: 控制端的 `_EXEC_LOCKS` 是**进程内** threading 锁, 挡不住两个
harness 进程静默共卡 —— 真机实测 (I68, H20, 4096³ fp32): 串行 6.2ms, 两进程
并发 13.6/11.6ms (2.1× 膨胀), 其中一条仍判 `valid`、`gpu_util_pct=0`。现有
两个采样对这种场景都是瞎的:

  * `nvidia-smi utilization.gpu` 是 ~1 秒窗口的**时刻**采样;
  * `--query-compute-apps` 只看得见**已建 CUDA 上下文**的进程 —— 两个 ssh
    进程同时起时, 对手还在 import torch, 我们的采样早就结束了。

claim 文件的生命期 = **整个 harness 进程** (run_kernel import 时写入、退出时
atexit 拆除): 对手哪怕还在 import, 它的占卡意图已经可见。

落点纪律 (doc 52 / #69): 会合目录是**固定已知**的 (协议要求 —— 别人要能找到),
但文件名不可预测 (pid + 随机后缀)、以 O_CREAT|O_EXCL + 0600 写入; 只**读**
别人的声明, 从不执行它们。整个协议尽力而为: 写不进 (权限/只读文件系统) 不许
挡测量本身 —— 声明是观测面, 不是入场券。

与 `gpu_other_procs` 的分工: 那个键数的是**任意**计算进程 (8 租户共享卡上几乎
恒真, 只能作提示); 本协议只数 **harness 进程** —— 同一个测量框架的另一个运
行实例, 它一旦开始测量就是硬竞争 (I68 的 2.1×), 所以能当判据。
"""
import atexit
import json
import os
import time
import uuid


def claim_dir() -> str:
    """声明的会合目录: host 上一处固定路径 (env 可覆盖, 单测注入用)。

    放 `~` 而不是 `/tmp`: claim 只承诺看见**同用户**的 harness 进程 (I68 的目标
    场景就是「两个 agent / 两个任务」); 跨用户的干扰仍由 util 采样兜底。`~` 还
    避开了 /tmp 的世界可写区 (doc 52 的落点纪律)。
    """
    return os.environ.get("OPERATOR_AGENT_GPU_CLAIM_DIR") or os.path.expanduser(
        "~/.operator_agent_gpu_claims")


def write_claim(gpu_id: int, *, pid: int = None, directory: str = None,
                runner: str = "") -> "str | None":
    """写本进程的占卡声明, 返回文件路径; 写不进返回 None (尽力而为, 不抛)。

    `pid`/`directory` 是给单测注入的; 只有 pid 省略 (= 本进程) 时才注册 atexit
    拆除 —— 测试造的假 pid 不许挂到真实进程的退出钩子上。
    """
    pid = os.getpid() if pid is None else pid
    directory = directory or claim_dir()
    try:
        os.makedirs(directory, exist_ok=True)
        # 名字带随机后缀: 固定名会让两次并发启动互相覆盖 (还会被 /tmp 那类
        # symlink 竞态利用, doc 52); O_EXCL 保证即便撞名也绝不改写别人。
        path = os.path.join(directory, f"claim-{pid}-{uuid.uuid4().hex[:8]}.json")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"pid": pid, "gpu": int(gpu_id),
                           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                           "started_at": round(time.time(), 3),
                           "runner": runner}, fh)
        except Exception:
            _unlink_quietly(path)      # 写了一半的声明比没有更糟 (scan 会跳过, 但留垃圾)
            raise
    except OSError:
        return None
    if pid == os.getpid():
        atexit.register(remove_claim, path)
    return path


def remove_claim(path) -> None:
    """拆除自己的声明。不存在/已拆/权限问题一律静默 (atexit 里不许抛)。"""
    if not path:
        return
    _unlink_quietly(path)


def _unlink_quietly(path) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def pid_alive(pid: int) -> bool:
    """同宿主机上这个 pid 活着吗。别的用户的进程 → PermissionError = 活着。"""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True          # 活着, 只是不归我们信号
    except OSError:
        return False         # 坏 pid 之类「查不了」: 不算活 (宁可漏报陈旧声明)


def scan_claims(gpu_id: int, *, exclude_pid: int, directory: str = None) -> list:
    """同卡上**还活着**的其他 harness 声明 (排除自己)。

    坏文件 (崩溃写了一半 / 非本协议的 .json) 与死进程的陈旧声明一律跳过 ——
    scan 是观测面, 任何输入都不许让它抛。
    """
    directory = directory or claim_dir()
    peers = []
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return peers          # 目录不存在 = 没有任何声明, 不是错误
    for name in names:
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(directory, name), encoding="utf-8") as fh:
                claim = json.load(fh)
        except (OSError, ValueError):
            continue
        pid, gpu = claim.get("pid"), claim.get("gpu")
        if isinstance(pid, bool) or not isinstance(pid, int):
            continue
        if isinstance(gpu, bool) or not isinstance(gpu, int):
            continue
        if pid == exclude_pid or gpu != gpu_id:
            continue
        if not pid_alive(pid):
            continue          # 崩溃没拆的陈旧声明: 死人不算占用
        peers.append(claim)
    return peers
