"""
远程代码执行器
支持两种连接模式:
  1. SSH 模式 (推荐): 通过 SSH 连接远程 GPU 服务器，自动部署并执行脚本
  2. HTTP 模式: 通过 HTTP API 调用 (如果远程有接口服务)
  3. 本地模式: fallback 到本地 CPU 执行

环境变量 (platform="nvidia"):
  REMOTE_SSH_HOST     - SSH 主机或 ~/.ssh/config 里的 Host 别名 (默认 algorithm-h20)
  REMOTE_SSH_PORT     - SSH 端口 (默认不设, 交由 ssh config 的 Port 解析)
  REMOTE_SSH_USER     - SSH 用户名 (默认不设, 交由 ssh config 的 User 解析)
  REMOTE_SSH_KEY      - SSH 私钥路径 (默认不设, 交由 ssh config 的 IdentityFile 解析)
  REMOTE_WORK_DIR     - 远程工作目录 (默认 ~/operator_agent)
  REMOTE_RUNNER_PATH  - 远程脚本路径 (默认 {work_dir}/run_kernel.py)
  REMOTE_HTTP_URL     - HTTP 接口地址 (默认不设)
  LLM_BASE_URL        - LLM 端点 (必填, 见 llm_client.py)

  真实地址/端口/用户名/密钥一律不写进仓库, 统一放在 ~/.ssh/config 与环境变量里。

环境变量 (platform="kunlunxin", 服务器已下线, 保留作为接入其他 GPU 平台的范例):
  REMOTE_SSH_HOST_KL / REMOTE_SSH_PORT_KL / REMOTE_SSH_USER_KL / REMOTE_SSH_KEY_KL
  REMOTE_WORK_DIR_KL / REMOTE_RUNNER_PATH_KL / REMOTE_PYTHON_KL
  脱敏约定与 nvidia 分支一致: host 默认只是 ssh 别名 kunlunxin-p800,
  port / user / key 默认不设, 交由 ~/.ssh/config 的 Host 块解析。
"""

import os
import json
import shlex
import subprocess
import tempfile
import threading
import hashlib
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Optional

from .environment import EnvironmentChecker, DependencyAnalyzer, EnvironmentReport, CheckItem
from .backend_caps import Backend, Cap
from .hardware_caps import detect_hardware_caps
from .measurement import (TRANSPORT_ANSWERED, TRANSPORT_KEY, transport_outcome)


# 按执行目标串行化 execute/benchmark, 防止并发任务共用同一 GPU 互相污染 benchmark
_EXEC_LOCKS: dict[str, threading.Lock] = {}
_EXEC_LOCKS_GUARD = threading.Lock()


def _get_exec_lock(key: str) -> threading.Lock:
    with _EXEC_LOCKS_GUARD:
        return _EXEC_LOCKS.setdefault(key, threading.Lock())


def _transport_outcome_of(exc) -> str:
    """一次调用失败时, 归到「听到回答」/「没听到回答」那一格 (单一实现, 四个出口共用)

    `ssh_rc` 是 `_ssh_exec` 挂在异常上的那个码: 255 是 ssh 自己失败的保留码, 正落在
    「命令没跑 / 跑了但回执丢了 / 还在飞」这三种对客户端同形的世界里 (#102 第 1 片,
    依据 Limbo `2609.29095v1` 行 226-232)。判据放在 `measurement.transport_outcome` 里,
    因为读它的报表与写它的执行侧必须共用一套词表 (I22 那一族)。
    """
    return transport_outcome(exc, remote_rc=getattr(exc, "ssh_rc", None))


def _transport_failure(exc) -> dict:
    """三个「返回 dict」的远程出口共用的失败载荷形状"""
    return {"success": False, "error": str(exc), TRANSPORT_KEY: _transport_outcome_of(exc)}


@dataclass
class RemoteExecutionResult:
    """远程执行结果"""
    success: bool = True
    output_shape: Optional[list] = None
    output_sample: Optional[list] = None
    execution_time_ms: float = 0.0
    gflops: float = 0.0
    memory_bandwidth_gbs: float = 0.0
    error: str = ""
    device: str = ""
    #: 「听到回答」/「没听到回答」那一格 (#102 第 1 片, 依据 Limbo `2609.29095v1` 行 226-232)。
    #: 默认 `answered` 不是猜的 —— 能构造出这个对象就说明手里有一份远端返回; 只有 except 分支
    #: 会把它改成 `outcome_unknown`。默认选**不报警**那一侧, 因为下游拿它做的是幂等清点。
    transport_outcome: str = TRANSPORT_ANSWERED


@dataclass
class Resolution:
    """门面「到底解析成了什么」的记录 —— 降级必须是**数据**, 不是一行 print

    `attempts` 保留每一级探测的 `(label, reason)`: 旧的 `check_connection() -> bool`
    把异常吞成 False, 于是「连不上」在日志里只剩一行 print, 而**为什么**连不上
    (DNS / 认证 / 超时) 永远查不到 (DSH postmortem 0004: 保留下层拥有的结构化失败)。
    """

    requested: str                       # 请求的 mode
    resolved: str = ""                   # 解析到的后端 label; 抛 BackendUnavailable 时为空
    degraded: bool = False               # 请求的与实际拿到的不一致
    attempts: list = field(default_factory=list)


class BackendUnavailable(RuntimeError):
    """请求的执行侧连不上。**不降级**

    `mode="ssh"` 以前只是**偏好**: 连不上就试 HTTP, 再连不上就无条件落到 LocalExecutor,
    唯一的信号是一行 `print("回退到本地 CPU 模式")` —— 不抛异常、不存标志、没有 strict
    模式 (doc 49 §3.3 S3)。加上 `LocalExecutor.check_connection()` 恒 True (S4),
    这个梯子**永远以「成功」结束**。于是驱动只能靠 `"LocalExecutor" in remote.name`
    这种显示名子串去猜到底连上没有 (A3), 猜错时**凭空造一台 SM 数是真的假 H20**。

    现在: `"ssh"`/`"http"` 是承诺, 兑现不了就在**第一次访问执行侧时**抛这个异常;
    `"auto"` 仍是偏好, 可以一路降到 CPU, 但降级写进 `Resolution.degraded` 并由
    `capabilities` 变成空集暴露出来。DSH `2026-08-19` 笔记的三条要求正好对上:
    "configuration errors must fail explicitly"、"explicit first-access failure …
    without allowing silent degradation"、"it never degrades to a default value"。
    """

    def __init__(self, resolution: Resolution):
        self.resolution = resolution
        detail = "; ".join(f"{label}: {reason or '未知原因'}"
                           for label, reason in resolution.attempts) or "没有可用的探测目标"
        super().__init__(
            f"mode={resolution.requested!r} 是一次承诺, 但执行侧不可用 —— {detail}。"
            f"不降级到 CPU: 那会把「没有加速卡」翻译成「有一个更慢的」, 而产物里看不出区别。"
            f"要显式跑本机 CPU 请写 mode=\"local\", 要允许降级请写 mode=\"auto\"。")


class HardwareProbeFailed(RuntimeError):
    """硬件信息**探测不到**, 而且这里没有「退一步给个保守值」这种退路 (doc 49 §3.3 S1/S2)。

    旧写法在失败路径上返回一整套 A100 量级的虚构字典 (128 SM / 40GB / 2TB/s /
    128KB/SM / 19TB/s), 而它唯一的读者是 Designer 的 tiling 约束
    (`agents.py` 读 `memory.local.size`) —— 于是「没探测过硬件」在产物里长得和
    「这是一台 A100」一模一样, 而 H20 的共享内存是 228KB/SM: 编造的**更小**,
    所以 tiling 会朝保守方向收缩, 不会报错, 只会一直不快。

    与 `BackendUnavailable` 的分工: 那个回答「执行侧连不上」(mode 承诺兑现不了),
    这个回答「连上了, 但问不到它是谁」。合用一个类会让日志里「SSH 不通」和
    「SSH 通了但 runner 输出不是 JSON」看不出区别 —— 这正是 postmortem 0004 的形状。
    """

    def __init__(self, backend_label: str, reason: str, endpoint_hint: str = ""):
        self.backend_label = backend_label
        self.reason = reason
        msg = (f"后端 {backend_label} 探测不到硬件信息 ({reason}) —— "
               f"不返回虚构规格: 编造一台 A100 会把「不知道这台卡是什么」变成"
               f"一个能被 Designer 当约束用的假分母。")
        if endpoint_hint:
            msg += endpoint_hint
        else:
            msg += ("先跑一次 `probe_connection()` 看传输层是否可达, "
                    "再看远端 runner 的 `gpu_info` 为什么没返回字段。")
        super().__init__(msg)


class SSHExecutor(Backend):
    """
    通过 SSH 在远程 GPU 服务器上执行 kernel 代码。
    
    工作原理:
    1. 首次连接时自动将 remote_scripts/run_kernel.py 部署到远程
    2. 后续调用通过 SSH stdin 传入 JSON payload，远程脚本执行后 stdout 返回结果
    3. 支持 torch GPU 加速 (远程有 A100)，无 torch 时回退 numpy
    
    依赖: 本机需要 ssh 命令行工具 (macOS/Linux 自带)
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        key_path: Optional[str] = None,
        remote_runner_path: Optional[str] = None,
        work_dir: Optional[str] = None,
        timeout: int = 300,
        platform: str = "nvidia",  # 接入其他 GPU 平台时在此增加分支
    ):
        self.platform = platform
        if platform == "kunlunxin":
            # 昆仑芯 P800 (服务器已下线)。分支保留作为"非 NVIDIA 平台"的接入范例:
            # 不走 uv, 用厂商预装 conda python + 专用 runner, 且需要额外的 LD_LIBRARY_PATH
            #
            # 连接信息与下面 nvidia 分支同一套脱敏约定: host 只给 ~/.ssh/config 里的
            # 别名, port / user / key_path 默认 None 交由该 Host 块解析, 仓库中因此
            # 不出现真实 IP、端口、用户名或密钥路径。
            self.host = host or os.environ.get("REMOTE_SSH_HOST_KL", "kunlunxin-p800")
            _env_port_kl = os.environ.get("REMOTE_SSH_PORT_KL", "")
            self.port = port or (int(_env_port_kl) if _env_port_kl else None)
            self.user = user or os.environ.get("REMOTE_SSH_USER_KL") or None
            self.key_path = key_path or os.environ.get("REMOTE_SSH_KEY_KL") or None
            self.work_dir = work_dir or os.environ.get(
                "REMOTE_WORK_DIR_KL", "~/operator_agent_runner"
            )
            self.remote_runner_path = remote_runner_path or os.environ.get(
                "REMOTE_RUNNER_PATH_KL", f"{self.work_dir}/run_kernel_xpu.py"
            )
            self._python_cmd = os.environ.get(
                "REMOTE_PYTHON_KL", "/data1/miniconda3/envs/vllm_xpu/bin/python"
            )
            self._runner_script = "run_kernel_xpu.py"
        else:
            # 默认走 ~/.ssh/config 里的 Host 别名: 端口 / 用户名 / 密钥由该 Host 块提供,
            # 仓库中因此不出现真实 IP、端口、用户名或密钥路径。
            # 换机器只需改 ssh config, 或用 REMOTE_SSH_* 环境变量覆盖。
            self.host = host or os.environ.get("REMOTE_SSH_HOST", "algorithm-h20")
            _env_port = os.environ.get("REMOTE_SSH_PORT", "")
            self.port = port or (int(_env_port) if _env_port else None)
            self.user = user or os.environ.get("REMOTE_SSH_USER") or None
            self.key_path = key_path or os.environ.get("REMOTE_SSH_KEY") or None
            self.work_dir = work_dir or os.environ.get(
                "REMOTE_WORK_DIR", "~/operator_agent"
            )
            self.remote_runner_path = remote_runner_path or os.environ.get(
                "REMOTE_RUNNER_PATH", f"{self.work_dir}/run_kernel.py"
            )
            self._python_cmd = None  # nvidia 走 uv run python
            self._runner_script = "run_kernel.py"
        self.timeout = timeout
        self.name = f"SSHExecutor@{self._ssh_target()}({platform})"
        self._deployed = False
        self._uv_setup = False
        self._gpu_info_cache: Optional[dict] = None
        # 固定的 GPU 卡号: None=未解析, -1=解析过但不固定, >=0=固定该卡
        self._gpu_pin: Optional[int] = None
        self._uv_path = "~/.local/bin/uv"   # ensure_environment 复用

    @property
    def label(self) -> str:
        return self.name

    @property
    def capabilities(self) -> frozenset:
        """真卡 + 全套远端协议能力

        ⚠️ 词汇表里刻意**没有**「硬件信息是真探测的」这一项, 而且 S1 修好之后仍然不加。
        doc 49 §3.3 原本写的是「等 S1 改成『要么真探测、要么抛』再加」—— 那个条件
        现在满足了, 但满足了不等于该加: 加一项能力必须有**读者**按它分支, 而这里没有
        任何调用方会写 `if Cap.HARDWARE_PROBED in capabilities` —— `conductor` 无论
        如何都必须拿到硬件信息, 拿不到就该抛。一个没有读者的 Cap 就是「算了没人读」
        的又一个副本, 而 `raise` 已经让坏状态**不可表达**, 不需要再多一层声明。
        """
        caps = {Cap.REAL_DEVICE, Cap.REMOTE_VERIFY, Cap.SAME_DEVICE_BASELINE,
                Cap.ENV_PROVISION, Cap.DEPLOY, Cap.DEP_ANALYSIS}
        if self.platform == "nvidia":
            # `_pinned_gpu()` 对非 nvidia 平台第一行就 return None (它依赖 nvidia-smi),
            # 于是「跨轮固定同一张卡」在那个平台上根本不发生 —— 不声明就是不说谎
            caps.add(Cap.GPU_PINNING)
        return frozenset(caps)

    def _ssh_target(self) -> str:
        """连接目标。user 为空时只给 host, 用户名交由 ~/.ssh/config 的 Host 块解析。"""
        return f"{self.user}@{self.host}" if self.user else self.host

    def _build_ssh_base(self) -> list:
        # port / user / key_path 未显式给出时不写进命令行, 由 ~/.ssh/config 解析。
        # 这样仓库里只需保留一个 ssh 别名, 不必写死真实 IP、端口、用户名和密钥路径。
        cmd = ["ssh", "-o", "StrictHostKeyChecking=no",
               "-o", "ConnectTimeout=10", "-o", "BatchMode=yes"]
        if self.port:
            cmd.extend(["-p", str(self.port)])
        if self.key_path:
            cmd.extend(["-i", os.path.expanduser(self.key_path)])
        cmd.append(self._ssh_target())
        return cmd

    def _ssh_exec(self, remote_cmd: str, stdin_data: Optional[str] = None) -> str:
        """执行远程命令并返回 stdout

        说明:
        - 远端 uv / triton 会往 stderr 打 deprecation 或 JIT warning，这类输出
          并不影响结果，并且即便 rc != 0 只要 stdout 里有可用 JSON 通常也可接受。
          因此 rc != 0 时我们先把 stderr/stdout 都拿到 (不直接抛错)，
          由上层 _ssh_exec_json 根据 stdout 自行解析，只有 JSON 解析失败才报错。
        """
        full_cmd = self._build_ssh_base() + [remote_cmd]
        proc = subprocess.run(
            full_cmd,
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
        # ── rc != 0 时: 仅当 stdout 完全空才抛错 ──
        stderr_out = proc.stderr.strip()
        stdout_out = proc.stdout.strip()
        if not stdout_out:
            combined = []
            if stderr_out:
                tail = stderr_out[-3000:] if len(stderr_out) > 3000 else stderr_out
                combined.append(f"[stderr ({len(stderr_out)} chars, tail shown)]\n{tail}")
            detail = ("\n\n".join(combined) if combined else "(no output captured)")
            err = RuntimeError(
                f"SSH exec failed [rc={proc.returncode}]: {detail}"
            )
            # rc 挂在异常上而不是换异常类型: 文件里有三处 `except RuntimeError` 靠这个类型
            # 走探测分支, 换类型会静默改掉它们的行为。而 rc=255 是 ssh **自己**失败的保留码
            # (远端命令的返回码不会被它覆盖) —— 那一格恰恰是「命令可能没跑、也可能跑完了
            # 但回执丢了」, 所以 `transport_outcome` 要能看见它 (#102 第 1 片)。
            err.ssh_rc = proc.returncode
            raise err
        # stdout 非空时，把 stderr 以 "warning:" 前缀附加到 stdout 末尾，
        # 方便 _ssh_exec_json 或调用方在 JSON 解析失败后诊断真正原因。
        if stderr_out:
            tail = stderr_out[-1500:] if len(stderr_out) > 1500 else stderr_out
            # 用 JSON 安全的前缀把 stderr 诊断信息塞进输出
            # _ssh_exec_json 会按行找最底部 "{"，这些 #// 前缀会被自动跳过
            diagnostics = (
                "\n# [_ssh_exec rc={} diagnostics, stderr tail]:\n".format(proc.returncode)
                + "\n".join(f"# {ln}" for ln in tail.splitlines()[-30:])
            )
            stdout_out = stdout_out + diagnostics
        return stdout_out

    def _pinned_gpu(self) -> Optional[int]:
        """选定一张卡并在本 executor 生命周期内固定使用。

        run_kernel.py 每次调用都是独立进程、会各自重新选卡; 共享机器上
        L1/L2/L3 落在不同卡会让加速比不可比, 故由控制端选定后下发。
        返回 None 表示不固定, 交由远端自动选择。
        """
        if self._gpu_pin is not None:
            return self._gpu_pin if self._gpu_pin >= 0 else None
        self._gpu_pin = -1
        # 依赖 nvidia-smi, 非 NVIDIA 平台不适用, 交由 runner 自行选卡
        if self.platform != "nvidia":
            return None
        try:
            out = self._ssh_exec(
                "nvidia-smi --query-gpu=index,memory.free,utilization.gpu "
                "--format=csv,noheader,nounits"
            )
            best = None
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    continue
                try:
                    idx, free, util = int(parts[0]), int(parts[1]), int(parts[2])
                except ValueError:
                    continue
                # 利用率优先: 显存余量大但被跑满的卡测不出可信加速比
                key = (-util, free)
                if best is None or key > best[0]:
                    best = (key, idx, util, free)
            if best:
                self._gpu_pin = best[1]
                print(f"  [SSHExecutor] 固定使用 GPU {best[1]} "
                      f"(util={best[2]}%, free={best[3] / 1024:.1f}GB)")
        except Exception as e:
            print(f"  [SSHExecutor] 选卡失败, 交由远端自动选择: {e}")
        return self._gpu_pin if self._gpu_pin >= 0 else None

    def _ssh_exec_json(self, payload: dict) -> dict:
        """通过 SSH 在远程执行 runner 脚本并解析 JSON 结果"""
        json_str = json.dumps(payload, ensure_ascii=False)
        if self.platform != "nvidia":
            # 非 NVIDIA 平台: 用厂商预装 conda python, 需要额外的 LD_LIBRARY_PATH
            remote_cmd = (
                f"mkdir -p {self.work_dir} && "
                f"export LD_LIBRARY_PATH=/data1/miniconda3/envs/vllm_xpu/xcudart/lib:"
                f"/data1/miniconda3/envs/vllm_xpu/lib:$LD_LIBRARY_PATH && "
                f"{self._python_cmd} {self.remote_runner_path}"
            )
        else:
            gpu = self._pinned_gpu()
            env = f"OPERATOR_AGENT_GPU={gpu} " if gpu is not None else ""
            remote_cmd = (
                f"cd {self.work_dir} && {env}"
                f"~/.local/bin/uv run python {self.remote_runner_path}"
            )
        output = self._ssh_exec(remote_cmd, stdin_data=json_str)
        for line in reversed(output.split("\n")):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        raise ValueError(f"No JSON found in remote output: {output[:500]}")

    def _setup_uv_env(self) -> bool:
        """在远程工作目录下自动配置 uv 环境 (nvidia)，或使用厂商预装 conda 环境 (其他平台)"""
        if self._uv_setup:
            return True

        if self.platform != "nvidia":
            # 非 NVIDIA 平台使用预装 conda 环境，只需创建工作目录和修复 JIT 缓存权限
            print(f"  [SSHExecutor] {self.platform} 平台，使用预配置 conda 环境: {self.work_dir}")
            try:
                self._ssh_exec(f"mkdir -p {self.work_dir}")
                fix_cmd = (
                    "mkdir -p /tmp/xpujitc-temp /tmp/kXPU3 /tmp/.xblas_cache; "
                    "sudo -n chmod -R 777 /tmp/xpujitc-temp /tmp/kXPU3 /tmp/.xblas_cache 2>/dev/null; "
                    "touch /tmp/.xblas_jit.lock && chmod 777 /tmp/.xblas_jit.lock 2>/dev/null; "
                    "true"
                )
                self._ssh_exec(fix_cmd)
                self._uv_setup = True
                return True
            except Exception as e:
                # 旧写法在这里 print 一行「警告」然后 `_uv_setup = True; return True`
                # (doc 49 §3.3 S7): 工作目录建不起来 = 后面每一次 scp 都会失败, 而
                # `deploy()` 读的是本函数的返回值, 于是它先被告知「环境就绪」。
                # 更糟的是 `_uv_setup = True` 会把这句谎话**缓存住** —— 第二次调用在
                # 函数第一行就 return True, 连再试一次都没有。
                # `fix_cmd` 以 `; true` 收尾, 它自己不会失败; 走到这里就是传输层坏了。
                print(f"  [SSHExecutor] {self.platform} 环境配置失败, 不报成功: {e}")
                return False

        print(f"  [SSHExecutor] 开始配置远程 uv 环境: {self.work_dir}")

        try:
            self._ssh_exec(f"mkdir -p {self.work_dir}")
            print(f"  [SSHExecutor] 创建工作目录: {self.work_dir}")
        except Exception as e:
            print(f"  [SSHExecutor] 创建目录失败: {e}")
            return False

        try:
            self._ssh_exec("test -f ~/.local/bin/uv")
            print(f"  [SSHExecutor] uv 已安装")
        except RuntimeError:
            print(f"  [SSHExecutor] 安装 uv...")
            try:
                self._ssh_exec("curl -LsSf https://astral.sh/uv/install.sh | sh")
                print(f"  [SSHExecutor] uv 安装成功")
            except Exception as e:
                print(f"  [SSHExecutor] uv 安装失败: {e}")
                return False

        try:
            self._ssh_exec(f"test -f {self.work_dir}/pyproject.toml")
            print(f"  [SSHExecutor] uv 项目已初始化")
        except RuntimeError:
            print(f"  [SSHExecutor] 初始化 uv 项目...")
            try:
                python_path = self._ssh_exec("which python3").strip()
                self._ssh_exec(
                    f"cd {self.work_dir} && ~/.local/bin/uv init --python {python_path}"
                )
                print(f"  [SSHExecutor] uv 项目初始化成功")
            except Exception as e:
                print(f"  [SSHExecutor] uv 项目初始化失败: {e}")
                return False

        # .python-version 必须在 uv add 之前与 uv init 所用的解释器对齐。
        # 放后面会导致残留的错误 pin 先让 uv add 失败并 return，本段永远执行不到。
        # 写死 3.12 会让 uv 去下载不存在的 CPython 3.12 (镜像 404) 而全盘失败。
        try:
            pyver = self._ssh_exec(
                "python3 -c 'import sys;print(\"%d.%d\"%sys.version_info[:2])'"
            ).strip().splitlines()[-1]
            self._ssh_exec(
                f"echo '{pyver}' > {self.work_dir}/.python-version"
            )
            print(f"  [SSHExecutor] 固定 .python-version = {pyver} (与远端解释器一致)")
        except Exception as e:
            # 同 S7 的第 5 处: 这一段的**全部意义**是把 `.python-version` 对齐到远端
            # 解释器 (上一段注释记着残留错误 pin 会让 `uv add` 全盘失败)。写不进去却
            # 继续往下走、最后 `return True`, 等于「防某个已知失败的步骤没做成, 而
            # 调用方被告知环境已就绪」。这里没有「保守值」可用 —— 不知道 uv 实际用了
            # 哪个解释器, 就是不知道。
            print(f"  [SSHExecutor] 固定 .python-version 失败, 不继续装依赖: {e}")
            return False

        try:
            self._ssh_exec(
                f"cd {self.work_dir} && ~/.local/bin/uv add numpy torch triton"
            )
            print(f"  [SSHExecutor] 依赖安装成功")
        except Exception as e:
            print(f"  [SSHExecutor] 依赖安装失败: {e}")
            return False

        self._uv_setup = True
        print(f"  [SSHExecutor] uv 环境配置完成: {self.work_dir}")
        return True

    def _remote_files(self) -> list:
        """需要部署到远端的 (本地路径, 远端路径) 列表, 按推送顺序排列

        runner 的同级依赖必须排在 runner 前面: 这样一次部署即使中途失败, 也只会
        出现「新依赖 + 旧 runner」(能跑), 不会出现「新 runner + 缺依赖」(直接
        ImportError)。
        """
        repo_root = Path(__file__).parent.parent
        remote_dir = str(PurePosixPath(self.remote_runner_path).parent)
        return [
            # L2 数值门的容差表与判定式, runner 以同级模块 import
            (repo_root / "operator_agent" / "tolerance.py",
             f"{remote_dir}/tolerance.py"),
            # 运行时反作弊屏蔽, 与控制端 static_audit 共用同一份禁用清单
            (repo_root / "operator_agent" / "anticheat.py",
             f"{remote_dir}/anticheat.py"),
            (repo_root / "remote_scripts" / self._runner_script,
             self.remote_runner_path),
        ]

    def _remote_hashes(self, paths: list) -> dict:
        """一次 SSH 往返取回多个远端文件的 sha256 前 12 位; 不存在的记为空串"""
        py = self._python_cmd if self._python_cmd else "python3"
        script = (
            "import hashlib, os\n"
            f"for p in {paths!r}:\n"
            # 远端路径默认是 ~/operator_agent/... 形式。shell 与 scp 会展开 ~,
            # 但 python 的 open() 不会 —— 少了这一行 hash 查询会恒定失败, 于是
            # 每个任务都白推一次脚本 (改动前的实现就是这个状态)。
            "    q = os.path.expanduser(p)\n"
            "    try:\n"
            "        with open(q, 'rb') as f:\n"
            "            print(p, hashlib.sha256(f.read()).hexdigest()[:12])\n"
            "    except OSError:\n"
            "        print(p, '')\n"
        )
        try:
            out = self._ssh_exec(f"{py} -c {shlex.quote(script)}")
        except RuntimeError:
            return {p: "" for p in paths}
        hashes = {}
        for line in out.splitlines():
            parts = line.rsplit(" ", 1)
            if len(parts) == 2:
                hashes[parts[0]] = parts[1]
        return hashes

    def _scp(self, local: Path, remote_path: str) -> None:
        scp_cmd = ["scp", "-o", "StrictHostKeyChecking=no"]
        if self.port:
            scp_cmd.extend(["-P", str(self.port)])
        if self.key_path:
            scp_cmd.extend(["-i", os.path.expanduser(self.key_path)])
        scp_cmd.extend([str(local), f"{self._ssh_target()}:{remote_path}"])
        subprocess.run(scp_cmd, capture_output=True, text=True, timeout=60, check=True)

    def deploy(self) -> bool:
        """部署 runner 及其同级依赖到远程服务器 (逐文件按内容 hash 判断是否需要更新)"""
        if not self._setup_uv_env():
            print(f"  [SSHExecutor] 环境配置失败，跳过部署")
            return False

        files = self._remote_files()
        missing = [str(local) for local, _ in files if not local.exists()]
        if missing:
            print(f"  [SSHExecutor] 本地脚本不存在: {', '.join(missing)}")
            return False

        local_hashes = {
            remote: hashlib.sha256(local.read_bytes()).hexdigest()[:12]
            for local, remote in files
        }
        remote_hashes = self._remote_hashes([remote for _, remote in files])

        stale = [remote for remote in local_hashes
                 if remote_hashes.get(remote, "") != local_hashes[remote]]
        if not stale:
            self._deployed = True
            print(f"  [SSHExecutor] 远程脚本已是最新 "
                  f"(hash={local_hashes[self.remote_runner_path]}): {self.remote_runner_path}")
            return True

        try:
            # 按 _remote_files() 的顺序推: 依赖先于 runner
            for local, remote in files:
                if remote in stale:
                    action = "更新" if remote_hashes.get(remote) else "部署"
                    self._scp(local, remote)
                    print(f"  [SSHExecutor] 脚本已{action}到 {self.host}:{remote} "
                          f"(hash={local_hashes[remote]})")
            self._deployed = True
            return True
        except Exception as e:
            print(f"  [SSHExecutor] 部署失败: {e}")
            return False

    def probe_connection(self) -> tuple:
        """探一次连通性, 返回 `(ok, reason)`

        `reason` 必须带出来: 旧的 `check_connection() -> bool` 把异常吞成一个 False,
        于是「连不上」在日志里只剩一行 print, 而**为什么**连不上 (DNS? 认证? 超时?)
        永远查不到。降级梯子的每一级都要把这个 reason 记进 `Resolution.attempts`
        (doc 49 §3.3 S3/S4; DSH postmortem 0004: 保留下层拥有的结构化失败)。
        """
        try:
            self._ssh_exec("echo ok")
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    @property
    def selected_gpu(self) -> Optional[int]:
        """`_pinned_gpu()` 已经解析出来的那张卡; 没解析出来就回 None

        刻意**不**在这里触发一次选卡: 这是一个给报表读的只读视图, 触发往返会让「问它在用
        哪张卡」变成「再花一次 ssh」。`_gpu_pin` 的三态 (None 未解析 / -1 解析过但不固定 /
        >=0 固定该号) 在这里折叠成两态, 因为 -1 与「还没解析」对读者的意思是一样的:
        **这一轮没有固定卡号** (doc 49 §7.12)。
        """
        pin = self._gpu_pin
        return pin if pin is not None and pin >= 0 else None

    def lock_identity(self) -> str:
        """按**目标机器**取键, 不带 platform: 锁保护的是那张卡, 不是那份配置"""
        return f"ssh:{self._ssh_target()}"

    def ensure_environment(self, required_pips: Optional[list] = None) -> tuple:
        """
        前置环境自检 + 自动配置 + 返回依赖清单

        流程:
        1. deploy() 安装 uv + 基础依赖 + 同步 run_kernel.py (已含 hash 同步)
        2. EnvironmentChecker 全链路自检 (ssh→uv→python→torch→triton→cuda→gpu)
        3. 缺失依赖按需 uv add 自动安装
        4. 仍失败则返回 (False, 依赖清单文本) 供用户人工配置

        Returns:
            (all_passed, manifest_text)
        """
        # 步骤 1: uv 环境 + 脚本部署 (deploy 内部会按 hash 同步)
        if not self.deploy():
            report = EnvironmentReport()
            report.add(CheckItem("deploy", False, "uv 环境配置或脚本部署失败",
                                 fix_hint=f"检查远程工作目录 {self.work_dir} 权限与网络"))
            return False, report.dependency_manifest()

        # 步骤 2: 全链路自检
        checker = EnvironmentChecker(self._ssh_exec, self._uv_path, self.work_dir)
        report = checker.check(required_pips=required_pips or [])

        # 步骤 3: 缺失的 pip 依赖尝试自动安装
        missing = [it.name.split("dep:")[1] for it in report.items
                   if not it.passed and it.name.startswith("dep:")]
        if missing:
            print(f"  [SSHExecutor] 尝试自动安装缺失依赖: {missing}")
            try:
                self._ssh_exec(
                    f"cd {self.work_dir} && {self._uv_path} add {' '.join(missing)}"
                )
                print(f"  [SSHExecutor] 依赖安装成功, 重新自检")
                report = checker.check(required_pips=required_pips or [])
            except Exception as e:
                print(f"  [SSHExecutor] 自动安装失败: {e}")

        if report.all_passed:
            print(f"  [SSHExecutor] 环境自检通过:\n{report.summary()}")
        else:
            print(f"  [SSHExecutor] 环境自检未通过:\n{report.summary()}")

        return report.all_passed, report.dependency_manifest()

    def analyze_dependencies(self, kernel_code: str) -> list:
        """从生成的 kernel 代码提取所需 pip 依赖名"""
        modules = DependencyAnalyzer.extract_imports(kernel_code)
        return DependencyAnalyzer.to_pip_names(modules)

    def get_gpu_info(self) -> dict:
        """获取远程 GPU 信息"""
        if self._gpu_info_cache:
            return self._gpu_info_cache
        try:
            if not self._deployed:
                self.deploy()
            result = self._ssh_exec_json({"action": "gpu_info"})
            if self.platform != "nvidia":
                # 非 NVIDIA 平台: runner 字段名与语义都不同 (无 compute_capability / triton)。
                # 带宽与共享内存取不到就标 unknown, 不要写死某个具体型号的数值 ——
                # 这个分支是接入下一个平台的模板, 写死会让新平台拿到错的硬件约束。
                self._gpu_info_cache = {
                    "name": result.get("device_name", "Unknown"),
                    "vendor": result.get("vendor", self.platform),
                    "host": self.host,
                    "compute_units": result.get("multiprocessor_count", 0),
                    "warp_size": result.get("warp_size", 32),
                    "vector_width": 128,
                    "memory": {
                        "global": {
                            "size": f"{result.get('total_memory_gb', 0)}GB",
                            "bandwidth": "unknown",
                        },
                        "local": {"size": "unknown", "bandwidth": "unknown"},
                    },
                    "device": result.get("device", f"{self.platform}_accelerator"),
                    "compute_capability": result.get("compute_capability", "unknown"),
                    "torch_version": result.get("torch_version", "unknown"),
                    "triton_version": "not_supported",
                    "device_count": result.get("device_count", 1),
                    "selected_gpu": result.get("selected_gpu", 0),
                    "gpu_memory": result.get("gpu_memory", []),
                }
            else:
                # runner 不上报带宽/共享内存, 从 hardware_caps 取 (单一真源)。
                # 此前写死 "2TB/s" + "128KB/SM" 是 A100 量级, 会误导 Designer 的 tiling:
                # H20 实为 4.0TB/s + 228KB/SM。取不到就标 unknown, 不再编造数字。
                caps = detect_hardware_caps(result)
                bw_gbs = caps.global_mem_bandwidth_gbs
                smem_kb = caps.shared_mem_per_sm_kb
                self._gpu_info_cache = {
                    "name": result.get("name", "Unknown GPU"),
                    "host": self.host,
                    "compute_units": result.get("multiprocessor_count") or caps.sm_count,
                    "warp_size": caps.sub_group_size or 32,
                    "vector_width": (caps.vector_width_bytes or 16) * 8,
                    "memory": {
                        "global": {
                            "size": f"{result.get('total_memory_gb', 0)}GB",
                            "bandwidth": f"{bw_gbs / 1000:.1f}TB/s" if bw_gbs else "unknown",
                        },
                        "local": {
                            "size": f"{smem_kb}KB/SM" if smem_kb else "unknown",
                            # L1/共享内存带宽在**两个真源里都没有字段**: runner 的 gpu_info 不报,
                            # HardwareCaps 只有 `shared_mem_bank_count`。以前这里写死 "19TB/s"
                            # (doc 49 §3.3 第 4 处, 就在「取不到就标 unknown, 不再编造数字」
                            # 那行注释的下面两行) —— 编造值唯一的读者本来没有, 它的作用只是让
                            # 日志看起来像探测过。
                            "bandwidth": "unknown",
                        },
                    },
                    "device": result.get("device", "unknown"),
                    "compute_capability": result.get("compute_capability", "unknown"),
                    "torch_version": result.get("torch_version", "unknown"),
                    "triton_version": result.get("triton_version", "unknown"),
                    "device_count": result.get("device_count", 1),
                    "selected_gpu": result.get("selected_gpu", 0),
                    "gpu_memory": result.get("gpu_memory", []),
                }
            # 打印 GPU 选择日志
            gpu_log = result.get("gpu_selection_log", "")
            if gpu_log:
                for line in gpu_log.split("\n"):
                    print(f"  [SSHExecutor] {line}")
            return self._gpu_info_cache
        except Exception as e:
            print(f"  [SSHExecutor] 获取 GPU 信息失败, 抛出而不是编造规格: {e}")
            raise HardwareProbeFailed(self.label, f"{type(e).__name__}: {e}") from e

    def get_hardware_info(self) -> dict:
        """返回硬件信息供 Designer Agent 使用（兼容 MockGPU 接口）"""
        return self.get_gpu_info()

    def compile(self, kernel_code: str) -> bool:
        try:
            if not self._deployed:
                self.deploy()
            result = self._ssh_exec_json({"action": "compile", "code": kernel_code})
            if not result.get("success", False):
                raise SyntaxError(result.get("error", "Remote compile failed"))
            return True
        except Exception as e:
            raise SyntaxError(f"Remote compile error via SSH: {e}")

    def execute(self, kernel_code: str, inputs: dict) -> RemoteExecutionResult:
        try:
            if not self._deployed:
                self.deploy()
            payload = {
                "action": "execute",
                "code": kernel_code,
                "inputs": self._serialize_inputs(inputs),
            }
            result = self._ssh_exec_json(payload)
            return RemoteExecutionResult(
                success=result.get("success", False),
                output_shape=result.get("output_shape"),
                output_sample=result.get("output_sample"),
                execution_time_ms=result.get("execution_time_ms", 0),
                gflops=result.get("gflops", 0),
                memory_bandwidth_gbs=result.get("memory_bandwidth_gbs", 0),
                error=result.get("error", ""),
                device=result.get("device", ""),
            )
        except Exception as e:
            return RemoteExecutionResult(success=False, error=str(e),
                                   transport_outcome=_transport_outcome_of(e))

    def benchmark(self, kernel_code: str, inputs: dict, warmup: int = 3, repeat: int = 10) -> dict:
        try:
            if not self._deployed:
                self.deploy()
            payload = {
                "action": "benchmark",
                "code": kernel_code,
                "inputs": self._serialize_inputs(inputs),
                "warmup": warmup,
                "repeat": repeat,
            }
            return self._ssh_exec_json(payload)
        except Exception as e:
            return _transport_failure(e)   # 带「有没有听到回答」那一格 (见 _transport_failure)

    def verify_function(self, kernel_code: str, inputs: dict, kernel_type: str,
                        max_error: float, precision: str = "",
                        reference_source: str = "") -> dict:
        """Compare candidate/reference on the remote GPU and return a compact summary.

        `reference_source` 非空时, 远端用它做参考 (新架构算子), 不再按 kernel_type
        查内建表。空串 = 保持旧行为。
        """
        try:
            if not self._deployed:
                self.deploy()
            return self._ssh_exec_json({
                "action": "verify", "code": kernel_code,
                "inputs": self._serialize_inputs(inputs), "kernel_type": kernel_type,
                "max_error": max_error, "precision": precision,
                "reference_source": reference_source or "",
            })
        except Exception as e:
            return _transport_failure(e)   # 带「有没有听到回答」那一格 (见 _transport_failure)

    def benchmark_with_baselines(self, kernel_code: str, inputs: dict,
                                 kernel_type: str, warmup: int = 10,
                                 repeat: int = 30, precision: str = "",
                                 baseline_priority=None,
                                 reference_source: str = "") -> dict:
        """Time candidate vs one interleaved baseline on the same remote device.

        `baseline_priority` 是控制端的有序偏好, 远端只测第一个能跑的基线 —— 谁公平
        是个策略问题 (fp16 候选该比 cuBLAS fp16), 但「能不能跑」只有远端知道。
        `reference_source` 同 `verify_function`。
        """
        try:
            if not self._deployed:
                self.deploy()
            return self._ssh_exec_json({
                "action": "benchmark_with_baselines", "code": kernel_code,
                "inputs": self._serialize_inputs(inputs), "kernel_type": kernel_type,
                "warmup": warmup, "repeat": repeat, "precision": precision,
                "baseline_priority": list(baseline_priority or []),
                "reference_source": reference_source or "",
            })
        except Exception as e:
            return _transport_failure(e)   # 带「有没有听到回答」那一格 (见 _transport_failure)

    @staticmethod
    def _serialize_inputs(inputs: dict) -> dict:
        """将输入序列化为轻量级描述 (shape+dtype+seed), 远程端按描述本地生成数据。
        避免大矩阵 JSON 序列化/传输瓶颈 (如 4096x4096 单矩阵 ~67MB JSON)。

        seed 用 zlib.crc32 而非内建 hash(): 后者受 PYTHONHASHSEED 随机化, 每次进程
        都不同, 会让跨运行的输入不可复现、L3 加速比不可比。键名必须参与派生, 否则
        同 shape+dtype 的两个输入 (方阵 matmul 的 A/B) 会生成出同一份数据。
        """
        import zlib
        import numpy as np
        specs = {}
        for k, v in inputs.items():
            arr = np.asarray(v)
            ident = f"{k}|{arr.shape}|{arr.dtype}".encode()
            specs[k] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "seed": zlib.crc32(ident) % (2 ** 31 - 1),
            }
        return {"_generate": True, "specs": specs}


class HTTPExecutor(Backend):
    """
    通过 HTTP API 调用远程服务 (如果远程有接口服务)
    """

    def __init__(self, base_url: Optional[str] = None, timeout: int = 120):
        # 默认留空: 新环境没有本项目的 HTTP 服务, 8091 属于其他租户。
        # 空串会让 probe_connection 直接失败, auto 模式据此降级, 不会误连别人的服务。
        self.base_url = (base_url or os.environ.get("REMOTE_HTTP_URL", "")).rstrip("/")
        self.timeout = timeout
        self.name = f"HTTPExecutor@{self.base_url}"

    @property
    def label(self) -> str:
        return self.name

    @property
    def capabilities(self) -> frozenset:
        """**空集**, 而且是故意的

        它确实能在远端执行代码, 但: ① 问不到对面是谁 —— `get_hardware_info()` 抛
        `HardwareProbeFailed` (S2 已改成「要么真探测、要么抛」里的「抛」, 因为服务端没有
        硬件端点), 所以连「对面是不是加速卡」都没有凭据;
        ② 没有 `verify_function` / `benchmark_with_baselines` / `ensure_environment` /
        `deploy`, 基类那几个存根会把调用**抛出来**而不是返回 None。
        空集意味着任何 `require(Cap.REAL_DEVICE, …)` 的消费方都会在构造时就拒绝它 ——
        这正是想要的方向: 宁可响, 不要静默降级。

        ⚠️ 修掉 S2 **不等于**可以声明 `REAL_DEVICE`: 能力要跟着凭据走, 而现在唯一的
        凭据是「编译/执行都成功」, 它既可能来自 GPU 也可能来自 CPU。要加这一项,
        先加服务端的 `gpu_info` 端点。
        """
        return frozenset()

    def probe_connection(self) -> tuple:
        """探一次连通性, 返回 `(ok, reason)`。同 `SSHExecutor.probe_connection` 的理由"""
        if not self.base_url:
            return False, "未配置 base_url (REMOTE_HTTP_URL 为空)"
        try:
            req = urllib.request.Request(f"{self.base_url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True, ""
                return False, f"/health 返回 HTTP {resp.status}"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    def lock_identity(self) -> str:
        return f"http:{self.base_url}"

    def get_hardware_info(self) -> dict:
        """**抛** `HardwareProbeFailed` —— 这个协议上问不到对面是谁 (doc 49 §3.3 S2)。

        旧写法无条件返回一整套 A100 量级的虚构字典。它比 S1 更彻底: S1 至少先试了探测,
        这里连探测都没有, 所以任何读到它的地方 (Designer 的 tiling 约束、环境指纹)
        拿到的都是凭据为零的数。

        这一版**不**改成真探测, 因为服务端根本没有硬件端点 —— 这个类只讲得出
        `/health`、`/api/compile`、`/api/execute`、`/api/benchmark` 四件事,
        「对面是什么卡」不在其中。真探测需要先在服务端加一个 `gpu_info` 端点,
        那是协议改动, 不在本片的范围里。宁可抛, 不给一个能当分母用的假数。
        """
        raise HardwareProbeFailed(
            self.label,
            "HTTP 协议没有硬件端点 (只有 /health, /api/compile, /api/execute, /api/benchmark)",
            endpoint_hint="要在这个传输上拿到硬件信息, 先在远端服务加一个 gpu_info 端点;"
                          " 只是要跑真机请改用 mode=\"ssh\"。")

    def compile(self, kernel_code: str) -> bool:
        try:
            result = self._remote_call("/api/compile", {"code": kernel_code})
            if not result.get("success", False):
                raise SyntaxError(result.get("error", "Remote compile failed"))
            return True
        except urllib.error.URLError as e:
            raise SyntaxError(f"HTTP compile error: {e.reason}")

    def execute(self, kernel_code: str, inputs: dict) -> RemoteExecutionResult:
        payload = {"code": kernel_code, "inputs": self._serialize_inputs(inputs)}
        result = self._remote_call("/api/execute", payload)
        return RemoteExecutionResult(
            success=result.get("success", False),
            output_shape=result.get("output_shape"),
            output_sample=result.get("output_sample"),
            execution_time_ms=result.get("execution_time_ms", 0),
            gflops=result.get("gflops", 0),
            memory_bandwidth_gbs=result.get("memory_bandwidth_gbs", 0),
            error=result.get("error", ""),
        )

    def benchmark(self, kernel_code: str, inputs: dict, warmup: int = 3, repeat: int = 10) -> dict:
        payload = {"code": kernel_code, "inputs": self._serialize_inputs(inputs),
                  "warmup": warmup, "repeat": repeat}
        return self._remote_call("/api/benchmark", payload)

    def _remote_call(self, endpoint: str, payload: dict) -> dict:
        url = f"{self.base_url}{endpoint}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _serialize_inputs(inputs: dict) -> dict:
        import numpy as np
        out = {}
        for k, v in inputs.items():
            arr = np.asarray(v)
            out[k] = {"shape": list(arr.shape), "dtype": str(arr.dtype), "data": arr.flatten().tolist()}
        return out


class LocalExecutor(Backend):
    """
    本地 CPU 执行 (fallback)
    """

    def __init__(self):
        self.name = "LocalExecutor (CPU)"

    @property
    def label(self) -> str:
        return self.name

    @property
    def capabilities(self) -> frozenset:
        """**空集**: CPU 回退, 不是加速卡, 也不提供任何远端协议能力

        空集让降级后的门面**在能力上看得出来**是降级了 —— 调用方不必再去
        `"LocalExecutor" in remote.name` 里做子串匹配 (doc 49 §3.1 A3, 三个驱动各抄了一遍)。
        """
        return frozenset()

    def lock_identity(self) -> str:
        return "local"

    # 这里**故意没有** `check_connection` / `probe_connection`。
    # 它以前恒返回 True (doc 49 §3.3 S4), 与降级梯子 (S3) 合起来的结果是「梯子永远以
    # 成功结束」: 任何一级探测失败都会被它兜住, 于是「连不上」与「连上了一个 CPU 回退」
    # 在返回值上无法区分。对 CPU 回退问「能不能连上」本身没有意义 —— 它不是一个远端。
    # 删掉而不是改成返回 False: 降级到它是**梯子的终点**, 不是一次探测的结果。

    def get_hardware_info(self) -> dict:
        return {
            "name": "Local CPU",
            "host": "localhost",
            "compute_units": 1,
            "warp_size": 1,
            "vector_width": 1,
            "memory": {"global": {"size": "N/A", "bandwidth": "N/A"},
                       "local": {"size": "N/A", "bandwidth": "N/A"}},
        }

    def compile(self, kernel_code: str) -> bool:
        ns = {}
        exec(kernel_code, ns)
        if "kernel" not in ns or not callable(ns["kernel"]):
            raise SyntaxError("Missing 'kernel' function")
        return True

    def execute(self, kernel_code: str, inputs: dict) -> RemoteExecutionResult:
        import numpy as np
        import time
        from .anticheat import shield_torch
        result = RemoteExecutionResult(device="cpu")
        try:
            # 反作弊: 执行期间屏蔽 torch 原生算子回退
            with shield_torch():
                ns = {}
                exec(kernel_code, ns)
                kernel_func = ns["kernel"]
                start = time.perf_counter()
                output = kernel_func(**inputs)
                elapsed = time.perf_counter() - start
            out_arr = np.asarray(output)
            result.output_shape = list(out_arr.shape)
            flat = out_arr.flatten()
            # 全量返回输出样本，确保 L2 功能验证 100% 覆盖
            result.output_sample = flat.tolist()
            result.execution_time_ms = elapsed * 1000
            if "A" in inputs and "B" in inputs:
                a = np.asarray(inputs["A"]); b = np.asarray(inputs["B"])
                m, k = a.shape; k2, n = b.shape
                flops = 2 * m * n * k
                result.gflops = (flops / elapsed / 1e9) if elapsed > 0 else 0
                bytes_moved = (m * k + k * n + m * n) * 4
                result.memory_bandwidth_gbs = (bytes_moved / elapsed / 1e9) if elapsed > 0 else 0
            result.success = True
        except Exception as e:
            result.success = False
            result.error = str(e)
        return result

    def benchmark(self, kernel_code: str, inputs: dict, warmup: int = 3, repeat: int = 10) -> dict:
        import numpy as np
        import time
        ns = {}
        exec(kernel_code, ns)
        kernel_func = ns["kernel"]
        for _ in range(warmup):
            kernel_func(**inputs)
        times = []
        for _ in range(repeat):
            start = time.perf_counter()
            kernel_func(**inputs)
            times.append(time.perf_counter() - start)
        avg_time = sum(times) / len(times)
        gflops = 0.0
        bandwidth = 0.0
        if "A" in inputs and "B" in inputs:
            a = np.asarray(inputs["A"]); b = np.asarray(inputs["B"])
            m, k = a.shape; k2, n = b.shape
            flops = 2 * m * n * k
            gflops = (flops / avg_time / 1e9) if avg_time > 0 else 0
            bytes_moved = (m * k + k * n + m * n) * 4
            bandwidth = (bytes_moved / avg_time / 1e9) if avg_time > 0 else 0
        return {
            "success": True,
            "avg_time_ms": round(avg_time * 1000, 4),
            "min_time_ms": round(min(times) * 1000, 4),
            "max_time_ms": round(max(times) * 1000, 4),
            "gflops": round(gflops, 4),
            "memory_bandwidth_gbs": round(bandwidth, 4),
            "runs": repeat,
            "device": "cpu",
        }


class RemoteExecutor(Backend):
    """
    统一远程执行入口 —— `mode` 决定「连不上」意味着什么

    * `mode="ssh"` / `"http"`：**承诺**。探测失败就抛 `BackendUnavailable`, 不降级。
    * `mode="auto"`（默认）：**偏好**。SSH → HTTP → 本地 CPU 一路降, 但降级会记进
      `resolution.degraded`, 并由 `capabilities` 变成空集暴露出来。
    * `mode="local"`：显式选择本机 CPU。

    能力与锁的标识都**派生自解析后的内层后端**, 门面自己一个都不加 —— 所以「解析成了
    什么」是可读的数据 (`resolution`), 不需要靠显示名子串去猜。

    使用方式:
        # SSH 模式 (推荐) — 默认连 ~/.ssh/config 里的 algorithm-h20 Host 别名,
        # 端口 / 用户名 / 密钥均由该 Host 块提供, 仓库中不出现真实值
        executor = RemoteExecutor(mode="ssh")

        # 自动检测模式
        executor = RemoteExecutor()  # 尝试 SSH -> HTTP -> Local, 降级留痕
    """

    def __init__(
        self,
        mode: str = "auto",
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        key_path: Optional[str] = None,
        http_url: Optional[str] = None,
        work_dir: Optional[str] = None,
        timeout: int = 300,
        platform: str = "nvidia",  # 接入其他 GPU 平台时透传给 SSHExecutor
    ):
        self._mode = mode
        self._backend = None
        self._resolution: Optional[Resolution] = None
        self._platform = platform
        self._host = host
        self._port = port
        self._user = user
        self._key_path = key_path
        self._http_url = http_url
        self._work_dir = work_dir
        self._timeout = timeout

    def _init_backend(self):
        """延迟解析执行侧, 并把解析结果记成 `Resolution` 数据

        `mode` 的三种语义 (doc 49 §3.3 S3): `"ssh"`/`"http"` 是**承诺** —— 探测失败就抛
        `BackendUnavailable`, 不试别的传输、不落到 CPU; `"auto"` 是**偏好** —— 可以一路
        降到 LocalExecutor, 但降级会写进 `Resolution.degraded`; `"local"` 是显式选择本机 CPU。
        """
        if self._backend is not None:
            return

        if self._mode == "local":
            self._backend = LocalExecutor()
            self._resolution = Resolution("local", self._backend.label)
            print(f"  [RemoteExecutor] 使用本地 CPU 模式")
            return

        attempts = []
        if self._mode in ("ssh", "auto"):
            ssh = SSHExecutor(
                host=self._host,
                port=self._port,
                user=self._user,
                key_path=self._key_path,
                work_dir=self._work_dir,
                timeout=self._timeout,
                platform=self._platform,
            )
            ok, reason = ssh.probe_connection()
            if ok:
                self._backend = ssh
                self._resolution = Resolution(self._mode, ssh.label, attempts=attempts)
                print(f"  [RemoteExecutor] SSH 连接成功: {ssh.name}")
                return
            attempts.append((ssh.label, reason))
            print(f"  [RemoteExecutor] SSH 不可用 ({ssh.host}): {reason}")

        if self._mode in ("http", "auto"):
            http = HTTPExecutor(base_url=self._http_url, timeout=self._timeout)
            ok, reason = http.probe_connection()
            if ok:
                self._backend = http
                self._resolution = Resolution(self._mode, http.label, attempts=attempts)
                print(f"  [RemoteExecutor] HTTP 连接成功: {http.name}")
                return
            attempts.append((http.label, reason))
            print(f"  [RemoteExecutor] HTTP 不可用: {reason}")

        if self._mode != "auto":
            # 承诺没兑现。这里**不**给 self._backend 赋值, 于是下一次访问会再抛一次 ——
            # 「失败状态」不该被缓存成一个能用的对象 (DSH: it never degrades to a default)。
            raise BackendUnavailable(Resolution(self._mode, degraded=True, attempts=attempts))

        self._backend = LocalExecutor()
        self._resolution = Resolution("auto", self._backend.label, degraded=True,
                                      attempts=attempts)
        print(f"  [RemoteExecutor] auto 模式降级到本地 CPU 模式 "
              f"(试过 {len(attempts)} 个远端: {', '.join(l for l, _ in attempts) or '无'})")

    @property
    def backend(self):
        self._init_backend()
        return self._backend

    @property
    def resolution(self) -> Resolution:
        """解析结果 (会触发延迟初始化)。降级是**数据**, 不是一行 print"""
        self._init_backend()
        return self._resolution

    @property
    def name(self) -> str:
        return self.backend.name

    @property
    def label(self) -> str:
        return self.name

    @property
    def capabilities(self) -> frozenset:
        """**派生自解析后的内层后端**, 门面自己一个都不加

        这是「把解析结果记成数据」的落点: `mode="auto"` 降级到 LocalExecutor 之后,
        这里返回的就是空集 —— 降级**在能力上看得出来**, 而不是靠调用方去
        `"LocalExecutor" in remote.name` 里做子串匹配 (doc 49 §3.1 A3, 三个驱动各抄了一遍)。
        与 `name` 一样会触发延迟初始化。
        """
        return self.backend.capabilities

    @property
    def selected_gpu(self) -> Optional[int]:
        """转发内层的实际用卡 (#52): 报表问的是「跑在哪张卡上」, 那只有执行侧知道

        门面**不**自己记一份: 以前 `knobs.gpu` 读的是本地环境变量 (用户想的那张卡),
        于是「记录」与「事实」不同源。与 `capabilities` 一样会触发延迟初始化。
        """
        return self.backend.selected_gpu

    def get_hardware_info(self) -> dict:
        return self.backend.get_hardware_info()

    def ensure_environment(self, required_pips: Optional[list] = None) -> tuple:
        """前置环境自检 + 自动配置。**只转发**

        旧写法在内层没有这个方法时返回 `(True, "")`, 理由是「没有要 provision 的东西
        就是就绪的诚实答案」。对 LocalExecutor 诚实, 对 HTTPExecutor 不诚实 ——
        它把「这个后端没法准备自己的环境」上报成了「环境已就绪」(doc 49 §3.2 B1)。
        现在: 内层没有 ⇒ 基类存根抛 `CapabilityMissing`; 要不要调用由消费方按
        `Cap.ENV_PROVISION in backend.capabilities` 决定。

        Returns:
            (all_passed, manifest_text)
        """
        return self.backend.ensure_environment(required_pips=required_pips)

    def deploy(self) -> bool:
        """把 runner 及其同级依赖同步到远端。只转发, 不再把「没东西可部署」当成「部署成功」"""
        return self.backend.deploy()

    def compile(self, kernel_code: str) -> bool:
        return self.backend.compile(kernel_code)

    def _lock_key(self) -> str:
        """执行锁键: 同一执行侧的 execute/benchmark 进程内串行

        按**解析后的**后端取键 (doc 49 §3.3 S6)。旧写法按请求的 `self._mode`:
        `mode="auto"` 降级到本地之后键是 `auto:<host>`, 而 `if self._mode == "local"`
        永远不触发 ⇒ 一个真跑在本机 CPU 上的后端与另一个 `mode="local"` 的后端拿到
        **两把不同的锁**, 于是它们可以并发争同一份 CPU。
        顺带的好处: 取键会触发解析, 所以 `mode="ssh"` 连不上时在**取锁那一步**就抛,
        而不是等到锁里面。
        """
        return self.backend.lock_identity()

    def execute(self, kernel_code: str, inputs: dict) -> RemoteExecutionResult:
        with _get_exec_lock(self._lock_key()):
            return self.backend.execute(kernel_code, inputs)

    def benchmark(self, kernel_code: str, inputs: dict, warmup: int = 3, repeat: int = 10) -> dict:
        with _get_exec_lock(self._lock_key()):
            return self.backend.benchmark(kernel_code, inputs, warmup, repeat)

    def verify_function(self, kernel_code: str, inputs: dict, kernel_type: str,
                        max_error: float, precision: str = "",
                        reference_source: str = "") -> Optional[dict]:
        """远端紧凑比对协议。**只转发, 不判断内层有没有这个方法**

        旧写法是 `fn = getattr(self.backend, "verify_function", None); return fn(...) if fn else None`,
        于是「内层不支持」与「内层支持但返回了 None」共用一个值, 而 verifier 把 None 读成
        「走旧执行器兼容路径」—— doc 49 §3.2 B3/B4。现在内层是 `Backend`, 不支持时
        基类存根会抛 `CapabilityMissing`, 异常里带着**内层**的身份。
        """
        with _get_exec_lock(self._lock_key()):
            return self.backend.verify_function(kernel_code, inputs, kernel_type, max_error,
                                                precision=precision,
                                                reference_source=reference_source)

    def benchmark_with_baselines(self, kernel_code: str, inputs: dict,
                                 kernel_type: str, warmup: int = 10,
                                 repeat: int = 30, precision: str = "",
                                 baseline_priority=None,
                                 reference_source: str = "") -> Optional[dict]:
        """同机同卡基准协议。同上: 只转发"""
        with _get_exec_lock(self._lock_key()):
            return self.backend.benchmark_with_baselines(
                kernel_code, inputs, kernel_type, warmup, repeat,
                precision=precision, baseline_priority=baseline_priority,
                reference_source=reference_source)

    def analyze_dependencies(self, kernel_code: str) -> list:
        """解析源码里缺失的 import (L1 编译失败后的按需补装要用)

        **门面此前根本没有这个方法**, 而 conductor 的守卫是
        `hasattr(self.backend, "analyze_dependencies")` ⇒ 那条恢复路径在生产里从来没跑过
        (doc 49 §3.2 B6, 与已修好的 B5 完全同型)。基类存根 + 转发一起装上之后,
        `hasattr` 恒为真, 能力判断改由 `Cap.DEP_ANALYSIS in backend.capabilities` 承担。
        """
        return self.backend.analyze_dependencies(kernel_code)

    def __repr__(self) -> str:
        return self.name
