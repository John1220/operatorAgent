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
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Optional

from .environment import EnvironmentChecker, DependencyAnalyzer, EnvironmentReport, CheckItem
from .hardware_caps import detect_hardware_caps


# 按执行目标串行化 execute/benchmark, 防止并发任务共用同一 GPU 互相污染 benchmark
_EXEC_LOCKS: dict[str, threading.Lock] = {}
_EXEC_LOCKS_GUARD = threading.Lock()


def _get_exec_lock(key: str) -> threading.Lock:
    with _EXEC_LOCKS_GUARD:
        return _EXEC_LOCKS.setdefault(key, threading.Lock())


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


class SSHExecutor:
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
            raise RuntimeError(
                f"SSH exec failed [rc={proc.returncode}]: {detail}"
            )
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
                print(f"  [SSHExecutor] {self.platform} 环境配置警告: {e}")
                self._uv_setup = True
                return True

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
            print(f"  [SSHExecutor] 创建配置文件失败: {e}")

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

    def check_connection(self) -> bool:
        """检查 SSH 连接是否可用"""
        try:
            self._ssh_exec("echo ok")
            return True
        except Exception:
            return False

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
                            "bandwidth": "19TB/s",
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
            print(f"  [SSHExecutor] 获取 GPU 信息失败: {e}")
            return {
                "name": f"NVIDIA GPU (Remote {self.host})",
                "host": self.host,
                "compute_units": 128,
                "warp_size": 32,
                "vector_width": 128,
                "memory": {
                    "global": {"size": "40GB", "bandwidth": "2TB/s"},
                    "local": {"size": "128KB/SM", "bandwidth": "19TB/s"},
                },
            }

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
            return RemoteExecutionResult(success=False, error=str(e))

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
            return {"success": False, "error": str(e)}

    def verify_function(self, kernel_code: str, inputs: dict, kernel_type: str,
                        max_error: float, precision: str = "") -> dict:
        """Compare candidate/reference on the remote GPU and return a compact summary."""
        try:
            if not self._deployed:
                self.deploy()
            return self._ssh_exec_json({
                "action": "verify", "code": kernel_code,
                "inputs": self._serialize_inputs(inputs), "kernel_type": kernel_type,
                "max_error": max_error, "precision": precision,
            })
        except Exception as e:
            return {"success": False, "error": str(e)}

    def benchmark_with_baselines(self, kernel_code: str, inputs: dict,
                                 kernel_type: str, warmup: int = 10,
                                 repeat: int = 30, precision: str = "") -> dict:
        """Measure candidate/eager/torch.compile on the same remote device."""
        try:
            if not self._deployed:
                self.deploy()
            return self._ssh_exec_json({
                "action": "benchmark_with_baselines", "code": kernel_code,
                "inputs": self._serialize_inputs(inputs), "kernel_type": kernel_type,
                "warmup": warmup, "repeat": repeat, "precision": precision,
            })
        except Exception as e:
            return {"success": False, "error": str(e)}

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


class HTTPExecutor:
    """
    通过 HTTP API 调用远程服务 (如果远程有接口服务)
    """

    def __init__(self, base_url: Optional[str] = None, timeout: int = 120):
        # 默认留空: 新环境没有本项目的 HTTP 服务, 8091 属于其他租户。
        # 空串会让 check_connection 失败, auto 模式据此降级, 不会误连别人的服务。
        self.base_url = (base_url or os.environ.get("REMOTE_HTTP_URL", "")).rstrip("/")
        self.timeout = timeout
        self.name = f"HTTPExecutor@{self.base_url}"

    def check_connection(self) -> bool:
        try:
            req = urllib.request.Request(
                f"{self.base_url}/health",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def get_hardware_info(self) -> dict:
        return {
            "name": "NVIDIA GPU (Remote HTTP)",
            "host": self.base_url,
            "compute_units": 128,
            "warp_size": 32,
            "vector_width": 128,
            "memory": {
                "global": {"size": "40GB", "bandwidth": "2TB/s"},
                "local": {"size": "128KB/SM", "bandwidth": "19TB/s"},
            },
        }

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


class LocalExecutor:
    """
    本地 CPU 执行 (fallback)
    """

    def __init__(self):
        self.name = "LocalExecutor (CPU)"

    def check_connection(self) -> bool:
        return True

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


class RemoteExecutor:
    """
    统一远程执行入口 - 自动选择最佳后端
    
    优先级:
    1. SSH 模式 (直接连远程 GPU 服务器)
    2. HTTP 模式 (如果远程有接口服务)
    3. 本地模式 (fallback)
    
    使用方式:
        # SSH 模式 (推荐) — 默认连 ~/.ssh/config 里的 algorithm-h20 Host 别名,
        # 端口 / 用户名 / 密钥均由该 Host 块提供, 仓库中不出现真实值
        executor = RemoteExecutor(mode="ssh")
        
        # 自动检测模式
        executor = RemoteExecutor()  # 默认尝试 SSH -> HTTP -> Local
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
        self._platform = platform
        self._host = host
        self._port = port
        self._user = user
        self._key_path = key_path
        self._http_url = http_url
        self._work_dir = work_dir
        self._timeout = timeout

    def _init_backend(self):
        """延迟初始化后端，自动检测可用连接"""
        if self._backend is not None:
            return

        if self._mode == "local":
            self._backend = LocalExecutor()
            print(f"  [RemoteExecutor] 使用本地 CPU 模式")
            return

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
            if ssh.check_connection():
                self._backend = ssh
                print(f"  [RemoteExecutor] SSH 连接成功: {ssh.name}")
                return
            print(f"  [RemoteExecutor] SSH 不可用 ({ssh.host})")

        if self._mode in ("http", "auto"):
            http = HTTPExecutor(base_url=self._http_url, timeout=self._timeout)
            if http.check_connection():
                self._backend = http
                print(f"  [RemoteExecutor] HTTP 连接成功: {http.name}")
                return
            print(f"  [RemoteExecutor] HTTP 不可用")

        # Fallback to local
        self._backend = LocalExecutor()
        print(f"  [RemoteExecutor] 回退到本地 CPU 模式")

    @property
    def backend(self):
        self._init_backend()
        return self._backend

    @property
    def name(self) -> str:
        return self.backend.name

    def get_hardware_info(self) -> dict:
        return self.backend.get_hardware_info()

    def ensure_environment(self, required_pips: Optional[list] = None) -> tuple:
        """前置环境自检 + 自动配置, 委托给底层后端

        门面必须显式暴露本方法: conductor.py 的守卫是
        `if hasattr(self.backend, "ensure_environment")`, 而 driver 传进 Conductor
        的是 RemoteExecutor 门面。门面没有这个方法时守卫恒为 False, 于是「任务开始
        前的环境自检」在真机路径上被静默跳过 —— 不报错, 只是不执行。

        Returns:
            (all_passed, manifest_text)。底层后端无需配置环境时 (LocalExecutor /
            HTTPExecutor) 返回 (True, ""): 没有要 provision 的东西, 这就是「就绪」
            的诚实答案; manifest 只在 all_passed 为 False 时被消费。
        """
        fn = getattr(self.backend, "ensure_environment", None)
        if fn is None:
            return True, ""
        return fn(required_pips=required_pips)

    def deploy(self) -> bool:
        """把 runner 及其同级依赖同步到远端; 底层后端无需部署时视为已成功"""
        fn = getattr(self.backend, "deploy", None)
        return fn() if fn is not None else True

    def compile(self, kernel_code: str) -> bool:
        return self.backend.compile(kernel_code)

    def _lock_key(self) -> str:
        """执行锁键: 同一目标主机的 execute/benchmark 进程内串行"""
        if self._mode == "local":
            return "local"
        # 未显式传 host/base_url 时按各后端自己的默认解析, 否则不同目标都会落到
        # "ssh:None" 上被误判为同一主机 (过度串行, 不会漏锁, 但违背本方法的语义)。
        if self._mode == "http":
            return f"http:{self._http_url or HTTPExecutor().base_url}"
        host = self._host or os.environ.get("REMOTE_SSH_HOST", "algorithm-h20")
        return f"{self._mode}:{host}"

    def execute(self, kernel_code: str, inputs: dict) -> RemoteExecutionResult:
        with _get_exec_lock(self._lock_key()):
            return self.backend.execute(kernel_code, inputs)

    def benchmark(self, kernel_code: str, inputs: dict, warmup: int = 3, repeat: int = 10) -> dict:
        with _get_exec_lock(self._lock_key()):
            return self.backend.benchmark(kernel_code, inputs, warmup, repeat)

    def verify_function(self, kernel_code: str, inputs: dict, kernel_type: str,
                        max_error: float, precision: str = "") -> Optional[dict]:
        """Remote-only compact verification protocol; legacy backends return None."""
        with _get_exec_lock(self._lock_key()):
            fn = getattr(self.backend, "verify_function", None)
            return fn(kernel_code, inputs, kernel_type, max_error, precision=precision) if fn else None

    def benchmark_with_baselines(self, kernel_code: str, inputs: dict,
                                 kernel_type: str, warmup: int = 10,
                                 repeat: int = 30, precision: str = "") -> Optional[dict]:
        """Same-device benchmark protocol; legacy backends return None."""
        with _get_exec_lock(self._lock_key()):
            fn = getattr(self.backend, "benchmark_with_baselines", None)
            return fn(kernel_code, inputs, kernel_type, warmup, repeat, precision=precision) if fn else None

    def __repr__(self) -> str:
        return self.name
