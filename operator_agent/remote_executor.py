"""
远程代码执行器
支持两种连接模式:
  1. SSH 模式 (推荐): 通过 SSH 连接远程 GPU 服务器，自动部署并执行脚本
  2. HTTP 模式: 通过 HTTP API 调用 (如果远程有接口服务)
  3. 本地模式: fallback 到本地 CPU 执行

环境变量:
  REMOTE_SSH_HOST     - SSH 主机 (默认 152.136.252.171)
  REMOTE_SSH_PORT     - SSH 端口 (默认 22)
  REMOTE_SSH_USER     - SSH 用户名 (默认 zhanghan)
  REMOTE_SSH_KEY      - SSH 私钥路径 (默认 ~/.ssh/id_rsa)
  REMOTE_RUNNER_PATH  - 远程脚本路径 (默认 ~/operator_agent_runner/run_kernel.py)
  REMOTE_HTTP_URL     - HTTP 接口地址 (如 http://152.136.252.171:8091)
"""

import os
import json
import subprocess
import tempfile
import threading
import hashlib
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .environment import EnvironmentChecker, DependencyAnalyzer, EnvironmentReport, CheckItem


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
        port: int = 22,
        user: Optional[str] = None,
        key_path: Optional[str] = None,
        remote_runner_path: Optional[str] = None,
        work_dir: Optional[str] = None,
        timeout: int = 300,
        platform: str = "nvidia",  # "nvidia" or "kunlunxin"
    ):
        self.platform = platform
        if platform == "kunlunxin":
            self.host = host or os.environ.get("REMOTE_SSH_HOST_KL", "43.180.254.145")
            self.port = port or int(os.environ.get("REMOTE_SSH_PORT_KL", "22"))
            self.user = user or os.environ.get("REMOTE_SSH_USER_KL", "zhanghan")
            self.key_path = key_path or os.environ.get(
                "REMOTE_SSH_KEY_KL", os.path.expanduser("~/.ssh/yita_zhanghan.pem")
            )
            self.work_dir = work_dir or os.environ.get(
                "REMOTE_WORK_DIR_KL", "/home/zhanghan/operator_agent_runner"
            )
            self.remote_runner_path = remote_runner_path or os.environ.get(
                "REMOTE_RUNNER_PATH_KL", f"{self.work_dir}/run_kernel_xpu.py"
            )
            self._python_cmd = "/data1/miniconda3/envs/vllm_xpu/bin/python"
            self._runner_script = "run_kernel_xpu.py"
            self._needs_uv = False
        else:
            self.host = host or os.environ.get("REMOTE_SSH_HOST", "152.136.252.171")
            self.port = port or int(os.environ.get("REMOTE_SSH_PORT", "22"))
            self.user = user or os.environ.get("REMOTE_SSH_USER", "zhanghan")
            self.key_path = key_path or os.environ.get("REMOTE_SSH_KEY", "")
            self.work_dir = work_dir or os.environ.get(
                "REMOTE_WORK_DIR", "/data/zhanghan/operator_agent_runner"
            )
            self.remote_runner_path = remote_runner_path or os.environ.get(
                "REMOTE_RUNNER_PATH", f"{self.work_dir}/run_kernel.py"
            )
            self._python_cmd = None  # uses uv run python
            self._runner_script = "run_kernel.py"
            self._needs_uv = True
        self.timeout = timeout
        self.name = f"SSHExecutor@{self.user}@{self.host}({platform})"
        self._deployed = False
        self._uv_setup = False
        self._gpu_info_cache: Optional[dict] = None
        self._uv_path = "/home/zhanghan/.local/bin/uv"   # ensure_environment 复用

    def _build_ssh_base(self) -> str:
        cmd = ["ssh", "-p", str(self.port), "-o", "StrictHostKeyChecking=no",
               "-o", "ConnectTimeout=10", "-o", "BatchMode=yes"]
        if self.key_path:
            cmd.extend(["-i", os.path.expanduser(self.key_path)])
        cmd.append(f"{self.user}@{self.host}")
        return cmd

    def _ssh_exec(self, remote_cmd: str, stdin_data: Optional[str] = None) -> str:
        """执行远程命令并返回 stdout

        说明:
        - 昆仑芯 XPU 的 import hook (xpytorch_import_hook.py) 会在 stderr 输出
          pkg_resources deprecation warning，但这类 warning 并不影响结果，
          并且即便 rc != 0 只要 stdout 里有可用 JSON 通常也可接受。
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

    def _ssh_exec_json(self, payload: dict) -> dict:
        """通过 SSH 在远程执行 run_kernel.py 并解析 JSON 结果"""
        json_str = json.dumps(payload, ensure_ascii=False)
        if self.platform == "kunlunxin":
            # 昆仑芯使用 conda 环境的 Python
            remote_cmd = (
                f"mkdir -p {self.work_dir} && "
                f"export LD_LIBRARY_PATH=/data1/miniconda3/envs/vllm_xpu/xcudart/lib:"
                f"/data1/miniconda3/envs/vllm_xpu/lib:$LD_LIBRARY_PATH && "
                f"{self._python_cmd} {self.remote_runner_path}"
            )
        else:
            remote_cmd = f"cd {self.work_dir} && /home/zhanghan/.local/bin/uv run python {self.remote_runner_path}"
        output = self._ssh_exec(remote_cmd, stdin_data=json_str)
        for line in reversed(output.split("\n")):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        raise ValueError(f"No JSON found in remote output: {output[:500]}")

    def _setup_uv_env(self) -> bool:
        """在远程工作目录下自动配置 uv 环境 (nvidia) 或使用 conda 环境 (kunlunxin)"""
        if self._uv_setup:
            return True

        if self.platform == "kunlunxin":
            # 昆仑芯使用预安装的 conda 环境，只需要创建工作目录和修复权限
            print(f"  [SSHExecutor] 昆仑芯平台，使用预配置 conda 环境: {self.work_dir}")
            try:
                self._ssh_exec(f"mkdir -p {self.work_dir}")
                # 修复 xblas JIT 权限 (zhanghan 有免密 sudo)
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
                print(f"  [SSHExecutor] 昆仑芯环境配置警告: {e}")
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
            self._ssh_exec("test -f /home/zhanghan/.local/bin/uv")
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
                    f"cd {self.work_dir} && /home/zhanghan/.local/bin/uv init --python {python_path}"
                )
                print(f"  [SSHExecutor] uv 项目初始化成功")
            except Exception as e:
                print(f"  [SSHExecutor] uv 项目初始化失败: {e}")
                return False

        try:
            self._ssh_exec(
                f"cd {self.work_dir} && /home/zhanghan/.local/bin/uv add numpy torch triton"
            )
            print(f"  [SSHExecutor] 依赖安装成功")
        except Exception as e:
            print(f"  [SSHExecutor] 依赖安装失败: {e}")
            return False

        try:
            self._ssh_exec(
                f"echo '3.12' > {self.work_dir}/.python-version"
            )
            print(f"  [SSHExecutor] 创建 .python-version 配置文件")
        except Exception as e:
            print(f"  [SSHExecutor] 创建配置文件失败: {e}")

        self._uv_setup = True
        print(f"  [SSHExecutor] uv 环境配置完成: {self.work_dir}")
        return True

    def deploy(self) -> bool:
        """部署 runner 脚本到远程服务器（如果已存在则跳过）"""
        if not self._setup_uv_env():
            print(f"  [SSHExecutor] 环境配置失败，跳过部署")
            return False

        local_runner = Path(__file__).parent.parent / "remote_scripts" / self._runner_script
        if not local_runner.exists():
            print(f"  [SSHExecutor] 本地脚本不存在: {local_runner}")
            return False

        # 用内容 hash 判断是否需要更新 (替代"存在即跳过", 保证远程脚本始终最新)
        local_content = local_runner.read_bytes()
        local_hash = hashlib.sha256(local_content).hexdigest()[:12]
        try:
            py = self._python_cmd if self._python_cmd else "python3"
            remote_hash = self._ssh_exec(
                f"{py} -c \"import hashlib,sys; "
                f"print(hashlib.sha256(open('{self.remote_runner_path}','rb').read()).hexdigest()[:12])\""
            ).strip()
        except RuntimeError:
            remote_hash = ""   # 远程文件不存在

        if remote_hash == local_hash:
            self._deployed = True
            print(f"  [SSHExecutor] 远程脚本已是最新 (hash={local_hash}): {self.remote_runner_path}")
            return True

        try:
            scp_cmd = ["scp", "-P", str(self.port), "-o", "StrictHostKeyChecking=no"]
            if self.key_path:
                scp_cmd.extend(["-i", os.path.expanduser(self.key_path)])
            scp_cmd.extend([
                str(local_runner),
                f"{self.user}@{self.host}:{self.remote_runner_path}"
            ])
            subprocess.run(scp_cmd, capture_output=True, text=True, timeout=60, check=True)
            self._deployed = True
            action = "更新" if remote_hash else "部署"
            print(f"  [SSHExecutor] 脚本已{action}到 {self.host}:{self.remote_runner_path} (hash={local_hash})")
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
            if self.platform == "kunlunxin":
                self._gpu_info_cache = {
                    "name": result.get("device_name", "Kunlunxin P800"),
                    "vendor": result.get("vendor", "kunlunxin"),
                    "host": self.host,
                    "compute_units": result.get("multiprocessor_count", 8),
                    "warp_size": 32,
                    "vector_width": 128,
                    "memory": {
                        "global": {
                            "size": f"{result.get('total_memory_gb', 96)}GB",
                            "bandwidth": "1.6TB/s",
                        },
                        "local": {"size": "64KB/SM", "bandwidth": "20TB/s"},
                    },
                    "device": "kunlunxin_xpu",
                    "compute_capability": result.get("compute_capability", "kl3"),
                    "torch_version": result.get("torch_version", "unknown"),
                    "triton_version": "not_supported",
                    "device_count": 8,
                    "selected_gpu": 0,
                    "gpu_memory": [],
                }
            else:
                self._gpu_info_cache = {
                    "name": result.get("name", "Unknown GPU"),
                    "host": self.host,
                    "compute_units": result.get("multiprocessor_count", 128),
                    "warp_size": 32,
                    "vector_width": 128,
                    "memory": {
                        "global": {"size": f"{result.get('total_memory_gb', 40)}GB", "bandwidth": "2TB/s"},
                        "local": {"size": "128KB/SM", "bandwidth": "19TB/s"},
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
        避免大矩阵 JSON 序列化/传输瓶颈 (如 4096x4096 单矩阵 ~67MB JSON)。"""
        import numpy as np
        specs = {}
        for k, v in inputs.items():
            arr = np.asarray(v)
            # 基于 shape+dtype 的确定性种子, 同规格可复现
            shape_seed = abs(hash(tuple(arr.shape) + (str(arr.dtype),))) % (2 ** 31 - 1)
            specs[k] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "seed": shape_seed,
            }
        return {"_generate": True, "specs": specs}


class HTTPExecutor:
    """
    通过 HTTP API 调用远程服务 (如果远程有接口服务)
    """

    def __init__(self, base_url: Optional[str] = None, timeout: int = 120):
        self.base_url = (base_url or os.environ.get(
            "REMOTE_HTTP_URL", "http://152.136.252.171:8091"
        )).rstrip("/")
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
        # SSH 模式 (推荐)
        executor = RemoteExecutor(mode="ssh", host="152.136.252.171", user="root")
        
        # 自动检测模式
        executor = RemoteExecutor()  # 默认尝试 SSH -> HTTP -> Local
    """

    def __init__(
        self,
        mode: str = "auto",
        host: Optional[str] = None,
        port: int = 22,
        user: Optional[str] = None,
        key_path: Optional[str] = None,
        http_url: Optional[str] = None,
        work_dir: Optional[str] = None,
        timeout: int = 300,
        platform: str = "nvidia",  # "nvidia" or "kunlunxin"
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

    def compile(self, kernel_code: str) -> bool:
        return self.backend.compile(kernel_code)

    def _lock_key(self) -> str:
        """执行锁键: 同一目标主机的 execute/benchmark 进程内串行"""
        if self._mode == "local":
            return "local"
        return f"{self._mode}:{self._host}"

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
