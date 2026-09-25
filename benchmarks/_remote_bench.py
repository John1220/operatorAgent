"""Remote GPU Benchmark Infrastructure

Provides reusable utilities for benchmarking Triton operator implementations
against PyTorch baselines on remote GPU servers.

Usage:
    from benchmarks._remote_bench import run_remote_benchmark, BenchmarkResult
"""

import sys
import os
import json
import tempfile
import subprocess
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BenchmarkResult:
    """Benchmark result for a single operator"""
    operator: str
    shape: str
    dtype: str
    triton_fwd_ms: float = 0.0
    triton_bwd_ms: float = 0.0
    pytorch_fwd_ms: float = 0.0
    pytorch_bwd_ms: float = 0.0
    fwd_speedup: float = 0.0
    bwd_speedup: float = 0.0
    triton_gbps: float = 0.0
    pytorch_gbps: float = 0.0
    device: str = ""
    passed: bool = False
    error: str = ""


REMOTE_BENCH_SCRIPT = '''\
import json
import sys
import time
import importlib.util
import numpy as np

def run_benchmark(payload_path):
    import torch
    try:
        import triton
        import triton.language as tl
    except ImportError as e:
        return {"success": False, "error": f"Triton not available: {e}"}

    with open(payload_path) as f:
        payload = json.load(f)

    kernel_file = payload["kernel_file"]      # Path to kernel .py file
    reference_file = payload.get("reference_file", "")  # Path to reference .py file
    input_specs = payload["inputs"]
    operator = payload["operator"]
    dtype_str = payload.get("dtype", "fp16")
    warmup = payload.get("warmup", 10)
    repeat = payload.get("repeat", 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return {"success": False, "error": "No CUDA device available"}

    # Load kernel from file (Triton JIT requires file-based definition)
    spec = importlib.util.spec_from_file_location("kernel_module", kernel_file)
    kernel_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel_module)
    kernel_fn = getattr(kernel_module, "kernel", None)
    if kernel_fn is None:
        return {"success": False, "error": "No 'kernel' function found in kernel file"}

    # Load reference from file
    ref_fn = None
    if reference_file:
        ref_spec = importlib.util.spec_from_file_location("reference_module", reference_file)
        ref_module = importlib.util.module_from_spec(ref_spec)
        ref_spec.loader.exec_module(ref_module)
        ref_fn = getattr(ref_module, "reference", None)

    # Generate inputs locally (on remote GPU) using seed for reproducibility
    dtype = torch.float16 if dtype_str == "fp16" else torch.float32
    torch_inputs = {}
    for k, spec in input_specs.items():
        shape = spec["shape"]
        seed = spec.get("seed", 42)
        rng = np.random.default_rng(seed)
        data = (rng.standard_normal(shape, dtype=np.float32) * 0.1).astype(np.float32)
        torch_inputs[k] = torch.from_numpy(data).to(device).to(dtype)

    # ---- Forward benchmark ----
    for _ in range(warmup):
        with torch.no_grad():
            _ = kernel_fn(**torch_inputs)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = kernel_fn(**torch_inputs)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    triton_fwd_ms = sum(times) / len(times)

    # ---- Reference forward benchmark ----
    pytorch_fwd_ms = 0.0
    if ref_fn is not None:
        for _ in range(warmup):
            with torch.no_grad():
                _ = ref_fn(**torch_inputs)
        torch.cuda.synchronize()

        ref_times = []
        for _ in range(repeat):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = ref_fn(**torch_inputs)
            torch.cuda.synchronize()
            ref_times.append((time.perf_counter() - t0) * 1000)

        ref_times.sort()
        pytorch_fwd_ms = sum(ref_times) / len(ref_times)

    # Compute GB/s based on all input+output bytes (operator-agnostic)
    dtype_bytes = 2 if dtype_str == "fp16" else 4
    # Sum all input sizes as "read" bytes
    total_input_elements = sum(
        int(np.prod(spec["shape"])) for spec in input_specs.values()
    )
    # Output bytes = sum of all input shapes (assumes output is same size as largest input,
    # which is a reasonable approximation for elementwise ops; matmul is special-cased below)
    # For matmul: output is M×N, but we don't have that info generically.
    # Use total input as approximation (accounts for reads + writes of similar size)
    fwd_bytes = total_input_elements * dtype_bytes * 2  # read + write
    triton_fwd_gbps = fwd_bytes / (triton_fwd_ms / 1000) / 1e9
    pytorch_fwd_gbps = fwd_bytes / (pytorch_fwd_ms / 1000) / 1e9 if pytorch_fwd_ms > 0 else 0

    return {
        "success": True,
        "operator": operator,
        "device": device,
        "dtype": dtype_str,
        "triton_fwd_ms": round(triton_fwd_ms, 4),
        "triton_bwd_ms": 0.0,
        "pytorch_fwd_ms": round(pytorch_fwd_ms, 4),
        "pytorch_bwd_ms": 0.0,
        "fwd_speedup": round(pytorch_fwd_ms / triton_fwd_ms, 4) if triton_fwd_ms > 0 and pytorch_fwd_ms > 0 else 0,
        "bwd_speedup": 0.0,
        "triton_fwd_gbps": round(triton_fwd_gbps, 2),
        "pytorch_fwd_gbps": round(pytorch_fwd_gbps, 2),
        "triton_bwd_gbps": 0.0,
        "pytorch_bwd_gbps": 0.0,
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "Usage: python script.py <payload.json>"}))
        sys.exit(1)
    result = run_benchmark(sys.argv[1])
    print(json.dumps(result))
'''


def resolve_ssh_config(host=None, user=None, port=None, key=None, work_dir=None):
    """解析 SSH 连接参数: 未显式给出的项退回环境变量, 再退回 ssh 别名约定

    host 默认只是 ~/.ssh/config 里的一个 Host 别名; port / user / key 默认不设,
    交由该 Host 块解析。仓库中因此不出现真实 IP、端口、用户名或密钥路径 ——
    换机器只改 ssh config, 或用 REMOTE_SSH_* 环境变量覆盖。
    """
    if host is None:
        host = os.environ.get("REMOTE_SSH_HOST", "algorithm-h20")
    if user is None:
        user = os.environ.get("REMOTE_SSH_USER") or None
    if port is None:
        _env_port = os.environ.get("REMOTE_SSH_PORT", "")
        port = int(_env_port) if _env_port else None
    if key is None:
        key = os.environ.get("REMOTE_SSH_KEY", "")
    if work_dir is None:
        work_dir = os.environ.get("REMOTE_WORK_DIR", "~/operator_agent")
    return host, user, port, key, work_dir


def ssh_label(host, port) -> str:
    """显示用: port 未显式配置时来自 ~/.ssh/config, 不能打印成 None"""
    return f"{host}:{port}" if port else host


def _ssh_exec_base(host: str, port, user, key_path: str = "") -> list:
    """Build SSH command base

    port / user / key_path 为 None 或空时不写进命令行, 交由 ~/.ssh/config 的
    Host 块解析 —— 与 operator_agent/remote_executor.py 的 SSHExecutor 同一套
    约定, 这样仓库里只需保留一个 ssh 别名, 不必写死真实 IP、端口、用户名和密钥。
    """
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no",
           "-o", "ConnectTimeout=10", "-o", "BatchMode=yes"]
    if port:
        cmd.extend(["-p", str(port)])
    if key_path:
        cmd.extend(["-i", os.path.expanduser(key_path)])
    cmd.append(f"{user}@{host}" if user else host)
    return cmd


def _ssh_run(host: str, port, user, cmd: str,
             key_path: str = "", stdin_data: str = "", timeout: int = 300) -> str:
    """Execute remote command via SSH"""
    base = _ssh_exec_base(host, port, user, key_path)
    full_cmd = base + [cmd]
    proc = subprocess.run(
        full_cmd, input=stdin_data, capture_output=True, text=True, timeout=timeout
    )
    if proc.returncode != 0:
        raise RuntimeError(f"SSH exec failed [rc={proc.returncode}]: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _upload_file(host: str, port, user, local_path: str,
                 remote_path: str, key_path: str = "", timeout: int = 300):
    """Upload file via SSH stdin pipe (avoids command line length limits)"""
    with open(local_path, 'r') as f:
        content = f.read()
    base = _ssh_exec_base(host, port, user, key_path)
    full_cmd = base + [f"cat > {remote_path}"]
    proc = subprocess.run(
        full_cmd, input=content, capture_output=True, text=True, timeout=timeout
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Upload failed [rc={proc.returncode}]: {proc.stderr.strip()}")


def run_remote_benchmark(
    operator: str,
    kernel_code: str,
    inputs: dict,
    dtype: str = "fp16",
    reference_code: str = "",
    ssh_host: str = None,
    ssh_user: str = None,
    ssh_port: int = None,
    ssh_key: str = None,
    work_dir: str = None,
    warmup: int = 10,
    repeat: int = 50,
    do_backward: bool = True,
) -> BenchmarkResult:
    """Run a benchmark on remote GPU server

    Args:
        operator: Name of the operator (e.g., "layernorm")
        kernel_code: Python code defining a `kernel` function
        inputs: Dict with input specs: {"X": {"shape": [1024, 768], "seed": 42}, ...}
        dtype: Precision ("fp16" or "fp32")
        reference_code: Python code defining a `reference` function
        ssh_host: Remote host
        ssh_user: Remote user
        ssh_port: Remote port
        ssh_key: SSH key path
        work_dir: Remote working directory
        warmup: Warmup iterations
        repeat: Measurement iterations
        do_backward: Whether to benchmark backward pass

    Returns:
        BenchmarkResult with timing and speedup data
    """
    ssh_host, ssh_user, ssh_port, ssh_key, work_dir = resolve_ssh_config(
        ssh_host, ssh_user, ssh_port, ssh_key, work_dir
    )

    # Prepare input specs (shape + seed only, data generated on remote side)
    input_specs = {}
    for k, spec in inputs.items():
        input_specs[k] = {
            "shape": spec["shape"],
            "seed": spec.get("seed", 42),
        }

    payload = {
        "operator": operator,
        "kernel_code": kernel_code,
        "reference_code": reference_code,
        "inputs": input_specs,
        "dtype": dtype,
        "warmup": warmup,
        "repeat": repeat,
        "do_backward": do_backward,
    }

    # Build shape string from all input specs (operator-agnostic)
    shape_parts = []
    for k, spec in inputs.items():
        shape_parts.append(f"{k}={spec['shape']}")
    shape_str = ", ".join(shape_parts) if shape_parts else "unknown"

    try:
        # 1. Ensure work dir exists
        _ssh_run(ssh_host, ssh_port, ssh_user, f"mkdir -p {work_dir}", ssh_key)

        # 2. Write kernel code to a .py file on remote (Triton JIT requires file-based definition)
        kernel_filename = f"_kernel_{operator}.py"
        local_kernel_file = os.path.join(tempfile.gettempdir(), kernel_filename)
        with open(local_kernel_file, 'w') as f:
            f.write(kernel_code)
        remote_kernel_file = f"{work_dir}/{kernel_filename}"
        _upload_file(ssh_host, ssh_port, ssh_user, local_kernel_file, remote_kernel_file, ssh_key)

        # 3. Write reference code to a .py file on remote
        reference_filename = f"_reference_{operator}.py"
        local_ref_file = os.path.join(tempfile.gettempdir(), reference_filename)
        with open(local_ref_file, 'w') as f:
            f.write(reference_code)
        remote_ref_file = f"{work_dir}/{reference_filename}"
        _upload_file(ssh_host, ssh_port, ssh_user, local_ref_file, remote_ref_file, ssh_key)

        # 4. Write benchmark script to remote
        remote_script = f"{work_dir}/_bench_{operator}.py"
        local_script = os.path.join(tempfile.gettempdir(), f"_remote_bench_{operator}.py")
        with open(local_script, 'w') as f:
            f.write(REMOTE_BENCH_SCRIPT)
        _upload_file(ssh_host, ssh_port, ssh_user, local_script, remote_script, ssh_key)

        # 5. Write payload JSON to remote (references the kernel/reference file paths)
        payload["kernel_file"] = remote_kernel_file
        payload["reference_file"] = remote_ref_file
        remote_payload = f"{work_dir}/_bench_payload_{operator}.json"
        local_payload = os.path.join(tempfile.gettempdir(), f"_bench_payload_{operator}.json")
        with open(local_payload, 'w') as f:
            json.dump(payload, f)
        _upload_file(ssh_host, ssh_port, ssh_user, local_payload, remote_payload, ssh_key)

        # 6. Execute
        remote_cmd = (
            f"cd {work_dir} && "
            f"~/.local/bin/uv run python {remote_script} {remote_payload}"
        )
        output = _ssh_run(ssh_host, ssh_port, ssh_user, remote_cmd, ssh_key, timeout=600)

        # Parse result
        for line in reversed(output.split("\n")):
            line = line.strip()
            if line.startswith("{"):
                result_data = json.loads(line)
                break
        else:
            return BenchmarkResult(
                operator=operator, shape=shape_str, dtype=dtype,
                passed=False, error=f"No JSON in output: {output[:500]}",
            )

        if not result_data.get("success"):
            return BenchmarkResult(
                operator=operator, shape=shape_str, dtype=dtype,
                device=result_data.get("device", "unknown"),
                passed=False,
                error=result_data.get("error", "Unknown error"),
            )

        return BenchmarkResult(
            operator=operator,
            shape=shape_str,
            dtype=dtype,
            triton_fwd_ms=result_data.get("triton_fwd_ms", 0),
            triton_bwd_ms=result_data.get("triton_bwd_ms", 0),
            pytorch_fwd_ms=result_data.get("pytorch_fwd_ms", 0),
            pytorch_bwd_ms=result_data.get("pytorch_bwd_ms", 0),
            fwd_speedup=result_data.get("fwd_speedup", 0),
            bwd_speedup=result_data.get("bwd_speedup", 0),
            triton_gbps=result_data.get("triton_fwd_gbps", 0),
            pytorch_gbps=result_data.get("pytorch_fwd_gbps", 0),
            device=result_data.get("device", "remote"),
            passed=True,
        )

    except Exception as e:
        return BenchmarkResult(
            operator=operator, shape=shape_str, dtype=dtype,
            passed=False, error=str(e),
        )


def print_benchmark_table(results: list[BenchmarkResult], baseline_name: str = "PyTorch Eager(CUDA)"):
    """Pretty-print benchmark results as a table"""
    print("\n" + "=" * 120)
    print("  Operator Benchmark Results (Remote GPU A100)")
    print("=" * 120)
    header = (
        f"  {'Operator':<12} {'Shape':<30} {'Dtype':<8} {'Triton FWD':<12} {'PyTorch FWD':<12} "
        f"{'FWD Speedup':<12} {'Triton GB/s':<12} {'PyTorch GB/s':<12}"
    )
    print(header)
    print("-" * 120)
    for r in results:
        triton_fwd = f"{r.triton_fwd_ms:.4f}ms" if r.triton_fwd_ms > 0 else "N/A"
        pytorch_fwd = f"{r.pytorch_fwd_ms:.4f}ms" if r.pytorch_fwd_ms > 0 else "N/A"
        speedup = f"{r.fwd_speedup:.3f}x" if r.fwd_speedup > 0 else "N/A"
        triton_gbps = f"{r.triton_gbps:.1f}" if r.triton_gbps > 0 else "N/A"
        pytorch_gbps = f"{r.pytorch_gbps:.1f}" if r.pytorch_gbps > 0 else "N/A"
        row = (
            f"  {r.operator:<12} {r.shape:<30} {r.dtype:<8} {triton_fwd:<12} {pytorch_fwd:<12} "
            f"{speedup:<12} {triton_gbps:<12} {pytorch_gbps:<12}"
        )
        if not r.passed:
            row += f"  ⚠ {r.error[:30]}"
        print(row)
    print("=" * 120)

    passed = sum(1 for r in results if r.passed)
    improved = sum(1 for r in results if r.fwd_speedup > 1.0)
    print(f"\n  Total: {len(results)} benchmarks, {passed} passed, {improved} Triton faster")
    print(f"  Baseline: {baseline_name}")
    print(f"  Speedup = PyTorch_time / Triton_time (> 1.0 means Triton is faster)")
