"""Remote kernel runner for Kunlunxin P800 (KL3 architecture)
部署在昆仑芯服务器上，通过 SSH 调用。
支持 PyTorch (torch_xmlir 后端) 算子执行和 benchmark。

与 NVIDIA run_kernel.py 的区别:
  - 无 Triton 支持 (昆仑芯不支持 Triton DSL)
  - 使用 vllm_xpu conda 环境的 Python (torch 2.5.1+cu118 + torch_xmlir)
  - GPU 选择使用 xpu-smi
  - Baseline 使用 torch.matmul (自动映射到 xblas 昆仑芯 BLAS 库)
  - FP32 不使用 TF32 (昆仑芯 FP32 是原生 FP32 计算)
  - FP16 自动使用昆仑芯 Tensor Core (XPU-TC)

Actions:
  gpu_info  - query GPU info
  compile   - L1 compile/import check
  execute   - L2 functional run
  verify    - L2 functional correctness verification
  benchmark - L3 candidate-only timing
  benchmark_with_baselines - L3 candidate vs one baseline chosen by the control side

action 名与每个 action 读的 payload 键必须和 NVIDIA 的 run_kernel.py 一致, 由
tests/unit/test_runner_contract.py 从两边的源码里解析出来对照。
"""

import sys
import json
import time
import os
import subprocess

# ─── 环境配置 ─────────────────────────────────────────────
# 昆仑芯 XPU 运行时路径
XPU_RUNTIME_PATH = "/usr/local/xpu/so"
XPU_INCLUDE_PATH = "/usr/local/xpu/include"
CONDA_ENV_PATH = "/data1/miniconda3/envs/vllm_xpu"
CONDA_LIB_PATH = os.path.join(CONDA_ENV_PATH, "lib")
CONDA_XCUDART_PATH = os.path.join(CONDA_ENV_PATH, "xcudart", "lib")

# 设置 C++ 头文件搜索路径: torch/extension.h 会 include cuda_runtime_api.h,
# 昆仑芯 CUDA 兼容层头文件在 /usr/local/xpu/include (不在 gcc 默认搜索路径)
os.environ["CPATH"] = XPU_INCLUDE_PATH + ":" + os.environ.get("CPATH", "")
os.environ["CPLUS_INCLUDE_PATH"] = XPU_INCLUDE_PATH + ":" + os.environ.get("CPLUS_INCLUDE_PATH", "")

# 设置 LD_LIBRARY_PATH (在 import torch 前)
os.environ["LD_LIBRARY_PATH"] = (
    f"{CONDA_XCUDART_PATH}:{CONDA_LIB_PATH}:{XPU_RUNTIME_PATH}:"
    + os.environ.get("LD_LIBRARY_PATH", "")
)

# 将 conda env 的 bin 目录加入 PATH: load_inline 需要 ninja/gcc 进行 C++ JIT 编译
CONDA_BIN_PATH = os.path.join(CONDA_ENV_PATH, "bin")
os.environ["PATH"] = CONDA_BIN_PATH + ":" + os.environ.get("PATH", "")

# load_inline 编译缓存目录 (用户可写, 避免默认 /tmp 权限问题)
os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.expanduser("~/.torch_extensions"))
os.makedirs(os.environ["TORCH_EXTENSIONS_DIR"], exist_ok=True)

# 将 XBLAS JIT 缓存重定向到用户可写目录，避免 /tmp 权限问题
_user_cache = os.path.expanduser("~/.xpu_jit_cache")
os.makedirs(_user_cache, exist_ok=True)
# 用 sudo -n (免密 sudo, 无则静默失败) 修复 /tmp 下 xblas 相关目录权限
for tmp_path in ["/tmp/kXPU3", "/tmp/xpujitc-temp", "/tmp/.xblas_cache"]:
    try:
        if os.path.exists(tmp_path):
            subprocess.run(["sudo", "-n", "chmod", "-R", "777", tmp_path],
                           capture_output=True, timeout=10)
        else:
            os.makedirs(tmp_path, exist_ok=True)
            os.chmod(tmp_path, 0o777)
    except Exception:
        pass

try:
    lock_path = "/tmp/.xblas_jit.lock"
    if not os.path.exists(lock_path):
        with open(lock_path, "w") as f:
            f.write("")
    os.chmod(lock_path, 0o777)
except Exception:
    pass

import numpy as np


# ─── GPU 自动选择 ─────────────────────────────────────────────

_SELECTED_GPU_ID: int = 0
_GPU_SELECTION_LOG: str = ""


def _query_gpu_memory() -> list:
    """查询 GPU 显存状态，优先使用 xpu-smi"""
    try:
        import subprocess
        result = subprocess.run(
            ["/usr/local/xpu/bin/xpu-smi"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            gpus = []
            # Parse xpu-smi output format:
            # |   0  P800 OAM           | ... |  51010MiB / 98304MiB |
            lines = result.stdout.split("\n")
            current_idx = None
            for line in lines:
                # Match GPU index line
                import re
                m = re.match(r"\|\s+(\d+)\s+P800", line)
                if m:
                    current_idx = int(m.group(1))
                m2 = re.search(r"(\d+)MiB\s*/\s*(\d+)MiB", line)
                if m2 and current_idx is not None:
                    used_mb = int(m2.group(1))
                    total_mb = int(m2.group(2))
                    gpus.append({
                        "index": current_idx,
                        "name": "P800 OAM",
                        "free_mb": total_mb - used_mb,
                        "total_mb": total_mb,
                    })
                    current_idx = None
            if gpus:
                return gpus
    except Exception:
        pass

    # 回退: torch.cuda
    try:
        import torch
        if torch.cuda.is_available():
            gpus = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                free, total = torch.cuda.mem_get_info(i)
                gpus.append({
                    "index": i,
                    "name": props.name,
                    "free_mb": free // (1024 * 1024),
                    "total_mb": total // (1024 * 1024),
                })
            return gpus
    except Exception:
        pass
    return []


def _auto_select_gpu():
    gpus = _query_gpu_memory()
    if not gpus:
        return 0, "无 GPU 信息, 使用默认 device=0"
    best = max(gpus, key=lambda g: g["free_mb"])
    lines = [f"共 {len(gpus)} 张昆仑芯 P800, 自动选择余量最大的:"]
    for g in gpus:
        marker = " ← 选中" if g["index"] == best["index"] else ""
        free_gb = g["free_mb"] / 1024
        total_gb = g["total_mb"] / 1024
        lines.append(f"  XPU {g['index']} ({g['name']}): {free_gb:.1f}/{total_gb:.1f} GB 空闲{marker}")
    return best["index"], "\n".join(lines)


_SELECTED_GPU_ID, _GPU_SELECTION_LOG = _auto_select_gpu()

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_SELECTED_GPU_ID)
    _GPU_SELECTION_LOG += f"\n  已设置 CUDA_VISIBLE_DEVICES={_SELECTED_GPU_ID}"


def _ensure_gpu_device():
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            torch.cuda.set_device(0)  # After CUDA_VISIBLE_DEVICES, device 0 is our selected GPU
    except Exception:
        pass


def _load_module(code: str):
    import tempfile
    import importlib.util
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        spec = importlib.util.spec_from_file_location("_kernel_module", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "kernel") or not callable(module.kernel):
            raise RuntimeError("Missing 'kernel' function")
        return module
    finally:
        os.unlink(path)


def _detect_language(code: str) -> str:
    if "import triton" in code or "@triton.jit" in code:
        return "triton"  # Not supported on Kunlunxin, will error
    # C++ Extension: Python 文件内嵌 C++ 源码, 通过 load_inline JIT 编译
    if "load_inline" in code and ("_CPP_SOURCE" in code or "cpp_sources" in code):
        return "cuda_c"
    if "import torch" in code:
        return "torch"
    return "numpy"


def _deserialize_inputs(inputs: dict) -> dict:
    if inputs.get("_generate"):
        out = {}
        for k, spec in inputs["specs"].items():
            rng = np.random.default_rng(spec["seed"])
            shape = spec["shape"]
            dtype = spec["dtype"]
            # 先生成 float32 再转换，兼容旧版 numpy (standard_normal 不支持 dtype 参数)
            data = (rng.standard_normal(shape) * 0.1).astype(dtype)
            out[k] = data
        return out
    out = {}
    for k, v in inputs.items():
        arr = np.array(v["data"], dtype=v["dtype"]).reshape(v["shape"])
        out[k] = arr
    return out


def _compute_metrics(inputs: dict, elapsed: float) -> tuple:
    """Compute GFLOPS and bandwidth for common kernels."""
    # Find matrix shapes
    a = inputs.get("A")
    b = inputs.get("B")
    if a is not None and b is not None:
        M = a.shape[0] if a.ndim >= 2 else a.shape[0]
        K = a.shape[-1]
        N = b.shape[-1] if b.ndim >= 2 else 1
        flops = 2 * M * N * K
        nbytes = (a.size + b.size + M * N) * np.dtype(a.dtype).itemsize
        return flops / elapsed / 1e9, nbytes / elapsed / 1e9
    x = inputs.get("X")
    if x is not None:
        nbytes = x.size * np.dtype(x.dtype).itemsize * 3  # read x, write out
        return 0.0, nbytes / elapsed / 1e9
    return 0.0, 0.0


def _load_reference_from_source(source: str, inputs: dict):
    """Compile a traveler reference (`def reference(**inputs)`). See run_kernel.py."""
    if not source or not str(source).strip():
        return None, None
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    values = {}
    for k, v in inputs.items():
        if isinstance(v, np.ndarray):
            values[k] = torch.from_numpy(np.asarray(v)).to(device)
        else:
            values[k] = v
    ns = {}
    try:
        exec(source, ns, ns)  # noqa: S102
    except Exception as e:
        return None, f"reference_source_error:{type(e).__name__}"
    fn = ns.get("reference")
    if not callable(fn):
        return None, "reference_source_error:missing_reference"

    def _call(fn=fn, values=values):
        return fn(**values)

    return _call, "reference_source"


def _reference_torch(kernel_type: str, inputs: dict):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    values = {k: torch.from_numpy(np.asarray(v)).to(device) for k, v in inputs.items()}
    kind = (kernel_type or "").lower()
    if ("matmul" in kind or "gemm" in kind) and {"A", "B"} <= values.keys():
        return lambda: values["A"] @ values["B"], "pytorch_eager"
    if "softmax" in kind and "X" in values:
        import torch.nn.functional as F
        return lambda: F.softmax(values["X"], dim=-1), "pytorch_eager"
    if "relu" in kind and "X" in values:
        return lambda: torch.relu(values["X"]), "pytorch_eager"
    if "gelu" in kind and "X" in values:
        import torch.nn.functional as F
        return lambda: F.gelu(values["X"]), "pytorch_eager"
    return None, None


def _precision_refusal(precision: str):
    """声明精度本平台能不能如实兑现: 能则返回 None, 不能则返回拒绝原因

    拒绝而不是「收下再忽略」: 控制端按 `_REMOTE_L2_METRIC_KEYS` 那样的闭合名单搬运
    响应, 新加的留痕字段到不了任何产物里, `error` 是唯一保证被下游看到的通道。
    """
    p = (precision or "").lower()
    if p in ("", "fp32", "float32"):
        return None
    return (f"precision={p} 在昆仑芯 runner 上无法如实兑现: 本文件的数值门只有绝对项 "
            f"(max_abs_error <= max_error), 没有 NVIDIA 那边的相对项与 matched_ratio; "
            f"按声明精度喂 fp16 会误杀合法核, 不喂又等于拿 fp32 的成绩充当 fp16")


def _reference_refusal(reference_contract):
    if reference_contract is not None:
        reason = "CUDA native LayerNorm reference identity cannot be established on XPU"
        return {"success": False, "error": reason, "reference_error": reason,
                "reference_identity_unknown": True}
    return None


def run_gpu_info(reference_contract=None) -> dict:
    """Return GPU hardware info for the selected device."""
    refusal = _reference_refusal(reference_contract)
    if refusal:
        return refusal
    try:
        import torch
        if not torch.cuda.is_available():
            return {"success": False, "error": "XPU/CUDA not available"}
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        return {
            "success": True,
            "vendor": "kunlunxin",
            "name": "P800 OAM",
            "device_name": "P800 OAM",
            "compute_capability": "kl3",
            "multiprocessor_count": props.multi_processor_count,
            "total_memory_gb": round(total / 1024**3, 1),
            "free_memory_gb": round(free / 1024**3, 1),
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
            "selection_log": _GPU_SELECTION_LOG,
        }
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {e}"}


def run_compile(code: str) -> dict:
    lang = _detect_language(code)
    if lang == "triton":
        return {"success": False, "error": "Triton not supported on Kunlunxin P800"}
    try:
        module = _load_module(code)
        return {"success": True, "language": lang}
    except Exception as e:
        err_msg = f"Compile error: {type(e).__name__}: {e}"
        # cuda_c (load_inline) 编译错误可能极长 (gcc/ninja 输出), 截断到 2000 字符
        if lang == "cuda_c" and len(err_msg) > 2000:
            err_msg = err_msg[:1000] + "\n... [gcc 编译输出截断] ...\n" + err_msg[-1000:]
        return {"success": False, "error": err_msg}


def run_execute(code: str, inputs: dict) -> dict:
    lang = _detect_language(code)
    if lang == "triton":
        return {"success": False, "error": "Triton not supported on Kunlunxin P800"}
    real_inputs = _deserialize_inputs(inputs)
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _ensure_gpu_device()
        module = _load_module(code)
        torch_inputs = {k: torch.from_numpy(v).to(device).contiguous() for k, v in real_inputs.items()}
        # warmup
        for _ in range(3):
            _ = module.kernel(**torch_inputs)
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = module.kernel(**torch_inputs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if isinstance(output, torch.Tensor):
            out_arr = output.detach().cpu().float().numpy()
        else:
            out_arr = np.asarray(output, dtype=np.float32)
        gflops, bandwidth = _compute_metrics(real_inputs, elapsed)
        sample = out_arr.flatten()[:10].tolist()
        return {
            "success": True,
            "language": lang,
            "output_shape": list(out_arr.shape),
            "output_sample": sample,
            "execution_time_ms": round(elapsed * 1000, 4),
            "gflops": round(gflops, 4),
            "memory_bandwidth_gbs": round(bandwidth, 4),
            "device": "kunlunxin_xpu",
        }
    except Exception as e:
        import traceback
        return {"success": False, "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[:2000]}


def run_verify(code: str, inputs: dict, kernel_type: str, max_error: float,
               precision: str = "", reference_source: str = "", reference_contract=None) -> dict:
    refusal = _reference_refusal(reference_contract)
    if refusal:
        return refusal
    lang = _detect_language(code)
    if lang == "triton":
        return {"success": False, "error": "Triton not supported on Kunlunxin P800"}
    refusal = _precision_refusal(precision)
    if refusal:
        return {"success": False, "error": refusal}
    try:
        import torch
        real_inputs = _deserialize_inputs(inputs)
        reference, reference_type = _load_reference_from_source(reference_source, real_inputs)
        if reference is None:
            reference, reference_type = _reference_torch(kernel_type, real_inputs)
        if reference is None:
            return {"success": False, "error": f"No remote reference for {kernel_type}"}
        module = _load_module(code)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _ensure_gpu_device()
        candidate_inputs = {k: torch.from_numpy(v).to(device).contiguous()
                            for k, v in real_inputs.items()}
        with torch.no_grad():
            actual = module.kernel(**candidate_inputs)
            expected = reference()
        if not isinstance(actual, torch.Tensor):
            actual = torch.as_tensor(actual, device=device)
        if list(actual.shape) != list(expected.shape):
            return {"success": False, "error": "output shape mismatch",
                    "output_shape": list(actual.shape), "expected_shape": list(expected.shape)}
        actual_f = actual.detach().to(torch.float32)
        expected_f = expected.detach().to(torch.float32)
        abs_error = (actual_f - expected_f).abs()
        max_abs_error = float(abs_error.max().item()) if abs_error.numel() else 0.0
        denom = expected_f.abs().clamp_min(1e-12)
        max_rel_error = float((abs_error / denom).max().item()) if abs_error.numel() else 0.0
        return {
            "success": max_abs_error <= max_error,
            "reference_type": reference_type,
            "output_shape": list(actual.shape),
            "compared_elements": int(actual.numel()),
            "coverage": 1.0,
            "max_error": max_abs_error,
            "max_rel_error": max_rel_error,
            "error": "" if max_abs_error <= max_error else f"max abs error {max_abs_error:.2e}",
            "device": "kunlunxin_xpu",
        }
    except Exception as e:
        import traceback
        return {"success": False, "error": f"Remote verification error: {type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[:2000]}


def _bench_callable(fn, warmup: int, repeat: int, device_is_cuda: bool) -> dict:
    import torch
    # Fix permission issues for xblas JIT
    for _ in range(warmup):
        fn()
    if device_is_cuda:
        torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        if device_is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if device_is_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return {"min_time_ms": round(times[0], 4),
            "median_time_ms": round(times[len(times)//2], 4),
            "avg_time_ms": round(sum(times) / len(times), 4),
            "max_time_ms": round(times[-1], 4),
            "runs": repeat}


def run_benchmark(code: str, inputs: dict, warmup: int = 3, repeat: int = 10,
                  reference_contract=None) -> dict:
    """只测候选, 不测基线 —— 与 run_kernel.py 的同名 action 同口径。

    控制端这条 payload 不带 kernel_type, 所以本函数也不能需要它: 拿它去建基线的话,
    收到空 kernel_type 就只剩 "No remote reference for " 这一种结果。
    """
    refusal = _reference_refusal(reference_contract)
    if refusal:
        return refusal
    lang = _detect_language(code)
    if lang == "triton":
        return {"success": False, "error": "Triton not supported on Kunlunxin P800"}
    try:
        import torch
        real_inputs = _deserialize_inputs(inputs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _ensure_gpu_device()
        module = _load_module(code)
        torch_inputs = {k: torch.from_numpy(v).to(device).contiguous()
                        for k, v in real_inputs.items()}
        stats = _bench_callable(lambda: module.kernel(**torch_inputs), warmup, repeat,
                                device.type == "cuda")
        gflops, bandwidth = _compute_metrics(real_inputs, stats["median_time_ms"] / 1000)
        return {"success": True, "language": lang, "device": "kunlunxin_xpu", **stats,
                "gflops": round(gflops, 4), "memory_bandwidth_gbs": round(bandwidth, 4)}
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {e}"}


def run_benchmark_with_baselines(code: str, inputs: dict, kernel_type: str,
                                 warmup: int = 10, repeat: int = 30,
                                 precision: str = "", baseline_priority=None,
                                 reference_source: str = "", reference_contract=None) -> dict:
    """候选 vs **一个**基线, 基线由控制端的有序偏好决定。

    控制端只排序、远端取第一个能跑的并回报 `baseline_selected` (verifier.baseline_plan)。
    """
    refusal = _reference_refusal(reference_contract)
    if refusal:
        return refusal
    lang = _detect_language(code)
    if lang == "triton":
        return {"success": False, "error": "Triton not supported on Kunlunxin P800"}
    refusal = _precision_refusal(precision)
    if refusal:
        return {"success": False, "error": refusal}
    try:
        import torch
        real_inputs = _deserialize_inputs(inputs)
        reference, reference_type = _load_reference_from_source(reference_source, real_inputs)
        if reference is None:
            reference, reference_type = _reference_torch(kernel_type, real_inputs)
        if reference is None:
            return {"success": False, "error": f"No remote reference for {kernel_type}"}
        module = _load_module(code)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _ensure_gpu_device()
        candidate_inputs = {k: torch.from_numpy(v).to(device).contiguous()
                            for k, v in real_inputs.items()}
        candidate = lambda: module.kernel(**candidate_inputs)
        is_cuda = device.type == "cuda"

        # 只登记本平台真能建出来的基线。偏好名单里的 pytorch_eager_fp16 / pytorch_eager_tf32
        # 指的是 cuBLAS FP16 与 TF32 数据通路, 昆仑芯上没有对应物 —— 拿 xblas FP32 冒充
        # TF32 基线比不测更糟, 所以显式过滤掉。被过滤的名字记进 baseline_attempts 而不是
        # 单开一个字段: 控制端只搬运它认识的那几个键, 单开的字段到不了产物里。
        specs = {"pytorch_eager": lambda: reference}
        if hasattr(torch, "compile"):
            specs["torch_compile"] = lambda: torch.compile(reference)
        requested = list(baseline_priority or [])
        order = [n for n in requested if n in specs]
        attempts = [f"{n}: skipped: 昆仑芯 runner 建不出这个基线"
                    for n in requested if n not in specs]
        order += [n for n in specs if n not in order]

        chosen = None
        for name in order:
            try:
                fn = specs[name]()
                fn()                      # 一次实跑探可用性 (编译 / 显存 / 数值)
                if is_cuda:
                    torch.cuda.synchronize()
                chosen = (name, fn)
                break
            except Exception as e:
                attempts.append(f"{name}: {type(e).__name__}: {str(e)[:160]}")
        if chosen is None:
            return {"success": False, "baseline_attempts": attempts,
                    "error": f"No usable remote baseline for {kernel_type}; "
                             f"attempts={attempts}"}
        base_name, base_fn = chosen

        candidate_stats = _bench_callable(candidate, warmup, repeat, is_cuda)
        base_stats = _bench_callable(base_fn, warmup, repeat, is_cuda)
        elapsed = candidate_stats["median_time_ms"] / 1000
        gflops, bandwidth = _compute_metrics(real_inputs, elapsed)
        return {"success": True, "device": "kunlunxin_xpu", "candidate": candidate_stats,
                "baselines": {base_name: base_stats},
                "baseline_selected": base_name,
                "baseline_attempts": attempts,
                "reference_type": reference_type,
                "gflops": round(gflops, 4), "memory_bandwidth_gbs": round(bandwidth, 4)}
    except Exception as e:
        import traceback
        return {"success": False, "error": f"Remote benchmark error: {type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[:2000]}


# ─── Main entry point ─────────────────────────────────────────

def _without_seed_inputs(inputs_spec, run):
    """Unsupported seed transport must request concrete inputs before execution.

    This runner has no seed materializer or digest verifier. Returning the same
    explicit fallback signal as the NVIDIA runner prevents an unexpected seed
    request from silently executing with the default empty ``inputs`` mapping.
    Ordinary concrete-input requests retain their existing path.
    """
    if inputs_spec:
        return {"success": False, "input_seed_mismatch": True,
                "error": "XPU runner cannot reconstruct seed inputs; resend concrete inputs"}
    return run()


def main():
    """支持两种调用方式:
    1. python run_kernel_xpu.py <action> '<json_args>'
    2. echo '<json_payload>' | python run_kernel_xpu.py  (stdin 模式, payload 包含 action 字段)
    """
    args = {}
    action = None
    
    if len(sys.argv) > 1:
        action = sys.argv[1]
        if len(sys.argv) > 2:
            args = json.loads(sys.argv[2])
    else:
        # 从 stdin 读取完整 payload (包含 action 字段)
        raw = sys.stdin.read()
        payload = json.loads(raw)
        action = payload.get("action")
        args = payload
    
    if not action:
        print(json.dumps({"error": "Usage: run_kernel_xpu.py <action> [json_args] OR pipe JSON via stdin"}))
        sys.exit(1)
    
    dispatch = {
        "gpu_info": lambda: run_gpu_info(args.get("reference_contract")),
        "compile": lambda: run_compile(args.get("code", "")),
        "execute": lambda: run_execute(args.get("code", ""), args.get("inputs", {})),
        "verify": lambda: _without_seed_inputs(args.get("inputs_spec"), lambda: run_verify(
                                     args.get("code", ""), args.get("inputs", {}),
                                     args.get("kernel_type", ""),
                                     args.get("max_error", 1e-3),
                                     precision=args.get("precision", ""),
                                     reference_source=args.get("reference_source", ""),
                                     reference_contract=args.get("reference_contract"))),
        "benchmark": lambda: _without_seed_inputs(args.get("inputs_spec"), lambda: run_benchmark(
                                           args.get("code", ""), args.get("inputs", {}),
                                           warmup=args.get("warmup", 3),
                                           repeat=args.get("repeat", 10),
                                           reference_contract=args.get("reference_contract"))),
        "benchmark_with_baselines": lambda: _without_seed_inputs(
            args.get("inputs_spec"), lambda: run_benchmark_with_baselines(
            args.get("code", ""), args.get("inputs", {}),
            args.get("kernel_type", ""),
            warmup=args.get("warmup", 10), repeat=args.get("repeat", 30),
            precision=args.get("precision", ""),
            baseline_priority=args.get("baseline_priority") or None,
            reference_source=args.get("reference_source", ""),
            reference_contract=args.get("reference_contract"))),
    }
    fn = dispatch.get(action)
    if fn is None:
        print(json.dumps({"error": f"Unknown action: {action}"}))
        sys.exit(1)
    try:
        result = fn()
    except Exception as e:
        import traceback
        result = {"success": False, "error": f"{type(e).__name__}: {e}",
                  "traceback": traceback.format_exc()[:2000]}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
