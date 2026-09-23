"""Remote kernel runner - 部署在远程 GPU 服务器上执行 (当前: algorithm-h20, 8×H20)
Supports both Python (numpy/torch) and Triton kernel execution.

Actions:
  gpu_info  - query GPU info
  compile   - L1 compile check
  execute   - L2 functional run (single iteration)
  benchmark - L3 performance measurement (multiple iterations)

环境变量:
  OPERATOR_AGENT_GPU - 固定使用的 GPU 卡号; 未设置时按 (低利用率, 高显存余量) 自动选
"""

import sys
import json
import time
import os
import numpy as np


# ─── GPU 自动选择 ─────────────────────────────────────────────
# 在模块加载时完成，后续所有 CUDA 操作自动使用选中的设备

_SELECTED_GPU_ID: int = 0
_GPU_SELECTION_LOG: str = ""


def _query_gpu_memory() -> list[dict]:
    """查询所有 GPU 的显存与利用率。优先 nvidia-smi，回退 torch.cuda。"""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.free,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    util = None
                    if len(parts) >= 5:
                        try:
                            util = int(parts[4])
                        except ValueError:
                            util = None
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "free_mb": int(parts[2]),
                        "total_mb": int(parts[3]),
                        "util_pct": util,
                    })
            if gpus:
                return gpus
    except Exception:
        pass

    # 回退: torch.cuda (拿不到利用率)
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
                    "util_pct": None,
                })
            return gpus
    except Exception:
        pass

    return []


def _fmt_gpu(g: dict) -> str:
    util = g.get("util_pct")
    util_s = f"{util}%" if util is not None else "util=?"
    return (f"GPU {g['index']} ({g['name']}): "
            f"{g['free_mb'] / 1024:.1f}/{g['total_mb'] / 1024:.1f} GB 空闲, {util_s}")


def _sample_gpu_idle():
    """采样选中 GPU 的测量前状态: (利用率%, 其他进程数)。拿不到返回 (None, None)。

    利用率统计的是"有 kernel 在跑的时间占比", 我们自己的 benchmark 会把它顶到
    100%, 所以只能在测量前采样。其他计算进程数比利用率更可靠地反映是否被共享,
    故一并返回 (排除本进程自己)。
    """
    util, others = None, None
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", f"--id={_SELECTED_GPU_ID}",
             "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            util = int(r.stdout.strip().splitlines()[0])
        r2 = subprocess.run(
            ["nvidia-smi", f"--id={_SELECTED_GPU_ID}",
             "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if r2.returncode == 0:
            pids = set()
            for line in r2.stdout.strip().splitlines():
                line = line.strip()
                if line.isdigit():
                    pids.add(int(line))
            pids.discard(os.getpid())
            others = len(pids)
    except Exception:
        pass
    return util, others


def _auto_select_gpu() -> tuple[int, str]:
    """选择用于执行/基准测试的 GPU。返回 (device_id, log_message)。

    可通过 OPERATOR_AGENT_GPU 固定卡号 —— 共享机器上必须固定, 否则同一次运行的
    L1/L2/L3 会各自重选、落在不同卡上, 加速比不可比。
    """
    gpus = _query_gpu_memory()
    pinned = os.environ.get("OPERATOR_AGENT_GPU", "").strip()
    if pinned:
        try:
            pid = int(pinned)
        except ValueError:
            pid = None
        if pid is not None:
            g = next((x for x in gpus if x["index"] == pid), None)
            if g:
                return pid, f"使用固定 GPU (OPERATOR_AGENT_GPU={pid}): {_fmt_gpu(g)}"
            return pid, f"使用固定 GPU (OPERATOR_AGENT_GPU={pid}), 但 nvidia-smi 未列出该卡"

    if not gpus:
        return 0, "无 GPU 信息, 使用默认 device=0"

    # 以利用率为主、显存余量为辅: 一张卡可能余量很大却正被别人跑满,
    # 在那上面测出来的加速比是噪声。利用率未知时按最坏 (100%) 处理。
    def score(g):
        util = g.get("util_pct")
        return (-(100 if util is None else util), g["free_mb"])

    best = max(gpus, key=score)
    lines = [f"共 {len(gpus)} 张 GPU, 按 (低利用率, 高显存余量) 选择:"]
    for g in gpus:
        marker = " ← 选中" if g["index"] == best["index"] else ""
        lines.append(f"  {_fmt_gpu(g)}{marker}")
    return best["index"], "\n".join(lines)


_SELECTED_GPU_ID, _GPU_SELECTION_LOG = _auto_select_gpu()

# 尽早限制可见设备, 后续 import torch 自动只看到这张卡
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_SELECTED_GPU_ID)
    _GPU_SELECTION_LOG += f"\n  已设置 CUDA_VISIBLE_DEVICES={_SELECTED_GPU_ID}"
else:
    _GPU_SELECTION_LOG += (
        f"\n  CUDA_VISIBLE_DEVICES 已设置 ({os.environ['CUDA_VISIBLE_DEVICES']}), "
        f"选中 index={_SELECTED_GPU_ID} (将在运行时 set_device)"
    )


def _ensure_gpu_device():
    """确保 torch 使用自动选择的 GPU。在 import torch 之后调用。"""
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            # 当 CUDA_VISIBLE_DEVICES 未被我们设置时 (用户已预设),
            # _SELECTED_GPU_ID 是可见设备范围内的索引, 需显式 set_device
            if "CUDA_VISIBLE_DEVICES" in os.environ and _SELECTED_GPU_ID != 0:
                pass  # 我们已设置过 env var, torch 只看到一张卡
            torch.cuda.set_device(_SELECTED_GPU_ID)
    except Exception:
        pass


def _load_module(code: str):
    """Load generated code from an isolated temporary module."""
    import tempfile
    import importlib.util
    import os

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
        return "triton"
    if "import torch" in code:
        return "torch"
    return "numpy"


def run_compile(code: str) -> dict:
    lang = _detect_language(code)
    try:
        if lang == "triton":
            import triton
            import triton.language as tl
            import torch
        
        import tempfile
        import importlib.util
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            spec = importlib.util.spec_from_file_location("_kernel_module", temp_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if not hasattr(module, "kernel") or not callable(module.kernel):
                return {"success": False, "error": "Missing 'kernel' function"}
            return {"success": True, "language": lang}
        finally:
            os.unlink(temp_path)
    except ImportError as e:
        return {"success": False, "error": f"Import error ({lang}): {e}"}
    except Exception as e:
        return {"success": False, "error": f"Compile error: {e}"}


def _deserialize_inputs(inputs: dict) -> dict:
    """反序列化输入。支持两种格式:
    - 轻量模式: {"_generate": true, "specs": {"A": {"shape": [...], "dtype": "...", "seed": N}}}
    - 完整模式 (legacy): {"A": {"shape": [...], "dtype": "...", "data": [...]}}
    - 标量/布尔直通 (如 causal=False): 不具有 "shape"/"data" 键者原样透传
    """
    if inputs.get("_generate"):
        out = {}
        for k, spec in inputs["specs"].items():
            rng = np.random.default_rng(spec["seed"])
            shape = spec["shape"]
            dtype = spec["dtype"]
            # 生成与 controller 端一致的随机数据 (正态分布 × 0.1)
            out[k] = (rng.standard_normal(shape, dtype=np.float32) * 0.1).astype(dtype)
        return out

    # Legacy: 完整 JSON 数据 + 标量透传
    out = {}
    for k, v in inputs.items():
        if not isinstance(v, dict) or "shape" not in v or "data" not in v:
            out[k] = v
            continue
        try:
            arr = np.array(v["data"], dtype=v["dtype"]).reshape(v["shape"])
            out[k] = arr
        except Exception:
            out[k] = v
    return out


def run_execute(code: str, inputs: dict) -> dict:
    lang = _detect_language(code)
    real_inputs = _deserialize_inputs(inputs)

    if lang == "triton":
        return _execute_triton(code, real_inputs)
    return _execute_torch_numpy(code, real_inputs, lang)


def _reference_torch(kernel_type: str, inputs: dict):
    """Build the reference on the remote GPU, never on the controller host."""
    import torch
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    values = {k: torch.from_numpy(np.asarray(v)).to(device) for k, v in inputs.items()}
    kind = (kernel_type or "").lower()
    if ("matmul" in kind or "gemm" in kind) and {"A", "B"} <= values.keys():
        return lambda: values["A"] @ values["B"], "pytorch_eager"
    if "softmax" in kind and "X" in values:
        return lambda: F.softmax(values["X"], dim=-1), "pytorch_eager"
    if "layernorm" in kind or "layer_norm" in kind:
        if "X" in values:
            x_val = values["X"]
            if "weight" in values and "bias" in values:
                return lambda: F.layer_norm(x_val, x_val.shape[-1:],
                                            values["weight"], values["bias"]), "pytorch_eager"
            return lambda: F.layer_norm(x_val, x_val.shape[-1:]), "pytorch_eager"
    if "relu" in kind and "X" in values:
        return lambda: torch.relu(values["X"]), "pytorch_eager"
    if "gelu" in kind and "X" in values:
        return lambda: F.gelu(values["X"]), "pytorch_eager"
    # Attention: inputs Q, K, V (shapes: B,H,S,D or S,D), optional causal=True, sm_scale
    if ("attention" in kind or "flash" in kind or "attn" in kind) and {"Q", "K", "V"} <= values.keys():
        _q = values["Q"]
        _k = values["K"]
        _v = values["V"]
        _causal = bool(values.get("causal", False))
        # sm_scale: default 1/sqrt(head_dim)
        head_dim = _q.shape[-1]
        _sm_scale = values.get("sm_scale", None)
        if _sm_scale is None:
            _sm_scale = 1.0 / (head_dim ** 0.5)
        _sm_scale = float(_sm_scale)
        # Prefer F.scaled_dot_product_attention when available (PyTorch 2.x)
        if hasattr(F, "scaled_dot_product_attention"):
            # SDPA expects Q,K,V shapes: (B,H,S,D) or (B,S,H,D); we handle (B,H,S,D) or (S,D)
            if _q.ndim == 4:
                # (B, H, S, D) → SDPA expects (B, H, S, D) which works
                _ref_q, _ref_k, _ref_v = _q, _k, _v
            elif _q.ndim == 3:
                # (H, S, D) → add batch dim
                _ref_q = _q.unsqueeze(0)
                _ref_k = _k.unsqueeze(0)
                _ref_v = _v.unsqueeze(0)
            elif _q.ndim == 2:
                # (S, D) → single head, single batch
                _ref_q = _q.unsqueeze(0).unsqueeze(0)
                _ref_k = _k.unsqueeze(0).unsqueeze(0)
                _ref_v = _v.unsqueeze(0).unsqueeze(0)
            else:
                _ref_q, _ref_k, _ref_v = _q, _k, _v
            def _attn_ref(_rq=_ref_q, _rk=_ref_k, _rv=_ref_v, _c=_causal, _s=_sm_scale):
                out = F.scaled_dot_product_attention(_rq, _rk, _rv, is_causal=_c, scale=_s)
                # Squeeze back if we added dims
                if _q.ndim == 3 and out.ndim == 4:
                    out = out.squeeze(0)
                elif _q.ndim == 2 and out.ndim == 4:
                    out = out.squeeze(0).squeeze(0)
                return out
            return _attn_ref, "pytorch_sdpa"
        else:
            # Fallback: manual (Q@K^T) softmax @V (SLOW! but correct)
            def _attn_manual(_mq=_q, _mk=_k, _mv=_v, _c=_causal, _s=_sm_scale):
                # assume last dim = head_dim
                if _mq.ndim == 4:
                    # (B, H, S, D) @ (B, H, D, S) -> (B, H, S, S)
                    scores = torch.matmul(_mq, _mk.transpose(-2, -1)) * _s
                    if _c:
                        sz = scores.shape[-1]
                        mask = torch.triu(torch.ones(sz, sz, device=scores.device, dtype=torch.bool), diagonal=1)
                        scores = scores.masked_fill(mask, float("-inf"))
                    attn = F.softmax(scores, dim=-1)
                    return torch.matmul(attn, _mv)
                elif _mq.ndim == 3:
                    scores = torch.matmul(_mq, _mk.transpose(-2, -1)) * _s
                    if _c:
                        sz = scores.shape[-1]
                        mask = torch.triu(torch.ones(sz, sz, device=scores.device, dtype=torch.bool), diagonal=1)
                        scores = scores.masked_fill(mask, float("-inf"))
                    attn = F.softmax(scores, dim=-1)
                    return torch.matmul(attn, _mv)
                else:
                    scores = _mq @ _mk.T * _s
                    if _c:
                        sz = scores.shape[-1]
                        mask = torch.triu(torch.ones(sz, sz, device=scores.device, dtype=torch.bool), diagonal=1)
                        scores = scores.masked_fill(mask, float("-inf"))
                    attn = F.softmax(scores, dim=-1)
                    return attn @ _mv
            return _attn_manual, "pytorch_manual"
    return None, None


def run_verify(code: str, inputs: dict, kernel_type: str, max_error: float,
              precision: str = "") -> dict:
    """Run candidate and reference on one device; return only verification summary."""
    try:
        import torch
        real_inputs = _deserialize_inputs(inputs)
        reference, reference_type = _reference_torch(kernel_type, real_inputs)
        if reference is None:
            return {"success": False, "error": f"No remote reference for {kernel_type}"}

        module = _load_module(code)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        candidate_inputs = {}
        for k, v in real_inputs.items():
            if isinstance(v, np.ndarray):
                candidate_inputs[k] = torch.from_numpy(v).to(device).contiguous()
            else:
                # 标量 / 布尔等非数组参数 (如 causal=False) 原样透传
                candidate_inputs[k] = v
        # Convert candidate inputs to target precision when needed
        if (precision or "").lower() == "fp16":
            candidate_inputs = {
                k: (v.to(torch.float16) if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
                for k, v in candidate_inputs.items()
            }
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
            "device": str(device),
        }
    except Exception as e:
        return {"success": False, "error": f"Remote verification error: {type(e).__name__}: {e}"}


def _execute_triton(code: str, real_inputs: dict) -> dict:
    try:
        import triton
        import triton.language as tl
        import torch
        import tempfile
        import importlib.util
        import os

        if not torch.cuda.is_available():
            return {"success": False, "error": "CUDA not available, cannot run Triton kernel"}

        device = torch.device("cuda")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            spec = importlib.util.spec_from_file_location("_kernel_module", temp_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            kernel_func = module.kernel

            torch_inputs = {k: torch.from_numpy(v).to(device).contiguous() for k, v in real_inputs.items()}

            for _ in range(3):
                _ = kernel_func(**torch_inputs)
            torch.cuda.synchronize()

            start = time.perf_counter()
            output = kernel_func(**torch_inputs)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            if isinstance(output, torch.Tensor):
                out_arr = output.cpu().numpy()
            else:
                out_arr = np.asarray(output)

            gflops, bandwidth = _compute_metrics(real_inputs, elapsed)

            return {
                "success": True,
                "language": "triton",
                "output_shape": list(out_arr.shape),
                "output_sample": out_arr.flatten().tolist(),
                "execution_time_ms": round(elapsed * 1000, 4),
                "gflops": round(gflops, 4),
                "memory_bandwidth_gbs": round(bandwidth, 4),
                "device": str(device),
            }
        finally:
            os.unlink(temp_path)
    except Exception as e:
        return {"success": False, "error": f"Triton error: {type(e).__name__}: {e}"}


def _execute_torch_numpy(code: str, real_inputs: dict, lang: str) -> dict:
    try:
        if lang == "torch":
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch_inputs = {k: torch.from_numpy(v).to(device) for k, v in real_inputs.items()}
            ns = {"__name__": "__kernel__"}
            exec(code, ns)
            kernel_func = ns["kernel"]

            for _ in range(3):
                _ = kernel_func(**torch_inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter()
            output = kernel_func(**torch_inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            if isinstance(output, torch.Tensor):
                out_arr = output.cpu().numpy()
            else:
                out_arr = np.asarray(output)
        else:
            ns = {"__name__": "__kernel__"}
            exec(code, ns)
            kernel_func = ns["kernel"]
            start = time.perf_counter()
            output = kernel_func(**real_inputs)
            elapsed = time.perf_counter() - start
            out_arr = np.asarray(output)

        gflops, bandwidth = _compute_metrics(real_inputs, elapsed)

        return {
            "success": True,
            "language": lang,
            "output_shape": list(out_arr.shape),
            "output_sample": out_arr.flatten().tolist(),
            "execution_time_ms": round(elapsed * 1000, 4),
            "gflops": round(gflops, 4),
            "memory_bandwidth_gbs": round(bandwidth, 4),
            "device": str(device) if lang == "torch" else "cpu",
        }
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {e}"}


def run_benchmark(code: str, inputs: dict, warmup: int = 3, repeat: int = 10) -> dict:
    lang = _detect_language(code)
    real_inputs = _deserialize_inputs(inputs)

    if lang == "triton":
        return _benchmark_triton(code, real_inputs, warmup, repeat)
    return _benchmark_torch_numpy(code, real_inputs, warmup, repeat, lang)


def _timed_runs(fn, repeat: int, device_is_cuda: bool) -> list[float]:
    """重复执行 fn 并返回每轮耗时 (毫秒)。

    GPU 上使用 CUDA event 计时, 避免 CPU 端 time.perf_counter 引入的
    launch 调度与同步噪声, 提升 L3 性能验证精度 (对 autotune 选优更可靠)。
    """
    import torch
    times = []
    for _ in range(repeat):
        if device_is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))  # 毫秒
        else:
            start = time.perf_counter()
            fn()
            times.append((time.perf_counter() - start) * 1000)
    return times


def _bench_callable(fn, warmup: int, repeat: int, device_is_cuda: bool) -> dict:
    import torch
    for _ in range(warmup):
        fn()
    if device_is_cuda:
        torch.cuda.synchronize()
    times = _timed_runs(fn, repeat, device_is_cuda)
    times.sort()
    return {"min_time_ms": round(times[0], 4), "median_time_ms": round(times[len(times)//2], 4),
            "avg_time_ms": round(sum(times) / len(times), 4), "max_time_ms": round(times[-1], 4),
            "runs": repeat}


def run_benchmark_with_baselines(code: str, inputs: dict, kernel_type: str,
                                 warmup: int = 10, repeat: int = 30,
                                 precision: str = "") -> dict:
    """Benchmark candidate/eager/compile together on the remote execution device.

    For CUDA + matmul, also measures cuBLAS with TF32 enabled for a fair
    comparison against Triton (tl.dot uses TF32 by default on Ampere+).
    When precision is "fp16", also measures cuBLAS FP16 baseline.
    """
    try:
        import torch
        real_inputs = _deserialize_inputs(inputs)
        reference, reference_type = _reference_torch(kernel_type, real_inputs)
        if reference is None:
            return {"success": False, "error": f"No remote reference for {kernel_type}"}
        module = _load_module(code)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        candidate_inputs = {}
        for k, v in real_inputs.items():
            if isinstance(v, np.ndarray):
                candidate_inputs[k] = torch.from_numpy(v).to(device).contiguous()
            else:
                # 标量 / 布尔等非数组参数 (如 causal=False) 原样透传
                candidate_inputs[k] = v
        # Convert candidate inputs to target precision when needed
        if (precision or "").lower() == "fp16":
            candidate_inputs = {
                k: (v.to(torch.float16) if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
                for k, v in candidate_inputs.items()
            }
        candidate = lambda: module.kernel(**candidate_inputs)
        is_cuda = device.type == "cuda"
        candidate_stats = _bench_callable(candidate, warmup, repeat, is_cuda)
        eager_stats = _bench_callable(reference, warmup, repeat, is_cuda)
        baselines = {"pytorch_eager": eager_stats}

        # ---- FP16 baseline (for FP16 precision tasks) ----
        # When the task requires FP16, measure cuBLAS with FP16 inputs for fair comparison.
        is_matmul = ("matmul" in (kernel_type or "").lower() or
                      "gemm" in (kernel_type or "").lower())
        is_fp16 = (precision or "").lower() == "fp16"
        if is_cuda and is_matmul and is_fp16:
            try:
                # Convert inputs to FP16 for cuBLAS FP16 benchmark
                a_fp16 = candidate_inputs["A"].to(torch.float16)
                b_fp16 = candidate_inputs["B"].to(torch.float16)
                fp16_reference = lambda: a_fp16 @ b_fp16
                # warmup with FP16
                fp16_reference()
                if is_cuda:
                    torch.cuda.synchronize()
                eager_fp16_stats = _bench_callable(fp16_reference, warmup, repeat, is_cuda)
                baselines["pytorch_eager_fp16"] = eager_fp16_stats
            except Exception as e:
                baselines["pytorch_eager_fp16_error"] = str(e)[:300]

        # ---- TF32 baseline (fair comparison for Triton kernels on Ampere+) ----
        # Triton tl.dot() uses TF32 by default on Ampere; cuBLAS FP32 does not.
        # Skip TF32 baseline for FP16 tasks (FP16 baseline is more relevant).
        if is_cuda and is_matmul and hasattr(torch.backends.cuda, "matmul") and not is_fp16:
            try:
                original_tf32 = torch.backends.cuda.matmul.allow_tf32
                torch.backends.cuda.matmul.allow_tf32 = True
                # warmup with TF32
                reference()
                if is_cuda:
                    torch.cuda.synchronize()
                eager_tf32_stats = _bench_callable(reference, warmup, repeat, is_cuda)
                baselines["pytorch_eager_tf32"] = eager_tf32_stats
                torch.backends.cuda.matmul.allow_tf32 = original_tf32
            except Exception as e:
                baselines["pytorch_eager_tf32_error"] = str(e)[:300]
                try:
                    torch.backends.cuda.matmul.allow_tf32 = original_tf32
                except Exception:
                    pass

        if hasattr(torch, "compile"):
            try:
                compiled = torch.compile(reference)
                compiled()  # compilation is deliberately outside the timed region
                if is_cuda:
                    torch.cuda.synchronize()
                baselines["torch_compile"] = _bench_callable(compiled, warmup, repeat, is_cuda)
            except Exception as e:
                baselines["torch_compile_error"] = str(e)[:300]
        elapsed = candidate_stats["median_time_ms"] / 1000
        gflops, bandwidth = _compute_metrics(real_inputs, elapsed)
        return {"success": True, "device": str(device), "candidate": candidate_stats,
                "baselines": baselines, "reference_type": reference_type,
                "gflops": round(gflops, 4), "memory_bandwidth_gbs": round(bandwidth, 4)}
    except Exception as e:
        return {"success": False, "error": f"Remote benchmark error: {type(e).__name__}: {e}"}


def _benchmark_triton(code: str, real_inputs: dict, warmup: int, repeat: int) -> dict:
    import triton
    import triton.language as tl
    import torch
    import tempfile
    import importlib.util
    import os

    if not torch.cuda.is_available():
        return {"success": False, "error": "CUDA not available for Triton benchmark"}

    device = torch.device("cuda")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_path = f.name
    
    try:
        spec = importlib.util.spec_from_file_location("_kernel_module", temp_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        kernel_func = module.kernel
        torch_inputs = {k: torch.from_numpy(v).to(device).contiguous() for k, v in real_inputs.items()}

        for _ in range(warmup):
            _ = kernel_func(**torch_inputs)
        torch.cuda.synchronize()

        # CUDA event 计时, 提升精度 (毫秒 -> 秒)
        times = [t / 1000 for t in _timed_runs(lambda: kernel_func(**torch_inputs), repeat, True)]

        avg_time = sum(times) / len(times)
        gflops, bandwidth = _compute_metrics(real_inputs, avg_time)

        return {
            "success": True,
            "language": "triton",
            "avg_time_ms": round(avg_time * 1000, 4),
            "min_time_ms": round(min(times) * 1000, 4),
            "max_time_ms": round(max(times) * 1000, 4),
            "gflops": round(gflops, 4),
            "memory_bandwidth_gbs": round(bandwidth, 4),
            "runs": repeat,
            "device": str(device),
        }
    except Exception as e:
        return {"success": False, "error": f"Triton benchmark error: {type(e).__name__}: {e}"}
    finally:
        os.unlink(temp_path)


def _benchmark_torch_numpy(code: str, real_inputs: dict, warmup: int, repeat: int, lang: str) -> dict:
    try:
        if lang == "torch":
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch_inputs = {k: torch.from_numpy(v).to(device) for k, v in real_inputs.items()}
            ns = {"__name__": "__kernel__"}
            exec(code, ns)
            kernel_func = ns["kernel"]

            for _ in range(warmup):
                _ = kernel_func(**torch_inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # CUDA event 计时 (毫秒 -> 秒), CPU 回退到 perf_counter
            is_cuda = torch.cuda.is_available()
            times = [t / 1000 for t in _timed_runs(lambda: kernel_func(**torch_inputs), repeat, is_cuda)]
        else:
            ns = {"__name__": "__kernel__"}
            exec(code, ns)
            kernel_func = ns["kernel"]
            for _ in range(warmup):
                kernel_func(**real_inputs)
            times = [t / 1000 for t in _timed_runs(lambda: kernel_func(**real_inputs), repeat, False)]

        avg_time = sum(times) / len(times)
        gflops, bandwidth = _compute_metrics(real_inputs, avg_time)

        return {
            "success": True,
            "language": lang,
            "avg_time_ms": round(avg_time * 1000, 4),
            "min_time_ms": round(min(times) * 1000, 4),
            "max_time_ms": round(max(times) * 1000, 4),
            "gflops": round(gflops, 4),
            "memory_bandwidth_gbs": round(bandwidth, 4),
            "runs": repeat,
            "device": str(device) if lang == "torch" else "cpu",
        }
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {e}"}


def _compute_metrics(real_inputs: dict, elapsed: float) -> tuple:
    """计算算子 gflops 和 bandwidth，支持 matmul / layernorm / softmax / elementwise"""
    gflops = 0.0
    bandwidth = 0.0
    if elapsed <= 0:
        return gflops, bandwidth

    def _ndim(x):
        """兼容 numpy 和 torch 的 ndim 获取"""
        if hasattr(x, 'ndim'):
            return x.ndim
        if hasattr(x, 'dim'):
            return x.dim()
        return len(x.shape) if hasattr(x, 'shape') else 0

    def _itemsize(x):
        """兼容 numpy 和 torch 的 itemsize"""
        if hasattr(x, 'itemsize'):
            return x.itemsize
        # torch tensor
        if hasattr(x, 'element_size'):
            return x.element_size()
        return 4

    # Matmul
    if "A" in real_inputs and "B" in real_inputs:
        a = real_inputs["A"]; b = real_inputs["B"]
        m, k = a.shape; k2, n = b.shape
        flops = 2 * m * n * k
        gflops = flops / elapsed / 1e9
        bytes_moved = (m * k + k * n + m * n) * _itemsize(a)
        bandwidth = bytes_moved / elapsed / 1e9
        return gflops, bandwidth

    # Attention: Q, K, V with last D == head dim
    if "Q" in real_inputs and "K" in real_inputs and "V" in real_inputs:
        q = real_inputs["Q"]
        shape = q.shape
        # Shape interpretation based on ndim
        if len(shape) == 4:
            # (B, H, S, D)
            B, H, S, D = shape
        elif len(shape) == 3:
            # (H, S, D)  (batch=1)
            H, S, D = shape
            B = 1
        elif len(shape) == 2:
            # (S, D)  (B=1, H=1)
            S, D = shape
            B, H = 1, 1
        else:
            B, H, S, D = 1, 1, shape[0], shape[1] if len(shape) > 1 else 1
        # Attention FLOPs: QK^T matmul (2*B*H*S*S*D) + softmax(4*B*H*S*S) + PV matmul (2*B*H*S*S*D)
        # Ignore softmax (small) => total ~4*B*H*S*S*D FLOPs
        flops = 4.0 * B * H * S * S * D
        gflops = flops / elapsed / 1e9
        # Flash Attention data movement: Q (BHS D*2) K (BHS D*2) V (BHS D*2) O (BHS D*2) [FP16]
        # Actually HBM access = reads of Q once + K & V once each (re-read per tile) + write O
        # Use simpler estimate: 4 * (BHS * D * bytes_elem)
        dtype_bytes = _itemsize(q)
        total_elements = B * H * S * D
        bytes_moved = 4 * total_elements * dtype_bytes  # Q read, K read, V read, O write
        bandwidth = bytes_moved / elapsed / 1e9
        return gflops, bandwidth

    # LayerNorm / Layer Norm (有 weight/bias 参数)
    if "X" in real_inputs and ("weight" in real_inputs or "bias" in real_inputs):
        x = real_inputs["X"]
        if _ndim(x) >= 2:
            shape = x.shape
            m, n = shape[0], shape[-1]
            # 每元素约 11 FLOPs: mean(1) + sub(1) + var(2) + inv_std(1) + norm(1) + mul(1) + add(1) + reductions(2)
            flops_per_row = 11 * n
            total_flops = m * flops_per_row
            # Bytes: read X + read weight + read bias + write Y
            dtype_bytes = _itemsize(x)
            bytes_moved = (m * n + n + n + m * n) * dtype_bytes
            gflops = total_flops / elapsed / 1e9
            bandwidth = bytes_moved / elapsed / 1e9
            return gflops, bandwidth

    # Softmax / Elementwise (single input, no weight/bias)
    if "X" in real_inputs and "weight" not in real_inputs and "bias" not in real_inputs:
        x = real_inputs["X"]
        nd = _ndim(x)
        if nd >= 2:
            n = int(np.prod(x.shape)) if hasattr(x, 'shape') else 0
            # Softmax: 5 FLOPs/element + reduction
            flops_per_row = 5 * x.shape[-1] + x.shape[-1]
            total_flops = x.shape[0] * flops_per_row
            dtype_bytes = _itemsize(x)
            bytes_moved = 2 * n * dtype_bytes  # read + write
            gflops = total_flops / elapsed / 1e9
            bandwidth = bytes_moved / elapsed / 1e9
            return gflops, bandwidth
        elif nd == 1:
            n = x.shape[0]
            flops = 5 * n + n
            dtype_bytes = _itemsize(x)
            bytes_moved = 2 * n * dtype_bytes
            gflops = flops / elapsed / 1e9
            bandwidth = bytes_moved / elapsed / 1e9
            return gflops, bandwidth

    return gflops, bandwidth


def get_gpu_info() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            info = {
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gb": round(props.total_memory / 1e9, 2),
                "multiprocessor_count": props.multi_processor_count,
                "device": "cuda",
                "device_count": device_count,
                "selected_gpu": _SELECTED_GPU_ID,
                "gpu_selection_log": _GPU_SELECTION_LOG,
                "torch_version": torch.__version__,
            }
            # 多卡显存快照
            gpus = _query_gpu_memory()
            if gpus:
                info["gpu_memory"] = [
                    {"id": g["index"], "name": g["name"],
                     "free_gb": round(g["free_mb"] / 1024, 1),
                     "total_gb": round(g["total_mb"] / 1024, 1),
                     "util_pct": g.get("util_pct")}
                    for g in gpus
                ]
            try:
                import triton
                info["triton_version"] = triton.__version__
            except ImportError:
                info["triton_version"] = "not installed"
            return info
    except ImportError:
        pass
    return {"name": "CPU (no CUDA)", "device": "cpu"}


if __name__ == "__main__":
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        print(json.dumps({"success": False, "error": f"Invalid JSON: {e}"}))
        sys.exit(1)

    # 确保使用自动选择的最优 GPU
    _ensure_gpu_device()

    action = payload.get("action", "execute")
    code = payload.get("code", "")
    inputs = payload.get("inputs", {})

    if action == "gpu_info":
        result = get_gpu_info()
    elif action == "compile":
        result = run_compile(code)
    elif action == "execute":
        result = run_execute(code, inputs)
    elif action == "verify":
        result = run_verify(code, inputs, payload.get("kernel_type", ""),
                            float(payload.get("max_error", 1e-5)),
                            precision=payload.get("precision", ""))
    elif action in ("benchmark", "benchmark_with_baselines"):
        # 共享机器: 只在测量前采一次卡上占用情况。被其他租户占用会让加速比失真,
        # 把实测占用随结果返回, 由控制端标记污染, 而不是静默采信这个数字。
        util_before, other_procs = _sample_gpu_idle()
        if action == "benchmark":
            result = run_benchmark(code, inputs,
                                   payload.get("warmup", 3), payload.get("repeat", 10))
        else:
            result = run_benchmark_with_baselines(
                code, inputs, payload.get("kernel_type", ""),
                payload.get("warmup", 10), payload.get("repeat", 30),
                precision=payload.get("precision", ""),
            )
        if isinstance(result, dict):
            result["gpu_util_pct"] = util_before
            result["gpu_other_procs"] = other_procs
            result["selected_gpu"] = _SELECTED_GPU_ID
    else:
        result = {"success": False, "error": f"Unknown action: {action}"}

    print(json.dumps(result, ensure_ascii=False))
