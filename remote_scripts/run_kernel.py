"""Remote kernel runner - 部署在远程 GPU 服务器上执行 (当前: algorithm-h20, 8×H20)
Supports both Python (numpy/torch) and Triton kernel execution.

Actions:
  gpu_info  - query GPU info
  compile   - L1 compile check
  execute   - L2 functional run (single iteration)
  benchmark - L3 performance measurement (multiple iterations)

环境变量:
  OPERATOR_AGENT_GPU - 固定使用的 GPU 卡号; 未设置时按 (低利用率, 高显存余量) 自动选

同级依赖:
  tolerance.py - L2 数值门的容差表与判定式 (单一事实来源, 见该模块 docstring)。
  anticheat.py - 运行时屏蔽 shield_torch, 与控制端 static_audit 共用同一份禁用清单。
  gpu_claim.py - 跨进程的同卡占用声明 (CHECKLOG I68): import 本文件时写入、退出拆除。
  input_fidelity.py / remote_input_fidelity.py - 原值/storage 传输及实际 Torch 回读。
  Python 共享依赖由 SSHExecutor.deploy() 从 operator_agent/ 推送; gpu_claim.py 与本文件同在
  remote_scripts/ 下, 部署与仓库内直跑都在同一目录, 直接 import 即可。
"""

import sys
import json
import time
import os
import uuid
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
try:
    import tolerance
    from anticheat import shield_torch
    from remote_input_fidelity import (JSON_PROTOCOL, InputFidelityError,
        decode_json_inputs, to_torch_inputs, torch_input_readback, torch_output_digest)
except ImportError:          # 仓库内直接运行: 模块还在 operator_agent/ 下
    sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "operator_agent"))
    import tolerance
    from anticheat import shield_torch
    from remote_input_fidelity import (JSON_PROTOCOL, InputFidelityError,
        decode_json_inputs, to_torch_inputs, torch_input_readback, torch_output_digest)

import gpu_claim  # 同目录 (部署后 = 工作目录; 仓库内 = remote_scripts/)


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
    """采样选中 GPU 的测量前状态。

    返回 `(利用率%, 其他进程数, 两次 util 原始读数, 第二次的完整 GPU 状态字典)`。
    拿不到时前三个是 None/None/[]，第四个是 {}。第四个返回值顺路带回了时钟与
    throttle 读数 (D2) —— 见 `read_gpu_state`，那里说明了为什么合并进同一条查询。

    利用率统计的是"有 kernel 在跑的时间占比", 我们自己的 benchmark 会把它顶到
    100%, 所以只能在测量前采样。其他计算进程数比利用率更可靠地反映是否被共享,
    故一并返回 (排除本进程自己)。

    ⚠️ **采两次取最小值**, 中间留 1.2 s。原因 (真机实测, docs/research/33 第三节):
    `nvidia-smi` 的 utilization.gpu 是过去约 1 秒窗口内的占比, 所以"测量前"采样
    只挡住了**本进程**的核, 挡不住**上一个 runner 进程**的尾巴 —— L1/L2/L3 是
    三次独立的 ssh 调用, 间隔常常不到 1 秒。实测在一台真空闲的卡上 (连续 100 次
    采样全是 0%), 紧挨着自己一轮 benchmark 之后去采, 读到的是 10-34%。
    这个假阳性会直接判「基准被污染」→ 加速比不可比 → Conductor 不计停滞、
    不禁用配方, 也就是让闭环静默失效。

    取 min 而不是 max/mean: 自己的残留会在这个间隔里排空, 邻居的持续负载不会,
    所以两次里较小的那个才更接近"别人占了多少"。

    ⚠️ 两个当时没量、后来量出来与假设不符的事实 (真机探针, docs/research/34):
      1. 残留衰减得比「utilization.gpu 是约 1 秒窗口」这个说法快得多 —— 8192² fp16
         matmul 跑满 0.3 s / 2.0 s 之后, t+0s 读到 100%, **t+0.3s 已经回到 0%**。
         所以 1.2 s 的间隔有 4 倍以上是纯代价, 值得下调 (本轮未改, 因为 L3 墙钟
         数字被 doc 36 引用过)。
      2. 生产 L3 路径根本撞不上这个窗口: 每个 action 都是新的远端进程 (ssh + uv +
         import torch 要好几秒), 到自己的核排空之后才采样。连跑 24 次 L3, 现规则与
         「旧单次快照」规则的污染触发率都是 0%。所以这层是**保险** (进程复用 /
         HTTP 常驻模式下才用得上), 不是当前在响的警报器。

    两次原始读数一并回报 (第三个返回值): 「min 是在去自污染还是在掩盖真污染」只能靠
    first/second 的差来判断, 只留 min 的话上面这两条结论根本量不出来, 生产里也不可疑
    审计。
    """
    util, others = None, None
    samples = []
    state_second = {}
    try:
        import subprocess

        first = read_gpu_state()
        time.sleep(1.2)
        state_second = read_gpu_state()
        samples = [s for s in (first.get("util_pct"), state_second.get("util_pct"))
                   if s is not None]
        util = min(samples) if samples else None

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
    return util, others, samples, state_second


_STATE_QUERY_FIELDS = ("utilization.gpu,clocks.sm,clocks.mem,clocks.max.sm,"
                       "clocks.max.mem,clocks_throttle_reasons.active")


def read_gpu_state() -> dict:
    """一次 nvidia-smi 调用读回测量的旁证: 利用率 + SM/显存时钟 + throttle 位图

    为什么合成一条查询: 每次 `nvidia-smi` 进程调用实测中位 ~130 ms (最大见过 306 ms),
    单独为时钟再开一次就是给每次 L3 白加几百毫秒。util 的采样本来就要付这个钱,
    顺路把时钟读回来是免费的。

    ⚠️ 窗口前的读数取自 `_sample_gpu_idle()` 的**第二次**采样, 也就是 warmup 之前
    ~130 ms 那一刻; 不是 launch 那一瞬。窗口内部的涨落只能靠 before/after 之差捕捉,
    捕捉不到就是捕捉不到 —— 这条限制要连数据一起交给下游读。

    D2 (docs/research/20 第三节) 的第一步。为什么要看时钟：这台卡 8 租户共享,
    邻居的负载理论上会通过功耗/温度预算把我们的 SM 频率压下来 (DVFS)。那种情况下
    「这一轮比上一轮慢」描述的可能是**这一轮跑在 1.4 GHz、上一轮跑在 1.8 GHz**,
    而不是核的质量 —— 与缺口 1 处理过的两种情形同构 (launch 地板支配、邻居抢时间)。

    ⚠️ **只观测不判定**: 不预设阈值。本机的实测结论 (空载与持续 tensor-core 满载 40 s
    之间 `clocks.sm` 恒为 1980 MHz = 上限、throttle 位全 0, 而同批传感器里 util/功耗/温度
    都在动) 见 docs/research/40 —— 注意那里同时收回了「时钟被锁在 1980」这个说法:
    本驱动的 `clocks.applications.sm` 不是合法字段, 「锁频」与「无压力所以不降」区分不了。
    真要加判据, 得先在新平台上看到它动。
    """
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", f"--id={_SELECTED_GPU_ID}",
             f"--query-gpu={_STATE_QUERY_FIELDS}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return {}
        parts = [p.strip() for p in r.stdout.strip().splitlines()[0].split(",")]
        if len(parts) < 6:
            return {}

        def _num(s):
            # nvidia-smi 对不支持的字段回 "[N/A]"/"N/A", 不能直接 int()
            try:
                return int(s)
            except ValueError:
                return None

        # throttle 位图是十六进制串 ("0x0000000000000004" = SW power cap)
        try:
            throttle = int(parts[5], 16)
        except ValueError:
            throttle = None
        return {
            "util_pct": _num(parts[0]),
            "sm_clock_mhz": _num(parts[1]),
            "mem_clock_mhz": _num(parts[2]),
            "sm_clock_max_mhz": _num(parts[3]),
            "mem_clock_max_mhz": _num(parts[4]),
            "throttle_bits": throttle,
        }
    except Exception:
        return {}


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

# I68: 进程级占卡声明 —— 生命期与本进程相同 (atexit 拆除)。控制端的
# _EXEC_LOCKS 只在进程内互斥; 两个 harness 进程共卡时实测时间可整体偏移 2.1×
# 而档位仍 valid, 因为时刻采样看不见「还在 import torch 的对手」。claim 的
# 生命期覆盖对手的**整个进程**, 它还没建 CUDA 上下文也看得见。尽力而为:
# 写不进不挡测量 (声明是观测面, 不是入场券)。
gpu_claim.write_claim(_SELECTED_GPU_ID, runner="run_kernel")


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
        return {"success": False,
                "error": f"Import error ({lang}): {type(e).__name__}: {e}"}
    except Exception as e:
        # 异常类名必须带上。本文件其余每一条错误路径都是 `f"{type(e).__name__}: {e}"`,
        # 只有这两处原本写的是 `f"Compile error: {e}"` —— 于是 L1 失败的 error_log 里
        # **一个类名都没有**, 而 L1 恰恰是 Fixer 最需要「这是 SyntaxError 还是
        # AttributeError」的那一段。(真机量出来的: 三条失败全落进 unknown 桶,
        # 见 docs/research/45 第五节)
        return {"success": False,
                "error": f"Compile error: {type(e).__name__}: {e}"}


def _deserialize_inputs(inputs: dict) -> dict:
    """反序列化输入。支持三种格式:
    - 轻量模式: {"_generate": true, "specs": {...}, "scalars": {...}}
    - 完整模式 (legacy): {"A": {"shape": [...], "dtype": "...", "data": [...]}}
    - 标量/布尔直通 (如 causal=False): 不具有 "shape"/"data" 键者原样透传

    `scalars` 是 2026-10-01 补的: 在它之前, 轻量模式把**标量也**按
    「shape+dtype+seed 重新生成」处理, 于是 `causal=False` 变成一个 `astype(bool)`
    之后的 N(0,1) 抽样 —— 实测几乎总是 `True`, `eps=1e-5` 变成 `-0.0223`,
    `repeats=8` 变成 `0`。带标量参数的候选核因此在真机上验的是另一个算子 (而 L2
    会照样给出通过/不通过)。标量按值直传才与上面第三条「原样透传」的承诺一致。
    """
    if "protocol" in inputs:
        return decode_json_inputs(inputs)  # unknown versions fail closed
    if inputs.get("_generate"):
        out = {}
        for k, spec in inputs["specs"].items():
            rng = np.random.default_rng(spec["seed"])
            shape = spec["shape"]
            dtype = spec["dtype"]
            # 生成与 controller 端一致的随机数据 (正态分布 × 0.1)
            out[k] = (rng.standard_normal(shape, dtype=np.float32) * 0.1).astype(dtype)
        # 标量不在 specs 里 (没有 shape/seed 可言), 按值收下。
        for k, v in (inputs.get("scalars") or {}).items():
            out[k] = v
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


def _load_reference_from_source(source: str, inputs: dict):
    """Compile a traveler reference (`def reference(**inputs)`) sent by the control side.

    Used when the task carries `reference_source` (novel ops with no built-in
    reference). Same trust boundary as exec'ing the candidate kernel.
    Returns (callable, reference_type) or (None, None).
    """
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
    """Build the reference on the remote GPU, never on the controller host."""
    import torch
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    values = {k: (torch.from_numpy(np.array(v, copy=True)).to(device)
                  if isinstance(v, np.ndarray) else v) for k, v in inputs.items()}
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


def _error_shape_torch(bad, *, n_bad: int, total: int, has_nonfinite: bool):
    """坏点掩码 → 误差形态标签 (torch 侧的归约; 规则在 `tolerance`)

    与本地路径 (`verifier._error_shape_of`) 的分工完全对称: 归约在各自的数组后端上做
    (这里 torch, 那面 numpy), **分类规则只有一份** (`tolerance.classify_error_shape`,
    `deploy()` 会把同一个 tolerance.py 推到远端)。两边传进去的东西必须一样 ——
    这一点由 `tests/test_tolerance_error_shape.py` 的一致性用例钉住。

    形状折叠成二维 (除最后一维外全部折叠), 于是 (M,N) 与 (B,H,S,D) 用同一套规则。
    """
    if bad.dim() == 0 or bad.numel() == 0:
        # 拿不到二维视图时仍然问一次规则: `nonfinite` 不依赖任何形状
        return tolerance.classify_error_shape(tolerance.ErrorShapeInputs(
            n_bad=int(n_bad), total=int(total), rows=0, cols=0, bad_rows=0, bad_cols=0,
            rows_all_bad=0, cols_all_bad=0, bad_row_prefix=0, bad_row_suffix=0,
            bad_col_prefix=0, bad_col_suffix=0, has_nonfinite=bool(has_nonfinite)))
    cols = int(bad.shape[-1])
    if cols <= 0:
        return None
    mask = bad.reshape(-1, cols)
    n_rows = int(mask.shape[0])
    per_row = (mask.sum(dim=1) > 0).tolist()
    per_col = (mask.sum(dim=0) > 0).tolist()
    return tolerance.error_shape_from_axis_flags(
        per_row=per_row, per_col=per_col,
        n_bad=int(n_bad), total=int(total),
        rows_all_bad=int((mask.sum(dim=1) == cols).sum().item()),
        cols_all_bad=int((mask.sum(dim=0) == n_rows).sum().item()),
        has_nonfinite=bool(has_nonfinite))


def run_verify(code: str, inputs: dict, kernel_type: str, max_error: float,
              precision: str = "", reference_source: str = "") -> dict:
    """Run candidate and reference on one device; return only verification summary."""
    evidence = {}
    try:
        import torch
        real_inputs = _deserialize_inputs(inputs)
        exact_inputs = inputs.get("protocol") == JSON_PROTOCOL
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if exact_inputs:
            if _detect_language(code) == "triton" and device.type != "cuda":
                raise InputFidelityError("Triton 候选没有 CUDA 执行设备，不模拟为通过")
            candidate_inputs = to_torch_inputs(real_inputs, device)
            # Actual device tensors, before any candidate/reference source exec.
            evidence["input_fidelity"] = torch_input_readback(candidate_inputs)
        reference, reference_type = _load_reference_from_source(reference_source, real_inputs)
        if exact_inputs and reference_source and reference is None:
            reason = f"声明的 reference_source 未能建立: {reference_type}"
            return {"success": False, "error": reason, "reference_error": reason, **evidence}
        if reference is None:
            reference, reference_type = _reference_torch(kernel_type, real_inputs)
        if reference is None:
            reason = f"No remote reference for {kernel_type}"
            return {"success": False, "error": reason, "reference_error": reason, **evidence}

        module = _load_module(code)
        if not exact_inputs:
            candidate_inputs = {k: (torch.from_numpy(v).to(device).contiguous()
                if isinstance(v, np.ndarray) else v) for k, v in real_inputs.items()}
            # Legacy callers keep their precision semantics. Exact L2 transports
            # the declared dtype; precision selects tolerances, not input casts.
            if (precision or "").lower() == "fp16":
                candidate_inputs = {k: (v.to(torch.float16)
                    if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
                    for k, v in candidate_inputs.items()}
        # 运行时反作弊: 屏蔽只包住候选那一次调用。参考实现 (_reference_torch)
        # 恰恰要合法地调用 torch.matmul / F.softmax, 把 with 套在外面会连它一起打死。
        # 候选先跑、参考后跑这个顺序本身也是一道防护 (参考还没算出来, 候选无从
        # 回收它的显存), 见 docs/research/12 第四节。
        with shield_torch(), torch.no_grad():
            actual = module.kernel(**candidate_inputs)
        with torch.no_grad():
            expected = reference()
        if not isinstance(actual, torch.Tensor):
            actual = torch.as_tensor(actual, device=device)
        if exact_inputs:
            evidence["output_digest"] = torch_output_digest(actual)
        if list(actual.shape) != list(expected.shape):
            return {"success": False, "error": "output shape mismatch",
                    "output_shape": list(actual.shape), "expected_shape": list(expected.shape), **evidence}
        actual_f = actual.detach().to(torch.float32)
        expected_f = expected.detach().to(torch.float32)

        # ---- 数值门禁: 绝对 + 相对 (AND 形式) ----
        # 判据与容差表的定义在 operator_agent/tolerance.py, 由 deploy() 连同本文件
        # 一起推送到同目录。这里只做 torch 侧的元素级运算, 归约成标量后交给
        # tolerance.evaluate() —— 判定规则因此只有一份, 又不必把整个输出张量搬回
        # 主机 (attention 类算子的输出可以是 GB 量级)。
        #
        # 原实现是 `max_abs_error <= max_error`, 纯绝对容差: 实测在 4096³ fp16 上
        # 允许把结果整体缩小 14.93% 仍判通过 (docs/research/12 第一节)。
        atol, rtol, rho = tolerance.effective_atol(precision, max_error)
        abs_error = (actual_f - expected_f).abs()
        total = int(abs_error.numel())
        comparable = torch.isfinite(expected_f)
        n_compared = int(comparable.sum().item())
        n_ref_nonfinite = total - n_compared
        if total:
            # EPS=1e-8 而不是原先的 clamp_min(1e-12): 后者会让近零参考点上的
            # rel_err 再放大 4 个数量级。
            rel_error = abs_error / (expected_f.abs() + tolerance.EPS)
            # AND: 只有同时超绝对阈值和相对阈值才算坏点。这是不误杀合法 fp16
            # 累加核的关键 —— 那种核在近零参考处的 rel_err 可达 2.1e4。
            bad = (abs_error > atol) & (rel_error > rtol) & comparable
            n_bad = int(bad.sum().item())
            max_abs_error = float(abs_error[comparable].max().item()) if n_compared else 0.0
            max_rel_error = float(rel_error[comparable].max().item()) if n_compared else 0.0
            has_nonfinite = bool((~torch.isfinite(actual_f)).any().item())
            # 第一个坏元素: 展平下标 + 两侧取值 (LeetCUDA book_check 同款)。
            # 展平下标够 Fixer 定位坏 tile; 两侧取值分得清 scale 错 vs 局部炸。
            first_bad_index = None
            first_bad_actual = None
            first_bad_expected = None
            if n_bad:
                fb = int(torch.argmax(bad.reshape(-1).to(torch.int8)).item())
                first_bad_index = fb
                first_bad_actual = float(actual_f.reshape(-1)[fb].item())
                first_bad_expected = float(expected_f.reshape(-1)[fb].item())
        else:
            n_bad, max_abs_error, max_rel_error, has_nonfinite = 0, 0.0, 0.0, False
            first_bad_index = first_bad_actual = first_bad_expected = None

        passed, matched_ratio = tolerance.evaluate(n_bad, n_compared, has_nonfinite, rho)

        # 误差**形态** (设计 §8.2): 与本地路径 (verifier._error_shape_of) 用**同一份规则**
        # (`tolerance.classify_error_shape`), 这里只做 torch 侧的归约。
        # ⚠️ 触发条件是"判失败", 不是"有坏点": 含 NaN/Inf 时 `bad` 恒 False (NaN 比较恒假),
        # 挂在坏点计数上的话, 最该说出口的那一种形态会是唯一永远报不出来的。
        error_shape = None
        if not passed:
            error_shape = _error_shape_torch(bad, n_bad=n_bad, total=total,
                                             has_nonfinite=has_nonfinite)

        if has_nonfinite:
            reason = "输出含 NaN/Inf"
        elif not passed:
            reason = (f"matched_ratio {matched_ratio:.6f} < {rho} "
                      f"({n_bad}/{total} 个元素同时超 atol={atol:.1e} 与 rtol={rtol:.1e}; "
                      f"max_abs={max_abs_error:.2e}, max_rel={max_rel_error:.2e})")
            if first_bad_index is not None:
                reason += (f"; first_bad@{first_bad_index}"
                           f" (got {first_bad_actual:.6g}, want {first_bad_expected:.6g})")
        else:
            reason = ""
        return {
            "success": passed,
            "reference_type": reference_type,
            "output_shape": list(actual.shape),
            "compared_elements": n_compared,
            "coverage": n_compared / total if total else 1.0,
            "reference_nonfinite": n_ref_nonfinite,
            "max_error": max_abs_error,
            "max_rel_error": max_rel_error,
            "matched_ratio": matched_ratio,
            "atol": atol,
            "rtol": rtol,
            "required_matched_ratio": rho,
            "bad_elements": n_bad,
            "first_bad_index": first_bad_index,
            "first_bad_actual": first_bad_actual,
            "first_bad_expected": first_bad_expected,
            # 误差形态 (设计 §8.2): None = 判不了 (三态), 缺键与"判不了"在产物里同形,
            # 所以一律带上这个键 —— 控制端读的是它, 而不是去猜。
            "error_shape": error_shape,
            "error": reason,
            "device": str(device),
            **evidence,
        }
    except InputFidelityError as e:
        return {"success": False, "error": str(e), "input_fidelity_error": str(e), **evidence}
    except Exception as e:
        return {"success": False, "error": f"Remote verification error: {type(e).__name__}: {e}", **evidence}


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

            # 屏蔽包住 warmup + 计时整段, 而不是包住每一次 kernel 调用:
            # shield_torch 进出各要 setattr 二十多个属性, 在 18.69µs 的 launch
            # 底噪上是 2~3 倍开销, 逐次包裹会把 L3 计时彻底毁掉。
            with shield_torch():
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

            with shield_torch():
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
            # numpy DSL 同样要屏蔽: 「在 CPU 上用 torch 算出结果再返回」是这条路径
            # 下最省事的作弊写法, 静态审计漏掉时必须由运行时兜住。
            with shield_torch():
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


def _timed_runs(fn, repeat: int, device_is_cuda: bool, flush_l2: bool = True) -> list[float]:
    """重复执行 fn 并返回每轮耗时 (毫秒)。

    GPU 上使用 CUDA event 计时, 避免 CPU 端 time.perf_counter 引入的
    launch 调度与同步噪声, 提升 L3 性能验证精度 (对 autotune 选优更可靠)。

    每次计时前冲刷 L2, 做法直接取自 Triton 官方 `do_bench`
    (`triton/python/triton/testing.py:299,329` + `third_party/nvidia/backend/
    driver.py:398-408`): 一个 256 MB 的 buffer, 每轮 `zero_()` 一遍, 且冲刷在
    event 窗口**之外** (同流有序, 所以 start.record() 一定在冲刷完成之后)。

    为什么必须跟: H20 的 L2 是 60 MiB。layernorm 1024×768 读写共 3.15 MB, 第一轮
    之后就整份常驻 L2, 此后每次"测量"量的都是 L2 命中而不是 HBM; 而 verifier 拿
    这个时间去和 3.68 TB/s 的 **HBM** 峰值比, 算出的带宽利用率因此系统性偏高。
    matmul 4096³ 的 100 MB 流量超出 L2 只有部分命中 —— 偏差方向相同但幅度不同,
    也就是说候选与基线被偏袒的程度不一样, 连加速比本身都是脏的。
    """
    import torch
    times = []
    flush_buf = _l2_flush_buffer() if (device_is_cuda and flush_l2) else None
    for _ in range(repeat):
        if device_is_cuda:
            if flush_buf is not None:
                flush_buf.zero_()
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


_L2_FLUSH_BYTES = 256 * 1024 * 1024   # 与 Triton do_bench 一致
_l2_flush_cache = None


def _l2_flush_buffer():
    """惰性分配 256 MB 冲刷缓冲; 非 CUDA 环境下调用方不会走到这里"""
    global _l2_flush_cache
    if _l2_flush_cache is None:
        import torch
        _l2_flush_cache = torch.empty(_L2_FLUSH_BYTES // 4, dtype=torch.int, device="cuda")
    return _l2_flush_cache


# 时间预算 (毫秒) 而不是固定次数决定重复轮数, 同样取自 Triton do_bench
# (`testing.py:273-275,313-314`: warmup=25 / rep=100 的单位是**毫秒**,
# n_repeat = max(1, int(rep / estimate_ms)))。
# 固定 repeat 对一个 20 µs 的核只有 0.2 ms 的总测量窗口, 统计上毫无意义, 而 L3
# 的加速比要拿去做配方禁用这种**永久**决策。取 25/50 而不是官方的 25/100:
# 一次 L3 要测候选 + 四条基线共五组, 且这台机器是共享的、每次评测都在 LLM 循环
# 的关键路径上。
WARMUP_MS_BUDGET = 25
REPEAT_MS_BUDGET = 50
MAX_REPEATS = 5000        # 上限兜底: 极快的核不至于把一次 L3 拖到几十秒


def _resolve_counts(fn, warmup: int, repeat: int, device_is_cuda: bool) -> tuple:
    """按时间预算把 (warmup, repeat) 两个「次数」折算成实际轮数

    估时只跑一次 (5 轮取均值, 不计入结果), 两个预算共用它 —— do_bench 也是先估
    一次再同时算 n_warmup 与 n_repeat (`testing.py:310-314`)。估时不冲 L2, 因为
    它只用来定轮数, 不进结果。

    结果不低于调用方给的次数, 所以慢核行为完全不变, 只有快核会被自动加采样。
    估时失败时退回原值: 预算是增强, 不该成为新的失败点。
    """
    if not device_is_cuda:
        return warmup, repeat
    try:
        est = sum(_timed_runs(fn, 5, True, flush_l2=False)) / 5
    except Exception:
        return warmup, repeat
    if est <= 0:
        return warmup, repeat
    n_warmup = int(min(max(warmup, WARMUP_MS_BUDGET / est), MAX_REPEATS))
    n_repeat = int(min(max(repeat, REPEAT_MS_BUDGET / est), MAX_REPEATS))
    return n_warmup, n_repeat


def _percentile(sorted_times, q: float) -> float:
    """升序样本的 q 分位数 (q ∈ [0,100]), 线性插值, 与 numpy 默认口径一致。

    不引 numpy 是为了让这个函数在 runner 的任何执行路径上都能用 —— runner 会被
    scp 到远端单独跑, 依赖越少越好。
    """
    n = len(sorted_times)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_times[0]
    pos = (n - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_times[lo] * (1.0 - frac) + sorted_times[hi] * frac


def _sample_stats(times) -> dict:
    """把一批样本摊成统计量, 三条 benchmark 路径共用。

    p25/p75 是给离散度门用的 (docs/research/20 第四节): runner 本来就跑了 n 个样本,
    多报两个分位数**不花任何额外 GPU 时间**, 而下游因此能区分「核真的慢」和
    「这台机器此刻在抖」。判据 iqr/median 与 PyTorch Timer 的
    `_IQR_WARN_THRESHOLD` 同源。

    ⚠️ iqr/median 对**单个离群样本完全不敏感** (29 个 1.0 + 1 个 50.0 → p25=p75=1.0
    → iqr=0)。当决策统计量是 median 时这正是想要的性质 —— median 在 50% 样本被
    污染前都不动。但它意味着 iqr 看不见「测量窗口中途有邻居进程插进来」, 而
    gpu_util 只在测量**前**采一次快照, 同样看不见。所以额外报 max/median 作为
    「窗口内发生过抢占」的廉价旁证。
    """
    s = sorted(times)
    median = _percentile(s, 50)
    p25 = _percentile(s, 25)
    p75 = _percentile(s, 75)
    # ⚠️ 时间一律 round(..., 6) 而不是 4: 4 位小数 = 0.1 µs 的格点, 而 launch 主导的
    # 小核中位数就在 3 µs 量级 —— 0.1/3.1 = 3.2% 的量化步长会让「5 次重复一模一样」
    # 这种假象通过离散度门 (D3 真机对照时就是这样: 旧 runner 安静机器上 iqr 恰好 0.0000)。
    return {
        "min_time_ms": round(s[0], 6),
        "median_time_ms": round(median, 6),
        "avg_time_ms": round(sum(s) / len(s), 6),
        "max_time_ms": round(s[-1], 6),
        "p25_time_ms": round(p25, 6),
        "p75_time_ms": round(p75, 6),
        # 相对四分位距: 样本主体的散布。median 为 0 时不猜 (ZeroDivisionError 之源)。
        "dispersion_iqr_over_median": round((p75 - p25) / median, 4) if median > 0 else None,
        # 最慢样本 / 中位样本: 对离群点敏感, 与上面那个互补, 不可互相替代。
        "dispersion_max_over_median": round(s[-1] / median, 4) if median > 0 else None,
        "runs": len(s),
    }


def _bench_callable(fn, warmup: int, repeat: int, device_is_cuda: bool) -> dict:
    import torch
    fn()                       # 先跑一次吃掉 JIT/首次派发, 否则估时会严重偏大
    if device_is_cuda:
        torch.cuda.synchronize()
    n_warmup, n_repeat = _resolve_counts(fn, warmup, repeat, device_is_cuda)
    for _ in range(n_warmup):
        fn()
    if device_is_cuda:
        torch.cuda.synchronize()
    times = _timed_runs(fn, n_repeat, device_is_cuda)
    stats = _sample_stats(times)
    # 报实际跑的轮数, 不是调用方请求的: 快核会被自动加采样, 下游要知道
    # 这个数字有多少样本撑着 (verifier 会把它摊进 L3 metrics)
    stats["runs_requested"] = repeat
    return stats


def _baseline_specs(reference, cand_inputs: dict, is_cuda: bool,
                    is_matmul: bool, is_fp16: bool) -> dict:
    """基线**构造器**表: 名字 -> () -> (可调用, 善后函数或 None)。

    构造器只做「建张量 / 包 lambda / torch.compile / 打开 allow_tf32」, 不计时。
    `torch.compile` 的编译因此天然落在计时区之外 (与旧写法一致); TF32 的善后函数
    必须在**整个配对窗口结束后**才调用 —— 交错采样里两臂共用一段时间, 中途改标志位
    等于让其中一臂换了配置跑。
    """
    import torch
    # 四个基线一律用 specs["名字"] = 构造器 的形式登记: 控制端的偏好名单
    # (verifier.baseline_plan) 与 test_measurement_validity 的跨文件守卫都按这个形状
    # 认出可用的名字。写成 dict 字面量会让守卫看不见其中一个, 名单飘了也无人报警。
    specs = {}
    specs["pytorch_eager"] = lambda: (reference, None)

    # FP16 任务要拿 cuBLAS FP16 当基线, 否则是拿 fp16 候选比 fp32 参考
    if is_cuda and is_matmul and is_fp16:
        def build_fp16():
            a = cand_inputs["A"].to(torch.float16)
            b = cand_inputs["B"].to(torch.float16)
            return (lambda: a @ b), None
        specs["pytorch_eager_fp16"] = build_fp16

    # Triton tl.dot() 在 Ampere+ 默认走 TF32, cuBLAS fp32 默认不走 —— 不开这档
    # 就是拿 fp32-cuBLAS 的慢去充当候选的应得成绩。fp16 任务下这档无意义, 不建。
    if is_cuda and is_matmul and hasattr(torch.backends.cuda, "matmul") and not is_fp16:
        def build_tf32():
            original = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = True
            def restore():
                torch.backends.cuda.matmul.allow_tf32 = original
            return reference, restore
        specs["pytorch_eager_tf32"] = build_tf32

    if hasattr(torch, "compile"):
        def build_compile():
            return torch.compile(reference), None
        specs["torch_compile"] = build_compile
    return specs


def _additional_peak_bytes(fn, device_is_cuda: bool):
    """这一臂每次调用**额外**占用的峰值字节数; 非 CUDA / 失败时返回 None。

    为什么现在才量它: KernelBench-Verified (arXiv 2607.16241) 的三个结论里, 前两个
    (把基线换成 TF32 打开的 PyTorch、加隐藏测试集) 我们已经有对应机制, 第三个没有 ——
    「the best-performing model (GPT-5.5) ... On the memory front, 28% of GPU kernels
    generated by the best model increase peak GPU memory usage」。本仓库在 today 之前
    对显存是**零测量** (`max_memory` 全仓 0 命中), 也就是这个失效模式整个看不见。
    而 `Conductor` 刚把 speedup 写进跨运行语料 (docs/research/43), 「快但吃显存」的核
    从此会被固化成下一次的示范。

    口径 (与 doc 44 一起记):
      - 在**全部计时之后**单独跑, 不混进 event 窗口, 免得扰动 median;
      - 两臂都已热身过, allocator 的 workspace 已落成 cached-reserved (算进
        `memory_allocated` 之外的 reserved, 不计 here), 所以这里量到的是**每次调用的
        额外瞬时峰值**, 不是一次性的 autotune workspace;
      - 只观测不判定: 本机对「我们的核会增加多少」一个数都还没有, 先攒分布再谈阈值
        —— 与 D2 (SM 时钟) 同一处置, 改判条件写在 docs/research/44。
    """
    if not device_is_cuda:
        return None
    import torch
    try:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        fn()
        torch.cuda.synchronize()
        return int(torch.cuda.max_memory_allocated() - base)
    except Exception:
        return None


def _arm_identity(device_is_cuda: bool) -> dict:
    """**计时当时**这一臂实际所在的那张卡 (观测值, 不是声明值)。

    为什么按臂各读一次, 而不是开局读一次全局: 设计 §11-2 要求「候选与基线同卡」是
    **观测复核**, 契约里的 `same_device` 只是声明。只读一次全局的话, 任何在**两臂之间**
    改了当前设备的代码都不会在两臂对比里显形 —— 而这类事故在本仓真发生过 (doc 63 §四:
    `CUDA_VISIBLE_DEVICES=2 cd X && …python` 是**前缀赋值**, 只作用于内建的 `cd`, 于是
    变量没进环境、占位落在 GPU 0, 而现场每一条读数都自洽)。

    ⚠️ `visible_index` 是**可见集内**的索引, 不是物理卡号: `CUDA_VISIBLE_DEVICES=2` 时
    `current_device()` 返回 0, 而物理卡是 2。所以这里连 `CUDA_VISIBLE_DEVICES` 一起带回去
    —— 「index 0」被当成卡号读, 正是上面那次事故的失效方式。控制端只拿这三个字段做**同卡
    对比** (同一进程内两侧口径相同), 不去反推物理卡号。
    """
    if not device_is_cuda:
        return {"visible_index": -1, "name": "cpu", "count": 0,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}
    try:
        import torch
        idx = int(torch.cuda.current_device())
        props = torch.cuda.get_device_properties(idx)
        return {"visible_index": idx,
                "name": str(getattr(props, "name", "unknown")),
                "count": int(torch.cuda.device_count()),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}
    except Exception as e:      # noqa: BLE001 —— 读不到就明说读不到, 不猜一个值
        return {"visible_index": None, "name": f"读取失败: {type(e).__name__}",
                "count": None,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}


def _new_window_id() -> str:
    """一扇配对测量窗的新 id。

    为什么不是全局计数器: 常驻 runner (`runner_server.py`) 里两次调用必须在同一进程内
    也可区分, 而 uuid 顺手把「跨进程/跨机器」那半边也解决了 (计数器会撞名, 撞名 = 把
    跨窗读成同窗, 正是这道检查要抓的反面)。可读性靠控制端日志里的 `measurement_window_scope`。
    """
    return uuid.uuid4().hex[:12]


def _bench_pair_interleaved(cand_fn, base_fn, warmup: int, repeat: int,
                            device_is_cuda: bool) -> tuple:
    """候选与基线**按块交错**计时, 返回 (候选 stats, 基线 stats, paired stats)。

    为什么必须交错 (真机实测, docs/research/36 第三节):
      原来的写法是「候选跑完全部样本, 再跑基线」。共享机器上邻居的插队只会落进其中
      一个臂, 于是加速比带一个**方向固定**的偏置, 且不随样本数收敛。反向测量实测
      layernorm 16384×6144: A→B 给 1.4351, B→A 给 1.3276 —— 同样两个核, 顺序不同
      差 7.7%; 1024×768 这种 launch 主导的小核更狠: 119.57 vs 0.0112 (差 10⁴ 倍)。
      安静机器上三种日程一致到小数点后三位 —— 交错治的是邻居, 不是仪器误差。

    为什么是**按块**而不是逐样本:
      候选臂必须在 anticheat 的 shield_torch 里跑 (torch 原生算子被换成抛错桩),
      基线臂本身就是 torch 原生算子 —— 同一个窗口里两臂不可能共用一个 shield。
      逐样本包裹 shield 正是本文件 `_bench_triton` 里那段注释早已否决过的做法;
      按块切换则是每块一次 setattr 群, 且全部发生在 event 窗口之外, 不进测量值。
      块长上限 10 也是真机定的: 块太大时整块会被邻居的一个大 matmul 吞掉, 反而造出
      新的离群 (1024×768 上 k=50 出现过 median 1.006 / max 91.4), k≤10 时中位数稳定。

    shield 只包候选一侧的**全部**调用 (首跑 / 估时 / warmup / 计时), 与旧的
    `_bench_callable` 语义一致 —— 不让任何一次候选执行躲开反作弊屏蔽。

    与 test_anticheat_bypass 那套 AST 结构断言有两处约定, 动之前先看它:
      1. 这里**必须**直接写 `with shield_torch()`, 不能把 shield 当参数注进来 ——
         守卫按 `shield_torch` 这个字面名识别屏蔽区间, 换成注入的 callable 之后这条
         (唯一的) L3 路径会从断言视野里整个消失 (真实发生过, 见该测试文件开头注释);
      2. 两个臂按**形参位置**分类 (第 0 个候选、第 1 个基线), 所以形参顺序不能换,
         名字可以改。
    """
    import torch

    # 本次配对测量的**窗口 id** (§11-2 的「同一时间窗」那一半; CHECKLOG I47)。
    # 一次 `_bench_pair_interleaved` = 一扇窗: 两臂按块交错、同一批时钟/热状态。
    # 两臂各自把**它自己那批样本所属的窗**带回去 (下面 `paired["arm_windows"]`), 于是
    # 「同窗」是**被观测到的**而不是被假设的 —— 哪天某条臂的样本被复用/短路 (缓存),
    # 带回来的就是旧窗, 控制端那条 `pairing_conflict` 就会响。
    window_id = _new_window_id()

    # 各跑一次吃掉 JIT / 首次派发, 否则估时会严重偏大 (与 _bench_callable 同一处理)
    with shield_torch():
        cand_fn()
    base_fn()
    if device_is_cuda:
        torch.cuda.synchronize()

    # 两臂用同一个样本数: 配对要求第 i 个样本互相对齐, 不能各按各的时间预算取数
    with shield_torch():
        n_cand = _resolve_counts(cand_fn, warmup, repeat, device_is_cuda)[1]
    n_base = _resolve_counts(base_fn, warmup, repeat, device_is_cuda)[1]
    n = int(min(MAX_REPEATS, max(n_cand, n_base)))
    block = max(1, min(10, n // 20))   # 至少切成 20 块, 每块不超过 10 个样本

    n_warm = max(warmup, 3)
    with shield_torch():
        for _ in range(n_warm):
            cand_fn()
    for _ in range(n_warm):
        base_fn()
    if device_is_cuda:
        torch.cuda.synchronize()

    cand_times, base_times = [], []
    block_sizes = []
    cand_ident = base_ident = None
    i = 0
    while i < n:
        k = min(block, n - i)
        block_sizes.append(k)
        # 每臂各在自己第一块**计时前**读一次身份 (§11-2 配对检查的观测量)。
        # ⚠️ 位置要在两臂**之间**各读一次, 不能都在循环开头各读一次: 那样「候选块跑完、
        # 基线块跑之前有人改了当前设备」就看不见 —— 而这类改动正是这条检查要抓的东西。
        if cand_ident is None:
            cand_ident = _arm_identity(device_is_cuda)
        # 一块 = 一次 shield 进出, 且 shield 包的是**整块**而不是每一次调用
        # (与本文件 541-543 的原则一致); 事件窗口在 _timed_runs 内部逐样本开闭,
        # 所以 setattr 群不进测量值。
        with shield_torch():
            cand_times.extend(_timed_runs(cand_fn, k, device_is_cuda))
        if base_ident is None:
            base_ident = _arm_identity(device_is_cuda)
        base_times.extend(_timed_runs(base_fn, k, device_is_cuda))
        i += k

    cand_stats = _sample_stats(cand_times)
    base_stats = _sample_stats(base_times)
    ratios = [b / c for c, b in zip(cand_times, base_times) if c > 0]
    paired = {"paired_runs": len(ratios), "paired_block_samples": block,
              # 逐臂窗口观测 (§11-2 的「同一时间窗」)。两臂在这里**同值**是结论
              # (同一次交错调用), 不是空转: 它由**这一扇窗**生成、随两臂各自的样本回来,
              # 复用别窗样本的那一天就会不等。控制端 `observed_windows` 读这两个键。
              "arm_windows": {"candidate": window_id, "baseline": window_id},
              # 两臂计时当时的卡身份 + 跑完之后的复读。控制端的同卡配对检查读这里
              # (§11-2 / oracle.evaluate 的 candidate_device/baseline_device)。
              # 放在 `paired` 而不是加第四个返回值: `benchmarks/probe_block_series.py`
              # 按三元解包这个函数, 改元数会静默打断那个探针。
              "arm_identity": {"candidate": cand_ident, "baseline": base_ident,
                               "after": _arm_identity(device_is_cuda)}}
    if ratios:
        rs = sorted(ratios)
        med = _percentile(rs, 50)
        p25, p75 = _percentile(rs, 25), _percentile(rs, 75)
        # median-of-ratios: 逐轮配对后的比值中位数, 与 ratio-of-medians 互补
        paired["speedup_paired_median"] = round(med, 4)
        paired["speedup_paired_iqr_over_median"] = (
            round((p75 - p25) / med, 4) if med > 0 else None)

    # ---- 逐块比值 median (先观测, 不判定) ----
    # 为什么粒度是「块」而不是样本: 这台机器上的干扰是**邻居插队**, 一次插队污染的是
    # 一整块连续时间, 不是某个孤立样本。iqr/median 对单点离群完全不敏感 (doc 33 第四节
    # 已经栽过一次), 而把 30 个样本摊成一个数同样看不见「前 20 块干净、后 3 块被抢」。
    # 块序列才是「什么时候可以停」的唯一可判输入 —— 但那条规则要先把 false-stop 率量出来
    # (docs/research/46), 所以这里只把它带回来, 不参与任何判决。
    block_ratios, pos = [], 0
    for k in block_sizes:
        seg = [b / c for c, b in zip(cand_times[pos:pos + k], base_times[pos:pos + k])
               if c > 0]
        pos += k
        if seg:
            block_ratios.append(round(_percentile(sorted(seg), 50), 5))
    paired["block_ratio_medians"] = block_ratios

    # ---- 峰值显存: 与速度并列的第二项观测 (口径与理由见 _additional_peak_bytes) ----
    # 候选臂照旧包在 shield 里 —— 本文件的约定是「候选的每一次执行都在屏蔽区内」,
    # 多出来这一次也不例外; 少包一次就是给反作弊开一个洞。
    with shield_torch():
        cand_peak = _additional_peak_bytes(cand_fn, device_is_cuda)
    base_peak = _additional_peak_bytes(base_fn, device_is_cuda)
    paired["candidate_peak_bytes_extra"] = cand_peak
    paired["baseline_peak_bytes_extra"] = base_peak
    paired["peak_memory_ratio"] = (round(cand_peak / base_peak, 3)
                                   if cand_peak and base_peak else None)
    cand_stats["runs_requested"] = repeat
    base_stats["runs_requested"] = repeat
    return cand_stats, base_stats, paired


def run_benchmark_with_baselines(code: str, inputs: dict, kernel_type: str,
                                 warmup: int = 10, repeat: int = 30,
                                 precision: str = "",
                                 baseline_priority=None,
                                 reference_source: str = "") -> dict:
    """Time the candidate against exactly one baseline, interleaved, on device.

    `baseline_priority` is the control side's ordered preference; the first
    baseline that actually runs is the only one measured. Ordering matters:
    it encodes which comparison is fair (fp16 candidate vs cuBLAS fp16, etc).
    """
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
        is_matmul = ("matmul" in (kernel_type or "").lower() or
                     "gemm" in (kernel_type or "").lower())
        is_fp16 = (precision or "").lower() == "fp16"

        # 基线不再「全都测一遍再让控制端挑」, 而是按控制端给的偏好顺序, **第一个能跑的
        # 就定下来** (D3, docs/research/20 第三节)。旧写法对 layernorm 要测 eager +
        # torch_compile 两份、对 fp16 matmul 要测三份, 而 verifier 永远只用其中一份 ——
        # 白烧 2~3 倍 L3 的 GPU 时间, 而且测得越久, 撞上邻居插队的窗口越长。
        specs = _baseline_specs(reference, candidate_inputs, is_cuda, is_matmul, is_fp16)
        order = [n for n in (baseline_priority or []) if n in specs]
        order += [n for n in specs if n not in order]
        chosen, attempts = None, []
        for name in order:
            fn, cleanup = None, None
            try:
                fn, cleanup = specs[name]()
                fn()                      # 一次实跑探可用性 (编译 / 显存 / 数值)
                if is_cuda:
                    torch.cuda.synchronize()
                chosen = (name, fn, cleanup)
                break
            except Exception as e:
                attempts.append(f"{name}: {type(e).__name__}: {str(e)[:160]}")
                if cleanup is not None:
                    try:
                        cleanup()
                    except Exception:
                        pass
        if chosen is None:
            return {"success": False,
                    "error": f"No usable remote baseline for {kernel_type}; "
                             f"attempts={attempts}"}
        base_name, base_fn, base_cleanup = chosen

        # 两臂按块交错计时。候选臂全程在 shield_torch 内 (每一次执行都是), 基线臂不在
        # —— 基线本身就是 torch 原生算子, 套进 shield 会立刻抛 CheatDetectedError。
        try:
            candidate_stats, base_stats, paired = _bench_pair_interleaved(
                candidate, base_fn, warmup, repeat, is_cuda)
        finally:
            if base_cleanup is not None:
                base_cleanup()
        elapsed = candidate_stats["median_time_ms"] / 1000
        gflops, bandwidth = _compute_metrics(real_inputs, elapsed)
        return {"success": True, "device": str(device), "candidate": candidate_stats,
                "baselines": {base_name: base_stats},
                "baseline_selected": base_name,
                "baseline_attempts": attempts,
                "paired": paired,
                "reference_type": reference_type,
                "gflops": round(gflops, 4), "memory_bandwidth_gbs": round(bandwidth, 4)}
    except Exception as e:
        return {"success": False, "error": f"Remote benchmark error: {type(e).__name__}: {e}"}


def _triton_compile_metadata(module) -> dict:
    """从**编译产物**取「关于被测量那份码」的事实 (§10.3 条件 2 的真源)

    为什么必须从这里取: 控制端的 L1 只有 4 个键、L3 的 87 个键里没有一个关于线程数/
    寄存器/溢出 —— `num_warps`/`shared`/`n_regs`/`n_spills` 今天没人发布, 于是
    §10.3 条件 2 的第二把量具 (`occupancy_bound_evidence`) 在真机上不可能产出证据。
    而「被测量的是哪份码」这件事, 只有编译产物说得准: 控制端的 sketch 与 LLM 真出的
    码本来就允许不一致 (所以**不**在控制端解析源码文本)。

    真机实测 (H20, triton 3.8.0): `_k.device_caches[dev][0]` 是 {key: CompiledKernel}
    (3.8 已无 `.cache`); `CompiledKernel` 有 `.n_regs`/`.n_spills`/`.n_max_threads`,
    `.metadata` 是 KernelMetadata, 带 `num_warps`/`num_stages`/`shared`(=每 CTA 字节)。

    三态纪律: 取不到就**不发布这个键** —— 缺失既不等于 0, 也不等于「没有上限」。
    """
    out = {}
    try:
        from triton.runtime.jit import JITFunction
    except Exception:
        return out
    for obj in list(vars(module).values()):
        if not isinstance(obj, JITFunction):
            continue
        caches = getattr(obj, "device_caches", None) or {}
        for entry in caches.values():
            cache = entry[0] if isinstance(entry, tuple) else entry
            try:
                kernels = list(cache.values())
            except Exception:
                continue
            for ck in kernels:
                md = getattr(ck, "metadata", None)
                if md is not None:
                    for src, dst in (("num_warps", "num_warps"),
                                     ("num_stages", "num_stages"),
                                     ("shared", "shared_bytes_per_cta")):
                        value = getattr(md, src, None)
                        if isinstance(value, int) and not isinstance(value, bool):
                            out[dst] = value
                for src in ("n_regs", "n_spills", "n_max_threads"):
                    value = getattr(ck, src, None)
                    if isinstance(value, int) and not isinstance(value, bool):
                        out[src] = value
                if out:
                    out["compile_metadata_from"] = type(ck).__name__
                return out
    return out


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

        # 屏蔽包住 warmup + 整个 _timed_runs, 不逐次包裹 (开销理由同 _execute_triton)
        with shield_torch():
            call = lambda: kernel_func(**torch_inputs)
            n_warmup, n_repeat = _resolve_counts(call, warmup, repeat, True)
            for _ in range(n_warmup):
                _ = call()
            torch.cuda.synchronize()

            # CUDA event 计时, 提升精度 (毫秒 -> 秒)
            times = [t / 1000 for t in _timed_runs(call, n_repeat, True)]

        stats = _sample_stats([t * 1000 for t in times])
        # gflops / bandwidth 改用 median: 单个被邻居抢占的样本就能把 avg 拉高数倍
        # (29×1.0ms + 1×50ms → avg 2.63ms, median 1.0ms), 而 median 在 50% 样本被
        # 污染前都不动。含基线的那条路径 (run_benchmark_with_baselines) 一直用的是
        # median, 这两条以前用 avg, 属于口径不一致。
        gflops, bandwidth = _compute_metrics(real_inputs, stats["median_time_ms"] / 1000)

        return {
            "success": True,
            "language": "triton",
            **stats,
            "gflops": round(gflops, 4),
            "memory_bandwidth_gbs": round(bandwidth, 4),
            "device": str(device),
            # 编译产物的事实 (num_warps/shared/n_regs/n_spills): 取不到就不带这些键
            **_triton_compile_metadata(module),
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

            with shield_torch():
                call = lambda: kernel_func(**torch_inputs)
                is_cuda = torch.cuda.is_available()
                n_warmup, n_repeat = _resolve_counts(call, warmup, repeat, is_cuda)
                for _ in range(n_warmup):
                    _ = call()
                if is_cuda:
                    torch.cuda.synchronize()

                # CUDA event 计时 (毫秒 -> 秒), CPU 回退到 perf_counter
                times = [t / 1000 for t in _timed_runs(call, n_repeat, is_cuda)]
        else:
            ns = {"__name__": "__kernel__"}
            exec(code, ns)
            kernel_func = ns["kernel"]
            with shield_torch():
                call = lambda: kernel_func(**real_inputs)
                for _ in range(warmup):
                    call()
                times = [t / 1000 for t in _timed_runs(call, repeat, False)]

        stats = _sample_stats([t * 1000 for t in times])
        # gflops / bandwidth 改用 median: 单个被邻居抢占的样本就能把 avg 拉高数倍
        # (29×1.0ms + 1×50ms → avg 2.63ms, median 1.0ms), 而 median 在 50% 样本被
        # 污染前都不动。含基线的那条路径 (run_benchmark_with_baselines) 一直用的是
        # median, 这两条以前用 avg, 属于口径不一致。
        gflops, bandwidth = _compute_metrics(real_inputs, stats["median_time_ms"] / 1000)

        return {
            "success": True,
            "language": lang,
            **stats,
            "gflops": round(gflops, 4),
            "memory_bandwidth_gbs": round(bandwidth, 4),
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


def run_action(payload: dict) -> dict:
    """按 `action` 分派一次请求, 返回结果载荷。

    **一次性路径与常驻服务共用这一份分派** (`__main__` 与 `runner_server.py` 都只调
    这里)。常驻服务省下的正是「每次调用新起一个进程」那 ~4 s (实测 `import torch`
    1.09 s + CUDA 上下文 ~0.44 s + ssh 握手 0.57 s), 见 doc 62。抄第二套分派的代价
    是两边慢慢漂开, 所以宁可把这份链整段搬进来, 也不在服务里再写一个 if 链。

    每个 action 用到的键都在自己那一支里读, 不提到分派之前: tests/test_runner_contract.py
    按「这一支读了哪些键」对照两个 runner, 提到外面会让所有 action 都算成读了全部键。
    """
    # 确保使用自动选择的最优 GPU
    _ensure_gpu_device()

    action = payload.get("action", "execute")

    if action == "gpu_info":
        result = get_gpu_info()
    elif action == "compile":
        result = run_compile(payload.get("code", ""))
    elif action == "execute":
        result = run_execute(payload.get("code", ""), payload.get("inputs", {}))
    elif action == "verify":
        result = run_verify(payload.get("code", ""), payload.get("inputs", {}),
                            payload.get("kernel_type", ""),
                            float(payload.get("max_error", 1e-5)),
                            precision=payload.get("precision", ""),
                            reference_source=payload.get("reference_source", ""))
    elif action in ("benchmark", "benchmark_with_baselines"):
        # 共享机器: 只在测量前采卡上占用情况 (间隔 1.2 s 采两次, 见 _sample_gpu_idle)。
        # 被其他租户占用会让加速比失真, 把实测占用与两次原始读数随结果返回, 由控制端
        # 标记污染, 而不是静默采信这个数字。
        util_before, other_procs, util_samples, state_before = _sample_gpu_idle()
        # I68 的另一半: 上面的采样是**时刻**性的 (util ~1s 窗口 / compute-apps 只见
        # 已建上下文的进程), 看不见还在 import 的对手; claim 生命期 = 整个进程。
        # 前后各扫一次: 窗口中途新到的对手由后一次抓住, 取 max 当「窗口内有对手」。
        peers_before = gpu_claim.scan_claims(_SELECTED_GPU_ID, exclude_pid=os.getpid())
        if action == "benchmark":
            result = run_benchmark(payload.get("code", ""), payload.get("inputs", {}),
                                   payload.get("warmup", 3), payload.get("repeat", 10))
        else:
            result = run_benchmark_with_baselines(
                payload.get("code", ""), payload.get("inputs", {}),
                payload.get("kernel_type", ""),
                payload.get("warmup", 10), payload.get("repeat", 30),
                precision=payload.get("precision", ""),
                baseline_priority=payload.get("baseline_priority") or None,
                reference_source=payload.get("reference_source", ""),
            )
        # D2: 窗口之后的时钟/throttle 读数。窗口之前那次由 _sample_gpu_idle 顺路带回,
        # 所以整个 L3 只多付一次 nvidia-smi 调用 (~130 ms), 不是三次。
        state_after = read_gpu_state()
        peers_after = gpu_claim.scan_claims(_SELECTED_GPU_ID, exclude_pid=os.getpid())
        if isinstance(result, dict):
            result["gpu_util_pct"] = util_before
            result["gpu_util_samples"] = util_samples
            result["gpu_other_procs"] = other_procs
            # harness 同卡声明 (与 other_procs 分工: 那个是任意进程几乎恒真只作提示,
            # 这个只数 harness 实例, 能当判据)。两个原始计数一并回报 —— 「max 是在
            # 防漏报还是在夸大」要靠 before/after 的差审计, 与 util_samples 同一处置。
            result["gpu_harness_peers"] = max(len(peers_before), len(peers_after))
            result["gpu_harness_peer_scans"] = [len(peers_before), len(peers_after)]
            result["selected_gpu"] = _SELECTED_GPU_ID
            for side, snap in (("before", state_before), ("after", state_after)):
                for key, val in snap.items():
                    if key != "util_pct":      # util 已经有自己的三个键
                        result[f"{key}_{side}"] = val
    else:
        result = {"success": False, "error": f"Unknown action: {action}"}

    return result


if __name__ == "__main__":
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        print(json.dumps({"success": False, "error": f"Invalid JSON: {e}"}))
        sys.exit(1)
    print(json.dumps(run_action(payload), ensure_ascii=False))
