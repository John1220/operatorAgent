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
  两者都由 SSHExecutor.deploy() 与本文件一起推送; 在仓库内直接跑本文件时回退到
  operator_agent/ 下找。
"""

import sys
import json
import time
import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
try:
    import tolerance
    from anticheat import shield_torch
except ImportError:          # 仓库内直接运行: 模块还在 operator_agent/ 下
    sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "operator_agent"))
    import tolerance
    from anticheat import shield_torch


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
        if list(actual.shape) != list(expected.shape):
            return {"success": False, "error": "output shape mismatch",
                    "output_shape": list(actual.shape), "expected_shape": list(expected.shape)}
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
        if total:
            # EPS=1e-8 而不是原先的 clamp_min(1e-12): 后者会让近零参考点上的
            # rel_err 再放大 4 个数量级。
            rel_error = abs_error / (expected_f.abs() + tolerance.EPS)
            # AND: 只有同时超绝对阈值和相对阈值才算坏点。这是不误杀合法 fp16
            # 累加核的关键 —— 那种核在近零参考处的 rel_err 可达 2.1e4。
            bad = (abs_error > atol) & (rel_error > rtol)
            n_bad = int(bad.sum().item())
            max_abs_error = float(abs_error.max().item())
            max_rel_error = float(rel_error.max().item())
            has_nonfinite = bool((~torch.isfinite(actual_f)).any().item())
        else:
            n_bad, max_abs_error, max_rel_error, has_nonfinite = 0, 0.0, 0.0, False

        passed, matched_ratio = tolerance.evaluate(n_bad, total, has_nonfinite, rho)
        if has_nonfinite:
            reason = "输出含 NaN/Inf"
        elif not passed:
            reason = (f"matched_ratio {matched_ratio:.6f} < {rho} "
                      f"({n_bad}/{total} 个元素同时超 atol={atol:.1e} 与 rtol={rtol:.1e}; "
                      f"max_abs={max_abs_error:.2e}, max_rel={max_rel_error:.2e})")
        else:
            reason = ""
        return {
            "success": passed,
            "reference_type": reference_type,
            "output_shape": list(actual.shape),
            "compared_elements": total,
            "coverage": 1.0,
            "max_error": max_abs_error,
            "max_rel_error": max_rel_error,
            "matched_ratio": matched_ratio,
            "atol": atol,
            "rtol": rtol,
            "required_matched_ratio": rho,
            "bad_elements": n_bad,
            "error": reason,
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
    i = 0
    while i < n:
        k = min(block, n - i)
        # 一块 = 一次 shield 进出, 且 shield 包的是**整块**而不是每一次调用
        # (与本文件 541-543 的原则一致); 事件窗口在 _timed_runs 内部逐样本开闭,
        # 所以 setattr 群不进测量值。
        with shield_torch():
            cand_times.extend(_timed_runs(cand_fn, k, device_is_cuda))
        base_times.extend(_timed_runs(base_fn, k, device_is_cuda))
        i += k

    cand_stats = _sample_stats(cand_times)
    base_stats = _sample_stats(base_times)
    ratios = [b / c for c, b in zip(cand_times, base_times) if c > 0]
    paired = {"paired_runs": len(ratios), "paired_block_samples": block}
    if ratios:
        rs = sorted(ratios)
        med = _percentile(rs, 50)
        p25, p75 = _percentile(rs, 25), _percentile(rs, 75)
        # median-of-ratios: 逐轮配对后的比值中位数, 与 ratio-of-medians 互补
        paired["speedup_paired_median"] = round(med, 4)
        paired["speedup_paired_iqr_over_median"] = (
            round((p75 - p25) / med, 4) if med > 0 else None)
    cand_stats["runs_requested"] = repeat
    base_stats["runs_requested"] = repeat
    return cand_stats, base_stats, paired


def run_benchmark_with_baselines(code: str, inputs: dict, kernel_type: str,
                                 warmup: int = 10, repeat: int = 30,
                                 precision: str = "",
                                 baseline_priority=None) -> dict:
    """Time the candidate against exactly one baseline, interleaved, on device.

    `baseline_priority` is the control side's ordered preference; the first
    baseline that actually runs is the only one measured. Ordering matters:
    it encodes which comparison is fair (fp16 candidate vs cuBLAS fp16, etc).
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
        # 共享机器: 只在测量前采卡上占用情况 (间隔 1.2 s 采两次, 见 _sample_gpu_idle)。
        # 被其他租户占用会让加速比失真, 把实测占用与两次原始读数随结果返回, 由控制端
        # 标记污染, 而不是静默采信这个数字。
        util_before, other_procs, util_samples, state_before = _sample_gpu_idle()
        if action == "benchmark":
            result = run_benchmark(code, inputs,
                                   payload.get("warmup", 3), payload.get("repeat", 10))
        else:
            result = run_benchmark_with_baselines(
                code, inputs, payload.get("kernel_type", ""),
                payload.get("warmup", 10), payload.get("repeat", 30),
                precision=payload.get("precision", ""),
                baseline_priority=payload.get("baseline_priority") or None,
            )
        # D2: 窗口之后的时钟/throttle 读数。窗口之前那次由 _sample_gpu_idle 顺路带回,
        # 所以整个 L3 只多付一次 nvidia-smi 调用 (~130 ms), 不是三次。
        state_after = read_gpu_state()
        if isinstance(result, dict):
            result["gpu_util_pct"] = util_before
            result["gpu_util_samples"] = util_samples
            result["gpu_other_procs"] = other_procs
            result["selected_gpu"] = _SELECTED_GPU_ID
            for side, snap in (("before", state_before), ("after", state_after)):
                for key, val in snap.items():
                    if key != "util_pct":      # util 已经有自己的三个键
                        result[f"{key}_{side}"] = val
    else:
        result = {"success": False, "error": f"Unknown action: {action}"}

    print(json.dumps(result, ensure_ascii=False))
