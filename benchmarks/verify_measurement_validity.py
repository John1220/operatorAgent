"""真机验证测量有效性三态 (docs/research/00 缺口 1)

在控制端运行 (会触发 deploy(), 需要能连到 GPU 机器):
    python3 benchmarks/verify_measurement_validity.py

要验证的是 docs/research/01 第四节那张表在真机上成立:
  layernorm 1024×768   访存 3.1 MB,  roofline 下界 0.86 µs  → 应判 uninformative
  layernorm 16384×6144 访存 402.7 MB, roofline 下界 109.4 µs → 应脱离地板
  matmul 4096³         访存 100 MB,  roofline 下界 27 µs    → 应判 valid

第一行是项目已经踩过的那个坑: 改动前它报出「0.65x / 带宽利用率 5.4% /
memory-bound 需优化」, Conductor 拿这套数字去禁用配方并累计停滞轮次, 而这三个数
描述的都是 kernel launch 开销, 不是核的质量。

D4 之后每个 case 还顺带跑一遍四态判决与 roofline 收敛判定 (同一份真机 metrics),
看达成率的量级和带宽相对实际散布是宽还是窄 —— 那两条判据决定了循环会不会提前停。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from operator_agent.remote_executor import RemoteExecutor
from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec, TargetSpec,
                                    TaskComplexity, TaskSchema, TaskType)
from operator_agent.verifier import Verifier
from operator_agent.measurement import (
    DISPERSION_DEGRADED, DISPERSION_UNINFORMATIVE, INTRUDER_MAX_OVER_MEDIAN,
    NOISE_BAND, ROOFLINE_CONVERGED_PCT, Validity, binding_attainment,
    roofline_converged, verdict,
)

TRITON_LN = '''
import torch, triton, triton.language as tl

@triton.jit
def _ln(x_ptr, w_ptr, b_ptr, y_ptr, n_rows, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / n_cols
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * n_cols + cols, (xc * rstd * w + b).to(y_ptr.dtype.element_ty), mask=mask)

def kernel(X, weight, bias):
    n_rows, n_cols = X.shape
    Y = torch.empty_like(X)
    _ln[(n_rows,)](X, weight, bias, Y, n_rows, n_cols, 1e-5,
                   BLOCK=triton.next_power_of_2(n_cols))
    return Y
'''

TRITON_MM = '''
import torch, triton, triton.language as tl

@triton.jit
def _mm(a_ptr, b_ptr, c_ptr, M, N, K,
        sam, sak, sbk, sbn, scm, scn,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    off_m = pid_m * BM + tl.arange(0, BM)
    off_n = pid_n * BN + tl.arange(0, BN)
    a_ptrs = a_ptr + off_m[:, None] * sam + tl.arange(0, BK)[None, :] * sak
    b_ptrs = b_ptr + tl.arange(0, BK)[:, None] * sbk + off_n[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K, BK):
        acc += tl.dot(tl.load(a_ptrs), tl.load(b_ptrs))
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    tl.store(c_ptr + off_m[:, None] * scm + off_n[None, :] * scn, acc.to(tl.float16))

def kernel(A, B):
    M, K = A.shape
    N = B.shape[1]
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)
    BM, BN, BK = 64, 64, 32
    _mm[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1),
        C.stride(0), C.stride(1), BM=BM, BN=BN, BK=BK)
    return C
'''


def task(kernel_type, precision=Precision.FP16):
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type=kernel_type, complexity=TaskComplexity.MEDIUM),
        target=TargetSpec(hardware="nvidia_gpu", dsl="triton", compute_units=78, warp_size=32),
        constraints=ConstraintSpec(precision=precision, min_speedup=0.0, max_error=1e-1))


def ln_inputs(rows, cols, dtype=np.float16):
    rng = np.random.default_rng(11)
    return {"X": rng.standard_normal((rows, cols)).astype(dtype),
            "weight": np.ones(cols, dtype=dtype),
            "bias": np.zeros(cols, dtype=dtype)}


def mm_inputs(n):
    rng = np.random.default_rng(7)
    return {"A": (rng.standard_normal((n, n)) * 0.1).astype(np.float16),
            "B": (rng.standard_normal((n, n)) * 0.1).astype(np.float16)}


backend = RemoteExecutor(mode="ssh")
print("=== deploy ===")
if not backend.deploy():
    sys.exit(1)
v = Verifier(backend)
print("=== 探测到的硬件 ===")
peak = v._detect_theoretical_peak()
print(f"  name={peak['name']!r} cc={peak['compute_capability']!r} "
      f"launch_floor_us={peak['launch_floor_us']}")
if peak["launch_floor_us"] is None:
    print("  ⚠️ 地板为 None: hardware_caps 未命中这台设备, 地板支配检查会被跳过")
# 缺口 3: roofline 分母是哪一行、凭什么。未核口径的行会关掉「已到 roofline」出口
# (measurement.roofline_converged), 所以下面每个 case 的 converged 都要连着这行读。
print(f"  分母行 [{peak['peak_table_key']}] fp16={peak['fp16_tflops']} TFLOPS / "
      f"bw={peak['memory_bandwidth_gbs']} GB/s — {peak['peak_source']} "
      f"(trusted={peak['peak_trusted']})")

CASES = [
    ("layernorm 1024×768 (docs/research/01 判定为无意义的那一档)",
     TRITON_LN, "layernorm", ln_inputs(1024, 768), 3.1, 0.86),
    ("layernorm 16384×6144 (docs/research/01 推荐的 shape)",
     TRITON_LN, "layernorm", ln_inputs(16384, 6144), 402.7, 109.4),
    ("matmul 4096³ (docs/research/01 判定为可信的那一档)",
     TRITON_MM, "matmul", mm_inputs(4096), 100.0, 27.0),
]

for label, code, kernel_type, inputs, traffic_mb, roof_us in CASES:
    r = v._verify_performance(code, inputs, 0.0, task(kernel_type))
    m = r.metrics
    print(f"\n{label}")
    print(f"  访存 {traffic_mb} MB, roofline 下界 {roof_us} µs")
    print(f"  候选 median = {m.get('median_time_ms')} ms   "
          f"基线 = {m.get('baseline_time_ms')} ms ({m.get('baseline_display')})")
    print(f"  launch_dominance = {m.get('launch_dominance')}  "
          f"(地板 {m.get('launch_floor_us')} µs)")
    print(f"  → validity = {m.get('measurement_validity')}   "
          f"can_gate={m.get('measurement_can_gate')} "
          f"can_disable_recipe={m.get('measurement_can_disable_recipe')}")
    for reason in m.get("measurement_validity_reasons") or []:
        print(f"    - {reason}")
    print(f"  speedup={m.get('speedup_vs_baseline')}  "
          f"runs={m.get('runs')}(请求 {m.get('runs_requested')})  "
          f"contaminated={m.get('benchmark_contaminated')} "
          f"(util={m.get('gpu_util_pct')}%, 其他进程={m.get('gpu_other_procs')})")

    # ---- 离散度门是否真的通了 ----
    # 缺口 1 原来的失效方式不是「没算」而是「算了没人读」。新加的 p25/p75/两个
    # dispersion 字段走的正是同一条 bench["candidate"] → metrics 的搬运路径, 所以
    # 必须在真机上确认它们**到了**, 否则整个门是空转的。
    disp_iqr = m.get("dispersion_iqr_over_median")
    disp_max = m.get("dispersion_max_over_median")
    print(f"  样本分布: p25={m.get('p25_time_ms')} p75={m.get('p75_time_ms')} "
          f"min={m.get('min_time_ms')} max={m.get('max_time_ms')} ms")
    print(f"  iqr/median = {disp_iqr}   max/median = {disp_max}   "
          f"(判据阈值 {DISPERSION_DEGRADED} / {DISPERSION_UNINFORMATIVE}; "
          f"抢占阈值 {INTRUDER_MAX_OVER_MEDIAN} 仅诊断, 不改档位)")
    if disp_iqr is None or disp_max is None:
        print("  ❌ runner 没把分位数搬进 metrics —— 离散度门在此路径上空转")
    if m.get("measurement_intruder_suspected") and \
            m.get("measurement_validity") == "valid":
        print("  ✓ 抢占观察已上报但未改变档位 (符合当前设计)")

    # ---- D4: 拿本轮的真机数字跑一遍四态判决与收敛判定 ----
    #
    # 这两个判据只在离线单测里见过合成的 metrics, 到这里才算吃过真数。要看的不是
    # 「对不对」而是「量级合不合理」: 达成率是不是真在 0~100% 附近、判决带宽是不是
    # 比这一档的实际散布宽 (窄了会把噪声当改善, 宽了会永远判 SAME)。
    att = binding_attainment(m)
    print(f"  绑定轴达成率 = {att}%   roofline_converged = {roofline_converged(m)} "
          f"(阈值 {ROOFLINE_CONVERGED_PCT:.0f}%)")
    validity = Validity(m.get("measurement_validity", Validity.VALID.value))
    band = NOISE_BAND[validity]
    cur = float(m.get("gflops", 0) or 0)
    for tag, prev in (("与上一轮持平", cur), ("上一轮差 3 倍", cur / 3.0),
                      ("上一轮好 1.5 倍", cur * 1.5)):
        vd, why, _used_band = verdict(cur, prev, validity)
        print(f"    verdict(cur={cur:.1f} vs {tag}) = {vd.value}  "
              f"({why}; 本档带宽 ±{band:.2%})")
