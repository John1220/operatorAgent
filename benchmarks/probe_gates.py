"""三种 L2 门禁在同一份实测数据上的判定对照

门禁定义 (全部一手核实):
  OURS   remote_scripts/run_kernel.py:422   max_abs <= max_error          (纯 atol, 无相对项)
  KB     KernelBench eval.py:802-805        allclose(atol=t, rtol=t)      t=fp16:1e-2
  FIB    flashinfer-bench utils.py:113-118  (abs>atol)&(rel>rtol) 的元素占比
                                            matched_ratio >= rho

候选核:
  correct_fp32acc  fp32 累加 —— 好的 Triton kernel (tl.dot 默认 fp32 累加器) 应当如此
  correct_fp16acc  fp16 累加 —— 合法但精度差的实现, 不应被判作弊
  scaled_097       0.97 * ref —— 错误答案, 必须被拦下
"""
import numpy as np

M = K = N = 4096
MAX_ERROR = 5e-1     # test_nvidia_matmul.py:114
KB_TOL = 1e-2        # KernelBench eval.py:98  torch.float16
FIB_ATOL, FIB_RTOL, RHO = 1e-2, 1e-2, 1.0

rng = np.random.default_rng(12345)
A = (rng.standard_normal((M, K), dtype=np.float32) * 0.1).astype(np.float16)
B = (rng.standard_normal((K, N), dtype=np.float32) * 0.1).astype(np.float16)
A32, B32 = A.astype(np.float32), B.astype(np.float32)

ref = A32 @ B32                                   # cuBLAS fp16 GEMM ≈ fp32 累加

def fp16_accumulate(a, b, chunk=64):
    """分块 fp16 累加, 模拟用 fp16 累加器的 kernel"""
    acc = np.zeros((a.shape[0], b.shape[1]), dtype=np.float16)
    for i in range(0, a.shape[1], chunk):
        acc = (acc + (a[:, i:i+chunk].astype(np.float32) @
                      b[i:i+chunk, :].astype(np.float32)).astype(np.float16)).astype(np.float16)
    return acc.astype(np.float32)

cands = {
    "correct_fp32acc": ref.copy(),
    "correct_fp16acc": fp16_accumulate(A, B),
    "scaled_0.97":     ref * 0.97,
    "scaled_0.999":    ref * 0.999,
}

def gate_ours(a, e):
    return float(np.abs(a - e).max()) <= MAX_ERROR

def gate_kb(a, e):
    return bool(np.all(np.abs(a - e) <= KB_TOL + KB_TOL * np.abs(e)))

def gate_fib(a, e):
    abs_e = np.abs(a - e)
    rel_e = abs_e / (np.abs(e) + 1e-8)              # FIB eps=1e-8, 非 clamp_min(1e-12)
    bad = (abs_e > FIB_ATOL) & (rel_e > FIB_RTOL)
    return (1.0 - bad.mean()) >= RHO

print(f"4096^3 fp16 | OURS atol={MAX_ERROR} | KB atol=rtol={KB_TOL} | FIB atol=rtol={FIB_ATOL} rho={RHO}")
print(f"{'候选':<17}{'max_abs':>11}{'max_rel':>11}{'p99.9_abs':>11}   OURS    KB    FIB   期望")
print("-" * 84)
expect = {"correct_fp32acc": "PASS", "correct_fp16acc": "PASS",
          "scaled_0.97": "fail", "scaled_0.999": "PASS(边界)"}
for name, a in cands.items():
    abs_e = np.abs(a - ref)
    rel_e = abs_e / (np.abs(ref) + 1e-8)
    row = [f"{name:<17}", f"{abs_e.max():>11.3e}", f"{rel_e.max():>11.3e}",
           f"{np.percentile(abs_e,99.9):>11.3e}"]
    for g in (gate_ours, gate_kb, gate_fib):
        row.append(f"{'PASS' if g(a, ref) else 'fail':>7}")
    row.append(f"  {expect[name]}")
    print("".join(row))
