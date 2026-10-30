"""复现 remote_scripts/run_kernel.py:382-431 的 L2 判定, 量化 max_error=5e-1  admits 什么

配方严格照抄 runner:
  输入   _deserialize_inputs:261  -> (rng.standard_normal(shape, fp32) * 0.1).astype(fp16)
  参考   _reference_torch:296     -> A @ B
  比对   run_verify:415-422       -> float32 下 |actual-expected| 的 max, 与 max_error 比
  门禁   run_verify:422           -> success = max_abs_error <= max_error   (纯 atol, 无 rtol)
"""
import numpy as np

M = K = N = 4096
MAX_ERROR = 5e-1          # test_nvidia_matmul.py:114 实际传的值
SEED = 12345              # runner 由 shape+dtype 派生, 这里取任意固定值

rng = np.random.default_rng(SEED)
A = (rng.standard_normal((M, K), dtype=np.float32) * 0.1).astype(np.float16)
B = (rng.standard_normal((K, N), dtype=np.float32) * 0.1).astype(np.float16)

# cuBLAS fp16 GEMM 走 fp32 累加; 本地用 float32 累加复现同一语义
ref = (A.astype(np.float32) @ B.astype(np.float32))

print(f"shape {M}x{K}x{N} fp16, max_error={MAX_ERROR}")
print(f"参考输出 |ref|: max={np.abs(ref).max():.4f}  mean={np.abs(ref).mean():.4f}  "
      f"p99={np.percentile(np.abs(ref),99):.4f}")
print(f"→ 门禁 5e-1 相当于允许 {MAX_ERROR/np.abs(ref).max()*100:.1f}% 的最大幅值偏差\n")

print(f"{'缩放因子':>10} {'max_abs_err':>13} {'判定':>8} {'等价 rtol':>11}")
print("-" * 48)
for scale in (1.0, 0.99, 0.97, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.0):
    actual = ref * scale
    abs_err = np.abs(actual - ref)
    mae = float(abs_err.max())
    denom = np.maximum(np.abs(ref), 1e-12)      # runner: clamp_min(1e-12)
    mre = float((abs_err / denom).max())
    verdict = "PASS" if mae <= MAX_ERROR else "fail"
    tag = "  ← 正确核" if scale == 1.0 else ("  ← 全零核!" if scale == 0.0 else "")
    print(f"{scale:>10.2f} {mae:>13.4e} {verdict:>8} {mre:>11.4f}{tag}")

# 二分找出仍能通过的最大偏差
lo, hi = 0.0, 1.0
for _ in range(40):
    mid = (lo + hi) / 2
    if float(np.abs(ref * mid - ref).max()) <= MAX_ERROR:
        hi = mid
    else:
        lo = mid
print(f"\n仍能通过门禁的最小缩放因子 = {hi:.6f}")
print(f"即: 一个把结果整体缩小 {(1-hi)*100:.2f}% 的核, 在 4096^3 fp16 matmul 上判为「功能验证通过」")
