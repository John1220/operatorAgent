"""标定 L2 容差表: 我们的 cuBLAS fp16 参考自身离真值有多远?

在远端 GPU 机器上运行 (需要 torch/triton):
    ssh <别名> 'cd ~/operator_agent && OPERATOR_AGENT_GPU=2 \
        ~/.local/bin/uv run python -' < benchmarks/calibration/calibrate_l2_tolerance.py

结论记录在 docs/research/30-L2数值门真机验证.md 第二、三节。

原题:: 我们的 cuBLAS fp16 参考自身离真值有多远?

docs/research/12 第五节留的未决问题: 若 allow_fp16_reduced_precision_reduction
为 True, 则 fp16 GEMM 的中间归约可能就在 fp16 里做, 那么「候选与参考之差」里有
一部分是参考的误差而不是候选的 —— 这直接决定容差表能否照抄 KernelBench 的 1e-2。
"""
import os, torch, numpy as np

print("torch", torch.__version__, "| triton", __import__("triton").__version__)
print("\n=== PyTorch 后端默认值 (doc 12 未决问题) ===")
print("  matmul.allow_tf32                        =", torch.backends.cuda.matmul.allow_tf32)
print("  cudnn.allow_tf32                         =", torch.backends.cudnn.allow_tf32)
print("  matmul.allow_fp16_reduced_precision_red  =", torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
print("  matmul.allow_bf16_reduced_precision_red  =", torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)

dev = torch.device("cuda")
M = K = N = 4096
rng = np.random.default_rng(12345)
# 与 run_kernel.py:_deserialize_inputs 同一配方: standard_normal × 0.1 → fp16
A = torch.from_numpy((rng.standard_normal((M, K), dtype=np.float32) * 0.1).astype(np.float16)).to(dev)
B = torch.from_numpy((rng.standard_normal((K, N), dtype=np.float32) * 0.1).astype(np.float16)).to(dev)

with torch.no_grad():
    ref_fp16 = A @ B                                  # _reference_torch 走的就是这条
    truth = (A.float() @ B.float())                   # fp32 累加 = 真值
    cand_fp32acc = (A.float() @ B.float()).half()     # 一个「正确的」fp16 kernel 的输出

def stats(name, actual, expected):
    a = actual.detach().float(); e = expected.detach().float()
    abs_e = (a - e).abs()
    rel_e = abs_e / (e.abs() + 1e-8)                  # tolerance.EPS
    for atol, rtol, tag in ((1e-2, 1e-2, "表值 fp16"), (1e-5, 1e-2, "driver 5e-1 被 min 压到表值")):
        bad = (abs_e > atol) & (rel_e > rtol)
        ratio = 1.0 - bad.float().mean().item()
        print(f"    [{tag}] atol={atol:.0e} rtol={rtol:.0e} → bad={int(bad.sum())}/{abs_e.numel()} "
              f"matched_ratio={ratio:.6f} {'PASS' if ratio >= 1.0 else 'FAIL'}")
    print(f"  {name}: max_abs={abs_e.max().item():.4e} max_rel={rel_e.max().item():.4e} "
          f"|ref|max={e.abs().max().item():.4f}")

print(f"\n=== {M}x{K}x{N} fp16, 输入 standard_normal×0.1 ===")
print("\n[A] 参考 (cuBLAS fp16) vs fp32 累加真值 —— 参考自身的误差")
stats("ref_fp16 vs truth", ref_fp16, truth)
print("\n[B] 一个正确的 fp32 累加候选 vs 我们的参考 —— 这才是门禁实际看到的")
stats("cand_fp32acc vs ref_fp16", cand_fp32acc, ref_fp16)
print("\n[C] 整体缩放 0.97 的错核 vs 参考 —— 必须被拦下")
stats("0.97*ref vs ref", ref_fp16 * 0.97, ref_fp16)
print("\n[D] 整体缩放 0.999 的错核 vs 参考 —— doc 12 认为该量级在噪声内, 放行是对的")
stats("0.999*ref vs ref", ref_fp16 * 0.999, ref_fp16)
