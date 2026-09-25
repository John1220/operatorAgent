"""Benchmark: Triton Official MatMul vs cuBLAS

Compares the official Triton matmul implementation from:
  triton/python/tutorials/03-matrix-multiplication.py
against cuBLAS (PyTorch @) on various matrix sizes.

Key techniques in the official implementation:
  1. @triton.autotune with 18 configs (BLOCK_SIZE_M/N/K, num_warps, num_stages)
  2. L2 cache grouping (GROUP_SIZE_M=8) for better data reuse
  3. tl.assume() for pointer optimization
  4. Pointer advance (a_ptrs += BLOCK_SIZE_K * stride_ak)
  5. 3-parameter tl.dot(a, b, accumulator) for fused accumulation
  6. Modulo (%) for M/N boundary handling (compiler optimizes to bitwise AND)
  7. K-dimension mask: offs_k < K - k * BLOCK_SIZE_K

Usage:
    python benchmarks/test_matmul.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks._remote_bench import (run_remote_benchmark, print_benchmark_table,
                                     resolve_ssh_config, ssh_label)


# ─── Official Triton MatMul Implementation ──────────────────────────────
# Extracted from triton/python/tutorials/03-matrix-multiplication.py
# Uses: autotune (18 configs), L2 cache grouping, tl.assume, pointer advance,
# 3-param tl.dot, modulo boundaries, optimized K-mask

TRITON_MATMUL_CODE = '''
import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        # FP8-friendly configs (also work for fp16)
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # Additional high-tile configs
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if ACTIVATION == "leaky_relu":
        accumulator = tl.where(accumulator >= 0, accumulator, 0.01 * accumulator)
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def kernel(A, B):
    """Official Triton matmul (FP16) with autotune"""
    assert A.shape[1] == B.shape[0], "Incompatible dimensions"
    assert A.is_contiguous(), "Matrix A must be contiguous"
    M, K = A.shape
    K2, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        ACTIVATION="",
    )
    return C
'''


# ─── cuBLAS Reference (PyTorch @) ──────────────────────────────────────

CUBLAS_REFERENCE_CODE = '''
import torch

def reference(A, B):
    """cuBLAS matmul via PyTorch @ operator (uses cuBLAS internally)"""
    return A @ B
'''


# ─── Test Shapes ─────────────────────────────────────────────────────────

# Covering: small, medium, large, square, and non-square
# All are common LLM dimensions (hidden_size, intermediate_size, etc.)
TEST_SHAPES = [
    # (M, K, N)
    (1024, 1024, 1024),      # Square 1K
    (2048, 2048, 2048),      # Square 2K
    (4096, 4096, 4096),      # Square 4K
    (1024, 768, 1024),       # BERT-like
    (1024, 3072, 1024),      # GPT-3 FFN
    (2048, 768, 2048),       # Larger batch
    (4096, 512, 4096),       # Tall-skinny
    (512, 4096, 512),       # Short-wide
    (1024, 4096, 1024),      # Large K
    (2048, 1024, 2048),      # Balanced large
]


def main():
    print("=" * 70)
    print("  Benchmark: Triton Official MatMul vs cuBLAS (PyTorch @)")
    print("  Source: triton/python/tutorials/03-matrix-multiplication.py")
    print("  Techniques: autotune(18 configs), L2 cache grouping, tl.assume,")
    print("              pointer advance, 3-param tl.dot, modulo boundaries, optimized K-mask")
    print("=" * 70)

    ssh_host, ssh_user, ssh_port, _, _ = resolve_ssh_config()

    print(f"\n  Remote GPU: {ssh_label(ssh_host, ssh_port)}")
    print(f"  Test shapes: {len(TEST_SHAPES)} configurations")
    print(f"  Precision: FP16 (forward pass only)")
    print()

    results = []
    shape_info = []  # Store (M, K, N) for each result
    for M, K, N in TEST_SHAPES:
        print(f"  Benchmarking M={M:5d}, K={K:5d}, N={N:5d}...", end=" ", flush=True)

        inputs = {
            "A": {"shape": [M, K], "seed": 42},
            "B": {"shape": [K, N], "seed": 789},
        }

        result = run_remote_benchmark(
            operator="matmul",
            kernel_code=TRITON_MATMUL_CODE,
            reference_code=CUBLAS_REFERENCE_CODE,
            inputs=inputs,
            dtype="fp16",
            ssh_host=ssh_host,
            ssh_user=ssh_user,
            ssh_port=ssh_port,
            warmup=10,
            repeat=50,
            do_backward=False,
        )

        results.append(result)
        shape_info.append((M, K, N))

        if result.passed:
            # TFLOPS = FLOPs / time / 1e12, where FLOPs = 2*M*N*K (1 mul + 1 add per element)
            flops = 2.0 * M * N * K
            triton_tflops = flops / (result.triton_fwd_ms / 1000) / 1e12
            pytorch_tflops = flops / (result.pytorch_fwd_ms / 1000) / 1e12 if result.pytorch_fwd_ms > 0 else 0
            speedup = f"{result.fwd_speedup:.3f}x" if result.fwd_speedup > 0 else "N/A"
            triton_ms = f"{result.triton_fwd_ms:.4f}ms"
            pytorch_ms = f"{result.pytorch_fwd_ms:.4f}ms"
            print(f"OK | Triton: {triton_ms} ({triton_tflops:.1f} TF), cuBLAS: {pytorch_ms} ({pytorch_tflops:.1f} TF), Speedup: {speedup}")
        else:
            print(f"FAILED: {result.error[:80]}")

    # Print summary table
    print_benchmark_table(results, baseline_name="cuBLAS (PyTorch @)")

    # Enhanced summary with TFLOPS
    passed = [(r, m, k, n) for r, (m, k, n) in zip(results, shape_info) if r.passed]
    if passed:
        avg_speedup = sum(r.fwd_speedup for r, _, _, _ in passed) / len(passed)
        min_speedup = min(r.fwd_speedup for r, _, _, _ in passed)
        max_speedup = max(r.fwd_speedup for r, _, _, _ in passed)

        # A100 FP16 Tensor Core peak: 312 TFLOPS
        peak_tflops = 312.0

        # Calculate TFLOPS for each shape
        tflops_data = []
        for r, M, K, N in passed:
            flops = 2.0 * M * N * K
            triton_tf = flops / (r.triton_fwd_ms / 1000) / 1e12
            cublas_tf = flops / (r.pytorch_fwd_ms / 1000) / 1e12
            tflops_data.append((M, K, N, triton_tf, cublas_tf, r.fwd_speedup))

        max_triton_tflops = max(d[3] for d in tflops_data)
        max_cublas_tflops = max(d[4] for d in tflops_data)

        print(f"\n  Summary (forward pass, FP16):")
        print(f"    Average speedup (Triton / cuBLAS): {avg_speedup:.3f}x")
        print(f"    Min speedup:     {min_speedup:.3f}x")
        print(f"    Max speedup:     {max_speedup:.3f}x")
        print(f"    Peak Triton:  {max_triton_tflops:.1f} TFLOPS ({max_triton_tflops/peak_tflops*100:.1f}% of A100 312T peak)")
        print(f"    Peak cuBLAS:  {max_cublas_tflops:.1f} TFLOPS ({max_cublas_tflops/peak_tflops*100:.1f}% of A100 312T peak)")

        # Per-shape TFLOPS table
        print(f"\n  Per-Shape TFLOPS Breakdown:")
        print(f"  {'Shape (M,K,N)':<20} {'Triton TF':<12} {'cuBLAS TF':<12} {'Speedup':<10} {'Triton %peak':<14} {'cuBLAS %peak':<14}")
        print(f"  {'-'*82}")
        for M, K, N, triton_tf, cublas_tf, sp in tflops_data:
            print(f"  ({M:5d},{K:5d},{N:5d})   {triton_tf:<12.1f} {cublas_tf:<12.1f} {sp:<10.3f}  {triton_tf/peak_tflops*100:<12.1f}%  {cublas_tf/peak_tflops*100:<12.1f}%")

        # Find the best shape for Triton (best absolute TFLOPS)
        best = max(tflops_data, key=lambda d: d[3])
        worst = min(tflops_data, key=lambda d: d[3])
        print(f"\n  Best Triton shape: ({best[0]},{best[1]},{best[2]}) = {best[3]:.1f} TFLOPS ({best[5]:.3f}x cuBLAS)")
        print(f"  Worst Triton shape: ({worst[0]},{worst[1]},{worst[2]}) = {worst[3]:.1f} TFLOPS ({worst[5]:.3f}x cuBLAS)")

        if avg_speedup > 0.9:
            print(f"\n    ⭐ EXCELLENT: Triton official is within 10% of cuBLAS!")
        elif avg_speedup > 0.7:
            print(f"\n    ✅ GOOD: Triton official reaches {avg_speedup*100:.0f}% of cuBLAS performance")
        elif avg_speedup > 0.5:
            print(f"\n    ⚠️  MODERATE: Triton official reaches {avg_speedup*100:.0f}% of cuBLAS")
        else:
            print(f"\n    ❌ BELOW TARGET: Triton official only reaches {avg_speedup*100:.0f}% of cuBLAS")

        # Target for Agent: 80% of Triton's performance
        agent_target = avg_speedup * 0.8
        print(f"\n    Target for Agent (0.8x of Triton): {agent_target:.3f}x cuBLAS")
        print(f"    (Agent should reach {agent_target*100:.0f}% of cuBLAS performance)")

    return results


if __name__ == "__main__":
    main()
