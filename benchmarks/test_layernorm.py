"""Benchmark: Triton Official LayerNorm vs PyTorch Eager

Compares the official Triton LayerNorm implementation from:
  triton/python/tutorials/05-layer-norm.py
against PyTorch's native F.layer_norm (cuDNN kernel).

Usage:
    python benchmarks/test_layernorm.py

Output: Speedup ratio (PyTorch_time / Triton_time) for various shapes.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks._remote_bench import run_remote_benchmark, print_benchmark_table


# ─── Official Triton LayerNorm Forward Kernel ────────────────────────────
# Extracted from triton/python/tutorials/05-layer-norm.py
# Key techniques:
#   1. Multi-block reduction (BLOCK_SIZE=2048, process N in chunks)
#   2. fp32 accumulation for numerical stability
#   3. Fused normalize + scale + bias in second pass
#   4. Heuristic num_warps based on BLOCK_SIZE

TRITON_LAYERNORM_CODE = '''
import triton
import triton.language as tl
import torch

# Optimized forward-only LayerNorm (removes mean/rstd global writes, num_ctas=1 barrier)
@triton.jit
def _layer_norm_fwd_inference(
    X, Y, W, B,
    stride, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute mean
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Normalize and apply linear transformation (fused)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        b = tl.load(B + cols, mask=mask).to(tl.float32)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)


def kernel(X, weight, bias, eps=1e-5):
    """Optimized Triton LayerNorm for inference (forward only)"""
    orig_shape = X.shape
    N = orig_shape[-1]
    x_arg = X.reshape(-1, N).contiguous()
    M = x_arg.shape[0]
    y = torch.empty_like(x_arg)

    MAX_FUSED_SIZE = 65536 // X.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_SIZE:
        BLOCK_SIZE = triton.next_power_of_2(N)
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

    _layer_norm_fwd_inference[(M,)](
        x_arg, y, weight, bias,
        x_arg.stride(0), N, eps,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
    )
    return y.reshape(orig_shape)
'''


# ─── PyTorch Reference ───────────────────────────────────────────────────

PYTORCH_REFERENCE_CODE = '''
import torch
import torch.nn.functional as F

def reference(X, weight, bias, eps=1e-5):
    """PyTorch native LayerNorm (cuDNN kernel on CUDA)"""
    return F.layer_norm(X, (X.shape[-1],), weight, bias, eps)
'''


# ─── Test Shapes ─────────────────────────────────────────────────────────

TEST_SHAPES = [
    # (M, N) - covering small, medium, and large hidden sizes
    (1024, 768),      # BERT-base hidden size
    (1024, 1024),     # Square
    (1024, 2048),     # Larger hidden
    (1024, 4096),     # Large hidden
    (4096, 768),      # More rows
    (2048, 3072),     # GPT-3 style
]


def main():
    print("=" * 70)
    print("  Benchmark: Triton Official LayerNorm vs PyTorch Eager (cuDNN)")
    print("  Source: triton/python/tutorials/05-layer-norm.py")
    print("=" * 70)

    # Connect to remote GPU
    ssh_host = os.environ.get("REMOTE_SSH_HOST", "152.136.252.171")
    ssh_user = os.environ.get("REMOTE_SSH_USER", "zhanghan")
    ssh_port = int(os.environ.get("REMOTE_SSH_PORT", "22"))

    print(f"\n  Remote GPU: {ssh_host}:{ssh_port}")
    print(f"  Test shapes: {len(TEST_SHAPES)} configurations")
    print(f"  Precision: FP16 (forward pass only)")
    print()

    results = []
    for M, N in TEST_SHAPES:
        print(f"  Benchmarking M={M}, N={N}...", end=" ", flush=True)

        inputs = {
            "X": {"shape": [M, N], "seed": 42},
            "weight": {"shape": [N], "seed": 123},
            "bias": {"shape": [N], "seed": 456},
        }

        result = run_remote_benchmark(
            operator="layernorm",
            kernel_code=TRITON_LAYERNORM_CODE,
            reference_code=PYTORCH_REFERENCE_CODE,
            inputs=inputs,
            dtype="fp16",
            ssh_host=ssh_host,
            ssh_user=ssh_user,
            ssh_port=ssh_port,
            warmup=10,
            repeat=50,
            do_backward=False,  # Forward pass comparison only
        )

        results.append(result)
        if result.passed:
            speedup = f"{result.fwd_speedup:.3f}x" if result.fwd_speedup > 0 else "N/A"
            triton_ms = f"{result.triton_fwd_ms:.4f}ms"
            pytorch_ms = f"{result.pytorch_fwd_ms:.4f}ms"
            print(f"OK | Triton: {triton_ms}, PyTorch: {pytorch_ms}, Speedup: {speedup}")
        else:
            print(f"FAILED: {result.error[:80]}")

    # Print summary
    print_benchmark_table(results, baseline_name="PyTorch Eager(CUDA) - cuDNN LayerNorm")

    # Analysis
    passed = [r for r in results if r.passed]
    if passed:
        avg_speedup = sum(r.fwd_speedup for r in passed) / len(passed)
        min_speedup = min(r.fwd_speedup for r in passed)
        max_speedup = max(r.fwd_speedup for r in passed)
        print(f"\n  Summary (forward pass):")
        print(f"    Average speedup: {avg_speedup:.3f}x (Triton vs PyTorch cuDNN)")
        print(f"    Min speedup:     {min_speedup:.3f}x")
        print(f"    Max speedup:     {max_speedup:.3f}x")
        if avg_speedup > 1.0:
            print(f"    Verdict: Triton OFFICIAL is {avg_speedup:.1f}% FASTER than PyTorch cuDNN")
        else:
            print(f"    Verdict: Triton OFFICIAL is {(1-avg_speedup)*100:.1f}% SLOWER than PyTorch cuDNN")
        print(f"    Target for Agent: 0.8x of this speedup (i.e., Agent should reach {avg_speedup*0.8:.3f}x)")
    else:
        print("\n  All benchmarks failed. Check remote GPU connection.")

    return results


if __name__ == "__main__":
    main()
