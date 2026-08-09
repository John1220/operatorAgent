"""Benchmark: Triton Official Fused Attention vs PyTorch SDPA (cuDNN/Flash)

Compares the official Triton fused attention implementation from:
  triton/python/tutorials/06-fused-attention.py
against PyTorch's scaled_dot_product_attention (uses cuDNN Flash Attention backend).

Key techniques in the official implementation:
  1. Flash Attention v2 algorithm (online softmax with exp2)
  2. Tiled K/V streaming (blocked attention)
  3. @triton.autotune with multiple configs
  4. Online normalization (m_i, l_i tracking)
  5. Fused QK^T + softmax + PV in a single kernel

Usage:
    python benchmarks/test_attention.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks._remote_bench import run_remote_benchmark, print_benchmark_table


# ─── Official Triton Fused Attention (Forward Only) ─────────────────────
# Simplified from triton/python/tutorials/06-fused-attention.py
# A100 (SM 80) - no tensor descriptor support, no warp specialization
# Uses raw pointer-based access with autotune

TRITON_ATTENTION_CODE = '''
import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128}, num_stages=2, num_warps=8),
    ],
    key=['N_CTX', 'HEAD_DIM'],
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, O,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    sm_scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    # Initialize online softmax accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 0.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Load Q: [BLOCK_M, HEAD_DIM]
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs)

    # Loop over K/V blocks (Flash Attention v2 tiled)
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_offs = start_n + tl.arange(0, BLOCK_N)

        # Load K transposed: [HEAD_DIM, BLOCK_N]
        k_ptrs = K + k_offset + offs_d[:, None] * stride_kk + k_offs[None, :] * stride_kn
        k = tl.load(k_ptrs)

        # QK^T: [BLOCK_M, HEAD_DIM] @ [HEAD_DIM, BLOCK_N] = [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, k)
        qk = qk * sm_scale

        # Online softmax: update max, compute probs
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # Correct old accumulator
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]

        # Load V: [BLOCK_N, HEAD_DIM]
        v_ptrs = V + v_offset + k_offs[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs)

        # PV: [BLOCK_M, BLOCK_N] @ [BLOCK_N, HEAD_DIM]
        p_fp16 = p.to(tl.float16)
        acc = tl.dot(p_fp16, v, acc)

        # Update running statistics
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    # Normalize and write
    acc = acc / l_i[:, None]
    o_ptrs = O + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(tl.float16))


def kernel(Q, K, V, causal=False, sm_scale=None):
    """Triton fused attention (forward only) with autotune"""
    BATCH, N_HEADS, N_CTX, HEAD_DIM = Q.shape
    if sm_scale is None:
        sm_scale = HEAD_DIM ** -0.5

    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    O = torch.empty_like(Q)

    grid = lambda META: (
        triton.cdiv(N_CTX, META['BLOCK_M']),
        BATCH * N_HEADS,
        1,
    )

    _attn_fwd_kernel[grid](
        Q, K, V, O,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        BATCH, N_HEADS, N_CTX,
        sm_scale,
        HEAD_DIM=HEAD_DIM,
    )
    return O
'''


# ─── PyTorch SDPA Reference (cuDNN Flash Attention) ─────────────────────

PYTORCH_REFERENCE_CODE = '''
import torch
import torch.nn.functional as F

def reference(Q, K, V, causal=False, sm_scale=None):
    """PyTorch scaled_dot_product_attention (uses cuDNN/Flash Attention)"""
    if sm_scale is None:
        sm_scale = Q.shape[-1] ** -0.5
    # SDPA expects [B, H, S, D] format
    is_causal = causal
    return F.scaled_dot_product_attention(
        Q, K, V,
        is_causal=is_causal,
        scale=sm_scale,
    )
'''


# ─── Test Shapes ─────────────────────────────────────────────────────────

# Covering: different batch/head/sequence length combinations
# Representative of LLM inference scenarios (small batch, long context)
TEST_SHAPES = [
    # (BATCH, N_HEADS, N_CTX, HEAD_DIM)
    (1, 8, 1024, 64),       # Small batch, short seq
    (1, 8, 2048, 64),       # Small batch, medium seq
    (1, 8, 4096, 64),       # Small batch, long seq
    (2, 8, 2048, 64),       # Medium batch
    (4, 8, 4096, 64),       # Larger batch
    (1, 8, 2048, 128),      # Larger head dim
    (1, 8, 4096, 128),      # Long seq, large head dim
    (2, 8, 1024, 128),      # Medium batch, large head
]


def main():
    print("=" * 70)
    print("  Benchmark: Triton Official Fused Attention vs PyTorch SDPA")
    print("  Source: triton/python/tutorials/06-fused-attention.py")
    print("  Algorithm: Flash Attention v2 (online softmax, tiled K/V)")
    print("=" * 70)

    ssh_host = os.environ.get("REMOTE_SSH_HOST", "152.136.252.171")
    ssh_user = os.environ.get("REMOTE_SSH_USER", "zhanghan")
    ssh_port = int(os.environ.get("REMOTE_SSH_PORT", "22"))

    print(f"\n  Remote GPU: {ssh_host}:{ssh_port} (A100-SXM4-40GB)")
    print(f"  Test shapes: {len(TEST_SHAPES)} configurations")
    print(f"  Precision: FP16 (forward pass only, non-causal)")
    print()

    results = []
    shape_info = []
    for BATCH, N_HEADS, N_CTX, HEAD_DIM in TEST_SHAPES:
        print(f"  Benchmarking B={BATCH}, H={N_HEADS}, S={N_CTX}, D={HEAD_DIM}...", end=" ", flush=True)

        inputs = {
            "Q": {"shape": [BATCH, N_HEADS, N_CTX, HEAD_DIM], "seed": 42},
            "K": {"shape": [BATCH, N_HEADS, N_CTX, HEAD_DIM], "seed": 123},
            "V": {"shape": [BATCH, N_HEADS, N_CTX, HEAD_DIM], "seed": 456},
        }

        result = run_remote_benchmark(
            operator="attention",
            kernel_code=TRITON_ATTENTION_CODE,
            reference_code=PYTORCH_REFERENCE_CODE,
            inputs=inputs,
            dtype="fp16",
            ssh_host=ssh_host,
            ssh_user=ssh_user,
            ssh_port=ssh_port,
            warmup=5,
            repeat=20,
            do_backward=False,
        )

        results.append(result)
        shape_info.append((BATCH, N_HEADS, N_CTX, HEAD_DIM))

        if result.passed:
            # FLOPs for attention = 4 * B * H * S^2 * D (QK^T + PV matmuls)
            flops = 4.0 * BATCH * N_HEADS * N_CTX * N_CTX * HEAD_DIM
            
            # Accurate bandwidth for Flash Attention
            # Flash Attention reads Q once, K/V in tiles, writes O once
            # Input: Q (B,H,S,D) + K (B,H,S,D) + V (B,H,S,D)
            # Output: O (B,H,S,D)
            # In fp16: 2 bytes per element
            dtype_bytes = 2  # fp16
            input_elements = BATCH * N_HEADS * N_CTX * HEAD_DIM * 3  # Q, K, V
            output_elements = BATCH * N_HEADS * N_CTX * HEAD_DIM     # O
            total_bytes = (input_elements + output_elements) * dtype_bytes
            
            triton_tflops = flops / (result.triton_fwd_ms / 1000) / 1e12
            pytorch_tflops = flops / (result.pytorch_fwd_ms / 1000) / 1e12 if result.pytorch_fwd_ms > 0 else 0
            
            # Bandwidth utilization
            triton_bw = total_bytes / (result.triton_fwd_ms / 1000) / 1e9
            pytorch_bw = total_bytes / (result.pytorch_fwd_ms / 1000) / 1e9 if result.pytorch_fwd_ms > 0 else 0
            
            speedup = f"{result.fwd_speedup:.3f}x" if result.fwd_speedup > 0 else "N/A"
            triton_ms = f"{result.triton_fwd_ms:.4f}ms"
            pytorch_ms = f"{result.pytorch_fwd_ms:.4f}ms"
            print(f"OK | Triton: {triton_ms} ({triton_tflops:.1f} TF, {triton_bw:.1f} GB/s), PyTorch: {pytorch_ms} ({pytorch_tflops:.1f} TF, {pytorch_bw:.1f} GB/s), Speedup: {speedup}")
        else:
            print(f"FAILED: {result.error[:120]}")

    # Print summary table
    print_benchmark_table(results, baseline_name="PyTorch SDPA (cuDNN/Flash Attention)")

    # Enhanced summary with bandwidth analysis
    passed = [(r, b, h, s, d) for r, (b, h, s, d) in zip(results, shape_info) if r.passed]
    if passed:
        avg_speedup = sum(r.fwd_speedup for r, _, _, _, _ in passed) / len(passed)
        min_speedup = min(r.fwd_speedup for r, _, _, _, _ in passed)
        max_speedup = max(r.fwd_speedup for r, _, _, _, _ in passed)

        # A100 hardware specs
        peak_tflops = 312.0      # FP16 Tensor Core
        peak_bw_gbps = 2039.0    # HBM bandwidth
        dtype_bytes = 2          # FP16

        # Calculate TFLOPS and bandwidth for each shape
        analysis_data = []
        for r, B, H, S, D in passed:
            flops = 4.0 * B * H * S * S * D
            input_elements = B * H * S * D * 3  # Q, K, V
            output_elements = B * H * S * D     # O
            total_bytes = (input_elements + output_elements) * dtype_bytes
            
            triton_tf = flops / (r.triton_fwd_ms / 1000) / 1e12
            cublas_tf = flops / (r.pytorch_fwd_ms / 1000) / 1e12
            triton_bw = total_bytes / (r.triton_fwd_ms / 1000) / 1e9
            cublas_bw = total_bytes / (r.pytorch_fwd_ms / 1000) / 1e9
            
            # Arithmetic intensity
            ai = flops / total_bytes
            
            # Bottleneck classification
            ridge_point = peak_tflops * 1e12 / (peak_bw_gbps * 1e9)
            if ai < ridge_point * 0.8:
                bottleneck = "Memory-bound"
            elif ai < ridge_point * 1.5:
                bottleneck = "Balanced"
            else:
                bottleneck = "Compute-bound"
            
            analysis_data.append({
                'B': B, 'H': H, 'S': S, 'D': D,
                'triton_tf': triton_tf, 'cublas_tf': cublas_tf,
                'triton_bw': triton_bw, 'cublas_bw': cublas_bw,
                'triton_tf_pct': triton_tf / peak_tflops * 100,
                'cublas_tf_pct': cublas_tf / peak_tflops * 100,
                'triton_bw_pct': triton_bw / peak_bw_gbps * 100,
                'cublas_bw_pct': cublas_bw / peak_bw_gbps * 100,
                'speedup': r.fwd_speedup,
                'ai': ai,
                'bottleneck': bottleneck,
            })

        max_triton_tflops = max(d['triton_tf'] for d in analysis_data)
        max_cublas_tflops = max(d['cublas_tf'] for d in analysis_data)
        max_triton_bw = max(d['triton_bw'] for d in analysis_data)
        max_cublas_bw = max(d['cublas_bw'] for d in analysis_data)

        print(f"\n  {'='*90}")
        print(f"  SUMMARY: Triton vs PyTorch SDPA (A100, FP16, Forward)")
        print(f"  {'='*90}")
        print(f"\n  【性能概览】")
        print(f"    Average speedup (Triton / PyTorch SDPA): {avg_speedup:.3f}x")
        print(f"    Min/Max speedup:     {min_speedup:.3f}x / {max_speedup:.3f}x")
        
        print(f"\n  【算力分析 (TFLOPS)】")
        print(f"    Peak Triton:   {max_triton_tflops:.1f} TFLOPS ({max_triton_tflops/peak_tflops*100:.1f}% of A100 312T)")
        print(f"    Peak PyTorch:  {max_cublas_tflops:.1f} TFLOPS ({max_cublas_tflops/peak_tflops*100:.1f}% of A100 312T)")
        
        print(f"\n  【带宽分析 (GB/s)】")
        print(f"    Peak Triton:   {max_triton_bw:.1f} GB/s ({max_triton_bw/peak_bw_gbps*100:.1f}% of A100 2039G)")
        print(f"    Peak PyTorch:  {max_cublas_bw:.1f} GB/s ({max_cublas_bw/peak_bw_gbps*100:.1f}% of A100 2039G)")

        # Detailed per-shape table
        print(f"\n  {'='*90}")
        print(f"  【详细性能表】")
        print(f"  {'='*90}")
        header = (
            f"  {'Shape (B,H,S,D)':<20} {'AI':<8} {'瓶颈':<14} "
            f"{'Triton TF':<11} {'Triton %peak':<13} "
            f"{'Triton BW':<11} {'Triton %bw':<10} "
            f"{'PyTorch TF':<11} {'PyTorch BW':<11} "
            f"{'Speedup':<10}"
        )
        print(header)
        print(f"  {'-'*143}")
        for d in analysis_data:
            shape_str = f"({d['B']},{d['H']},{d['S']},{d['D']})"
            print(
                f"  {shape_str:<20} {d['ai']:<8.0f} {d['bottleneck']:<14} "
                f"{d['triton_tf']:<11.1f} {d['triton_tf_pct']:<12.1f}% "
                f"{d['triton_bw']:<11.1f} {d['triton_bw_pct']:<9.1f}% "
                f"{d['cublas_tf']:<11.1f} {d['cublas_bw']:<11.1f} "
                f"{d['speedup']:<10.3f}x"
            )

        # Bottleneck distribution
        print(f"\n  {'='*90}")
        print(f"  【瓶颈分布分析】")
        print(f"  {'='*90}")
        from collections import Counter
        bn_counts = Counter(d['bottleneck'] for d in analysis_data)
        for bn, count in sorted(bn_counts.items()):
            avg_tf = sum(d['triton_tf'] for d in analysis_data if d['bottleneck'] == bn) / count
            avg_bw = sum(d['triton_bw'] for d in analysis_data if d['bottleneck'] == bn) / count
            print(f"    {bn}: {count} shapes, Avg Triton: {avg_tf:.1f} TFLOPS, {avg_bw:.1f} GB/s")

        # Best/worst by compute
        best = max(analysis_data, key=lambda d: d['triton_tf'])
        worst = min(analysis_data, key=lambda d: d['triton_tf'])
        print(f"\n  【算力最优/最差】")
        print(f"    Best:  ({best['B']},{best['H']},{best['S']},{best['D']}) = {best['triton_tf']:.1f} TFLOPS ({best['speedup']:.3f}x PyTorch)")
        print(f"    Worst: ({worst['B']},{worst['H']},{worst['S']},{worst['D']}) = {worst['triton_tf']:.1f} TFLOPS ({worst['speedup']:.3f}x PyTorch)")

        best_bw = max(analysis_data, key=lambda d: d['triton_bw'])
        worst_bw = min(analysis_data, key=lambda d: d['triton_bw'])
        print(f"\n  【带宽最优/最差】")
        print(f"    Best:  ({best_bw['B']},{best_bw['H']},{best_bw['S']},{best_bw['D']}) = {best_bw['triton_bw']:.1f} GB/s ({best_bw['triton_bw_pct']:.1f}% 峰值)")
        print(f"    Worst: ({worst_bw['B']},{worst_bw['H']},{worst_bw['S']},{worst_bw['D']}) = {worst_bw['triton_bw']:.1f} GB/s ({worst_bw['triton_bw_pct']:.1f}% 峰值)")

        # Efficiency verdict
        print(f"\n  {'='*90}")
        print(f"  【效率评估】")
        print(f"  {'='*90}")
        
        # Compute efficiency
        compute_pct = max_triton_tflops / peak_tflops * 100
        if compute_pct > 40:
            compute_verdict = "✅ 优秀 (Triton Tensor Core 利用率高)"
        elif compute_pct > 30:
            compute_verdict = "⚠️  良好 (可进一步优化)"
        else:
            compute_verdict = "❌ 较低 (存在较大提升空间)"
        print(f"\n  算力效率: {compute_pct:.1f}% {compute_verdict}")
        
        # Bandwidth efficiency  
        bw_pct = max_triton_bw / peak_bw_gbps * 100
        if bw_pct > 10:
            bw_verdict = "✅ 优秀 (带宽利用率高)"
        elif bw_pct > 5:
            bw_verdict = "⚠️  良好 (可进一步优化)"
        else:
            bw_verdict = "❌ 较低 (Memory-bound 场景需优化)"
        print(f"  带宽效率: {bw_pct:.1f}% {bw_verdict}")
        
        # Overall verdict
        if avg_speedup > 0.9:
            overall = "⭐ EXCELLENT"
        elif avg_speedup > 0.7:
            overall = "✅ GOOD"
        elif avg_speedup > 0.5:
            overall = "⚠️ MODERATE"
        else:
            overall = "❌ BELOW TARGET"
        print(f"  综合评分: {overall} (Triton 官方达 PyTorch SDPA {avg_speedup*100:.0f}%)")

        # Agent target
        agent_target = avg_speedup * 0.8
        print(f"\n  {'='*90}")
        print(f"  【Agent 性能目标】")
        print(f"    Agent target (0.8x of Triton official): {agent_target:.3f}x PyTorch SDPA")
        print(f"    Agent Tensor Core target: {max_triton_tflops * 0.8:.1f} TFLOPS ({max_triton_tflops * 0.8 / peak_tflops * 100:.1f}%)")
        print(f"    Agent bandwidth target: {max_triton_bw * 0.8:.1f} GB/s ({max_triton_bw * 0.8 / peak_bw_gbps * 100:.1f}%)")
        print(f"  {'='*90}")

    return results


if __name__ == "__main__":
    main()
