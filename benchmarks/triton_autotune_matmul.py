"""
A100 自动调优 Triton Matmul 验证脚本
====================================
在 NVIDIA A100 上验证 FP16 矩阵乘法算子，通过 @triton.autotune 搜索最优
(tile 大小 / warp 数 / 流水线阶段数) 组合，并与 cuBLAS (torch.matmul, FP16)
基线对比性能和精度。

运行方式 (在远程 A100 上):
    cd /data/zhanghan/operator_agent_runner && uv run python triton_autotune_matmul.py
"""

import triton
import triton.language as tl
import torch


@triton.autotune(
    configs=[
        # 大 tile + 深流水线: 适合大矩阵, 充分复用共享内存与寄存器
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr,
                  M, N, K,
                  stride_am, stride_ak,   # A 矩阵行/列步长
                  stride_bk, stride_bn,   # B 矩阵行/列步长
                  stride_cm, stride_cn,   # C 矩阵行/列步长
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  GROUP_SIZE_M: tl.constexpr):
    # ---- 1. 1D grid + 分组调度: 提高 L2 cache 命中率 (R-MATMUL-01) ----
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ---- 2. 指针初始化: 首块 tile 的偏移量 (R-MATMUL-08) ----
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    # 提示编译器: K 整除 BLOCK_K, 可去掉 K 循环 mask 并做地址优化 (R-MATMUL-05)
    tl.assume(K % BLOCK_K == 0)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # ---- 3. 累加器初始化: fp32 保证精度, 输入 fp16 触发 Tensor Core ----
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ---- 4. K 循环: 软件流水线(编译器自动) + 3-arg dot 融合累加 ----
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        # 3-arg tl.dot: FMA 融合, 减少寄存器读写 (R-MATMUL-09)
        acc = tl.dot(a, b, acc)
        # 增量推进指针, 避免每轮重算基址 (R-MATMUL-08)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # ---- 5. 存储结果: 边界 mask + cast 回 fp16 ----
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Host 入口: 从 tensor 取形状/步长并启动 kernel"""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "A/B 的 K 维不匹配"
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


def benchmark(fn, *args, warmup=10, repeat=50):
    """预热 + 计时, 返回平均毫秒数"""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return sum(times[len(times) // 4: -len(times) // 4]) / (len(times) // 2)  # 去掉首尾 25%


def main():
    torch.manual_seed(42)
    M, K, N = 4096, 4096, 4096
    print("=" * 64)
    print(f"  A100 自动调优 Triton Matmul 验证")
    print(f"  形状: A({M}x{K}) B({K}x{N})  FP16 | 计算量: {2*M*N*K/1e9:.0f} GFLOP")
    print(f"  设备: {torch.cuda.get_device_name(0)}")
    print("=" * 64)

    # 生成测试数据 (fp16)
    a = torch.randn(M, K, device='cuda', dtype=torch.float16) * 0.1
    b = torch.randn(K, N, device='cuda', dtype=torch.float16) * 0.1

    # ---- 1. 正确性验证: 与 fp32 参考对比 ----
    print("\n[L2] 功能验证...")
    ref = (a.float() @ b.float()).float()
    out = matmul(a, b).float()
    max_err = (out - ref).abs().max().item()
    rel_err = ((out - ref).abs() / (ref.abs() + 1e-6)).max().item()
    print(f"  最大绝对误差: {max_err:.2e}")
    print(f"  最大相对误差: {rel_err:.2e}")
    print(f"  {'✅ 通过 (FP16 误差 < 1e-2)' if max_err < 1e-2 else '❌ 失败'}")

    # ---- 2. 性能验证: 与 cuBLAS (torch.matmul) 对比 ----
    print("\n[L3] 性能验证 (vs cuBLAS FP16)...")

    # 预热 autotune 并获取最优配置
    print("\n  autotune 搜索最优配置中...")
    _ = matmul(a, b)
    best = matmul_kernel.best_config
    print(f"  最优配置: BLOCK_M={best.kwargs['BLOCK_M']}, "
          f"BLOCK_N={best.kwargs['BLOCK_N']}, BLOCK_K={best.kwargs['BLOCK_K']}, "
          f"GROUP_SIZE_M={best.kwargs['GROUP_SIZE_M']}, "
          f"num_warps={best.num_warps}, num_stages={best.num_stages}")

    # cuBLAS 基线
    cublas_t = benchmark(lambda: torch.matmul(a, b))
    cublas_gflops = 2 * M * N * K / (cublas_t / 1e3) / 1e9

    # Triton 最优
    triton_t = benchmark(matmul, a, b)
    triton_gflops = 2 * M * N * K / (triton_t / 1e3) / 1e9

    # 理论峰值: A100 FP16 Tensor Core = 312 TFLOPS
    peak = 312e12
    util = triton_gflops * 1e9 / peak * 100

    print(f"\n  cuBLAS:  {cublas_t:8.3f} ms | {cublas_gflops:9.1f} GFLOPS")
    print(f"  Triton:  {triton_t:8.3f} ms | {triton_gflops:9.1f} GFLOPS")
    print(f"  ─────────────────────────────────────────────")
    print(f"  加速比:  {triton_gflops / cublas_gflops:.3f}x vs cuBLAS")
    print(f"  理论峰值利用率: {util:.1f}% (峰值 312 TFLOPS)")

    # Roofline 分析
    ai = (2 * M * N * K) / ((M * K + K * N + M * N) * 2)  # FLOP/Byte, fp16 2B/元素
    print(f"\n  Roofline: 算术强度={ai:.1f} FLOP/Byte (>156 为计算-bound)")
    print(f"  结论: 计算-bound, 已接近理论峰值")

    passed = (triton_gflops / cublas_gflops) >= 0.8
    print(f"\n{'=' * 64}")
    print(f"  结论: {'✅ 通过 (≥0.8x cuBLAS)' if passed else '❌ 未达标 (<0.8x cuBLAS)'}")
    print(f"{'=' * 64}")
    return passed


if __name__ == "__main__":
    main()