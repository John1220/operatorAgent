"""Triton 性能优化规则库

基于 Roofline 模型和硬件特性的启发式优化规则。
这些规则在 SearchAgent 的 LLM 优化失败时作为回退策略。
"""

from typing import Dict, Tuple
from operator_agent.schemas import TilingStrategy


class TritonOptimizationRules:
    """Triton 算子性能优化规则引擎"""

    # A100 (SM 8.0) 硬件参数
    A100_PEAK_GFLOPS = 3120  # FP32 peak
    A100_PEAK_BANDWIDTH = 1935  # GB/s
    A100_SHARED_MEM_PER_SM = 16384  # bytes (16KB)
    A100_MAX_WARPS_PER_SM = 32
    A100_WARP_SIZE = 32

    # 推荐的 tile size 组合 (基于硬件特性)
    TILE_RECOMMENDATIONS = {
        "a100": {
            "compute_bound": [
                (128, 128, 32),
                (128, 64, 32),
                (64, 128, 32),
            ],
            "memory_bound": [
                (64, 64, 64),
                (64, 64, 128),
                (32, 64, 128),
            ],
        },
        "default": {
            "compute_bound": [(64, 64, 32), (64, 32, 32)],
            "memory_bound": [(32, 32, 64), (32, 64, 64)],
        },
    }

    @classmethod
    def recommend_tiling(
        cls,
        current_tiling: TilingStrategy,
        bound_type: str,
        hw_name: str = "unknown",
        hw_cc: str = "unknown",
    ) -> Tuple[TilingStrategy, str]:
        """
        根据当前 tiling 和性能瓶颈推荐新的 tiling

        Args:
            current_tiling: 当前 tiling
            bound_type: compute_bound | memory_bound | mixed
            hw_name: 硬件名称
            hw_cc: compute capability

        Returns:
            (新 tiling, 建议说明)
        """
        old_bm, old_bn, old_bk = current_tiling.block_m, current_tiling.block_n, current_tiling.block_k

        # 确定硬件平台
        platform = "a100" if "a100" in hw_name.lower() else "default"
        recommendations = cls.TILE_RECOMMENDATIONS[platform]

        # 根据 bound type 选择策略
        if bound_type == "compute_bound":
            new_tiling = cls._optimize_compute_bound(old_bm, old_bn, old_bk, recommendations["compute_bound"])
            reason = f"compute-bound: 增大 BLOCK_M/BLOCK_N 提升 SM 占有率和计算密度"
        elif bound_type == "memory_bound":
            new_tiling = cls._optimize_memory_bound(old_bm, old_bn, old_bk, recommendations["memory_bound"])
            reason = f"memory-bound: 增大 BLOCK_K 提升数据复用率，优化内存合并"
        else:
            new_tiling = cls._optimize_mixed(old_bm, old_bn, old_bk)
            reason = f"mixed-bound: 均衡调整 tiling"

        return new_tiling, reason

    @classmethod
    def _optimize_compute_bound(
        cls, bm: int, bn: int, bk: int, candidates: list
    ) -> TilingStrategy:
        """compute-bound: 增大 BLOCK_M/BLOCK_N 提升计算密度"""
        # 如果当前 tile 不在推荐列表中，选择第一个推荐项
        if (bm, bn, bk) not in candidates:
            return TilingStrategy(*candidates[0])

        # 当前 tile 在推荐列表中，尝试下一个
        idx = candidates.index((bm, bn, bk))
        if idx + 1 < len(candidates):
            return TilingStrategy(*candidates[idx + 1])

        # 已到最后一个，尝试增大 BLOCK_M/BLOCK_N
        new_bm = min(bm * 2, 256)
        new_bn = min(bn * 2, 256)
        return TilingStrategy(new_bm, new_bn, bk)

    @classmethod
    def _optimize_memory_bound(
        cls, bm: int, bn: int, bk: int, candidates: list
    ) -> TilingStrategy:
        """memory-bound: 增大 BLOCK_K 提升数据复用"""
        if (bm, bn, bk) not in candidates:
            return TilingStrategy(*candidates[0])

        idx = candidates.index((bm, bn, bk))
        if idx + 1 < len(candidates):
            return TilingStrategy(*candidates[idx + 1])

        # 已到最后一个，尝试增大 BLOCK_K
        new_bk = min(bk * 2, 256)
        return TilingStrategy(bm, bn, new_bk)

    @classmethod
    def _optimize_mixed(cls, bm: int, bn: int, bk: int) -> TilingStrategy:
        """mixed-bound: 均衡调整"""
        if bm < 64:
            return TilingStrategy(64, bn, bk)
        if bn < 64:
            return TilingStrategy(bm, 64, bk)
        if bk < 64:
            return TilingStrategy(bm, bn, 64)
        # 都已 >= 64，尝试增大 BLOCK_K
        return TilingStrategy(bm, bn, min(bk * 2, 128))

    @classmethod
    def validate_tiling(cls, tiling: TilingStrategy, dtype: str = "float32") -> list[str]:
        """
        验证 tiling 是否合理，返回警告列表

        Args:
            tiling: 待验证的 tiling
            dtype: 数据类型

        Returns:
            警告信息列表
        """
        warnings = []
        bm, bn, bk = tiling.block_m, tiling.block_n, tiling.block_k

        # 检查 warp 对齐
        if bm % cls.A100_WARP_SIZE != 0:
            warnings.append(f"BLOCK_M={bm} 不是 warp size ({cls.A100_WARP_SIZE}) 的倍数")
        if bn % cls.A100_WARP_SIZE != 0:
            warnings.append(f"BLOCK_N={bn} 不是 warp size ({cls.A100_WARP_SIZE}) 的倍数")

        # 检查共享内存限制 (估算)
        shared_mem_estimate = bm * bk * 4 + bk * bn * 4 + bm * bn * 4
        if shared_mem_estimate > cls.A100_SHARED_MEM_PER_SM:
            warnings.append(
                f"共享内存估算 {shared_mem_estimate} bytes > 限制 {cls.A100_SHARED_MEM_PER_SM} bytes"
            )

        # 检查 tile 大小是否合理
        if bm * bn * bk > 1024 * 1024:
            warnings.append(f"tile 过大 ({bm}x{bn}x{bk})，可能导致寄存器溢出")

        return warnings

    @classmethod
    def generate_optimization_hints(
        cls,
        bound_type: str,
        current_gflops: float,
        target_gflops: float,
        tiling: TilingStrategy,
        dtype: str = "float32",
    ) -> list[str]:
        """
        根据性能数据生成优化提示

        Args:
            bound_type: 瓶颈类型
            current_gflops: 当前 GFLOPS
            target_gflops: 目标 GFLOPS
            tiling: 当前 tiling

        Returns:
            优化提示列表
        """
        hints = []
        ratio = current_gflops / target_gflops if target_gflops > 0 else 0

        if bound_type == "compute_bound":
            hints.append("使用 tl.dot() 触发 Tensor Core")
            hints.append("增大 BLOCK_M/BLOCK_N 提升计算密度")
            hints.append("使用 float16 累加器提升吞吐量")
            if ratio < 0.5:
                hints.append("当前计算利用率低于 50%，考虑重新设计 tiling 策略")

        elif bound_type == "memory_bound":
            hints.append("优化内存访问模式，确保 coalesced 访问")
            hints.append("增大 BLOCK_K 提升每次加载的数据复用率")
            hints.append("使用 tl.cache 优化数据缓存")
            if ratio < 0.5:
                hints.append("当前带宽利用率低于 50%，检查内存访问是否合并")

        else:
            hints.append("检查内存访问模式和计算效率")

        # 添加 tiling 相关提示
        validation = cls.validate_tiling(tiling, dtype)
        if validation:
            hints.extend(validation)

        return hints
