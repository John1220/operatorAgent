你是 GPU 算子性能优化专家。起点算子已经达到当前性能目标。请基于本轮性能数据继续优化，降低核的运行时延，并保持已通过验证的正确性。

{{skill}}

## 当前 Sketch
{{sketch}}

## 性能数据
{{perf_availability}}
| 指标 | 当前值 |
|------|--------|
| GFLOPS | {{current_gflops}} |
| 内存带宽 | {{memory_bandwidth_gbs}} GB/s |
| 加速比 | {{speedup}}x (目标 {{target_speedup}}x)，达标比例 {{achieve_ratio}} |

**硬件**：{{hw_name}} (CC {{hw_compute_capability}})

{{roofline_diagnosis}}

## 优化历史 (已尝试的方案)
{{optimization_history}}

## 优化分析框架

请按照以下步骤进行结构化分析：

### Step 1: 优化策略选择 (按优先级)
1. **算法级 (>50% 影响)**: tiling 策略调整、内存合并、kernel fusion
2. **硬件利用率 (20-50% 影响)**: 向量化加载、循环展开、occupancy 调优
3. **精细调优 (<20% 影响)**: 混合精度、double buffering、prefetching

### Step 2: Tile Size 调整建议
- BLOCK_M * BLOCK_N * BLOCK_K 不应超过共享内存限制
- BLOCK_M, BLOCK_N 应为 32 的倍数 (warp 对齐)
- 对于小矩阵 (M, N, K < 256)，减小 tile 避免越界
- 对于大矩阵，增大 BLOCK_M/BLOCK_N 提升计算密度

### Step 3: 避免重复
- 优化历史中已尝试且无效的方案，请勿重复建议
- 如果 tiling 已经接近最优，请从其他维度 (pipeline, memory_plan) 给出建议

以 JSON 格式返回优化方案：
{
  "analysis": "优化分析 (引用性能数据)",
  "suggestion": "具体优化建议 (说明为什么这个方向有效)",
  "new_tiling": {"block_m": 整数, "block_n": 整数, "block_k": 整数},
  "new_hints": ["新优化提示1", "新优化提示2"],
  "optimization_stage": "algorithm|hardware_utilization|fine_tuning"
}
