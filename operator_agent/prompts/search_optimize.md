你是 GPU 算子性能优化专家。当前算子性能未达标，请基于 Roofline 模型分析瓶颈并给出优化方案。

{{skill}}

## 当前 Sketch
{{sketch}}

## 性能数据
{{perf_availability}}
| 指标 | 当前值 | 硬件理论峰值 | 利用率 |
|------|--------|-------------|--------|
| GFLOPS | {{current_gflops}} | {{hw_peak_gflops}} | {{gflops_utilization_pct}} |
| 内存带宽 | {{memory_bandwidth_gbs}} GB/s | {{hw_peak_bandwidth_gbs}} GB/s | {{bandwidth_utilization_pct}} |
| 算数强度 | {{arithmetic_intensity}} FLOP/Byte | Roofline拐点: {{ridge_point}} | - |
| 加速比 | {{speedup}}x (目标 {{target_speedup}}x) | - | {{achieve_ratio}} |

**Roofline 判定**: {{roofline_bound}} (算子名分类: {{name_bound}})
**硬件**: {{hw_name}} (CC {{hw_compute_capability}})

## 优化历史 (已尝试的方案)
{{optimization_history}}

## 优化分析框架

请按照以下步骤进行结构化分析：

### Step 1: 瓶颈定位
- 如果 GFLOPS 利用率 < 30%: 核心瓶颈是 **计算资源未充分利用**
  - 检查 tile size 是否过小 (SM 利用率低)
  - 检查是否有足够的并行度 (grid 太小)
  - 考虑增大 BLOCK_M/BLOCK_N 提升 SM 占有率
- 如果带宽利用率 < 30%: 核心瓶颈是 **访存效率低**
  - 检查是否有非合并内存访问 (non-coalesced)
  - 检查是否有 bank conflict
  - 考虑向量化加载 (tl.load 的 mask 检查)
- 如果 ROOFLINE 判定为 memory-bound 但带宽利用率 < 30%:
  - 说明瓶颈在访存而非计算，需要优化数据加载模式
- 如果 ROOFLINE 判定为 compute-bound 但 GFLOPS 利用率 < 30%:
  - 说明瓶颈在计算，需要优化计算密度和 Tensor Core 利用率

### Step 2: 优化策略选择 (按优先级)
1. **算法级 (>50% 影响)**: tiling 策略调整、内存合并、kernel fusion
2. **硬件利用率 (20-50% 影响)**: 向量化加载、循环展开、occupancy 调优
3. **精细调优 (<20% 影响)**: 混合精度、double buffering、prefetching

### Step 3: Tile Size 调整建议
- BLOCK_M * BLOCK_N * BLOCK_K 不应超过共享内存限制
- BLOCK_M, BLOCK_N 应为 32 的倍数 (warp 对齐)
- 对于小矩阵 (M, N, K < 256)，减小 tile 避免越界
- 对于大矩阵，compute-bound 时增大 BLOCK_M/BLOCK_N 提升计算密度
- memory-bound 时增大 BLOCK_K 提升每次加载的复用率

### Step 4: 避免重复
- 优化历史中已尝试且无效的方案，请勿重复建议
- 如果 tiling 已经接近最优，请从其他维度 (pipeline, memory_plan) 给出建议

以 JSON 格式返回优化方案：
{
  "analysis": "瓶颈分析 (引用 Roofline 数据和利用率数字)",
  "suggestion": "具体优化建议 (说明为什么这个方向有效)",
  "new_tiling": {"block_m": 整数, "block_n": 整数, "block_k": 整数},
  "new_hints": ["新优化提示1", "新优化提示2"],
  "optimization_stage": "algorithm|hardware_utilization|fine_tuning"
}