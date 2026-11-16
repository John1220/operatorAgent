## Roofline 诊断

| 指标 | 硬件理论峰值 | 利用率 |
|------|-------------|--------|
| GFLOPS | {{hw_peak_gflops}} | {{gflops_utilization_pct}} |
| 内存带宽 | {{hw_peak_bandwidth_gbs}} GB/s | {{bandwidth_utilization_pct}} |

**算数强度**：{{arithmetic_intensity}} FLOP/Byte
**Roofline拐点**：{{ridge_point}}
**Roofline 判定**：{{roofline_bound}}（算子名分类：{{name_bound}}）

### 瓶颈定位

- 对于 memory-bound 且带宽利用率偏低的核，检查内存访问是否连续，优先考虑向量化加载、减少重复读取和提高数据复用。
- 对于 compute-bound 且 GFLOPS 利用率偏低的核，检查 tile、并行度和计算密度，评估适用的 Tensor Core 路径。
- 在优化分析中引用可用的 Roofline 数据和利用率数字，说明建议如何针对观测到的瓶颈。
