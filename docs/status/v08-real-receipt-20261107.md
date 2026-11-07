# v0.8 真机回执：H20 窗口、L3 有效读数与 preflight 修两处（2026-11-07）

| 项目 | 值 |
|---|---|
| 机器 | `h20`（SSH 别名），192 核，8×NVIDIA H20 |
| 窗口 | 13:36–14:10 CST；load average 5.35 / 4.95 / 4.90，八卡 `utilization.gpu` 全 0 |
| 用途 | ① 抓一份真实 `valid` L3 回执验证诊断器；② 验证 `preflight --shape` 是否量对了通道 |
| 证据 | 仓外 `output/v08-real-receipt-20261107/`（`run/verify-1`、`run/verify-2`、`preflight-4096.json`、`preflight-1024.json`、两份候选源码、探针脚本） |

本记录只写这次真机观测与它改掉的代码，不构成性能收益主张。

## 1. 窗口噪声（三档，两种口径）

产品口径（`preflight --shape`，`--reps 5 --repeat 8`，远端 CUDA event + 每轮 L2 冲刷）：

| shape | median | iqr/median | 3× 判决带宽 | `action`（对 1% 效应） |
|---|---|---|---|---|
| 4096³ | 1.0396 ms | 0.0677% | 0.203% | `start` |
| 1024³ | 0.0258 ms | 1.05% | 3.16% | `hold` |

手工口径（`output/…/gpu_repeatability_probe.py`，torch matmul，无 L2 冲刷，8 次单样本）：
128³ 的 iqr/median = 7.40%（launch 地板 21 µs 对 median 21.6 µs，等于地板占满）。

**结论边界**：上一轮八格记录的「判决带宽 12.97%、环境不可判」是**窗口特异性**的；
同一个 4096³ 在本窗口只有 0.07%。`relu 1024×768` 这类小 shape 仍然不可判——那由 shape
和地板决定，不是机器变吵。

## 2. 真实 L3 回执（fp16 4096³，triton，stride-aware）

| 字段 | 值 |
|---|---|
| levels | L1 / L2 / L3 全 `true` |
| `measurement_validity` | **`valid`**，`can_gate/can_disable_recipe/can_count_stagnation` 全 `true` |
| median / baseline | 1.040016 ms / 1.044096 ms |
| `speedup_vs_baseline` | 1.0039（与 torch 基线持平，**不是收益**） |
| `runs` / `dispersion_iqr_over_median` | 46 / 0.0016 |
| `roofline.bound_type` | `compute`，`compute_used=fp16_tensor_core` |
| 达成率 | `gflops_utilization_pct=89.3`（`bandwidth_utilization_pct=2.4`） |
| `launch_dominance` / `launch_floor_us` | 0.0191 / 20.24 |
| 分母出处 | `peak_trusted=true`、`reference_lower_bound=false`（规格峰值，非参考下界） |

第一版候选假设连续布局，`auto-k4-m32-n32-fp16-random-strided` 派生用例失败（误差 9.69）——
与八格里 `relu-small-B` 同一个失败模式。改成 stride-aware 后全通过。

## 3. 这次改动

| # | 问题 | 修法 | 提交 |
|---|---|---|---|
| 1 | 诊断器建议**已满足**的轴（核已在 tensor core 上，还建议换 dtype） | `_eligible_experiments` 按证据过滤 | `34f1177` |
| 2 | 余量口径缺失：分母是规格峰值却报 `unverified/null` | 加 `audited_peak` 口径，报 `remaining_pct` 并注明分母性质 | `34f1177` |
| 3 | `preflight` 用小参照核，SSH 上落在远端 `perf_counter`（CPU）分支 | 加 `--shape M,N,K`：目标尺寸 triton 探针，走 CUDA event + L2 冲刷 | 本次 |
| 4 | 探针每个 episode 只采 1 个样本，5 个点的 iqr 把 0.07% 放大成 2.02% | `--repeat`（默认 8）：episode 内多采、跨 episode 看中位散布 | 本次 |

第 4 条是这次最容易踩的坑：先跑出来的回执说 4096³ 的判决带宽是 **6.07%**、`hold`，
与同一窗口的真实 L3 读数（`dispersion=0.0016`、median 1.040 ms）自相矛盾。改成每
episode 8 样本后，median 1.0396 ms 与 L3 的 1.040016 ms 对得上——**探针读数必须能被
同窗口的 L3 复现，否则是探针自己坏了**。

## 4. 仍未解决

- `sampling_comparability` 仍 `not_established`（A 臂无同等硬门、账本未记 `temperature/n/enable_thinking`）。
- R2（v0.7.2 对 v0.8 的 B/C 配对）未开始；本文只立起了它的启动门，没有跑配对。
- 本窗口的证据只覆盖 fp16 matmul 4096³ 与 1024³；layernorm 16384×6144 未测。