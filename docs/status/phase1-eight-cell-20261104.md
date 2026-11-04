# 一期八格对照：第一轮完整终态（2026-11-04）

| 项目 | 说明 |
|---|---|
| 文档性质 | 真机执行记录；**不是**收益结论，也不是 v0.7 发布验收 |
| 计划 | `phase1-agent-comparison-v2`（`target_plan_id`），plan hash `77a54047…` |
| 预登记 | `da00f988…`（据此冻结采样与预算） |
| 实例 | H20 物理 3 号卡 `GPU-634bbc76-…`，root@h20，`/root/op-agent-runtime`，runner 15 文件全等 |
| 采样 | `max_tokens=32768`、`enable_thinking=true`、`reasoning_effort=medium`、`temperature=0.1`、`n=1` |
| 预算 | 每格 30 次模型调用 / 2400 s 墙钟 / 20 次 oracle |
| 工件 | 仓外 `…/operator-v07-candidate-20261103/execution-s5/`（`summary.json`、八格 `cells/`、`transport-failures.jsonl`） |

## 1. 八格终态

| 格 | 终态 | 归因 | 耗时 | 调用 | L1/L2/L3 | speedup |
|---|---|---|---|---|---|---|
| matmul-4096-A | failed | `no_gain_2_rounds` | 474 s | 30 | — | — |
| matmul-4096-B | done | `converged` | 358 s | 19 | ✓✓✗ | 0.944 |
| relu-large-B | done | `passed` | 368 s | 19 | ✓✓✓ | 1.009 |
| relu-large-A | done | `native_host_returned` | 605 s | 29 | ✓✓✓ | 未落盘 |
| softmax-4096-A | failed | `budget_calls_exhausted` | 666 s | 30 | ✓✓✗ | — |
| **softmax-4096-B** | done | **`passed`** | **228 s** | **10** | ✓✓✓ | **1.8815** |
| relu-small-B | done | `fix_error` | 569 s | 28 | ✓✗ | — |
| relu-small-A | done | `native_host_returned` | 507 s | 30 | ✓✓✗ | — |

合计 **195 次模型调用**，八格全部有归宿（无 `not_dispatched`），总墙钟 74 分钟。

## 2. 可归因的三条停止理由实机验证

| 归因 | 出现格 | 判据来源 |
|---|---|---|
| `no_gain_2_rounds` | matmul-4096-A | 连续 2 次 **valid** 的「跑对但没更快」⇒ 关闭验证服务；两次 `degraded` 读数按设计**不计入** |
| `budget_calls_exhausted` | softmax-4096-A | 读账本 `remaining_calls == 0`，不靠错误文本匹配 |
| `fix_error` | relu-small-B | L2 始终不过，候选失败如实记录 |

三条此前都不存在或被误记（曾把编译错误判成 `infra_retry`、把预算耗尽判成 `native_host_failed`）。

## 3. 断流：本轮 0 次

178 条流观测记录，**失败 0**；单条最大 **1,528,144 字节**（约 1.46 MiB，已越过旧 1 MiB 上限 50%）。

旧上限的失效方式已复现过：同一请求重复 6 次，5 次在 `1,048,xxx` 字节处被掐断，
`code=response_limit`，而子进程只看到 `Stream ended without finish_reason`。
修复后大输出档为 32 MiB / 65536 块（4K 档保持原值）。

## 4. 哪些数能用，哪些不能用

**能用的只有一条**：`softmax-4096-B = 1.8815`。它比当前判决带宽（见下）宽 6.8 倍，
量级差异在这台机器上站得住。

**不能当结论的两条**：

- `relu-large-B = 1.009`：同一格在另一次运行里是 `1.0769`，两次都判 `passed`，
  差值 7 倍于判决带宽 ⇒ 只能记「无可判差异」。
- `matmul-4096-B = 0.944`：落在带内偏低，方向（打不过 cuBLAS）与历史一致，
  但不能据此给出精度结论。

**结构性不可判的**：`relu-small`（1024×768，launch 地板支配 55–68%）、
`matmul-4096-A`（连续无增益）、`softmax-4096-A`（预算耗尽）。

## 5. 读数环境：这与结论强相关

| 时刻 | 主机 load average | 本卡 valid 判决带宽 |
|---|---|---|
| 10:05（当日首次标定） | 7.15 / 9.65 / 8.62 | **1.23%** |
| 20:48–22:26（本轮执行） | **188.8 / 188.4 / 188.9**（192 核） | 10–14% |

同日另做了独占卡对照（停掉 GPU6 上的 `qwen8004.service` 后测）：valid 带宽
**12.97%**，锁频 1980 后 **14.39%** —— **独占并没有更安静**，主导因素是主机 CPU 负载，
不是卡片共享。因此「换独占卡」不是降噪手段；需要的是主机空闲窗口。

## 6. 结论边界

- 本轮 4 组 A/B 中，**只有 softmax 这一对可判**（A 臂 L3 未过，B 臂 1.88x 达标）。
- 按预登记的达标定义（完整 L2 + 至少一次 valid L3 + speedup ≥ 1.0），B 臂记 1 格达标、
  A 臂 0 格。**这不足以支持「插件有收益」**：4 格中 3 格各有不可判原因，
  且 +1 的最小可检出单位要求两臂 4 格全部可判。
- `sampling_comparability` 仍为 `not_established`：DSH 外层 usage 对账缺失，
  费用保持 unknown。
- 本轮所有 L3 数字都带 `measurement_validity` 标签，`degraded` 的不进入任何性能结论。

## 7. 本轮之外、同日的实现变更（进入本版身份）

| 变更 | 证据 |
|---|---|
| 网关流上限按输出上限缩放（32K ⇒ 32 MiB / 65536 块） | `tests/unit/phase1/test_study_sampling.py::TestStreamLimitScalesWithOutputCap` |
| 常驻会话默认开启（L1+L2 从 384.9 s → 60.0 s，全量 405 s → 31.9 s） | 同文件 relay 用例钉住 `persistent=True` |
| 单 case 传输未知补发一次 | `test_correctness_suite.py::test_single_case_transport_unknown_is_resent_once` |
| A 臂「连续 2 次有效无增益即停」 | `test_native.py::test_consecutive_no_gain_closes_the_native_verifier` |
| 编译错误优先于基础设施判定（`_INFRA_MARKERS` 去掉裸词 `ssh`） | `test_failure_route.py::TestRemoteCompileErrorIsNotInfra` |
| PDL 配方纠正：`launch_pdl` 只能写在启动点 | `test_recipe_caps_gate.py::TestPdlRecipeDoesNotFailItsOwnCompile` |
## 8. 尝试过但无效的降噪手段：CPU 亲和

设想：抖动来自主机 CPU（SM 时钟恒 1980、显存稳定），把测量进程钉到固定核上应能减少迁移与抢占。

实测（同一张 GPU3、同一脚本、`taskset -c 16,43,172,181` + `OMP_NUM_THREADS=4`）：

| 口径 | valid 带宽 | degraded 带宽 | 散布 p95 |
|---|---|---|---|
| 不钉核 | 12.97% | 30.07% | 0.664 |
| 钉 4 核 | **13.24%** | 40.11% | — |
| 钉核 + 锁频 1980 | 14.39% | 35.93% | — |

**没有改善。** 原因是这张 192 核机上**最空闲的核也只有约 10% 空闲**（其余 ~90% 被占），
没有任何核可以「隔离出来给测量用」；亲和性只避免迁移，拿不到安静时间。

因此在本机当前负载下（load ≈ 188/192），**不存在能把判决带宽压回个位数百分比的软件手段**；
要可判的性能对照，只能换一台安静的机器，或等到主机负载实质性下降。
