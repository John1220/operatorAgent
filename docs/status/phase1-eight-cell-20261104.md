# 一期八格对照：第一轮完整终态（2026-11-04）

| 项目 | 说明 |
|---|---|
| 文档性质 | 真机接入执行记录，供复核正确性、终态与测量有效性。正式对照及安装工件验收见 §6 |
| 计划 | 接入计划 `phase1-entry-integration-v1`，目标计划 `phase1-agent-comparison-v2`，plan hash `77a54047…` |
| 预登记 | `da00f988…`（据此冻结采样与预算） |
| 实例 | H20 物理 3 号卡 `GPU-634bbc76-…`，root@h20，`/root/op-agent-runtime`，runner 15 文件全等 |
| 采样 | `max_tokens=32768`、`enable_thinking=true`、`reasoning_effort=medium`、`temperature=0.1`、`n=1` |
| 预算 | 每格 30 次模型调用 / 2400 s 墙钟 / 20 次 oracle |
| 运行留档 | 仓外 `…/h20-bind-20261104/execution-s5/`（`plan.json`、`summary.json`、八格 `cells/`） |

## 1. 八格终态

| 格 | 终态 | 归因 | 耗时 | 调用 | 独立终评 L1 / L2 / L3 | 独立终评倍率 | L3 测量有效性 |
|---|---|---|---|---|---|---|---|
| matmul-4096-A | failed | `no_gain_2_rounds` | 474 s | 30 | ✓ / ✓ / ✗ | 0.9393× | `valid` |
| matmul-4096-B | done | `converged` | 358 s | 19 | ✓ / ✓ / ✗ | 0.9670× | `degraded` |
| relu-large-B | done | `passed` | 368 s | 19 | ✓ / ✓ / ✓ | 1.0011× | `degraded` |
| relu-large-A | done | `native_host_returned` | 605 s | 29 | ✓ / ✓ / ✓ | 1.0759× | `degraded` |
| softmax-4096-A | failed | `budget_calls_exhausted` | 666 s | 30 | ✓ / ✓ / ✗ | 2.6839× | `uninformative` |
| softmax-4096-B | done | `passed` | 228 s | 10 | ✓ / ✓ / ✓ | 1.8873× | `degraded` |
| relu-small-B | done | `fix_error` | 569 s | 28 | ✓ / ✗ / 未执行 | — | — |
| relu-small-A | done | `native_host_returned` | 507 s | 30 | ✓ / ✓ / ✗ | 0.9696× | `uninformative` |

合计 **195 次模型调用**，八格全部留有终态和独立终评，总墙钟 74 分钟。**7/8 格通过完整 L2**。
`relu-small-B` 的连续布局主用例通过，失败发生在 `auto-cols32-rows8-fp16-random-strided` 派生用例。
终评列取自各格 `cells/<格>/final/final-result.json` 和 `final/verify-1.json`。`passed` 是格位的停止归因，L3 勾选是验证器回执，正式达标还要求 `measurement_validity=valid`。

## 2. 可归因的三条停止理由实机验证

| 归因 | 出现格 | 判据来源 |
|---|---|---|
| `no_gain_2_rounds` | matmul-4096-A | 连续 2 次 **valid** 的「跑对但没更快」⇒ 关闭验证服务；两次 `degraded` 读数按设计**不计入** |
| `budget_calls_exhausted` | softmax-4096-A | 读账本 `remaining_calls == 0`，不靠错误文本匹配 |
| `fix_error` | relu-small-B | 终稿的独立 L2 在 strided 派生用例失败 |

三条此前都不存在或被误记（曾把编译错误判成 `infra_retry`、把预算耗尽判成 `native_host_failed`）。

## 3. 断流：本轮 0 次

178 条流观测记录，**失败 0**。单条最大 **1,528,144 字节**（约 1.46 MiB，比旧 1 MiB 上限大约 46%）。

旧上限的失效方式已复现过：同一请求重复 6 次，5 次在 `1,048,xxx` 字节处被掐断，
`code=response_limit`，而子进程只看到 `Stream ended without finish_reason`。
修复后大输出档为 32 MiB / 65536 块（4K 档保持原值）。

## 4. 性能观测与测量有效性

- `softmax-4096-B` 的开发测量为 **1.8815×**，开发确认测量为 **1.8661×**，独立终评为 **1.8873×**，三份回执均为 `degraded`。终评记录了 launch 地板占比 28.7% 和同卡的另一 harness 进程。这些倍率只保留为观测值。
- `relu-large-B` 的开发测量为 **1.009×**，独立终评为 **1.0011×**，均为 `degraded`。A 臂独立终评也为 `degraded`。
- `matmul-4096-A` 的独立终评为 `valid`，**0.9393×** 低于 1.0× 门槛。B 臂终评为 `degraded`。
- `softmax-4096-A` 和 `relu-small-A` 的终评 L3 为 `uninformative`。`relu-small-B` 在独立终评的 L2 阶段失败，未执行 L3。

复核 Softmax B 时可对照同一格的 `host-terminal-status.json`、`work/output/softmax/nvidia_gpu/20261104_224405_f0ddee44/confirmation.json` 和 `final/verify-1.json`。

## 5. 读数环境：这与结论强相关

| 时刻 | 主机 load average | 本卡 valid 判决带宽 |
|---|---|---|
| 10:05（当日首次标定） | 7.15 / 9.65 / 8.62 | **1.23%** |
| 20:48–22:26（本轮执行） | **188.8 / 188.4 / 188.9**（192 核） | 10–14% |

同日另做了独占卡对照（停掉 GPU6 上的 `qwen8004.service` 后测）：valid 带宽
**12.97%**，锁频 1980 后 **14.39%** —— **独占并没有更安静**，主导因素是主机 CPU 负载，
不是卡片共享。因此「换独占卡」不是降噪手段；需要的是主机空闲窗口。

## 6. 结论边界

- 按[预登记条件](../testing/phase1-study.md#1-当前固定设计)，达标要求完整 L2、至少一次 `valid` L3 且倍率 ≥ 1.0×。本轮独立终评未确认任何格位达标，四组 A/B 均缺少可比较的终评证据。原先的「仅 Softmax 可判，B 达标 1 格 / A 达标 0 格」计数撤回。A/B 达标差值保持未知。
- 八格预算均可审计。`sampling_comparability` 仍为 `not_established`，DSH 外层 usage 缺少对账，费用和人工分钟保持未知。
- 原始 `summary.json` 记录 `dsh_entry_verified=true`、`formal_comparison_credit=false`、`installed_plugin_acceptance=false`，分别对应接入路径实跑、正式对照资格及最终安装工件验收状态。

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
