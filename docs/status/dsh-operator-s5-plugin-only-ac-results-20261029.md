# S5 A/C 插件限定实测（2026-10-29）

状态：探索性批次因验证设施 unknown 提前停止，不构成正式 S5 证据，也不允许插件增益推断。

## 冻结条件

- 比较 A：原生 DSH；C：DSH + 已发布算子开发插件。任务固定为 Matmul-tail-fp16、LayerNorm-tail、Softmax-extreme；共 6 格，按任务交错 A/C。
- 模型 `deepseek-v4.1-flash-1017`，Responses，reasoning medium，单次输出上限 32,768 token；每格最多 24 次调用、总输出预留 786,432 token、墙钟 2,400 秒。
- 独立验证使用 algorithm-h20 GPU2：H20 UUID `GPU-13c6c4b4-45ce-8963-1122-df2b3d72fb76`。每格开始前核对设备身份、空闲显存至少 8,192 MiB、利用率不高于 5%、进程只来自冻结允许名单。此为共享 GPU 观察，不是独占保证。
- 冻结计划哈希：`22830ea3e66ac3c8b062cba1cbffc7faa7dbafab52834f84030ba2a775b792b2`。
- 汇总哈希：`7fc3ff37b52ad6f41016ab3445349ffc7090ed5325985aaf06a68ae89629fd72`。
- 证据目录：`output/plugin_completion/20261029-s5-plugin-only-ac-v1/`。第一次接线失败保存在 `execution-v1/`；真正派发的冻结计划、原始输出与格记录在 `frozen-plan-v2/`、`readiness-v2/` 和 `execution-v2/`。

## 结果

| 任务 | 组别 | 终态 | 模型调用预留 | 开发 / 最终验证 | 格耗时 |
|---|---|---|---:|---:|---:|
| Matmul-tail-fp16 | A | 最终 L2 通过 | 8 | 1 / 1 | 407.036 秒 |
| Matmul-tail-fp16 | C | 最终 L2 通过 | 8 | 1 / 1 | 479.934 秒 |
| LayerNorm-tail | C | unknown：最终 CorrectnessSuite 传输结果未知 | 9 | 1 / 1（最终 unknown） | 228.504 秒 |
| LayerNorm-tail | A | 未派发 | - | - | - |
| Softmax-extreme | A | 未派发 | - | - | - |
| Softmax-extreme | C | 未派发 | - | - | - |

实际派发 3/6 格，预算共预留 25 次模型调用。Matmul 配对两组都通过最终 L2，调用数相同；C 比 A 多约 73 秒（约 18%）。这只是一个任务的一次 A/C 配对，不能区分插件效应与模型/运行波动。固定六格汇总保留 passed 2、failed 0、unknown 4；设施 unknown 触发停止规则，未执行三格不能当作失败或通过。

服务商原始 usage 缺失，25 次调用的服务商费用未知；本地调用账本不是服务商账单。Matmul-C 发现并调用了 `operator_capabilities`，但开发验证通过 A/C 共用的公开 Python 测试入口提交；审计没有观察到插件 `operator_verify` 提交。插件能力虽然可用，本批并未证明模型实际使用插件验证功能能改善结果。

LayerNorm-C 的开发 L2 通过。最终验证中，基础 L1/L2 结果通过，但 CorrectnessSuite 在 `auto-cols32-rows1-fp32-random-contiguous` 的重复检查遇到远端传输结果 unknown，后续 case 未继续。网关故障快照没有上游失败记录，因此当前证据将其归为验证设施 unknown，而非模型请求失败或候选数值错误。该格不重跑，后续格也按停批策略未派发。

## 接线修复与限制

第一次真实启动因 `s5_plugin_env.build` 重复传递 `tools=True` 而在 DSH 启动前中断。该尝试的账本显示 0 次模型调用、0 次 GPU 验证。修复后，`tests/test_s5_plugin_env.py` 与 `tests/test_s5_plugin_workflow.py` 共 12 项通过；新无模型 readiness 的 A/C 脚本化探针各完成 10 次请求，真实模型和 GPU 仍为 0，随后才冻结 v2 计划。

GPU2 结束后观察为空闲：22,246 MiB 可用、利用率 0%；没有停止其他计算进程。readiness 仍标记 `network_isolation_verified=false`，也未证明网络只可连接预算网关或抵御同 UID 恶意进程，所以本批不计正式 S5 分母。

若优先加速下一轮，可先跑一个固定任务的 A/C 配对，共 2 格，减少三分之二的格数；代价是结论只覆盖该任务。开始下一批前应先定位远端重复检查的传输 unknown，之后重新冻结计划与源码身份，不复用本批 unknown 格。
