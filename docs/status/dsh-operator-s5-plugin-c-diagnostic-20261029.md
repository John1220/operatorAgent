# S5：C-only 三任务诊断（2026-10-29）

状态：三个 C 格全部通过开发与独立最终 L2，汇总为 passed 3、failed 0、unknown 0。模型未使用插件 `operator_verify`，因此本轮只确立“安装插件的 DSH 环境可完成三个任务”，没有确立插件验证入口的真实模型使用效果或 A/C 增益。S5/S6 未封版。

## 范围与固定条件

用户要求先把 C 的三个 case 跑通。为此给现有驱动增加 `freeze --mode c-only`，采用独立的 `s5-plugin-c-diagnostic-plan-v1` 与结果汇总；默认 A/C 六格协议保留。诊断只派发 Matmul-tail-fp16、LayerNorm-tail、Softmax-extreme 的 C 格，不产生 A/C 差值或增益推断。

- 模型：`deepseek-v4.1-flash-1017`，Responses，reasoning medium；单次最多 32,768 输出 token。
- 每格预算：24 次模型调用、总输出预留 786,432 token、墙钟 2,400 秒；最多 2 次开发 L2 + 1 次独立最终 L2，所有验证入口共用额度。
- GPU：algorithm-h20 GPU2，NVIDIA H20，UUID `GPU-13c6c4b4-45ce-8963-1122-df2b3d72fb76`；沿用共享条件，逐格核对身份、至少 8,192 MiB 可用显存、利用率至多 5% 和已知进程名单。
- 重新运行无模型 readiness：A/C 各 10 次脚本化请求通过；真实模型调用 0，GPU 执行 0。C 的真实 MCP schema/instructions 与验证回执在 mock 后端下可用。
- 计划哈希：`7657260bfbefcef3235fe8cd248fe8a48322ebb7dacf8b7c92296defc3fb3f40`。
- 汇总哈希：`4ca123267510aae5415c4cfdd437870ed772413e3889392064e49f514f888914`。

## 结果

| 任务 | 模型调用预留 / 完成 | 开发验证 | 独立最终验证 | 格耗时 | 模型的开发验证入口 |
|---|---:|---|---|---:|---|
| Matmul-tail-fp16 | 6 / 6 | L1/L2 通过，15 case | L1/L2 通过，15 case | 363.758 秒 | 公共 `check.py` |
| LayerNorm-tail | 7 / 7 | L1/L2 通过，12 case | L1/L2 通过，12 case | 269.738 秒 | 公共 `check.py` |
| Softmax-extreme | 7 / 7 | L1/L2 通过，12 case | L1/L2 通过，12 case | 330.161 秒 | 公共 `check.py` |

共预留并完成 20 次模型调用，未耗尽额度，网关故障记录为空。三格累计墙钟 963.657 秒，约 16 分 4 秒；开发与最终验证累计 717.589 秒，占约 74.5%。服务商原始 usage 均缺失，实际 token 使用和费用未知；本地调用账本不替代服务商账单。

本批 LayerNorm 最终复查通过，之前 A/C 批次的传输 unknown 没有再次出现。尚不能确定之前传输异常的底层根因；旧结果继续保留 unknown。本次是新计划、新候选的独立诊断，不能覆盖或补齐旧 A/C 批次。

## 插件能力的证据边界

Matmul、LayerNorm 的真实模型调用了 `operator_capabilities`；Softmax 没有插件工具调用。三格的开发验证均经公共 Python 入口，转录审计均为完整，但 `plugin_verification_observed=false`。独立最终验证由监督器发起，也不能记作模型主动使用插件。

因此，下一项产品验收应确认模型在正常插件说明引导下实际使用 `operator_verify`，取得候选、MCP 调用、GPU 反馈和最终通过之间的完整记录。若明确要求走插件工具，应单列为插件功能验收；后续正式 A/C 比较继续使用相同初始任务说明，不把单独的实验提示当作产品效果。

手动打开 DSH 有助于观察工具发现、参数填写和反馈是否好用。保持相同完整套件时，手动操作仍需承担 GPU 验证耗时；这轮大约四分之三墙钟已用于验证，无需用户接手重跑本批。手动体验应独立记录，不能作为旧 A/C 未执行格的替代结果。

readiness 的 `network_isolation_verified=false` 仍成立；本轮不是网络隔离或同 UID 对抗验证，也未执行 L3 性能评测。`formal_s5_credit=false`、`inference_allowed=false`。

## 留存与实现验证

证据目录：`output/plugin_completion/20261029-s5-plugin-c-diagnostic-v1/`，包含 `readiness/`、`frozen-plan/`、`execution/aggregate.json`、逐格预算、事件、工作区归档、开发/最终验证和审计回执。`frozen-source/` 留存本次计划绑定的 29 个实现文件，复制时逐一校验 SHA-256；后续维护不覆盖本批源码身份。

C-only 新增检查覆盖冻结三格、汇总保留 unknown、拒绝异批记录、只派发 C、设施 unknown 后停止和 A/C CLI 推断字段保持一致。相关环境、驱动与汇总回归共 **52 项通过**。执行后 GPU2 仍为 22,246 MiB 可用、0% 利用率、原有 7 个已知进程；未停止其他服务。

相关：[A/C 停批记录](dsh-operator-s5-plugin-only-ac-results-20261029.md)、[A/C 设计](../design/s5-plugin-only-comparison-20261029.md)。
