# v0.7.1 DSH 模型委托验收（2026-11-05）

本轮验证显式启用的固定 DSH 路由：Python 继续构造角色消息、执行生成/修复循环并记录任务预算；实际模型请求通过插件的回环转发交给 DSH。当前为开发预览，未封版或公开发布。v0.8 的优化增强范围保持原路线。

## 安装身份与范围

在隔离 HOME、DSH_HOME 和全新 Python 3.12 venv 中安装 wheel 与 tgz，Python 以 `-I` 启动。宿主为已安装的 DSH `0.2.1-rc.2`，运行时 Node `24.14.1`。个人 desktop profile 保持原样。

| 安装包 | SHA256 |
|---|---|
| `operator_agent-0.7.1.dev1-py3-none-any.whl` | `57cb44b4142d20eb53c7068b793c8a70820bf3ece9574755dc362259302eb4b9` |
| `operator_agent-0.7.1.dev1.tar.gz` | `01be69c0c91ca0b79ef2658a1b5736b479dded2c4a1dd2b0cc66256eb0dc2bf7` |
| `dsh-operator-agent-0.2.1-rc.2.tgz` | `b82ad7abf8aabe3f1898a3ea1314ea3f00cc72d3ae63f9cda0b5984e4d05f28f` |

安装包源码清单 SHA256 为 `bdfb5f32c90463ca54a32f5da68475d97ce79950cb23372f535992547661f72a`。验收后的文档补充另存于交付指南，保留上述已安装包的原字节和构建源码快照。

外层由确定性测试脚本驱动真实 DSH 的 MCP 工具调用，内层使用真实服务商模型。因此本轮证明工具与模型委托链路，不能据此推断外层模型自主选择工具的成功率。凭据由 DSH 读取，Python worker 只收到单独的本地转发凭据。没有新增付费模型或隐式 fallback。

## 验收结果

最终验证路由为已有阿里云 OpenAI Completions 路由，模型 `deepseek-v4.1-flash-1017`。生成、修复、取消和 H20 验收采用同一兼容配置。转发默认关闭，路由在隔离 profile 中固定。

| 流程 | 结果 | 边界 |
|---|---|---|
| FakeBackend + 真实模型生成 | L1/L2 通过，3 次模型调用 | NumPy CPU 执行，保持 `FAKE / test_only` |
| 从错误候选修复 | 初始 L2 失败，Fixer 修复后 L2 通过，1 次模型调用 | 真实模型调用，没有替换为参考答案 |
| 在飞取消 | 任务 `cancelled`、worker 已回收，DSH 流观察到 abort | 服务商是否停止执行/计费仍为 unknown |
| H20 FP16 Softmax `4096×4096` | 新生成候选 L2 通过，独立终评 12/12 个套件用例通过，源码和报告已导出 | L3 未达标，测量为 `degraded` |

每个流程分别保留工具回包、Python 模型账本和 DSH 请求/结束/用量观察记录。H20 的 5 次调用均正常回答：Designer 2 次、Coder 2 次、Search 1 次。成功生成、修复及 H20 的共 9 次回答逐一核对了 Python 派发数、宿主观察数、temperature/maxTokens 和用量；实际出站请求使用 `enable_thinking: false`，未提高 Designer 的 2048 token 上限。

H20 模型账本观测到：非缓存输入 18,000 token，缓存读取 8,192 token，输出 4,406 token，服务商总计 30,598 token。5 次调用的总 token 均可核对；未提供缓存字段的调用仍按字段保持 unknown，费用保持 unknown。

## H20 独立终评身份

生成任务预算为最多 2 轮、12 次模型调用、600 秒和 20 个 oracle 单位。独立终评使用新 run 目录和原冻结契约，复核选中源码的完整哈希。验收前后均核对同一 GPU、驱动、运行时及 15 个 runner 文件哈希，不部署或替换远端环境。

- 运行时：Python `3.10.12`、Torch `2.14.1+cu130`、Triton `3.8.0`、CUDA `13.0`、驱动 `580.105.08`。
- 契约 SHA256：`8fd7e3ed178667c3e6ade373b032c98edef82454de6393edcde72bdda6b8413c`。
- 新候选 SHA256：`bad6f5c5f663f3718cf8372ede83356f26d069bf9e877e37889f3b44f3982faf`。
- 独立终评报告 SHA256：`35f96366b3b39156cc4475a2736df6da7a3aea8987547fb596deadfcbc5e3062`。
- 正确性套件 SHA256：`0453674dc3138df396e7455a784e27a0f885a398fba55834d6073f0fbb706ecd`；12 个声明用例全部通过，不外推未声明输入。

独立终评读数为 speedup `0.803`，`L3=false`、`validity=degraded`；launch 地板占实测时间 `14.7%`，超过 10% 判据。该读数没有建立性能提升或进展资格，没有进入长期记忆。本轮仅验收委托链路和正确性，未形成公平 A/B 或业务收益结论。

## 保留的失败与修复

1. **开发前包的 FakeBackend 提示缺少执行限制。** 真实模型生成 Torch 候选，FakeBackend 仅允许 NumPy/math，L1/L2 失败。现由 backend 显式提供 `generation_instructions`，统一注入默认 PromptManager；生成和独立 `operator_repair` 都接入该限制，并增加回归。安装包升级到 Python dev1 / 插件 rc.2 后复验通过。
2. **desktop 当前默认免费路由配额耗尽。** 生成失败，修复流明确返回 `402 free_request_quota_exhausted`；取消验收通过。所有已派发调用保留，没有将失败计作成功或退还预算。后续使用已有阿里云路由，未把默认免费路由记录为已可用。
3. **原阿里云兼容配置没有发出关闭思考字段。** FakeBackend 流程虽通过，首个 H20 Designer 在 2048 输出 token 处 `max-tokens`，转发层拒绝不完整输出。出站观察显示原配置使用 `max_completion_tokens`，未包含 `enable_thinking`。在隔离模型条目配置 `compat.thinkingFormat: qwen`、`compat.maxTokensField: max_tokens` 后，短请求及 H20 全流程均核对了实际字段。成功轮沿用相同安装包和预算，没有放宽截断规则。

配置说明见[模型转发配置](../../integrations/dsh-operator/README.md#31-v071--通过-dsh-发出内层模型请求)。上述 `qwen` 是宿主适配器的字段格式名称，其他服务需核对其协议。

## 回归与后续范围

- Python 相关生成、修复、后端及仓库守卫：**310 passed / 12 subtests passed**，14.33 秒。
- 同一已安装 wheel/tgz 的完整插件测试：**66 passed / 0 failed / 0 skipped**，包括真实 DSH 宿主加本地模型 fixture。
- 这是相关范围回归，不替代 v0.7.0 已完成的全量回归，也不把本轮记作新的全量通过。

本轮交付允许在已验证固定路由上试用。按会话选择宿主路由、将内层调用归因到主会话，以及扩大服务商兼容范围仍待开发和验收；主会话的用户/项目上下文不会自动混入算子角色消息。性能诊断与搜索增强继续归 v0.8。本轮没有发布 `v0.7.1` 正式版。

本文件保留开发期 dev1 / rc.2 的工件身份与验收事实；后续正式工件的本地封版另见[交付记录 §12](../testing/v07-delivery.md#12-v071-本地封版)，不将本文件的哈希或实际模型调用改记到新工件。
