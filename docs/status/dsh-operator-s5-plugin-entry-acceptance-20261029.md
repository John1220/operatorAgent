# DSH `operator_verify` 真实插件入口验收（2026-10-29）

状态：**单任务入口验收通过**。已确认候选实际经 DSH 的 `mcp__operator__operator_verify` 提交到 H20 GPU2，开发期 L2 反馈回到 DSH，随后独立终评的最终候选通过完整 L2。该结果不计入正式 S5 分母，不构成 A/C 对比或插件收益结论。

## 冻结范围

- 冻结计划：`eadca10deb7ea847f18a5b730a5893995b7fc616c58300b41cb6aae927ef028d`。
- 单格：`layernorm-tail`，仅 C（DSH + 插件），一次模型会话。
- 模型：DeepSeek `deepseek-v4.1-flash-1017`，reasoning `medium`，每次最多 32768 输出 token。
- 模型预算：最多 24 次调用，本次预留并完成 5 次；DSH 汇总耗时 256.446 秒。调用账本的实际 token 统计见执行记录；服务商账单金额未知。
- 验证目标：H20 `algorithm-h20`，GPU 2，UUID `GPU-13c6c4b4-45ce-8963-1122-df2b3d72fb76`。执行前资源快照为 0% 利用率、22246 MiB 空闲；当时有 7 个其他计算进程，因此这是共享 GPU 环境，不是独占租约。远端日志确认固定使用 GPU 2 并设置 `CUDA_VISIBLE_DEVICES=2`。

## 调用链结果

1. DSH 经 MCP relay 发起一次 `mcp__operator__operator_verify`，参数为 `backend=ssh`、`stop_after=L2`。relay 记录到 `operator_verify` 的 `tools/call` 已应答；DSH 转录中的 tool call/result 配对，和 development audit 的调用 ID、服务响应哈希相符。
2. 开发候选 SHA256 为 `374a9981093d74f3a7ef538f0cf0fc64df01fd2bab6bb567f88c90eb4e6b28ea`。服务返回 L1、L2 均通过，包含生产 CorrectnessSuite 的 12-case 通过结果；DSH 收到同一服务响应，继续完成最终源码。
3. DSH 最终候选 SHA256 为 `df189e4c0ab4bef63000147d5ad2f5844b805ea1797c099df21dd2edce69e0fd`。独立终评结果的 `source_sha256` 与候选一致，执行后端为 H20/SSH，`remote_backend_executed=true`、`mock_only=false`、`full_l2_passed=true`。L1/L2 均通过，CorrectnessSuite 12/12 通过，最大误差 `2.384185791015625e-7`。

## 汇总修正

首次汇总为 `not_demonstrated`，原因是最终服务适配层把完整 L2 布尔值存为 `passed`，而入口汇总只检查 `full_l2_passed`。独立终评原始结果中该字段明确为 `true`，候选哈希也一致。已让服务响应显式带出 `full_l2_passed`，并令汇总器兼容旧 `passed` 字段（只有字段缺失时才回退；显式 `null` 仍判为未证明）。新增回归覆盖了旧格式和显式 unknown。没有重跑模型/GPU，也没有改写原始 run record；通过原始转录、服务账本和终评结果重算后，三个验收项均为 true。

## 证据与边界

证据目录：`output/plugin_completion/20261029-s5-plugin-entry-acceptance-v1/`。关键文件为冻结计划、执行 `aggregate.json`、`acceptance-audit.json`、DSH `events.json`、`development-audit.json`、MCP relay 快照、GPU 资源检查以及 `supervised-verification/002-final/evaluation/result.json` 和底层 12-case 评测记录。该目录由 Git 忽略，应作为本机私有验收材料保留。

本次证明的是一次真实产品入口端到端可用：插件提交、远端 GPU 验证、反馈回到 DSH、最终候选独立通过。单个 LayerNorm case 不能证明插件提升开发能力、覆盖其他算子或失败路径、改善耗时/调用数，也不能证明 DSH 因这次反馈而改进候选；development audit 的 `feedback_causality_proven` 仍为 false。此次不授予正式 S5 credit，不替代 A/C 实验。

相关汇总修正的定向回归：`python3.9 -m pytest tests/test_s5_plugin_workflow.py tests/test_s5_workflow.py -q`，46 项通过。
