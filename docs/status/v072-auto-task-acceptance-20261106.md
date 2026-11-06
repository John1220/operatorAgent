# 自动任务入口验收（2026-11-06）

Python `0.7.2rc3` / 插件 `0.2.2-rc.6`。用户无需 `/op new`，DSH 根据新用户轮次的任务要求决定 prepare；普通讨论保持只读，含糊的“继续”应先澄清。意图由 DSH 模型判断，插件确定性检查轮次、目录、旧任务清理和预算，不能证明模型的语义理解必然正确。

成功 prepare 每轮最多一次，新任务须全新目录。原生用户消息之后才允许自动入口，runtime-context/skill/tool/fork 不授予入口；旧任务未结算、未知 prepare 或 worker 未回收则拒绝切换。旧任务额度、截止、结果不重置。`/op new` 保留手动兼容。

新增纯回放、hook 与真实 DSH 本地 provider 测试，验证无命令两项任务、相同 session 不同 task_id、旧模型账本逐字未变及跨进程恢复。没有使用真实付费模型或 GPU。第一次真实宿主测试暴露 tool/call 早于 guard 的时序，已修为核对同 callId，未跳过测试；初轮 relay 网络 ECONNRESET 单独保留日志并复验。

最终 wheel/tgz 原生集成与全插件回归：**142 passed / 0 failed / 0 skipped**；MCP 与 hygiene：**78 passed**。日志分别为 `/private/tmp/operator-auto-packaged.log`、`/private/tmp/operator-auto-python.log`。干净 rc3 安装与 pip check 通过；DSH NEXT 插件 rc.6 两个组件启用，原会话能力命令确认 Python rc3。未代用户执行真实模型新 case。

本地工件：wheel SHA256 `453da6811fa1d6390b000f492fea2eda9d4990baf8d3541c359b8516ec086c8a`；tgz SHA256 `784b8040af0359747051f6e2536f3a6502a22d4e05fcb1e1b7bbf0ae549d6af3`。持久快照及日志位于 `/Users/.../op-agent-delivery/v0.7.2rc3-auto/`，原模型账本保持 rc2 验收时的哈希。


## 后续真实用户验收补充

本机安装完成后，用户在同一 session 运行了[失败修复 case](../examples/v072-repair-user-review.md)以及[普通讨论和简短自然语言新任务](../examples/v072-natural-user-review.md)。前者 21 项核对通过；后者 16 项通过，旧手动授权早已消费且无新增 command/run，确认实际无命令自动入口。均为真实 DSH 模型与 CPU FakeBackend，不扩大 GPU 或性能资格。上述“未代用户执行”描述保留安装时点事实，不表示用户之后没有完成试跑。
