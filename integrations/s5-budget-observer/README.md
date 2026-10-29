# S5 实验性 DSH 外层预算拦截器

这个目录是 S5 **离线实验组件**，不安装、不启用到用户 DSH profile。它不是完整的三组对照，也不能证明外层与内层的所有模型请求都已被拦截。

`interceptor.mjs` 向调用者提供 `createS5Interceptor(ledger)` 和 `installS5Interceptor(ctx, ledger)`：后者在 DSH `llm/stream` 上注册全局、prepend 的 waterfall。只有消费返回的流时才异步预留；提交预留和派发标记成功后才调用 `next()`。预留前要求明确的 `maxTokens`，计算消息和配置的 SHA256，但不保存或输出原文。hook 不按 session ID 过滤：普通、标题和 compaction 请求必须共用预算；未知 purpose 拒绝。请求不符合可记账契约或账本操作失败会在本 hook 实例锁存失败，后续调用拒绝；已预留额度不退款。

`cli-ledger.mjs` 将外层接到 `operator_agent.s5_shared_budget` 的 SQLite 同机账本，要求绝对路径与冻结任务 ID，首次核对任务 ID 后固定文件身份，后续操作前再次核对。参数只带摘要、调用 ID 和上限，子进程不继承 API 密钥。共享账本是协作式的，不是针对恶意同 UID 进程的隔离。外层和内层实验路径必须指向相同账本；即使内层入口已接线，也不能仅凭此认定所有可能请求都经过拦截。

DSH/pi-ai 映射的 usage 可能是缺失响应被初始化后的零值：hook 只记录出现了归一化 usage（`unverified`）或未出现（`missing`），不会把这些值当可信服务端 token，也不会填零。未来要报告可信 usage，必须另接可核对原始响应、绑定本次调用的采集并经测试，而不是信任此 hook 或调用方声明。`dispatch` 是在 `next()` 前持久化的保守意图，进程在标记与调用之间被中断时可多计，不证明服务端收到请求。预留后、派发前取消留下未派发且未完成的预留；派发/完成回执失败、进程突然退出或子进程超时会保留未知并要求停止批次，不推断费用或退款。

## 离线测试

```sh
node --test integrations/s5-budget-observer/test/*.test.mjs
S5_DSH_ROOT='/Applications/DSH Desktop.app/Contents/Resources/app/node_modules/@deepseek-ai' \
  node --test integrations/s5-budget-observer/test/host-integration.test.mjs
```

测试使用本地 SQLite、Python 进程、实际已安装的 DSH waterfall 和假 adapter；不会发送真实模型请求。找不到指定 DSH 安装时对应宿主测试为 skip。覆盖共享额度、异常、取消、重试计数、`next()` 异常、未终止流、重复消费、无 session 辅助调用、伪零 usage、身份漂移和并发失败熔断。

## 正式使用前仍需解决

- 单个 hook 的存在不证明所有 DSH 进程、子 agent、后台辅助调用和其他网络路径都经过它；加载顺序和绕过路径须以运行中实际宿主/profile做端到端校验。某个优先级更高的 hook 可短路 waterfall。没有验证过的请求来源必须停止正式对照。
- 文件身份和任务 ID 检查与实际数据库操作之间仍有竞争窗口，并非针对同 UID 的安全保证；跨主机 SQLite 文件、provider 内部网络重试和服务商账单也不在此组件的保证范围。无真实 provider usage 证据时保留 unknown。
- 可写的共享账本接收不可逆 abort 后，会拒绝后续预留与派发，但已通过派发边界的调用仍可能执行。身份变化、损坏或不可用时无法证明 abort 已成功：全批次监督器必须停止所有参与进程，不能仅依赖本 hook 的锁存。当前监督和全路径覆盖尚未完成。
