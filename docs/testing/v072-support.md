# v0.7.2 支持清单（有限支持）

更新日期：2026-11-06。正式本地版身份为 Python **0.7.2** / DSH 插件 **0.2.2**（npm 已发布 0.2.4/0.2.5，见插件 CHANGELOG）；Python 包以分布名 **`dsh-operator-agent`** 发布在 PyPI（import 名 `operator_agent`），原名 `operator-agent` 在 PyPI 不可用。工具契约 **12**。本文描述实现与支持边界；正式工件的安装状态以本地安装回执和实际 UI 核对为准，不沿用候选安装结论。

原生集成测试基线为 DSH **0.2.1-rc.2**、Node **24.14.1**；DSH NEXT **0.2.1-rc.4** 已有真实用户 CPU 功能闭环。两类证据分别说明下述范围，不扩大为完整宿主兼容矩阵；控制面默认 Python **3.12**。其他组合需另行验收。前版 [v0.7.1 支持清单](v071-support.md)和[交付回退记录](v07-delivery.md#12-v071-本地封版)保留。

## 1. 内外层职责与工具面

| 责任 | 本版归属与边界 |
|---|---|
| 用户、项目与模型上下文 | DSH 保留会话，选择阶段材料并组装最终 messages；主模型生成、修复候选，决定下一步或结束 |
| 权威领域事实 | Python 插件冻结任务、ABI、容差和 L2 套件，提供生成规约、技能、确定性诊断及资源预检；聊天与原始角色模板不能覆盖冻结事实 |
| 执行、验证与证据 | 插件验证外层给出的源码，持久保存候选、报告、预算与完整证据；完整 L2 和独立终评资格由硬门判定 |
| host 工具 | capabilities / prepare / context / verify / resume / status / cancel；隐藏并拒绝内层 run / repair / bench，内部模型调用为 0 |
| 兼容入口 | 独立 CLI 保留；legacy MCP 需同时设置 `OPERATOR_AGENT_WORKFLOW=legacy` 和 `hostModelBudget: false`，原有 modelRelay 仍为显式启用 |

context 返回材料而非渲染好的模型消息。必需的冻结事实、ABI、生成规约与绑定反馈可核对版本和哈希；模板、技能按需组合。输入值、seed 和参考源码不导出。材料上限 40 KiB，完整 MCP context 文本上限 48,000 UTF-16 字符；超限明确拒绝，不静默裁掉约束。

## 2. 任务预算与有限会话支持

默认 `host_budget` 为墙钟 **600 s**、验证派发 **6 次**（其中 **1 次**固定保留独立终评）、绑定后主模型派发 **12 次**、单次输出上限 **8192 token**、累计输出预留 **98304 token**。prepare 时冻结这些值；重复 prepare、失败、取消及恢复均不补额度。

| 项目 | 支持行为 | 限制 |
|---|---|---|
| 验证预算 | Python 跨调用账本约束派发及墙钟；final 要求同源码已有完整 L2，使用新的完整证据，并关闭开发 | 单位是验证派发，不是逐 case 调用或 GPU 秒数；直接 Python 库调用没有 MCP supervisor 的在飞终止能力 |
| 主模型硬门 | 默认组合包 `hostModelBudget: {}` 在成功 prepare/resume 后绑定原任务；输出参数受限，并在 provider 前原子预留调用和输出额度 | 仅限绑定后的原生主模型派发；绑定前准备、标题等辅助调用、legacy 转发及任务结束后的普通讨论不计入；不是全会话或精确费用上限 |
| 结算 | 失败、取消、遗留未完成及输出少于预留均不退款；有效 API usage 单列，缺失或不一致保持 unknown | 本地 abort 不证明服务商已停止执行或计费；观测 provider 输出越界后拒绝后续请求，不能撤销已有输出 |
| 会话范围 | 一次一个活动任务，结束后可在同一主会话直接提出新的自然语言需求，自动开启下一项；final 后原轮汇总仍计入原任务 | 活动任务仅七个算子工具；讨论阶段只读算子工具，普通讨论按 DSH 正常计费；fork/委托会话不支持 host 开发 |
| 项目取材 | 在 prepare 前读取项目材料；已有用户/项目背景及后续用户消息仍由 DSH 保留；绑定后通过 context 取得领域材料，验证/resume/status 取得候选与证据引用 | 本版没有通用的绑定后文件读取工具；白名单是预算支持范围，不是分工必然要求 |

`hostModelBudget` 的配置只接受 `{}` 或关闭值 `false`，限额来自冻结 Python policy；不能另配一套 profile 额度。只用 Python policy 或看到有效模型收据，不证明其他 MCP 客户端已安装主模型发送前硬门。

模型截止同时受原墙钟、持久的本机单调时间锚点和本进程定时器约束，时钟回拨或 DSH 停机不会续时。恢复需确认同一次系统启动：当前实现读取 macOS 的 `kern.bootsessionuuid` 或 Linux 的 `boot_id`；不能读取、系统重启或时间证据缺失时明确拒绝模型派发，仍可直接查询和取消，不猜测剩余时间。

## 3. 恢复、干预与无需模型的出口

同一 DSH session 重启会回放自己的原生用户轮次、prepare/resume 回执、可选的 `/op new` command/run 与成功 command/done，以及 turn/end，再结合持久账本恢复当前阶段，在首个后续主模型请求前生效。已结束或已切换的任务不会仅因旧 prepare 回执重新绑定。原 task_id、契约、deadline、候选和调用记录保留；旧 owner 无法确认退出时拒绝接管，确认退出后的遗留请求可标 abandoned，额度仍不退款。旧 policy 缺少模型字段时不自动补模型信用，损坏账本或契约漂移明确拒绝。

`operator_resume(run_dir, backend)` 只读返回原预算、最近完成轮、候选/报告引用、`next_action` 与按需取材参数，不调用模型、不重新 prepare、不清理未知验证 owner，也不重新打开已终评开发。它不返回完整源码正文。用户追加要求由 DSH 原生会话、steer 或 followup 进入后续模型上下文；保持契约时继续原预算。改变 shape、精度等冻结判据属于独立任务；用户在新轮次明确提出新需求后，可在同一主会话自动开启，原任务不可重置。

模型预算可能在终评或汇总前耗尽；验证 final 预留不会额外授予模型调用。此时如实报告未完成，不能把普通完整 L2 候选称为已独立终评。用户仍可直接使用：

```text
/op new [new_run_dir]
/op status
/op status <run_dir>
/op cancel <run_dir> <invocation_id>
```

这些命令不调用模型。无参 `/op status` 查询本会话执行、讨论或等待新任务的阶段；`/op status <run_dir>` 显示剩余预算、候选或最近失败验证的源码/报告路径，以及在飞工具实例；cancel 请求后还需查清理与终态。`host_enforced_receipt` 表示 Python 核对了宿主账本收据，核对范围是身份、哈希和累计事实；`not_bound`、`receipt_invalid`、`legacy_policy_without_model_limits` 不作为有模型额度的证明。

日常直接描述新需求即可，无需 `/op new`。意图由 DSH 模型判断；解释旧结果保持讨论，含糊的“继续”应先澄清。插件只核对新用户轮次、目录、账本与清理状态，每轮最多成功 prepare 一个新任务，不在同轮自动循环建目录。`/op new [new_run_dir]` 保留为用户直接输入的手动入口；下一次 prepare 才创建新契约与预算。旧任务的截止、消耗和记录不重置，不允许复用旧任务目录补额度。终评后的原轮汇总仍计入旧任务；原轮结束或预算停止后进入讨论阶段，普通讨论不写旧任务模型账本，但 DSH 仍正常调用模型并计费。讨论阶段只开放领域材料/状态等只读算子入口，不能继续旧任务开发。未结算模型请求、在飞验证或未知清理状态会阻止新执行，须先查询/取消并确认终态。

## 4. 验收范围与未建立的结论

历史 dev2 工件已完成真实 DSH 外层 FakeBackend 生成/修复、H20 FP16 Softmax `4096×4096` 主用例全量比对及 **12 个 L2 case**、同源码独立终评和取消后原预算恢复，身份与限制见[开发验收记录](../status/v072-host-context-acceptance-20261105.md)。其 `not_enforced_by_plugin` 是当时模型预算的事实，不用正式版身份改写。

rc1 新增预算已有真实 DSH + 本地测试 provider 的调用耗尽、输出预留耗尽、在飞模型取消及同 session 重启测试，证明测试请求在 provider 前被硬门阻止且不补额度。rc1 的真实服务商恢复/用户干预、完整回归及 H20 记录见[封版待办](../status/v072-todo.md)与所链验收文档。rc2/rc.5 生命周期修正的回归、候选包安装与新试跑证据以[生命周期验收记录](../status/v072-session-lifecycle-acceptance-20261106.md)为准。rc3/rc.6 自动入口的原生测试与回归见[自动入口验收](../status/v072-auto-task-acceptance-20261106.md)。这些记录保留各自候选身份，不能冒称正式工件已完成同一轮安装或付费模型验收。

DSH NEXT `0.2.1-rc.4` 的用户功能证据包括 rc1/rc.4 的[ReLU 试跑](../examples/v072-relu-user-review.md)，以及 rc3/rc.6 的[失败修复](../examples/v072-repair-user-review.md)、[普通讨论与简短自然语言新任务](../examples/v072-natural-user-review.md)。后者证明无需指定工具、预算、目录或 `/op new` 的实际自动入口；全部使用 CPU FakeBackend，不代表 RC4 全矩阵兼容或 GPU 性能验收。

不扩大 v0.7.1 的算子、精度、布局、硬件或运行时承诺。FakeBackend 恒标 `FAKE / test_only`，不能取得 GPU 性能资格。H20 历史读数为 `uninformative`，不证明性能提升；本版控制面交付不改变 L3 判据，性能诊断与搜索增强继续归 v0.8。

多代理并发、第二宿主、通用 Watchdog、业务框架接入、任意失控核恢复和敌意代码隔离均不在本轮范围。执行源码/runner、提示或 skills、冻结判据、GPU/运行时、DSH 适配器或服务商协议变化时应重新核对对应证据，不能直接沿用历史资格。
