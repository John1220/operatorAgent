# 生产入口任务预算

2026-10-26，落实开发评审 R2。适用入口为 MCP `operator_run`、CLI `run` 及其无人值守执行进程。预算接线时工具契约版本为 5；S1 独立证据留存升级为 6，预算语义保持。

2026-10-27 的工具契约 9 将相同参数接入 MCP `operator_repair` 和 CLI `repair`：每次修复任务独立持久化模型与验证预算，CLI/MCP 受监督墙钟截止；直接 Python 库调用只有合作式截止。修复只消费非确认预留，未派发不得伪装成判定通过。生产账本另持久化调用时间和经验证的 API usage；缺失 usage 仍为未知。下文关于无人值守入口的历史验收数字仍是当时结果，不表示 S4 真机验收已完成；当前状态见 [S4 状态](../status/dsh-operator-s4-20261027.md)。

## 配置

| 参数 | 有限默认值 | 计量口径 |
|---|---:|---|
| `wall_clock_s` / `--wall-clock-s` | 1800 秒 | 从本机监督进程开始计时的任务执行时限 |
| `max_llm_calls` / `--max-llm-calls` | 24 次 | 所有生产 Agent 共用的派发前调用预留；失败不退款 |
| `oracle_total` / `--oracle-total` | 20 单位 | 按调用方提供的输入组计的候选验证派发预留；最终确认额度不可挪用 |

这些是运行选项，不修改冻结的算子语义、参考实现或容差。布尔值、非有限数、非正数、错误类型和未知选项会被拒绝。配置写入 `autonomous.json.task_budget`；旧配置缺少该字段时使用上述有限默认值。直接构造 `Conductor` 或严格单候选实验仍由其调用者提供配置，不把旧证据追认为使用了新预算。

```sh
python -m operator_agent.cli run --request 'matmul fp32' --shapes '8,8,8' \
  --run-dir /tmp/my-operator-run --backend mock \
  --wall-clock-s 1800 --max-llm-calls 24 --oracle-total 20
```

`backend=mock` 仍需要内层模型配置；上述示例不是无需模型的离线模式。本次实现验证使用本地假模型服务，没有调用真实模型或 GPU。

## 模型账本

生产入口使用 `ProductionLLMCallBudget`，所有 Designer、Coder、Fixer、Search、Planner、Reflexion 调用共享一份账本。当前无人值守入口仍关闭 CoderPipeline；显式调用标签白名单不意味着入口默认启用了所有策略。

派发前先持久化预留，失败、JSON 解析失败和预检后的显式重试都不退款。普通请求失败不毒化剩余额度，Agent 仍按原来的显式策略决定是否再调用；预算耗尽、身份漂移或日志失败不能换账本继续派发。生产任务关闭隐式 fallback，防止未记账调用或静默改变模型身份。

每次请求使用单次超时、调用预算超时与任务剩余时间中的最小值。额外保留每次 40960 输出 token 上限、总输出预留上限和请求/响应各 1 MiB 字节上限。这些不代表精确输入 token 或服务商账单硬上限。

`model-journal/` 保存 limits、模型配置身份、调用元数据、哈希、预留和结果状态；不保存原始请求消息或响应源码。`summary.json` 在绑定、预留、完成、拒绝和解析失败后刷新。监督快照核对摘要、调用文件集合和内容、limits、identity；日志缺失、损坏或失效时用量保持未知，不能显示为 0。

严格单候选原有 `LLMCallBudget` 仍保留失败后终止、原始审计数据及一次调用约束。

## 验证预留

有限 `BudgetPlan.oracle_total` 采用 `candidate_verification_dispatch_per_input_group` 单位：

- 一次完整验证，在派发前按输入组数预留；L1 就失败或后端异常也不退款。
- 决赛候选的 L1/L2、随后单独派发的 L3、最终独立确认分别预留。
- 确认使用原有独立份额；探索和精修不能挪用。确认额度不足不会调用后端，也不会补造确认成功。
- 内部正确性套件的多种 seed、布局、计时重复和原始 GPU launch 不按此单位逐次计数。因此它不是 GPU launch 次数或 GPU 秒数上限。
- 既有标定规则保持：标定不因额度不足而跳过，先记账并显式报告超计划；它是总额度的已声明例外，仍受任务总时限约束。

`verification-budget.json` 在初始状态、预留和拒绝时更新，包含已花、剩余、确认预留和单位。最终 `result.json` 保存确认之后的预算与停止原因，并支持加载。预算阻止首次验证时会生成明确的未派发失败记录，避免空验证列表被当作通过。

已有实验驱动的新计划同步迁移计量单位：固定搜索和 LLM 搜索额度由 5 调整为 7，有界单候选生成由 4 调整为 20，以覆盖原定两轮及两个输入组的确认预留。候选、模型调用、轮数、RPC 和测量窗口上限保持原值；这次只调整新计划的预留单位，不扩大既定实测序列，也不改写旧 sealed 计划或证据。新计划顶层记录 `budget_accounting`，旧计划继续按原身份接受漂移检查。

## 总时限与状态

`autonomous.main()` 是独立监督进程，实际工作放在另一个进程。监督进程持续观察共享 monotonic 截止点，在到期时终止本机 worker 及观察到的后代进程组，执行有界清理；阻塞模型调用和本机 native/HTTP guard 不依赖下一轮开始才停止。进程组发现失败会记录 `cleanup_unknown`，不会声称已证明清理了所有后代。正常、异常和截止退出都会清理已观察的组。

状态中的 PID 属于监督进程，另有 `worker_pid`。每次运行有独立 invocation 身份，不能用旧 worker 结果顶替本次终态；截止前已完成的普通失败，不能仅因监督进程稍晚读取就被改写成预算停止。

S2 起阶段为 `running / done / failed / cancelled / lost`。`done` 只表示本次流程结束，可能是 `budget_llm_calls`、`budget_wall_clock`、验证额度耗尽等；必须同时查看 `stopped_reason`、`all_passed` 和验证证据。任务预算停止不会被摘要报成全部完成验证；原候选验证是否通过另外保留在 `verification_all_passed` 中。

`operator_status` 返回 `task_budget`、`budget`、`stopped_reason`、生成证据路径及清理状态。缺少账本的旧状态文件不补造用量或上限。运行期间定期刷新，终态保存在 `run-status.json` 和 `conductor.json`；worker 自身结果单独写入 `worker-result.json`，避免多个进程竞争写状态文件。

本机超时终止不证明服务商停止推理或远端 GPU 已取消执行；远端行为仍受后端自身的执行保护约束。报告明确保留 `remote_cancellation_proven=false`、`provider_billing_cap=false`、`gpu_seconds=null`。监督时限不包含 MCP 的前置契约准备和 Python 启动开销，退出后的进程发现/清理也可能有短暂开销。

## 验收

本地替身覆盖：调用额度在 Designer 与 Coder 之间耗尽、预检重试共享账本、失败请求不退款、最终确认预留、输入组计数、阻塞 HTTP 超时、独立 session 后代清理、缺失日志保持未知、普通失败完成时点、配置透传和旧状态兼容。真实模型质量、GPU 性能和 Q1 内外层对照不在本次变更范围。

本次验收：Python 全量 **5346 通过、1 跳过**；DSH 插件 **53 通过**；Golden 本轮 **4/4 通过**；`git diff --check` 通过。Golden 原基线与容忍范围未修改，模拟后端结果不构成真实 GPU 性能结论。

S2 已为监督任务、取消与 worker 完成时点统一系统单调时钟，兼容 macOS Python 3.9；新增按实例取消和不可退款的取消快照。当前行为见 [取消协议](task-cancellation.md)，本页上述 5346 项记录保留为任务预算首次验收的历史结果。
