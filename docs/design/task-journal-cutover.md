# 任务状态收口：journal 切流计划

状态：**权威层已完成并有守卫；尚未接任何写入方。** 本文只记「接下来怎么切」，不重复
设计理由——那些在 `operator_agent/core/task_journal.py` 的模块文档里。

## 已经定下的决定

| 决定 | 取值 | 理由 |
|---|---|---|
| journal 粒度 | **每任务一条 `journal.jsonl`** | 一个任务没有并发写者；每 invocation 一条只会让 resume 变扫描 |
| 迁移方式 | **一次性迁移，不做双写** | 两个权威并存正是要消灭的形状 |
| 封账门 | 只有 `task_closed` | 单个 Job 的终态只约束它自己；一个 run 会做三次 verify |
| `kind` | 闭合词表 | 拼错的事件不许静默变成空操作 |
| `unknown` | `none / received / unknown` 三态 + `result_reason` | 诊断入口不能被下一条事件覆盖 |
| JS 侧 | **不加 npm 依赖**，经 Python 入口提交事件 | `rev` 分配与落盘必须只有一处 |
| 预算语义 | `result_unknown` 仍按 `failed` 结 | 不退款、不给 candidate credit；变化只在**记录诚实** |

## 已发现的语义缺陷（接写入方时要一并修）

`operator_agent/app/tool_calls.py:207`

```python
else:
    self.status.update(phase='failed', stopped_reason='worker_failed')
    self.host_budget.settle_owner(self.invocation_id,
        outcome='failed', reject_completed=True)
```

worker 退出了、但 `worker-result.json` 读不到或身份对不上时，这里记成**失败**，并按
「什么都没有完成」结算。可远端那次执行**很可能已经跑完**了。

落盘本身没有问题：`tool_calls.main` 在退出前已经原子写了带 `invocation_id` 的
`worker-result.json`，父进程也校验了身份与时间序。缺的是**分类**——这一支应当写成

```python
job_finished(outcome='failed', result='unknown', result_reason='receipt_absent')
```

并且**不再** `reject_completed`。账不变，记录变诚实。

## 实测影响面

| 面 | 数量 | 位置 |
|---|---|---|
| Python 状态写入方 | 4 | `tool_calls` / `autonomous` / `task_cancellation` / `hitl_tools` |
| JS 状态读取方 | 2 | `index.mjs` / `host-model-budget.mjs:89` |
| 直接构造 `ToolCall` 的测试 | 6 个文件 | 见下「逐文件」 |
| 引用这两个状态文件的测试/脚本 | 8 单测 + 2 JS + 5 benchmark | — |

## 落地顺序（每步独立可绿）

### 1. `ToolCall` 接 journal

`prepare` 发 `task_prepared`；第一个 Job 的发起方发 `task_activated`；`ToolCall` 只发
`job_*`。

> **不要**让 `ToolCall` 在 journal 缺失时自己补一条 `task_prepared`。那会为一个**没有
> 契约的目录**伪造「已准备」事件——那是这套东西里最贵的一种谎。

同一提交内必须删掉 Python 侧对 `run-status.json` 的写入，否则就是双写。

验收：新 journal 事件流与旧的 `run-status.json` 内容在语义上逐项对齐（相位、结果、
退出原因），并且 `grep -rn "run-status.json" operator_agent/` 只剩读者、没有写者。

### 2. Python 侧加 `journal_append` 入口，JS 改调它

JS 不再自己 append。理由：`rev` = 「读当前最大值 + 1」，两侧各读各的会撞号；而
`proper-lockfile` 用 `mkdir` 取目录锁，与 Python 的 `flock` **不是同一个互斥协议**，
同名的 `.lock` 提供不了共同互斥。

验收：`host-model-budget.mjs` 里不再出现 `run-status.json`；`host-model-ledger.mjs`
的 `host-task.json` 读写也走同一入口。

### 3. 读者改读投影

`status` / `best` / `resume` 一律 `fold(journal)`，不再拼多个文件。

验收：`grep -rn "run-status.json" operator_agent/ integrations/` 为空。

### 4. `host_task_budget` 的 `host-task.json` 汇入 journal

预算事件用已经预留在词表里的 `budget_reserved` / `budget_settled`（当初留它们就是为了
这一跳不用再 bump schema）。

验收：同一份预算在 journal 投影里与旧 `host-task.json` 的每个数字一致。

### 5. 迁移脚本 + 4 个 repro 包

`benchmarks/repro/` 下 4 个包是**已发布证据**。转换脚本必须能证明转换后重放结果不变：
包内的 `replay-*.json` 重跑仍逐字段相同，否则不许合并。

验收：4 个包依次 replay 全绿；脚本对「无法归属的旧流」显式拒绝而不是猜。

## 逐文件：直接构造 `ToolCall` 的测试

```
tests/unit/test_mcp_server.py
tests/unit/test_host_workflow.py
tests/unit/test_rpc_deadline.py
tests/unit/test_owned_cleanup_defaults.py
tests/unit/test_repair_task_budget.py
tests/unit/fake_backend/test_provenance.py
```

这几处会在第 1 步一起改。其中 `test_host_workflow.py` 是那个「一个 run 做三次 verify」
的现场，第 1 步做完它必须仍然全绿——它是这次拆两层的直接理由。

## 已知噪声（与本次迁移无关）

- `tests/unit/test_remaining_deadline_interrupts_blocked_model_request` 在全量负载下会偶发
  超时，单跑不复现。它是时序抖动，不是状态问题。
- 3 条需要 `ps` 或嵌套沙箱的用例在本机会红；`tests/acceptance` 已改成按**能力探针**跳过
  （见 `s5_runtime.sandbox_probe` / `_process_introspection_usable`）。