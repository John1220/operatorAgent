# 任务状态收口：journal 切流计划

状态：**权威层已完成并有守卫；写入方已接两个**（`prepare` 发 `task_prepared`，
`ToolCall` 发 `task_activated` + `job_*`）。本文只记「接下来怎么切」，不重复
设计理由——那些在 `operator_agent/core/task_journal.py` 的模块文档里。

## 进度（2026-11-24，接手先读这一节）

三提交计划里**前两个已合并**，第三个（迁移 0.8.2 完整优化流程）未开始。

| 提交 | 内容 |
|---|---|
| `48f3bd2` | 工具契约收齐：两处 DSH 验收输入补 `operator`、精度词表收敛成一份、`tool_version` 12→13 |
| `091f813` | journal 状态转换：`admit()` 让 `append` 与 `fold` 共用顺序判据（终态不可回退、Job 事件要求 active） |
| `8e8f9c0` | 旧任务只读恢复：`legacy_task_view()` 按各自身份读回契约 / 两本账 / 交付源码，A4 真实产物做回归 |
| 本轮的 `feat(journal)` | **落地顺序第 1 步**：`ToolCall` 接 journal（详见下面第 1 步的「已完成」） |

接下来两件事，按此顺序：

1. **接 journal 的写入方**（本文下面的五步）。**第 1 步已做完**；剩第 2–5 步。
2. **profile 搬完**：`ncu_profile` / `gpu_lease` / `profile_budget` 已搬；
   还差把租约接进 `HostTaskBudget.reserve`，判据是
   `tests/unit/profile/test_gpu_lease_across_tasks.py` 里那 4 条 skip
   **自动转绿**（它们按源码探针跳过，租约一接上就不跳了）。

### 已知噪声，别当回归

- 3 条需要 `ps` 或嵌套沙箱的用例在本机会红（`test_supervisor_kills_blocked_worker…`、
  `test_cancel_cleans_owned_sessions…`、`test_the_c_arm_runs_inside_a_repo_blind_sandbox`）。
- `test_remaining_deadline_interrupts_blocked_model_request` 在全量负载下偶发超时，单跑不复现。
- `tests/acceptance/phase1/test_rehearsal.py` 在**与别的套件同进程连跑**时偶发一次
  `result['phase'] == 'failed'`（八格 worker 里有格子超时）；单跑与第二次连跑都不复现。
  断它之前先单跑一遍。
- `tests/hygiene/.../test_no_credential_shaped_content` 是**真红**：`tests/unit/resume/
  test_legacy_resume.py:16` 里写死了本机 A4 产物的绝对路径。与本次切流无关。
- `tests/acceptance` 已改成按能力探针跳过；跑之前先 `python -m benchmarks.experiments.s5.s5_rebuild_deps`
  重建依赖树，否则 118 条会 skip。

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

> **已修（第 1 步）**：终态现在分两支。回执读不到 ⇒ `result=unknown` +
> `result_reason ∈ {receipt_absent, invocation_mismatch, invalid_timestamp}`，
> 结算用 `reject_completed=False`（`settlement_basis='worker_reaped'`，已完成记录不动）；
> 回执读到了、只是落在停止之后 ⇒ `result=received` + `outcome=cancelled|failed`，
> 这一支才用 `reject_completed=True`。`result=none` 只用于「根本没有 worker 起过」。

## 实测影响面

| 面 | 数量 | 位置 |
|---|---|---|
| Python 状态写入方 | **1 条路径**（原 4） | `autonomous`（含 `hitl_tools.run_autonomous` 的启动态）—— `tool_calls` 已切走 |
| Python 状态读取方 | 2 | `hitl_tools.autonomous_status`（跑旧目录时）、`task_cancellation.call_status`（兜底） |
| JS 状态读取方 | 1 | `host-model-budget.mjs` 的 `cleanupPending`（只在目录没有账本时读旧文件） |
| 直接构造 `ToolCall` 的测试 | 6 个文件 | 见下「逐文件」；本轮只改了 `test_owned_cleanup_defaults.py` |
| 引用这两个状态文件的测试/脚本 | 8 单测 + 2 JS + 5 benchmark | benchmark 那 5 处等第 3 步一起搬 |

## 落地顺序（每步独立可绿）

### 1. `ToolCall` 接 journal

`prepare` 发 `task_prepared`；第一个 Job 的发起方发 `task_activated`；`ToolCall` 只发
`job_*`。

> **不要**让 `ToolCall` 在 journal 缺失时自己补一条 `task_prepared`。那会为一个**没有
> 契约的目录**伪造「已准备」事件——那是这套东西里最贵的一种谎。

同一提交内必须删掉 Python 侧对 `run-status.json` 的写入，否则就是双写。

验收：新 journal 事件流与旧的 `run-status.json` 内容在语义上逐项对齐（相位、结果、
退出原因），并且 `grep -rn "run-status.json" operator_agent/` 只剩读者、没有写者。

#### 已完成（2026-11-24）

写者：

| 位置 | 事件 |
|---|---|
| `hitl_tools._record_prepared` | `task_prepared`（幂等；账本读不动时拒绝准备） |
| `tool_calls._JobJournal` | `task_activated`（相位是 prepared 时）+ `job_queued` / `job_started` / `job_finished` |

读者（同一个投影，不再解析写入方维护的文件）：
`task_journal.invocation_view()` → `hitl_tools.autonomous_status`（tool-call 目录）
与 `task_cancellation.call_status`（取消前的相位/owner 判据）。
`host-model-budget.mjs` 的 `cleanupPending` 也改读 `journal.jsonl`——**读者**不是第二写者，
但那一格决定「旧任务能不能放行」，留着读死文件会让任务永远关不上。

`job_started` **在起 worker 之前**写：起不来也是一次「开始了、没成」，而不是一条永远
没有下文的排队记录。

`job_finished` 带 `cleanup`，并且**在 worker 收完之后**才写：所以「终态存在」本身
就是「这一次调用已经收干净」。`worker_start_failed` 那一支记 `result=none` +
`cleanup.status='worker_not_started'`（没有 worker 要收，也不是「未知」）。

三处权威层的动作（趁还没有存量流）：`OPERATIONS` 加 `repair`（它同样是一次
有始有终的监督调用）；`RESULT_REASONS` 加 `stopped_before_receipt`（被外部停掉、
回执没等到——与 worker 自己没了要分开）；`task_id` 缺省时从事件流继承（身份只由
prepare 建立一次）。

**两处与原文不一致，都是有意的：**

1. **autonomous 那条路仍写 `run-status.json`。** 它不是同一件事：那个文件里有每秒跳
   一次的预算 ticker 与 `stage`，是**活进程的现场**，不是任务状态；把它改成 append-only
   事件流会把「现场」错当成「转移」。在它切过来之前，那条路仍然只有一个写者
   （`run-status.json` 自己），**没有双写**。全局 grep 的验收因此推到那一步；
   到时候 `autonomous_status` 的 lost 判据要一起搬，不能只换写方。
2. `task_closed` 仍没有写入方。它是「谁有权断言整段工作结束了」这个问题，不该由
   第 1 步顺手发明。

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