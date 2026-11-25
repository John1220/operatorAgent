# 任务状态收口：journal 切流计划

状态：**权威层已完成并有守卫；写入方已接两个**（`prepare` 发 `task_prepared`，
`ToolCall` 发 `task_activated` + `job_*`）。本文只记「接下来怎么切」，不重复
设计理由——那些在 `operator_agent/core/task_journal.py` 的模块文档里。

## 进度（2026-11-24，接手先读这一节）

三提交计划里**前两个已合并**；第三个（迁移 0.8.2 完整优化流程）未开始。
**journal 切流的五步已全部落地**（第 5 步的真机 replay 对账待 H20）。

| 提交 | 内容 |
|---|---|
| `48f3bd2` | 工具契约收齐：两处 DSH 验收输入补 `operator`、精度词表收敛成一份、`tool_version` 12→13 |
| `091f813` | journal 状态转换：`admit()` 让 `append` 与 `fold` 共用顺序判据（终态不可回退、Job 事件要求 active） |
| `8e8f9c0` | 旧任务只读恢复：`legacy_task_view()` 按各自身份读回契约 / 两本账 / 交付源码，A4 真实产物做回归 |
| `cefe7f9` | **落地顺序第 1 步**：`ToolCall` 接 journal（详见下面第 1 步的「已完成」） |
| `d00193c` | 文档：把上面的行指向它 |
| `7d0ff8d` | 工具契约的**插件侧**补齐：`EXPECTED_TOOL_VERSION` / `PAIRED_PYTHON` 改从构建期清单取（此前手抄 `12`，真机上 doctor 反着误报） |
| `243f5de` | journal 的**重试**行为钉住（被拒不留痕、不占号、终态不被同事件覆盖、并发重试只进一条） |
| `124c058` | **落地顺序第 3 步**：读者一律读投影；autonomous 那条路一起切（最后一个写者消失） |
| `6e2aae4` | **落地顺序第 2 步**：`journal-append` 入口；宿主提交 `task_closed` 与 `budget_*` 快照 |
| `8d91ab4` | **落地顺序第 4 步**：预算状态搬进账本（`budget_opened` / `budget_reserved` / `budget_settled`） |
| `b490d2a` | **落地顺序第 5 步**：`journal-migrate` 迁移脚本 + 回执对账工具 |

状态对照（三条计划的收口情况）：

1. **工具契约收齐** —— **已完成**（`48f3bd2` + `7d0ff8d`）。注意后者是补齐：
   前者把 Python 侧与清单提到 13，插件那份手抄的 12 落下了，真机 `/op doctor`
   会把正确的工具面判成「13 ≠ 期望 12」。
2. **恢复语义** —— A4 真实产物的回归件在 `tests/unit/resume/`（7 条，本机跑真产物，
   缺产物时 skip）；journal 的乱序与重试用例齐了。**状态读写方两条路都切完了**
   （tool-call 见 `cefe7f9`；autonomous 见下面第 3 步的「已完成」）。
3. **迁移 0.8.2 完整优化流程** —— **NCU profile 执行链已接入并完成真机验收**。
   `HostTaskBudget.reserve` 在预留时取每 run + 机器级 GPU 租约，`finish` / `settle_owner`
   在结算后按 owner 释放；跨任务同卡、不同卡、profile/L3 互斥与未声明卡冲突的 4 条用例
   已直接运行并通过。远端 runner 的 `action=profile`、SSHExecutor 的 NCU 采集/清理、
   `hitl_tools.profile`、`ToolCall` 监督和 host 工具清单（`tool_version=17`）均已接线。
   `final_selection` 也已迁移：L3 性能事实、噪声带、可比窗口与最快合格候选由预算账本
   投影；final 默认送评推荐源码，只有显式 `override_reason` 才覆盖。选择回执与预留
   原子写入同一条 `budget_reserved` 快照，`status`/`resume` 只读该投影，不再写
   `best.json` 或 `final-selection.json`。宿主 L3 现在验证全部声明形状；SSH 调用参数
   参与机器级租约；`seed_source`/`preflight_receipt`、baseline 门与 optimize/final
   恢复状态机也已接入。搬运源在 `codex/v0.8.2-profile-optimize`。

   **2026-11-24 H20 验收**：在 GPU3 上对 `relu fp16 128x128` 的 round 2（完整 L2
   12/12）实跑 NCU 2025.3。回执只含 1 个 `_relu_kernel`，位于 `candidate` NVTX 范围，
   `decoy_seen=false`，raw 与 re-export 逐字段一致；receipt/complete 及其余产物的
   SHA256 均独立复核相同。随后用 8 秒绝对 deadline 实跑超时探针：返回
   `profile_timeout`，`cleanup=attempted`、清理约 1.56 秒；两个 NCU 进程组均已确认
   消失，失败仍占满第 2/2 个 profile 名额，run 与机器级 GPU 租约均释放。
   真机暴露的两处接线问题也已修：能力判断走 `Cap.NCU_PROFILE`，NCU 使用每次采集
   独立的 `TMPDIR`，避免共享 `/tmp` 的权限污染。小核 L3 读数仍受共享卡与 launch
   地板影响，**不作为性能结论**。

接下来一件事：

**v0.8.2 执行链迁移已完成**：`operator_profile` 真机验收、`final_selection`、全形状 L3、
调用感知 GPU 租约、seed/preflight 与跨轮 NCU 实验记录均已落地；主机通配租约会在
取得实测 UUID 后原子升级。journal 的真机 replay 对账已在本文第 5 步完成。当前剩余
边界是真实模型 `forward` 验收与正式 A/B 成本对照。

### 已知噪声，别当回归

- 3 条需要 `ps` 或嵌套沙箱的用例在本机会红（`test_supervisor_kills_blocked_worker…`、
  `test_cancel_cleans_owned_sessions…`、`test_the_c_arm_runs_inside_a_repo_blind_sandbox`）。
- `test_remaining_deadline_interrupts_blocked_model_request` 在全量负载下偶发超时，单跑不复现。
- `tests/acceptance/phase1/test_rehearsal.py` 在**与别的套件同进程连跑**时偶发一次
  `result['phase'] == 'failed'`（八格 worker 里有格子超时）；单跑与第二次连跑都不复现。
  断它之前先单跑一遍。
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
| Python 状态写入方 | **0** | 两条路（tool-call / autonomous）都写 `journal.jsonl` |
| Python 状态读取方 | 0 | `autonomous_status` / `call_status` / `resume` 一律读投影 |
| JS 状态读取方 | 0 | `cleanupPending` 读账本；没有记录 = 未结算 |
| 直接构造 `ToolCall` 的测试 | 6 个文件 | 见下「逐文件」 |
| 引用旧状态文件的测试/脚本 | 8 单测 + 2 JS + 5 benchmark | 单测与 JS 已改完；benchmark 4 处改读账本，剩 `s4_case_bundle` 一处读**已发布**的旧证据（第 5 步的迁移脚本负责） |

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

#### 已完成（2026-11-24）

入口是 CLI 子命令 `operator-agent journal-append --run-dir <dir>`，事件走 stdin，
回一条**已写入**的事件（含分配到的 `rev`）；拒绝回 `ok:false` + 稳定 `code`
（退出码 2）。JS 侧 `integrations/dsh-operator/journal.mjs` 只做提交，不分配、不落盘。

宿主（DSH 插件）现在提交两类事件，都**只在目录确实有账本时**提交：

| 时机 | 事件 | 说明 |
|---|---|---|
| 相位最终成为 `discussion`（封账） | `task_closed` | 一次任务只提交一次；终态与停因从 `host-task.json` 与账本读出，不猜 |
| 每次模型派发 / 结算 | `budget_reserved` / `budget_settled` | 载荷是 `{model: <账本快照>}`，与预算事件给 `fold` 的合并语义一致（快照，不是增量）|

封账落在**一个**位置：「相位最终成了 discussion」那一刻。原因是关任务的判定有两个
来源（预算分支与 `turn/end` 的生命周期规则），就地各写一份会出现「同一次关闭提交两条」
或者「有一条路忘了提交」。只记 `taskId` 判重，两个来源都走它。

失败语义：CLI 不在（`journal_unavailable`）或账本拒绝（`journal_closed` 等）都只是
回执 —— 模型预算门不该因为一条事件写不进去就崩；但**只有** `journal_closed`
（已经封过账）当作正常，其他错误一律进 `ctx.logger.error`。

JS 不再自己 append。理由：`rev` = 「读当前最大值 + 1」，两侧各读各的会撞号；而
`proper-lockfile` 用 `mkdir` 取目录锁，与 Python 的 `flock` **不是同一个互斥协议**，
同名的 `.lock` 提供不了共同互斥。

验收：`host-model-budget.mjs` 里不再出现 `run-status.json`；`host-model-ledger.mjs`
的 `host-task.json` 读写也走同一入口。

### 3. 读者改读投影

`status` / `best` / `resume` 一律 `fold(journal)`，不再拼多个文件。

验收：`grep -rn "run-status.json" operator_agent/ integrations/` 为空。

#### 已完成（2026-11-24）

验收达成：`grep -rn "run-status" operator_agent/ integrations/` **无命中**（两条路都算）。
剩下的两处命中都是「断言这个文件**不存在**」的用例，不是读它。

两层的读者各有入口，都只是 `fold` 的投影：

| 入口 | 回答 | 谁在用 |
|---|---|---|
| `task_journal.invocation_view(dir)` | 一次调用（Job 层） | `autonomous_status`（tool-call 目录）、`call_status`、4 个 benchmark |
| `task_journal.task_view(dir)` | 整段运行（任务层） | `autonomous_status`（运行根）、`call_status` |

**autonomous 那条路也切了**（这一步绕不开它：它是「仓库里不再有人写这个文件」的最后
一道门）。换的时候有三件事值得记：

1. **每秒一次的 ticker 没有了**。那个文件里跳的是预算的**现值**，不是转移；把现值
   写进 append-only 流等于把「现场」当成「状态」，几十秒就是几百条同义事件。现在
   读者从预算账本现算：事件里带 `started_monotonic`（监督者那侧）与 `started_at`
   （epoch），`now = started_monotonic + (epoch 至今)` —— 两个时钟各自都不可跨进程
   比较，这个组合可以。
2. **停因/错误/清理进 `task_closed`**，读的时候原样取出。所以「这次运行怎么结束的」
   只有一个来源，`conductor.json` 回到它本来的身份：**产物摘要**，不是状态。
3. **旧目录不再被解释**。没有账本但有 `autonomous.json` / `conductor.json` /
   `worker-result.json` 的目录，回答 `legacy_run_directory`（只读）：它能读回的是
   契约、两本账、交付源码（`operator_resume`），不是「它当时在不在跑」。
   `prepare` 失败时同样没有账本 —— 没有任务就没有可记的转移，不伪造一条
   「已准备然后失败」。

### 4. `host_task_budget` 的 `host-task.json` 汇入 journal

预算事件用已经预留在词表里的 `budget_reserved` / `budget_settled`（当初留它们就是为了
这一跳不用再 bump schema）。

验收：同一份预算在 journal 投影里与旧 `host-task.json` 的每个数字一致。

#### 已完成（2026-11-24）

预算状态**搬到账本里**，`host-task.json` 不再写（只剩「认旧目录」这一个用处）。
每条事件带的是**整份快照**（`budget={'host_task': state}`），与 `fold` 对 `budget`
的合并语义一致：读的人拿最后一条就是现值，不存在「增量要不要重放」的问题。

四种转移各有自己的事件：

| 时机 | 事件 |
|---|---|
| 开账（`initialize_host_task`） | `budget_opened` |
| 预占一个验证名额（`reserve`） | `budget_reserved` |
| 结算（`finish` / `settle_owner`） | `budget_settled` |

**一处与原文不一致**：新增了 `budget_opened`。把开账写成一次 `budget_reserved`
会让「有人预占了一个名额」凭空出现一次，而开账时一个名额都没占；词表是闭合的，
多一个词比多一个谎言便宜。schema 仍是 `operator-task-journal-v2`。

三件顺手必须一起做的事：

1. **`snapshot()` 不再落盘**。以前每次查询都把 `max_elapsed_s` 写一次文件；账本只追加，
   「查一下」不是一次转移。反悔不了的保证不靠它：`_elapsed` 取
   `max(已落账的 max_elapsed_s, 单调钟走过的, 墙钟走过的)`，而落账发生在**每一次真正的
   转移**上。`max_elapsed_s` 仍是高水位，`reserve`/`finish`/`settle` 都会把它推上去。
2. **JS 侧改读同一条事件**（`journal.mjs` 的 `readJournal` / `readHostTask`）：
   `task_id` / `policy_hash` / `started_*` / `max_elapsed_s` / `records` 全在
   `budget_opened` 之后的最新快照里。旧目录（没有账本）**失败即关闭**：
   `host_model_journal`，不重建额度、不解释旧文件。
3. **软链账本一律拒绝**（Python 与 JS 各一处）：账本被换掉意味着「写下去的额度落在别处」。

验收口径落成了测试：`test_host_task_budget.py` 的 52 条全绿（含「手写一条快照不进额度」——
账本的连续性是那条链），`test_host_resume.py` 里「resume 只读」改成对 `journal.jsonl`
字节不变，插件侧 168 条全绿（其中 `test_real_javascript_ledger_roundtrips_to_python_receipt`
仍是跨语言往返：JS 写、Python 校验回执）。

### 5. 迁移脚本 + 4 个 repro 包

`benchmarks/repro/` 下 4 个包是**已发布证据**。转换脚本必须能证明转换后重放结果不变：
包内的 `replay-*.json` 重跑仍逐字段相同，否则不许合并。

验收：4 个包依次 replay 全绿；脚本对「无法归属的旧流」显式拒绝而不是猜。

#### 本地已完成（2026-11-24）；replay 那一步要 H20

先把一件事说清：**这四个包里没有旧相位文件**。它们进的是 `hitl_tools.prepare/verify`
的进程内路径，从来只写过 `worker-result.json` 与产物目录；`failure_replay.py` 里
一个 `status` 都不提。所以对它们来说，第 5 步不是「转换」，而是**证明没变**：

```
python -m benchmarks.repro.compare_replays --fresh logs/replays.XXXXXX
```

逐字段对账（只放过 `evidence` / `run_dir` / `final_evidence` 三个「这一次落在哪」的
字段）。这个工具自己也有守卫：改判据要红、少一份回执要红、只改路径不算漂移 ——
否则它会替「重放不成立」签下「成立」。

真正需要转换的是**旧证据目录**，脚本是新的 CLI 子命令：

```
operator-agent journal-migrate --run-dir <旧任务目录> [--dry-run]
```

四条纪律，全部落成了测试（`tests/unit/job/test_legacy_migration.py`）：

* **能归属才搬**：`declaration.json` + `contract.json` 缺一个就拒绝（`not_migratable`），
  不从孤零零的相位文件反推任务身份；
* **不覆盖已有流**（`already_migrated`），也不改旧文件（它们原地不动，是那一年运行的记录）；
* **未知相位拒绝**（`unknown_phase` / `unattributable_status`），不猜；
* **旧文件里没有的东西不补**：`result` 这一格旧相位里根本不存在 —— 能确定的写
  `received`，不能确定的写 `unknown` + `receipt_absent` / `stopped_before_receipt`。
  旧相位里的 `lost` 只搬成「激活过」，读者用同一条判据重新得出 lost。

`benchmarks/experiments/s4/s4_case_bundle.py` 的那个读点改成读投影：没迁移的目录
直接报错并给出上面那条命令，不回去读那个已经没人写的文件。

**真机 replay 与对账：2026-11-24 在 H20 上跑通。** `sh benchmarks/repro/run_all_replays.sh`
八个回放 8/8 绿：四个「修前」都按冻结预期复现（契约哈希 + 源码身份 + 级别 + 具名用例及误差），
四个补丁都过完整 L2（32/32、32/32、15/15、8/8）。随后
`python -m benchmarks.repro.compare_replays --fresh logs/replays.ekmrZv` → `ok: true`，
`problems: []`：重跑的回执与仓里那份逐字段相同。

跑之前遇到一件必须先修的事：**四份 `case.json` 的 `declaration` 都冻结在 `operator`
变成必填之前**，所以 `prepare` 当场拒绝（`explicit_declaration_required`）。补上的
`operator` 是各包**当年推断出的同一个**算子类型（`fused_layernorm_residual` / `matmul`
/ `attention`），并且用重算契约哈希证明被评判的东西没变：四份重算值分别等于
`case.json` 里记录的 `093339587d8c…` / `093339587d8c…` / `6e337b0f1be6…` / `e9f75d60cf1c…`。
每份 `case.json` 里加了 `declaration_amendment`，把「补了什么、为什么、契约哈希未变」
写在证据旁边 —— 改的是写法，不是判据。

顺带一个观测（**未改**）：`matmul_strided_20261107/case.json` 的 `correctness_profile`
写着 `builtin-layernorm-fp16-cuda-v2`，而它冻结契约重算出来是 `builtin-l2-v1`。
回放不比较这个字段（判据是契约），所以它不影响结论，但那个值本身是错的。

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