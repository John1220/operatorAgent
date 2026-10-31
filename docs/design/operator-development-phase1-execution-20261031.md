# 第一期执行清单：交付身份收口 + 阶段 A 需求与基线

| 项目 | 说明 |
|---|---|
| 文档性质 | 一期可勾选执行清单；不含已完成事项，不代替验收报告 |
| 版本 | v0.2 |
| 创建 / 最近更新 | 2026-10-31 / 2026-10-31（`HEAD = f3c9c03`，分包重构之后） |
| 上游 | [产品路线 §5 阶段 A](operator-development-product-roadmap.md#L155-L158)、[§6 评价问题与口径](operator-development-product-roadmap.md#L186-L215) |
| 事实来源 | [发布形态](../status/dsh-operator-release-readiness-20261030.md)、[GPU 中断恢复](../status/gpu-interrupt-recovery-20261030.md)、[S5 结项](../status/dsh-operator-s5-closeout-20261030.md)、[仓库瘦身与分包](repo-slimming-20261031.md)、[可迁移性](dsh-operator-portability-4090.md) |
| 本期资源前提 | 零 GPU 机时、零模型调用；H20 与 P800 已下线，读数不可再生 |

本期只做两件事：**把已验证的能力交付身份钉死**，**建立阶段 A 的任务与基线**。不新增能力。三项工作按 W1 → W2 → W3 顺序推进，W3 依赖 W2 的场景选择结论。

v0.1 写成之后，工作区被四个提交扫过（证据出仓 + 布局守卫、`tests/` 分目录、`benchmarks/` 分目录、`operator_agent/` 分五层）。**本期目标不变，W1 的形状变了**：原来缺的是"提交在盘改动"，现在缺的是"**已交付工件与仓库 HEAD 不再是同一身份**"。

## 0. 开工前先读的三条边界

- 路线的阶段 A 是 B 的门槛，不是可跳过的文书工作。本期结束时必须能回答"谁在哪个环节花时间、怎么证明改善"。
- 本期任何"通过"都不得写成插件增益。[S5 已按证据不足收口](../status/dsh-operator-s5-closeout-20261030.md)，插件整体优于原生 DSH 这条主张永久不可补，本期也不试图补。
- 需要真机读数才能定论的判断，本期只交量具与判据清单，判读留给人；不搭对照实验。

## 1. 开工核对（本文档写完即跑过，读数条件一并记录）

**读数取自 2026-10-31 本机、`HEAD = f3c9c03`、工作区干净。** v0.1 那节的 `HEAD = de1918f` 读数已失效，按下面五条替换。

| # | 事实 | 本次实算证据 |
|---|---|---|
| F1 | 默认清理修复**已入库**，v0.1 里"在盘未提交"这条缺口不成立了 | `git log -S'owned_worker_scope'` 指向 `ef6acea`（`tests/` 分目录那个提交）；`6cb1fca` 上 `backend_ownership.py` 内该符号出现 **0** 次。即：修复与一次 578 文件重构挤在同一提交里，交付动作本身没留独立记录 |
| F2 | **在盘 0.6.1 wheel 与 HEAD 不是同一身份**，且版本号没变 | ① 布局：wheel 内是包根平铺（`operator_agent/tool_calls.py`），HEAD 包根只剩 `__init__.py`+`__main__.py`，模块进五层（`ls` 计 24/31/16/22/10 含各层 `__init__`）② 入口：wheel `entry_points.txt` = `operator_agent.cli:main` / `.mcp_server:main`，HEAD [pyproject.toml:21](../../pyproject.toml#L21-L22) = `operator_agent.app.cli:main` / `app.mcp_server:main` ③ 两处都自称 `0.6.1`（[pyproject.toml:6](../../pyproject.toml#L6)、[operator_agent/__init__.py:20](../../operator_agent/__init__.py#L20)） |
| F3 | **协议身份断链**，比 F2 更硬 | `protocol_identity()` 的清单仍是 73 条、文件名集合完全不变，但 **70 条路径键**从平铺变成带层（`operator_agent/conductor.py` → `operator_agent/app/conductor.py`）。比对方式：解包 wheel 读 `run_contract.py` 的 `files` 元组与 [core/run_contract.py:71](../../operator_agent/core/run_contract.py#L71) 逐条对照，不依赖运行期（本机默认解释器无 numpy，`import operator_agent` 起不来）。⇒ HEAD 算出的身份与在盘 wheel、与引用「协议身份 73 文件」的[安装工件验收](../status/dsh-operator-s5-installed-gpu-acceptance-20261030.md#L7)、[S6 准备记录的「70 文件」](../status/dsh-operator-s6-preparation-20261027.md#L11) 都不同源；[分包记录自己写明](repo-slimming-20261031.md#L115-L118)「引用旧身份的冻结记录不能与新身份互换」 |
| F4 | 全量回归读数现在有**四个口径**，最新那份在一页内自相矛盾 | 6172 passed / 1 skipped（[中断恢复](../status/gpu-interrupt-recovery-20261030.md#L47)）；6159 passed + 15 failed + 4 errors（[发布形态](../status/dsh-operator-release-readiness-20261030.md#L53)，沙箱内）；6185 passed / 1 skipped（[瘦身记录第 5 行](repo-slimming-20261031.md#L5)）；6183 passed / 1 skipped（[同一文件 §4](repo-slimming-20261031.md#L95)）。**不得预先命名成因**；先把每个数钉到提交号、解释器与目录集合 |
| F5 | 工作树布局多了守卫，**在仓内落产物会先把守卫跑红** | `output/` 已是指向仓外证据树的符号链接，位置与单副本警告写在 [evidence-location.json](../research/evidence-location.json)；[tests/hygiene/test_repo_layout.py](../../tests/hygiene/test_repo_layout.py) 管根目录白名单、单层条目预算、禁缓存/构建产物/证据入库、`output/` 只能是符号链接。v0.1 里"日志落 `output/plugin_completion/`"仍可用（符号链接照旧解析，已实测可读），但构建目录必须落在仓外 |

工件哈希（本次实算，与发布形态记录一致）：

```
097918d395d08bc2859b07edd3db81cd543a3c10b9b588856fd3ea8100df6de4  operator_agent-0.6.1-py3-none-any.whl
8245f1689c9c5b8145986248afd71631c1d6a50a958e194be7a4d222e65bab0d  dsh-operator-agent-0.1.10.tgz
```

- [ ] 复核上述两行哈希，确认文档与在盘工件一致
- [ ] 把 F4 的四个读数钉到「哪个提交、哪个解释器、哪个目录集合」，结论写回本节，不改进判定
- [ ] 记下 F3 是**结构性断链**：同名文件集合不变、70 条路径键变，因此不能靠"内容没改"解释掉

## 2. W1 交付身份收口

目标一句话：**任何人从仓库检出即可构建出与安装说明同身份的 wheel 与插件包，且没有任何一份冻结记录被当成新身份引用。** 不需要 GPU，不需要模型。

v0.1 的 T1.1（提交在盘改动）已被重构顺带完成，但换来的是 F2/F3：**已交付工件与 HEAD 成了两个身份、共用一个版本号**。所以本期 W1 的第一动作从"提交"变成"重绑身份"。

- [ ] **T1.1 写一份身份重绑记录（本期第一个产出）。** 逐条列旧身份 → 新身份：协议身份（73 条清单、文件名集合不变、70 条路径键带层）、console script 入口（`operator_agent.cli` → `operator_agent.app.cli`）、包布局（平铺 → 五层）。点名从此**只能读、不能引**的冻结记录：[S5 安装工件验收](../status/dsh-operator-s5-installed-gpu-acceptance-20261030.md#L7)（协议身份 73 文件）、[S6 准备](../status/dsh-operator-s6-preparation-20261027.md#L11)（70 文件）、[S6 验收](../status/dsh-operator-s6-acceptance-20261030.md#L7)、[中断恢复](../status/gpu-interrupt-recovery-20261030.md)、[发布形态](../status/dsh-operator-release-readiness-20261030.md)。不改判据、不重跑 GPU、不刷新任何哈希。
- [ ] **T1.2 版本号强制递增。** `0.6.1` 现在指两种结构（F2），必须升到 `0.6.2`，同步 [pyproject.toml:6](../../pyproject.toml#L6) 与 [operator_agent/__init__.py:20](../../operator_agent/__init__.py#L20)，插件包同步 `0.1.11`；在盘 `0.6.1`/`0.1.10` 工件与其 SHA256 原样保留，不覆盖、不复用其验收。v0.1 里这条是"若判定需要"，现在升级为硬条件。
- [ ] **T1.3 仓外干净检出重建。** 构建目录落在**仓外**（F5：[布局守卫](../../tests/hygiene/test_repo_layout.py) 禁构建产物入库），`rsync` 导出排除 `output/`、`logs/`、`triton/`、`.git/`。新 wheel 三条断言：模块集合是五层、`entry_points.txt` 指向 `operator_agent.app.*`、许可与 11 prompts / 2 skills 齐备。`tests/hygiene/test_packaging.py` 与布局守卫同时绿才算过。
- [ ] **T1.4 反向对照。** 在 `6cb1fca`（本次实算：该提交 `backend_ownership.py` 内 `owned_worker_scope` 出现 0 次）跑 [tests/unit/test_owned_cleanup_defaults.py](../../tests/unit/test_owned_cleanup_defaults.py)，必须以缺失/失败告终。这一条不过，T1.3 的绿灯不算数——说明这组测试锁不住行为，只证"这条分支没人走"。
- [ ] **T1.5 沙箱外全量 pytest 一次留档。** 缺口见[发布形态 §3](../status/dsh-operator-release-readiness-20261030.md#L57-L59)：`ps`／`sandbox-exec`／真 DSH 的 19 项集成用例仍只有历史轮次读数。本次要顺带把 F4 的四个口径钉到提交号与解释器上。日志经 `output/` 符号链接落仓外证据树，逐文件哈希入清单。**已知顺序相关偶发**（[分包记录 §未处理项](repo-slimming-20261031.md#L140-L142) 的 `test_hitl_tools::…gets_two_consequences`）按偶发单列，不并进"全绿"。
- [ ] **T1.6 CI 首跑。** 工作流从未在远端真实执行过；首跑红/绿都要记录，失败不靠改判据变绿。注意 CI 跑的是分层后的新布局，与 `6cb1fca` 那次定义已有差异。
- [ ] **T1.7 安装说明与身份绑定。** [插件 README](../../integrations/dsh-operator/README.md) 的安装段落按新工件版本与哈希写；命令名（`operator-agent` / `operator-agent-mcp`）不变但背后模块路径变了，段落里要写明这点，避免按旧模块路径排查。

本期**不做** `npm publish` / `twine upload`：会产生不可撤回的公开版本号，属发布决定，不在一期范围。

## 3. W2 阶段 A：任务复盘、场景、基线

目标是产出**下一轮能直接被引用**的三张表，不是访谈记录本身。样本用于发现问题和选试点，不推断市场比例。

- [ ] **T2.1 选 3–5 个真实任务。** 优先"最近完成或卡住"的算子优化任务，每例需有可描述的输入范围、精度要求、目标硬件、参考实现与预算。允许用当事人自己的既有任务、代码与实验记录还原流程，不必新写题目。
- [ ] **T2.2 每例固定字段。** 任务与验收标准 / 当时用的工具与助手 / 逐环节人工投入与实际等待 / 反复失败的原因 / 最终是否接入 / 哪些环节愿意委托。缺一格就写"未知"，不用估算填。
- [ ] **T2.3 外部吸纳补齐基线。** 无当事人可用时，用公开一线流程做**对照基线**并明确标注"非本项目用户访谈"——[路线 §2.3](operator-development-product-roadmap.md#L60-L68) 的三条来源是唯一起点；新增来源要记核查日期，页面可能下线。
- [ ] **T2.4 输出场景表并排序。** 按"任务频率 × 问题成本"排序，选 **1 个**算子族作为阶段 B 的首条完整工作流。选择理由写成一句可反驳的话，而不是清单堆叠。
- [ ] **T2.5 冻结评价问题与门槛（跑之前）。** 从[§6.1 两个问题](operator-development-product-roadmap.md#L186-L191)里选**一个**作主问题，写清验收门槛、重复方式、停止规则、分母与"失败/未知/未执行"如何单列。这一步做完才允许碰任何 GPU。
- [ ] **T2.6 口径落表。** 把[§6.3](operator-development-product-roadmap.md#L202-L215) 八项指标换成具体单位与取数命令：模型调用预留、验证 case 次数、GPU 时间、账单金额**不得混成同一单位**；人工投入记活跃时间，不用模型墙钟代替。
- [ ] **T2.7 记下无 GPU 就能算的存量问题。** 现有证据里还有哪些能再榨一层（如 Golden 首测 3 PASS/1 FAIL 与诊断复核 4 PASS 的 CPU 计时长尾、`docs/research/` 系列未复核的假设），列成待办但不占用本期工时。

## 4. W3 冻结"开机即跑"的阶段 B 计划

避免下一次拿到卡时现编实验——那正是 S5 24 格只跑 2 格的成因。

- [ ] **T3.1 选卡。** 按[可迁移性 §6 最小动作](dsh-operator-portability-4090.md) 与 G1–G5 缺口评估目标设备；优先级取"一次非 CUDA 真迁移 > 再加一张 N 卡"，红线是走 triton-ascend 路线、不学厂商私有 C 接口。
- [ ] **T3.2 迁移门先行。** 未知卡 launch 地板、共享内存解析、roofline 分母口径、噪声带标定这四项在没有真卡时**只能写探针、不能声称已验证**；探针本身作为 W3 产出交付。
- [ ] **T3.3 控制变量清单。** 模型、硬件、任务输入、起始代码、参考实现、预算上限逐项固定，必要差异写明；开发基线用目标用户实际在用的 Agent 与常用脚本，保留其正常验证与执行能力。
- [ ] **T3.4 记录绕过与额外成本。** 插件发现失败、工具调用未成功、用户绕过工具、额外配置与纠错耗时都要有栏位；只统计成功样本的表不予接受。
- [ ] **T3.5 产出 runbook。** 一条命令序列 + 预期读数 + 停止条件，落在拿到卡的那天可直接执行；GPU 型号与 UUID 运行时探测，不写死常量。注意中断恢复驱动已随分目录搬到 [benchmarks/experiments/misc/gpu_interrupt_recovery.py](../../benchmarks/experiments/misc/gpu_interrupt_recovery.py)，runbook 里别再写平铺老路径。
- [ ] **T3.6 复用既有 workload 前先重绑 evaluator 身份。** [Phase 3 记录](repo-slimming-20261031.md#L47-L50) 与 [Phase 2 记录](repo-slimming-20261031.md#L115-L118) 都写明：`workload_decision_h20.evaluator_sources()`、`workload_llm_search_h20.sources()`、`quality_evaluation.evaluator_identity()`、`s5_gateway_real_preflight._identities()` 拿文件名当哈希键，分目录后键值变了。⇒ 新批次不得与引用旧身份的 H20 冻结记录混在同一张表里报数，也不得声称"接着上轮跑"。

## 5. 本期出口判据

| 判据 | 通过条件 |
|---|---|
| 交付身份 | T1.1 重绑记录入库，且 T1.3 与 T1.4 同时成立——仓外干净检出可复现工件，去掉修复时覆盖确实变红 |
| 版本一致 | T1.2 完成后，`版本号 → 内容` 重新一一对应；旧工件哈希未被覆盖（这正是发布形态 §1 第 3 项修过、又被重构冲掉的缺陷） |
| 验证留档 | T1.5 与 T1.6 各有一份日志与哈希，红/绿与偶发原样记录 |
| 阶段 A | T2.4 选出 1 个场景、T2.5 预登记完成，能回答"谁在哪一步花时间" |
| B 就绪 | T3.5 runbook 可在无卡条件下被第三方逐步复核，且 T3.6 的身份绑定已写明 |

五条缺一不进入阶段 B。

## 6. 本期明确不做，以及各自的解除条件

| 不做 | 解除条件（来自[路线 §5.2](operator-development-product-roadmap.md#L173-L182)） |
|---|---|
| 性能剖析工具（`profiler` 当前为 `None`，见 [generate/agent_runtime.py:154](../../operator_agent/generate/agent_runtime.py#L154)） | 阶段 A 显示瓶颈分析是主要耗时环节，且有可用卡验证增量 |
| 形状扫描 `shape_sweep`（同上 `None`；L3 只测 `test_inputs[0]`） | 真实任务出现"单形状达标但输入分布回退"的实际案例 |
| 多候选比较升为工具（现仅 `operator_bench` 探索级、不可裁决；配对逻辑在 `benchmarks/` 驱动里） | 新增正确或达标结果足以覆盖额外调用成本，且同预算对照可跑 |
| 终稿/验收报告生成（现只有逐轮 `report.md`） | 阶段 B 出现"结果正确但用户无法据此验收"的实际阻塞 |
| 框架/模型端到端接入 | 有明确接入任务与可维护测试基线 |
| 第二宿主、独立应用 | 首入口收益成立且用户确实集中在其他环境 |
| **回退分层布局去复原旧协议身份** | 不回退。分层与[层依赖棘轮守卫](../../tests/hygiene/test_repo_layout.py)是本轮定下的方向；旧身份只能被记录为断链，不能靠回滚布局冒充同源 |

## 7. 已知不可解，别当本期缺陷

- 插件整体增益永久不可补：H20 与 P800 已下线，同条件对照的原始读数不可再生。
- 私有原始证据为单副本（2026-10-30 已接受此口径），本期不再重提备份方案；证据树现在在仓外，仓内 `output/` 只是符号链接，[位置与重建说明](../research/evidence-location.json) 必须在守卫里存在。
- **协议身份的旧值不可复现**：`protocol_identity()` 现在按带层路径取键，要得到与 wheel/冻结记录同源的哈希只能回到平铺布局。本期把它当既成事实记录，不试图对齐。
- **历史文档里的旧路径不改**：[Phase 3 记录末尾](repo-slimming-20261031.md#L57-L59) 已定口径——带日期的 `docs/status/**` 与 `docs/research/**` 是当时快照，里面的 `tests/test_x.py`、`benchmarks/x.py` 保留原样。本清单引用它们时同样不改写；只有当前入口类文档随迁移更新。
- `exec_guard` 不是沙箱；MCP 工具不经 DSH 审批层；网络唯一出口与敌意候选隔离仍属后移项。
- 中断恢复验收只覆盖受信任固定候选、常驻 SSH runner 与显式新任务，不证明任意失控 kernel 的 watchdog、服务商停止或全部后代退出。

## 修订记录

| 日期 | 变更 | 依据 |
|---|---|---|
| 2026-10-31 | 创建 v0.1：把路线阶段 A 改写为可勾选一期清单，含 W1 交付身份收口、W3 冻结 B 计划 | 本次实算的工件与 `HEAD`/工作区哈希、发布形态与中断恢复记录、可迁移性评估 |
| 2026-10-31 | v0.2：适配四层重构（`HEAD = f3c9c03`）。§1 读数整段替换为 F1–F5；W1 第一动作由"提交在盘改动"改为"身份重绑"，版本号递增从条件项变硬条件，构建改到仓外，反向对照基线钉到 `6cb1fca`；W3 新增 T3.6 evaluator 身份重绑；出口判据四条扩为五条；§6 加"不回退布局"，§7 加旧身份不可复现与历史文档不改写 | 本次实算：wheel 与 HEAD 的 `entry_points`/模块集合/路径键差异（73 条清单里 70 条变）、`git log -S` 定位 `ef6acea`、`6cb1fca` 上符号 0 次、四个全量读数、布局守卫与符号链接 |
