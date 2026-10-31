# 第一期执行清单：交付身份收口 + 阶段 A 需求与基线

| 项目 | 说明 |
|---|---|
| 文档性质 | 一期可勾选执行清单；不含已完成事项，不代替验收报告 |
| 版本 | v0.2 |
| 创建 / 最近更新 | 2026-10-31 / 2026-10-31（`HEAD = f3c9c03`，分包重构之后） |
| 上游 | [产品路线 §5 阶段 A](operator-development-product-roadmap.md#L157-L163)、[§6 评价问题与口径](operator-development-product-roadmap.md#L186-L215) |
| 事实来源 | [发布形态](../status/dsh-operator-release-readiness-20261030.md)、[GPU 中断恢复](../status/gpu-interrupt-recovery-20261030.md)、[S5 结项](../status/dsh-operator-s5-closeout-20261030.md)、[仓库瘦身与分包](repo-slimming-20261031.md)、[可迁移性](dsh-operator-portability-4090.md) |
| 本期资源前提 | 零 GPU 机时、零模型调用；H20 与 P800 已下线，读数不可再生 |

本期只做两件事：**把已验证的能力交付身份钉死**，**建立阶段 A 的任务与基线**。不新增能力。三项工作按 W1 → W2 → W3 顺序推进，W3 依赖 W2 的场景选择结论。

v0.1 写成之后，工作区被四个提交扫过（证据出仓 + 布局守卫、`tests/` 分目录、`benchmarks/` 分目录、`operator_agent/` 分五层）。**本期目标不变，W1 的形状变了**：原来缺的是"提交在盘改动"，现在缺的是"**已交付工件与仓库 HEAD 不再是同一身份**"。

## 0. 开工前先读的三条边界

- 路线的阶段 A 是 B 的门槛，不是可跳过的文书工作。本期结束时必须能回答**两问**：哪些环节会失败、单次代价多少（机器侧）；人在哪些环节出现、为什么（贡献矩阵）。按环节的人工分钟数不在其中——2026-10-31 已按「甲」把它从门槛里移出（见 §5.1），因为历史没记过、外部也没有这个读数。
- 本期任何"通过"都不得写成插件增益。[S5 已按证据不足收口](../status/dsh-operator-s5-closeout-20261030.md)，插件整体优于原生 DSH 这条主张永久不可补，本期也不试图补。
- 需要真机读数才能定论的判断，本期只交量具与判据清单，判读留给人；不搭对照实验。

## 1. 开工核对（本文档写完即跑过，读数条件一并记录）

**读数取自 2026-10-31 本机、`HEAD = f3c9c03`、工作区干净。** v0.1 那节的 `HEAD = de1918f` 读数已失效，按下面五条替换。

| # | 事实 | 本次实算证据 |
|---|---|---|
| F1 | 默认清理修复**已入库**，v0.1 里"在盘未提交"这条缺口不成立了 | `git log -S'owned_worker_scope'` 指向 `ef6acea`（`tests/` 分目录那个提交）；`6cb1fca` 上 `backend_ownership.py` 内该符号出现 **0** 次。即：修复与一次 578 文件重构挤在同一提交里，交付动作本身没留独立记录 |
| F2 | **在盘 0.6.1 wheel 与 HEAD 不是同一身份**，且版本号没变 | ① 布局：wheel 内是包根平铺（`operator_agent/tool_calls.py`），HEAD 包根只剩 `__init__.py`+`__main__.py`，模块进五层（`ls` 计 24/31/16/22/10 含各层 `__init__`）② 入口：wheel `entry_points.txt` = `operator_agent.cli:main` / `.mcp_server:main`，HEAD [pyproject.toml:21](../../pyproject.toml#L21-L22) = `operator_agent.app.cli:main` / `app.mcp_server:main` ③ 两处都自称 `0.6.1`（[pyproject.toml:7](../../pyproject.toml#L7)、[operator_agent/__init__.py:20](../../operator_agent/__init__.py#L20)） |
| F3 | **协议身份断链**，比 F2 更硬，且已有具体数值 | 两侧 `protocol_identity()` 用仓库自己的 `digest()` 归约：旧 `78993c98…`（在盘 wheel，解包到独立目录计算）→ 新 `e8c7f2c6…`（`HEAD = 24f4fc2`）。清单仍是 73 条、文件名集合一个没变，但 **14 条是纯搬运（内容哈希不变、仅路径键变）、59 条内容也随 import 重写而变**，`remote_scripts/` 三件全变 ⇒ 远端部署集合的身份同时改变。机制后果与失效清单写在 [交付身份重绑](../status/delivery-identity-rebind-20261031.md)：分层前落盘的经验库会被 [corpus_admission 的 drift 门](../../operator_agent/core/corpus_admission.py#L36-L42) 直接拒收，旧批次冻结计划被 workload driver 的比对挡下 |
| F4 | 全量回归读数的四个口径已钉住，其中两个**只有文字、没有留档** | ① 6172 passed / 1 skipped（[中断恢复 §L47](../status/gpu-interrupt-recovery-20261030.md#L47)）——证据树里有 `20261030-gpu-interrupt-recovery-validation/pytest-full.log`，506.34s，但**日志未记解释器** ② 6159 passed + 15 failed + 4 errors（[发布形态 §L53](../status/dsh-operator-release-readiness-20261030.md#L53)）——沙箱内，19 项因 `ps`/`sandbox-exec`/真 DSH 被拒 ③ 6185（[瘦身记录 §L5](repo-slimming-20261031.md#L5)）与 ④ 6183（[同一文件 §L95](repo-slimming-20261031.md#L95)）——**整个证据树里 grep 不到任何一个 20261031 的重跑日志**，两个数都无留档，因此 6185/6183 的差异此刻无法归因。本期新读数见下方 §1.1 |
| F5 | 工作树布局多了守卫，**在仓内落产物会先把守卫跑红** | `output/` 已是指向仓外证据树的符号链接，位置与单副本警告写在 [evidence-location.json](../research/evidence-location.json)；[tests/hygiene/test_repo_layout.py](../../tests/hygiene/test_repo_layout.py) 管根目录白名单、单层条目预算、禁缓存/构建产物/证据入库、`output/` 只能是符号链接。v0.1 里"日志落 `output/plugin_completion/`"仍可用（符号链接照旧解析，已实测可读），但构建目录必须落在仓外 |

工件哈希（本次实算，与发布形态记录一致）：

```
097918d395d08bc2859b07edd3db81cd543a3c10b9b588856fd3ea8100df6de4  operator_agent-0.6.1-py3-none-any.whl
8245f1689c9c5b8145986248afd71631c1d6a50a958e194be7a4d222e65bab0d  dsh-operator-agent-0.1.10.tgz
```

- [ ] 复核上述两行哈希，确认文档与在盘工件一致
- [x] 把 F4 的四个读数钉到「哪个提交、哪个解释器、哪个目录集合」→ 结论已写回 F4 与下面 §1.1；判定未改
- [x] 记下 F3 是**结构性断链**：文件名集合不变，但只有 14 条是纯搬运、59 条内容也变了，因此不能靠"内容没改"解释掉 → 见 [交付身份重绑](../status/delivery-identity-rebind-20261031.md)

### 1.1 本期新读数（T1.5，第一次同时有日志与解释器）

| 项 | 值 |
|---|---|
| 命令 | `~/opt/miniconda3/bin/python -m pytest tests/ -q -rfE` |
| 解释器 | Python 3.9.12 / numpy 2.0.2 / torch 2.8.0 / pytest 8.4.2（与 `cpython-39-pytest-8.4.2` 的 `.pyc` 标签一致） |
| 提交 | `37f09fd`，分支 `feat/phase1-delivery-identity` |
| 结果 | **6185 passed / 1 skipped / 2 warnings，467.05s**，退出码 0 |
| 留档 | `output/plugin_completion/20261031-phase1-identity-v1/`（日志 + `manifest.json` + 工件与哈希） |

三条附带事实，都是本次实跑出来的：

1. **第一次跑错解释器就撞在收集阶段**：`/usr/bin/python3` 3.9.6（有 numpy、无 torch）在 6.68 秒内以 2 个 `ModuleNotFoundError: No module named 'torch'` 中断收集（`tests/unit/test_pairing_window.py`、`tests/unit/test_remote_input_fidelity.py`）。[CI 定义的注释](../../.github/workflows/ci.yml)与 [completion-plan:243](dsh-operator-completion-plan-20261026.md#L243) 早就写明这一点，是读数的解释器条件，不是仓库缺陷。该次失败日志被重跑覆盖，读数只保留在 `manifest.json` 里。
2. **6185 在含版本 bump 的 HEAD 上原样复现**，说明 T1.2 的 8 处改动既没加测试也没破测试；至于 6183，因为没有留档日志，**本期不给解释**。
3. **本清单自己触发了一次公开仓卫生守卫**：写解释器时用了家目录绝对路径，`tests/hygiene/test_repo_hygiene.py::test_no_credential_shaped_content` 立刻变红（`/Users/[A-Za-z0-9_-]+` 命中本文档一行），改成 `~` 写法后回绿。origin 是公开仓，这条命中是真命中，不是误报。

## 2. W1 交付身份收口

目标一句话：**任何人从仓库检出即可构建出与安装说明同身份的 wheel 与插件包，且没有任何一份冻结记录被当成新身份引用。** 不需要 GPU，不需要模型。

v0.1 的 T1.1（提交在盘改动）已被重构顺带完成，但换来的是 F2/F3：**已交付工件与 HEAD 成了两个身份、共用一个版本号**。所以本期 W1 的第一动作从"提交"变成"重绑身份"。

- [x] **T1.1 写一份身份重绑记录（本期第一个产出）。** 已落 [交付身份重绑](../status/delivery-identity-rebind-20261031.md)：两侧聚合摘要 `78993c98…` → `e8c7f2c6…`、14 纯搬运 / 59 内容变、入口与布局差异、三处硬失败机制、五份冻结记录的失效清单。复算命令按文档原样跑过（`/usr/bin/python3` 3.9.6，`HEAD = 24f4fc2`）。未改任何判据、未重跑 GPU、未覆盖任何哈希。
- [x] **T1.2 版本号强制递增。** Python `0.6.1` → `0.6.2`，插件壳 `0.1.10` → `0.1.11`。改了 8 处活副本：[pyproject.toml:7](../../pyproject.toml#L7)、[operator_agent/__init__.py:20](../../operator_agent/__init__.py#L20) 与同文件首行 docstring、[README.md:7](../../README.md#L7)、[package.json:3](../../integrations/dsh-operator/package.json#L3)、[CHANGELOG.md](../../integrations/dsh-operator/CHANGELOG.md) 新增 `## 0.1.11` 段、插件 README 的安装命令与发布形态段落。带日期的状态文档按「历史快照不改写」保留 `0.6.1` 原文。读数：`tests/hygiene` **59 passed**、`npm test` **58/58**、`capabilities()` 返回 `version=0.6.2`、`tool_version=9`（工具契约未变）。
- [x] **T1.3 仓外干净检出重建。** 已建：`git clone` 到临时目录 → `uv build --wheel`，产出 `operator_agent-0.6.2-py3-none-any.whl`（968,687 bytes，`11683240…`）与 `dsh-operator-agent-0.1.11.tgz`（31,751 bytes，`0ce665a7…`）。三条断言逐条查看过：包内是五层 + `ops` + `prompts`(11)/`skills`(2)、`dist-info/licenses/` 两份齐备、`entry_points.txt` 指向 `operator_agent.app.*`。**关键一条：解包后 `digest(protocol_identity())` 等于 HEAD 的 `e8c7f2c6…`，即干净检出确实复现同一身份。** 读数条件：本机 `/usr/bin/python3` 3.9.6（pip 21.2.4、setuptools 58 都低于 `pyproject` 要求，故走 `uv build` 自带的隔离构建）。工件与哈希落 `output/plugin_completion/20261031-phase1-identity-v1/`，详见[重绑记录 §5](../status/delivery-identity-rebind-20261031.md)。附带抓到一次真约束：构建残留 `build/`、`operator_agent.egg-info` 会让布局守卫变红——所以「构建落在仓外」不是洁癖。
- [x] **T1.4 反向对照（双向都做了）。** 在临时克隆里跑，不在共享工作树切提交。同一份 [tests/unit/test_owned_cleanup_defaults.py](../../tests/unit/test_owned_cleanup_defaults.py)：在 `ef6acea`（修复所在提交）**11 passed**；在 `6cb1fca`（该提交 `backend_ownership.py` 内 `owned_worker_scope` 出现 0 次）**11 failed**，失败消息是行为缺失（`owned local worker did not reach the expected state`），不是导入报错。⇒ 这组测试确实锁住了行为，T1.3 的绿灯有效。
- [x] **T1.5 沙箱外全量 pytest 一次留档。** 已做，读数见 §1.1：**0 failed、0 error**，退出码 0，日志里不再出现上一轮的 `Operation not permitted` 字样，所以 [发布形态 §3](../status/dsh-operator-release-readiness-20261030.md#L57-L59) 那 19 项被拒用例本轮没有被拒。已知顺序相关偶发（[分包记录 §未处理项](repo-slimming-20261031.md#L140-L142) 的 `test_hitl_tools::…gets_two_consequences`）**本轮未复现**，按一次读数记录，不宣称已定位污染源。**本轮量具自身有一处缺口**：只加了 `-q -rfE`，没加 `-rs`，那 1 项 skip 的名字与原因没有留档（`-q` 不打印 skip 详情），下一轮补。F4 的四个口径已钉住，其中 6185/6183 无留档日志这一事实写在 §1。
- ~~**T1.6 CI 首跑**~~ **本期不纳入**（用户 2026-10-31 决定：一期做到「本地可发布」为止，不 push）。工作流仍未在远端真实执行过；解除条件是先过凭据/脱敏扫描并单独确认，因为 origin 是公开库、`design/autonomous-spine` 至今零 push。
- [x] **T1.7 安装说明与身份绑定。** [插件 README](../../integrations/dsh-operator/README.md) 的安装段落已随 T1.2 换成 `0.6.2`/`0.1.11`，并在发布形态段落指向重绑记录。**工件哈希不写进被打包的 README 本身**（写进去就自指，改了文字哈希必变），改记在[重绑记录 §5](../status/delivery-identity-rebind-20261031.md) 与证据树的 `artifacts-SHA256.txt`。
- [ ] **T1.8 入口说明与包布局矛盾。** 两处同类漂移，都是重构扫过时留下的「文字对了、路径错了」：① [operator_agent/__init__.py:4-15](../../operator_agent/__init__.py#L4-L15) 的「模块结构」还在列平铺名（`schemas.py`、`kernel_db.py`、`mock_gpu.py`…），分层后这些都已带层前缀；② 变异表头部的跑法命令——如 [benchmarks/mut_specs/transport_error_mask.py:5](../../benchmarks/mut_specs/transport_error_mask.py#L5) 写 `-m benchmarks.mutation_calibrate`，而 Phase 3 之后真实模块是 `benchmarks.calibration.mutation_calibrate`（本次实测照旧写法起不来，新表 `human_ledger.py` 已按真实路径写）。与版本 bump 分开放，避免一次改动混两类事实。

本期**不做** `npm publish` / `twine upload`：会产生不可撤回的公开版本号，属发布决定，不在一期范围。

## 3. W2 阶段 A：任务复盘、场景、基线

目标是产出**下一轮能直接被引用**的三张表，不是访谈记录本身。样本用于发现问题和选试点，不推断市场比例。**本期产出全部写在 [阶段 A 台账与基线](../research/80-阶段A真实任务台账与基线.md)。**

- [x] **T2.1 选真实任务。** 按路线 §2.4「用现有任务、代码与实验记录还原流程」，从存量证据选定 **8 条真实任务**（台账 L1–L8）：30 次三臂对照、24 格协作研究、性能 A/C、端到端成功率量具、插件修复验收、手搓 vs 插件会话、LeetCUDA 20 题、Golden 4 例。**当事人自述的 3–5 例仍缺**，见 T2.2。
- [ ] **T2.2 每例固定字段。** 机器侧字段已逐条填好（结局、调用、token、墙钟、卡点与出处）；**人工活跃时间、等待、是否接入、愿意委托哪些环节一律未知**——全仓 grep 不到任何人时读数，唯一近似是一次会话的 79 次工具调用与「45 次 bash ≈2–4 分钟」的散文估算（[hitl-bench-sweep:8-16](../design/hitl-bench-sweep.md#L8-L16)），不可复现且计的是调用数不是人时。**当事人已答复「没记录过」**，且本轮查证：**外部也没有「按环节的人工时间百分比」这种读数**（台账 §3.1，八条来源逐条核过）。⇒ 此条不再当作「等人回忆」的待办，改由 **T3.7 的量具**承接；历史人时永久不可得，这是已知缺口而非本期欠账。「是否接入」一条可直接定：代码无接入阶段（状态机止于 `REPORT`），三条候选任务一律记**未接入**。
- [x] **T2.3 外部吸纳补齐基线。** KernelAgent（PyTorch 博客 2026-03-06）全文本次抓取核对：六段流程、1.56×/2.02×/89% roofline、matvec 9.52→1.95 ms 四次架构改动、「手工要数天到数周」。**已标注非本项目访谈**；GDPA 与 NVIDIA 工具链两条未重核，沿用路线原记录。对照结论：他们最前两段（Profile/Diagnose）在本仓是 `None`。另查八条来源做「能否用外部读数代替」的判定（台账 §3.1）：KEET 点名 profile 解读为成本环节、Harness Engineering 附录 Table 5 给人与 agent 的贡献矩阵、FuriosaAI 的 12 天/283 小时属**总耗时非人时**、2609.21058 给端到端可改份额 80–86% 卡在 cuBLAS/FA ⇒ 端到端上限约 1%、AgentKernelArena 的主导失败是入口契约。**没有任何一条给出按环节的人时百分比。**
- [x] **T2.4 输出场景表并排序。** 环节排序五位（L3 判定 → L1/形状契约 → 委托被绕过 → 传输中断 → 测量有效性）+ 一条「接入/端到端＝无读数不可建基线」。选定场景：**目标硬件上对已有正确实现做性能确认与优化**。理由与推翻条件写在台账 §5，是一句可反驳的话。
- [ ] **T2.5 冻结评价问题与门槛（跑之前）。** 骨架已就位（主问题推荐**固定开发预算**、重复方式、停止规则、分母口径），但**收益门槛待人确认**——两个约束已摆明：门槛设在 1.00x 时量到的是门槛位置（实测中位 0.99x、±3% 刀刃），而预注册 1.05× 在 A/C 上只有 1.34%/0.24%。这一步做完才允许碰任何 GPU。
- [x] **T2.6 口径落表。** 八项指标 × 单位 × 取数字段 × 已知陷阱（台账 §8）：token 曾被跨运行累加**高估 3.25 倍**、「不要引用省 61%」、`workflow_progress` 是 ordinal 非时长、`utilization.memory` 是时间占比、人工投入**禁止用模型墙钟代替**、账单缺失记未知。
- [x] **T2.7 记下无 GPU 就能算的存量问题。** 五条待办（台账 §9）：LeetCUDA 批量 `llm_calls=0` 的来源、540114 错值是否还被引用、语料带轴为 0 不可作分母、44/47/76/79 欠复核、以及**工具与传输吃掉绝大部分墙钟**必须单列否则会被算成收益。

## 4. W3 冻结"开机即跑"的阶段 B 计划

避免下一次拿到卡时现编实验——那正是 S5 24 格只跑 2 格的成因。

- [ ] **T3.1 选卡。** 按[可迁移性 §6 最小动作](dsh-operator-portability-4090.md) 与 G1–G5 缺口评估目标设备；优先级取"一次非 CUDA 真迁移 > 再加一张 N 卡"，红线是走 triton-ascend 路线、不学厂商私有 C 接口。
- [ ] **T3.2 迁移门先行。** 未知卡 launch 地板、共享内存解析、roofline 分母口径、噪声带标定这四项在没有真卡时**只能写探针、不能声称已验证**；探针本身作为 W3 产出交付。
- [ ] **T3.3 控制变量清单。** 模型、硬件、任务输入、起始代码、参考实现、预算上限逐项固定，必要差异写明；开发基线用目标用户实际在用的 Agent 与常用脚本，保留其正常验证与执行能力。
- [ ] **T3.4 记录绕过与额外成本。** 插件发现失败、工具调用未成功、用户绕过工具、额外配置与纠错耗时都要有栏位；只统计成功样本的表不予接受。
- [ ] **T3.5 产出 runbook。** 一条命令序列 + 预期读数 + 停止条件，落在拿到卡的那天可直接执行；GPU 型号与 UUID 运行时探测，不写死常量。注意中断恢复驱动已随分目录搬到 [benchmarks/experiments/misc/gpu_interrupt_recovery.py](../../benchmarks/experiments/misc/gpu_interrupt_recovery.py)，runbook 里别再写平铺老路径。
- [ ] **T3.6 复用既有 workload 前先重绑 evaluator 身份。** [Phase 3 记录](repo-slimming-20261031.md#L47-L50) 与 [Phase 2 记录](repo-slimming-20261031.md#L115-L118) 都写明：`workload_decision_h20.evaluator_sources()`、`workload_llm_search_h20.sources()`、`quality_evaluation.evaluator_identity()`、`s5_gateway_real_preflight._identities()` 拿文件名当哈希键，分目录后键值变了。⇒ 新批次不得与引用旧身份的 H20 冻结记录混在同一张表里报数，也不得声称"接着上轮跑"。
- [x] **T3.7 把「人在哪个环节出现」做成量具，替代追不回来的人时。** 外部查证（台账 §3.1）显示按环节的人时百分比**全行业都没有**，可执行的替代口径是 [Harness Engineering 附录 Table 5「Human and Agent Contribution Matrix」](https://arxiv.org/abs/2607.17979) 那个形态：**不追分钟，追「谁做了哪类动作 + 为什么」**。具体做法：每次运行按环节记四元组（人是否介入 / 介入动作 / 介入原因 / 可选的分钟量级），字段默认**留空而不是 0**，未填一律计入 unknown 并进分母；报表必须把「人自报量级」与「机器墙钟」两栏分开印。这样阶段 A 那句「谁在哪个环节花时间」就有了可测形式，且零机时即可先把记录接口、报表列与变异表写完。**判读仍留给人**：本期只落量具，不顺手把对照实验搭起来跑。

  落地读数（2026-10-31 本次实跑）：量具在 [operator_agent/core/human_ledger.py](../../operator_agent/core/human_ledger.py)——六个环节逐字取自路线 §5.1 的过程句，`involved` 缺省 `unknown`、分母恒为全部环节、人自报档位与机器秒分列且**不提供相加入口**；拒绝消息一律回显合法取值（沿用[发布形态 §1 第 5 项](../status/dsh-operator-release-readiness-20261030.md#L13)修好的那个姿势）。用例 [tests/unit/test_human_ledger.py](../../tests/unit/test_human_ledger.py) **15 条通过**，正反配对（默认 unknown ↔ 显式 no 才计 no）。变异表 [benchmarks/mut_specs/human_ledger.py](../../benchmarks/mut_specs/human_ledger.py) 八行，`--check-only` 核过锚点与证人，实跑 **行为分母 8/8 全部 KILLED、结构分母 0/0**，跑完工作树复原；其中 M8 守的是 `isinstance(True, int)` 这个语言层既成事实。**边界**：量具没有接进任何 MCP 工具回包——那会改工具契约，属另一件事，需单独设计与验收。

## 5. 本期出口判据

| 判据 | 通过条件 | 现状（2026-10-31） |
|---|---|---|
| 交付身份 | T1.1 重绑记录入库，且 T1.3 与 T1.4 同时成立——仓外干净检出可复现工件，去掉修复时覆盖确实变红 | **已达**：记录入库、新 wheel 解包后摘要等于 HEAD、对照双向 11 红/11 绿 |
| 版本一致 | T1.2 完成后，`版本号 → 内容` 重新一一对应；旧工件哈希未被覆盖（这正是发布形态 §1 第 3 项修过、又被重构冲掉的缺陷） | **已达**：`0.6.2`/`0.1.11` 与 `e8c7f2c6…` 绑定，旧 `0.6.1`/`0.1.10` 原样保留 |
| 验证留档 | T1.5 与 T1.6 各有一份日志与哈希，红/绿与偶发原样记录 | **T1.5 已达**（§1.1：日志 + 解释器 + 0 failed/0 error）；**T1.6 按用户决定划出一期**，CI 仍未真实跑过 |
| 阶段 A | （按 §5.1 改写后的形式）指出哪些环节会失败、单次代价多少；并指出人在哪些环节出现、为什么 | **第一半已达**：两处失败集中点各有两批独立复现 + 外部同向（台账 §4）。**第二半量具已落**：T3.7 的 15 条用例与 8/8 变异拦截，读数本身要等阶段 B 的运行当场产生 |
| B 就绪 | T3.5 runbook 可在无卡条件下被第三方逐步复核，且 T3.6 的身份绑定已写明 | **未开始** |

五条缺一不进入阶段 B。**W1 的三项（交付身份、版本一致、T1.5 留档）本期已达；W2 的场景、排序与口径已就位，欠 T2.5 的收益门槛确认；T1.6 CI 首跑按决定划出一期；W3 的 runbook 与 T3.7 量具未动工。**

### 5.1 阶段 A → B 的门槛：已按**甲**改写（2026-10-31 定案）

路线原句把「能明确指出谁在什么环节花费时间，以及怎样证明改善」当成推进到 B 的条件（[§5 表 A 行](operator-development-product-roadmap.md#L159)）。当事人答复没记录过，本轮又查证外部没有同类读数，于是三个选项里只有一个可执行，**已采纳甲**：

| 方案 | 内容 | 处置 |
|---|---|---|
| **甲** | 门槛拆两半：①哪些环节会失败、单次代价多少（机器侧）；②人在哪些环节出现、为什么（贡献矩阵，当场记录）。历史人时明确记为不可追 | **已采纳**：路线 [§5 A 行](operator-development-product-roadmap.md#L159) 已改，[§2.4](operator-development-product-roadmap.md#L70-L76) 与 [§6.3「人工实际投入」](operator-development-product-roadmap.md#L211) 同步写明可行形式；路线自己的修订记录加了 v0.2 一行 |
| 乙 | 维持原句，阶段 B 无限期停在门槛外 | 未采纳：这条门槛从未被任何公开工作满足过 |
| 丙 | 用外部定性表述（「数天到数周」「profile 解读最贵」）冒充我方基线 | **判为不可接受**：读数条件错位，等于把别人的工具当成我们的证据 |

引用路线这些行锚时要重核：改写后的路线图行号会随编辑漂移，本清单只保证本次核过（本次 `grep -n` 逐条实定位）。

## 6. 本期明确不做，以及各自的解除条件

| 不做 | 解除条件（来自[路线 §5.2](operator-development-product-roadmap.md#L175-L184)） |
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
| 2026-10-31 | v0.3：T1.1、T1.2 完成并打勾，F3 换成具体数值（聚合摘要 `78993c98…`→`e8c7f2c6…`、14 纯搬运 / 59 内容变、`remote_scripts` 三件全变、corpus drift 门这一机制后果）；按用户决定把 T1.6 CI 首跑划出一期范围（本地做完，不 push）；T1.7 拆出「等 T1.3 再填哈希」；新增 T1.8 记入口 docstring 与包布局矛盾 | [交付身份重绑](../status/delivery-identity-rebind-20261031.md)；两侧 `protocol_identity()`/`digest()` 复算；`tests/hygiene` 59 passed、`npm test` 58/58、`capabilities()` 报 `0.6.2` / 契约 9 |
| 2026-10-31 | v0.4：T1.3、T1.4、T1.7 完成。新工件 `0.6.2`（`11683240…`）解包后身份摘要等于 HEAD，插件包 `0.1.11`（`0ce665a7…`）；反向对照做成双向（`ef6acea` 11 绿 / `6cb1fca` 11 红，且失败原因是行为缺失）；§5 出口判据加「现状」列——交付身份与版本一致两条已达，验证留档是本期唯一欠账 | 仓外 `uv build --wheel` 与 `npm pack` 实跑；解包目录逐项查看布局/入口/prompts-skills；双向对照各跑一次；构建残留触发布局守卫变红一次（记为约束证据） |
| 2026-10-31 | v0.5：T1.5 完成并新增 §1.1（6185 passed / 1 skipped / 0 failed / 0 error，467.05s，`37f09fd`，解释器与日志同时留档）；F4 四个口径钉住并查明 **6185 与 6183 在任何地方都没有留档日志**、6172 的日志未记解释器；记下本期量具自身的缺口（没加 `-rs`，那 1 项 skip 的名字与原因没留住）；跑错解释器导致合集中断一次，已归到读数条件而非仓库缺陷 | `~/opt/miniconda3/bin/python`（3.9.12）实跑与日志复核；`/usr/bin/python3`（3.9.6，无 torch）那次 6.68s 中断；证据树 `grep` 查档；`manifest.json` 落档 |
| 2026-10-31 | v0.6：**W2 落地**，T2.1/T2.3/T2.4/T2.6/T2.7 完成、T2.2 与 T2.5 留为待补，产出 [阶段 A 台账与基线](../research/80-阶段A真实任务台账与基线.md)。核心事实：**全仓从未记录人工活跃时间**（唯一近似是一次会话的 79 次工具调用与散文估算），所以「谁在哪个环节花时间」不能靠存量证据回答；机器侧基线已能建，失败集中在 L3 判定（L2 10/10 对 L3 5/10、中位 0.99x）与 L1/形状契约（matmul 三臂 0/3、33% 首轮 L1 失败每轮 ≈20s+22k token）两处；外部 KernelAgent 核查显示其最前两段（Profile/Diagnose）在本仓仪器为 `None` | 台账八条任务逐条实读出处（S5 四批、research 48、hitl-bench-sweep、`leetcuda_bench/agent_full20/results.json` 的 aggregate、Golden 记录）；KernelAgent 博客全文抓取核对；`docs/` 全量 grep 人时字段 |
| 2026-10-31 | v0.7：当事人答复「没记录过人时」，遂把「能否用外部读数代替」查到底。T2.3 补入八条外部来源与**负结果**（按环节的人时百分比全行业都没有）；T2.2 从「等人回忆」改为已知缺口，改由新任务 **T3.7**（借 Harness Engineering 附录 Table 5 的贡献矩阵形态：记「人是否介入 / 动作 / 原因 / 可选量级」，默认留空不填 0）承接；「是否接入」直接定为**未接入**（代码无接入阶段）；新增 §5.1 **阶段 A→B 门槛改写提案**（甲/乙/丙，丙判为不可接受）待点头；§5 表补回被截断的 B 就绪行并更新现状 | 台账 §3.1 与其 v0.2 修订记录；`output/harness/*/pass_rate.json` 两批 aggregate；[harness_pass_rate.py:283-290](../../benchmarks/drivers/harness_pass_rate.py#L283-L290) |
| 2026-10-31 | v0.8：**用户选定甲**，门槛改写落地。① 路线 §5 A 行改成两半式条件、§2.4 加「历史不可追」、§6.3 允许自报量级档并禁止与秒折叠、路线自己的修订记录加 v0.2 行；② **T3.7 落地**：`core/human_ledger.py`（六环节取自路线 §5.1；unknown 缺省；分母恒全；两列不折叠；拒绝消息回显合法取值）+ 15 条用例 + 8 行变异表（行为分母 8/8 KILLED，跑后复原）；③ §5.1 从提案改为定案记录，阶段 A 判据按新形式重写为「第一半已达 / 第二半量具已落、读数待阶段 B 产生」；④ 因路线图插行，**逐条重算并修正了引用它的行锚**（+2 漂移），并把「引用锚点需重核」写进 §5.1；⑤ T1.8 扩为两处入口说明漂移，新增变异表头部命令路径已失效一条 | `pytest tests/unit/test_human_ledger.py` 15 passed；`benchmarks.calibration.mutation_calibrate --spec … --check-only` 与实跑输出；`grep -n` 逐条重定位路线行号；`git status` 确认变异后工作树复原 |
