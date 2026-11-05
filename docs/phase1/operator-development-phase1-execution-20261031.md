# 第一期执行清单：v0.7 基础可用版

| 项目 | 说明 |
|---|---|
| 文档性质 | 一期可勾选执行清单与进度；历史读数保留各自身份，不代替验收报告 |
| 文档修订 | v0.22 |
| 目标产品版本 | v0.7 基础可用版；开发中，尚未发布；不同于本文修订号与实验计划编号 |
| 创建 / 最近更新 | 2026-10-31 / 2026-11-02（本轮修订：A/B 离线编排与证据汇总） |
| 上游 | [产品路线 §5 版本迭代与 §6 评价口径](../design/operator-development-product-roadmap.md) |
| 事实来源 | [发布形态](../status/dsh-operator-release-readiness-20261030.md)、[GPU 中断恢复](../status/gpu-interrupt-recovery-20261030.md)、[S5 结项](../status/dsh-operator-s5-closeout-20261030.md)、[仓库瘦身与分包](../design/repo-slimming-20261031.md)、[可迁移性](../design/dsh-operator-portability-4090.md) |
| 本期资源前提 | 最初离线准备为零 GPU、零模型；后续 RTX 3080 试跑见 runbook §7.1–§7.4；新轮尚未执行。H20 与 P800 已下线 |

一期最初围绕交付身份、任务基线和执行准备安排 W1 → W2 → W3。**按用户本轮要求，当前以交付 v0.7 完整可用版本为出口**：在支持范围内，可从安装入口完成声明任务、Agent 生成/修复、状态/取消、独立终评和源码/报告导出。原 W1–W3 是本版内工作，完成文档或准备不等于交付版本；自动对照、采集与最终安装回归的待办统一列在 §5.2。v0.8 及以后新增能力不提前打包进一期。

v0.1 写成之后，工作区被四个提交扫过（证据出仓 + 布局守卫、`tests/` 分目录、`benchmarks/` 分目录、`operator_agent/` 分五层）。**本期目标不变，W1 的形状变了**：原来缺的是"提交在盘改动"，现在缺的是"**已交付工件与仓库 HEAD 不再是同一身份**"。

## 0. 开工前先读的三条边界

- 阶段 A 的出口是指出哪些环节失败、单次机器代价多少，并冻结可执行的对照与验收。**2026-11-01 按用户本轮要求，人工贡献矩阵与填表完整度也移出门槛**；两臂均由 Agent 执行并自动记录可观测介入事件，无法观测的部分保持 unknown，不要求用户补填。当前方案见 [runbook §0](stage-b-runbook-20261101.md#0-当前方案agent-自动对照与事件记录2026-11-01-修订)。
- 本期任何"通过"都不得写成插件增益。[S5 已按证据不足收口](../status/dsh-operator-s5-closeout-20261030.md)，插件整体优于原生 DSH 这条主张永久不可补，本期也不试图补。
- 原离线准备只交量具与判据；已有试跑不等于跨臂收益验收。新轮按 runbook §0 先实现执行与自动记录，再冻结身份、运行和终评；本次调整只改方案，不宣称实施完成。

## 1. 开工核对（本文档写完即跑过，读数条件一并记录）

**读数取自 2026-10-31 本机、`HEAD = f3c9c03`、工作区干净。** v0.1 那节的 `HEAD = de1918f` 读数已失效，按下面五条替换。

| # | 事实 | 本次实算证据 |
|---|---|---|
| F1 | 默认清理修复**已入库**，v0.1 里"在盘未提交"这条缺口不成立了 | `git log -S'owned_worker_scope'` 指向 `ef6acea`（`tests/` 分目录那个提交）；`6cb1fca` 上 `backend_ownership.py` 内该符号出现 **0** 次。即：修复与一次 578 文件重构挤在同一提交里，交付动作本身没留独立记录 |
| F2 | **在盘 0.6.1 wheel 与 HEAD 不是同一身份**，且版本号没变 | ① 布局：wheel 内是包根平铺（`operator_agent/tool_calls.py`），HEAD 包根只剩 `__init__.py`+`__main__.py`，模块进五层（`ls` 计 24/31/16/22/10 含各层 `__init__`）② 入口：wheel `entry_points.txt` = `operator_agent.cli:main` / `.mcp_server:main`，HEAD [pyproject.toml:21](../../pyproject.toml#L21-L22) = `operator_agent.app.cli:main` / `app.mcp_server:main` ③ 两处都自称 `0.6.1`（[pyproject.toml:7](../../pyproject.toml#L7)、[operator_agent/__init__.py:20](../../operator_agent/__init__.py#L20)） |
| F3 | **协议身份断链**，比 F2 更硬，且已有具体数值 | 两侧 `protocol_identity()` 用仓库自己的 `digest()` 归约：旧 `78993c98…`（在盘 wheel，解包到独立目录计算）→ 新 `e8c7f2c6…`（`HEAD = 24f4fc2`）。清单仍是 73 条、文件名集合一个没变，但 **14 条是纯搬运（内容哈希不变、仅路径键变）、59 条内容也随 import 重写而变**，`remote_scripts/` 三件全变 ⇒ 远端部署集合的身份同时改变。机制后果与失效清单写在 [交付身份重绑](delivery-identity-rebind-20261031.md)：分层前落盘的经验库会被 [corpus_admission 的 drift 门](../../operator_agent/core/corpus_admission.py#L36-L42) 直接拒收，旧批次冻结计划被 workload driver 的比对挡下 |
| F4 | 全量回归读数的四个口径已钉住，其中两个**只有文字、没有留档** | ① 6172 passed / 1 skipped（[中断恢复 §L47](../status/gpu-interrupt-recovery-20261030.md#L47)）——证据树里有 `20261030-gpu-interrupt-recovery-validation/pytest-full.log`，506.34s，但**日志未记解释器** ② 6159 passed + 15 failed + 4 errors（[发布形态 §L53](../status/dsh-operator-release-readiness-20261030.md#L53)）——沙箱内，19 项因 `ps`/`sandbox-exec`/真 DSH 被拒 ③ 6185（[瘦身记录 §L5](../design/repo-slimming-20261031.md#L5)）与 ④ 6183（[同一文件 §L95](../design/repo-slimming-20261031.md#L95)）——**整个证据树里 grep 不到任何一个 20261031 的重跑日志**，两个数都无留档，因此 6185/6183 的差异此刻无法归因。本期新读数见下方 §1.1 |
| F5 | 工作树布局多了守卫，**在仓内落产物会先把守卫跑红** | `output/` 已是指向仓外证据树的符号链接，位置与单副本警告写在 [evidence-location.json](../research/evidence-location.json)；[tests/hygiene/test_repo_layout.py](../../tests/hygiene/test_repo_layout.py) 管根目录白名单、单层条目预算、禁缓存/构建产物/证据入库、`output/` 只能是符号链接。v0.1 里"日志落 `output/plugin_completion/`"仍可用（符号链接照旧解析，已实测可读），但构建目录必须落在仓外 |

工件哈希（本次实算，与发布形态记录一致）：

```
097918d395d08bc2859b07edd3db81cd543a3c10b9b588856fd3ea8100df6de4  operator_agent-0.6.1-py3-none-any.whl
8245f1689c9c5b8145986248afd71631c1d6a50a958e194be7a4d222e65bab0d  dsh-operator-agent-0.1.10.tgz
```

- [x] 复核上述两行哈希，确认文档与在盘工件一致 → 本次 `shasum -a 256` 实算与发布形态记录一致；其后因文档迁入 `docs/phase1/` 改了随包文字，工件已升到 `0.6.3`/`0.1.12`（再往后是 `0.6.4`/`0.1.13`），两代哈希与身份核对见[重绑记录 §5 与 §5.1](delivery-identity-rebind-20261031.md)
- [x] 把 F4 的四个读数钉到「哪个提交、哪个解释器、哪个目录集合」→ 结论已写回 F4 与下面 §1.1；判定未改
- [x] 记下 F3 是**结构性断链**：文件名集合不变，但只有 14 条是纯搬运、59 条内容也变了，因此不能靠"内容没改"解释掉 → 见 [交付身份重绑](delivery-identity-rebind-20261031.md)

### 1.1 本期新读数（T1.5，第一次同时有日志与解释器）

| 项 | 值 |
|---|---|
| 命令 | `~/opt/miniconda3/bin/python -m pytest tests/ -q -rfE` |
| 解释器 | Python 3.9.12 / numpy 2.0.2 / torch 2.8.0 / pytest 8.4.2（与 `cpython-39-pytest-8.4.2` 的 `.pyc` 标签一致） |
| 提交 | `37f09fd`，分支 `feat/phase1-delivery-identity` |
| 结果 | **6185 passed / 1 skipped / 2 warnings，467.05s**，退出码 0 |
| 留档 | `output/plugin_completion/20261031-phase1-identity-v1/`（日志 + `manifest.json` + 工件与哈希） |

**第二次全量（加入 T3.7 量具之后）**：读数条件是 `HEAD = aa55a39` + 三个尚未跟踪的新文件（`core/human_ledger.py`、`tests/unit/test_human_ledger.py`、`benchmarks/mut_specs/human_ledger.py`，内容即随后提交的 `2539d42`）——当时跑的就是这份树，不是 `2539d42` 这个检出本身。结果 **6200 passed / 1 skipped / 2 warnings**，472.88 s，退出码 0，日志 `full-regression-with-ledger.log`。6200 = 上一轮 6185 + 新增 15 条用例，**没有别数变化**；本轮补了 `-rs`，那条 skip 的名字与原因终于留住：`tests/unit/test_remote_input_fidelity.py:365 — requires real CUDA; CPU protocol tests do not count`。§1.1 上面记的「量具自身缺口（没加 `-rs`）」就此闭合。

**第三次全量（`0.6.4` 树，W3 文档与 T1.8 修正之后）**：**6200 passed / 1 skipped / 2 warnings**，455.76 s，0 failed / 0 error，退出码 0，日志 `output/plugin_completion/20261101-phase1-docindex-v1/full-regression.log`。与第二次**总数相同**，说明这轮的改动（版本号字符串、包根 docstring 分层清单、34 份变异表死路径、四份文档搬家）零行为影响；同一条 skip 仍是 `test_remote_input_fidelity.py:365`（requires real CUDA）。先前那条「`0.6.3`/`0.6.4` 这棵树没跑过全量」的缺口至此闭合。

**第四次全量（`0.6.5` 树，CLI 与迁移门入口落地之后）**：**6226 passed / 1 skipped / 2 warnings**，459.31 s，0 failed / 0 error，退出码 0，日志 `full-regression-065.log`。6226 = 上一轮 6200 + 本次新增 26 条（迁移门 14 + CLI 与量具面 12），**逐项对得上，无别数变化**；同一条 skip 仍是那条 requires real CUDA。

三条附带事实，都是本次实跑出来的：

1. **第一次跑错解释器就撞在收集阶段**：`/usr/bin/python3` 3.9.6（有 numpy、无 torch）在 6.68 秒内以 2 个 `ModuleNotFoundError: No module named 'torch'` 中断收集（`tests/unit/test_pairing_window.py`、`tests/unit/test_remote_input_fidelity.py`）。[CI 定义的注释](../../.github/workflows/ci.yml)与 [completion-plan:243](../design/dsh-operator-completion-plan-20261026.md#L243) 早就写明这一点，是读数的解释器条件，不是仓库缺陷。该次失败日志被重跑覆盖，读数只保留在 `manifest.json` 里。
2. **6185 在含版本 bump 的 HEAD 上原样复现**，说明 T1.2 的 8 处改动既没加测试也没破测试；至于 6183，因为没有留档日志，**本期不给解释**。
3. **本清单自己触发了一次公开仓卫生守卫**：写解释器时用了家目录绝对路径，`tests/hygiene/test_repo_hygiene.py::test_no_credential_shaped_content` 立刻变红（`/Users/[A-Za-z0-9_-]+` 命中本文档一行），改成 `~` 写法后回绿。origin 是公开仓，这条命中是真命中，不是误报。

## 2. W1 交付身份收口

目标一句话：**任何人从仓库检出即可构建出与安装说明同身份的 wheel 与插件包，且没有任何一份冻结记录被当成新身份引用。** 不需要 GPU，不需要模型。

v0.1 的 T1.1（提交在盘改动）已被重构顺带完成，但换来的是 F2/F3：**已交付工件与 HEAD 成了两个身份、共用一个版本号**。所以本期 W1 的第一动作从"提交"变成"重绑身份"。

- [x] **T1.1 写一份身份重绑记录（本期第一个产出）。** 已落 [交付身份重绑](delivery-identity-rebind-20261031.md)：两侧聚合摘要 `78993c98…` → `e8c7f2c6…`、14 纯搬运 / 59 内容变、入口与布局差异、三处硬失败机制、五份冻结记录的失效清单。复算命令按文档原样跑过（`/usr/bin/python3` 3.9.6，`HEAD = 24f4fc2`）。未改任何判据、未重跑 GPU、未覆盖任何哈希。
- [x] **T1.2 版本号强制递增。** Python `0.6.1` → `0.6.2`，插件壳 `0.1.10` → `0.1.11`。改了 8 处活副本：[pyproject.toml:7](../../pyproject.toml#L7)、[operator_agent/__init__.py:20](../../operator_agent/__init__.py#L20) 与同文件首行 docstring、[README.md:7](../../README.md#L7)、[package.json:3](../../integrations/dsh-operator/package.json#L3)、[CHANGELOG.md](../../integrations/dsh-operator/CHANGELOG.md) 新增 `## 0.1.11` 段、插件 README 的安装命令与发布形态段落。带日期的状态文档按「历史快照不改写」保留 `0.6.1` 原文。读数：`tests/hygiene` **59 passed**、`npm test` **58/58**、`capabilities()` 返回 `version=0.6.2`、`tool_version=9`（工具契约未变）。
- [x] **T1.3 仓外干净检出重建。** 已建：`git clone` 到临时目录 → `uv build --wheel`，产出 `operator_agent-0.6.2-py3-none-any.whl`（968,687 bytes，`11683240…`）与 `dsh-operator-agent-0.1.11.tgz`（31,751 bytes，`0ce665a7…`）。三条断言逐条查看过：包内是五层 + `ops` + `prompts`(11)/`skills`(2)、`dist-info/licenses/` 两份齐备、`entry_points.txt` 指向 `operator_agent.app.*`。**关键一条：解包后 `digest(protocol_identity())` 等于 HEAD 的 `e8c7f2c6…`，即干净检出确实复现同一身份。** 读数条件：本机 `/usr/bin/python3` 3.9.6（pip 21.2.4、setuptools 58 都低于 `pyproject` 要求，故走 `uv build` 自带的隔离构建）。工件与哈希落 `output/plugin_completion/20261031-phase1-identity-v1/`，详见[重绑记录 §5](delivery-identity-rebind-20261031.md)。附带抓到一次真约束：构建残留 `build/`、`operator_agent.egg-info` 会让布局守卫变红——所以「构建落在仓外」不是洁癖。
- [x] **T1.4 反向对照（双向都做了）。** 在临时克隆里跑，不在共享工作树切提交。同一份 [tests/unit/test_owned_cleanup_defaults.py](../../tests/unit/test_owned_cleanup_defaults.py)：在 `ef6acea`（修复所在提交）**11 passed**；在 `6cb1fca`（该提交 `backend_ownership.py` 内 `owned_worker_scope` 出现 0 次）**11 failed**，失败消息是行为缺失（`owned local worker did not reach the expected state`），不是导入报错。⇒ 这组测试确实锁住了行为，T1.3 的绿灯有效。
- [x] **T1.5 沙箱外全量 pytest 一次留档。** 已做，读数见 §1.1：**0 failed、0 error**，退出码 0，日志里不再出现上一轮的 `Operation not permitted` 字样，所以 [发布形态 §3](../status/dsh-operator-release-readiness-20261030.md#L57-L59) 那 19 项被拒用例本轮没有被拒。已知顺序相关偶发（[分包记录 §未处理项](../design/repo-slimming-20261031.md#L140-L142) 的 `test_hitl_tools::…gets_two_consequences`）**本轮未复现**，按一次读数记录，不宣称已定位污染源。**本轮量具自身有一处缺口**：只加了 `-q -rfE`，没加 `-rs`，那 1 项 skip 的名字与原因没有留档（`-q` 不打印 skip 详情）——**已在 T3.7 之后的第二次读数里闭合**（§1.1：`test_remote_input_fidelity.py:365`，requires real CUDA）。F4 的四个口径已钉住，其中 6185/6183 无留档日志这一事实写在 §1。
- ~~**T1.6 CI 首跑**~~ **本期不纳入**（用户 2026-10-31 决定：一期做到「本地可发布」为止，不 push）。工作流仍未在远端真实执行过；解除条件是先过凭据/脱敏扫描并单独确认，因为 origin 是公开库、`design/autonomous-spine` 至今零 push。
- [x] **T1.7 安装说明与身份绑定。** [插件 README](../../integrations/dsh-operator/README.md) 的安装段落已随 T1.2 换成 `0.6.2`/`0.1.11`，并在发布形态段落指向重绑记录。**工件哈希不写进被打包的 README 本身**（写进去就自指，改了文字哈希必变），改记在[重绑记录 §5](delivery-identity-rebind-20261031.md) 与证据树的 `artifacts-SHA256.txt`。
- [x] **T1.8 入口说明与包布局矛盾。** 两处都修了。① [operator_agent/__init__.py](../../operator_agent/__init__.py) 的「模块结构」从平铺清单改写成分层清单（core/decide/execute/generate/app + ops/prompts/skills，只列实际存在的模块名，全量归属指向分包记录 §5）。② 变异表头部的死命令 `-m benchmarks.mutation_calibrate` → `-m benchmarks.calibration.mutation_calibrate`：**实测 34 份文件带死路径**（30 份 mut_specs + runner 自身文档串 + 2 份测试头部）。先前那句「59 份」是把 `__pycache__` 数进去了，作废。修完用官方 runner `--check-only` 跑一条做正例（11 行锚点与证人齐备）。带日期快照里的 3 处旧命令按「历史文档不改写」保留不动。

本期**不做** `npm publish` / `twine upload`：会产生不可撤回的公开版本号，属发布决定，不在一期范围。

## 3. W2 阶段 A：任务复盘、场景、基线

目标是产出**下一轮能直接被引用**的三张表，不是访谈记录本身。样本用于发现问题和选试点，不推断市场比例。**本期产出全部写在 [阶段 A 台账与基线](stage-a-ledger-20261031.md)。**

- [x] **T2.1 选真实任务。** 按路线 §2.4「用现有任务、代码与实验记录还原流程」，从存量证据选定 **8 条真实任务**（台账 L1–L8）：30 次三臂对照、24 格协作研究、性能 A/C、端到端成功率量具、插件修复验收、手搓 vs 插件会话、LeetCUDA 20 题、Golden 4 例。**当事人自述的 3–5 例仍缺**，见 T2.2。
- ~~**T2.2 每例固定字段**~~ **已划出范围**（2026-11-01 定案，与 T1.6 同类处理）。机器侧字段已逐条填好（结局、调用、token、墙钟、卡点与出处）；人时段（活跃时间、等待、愿意委托哪些环节）——全仓 grep 不到任何人时读数，唯一近似是一次会话的 79 次工具调用与「45 次 bash ≈2–4 分钟」的散文估算（[hitl-bench-sweep:8-16](../design/hitl-bench-sweep.md#L8-L16)），不可复现且计的是调用数不是人时；当事人答复「没记录过」，且八条外部来源逐条核过后**外部也没有「按环节的人工时间百分比」这种读数**（台账 §3.1）。历史四格因此永久不可得，挂在待办里只会让人误以为本期欠着；**原先的 T3.7 当场填表安排已在本轮取消，未来由 Agent 自动记录有来源的介入事件，未观测到的内容保持未知**。「是否接入」这一格反而**不需要回忆**：代码无接入阶段（状态机止于 `REPORT`），三条候选任务一律记**未接入**——据此「业务交付」这一档应从一期成功标准里剔除，不能由 kernel 读数外推。
- [x] **T2.3 外部吸纳补齐基线。** KernelAgent（PyTorch 博客 2026-03-06）全文本次抓取核对：六段流程、1.56×/2.02×/89% roofline、matvec 9.52→1.95 ms 四次架构改动、「手工要数天到数周」。**已标注非本项目访谈**；GDPA 与 NVIDIA 工具链两条未重核，沿用路线原记录。对照结论：他们最前两段（Profile/Diagnose）在本仓是 `None`。另查八条来源做「能否用外部读数代替」的判定（台账 §3.1）：KEET 点名 profile 解读为成本环节、Harness Engineering 附录 Table 5 给人与 agent 的贡献矩阵、FuriosaAI 的 12 天/283 小时属**总耗时非人时**、2609.21058 给端到端可改份额 80–86% 卡在 cuBLAS/FA ⇒ 端到端上限约 1%、AgentKernelArena 的主导失败是入口契约。**没有任何一条给出按环节的人时百分比。**
- [x] **T2.4 输出场景表并排序。** 环节排序五位（L3 判定 → L1/形状契约 → 委托被绕过 → 传输中断 → 测量有效性）+ 一条「接入/端到端＝无读数不可建基线」。选定场景：**目标硬件上对已有正确实现做性能确认与优化**。理由与推翻条件写在台账 §5，是一句可反驳的话。
- [x] **T2.5 冻结评价问题与门槛（跑之前）。** 当事人给定预算：**每任务 ≤30 次模型调用 / ≤40 分钟（2400 s）墙钟**。已按[阶段 B runbook §1](stage-b-runbook-20261101.md) 冻结：主问题＝固定开发预算；达标＝完整 L2 + 一次 `valid` 的 L3 + 终稿哈希绑定一致三条件同时满足；设计 4 任务 × 2 臂；最小可检出单位 +1（并写明它不是统计显著性）；停止规则四条；**墙钟是开发预算不是 GPU 时间上限**（路线 §5.4 禁的正是这种混用）。CLI 默认是 24 次 / 1800 s，必须显式覆盖——这一条已从 `cli.py:39-145` 实读确认。
- [x] **T2.6 口径落表。** 八项指标 × 单位 × 取数字段 × 已知陷阱（台账 §8）：token 曾被跨运行累加**高估 3.25 倍**、「不要引用省 61%」、`workflow_progress` 是 ordinal 非时长、`utilization.memory` 是时间占比、人工投入**禁止用模型墙钟代替**、账单缺失记未知。
- [x] **T2.7 记下无 GPU 就能算的存量问题。** 五条待办（台账 §9）：LeetCUDA 批量 `llm_calls=0` 的来源、540114 错值是否还被引用、语料带轴为 0 不可作分母、44/47/76/79 欠复核、以及**工具与传输吃掉绝大部分墙钟**必须单列否则会被算成收益。

## 4. W3 冻结"开机即跑"的阶段 B 计划

避免下一次拿到卡时现编实验——那正是 S5 24 格只跑 2 格的成因。

- [x] **T3.1 选卡。** 按[可迁移性 §6 最小动作](../design/dsh-operator-portability-4090.md) 与 G1–G5 缺口评估目标设备；优先级取"一次非 CUDA 真迁移 > 再加一张 N 卡"，红线是走 triton-ascend 路线、不学厂商私有 C 接口。**定卡结果：RTX 4070（CC 8.9，独占、ssh 可达）作为第一张，用来拿阶段 A 的第二半读数与阶段 B 的 4×2 格**。按证据判它可用也记它的四处代价（8.9 兜底行按 L40S 偏大 ⇒ 判停出口基本不打开；`agents.py` 缺 8.9 分支 —— **同日已修**，见 T3.2，代价换成「换卡必须连 runner 一起部署」；预检阈值 —— **同日更正**，独占默认档是 `min_free_mib=4096`、12 GB 卡可过，8192 只属 `shared-known-processes` 那档且被 `validate_policy` 钉死，两条都只在 S5 驱动层生效；带宽约 H20 的 1/7 ⇒ 别期待复现旧读数）。**「一次非 CUDA 真迁移」是顺延，不是放弃**：4070 是最短路径，昇腾 + triton-ascend 那条路仍待核覆盖度。判据与处置见 [runbook §7](stage-b-runbook-20261101.md)。**同日卡源有变**（改成公开算力市场的价目表）：市场 20 档机型的逐行判定、两处表内缺陷（V100 行给了 Volta 没有的 bf16 分母、A100-40GB 命中 80GB 口径而 `peak_trusted` 仍 True）与排序见 [`docs/research/80-算力市场机型可用性与连接方式核查.md`](../research/80-算力市场机型可用性与连接方式核查.md)；能力齐全里首选 **RTX 3070 8 GB**，要与 H20 同代讲同一个故事则 **3090 24 GB**。**判据一条没改**，改的只是机型与它的代价清单。
- [x] **T3.2 迁移门先行。** 未知卡 launch 地板、共享内存解析、roofline 分母口径、噪声带标定这四项在没有真卡时**只能写探针、不能声称已验证**；探针本身作为 W3 产出交付 —— **已交付**：[unknown_card_gates.py](../../benchmarks/probes/unknown_card_gates.py) 五道门 G1–G5，规则是「读数来源必须是本卡 UUID，否则 fail；缺读数 unknown；**只有全 pass 才允许开跑**」。门的输入侧也补了采集器 `benchmarks/probes/collect_unknown_card_probe.py`（读设备属性与 UUID，**拒绝给共享形态和 smem 编兜底值**；字段契约由用例钉住，防的是「采集器写一个键、门读另一个键」这种只会全 unknown 不会崩的错）。**同日把两处缺口补掉**：① `remote_scripts/run_kernel.py` 的 `gpu_info` 自描述 `hardware_report`（每 SM 共享内存 / opt-in 字节 / L2 / 线程数 / 寄存器 / warp，读不到就省键 + 一条 `hardware_report_missing`），Designer 侧提出 `smem_budget()`（实测 > cc 表 > legacy 申报 > 抛），两处 `128` 兜底删掉；② 标定输出接进门的 `peak`/`launch_floor` 段——两份测量脚本尾巴各打一行 `PROBE_JSON {...}`（带本卡 uuid），采集器 `--from-probe` 读它，**uuid 不符或缺 uuid 直接拒**，所以 G3 不再是「永远 unknown 但没人知道为什么」。**用例 26→34→59 条（门 21 / 采集器 19 / Designer 与 runner 接面 19），变异表 7→9→21 行，本轮 21/21 拦住（行为 18 / 结构 3）**。顺带查出两个会让 4070 开不了跑的地方：`shared_memory_per_multiprocessor` 不是每个 torch build 都暴露 ⇒ 门与采集器改成**两个上限有一个就够**（全没有才拒），Designer 只有 opt-in 时按 opt-in 取值（Ada 99 ≤ 100，保守方向）；`128` 兜底在 legacy 分支里还留了一处 `else 128`，一起删。改到随包文件 ⇒ 再升 `0.6.6`/`0.1.15`，协议摘要 `e8c7f2c6…` → **`b937fd8e…`**（断链记录 §5.4；`4008621` 提交说明里写的 `19ab44fc…` 是**中途树**的摘要，那次算之后我又改了 opt-in 兜底那一处，此值在任何一棵树上都不存在，已作废并记入勘误）。**五道门至今零读数**——它们要真实卡上的探测输出，形状不是读数。**2026-11-01 更正：G4/G5 的产出方已补齐并在 RTX 3080 20 GB 上跑出第一份读数，五门全绿**（门之外的探针形状未变，见 [runbook §7.1](stage-b-runbook-20261101.md) 与 §5 末段）。
- [x] **T3.3 控制变量清单。** 见 [runbook §2](stage-b-runbook-20261101.md)：模型与采样（含 `reasoning effort`，S5 那条 8192 预算被内部推理吃满的教训）、同一卡与运行时 UUID、同一输入与参考实现哈希、同一 seed 候选哈希、三档预算分列、基线臂保留正常验证与执行能力；文件系统可见性必须如实声明——S5 曾因「隔离没有建立」把 A 臂当成裸环境而返工。
- [x] **T3.4 记录字段方案已落。** 新口径见 [runbook §0.3 与 §3](stage-b-runbook-20261101.md)：自动关联工具轨迹、调用账本、配置/纠错机器耗时和可观测人工事件，缺来源写 unknown。插件臂绕过委托的格不计插件工作流收益；基线正常 shell 与 Agent 动作不算人工绕过。自动采集实施另列 §5.2，不能由本勾选推断已实现。
- [x] **T3.5 runbook 已落。** 新流程为迁移门 → 冻结契约 → Agent 执行两臂 → 查询/取消 → 独立终评 → 自动归档汇总；取消强制填人时表。G1–G5 未满足即停，GPU 与 UUID 运行时探测。新轮执行入口和自动汇总的实施缺口见 runbook §0.4。
- [x] **T3.6 复用既有 workload 前先重绑 evaluator 身份。** 见 [runbook §5](stage-b-runbook-20261101.md)，四条新摘要今日实算（decision 121 条 / codegen 124 / llm_search 124 / quality）。另有两点是查出来的事实而不是规则：① 协议身份历史上**不止一代**——64 条 `fdb76a11…`（S2 留存 1 份）、70 条 `9914985a…`（S4 留存 6 份）、73 条 `78993c98…`（在盘 wheel）、当前 `e8c7f2c6…`；② **全证据树没有任何 JSON 带 `evaluator_sources` 字段**（grep 0 命中），所以旧 evaluator 身份不可复算，缺摘要就记「不可核」，不倒推。以后写「协议身份 N 文件」必须同时给摘要。
- [x] **T3.7 把「人在哪个环节出现」做成量具，替代追不回来的人时。** 外部查证（台账 §3.1）显示按环节的人时百分比**全行业都没有**，可执行的替代口径是 [Harness Engineering 附录 Table 5「Human and Agent Contribution Matrix」](https://arxiv.org/abs/2607.17979) 那个形态：**不追分钟，追「谁做了哪类动作 + 为什么」**。具体做法：每次运行按环节记四元组（人是否介入 / 介入动作 / 介入原因 / 可选的分钟量级），字段默认**留空而不是 0**，未填一律计入 unknown 并进分母；报表必须把「人自报量级」与「机器墙钟」两栏分开印。这样阶段 A 那句「谁在哪个环节花时间」就有了可测形式，且零机时即可先把记录接口、报表列与变异表写完。**判读仍留给人**：本期只落量具，不顺手把对照实验搭起来跑。

  落地读数（2026-10-31 本次实跑）：量具在 [operator_agent/core/human_ledger.py](../../operator_agent/core/human_ledger.py)——六个环节逐字取自路线 §5.1 的过程句，`involved` 缺省 `unknown`、分母恒为全部环节、人自报档位与机器秒分列且**不提供相加入口**；拒绝消息一律回显合法取值（沿用[发布形态 §1 第 5 项](../status/dsh-operator-release-readiness-20261030.md#L13)修好的那个姿势）。用例 [tests/unit/test_human_ledger.py](../../tests/unit/test_human_ledger.py) **15 条通过**，正反配对（默认 unknown ↔ 显式 no 才计 no）。变异表 [benchmarks/mut_specs/human_ledger.py](../../benchmarks/mut_specs/human_ledger.py) 八行，`--check-only` 核过锚点与证人，实跑 **行为分母 8/8 全部 KILLED、结构分母 0/0**，跑完工作树复原；其中 M8 守的是 `isinstance(True, int)` 这个语言层既成事实。**边界**：量具没有接进任何 MCP 工具回包（那会改工具契约，属另一件事）；**填写入口已落为 CLI 三条子命令** `operator-agent ledger-new|ledger-fill|ledger-report`，用例 10 条、含「重复 new 必须拒绝且已答格子原样还在」。另配 [七行变异表](../../benchmarks/mut_specs/unknown_card_gates.py)：初版六行里两行 ESCAPED，暴露 G4 当时只要求「写了共享形态」就放行——**先修门再补用例**，才变成七行全拦。

**T3.7 当前用途（2026-11-02 更新）**：上述人时 CLI 的实现与历史测试事实保留，降为可选自报工具；不再要求用户填写贡献矩阵。离线事件汇总与来源/覆盖核对已实现，正式宿主会话和外部编辑观测源仍待接入。缺人工观测不阻止阶段推进，也不能据此声称没有人工投入或已经省人时。

## 5. 本期出口判据

| 判据 | 通过条件 | 当前状态（2026-11-01 修订） |
|---|---|---|
| 交付身份 | 重绑记录、可复现工件与行为反向对照齐备，均绑定对应树 | **旧工件已达，新轮待绑定**：历代重建与双向 11 红/11 绿保留；真机修复后工作区与已交付工件不再同身份，按 §5.2 冻结 |
| 版本一致 | `版本号 → 内容` 一一对应，旧工件及哈希不覆盖 | **旧工件已达，新树未收口**：已交付至 `0.6.7`/`0.1.16`；当前未提交修复尚未升版、重建。历史工件表见[重绑记录](delivery-identity-rebind-20261031.md) |
| 验证留档 | T1.5 及后续受影响验证各有日志与条件，红/绿与偶发原样记录；T1.6 已划出 | **T1.5 已达**（§1.1）；新轮只引用其对应树的检查，不跨身份复用通过结论。CI 仍未真实跑过 |
| 阶段 A | 指出哪些环节失败、单次机器代价多少，并定义可执行对照与验收；不以人工记录完整度为条件 | **已达**：台账 §4 与新 runbook §0；人工活跃时间保持未知，本期不证明节省人时 |
| B 就绪 | 新计划、对照入口、共同预算、自动记录与身份条件已明确；实施完成后才能派发 | **离线编排/汇总已实现，正式接线未完成**：runbook §0.4 的源码/量具身份冻结、正式基线/宿主入口及观测源仍待做；不能复用旧轮“已就绪”作为新轮派发许可 |
| v0.7 可用交付 | 支持清单内完整入口安装可用，任务结果可导出，交付包/示例/已知问题/回退与身份清单齐备，前版承诺能力回归 | **待验收、未发布**：原准备工作与旧工件读数不足以勾选；按 §5.2 完成本版交付检查。是否优于基线依据对照另报，不由“可用”推出 |

**旧离线准备曾按 23 项完成 / 2 项划出收口，此数不代表当前整期无待办。** 人工记录已不再阻塞推进；新轮的实施与冻结按 §5.2 执行。旧交付身份、版本与回归记录只覆盖当时的树，当前未提交的真机修复仍需重新绑定，不能把旧工件的通过结论外推到新树。CI 首跑与历史人时仍不纳入。

**2026-11-01 追加（第 0 步已在另一张卡上跑到底）**：G4/G5 原先**没有任何产出方**（不是"没跑"，
是"没人能填"），本轮补了两个采集器并在一台 **RTX 3080 20 GB（cc 8.6）** 上跑出第一份真机读数，
**五门全绿、退出码 0**。三条要带走：① 这张卡的 cc 8.6 兜底行对 tf32 偏大 **9.35 倍**、对带宽偏大
2.20 倍，而 **fp32 偏小 0.88 倍**（偏小会提前停，是危险方向）；② G4 的阈值在这张卡上轮间差 2–6 倍，
degraded 档只能到 20% 量级 ⇒ 该档读数支撑不了小于 ~22% 的判决；③ 两张地板读数（38.9 µs 无冲刷 vs
3.07 µs 有冲刷）**不是同一把尺**，混用会算出物理上不可能的"地板支配 292%"。读数与缺口全文见
[runbook §7.1](stage-b-runbook-20261101.md)。**换卡不改变 T3.1 判据，但 4×2 的结论仍须按新身份另表。**

### 5.1 阶段 A → B 的门槛：历史定案与本轮修订

**当前有效口径（2026-11-01）**：用户认为人工执行与填表不合理，已要求调整方案。阶段 A 以机器侧问题基线和可执行验收为出口；人工贡献矩阵、分钟数及补填完整度均不再作为门槛。自动记录可观测介入，缺失 unknown；历史方案「甲」与下表作为决策经过保留，不再要求完成其中的人工记录才开跑。

路线原句把「能明确指出谁在什么环节花费时间，以及怎样证明改善」当成推进到 B 的条件（[§5 表 A 行](../design/operator-development-product-roadmap.md#L161)）。当事人答复没记录过，本轮又查证外部没有同类读数，于是三个选项里只有一个可执行，**已采纳甲**：

| 方案 | 内容 | 处置 |
|---|---|---|
| **甲** | 门槛拆两半：①哪些环节会失败、单次代价多少（机器侧）；②人在哪些环节出现、为什么（贡献矩阵，当场记录）。历史人时明确记为不可追 | **已采纳**：路线 [§5 A 行](../design/operator-development-product-roadmap.md#L161) 已改，[§2.4](../design/operator-development-product-roadmap.md#L70-L78) 与 [§6.3「人工实际投入」](../design/operator-development-product-roadmap.md#L255) 同步写明可行形式；路线自己的修订记录加了 v0.2 一行 |
| 乙 | 维持原句，阶段 B 无限期停在门槛外 | 未采纳：这条门槛从未被任何公开工作满足过 |
| 丙 | 用外部定性表述（「数天到数周」「profile 解读最贵」）冒充我方基线 | **判为不可接受**：读数条件错位，等于把别人的工具当成我们的证据 |

引用路线这些行锚时要重核：改写后的路线图行号会随编辑漂移，本清单只保证本次核过（本次 `grep -n` 逐条实定位）。

### 5.2 当前执行待办（新轮，不计作已完成）

**rc2 复验更新**：安装工件已在兼容 DSH 上完成一条 H20 FP16 Softmax 的任务、独立终评和导出，另有持久 SSH 取消回执。性能资格和更广支持范围仍按各自证据验收。最新结果见 [rc2 同版本复验与交付清单](../testing/v07-delivery.md#10-rc2-同版本复验与交付清单)和[八格接入记录](../status/phase1-eight-cell-20261104.md)。

- [x] 实现显式离线 FakeBackend 与脚本模型，接入正常任务、修复、预算、状态/取消和归档；全链路标记 test_only，隔离测试语料，L3 不具备 GPU 验收资格。用法与回归范围见 [fake 后端说明](../testing/fake-backend.md)。这是组件检查完成，不代表以下对照与版本交付已完成。
- [x] 实现 [A/B 离线编排与自动汇总](../testing/phase1-rehearsal.md)：独立基线循环与生产自主 worker，串行交错八个小 fixture 格位；复用模型共同预算与验证预留，绑定终稿/契约做独立终评，保留失败、取消、未执行、未知、事件来源与覆盖，支持只读回放。只计离线演练，不计正式八格或 DSH 宿主验证。
- [x] 实现[模型 / SSH / DSH 接入代码与候选支持矩阵](../testing/phase1-connected.md)：独立模型基线、生产 worker 和真实 DSH 启动配置，外层/内层共用模型预算，固定工具参数与单次委派、身份检查、独立终评及回放；修正 FP16 LayerNorm v2 的 X-only 契约与声明精度覆盖。此项只表示接入实现，不等于正式同口径或交付验收。
- [x] 完成[无需租卡的候选安装与交付准备](../testing/v07-delivery.md)：Python 3.12 干净 wheel 安装、最终 tgz 的实际 DSH 离线工具流程、六种随包示例、完整 L2 独立终评、源码/报告导出、隔离 profile 安装/卸载及无网络卸载重装；提供候选工件、匹配本机的依赖、源码快照、哈希和可重复验收工具。用户个人 profile 的实测与本轮隔离环境分别记载，真机与正式发布仍待验收。
- [ ] 固定源码、远端 runner、量具与交付身份，冻结 `phase1-agent-comparison-v2` 的四任务八格及 A/B 实际入口；旧试跑只作探索记录。
- [x] 完成[租卡前预登记与实例绑定实现](../testing/phase1-study.md)：固定四任务完整材料、实际原生 DSH / 插件两臂、统一 wire 采样、预算与停止规则、本地身份；提供真实实例只读采集、绑定和逐格漂移检查。GPU 事实与真实 provider 验证尚未产生，正式执行仍待上一项的实例绑定。
- [ ] 在已接线入口上核定实际模型请求采样参数，冻结正式基线与 DSH 插件臂；接入所需真实观测源并标明覆盖。当前内层角色参数与 DSH 外层参数尚未建立可比性，外部编辑/人工会话仍无完整观测，不能用本地接入测试补齐正式结果，不依赖用户填表。
- [ ] 同卡串行交错执行八格、独立终评并归档全部终态；报告噪声、共享观测、未知与失败，不以自动化执行代替收益证明。
- [ ] 冻结 v0.7 的硬件/运行时/算子/dtype/布局支持清单；干净环境从 DSH 插件安装入口跑通声明→生成/修复→状态/取消→独立终评→源码/报告导出，并回归前版承诺能力。CLI 试跑不能代替宿主入口验收。
- [ ] 形成可用交付包：安装依赖、可运行示例、结果说明、已知问题与回退方式、产品版本到各包版本及哈希的清单；功能/收益/交付状态分别记载。本地验收后可作为本版交付，公开发布仍不在本清单既定范围。

## 6. 本期明确不做，以及各自的解除条件

| 不做 | 解除条件（来自[路线 §5.2](../design/operator-development-product-roadmap.md#L182-L191)） |
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
- **历史文档里的旧路径不改**：[Phase 3 记录末尾](../design/repo-slimming-20261031.md#L57-L59) 已定口径——带日期的 `docs/status/**` 与 `docs/research/**` 是当时快照，里面的 `tests/test_x.py`、`benchmarks/x.py` 保留原样。本清单引用它们时同样不改写；只有当前入口类文档随迁移更新。
- `exec_guard` 不是沙箱；MCP 工具不经 DSH 审批层；网络唯一出口与敌意候选隔离仍属后移项。
- 中断恢复验收只覆盖受信任固定候选、常驻 SSH runner 与显式新任务，不证明任意失控 kernel 的 watchdog、服务商停止或全部后代退出。

## 修订记录

| 日期 | 变更 | 依据 |
|---|---|---|
| 2026-11-03 | v0.22：接通显式模型基线、SSH 和实际 DSH + 源码插件入口，共享外层/内层预算、固定单次委派及独立终评；明确四算子 FP16/FP32 候选范围，修正 LayerNorm v2 输入与精度覆盖。正式采样可比性、同卡八格与交付验收仍待完成 | [接入说明与本地回归](../testing/phase1-connected.md)；实际 DSH 使用本地假模型，不能作真机收益证据 |
| 2026-11-02 | v0.21：新增 A/B 离线编排、共同预算核对、独立终评、事件来源与缺失覆盖汇总、取消和只读回放；共享额度耗尽按正常预算停止分类。正式 Agent/宿主/真机入口接线仍未完成 | 用户要求先开发自动执行和汇总；[演练说明](../testing/phase1-rehearsal.md) |
| 2026-10-31 | 创建 v0.1：把路线阶段 A 改写为可勾选一期清单，含 W1 交付身份收口、W3 冻结 B 计划 | 本次实算的工件与 `HEAD`/工作区哈希、发布形态与中断恢复记录、可迁移性评估 |
| 2026-10-31 | v0.2：适配四层重构（`HEAD = f3c9c03`）。§1 读数整段替换为 F1–F5；W1 第一动作由"提交在盘改动"改为"身份重绑"，版本号递增从条件项变硬条件，构建改到仓外，反向对照基线钉到 `6cb1fca`；W3 新增 T3.6 evaluator 身份重绑；出口判据四条扩为五条；§6 加"不回退布局"，§7 加旧身份不可复现与历史文档不改写 | 本次实算：wheel 与 HEAD 的 `entry_points`/模块集合/路径键差异（73 条清单里 70 条变）、`git log -S` 定位 `ef6acea`、`6cb1fca` 上符号 0 次、四个全量读数、布局守卫与符号链接 |
| 2026-10-31 | v0.3：T1.1、T1.2 完成并打勾，F3 换成具体数值（聚合摘要 `78993c98…`→`e8c7f2c6…`、14 纯搬运 / 59 内容变、`remote_scripts` 三件全变、corpus drift 门这一机制后果）；按用户决定把 T1.6 CI 首跑划出一期范围（本地做完，不 push）；T1.7 拆出「等 T1.3 再填哈希」；新增 T1.8 记入口 docstring 与包布局矛盾 | [交付身份重绑](delivery-identity-rebind-20261031.md)；两侧 `protocol_identity()`/`digest()` 复算；`tests/hygiene` 59 passed、`npm test` 58/58、`capabilities()` 报 `0.6.2` / 契约 9 |
| 2026-10-31 | v0.4：T1.3、T1.4、T1.7 完成。新工件 `0.6.2`（`11683240…`）解包后身份摘要等于 HEAD，插件包 `0.1.11`（`0ce665a7…`）；反向对照做成双向（`ef6acea` 11 绿 / `6cb1fca` 11 红，且失败原因是行为缺失）；§5 出口判据加「现状」列——交付身份与版本一致两条已达，验证留档是本期唯一欠账 | 仓外 `uv build --wheel` 与 `npm pack` 实跑；解包目录逐项查看布局/入口/prompts-skills；双向对照各跑一次；构建残留触发布局守卫变红一次（记为约束证据） |
| 2026-10-31 | v0.5：T1.5 完成并新增 §1.1（6185 passed / 1 skipped / 0 failed / 0 error，467.05s，`37f09fd`，解释器与日志同时留档）；F4 四个口径钉住并查明 **6185 与 6183 在任何地方都没有留档日志**、6172 的日志未记解释器；记下本期量具自身的缺口（没加 `-rs`，那 1 项 skip 的名字与原因没留住）；跑错解释器导致合集中断一次，已归到读数条件而非仓库缺陷 | `~/opt/miniconda3/bin/python`（3.9.12）实跑与日志复核；`/usr/bin/python3`（3.9.6，无 torch）那次 6.68s 中断；证据树 `grep` 查档；`manifest.json` 落档 |
| 2026-10-31 | v0.6：**W2 落地**，T2.1/T2.3/T2.4/T2.6/T2.7 完成、T2.2 与 T2.5 留为待补，产出 [阶段 A 台账与基线](stage-a-ledger-20261031.md)。核心事实：**全仓从未记录人工活跃时间**（唯一近似是一次会话的 79 次工具调用与散文估算），所以「谁在哪个环节花时间」不能靠存量证据回答；机器侧基线已能建，失败集中在 L3 判定（L2 10/10 对 L3 5/10、中位 0.99x）与 L1/形状契约（matmul 三臂 0/3、33% 首轮 L1 失败每轮 ≈20s+22k token）两处；外部 KernelAgent 核查显示其最前两段（Profile/Diagnose）在本仓仪器为 `None` | 台账八条任务逐条实读出处（S5 四批、research 48、hitl-bench-sweep、`leetcuda_bench/agent_full20/results.json` 的 aggregate、Golden 记录）；KernelAgent 博客全文抓取核对；`docs/` 全量 grep 人时字段 |
| 2026-10-31 | v0.7：当事人答复「没记录过人时」，遂把「能否用外部读数代替」查到底。T2.3 补入八条外部来源与**负结果**（按环节的人时百分比全行业都没有）；T2.2 从「等人回忆」改为已知缺口，改由新任务 **T3.7**（借 Harness Engineering 附录 Table 5 的贡献矩阵形态：记「人是否介入 / 动作 / 原因 / 可选量级」，默认留空不填 0）承接；「是否接入」直接定为**未接入**（代码无接入阶段）；新增 §5.1 **阶段 A→B 门槛改写提案**（甲/乙/丙，丙判为不可接受）待点头；§5 表补回被截断的 B 就绪行并更新现状 | 台账 §3.1 与其 v0.2 修订记录；`output/harness/*/pass_rate.json` 两批 aggregate；[harness_pass_rate.py:283-290](../../benchmarks/drivers/harness_pass_rate.py#L283-L290) |
| 2026-10-31 | v0.8：**用户选定甲**，门槛改写落地。① 路线 §5 A 行改成两半式条件、§2.4 加「历史不可追」、§6.3 允许自报量级档并禁止与秒折叠、路线自己的修订记录加 v0.2 行；② **T3.7 落地**：`core/human_ledger.py`（六环节取自路线 §5.1；unknown 缺省；分母恒全；两列不折叠；拒绝消息回显合法取值）+ 15 条用例 + 8 行变异表（行为分母 8/8 KILLED，跑后复原）；③ §5.1 从提案改为定案记录，阶段 A 判据按新形式重写为「第一半已达 / 第二半量具已落、读数待阶段 B 产生」；④ 因路线图插行，**逐条重算并修正了引用它的行锚**（+2 漂移），并把「引用锚点需重核」写进 §5.1；⑤ T1.8 扩为两处入口说明漂移，新增变异表头部命令路径已失效一条 | `pytest tests/unit/test_human_ledger.py` 15 passed；`benchmarks.calibration.mutation_calibrate --spec … --check-only` 与实跑输出；`grep -n` 逐条重定位路线行号；`git status` 确认变异后工作树复原 |
| 2026-10-31 | v0.9：T3.7 之后补第二次全量留档 —— **6200 passed / 1 skipped**，472.88 s，0 failed / 0 error；6200 = 6185 + 新增 15 条，无别数变化。本轮带 `-rs`，T1.5 自记的量具缺口（skip 无名）**闭合**：`test_remote_input_fidelity.py:365 — requires real CUDA`。`manifest.json` 增 `full_regression_after_t37` 与 `t37_instrument` 两段 | `output/plugin_completion/20261031-phase1-identity-v1/full-regression-with-ledger.log`；本次实跑与 tail 复核 |
| 2026-11-01 | v0.10：**一期文档迁入 `docs/phase1/`**（清单 + 两份记录 + 新增目录索引 `README.md`）。台账去掉 `80-` 系列号改名 `stage-a-ledger-20261031.md`，并从 `docs/research/README.md` 索引摘行——保留前缀会让研究索引守卫变红。搬动触发 **55 处断链**，由「按 basename 定位真实路径再重写」的解析器全部修掉（零歧义），与 HEAD 基线做差集复查新增断链为 0。随包文字（插件 README/CHANGELOG 与 wheel 内 `human_ledger.py` 的文档路径）变了 ⇒ 按「版本号 ↔ 内容一一对应」升 **`0.6.3` / `0.1.12`** 并重建；`0.6.2`/`0.1.11` 的哈希原样保留不覆盖。**T2.2 改判为已划出范围**（历史四格永久不可得、未来四格由 T3.7 当场产生；「是否接入」直接定为未接入，据此「业务交付」档移出本期成功标准）。前置修复两条：`.workbuddy/` 进 `.gitignore` + `ALLOWED_ROOT` 使布局守卫回绿；并发会话的路线 v0.3 与 526 行事件驱动提案原样落库。另：解析器第一版扫了全仓、顺手改了 `docs/design/hitl-generation-boundary.md` 的 13 处存量断链——**已还原**，存量断链是重构遗留、本次不清理。**并且基线数不能拿临时工作树去扫**——worktree 里没有 `output/` 符号链接，带日期文档里的 `../../output/...` 会被整批判成断链（实测 `dsh-operator-s5-closeout` 一份就贡献 19 条），所以先前那句「基线 333 处」是测量方法错误造成的虚高，作废 | `tests/hygiene` 59 passed（迁移前后各跑）；断链差集对比 HEAD 基线；`date` 核对日历日为 11-01（一期文件名保留 10-31 创建日不改） |
| 2026-11-01 | v0.11：**W3 落地 + T1.8 清完**。新增 [阶段 B runbook](stage-b-runbook-20261101.md)，T2.5（预算 ≤30 调用 / ≤2400 s 墙钟由当事人给定；达标三条件；4×2 设计，最小可检出单位 +1 且注明不是显著性；四条停止规则；墙钟≠GPU 时间）与 T3.3–T3.6 一并打勾；T3.6 查出**协议身份至少四代**（64/70/73平铺/73带层）且**全证据树没有一份 JSON 带 `evaluator_sources`**，旧 evaluator 身份因此不可复算。T1.8 两处都修：包根 docstring 改成分层清单（模块名逐个对过树）、变异 runner 死路径 **34 份**文件修正（先前「59 份」把 `__pycache__` 数了进去，作废），修完 `--check-only` 跑一条正例。包根 docstring 在 wheel 内 ⇒ 再升 `0.6.4`/`0.1.13`（哈希见重绑记录 §5.2），协议摘要仍 `e8c7f2c6…`。§1 哈希复核项闭合，B 就绪判据转为已达——五条里只剩阶段 A 第二半等人时读数 | `git clone` 自 `ba29424` 重建 + `shasum` 实算；`uv build` 输出；`npm pack` 输出；解包内 `digest(protocol_identity())` 复算；`tests/hygiene` 59 passed；`runner --check-only` 11 行全唯一 |
| 2026-11-01 | v0.12：落 **T3.2 探针形状**与**量具 CLI 入口**（一期零机时的最后两块）。新增 `benchmarks/probes/unknown_card_gates.py`（G1–G5 五道门，来源必须是本卡、缺读数 unknown、有 fail/unknown 就非零退出）+ `operator-agent ledger-new/fill/report` 三条子命令；runbook 第 0 步与第 6 步改为直接调这两个入口。**T3.7 的「量具没接任何填写面」这条边界被补上**（仍不进 MCP 工具契约）。变异表第一轮 **M2、M3 ESCAPED**：M3 揭出 G4 只要写了 sharing_pattern 就放行，继承值配一句说明即绿——**修门（G4 必须给 `source_uuid`）再补 4 条用例**，最终七行 7/7 全拦、26 条用例通过。CLI 在 wheel 内 ⇒ 再升 `0.6.5`/`0.1.14` | `mutation_calibrate --spec … `（7/7 KILLED，跑后工作树复原）；`pytest tests/unit/test_unknown_card_gates.py` + `test_cli_ledger.py` 26 passed；`--selftest` 与 CLI 手跑正反例；`tests/hygiene` 59 passed |
| 2026-11-01 | v0.13：**T3.1 定卡 = RTX 4070**（当事人确认独占 + ssh 可达），四处代价记入 runbook §7；非 CUDA 真迁移**顺延不放弃**。T3.2 补齐输入侧：新增采集器 + 契约用例，变异表 7→9 行、用例 26→34 条，**9/9 KILLED**。**一期账面 23 项全完成 / 2 项划出**，剩下的都是执行而非准备 | `--check-only` 与实跑（9/9）；契约用例 8 条；`git status` 确认变异后工作树复原；采集器与门分开实现，未触碰随包文件 ⇒ **本轮无需升号**（`0.6.5`/`0.1.14` 仍是当前身份） |
| 2026-11-01 | v0.14：**补掉两处缺口**（一期最后一批零机时项）。① runner `gpu_info` 自描述 `hardware_report`，Designer 提出 `smem_budget()`（实测 > cc 表 > legacy 申报 > 抛），两处 `128` 兜底删掉；② 标定/地板脚本打 `PROBE_JSON` 行、采集器 `--from-probe` 按 uuid 核对来源后填进门的 `peak`/`launch_floor` 段。顺带修掉一处会让 4070 开不了跑的口径问题（`shared_memory_per_multiprocessor` 不是每个 torch build 都暴露 ⇒ 两个上限有一个就够）。§7 第 4 条的「8192」按实读更正为两档。**升 `0.6.6`/`0.1.15`，协议摘要本期第一次换号：`e8c7f2c6…` → `b937fd8e…`**（`4008621` 提交说明里的 `19ab44fc…` 作废，见目录勘误二）。**读数按当事人指示收窄：本轮只跑受影响子集 196 passed / 4.54s，第五次全量留档未做** | 变异表 21 行实跑 **21/21 KILLED**（行为 18 / 结构 3），跑后 `git status` 复原、grep 不到 `if False:`；契约用例 59 条（门 21 / 采集器 19 / 接面 19）；`--check-only` 与实跑；`git clone` 自 `4008621` + `uv build` + `npm pack`，解包与 HEAD 两侧 `digest()` 相等、`0.6.5` wheel 复算作反向对照；`shasum -a 256` 逐件实算入 `20261101-phase1-smem-selfdescribe-v1/` |
| 2026-11-01 | v0.15：**卡源变了**——从「租一张 4070 实例」改成公开算力市场按小时选卡。新增 [`docs/research/80`](../research/80-算力市场机型可用性与连接方式核查.md)：价目表 20 档**逐行判定**（P4/P40/P100/TITAN X 出局：CC<7.0 编不出来，而 `_INFRA_MARKERS` 不含架构错 ⇒ **环境问题会被记成候选失败**；V100/TITAN V 落在工具链收口区；2080 Ti/T4 可用但异代）、连接方式证实是真 SSH（⇒ 插件臂可跑，Colab 那种没有 sshd 的运行时不行），首选 **3070 8 GB**、同代 **3090 24 GB**。同时更正 runbook §7 第 1 条的**归因错误**（判停出口认的是「分母实测过」，不是 `peak_trusted`）。T3.1 的账面结论**不变**：判据一条没动，只是代价清单换了一张 | 三个文档页 WebFetch 实取（可达性按主机名记：这个域名直连可读，与 GitHub/arXiv 不同口径）；`_PEAK_BY_CC` / `_PEAK_CC_SOURCE` / `_KNOWN_HARDWARE` / `verifier.py:1875,2524` / `cli.py:105` 逐行实读；「3090 Ti 8 GB」按切分卡处理是因为真卡是 24 GB——**独占举证不了就不进表** |
| 2026-11-01 | v0.16：**CC 下限预检落地**。三次选卡提问（TITAN X / P4 / 市场 20 档）暴露的是同一处洞：仓里没有任何门拦得住 Triton 不支持的卡。`core/hardware_caps.py` 新增 `require_triton_support()` 与 `TritonUnsupported`，接在 [`_execute_inner` 的硬件探测之后](../../operator_agent/app/conductor.py#L1482) ⇒ 一台 P4 走完运行体是**抛 + `llm.calls == 0`**（端到端用例拿「被调就炸」的 LLM 假件证）。**「解析不出 cc」判 unknown 后放行**：照旧写法 `_nvidia_cc_major` 把失败折成 0，那道门会把 MockGPU 与 CPU 回退一起判死——防假表的门自己不能变成新的静默降级。新用例 11 条、新变异表 4 行 **4/4 KILLED**。读数按当天定下的三档规则：本轮属「改被广泛读的文件」⇒ unit+hygiene **5633 passed / 1 skipped / 0 failed**（4:26），其后只补了用例与新表文件，最终子集 163 passed + hygiene 复跑为绿；`tests/` 全量（含 acceptance）仍未跑，按指示保持精简。升 `0.6.7`/`0.1.16`，**摘要本期第二次换号** `b937fd8e…` → `5b9b0c8f…`（断链记录 §5.5），这次只在最终树上算一次。**两条记成缺口而不是已修**：`verify`/`repair` 不经 Conductor（那条路上架构不支持仍记成候选失败，加 `_INFRA_MARKERS` 会变无限重试，需要「环境不可用」终态）；7.x 的 `sunset` 分支只是 print、没有活入口 | `tests/unit/test_triton_cc_floor.py` 11 passed（含 AST 接线守卫与非空转证明）；`mutation_calibrate --spec …` 4/4；解包后的 `0.6.7` wheel 内实跑 `6.1/7.5/8.6/None` 四种申报；三侧 `digest()`（HEAD / 0.6.7 / 0.6.6 对照）；`shasum -a 256` 逐件实算入 `20261101-phase1-cc-floor-preflight-v1/` |
| 2026-11-01 | v0.17：**第 0 步第一次在真卡上跑到底**。G4/G5 原先是**结构性缺口**（没有任何脚本产出这两段 ⇒ 门永远 unknown ⇒ runbook 第 0 步必然停住），本轮补上两个产出方：`benchmarks/calibration/calibrate_dispersion_thresholds.py` 加 `PROBE_JSON` 与 `--runs`（G4 三阈值）、新增 `benchmarks/probes/collect_shape_inventory.py`（G5 形状清单），采集器按「只认本卡 uuid + 三键必须齐 + band 键集必须对」合并。在 **RTX 3080 20 GB（cc 8.6，gpushare）** 上 **五门全绿 / exit 0**。同时查实三处口径缺陷：① 两张地板（38.9 µs 无冲刷 vs 3.07 µs 有冲刷）混用会算出「地板支配 292%」，标定脚本现在直接拒；② `nvidia-smi` util 在刚跑完时读的是自己（首轮污染阈值被抬到 **71%**，加静置窗后是 1.0）；③ `torch.cuda.Event` 量化步长会把跨轮 median 的 iqr 打成 **0** ⇒ 带宽 0 会让任何抖动都判显著，故取 median/均值两版的最大值并拒绝零带宽。修掉 `measure_launch_floor.py` 一处**报表缺陷**：尾部 `PROBE_JSON` 里的 `launch_floor` 读的是被 `crossover` 解包覆写的 `med` 与上一轮循环的 `mn`（实测报 600.064/1160.192，真值 47.10/38.91，`min>median` 自相矛盾）——**G1 门拦不住这种错**，它只验来源与样本数，已记成缺口 | 五门 all_green=True；用例 59→81 条、变异 **9/9 KILLED**（含 M2b）；三份标定连跑对比（band.degraded 0.2225/0.0351/0.2299）；本卡实测 706.4 GB/s、fp16 65.44 / bf16 65.90 / tf32 33.36 / fp32 22.14 TFLOPS，与 cc 8.6 行逐格比对；`md5sum` 两侧逐件核对；读数全文与缺口见 [runbook §7.1](stage-b-runbook-20261101.md) |
| 2026-11-01 | v0.18：按用户要求取消人工执行与强制填表，人工贡献矩阵移出阶段 A 出口；T3.4/T3.5 改为自动记录方案，T3.7 降为可选自报。§5 区分旧树已达与新树待绑定，§5.2 列新轮入口、预算控制、自动归档及八格执行待办 | 本轮用户要求调整方案；runbook §0；旧轮四格终评与未提交工作区事实 |
| 2026-11-01 | v0.19：一期归属 v0.7 基础可用版；原 W1–W3 纳入版本内工作，新增支持矩阵、干净安装、DSH 完整入口、源码/报告导出及可用交付包验收，区分功能可用、收益与发布状态 | 用户要求每个版本均可用；长期版本路线 §5/§5.6 |
| 2026-11-02 | v0.20：实现始终离线的 FakeBackend、默认脚本模型、七种场景与三种测试规格；复用真实生成/修复、验证、预算、归档及监督进程，禁止 fake 读数取得 GPU 验收和生产语料资格；本项纳入 v0.7，尚未发布工件 | 用户要求租 GPU 前可验证其他组件；实现及回归见 [fake 后端说明](../testing/fake-backend.md) |
