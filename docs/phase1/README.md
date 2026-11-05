# 第一期文档目录

一期对应 **v0.7 基础可用版**，本地有限支持范围与工件见 [v0.7.0 封版记录](../testing/v07-delivery.md#11-v070-本地封版)。原阶段 A 的任务基线、交付身份收口与对照准备都是该版本内的工作。**目录按交付期分组，不改变文档类型职责**：
清单是计划、两份记录分别是验收类与研究类快照，职责仍按[路线 §9](../design/operator-development-product-roadmap.md) 划分。

**当前方案（2026-11-01 修订）**：两臂均由 Agent 执行；取消用户手工补跑和逐格填人时表的要求，人工记录完整度移出阶段出口。自动汇总机器指标与有来源的可观测介入事件，缺失 unknown，本期不证明节省人时。新轮 `phase1-agent-comparison-v2` 的实施待办与冻结条件见 [runbook §0](stage-b-runbook-20261101.md) 和[执行清单 §5.2](operator-development-phase1-execution-20261031.md)；离线执行与汇总已实现，真实入口和宿主观测源接线仍待做，旧轮四格插件终评不冒充新轮对照结果。

**本版可用出口**：在公布的支持范围内，从安装入口完成任务声明、Agent 生成/修复、状态/取消、独立终评与源码/报告导出；交付包含依赖、示例、支持清单、已知问题和回退说明。对照结果如实披露，功能可用与优于基线分别验收。完整版本顺序见[长期版本路线 §5](../design/operator-development-product-roadmap.md)，本版待办见执行清单 §5.2。

**2026-11-02 实现进展**：[FakeBackend 离线检查](../testing/fake-backend.md)已接入 CLI/MCP 自主任务，默认 fake 模型零 API、零 GPU，可检查生成/修复、预算、归档与取消。结果标记 test_only、独立存放，不替代新轮八格对照、自动事件采集或真机验收；v0.7 仍未发布。

同日新增 [A/B 离线演练入口](../testing/phase1-rehearsal.md)：自动串行交错八个小 fixture 格位、共同预算核对、独立终评、事件与结果汇总、取消和只读回放。A 为独立基线循环，B 为生产自主 worker；人工会话/编辑来源未接入，总介入数仍 unknown。此项不计正式真机八格或 DSH 宿主验收。

| 文档 | 类型 | 内容 |
|---|---|---|
| [operator-development-phase1-execution-20261031.md](operator-development-phase1-execution-20261031.md) | 一期计划 | 可勾选执行清单（W1 交付身份 / W2 阶段 A / W3 B 计划冻结）、出口判据与现状、本期不做项 |
| [delivery-identity-rebind-20261031.md](delivery-identity-rebind-20261031.md) | 事实记录 | 平铺 → 五层造成的协议身份断链（`78993c98…` → `e8c7f2c6…`）、失效清单、六代工件（`0.6.2`→`0.6.7`）逐件哈希与身份核对；本期**两次摘要换号**：`e8c7f2c6…` → `b937fd8e…`（§5.4，runner 自描述）→ `5b9b0c8f…`（§5.5，CC 下限预检） |
| [stage-a-ledger-20261031.md](stage-a-ledger-20261031.md) | 研究记录 | 八条真实任务台账、「人时全仓从未记录」的缺口证明、外部一线流程核查、环节排序与场景选择 |
| [stage-b-runbook-20261101.md](stage-b-runbook-20261101.md) | 执行方案与试跑记录 | §0 当前自动对照方案与实施待办；§1–§6 原预算、控制变量和执行约定；§7.1–§7.4 迁移门、插件臂四格及独立终评事实 |

2026-11-03 接入更新：新增[模型 / SSH / DSH 入口检查](../testing/phase1-connected.md)，A 使用真实模型客户端独立生成/验证，B 显式选择生产 worker 或已安装 DSH + 源码插件。外层和内层共用模型预算，任务/后端/验证预算固定，单次委派后只允许查询和取消。候选支持范围明确为四算子 × FP16/FP32，LayerNorm 为非 affine、最后一轴、eps=1e-5；FP16 v2 输入契约已修正。正式采样参数可比性、真机四任务八格和干净安装交付仍待验收，人工总数继续 unknown。

## 命名说明

台账原名 `80-阶段A真实任务台账与基线.md`，属于 `docs/research/` 的编号系列。搬进本目录后系列号失去意义，
且[研究索引守卫](../../tests/hygiene/test_repo_hygiene.py)要求 `NN-*` 引用必须能在 `docs/research/` 解析——
保留前缀会让守卫变红，因此改名为 `stage-a-ledger-20261031.md`，并从 `docs/research/README.md` 的索引里摘掉那一行。

## 读数与证据在哪里

机器读数与工件不在仓库里（`output/` 是指向仓外证据树的符号链接，布局守卫要求它只能是符号链接）：

```sh
ls output/plugin_completion/20261031-phase1-identity-v1/
#   artifacts/            0.6.2 wheel + 0.1.11 插件包（后续几代另放三个目录）
#   artifacts-SHA256.txt  逐文件哈希
#   full-regression.log / full-regression-with-ledger.log   两次全量留档
#   manifest.json         命令、解释器、对照、守卫、量具缺口
ls output/plugin_completion/20261101-phase1-docindex-v1/artifacts/     # 0.6.3→0.6.5 三代
ls output/plugin_completion/20261101-phase1-smem-selfdescribe-v1/      # 0.6.6 / 0.1.15
ls output/plugin_completion/20261101-phase1-cc-floor-preflight-v1/     # 0.6.7 / 0.1.16
#   scoped-regression.log 各代留档的是**当时那一棵树的读数**，条件写在同目录 manifest.json 里
```

## 读数分档（当事人 2026-11-01 指示「精简，不用跑这么多」之后定下）

| 改动落在哪儿 | 跑什么 | 实测耗时 |
|---|---|---|
| `benchmarks/probes` / `mut_specs` / `tests` / `docs` | 只跑受影响文件 | 秒级 |
| `operator_agent/**` 的叶子模块（例：`agents.py` 提出 `smem_budget()`） | 受影响子集 + `tests/hygiene` | ~3 s |
| **被广泛读的文件**（`conductor` / `hardware_caps` / `run_contract` / 后端） | `tests/unit` + `tests/hygiene` | 4:26（5633 passed / 1 skipped / 0 failed） |
| 发版 / 身份重绑那一次 | 才跑 `tests/`（含 acceptance） | 上一条留档是 `0.6.5` 树 6226 passed |

所以本期最后两代（`0.6.6`/`0.6.7`）**没有** `tests/` 全量读数：`0.6.6` 只留了受影响子集（196 passed），
`0.6.7` 留了 unit+hygiene 全量 5633 + 最终子集 163。缺口写明，不留空框。

证据位置与「单副本、H20/P800 已下线不可再生」的警告写在
[docs/research/evidence-location.json](../research/evidence-location.json)。

## 勘误：本分支有三条提交的标题写错了（历史不动）

三条都复用了**先前某条提交的标题原文**，看 `git log --oneline` 会读成重复劳动，实际内容各不相干。
当事人 2026-11-01 决定不改写历史，因此在此逐条指认。核对方式：`git show --stat <hash>`。

| 提交 | 标题重复了 | 实际内容 |
|---|---|---|
| `1961147` | `4291705`（一期 W2 台账落地） | 阶段 A 出口条件按**甲**改写：路线 §5 A 行拆成两半、§2.4 加「历史不可追」、§6.3 允许自报量级档并禁止与秒折叠，加行锚修正（3 文件，+29/−18） |
| `ba29424` | `2539d42`（人时量具本体） | 阶段 B runbook 落地（预算/达标定义/控制变量/绕过记录/身份重绑）+ T1.8 两处入口漂移修正（含 34 份变异表的死模块路径）+ 升号 `0.6.4`/`0.1.13`（45 文件） |
| `6772930` | `f6df8ad`（第二次全量读数） | **第四次**全量读数：`0.6.5` 树 6226 passed / 1 skipped（= 上一轮 6200 + 新增 26 条），1 文件 +2 行 |

写成这样是我的流程问题，不是仓库的：连续三次在提交时复用了上一条的标题字符串。今后 `git commit` 前先把标题与上一条对拍一遍，写错就当场说、当场请求处置，不等被问。

## 勘误二：`4008621` 提交说明里的协议摘要是中途树的值（历史不动）

那条提交写的是 `e8c7f2c6…` → `19ab44fc…`。**`19ab44fc…` 在任何一棵树、任何一件工件上都不存在**：
它是我在「两处兜底删完之后、加 opt-in 那一档之前」那个中途状态算出来的，算完又改了 `smem_budget()`。
真实值以证据树与 [`delivery-identity-rebind` §5.4](delivery-identity-rebind-20261031.md) 为准：
**`b937fd8ec655604407a7674c091492d5b2532e9de643c3e4fe5d078b1b7b2c43`**（本树 HEAD 与解包后的 `0.6.6`
wheel 两侧算出来一样，`0.6.5` wheel 复算仍是 `e8c7f2c6…` 作反向对照）。
教训归到量具上：**摘要必须在最终树上重算一次再写**，中途值一旦进文档就伪装成「证据被篡改」。

## 新增文件时的两条规矩

1. 只放**属于一期**的文档；一期的验收结论仍进 `docs/status/`（按日期命名），调研与核查进 `docs/research/`（按编号命名）。
2. `tests/hygiene/test_repo_layout.py` 给本目录设了条目预算（8 条）。超预算说明这里已经在变成堆放处，应先收敛而不是提预算。
