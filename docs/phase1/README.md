# 第一期文档目录

一期（阶段 A 需求与基线 + 交付身份收口）的执行文档集中在这里。**目录按交付期分组，不改变文档类型职责**：
清单是计划、两份记录分别是验收类与研究类快照，职责仍按[路线 §9](../design/operator-development-product-roadmap.md) 划分。

| 文档 | 类型 | 内容 |
|---|---|---|
| [operator-development-phase1-execution-20261031.md](operator-development-phase1-execution-20261031.md) | 一期计划 | 可勾选执行清单（W1 交付身份 / W2 阶段 A / W3 B 计划冻结）、出口判据与现状、本期不做项 |
| [delivery-identity-rebind-20261031.md](delivery-identity-rebind-20261031.md) | 事实记录 | 平铺 → 五层造成的协议身份断链（`78993c98…` → `e8c7f2c6…`）、失效清单、重建后的 `0.6.3` 工件与身份核对 |
| [stage-a-ledger-20261031.md](stage-a-ledger-20261031.md) | 研究记录 | 八条真实任务台账、「人时全仓从未记录」的缺口证明、外部一线流程核查、环节排序与场景选择 |
| [stage-b-runbook-20261101.md](stage-b-runbook-20261101.md) | 执行前冻结计划 | W3 产出：T2.5 预算与达标定义、T3.3 控制变量、T3.4 绕过与额外成本字段、T3.5 七步 runbook、T3.6 四代协议身份与 evaluator 摘要 |

## 命名说明

台账原名 `80-阶段A真实任务台账与基线.md`，属于 `docs/research/` 的编号系列。搬进本目录后系列号失去意义，
且[研究索引守卫](../../tests/hygiene/test_repo_hygiene.py)要求 `NN-*` 引用必须能在 `docs/research/` 解析——
保留前缀会让守卫变红，因此改名为 `stage-a-ledger-20261031.md`，并从 `docs/research/README.md` 的索引里摘掉那一行。

## 读数与证据在哪里

机器读数与工件不在仓库里（`output/` 是指向仓外证据树的符号链接，布局守卫要求它只能是符号链接）：

```sh
ls output/plugin_completion/20261031-phase1-identity-v1/
#   artifacts/            0.6.3 wheel + 0.1.12 插件包
#   artifacts-SHA256.txt  逐文件哈希
#   full-regression.log / full-regression-with-ledger.log   两次全量留档
#   manifest.json         命令、解释器、对照、守卫、量具缺口
```

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

## 新增文件时的两条规矩

1. 只放**属于一期**的文档；一期的验收结论仍进 `docs/status/`（按日期命名），调研与核查进 `docs/research/`（按编号命名）。
2. `tests/hygiene/test_repo_layout.py` 给本目录设了条目预算（8 条）。超预算说明这里已经在变成堆放处，应先收敛而不是提预算。
