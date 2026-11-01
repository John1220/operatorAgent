# 第一期文档目录

一期（阶段 A 需求与基线 + 交付身份收口）的执行文档集中在这里。**目录按交付期分组，不改变文档类型职责**：
清单是计划、两份记录分别是验收类与研究类快照，职责仍按[路线 §9](../design/operator-development-product-roadmap.md) 划分。

| 文档 | 类型 | 内容 |
|---|---|---|
| [operator-development-phase1-execution-20261031.md](operator-development-phase1-execution-20261031.md) | 一期计划 | 可勾选执行清单（W1 交付身份 / W2 阶段 A / W3 B 计划冻结）、出口判据与现状、本期不做项 |
| [delivery-identity-rebind-20261031.md](delivery-identity-rebind-20261031.md) | 事实记录 | 平铺 → 五层造成的协议身份断链（`78993c98…` → `e8c7f2c6…`）、失效清单、重建后的 `0.6.3` 工件与身份核对 |
| [stage-a-ledger-20261031.md](stage-a-ledger-20261031.md) | 研究记录 | 八条真实任务台账、「人时全仓从未记录」的缺口证明、外部一线流程核查、环节排序与场景选择 |

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

## 新增文件时的两条规矩

1. 只放**属于一期**的文档；一期的验收结论仍进 `docs/status/`（按日期命名），调研与核查进 `docs/research/`（按编号命名）。
2. `tests/hygiene/test_repo_layout.py` 给本目录设了条目预算（8 条）。超预算说明这里已经在变成堆放处，应先收敛而不是提预算。
