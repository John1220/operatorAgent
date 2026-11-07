# v0.8 进度入口

2026-11-07 · 本文件是 v0.8 的**单一入口**：做到哪、下一步、卡在哪、证据在哪。
细节在各专题文档里（§5 索引），本文件不重复它们的论证。

## 1. 一句话现状

v0.8 的"优化增强"能力（把性能事实与确定性诊断交给宿主）**已落地并用真机验证**；
attention 族的可判定化走到 C4/5——**C5 因受信任参考的执行身份与远端环境不一致而停下**，
需要你在两种修法里选一个（§4 第 1 条）。

## 2. 已完成

| 交付 | 提交 | 验收证据 |
|---|---|---|
| host 侧性能事实有界投影（绑定轴/地板/噪声档/用途边界） | `d6ec795` | `tests/unit/test_host_context.py`（16 项） |
| optimize 阶段确定性诊断（5 状态、轴过滤、地板串行上限、余量三口径） | `bcbbe75` `34f1177` `097e999` | 真机回执驱动的 3 次修正，见 [真机回执](v08/real-receipt-20261107.md) |
| 噪声 preflight（`--shape/--op/--dtype/--repeat`，CUDA event 通道） | `30eca6d` `29264d4` `097e999` `409ef2a` | [preflight 文档](../testing/v08-noise-preflight.md)；4096³ 带宽 0.20%、1024³ 3.16% |
| 两臂模型计量对账（两个 verdict、四条 blocker） | `7b06f99` | [对账文档](../testing/v08-arm-metering.md) |
| R2 预登记 + 校验器（fp16 / fp32 两份计划已冻结） | `3681218` `6ba557b` | hash `58bde16da8fbb252` / `8d460f9454ad2531` |
| R2 第一对（管道跑通，增量未确立） | `afc5c78` | [第一对记录](v08/r2-pair-1-20261107.md) |
| 候选任务余量实测（5 个候选） | `409ef2a` | [余量表](v08/candidate-headroom-20261107.md) |
| attention 族 C1–C4（布局/标量、参考+容差、套件+反作弊、roofline） | `1ea7448` `c213be3` `7673937` `4ea75dd` `a8b9cd0` | [attention 设计 §13–16](../design/v08/attention-family.md) |
| 模型路线（qwen3.8-2.4t-a95b 可用但强制思考） | `6397e82` | [model-switch](v08/model-switch-qwen-20261107.md) |

测试基线：`tests/unit` + `tests/hygiene` **6320 passed / 0 failed / 1 skipped**（2026-11-07 21:05）。

## 3. 进行中 / 阻塞

| 项 | 状态 | 说明 |
|---|---|---|
| **C5 attention 端到端** | **阻塞** | 核已写好、静态审计通过，但 L2 在"执行身份"处 fail closed：pin 的 torch 2.14.0 ≠ runner `.venv` 2.14.1 ≠ 系统 2.13.0。详见 [C5 卡点](v08/c5-identity-blocker-20261107.md) |
| R2 第二对 | 未开始 | 任务已选（fp32 4096³，余量 66%），但串味问题要求原生 profile 路径或子代理隔离 |
| `sampling_comparability` | 未成立 | A 臂无同等硬门；账本不记 temperature/n/enable_thinking |

**这条阻塞的影响面比 attention 大**：任何受信任参考任务都会在 L2 前被拦，包括 layernorm v2。

## 4. 待你决定

1. **受信任身份 vs 环境**（阻塞 C5）
   - **A（建议）**：部署时 `torch==<attested>`（版本从 `PINNED_RUNTIME` 取），两份 profile 继续有效；
   - B：在 2.14.1 上重做隔离审计并给两份 profile 升版。
2. **契约 `12 → 13`**：是否把 `preflight` / `usage-reconcile` 暴露成 host 工具。
   我建议 R2 期间不做（会让 C 臂多两个工具、污染"两臂只差 optimize"的立论）。
3. **qwen 用在哪个系列**：建议 R3（测"模型变弱时诊断增益是否变大"），不进 R2。
4. **R2 第二对的驱动方式**：原生 profile 会话（干净但需切一次）或两个子代理（隔离，但预算靠记账）。

## 5. 文档索引

**设计**：[v0.8 准备提案](../design/v08-preparation.md)（原文照录）·
[attention Scope A 设计](../design/v08/attention-family.md)（含 C1–C4 实测）

**测试/操作**：[noise preflight](../testing/v08-noise-preflight.md) ·
[两臂计量对账](../testing/v08-arm-metering.md) ·
[R2 预登记](v08-r2-preregistration.md) · [R2 跑法](../testing/v08-r2-runbook.md)

**状态/证据**：[真机回执](v08/real-receipt-20261107.md) ·
[候选余量](v08/candidate-headroom-20261107.md) · [R2 第一对](v08/r2-pair-1-20261107.md) ·
[模型切换](v08/model-switch-qwen-20261107.md) · [C5 卡点](v08/c5-identity-blocker-20261107.md)

## 6. 仓外证据（不在工作树内）

| 目录 | 内容 |
|---|---|
| `output/v08-real-receipt-20261107/` | 三份真机回执（matmul valid / relu degraded / layernorm unknown）+ 三份 preflight |
| `output/v08-candidates-20261107/` | 5 个候选的 L3 与 `gate-fp32.json` |
| `output/v08-r2-20261107/` | 两份冻结计划、pair-1 的 gate/两臂 run/`pair.json` |
| `output/v08-attention-20261107/` | C5 的两次 verify、attention 核、指纹探针 |

关键哈希：layernorm 参考 `8f779954d3d2b3af…`、attention 参考 `145c6cce7c6a3192…`、
R2 计划 `58bde16da8fbb252…` / `8d460f9454ad2531…`。

## 7. 只有提交信息、没进文档的细节（如实说明）

- 每次改动**为什么**改（例如"地板不能冒充竞争轴"那条推导）只在提交信息里；
  专题文档给出结论，不给完整推演。
- 三处"标量不是数组"的既有缺口分布在 `hitl_tools.prepare`、`remote_scripts/run_kernel.py`、
  `core/contract.py`，结论写在 [attention 设计 §16](../design/v08/attention-family.md)，
  逐个的现场报错文本只在提交信息里。
- 本文件之前，v0.8 **没有入口文档**（v0.7.x 有 `v072-todo.md`）—— 这就是你现在看的这份。