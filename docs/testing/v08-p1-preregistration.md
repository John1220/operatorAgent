# P1 预登记：A/C 交付核的性能 parity（不是增益，不是打库）

状态：**已冻结**（2026-11-14）。计划 `operator-v08-workflow-p1-parity-v1`，
hash `529b499ea1c655937986775e4a0cf62454b4dcbee129a1d2057d2f5acabeef4f`，
冻结件 `output/v08-p1-20261114/preregister-p1-parity-v1.json`（仓外证据树）。
量具（计划校验器 + 判读函数）已落地并单测；**真机 L3 取数待机时窗口**，本文不发明任何读数。

## 1. 这轮只问一个问题

v0.8 A/C 复现建立了弱宿主（27B/xhigh）的**交付数**增益，但**性能零证据**——独立终评止于 L2，
每个交付配对 `validity=None`。P1 对**两臂都交付**的配对重跑独立 L3，只问：

> 插件臂（C）交付的核，在同一冻结契约下，是否比裸臂（A）交付的核**更慢**（超出噪声带）？

这是 **parity（不劣于）** 问题，**不是**「C 更快」，**不是**「打赢 cuBLAS/SDPA」。

## 2. 一处对 earlier 路线图的更正（重要）

之前说「softmax/tall-K 窗口探针」不准确。从 v6 两轮 cells 核出：

- **softmax-16384x6144-fp16：两臂从未都交付**（A 两轮都失败、C 两轮都交付）⇒ **没有 softmax 配对**，无从测 parity。
- **tall-K（matmul-2048x16384x2048）不在 v6 复现任务里**（那是更早一轮的任务）。
- 实际可测配对**只有 3 个**：matmul-4096-fp32 两轮 + attention-prefill-4096 一轮。

## 3. 登记的配对（哈希绑定，不用绝对路径）

| pair_id | 任务 | 轮 | 契约哈希 | A 源码 sha256 | C 源码 sha256 |
|---|---|---|---|---|---|
| p1-matmul-r1 | matmul-4096-fp32 | v6r1 | `f6062c86…` | `cadea73a…` | `6aea22b3…` |
| p1-matmul-r2 | matmul-4096-fp32 | v6r2 | `f6062c86…` | `f45a5b72…` | `b9720915…` |
| p1-attention-r2 | attention-prefill-4096 | v6r2 | `e9f75d60…` | `8f84a010…` | `abb98d4d…` |

源码在仓外证据树，测量时按 `(round, task, arm)` 解析并**重算哈希核对**；契约哈希由声明重冻结核对
（校验器已验证两个声明逐字 reproduces 上表契约）。哈希不符 ⇒ 拒配（接线问题，不是候选问题）。

## 4. 基线纪律（与方向1 定案一致）

两个契约都用**库基线**：matmul = cuBLAS IEEE fp32，attention = SDPA fp16。

- **parity 与基线无关**：两臂同契约同基线，比的是 C 与 A 的相对快慢。
- **vs 库的绝对加速比**（历史 matmul ~0.37×、attention 0.8608×，预期 < 1）只作**上下文**记录，
  **绝不**作为插件性能主张。
- 方向1 的「**赢 eager 组合**」正主张需要**新的 eager 基线任务**，明确**不在本轮**（见
  [任务族选型定案](../design/v08/task-family-selection.md) §6）。本轮不登记任何 vs-eager 读数。

## 5. 探针（逐任务，窗口必须重跑）

沿用 R2 已标定的两条通道（同形状/dtype/mask）：

| 任务 | 通道 | 历史带宽（仅参考） |
|---|---|---|
| matmul-4096-fp32 | `matmul-4096x4096x4096-fp32-cuda-event` | 0.1%（2026-11-07） |
| attention-prefill-4096 | `attention-1x16x4096x128-fp16-nocausal-cuda-event` | 0.1%（2026-11-08） |

**P1 窗口必须重跑探针**，`action == "start"` 才开测；历史带宽只作参考，不许继承为本窗读数。
探针命令见冻结件 `probes`（`--dtype`/`--op`/`--shape`/`--causal` 与任务逐字一致，校验器强制）。

## 6. 有效性门与判读规则（已落地为可执行量具）

一个配对**可判**当且仅当：两臂都在绑定源码哈希上过完整 L2、两臂 `measurement_validity == "valid"`、
重冻结契约 == 登记契约、各臂独立终评（`purpose=final`, `stop_after=L3`）。

判读（`p1_plan.judge_pair` / `judge_series`）：

- 单对：`diff_pct = (C_median − A_median) / A_median × 100`；`> band` ⇒ `C_worse`，
  `< −band` ⇒ `C_faster`，其间 ⇒ `parity_within_band`；任一臂非 valid ⇒ `not_decidable`。
- 整轮：`parity_holds`（`formal_parity_claim=true`）当且仅当 **≥1 对可判 且 无可判对为 C_worse**；
  有 C_worse ⇒ `C_worse_observed`；无可判对 ⇒ `unknown`（这是关于窗口的结论，不是候选）。

## 7. 停止规则

- 探针 `action != "start"` ⇒ hold，不开测、不耗预算。
- 契约/源码哈希不符 ⇒ 拒配。
- **连续两对无 valid L3 ⇒ 停**，整轮记 unknown，不追加第三对。
- 预算冻结，中途不许抬高去凑阈值。

## 8. 预算（冻结）

`pairs=3`、`max_model_calls=0`（**不生成**，只重测已有交付）、`max_verifications=6`（每臂每对一次独立 L3）、
`wall_clock_s=2400`、`gpu_minutes=60`。

## 9. 边界（不能外推）

- 单卡（H20）、单模型族（27B/xhigh）；**n=3 对、2 任务，无显著性**。
- 候选是弱宿主**一次交付**的核，**不是优化核**：本轮测交付性能，**不测 optimize 闭环**（那是 P2）。
- 库基线；vs 库绝对值只作上下文；vs-eager 正主张不在本轮。
- 无开发人时主张；除这两任务上的 parity 外无任何性能主张。
- 对外表述禁列表仍有效：不写「性能全面达标」「显著优于 DSH」，不把单形状历史读数归因于插件。

## 10. 复现

```bash
.venv/bin/python -c "from benchmarks.experiments.v08 import p1_plan; \
  print(p1_plan.validate(p1_plan.design()))"          # 期望 []
.venv/bin/python -m pytest tests/unit/v08/test_p1_plan.py -q   # 11 项
```

冻结件与判读量具：`benchmarks/experiments/v08/p1_plan.py`；
证据：`output/v08-p1-20261114/preregister-p1-parity-v1.json`。