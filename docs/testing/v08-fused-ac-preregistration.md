# 融合任务 A/C 预登记草案：插件工作流能否在同等预算内交付**达标**的 fused 核

**2026-11-17 起草 · 2026-11-17 评审后修订（v2）。** 头寸门（[融合头寸预登记](v08-fused-headroom-preregistration.md)）
已通过：手写 fused 核比 eager 两 kernel 快 55.14%。本轮问的是**工作流**：
同样的任务、同样的预算，裸 DSH（A）与 DSH+插件（C）各自能不能**自己交付**一个
**既正确、又真正融合、且达到冻结性能门槛**的核。

本文件不预设结论。用户已确认模型档位/对数/墙钟/A 臂套件（§11）；评审提出的四处修订已并入
§3（达标定义）、§8（终稿选择与终评预算）、§7（预算上限）、§9/§10（判读与失败规则）。
冻结后不许原位改预算或判据。

## 1. 研究问题

> 在同一个 `residual add + LayerNorm` 任务上，A/C 两臂从**空起点**出发、使用**相同模型/采样/预算/
> 独立终评**，C 臂（插件 + 其文档化工作流）交付的**达标核数量**，是否比 A 臂（裸 DSH + 等价事实）更多？

主指标是**达标交付数**；「仅通过 L2」只作次级观测。调用数、墙钟是成本观测，不授正式信用。

## 2. 与已有轮次的关系（别混轴）

| 轮 | 问什么 | 结论 |
|---|---|---|
| v6 A/C | 弱宿主上交付数（仅 L2） | `reproduced`（每轮 +1，插件臂 3–9 次调用） |
| P1 | 一次交付核的性能 parity | `C_worse_observed` |
| P2 | optimize 诊断提示的增量 | `no_increment_observed` |
| D2 | 参数层/结构层余量 | `no_increment` / `structure_hurts` |
| 融合头寸 | 融合本身有没有提速空间 | `headroom_confirmed`（+55.14%，手写核） |
| **本轮** | **谁能交付达标（正确+真融合+够快）的核** | **待测** |

头寸门只证明“任务值得做”；它**不**证明插件能交付这样的核，也不证明 A 交付不了。

## 3. 任务与判据

- **任务**：`fp16 32768x6144 residual add layernorm`，`Y = LayerNorm(X + R)`，末轴、总体方差、
  `eps=1e-5`、无 affine，X/R/Y 同形 fp16。
- **冻结契约**：`093339587d8c462c…`（`min_speedup=1.5`）；profile
  `builtin-fused-layernorm-residual-fp16-cuda-v1`；atol `5e-3`、rtol `1e-2`、rho `1.0`；
  参考 digest `c9aea6ac…`；套件 digest `652f65b1…`（32 case）。
- **达标交付（主指标，全部满足）**：
  1. 最终源码在 supervisor 独立终评里通过**完整 L2**（32 case + 主用例）；
  2. 同一次终评的 L3 `measurement_validity=valid`；
  3. L3 `speedup_vs_baseline ≥ 1.5`（冻结在契约 `min_speedup`，基线 `pytorch_eager`）；
  4. **融合静态核验通过**：源码只有**一个** `@triton.jit` 入口核，且 timed wrapper `kernel()`
     只发起**一次**该核 launch；禁止 `add` 与 `LayerNorm` 两个 jit 核串联充数。
     静态核验记录 `fusion_static=passed/failed + reason`；动态证据以第 3 条性能门槛为准
     （两核串联的实测通常 ≈1.0×，过不了 1.5×）。
- **仅正确交付（次级，不授正式信用）**：满足第 1 条、但 L3 未达 1.5× 或未融合。
- **到独立终评达标的时间**：从该格第一条模型请求时间戳到**终评达标回执**的时间；不达标记 `—`。
- **性能门槛的由来**：头寸手写核 2.229×，eager 1.0×，两 kernel 串联 ≈1.0×；1.5× 是
  “消掉至少一整遍读写”的保守行为门槛，不是手写核成绩的复刻。

## 4. 两臂

| 臂 | 工具面 | 拿到规格的途径 | 验证 |
|---|---|---|---|
| **A（裸 DSH）** | 普通 shell / 读写 / 编辑 + `check.py`（stdlib 客户端，提交走 supervisor） | context pack：任务声明、X/R/Y ABI、参考语义、容差、32 条 case 布局/值模式、规则、预算 | 同一 supervisor 契约；**8 次总预算**（6 次开发 + 2 次终评） |
| **C（DSH + 插件）** | 普通工具关掉，只有插件 MCP 工具（`operator_context` / `operator_verify` / …） | `operator_context` 一次性发布，与 A 的 context pack 在**任务事实**上逐字等价 | `operator_verify` 同一冻结契约；**8 次总预算**（6 次开发 + 2 次终评） |

两臂共用同一个 supervisor 判据、同一份冻结契约、同一张 H20、同一次独立终评。
C 的待遇是“插件 + 其文档化工作流”（v6 已定：工作流本身属于 treatment）。

## 5. 公平控制（继承 v6 + 本轮修订）

1. **Repo-blind**：两臂工作区都看不到本仓、历史回执、计划与插件实现。
2. **ABI 等价**：A 的 context pack 与 C 的 `operator_context` 在任务事实上逐字等价；差别只能是
   C 多出的工具与工作流。
3. **采样六字段逐字一致**：`model / temperature / n / max_tokens / enable_thinking /
   reasoning_effort`；从 `cell/outer-requests/*.json` 核对。
4. **预算与终评一致**：调用、预留、墙钟、验证次数、GPU UUID 完全相同；两臂终稿**选择规则相同**（§8）。
5. **空起点**：不提供融合核、不提供 seed 源码。
6. **同一冻结件驱动执行器**：预算、采样、模型端点哈希、GPU index/UUID/name 都从计划读取；
   实际回执与冻结件不符 ⇒ 该格作废（`unknown`），不许按候选列表自动换卡。
7. **仪器失败 ≠ 候选失败**：只有**有证据的设施故障**（钉卡失败、被抢卡 OOM、沙箱/网关故障、
   远端断连）记 `unknown`；**正常用完 2400 s 仍未达标记 `not_delivered`（失败）**。
8. **v6 七个仪器缺陷复核**：输出上限（**线上 wire `max_tokens=32768`，40960 是账本预留**）、
   流式截止按实测吞吐、MCP 超时覆盖分钟级 L2、两臂采样一致、网关不拒 40960、A 隔离、C 工具面。

## 6. 模型与采样（已确认）

- `Qwen3.8-27B`（本地 vLLM），`reasoning_effort=xhigh`，`enable_thinking=true`，
  `max_tokens=32768`（wire），单次预留 `40960`（账本），`n=1`，`temperature=0.1`。
- **冻结前 smoke（不计入正式格）**：用当前 Coder 全量提示（11 配方）连跑 2 次，确认
  `finish=stop`、核源码完整发出；若撞 wire 32768，**先停下**决定换 medium 还是收窄提示，
  并把决定写进冻结件——不许在正式格里换档。

## 7. 预算（已确认 + 评审修订）

| 项 | 值 |
|---|---|
| 每臂每格模型调用 | 20 次 |
| 单次 wire `max_tokens` / 账本预留 | 32768 / 40960 |
| 每格验证派发 | **8 次总预算**；终评 run 占 2 次（1 次 explore L2 + 1 次 `purpose=final`），其余 6 次给开发（host 的 `final_reserve` 仍为 1，2 次是计数口径） |
| 墙钟 | **2400 s/格，含终评** |
| 正式格 | 2 对 = 4 格；**模型调用上限 4×20 = 80 次** |
| 重跑额度 | 只给有证据的 `unknown` 格，最多 2 格（+40 次调用、+2×2400 s）；总上限 **120 次 / 6 格墙钟** |
| 测量卡 | GPU7，`GPU-52ca6155-d8bc-a416-847a-6f0c71db1552`（从冻结件读取，不自动换卡） |
| 起点 | 空（无 seed 核） |

## 8. 终稿选择与终评（评审新增，两臂统一）

1. **选择规则**：终稿 = 该格预算内**最后一次实际执行且通过完整 L2** 的提交。
   被预算拒绝、超墙钟、未实际运行的提交**不得**更新终稿；A 臂当前在 `develop()` 里先写
   `selected_source` 再判预算，必须改成“实际验证通过后才更新”。
2. **收尾**：两臂都用这个终稿新开**一个** `final-run`：先 explore L2、再 `purpose=final`、
   `stop_after=L3`；两次派发计入 §7 的 8 次与 2400 s。
3. **不重试终评**（设施 `unknown` 除外，走重跑额度）；终评 L2 失败或 L3 非 `valid` ⇒ 不达标。
4. **计数**：`_counts` 必须把 `final-run/verify-*.json` 计入验证数与墙钟；两臂用同一套计数。
5. 终评源码哈希必须等于终稿哈希；不等 ⇒ 该格作废。

## 9. 判读

- **单对**：
  - C 达标、A 不达标 ⇒ `+1`；
  - A 达标、C 不达标 ⇒ `−1`；
  - 两臂同结果 ⇒ `0`，并记录子型：`both_qualified` / `both_correct_only` / `both_failed`（可混合）。
  - 次级同时记录：仅正确交付、调用数、到首次完整 L2 与到终评达标的时间。
- **整轮**：
  - 两对均 `+1` ⇒ **`reproduced`**，`formal_gain_claim=true`；
  - 两对均 `0` ⇒ **`delivery_parity`**（附两对子型）；
  - 一对 `+1`、一对 `0` ⇒ **`not_established`**（`formal_gain_claim=false`，报 partial 信号）；
  - 任一对 `−1` ⇒ **`contrary`**；
  - 重跑额度用完仍不可判 ⇒ **`unknown`**。
- **禁止**：把“调用数更少”写成“开发更快/人时更少”；把结论外推到其它任务族/强模型；
  把仅正确交付说成达标交付。

## 10. 失败、重跑与停止

- **失败**：正常耗尽 20 次调用或 2400 s 仍未达标 ⇒ `not_delivered`（计入分母），不是 `unknown`。
- **unknown**：仅有证据的设施故障；保留失败/未知回执，不删证据。
- **重跑**：只对 `unknown` 格，最多 2 格；新工作区、同冻结件；重跑格与原格的证据都保留；
  重跑仍 `unknown` ⇒ 整轮 `unknown`，不追加第三对。
- **停止**：预算耗尽即停；不中途改判据/容差/case；契约或模型端点哈希漂移即停。
- smoke 未过（§6）⇒ 不冻结、不开正式格。

## 11. 决策状态（2026-11-17 用户已定）

| 项 | 定值 |
|---|---|
| 模型档位 | `Qwen3.8-27B / xhigh`（冻结前须过 §6 smoke） |
| 正式对数 | 2 对 |
| 墙钟 | 2400 s/格（含终评） |
| A 臂套件清单 | **给**（与 C 的 `operator_context` 等价） |
| 达标门槛 | **L2 32/32 + L3 `valid` + speedup ≥ 1.5 + 单核融合静态核验**（评审后新增） |
| 终评预算 | 终评 run 占 8 次中的 2 次（explore L2 + 预留 final），计入 2400 s |

## 12. 执行前置（A/C harness 待接线，冻结前完成）

1. `ac_plan` 的 `_supported_kernel_types()` 加入已注册融合族（现为 `OPERATORS | {'attention'}`）；
   任务表加入本任务（声明、契约 `09333958…`、32 case、`min_speedup=1.5`、fusion 检查项）。
2. **统一终稿选择**：修 `phase1/native.py::develop`（A）——实际验证通过完整 L2 后才更新终稿；
   C 已按“最近一次 L2 通过”选择，核对两臂回执字段一致。
3. **终评**：两臂 `_finish_cell` 统一 `stop_after=L3`；`_counts` 纳入 `final-run` 派遣与墙钟；
   预算按 6+2 派发。
4. **执行器读冻结件**：GPU 由计划 UUID 钉死（去掉 `select_gpu` 候选自动选择）；预算/采样/端点哈希
   运行时核对；回执记录实际 GPU UUID、采样六字段。
5. **融合静态核验**：A/C 收尾加一个源码 AST 检查（单一 `@triton.jit` + `kernel()` 单次 launch），
   与性能门槛一起决定“达标”；检查器仅用于本轮，不改核心 verifier。
6. 仪表复核：§5 第 8 条逐项；`_counts` 与墙钟口径；终评证据独立性。
7. 先做**不计入正式格**的 fixture 试跑（脚本模型）走通 A/C 终稿、终评、计数与回执；
   再做 §6 的 xhigh smoke。全过后冻结计划编号/hash，才开正式格。

**接线进度（2026-11-17）**

| 项 | 状态 |
|---|---|
| `ac_plan` v7：融合任务、`_supported_kernel_types`、execution（GPU/采样/wire 32768 / 预留 40960/2 对）、20 次/8 验证（6+2）/2400 s 预算 | 已接线 + 测试 |
| 终稿规则：A 臂改为 `last_full_l2_submission`（实际执行且 L2 通过）；C 臂保持最近 L2 通过 | 已接线 + 测试 |
| 终评：两臂 `stop_after=L3`；`_counts` 纳入 `final-run` 两次派发与总墙钟；新增 `qualified/delivery_class/fusion_static/final_speedup/final_validity` | 已接线 + 测试 |
| GPU 钉卡：计划 `execution.gpu` 的 UUID 与实际回执核对，不符即中止；去掉自动选卡 | 已接线 + 测试 |
| 融合静态核验：`fusion_check.py`（单一 `@triton.jit` + 单一 launch）接入收尾 | 已接线 + 测试 |
| A 臂 context pack 加 32 条 case 清单（与 C 的 `operator_context` 等价） | **已接线**：两臂 prompt 都带同一段 `correctness_cases`（32 条 case_id/形状/dtype/pattern/layout） |
| v7 聚合器按 `qualified` 判读（+1/0/−1、partial、子型） | **已接线**：[fused_ac_aggregate.py](../../benchmarks/experiments/v08/fused_ac_aggregate.py)，8 项测试 |
| 总墙钟 >2400 s 的格判 `not_delivered` | **已接线**：`_finish_cell` 的 `over_wall` 直接降为 failed，不记 unknown |
| fixture 试跑（A 臂） | **已跑通**：scripted 模型 → `check.py` → supervisor → 终评；L2 32/32、L3 `valid`、2.2392×、`qualified`、验证 3 次 |
| fixture 试跑（C 臂） | **已跑通**：scripted 模型走插件工具；L2 32/32、L3 `valid`、2.2352×、`qualified`、验证 3 次 |
| 两格进 v7 聚合器 | **已跑通**：`pair_verdict` → `both_qualified`、配对差 0（fixture 仅供接线，不作成绩） |
| xhigh smoke（内部 Coder 路径） | **不通过（2/2）**：真实 Coder prompt + wire 32768 ⇒ 两次都 `finish=length`、completion=32768 全部写进思考、核未发出（702 s / 697 s）。这说明 P2 起点核那条**单次整核生成**路径在 xhigh 下不可行；它**不是** A/C 臂的路径（A/C 两臂都是 DSH agent 用工具自己写） |
| xhigh smoke（真实 A 臂） | **不通过**：一格真实 A 臂，首次模型调用 `outputTokens=32768`、`turn_end: max-tokens`、**零次工具调用、零次提交**。xhigh 在 wire 32768 下结构性不可用（两条路径都证实：内部 Coder 2/2、DSH agent 1/1） |
| 冻结实际计划文件 | 待 smoke 后执行 |

**fixture 暴露并修掉的四个真缺陷（都会直接影响正式格）**

1. **终评 UTC 回收**（下述）。
2. **A 臂会话提交根本没进判据**：`VerificationService` 缺 `deadline_monotonic` ⇒ `develop()`
   `KeyError` ⇒ 服务回 400 ⇒ `check.py` 只报 `development_transport_failed`。此前 A 臂
   "完成"只是因为终评直接用了落盘的 candidate 文件。
3. **DSH 的 `workspace-write` 沙箱在本机不可用** ⇒ bash 工具被拒；fixture 改用
   `danger-full-access`（`test_only`，不是公平性声明）。
4. **MCP 工具 60 s 超时会掐断融合 L2**（v6 第 5 号缺陷翻版）；按格墙钟放宽
   `toolCallTimeoutMs` 与 bash 超时后 C 臂才通。
5. **A 臂派发被数两次**：一次会话提交同时留下 `native-verification/result-N.json` 与
   `run/verify-N.json`；按臂取其一后两臂计数对称（各 3 次）。

**终评 UTC 回收（会影响正式格）**：终评的远端持久 worker 不回收，会留在同卡被
下一段质量检查记成 `harness peer`，L3 判 `degraded` ⇒ 达标门永远过不去。现在 `final_evaluation`
的每次派发各自包一层 `owned_worker_scope`（独立目录，避免 write-once 审计文件相撞），
A 臂 fixture 已复现修复前后的差别：修复前 `degraded/correct_only`，修复后 `valid/qualified`。

## 13. 冻结身份（冻结时填）

- 计划编号/hash、任务契约 `093339587d8c462c…`、`min_speedup=1.5`、profile/atol/参考与套件 digest、
  模型端点哈希、采样六字段、A/C context pack 与提示哈希、融合静态检查器版本、
  两臂工具面清单、预算（含重跑额度）、GPU UUID、supervisor 独立终评规则（`stop_after=L3`）。

## 14. 边界

- 结论限：这一条融合任务、32768×6144 fp16、弱宿主 27B/xhigh、登记的 H20/GPU7、2 对正式格。
- 达标交付含性能与融合要求；仅正确交付只作次级观测，不构成性能主张。
- 头寸轮的 55.14% 是手写核读数，不能当作任一臂的交付性能。
