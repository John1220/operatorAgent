# v0.8.1 优化分支进度入口

更新于 2026-11-14 · 分支 `codex/v0.8.1-optimize`（当日由 `codex/v0.9-optimize` 改名）。

> **为什么是 v0.8.1 而不是 v0.9**：产品路线里的 **v0.9 是「业务接入版」**（把已验证核接进真实
> 框架/模型、端到端回归，见 [产品路线](../../design/operator-development-product-roadmap.md) §5.3），
> 尚未开始。本分支做的是判定缓存、异步验证、任务族选型与性能 parity 测量，是 **v0.8 优化工作的
> 延续**，故按 minor 递增记 v0.8.1，不占用路线的 v0.9 语义。基线分支 `codex/v0.8`。

## 1. 当前状态

两条主线各有结论：

- **验证链路增强（已提交 `9fb3ce7`）**：判定缓存与异步验证落地，P0 计量漏计修复 + 文档勘误，
  方向1 任务族选型定案。
- **P1 性能 parity（真机已跑完，量具批次未提交）**：对 v6 两臂都交付的 3 个配对重跑独立 L3，
  判 **`C_worse_observed`、`formal_parity_claim=false`**——三对里插件臂（C）交付的核都比裸臂（A）
  慢。**性能 parity 主张被否证**。这与 v6 的「交付数增益 reproduced」分属两轴，互不推翻（见 §3）。

## 2. 已完成

| 交付 | 提交或状态 | 验收证据 |
|---|---|---|
| **判定缓存**（五元组键：契约×源码×stop_after×环境身份×实现；只回放决定性 L1/L2，不回放 L3/unknown/final；环境身份探测失败 fail-closed；命中也按次扣预算） | `9fb3ce7` | `operator_agent/decide/verdict_cache.py`；`tests/unit/v08/test_verdict_cache.py`（13 项） |
| **异步验证**（`operator_verify` 增 `wait=false` → 脱离提交返回 `invocation_id`，`operator_status` 带终态 `result`，`operator_cancel` 不变） | `9fb3ce7` | `operator_agent/app/mcp_server.py`；`tests/unit/test_mcp_server.py::TestAsyncVerify`（3 项） |
| **P0 计量漏计修复**（A/C 两条早退路径都过 `_counts`，不再漏成 None） | `9fb3ce7` | `tests/unit/v08/test_ac_round.py::test_failed_cell_early_return_still_meters_dispatches`（变异验证） |
| **P0 文档勘误**（报告 §7 两级正式信用 + n=1 措辞、README §3、`ac_aggregate` docstring；冻结回执不改，round-1 哈希复算 intact） | `9fb3ce7` | `docs/status/v08/ac-workflow-round-report-20261110.md` |
| **方向1 任务族选型定案**（性能基线认 eager 组合，不认厂商库；只有库基线的任务只进交付数对照） | `9fb3ce7` | [任务族选型定案](../../design/v08/task-family-selection.md) |
| **P1 预登记 + 判读量具 + 测量执行器** | 工作树，**未提交** | `benchmarks/experiments/v08/p1_plan.py`（judge_pair/judge_series/validate/freeze）、`p1_run.py`；`tests/unit/v08/test_p1_plan.py`（11）+ `test_p1_run.py`（8）；[P1 预登记](../../testing/v08-p1-preregistration.md) |
| **P1 真机取数 + 判读** | 已归档（仓外证据树） | `output/v08-p1-20261114/run3/series.json` + 三对 `pair.json`；计划哈希 `529b499e…` |

协议身份（`protocol_identity`）随判定缓存加入扩到 **77 文件**（`tests/unit/test_input_seed_spec.py` 钉住）；
旧证据不会命中新缓存（实现身份已变）。

## 3. P1 性能 parity 判读（核心结果）

**问题**（预登记写死）：同一冻结契约下，C 交付的核是否比 A 交付的核**更慢超噪声带**？
是 parity（不劣于），**不是**「C 更快」，**不是**打赢 cuBLAS/SDPA。

三对全部可判（两臂都过完整 L2、`measurement_validity=valid`、绑定各自提交源码哈希、独立终评
`purpose=final`/`stop_after=L3`）：

| 配对 | 任务 | A median | C median | 配对差 (C−A)/A | A vs库 | C vs库 | 判 |
|---|---|---|---|---|---|---|---|
| p1-matmul-r1 | matmul-4096-fp32 | 5.725 ms | 7.150 ms | **+24.89%** | 0.79× | 0.63× | C_worse |
| p1-matmul-r2 | matmul-4096-fp32 | 5.085 ms | 5.179 ms | **+1.85%** | 0.89× | 0.87× | C_worse |
| p1-attention-r2 | attention-prefill-4096 | 1.186 ms | 1.638 ms | **+38.13%** | 0.89× | 0.64× | C_worse |

系列终判：**`C_worse_observed`**，`decidable_pairs=3`、`worse_pairs=3`、`formal_parity_claim=false`。

**口径（不许临场改）**
- 主指标是 **C vs A 的配对差**；「vs库」两列（cuBLAS IEEE fp32 / SDPA fp16）只作上下文，两臂都慢于库，
  与 parity 是两条轴，别混。
- 噪声带 0.1%（门 `action=start` 放行）。matmul-r2 的 +1.85% 幅度小，但超带即判 C_worse，规则写死。
- 范围：**n=3 对、2 任务、单卡（H20 GPU3）、单模型族（v6 弱宿主 27B/xhigh）**，无显著性。
  候选是弱宿主**一次交付**的核，**不是优化核**——本轮测交付性能，不测 optimize 闭环（那是 P2）。
- C 臂 matmul-r1 终评遇 transport-unknown，执行器自动换新 run_dir 重试（attempt 2 拿到决定性读数）；
  那是基础设施抖动，不是候选失败。

**怎么读这个负结果**：插件提供的是验证/反馈/预算闭环，**不抬高核的峰值性能上限**。v6 证明它能让弱宿主
「更少调用交付正确核」（交付数轴）；P1 证明它交付出来的核**并不更快**（性能轴）。两者兼容。
「赢 eager 组合」的正主张需要新的 eager 基线任务，明确不在本轮（方向1 定案 §6）。

## 4. 进行中 / 待办

| 项 | 状态 | 说明 |
|---|---|---|
| **方向3 真机标定** | **已标定（2026-11-14）** | 异步全链路 6/6 过；身份+缓存 8/9 过，捕到一处环境身份缺口（见下） |
| **P1 量具批次提交** | 待用户批准 | `p1_plan.py`/`p1_run.py`/两份测试/预登记文档/README 指针/`test_verdict_cache.py` 迁移 |
| 方向2 autotune 拆分 | 未开始 | 参数层确定性扫描 + LLM 语义重构 |
| P2 修复/诊断因果轮 | 未开始 | 测 optimize 闭环（修复型任务），沿用方向1 选型 |
| P3 中档模型探点 | 未开始 | 测「模型变弱时诊断增益是否变大」 |
| **缓存环境身份补 UUID**（标定捕到的缺口） | 待用户决定 | 见下「方向3 标定读数」；要不要在 runner+executor 两层补 `gpu_uuid/driver/cuda`，把缓存键绑到卡实例 |

### 方向3 标定读数（2026-11-14，H20 GPU3）

量具 `benchmarks/experiments/v08/d3_calibrate.py`，候选复用 P1 绑定哈希的 attention 核
（`abb98d4d…`），不新生成、不调模型。回执 `output/v08-d3-20261114/{identity-cache,async-chain}.json`。

**A. 身份 + 缓存（8/9 过）**：环境身份回读到真实 `torch 2.14.0+cu130 / triton 3.8.0 /
NVIDIA H20 / cc 9.0`；同源码重复 verify 第二次 `cache_hit=True` 且 `cached_from` 指回第一轮、
换源码哈希不命中、预算按派发扣（`reserved` 1→2→3，命中也扣）——**缓存机制全对**。

**唯一 FAIL：`environment_has_gpu_uuid`**。根因两层都没产出这三个字段：
- 远端 runner [run_kernel.py:1808](../../../remote_scripts/run_kernel.py#L1808) `get_gpu_info` 没采 `gpu_uuid/driver_version/cuda_version`；
- 控制端 [remote_executor.py:1397](../../../operator_agent/execute/remote_executor.py#L1397) 重建字典也没透传。

⇒ 缓存设计里的 `_EXTRA_IDENTITY_KEYS=(driver_version, cuda_version, gpu_uuid)` 在 SSH 上永远收不到，
实际环境身份只绑到 **GPU 型号级**（name+device+cc+torch+triton+backend），**没绑到具体卡 UUID**。
缓存只回放决定性 L1/L2（正确性），同型号+同 torch/triton ⇒ 正确性一致，故**功能上正确**；
但设计声称的 UUID 绑定没实现。要绑到卡实例需两层补字段——**未擅自改，留待决定**。

**B. 异步全链路（6/6 过）**：spawn 真实 `operator-agent-mcp` 走 JSON-RPC，分钟级 attention L2：
- `operator_verify(wait=false)` **21 ms** 立即返回 invocation_id（`phase=running`，不阻塞）；
- `operator_status` 轮询 50 次 / **246.9 s** 到终态 `done`，`L1/L2=True`、`ok=True`（后台 worker 真跑出决定性判定）；
- `operator_cancel`  accepted，终态 `cancelled`。

**结论**：异步量具完全可用；缓存量具机制正确但环境身份比设计粗一格（缺 UUID）。标定到此为止，判读与是否补字段留给人。

## 5. 边界与对外表述（继承，仍有效）

- 不写「插件显著优于原生 DSH」「性能全面达标」「提高正确率 X%」「模型调用下降 X%」。
- 不把单形状历史读数（手写核 CLI/core 路径）归因于插件；不把 case 数写成「N 个算子」或「正确率 100%」。
- P1 只能报「这 3 对上 C 慢于 A、parity 不成立」，带 §3 的 scope；不外推到优化核、其它任务族或强模型。
- 交付数增益（v6 reproduced）逐字带 scope：*weak host (Qwen3.8-27B, xhigh) on tasks that one
  generation can deliver*；无性能/人时主张。

## 6. 文档索引

**设计**：[任务族选型定案](../../design/v08/task-family-selection.md) ·
[产品路线（v0.9=业务接入）](../../design/operator-development-product-roadmap.md)

**测试/操作**：[P1 性能 parity 预登记](../../testing/v08-p1-preregistration.md) ·
[noise preflight](../../testing/v08-noise-preflight.md)

**上游进度**：[v0.8 进度入口](../v08/README.md)（A/C 复现 reproduced 的完整过程与回执）·
[A/C 工作流轮报告](../v08/ac-workflow-round-report-20261110.md)

## 7. 仓外证据（不在工作树内）

| 目录 | 内容 |
|---|---|
| `output/v08-p1-20261114/preregister-p1-parity-v1.json` | P1 冻结计划，hash `529b499e…` |
| `output/v08-p1-20261114/run3/series.json` | 系列终判 `C_worse_observed` |
| `output/v08-p1-20261114/run3/p1-{matmul-r1,matmul-r2,attention-r2}/pair.json` | 三对判读 + 两臂源码哈希 |
| `output/v08-p1-20261114/run3/<pair>/<arm>/verify-2/result.json` | 各臂独立终评 L3 回执 |
| `output/v08-d3-20261114/identity-cache.json` | 方向3 身份+缓存标定（8/9，含环境身份缺口） |
| `output/v08-d3-20261114/async-chain.json` | 方向3 异步全链路标定（6/6） |

GPU 身份钉在 GPU3（UUID 见冻结件与 `series.json`）；源码哈希按文件重算与预登记逐字一致。

## 8. 测试现状

**2026-11-14 聚焦读数：73 passed**（`test_p1_plan` + `test_p1_run` + `test_verdict_cache` +
`test_mcp_server` + `test_ac_round` 五个文件，21.8 s）。全量基线最近一次记录见
[v0.8 入口 §8](../v08/README.md)（2026-11-08：6439 passed）；本会话未重跑全量。