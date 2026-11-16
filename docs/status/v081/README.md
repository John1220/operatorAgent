# v0.8.1 优化分支进度入口

更新于 2026-11-16 · 分支 `codex/v0.8.1-optimize`（11 月 14 日由 `codex/v0.9-optimize` 改名）。

> **为什么是 v0.8.1 而不是 v0.9**：产品路线里的 **v0.9 是「业务接入版」**（把已验证核接进真实
> 框架/模型、端到端回归，见 [产品路线](../../design/operator-development-product-roadmap.md) §5.3），
> 尚未开始。本分支做的是判定缓存、异步验证、任务族选型与性能 parity 测量，是 **v0.8 优化工作的
> 延续**，故按 minor 递增记 v0.8.1，不占用路线的 v0.9 语义。基线分支 `codex/v0.8`。

## 1. 当前状态

- **验证链路增强（已提交 `9fb3ce7`）**：判定缓存与异步验证落地，P0 计量漏计修复 + 文档勘误，
  方向1 任务族选型定案。
- **P1 性能 parity 已完成。** `0c83cd1` 已提交测量和判读工具。真机复测 v6 两臂都交付的三个配对，
  三对 C 臂时延均更高，判为 `C_worse_observed`，`formal_parity_claim=false`（见 §3）。
- **P2 优化诊断闭环已完成正式判读。** 正式两对（pair-1/2）均完整 L2、独立 L3 `valid`；
  pair-1 诊断组低 8.49%、pair-2 仅 0.15%（落当窗带内）⇒ 整轮 `no_increment_observed`，
  `formal_gain_claim=false`；四份终稿均慢于起点核。见 [P2 正式结果](../../testing/v08-p2-preregistration.md#p2-formal-result)。

## 2. 已完成

| 交付 | 提交或状态 | 验收证据 |
|---|---|---|
| **判定缓存**（契约 × 源码 × 判定深度 × 环境身份 × 实现。回放范围为探索阶段的决定性 L1/L2，命中也按次扣预算） | `9fb3ce7` + `db176f8` | `tests/unit/v08/test_verdict_cache.py`（15 项，含 `unknown` 重试和仅返回 `backend` 时的两条回归） |
| **异步验证**（`operator_verify` 增 `wait=false` → 脱离提交返回 `invocation_id`，`operator_status` 带终态 `result`，`operator_cancel` 不变） | `9fb3ce7` | `operator_agent/app/mcp_server.py`；`tests/unit/test_mcp_server.py::TestAsyncVerify`（3 项） |
| **P0 计量漏计修复**（A/C 两条早退路径都过 `_counts`，不再漏成 None） | `9fb3ce7` | `tests/unit/v08/test_ac_round.py::test_failed_cell_early_return_still_meters_dispatches`（变异验证） |
| **P0 文档勘误**（报告 §7 两级正式信用 + n=1 措辞、README §3、`ac_aggregate` docstring；冻结回执不改，round-1 哈希复算 intact） | `9fb3ce7` | `docs/status/v08/ac-workflow-round-report-20261110.md` |
| **方向1 任务族选型定案**（性能基线认 eager 组合，不认厂商库；只有库基线的任务只进交付数对照） | `9fb3ce7` | [任务族选型定案](../../design/v08/task-family-selection.md) |
| **P1 预登记 + 判读量具 + 测量执行器** | `0c83cd1` | `benchmarks/experiments/v08/p1_plan.py`（judge_pair/judge_series/validate/freeze）、`p1_run.py`；`tests/unit/v08/test_p1_plan.py`（11）+ `test_p1_run.py`（8）；[P1 预登记](../../testing/v08-p1-preregistration.md) |
| **P1 真机取数 + 判读** | 已归档（仓外证据树） | `output/v08-p1-20261114/run3/series.json` + 三对 `pair.json`；计划哈希 `529b499e…` |
| **P2 诊断开关量具 + 起点核 + 正式两对判读** | `f318910` | `p2_plan.py` / `p2_run.py` / `p2_optimize.py` / `p2_seed.py` + 两组提示；`tests/unit/v08/test_p2_*`（29 项）；[P2 预登记](../../testing/v08-p2-preregistration.md#p2-formal-result)。整轮 `no_increment_observed` |

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
| **方向3 真机标定** | **已标定（2026-11-14）** | 异步全链路 6/6 过；身份+缓存机制全对，环境身份为型号级（已接受，见下） |
| 方向2 autotune 拆分 | 未开始 | 参数层确定性扫描 + LLM 语义重构 |
| P2 诊断增量 | **已完成正式判读（2026-11-16）** | 两对可判：pair-1 `optimize_helps`（+8.49%），pair-2 `no_increment`（+0.15%，带内）⇒ 整轮 `no_increment_observed`，`formal_gain_claim=false`。见 [P2 正式结果](../../testing/v08-p2-preregistration.md#p2-formal-result)。 |
| P3 中档模型探点 | 未开始 | 测「模型变弱时诊断增益是否变大」 |
| **缓存环境身份边界**（标定捕到） | **已接受型号级身份**（2026-11-15） | 当前工作树已拒绝仅有 `backend` 的身份。后端返回 `Unknown GPU` 等占位值时仍可能命中。见 [`_identified`](../../../operator_agent/decide/verdict_cache.py#L79)。 |

### 方向3 标定读数（2026-11-14，H20 GPU3）

量具 `benchmarks/experiments/v08/d3_calibrate.py`，候选复用 P1 绑定哈希的 attention 核
（`abb98d4d…`），不新生成、不调模型。回执 `output/v08-d3-20261114/{identity-cache,async-chain}.json`。

**A. 身份 + 缓存（8/9 过）**：环境身份回读到真实 `torch 2.14.0+cu130 / triton 3.8.0 /
NVIDIA H20 / cc 9.0`；同源码重复 verify 第二次 `cache_hit=True` 且 `cached_from` 指回第一轮、
换源码哈希不命中、预算按派发扣（`reserved` 1→2→3，命中也扣）——**缓存机制全对**。

**型号级身份（已接受的边界）**：标定初版有一项 `environment_has_gpu_uuid` 检查未过——
SSH 的环境身份不含 `gpu_uuid/driver_version/cuda_version`。根因两层都没产出：
- 远端 runner [run_kernel.py:1808](../../../remote_scripts/run_kernel.py#L1808) `get_gpu_info` 没采这三键；
- 控制端 [remote_executor.py:1397](../../../operator_agent/execute/remote_executor.py#L1397) 重建字典也没透传。

⇒ 缓存的 `_EXTRA_IDENTITY_KEYS=(driver_version, cuda_version, gpu_uuid)` 在 SSH 上机会性收录、
当前收不到，有效环境身份是 **GPU 型号级**（name+device+cc+torch+triton+backend），不绑具体卡 UUID。
缓存只回放决定性 L1/L2（正确性），同型号+同 torch/triton ⇒ 正确性一致，故功能正确。
**2026-11-15 用户决定：接受型号级身份，不补字段**；设计措辞已改实（`verdict_cache` docstring/注释
写明 SSH 为型号级、额外键后端产出才收），标定预期同步改为型号级完整性检查、UUID 降为观测项。
要绑卡实例需 runner+executor 两层补采集，届时三键自动生效。

**B. 异步全链路（6/6 过）**：spawn 真实 `operator-agent-mcp` 走 JSON-RPC，分钟级 attention L2：
- `operator_verify(wait=false)` **21 ms** 立即返回 invocation_id（`phase=running`，不阻塞）；
- `operator_status` 轮询 50 次 / **246.9 s** 到终态 `done`，`L1/L2=True`、`ok=True`（后台 worker 真跑出决定性判定）；
- `operator_cancel`  accepted，终态 `cancelled`。

**结论**：异步量具完全可用；缓存量具机制正确，环境身份为型号级（已接受边界，设计措辞已改实）。

### P2 接线前置（2026-11-16，H20 GPU3）

起点核量具 `benchmarks/experiments/v08/p2_seed.py`（`--mode probe|generate`），复用 phase1 已标定
的 study 机制，不改核心默认值。已确认的接线与标定：

- **隧道已确认可用。** 重启本地 SSH 转发后，`/v1/models` 返回 `Qwen3.8-27B`。
- **测量卡 GPU3**：UUID `GPU-634bbc76-9ab6-220a-48a8-27247d7acbe2`（与 P1/方向3 同卡；当前最空，
  单个睡眠邻居共存，不 kill）。状态易变，正式轮以当窗 preflight 噪声门为准。
- **内层采样身份（用户 11-16 定：接成真 xhigh）**：v6 的 xhigh 原是**外层** DSH 宿主经 phase1
  `wire_sampling` 注入的 `reasoning_effort`；内层 Conductor 的 `LLMClient` 默认思考关/4096/无
  reasoning_effort（[autonomous.py:102](../../../operator_agent/app/autonomous.py#L102)）。本量具用
  `connected_worker.run`（`environment+bounded` 上下文进程内跑 `_worker_main`），把冻结 cell config 的
  `sampling`（reasoning_effort=xhigh / enable_thinking=True / max_tokens=32768）normalize 进内层 bounded
  client 请求体。**正控全过**：`sampling-inner.jsonl` 记 requested{思考关,4096,0.3} → effective{xhigh,思考开,32768,0.1,n=1}。
- **凭据**：本机 vLLM 27B 无鉴权（不带 key 也 200，已实证）；client 强制非空 key，用**非密占位**
  `LLM_API_KEY=EMPTY`（用户 11-16 定），不落盘真实密钥。
- **起点核声明（本地纯函数标定）**：`request='fp16 16384x6144 layernorm'`、shapes[[16384,6144]]、
  fp16/triton/nvidia_gpu ⇒ kernel_type=layernorm、profile=`builtin-layernorm-fp16-cuda-v2`、
  contract_hash `642b52781c88fb890fb7f96110fda1fa42b71574438a3c7f6c8175721499cee7`。SSH 后端代码型
  （`param_space` 继承返回 None），起点核确由 LLM 生成而非模板参数空间。

**两个标定发现（先复现再修，已落进量具）**：
1. **EXISTING_REMOTE_ONLY provisioning 坑**：`connected_worker.environment` 设
   `OPERATOR_AGENT_EXISTING_REMOTE_ONLY=1`，deploy 只**检查**现有环境（`.venv/bin/python`+脚本哈希）、
   不 provisioning（[remote_executor.py:1208](../../../operator_agent/execute/remote_executor.py#L1208)）。
   首跑新 work_dir 时检查失败 ⇒ 参考身份 unknown ⇒ Conductor 在 Designer 前早退（rounds=0、无模型调用）。
   修：`generate` 前先**不带**该 flag 跑 `reference_preflight`（触发完整 provisioning + 校验冻结参考身份）。
2. **xhigh Coder 输出预算（超时 → 撞顶 → 内层硬上限；成分已查清、用户定起点核降 medium）**：三次真机生成逐步暴露——
   - ① phase1 `request_timeout` 用 16s/1k，×32768=524s；本机 vLLM ~53 tok/s，xhigh Coder 把整核写进
     思考需 ~618s ⇒ call-003 `wall=524.3s` **超时** failed（api_usage=null）。放宽 `sampling.request_timeout`
     到 30s/1k（`--call-s-per-1k`）后超时解除。
   - ② 超时解除后 call-003 `completion=32768` **写满撞顶截断**、核 JSON 未发出 ⇒ bounded failed。
     根因：phase1 POLICY 默认 max_tokens=32768 把 Coder 正常请求值（40960，见 agents.py「Coder 0.1/40960」）
     **压低**了。提到 40960。
   - ③ 40960 仍 `completion=40960` **写满截断** failed（wall=829.5s）。**40960 是内层 bounded 预算硬上限**
     （[task_budget.py:41](../../../operator_agent/execute/task_budget.py#L41) `llm_limits` 硬编
     `max_output_tokens=40960`；[llm_budget.py:208](../../../operator_agent/execute/llm_budget.py#L208)
     `output <= max_output_tokens` 拒超）。xhigh 真实 Coder（11 配方+RAG）思考写满 40960 仍截断 ⇒ 起点核未冻住。

**截断成分探查（11-16，`p2_truncation_probe.py`，仓外 `output/v08-p2-trunc-{xhigh,medium}-20261116/`）**：
用忠实但简化的 Coder prompt（无 11 配方、ref_code='无'）直发 vLLM，读 `usage.completion_tokens_details.reasoning_tokens`
拆分思考/代码（vLLM qwen3 用 `message.reasoning` 字段，非 `reasoning_content`）：

| 档位 | prompt | finish | 思考 token | 代码 token | 思考占比 | 完整核 |
|---|---|---|---|---|---|---|
| medium | 简化 | stop | 7963 | 787 | 91% | ✓ |
| xhigh | 简化 | stop | 29743 | 1552 | 95% | ✓（31295<40960）|
| xhigh | 真实 Coder（11 配方） | length | ≥40960 | 未发出 | — | ✗ |

**关键 nuance**：代码本身极小（~0.8–1.5k token），**思考占 91–95% 是撞顶主因**。xhigh 思考随 prompt 复杂度涨——
简化 prompt 31295 能 fit，真实 Coder 加 11 配方就把思考推过 40960；medium 思考量只有 xhigh ~1/4。
故「xhigh 必然 >40960」的早先措辞不准确：是**配方复杂度×xhigh** 共同把思考推过上限。

**用户定案（11-16）：起点核生成降 medium**（保 40960 上限、零核心改动；`--reasoning-effort medium`）。
起点核只需正确，medium 思考量轻松 fit；optimize 闭环采样身份另定（Search 诊断调用产小 JSON，非整核，可仍 xhigh）。
不提内层硬上限（避免改核心预算、影响其它路径冻结身份）。v6 提示做法（源码只进最终输出）保留，对 medium 同样有益。

**起点核状态：已冻住（2026-11-16，medium + bounded 修复循环）**。`output/v08-p2-seed7-medium-20261116/`：
Coder 生成（strided 挂）→ Fixer 修 stride（根因「硬编码 row*N+col，未读运行时 tensor.stride()」）→
CorrectnessSuite 32 case 全过 → L1/L2/L3 全过（generate+repair，未进 optimize）。独立验收 explore_l2=True /
final_l2=True / correctness=passed / delivered=True ⇒ FROZEN。
**seed source_hash=`cb851b67bc11530fe4da0b7043f5aaf2fb296c95dedd4b529409bff60d65ee96`**，契约 `642b5278…99cee7`。
L3 speedup=**1.4615× vs pytorch_eager**（min_speedup=1.0），但 `measurement_validity=degraded`
（launch 地板 12.7%>10% + 1 个 harness 共卡进程；can_verdict/can_tune=True，can_persist=False）。
xhigh 失败回执在 `output/v08-p2-seed{2,3,4,5,6}-20261116/`（①超时 ②32768 撞顶 ③④40960 撞顶 ⑤medium strided L2 挂）。

**step 2 机制（L3-gate 发现 → 用户定方案②，11-16）**：种子 L3 已达标（1.46×≥1.0），内层 Conductor
optimize **只在 L3 未过且 round<max_rounds 时才进**（[conductor.py:1822](../../../operator_agent/app/conductor.py#L1822) /
[2627](../../../operator_agent/app/conductor.py#L2627) 全过即停）；且升级门缺省拒绝只在 L3-fail 轮重算
（[2660](../../../operator_agent/app/conductor.py#L2660)/[2766](../../../operator_agent/app/conductor.py#L2766)）⇒
已达标种子强制续跑会两臂都退规则搜索、prompt 差分不渲染。用户选择方案②，用 [固定轮数驱动](../../../benchmarks/experiments/v08/p2_optimize.py)
直接调用 `SearchAgent.run`。两组使用共同的已达标任务描述，开启组插入 Roofline 诊断块。本地测试逐字比对两组提示差异，
并验证模型调用、提示哈希和规则回退记录。

**32768×6144 起点复测。** 输入传输上限由 256 MiB 提到 512 MiB。同一份源码在新契约 `cc9122d1…` 下独立终评，
完整 L2 为 32/32。L3 中位时延 0.268672 ms，相对 PyTorch `F.layer_norm` 为 1.4939×。
启动开销占 **7.01%**，低于 **10%** 门槛。当时 H20 GPU3 上有两个测试进程共卡，L3 档位为 `degraded`。
回执见 §7。

**恢复后的首对试跑。** [双组驱动](../../../benchmarks/experiments/v08/p2_run.py) 已接通真实模型、候选验算和独立终评。
起始噪声带为 0.430926%。起点核独立 L3 为 `valid`，中位时延 0.267744 ms，同卡测试进程为零。
两臂各完成一次 Search 和一次 Coder，终评前的噪声带为 0.88191%。开启组中位时延 0.399200 ms，关闭组 0.749248 ms。
各臂完整 L2 均为 32/32，独立 L3 均为 `valid`。见 [P2 真机试跑](../../testing/v08-p2-preregistration.md#p2-pilot-result)。

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
[P2 优化闭环预登记草案](../../testing/v08-p2-preregistration.md) ·
[noise preflight](../../testing/v08-noise-preflight.md)

**上游进度**：[v0.8 进度入口](../v08/README.md)（A/C 复现 reproduced 的完整过程与回执）·
[A/C 工作流轮报告](../v08/ac-workflow-round-report-20261110.md)

## 7. 实验回执（不纳入 Git）

| 目录 | 内容 |
|---|---|
| `output/v08-p1-20261114/preregister-p1-parity-v1.json` | P1 冻结计划，hash `529b499e…` |
| `output/v08-p1-20261114/run3/series.json` | 系列终判 `C_worse_observed` |
| `output/v08-p1-20261114/run3/p1-{matmul-r1,matmul-r2,attention-r2}/pair.json` | 三对判读 + 两臂源码哈希 |
| `output/v08-p1-20261114/run3/<pair>/<arm>/verify-2/result.json` | 各臂独立终评 L3 回执 |
| `output/v08-d3-20261114/identity-cache.json` | 方向3 身份+缓存标定（8/9，含环境身份缺口） |
| `output/v08-d3-20261114/async-chain.json` | 方向3 异步全链路标定（6/6） |
| `output/v08-p2-seed7-medium-20261116/seed.json` | 起点核冻结记录，源码 `cb851b67…`，原尺寸契约 `642b5278…` |
| `output/v08-p2-probe-l3-32768c-20261116/verify-2/result.json` | 32768×6144 独立终评，新契约 `cc9122d1…`，L2 32/32，L3 `degraded` |
| `logs/p2-pilot.kdTGD8/run/seed/evaluation/verify-2/result.json` | 恢复后起点核独立终评，完整 L2 通过，L3 `valid` |
| `logs/p2-pilot.kdTGD8/run/pilot.json` | 首对试跑判读，`optimize_helps`，`formal_gain_claim=false` |
| `logs/p2-pilot.kdTGD8/formal-plan.json` | 正式轮冻结文件，哈希 `b000d8d6123f…`，两对按相反顺序执行 |
| `logs/p2-formal-pair1-20261116/pilot.json` | 正式 pair-1 判读，`optimize_helps`（+8.49%），带宽 0.3376% |
| `logs/p2-formal-pair2-20261116/pilot.json` | 正式 pair-2 判读，`no_increment`（+0.15%），带宽 0.5621% |
| `logs/p2-formal-series-20261116.json` | 正式两对聚合，`no_increment_observed`，`decidable_pairs=2`、`helps=1`、`hurts=0` |

GPU 身份钉在 GPU3（UUID 见冻结件与 `series.json`）；源码哈希按文件重算与预登记逐字一致。

## 8. 测试现状

**2026-11-14 聚焦读数：73 passed**（`test_p1_plan` + `test_p1_run` + `test_verdict_cache` +
`test_mcp_server` + `test_ac_round` 五个文件，21.8 s）。全量基线最近一次记录见
[v0.8 入口 §8](../v08/README.md)（2026-11-08：6439 passed）；本会话未重跑全量。

**2026-11-16 收尾复跑同五文件：73 passed, 2 failed**。两条失败都是 `test_ac_round` 的 C 臂
DSH-headless 沙箱集成测试（`test_the_c_arm_session_delivers…` / `test_the_c_arm_runs_inside_a_repo_blind_sandbox`），
**环境性、不在本批改动代码路径**：repo-blind 沙箱挡住系统 CommandLineTools Python3 的 dylib
（`dyld: Library not loaded … Python3 (blocked by sandbox)`）⇒ operator-agent MCP server 子进程起不来
⇒ `mcp-operator … tool synchronization failed` ⇒ 无通过候选 ⇒ `state=crashed`。缓存修复批次本身绿
（`test_verdict_cache` 15 项含两条新回归全过）。这两条要绿需沙箱放行一个 dylib 可达的解释器（如 .venv），
属 DSH 沙箱环境接线，另议，不在 P2 收尾里动。

**2026-11-16 本地复核 383 passed，1 skipped，30 subtests passed。** 覆盖 P2 开关和起点核验收、缓存、
输入传输、步幅诊断、搜索、提示模板、测量有效性与仓库布局。

**首对试跑驱动相关回归 175 passed，30 subtests passed。** 含提示差分、出窗暂停和宿主终评状态检查。

**2026-11-16 P2 正式轮后复核：`test_p2_*` 29 passed；缓存/步幅 125 passed。** 正式轮未改代码路径，
两批提交为 `db176f8`（缓存身份/unknown 加固）与 `f318910`（P2 量具与预登记）。
