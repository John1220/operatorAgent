# v0.8 进度入口

更新于 2026-11-08 · 分支 `codex/v0.8`。详细回执见第 5、6 节。

> **本轮完整报告**：`docs/status/v08/ac-workflow-round-report-20261110.md`
> （结论：强模型两臂打平、弱模型 **reproduced**；七个仪器缺陷、限制与证据索引）。

## 1. 当前状态

FP32 Matmul 的 IEEE 基线与计算路径判断已修复，并通过 H20 回归。修正后的分块复测和诊断开关首对已归档。普通 DSH（A）对 v0.8 插件（C）的对照**已完成**：强模型两臂打平；弱宿主（27B/xhigh）预登记 2 轮复现判 `reproduced`（范围限一次生成可交付的任务，不含性能与开发人时），见本轮完整报告。

**attention 族 C5 已于 2026-11-08 在 H20 跑通**：L1/L2（8 条 case 全过）通过，拿到第一份 L3 与绑定轴 `compute`，
测量有效性 `degraded`（launch 地板 17.3%）。这条路径上暴露出并修掉了三处"身份门之后的缺陷"。见
[C5 端到端](c5-attention-e2e-20261108.md)。

## 2. 已完成

| 交付 | 提交或状态 | 验收证据 |
|---|---|---|
| host 侧性能事实有界投影（绑定轴/地板/噪声档/用途边界） | `d6ec795` | `tests/unit/test_host_context.py`（16 项） |
| optimize 阶段确定性诊断（5 状态、轴过滤、地板串行上限、余量三口径） | `bcbbe75` `34f1177` `097e999` | 真机回执驱动的 3 次修正，见 [真机回执](real-receipt-20261107.md) |
| 噪声 preflight（`--shape/--op/--dtype/--repeat`，CUDA event 通道） | `30eca6d` `29264d4` `097e999` `409ef2a` | [preflight 文档](../../testing/v08-noise-preflight.md)；4096³ 带宽 0.20%、1024³ 3.16% |
| 两臂模型计量对账（两个 verdict、四条 blocker） | `7b06f99` | [对账文档](../../testing/v08-arm-metering.md) |
| R2 预登记 + 校验器（fp16 / fp32 两份计划已冻结） | `3681218` `6ba557b` | hash `58bde16da8fbb252` / `99b575c17f0101e2`；fp32 首版 `8d460f9454ad2531` 已撤回（门控继承了 fp16 探针），见 [候选余量](candidate-headroom-20261107.md) |
| R2 FP16 试跑（管道跑通，增量未确立） | `afc5c78` | [FP16 试跑记录](r2-pair-1-20261107.md) |
| R2 FP32 v2 原生首对 | 已归档 | 原 TF32 基线下，B `0.3820×`，C `0.3674×`。两组通过完整 L2，L3 测量有效。[FP32 首对记录](r2-fp32-pair-1-20261108.md) |
| FP32 Matmul 基线与计算路径 | 工作树，未提交 | CUDA 真机内建 Matmul 明确请求 IEEE FP32 基线，回执须确认选中。诊断依据候选 PTX，路径不明时不报计算余量或据此判停。原起点 `5.490912 ms`，同精度基线 `4.520304 ms`。[真机回归](ieee-fp32-measurement-20261108.md) |
| FP32 分块宽度单变量复测 | 已归档 | 原 C 终稿仅改 `BLOCK_N`（64 → 128），同卡交替测得时延从 `5.268592 ms` 降至 `5.071544 ms`（下降 `3.74%`）。四轮均通过完整 L2，L3 测量均有效。[分块复测](ieee-fp32-measurement-20261108.md#单变量复测) |
| FP32 诊断开关对照 | 首对已归档 | 开启组 `5.060272 ms`，关闭组 `5.162624 ms`。两组完整 L2 通过、L3 测量有效，各 8 次验证、11 次绑定后模型派发。开启组时延低 `1.98%`，未达登记的 `2.5%`。[开关对照](ieee-fp32-measurement-20261108.md#诊断开关对照) |
| 候选任务余量实测（5 个候选） | `409ef2a` | [余量表](candidate-headroom-20261107.md) |
| attention 族 C1–C4（布局/标量、参考+容差、套件+反作弊、roofline） | `1ea7448` `c213be3` `7673937` `4ea75dd` `a8b9cd0` | [attention 设计 §13–16](../../design/v08/attention-family.md) |
| **attention C5 端到端**（L2 八条全过 + 第一份 L3，`compute` 轴，`degraded`） | 工作树，未提交 | [C5 端到端](c5-attention-e2e-20261108.md)；v2 回执里第二处 `inputs["X"]` 与 case 身份两侧不一致同时修掉 |
| **A1 部署钉 torch 版本**（`torch==2.14.0`，装完回读核对，漂了就拒放行） | 工作树，未提交 | 全新目录与**默认目录**都自动回到 `2.14.0+cu130`，身份门通过、L2 八条全过；[A1/A2 记录](a1-a2-deploy-pin-and-attention-gate-20261108.md) |
| **A2 attention 窗口门**（`--op attention --shape B,H,S,D [--causal]`） | 工作树，未提交 | H20 实测 channel `attention-1x16x1024x128-fp16-causal-cuda-event`，带宽 `0.1%`、`action=start`；探针 `0.09728 ms` vs 同窗 L3 `0.09648 ms` |
| **A3 attention 进 R2**（`operator-v08-increment-r2-attention-v1` 已冻结） | 工作树，未提交 | 计划哈希 `2669865924bad093…`，门控通道 `…-fp16-nocausal-cuda-event`、带宽 `0.1%`；校验器把门控绑定从 dtype 扩到算子/尺寸/mask；[A3 记录](a3-attention-r2-preregistration-20261108.md) |
| **A4 支持清单**（`capabilities().correctness_profiles.families.attention`） | 工作树，未提交 | 报出 profile/dtype/Scope A/排除项/8 条 case/粗估 roofline/窗口门；`operators` 标明「不是支持清单」；[支持矩阵](../../testing/v08-support.md) |
| **A5 Scope A 几何校验**（`attention_scope_problem` + `attention_scope` 拒绝） | 工作树，未提交 | `S=3072`、`B=8/H=8/D=64`、`(B,H,S)` 三元组都从静默冻结变成具名拒绝；只收紧 attention，layernorm/matmul 的 profile 未动 |
| **attention 的 `valid` L3**（S=4096） | 工作树，未提交 | `1.212096 ms` vs SDPA `1.043360 ms` ⇒ `0.8608×`，地板占比 1.64%、散布 0.07%、47 次采样，`validity=valid`；8 条 case 全过 |
| **R2 计划 v2**（`…-attention-v2`，任务 S=4096） | 已冻结 | hash `756f17d44e148788…`；v1（S=1024）按名字撤回——那个核的地板占 17.3%，每次配对结构性不可判 |
| **A 臂预算硬门 + 配对比较器** | 工作树，未提交 | `arm_gate.py`（调用/输出/墙钟三格硬门，写 `usage-reconcile` 读的同一份日志）+ `r2_compare.py`（终评绑定、源码重算哈希、两臂 `valid` 才可比、连续两对不可判即停）；[记录](r2-wiring-and-valid-l3-20261108.md) |
| **attention 的 optimize 闭环（真机）** | 工作树，未提交 | 第一次跑 `phase=optimize`：诊断 `compute_bound`/76.6%；三次单变量实验里 tiling 两个方向都**更慢**（+34.6% / +10.8%），`num_stages 2→3` 两次 **−2.14% / −1.86%**；全部 L2 通过、L3 `valid`。[记录](attention-optimize-round-20261108.md) |
| **optimize 上下文字节预算**（超限时裁模板正文并登记，裁不动才抛错） | 工作树，未提交 | attention S=4096 的 optimize 体 41.3 KB > 40 KB，原本直接建不出来；现在 `truncated=true` + `dropped=[…]`，3 条测试钉住 |
| **诊断措辞收口**（实验表改成维度菜单，去掉方向性祈使句） | 工作树，未提交 | 6 行 `change` 全部重写 + `POLICY` 补 MENU OF DIMENSIONS；两条测试钉住"不再出现 raise/increase/make 这类开头"；`context_hash` 随之改变 |
| **A/C 工作流轮 P0-1**（计划已冻结） | 已冻结 | `operator-v08-workflow-ac-v1`，hash `41439fca11149c76…`，8 格（4 任务 × A/C）、`gpu_minutes=180`；[待办 §2b](ac-workflow-round-todo-20261108.md) |
| **A/C 工作流轮 P0-2**（插件侧臂名参数化） | 工作树，未提交 | `normalize_host_ledger(..., arm='B')` + `--plugin-arm`；归档 C 账本复算 `9 次派发 / 19594 tokens` 与 `pair.json` 一致，除臂名外逐字节相同 |
| **A/C 工作流轮 P0-3**（预算硬门接进派发路径） | 工作树，未提交 | `arm_gate` 有了 CLI 入口 + `wrap()`，接进 `model_gateway(..., arm_gate=…)`（默认关）；第 31 次抛 `arm_call_budget`、日志 30 行、解析通过 |
| **A/C 工作流轮 P0-4 ①②**（臂面核对 + supervisor 终评） | 工作树，未提交 | A 臂工具齐全且无插件入口、C 臂两个入口都在（`output/v08-ac-20261108/surface-arm-{A,C}.json`）；`prepare → 同源码 L2 → purpose=final` 三步跑通、`remaining_final=0` |
| **A/C 工作流轮 P0-4 ③（可验证部分）**（两份 prompt + 一格记录链） | 工作树，未提交 | `arm_prompt`：A 无插件词汇、C 写死插件工作流、两臂预算/声明逐字一致；`a_cell` 跑出 `delivered` + `purpose=final`（`output/v08-ac-20261108/cell-smoke.json`） |
| **A/C 工作流轮 P0-5（A 臂）**（scripted upstream 真跑一格） | 工作树，未提交 | 真 headless DSH → bash → `check.py` → supervisor 服务 → 选中源码 → `purpose=final` ⇒ `delivered`；`scripted_fixture=true`、2 次 fixture 请求、`remaining_final=0`（`output/v08-ac-20261108/fixture-cell.json`） |
| **C 臂一跳（方案 1 跑通）** | 工作树，未提交 | 插件用 host 模式 MCP 提交、终评由 supervisor 用**同一套** host task 产出（A/C 同源）：`delivered`、`purpose=final`、L2 通过、3 次 fixture 请求（`output/v08-ac-20261108/fixture-cell-c.json`） |
| **P1-2 首次真模型尝试（A 臂）** | 失败，已定因 | 30 次模型调用全部用于 `grep/sed` 读本仓、0 次提交 ⇒ **A 臂没有工作区隔离**，能读到插件实现与历史回执；先接 `s5_seatbelt` 再谈一对（todo §2g） |
| **P1-2 第二次（预算 20 + 沙箱）** | 失败，已定因 | 20 次调用、1 次提交、`L1: Missing 'kernel' function` ⇒ A 臂材料里缺 **ABI**（插件会通过 `operator_context.interface` 给，A 臂没有等价物）（todo §2h） |
| **P1-2 第三次（补 ABI）** | 结果：A 臂 0 交付 | L1 **通过**、L2 失败（`1.16e-03` vs 阈值 `1e-5`）；20/20 调用、1 次提交（todo §2i）。C 臂尚无同预算对照，故不能下「A 输给 C」的结论 |
| **C 臂进沙箱** | 工作树，未提交 | 插件 / `operator_agent` / `remote_scripts` / numpy 全部搬进格子目录，patch 指向拷贝、`run_dir` 进写白名单；scripted 模型在 repo-blind 沙箱里 `delivered`（todo §2j） |
| **第一对真实数据（20 次预算）** | **已撤回，不能当 A/C 对照** | C 臂的工具列表里没有任何 `mcp__operator__*` ——它跑成了第二个裸 DSH（todo §2l）。链路本身是好的：已知合格 fp32 核在同一契约下 L2 通过、误差 0 |
| **预算口径修正 + 计划 v2** | 已冻结 | 线上 `32768`、每次预留 `40960`（v1 抄了插件默认的 `8192`，把每步输出截半）；v2 `operator-v08-workflow-ac-v2` hash `ef6491cb…`，20 次调用/总预留 819200；新增预算自洽与工具面两条 fail-closed 断言（todo §2m） |
| **真凶：`--arm C` 一直在跑 A 臂** | 已修 | `ac_live.main()` 写死 `run_a_live`；修好后 C 臂工具面才有 7 件 operator 工具。§2k 的对照结论作废（todo §2n/§2o） |
| **真 C 臂首跑（20 次）** | 0 提交，已定因 | 验证要在沙箱里直接 ssh H20，而 `~/.ssh` 不在读白名单；A 臂走 supervisor 不受影响（todo §2o，两条修法待选） |
| **ssh 沙箱打通 + C 臂真用上插件** | 工作树，未提交 | 显式 host/port/user/key + 只白名单那把私钥；C 臂调用 `operator_context`/`operator_verify`。但 receipt 绑到了别的 contract（`f6062c86…`），且 C 臂有 bash 时会绕过插件自己 ssh 上卡（todo §2p） |
| **臂的模型换成本地 vLLM** | 工作树，未提交 | `Qwen3.8-27B`（H20 上的 vLLM，不限量），经 ssh 隧道走 `127.0.0.1:18003`（明文只许回环这条策略得以保留）；smoke 10 次请求全部被服务 |
| **`reasoning_effort` 可用**（`none` 关思考） | 实测 | `none` 0.6s/25tok、`medium` 2.9s/151tok、`high` 400；不用改 `chat_template_kwargs` 禁令。但关思考后 27B 在该任务上明显更弱（todo §2r） |
| **medium 跑的一对** | 两臂都不交付，原因不同 | A：单次生成 524 s 撞流式截止（`TRANSPORT`）；C：17 次 `exit_plan_mode` 打转、0 提交（todo §2s） |
| **更正：截止系数 vs 吞吐** | 已改 | 实测 53 tok/s，而截止按 62.5 tok/s 算 ⇒ 写满 32k 必被掐；系数改为可配（默认 24 s/1k）；并关掉 plan 模式（C 曾 17/20 次耗在 `exit_plan_mode`） |
| **第一格真实交付（A 臂）** | 里程碑 | `delivered`：L1/L2 通过、误差 `0.00e+00`、15 条 case、`purpose=final`、20 次调用。C 臂仍失败：`operator_context` 过大 + medium 思考把 32k 输出吃满（todo §2u） |
| **第一对出现差异** | 工作树，未提交 | qwen3.8-2.4t、medium、20 次：A 未交付（L2 `1.16e-03`）、C **交付**（`0.00e+00`、15/15）；`paired_difference=+1`、`clears_minimum_detectable_unit=true`，但 n=1 且整轮不完整，`formal_comparison_credit=false`（todo §2w） |
| **剩余 6 格** | 未执行（阻塞） | 上游 `qwen3.8-2.4t` 免费额度用尽 → 403；6 格记为 `unknown`。同账号 `deepseek-v4.1-flash-1017` 与本地 vLLM 仍可用（todo §2x） |
| **剩余 6 格（deepseek-pro 重跑）** | 2 跑完 4 未执行 | attention/A 真跑但 L1 签名不符；attention/C 跑一半额度断流；softmax/tall-K 四格 403（todo §2y）。可用：`deepseek-v4.1-flash-1017` 与本地 vLLM |
| **完整一轮（8 格）** | 工作树，未提交 | **C 2/4、A 1/4，配对差 +1**，`round_complete=true`、`clears_minimum_detectable_unit=true`；但跨格换过模型、无计量回执 ⇒ `formal_comparison_credit=false`（todo §2z） |
| **attention 两格查因** | A 的失败是我方 bug | 我给裸臂的 ABI 只含张量、漏了标量 `causal`，A 照抄后 L1 挂；插件给的签名是对的。已修 `calling_convention`（含 `case.scalars`）并加测试，**A 的 attention 格需重跑**（todo §3a） |
| **attention/C 为何写错** | 已定因 | 候选是真正的 flash-attention（在线 softmax/exp2/causal 分支），属数值 bug 而非不会写；插件三处可改进：反馈太粗、缺廉价语义自检、scope 事实没进 context（todo §3b） |
| **两处插件改进** | 已落地 | L2 失败消息带 `matched_ratio`/`first_bad@`；context 新增 `suite_intent` （声明值 ≠ 被测值，attention 的 causal 两种取值都会被测）；6467 passed（todo §3c） |
| **修 ABI 后重跑 attention** | 结果反转 | A **交付**（`6.10e-05`、suite 8/8）；C 未交付。整轮重算：A 2/4、C 2/4，**配对差 0**（todo §3d）——之前的 +1 是我方 ABI bug 造成的假象 |
| **C 的 attention 也交付了** | 已定因 | MCP 工具调用 60 s 超时（我为保住 host 模式去掉了 `development`，超时随之掉回默认值）；抬高后 L2 通过、终评交付。整轮 A 2/4、C 3/4，但**同模型下是 2–2 平手**，+1 只来自跨模型那一格（todo §3e） |
| **softmax：C 交付、A 挂反作弊** | 工作树，未提交 | context 公布被测布局 + suite 消息点名 `layout=strided`；C 交付（`1.91e-06`、12/12），A 因**调用 torch 原版 softmax** 被判反作弊——而这条规则两臂都没被告知。整轮 A 2/4、C 4/4（todo §3f） |
| **反作弊规则公布后** | 结果收敛 | A 的 softmax 也交付（`3.81e-06`、12/12）。整轮 A 3/4、C 4/4；**同模型三格 3–3 平手**，唯一差异来自跨模型那一格（todo §3g） |
| **弱宿主（27B, xhigh）attention 一对** | 首个正证据 | A 输出 NaN（`nonfinite`）、C **交付**（`6.10e-05`、8/8）；服务端只支持 `xhigh/medium/low`（`high` 报 400）（todo §3h） |
| **attention 重复 3 次（27B/xhigh）** | 强信号 | C **3/3** 且每次只用 **3 次调用**；A **1/3** 且每次用满 20 次（失败原因一次 NaN、一次编译错）——交付率与代价双重差距（todo §3i） |
| **弱宿主完整一轮（27B/xhigh）** | **A 1/4、C 4/4，配对差 +3** | A 的三次失败分别是编译错、NaN、strided 布局；C 每格只用 **3 次调用**、A 每格用满 20 次 ⇒ 交付率与代价双重差距（todo §3j） |
| **弱宿主一轮（修正采样后）** | **A 1/4、C 4/4，配对差 +3，采样身份 established** | C 每格 3–6 次调用交付、A 每格 20 次且 3/4 未交付；`sampling_identity` 六字段逐格核对一致（todo §3l） |
| **复现计划 v3 + 聚合器** | 已冻结/在跑 | `operator-v08-workflow-ac-v3`（`031824ff…`）：3 轮 × 3 格，严格规则（每轮 ≥ +1，否则 null 且停止）；`ac_aggregate` 才给 `formal_gain_claim`（todo §3n） |
| **复现计划 v4（2 轮）** | 已冻结/在跑 | v3 按名字撤回（用户把轮数从 3 降到 2）；v4 `2252118c…`：2 轮 × 3 格，严格规则不变；GPU 3 + 验证前空闲预检（todo §3p） |
| **复现计划 v6（2 轮，40960）** | 已冻结/在跑 | v5 作废（网关硬顶每次预留 ≤40960，64k 直接 429）；v6 `42e01fca…` 只带 prompt 修正「别在思考里复述源码」，两臂同源（todo §3s） |
| **复现结果 reproduced** | **结论成立** | 两轮配对差均 +1（A 1/3、2/3；C 2/3、3/3），采样身份两轮 established ⇒ `formal_gain_claim=true`，范围限 27B/xhigh + 一次生成可交付（todo §3u） |
| **A/C 工作流轮的判读量具**（整轮预登记 + 交付数比较器） | 工作树，未提交 | `ac_plan.py`（4 任务 × 2 臂、首要指标=交付数、最小可检出单位 +1 任务、留出槽按 family/configuration 区分）+ `ac_compare.py`（交付/失败/未知/未执行四态分离，A 臂没过 L2 是**数据**不是坏输入，源码哈希按文件重算，`formal_gain_claim` 恒 False）；25 条测试。[A/C 待办](ac-workflow-round-todo-20261108.md) |
| 审查后的 verify 修复（输出设备、标量物化、`torch.ops` 访问与预先保存） | 工作树，未提交 | [三处阻塞记录](review-fixes-20261107.md) |
| 绝对容差按 profile **相等绑定**（attention 不再继承 layernorm 的 `1e-5`） | 工作树，未提交 | [C5 §7](c5-identity-blocker-20261107.md)；attention 契约 `33a95e6c…` → `984834bd…`，layernorm 契约哈希逐字不变 |
| 模型路线（qwen3.8-2.4t-a95b 可用但强制思考） | `6397e82` | [model-switch](model-switch-qwen-20261107.md) |

历史测试基线：`tests/unit` + `tests/hygiene` **6320 passed / 0 failed / 1 skipped**（2026-11-07 21:05）。
**2026-11-08 19:20 实测：6414 passed / 0 failed / 1 skipped**。
**2026-11-08 21:13 实测：6439 passed / 0 failed / 1 skipped / 625 subtests passed**（+25 = A/C 量具），见 §8。
最近一轮相关单测 559 项通过，按迁移后的追踪路径用临时 Git 索引核对仓库守卫，59 项通过。

## 3. 进行中 / 阻塞

| 项 | 状态 | 说明 |
|---|---|---|
| **普通 DSH（A）对 v0.8 插件（C）** | **已完成：弱宿主 reproduced** | 预登记 2 轮 × 3 任务，每轮配对差 +1，聚合判 `reproduced`（范围：27B/xhigh、一次生成可交付的任务；不含性能/人时）；强模型打平。过程与判读边界见 [A/C 工作流轮待办](ac-workflow-round-todo-20261108.md) 与 [本轮报告](ac-workflow-round-report-20261110.md) |
| **C5 attention 端到端** | **已跑通（`degraded`）** | 契约 `556b76336accdb41…`、候选 `90f83ca50d2e…`。主用例 max_abs `1.22e-04`、八条派生 case 全过；L3 候选中位 `0.09648 ms` vs 同卡 SDPA `0.08515 ms`（`0.8826×`），绑定轴 `compute`、达成率 60.2%。`degraded` 的成因是 launch 地板 17.3%（96 µs 核 + 20 µs 地板），**不是**候选错。要 `valid` 读数需要换 S=4096 主用例或更快的核，属 R2 取数。见 [C5 端到端](c5-attention-e2e-20261108.md) |
| `sampling_comparability` | v6 两轮 established | 六字段（model/temperature/n/max_tokens/enable_thinking/reasoning_effort）从每格 `outer-requests/*.json` 逐字核对一致；早期试跑的缺口由 `ac_metering.sampling_identity()` 补上。 |

默认 H20 runner 的 torch 漂移已由 A1 部署钉版本解决（`torch==2.14.0`，装完回读核对 `2.14.0+cu130`，漂了拒放行）；受信任参考任务在钉版本后通过身份门。

## 4. 下一步

### attention 进 R2 之前还剩

1. **attention 的配对形式要重新登记**：现有计划按 "B=v0.7.2 / C=v0.8" 写，而 v0.7.2
   **准备不了 attention**（没有这个 profile）。可行形式是同一份 v0.8 开关 optimize 诊断
   （`OPERATOR_R2_DIAGNOSIS=off/on`，与 fp32 那次同构），需要新编号。
2. A 臂 agent 已在 A/C 工作流轮真跑过多轮；R2 attention 配对本身尚未登记。插件侧采样字段（契约 `12 → 13`）仍推迟，所以 R2 配对结论只能报观测。

### A/C 首对准备（已完成）

首对与复现轮均已跑完：结论、成本读数与限制见 [本轮报告](ac-workflow-round-report-20261110.md)，
逐条过程记录以 [A/C 工作流轮待办](ac-workflow-round-todo-20261108.md) 为准。
后续轮次（修复型任务、更多任务族、中档模型探点）尚未登记。

### P1 性能 parity（已跑完：C_worse_observed）

对 v6 **两臂都交付**的 3 个配对（matmul-4096-fp32 ×2 轮 + attention-prefill-4096 ×1 轮）重跑独立 L3，
只问「C 是否比 A 更慢超噪声带」（parity，不是增益、不是打库）。2026-11-14 在 H20 GPU3 跑完：
三对全部可判、全部 C_worse（+24.89% / +1.85% / +38.13%），系列判 `C_worse_observed`、
`formal_parity_claim=false`——**性能 parity 主张被否证**（与交付数增益分属两轴，互不推翻）。
见 [P1 预登记](../../testing/v08-p1-preregistration.md) 与 [v0.8.1 进度入口 §3](../v081/README.md)。

### 其他待决定事项

1. **默认 runner 的参考环境**（C5 复测已用隔离环境）
   - **A（建议）**：部署时 `torch==<attested>`（版本从 `PINNED_RUNTIME` 取），两份 profile 继续有效；
   - B：在 2.14.1 上重做隔离审计并给两份 profile 升版。
2. **契约 `12 → 13`**：是否把 `preflight` / `usage-reconcile` 暴露成 host 工具。
   我建议 R2 期间不做（会让 C 臂多两个工具、污染"两臂只差 optimize"的立论）。
3. **qwen 用在哪个系列**：建议 R3（测"模型变弱时诊断增益是否变大"），不进 R2。

## 5. 文档索引

**设计**：[v0.8 准备提案](../../design/v08-preparation.md)（原文照录）·
[attention Scope A 设计](../../design/v08/attention-family.md)（含 C1–C4 实测）·
[任务族选型定案](../../design/v08/task-family-selection.md)（性能基线认 eager 组合，不认厂商库）

**测试/操作**：[noise preflight](../../testing/v08-noise-preflight.md) ·
[两臂计量对账](../../testing/v08-arm-metering.md) ·
[R2 预登记](../../testing/v08-r2-preregistration.md) · [R2 跑法](../../testing/v08-r2-runbook.md) ·
[P1 性能 parity 预登记](../../testing/v08-p1-preregistration.md)

**状态/证据**：[真机回执](real-receipt-20261107.md) ·
[候选余量](candidate-headroom-20261107.md) · [R2 FP16 试跑](r2-pair-1-20261107.md) ·
[R2 FP32 原生首对](r2-fp32-pair-1-20261108.md) ·
[IEEE 基线、分块宽度和诊断开关复测](ieee-fp32-measurement-20261108.md) ·
[模型切换](model-switch-qwen-20261107.md) · [C5 卡点](c5-identity-blocker-20261107.md) ·
[审查后三处阻塞修复](review-fixes-20261107.md) ·
[C5 attention 端到端](c5-attention-e2e-20261108.md) ·
[A1 部署钉版本 + A2 attention 窗口门](a1-a2-deploy-pin-and-attention-gate-20261108.md) ·
[A3 attention 进 R2](a3-attention-r2-preregistration-20261108.md) ·
[valid L3 + R2 接线](r2-wiring-and-valid-l3-20261108.md) ·
[attention optimize 闭环](attention-optimize-round-20261108.md) ·
[A/C 工作流轮待办](ac-workflow-round-todo-20261108.md) ·
[支持矩阵](../../testing/v08-support.md)

## 6. 仓外证据（不在工作树内）

| 目录 | 内容 |
|---|---|
| `output/v08-real-receipt-20261107/` | 三份真机回执（matmul valid / relu degraded / layernorm unknown）+ 三份 preflight |
| `output/v08-candidates-20261107/` | 5 个候选的 L3 与 `gate-fp32.json` |
| `output/v08-r2-20261107/fp32-v2-pair-1/` | 原 R2 FP32 首对的冻结计划、噪声门、两臂原生会话及 `pair.json` |
| `output/v08-r2-20261107/ieee-fp32-tile-control-20261108/` | IEEE 起点回归和四轮分块宽度复测，含 `summary.json` |
| `output/v08-r2-20261107/diagnosis-ab-ieee-20261108/` | 修正后 v0.8 的诊断开关对照，含冻结计划、两臂原生会话及 `pair.json` |
| `output/v08-attention-20261107/` | C5 原始验证、Attention 核、指纹探针和隔离 runner 的四轮复测 |
| `output/v08-attention-20261108/` | C5 最终两次运行与回执、attention 窗口门（causal/nocausal）与冻结的 R2 attention 计划 |

关键哈希：layernorm 参考 `8f779954d3d2b3af…`、attention 参考 `145c6cce7c6a3192…`、
R2 计划 `58bde16da8fbb252…` / `99b575c17f0101e2…`（fp32 首版 `8d460f9454ad2531…` 已撤回）。

## 7. 只有提交信息、没进文档的细节（如实说明）

- 每次改动**为什么**改（例如"地板不能冒充竞争轴"那条推导）只在提交信息里；
  专题文档给出结论，不给完整推演。
- 三处"标量不是数组"的既有缺口分布在 `hitl_tools.prepare`、`remote_scripts/run_kernel.py`、
  `core/contract.py`，结论写在 [attention 设计 §16](../../design/v08/attention-family.md)，
  逐个的现场报错文本只在提交信息里。
- 本文件之前，v0.8 **没有入口文档**（v0.7.x 有 `v072-todo.md`）—— 这就是你现在看的这份。

## 8. 测试现状（2026-11-08 18:25 实测）

`tests/unit` + `tests/hygiene`：**6409 passed / 0 failed / 1 skipped**（唯一 skip 是需要真实 CUDA 设备的那条）。
本轮清掉的六条：

| 条数 | 测试 | 处理 |
|---:|---|---|
| 4 | `test_budget_gate`×2、`test_escalation`、`test_search_objective` | **真回归，已修**：计算路径未确立时 `gflops_utilization_pct=None` 进 `agents._num()` 抛 `TypeError`，被吞成"LLM 优化失败⇒回退规则优化"，结构搜索静默消失。`_num` 现在显式区分 `refused / None / 数值`，并把"峰值=0（不知道）"也印成"无可用读数"。新增 `tests/unit/v08/test_unknown_compute_path.py`（3 条） |
| 1 | `test_pluggable_reference` | 守卫从"数源码里某个表达式"改成看 AST 关键字：少传一次就红，换写法不红 |
| 1 | `test_repo_layout` | `docs/status: 52 > 51` 是因为 `v08-todo.md` 的删除尚未进索引；已 `git add` 这条删除（其余改动仍未暂存），按提交后的状态计数正好 51 |

**2026-11-08 21:13 追加实测：6439 passed / 0 failed / 1 skipped / 625 subtests passed。**
多出的 25 项是 [A/C 工作流量具](ac-workflow-round-todo-20261108.md)（`tests/unit/v08/test_ac_workflow_round.py`）。
上面 18:25 那行是**那一窗的读数**，不改写；本节标题里的时间戳跟着那次读数走，最新读数在本节结尾。
