# P2 预登记草案：optimize 诊断驱动闭环能不能把核改快（因果增量）

**2026-11-16 更新。** 首对真机试跑已完成，正式计划已冻结，正式两对已跑完并判读。试跑读数见 §13，
正式计划见 §14，正式结果见 [§14 末](#p2-formal-result)。
量具骨架沿用 [R2 计划](../../benchmarks/experiments/v08/r2_plan.py)（`v08-increment-preregistration-v1`）
与 [P1 计划](../../benchmarks/experiments/v08/p1_plan.py) 的结构。

## 1. 研究问题

> 两组从**同一份通过完整 L2 的起点核**开始，在相同轮数和预算下继续优化。
> 给其中一组提供 Roofline 诊断提示后，该组终稿的时延降幅能否达到效应门槛并超出当窗噪声带？

结论适用于本轮固定轮数驱动下的诊断提示效果。

## 2. 与 P1 / R2 的关系（别混轴）

| 轮 | 问什么 | 对照 | 结论 |
|---|---|---|---|
| v6 A/C | 交付数 | 裸 DSH vs 插件 | reproduced（弱宿主，一次生成可交付） |
| P1 | 一次交付核的性能 parity | A 交付核 vs C 交付核 | C_worse_observed（parity 否证） |
| R2 | fp32 matmul 诊断开关 | 同插件 optimize on/off | 1.98% < 登记 2.5%（未达） |
| **P2** | **诊断提示带来的时延改善** | **FP16 LayerNorm，两臂按同一预算继续优化** | **待测** |

本轮使用有已标定正确性参考的 FP16 LayerNorm，登记的性能基线见 §6。

## 3. 固定轮数的诊断开关对照

两臂使用同一起点源码与契约、相同模型和采样设置、相同轮数与预算。
实验驱动直接调用 `SearchAgent.run`，按相同规则生成、验证和选择候选。各臂最终候选单独做
`purpose=final`、`stop_after=L3` 的终评。

- **关闭组**收到共同的任务描述、硬件信息和测速反馈，照常由模型提出优化建议。
- **开启组**在相同输入上额外收到 Roofline 数据和瓶颈分析提示。
- 首轮使用同一起点的测量反馈。后续各臂按同一规则使用各自候选的验证反馈。

## 4. 任务选型

**硬前提**：任务必须有已标定容差 profile（L2 前提）。当前注册表只有两个 profile：
`builtin-layernorm-fp16-cuda-v2`（atol 1e-5）、`builtin-attention-fp16-cuda-v1`（atol 3e-4）。

| 候选任务 | profile | 性能基线 | 首轮安排 |
|---|---|---|---|
| layernorm fp16 | 有 | `F.layer_norm` 原生 eager | **首轮试跑** |
| attention prefill fp16 | 有 | 候选 `QK^T*scale → softmax(-1) → @V` | 后续候选 |
| fused layernorm+residual / rmsnorm / softmax+mask | 待标定 | 待登记 | 后续候选 |

## 5. 起点核及契约绑定

已用 27B/medium 生成起点核，在预算内修复非连续输入的步幅问题，完整 L2 和独立终评通过后冻结源码。
`output/v08-p2-seed7-medium-20261116/seed.json` 绑定原尺寸 16384×6144 的契约 `642b5278…`。
同一源码在 32768×6144 上已通过完整 L2。新尺寸试跑用该源码及新契约 `cc9122d1…`，绑定见 §13。

## 6. 基线纪律（方向1，2026-11-15 定 A）

- **本轮基线是 `pytorch_eager`，即 PyTorch 原生 `F.layer_norm`。** 历史 1.4135× 来自 FP32 的单形状读数。
- LayerNorm v2 绑定 `pytorch_eager` 作为性能基线，并使用 `native_layernorm` 作为正确性参考。见 [Verifier 接线](../../operator_agent/decide/verifier.py#L1645)。
- **P2 主指标是 on/off 配对差，两臂同基线，基线选择在差值中对消**；绝对「赢 eager」是次要上下文。
- direction-1 的「未融合组合（mean→var→rsqrt）」理想记为更严后续选项（B），不在本轮。
- `baseline_type` 认机器值（`pytorch_eager`），不认显示串。

## 7. 探针（逐任务，窗口必须重跑）

按形状/算子/dtype 单独做噪声 preflight，**不许继承** layernorm 16384×6144 或别的窗口读数。
`action == "start"` 才开测；无合格探针窗口的任务性能记 `unknown`，不进主张。
探针通道与带宽在冻结件里逐任务绑定。

## 8. 有效性门与判读骨架（落地为量具时写死）

一个配对**可判**当且仅当：两臂都在绑定起点/终稿哈希上过完整 L2、两臂 `measurement_validity=="valid"`、
重冻结契约 == 登记契约、各臂独立终评。

- 单对：`diff_pct = (off_median − on_median) / off_median × 100`（on 更快为正）；
  `≥ effect_threshold` 且 `> band`（超当窗噪声）⇒ `optimize_helps`；`< −band` ⇒ `optimize_hurts`；其间 ⇒ `no_increment`。
- 整轮：`increment_established`（`formal_gain_claim=true`）当且仅当 **≥2 对可判且同向 optimize_helps
  且 无可判对为 optimize_hurts**；否则 `no_increment_observed` / `unknown`。
- `effect_threshold = 2.5%`（沿用 R2，用户定）；`band` 取各任务当窗 preflight 噪声，逐任务冻结。

## 9. 停止规则（沿用 R2）

- 探针 `action != "start"` ⇒ hold，不开测、不耗预算。
- 契约/源码哈希不符 ⇒ 拒配。
- 连续两对无 valid L3 ⇒ 停，整轮 `unknown`，不追加第三对。
- 预算冻结，中途不许抬高去凑阈值。

## 10. 预算（已随正式计划冻结）

正式计划 `b000d8d6…` 冻结：2 对、每臂 1 轮、最多 4 次模型调用、单次输出上限 32768、
总输出预留 131072、调用墙钟 1000 s、campaign 1800 s、每候选最多 2 次验证。
模型预算计入搜索和生成的实际派发，每臂各留一次独立终评；聚合器只认计划冻结值，
中途不抬高去凑阈值。

## 11. R2 静默降级威胁——冻结前必须核查

[R2 可解释性风险](../status/v08/README.md)：fp32 计算路径无 PTX 证据时 roofline 出 `compute_used="unknown"`，
下游 prompt 格式化 TypeError 被宽 except 吞掉 ⇒ **静默回退规则搜索，模型调用没发生，metrics 无降级记录**。
已归档的 R2 两批经查未受影响。P2 的本地 mock 已用异常验证回退路径可记录。真机驱动接入后，按轮保存
搜索走 LLM 或规则的路径、回退原因和模型派发回执，供配对判读核查。

## 12. 边界与对外表述（继承）

- 实验结论适用于登记的 H20、模型、输入形状及固定轮数的优化条件。
- 不写「性能全面达标」「显著优于 DSH」；不把单形状历史手写核读数归因于插件。
- `formal_gain_claim` 在量具里恒 False，只有聚合器在满足 §8 规则后才给。
- 交付数增益（v6）与本轮性能增量分属两轴，互不替代。

## 13. 决策状态（2026-11-15，2026-11-16 更新）

| 项 | 状态 | 定值 |
|---|---|---|
| 首轮任务 | **已定** | FP16 LayerNorm 一对接线试跑（attention/融合族不进首轮） |
| 效应阈值 | **已定** | 2.5%（沿用 R2）且须超当窗噪声带 |
| 整轮判据 | **已定** | ≥2 对有效且同向 optimize_helps，无 optimize_hurts |
| 起点核来源 | **源码已冻住**（11-16） | 27B/medium 生成并修复步幅，源码 `cb851b67bc11530fe4da0b7043f5aaf2fb296c95dedd4b529409bff60d65ee96`。两臂共用。 |
| 内层模型宿主 | **27B 接线已验证** | `p2_seed.py` 复用 phase1 `bounded` 注入采样。起点生成使用 `medium`，优化阶段的采样设置待定。 |
| 性能基线 | **已定 A** | `pytorch_eager` 对应 `F.layer_norm`。主指标是两臂独立终评的时延差。 |
| optimize 开关接线 | **方案②，本地模拟通过** | `p2_optimize.py` 用共同的已达标描述渲染两组提示，开启组加入 Roofline 诊断块。每组保存实际渲染提示的哈希和搜索回退记录。 |
| 形状与有效性 | **32768×6144 完整 L2 通过，L3 `degraded`** | 起点源码绑定新契约 `cc9122d1…`。H20 GPU3 L2 32/32，中位时延 0.268672 ms，相对基线 1.4939×，启动开销 7.01%。当时有两个测试进程共卡。 |
| 内层输出预算 | **起点核生成已验证** | `p2_truncation_probe.py` 在简化提示下测得思考占输出的 91–95%。完整 Coder 提示的 `xhigh` 曾用满 40,960 token。实际冻结的起点核使用 `medium`。 |

32768×6144 的详细回执为 `output/v08-p2-probe-l3-32768c-20261116/verify-2/result.json`。
输入传输上限已从 256 MiB 提到 512 MiB，测量有效性仍按启动开销占比 10% 的门槛判读。

11 月 16 日恢复后，[试跑驱动](../../benchmarks/experiments/v08/p2_run.py) 已接通 Search、Coder 和独立终评。
本地模拟覆盖正常配对、起始窗口拒绝和终评窗口拒绝。真机试跑采用 Qwen3.8-27B/medium，每组一次搜索和最多三次
Coder 尝试，每次最多预留 32,768 个输出 token。两组从相同起点反馈开始，生成结束后再做一次当窗探针，随后评估两份终稿。

<a id="p2-resume"></a>

### 试跑工作顺序

1. **已离线校准两组提示。** 给两组统一任务描述和已达标状态，共用原始测速、硬件信息、通用建议和输出格式。
   用同一份输入渲染两组完整提示，校验差异范围为 Roofline 数据和诊断提示，覆盖模型搜索和规则回退。
2. **已接通端到端配对驱动并做本地试跑。** 驱动调用 `SearchAgent.run`、`CoderAgent.run`、验证和
   独立终评，绑定 32768×6144 的契约和起点源码哈希。确定试跑的模型采样、轮数及预算，按轮保存实际派发和回退。
3. **安排干净的测速窗口，先校验起点，再跑一对。** 同形状当窗探针 `action=start` 后，重测冻结源码。
   起点获得 `valid` 的独立 L3 后才花模型预算。两臂各自通过完整 L2 并获得 `valid` 的独立 L3 后，进入正式预登记。

<a id="p2-pilot-result"></a>

### 2026-11-16 首对真机试跑

| 对象 | 完整 L2 | 独立 L3 有效性 | 中位时延 |
|---|---:|---|---:|
| 起点核 | 32/32 | `valid` | 0.267744 ms |
| 关闭诊断 | 32/32 | `valid` | 0.749248 ms |
| 开启诊断 | 32/32 | `valid` | 0.399200 ms |

两臂各派发一次 Search 和一次 Coder，均没有规则回退。终评前当窗噪声带为 0.88191%。开启组时延比关闭组低
46.72%，试跑判为 `optimize_helps`。两份候选时延均高于起点核。三次独立终评均在验证预算内完成，所属远端会话全部清理。
回执为 `logs/p2-pilot.kdTGD8/run/pilot.json`。正式轮重新生成两对候选，试跑回执的 `formal_gain_claim=false`。

## 14. 正式计划与判读

冻结文件为 `logs/p2-pilot.kdTGD8/formal-plan.json`，哈希 `b000d8d6123f…`。
固定 H20 GPU3、起点源码、32768×6144 契约和试跑的 27B/medium 采样及预算。计划登记两对新候选，第一对先关闭后开启，
第二对先开启后关闭。运行 `p2_run.py` 时传入 `--formal-plan` 和对应的 `--pair-id pair-1` 或 `--pair-id pair-2`。
两对完成后，`p2_plan.judge_series` 从当窗探针、实际模型派发账本和独立终评复算相对时延差，并按 §8 判读。
结果同时记录每个终稿相对起点核的时延变化。

<a id="p2-formal-result"></a>

### 2026-11-16 正式两对结果

两对按冻结计划跑完，`p2_plan.judge_series` 聚合为 `logs/p2-formal-series-20261116.json`：

| 对 | 臂序 | off 中位 | on 中位 | 配对差 (off−on)/off | 当窗带宽 | 单对判 |
|---|---|---:|---:|---:|---:|---|
| pair-1 | off→on | 0.390544 ms | 0.357376 ms | **+8.49%** | 0.3376% | `optimize_helps` |
| pair-2 | on→off | 0.399744 ms | 0.399136 ms | **+0.15%** | 0.5621% | `no_increment` |

整轮：`decidable_pairs=2`、`helps_pairs=1`、`hurts_pairs=0` ⇒ **`no_increment_observed`**，
`formal_gain_claim=false`（§8 要求 ≥2 对同向 `optimize_helps`）。

- 两对四份终稿均完整 L2（32 case）、独立 L3 `measurement_validity=valid`；搜索全部走 LLM、
  无规则回退；每臂 2 次模型派发（Search + Coder），均在冻结预算内。
- 四份终稿都慢于起点核：pair-1 off −45.69% / on −33.32%，pair-2 on −48.97% / off −49.19%；
  起点核当窗中位 0.267936–0.268064 ms。
- 当窗门两对均 `action=start`：pair-1 起始 0.5063% / 终评前 0.3376%，pair-2 起始 0.2813% /
  终评前 0.5621%，均低于登记的 2.5% 效应阈值。

**判读**：诊断在 pair-1 带来 8.49% 的配对改善，但 pair-2 只有 0.15%（落在 0.56% 噪声带内），
两对未同向越过 2.5% 门槛，故本轮**没有可复现的诊断增量**。方向与试跑一致（诊断让 on 臂少改坏），
但「固定轮数下诊断提示能把核改快」未被确立。范围限一条 LayerNorm fp16 32768×6144 任务、
H20 GPU3、Qwen3.8-27B/medium、每臂 1 轮；不外推到其它算子族、模型或更多轮数。
