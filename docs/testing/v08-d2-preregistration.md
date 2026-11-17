# D2 预登记草案：确定性参数扫描单独能不能把冻结起点核改快（Q1）

**2026-11-17 起草。** 方向2「autotune 拆分」的首轮。量具已落地并本地 mock 标定全绿，
冻结计划已生成（`output/v08-d2-20261117/sweep-plan.json`，哈希 `d6f57589b4c8…`），真机未跑。
身份与终评校验复用 [P2 预登记](v08-p2-preregistration.md) 的冻结件（同一起点核、同一契约）。

## 1. 研究问题（Q1）

> 对同一份通过完整 L2 的冻结起点核，**只做确定性参数扫描**（不动核结构、零模型调用），
> 扫出的最优参数点相对起点核的时延降幅，能否达到效应门槛并超出当窗噪声带？

Q1 是单比较（best 点 vs 起点），不是 A/B 配对系列。**Q2（拆分本身的因果增量：
sweep-only vs sweep+语义重构）明确不在首轮**，待 Q1 有读数后另登记。

## 2. 与 P1 / P2 / R2 的关系（别混轴）

| 轮 | 问什么 | 对照 | 结论 |
|---|---|---|---|
| v6 A/C | 交付数 | 裸 DSH vs 插件 | reproduced（弱宿主，一次生成可交付） |
| P1 | 一次交付核性能 parity | A 交付核 vs C 交付核 | C_worse_observed |
| P2 | 诊断提示的优化增量 | optimize on/off | no_increment_observed（四份终稿均慢于起点） |
| **D2 Q1** | **参数层单独有无余量** | **best 参数点 vs 冻结起点** | **待测** |

P2 暴露的病灶：固定预算下让 LLM 整核重写大概率改坏。D2 把「参数」与「语义」拆开，
先单独测参数层——它不烧模型预算，是最便宜的一问；负结果也有价值（参数层无余量 ⇒
剩余余量在结构层，语义层的门槛要据此重定）。

## 3. 拆分定义

- **参数层（本轮，零模型调用）**：冻结核结构不动，机械枚举真实 launch 旋钮
  （BLOCK_SIZE × num_warps），逐点注入源码 → 每点完整 L2 → L3 测速 → 择优。
- **语义层（不在本轮）**：改核结构（单遍归约、persistent、融合），即现有 Search+Coder。

**为什么现有规则搜索不算参数层**：SearchAgent 的格点邻域动的是 sketch 的
`block_m/n/k`（matmul 形状），机械改写只对齐 `BLOCK_M/N/K`、明写跳过 `BLOCK_SIZE`
（[coder_pipeline.py:311](../../operator_agent/generate/coder_pipeline.py#L311)）。
起点核的真实旋钮是 wrapper 里的 `BLOCK_SIZE`/`num_warps`，现有参数搜索对它结构性失明。
D2 补的正是这层源码级确定性扫描。

## 4. 冻结扫描格（16 点，用户 2026-11-17 认）

`BLOCK_SIZE ∈ {256, 512, 1024, 2048}` × `num_warps ∈ {2, 4, 8, 16}`，轴序即枚举序
（BLOCK_SIZE 外轴、num_warps 内轴）。种子点 `(512, 4)` 在格内。N=6144，4096 以上浪费
lane，不进格。每点源码哈希在冻结件里逐点登记，16 个哈希互异。

## 5. 机械注入规则（字面量，零模型依赖）

[d2_rewrite.py](../../benchmarks/experiments/v08/d2_rewrite.py)：AST 定位 wrapper 唯一
launch 调用的 `BLOCK_SIZE`/`num_warps` 关键字，按 Constant 精确 span 替换，其余字节逐字保留。
`BLOCK_SIZE` 支持「字面量」与「赋值变量」两条解析路径（起点核是后者），每处只认唯一常量。

**fail-closed**：launch 不唯一、关键字缺失/重复、非常量、多处赋值、非 2 的幂目标 ⇒ 拒绝改写，
不猜值。**标定不变量**（本地 mock 已验）：
- 种子点 `(512,4)` 改写逐字还原起点源码（哈希 `cb851b67…` 不变）；
- `@triton.jit` 核体 AST 改写前后逐字相等；
- 差分 ≤ 2 行且每行只是数字替换；改写后回读等于目标值。

## 6. 逐点验证与出局规则

每点一次 explore 验证（`stop_after=L3`，完整 L2 32 case + 时延），绑定该点源码哈希与契约。
该 explore 派发独占一个 host task（`max_verifications=2` 上限，用 1 个 explore 槽，host task
停在 `awaiting_final`，不要求 `final_complete`——那是终评相位）。
- BLOCK_SIZE 改变 fp32 归约树 ⇒ 数值会变，正确性逐点重验（源码哈希不同，判定缓存不串）。
- **L2 挂的点记录后出局，不许「修」**（修是语义层的事，不在本轮）。
- `measurement_validity != valid`（含 degraded）的点记录后出局，不进择优。

## 7. 择优与主张分离

- **择优**：合格点（完整 L2 + valid）按 explore 中位时延取 argmin，打平取枚举序先。
- **主张读数**：best 点与起点核**各自** `purpose=final`/`stop_after=L3` 的独立终评，
  终评前重做当窗探针。explore 读数只用于选择，不进主张。host 契约要求终评所在 host task
  内先有该源码的 explore L2 通过，故每份终评内部是 `explore(L2)→final(L3)` 两次派发；
  best 与 seed 分属不同 host task/证据，独立性不变。
- best 恰为种子点 ⇒ 两份独立终评仍跑，判 `no_increment`（`best_is_seed_point=true`）。

## 8. 有效性门与判读骨架（量具写死）

可判当且仅当：best 与起点都在绑定哈希上过完整 L2、两臂 `measurement_validity=valid`、
重冻结契约 == 登记契约、各自独立终评、终评晚于当窗门。

- `diff_pct = (seed_median − best_median) / seed_median × 100`（best 更快为正）；
  `≥ 2.5%` 且 `> band` ⇒ `sweep_helps`；`< −band` ⇒ `sweep_hurts`；其间 ⇒ `no_increment`。
- `effect_threshold = 2.5%`（沿用 R2/P2）；`band` 取当窗 preflight 噪声。
- 无合格点 ⇒ `unknown`；窗口拒绝 ⇒ `held`。
- `formal_gain_claim` 在量具里恒 False；Q1 是 n=1 任务单比较，主张留给人判读。

## 9. 停止规则

- 探针 `action != start` ⇒ hold，不开测、不耗预算。
- 契约/源码哈希不符 ⇒ 拒。
- campaign 墙钟超冻结上限 ⇒ hold（`campaign_deadline`）。
- 预算冻结，中途不许抬高去凑阈值。

## 10. 预算（已随计划冻结）

`max_points=16`、每点 1 次 explore 验证、2 份独立终评（best+seed，各含 explore L2 + final L3
两次派发，共 20 次验证派发）、**实际模型调用 0 次**、单次验证墙钟 600 s、campaign 5400 s。
host 预算上限 `max_verifications=2`、`max_model_calls=1` 是 `HostTaskLimits` 的硬约束（≥2 / ≥1），
只是派发上限——零模型靠「不派发」保证，不靠把上限设 0。复算器（`replay_sweep`）见到任何
模型派发证据即拒。

## 11. 基线纪律（继承方向1）

性能基线 `pytorch_eager`（`F.layer_norm`）。**主指标是 best vs 起点的配对差，基线在差值中
对消**；绝对「赢 eager」是次要上下文。`baseline_type` 认机器值。

## 12. 边界与对外表述

- 结论限登记的 H20 **GPU7**（实跑卡；GPU3 满载后用户定换 GPU7，计划重冻绑定 GPU7 UUID
  `GPU-52ca6155…`）、LayerNorm fp16 32768×6144、16 点格、零模型调用条件；不外推其它
  算子族/形状/模型。换卡只改 D2 自身绑定，P2 的 GPU3 冻结身份不动。
- 不写「性能全面达标」「显著优于 DSH」；不把单形状读数归因于插件整体。
- **余量可能很小**：起点 0.268 ms 对 384 MiB 输入已接近带宽限（两遍读+一遍写），参数层未必
  有 ≥2.5% 空间。真正的结构余量在「减一遍读」，那属语义层（Q2），不在本轮主张。
- Q1 负结果 ⇒ 「参数层无余量」，把问题精确推到结构层；不读成「插件无价值」（交付数轴另立）。

## 13. 决策状态（2026-11-17）

| 项 | 状态 | 定值 |
|---|---|---|
| 首轮范围 | **用户定** | 只答 Q1；Q2 另登记 |
| 扫描格 | **用户认** | 16 点（§4） |
| 注入方式 | **用户定** | 字面量机械注入（零模型依赖） |
| 测量卡 | **用户定** | GPU3 满载 → 换 GPU7（UUID `GPU-52ca6155…`） |
| 起点核 / 契约 | 继承 P2 | 源码 `cb851b67…`，契约 `cc9122d1…`（32768×6144） |
| 量具 | **已落地 + 真机标定** | d2_rewrite/d2_plan/d2_sweep + 43 项测试；P1/P2/噪声回归合计 101 过 |
| 冻结计划 | **已生成（GPU7）** | `output/v08-d2-gpu7-20261117/sweep-plan.json`，哈希 `d8c2f735fd70…` |
| 真机 | **已跑（2026-11-17）** | `output/v08-d2-gpu7-20261117/sweep3/`；复算器重验 58 件证据一致 |
| Q1 读数 | **no_increment** | best=bs1024-w4 0.264128ms vs 起点 0.267632ms，快 1.31%（>band 0.714%，<阈 2.5%）；15/16 合格；formal_gain_claim=False；主张留给人 |

真机暴露并修复的三个量具缺陷（离线 mock 未抓到，均已重标定 + 复算重验）：
1. **host 预算契约**：`HostTaskLimits` 要求 `max_verifications≥2`、`max_model_calls≥1`；终评
   host task 内须 explore(L2) 先行再 final(L3)。live_evaluation 改为 explore 点单次 explore(L3)、
   终评 explore(L2)+final(L3)；零模型靠不派发保证（上限非用满）。
2. **参考服务自污染**：`_establish_reference` 的 persistent runner_server linger 成同卡 harness
   peer（I68），每点判 degraded。参考标定包进 owned_worker_scope 审计回收 → peer 归零。
3. **复算器终评结构**：replay_sweep 改认 verify-2 两条记录（explore+final），root 解析绝对路径
   （output/ 为符号链接，运行时存解析路径）。

## 14. 量具与回执索引

- [d2_rewrite.py](../../benchmarks/experiments/v08/d2_rewrite.py)：确定性改写器。
- [d2_plan.py](../../benchmarks/experiments/v08/d2_plan.py)：freeze/validate/explore_metrics/
  select_best/judge_sweep/replay_sweep。
- [d2_sweep.py](../../benchmarks/experiments/v08/d2_sweep.py)：执行器（freeze/run），注入式
  preflight/evaluate 离线可验；run 复用 p2_seed_probe 的参考身份确立与 GPU env 钉卡。
- 测试：`tests/unit/v08/test_d2_{rewrite,plan,sweep}.py`（43 项，含真实 `HostTaskLimits` 契约校验）。
- 冻结计划：`output/v08-d2-gpu7-20261117/sweep-plan.json`（仓外证据，不纳入 Git；GPU7 绑定）。
- 真机回执：`output/v08-d2-gpu7-20261117/sweep3/`（points/、best/、seed/、preflight-*.json、
  sweep.json、replay.json）；判读复算 `python -m benchmarks.experiments.v08.d2_plan judge --plan … --sweep …`。

<a id="d2-structure-probe-result"></a>

## 15. Q1 后的整行加载试验（2026-11-17）

在同一张 H20 GPU7 上复测 Q1 的 `BLOCK_SIZE=1024, num_warps=4` 最佳版，并验证一份手工改写。
改写版沿用 4 个 warp 和 FP32 统计，每行用一次 `tl.load` 读入，统计和输出复用这份数据。两份源码
使用相同的 FP16 LayerNorm `32768×6144` 契约，各自通过全部 32 项 L2 用例，独立 L3 均为 `valid`。

| 实现 | L3 中位时延 |
|---|---:|
| Q1 最佳版（本轮对照） | 0.264160 ms |
| 整行加载复用 | 0.333408 ms |

整行加载版时延高 **26.21%**，超过当窗实测噪声带 **0.51%**，判为 `structure_hurts`。进入 Q2 的
门槛是提速至少 2.5% 且高于当窗噪声。暂缓 Q2，保留 Q1 最佳版作对照。结论覆盖该形状和这一份
手工实现。全程零模型调用。

[候选源码](../../logs/d2-structure-20261117.bk5DOj/candidate.py) ·
[本地结果回执](../../logs/d2-structure-20261117.bk5DOj/trial-1/result.json)

后续先归档 D2 试验，再为一条真实融合任务验证提速空间。[v0.8.1 后续计划](../status/v081/README.md#v081-next-steps)
