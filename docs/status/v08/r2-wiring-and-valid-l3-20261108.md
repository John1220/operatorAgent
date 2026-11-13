# R2 接线：A 臂预算硬门 + 统一配对比较器；attention 拿到 `valid` L3（2026-11-08）

| 项 | 值 |
|---|---|
| attention L3（S=4096） | `1.212096 ms` vs 同卡 SDPA `1.043360 ms` ⇒ **0.8608×**，`validity=valid`，8 条 case 全过 |
| 门控（S=4096） | `attention-1x16x4096x128-fp16-nocausal-cuda-event`，有效带宽 `0.1%`，`action=start`，中位 `1.210976 ms`（与 L3 差 0.09%） |
| 计划 | `operator-v08-increment-r2-attention-v2` 已冻结，hash `756f17d44e148788…`；v1（S=1024）**按名字撤回** |
| 新代码 | `benchmarks/experiments/v08/arm_gate.py`（A 臂预算硬门 + arm log）、`r2_compare.py`（配对比较器） |
| 证据 | `output/v08-attention-20261108/c5/run-c5-4096/`、`gate-attention-4096-20261108.json`、`preregister-attention-v2.json` |

## 1. S=1024 的计划为什么必须撤

同一份候选核，两个规模：

| 规模 | 候选中位 | 地板占比 | 有效性 | 能不能配对 |
|---|---:|---:|---|---|
| S=1024 | 0.09648 ms | 17.3% | `degraded` | **不能**：每次配对都非 `valid`，两对之后系列记 `unknown` |
| S=4096 | 1.21210 ms | 1.64% | **`valid`** | 能 |

地板是 20.24 µs，S=1024 的核只有 96 µs —— 这个比例是结构的，不是窗口的。所以
`operator-v08-increment-r2-attention-v1` 不是"跑砸了"，而是**设计上判不出**，在任何一对开跑之前就撤回，
理由按 `WITHDRAWN_PLAN_IDS` 的规矩具名写进计划表（与 fp32 v1 同一条纪律：撤回要指名，不能落进"编号不认识"）。

S=4096 的回执：L2 主用例 max_abs `6.10e-05`（atol `3e-4`）、8 条 case 全过；
L3 达成率 76.6%（粗估模型 `attention_coarse_v1`，AI=2048 FLOP/byte、compute-bound），
`launch_dominance=1.64%`、`dispersion_iqr_over_median=0.07%`、47 次采样。

## 2. A 臂：预算从"计划里的一句话"变成硬门

`usage-reconcile` 的 blocker 一直写着"**A 臂没有同等的调用硬门**"。这个模块就是它：

| 约束 | 行为 |
|---|---|
| 模型调用数 | 发请求**之前** `reserve()` 记账；超了抛 `arm_call_budget`，且**不留下那条被拒的派发** |
| 单次输出 | 超过 `max_output_tokens_per_call` 抛 `arm_output_cap` |
| 累计输出预留 | 超过 `total_output_token_reservations` 抛 `arm_token_budget` |
| 墙钟 | 超过 `wall_clock_s` 抛 `arm_deadline` |
| 日志 | 一次派发一行，`reserved` → `answered/failed/cancelled/abandoned`；崩溃只会留下"已开始"，不会假装没发生 |
| 采样字段 | **能记就记**：没给的项不写键，对账会以 `unrecorded` 点名，不会被填成相等 |
| 复用 | 同一路径重开直接拒绝：两段运行的预算写进一个文件，对账会把"两次预算"读成"一次超支" |

它写出来的就是 `usage-reconcile` 读的那份文档（`decide/arm_metering.normalize_arm_document`）——
A 与 B/C **同一个读取器**，这是"同口径"的前提，而不是结论。测试里直接拿真读取器过一遍：
缺 usage 的那一条让该臂 token 合计变成 `null`（半截求和长得像总量，是最容易引用错的格）。

## 3. 比较器：判据从"人读表"变成代码

`r2_compare.py` 把跑法 §3 的规则落成代码，输入只有冻结计划、两臂终评回执、可选的门控与对账回执：

| 规则 | 行为 |
|---|---|
| 计划 | 必须 `validate()` 通过且 `status=frozen`，hash 一致；撤回的编号按名字拒绝 |
| 终评 | 两臂都必须是 `verification_purpose=final`、L2 通过、有 L3 |
| 源码 | **按文件重算哈希**与回执比对；不一致 = 判的不是提交的那份代码 ⇒ 抛错停止 |
| 可比 | 两臂 L3 都 `valid` **且**契约哈希相同；否则记 reasons（`degraded` 不给差值、`uninformative` 让这一对不可判） |
| 主指标 | 终稿加速比的配对差值 + 相对百分比，与计划登记的 `min_detectable_effect_pct` 比 |
| 系列 | 连续两对不可判 ⇒ `stop=True`（用计划自己的停止规则，不写第二份） |
| 信用 | `formal_gain_claim` / `formal_comparison_credit` 恒为 `False` |

"坏输入"与"不可比"分开：源码漂移、半份回执、计划未冻结 → **抛错**（没有可判的东西）；
数据合法但不可判 → 记 reasons 并继续（那是关于窗口的结论，不是关于候选的）。

## 4. 还没做的（如实）

1. **A 臂的 agent 本身**：本轮交付的是"派发经过它就被硬管"的机制与统一判据，
   没有跑一次真实的 A 臂会话——那需要一份不带插件的 DSH profile 与一次真实模型预算。
2. **插件侧的采样字段**（`temperature/n/enable_thinking` 进账本）仍是契约 `12 → 13` 的事，
   明确被推迟；所以 `sampling_comparability` 依旧 `not_established`，配对结论只能报观测。
3. 配对需要两臂各一次真实运行；本轮只把"跑完之后怎么判"变成了可复现的代码。