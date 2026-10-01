# 搜索侧的信念门与 bandit —— I52 那一格的下游全部收口

> 起点是一句设计承诺和一处实证的落差。设计 §4.3 写「**资格决定这个数被允许造成什么
> 后果**」，而那一格具体是这样的（CHECKLOG I52 的原话）：

> `conductor.py:1299-1322` 在「L3 未通过」分支里**无条件**把 `l3_result.metrics` 拼成
> `perf_data`（gflops/speedup/`roofline_bound`/util 全在里面），再喂给邻居排序
> （`agents.py:680` 读 `roofline_bound` 决定往大 BLOCK_K 还是小 BLOCK_M/N 排）与 LLM
> 先验（`agents.py:708` 读 `speedup`）。于是一个档位 `uninformative`、五位全 False、
> `can_tune=False` 的读数**照样决定下一轮往哪搜**。

本篇记的是这一格怎么收口的，以及收口之后**还剩什么没做**。

---

## 1. 判据从哪来（不是我发明的）

设计 §10.4 的主循环伪码就是判据本身，而且它给的答案比「停下来」精确得多：

```python
obs = oracle.evaluate(c, contract)          # 值 + 不确定度 + 溯源 + 资格
if obs.eligibility.can_tune:
    belief.update(c, obs)                   # FAST / SAME / SLOW / AMBIGUOUS
if obs.eligibility.can_count_progress:
    stagnation.update(belief)
```

`can_tune` 门的是 **`belief.update`** —— 那个数**不许进信念**，于是下一次 `propose()`
只能从**上一次有资格的读数**构成的信念出发。它**不是**「这一轮结束」：
§4.3 的表里 `DEGRADED` 那一档明写 `can_verdict = can_tune = can_report = True`，
只有 `UNINFORMATIVE` 才全 False；而「噪声轮次必须跑满预算、不许提前终止」这条策略
真机用例一直在守（`tests/test_measurement_validity.py::TestConductorIgnoresUninformativeMeasurement`）。

第一版实现我写成了「没资格 ⇒ 记停因、`break`」——**跑起来立刻被 7 条既有用例按红**：
「① 不可信测量不得被当成性能未改善 ② 不许禁用配方 ③ 跑满预算」是仓里早已定下的取向。
伪码读对之后这 7 条全绿，而且改动比我原来那版**更小**：门加在「这份读数进不进信念」上，
循环骨架一行没动。

## 2. 四处收口（同一个后果，四条通道）

一个数能「指挥下一轮往哪搜」有四条通道，漏一条门就形同虚设：

| # | 通道 | 落地 | 判据 |
|---|---|---|---|
| 1 | `perf_data`（当轮数字：gflops/speedup/roofline/util） | `conductor.py:1366-1381` | `can_tune` ⇒ 进信念并存为 `last_tunable_perf`；没资格 ⇒ **沿用上一次有资格的信念**；本次运行还没有过有资格的读数 ⇒ 一个**不含任何数字**的结构先验 + `tuning_refused` |
| 2 | `optimization_history`（会印进下一轮 prompt 的历史） | `conductor.py:1431` | 只在有资格时累加 —— 它是**信念的另一种写法**，从这条通道漏进去与从 `perf_data` 漏进去是同一件事 |
| 3 | bandit 回报统计 | `conductor.py:1343` | `can_count_progress`（§10.5-3「毒不入环」在搜索侧的唯一实现）；没资格的那一轮只记 `pulls`，回报不进均值 |
| 4 | prompt 里的数字槽 | `agents.py:714-760` | `tuning_refused` 时所有数字槽印「无可用读数」，**不许**用 `.get(k, 0)` 编一个 `0.00` —— 那是「缺键 ≠ 0」这条纪律在 prompt 里的同一件事 |

第 4 条是顺手堵的、但它是**唯一保证被模型看到的通道**：上游把数字挡在信念之外，
如果 prompt 层再把「没有数」渲染成 `0.00x`，模型会拿一个不存在的"事实"去改方向 ——
门修好了，事实照旧被发明。`target_speedup` 例外且刻意保留：它是**任务常量**（标尺），
不是观测。

## 3. bandit：`operator_agent/bandit.py`

今天的搜索顺序是「LLM 先验 + 确定性格点邻域 + 确定性排序 + **FIFO 轮换**」。轮换与读数
**无关** —— 它免疫于脏数据，但也**无知**：同一批候选里哪个上次变快了，队列不知道。
`bandit.py` 补的就是这一格，形状刻意压到最小：

* **确定性 UCB1 + 冷启动，无 RNG**。没观测过的臂排最前（UCB 的无限乐观初始化），
  它们之间**保持先验顺序**（LLM/规则先验是**先验**不是采纳，P1-4；稳定排序即实现）；
  有观测的按 `mean + c·sqrt(2 ln N / n)` 降序。
* 为什么不用 ε-greedy：它要随机源，同一份血缘重跑两次顺序就不同。而「跑完能离线复算」
  是这个仓的基本要求（血缘里刚补上资格位与永久后果，再塞个 RNG 就把可复算性卖了）。
* `update(arm, reward, *, eligible)`：调用方传**布尔**，不传档位字符串 ——
  档位→后果的映射只许有一处（`eligibility.derive_eligibility`），搜索侧不许再解释一遍。
* 臂 = **tiling 三元组**（`Conductor._arm_of`），写法与 `optimization_history`/血缘里
  那个一致。奖励口径 = **相对本轮之前最好 gflops 的改进**，与 `best_gflops`/停滞判定
  同一个量，不另立口径。
* `as_dict()` 落进 `artifact.metadata["bandit"]`：产物里必须能看见「有几轮的数进了统计」
  （`total_scored` / `unscored_updates`），否则排序就是个没有出处的黑箱 ——
  与 `stop`/`selection_rejections` 同一处置。

接线在 `conductor.py:1419`：Search 产出的候选队列过一次 `bandit.order(..., key=_arm_of)`。
**没有**接成「bandit 直接选下一个候选」：真机 L3 是裁判这件事不变，bandit 只改**顺序**。

## 4. 守卫与变异（8 处，每处都真跑过变红）

| 变异（把判据改回坏的样子） | 变红的用例 |
|---|---|
| 撤掉信念门：没资格也 `last_tunable_perf = perf_data` | `test_the_belief_is_reused_instead_of_rewritten`（信念被改写：`speedup` 变了） |
| 撤掉 history 门：没资格也往 `optimization_history` 追加 | 同上（第 3 次问话时 history 长度 1 → 2） |
| bandit 忽略 `can_count_progress`（`eligible=True`） | 同上（`arm 128x64x32` 的 `scored` 0 → 1） |
| prompt 回到 `.get(k, 0)` | `test_a_refused_reading_prints_no_numbers` |
| 队列不过 bandit | `test_the_consumed_order_is_the_bandits_output`（桩反转顺序后消费仍是先验序） |
| 臂名退化成 `id(sketch)` | 同上 + `test_order…`（同一个 tiling 的不同版本被拆成两个臂） |
| bandit 把 `None` 当 0 收下 | `test_a_none_reward_is_none_not_zero` / `test_bool_and_non_numeric_…` |
| 冷启动丢先验顺序（`reversed(unknown)`） | `test_untried_arms_keep_the_prior_order` 等 3 条 |

用例分三层，**层次是刻意的**：
`tests/test_bandit.py`（11 条，纯策略：冷启动/资格门/UCB 探索项/稳定序/确定性 + 一条
「源码里不许出现随机源」的机械守卫）、`tests/test_search_belief_gate.py`
的 prompt 组（3 条，直接看渲染出来的 prompt）、同一文件的**生产链路组**（6 条，走真
`Conductor.execute`，用记录型假 Search 看它真正收到了什么）。

**一处观测上的坑值得记**：验「信念沿用」时，`optimization_history` 是**同一个 list 对象**，
假 Search 若只 `dict(perf_data)`，首轮快照会跟着一起长 ⇒ 「有没有偷偷累加」根本观测不到；
而且只有 3 轮时 Search 只在第 1/2 轮末尾被问（append 发生在问话之后）⇒ 追加不可见。
两处都补掉之后（快照拷 list + 该用例跑 4 轮）变异才变红。**守卫不是写出来就有效，
观测口没对准就是空转。**

## 5. 明写的边界（不许被读成更强的结论）

* **bandit 的证据只覆盖本次运行**，与邻居队列同命运：**不进快照**。跨运行的臂统计要有
  §7 的硬件分区记忆，而那条今天还没有产物。理由沿用 `conductor.py` 里那句早就写下的
  原话 —— 断点续跑时机器状态已变，旧排序作废。
* `c`（探索系数）是**拍的**，没有标定样本。偏大只会让排序更接近轮询（更保守）。
* **臂空间只有 tiling**。设计 §5 要的空间是 (BM, BN, BK, stages, warps, 向量宽度,
  layout, split-K)，其余维度今天根本不在候选集里 —— §12 的 `param_space()` 仍是零实现，
  「模板后端」也还是 ⏳。
* 排序是**软后果**：不改 `best`、不写记忆、不禁配方。所以它不适用 `can_persist` 那一档
  的严苛度；但**读一个没有资格的数**同样不允许，那道门在调用方（§2 的第 1 条）。
* 没资格的那一轮**照样花机时**（设计取向如此：跑满预算、不提前终止）。本次改动**没有**
  减少任何机时，它减的是「拿一个撑不住结论的数去指挥搜索」这件事。
* **真机未复跑**。这一片全在离线假 GPU 上验（本机无卡），`LLM_API_KEY` 也没设 ⇒
  在这里能声称的只有「判据与接线有产物且变异能红」，不是「真机行为已改」。
* **`can_report` 仍然零生产读者**（`grep -rn "can_report" operator_agent/` 除
  `eligibility.py` 自身外零命中）。今天报告里的「能不能对外报数」问的是**资格位在不在**
  （收口点跑过的证据），不是这一位。I52 标题里的两项只修掉了一项，另一项仍在台账上。
* 消费端**三套词汇**（`measurement_validity`→`measurement_ok`、`measurement_can_*`、
  `eligibility.can_*`）今天仍并存。三者由同一个 validity 派生所以不会打架，但设计 §4.3
  的五位是唯一契约 —— 这条是 I52 里那件「同族的次要一件」，**未做**。
* 沿用的「上一次有资格读数」在**机器状态已变**时未必仍然可比（§9.3 的 partition 漂移）。
  今天没有门拦这一条：漂移那一路由 `measurement_validity` 的降级表达，而没资格的那一轮
  本身就没进信念 —— 这条留给 §7 的记忆层。

## 6. 文件索引与复现

| 位置 | 内容 |
|---|---|
| `operator_agent/bandit.py` | 确定性 UCB1（`update` / `score` / `order` / `as_dict`） |
| `operator_agent/conductor.py:1325-1381` | 信念门（`can_tune`）+ bandit 记账（`can_count_progress`） |
| `operator_agent/conductor.py:1419` / `:1564` / `:1612` | 队列过 bandit / 状态落盘 / 臂定义 |
| `operator_agent/agents.py:714-760` | prompt 里的「不许编数字」 |
| `operator_agent/prompts/search_optimize.md` | 新增 `{{perf_availability}}` 槽；`%` 收进值里 |
| `tests/test_bandit.py` / `tests/test_search_belief_gate.py` | 11 + 9 条 |

```bash
python3 -m pytest tests/test_bandit.py tests/test_search_belief_gate.py -q   # 20 条
python3 -m pytest tests/ -q                                                  # 全量 1697
```

台账：**CHECKLOG I52**（⚠️ → ✅，`can_report` 那半仍在）；设计状态表 §5 那一行由
「⏳ 未做」改为 ✅ 并拆出 §12 单列。