# score(c) 与 ε-greedy：可复算的抽签、两个词表的碰撞、分相预算真的限制选择（CHECKLOG I63）

Stage 3 的第三片。上一片（doc 74 / I62）把参数空间接进了主循环，顺位由**调用方**给 —— 而
「调用方」当时还不存在：`select_template_candidate(order=None)` 就退回枚举序。这一片把那句
「调用方」写出来（`operator_agent/selection.py`），并把设计 §10.3 逐字落地：

> `score(c) = P(beats_best | 观测) × 预期增益 / 预期 GPU 秒`
> 选择：ε 概率 → 最不确定 / 未探索区域；1-ε → 期望增益最高。

落地时撞出三个「纸面 vs 代码」的缺口，各自有一条**有名字的**处置。

## 1. ε-greedy 要不要随机源？——不要，因为抽签是 (seed, 轮次) 的纯函数

`bandit.py` 的模块注释里早有一句「为什么不用 ε-greedy：它需要随机源…可复算性就卖了」。
那句**理由**仍然成立（有状态随机源确实会毁掉可复算性），但它不构成禁令：

```python
def explore_draw(seed, round_num):
    digest = hashlib.sha256(f"{seed}|{round_num}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)
```

种子 = `f"{task_id}|{backend.label}"`（没有 `task_id` 用 `"<unknown>"`），落进产物
`metadata["selection_policy"] = {epsilon, seed, rule, seconds_source}`。于是**逐轮的抽签值可以
离线复算**，不必把每个随机数记下来；`tests/test_selection.py` 里那条用例就是拿产物里的
`seed` + 血缘里的 `round` 把 `draw` 算回来逐轮比对（这是「可复算」的**观察面**，不是一句
自我声明）。同时钉住 `selection.py` 里没有 `import random` / `shuffle` / `urandom`。

## 2. 一个真事故：模式名与预算相位名是两个词表

第一版 `choose` 拿**预算相位**的名字去和**选择模式**的名字比：

```python
allowed_modes = tuple(allowed)                    # ("explore", "refine") —— 相位
mode = EXPLORE if draw < epsilon else EXPLOIT     # "exploit" —— 模式
if EXPLOIT in allowed_modes:  ...                 # 永远为假!
```

`EXPLOIT == "exploit"` 而相位叫 `"refine"` ⇒ 生产路径上**利用那一支从来没有发生过**，每轮
都被「精修那一格没有计划额度了」逼回探索；而单元用例因为我自己传的是模式名，全绿。这正是
这个仓反复记的那类失效：**两个词表的碰撞不会报错，只会悄悄退化成一个分支**。

处置不是改一个字符串，而是把词表边界变成**有名字的转换 + 拒收**：

```python
MODE_TO_PHASE = {EXPLORE: "explore", EXPLOIT: "refine"}       # 利用了就要花「精修」那一格
PHASE_TO_MODE = {v: k for k, v in MODE_TO_PHASE.items()}
def modes_from_phases(phases):        # None ⇒ 不限;  calibration/confirm ⇒ 忽略 (不是可选模式)
    ...                               # 其余词 ⇒ ValueError: "不是一个词表"
```

入参也从 `allowed` 改名成 `phases`（名字里带单位）；顺带把**相位映射**写成
`MODE_TO_PHASE`。同样的错在记账处被抓过一次：`oracle_spend(round_phase)` 早先写死
`"explore"`，`phase` 一度写成 `Selection.mode` ⇒ `oracle_spend("exploit")` 当场被
`budget` 的词汇表检查炸掉（fail-closed 的词汇表救了这里）。

⚠️ 相位**不是**恒等地等于模式：`exploit → refine`。第一版注释写着「且是恒等的」，错的。

## 3. 分数三块原料各自的**三态**

| 原料 | 有 | 缺 | 「有但是零」 |
|---|---|---|---|
| `P(beats_best)` | Jeffreys `(wins+0.5)/(scored+1)` | 没观测过 ⇒ `None` | 观测过但没赢过 ⇒ `0.25`（**不是 0**：一次没赢过 ≠ 永远赢不了） |
| 预期增益 | `win_sum/wins`（只算赢的那些轮） | 一次都没赢过 ⇒ `None` | — |
| 预期秒 | 本臂平均测量秒 | 本臂没有 ⇒ 全局均值兜底（并**说明**）；全局也没有 ⇒ `None` | 成本与「有没有数」解耦：`eligible=False` 的轮次照样花秒、照样进账 |

任一块缺 ⇒ `score=None` 且 `missing` 写明是哪一块（不许拿 0 或 1.0 编一个分数出来）。
两个刻意的口径：

- **分母的货币**是**控制端观测到的测量墙钟**（`time.monotonic()` 包住 L3 那一段），与 §10.2
  的墙钟预算同一个货币 —— 不是卡上时间（那是探针/心跳那些量具的事），也不是 LLM 墙钟。
- **赢了才谈增益**：`mean(reward)` 会把「试过而且更慢」的负回报摊进来，于是「试过且不好」的臂
  看起来比没试过的臂更有吸引力 —— 那是把负证据读成正证据。所以增益只从 `wins` 那一批算。

## 4. 「分相预算」从一句注释变成**限制选择的输入**

`BudgetLedger` 早就有 `caps`（55% 探索 / 30% 精修 那一组计划格子），但它只是 `oracle_may`
里一个**总数**检查 —— 相位名只在 `spent` 的字典键里出现。这一片：

- `BudgetLedger.phase_room(kind)`：那一格**自己**还剩多少（`None` = 没有计划，不是 0）；
- `phases_with_room()`：只剩可选的两相；
- `Conductor._plan_phases()`：`phases_with_room() or None` —— 有格子就传给选择器当约束；
  **一格都没有 ⇒ `None`（不拿计划当约束）而不是空元组**。两者含义不同：空元组在 `choose`
  里是「没有可选的相位 ⇒ 不许选」，而这里要说的是「这份计划没有可分配的空间」（总数太小，
  5%/55%/30% 被 `int()` 抹平成 0）。硬边界仍只有总数与**不可挪用的最后 10%**那一条。
- `oracle_spend` 记 `over_plan[kind]`：计划被突破要**留痕**（与 `calibration_over_plan` 同一
  处置），但不停整个运行 —— 交付 best-known-good 比多花一格更重要。

不变量（用例钉住）：选择器的模式与**实际用的那个候选**的相位一致 —— 探索那一支只取未观测的
候选（相位 = `explore`），利用那一支取已观测的（相位 = `refine`）。**一个例外要看得见**：
队列按顺位实例化，排第一的点**建不出来**时会顺位回退到别的点，而那个点的观测状态可能与模式
不同。那时 `phase` 说的是「实际花了哪一格」（写实的），`selection.mode` 说的是「选择器想干
什么」，两者由 `template_selection.chosen` 与 `selection.chosen_label` 的差异显式区分。

## 5. 排序权威只有一处：选择器看**剪枝后**的候选

第一版把 `_space.points()`（全部点）喂给选择器，队列随后把剪掉的点过滤掉 ⇒「想选谁」与
「用了谁」分叉，连花哪一格预算都跟着错（真机剪枝常常只留一个点，所以这是常态而不是罕见边）。
处置：`candidates.template_points()` = 「枚举 + 便宜代理剪枝」这一对的名字（剪枝实现仍只有
`prune_by_smem` 一处），选择器与队列拿的是**同一批**候选；队列侧的 `order` 完整性检查因此
变成恒真（它守的是另一种调用方）。

同一族的一处收拢：队列**入队不排序**（保持先验顺序），排序与「取哪一个」是同一个判断，发生
在**消费那一轮**（证据最新，且一次判断一个地方做）。之前入队时排一遍、消费时再排一遍 —— 真实
`Bandit.order` 是幂等的（巧合），而一个会反转的排序器会让它露馅（`test_search_belief_gate.py`
里那个 `_ReversingBandit` 桩就是靠这一点活的）。

## 6. 产物

| 格子 | 内容 |
|---|---|
| 血缘（逐轮） | `selection`（`mode`/`phase`/`draw`/`epsilon`/`forced`/`note`/`ranking`/`scores`/`chosen_label`/`applies_to`）、`phase`（实际花的那一格）、`measurement_wall_s`（`None` = 这一轮没有真值测量） |
| `metadata["selection_policy"]` | `epsilon` / `seed` / `rule` / `seconds_source`（逐轮抽签值可离线复算） |
| `metadata["bandit"]` | 每臂 `pulls`/`wins`/`p_beats_best`/`gain_if_win`/`mean_cost_s`，顶层 `mean_cost_s`（兜底分母） |
| `metadata["budget"]` | `spent` / `phase_room` / `over_plan`（计划被突破了几次） |

`applies_to` 是必须的：模板型轮选择器选的是**本轮**要用的点（`this_round`），代码型轮选的是
**下一轮**的 sketch（`next_round`）—— 两条记录长得一样但时间指向不同，合成一格会让复盘的
人把下一轮的理由读到这一轮头上。

## 7. 变异矩阵（19 处，全红）

| # | 变异 | 结果 |
|---|---|---|
| M1 | 抽签值带上随机源（不可复算） | ✅ |
| M2 | 抽签忽略轮次（同一个值每轮） | ✅ |
| M3 | 分数用含负回报的均值当增益 | ✅ |
| M4 | 没观测过也给一个分数（拿 0 当 P） | ✅ |
| M5 | 没有秒口径就拿 1.0 当分母 | ✅ |
| M6 | 模式名与相位名不再区分（原事故） | ✅ |
| M7 | 探索那一支取裸的先验顺序（绕过排序器） | ✅ |
| M8 | 被迫换池子时不看 `phases` | ✅ |
| M9 | ε 分支写反（抽签大走探索） | ✅ |
| M10 | 相位映射换成恒等（`exploit→exploit`） | ✅ |
| M11 | 记账仍写死 `explore` | ✅ |
| M12 | 测量秒不喂给 bandit | ✅ |
| M13 | 血缘不落选择记录 | ✅ |
| M14 | 计划相位不传给选择器（计划变注释） | ✅ |
| M15 | 计划格子的取法改成空元组 | ✅ |
| M16 | 超计划不记账 | ✅ |
| M17 | 候选不先剪枝（选择器看到剪掉的点） | ✅ |
| M18 | 赢了不记（`P(beats_best)` 永远 0.5） | ✅ |
| M19 | 成本不记账（分母永远缺） | ✅ |

**第一次是 16 红 / 2 逃逸 / 1 无效**，三条都不是「变异太强」而是**观察面缺失**：

- M11（记账写死 `explore`）逃逸 ⇒ 没有用例断言「第 2 轮复用已观测的点 ⇒ `spent["refine"]>0`」。
  补 `test_the_charged_bucket_is_the_rounds_phase`（分相预算的意义就是两格的花费分别看得见）。
- M17（不先剪枝）逃逸 ⇒ 没有用例把选择器的候选集与队列的候选集对上。补
  `test_the_selector_only_sees_the_pruned_candidates`（并因此加了 `chosen_label` 这一格）。
- M14 无效：锚点 `phases=self._plan_phases())` 命中了 2 次（模板侧与消费侧）⇒ 锚点加长到带
  `_universe.kept` 那一行。

同一轮里把上两片的矩阵一起重跑（`/tmp/queue_mutations.py` 14 处、`/tmp/template_mutations.py`
12 处）：它们各有 2 处 / 1 处锚点因这一片改了签名而失效（`select_template_candidate(bandit=…)`
→ `order=…`、`prune_by_smem(points…)` → `template_points`、控制器缺口守卫已删除）—— 锚点跟着
代码走，不能用 `git checkout` 糊过去，也不许把失效记成绿色。更新后 14/14、12/12 全红。

## 8. 血统与顺带纠正

- `bandit.py` 的模块注释点了名：ε 那一支的可复算性由 `selection.explore_draw` 保证，不是
  「不用 ε-greedy」。
- `bandit.update(..., cost_s=)` 的**成本与资格解耦**：`eligible=False`（这轮没数）照样记成本，
  否则「试了但测不出数」的候选在分母上会显得免费。
- `selection.choose` 的探索那一支顺序 = **排序器的输出**（`Bandit.order`：无观测的先试、各自
  保持先验顺序），只是把已观测的挪到后面 —— 排序权威在排序器里，选择器不用「原列表顺序」重排。

## 9. 边界（➖/⏳）

- ➖ **没有真模板后端**：CUTLASS/TVM 的实例化器与其编译/测量路径仍然没有；这一片的选择器在
  假后端上跑（`tests/test_template_queue.py` 的 `_TemplateBackend`）。
- ➖ **L1 的两道静态守卫对 C++/CUDA 模板源是空转的**（`anticheat` / `interface_contract` 是
  Python 形状的，判不了就倾向放行）—— 模板源现在真的会执行，这是**最紧急**的那条边界。
- ➖ **参数轴仍只有 tiling 三维**有产出者（`stages`/`warps`/向量宽度/layout/split-K 还没有）。
- ⏳ **§10.4 的停止条件**只做了一半：「达标 / 预算耗尽 / 不安全中止」在，可靠停滞只有
  `MAX_STAGNANT_ROUNDS=2` 那一个局部常量，「有希望的区域已穷尽」还没有（它需要本片的选择器：
  「ε 那一支连续 K 轮都在同一小片区域里转」才是可判的）。
- ⏳ **`phases_with_room()` 没有生产者之外的消费者**：两格都用尽时靠 `_plan_phases()` 退化成
  「不约束」继续跑（`over_plan` 记账），而不是停下来要求重规划 —— 「计划用完 ⇒ 换相位」今天
  由 `choose` 的被迫换池子完成 ✓，但「两格都空 ⇒ 该不该停」这个决定还没人做。
- ➖ 真机未复跑（本机无 GPU）：ε、分相、成本口径都只在 Mock 与假后端上验证过。