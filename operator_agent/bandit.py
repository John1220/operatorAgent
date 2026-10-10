"""§5 bandit 局部精修器 —— 让搜索吃自己这一轮的证据, 而且只吃**有资格**的那部分

设计 §5: 「bandit 主搜索」; §10.5-3: 「毒不入环」。本仓今天的搜索是: LLM 先验
(结构级判断) + 确定性格点邻域 (±1 步/维, 只 triton) + 确定性排序 + **FIFO 轮换**
(Conductor 的 `search_alternatives`)。轮换与读数**无关** —— 它免疫于脏数据, 但也**无知**:
同一批候选里哪个上次变快了、哪个上次更慢, 队列不知道。

    [Conductor] 轮换邻域候选: tiling=128x128x32 (队列剩 2 个)

这一格与 §13.5 Roofline 那个 F1 是同一族: **判据算了、没人读**。本模块补的是
「用本轮累积的证据排序」, 并在这里把资格位接上 (CHECKLOG I52: `can_tune` 此前零读者):

  * 奖励只在**这一轮有资格计入进展** (`can_count_progress is True`) 时才进统计 ——
    §10.5-3「毒不入环」在搜索侧的实现。被污染/无指称的那一轮仍然算「试过」
    (`pulls` +1), 只是回报不进均值: 「试过但没数」与被读成另外两件事都不是一回事 ——
    它不是「没试过」(那样会重复烧机时), 也不是「试过且不好」(那样会让噪声永久拉黑一个
    臂, 正是 §16 那条反模式)。
  * 调用方传**布尔** `eligible`, 不传档位字符串: 档位→后果的映射只有一处
    (`eligibility.derive_eligibility`), 搜索侧不许再解释一遍分档表。

UCB1 + 冷启动, **全确定性, 无 RNG**:

  * 没观测过的臂排在最前 (UCB 的无限乐观初始化), 它们之间**保持先验顺序** ——
    LLM/规则先验是**先验**不是采纳 (P1-4), 加了 bandit 也不改变这一点;
  * 有观测的臂按 `mean + c * sqrt(2 ln N / n)` 降序, 同分保持先验顺序 (稳定排序)。

为什么这里没有随机源 (以及 §10.3 的 ε-greedy 后来怎么落地的): 一个**状态型**随机源会让
同一份血缘重跑两次给出不同的队列, 而「跑完能离线复算」是这个仓的基本要求。UCB1 是
确定性的, 同样的历史必然给同样的顺序。

2026-10-02 (CHECKLOG I63): §10.3 的 ε-greedy **不需要**打破这条 —— 抽签值取
`(seed, round)` 的**纯函数** (`selection.explore_draw`), 于是分布上仍然是「ε 概率去最
不确定的区域」, 而重跑逐位可复算, 且 `seed` 与抽签值都落进产物。本模块仍然不带 RNG:
「ε 那半」在 `selection.py`, 这里只提供证据 (`p_beats_best` / `gain_if_win` /
`mean_cost`)。

边界 (明写, 不许被读成更强的结论):
  * 这里的证据只覆盖**本次运行**。跨运行的臂统计要有 §7 的硬件分区记忆才能谈, 而那条
    今天还没有产物 —— 所以 bandit 状态与邻居队列同命运: **不进快照** (断点续跑时机器
    状态已变, 旧排序作废, 这是 `conductor.py` 里那条注释早就写下的理由)。
  * `c` (探索系数) 是拍的, 没有标定样本。它偏大只会让排序更接近轮询 (更保守)。
  * 排序是**软后果**: 它不改 `best`、不写记忆、不禁配方。所以它不适用 `can_persist`
    那一档的严苛度; 但**读一个没有资格的数**同样不允许 (`can_tune`) —— 那是调用方的门,
    不在这层。
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional


class Bandit:
    """按臂累积证据的确定性排序器 (UCB1)。臂 = 调用方给的任意可哈希键。"""

    def __init__(self, c: float = 1.0):
        #: 探索系数。偏大 ⇒ 更接近轮询; 偏小 ⇒ 更快锁死在当前最好的臂上。
        self.c = float(c)
        #: arm -> {"pulls": 试过几次, "scored": 有几次回报被采纳, "sum": 采纳回报之和,
        #:         "wins": 采纳回报里**为正**的次数 (= 打赢了**当时**的最好),
        #:         "win_sum": 那些正回报之和, "cost_sum"/"cost_n": 这条臂花掉的测量秒}
        #
        # 「赢」的定义不是另立一个口径: 调用方给的 reward **就是**「相对本轮之前最好
        # 冻结搜索目标的改进」(显式 workload 是聚合加速比，legacy 是 GFLOPS), 所以 `reward > 0` 恰好是
        # §10.3 分数里那个 `P(beats_best)`, 不需要第二次比较。
        self.stats: Dict[Any, Dict[str, float]] = {}
        #: 试过、但回报没被采纳的次数 (没资格 / 没数)。审计用: 「有几轮的数没进统计」。
        self.unscored_updates = 0

    # ─── 记账 ──────────────────────────────────────────────────

    def update(self, arm: Any, reward: Optional[float], *, eligible: bool,
               cost_s: Optional[float] = None) -> None:
        """记一次尝试。`eligible=False` ⇒ 只记「试过」, **不记回报**。

        `reward=None` (算不出相对改进) 与 `eligible=False` (没资格) 对统计的效果一样
        (都没有证据), 但它们是两件事, 所以 `unscored_updates` 这个计数器存在, 别把它们
        合成一个「坏数」。

        `cost_s` 是这次尝试**花掉的测量秒** (控制端口径, 见 `selection.py`)。它与
        `eligible` **无关**: 一次没资格的测量同样花掉了机时, 那是事实。`None` =
        不知道 (不是 0 —— §10.2 那三态在这里同样成立)。
        """
        st = self.stats.setdefault(arm, {"pulls": 0, "scored": 0, "sum": 0.0,
                                         "wins": 0, "win_sum": 0.0,
                                         "cost_sum": 0.0, "cost_n": 0})
        st["pulls"] += 1
        ok = (eligible is True and isinstance(reward, (int, float))
              and not isinstance(reward, bool) and math.isfinite(reward))
        if (isinstance(cost_s, (int, float)) and not isinstance(cost_s, bool)
                and math.isfinite(cost_s) and cost_s > 0):
            st["cost_sum"] += float(cost_s)
            st["cost_n"] += 1
        if not ok:
            self.unscored_updates += 1
            return
        st["scored"] += 1
        st["sum"] += float(reward)
        if float(reward) > 0:
            st["wins"] += 1
            st["win_sum"] += float(reward)

    # ─── 打分与排序 ────────────────────────────────────────────

    @property
    def total_scored(self) -> int:
        return sum(int(s["scored"]) for s in self.stats.values())

    def pulls(self, arm: Any) -> int:
        """这条臂被**试过**几次 (含没被采纳的那些)。0 = 未探索区域。"""
        st = self.stats.get(arm)
        return int(st["pulls"]) if st else 0

    def p_beats_best(self, arm: Any) -> Optional[float]:
        """§10.3 分数里的 `P(beats_best | 观测)`。

        Jeffreys 平滑 `(wins + 0.5) / (scored + 1)`: 不这么做的话「试过一次、赢过一次」
        会给 `P=1.0` —— 一次观测撑不起一个必然性, 而分数正是拿它当乘数用的。
        `None` = 没有观测 (没有这个概率可谈, 不是 0)。
        """
        st = self.stats.get(arm)
        if not st or int(st["scored"]) == 0:
            return None
        return (float(st["wins"]) + 0.5) / (float(st["scored"]) + 1.0)

    def gain_if_win(self, arm: Any) -> Optional[float]:
        """§10.3 分数里的「预期增益」= **赢的那些次的平均改进**。

        刻意不用包含负回报的 `mean()`: 分数是 `P × 增益`, 负的那部分已经由 `P`(赢的
        概率) 表达了; 再把负回报混进「赢了能拿多少」就是把同一件事扣两次。
        `None` = 一次都没赢过 (不是 0: 「没赢过」与「赢了但增益为 0」不是一回事)。
        """
        st = self.stats.get(arm)
        if not st or int(st["wins"]) == 0:
            return None
        return float(st["win_sum"]) / float(st["wins"])

    def mean_cost(self, arm: Any) -> Optional[float]:
        """这条臂的**平均测量秒** (控制端口径)。`None` = 没有成本证据。"""
        st = self.stats.get(arm)
        if not st or int(st["cost_n"]) == 0:
            return None
        return float(st["cost_sum"]) / float(st["cost_n"])

    def mean_cost_all(self) -> Optional[float]:
        """所有臂的平均测量秒 —— 某条臂没有成本证据时的**兜底分母**。

        `None` = 本次运行一条成本证据都没有 ⇒ 分数里那个分母**算不出来** (那时选择器
        退回「没有分数」的那一档并明说, 而不是拿 1.0 当分母编一个分数出来)。
        """
        total = sum(float(s.get("cost_sum", 0.0)) for s in self.stats.values())
        n = sum(int(s.get("cost_n", 0)) for s in self.stats.values())
        return (total / n) if n > 0 else None

    def mean(self, arm: Any) -> Optional[float]:
        st = self.stats.get(arm)
        if not st or int(st["scored"]) == 0:
            return None                   # 无观测 ⇒ 没有均值可言 (不是 0)
        return st["sum"] / st["scored"]

    def score(self, arm: Any) -> Optional[float]:
        """UCB1 分数。`None` = 无观测 (没有均值可谈)。"""
        st = self.stats.get(arm)
        if not st or int(st["scored"]) == 0:
            return None
        n = int(st["scored"])
        total = self.total_scored
        bonus = self.c * math.sqrt(2.0 * math.log(max(total, 1)) / n)
        return self.mean(arm) + bonus

    def order(self, candidates: Iterable, key=None) -> List:
        """按 UCB1 排序候选; 无观测的排前面, 各自保持**先验顺序**。

        稳定排序 (Python 的 `sorted` 是稳定的) 就是「同分保持先验顺序」这条的实现 ——
        不需要额外的序号字段。
        """
        keyfn = key or (lambda x: x)
        items = list(candidates)
        unknown = [x for x in items if self.score(keyfn(x)) is None]
        known = [x for x in items if self.score(keyfn(x)) is not None]
        known.sort(key=lambda x: self.score(keyfn(x)), reverse=True)
        return unknown + known

    def as_dict(self) -> Dict[str, Any]:
        """落盘用 (产物里要能看见「有几轮的数进了统计」, 而不是只看见顺序)"""
        arms = {}
        for arm, st in self.stats.items():
            arms[str(arm)] = {"pulls": int(st["pulls"]), "scored": int(st["scored"]),
                              "mean": (round(st["sum"] / st["scored"], 6)
                                       if int(st["scored"]) else None),
                              "score": (round(self.score(arm), 6)
                                        if self.score(arm) is not None else None),
                              # §10.3 分数(c) 的三块原料 —— 落盘是为了让「为什么先试它」
                              # 离线可复算, 而不是只能读一行 print。
                              "wins": int(st["wins"]),
                              "p_beats_best": (round(self.p_beats_best(arm), 6)
                                               if self.p_beats_best(arm) is not None else None),
                              "gain_if_win": (round(self.gain_if_win(arm), 6)
                                              if self.gain_if_win(arm) is not None else None),
                              "mean_cost_s": (round(self.mean_cost(arm), 6)
                                              if self.mean_cost(arm) is not None else None)}
        return {"c": self.c, "total_scored": self.total_scored,
                "unscored_updates": self.unscored_updates, "arms": arms,
                "mean_cost_s": (round(self.mean_cost_all(), 6)
                                if self.mean_cost_all() is not None else None)}
