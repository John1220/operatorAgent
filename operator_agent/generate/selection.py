"""§10.3 的候选选择: `score(c)` + ε-greedy + 分相预算的接口

设计 §10.3 逐字:

```
score(c) = P(beats_best | 观测) × 预期增益 / 预期 GPU 秒

选择：
  ε 概率 → 最不确定 / 未探索区域（信息优先）
  1-ε    → 期望增益最高（收益优先）
```

本仓 2026-10-02 之前只有 `Bandit.order`（UCB1，确定性的「无观测先试 + 有观测按分数」），
而 `bandit.py` 的模块注释当时写的是「**为什么不用 ε-greedy**：它需要随机源……可复算性
就卖了」。这一片把那句话的**理由**与**做法**分开了：

* 理由仍然成立 —— **状态型**随机源确实会毁掉可复算性；
* 做法不需要随机源：ε 的抽签值取 `(seed, round)` 的**纯函数**（`explore_draw`）。
  分布上仍然是「ε 概率去最不确定的区域」，而同一份 (seed, 轮次) 必然给同一个抽签值 ——
  `seed` 与抽签值都落进产物，于是「跑完能离线复算」与 §10.3 的 ε-greedy 同时成立。

三块拦路的东西各自的三态（都不许被读成 0）：

| 原料 | 有 | 没有 | 为什么不能合成 |
|---|---|---|---|
| `P(beats_best)` | `wins/scored` 的 Jeffreys 平滑 | `None` = 没观测过 | 「没试过」与「试过但没赢过」的下一步动作不同（前者要探索，后者要放弃） |
| `预期增益` | 赢的那些次的平均改进 | `None` = 一次没赢过 | 「没赢过」与「赢了但增益 0」不是一回事 |
| `预期 GPU 秒` | 这条臂的平均测量秒 | `None` = 没有成本证据 ⇒ 用**全局**均值为兜底分母；全局也没有 ⇒ 分数**算不出来** | 拿 1.0 当分母就是把一个编出来的数放进排序 |

「预期 GPU 秒」的口径是**控制端观测到的测量墙钟**（`conductor` 包住 L3 那一段），
与 §10.2 的墙钟预算**同一个货币**（同一个进程、同一个 `time.monotonic()`）。它**不是**
卡上时间：真机的卡上时间由心跳/probe 那一族仪器负责，这里要的是「这一轮把秒换成数字花掉
了多少」，而预算本来就是按这个口径拦人的。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: 两个**选择模式**。刻意不叫 explore/refine: 后者是**预算相位**的名字
#: (`budget.PHASES`), 而这里的模式是「这一轮在干什么」。两者的映射只有一处
#: (`MODE_TO_PHASE`) —— 利用了就要花「精修」那一格, 所以它**不是恒等的**
#: (`exploit → refine`), 早先把这两个词当同一个名字用, 当场被 fail-closed 的
#: `oracle_spend("exploit")` 拦下来 (词汇表外的相位会炸)。
EXPLORE = "explore"      # ε 那一支: 去最不确定 / 未探索的区域 (信息优先)
EXPLOIT = "exploit"      # 1-ε 那一支: 去分数最高的那个 (收益优先)

#: 模式 → 预算相位的**唯一**映射。
MODE_TO_PHASE = {EXPLORE: "explore", EXPLOIT: "refine"}
#: 反向映射。存在的理由是一次**真事故**: `choose` 原来拿预算相位的名字 (explore/refine)
#: 去和模式名 (explore/**exploit**) 比 —— 于是 `EXPLOIT in allowed` 永远为假, 生产路径上
#: 的「利用那一支」**从来没有发生过**, 而单元用例因为传的是模式名, 全绿。两个词表混用是
#: 这类 bug 的温床, 所以这里把它变成一个**有名字的转换**, 并且在 `phases` 里拒收模式名。
PHASE_TO_MODE = {v: k for k, v in MODE_TO_PHASE.items()}


def modes_from_phases(phases: Optional[Sequence[str]]) -> Optional[Tuple[str, ...]]:
    """把**预算相位**的名字翻成选择模式。`None` = 不限 (两相都可以选)。

    标定与预留 (`calibration` / `confirm`) **不是可选的模式** ⇒ 忽略 (它们照样在
    `budget.PHASES` 里, 只是不参与候选选择); 其他词 ⇒ 当场报错 (fail-closed, 与
    `budget` 的词汇表纪律一致)。传 `"exploit"` 这种**模式名**会被拒 —— 那个词不是相位。
    """
    if phases is None:
        return None
    out = []
    for name in phases:
        if name in ("calibration", "confirm"):
            continue
        if name not in PHASE_TO_MODE:
            raise ValueError(
                f"`phases` 只接受预算相位名 {tuple(PHASE_TO_MODE)}, 收到 {name!r} —— "
                "模式名 (explore/exploit) 与相位名 (explore/refine) 不是一个词表")
        mode = PHASE_TO_MODE[name]
        if mode not in out:
            out.append(mode)
    return tuple(out)

#: ε 的默认值。设计没给具体数 ⇒ 取 0.2 偏探索: 自动跑里「未探索区域」的信息价值更高,
#: 而探索的机时由 §10.2 的 55% 那一格单独管着 (不是无限的)。
DEFAULT_EPSILON = 0.2


def explore_draw(seed: str, round_num: int) -> float:
    """ε-greedy 的抽签值 ∈ [0,1)。**纯函数** —— 同一个 (seed, 轮次) 必然同一个值。

    `seed` 由调用方给 (本次运行的 `task_id` + 后端标签), 并落进产物: 于是
    「第 3 轮为什么走了探索那一支」可以在跑完之后**逐位复算**, 而不必信任一行 print。
    """
    digest = hashlib.sha256(f"{seed}|{round_num}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def phase_of(bandit: Any, arm: Any) -> str:
    """这一轮的候选值哪一相预算: **没观测过 ⇒ 探索**, 观测过 ⇒ 精修。

    这条规则与 `choose` 的模式是同一件事的两半（探索那一支只取未观测的候选）——
    所以 `Selection.mode == phase_of(...)` 是一个可以被钉住的不变量, 而不是巧合。
    """
    mode = EXPLORE if bandit.pulls(arm) == 0 else EXPLOIT
    return MODE_TO_PHASE[mode]


@dataclass(frozen=True)
class CandidateScore:
    """一个候选的 §10.3 分数, 以及它**为什么**是这个数 (或为什么算不出来)。"""

    arm: Any
    p_beats_best: Optional[float] = None
    gain_if_win: Optional[float] = None
    expected_seconds: Optional[float] = None
    score: Optional[float] = None
    missing: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"arm": str(self.arm), "p_beats_best": self.p_beats_best,
                "gain_if_win": self.gain_if_win,
                "expected_seconds": self.expected_seconds,
                "score": self.score, "missing": self.missing}


def score_candidate(bandit: Any, arm: Any, *,
                    fallback_seconds: Optional[float] = None) -> CandidateScore:
    """`score(c) = P(beats_best) × 预期增益 / 预期秒`, 缺任何一块 ⇒ `score=None` + 理由。"""
    p = bandit.p_beats_best(arm)
    gain = bandit.gain_if_win(arm)
    seconds = bandit.mean_cost(arm)
    source = "本臂"
    if seconds is None:
        seconds = fallback_seconds
        source = "全局均值" if seconds is not None else ""
    if p is None:
        return CandidateScore(arm=arm, p_beats_best=None, gain_if_win=gain,
                              expected_seconds=seconds,
                              missing="没观测过 (没有 P(beats_best) 可谈)")
    if gain is None:
        return CandidateScore(arm=arm, p_beats_best=p, gain_if_win=None,
                              expected_seconds=seconds,
                              missing="一次都没赢过 (没有『赢了能拿多少』)")
    if seconds is None or seconds <= 0:
        return CandidateScore(arm=arm, p_beats_best=p, gain_if_win=gain,
                              expected_seconds=None,
                              missing="没有测量秒口径 (本臂与全局都没有), 分母算不出来")
    return CandidateScore(arm=arm, p_beats_best=p, gain_if_win=gain,
                          expected_seconds=seconds,
                          score=(p * gain) / seconds,
                          missing=f"分母取{source}的平均测量秒")


@dataclass
class Selection:
    """这一轮的选择: 模式、顺序、为什么 (以及被计划格子逼成什么样)。"""

    mode: str = EXPLOIT
    ranking: List[Any] = field(default_factory=list)
    scores: List[CandidateScore] = field(default_factory=list)
    draw: float = 0.0
    epsilon: float = DEFAULT_EPSILON
    forced: str = ""
    #: 人读的一行 (日志/血缘)。它是**字段**不是方法: 「这一轮为什么选它」是一份数据,
    #: 要能原样落进产物, 而不是一个每次读都重新算一遍的渲染结果。
    note: str = ""

    @property
    def chosen(self) -> Any:
        """顺序里的第一个 —— 调用方可能还要往后试 (模板路径: 建不出来就下一个)。"""
        return self.ranking[0] if self.ranking else None

    def as_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode, "phase": MODE_TO_PHASE[self.mode], "draw": self.draw,
                "epsilon": self.epsilon, "forced": self.forced, "note": self.note,
                "ranking": [str(a) for a in self.ranking],
                "scores": [s.as_dict() for s in self.scores]}


def choose(candidates: Sequence[Any], bandit: Any, key, *,
           epsilon: float = DEFAULT_EPSILON, draw: float = 0.0,
           phases: Optional[Sequence[str]] = None,
           fallback_seconds: Optional[float] = None) -> Optional[Selection]:
    """从候选里按 §10.3 选一个, 并给出**完整顺序** (调用方可能往后试)。

    `phases` 是「还有计划格子的**预算相位**」(`budget.phases_with_room()`): 它**约束**模式
    —— 探索那一格用完了就只许精修, 反之亦然; 一个可选的相位都没有 ⇒ 返回 `None` (调用方
    停, 不是这里替它决定)。这一条就是「分相预算」落地的样子: 计划不是一句注释, 而是
    **限制选择的输入**。入参是相位名, 由 `modes_from_phases` 翻成模式 —— 两个词表不许混用。

    两个池子都由**观测状态**定义 (不是另立一个模式字段): 未观测 = 探索池, 观测过 = 精修池。
    某个池子空时**不硬选** —— 换到另一个池子并说明 (`forced`), 因为「没得选」与「选错了」
    在复盘里必须能分开。
    """
    items = list(candidates)
    if not items:
        return None
    converted = modes_from_phases(phases)
    allowed_modes = converted if converted is not None else (EXPLORE, EXPLOIT)
    explore_pool = [c for c in items if bandit.pulls(key(c)) == 0]
    exploit_pool = [c for c in items if bandit.pulls(key(c)) > 0]

    forced = ""
    mode: Optional[str] = None
    if EXPLORE in allowed_modes and EXPLOIT in allowed_modes:
        mode = EXPLORE if float(draw) < float(epsilon) else EXPLOIT
    elif EXPLORE in allowed_modes:
        mode, forced = EXPLORE, "精修那一格没有计划额度了 ⇒ 这一轮只能探索"
    elif EXPLOIT in allowed_modes:
        mode, forced = EXPLOIT, "探索那一格没有计划额度了 ⇒ 这一轮只能精修"

    if mode is None:
        # 两格都没有额度 (或调用方给的相位表是空的): 不是「选不出来」, 是**没得花**。
        return None
    if mode == EXPLORE and not explore_pool:
        if EXPLOIT not in allowed_modes:
            return None            # 没得换: 计划只许探索, 而未探索区域已空
        mode, forced = EXPLOIT, "未探索区域已空 ⇒ 回利用"
    elif mode == EXPLOIT and not exploit_pool:
        if EXPLORE not in allowed_modes:
            return None            # 没得换: 计划只许精修, 而一个已观测的候选都没有
        mode, forced = EXPLORE, "还没有已观测的候选 ⇒ 回探索"

    scores = [score_candidate(bandit, key(c), fallback_seconds=fallback_seconds)
              for c in items]
    by_arm = {s.arm: s for s in scores}

    # 两个顺序都用**已有的** `Bandit.order` 那一套口径 (无观测的先试, 有观测的按 UCB1),
    # 只在「利用那一支」把有分数的那批按 §10.3 分数重排 —— 不另写第三个排序器。
    ucb_order = bandit.order(items, key=key)
    prior_index = {id(c): i for i, c in enumerate(items)}
    scored = [c for c in items if by_arm[key(c)].score is not None]
    scored.sort(key=lambda c: (-by_arm[key(c)].score, prior_index[id(c)]))
    unscored = [c for c in ucb_order if by_arm[key(c)].score is None]

    if mode == EXPLORE:
        # 探索那一支的顺序**就是排序器的输出** (`Bandit.order`: 无观测的先试、各自保持
        # 先验顺序), 只是把已观测的那批挪到后面。这一句是刻意的: 排序权威在排序器里,
        # 不在这里用「原列表顺序」重排一遍 —— 否则「顺序来自排序器」这条接线就没有观察面。
        ranking = ([c for c in ucb_order if bandit.pulls(key(c)) == 0]
                   + [c for c in ucb_order if bandit.pulls(key(c)) > 0])
    else:
        ranking = [c for c in scored if bandit.pulls(key(c)) > 0] + unscored
        if not ranking:
            ranking = list(ucb_order)

    sel = Selection(mode=mode, ranking=ranking, scores=scores, draw=float(draw),
                    epsilon=float(epsilon), forced=forced)
    sel.note = (_note(sel, by_arm, key))
    return sel


def _note(sel: Selection, by_arm: Dict[Any, CandidateScore], key) -> str:
    head = sel.chosen
    if head is None:
        return "没有可选的候选"
    lost = [s for s in sel.scores if s.score is None]
    parts = [f"{sel.mode.upper()} (抽签 {sel.draw:.3f} vs ε={sel.epsilon:.2f})",
             f"选中 {head if not hasattr(head, 'template_id') else _point_label(head)}"]
    if sel.forced:
        parts.append(sel.forced)
    if lost:
        reasons = sorted({s.missing for s in lost})
        parts.append(f"{len(lost)} 个候选没有分数: " + "; ".join(reasons))
    chosen_score = by_arm.get(key(head))
    if chosen_score is not None and chosen_score.score is not None:
        parts.append(f"score={chosen_score.score:.4g} "
                     f"(P={chosen_score.p_beats_best:.2f} × 增益"
                     f"{chosen_score.gain_if_win:.3f} / {chosen_score.expected_seconds:.3f}s)")
    return " | ".join(parts)


def _point_label(cand: Any) -> str:
    from ..core.candidates import point_label      # 局部 import: 避免模块级循环依赖
    return point_label(cand)