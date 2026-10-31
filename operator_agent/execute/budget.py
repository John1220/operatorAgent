"""§10.2 三种预算，分开管 —— 账本 + 那条「最后 10% 不可挪用」

设计 §10.2 的表:

| 预算 | 角色 | 典型分配 |
|---|---|---|
| wall clock | 硬上限，兜底 | 任务级 |
| **真值 oracle 次数**（GPU 测量次数） | 真正的货币 | 5% 预检标定 / 55% 探索 / 30% 精修 / **10% 最终确认** |
| LLM token | 只在结构级介入时消耗 | 独立管；超了退化成纯数值搜索 |

> 最后 10% 的"最终确认"预算**不可挪用**：任何要写进长期记忆或对外报出的结论，
> 必须留有**一次独立重测**的预算。

这个模块只做两件事：**切分**（plan 的份额）与**记账/放行**（ledger）。三种预算分开管
的落点就是三条互不相干的通道：

  * `oracle_may(kind)` —— 真值测量次数的门。**非 confirm 的开销一律不许碰预留**
    （`spent_all() + n <= total - confirm 份额`），`confirm` 只能花自己那一格。
    这是「不可挪用」的实现，也是本模块存在的唯一理由。
  * `within_wall_clock(elapsed_s)` —— 硬上限。调用方传**已经过了多少秒**，账本不读钟：
    读钟会让这个模块不可测（真跑一秒和假过一秒必须长得一样）。
  * `tokens_exhausted(used)` —— LLM token 的独立上限；超了就退化成纯数值搜索
    （调用方那侧的后果，不在账本里）。

三条边界，明写：

  1. **三种预算全部默认关**（`None`）。没设上限 ⇒ 不拦，也**不假装拦过**。理由：预算属于
     设计 §3.5 的**允许配置**那一栏（「属于预算与搜索强度，不属于安全语义」），而「本轮
     要不要花这次钱」是任务级的决定，不是模块能替调用方拍的。⚠️ 反过来说，安全语义那
     一栏**不许**照这个模式做默认关的开关（§3.5 规则 2）—— 同仓 `长期记忆的独立确认`
     曾经就是一个默认关的构造参数，2026-10-03 已改成无条件（CHECKLOG I79）。
  2. 份额切分按设计原比例，**先切预留**、余数归 refine，使 `sum(份额) == total` 恰好
     成立（账目恒等式，有用例钉住）。顺序不能反：余数归 confirm 会让预留不单调
     （total=3 有 1 次而 total=4 是 0 次 —— 取整事故，不是设计意图）。`total < 3` 时
     放不下三格，收 explore 与标定的保底；只剩 1 次预算时**全留给预留**（宁可这次不探索）。
  3. 标定（心跳）**只记账、不被拦**。让预算饿死标定，等于把「尺子有没有坏」变成一件
     可以因为缺钱而跳过的事 —— 那是 §10.5-5（agent 不能编辑自己的尺子）的同族病。
     超支只报警（`calibration_over_plan()`），该跑还得跑。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

#: 相位的闭集。设计与本模块都只有这四格；写错一个名字要当场炸，不许静默算成「别的相」。
PHASES = ("calibration", "explore", "refine", "confirm")

#: 设计 §10.2 的份额。唯一一处；改比例只需要改这里。
DEFAULT_SPLIT = {"calibration": 0.05, "explore": 0.55, "refine": 0.30, "confirm": 0.10}

#: 从这一次总预算起，预留恒 ≥ 1 次（更小的预算里三次「优先切」放不下，只能收 explore）。
MIN_TOTAL_FOR_RESERVE = 3


@dataclass(frozen=True)
class BudgetPlan:
    """三种预算的上限。全 `None` = 全关（默认）。"""

    oracle_total: Optional[int] = None
    wall_clock_s: Optional[float] = None
    llm_tokens: Optional[int] = None

    def __post_init__(self):
        for name, value in (("oracle_total", self.oracle_total),
                            ("llm_tokens", self.llm_tokens)):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须是正整数: {value!r}")
        if isinstance(self.wall_clock_s, bool) or (
                self.wall_clock_s is not None
                and (not isinstance(self.wall_clock_s, (int, float))
                     or self.wall_clock_s <= 0)):
            raise ValueError(f"wall_clock_s 必须是正数: {self.wall_clock_s!r}")

    @property
    def any_set(self) -> bool:
        return any(v is not None for v in
                   (self.oracle_total, self.wall_clock_s, self.llm_tokens))

    def phase_caps(self) -> Optional[Dict[str, int]]:
        """把 `oracle_total` 切成四格。四格之和恒等于总数（账目恒等式）。

        **切分顺序是判据的一部分**：先切 `confirm`，再 `explore`、`calibration`，
        余数给 `refine`。为什么先切预留 —— 设计说它**不可挪用**，那就不能让它成为
        「前面三格取整剩下的东西」：按 `int()` 顺次切、余数归 confirm 会让预留**不单调**
        （试过：total=3 有 1 次、total=4 反而 0 次），而「3 次预算能留 1 次、4 次留不下」
        是纯粹的取整事故，不是设计意图。次数不够时从**低优先级**收（先 calibration 的
        1 次保底、再 explore），预留那一格不动。
        """
        total = self.oracle_total
        if total is None:
            return None
        caps = {p: int(total * DEFAULT_SPLIT[p]) for p in PHASES}
        # 5% 标定在很小的预算下会被取整成 0；给到 1，因为「一次都不标定」与「标定额度
        # 用完了」在账本里会长得一样。这一格是**可收的** (收在 explore 之前)。
        caps["calibration"] = max(1, caps["calibration"])
        caps["confirm"] = max(1, caps["confirm"])
        total_used = caps["calibration"] + caps["explore"] + caps["confirm"]
        if total_used > total:
            for phase in ("calibration", "explore"):
                take = min(total_used - total, caps[phase])
                caps[phase] -= take
                total_used -= take
                if total_used <= total:
                    break
        caps["refine"] = total - (caps["calibration"] + caps["explore"]
                                  + caps["confirm"])
        if caps["refine"] < 0:            # total 小到连预留都放不下 ⇒ 预留优先
            deficit = -caps["refine"]
            take = min(deficit, caps["explore"])
            caps["explore"] -= take
            caps["refine"] = total - (caps["calibration"] + caps["explore"]
                                      + caps["confirm"])
            if caps["refine"] < 0:        # 只剩 1 次预算: 全留给预留
                caps["calibration"] = 0
                caps["explore"] = 0
                caps["refine"] = 0
                caps["confirm"] = total
        return caps

    def as_dict(self) -> Dict[str, Any]:
        return {"oracle_total": self.oracle_total, "wall_clock_s": self.wall_clock_s,
                "llm_tokens": self.llm_tokens, "split": dict(DEFAULT_SPLIT),
                "phase_caps": self.phase_caps()}


@dataclass
class BudgetLedger:
    """一次运行的账本。**每次运行新建一个**（跨任务累计会让第二个任务少花钱）。"""

    plan: BudgetPlan = field(default_factory=BudgetPlan)
    spent: Dict[str, int] = field(default_factory=lambda: {p: 0 for p in PHASES})
    #: 超计划的标定次数（心跳不被拦，但超了要看得出）。
    calibration_over_plan: int = 0
    #: 计划格子被**突破**的次数（按相计）。计划是计划，不是硬边界 —— §10.3 的选择器
    #: 拿它决定这一轮探索还是精修，但真到了「只剩一格有额度」的时候，硬停整个运行比
    #: 超一格更坏（§10.4 的「交付 best-known-good，不装作成功」说的是没得花，不是计划用完）。
    #: 记下来是为了让「计划被突破了几次」在产物里可读 —— 与 `calibration_over_plan` 同一处置。
    over_plan: Dict[str, int] = field(default_factory=dict)
    #: 因为没额度而**没花**的那几次（复盘用：谁被预算挡住了）。
    refusals: Dict[str, int] = field(default_factory=dict)
    wall_clock_checked_s: float = 0.0
    #: 最近一次读到的 LLM token 用量（**本次运行**口径，不是进程累计）。
    llm_tokens_used: int = 0
    #: 因为 token 预算用尽而没去问 LLM 的轮数（「退化成纯数值搜索」的轮数）。
    llm_degraded_rounds: int = 0

    # ─── oracle 次数 ──────────────────────────────────────────

    @property
    def caps(self) -> Optional[Dict[str, int]]:
        return self.plan.phase_caps()

    @property
    def spent_all(self) -> int:
        return sum(self.spent.values())

    def oracle_may(self, kind: str = "explore", n: int = 1) -> bool:
        """现在能不能再花 `n` 次真值测量。`kind` 必须是四相之一（写错当场炸）。"""
        if kind not in PHASES:
            raise ValueError(f"相位词汇外的 kind: {kind!r} (词汇表 {PHASES})")
        if self.plan.oracle_total is None:
            return True
        caps = self.caps
        if kind == "confirm":
            # 预留只能花在自己的格子里；也不许被别处预支。
            return self.spent["confirm"] + n <= caps["confirm"]
        # 非 confirm：一律不许碰预留 —— 这就是「最后 10% 不可挪用」。
        return self.spent_all + n <= self.plan.oracle_total - caps["confirm"]

    def phase_room(self, kind: str = "explore") -> Optional[int]:
        """**这一相自己的计划格子**还剩几次。`None` = 没有计划 (无上限)。

        与 `oracle_may` 是**两个问题**, 刻意不合成一个名字:

        * `oracle_may(kind)` = 「现在能不能花」—— 总池 + 不可挪用的预留 (硬边界);
        * `phase_room(kind)` = 「这个**计划格子**还剩多少」—— 计划, §10.3 拿它来决定
          这一轮该**探索**还是**精修** (计划里 55% / 30% 要真的体现在花的钱上)。

        合成一个名字的代价在 I63 那一片里是具体的: 用 `oracle_may` 当计划格子会得到
        「探索用完 ⇒ 整个运行停」, 而设计要的是「探索用完 ⇒ 后面都去精修」。
        """
        if kind not in PHASES:
            raise ValueError(f"相位词汇外的 kind: {kind!r} (词汇表 {PHASES})")
        caps = self.caps
        if caps is None:
            return None
        return int(caps[kind]) - int(self.spent[kind])

    def phases_with_room(self) -> tuple:
        """还剩计划格子的可用相位 (探索/精修那两个 —— 标定与预留不参与选择)。"""
        return tuple(p for p in ("explore", "refine")
                     if self.phase_room(p) is None or self.phase_room(p) > 0)

    def oracle_spend(self, kind: str = "explore", n: int = 1) -> None:
        """记账。**不检查**放行 —— 检查是 `oracle_may` 的事。

        为什么分开：标定（心跳）必须能记账却**不许**被拦（见模块 docstring 边界 3），
        所以「该不该花」与「花了多少」不能是同一个函数。
        """
        if kind not in PHASES:
            raise ValueError(f"相位词汇外的 kind: {kind!r} (词汇表 {PHASES})")
        self.spent[kind] += n
        caps = self.caps
        if caps is not None and self.spent[kind] > caps[kind]:
            self.over_plan[kind] = self.over_plan.get(kind, 0) + n

    def oracle_refuse(self, kind: str = "explore") -> None:
        self.refusals[kind] = self.refusals.get(kind, 0) + 1

    def note_calibration_run(self, n: int = 1) -> None:
        """心跳该跑就跑，只记账 + 记超支（不拦）。"""
        self.oracle_spend("calibration", n)
        caps = self.caps
        if caps is not None and self.spent["calibration"] > caps["calibration"]:
            self.calibration_over_plan += n

    # ─── wall clock ───────────────────────────────────────────

    def within_wall_clock(self, elapsed_s: float) -> bool:
        self.wall_clock_checked_s = float(elapsed_s)
        if self.plan.wall_clock_s is None:
            return True
        return float(elapsed_s) < float(self.plan.wall_clock_s)

    # ─── LLM token ────────────────────────────────────────────

    def tokens_left(self, used: int) -> Optional[int]:
        if self.plan.llm_tokens is None:
            return None
        return int(self.plan.llm_tokens) - int(used)

    def tokens_exhausted(self, used: int) -> bool:
        left = self.tokens_left(used)
        return left is not None and left <= 0

    def note_llm_usage(self, used: int) -> bool:
        """记下用量并回答「是不是超了」。超了 ⇒ 调用方退化成纯数值搜索（§10.2 第三行）。"""
        self.llm_tokens_used = int(used)
        if self.tokens_exhausted(used):
            self.llm_degraded_rounds += 1
            return True
        return False

    # ─── 审计 ─────────────────────────────────────────────────

    def as_dict(self) -> Dict[str, Any]:
        """落进产物。读者要能回答三个问题：计划是什么、花了多少、谁被挡住了。"""
        return {"plan": self.plan.as_dict(), "spent": dict(self.spent),
                "spent_all": self.spent_all,
                "phase_room": {p: self.phase_room(p) for p in PHASES},
                "over_plan": dict(self.over_plan),
                "refusals": dict(self.refusals),
                "calibration_over_plan": self.calibration_over_plan,
                "wall_clock_checked_s": round(self.wall_clock_checked_s, 3),
                "llm_tokens_used": self.llm_tokens_used,
                "llm_degraded_rounds": self.llm_degraded_rounds}