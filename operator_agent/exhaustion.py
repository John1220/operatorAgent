"""§10.4 的「有希望的区域已穷尽」——停滞出口的**另一半**判据

设计 §10.4 逐字：

```
| **可靠停滞** | K 轮 valid 无改善，且**有希望的区域已穷尽** | 必须"穷尽"才算, 不能只因轮数 |
```

`max_stagnant_rounds` 早就实现了前半句（K 轮 valid 无改善），后半句在代码里**不存在** ——
于是循环在「轮数够了」的那一刻就宣布停滞，而设计恰恰点名这种读法是错的：
**「无法再改进」这个结论本身必须可信。**

这个模块只回答一个问题：**候选空间被覆盖了吗？** 三种答案，缺一不可：

* `exhausted`  —— 有一个**可枚举**的候选全集，且每一个都至少被观测过一次（覆盖是**列举出来**的，
  不是推断出来的）；血缘里记下试过几个 / 没试的是哪几个；
* `unexplored` —— 可枚举，但还有没试过的（附上标签，于是「为什么不宣布停滞」可复盘）；
* `unknown`    —— **无法确立**：候选空间由 LLM / 规则**按需产生**（代码型路径每轮都可能提出新的
  邻居，甚至换结构），覆盖没有可列举的全集可言。这不是「已穷尽」，也不是「还有候选」，
  所以它**不许**被读成任何一边 —— 拿它去停就是「只因轮数」，拿它去继续就是假装还有希望。

三态而不是布尔：这个仓反复记的那类失效正是「缺 ≠ 0」被压成一个位（`exclusive=None`、
`external_load=None`、`shared_mem_per_kb=0` 都是同一族）。这里被压掉的位会让**整个优化的
终止条件**变成一个不成立的结论。

## 谁在什么时候问它

代码型路径今天恒返回 `unknown`（见上）⇒ 停滞出口**不成立**（`stagnation_declined` 记账，
运行由轮次上限与 §10.2 的预算收口）。模板型路径有可枚举的参数空间点，所以它是可判的 ——
这也是 §10.3 的选择器接上之后才可能的事：候选是谁、试过没有，现在都有唯一的记账处
（`bandit.pulls`）。
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

#: 覆盖判据的三个状态（一个判定一处实现：别的模块读这两个名字，不自己拼字符串）
EXHAUSTED = "exhausted"
UNEXPLORED = "unexplored"
UNKNOWN = "unknown"

#: 候选全集的来源（产物里要能看出「穷尽」是对**哪个**集合说的）
FROM_TEMPLATE_SPACE = "template_space"
FROM_LLM_ON_DEMAND = "llm_on_demand"


@dataclass(frozen=True)
class Coverage:
    """一次覆盖判定的结果（数据，不是判决 —— 判决在调用方）"""

    state: str
    universe: str
    total: Optional[int] = None
    tried: Optional[int] = None
    untried: Tuple[str, ...] = ()
    reason: str = ""

    @property
    def exhausted(self) -> Optional[bool]:
        """三态视图：`True` / `False` / `None`(=无法确立)。

        与 `state` 是同一个事实的两种写法（这里只有一个实现），调用方想用哪个都行 ——
        但**不许**把 `None` 当 `False` 用（那会把「不知道」读成「还有希望」）。
        """
        if self.state == EXHAUSTED:
            return True
        if self.state == UNEXPLORED:
            return False
        return None

    def as_dict(self) -> Dict[str, Any]:
        return {"state": self.state, "universe": self.universe, "total": self.total,
                "tried": self.tried, "untried": list(self.untried),
                "exhausted": self.exhausted, "reason": self.reason}

    def note(self) -> str:
        if self.state == EXHAUSTED:
            return (f"候选区域已穷尽 (来源 {self.universe}: {self.tried}/{self.total} 个点"
                    f"都试过) ⇒ 停滞可以成立")
        if self.state == UNEXPLORED:
            shown = ", ".join(self.untried[:4])
            more = "" if len(self.untried) <= 4 else f" 等 {len(self.untried)} 个"
            return (f"还有 {self.total - self.tried} 个候选没试过 ({shown}{more}) ⇒ "
                    f"不许宣布停滞")
        return f"覆盖**无法确立** ({self.reason}) ⇒ 停滞条件不成立"


def undeclared(universe: str, reason: str) -> Coverage:
    """「无法确立」那一态：候选空间不可枚举 ⇒ 不许用覆盖去支持任何结论"""
    return Coverage(state=UNKNOWN, universe=universe, reason=reason)


def assess_coverage(candidates: Optional[Sequence[Any]], bandit: Any, key: Callable[[Any], Any],
                    *, label: Callable[[Any], str] = str,
                    universe: str = FROM_TEMPLATE_SPACE) -> Coverage:
    """可枚举候选的覆盖判定。

    `candidates` **必须**是这一轮真正可枚举的那批（模板型 = 剪枝后的点）。`None` 或空 ⇒
    这是「没有可枚举的候选」而不是「已穷尽」：空集上宣布穷尽正是最危险的那种自证
    （一个空队列会让每一轮都满足「已穷尽」）⇒ 返回 `unexplored` 并把 `total=0` 记下来，
    调用方自己决定这种状态意味着什么。
    """
    if candidates is None or len(candidates) == 0:
        return Coverage(state=UNEXPLORED, universe=universe, total=0, tried=0, untried=(),
                        reason="没有可枚举的候选 (空集不能算作已穷尽)")
    untried = [label(c) for c in candidates if bandit.pulls(key(c)) == 0]
    total = len(candidates)
    tried = total - len(untried)
    if not untried:
        return Coverage(state=EXHAUSTED, universe=universe, total=total, tried=tried,
                        untried=(), reason="可枚举的候选全部观测过")
    return Coverage(state=UNEXPLORED, universe=universe, total=total, tried=tried,
                    untried=tuple(untried), reason="还有未观测的候选")


@dataclass
class CoverageLedger:
    """把逐轮的覆盖判定收成一份运行级的记录（停不成的原因要能被复盘点清）"""

    rounds: int = 0
    #: 达到停滞阈值却**没有**停的次数（按键: 覆盖状态）。这是「我们没有宣布停滞」的证据 ——
    #: 没有它, 「因为没穷尽所以继续跑」与「停滞判据根本没人读」在产物里长得一样。
    declined: Dict[str, int] = field(default_factory=dict)
    last: Optional[Coverage] = None

    def record(self, coverage: Coverage, *, threshold_reached: bool) -> None:
        self.rounds += 1
        self.last = coverage
        if threshold_reached:
            self.declined[coverage.state] = self.declined.get(coverage.state, 0) + 1

    def as_dict(self) -> Dict[str, Any]:
        return {"rounds": self.rounds, "declined": dict(self.declined),
                "last": self.last.as_dict() if self.last is not None else None}