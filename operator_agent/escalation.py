"""§10.3 的三条件升级门: **什么时候才许让 LLM 做结构性改动**

设计 §10.3 的原话：

> **何时才升级到 LLM 做结构性改动**（三条件同时满足，否则不许调用）：
> 1. 参数搜索已经平台化（K 轮 valid 且 SAME/AMBIGUOUS）；
> 2. 平台化有**可归因的硬限制**证据（spill-bound、smem-bound、占用率上限、bank conflict
>    报表）—— 不是"感觉调不动了"；
> 3. 还存在一个可探索的结构方向（融合 / split-K / 算法替换）。
>
> 这条门禁很重要：LLM 在自动跑里最大的价值是**结构性跳跃**，最大的风险是被当成
> 万能调参器反复重写同一份代码。

改前（CHECKLOG I53）这三条**零实现**：每一轮优化都无条件问 LLM 要结构改动，而上面那句
「最大的风险」正好就是当时的行为。本模块把三条判据写在一处，并坚持三件事：

1. **只数 valid 轮**（条件 1 的「K 轮 valid」）：一轮 `uninformative` / `degraded` 的读数
   不能替一个 valid 轮 —— 「没测准」与「调不动了」是两件事，前者不构成平台化证据。
   调用方喂进来的判决里如果混进了非 valid 轮的判决，那是调用方**用错了**（本模块不替它猜：
   判据的输入口径写在 `EscalationInputs.verdicts` 上，生产侧那一个点只会有 valid 轮）。
2. **缺证据 = 条件不成立**（fail-closed）：条件 2 只认**能指出出处**的证据；今天仓里有量具的
   只有 `smem-bound`（`resource_check` 的估算 + `HardwareCaps.shared_mem_per_sm_kb`），
   spill / bank conflict / 占用率上限**都还没有量具** —— 那就一条证据都没有、门就是关的，
   而不是用一个「感觉调不动了」的代理量把门撬开。
3. **SLOW 不算平台化**：退步是「还在动」（而且它自己有一条出路：禁用配方），
   平台化要的是「既没变好也没变坏」的平坦区。

条件 3 的方向来自**配方目录**（`optimization_planner`）：`selected_recipes` 里那些
`applied_recipes` 还没有的 id。这不是新量具 —— 判定与候选都由 planner 出，本模块只做差集。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: 三条件的原文（日志/产物里要能对回设计那一句）
COND_PLATEAU = "参数搜索已平台化 (K 轮 valid 且 SAME/AMBIGUOUS)"
COND_HARD_LIMIT = "平台化有可归因的硬限制证据"
COND_DIRECTION = "还存在可探索的结构方向"

#: 设计的 K 在**生产侧**取自停滞上限 (`Conductor.MAX_STAGNANT_ROUNDS`), 见那里的说明。
#: 这里不给默认值: 两个 K 会长成「停得比门早」(门永远开不了) 或者「门开得比停早」
#: (在同一段平台上反复放行) —— 两种都是谁也没设计过的行为。

#: 平台化容忍的判决词。SLOW 刻意不在内: 退步是「还在动」, 不是「调不动了」。
PLATEAU_VERDICTS = ("SAME", "AMBIGUOUS")


@dataclass(frozen=True)
class EscalationInputs:
    """三条件的输入。每个字段**都可以缺** —— 缺就是「这条不成立」(fail-closed)。"""

    #: 条件 1: 按时间顺序的判决词, **只放 valid 轮**（生产侧的喂法见模块 docstring）
    verdicts: Tuple[str, ...] = ()
    #: K。**必填**: 生产侧用停滞上限那一个数（见 `Conductor.MAX_STAGNANT_ROUNDS`）
    k_valid_rounds: int = 0
    #: 条件 2: 硬限制证据, 每条都要能指出出处（谁算的、拿什么算的）
    hard_limits: Tuple[str, ...] = ()
    #: 条件 3: 还没试过的结构方向（配方 id）
    untried_directions: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.k_valid_rounds < 1:
            raise ValueError("k_valid_rounds 必须 ≥ 1（K=0 会让条件 1 恒真; 不给默认值"
                             "是因为生产侧只有一个 K = 停滞上限）")


@dataclass(frozen=True)
class EscalationDecision:
    allowed: bool
    conditions: Tuple[Tuple[str, bool], ...]
    refusals: Tuple[str, ...]
    detail: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "conditions": {c: ok for c, ok in self.conditions},
            "refusals": list(self.refusals),
            "detail": dict(self.detail),
        }

    def refusal_note(self) -> str:
        return "; ".join(self.refusals)


def plateaued(verdicts: Sequence[str], k: int) -> bool:
    """最近 K 轮**全是** SAME/AMBIGUOUS？轮数不够 K ⇒ 不成立（不是平台化，是还没跑够）

    看的是**尾部** K 轮而不是「累计 K 轮」: 中间只要出现过一次 FAST/SLOW, 平台区就断了,
    后面得重新攒够 K 轮 —— 否则「跳了一下之后卡住」会被读成「一直在平台上」。
    """
    if k < 1 or len(verdicts) < k:
        return False
    return all(str(v).upper() in PLATEAU_VERDICTS for v in list(verdicts)[-k:])


def smem_bound_evidence(tile_smem_bytes: Optional[int],
                        limit_bytes: Optional[int]) -> Tuple[str, ...]:
    """`smem-bound` 证据（唯一今天有量具的那一条）。缺任一侧 ⇒ 空（= 没有证据）

    判据是「tile 已经**贴到**上界」（`est ≥ limit`，不带宽松系数）：宽松系数是给
    「拦不拦」用的，而这里是**归因**用的 —— 说「这套 tile 再也长不大了」比说
    「它离上界还有 20% 余量」强得多，后者不构成硬限制。
    """
    if not tile_smem_bytes or not limit_bytes or limit_bytes <= 0:
        return ()
    if tile_smem_bytes < limit_bytes:
        return ()
    return (f"smem-bound: 这套 tile ≈{tile_smem_bytes / 1024:.0f}KiB 已达每 SM 上界 "
            f"{limit_bytes / 1024:.0f}KiB (resource_check 估算 + HardwareCaps)",)


def untried_directions(selected_recipe_ids: Sequence[str],
                       applied_recipe_ids: Sequence[str],
                       disabled_recipe_ids: Sequence[str] = ()) -> Tuple[str, ...]:
    """planner 选中但**没试过**的配方 = 还剩的结构方向

    已禁用的不算方向: 它是被证据否掉的（`SLOW`），不是没探索的。顺序沿用 planner 的选择序
    （稳定排序不必再加一层: planner 自己按 priority 排）。
    """
    tried = set(applied_recipe_ids) | set(disabled_recipe_ids)
    seen: List[str] = []
    for rid in selected_recipe_ids:
        if rid not in tried and rid not in seen:
            seen.append(rid)
    return tuple(seen)


def decide(inputs: EscalationInputs) -> EscalationDecision:
    """三条件同时满足才放行。任一条缺证据 ⇒ 拒绝，并把**缺哪条**说清楚。"""
    cond1 = plateaued(inputs.verdicts, inputs.k_valid_rounds)
    cond2 = bool(inputs.hard_limits)
    cond3 = bool(inputs.untried_directions)
    conditions = ((COND_PLATEAU, cond1), (COND_HARD_LIMIT, cond2),
                  (COND_DIRECTION, cond3))
    refusals = [name for name, ok in conditions if not ok]
    detail = {
        "k_valid_rounds": inputs.k_valid_rounds,
        "verdicts_tail": list(inputs.verdicts)[-inputs.k_valid_rounds:],
        "valid_rounds_seen": len(inputs.verdicts),
        "hard_limits": list(inputs.hard_limits),
        "untried_directions": list(inputs.untried_directions),
    }
    return EscalationDecision(allowed=not refusals, conditions=conditions,
                              refusals=tuple(refusals), detail=detail)