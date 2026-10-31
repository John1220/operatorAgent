"""§8 失败分类与路由 —— 「价值密度最高的一层」

设计 `docs/design/autonomous-operator-agent-design.md` §8 原表:

| 类别 | 表现 | 策略 | 是否算迭代 |
|---|---|---|---|
| 编译失败 | 类型/layout/DSL 误用 | LLM 拿错误修, 便宜 | 是 |
| 正确性失败 | 数值错、边界错 | 调试模式修 (§19.3: **修 > 重写**, 省 92.8% token) | 是 |
| 性能不达标 | 能跑对但慢 | **搜索失败不是 bug**: 换配置/结构, 别修代码 | 是 |
| 基础设施 | SSH/驱动/OOM | 退避重试, **不污染学习也不计进度** | 否 |
| 静默退化 | 全过但不是你以为的 | 溯源检查拦 (§4.3) | 否 |

与既有 `failure_class` / `stage_directive` 的分工:
  * `kernel_db.failure_class_of` —— 细粒度签名桶 (interface/numeric/cheat/…),
    给 Fixer 取向与复发门用
  * **本模块** —— 粗粒度五路**动作路由**, 给 Conductor 决定「这轮该怎么办、
    算不算进度、要不要学」

§19.3 修正 2 (外部证据 Don't Regenerate, Debug): 编译通过、仅数值失败时
**修 > 重写** —— 路由里直接给出 `fix_not_rewrite=True`, 不靠调用方自觉。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from ..core import tolerance
from ..decide.measurement import TRANSPORT_KEY, TRANSPORT_UNKNOWN


# 五路动作 (§8 表的「策略」列)
ROUTE_FIX = "fix"                 # 编译失败: 就地修
ROUTE_DEBUG_FIX = "debug_fix"     # 正确性失败: 调试式修, **不重写**
ROUTE_SEARCH = "search"           # 性能不达标: 换配置/结构
ROUTE_RETRY = "retry"             # 基础设施: 退避重试
ROUTE_HOLD = "hold"               # 静默退化/参考侧无对象: 不动候选


@dataclass(frozen=True)
class Route:
    route: str
    kind: str                      # compile | numeric | performance | infra | silent
    counts_as_iteration: bool
    pollutes_learning: bool
    fix_not_rewrite: bool = False
    reason: str = ""
    #: 数值失败的**误差形态** (设计 §8.2); `None` = 判不了 (三态)。
    #: 它是**证据不是判据**: 候选的去留仍由 route/kind 决定, 形态只让 reason 说得出
    #: "该往哪儿修" —— 判歪的代价是提示指歪, 不是误杀。
    error_shape: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "route": self.route, "kind": self.kind,
            "counts_as_iteration": self.counts_as_iteration,
            "pollutes_learning": self.pollutes_learning,
            "fix_not_rewrite": self.fix_not_rewrite,
            "reason": self.reason,
            "error_shape": self.error_shape,
        }


#: 形态 → "该往哪儿修"的一句话 (设计 §8.2)。**只影响提示文本, 不影响路由决策**:
#: 这几句是给 Fixer 的处方方向, 判错了最坏是让它先看错地方, 而不会让错核通过 ——
#: 那是 `fix_not_rewrite` 与 L2 判据的活。
ERROR_SHAPE_HINTS = {
    tolerance.ERROR_SHAPE_TAIL_BLOCK:
        "检查尾块/边界守卫 (非对齐尾块、越界、mask 少盖了一块)",
    tolerance.ERROR_SHAPE_WHOLE_ROW:
        "检查行方向的索引/广播 (整行错: 少一维、轴选错、mask 写反)",
    tolerance.ERROR_SHAPE_BROADCAST_AXIS:
        "检查列方向的索引/广播 (整列错: 轴选错、mask 写反)",
    tolerance.ERROR_SHAPE_WHOLE_TENSOR:
        "整张量错: 先核对数值语义与输出缩放 (dtype / 累加精度 / 是否漏了一次 scale)",
    tolerance.ERROR_SHAPE_NONFINITE:
        "数值稳定性: 除零、溢出、未初始化 (NaN/Inf 会让其它位置统计失去意义)",
}


def _error_shape_of(results) -> Optional[str]:
    """从失败结果里读形态 —— 读的是**产物里记的那一份**, 不重算

    与 `static_guards` / `correctness_verified` 同一条纪律: 判定看的必须就是产物里记的,
    否则「记录说 A、判定按 B」这种分叉永远查不出来。
    """
    for r in results:
        shape = (getattr(r, "metrics", None) or {}).get("error_shape")
        if shape:
            return str(shape)
    return None


_INFRA_MARKERS = (
    "ssh", "timeout", "connection", "network", "oom", "out of memory",
    "transport", "deploy", "no usable remote baseline", "hardware probe",
    "backendunavailable", "connectionreset", "broken pipe",
)

_SILENT_MARKERS = (
    # 不用「参考侧无对象」这个串: 那是 `kernel_db` 的 marker, 有一条守卫在数
    # 「产它的只有 verifier, 认它的只有词表」的文件集合 (test_reference_uninformative_route)。
    "reference_uninformative", "无参考输出", "没有可比对象", "no remote reference",
)


def classify(verify_results: Iterable,
             *, backend_error: Optional[str] = None,
             lib_fallback: bool = False) -> Route:
    """把一次验证回合归到 §8 的五路。`verify_results` 是 VerificationResult 序列。

    优先级 (从高到低, 与 §8 表的危险程度对齐):
      5 静默退化 / 参考侧无对象 —— 什么都没判, 不许动候选、不许学
      4 基础设施            —— 不算迭代、不学
      3 性能不达标          —— 走搜索, **不是 bug**
      2 正确性失败          —— 调试式修, **fix_not_rewrite** (§19.3)
      1 编译失败            —— 就地修

    `lib_fallback=True` 一律按静默退化处理 (§4.2 第 1 条, 最常见的自欺)。
    """
    results = list(verify_results or ())
    if lib_fallback:
        return Route(ROUTE_HOLD, "silent", False, False, False,
                     "检测到 lib fallback: 候选不是生成代码, 拒绝一切动作")

    # 载荷级基础设施错误 (没到 VerificationResult 就炸了)
    if backend_error:
        low = backend_error.lower()
        if any(m in low for m in _INFRA_MARKERS):
            return Route(ROUTE_RETRY, "infra", False, False, False,
                         f"基础设施失败: {backend_error[:120]}")

    if not results:
        return Route(ROUTE_HOLD, "silent", False, False, False,
                     "没有验证结果: 什么都没判")

    failed = [r for r in results if not getattr(r, "passed", True)]
    if not failed:
        return Route(ROUTE_HOLD, "silent", False, False, False,
                     "全部通过: 无需动作")

    if any((getattr(r, "metrics", {}) or {}).get(TRANSPORT_KEY) == TRANSPORT_UNKNOWN
           for r in failed):
        return Route(ROUTE_HOLD, "infra", False, False, False,
                     "传输结果未知: 不修候选、不重试在飞请求; 先确认 owned runner 静默")

    if any((getattr(r, "metrics", {}) or {}).get("correctness", {}).get("verdict")
           == "unknown" for r in failed):
        return Route(ROUTE_HOLD, "silent", False, False, False,
                     "CorrectnessSuite 无法确立: 不修候选、不沉淀错误教训")

    levels = {getattr(getattr(r, "level", None), "value", str(getattr(r, "level", "")))
              for r in failed}
    texts = " | ".join((getattr(r, "error_log", "") or getattr(r, "message", "") or "")
                       for r in failed)
    low = texts.lower()

    # 5) 参考侧无对象 / 参考侧没有对象
    if any(m in texts for m in _SILENT_MARKERS) or "reference_uninformative" in low:
        return Route(ROUTE_HOLD, "silent", False, False, False,
                     "参考侧没有可比对象: 不动候选、不沉淀教训")

    # 4) 基础设施
    if any(m in low for m in _INFRA_MARKERS):
        lv_s = "+".join(sorted(levels))
        return Route(ROUTE_RETRY, "infra", False, False, False,
                     f"基础设施失败 ({lv_s}): 退避重试, 不计进度")

    # 3) 只有 L3 失败 = 能跑对但慢 → 搜索, 不是 bug
    if levels and levels <= {"l3", "L3", "l3_performance", "L3_PERFORMANCE"}:
        return Route(ROUTE_SEARCH, "performance", True, True, False,
                     "性能不达标: 搜索失败不是 bug, 换配置/结构")
    l3_only = all("L3" in str(lv).upper() or "perf" in str(lv).lower() for lv in levels)
    if l3_only:
        return Route(ROUTE_SEARCH, "performance", True, True, False,
                     "性能不达标: 搜索失败不是 bug")

    # 2) L2 失败 (且不是 L1 编译错) → 数值/边界, 调试式修
    only_l2 = all(("L2" in str(lv).upper() or "function" in str(lv).lower()) for lv in levels)
    if only_l2 and "L1" not in "".join(levels).upper():
        shape = _error_shape_of(failed)
        reason = "正确性失败: 调试式修, **不重写** (§19.3 修>重写, 省 92.8% token)"
        if shape:
            hint = ERROR_SHAPE_HINTS.get(shape, "")
            reason += f"; 误差形态={shape}" + (f" ⇒ {hint}" if hint else "")
        else:
            # 说不出来也要说 (与"空章节必须带依据"同一条纪律): 沉默会被读成"形态没问题"。
            reason += "; 误差形态未确立 (坏点分布不落在已知几类上) —— 按首个坏点定位, 别猜"
        return Route(ROUTE_DEBUG_FIX, "numeric", True, True, True, reason,
                     error_shape=shape)

    # 1) 含 L1 → 编译/接口, 就地修
    return Route(ROUTE_FIX, "compile", True, True, False,
                 "编译/接口失败: 就地修 (便宜)")
