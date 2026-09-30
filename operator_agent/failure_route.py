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

    def as_dict(self) -> dict:
        return {
            "route": self.route, "kind": self.kind,
            "counts_as_iteration": self.counts_as_iteration,
            "pollutes_learning": self.pollutes_learning,
            "fix_not_rewrite": self.fix_not_rewrite,
            "reason": self.reason,
        }


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
        return Route(ROUTE_DEBUG_FIX, "numeric", True, True, True,
                     "正确性失败: 调试式修, **不重写** (§19.3 修>重写, 省 92.8% token)")

    # 1) 含 L1 → 编译/接口, 就地修
    return Route(ROUTE_FIX, "compile", True, True, False,
                 "编译/接口失败: 就地修 (便宜)")