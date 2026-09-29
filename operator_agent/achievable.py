"""可达上界: 从已有产物聚合「这台卡到底跑到过多少」(缺口 3 的分母, doc 51 §五 A 片)

为什么不是公式而是聚合: doc 42 量出可达上界是**按 shape 索引的一条曲线**, 标量分母装不下;
doc 48 又量出「离标称峰值还有多远」把「厂商手写核也到不了标称」混了进来 (94.7% vs 86.8%)。
所以分母只能从**这台卡自己的读数**里长出来。原料是 #63 (`b0505b1`) 才落进血缘的
`iteration_history[].roofline` + `gflops`。

一个曲线点的算法 (不需要新字段):
    有效带宽 GB/s = gflops / arithmetic_intensity
因为 AI = FLOPs / Bytes ⇒ FLOPs/s ÷ (FLOPs/Byte) = Byte/s。这是**有效**带宽
(按理论传输量算的), 与 roofline 的横轴同口径, 不是 profiler 计数 —— 所以它可比,
但它不是硬件实测的 HBM 利用率。

三条纪律 (每条对应本仓已经犯过的一个错):

1. **只有 `measurement_validity == "valid"` 的读数有资格定义天花板。** doc 50 §7.4 那批
   全是 `uninformative` (launch 地板占 70~72%) —— 拿它们当上限会把派发开销当成「这台卡的
   可达速率」。被排除的条数**照样报出来**: 「拒了 0 条」与「压根没被问」不许长得一样 (doc 48)。
2. **样本不够就拒绝给百分数**, 并仍然给原始散布 (doc 48 那条「分母 <4 拒绝打印百分数」)。
   这不是保守, 是因为一个 n=1 的「可达上界」会被下游当分母用, 而它的失效方式是**静默的**。
3. **没有轴的产物算「没有轴」, 不算「速率为 0」** (缺键 ≠ 零值, doc 49 §7.9 那条三态纪律)。

跑法 (读一份 output 树, 只打印事实, 不改任何判决):
    /usr/bin/python3 -B -m operator_agent.achievable [output_dir]
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# 一个格子至少要几个样本才允许打印百分数 —— 与 doc 48 的 `MIN_DENOMINATOR` 同一个数,
# 故意不注释成「经验值」: 它是个**门槛**, 改它就是在改判决资格, 必须显式。
MIN_SAMPLES_FOR_PCT = 4


@dataclass(frozen=True)
class Key:
    hardware: str
    precision: str
    kernel_type: str
    ai: float        # 算术强度, 按**观测值**分组; doc 51 §六 第 1 条: 分档还没定, 所以不分档
    l2_band: str     # "fits_l2" / "exceeds_l2" / "unknown"


@dataclass
class Cell:
    key: Key
    bandwidth_gbs: list[float] = field(default_factory=list)
    gflops: list[float] = field(default_factory=list)
    excluded: dict = field(default_factory=dict)   # 档位 -> 被排除的条数

    @property
    def n(self) -> int:
        return len(self.bandwidth_gbs)

    def peak_bandwidth_gbs(self) -> float | None:
        return max(self.bandwidth_gbs) if self.bandwidth_gbs else None

    def spread_pct(self) -> float | None:
        """同一格子内 max/min 相对中位的散布。样本 <2 时是 None (不是 0.0)。"""
        if len(self.bandwidth_gbs) < 2:
            return None
        s = sorted(self.bandwidth_gbs)
        mid = s[len(s) // 2]
        return None if mid <= 0 else (s[-1] - s[0]) / mid * 100.0


def _l2_band(bytes_transferred: float | None, l2_cache_kb: float | None) -> str:
    """工作集是否溢出 L2。缺任何一侧都返回 "unknown", 不猜。

    只做两档 (doc 51 §四 吸纳 ②): 四层 hierarchical roofline 需要每层各自的标定与读者,
    而今天这条链上只有一个轴。
    """
    if not bytes_transferred or not l2_cache_kb:
        return "unknown"
    return "fits_l2" if bytes_transferred <= l2_cache_kb * 1024.0 else "exceeds_l2"


def points_from_rounds(rounds: Iterable[dict]) -> list[tuple[Key, float, float, str]]:
    """把一轮的血缘记录换成 (格子, 有效带宽, gflops, 档位)。

    没有 `roofline` 键、没有 `gflops`、或 AI <= 0 的轮次**直接不产点** —— 调用方要单独
    统计「没有轴」的条数, 所以这里不吞成 0。
    """
    out = []
    for r in rounds or []:
        axes = r.get("roofline")
        gflops = r.get("gflops")
        if not isinstance(axes, dict) or not axes or not gflops:
            continue
        ai = axes.get("arithmetic_intensity") or 0.0
        if ai <= 0:
            continue
        key = Key(hardware=str(axes.get("hw_compute_capability") or ""),
                  precision=str(axes.get("compute_used") or ""),
                  kernel_type=str(r.get("kernel_type") or ""),
                  ai=round(float(ai), 4),
                  l2_band=_l2_band(axes.get("bytes_transferred"),
                                   axes.get("l2_cache_size_kb")))
        out.append((key, float(gflops) / float(ai), float(gflops),
                    str(r.get("measurement_validity") or "")))
    return out


def collect(artifacts_dir: str | Path) -> tuple[dict, dict]:
    """扫一棵产物树, 返回 (格子表, 统计)。

    统计里必须有「读不到」的那一类: 一个聚合器只报它成功读到多少, 就又是一份「算了没人读」。
    """
    root = Path(artifacts_dir)
    cells: dict[Key, Cell] = {}
    stats = {"result_files": 0, "rounds": 0, "no_axes": 0, "invalid_points": 0,
             "by_validity": {}, "unreadable_files": 0}
    for path in sorted(root.rglob("result.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            stats["unreadable_files"] += 1
            continue
        stats["result_files"] += 1
        rounds = data.get("iteration_history") or []
        for r in rounds:
            r.setdefault("kernel_type", ((data.get("task") or {}).get("source") or {})
                         .get("kernel_type", ""))
        stats["rounds"] += len(rounds)
        pts = points_from_rounds(rounds)
        if not pts:
            stats["no_axes"] += len(rounds)
            continue
        for key, bw, gf, validity in pts:
            stats["by_validity"][validity or "(空)"] = \
                stats["by_validity"].get(validity or "(空)", 0) + 1
            cell = cells.setdefault(key, Cell(key=key))
            if validity == "valid":
                cell.bandwidth_gbs.append(bw)
                cell.gflops.append(gf)
            else:
                # 排除不等于消失: 被排除的条数本身就是「这批机时够不够」的答案
                cell.excluded[validity or "(空)"] = cell.excluded.get(validity or "(空)", 0) + 1
                stats["invalid_points"] += 1
    return cells, stats


def report(cells: dict[Key, Cell], stats: dict) -> list[str]:
    """只打印有资格的话。没有可信样本时**明说测不到**, 不给一个看着像数的 0。"""
    lines = [f"产物 {stats['result_files']} 份 / 轮 {stats['rounds']} 条 / "
             f"没有轴的轮 {stats['no_axes']} 条 / 读不动的文件 {stats['unreadable_files']} 份",
             f"档位分布: {stats['by_validity'] or '(一个点都没有)'} / "
             f"被排除(不足以定义天花板)的点: {stats['invalid_points']} 个"]
    if not cells:
        lines.append("可达上界: **一个格子都没有** —— 产物里没有带 roofline 轴的轮次。"
                     "这不是 0, 是测不到。")
        return lines
    trusted = {k: c for k, c in cells.items() if c.n > 0}
    if not trusted:
        lines.append(f"可达上界: **测不到** —— {len(cells)} 个格子里没有一个有 valid 读数 "
                     f"(全被档位门挡住)。分母需要先补机时, 不是先补公式。")
        return lines
    for key, cell in sorted(trusted.items(), key=lambda kv: str(kv[0])):
        if cell.n >= MIN_SAMPLES_FOR_PCT:
            spread = cell.spread_pct()
            pct = f"  散布 {spread:.1f}%" if spread is not None else ""
        else:
            pct = f"  (样本 {cell.n} < {MIN_SAMPLES_FOR_PCT}, 拒绝打印任何百分数)"
        lines.append(f"  {key.kernel_type or '?'} / cc={key.hardware or '?'} / "
                     f"{key.precision or '?'} / AI={key.ai} / L2={key.l2_band}: "
                     f"峰值有效带宽 {cell.peak_bandwidth_gbs():.1f} GB/s, "
                     f"n={cell.n}, 排除 {sum(cell.excluded.values())} 个{pct}")
    return lines


def main(argv: list[str]) -> int:
    target = argv[1] if len(argv) > 1 else "output"
    cells, stats = collect(target)
    print(f"\n{'=' * 66}\n可达上界聚合 (doc 51 §五 A 片)  来源: {target}\n{'=' * 66}")
    for line in report(cells, stats):
        print(line)
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
