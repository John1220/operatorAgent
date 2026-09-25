"""D5 判别实验: 「按块的比值序列」到底看不看得见邻居插队, 以及截断判定的风险有多大

在**远端 GPU 机**上跑（自包含, 只用 torch + triton + 标准库）：

    scp remote_scripts/run_kernel.py remote_scripts/tolerance.py operator_agent/anticheat.py \\
        <ssh别名>:~/operator_agent/ 2>/dev/null; \\
    scp benchmarks/probe_block_series.py <ssh别名>:/tmp/ && \\
      ssh <ssh别名> "cd ~/operator_agent && CUDA_VISIBLE_DEVICES=2 \\
        ~/.local/bin/uv run python /tmp/probe_block_series.py"

要判的两个问题
--------------
① 生产口径的 `iqr/median` 对「一整块被邻居吞掉」敏感吗? doc 33 已经证明它对**单点离群**
   完全不敏感; 本次要证的是它对**成段污染**也不敏感 —— 因为 20 个块里脏 5 个, 在 30~60
   个样本的 IQR 里几乎看不见, 而 `_percentile` 的全局 median 会被其余干净样本按住。
   如果两条都成立, 那么 D5「噪声导向停止」就不能建在 iqr 上, 只能建在**块序列**上。

② 真要把固定采样数换成「判决稳定就停」, 必须先有 false-stop 率: 用同一个 run 的前 k 块
   重算判决, 与用全部块的判决比, 差多少次? 没有这条曲线, 「省墙钟且不改变结论」
   就只是推断。本脚本顺手把曲线在**受控污染**与**干净**两种 run 上各量一遍。

对照设计 (为什么这不是自我实现的预言)
------------------------------------
污染不是「希望它存在」的邻居负载, 而是**我们精确知道落在哪几个块**的人造抢占: 候选臂的
host 侧在运行时间的 55%~75% 区间睡 1.5 ms。哪些块被击中不靠预测 —— 跑完后用
「候选包装器的总调用数 − 5 次估时 − n_warm = n」反解出测量阶段的调用窗口, 再按 runner
自己报的块长切片。反解失败 (数字对不上) 会直接报 ❌, 不会把「没定位到」伪装成「没影响」。
"""
import json
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import triton
import triton.language as tl

# 被测的是**生产函数本身**, 不是它的仿写: 从已部署的 runner 里导入
sys.path.insert(0, os.path.expanduser("~/operator_agent"))
import run_kernel as rk  # noqa: E402


@triton.jit
def _scale_kernel(X, Y, n, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    tl.store(Y + o, tl.load(X + o, mask=m) * 2.0, mask=m)


def make_pair():
    dev = torch.device("cuda")
    x = torch.randn(1 << 20, device=dev)
    y = torch.empty_like(x)

    def cand():
        _scale_kernel[( (x.numel() + 1023) // 1024, )](x, y, x.numel(), BLOCK=1024)

    def base():
        torch.mul(x, 2.0, out=y)

    return cand, base


class TracingCandidate:
    """把候选臂的每一次调用记下来, 并可选地在一段时间窗内做 host 侧 sleep (=人造抢占)。

    候选臂全程在 shield_torch 内, 所以这里不能碰 torch 原生算子 —— triton launch 是允许的
    (屏蔽清单按本文件的既有约定放行 triton 路径)。
    """

    def __init__(self, fn, dirty_from=None, dirty_to=None, dirty_ms=0.0):
        self.fn = fn
        self.calls = 0
        self.dirty_from = dirty_from
        self.dirty_to = dirty_to
        self.dirty_ms = dirty_ms
        self.n_dirty = 0

    def __call__(self):
        self.calls += 1
        if (self.dirty_from is not None and self.calls > self.dirty_from
                and self.calls <= self.dirty_to):
            self.n_dirty += 1
            time.sleep(self.dirty_ms)
        self.fn()


def verdict(ratio, min_required, band):
    """把 D4 的四态判决复刻一份用于截断对比 (FAST/SAME/SLOW, 不算 AMBIGUOUS)"""
    if ratio > min_required * (1 + band):
        return "FAST"
    if ratio < 1.0 / max(min_required, 1e-9) * (1 - band):
        return "SLOW"
    return "SAME"


def block_medians(series, block):
    out = []
    for i in range(0, len(series), block):
        seg = series[i:i + block]
        if seg:
            out.append(statistics.median(seg))
    return out


def run_once(label, warmup=3, repeat=30, dirty_from=None, dirty_to=None, dirty_ms=0.0):
    cand, base = make_pair()
    # 候选臂的每一次调用都经过包装器: 总调用数是一个**可观测量**, 用它反解测量窗口
    tc = TracingCandidate(cand, dirty_from, dirty_to, dirty_ms)
    c_stats, b_stats, paired = rk._bench_pair_interleaved(tc, base, warmup, repeat, True)
    n_meas = paired["paired_runs"]
    block = paired["paired_block_samples"]
    # 反解: total = 5(估时) + n_warm + n_meas
    n_warm = tc.calls - 5 - n_meas
    if n_warm < 1:
        print(f"[{label}] ❌ 调用数反解失败: total={tc.calls} n_meas={n_meas} "
              f"→ 估时那 5 次可能没跑 (est 失败), 无法定位测量窗口")
        return None
    return {"label": label, "paired": paired, "block": block, "n": n_meas,
            "total_calls": tc.calls, "n_warm": n_warm, "n_dirty": tc.n_dirty,
            "cand_median": c_stats.get("median_time_ms"),
            "base_median": b_stats.get("median_time_ms"),
            "block_medians": paired.get("block_ratio_medians")}


def main() -> int:
    torch.cuda.init()
    print(f"[probe] 设备: {torch.cuda.get_device_name()}")
    print(f"[probe] _bench_pair_interleaved 来自已部署的 runner: {os.path.abspath(rk.__file__)}")

    clean = run_once("干净")
    if clean is None:
        return 1
    bm = clean["block_medians"]
    print(f"\n[干净] n={clean['n']} 块长={clean['block']} 块数={len(bm)} "
          f"候选 median={clean['cand_median']:.5f} ms 基线 median={clean['base_median']:.5f} ms")
    print(f"  比值序列 (每块 median): {[round(x, 4) for x in bm[:8]]} ... "
          f"{[round(x, 4) for x in bm[-3:]]}")
    print(f"  全局 iqr/median = {clean['paired'].get('speedup_paired_iqr_over_median')}")

    # 人造抢占: 打在中间那一段, 覆盖若干整块。窗口用**绝对调用序号**表示, 偏移量取自
    # 上面那次干净 run 反解出的结构 (5 次估时 + n_warm 次热身), 两次 run 的 n 一致时
    # 就是精确的; 不一致会在下面的定位里暴露出来。
    span = max(1, int(clean["n"] * 0.20))
    start = 5 + clean["n_warm"] + int(clean["n"] * 0.55)
    dirty = run_once("抢占", dirty_from=start, dirty_to=start + span, dirty_ms=0.0015)
    if dirty is None:
        return 1
    p2, blk2, bm2 = dirty["paired"], dirty["block"], dirty["block_medians"]
    hit_first = max(0, (start - 5 - dirty["n_warm"])) // blk2
    hit_last = min(len(bm2) - 1, (start + span - 5 - dirty["n_warm"]) // blk2)
    print(f"\n[抢占] 注入了 {dirty['n_dirty']} 次候选调用 (=1.5 ms host sleep), "
          f"应落在第 {hit_first}~{hit_last} 块")
    print(f"  n 从 {clean['n']} 变成 {dirty['n']} (块数 {len(bm)} → {len(bm2)}) —— "
          f"同一个核、同一台卡, 时间预算折算出的样本数本身就不稳, 这条对「固定次数 vs "
          f"按预算加采样」的对比是独立发现")
    clean_blocks = [v for i, v in enumerate(bm2) if not (hit_first <= i <= hit_last)]
    dirty_blocks = [v for i, v in enumerate(bm2) if hit_first <= i <= hit_last]
    if not dirty_blocks or not clean_blocks:
        print("  ❌ 定位失败: 被击中的块或干净块为空, 本条不作数")
        return 1
    med_clean = statistics.median(clean_blocks)
    # 方向: 比值 = 基线/候选, 所以「候选被拖慢」让比值**变小**。第一版这里写的是
    # `max(dirty)/med_clean`, 于是量具明明看见了 (×0.49), 脚本却报「看不见」——
    # 判据方向写反比判据缺失更难发现, 因为输出的 ❌ 看起来像是在自我批评。
    worst = min(dirty_blocks)
    inflation = med_clean / worst if worst > 0 else float("inf")
    print(f"  干净块 median={med_clean:.4f} 被污染块最差={worst:.4f} "
          f"→ 块级放大 ×{inflation:.2f} (候选被拖慢那么多倍)")
    print(f"  全局: 候选 median {clean['cand_median']:.5f} → {dirty['cand_median']:.5f} ms "
          f"(×{dirty['cand_median'] / clean['cand_median']:.2f})")
    print(f"  iqr/median: {clean['paired'].get('speedup_paired_iqr_over_median')} → "
          f"{p2.get('speedup_paired_iqr_over_median')}")

    print("\n=== 判读 ①: 全局信号看不看得见这段已知污染 ===")
    shift = abs(dirty["cand_median"] / clean["cand_median"] - 1)
    ok_block = inflation >= 2.0
    ok_blind = shift < 0.10
    print(f"  块级放大 ×{inflation:.2f} ≥2 → 序列**看得见**"
          if ok_block else f"  ❌ 块级放大只有 ×{inflation:.2f}: 序列也看不见, 粒度或注入都不可信")
    print(f"  全局 median 只动 {shift * 100:.2f}% <10% → 聚合口径**看不见**"
          if ok_blind else f"  ⚠️ 全局 median 动了 {shift * 100:.1f}%: 聚合口径也能看见, "
                           f"那 D5 用现成的 iqr 就够, 不必引入块序列")

    print("\n=== 判读 ②: 截断判决的 false-stop 曲线 ===")
    for tag, series in (("干净", bm), (f"污染×{inflation:.1f}", bm2)):
        full = verdict(statistics.median(series), 1.0, 0.004)
        diffs = []
        for k in range(1, len(series) + 1):
            v = verdict(statistics.median(series[:k]), 1.0, 0.004)
            diffs.append(0 if v == full else 1)
        first_bad = next((i + 1 for i, d in enumerate(diffs) if d), None)
        print(f"  [{tag}] 全量判决={full}; 前 k 块与全量不一致的 k 有 "
              f"{sum(diffs)}/{len(diffs)} 个" +
              (f", 最早在 k={first_bad}" if first_bad else ", 一个都没有"))

    print("\n结论与下一步:")
    print("  · 若 ① 成立: D5 的可判输入是块序列, 不是 iqr/median —— 生产口径里那个数")
    print("    对成段污染同样迟钝 (这是 doc 33 「iqr 对单点离群不敏感」的第二半)。")
    print("  · ② 给的是**受控污染**下的 false-stop 率; 真随机邻居的那一份要从生产语料里")
    print("    攒 `block_ratio_medians` (现已随 L3 进 metrics), 样本不够之前不设停止规则。")
    return 0 if (ok_block and ok_blind) else 2


if __name__ == "__main__":
    raise SystemExit(main())
