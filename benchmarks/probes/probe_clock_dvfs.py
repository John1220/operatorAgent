"""D2 判别实验: 空载/持续满载下的 SM 时钟到底动不动 (只观测, 先确认量具看得见)

在**远端 GPU 机**上跑（自包含，只用 torch + 标准库）：

    scp benchmarks/probes/probe_clock_dvfs.py <ssh别名>:~/ && \\
      ssh <ssh别名> "cd ~/operator_agent && CUDA_VISIBLE_DEVICES=2 \\
        ~/.local/bin/uv run python ~/probe_clock_dvfs.py"

要判的问题
----------
这台卡是 8 租户共享的。邻居的负载理论上会通过功耗/温度预算把我们的 SM 频率压下来
(DVFS)，那种情况下「这一轮比上一轮慢」描述的可能是**这一轮跑在 1.4 GHz、上一轮跑在
1.8 GHz**，而不是核的质量 —— 与缺口 1 处理过的两种情形同构。`docs/research/20`
第三节 D2 因此要求「每样本记一次 SM 频率」。

吸纳决策先要一个数: **本机上频率到底变不变**。若恒定, 加阈值就是写一条永不触发的
判据 (本项目已栽过三次的死条件), 该做的是把这份「空」记录下来并给出改判条件。

三道控制 —— 每一道都可能把零结果伪装成结论
------------------------------------------
`docs/research/34/35/36/39` 的同一类错误已犯四次: **拿看不见被测量级的量具报零结果**。
「时钟没变」有两成因, 读起来一模一样: (a) 真的没变, (b) 查询根本没返回可解析的值。
所以判「不变」之前必须逐个排除:

  控制 0 —— **字段在这台驱动上到底存不存在、值能不能解析**。生产代码查的是
    `clocks_throttle_reasons.active`; 新驱动把它拆成了 `clocks_event_reasons.*`。
    若返回 `[N/A]`, `read_gpu_state()` 得到 None, 而报表里 `or 0` 会把 None 折成 0 ——
    「throttle 位全 0」和「throttle 位没读到」于是长得一样。所以逐字段单独问一遍
    (⚠️ 必须分开问: `--query-gpu` 只要有一个字段不被支持就整条查询报错、返回码非 0,
     合并问会把「这台驱动没有这个字段」伪装成「查询失败」)。

  控制 1 —— **采样链路对负载有反应**。同一条查询里的 `utilization.gpu` 是已知的
    会随负载动的量。空载 ~0%、满载 ~100% → 采样活着; 若 util 也不动, 那是探针坏了,
    时钟的零结果不作数。

  控制 2 —— **功耗/温度确实被拉上去了**。时钟不动若是「离上限很远所以无事可降」,
    与「已顶到上限但驱动仍报不变」对**别的平台**的含义相反, 必须分清 —— 前者说明
    D2 在这类机器上永远不会触发, 后者说明上报口径本身不可信。

主实验
------
空载 12 s → 持续 tensor-core 满载 (8192² fp16 matmul 循环) 40 s → 恢复 12 s,
每秒采一次同一组字段, 报每段的中位/极值, 并直接给出 D2 的判定行。
"""
import argparse
import os
import statistics
import subprocess
import sys
import threading
import time

GPU_ID = os.environ.get("OPERATOR_AGENT_GPU", "2")

# 生产查询字段在前, 判别用的补充字段在后 —— 分开报, 免得把「生产看不看得到」混进
# 「这台机器上有没有」。
PROD_FIELDS = ["utilization.gpu", "clocks.sm", "clocks.mem",
               "clocks.max.sm", "clocks.max.mem", "clocks_throttle_reasons.active"]
EXTRA_FIELDS = ["pstate", "power.draw", "power.limit", "temperature.gpu",
                "clocks.applications.sm", "clocks.max.boost",
                "clocks_event_reasons.active", "utilization.memory"]
CANDIDATES = PROD_FIELDS + [f for f in EXTRA_FIELDS if f not in PROD_FIELDS]


def _ask(fields):
    """一次 nvidia-smi 调用, 返回 {字段: 原始字符串} 或 {'_raw_err': ...}"""
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--id={GPU_ID}",
             "--query-gpu=" + ",".join(fields),
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return {"_raw_err": (r.stderr or r.stdout).strip()[:120]}
        line = r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(fields):
            return {"_raw_err": f"返回 {len(parts)} 列, 期望 {len(fields)} 列"}
        return dict(zip(fields, parts))
    except Exception as e:
        return {"_raw_err": f"{type(e).__name__}: {e}"}


def parse(raw):
    """原始串 → (数值, 是否可解析)。`[N/A]`/`N/A`/空都归入「不可解析」"""
    if raw is None:
        return None, False
    if raw.lower().startswith("0x"):
        try:
            return int(raw, 16), True
        except ValueError:
            return None, False
    try:
        return int(raw), True
    except ValueError:
        try:
            return float(raw), True
        except ValueError:
            return None, False


SUPPORTED = []


def field_inventory():
    """控制 0: 逐字段问一遍, 区分「没这个字段」/「有但值 N/A」/「有但非数值」/「能用」

    ⚠️ 不能只看「parse 不成功」就归成 N/A: `pstate` 的值是 `P0` 这类字符串, 字段完全
    可用, 而 P0 正是「性能状态处于最高档」的证据 —— 误分类会把它从报表里丢掉。
    """
    global SUPPORTED
    print("=== 控制 0: 字段可得性 (逐字段单独查询) ===")
    usable, na, absent, nonnum = [], [], [], []
    for f in CANDIDATES:
        got = _ask([f])
        if "_raw_err" in got:
            print(f"  ❌ 不支持   {f:<32} {got['_raw_err'][:60]}")
            absent.append(f)
            continue
        val = got[f]
        num, ok = parse(val)
        tag = "生产用到" if f in PROD_FIELDS else "  补充  "
        head = "✅" if ok else ("⚠️ N/A" if _is_na(val) else "✅ 文本")
        print(f"  {head:<7} {tag} {f:<32} 原始={val!r:<20}")
        (usable if ok else na if _is_na(val) else nonnum).append(f)
    SUPPORTED = usable + nonnum
    prod_lost = [f for f in PROD_FIELDS if f not in usable]
    print(f"\n  可查 {len(usable) + len(nonnum)}/{len(CANDIDATES)}"
          f"   数值可用 {len(usable)}   有字段但非数值: {nonnum or '无'}"
          f"   值为 N/A: {na or '无'}   驱动不支持: {absent or '无'}")
    if prod_lost:
        print(f"  ❌ **生产查询里拿不到数值的字段**: {prod_lost}")
        print("     → 这些字段在 run_kernel 里恒为 None, 报表里被 `or 0` 折成 0,")
        print("       「全 0」其实是「没读到」。D2 关于这些字段的任何结论都不作数。")
    else:
        print("  ✅ 生产 6 个字段全部可解析 → 「时钟没变」不是量具瞎了")
    if not SUPPORTED:
        return None
    return _ask(SUPPORTED)


def _is_na(val):
    return val is None or val.strip().lower() in ("n/a", "[n/a]", "")


def load_loop(stop: threading.Event, n: int):
    """持续 tensor-core 满载: 这就是压 DVFS 的最坏情形 (功耗最高的一类核)"""
    import torch
    a = torch.randn(n, n, dtype=torch.float16, device="cuda")
    b = torch.randn(n, n, dtype=torch.float16, device="cuda")
    while not stop.is_set():
        for _ in range(20):
            a @ b
        torch.cuda.synchronize()


def summarize(label, rows):
    def col(f):
        return [parse(r.get(f))[0] for r in rows
                if f in r and parse(r.get(f))[1]]

    def fmt(vals, unit=""):
        if not vals:
            return "无可用读数"
        return (f"median={statistics.median(vals):.0f}{unit}  "
                f"范围 {min(vals):.0f}~{max(vals):.0f}{unit}  n={len(vals)}")

    print(f"\n  [{label}]")
    for f, unit in (("utilization.gpu", "%"), ("clocks.sm", " MHz"),
                    ("clocks.mem", " MHz"), ("power.draw", " W"),
                    ("temperature.gpu", " °C")):
        if f in SUPPORTED:
            print(f"    {f:<20} {fmt(col(f), unit)}")
    if "pstate" in SUPPORTED:
        print(f"    {'pstate':<20} {sorted({r.get('pstate') for r in rows if 'pstate' in r})}")
    for f in ("clocks_throttle_reasons.active", "clocks_event_reasons.active"):
        if f in SUPPORTED:
            vals = col(f)
            nz = sorted({v for v in vals if v})
            print(f"    {f:<20} 读数={len(vals)}/{len(rows)}  非零位={nz if nz else '无'}")
    return col


def sample_block(label, seconds, interval=1.0):
    rows = []
    t_end = time.time() + seconds
    while time.time() < t_end:
        got = _ask(SUPPORTED)
        if "_raw_err" not in got:
            rows.append(got)
        time.sleep(interval)
    summarize(label, rows)
    return rows


def med(rows, f):
    vals = [parse(r.get(f))[0] for r in rows if f in r and parse(r.get(f))[1]]
    return statistics.median(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idle", type=float, default=12.0)
    ap.add_argument("--busy", type=float, default=40.0)
    ap.add_argument("--recover", type=float, default=12.0)
    ap.add_argument("--n", type=int, default=8192)
    args = ap.parse_args()

    if field_inventory() is None:
        print("\n一个字段都拿不到, 后续实验没有意义。")
        return

    print(f"\n  → 启动持续 fp16 {args.n}² matmul {args.busy:.0f}s (压功耗/温度的最坏情形)")
    idle = sample_block(f"空载 {args.idle:.0f}s", args.idle)
    stop = threading.Event()
    worker = threading.Thread(target=load_loop, args=(stop, args.n), daemon=True)
    t0 = time.time()
    worker.start()
    try:
        busy = sample_block(f"满载 {args.busy:.0f}s", args.busy)
    finally:
        stop.set()
        worker.join(timeout=60)
    print(f"  → 负载结束, 实际持续 {time.time() - t0:.1f}s")
    after = sample_block(f"恢复 {args.recover:.0f}s", args.recover)

    print("\n===== 判定 =====")
    u0, u1 = med(idle, "utilization.gpu"), med(busy, "utilization.gpu")
    sees_load = u0 is not None and u1 is not None and u1 > u0 + 20
    print(f"  控制 1  采样链路: util 空载 {u0}% → 满载 {u1}%   "
          + ("✅ 活着" if sees_load else
             "❌ util 没随负载起来 → 探针坏了, 本实验的「时钟没变」不作数"))
    c0, c1 = med(idle, "clocks.sm"), med(busy, "clocks.sm")
    if c0 is None or c1 is None:
        print("  控制 0  clocks.sm 不可解析 → D2 无从判断 (别写判据)")
        return
    p1, plim = med(busy, "power.draw"), med(idle, "power.limit")
    drop = (c0 - c1) / c0
    frac = f"{p1 / plim:.0%}" if (p1 and plim) else "?"
    print(f"  控制 2  满载 SM 时钟 {c0:.0f} → {c1:.0f} MHz  降幅 {drop:+.2%}   "
          f"满载功耗 {p1} W / 上限 {plim} W = {frac}")
    thr_busy = [v for v in (parse(r.get("clocks_throttle_reasons.active"))[0]
                            for r in busy if "clocks_throttle_reasons.active" in r)
                if v]
    if thr_busy:
        print(f"  控制 2  满载期 throttle 位图非零: {sorted(set(thr_busy))}")
    if sees_load and abs(drop) < 0.005 and not thr_busy:
        why = ("离上限很远, 没有降频压力 (这类机器上 D2 永不触发)"
               if (p1 and plim and p1 < 0.9 * plim)
               else "已顶到功耗/温度上限而读数仍不变 (驱动上报口径可疑)")
        print(f"  → 本机 SM 时钟在 0%~100% 占用之间**不动**: {why}")
        print("  → D2 不加阈值 —— 这台平台上「这一轮比上一轮慢」不能用时钟解释。")
    else:
        print(f"  → 满载期时钟降幅 {drop:+.2%} 或 throttle 位非零 → D2 值得做,")
        print("    按 docs/research/20 第三节实现判定/归一化。")

    print("""
改判条件 (什么时候该回头做 D2 的判据)
------------------------------------
1. 换平台 / 换云实例: 只要 `clocks.sm` 在满载段相对空载段降幅 > 一个测量档位
   (`measurement.NOISE_BAND` 里 degraded 档是 0.4%), 跨轮比 gflops 就得先按时钟归一化。
   判据用**降幅**而不是绝对值: 锁频机器降幅为 0, 恰好说明「时钟低于上限就降级」这类
   阈值在独占机上永不触发 —— 那是一条写出来就不会响的警报。
2. `clocks_throttle_reasons.active` 出现非零位 (0x4=SW power cap, 0x8=HW slowdown 一类)
   → "慢"有外部归因, 走 `docs/research/20` C#1 的外部归因路径, 而不是让 Conductor
   把邻居造成的退步算成核的质量问题。
3. 若要在这台机器上做**归一化**而不是加阈值: 得先有跨轮时钟散布 (iqr/median > 0)。
   除以常数等于什么都没除 —— `verify_util_decontamination.py` 的
   「按时钟归一化收下 X% 的跨轮散布」就是这条的读数。
""")


if __name__ == "__main__":
    main()
