"""D8 判别实验: 把计时线程钉到某个 CPU 核, 对 launch 地板到底有没有用

在**远端 GPU 机**上跑（自包含，只用 numpy/torch/triton + 标准库）：

    scp benchmarks/probe_cpu_affinity.py <ssh别名>:/tmp/ && \\
      ssh <ssh别名> "cd ~/operator_agent && CUDA_VISIBLE_DEVICES=2 \\
        ~/.local/bin/uv run python /tmp/probe_cpu_affinity.py"

要判的问题
----------
`docs/research/01` 测出的「launch 地板 20.24 µs」有个尴尬性质：它是 **host 侧派发**
开销而非 GPU 执行时间 —— 同一个空核，被 256 MB 冲刷填满 GPU 后事件窗口只剩 4.64 µs
(`docs/research/32` 第一节)。既然主要是 host 的 CPU 时间，那把它钉在一个核上、避开
迁移与邻居，理论上就该受益。Google Benchmark 确实把 `sched_setaffinity` 列为**无需
root** 的降噪手段（✅ 一手文档）。吸纳决策 `docs/research/20` 第三节 D8 当时标 ⏸，
留了两条待验前提：(1) 钉到的核若被别人占满会更糟；(2) 收益不明。这个探针把它们分开测。

两道控制（第一版都栽了，写在这儿免得再栽）
------------------------------------------
`docs/research/34/35/36` 的同一类错误已犯三次：拿看不见被测量级的量具报零结果。这里
的零结果特别容易伪装成「钉核没用」，所以先把量具对准两个校准信号：

  控制 1 —— **争用必须可见**：另起一个**子进程**（不能是线程：同进程内 Python 自旋受
    GIL 约束，测出来的是 GIL 争用）专门 100% 占住某个核，再把计时线程钉到同一个核上，
    量它 1 s 墙钟里烧掉多少 CPU 秒。同一核 → 应得 ~0.5；空闲核 → 应得 ~1.0。
    ⚠️ 窗口必须是**秒级**：第一版用 2.2 ms 的纯计算采样，同一核上放着 100% 的邻居，
    中位数却纹丝不动 —— 短于 CFS 时间片的采样根本换不上一次上下文切换，量不出争用。
    地板测量本身正是这种短采样（每样本 20 µs），所以**钉核若有副作用，它落在样本的
    尾部而不是中位数**。这就是为什么主实验同时报 median 和 p90/max。

  控制 2 —— **事件窗口对 host 侧耗时敏不敏感**：往 `s.record()` 与 launch 之间注入已知的
    Python 工作量。抬得动 → 窗口里确有 host 段时间，D8 有作用路径可测；抬不动 →
    这个数字与 host 忙不忙无关，D8 直接不成立。⚠️ 这条控制只证明「敏感」，**不区分**
    「我们的线程在烧 CPU」与「GPU 空等命令到达」—— doc 32 第一节的读法是后者。
    （顺带回答一个口径问题：候选核若有很重的 Python 侧 launch 开销，我们的 L3 会不会
    把它藏起来 —— 控制 2 的读数说明不会，这也是 `docs/research/20` D7 拒绝 CUDA Graph
    计时的同一个理由。）

主实验：四臂按轮交错（doc 36 的教训：机器状态会漂，顺序测的差里混着顺序偏置），
每臂每轮报 地板 median / p90 / max、生产口径 median、以及该臂自己那 0.6 s 的有效核数。
`default` 是不钉（继承全部核），`idle` 钉到按 /proc/stat 选出的最闲核，
`hog` 钉到被我们自己那个 100% 子进程独占的核，`core0` 钉到 0 号核（最朴素的钉法）。
每轮对 default 做配对差分，报中位差 + 符号检验。
"""
import argparse
import os
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
import triton
import triton.language as tl

REPS = 9
FLOOR_SAMPLES = 1500        # 20 µs × 1500 = 30 ms/臂, 尾部统计量需要样本量
LN_SAMPLES = 150
FLUSH_BYTES = 256 * 1024 * 1024      # 与 Triton do_bench / 本项目 runner 一致
CONTEND_WINDOW_S = 0.6


@triton.jit
def empty_kernel(x_ptr):
    pass


@triton.jit
def _ln(x_ptr, w_ptr, b_ptr, y_ptr, n_rows, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / n_cols
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * n_cols + cols,
             (xc * rstd * w + b).to(y_ptr.dtype.element_ty), mask=mask)


def stat_field(pid, idx):
    """/proc/<pid>/stat 的第 idx 个字段 (1-based)。comm 可能含空格，所以从最后一个
    ')' 之后重新数"""
    with open(f"/proc/{pid}/stat") as f:
        parts = f.read().rsplit(")", 1)[1].split()
    return parts[idx - 3]


def effective_cores(window_s=CONTEND_WINDOW_S):
    """这段墙钟时间里本线程真正拿到多少核 —— 争用的直接读数

    不看耗时看「烧掉的 CPU 秒 / 墙钟秒」: 被抢占时墙钟照走, CPU 秒才反映真实份额。
    utime 精度 10 ms (CLK_TCK=100), 所以窗口必须秒级, 短采样测不出来。
    """
    clk = os.sysconf("SC_CLK_TCK")
    u0 = int(stat_field(os.getpid(), 14)) / clk
    t0 = time.perf_counter()
    spins = 0
    while time.perf_counter() - t0 < window_s:
        spins += 1
    wall = time.perf_counter() - t0
    cpu = int(stat_field(os.getpid(), 14)) / clk - u0
    return cpu / wall if wall else float("nan")


def per_core_busy(cores, gap=0.3):
    def snap():
        out = {}
        with open("/proc/stat") as f:
            for line in f:
                p = line.split()
                if (p[0].startswith("cpu") and p[0][3:].isdigit()
                        and int(p[0][3:]) in cores):
                    v = [int(x) for x in p[1:]]
                    out[int(p[0][3:])] = (v[3] + v[4], sum(v))
        return out
    a = snap()
    time.sleep(gap)
    b = snap()
    return {c: 1 - (b[c][0] - a[c][0]) / (b[c][1] - a[c][1]) for c in b if c in a}


def idle_cpus(n_cpu, gap=0.4):
    """按 /proc/stat 两次快照排每核忙闲, 返回 (busy 字典, 由闲到忙排序的核列表)"""
    def snap():
        out = {}
        with open("/proc/stat") as f:
            for line in f:
                p = line.split()
                if p[0].startswith("cpu") and p[0][3:].isdigit():
                    v = [int(x) for x in p[1:]]
                    out[int(p[0][3:])] = (v[3] + v[4], sum(v))
        return out
    a = snap()
    time.sleep(gap)
    b = snap()
    busy = {}
    for c in b:
        if c not in a:
            continue
        di, dt = b[c][0] - a[c][0], b[c][1] - a[c][1]
        busy[c] = 1.0 - (di / dt if dt else 1.0)
    order = sorted(busy, key=lambda c: busy[c])
    return busy, order


def quantiles(ts):
    q50, q90 = statistics.median(ts), float(np.percentile(ts, 90))
    return q50, q90, max(ts)


def timed_launches(fn, rep):
    ts = []
    for _ in range(rep):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    return ts


def measure_host_floor(rep):
    x = torch.zeros(1024, device="cuda")
    for _ in range(50):
        empty_kernel[(1,)](x)
    torch.cuda.synchronize()
    return timed_launches(lambda: empty_kernel[(1,)](x), rep)


def measure_ln_flushed(rep):
    rows, cols = 1024, 768
    rng = np.random.default_rng(11)
    X = torch.tensor(rng.standard_normal((rows, cols)), dtype=torch.float16, device="cuda")
    w = torch.ones(cols, dtype=torch.float16, device="cuda")
    b = torch.zeros(cols, dtype=torch.float16, device="cuda")
    Y = torch.empty_like(X)
    BLOCK = triton.next_power_of_2(cols)
    flush = torch.empty(FLUSH_BYTES, dtype=torch.int8, device="cuda")

    def one():
        _ln[(rows,)](X, w, b, Y, rows, cols, 1e-5, BLOCK=BLOCK)

    for _ in range(10):
        one()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rep):
        flush.zero_()                       # 冲掉 L2, 与生产口径一致
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        one()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    return ts


def spawn_hog(cpu):
    code = ("import os, time\n"
            f"os.sched_setaffinity(0, {{{cpu}}})\n"
            "end = time.time() + 600\n"
            "x = 0\n"
            "while time.time() < end:\n"
            "    x += 1\n")
    return subprocess.Popen([sys.executable, "-c", code],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


ARMS = ("default", "idle", "hog", "core0")


def control_host_work_visible():
    """控制 2: 往计时窗口里注入已知 host 工作量, 看事件窗口抬不抬"""
    x = torch.zeros(1024, device="cuda")
    out = []
    for inject in (0, 4000, 12000):
        for _ in range(30):
            empty_kernel[(1,)](x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(200):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            acc = 0
            for _ in range(inject):          # 必须在窗口内、launch 之前
                acc += 1
            empty_kernel[(1,)](x)
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) * 1000.0)
            assert inject == 0 or acc == inject
        out.append((inject, statistics.median(ts)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--samples", type=int, default=FLOOR_SAMPLES)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("需要 CUDA (这个探针测的是 GPU 派发路径)")
        return 1
    n_cpu = os.cpu_count()
    orig_mask = set(os.sched_getaffinity(0))
    busy, order = idle_cpus(n_cpu)
    idle_cpu = order[0]
    # 关键: hog 占的核必须与 idle 臂的核**分开**, 否则 idle 臂也在争用, 四臂塌成两臂。
    # (第一版就是 hog_cpu = idle_cpu, 于是 idle/core0/hog 全挤在同一个核上。)
    hog_cpu = next((c for c in order if c != idle_cpu and c != 0), n_cpu - 1)
    print("=== 机器状态 ===")
    print(f"  核数={n_cpu} 可用核={len(orig_mask)} loadavg={os.getloadavg()}")
    print(f"  平均 busy={sum(busy.values())/max(len(busy),1)*100:.1f}%  "
          f"最闲核={idle_cpu}({busy[idle_cpu]*100:.1f}%)  "
          f"给 hog 留的核={hog_cpu}({busy[hog_cpu]*100:.1f}%)")

    print("\n=== 控制 2: 事件窗口含不含 host 时间 ===")
    injected = control_host_work_visible()
    for inject, med in injected:
        print(f"  窗口内注入 {inject:6d} 次 Python 迭代 → median={med:7.2f} µs")
    base_med, inj_med = injected[0][1], injected[-1][1]
    if inj_med < base_med * 2:
        print("  ❌ 注入 host 工作量没抬高事件窗口 → 窗口与 host 耗时无关, D8 没有作用路径")
        return 1
    print("  ✅ 抬得动 → 事件窗口对 host 侧耗时敏感。注意这只证明「敏感」,"
          " 并不区分「我们的线程在烧 CPU」与「GPU 空等命令到达」")

    hog = spawn_hog(hog_cpu)
    time.sleep(1.2)
    print("\n=== 控制 1: 争用可见性 (秒级窗口才测得出) ===")
    try:
        for label, cpus in (("钉到 hog 核", {hog_cpu}), ("钉到空闲核", {idle_cpu}),
                            ("不钉", orig_mask)):
            os.sched_setaffinity(0, cpus)
            eff = effective_cores()
            print(f"  {label:12s} 有效核数={eff:.2f}  (同一核被独占应 ~0.50, 空闲应 ~1.00)")
        os.sched_setaffinity(0, orig_mask)
        hog_share = per_core_busy({hog_cpu}).get(hog_cpu, 0.0)
        print(f"  hog 所在核 {hog_cpu} 实测 busy={hog_share*100:.0f}% "
              f"(state={stat_field(hog.pid, 3)})")

        arm_cpu = {"default": None, "idle": idle_cpu, "core0": 0, "hog": hog_cpu}
        results = {a: {"p50": [], "p90": [], "mx": [], "ln": [], "eff": []} for a in ARMS}
        print(f"\n=== 主实验: 四臂 × {args.reps} 轮, 每臂 {args.samples} 个 launch 样本 ===")
        for rep in range(args.reps):
            order_arms = ARMS[rep % len(ARMS):] + ARMS[:rep % len(ARMS)]
            cells = []
            for arm in order_arms:
                cpu = arm_cpu[arm]
                os.sched_setaffinity(0, orig_mask if cpu is None else {cpu})
                host = measure_host_floor(args.samples)
                p50, p90, mx = quantiles(host)
                ln = statistics.median(measure_ln_flushed(LN_SAMPLES))
                eff = effective_cores(0.35)
                results[arm]["p50"].append(p50)
                results[arm]["p90"].append(p90)
                results[arm]["mx"].append(mx)
                results[arm]["ln"].append(ln)
                results[arm]["eff"].append(eff)
                cells.append(f"{arm}={p50:.2f}/{p90:.2f}/{mx:.1f}/{eff:.2f}")
            print(f"  轮 {rep+1:2d} (µs p50/p90/max, 有效核数): " + "  ".join(cells),
                  flush=True)
    finally:
        os.sched_setaffinity(0, orig_mask)
        hog.terminate()
        hog.wait()

    def sign(deltas):
        pos = sum(1 for d in deltas if d > 0)
        neg = sum(1 for d in deltas if d < 0)
        win, k = ("+", pos) if pos >= neg else ("-", neg)
        return f"{win}{k}/{len(deltas)}"

    print(f"\n=== 配对差 (相对 default, 同轮内相减, n={args.reps}) ===")
    print(f"{'臂':>8} {'p50 µs':>9} {'Δp50':>7} {'符号':>8} {'p90 µs':>9} {'Δp90':>7} "
          f"{'符号':>8} {'max µs':>9} {'生产口径':>9} {'Δ':>6} {'有效核数':>9}")
    for arm in ARMS:
        r, d = results[arm], results["default"]
        dp50 = [v - b for v, b in zip(r["p50"], d["p50"])]
        dp90 = [v - b for v, b in zip(r["p90"], d["p90"])]
        dln = [v - b for v, b in zip(r["ln"], d["ln"])]
        print(f"{arm:>8} {statistics.median(r['p50']):9.2f} "
              f"{statistics.median(dp50):7.2f} {sign(dp50):>8} "
              f"{statistics.median(r['p90']):9.2f} "
              f"{statistics.median(dp90):7.2f} {sign(dp90):>8} "
              f"{statistics.median(r['mx']):9.1f} "
              f"{statistics.median(r['ln']):9.2f} "
              f"{statistics.median(dln):6.2f} "
              f"{statistics.median(r['eff']):9.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
