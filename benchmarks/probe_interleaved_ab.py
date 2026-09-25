"""交错 A/B 计时的前提核验：顺序测到底有没有偏，偏多少（D3 的判别实验）

在**远端 GPU 机**上跑（自包含，只用 numpy/torch/triton）：

    scp benchmarks/probe_interleaved_ab.py <ssh别名>:/tmp/ && \\
      ssh <ssh别名> "cd ~/operator_agent && CUDA_VISIBLE_DEVICES=2 \\
        ~/.local/bin/uv run python /tmp/probe_interleaved_ab.py"

为什么先做实验再改代码
--------------------
D3 的说法是「候选与基线交错采样可以配对降噪」。这句话有两个可分离的可证伪成分：

  (1) **顺序偏置存在**：现在的 runner 是「先测完候选的全部样本，再测基线」。若机器
      状态在两次测量之间漂移，加速比 = med(base)/med(cand) 就带一个**方向固定**的偏置，
      而这个偏置不随样本数收敛（增加样本只收窄它周围的分布，不移动中心）。
      判别办法很干脆：**把顺序反过来测**。若 AB 与 BA 给出的加速比之差远大于各自的
      重复散布，顺序偏置就是实打实的。
  (2) **交错能消掉它**：交错后两臂共享同一段时间窗口，扰动同时作用于两侧。
      直接证据是逐轮配对的相关系数 corr(cand_i, base_i)：> 0 就说明存在共同时变成分，
      而比值 r_i = base_i/cand_i 的散布会**低于**「两臂独立」时应有的 sqrt(cv²+cv²)。

两个成分都用真机数据判，不采信任何外部说法。另外单独跑一个**故意造出来的漂移**
（`-d` 档：在两臂之间插一段忙等），用来检验这套统计量有没有 Detect 能力 —— 如果
连人为漂移都测不出来，那 AB≈BA 的零结果就没有意义（这是 doc 35 第五节的教训：
先确认量具能看见你要看的东西）。

对应决策：docs/research/20 第三节 D3；结果写进下一份 3x 报告。
"""
import argparse
import sys
import threading

import numpy as np
import torch
import triton
import triton.language as tl

REPS = 13            # 每种日程重复多少轮（每轮内部两臂各 N 个样本）
N_SAMPLES = 90       # 每轮每臂的样本数
BLOCKS = [1, 10, 50]   # 块交错的块长 (k=1 等价逐样本交错)
FLUSH_BYTES = 256 * 1024 * 1024      # 与 Triton do_bench / 本项目 runner 一致


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


def make_candidate(X, W, B):
    n_rows, n_cols = X.shape
    Y = torch.empty_like(X)
    blk = triton.next_power_of_2(n_cols)

    def run():
        _ln[(n_rows,)](X, W, B, Y, n_rows, n_cols, 1e-5, BLOCK=blk)
    return run


def make_baseline(X, W, B):
    def run():
        torch.nn.functional.layer_norm(X, (X.shape[-1],), W, B, 1e-5)
    return run


class Timer:
    def __init__(self, device):
        self.device = device
        self.buf = torch.empty(FLUSH_BYTES, dtype=torch.uint8, device=device) \
            if device.type == "cuda" else None

    def one(self, fn):
        if self.buf is not None:
            self.buf.zero_()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            return s.elapsed_time(e)
        import time
        t0 = time.perf_counter()
        fn()
        return (time.perf_counter() - t0) * 1000


def _dist(xs):
    """(median, iqr/median) —— iqr 口径与 runner 的 _sample_stats 一致"""
    a = np.asarray(sorted(xs), dtype=np.float64)
    med = float(np.median(a))
    q1, q3 = np.percentile(a, [25, 75])
    return med, float((q3 - q1) / med) if med > 0 else 0.0


def seq_schedule(tmr, cand, base, n, first, drift_ms=0.0, load=False):
    """先测完一臂的全部样本再测另一臂。first='A' 或 'B'。"""
    a, b = [], []
    order = [cand, base] if first == "A" else [base, cand]
    out = {}
    for i, fn in enumerate(order):
        second = i == 1
        if drift_ms and second:
            _busy_wait(tmr.device, drift_ms)
        ts = []
        if load and second:
            with Contention(tmr.device):
                for _ in range(n):
                    ts.append(tmr.one(fn))
        else:
            for _ in range(n):
                ts.append(tmr.one(fn))
        out["A" if fn is cand else "B"] = ts
    return out["A"], out["B"]


def interleaved_schedule(tmr, cand, base, n, load=False, load_from=0.5):
    """A/B/A/B… 每轮各一个样本, 共享同一段时间窗口。

    `load` 时争抢负载照样加, 只是加在**窗口的后半段** —— 这样三档对照才是同一种机器
    状态: 顺序档让某一臂整段被抢, 交错档让两臂各吃一半。若交错真能配对降噪, 它的重复
    散布应明显小于 |AB-BA| 这个顺序分歧。
    """
    a, b = [], []
    cut = int(n * load_from) if load else n
    for i in range(n):
        if load and i == cut:
            ctx = Contention(tmr.device)
            ctx.__enter__()
        a.append(tmr.one(cand))
        b.append(tmr.one(base))
    if load and cut < n:
        ctx.__exit__()
    return a, b


class Contention(threading.Thread):
    """在**一整段测量窗口内**持续抢占 GPU 的背景流。

    为什么必须是持续负载而不是一阵忙等: 第一版用 `_busy_wait` 在两臂之间插一段 matmul
    链, 结果 AB 与 BA 的加速比一模一样 —— 因为每次计时都自带 L2 冲刷 + synchronize,
    一段已经结束的负载不留任何痕迹, 也就是说那个注入根本不构成「顺序」相关的状态,
    用它得出的零结果是空转。换成与某个 arm 的测量窗口重叠的持续负载, 顺序才会真正要命:
    A→B 时只有 B 在被抢, B→A 时只有 A 在被抢。
    """

    def __init__(self, device):
        super().__init__(daemon=True)
        self.device = device
        self.stop = threading.Event()
        self.stream = None

    def run(self):
        self.stream = torch.cuda.Stream(device=self.device)
        x = torch.randn(4096, 4096, device=self.device, dtype=torch.float16)
        with torch.cuda.stream(self.stream):
            while not self.stop.is_set():
                x @ x
                self.stream.synchronize()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.join(timeout=30)
        torch.cuda.synchronize()


def block_schedule(tmr, cand, base, n, block, load=False, load_from=0.5):
    """块交错: 每块连测 `block` 个样本, 两块之间换臂。

    这是**能不能落地**的关键一档: 候选臂必须在 anticheat 的 shield_torch 里跑
    (torch 原生算子被换成抛错桩), 基线臂本身就是 torch 原生算子 —— 同一段时间窗口内
    两臂不可能共用一个 shield。所以只能按块切换 shield, 代价是每块一次 setattr 群。
    runner 里已经写明「逐次包裹会把 L3 计时毁掉」(run_kernel.py:541-543)。
    本函数量的就是: 块粒度多大才不会把配对收益丢掉。
    """
    a, b = [], []
    cut = int(n * load_from) if load else n
    ctx = None
    i = 0
    while i < n:
        k = min(block, n - i)
        if load and i <= cut < i + k and ctx is None:
            ctx = Contention(tmr.device).__enter__()
        for _ in range(k):
            a.append(tmr.one(cand))
        for _ in range(k):
            b.append(tmr.one(base))
        i += k
    if ctx is not None:
        ctx.__exit__()
    return a, b


def _busy_wait(device, ms):
    """旧注入 (保留做对照): 两臂之间跑一串 matmul, 但结束时不留痕迹"""
    x = torch.randn(2048, 2048, device=device, dtype=torch.float16)
    for _ in range(max(1, int(ms * 2))):
        x @ x
    torch.cuda.synchronize()


def summarize(tag, speedups):
    med, rel = _dist(speedups)
    print(f"    {tag:26s} 加速比 median={med:8.4f}  iqr/median={rel:7.4f}  "
          f"min={min(speedups):.4f} max={max(speedups):.4f}")
    return med, rel


def probe(rows_cols, device, drift_ms=0.0, load=False):
    rows, cols = rows_cols
    X = torch.randn(rows, cols, device=device, dtype=torch.float16)
    W = torch.ones(cols, device=device, dtype=torch.float16)
    B = torch.zeros(cols, device=device, dtype=torch.float16)
    cand, base = make_candidate(X, W, B), make_baseline(X, W, B)
    tmr = Timer(device)
    for fn in (cand, base):        # 各自吃掉 JIT/首次派发
        fn()
    torch.cuda.synchronize() if device.type == "cuda" else None

    print(f"\n===== layernorm {rows}×{cols} fp16  (REPS={REPS} × N={N_SAMPLES}"
          + (f"，第二臂全程加持续争抢负载" if load else "")
          + (f"，两臂之间插入 {drift_ms} ms 忙等" if drift_ms else "")
          + ") =====")
    sp_ab, sp_ba, sp_int, pairs = [], [], [], []
    sp_blk: dict = {}
    for _ in range(REPS):
        a, b = seq_schedule(tmr, cand, base, N_SAMPLES, "A", drift_ms, load)
        sp_ab.append(np.median(b) / np.median(a))
        a, b = seq_schedule(tmr, cand, base, N_SAMPLES, "B", drift_ms, load)
        sp_ba.append(np.median(b) / np.median(a))
        a, b = interleaved_schedule(tmr, cand, base, N_SAMPLES, load)
        sp_int.append(np.median(b) / np.median(a))
        pairs.append((a, b))
        for blk in BLOCKS:
            a, b = block_schedule(tmr, cand, base, N_SAMPLES, blk, load)
            sp_blk.setdefault(blk, []).append(np.median(b) / np.median(a))

    med_ab, rel_ab = summarize("顺序 A→B (现口径)", sp_ab)
    med_ba, rel_ba = summarize("顺序 B→A (反序)", sp_ba)
    med_int, rel_int = summarize("交错 A/B/A/B (D3)", sp_int)
    for blk in sorted(sp_blk):
        summarize(f"块交错 k={blk}", sp_blk[blk])

    gap = abs(med_ab - med_ba)
    spread = max(rel_ab, rel_ba, rel_int) * med_int
    print(f"\n    顺序偏置 |AB-BA| = {gap:.4f} ；三种日程里最大的重复散布 = {spread:.4f}")
    print(f"    → 偏置/散布 = {gap / spread if spread else float('inf'):.2f}×  "
          f"({'偏置淹没在噪声里，顺序不重要' if gap < spread else '顺序偏置真实存在'})")

    # 机制: 交错样本的同时刻相关性 + 比值散布 vs 独立假设的预测
    # 预测与实测必须同口径 (都用 iqr/median), 否则比较没有意义 —— 第一版这里误用了
    # MAD 口径 (约为 iqr 口径的一半), 得出「交错反而更吵」的假结论。
    A = np.concatenate([p[0] for p in pairs])
    Bt = np.concatenate([p[1] for p in pairs])
    r = np.corrcoef(A, Bt)[0, 1]
    ratio = Bt / A
    _, rel_r = _dist(ratio)
    cv_a, cv_b = _dist(A)[1], _dist(Bt)[1]
    pred_indep = (cv_a ** 2 + cv_b ** 2) ** 0.5
    print(f"\n    corr(候选, 基线) 逐样本 = {r:+.3f}   (>0 说明存在共同时变成分)")
    print(f"    单样本散布: 候选 iqr/med={cv_a:.4f}  基线 iqr/med={cv_b:.4f}")
    print(f"    逐样本比值 base/cand 的 iqr/median = {rel_r:.4f}   "
          f"两臂独立时预测 ≈ {pred_indep:.4f}")
    gain = pred_indep / rel_r if rel_r else float("nan")
    print(f"    → 比值散布/独立预测 = {1/gain if gain else float('nan'):.2f}×，"
          f"corr≈0 时 ≈1 即自洽 (对数正态比值的 iqr 口径偏大, 略 >1 属正常)")
    print(f"    决策相关的量是**下面三行的加速比重复散布**，不是这个逐样本比值")
    return dict(med_ab=med_ab, med_ba=med_ba, med_int=med_int,
                rel=(rel_ab, rel_ba, rel_int), gap=gap, corr=r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--drift-ms", type=float, default=0.0,
                    help="在两臂之间插入的人为漂移时长 (ms)，用于验证量具有检出能力")
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("-l", "--load", action="store_true",
                    help="第二臂的整个测量窗口内跑一条持续抢占的 matmul 流")
    args = ap.parse_args()
    globals()["REPS"] = args.reps

    if not torch.cuda.is_available():
        print("⚠️ 无 CUDA，CPU 计时只能验统计管线，不能用来判 D3")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        p = torch.cuda.get_device_properties(0)
        print(f"device = {p.name}  cc={p.major}.{p.minor}  "
              f"visible={torch.cuda.device_count()}")
    for shape in ((16384, 6144), (1024, 768)):
        probe(shape, device, args.drift_ms, args.load)


if __name__ == "__main__":
    sys.exit(main())
