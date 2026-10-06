"""标定「这台机器实际上跑得出来的峰值」到底是不是一个常数

在远端 GPU 机上跑 (需要 torch):

    ssh <别名> 'cd ~/operator_agent && CUDA_VISIBLE_DEVICES=<空闲卡> \\
        ~/.local/bin/uv run --no-project python -' \\
        < benchmarks/calibrate_achievable_peak.py

背景 (缺口 3 现在是有后果的了)
------------------------------
`operator_agent/verifier.py:_PEAK_BY_NAME["h20"]` 写的是 `(31.6, 71.6, 138.2, 138.3,
3680.0)`，注释说这些数是 2026-09-16 用 cuBLAS 8192³ / triad **实测**的，不是 datasheet。
也就是说我们已经采纳了「measured peak」这条外部做法 —— 但把它当成了一个标量常数。

吸纳 D4 之后这个常数开始有后果了: `measurement.reference_ceiling_reached()` 拿
「达成率 >= 90%」当**提前停止优化循环**的出口。而 doc 34/38 的真机读数里
matmul 4096³ 跑到 **140.0 TFLOPS > 138.2** —— 达成率 101.5%。一个以「离上界还有
多远」为判据的出口，上界本身却会被超过，说明那个数不是它声称的东西。

所以这个探针不是「再测一次峰值」，而是判一个具体假设:
**「achievable peak」随 shape 与精度显著变化，所以单个标量必然错。**

假设的来源是容量算术，不是直觉: 4096² fp16 的 A+B+C 共 96 MB，H20 的 L2 是
`torch.cuda.get_device_properties().L2_cache_size`（约 60 MB）—— 装不下；而 3072²
是 54 MB，装得下。若 L2 常驻真的抬高有效速率，那「8192³ 上测出来的 138.2」放在
4096³ 的判断上就是**偏低的上界**，于是达成率能超过 100%、收敛出口会提前打开。

三条会同时改变读数的口径，都单独记下来:
1. **TF32 开关**是 torch 的全局标志 (`torch.backends.cuda.matmul.allow_tf32`)，不是
   硬件属性。verifier 选 `tf32_tflops` 的条件是「kernel_type 含 matmul + cc>=8」，
   与这个标志无关 —— 于是「该除哪个峰值」可能和「机器实际跑了哪条路径」脱节。
   这里 fp32 行**强制关** TF32、tf32 行**强制开**，把两条路径分开量。
2. **L2 冲刷**排在 `start.record()` 之前 (与 `run_kernel._timed_runs` 同口径)。
   不冲刷的话小 shape 会被上一轮的残留喂饱，量出来的是缓存带宽而不是 HBM 带宽。
3. **cuBLAS 的算法选择随 shape 变** (启发式), 这本身就是「achievable」的一部分,
   不是噪声 —— 所以每个 shape 各测各的才是对的，取一个代表 shape 是错的。

带宽那一节同理: 拷贝尺寸从小到大，看 3680 GB/s 这个数是在哪一点测出来的。
"""
import statistics
import torch

FLUSH_BYTES = 256 * 1024 * 1024
WARMUP_MS_BUDGET = 25      # 与 run_kernel.py 一致
SAMPLES = 30
SIZES = [2048, 3072, 4096, 6144, 8192]
L2_BYTES = torch.cuda.get_device_properties(0).L2_cache_size

_flush_buf = torch.empty(FLUSH_BYTES // 4, dtype=torch.int, device="cuda")


def timed_once(fn) -> float:
    """与 run_kernel._timed_runs 同口径: 冲刷排在 start.record() 之前, 同流有序"""
    _flush_buf.zero_()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def median_ms(fn, samples=SAMPLES) -> float:
    fn()
    torch.cuda.synchronize()
    est = statistics.median(timed_once(fn) for _ in range(3))
    for _ in range(int(min(max(1, WARMUP_MS_BUDGET / est), 200)) if est > 0 else 1):
        fn()
    torch.cuda.synchronize()
    return statistics.median(timed_once(fn) for _ in range(samples))


def gemm_tflops(n, dtype, tf32=None):
    """返回 (TFLOPS, 三块张量合计 MB, 是否装得进 L2)"""
    if tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    c = torch.empty(n, n, device="cuda", dtype=dtype)
    ms = median_ms(lambda: torch.mm(a, b, out=c))
    tflops = (2.0 * n ** 3) / (ms * 1e-3) / 1e12
    foot = 3 * a.numel() * a.element_size()
    del a, b, c
    torch.cuda.empty_cache()
    return tflops, foot / 1e6, foot <= L2_BYTES


def copy_gbs(mb):
    """设备内 D2D 拷贝: 有效带宽 = 读+写 两倍字节数 / 时间"""
    n = int(mb * 1e6 / 2)
    src = torch.empty(n, dtype=torch.float16, device="cuda")
    dst = torch.empty_like(src)
    ms = median_ms(lambda: dst.copy_(src))
    gbs = (2 * n * 2) / (ms * 1e-3) / 1e9
    del src, dst
    torch.cuda.empty_cache()
    return gbs


def main():
    p = torch.cuda.get_device_properties(0)
    print(f"torch={torch.__version__}  "
          f"device={p.name}  cc={p.major}.{p.minor}")
    print(f"L2_cache_size = {L2_BYTES / 1e6:.0f} MB   "
          f"SM 数 = {p.multi_processor_count}   显存 = {p.total_memory / 1e9:.0f} GB")
    print(f"计时口径: {FLUSH_BYTES // 1024 // 1024} MB L2 冲刷在 start.record() 之前, "
          f"{SAMPLES} 样本取 median\n")

    rows = []
    print("=== 计算峰值阶梯 (cuBLAS/torch.mm, TFLOPS) ===")
    print(f"{'n':>6} {'dtype':>7} {'TF32标志':>9} {'TFLOPS':>9} "
          f"{'A+B+C MB':>9} {'装进L2?':>8}  相对 138.2")
    specs = [("fp16", torch.float16, None), ("bf16", torch.bfloat16, None),
             ("tf32", torch.float32, True), ("fp32", torch.float32, False)]
    for n in SIZES:
        for name, dt, tf32 in specs:
            tflops, mb, fits = gemm_tflops(n, dt, tf32)
            rows.append((n, name, tflops))
            ref = {"fp16": 138.2, "bf16": 138.3, "tf32": 71.6, "fp32": 31.6}[name]
            print(f"{n:>6} {name:>7} {('-' if tf32 is None else ('开' if tf32 else '关')):>9} "
                  f"{tflops:>9.2f} {mb:>9.0f} {('是' if fits else '否'):>8}  "
                  f"{tflops / ref:>7.1%}")

    print("\n=== 有效带宽阶梯 (D2D copy, GB/s; 表里的 H20 带宽 = 3680) ===")
    for mb in (8, 64, 256, 1024, 4096):
        gbs = copy_gbs(mb)
        print(f"  {mb:>5} MB → {gbs:>8.0f} GB/s   相对 3680 = {gbs / 3680:>6.1%}")

    print("\n=== 读法 ===")
    by = {}
    for n, name, tflops in rows:
        by.setdefault(name, []).append((n, tflops))
    for name, vals in by.items():
        lo = min(vals, key=lambda v: v[1])
        hi = max(vals, key=lambda v: v[1])
        spread = hi[1] / lo[1] - 1 if lo[1] else 0
        print(f"  {name}: 跨 shape 最高/最低 = {hi[1]:.1f}(n={hi[0]}) / "
              f"{lo[1]:.1f}(n={lo[0]})  相差 {spread:+.1%}")
    print("  若某个 dtype 的跨 shape 相差 > 一个测量档位 (measurement.NOISE_BAND 的 "
          "degraded 档 = 0.4%), 那么『一个 dtype 一个峰值标量』就是错的口径, "
          "达成率会随 shape 系统性地偏高或偏低。")


if __name__ == "__main__":
    main()
