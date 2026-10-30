"""测定目标 GPU 上 kernel launch 开销与"测量有意义"的最小工作量。

用途: 为缺口 1 (MeasurementValidity 的 uninformative 判定) 提供实测阈值,
      而不是拍一个任意的 shape 下限。
方法:
  1) 空核 launch 开销 -> 时间地板
  2) 纯 copy 核在不同字节量下的实测带宽 -> 找到 launch 开销占比降到 10% 以下的拐点
  3) 对照 layernorm 的真实 shape (1024x768 fp16) 落在拐点哪一侧

运行 (需在目标 GPU 机上, 用项目的 uv 环境):
  CUDA_VISIBLE_DEVICES=<空闲卡> ~/.local/bin/uv run python benchmarks/measurement/measure_launch_floor.py
本机通过 SSH 跑:
  scp benchmarks/measurement/measure_launch_floor.py <ssh别名>:~/ && \
    ssh <ssh别名> "cd ~/operator_agent && CUDA_VISIBLE_DEVICES=2 ~/.local/bin/uv run python ~/measure_launch_floor.py"

2026-09-24 于 algorithm-h20 (NVIDIA H20, GPU 2, 全机 0% util) 的结果见
docs/research/01-launch开销与shape充分性实测.md
"""
import torch
import triton
import triton.language as tl

PEAK_BW_GBS = 3680.0  # 2026-09-16 triad 实测


@triton.jit
def empty_kernel(x_ptr):
    pass


@triton.jit
def copy_kernel(src, dst, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(dst + offs, tl.load(src + offs, mask=m), mask=m)


def bench(fn, warmup=25, rep=200):
    """CUDA event 计时, 取 median (抗离群)"""
    import statistics
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rep):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)  # us
    return statistics.median(ts), min(ts)


dev = torch.device("cuda")
print(f"device: {torch.cuda.get_device_name(dev)}")

# ── 1. 空核 launch 开销 ──
x = torch.zeros(1024, device=dev)
med, mn = bench(lambda: empty_kernel[(1,)](x))
print(f"\n[1] 空核 launch: median={med:.2f}us min={mn:.2f}us")
floor = mn

# ── 2. copy 核: 扫描字节量, 找 launch 开销占比 < 10% 的拐点 ──
print(f"\n[2] copy 核扫描 (launch 地板 {floor:.2f}us)")
print(f"{'MiB':>8} {'median_us':>10} {'min_us':>9} {'GB/s':>9} {'%peak':>7} {'roofline_us':>12} {'倍数':>7} {'launch占比':>9}")
crossover = None
for mib in (0.03, 0.125, 0.5, 1, 3, 6, 12, 24, 48, 96, 192, 384):
    n = int(mib * 1024 * 1024 / 2)  # fp16 元素数
    src = torch.randn(n, device=dev, dtype=torch.float16)
    dst = torch.empty_like(src)
    BLOCK = 4096
    grid = (triton.cdiv(n, BLOCK),)
    med, mn = bench(lambda: copy_kernel[grid](src, dst, n, BLOCK=BLOCK))
    traffic = n * 2 * 2  # 读+写, 每元素 2 字节
    gbs = traffic / (med * 1e-6) / 1e9
    roof_us = traffic / (PEAK_BW_GBS * 1e9) * 1e6
    ratio = med / roof_us
    overhead_pct = floor / med * 100
    mark = ""
    if crossover is None and overhead_pct < 10:
        crossover = (mib, traffic, med, overhead_pct)
        mark = "  <-- launch 占比首次 <10%"
    print(f"{mib:8.3f} {med:10.2f} {mn:9.2f} {gbs:9.1f} {gbs/PEAK_BW_GBS*100:7.1f} "
          f"{roof_us:12.3f} {ratio:7.1f} {overhead_pct:8.1f}%{mark}")
    del src, dst

# ── 3. layernorm 真实 shape 落在哪一侧 ──
print("\n[3] layernorm 1024x768 fp16 (driver 当前用的 shape)")
rows, cols = 1024, 768
traffic = rows * cols * 2 * 2  # 读 X 写 Y
roof_us = traffic / (PEAK_BW_GBS * 1e9) * 1e6
print(f"    访存量 = {traffic/1e6:.3f} MB, roofline 下界 = {roof_us:.3f} us")
print(f"    launch 地板 {floor:.2f}us / 下界 {roof_us:.3f}us = {floor/roof_us:.1f}x"
      f"  -> launch 开销就已经是理想耗时的 {floor/roof_us:.0f} 倍")

print("\n[4] 结论")
if crossover:
    mib, traffic_c, med, oh = crossover
    print(f"    copy 核在 >= {mib} MiB/张 (访存 {traffic_c/1e6:.2f} MB) 时 launch 占比才降到 {oh:.1f}%")
    print(f"    => memory-bound 基准的单张张量访存量应达到 ~{traffic_c/1e6:.0f} MB 量级, 测量才有意义")
    print(f"    => layernorm 1024x768 只有 {traffic/1e6:.2f} MB, 差 {traffic_c/traffic:.0f}x")
