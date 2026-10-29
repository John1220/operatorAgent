"""Shared benchmarking scaffolding for the 4096^3 fp16 matmul study on H20.

Design goals:
  * Same measurement discipline for cuBLAS and for every Triton candidate,
    so the speedup number means something.
  * Report best / median / stddev, because "5% faster" needs a noise floor.
  * L2 flush between iterations (H20 has 60 MB L2; 3*4096*4096*2 = 100 MB,
    so a naive loop is partly an L2 benchmark, not a DRAM one).
"""
import os
import statistics

import torch

N = 4096
DTYPE = torch.float16


def setup(device="cuda:0"):
    torch.manual_seed(0)
    dev = torch.device(device)
    a = torch.randn(N, N, device=dev, dtype=DTYPE)
    b = torch.randn(N, N, device=dev, dtype=DTYPE)
    c = torch.empty(N, N, device=dev, dtype=DTYPE)
    return a, b, c


def tflops(ms):
    return 2.0 * N ** 3 / (ms * 1e-3) / 1e12


def _flush_buf(dev, bytes_per_gpu):
    # 256 MB scratch == well past L2, evicts both operands and the output tile.
    return torch.empty(bytes_per_gpu // 4, device=dev, dtype=torch.float32)


def time_fn(fn, iters=100, warmup=25, flush=None, use_cuda_graph=False):
    """Return dict of timings in ms (median/best/stddev) and the raw samples."""
    if use_cuda_graph:
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            fn()
        for _ in range(warmup):
            g.replay()
        torch.cuda.synchronize()
        samples = []
        ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(iters)]
        for i in range(iters):
            if flush is not None:
                flush.zero_()
            ev[i][0].record()
            g.replay()
            ev[i][1].record()
        torch.cuda.synchronize()
        for a, b in ev:
            samples.append(a.elapsed_time(b))
        return _stats(samples)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(iters)]
    for i in range(iters):
        if flush is not None:
            flush.zero_()
        ev[i][0].record()
        fn()
        ev[i][1].record()
    torch.cuda.synchronize()
    for a, b in ev:
        samples.append(a.elapsed_time(b))
    return _stats(samples)


def _stats(samples):
    s = sorted(samples)
    return {
        "n": len(s),
        "median_ms": statistics.median(s),
        "best_ms": s[0],
        "mean_ms": statistics.fmean(s),
        "stddev_ms": statistics.pstdev(s),
        "min_tflops": tflops(s[-1]),
        "median_tflops": tflops(statistics.median(s)),
        "best_tflops": tflops(s[0]),
    }


def parquet_free_report(name, st, tag=""):
    return (f"{name:<34s} {tag:<10s} med {st['median_ms']:.3f} ms  "
            f"best {st['best_ms']:.3f} ms  sd {st['stddev_ms']:.3f}  "
            f"{st['median_tflops']:.1f}/{st['best_tflops']:.1f} TF/s")