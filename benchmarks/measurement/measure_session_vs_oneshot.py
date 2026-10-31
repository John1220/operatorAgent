"""常驻会话 vs 每次新起进程: 真机上量三个数 (doc 62)

问题不是「谁更快」, 而是**换掉默认路径之前必须先知道三件事**:

  1. 每次测量的墙钟 —— 省下来的到底是不是固定开销 (拆解见 doc 62 §二)
  2. 一次心跳的成本 —— 它决定 `heartbeat_interval` 那个折中还需不需要
  3. **同一测量在常驻进程里连测 N 次的重复性** —— 这一条才决定它能不能替代今天的
     冷启动口径: 今天的读数带着「每次新进程」的冷启动痕迹 (doc 31 launch 地板 /
     doc 40 DVFS / doc 37 热冷两层), 换了之后那些已标定的阈值必须重测

另外顺路量两件同源的事:
  - 会话钉住哪张卡 (选卡发生在远端 import 时, 一次会话固定在同一张卡上)
  - 收尾是否到达静止: 只杀本地 ssh **不保证**远端进程停 (`benchmarks/_scratch.py` 的纪律),
    所以关掉会话后回远端 `pgrep -u $(whoami) -f runner_server.py` 数一遍

跑法 (需 ~/.ssh/config 里的 algorithm-h20):
    python3 benchmarks/measurement/measure_session_vs_oneshot.py
    python3 benchmarks/measurement/measure_session_vs_oneshot.py --n 5 --skip-triton
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

import numpy as np  # noqa: E402

# triton 那一对复用 61 号那次真机预演用的源码 —— 不抄第二份 (抄的那一刻是准的)
from measure_heartbeat_real import TRITON_GOOD_SRC  # noqa: E402
from operator_agent.decide.calibration import (KNOWN_GOOD_SOURCE,  # noqa: E402
                                        backend_bench_fn, default_inputs, heartbeat)
from operator_agent.execute.remote_executor import RemoteExecutor, SSHExecutor  # noqa: E402


def stats(xs):
    med = statistics.median(xs)
    mad = statistics.median([abs(x - med) for x in xs])
    return med, (mad / med if med else float("nan"))


def _pair(label, values):
    med, rel = stats(values)
    print(f"  {label:<26} {med:8.3f} ms  (n={len(values)}, mad/med={rel:.4f}, "
          f"min={min(values):.3f}, max={max(values):.3f})")
    return med


def timeit(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="每个臂的重复次数")
    ap.add_argument("--shape", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--skip-triton", action="store_true")
    args = ap.parse_args()

    A, B = default_inputs()
    rng = np.random.default_rng(0)
    t_in = {"A": rng.standard_normal((args.shape, args.shape)).astype(np.float32),
            "B": rng.standard_normal((args.shape, args.shape)).astype(np.float32)}

    print("[session-vs-oneshot] 连接 algorithm-h20 …")
    one = RemoteExecutor(mode="ssh", timeout=180)
    _ = one.label                                    # 触发解析/连接
    print(f"  一次性后端: {one.label}")

    # ── ① 每次测量的墙钟 (Python 分支参照核, 最便宜的那种) ─────────────
    print("\n" + "=" * 74)
    print("① 同一条测量 (24×24 朴素 matmul, Python 分支): 墙钟")
    print("=" * 74)
    oneshot_wall, oneshot_ms = [], []
    for i in range(args.n):
        p, dt = timeit(lambda: one.benchmark(KNOWN_GOOD_SOURCE, {"A": A, "B": B},
                                             warmup=args.warmup, repeat=args.repeat))
        oneshot_wall.append(dt)
        oneshot_ms.append(float(p.get("median_time_ms") or 0.0))
    _pair("一次性 墙钟", oneshot_wall)
    _pair("一次性 读数", oneshot_ms)

    ex = SSHExecutor(persistent=True, timeout=180)
    t0 = time.perf_counter()
    first, first_dt = timeit(lambda: ex.benchmark(KNOWN_GOOD_SOURCE, {"A": A, "B": B},
                                                 warmup=args.warmup, repeat=args.repeat))
    print(f"  常驻首次调用 (含 deploy + 服务端 import + 握手): {first_dt:.2f}s")
    print(f"  会话钉住的卡: {ex.session_gpu}")
    sess_wall, sess_ms = [], []
    for i in range(args.n):
        p, dt = timeit(lambda: ex.benchmark(KNOWN_GOOD_SOURCE, {"A": A, "B": B},
                                            warmup=args.warmup, repeat=args.repeat))
        sess_wall.append(dt)
        sess_ms.append(float(p.get("median_time_ms") or 0.0))
    _pair("常驻 墙钟", sess_wall)
    _pair("常驻 读数", sess_ms)
    print(f"  → 墙钟比: {statistics.median(oneshot_wall) / statistics.median(sess_wall):.1f}×")

    # ── ② 一次心跳的成本 ─────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("② 一次心跳 heartbeat(backend_bench_fn(backend)) 的成本")
    print("=" * 74)
    for label, be in (("一次性", one), ("常驻", ex)):
        r_, dt = timeit(lambda: heartbeat(backend_bench_fn(be, warmup=args.warmup,
                                                           repeat=args.repeat),
                                          epoch=1, window_id="hb-1"))
        print(f"  {label}: ok={r_.ok} 慢/好={r_.known_slow_ratio} 墙钟 {dt:.2f}s")

    # ── ③ 重复性 (GPU 分支: 这才是候选核走的通道) ─────────────────────
    if not args.skip_triton:
        print("\n" + "=" * 74)
        print(f"③ 重复性: triton matmul @{args.shape}³ (CUDA event 分支) 连测")
        print("=" * 74)
        g_one, g_sess = [], []
        for i in range(args.n):
            p, dt = timeit(lambda: one.benchmark(TRITON_GOOD_SRC, t_in,
                                                 warmup=args.warmup, repeat=args.repeat))
            if p.get("success") is False:
                print(f"  一次性失败: {str(p.get('error'))[:120]}")
                break
            g_one.append(float(p.get("median_time_ms") or 0.0))
        if g_one:
            _pair("一次性 GPU 读数", g_one)
        for i in range(args.n):
            p, dt = timeit(lambda: ex.benchmark(TRITON_GOOD_SRC, t_in,
                                                warmup=args.warmup, repeat=args.repeat))
            if p.get("success") is False:
                print(f"  常驻失败: {str(p.get('error'))[:120]}")
                break
            g_sess.append(float(p.get("median_time_ms") or 0.0))
        if g_sess:
            _pair("常驻 GPU 读数", g_sess)
            if g_one and g_sess:
                a, ra = stats(g_one); b, rb = stats(g_sess)
                print(f"  → 两条路的读数差 {abs(a - b) / a * 100:.2f}%  "
                      f"(相对离散度 {ra:.4f} vs {rb:.4f})")

    # ── 收尾: 到达静止了吗 ───────────────────────────────────────────
    print("\n" + "=" * 74)
    print("④ 收尾: 关掉会话, 再回远端数一遍 (判据必须不自匹配)")
    print("=" * 74)
    # ⚠️ 这里第一版写的是 `pgrep -f runner_server.py`, 它**匹配到自己那条命令行**
    # (那个 ssh 会话执行的 shell argv 里就带着这个字符串), 于是报出一个不存在的残留。
    # 精确判据有两个, 都不用模式匹配: ① 服务端握手时自报的 pid 还在不在 (`kill -0`);
    # ② ps 全表按**方括号技巧**过滤 (模式文本自身不匹配该正则) —— 后者才能看出有没有
    # 落下 uv 父进程之类的旁支。
    server_pid = (ex._session.call({"action": "ping"}) or {}).get("pid") if ex._session else None
    ex.close()
    t0 = time.perf_counter()
    alive = ex._ssh_exec(f"kill -0 {server_pid} 2>/dev/null && echo ALIVE || echo GONE")
    ps = ex._ssh_exec("ps -u $(whoami) -o pid=,args= | grep '[r]unner_server' || echo '(无)'")
    print(f"  服务端自报 pid={server_pid} → {alive.strip()}")
    print(f"  ps 全表里剩下的 runner_server 行: {ps.strip() or '(无)'}")
    print(f"  (核查耗时 {time.perf_counter() - t0:.2f}s, 两次往返)")
    getattr(one, 'close', lambda: None)()
    print("\n[session-vs-oneshot] 完。没起干扰负载, 没在共享 /tmp 留文件。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())