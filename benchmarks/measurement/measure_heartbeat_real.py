"""真机跑一次标定心跳 —— 参照对走远端那把尺子 (设计 §10.5-1)

首次把 `calibration.backend_bench_fn` 放到真机上, 回答四个问题:

  1. 参照对**到底量在哪**: 哪个语言分支、哪把秒表、哪个设备 (`device` 字段)
  2. 远端量出来的分离度是多少 (设计期望慢核 ≈8×), 与本地 MockGPU 差多少
  3. 一次心跳花多少时间 —— 它值不值得插在每 N 次真值测量前
  4. **换个参照对会怎样**: 同一台机、同一条 `backend.benchmark`, 把参照对换成
     triton 那对 (走 CUDA event 分支) —— 这是「按任务 DSL 选参照对」的预演

跑法 (需 `~/.ssh/config` 里的 algorithm-h20; 会 deploy runner):
    python3 benchmarks/measurement/measure_heartbeat_real.py
    python3 benchmarks/measurement/measure_heartbeat_real.py --n 1024 --rounds 3

⚠️ 这一份**不下任何干扰负载、不在共享 /tmp 留固定名** (参照核与 runner 走 stdin JSON,
没有落点文件)。共享机纪律见 `benchmarks/_scratch.py`。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from operator_agent.decide.calibration import (KNOWN_GOOD_SOURCE,  # noqa: E402
                                       KNOWN_SLOW_SOURCE, backend_bench_fn,
                                       default_inputs, heartbeat)
from operator_agent.execute.remote_executor import RemoteExecutor  # noqa: E402
from operator_agent.core.kernel_templates import (  # noqa: E402
    MATMUL_TRITON, jit_kernel_source,
)

# ── triton 那一对 (由 kernel_templates.MATMUL_TRITON **派生**, 不再手抄) ────
# 好核: 模板原样。慢核: 同一个 launcher 连跑 8 遍, 只返回最后一遍 —— 与 numpy 那对
# 的构造完全同构 (数值正确, 只是多算), 这样两对的分离度才可比。
#
# 这里曾经是一份**手抄**的核: 与模板语义相同、签名被重新折行 ⇒ 逐字比较抓不到,
# 而注释写着"取自 rule_templates.py"。模板一改, 这个量具就会静静地量另一个核,
# 注释继续撒谎。现在改为由单一真源派生, 派生关系由
# tests/unit/test_template_single_source.py 盯着。

TRITON_BODY = jit_kernel_source(MATMUL_TRITON)

TRITON_HEAD = "import triton\nimport triton.language as tl\nimport torch\n"


def _launcher(repeats: int) -> str:
    return f"""
def kernel(A, B):
    # 显式转 fp32: 真机实测 (2026-10-01) `default_inputs()` 是 Python float 列表 →
    # 远端按声明的 dtype 重建成 **fp64** → `tl.dot` 出 fp64, 累加器 fp32 编不过
    # ("Loop-carried variable acc ... re-assigned to fp64")。参照核必须自己钉住数值语义。
    A = A.float()
    B = B.float()
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    for _ in range({repeats}):
        matmul_kernel[grid](A, B, C, M, N, K, A.stride(0), A.stride(1),
                            B.stride(0), B.stride(1), C.stride(0), C.stride(1),
                            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return C
"""


TRITON_GOOD_SRC = TRITON_HEAD + TRITON_BODY + _launcher(1)
TRITON_SLOW_SRC = TRITON_HEAD + TRITON_BODY + _launcher(8)


class _Ref:
    """带源码形态的参照核替身。

    为什么需要它: `backend_bench_fn` 收的是**可调用**参照核 (它读 `__source__` 去远端量),
    而 triton 参照核在控制机上装不了/跑不了。这个替身只提供 `__source__` 与一个
    `__call__(A, B)` 形参表 (给 `inspect.signature` 看), 从不真的在本地被调用 ——
    于是「按 DSL 选参照对」这件事可以在不引入第二套 bench_fn 的前提下先试一遍。
    """

    def __init__(self, name: str, source: str):
        self.__name__ = name
        self.__source__ = source

    def __call__(self, A, B):  # pragma: no cover - 只在本地误调时触发
        raise RuntimeError("替身不该在控制机上被调用")


def raw(backend, label, code, inputs, warmup, repeat):
    """直接量一次并把**载荷原样**打出来 —— 语言分支/设备/秒表都在这几个键里"""
    t0 = time.perf_counter()
    p = backend.benchmark(code, inputs, warmup=warmup, repeat=repeat)
    dt = time.perf_counter() - t0
    if not isinstance(p, dict) or p.get("success") is False:
        print(f"  {label}: ✗ {p.get('error') if isinstance(p, dict) else p!r}  ({dt:.2f}s)")
        return None
    med = p.get("median_time_ms")
    print(f"  {label}: median={med} ms  language={p.get('language')!r} "
          f"device={p.get('device')!r}  往返 {dt:.2f}s")
    return med


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512, help="numpy 对照臂的方阵边长")
    ap.add_argument("--shape", type=int, default=2048, help="triton 臂的方阵边长")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--skip-triton", action="store_true")
    args = ap.parse_args()

    print("[heartbeat-real] 连接 algorithm-h20 …")
    t0 = time.perf_counter()
    backend = RemoteExecutor(mode="ssh", timeout=180)
    print(f"[heartbeat-real] backend={backend.label}  "
          f"gpu={getattr(backend, 'selected_gpu', '?')}  "
          f"(deploy 随首次调用发生, {time.perf_counter() - t0:.2f}s)")

    # ── ① 参照对量在哪 ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("① 现有参照对 (纯 Python, 24×24) —— 语言分支 / 秒表 / 设备")
    print("=" * 72)
    A, B = default_inputs()
    t0 = time.perf_counter()
    raw(backend, "known_good", KNOWN_GOOD_SOURCE, {"A": A, "B": B},
        args.warmup, args.repeat)
    first_call = time.perf_counter() - t0
    raw(backend, "known_slow", KNOWN_SLOW_SOURCE, {"A": A, "B": B},
        args.warmup, args.repeat)
    print(f"  (首次调用含 deploy, 共 {first_call:.2f}s)")

    # ── ② 同一条通道上, smoke_spine_h20 的 numpy 臂量在哪 ──────────────
    print("\n" + "=" * 72)
    print(f"② 对照: smoke_spine_h20 的 GOOD (numpy matmul @{args.n}³) 量在哪")
    print("=" * 72)
    rng = np.random.default_rng(0)
    big = {"A": rng.standard_normal((args.n, args.n)).astype(np.float32),
           "B": rng.standard_normal((args.n, args.n)).astype(np.float32)}
    smoke_good = ("import numpy as np\n\ndef kernel(A, B):\n"
                  "    return np.asarray(A) @ np.asarray(B)\n")
    med = raw(backend, f"numpy@{args.n}", smoke_good, big, args.warmup, args.repeat)
    if med is not None:
        print(f"  → 判据是载荷自己报的 device/language, 不是毫秒数: device='cpu' +")
        print("     language='numpy' 就说明这段被量在**远端 CPU 的 BLAS** 上, 与 H20 无关。")
        print("     (毫秒只在 n 大时才有区分度: 1024³ 在 H20 上约 0.05~0.2ms, 在 CPU 上约几 ms。)")

    # ── ③ 心跳本身 ───────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("③ heartbeat(backend_bench_fn(backend)) —— 真机第一次跑")
    print("=" * 72)
    for i in range(1, 4):
        t0 = time.perf_counter()
        r = heartbeat(backend_bench_fn(backend, warmup=args.warmup, repeat=args.repeat),
                      epoch=i, window_id=f"hb-{i}")
        dt = time.perf_counter() - t0
        print(f"  第 {i} 次: ok={r.ok}  好={r.known_good_ratio}  慢/好={r.known_slow_ratio}"
              f"  耗时 {dt:.2f}s")
        print(f"           reasons={list(r.reasons)}")

    # ── ④ 预演: 换成 triton 参照对 (CUDA event 分支) ──────────────────
    if not args.skip_triton:
        print("\n" + "=" * 72)
        print(f"④ 预演「按 DSL 选参照对」: triton matmul @{args.shape}³ (CUDA event 分支)")
        print("=" * 72)
        t = {"A": rng.standard_normal((args.shape, args.shape)).astype(np.float32),
             "B": rng.standard_normal((args.shape, args.shape)).astype(np.float32)}
        raw(backend, "triton_good (1×)", TRITON_GOOD_SRC, t, args.warmup, args.repeat)
        raw(backend, "triton_slow (8×)", TRITON_SLOW_SRC, t, args.warmup, args.repeat)
        t0 = time.perf_counter()
        r = heartbeat(backend_bench_fn(backend, warmup=args.warmup, repeat=args.repeat),
                      good_kernel=_Ref("known_good_kernel", TRITON_GOOD_SRC),
                      slow_kernel=_Ref("known_slow_kernel", TRITON_SLOW_SRC),
                      epoch=99, window_id="hb-triton")
        print(f"  triton 参照对: ok={r.ok}  好={r.known_good_ratio}  "
              f"慢/好={r.known_slow_ratio}  耗时 {time.perf_counter() - t0:.2f}s")
        print(f"           reasons={list(r.reasons)}")
        print("  (好核的绝对锚点仍缺 —— 只判分离度, 见 calibration 的边界说明)")

    print("\n[heartbeat-real] 完。没在共享 /tmp 留固定名文件, 没起干扰负载。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())