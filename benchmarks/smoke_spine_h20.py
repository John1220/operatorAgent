"""真机 §14 冒烟 —— 脊柱上 H20 跑 known-good / known-slow / 竞争负载

设计 §14 要「标定标定器本身」; MockGPU 等价物已经绿了, 这里补**真机**那一半:

  #1 已知好 kernel      → VALID, cv 不散
  #2 故意写慢的 kernel  → SLOW / 速度明显更差
  #4 后台占卡任务       → DEGRADED 而不是错误通过

跑法 (需 ~/.ssh/config 里的 algorithm-h20):
    python3 benchmarks/smoke_spine_h20.py
    python3 benchmarks/smoke_spine_h20.py --skip-contention   # 不占卡, 只测 1/2

输出走 oracle.Observation 的资格位 —— 不是「跑完了」, 是「这个数被允许造成什么后果」。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from operator_agent.contract import BaselineDef, DTypeSemantics, ShapeDist, freeze  # noqa: E402
from operator_agent.measurement import Validity, assess, verdict, Verdict  # noqa: E402
from operator_agent.oracle import evaluate  # noqa: E402
from operator_agent.placement import DeviceLease, DeviceKey, lease_notes  # noqa: E402
from operator_agent.remote_executor import RemoteExecutor  # noqa: E402
from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec,  # noqa: E402
                                    TargetSpec, TaskComplexity, TaskSchema, TaskType,
                                    VerificationLevel)

# 共享机的落点与拆除纪律 (#69 / test_shared_host_scratch_paths): 随机名 + 读回静默
from _scratch import remote_scratch_path, stop_interferer  # noqa: E402

# ── 参照核: 同一算子, 一个快一个慢 ─────────────────────────────
# 取 matmul 1024³: 足够大到 launch 地板不支配 (doc 31), 但一轮 <1s。
GOOD = """
import numpy as np
def kernel(A, B):
    return np.asarray(A) @ np.asarray(B)
"""

SLOW = """
import numpy as np
def kernel(A, B):
    # 故意多算 8 遍, 只返回第一遍 —— 慢但数值正确
    out = np.asarray(A) @ np.asarray(B)
    for _ in range(8):
        out = np.asarray(A) @ np.asarray(B)
    return out
"""

BURNER = """
import numpy as np
def kernel(A, B):
    # 站在卡上的占位: 大矩阵乘连跑
    for _ in range(200):
        _ = np.asarray(A) @ np.asarray(B)
    return np.asarray(A) @ np.asarray(B)
"""


def _burner_command(gpu: int, log_path: str) -> str:
    """远端占卡命令。落点来自 `remote_scratch_path` (随机名, 不在世界可写区固定留名),
    argv 里带 `MARK=<落点本体>` —— 拆除的 pkill/pgrep/rm 三个动作全靠这一个
    不可预测路径精确定位。旧写法两处违规: 固定名 burner_gpu<N>.log 落在共享 /tmp
    (公开路径, 任何人可换), 拆除用内容模式 `pkill -f '2048,2048'` (会误伤任何
    恰好带着这串的进程, 且只请求不读回)。
    """
    return (
        f"CUDA_VISIBLE_DEVICES={gpu} cd ~/operator_agent && "
        f"nohup .venv/bin/python -c '"
        f'MARK="{log_path}";'
        f"import numpy as np;"
        f"A=np.random.randn(2048,2048).astype(np.float32);"
        f"B=np.random.randn(2048,2048).astype(np.float32);"
        f"import time;t0=time.time();"
        f"[(A@B) for _ in range(200)];"
        f"print(time.time()-t0)' "
        f"> {log_path} 2>&1 &"
    )


def _task():
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.MEDIUM),
        target=TargetSpec(hardware="nvidia_gpu", dsl="python"),
        constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=1.0,
                                   max_error=1e-3),
    )


def _contract():
    return freeze(
        kernel_type="matmul",
        dtype_semantics=DTypeSemantics(dtype="fp32", tf32=False, accum="fp32"),
        baseline=BaselineDef(name="pytorch_eager", same_device=True,
                             same_window=True, same_dtype_semantics=True),
        shapes=ShapeDist(entries=(((1024, 1024, 1024), 1.0),)),
        max_error=1e-3,
        min_speedup=1.0,
    )


def _inputs(n=1024, seed=0):
    rng = np.random.default_rng(seed)
    return {"A": rng.standard_normal((n, n)).astype(np.float32),
            "B": rng.standard_normal((n, n)).astype(np.float32)}


def bench_code(backend, code, inputs, warmup=3, repeat=10):
    """用远端同一把尺子测 median ms。返回 (ok, median_ms, note)。"""
    try:
        r = backend.benchmark(code, inputs, warmup=warmup, repeat=repeat)
        if not r.get("success", True) and r.get("error"):
            return False, None, str(r.get("error"))[:120]
        ms = r.get("median_time_ms") or r.get("avg_time_ms") or r.get("min_time_ms")
        return True, float(ms), "ok"
    except Exception as exc:  # noqa: BLE001
        return False, None, f"{type(exc).__name__}: {exc}"


def observe(label, contract, value_ms, baseline_ms, validity_hint=None,
            extra=None):
    """把一次真机读数送进 oracle, 打印资格位。"""
    speedup = (baseline_ms / value_ms) if (value_ms and value_ms > 0) else None
    extra = extra or {}
    obs = evaluate(
        contract,
        lambda: {
            "value": speedup,
            "validity": validity_hint or "valid",
            "real_device": True,
            **extra,
        },
        record_hash=contract.hash or contract.compute_hash(),
        require_real_device=True,
        metric="speedup_vs_baseline", unit="x",
    )
    print(f"\n[{label}] value={value_ms:.4g} ms  baseline={baseline_ms:.4g} ms  "
          f"speedup={speedup if speedup is None else round(speedup, 4)}")
    print(f"  validity={obs.validity}  eligibility={obs.eligibility.as_dict()}")
    if obs.reasons:
        print(f"  reasons={obs.reasons}")
    if obs.refused:
        print(f"  refused={obs.refused}")
    return obs, speedup


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-contention", action="store_true",
                    help="不跑竞争臂 (不占卡)")
    ap.add_argument("--n", type=int, default=1024, help="方阵边长")
    args = ap.parse_args()

    print("[smoke] 连接 algorithm-h20 …")
    backend = RemoteExecutor(mode="ssh", timeout=180)
    print(f"[smoke] backend={backend.label}  gpu={getattr(backend, 'selected_gpu', '?')}")

    contract = _contract()
    task = _task()
    xs = _inputs(args.n)
    slow_verdict, slow_ratio, obs4 = "?", 0.0, None

    # ── #1 已知好 kernel ─────────────────────────────────────
    print("\n" + "=" * 62 + "\n§14-1 已知好 kernel (matmul @%d³)\n" % args.n + "=" * 62)
    ok, good_ms, note = bench_code(backend, GOOD, xs)
    if not ok:
        print(f"  ❌ 测不了: {note}")
        return 1
    # 参考 baseline: 同一核再测一次当「同卡同窗」对照 (真机冒烟不需要 cuBLAS)
    ok2, base_ms, note2 = bench_code(backend, GOOD, xs)
    obs1, sp1 = observe("known-good", contract, good_ms, base_ms, "valid")
    assert obs1.eligibility.can_verdict, "已知好核居然不能判决"

    # 跨次 cv: 同核再测 3 次
    samples = []
    for i in range(3):
        _, ms, _ = bench_code(backend, GOOD, xs, warmup=1, repeat=5)
        samples.append(ms)
    med = float(np.median(samples))
    mad = float(np.median(np.abs(np.array(samples) - med)))
    print(f"  跨次读数: {['%.3f' % s for s in samples]}  median={med:.3f}  "
          f"mad/med={mad/med if med else 0:.4f}")

    # ── #2 故意写慢 ─────────────────────────────────────────
    print("\n" + "=" * 62 + "\n§14-2 故意写慢 kernel\n" + "=" * 62)
    ok, slow_ms, note = bench_code(backend, SLOW, xs)
    if not ok:
        print(f"  ❌ 测不了: {note}")
    else:
        obs2, sp2 = observe("known-slow", contract, slow_ms, good_ms, "valid")
        ratio = slow_ms / good_ms if good_ms else None
        slow_ratio = ratio or 0.0
        print(f"  慢/好 = {ratio:.2f}× (期望 ≥2×)")
        # verdict 收的是「越大越好」的轴 (docstring: current/previous 同轴, 越大越好)。
        # 耗时是「越小越好」—— 直接塞 ms 会把「更慢」判成 FAST (真机冒烟抓到)。
        # 换成吞吐 1/ms 再比。
        thr_slow, thr_good = 1000.0 / slow_ms, 1000.0 / good_ms
        v, why, _ = verdict(current=thr_slow, previous=thr_good,
                            validity=Validity.VALID)
        slow_verdict = v.value
        print(f"  verdict(slow vs good) = {v.value}  ({why[:80]})")
        print(f"  (吞吐 1/ms: slow={thr_slow:.3f} vs good={thr_good:.3f})")
        assert ratio and ratio >= 2.0, f"慢核不够慢: {ratio}"
        assert v is Verdict.SLOW, f"慢核居然判成 {v.value} —— §14-2 不通过"

    # ── #4 竞争负载 → DEGRADED ──────────────────────────────
    if args.skip_contention:
        print("\n(--skip-contention: 跳过竞争臂)")
    else:
        print("\n" + "=" * 62 + "\n§14-4 后台占卡 → DEGRADED\n" + "=" * 62)
        # 同一条 SSH 会把前后两次 benchmark 串行化 (第一次冒烟 util=0 就是这么来的),
        # 所以占位进程必须**在远端后台**起, 与测量并发。
        import subprocess

        alias = "algorithm-h20"
        gpu = getattr(backend, "selected_gpu", 0) or 0
        log_path = remote_scratch_path("burner.log")

        def run_remote(cmd: str) -> str:
            r = subprocess.run(["ssh", "-o", "BatchMode=yes", alias, cmd],
                               check=False, capture_output=True,
                               timeout=30, text=True)
            return r.stdout or ""

        print(f"  起远端占位: GPU {gpu} ×200 次 2048³ matmul (日志 {log_path})")
        run_remote(_burner_command(gpu, log_path))
        time.sleep(2.0)
        try:
            r = backend.benchmark(GOOD, xs, warmup=1, repeat=5)
            util = r.get("gpu_util_pct")
            print(f"  竞争窗口 util={util}")
            contaminated = util is not None and util > 5
            obs4, sp4 = observe(
                "contention", contract, good_ms, base_ms,
                validity_hint="degraded" if contaminated else "valid",
                extra={"cv_exceeded": bool(contaminated),
                       "partition_drift": False},
            )
            print(f"  contaminated={contaminated} → "
                  f"can_persist={obs4.eligibility.can_persist} "
                  f"(期望 False 当 util>5)")
            if not contaminated:
                print("  ⚠️ 未能造出竞争 (占位可能已结束或没抢到卡) —— "
                      "这一臂**不通过**, 不算数")
        finally:
            # 拆除要读到静默 (#69 / DSH :19): pkill→pgrep→rm 三步对着同一个
            # 随机路径, 读回残留数 —— 有残留的卡不许被当成安静读数
            ev = stop_interferer(None, run_remote, log_path)
            if ev.get("remote_still_running"):
                print(f"  ⚠️ {ev['warning']}")

    print("\n" + "=" * 62)
    print("真机冒烟结论 (§14-1/2/4)")
    print("=" * 62)
    print(f"  #1 已知好 → can_verdict={obs1.eligibility.can_verdict} / "
          f"can_persist={obs1.eligibility.can_persist}")
    print(f"  #2 已知慢 → verdict={slow_verdict}  慢/好={slow_ratio:.2f}×")
    print(f"  #4 竞争   → can_persist="
          f"{obs4.eligibility.can_persist if obs4 else 'skipped'}")
    print("  ⚠️ 全绿之前不宣称资格审计有收益 (§19.4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
