"""I68 修复的真机验收: 两个 harness 进程同时测同一张卡, claim 声明必须让
「时刻采样看不见的对手」现形。

对照 (修复前, I68 实测, 2026-10-02 同机同形状 4096³ fp32):
  串行基线 6.2ms; 并发 A=13.6ms 档位 **valid** / B=11.62ms uninformative,
  两条臂 gpu_util_pct 都是 0 —— util 是 ~1s 窗口的时刻采样、compute-apps 只见
  已建 CUDA 上下文的进程, 对手还在 import torch 时全瞎。false-valid 就是这条。

修复 (CHECKLOG I68): runner import 时写**进程级** claim (~/.operator_agent_gpu_claims,
O_EXCL + 0600, atexit 拆除), benchmark 动作前后各扫一次同卡声明; 控制端
`measurement.assess(gpu_harness_peers=…)` 降 DEGRADED, `DeviceLease` 记 contended。

判据 (修复后, 全部要过):
  C1 两臂落同一张卡 (selected_gpu 相同) —— 否则这次跑不构成对照
  C2 每臂 gpu_harness_peers >= 1 (claim 扫描看见对手; util 仍可为 0)
  C3 档位 degraded 且理由点名 harness (I68 的 false-valid 不再发生)
  C4 租约 competition=contended, evidence 带「同卡 harness 进程」
并报 (对照读数, 不设门): 串行基线 ms、并发 ms、膨胀倍数、util、离散度。

跑法:
    python3 benchmarks/verify_cross_process_claims.py
    python3 benchmarks/verify_cross_process_claims.py --skip-serial   # 跳过基线臂
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ARM_TIMEOUT_S = 300


def run_arm(arm_id: str) -> int:
    """一个臂 = 一个独立 harness 进程 (真机 L3)。由 main() 以子进程方式再入本文件。"""
    import numpy as np

    from operator_agent.remote_executor import RemoteExecutor
    from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec,
                                        TaskSchema, TaskType, TargetSpec)
    from operator_agent.triton_matmul_template import instantiate
    from operator_agent.verifier import Verifier

    # 已知点 (I70 实测 5.7~6.2ms 串行): 足够大, launch 地板不支配; 足够快, 一臂 <1min
    inst = instantiate({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32,
                        "stages": 1, "warps": 4})
    task = TaskSchema(task_id=f"i68-claim-{arm_id}",
                      task_type=TaskType.CODE_TRANSLATION,
                      source=SourceSpec(kernel_type="matmul"),
                      target=TargetSpec(hardware="nvidia_gpu", dsl="triton"),
                      constraints=ConstraintSpec(precision=Precision.FP32,
                                                 min_speedup=0.1, max_error=1e-2))
    rng = np.random.default_rng(11)     # 与 replay/e2e 同一种子 ⇒ 读数直接对表
    inputs = {"A": rng.standard_normal((4096, 4096)).astype(np.float32),
              "B": rng.standard_normal((4096, 4096)).astype(np.float32)}
    t0 = time.monotonic()
    results = Verifier(RemoteExecutor(mode="ssh")).verify_l3_only(
        inst.source, inputs, 0.0, task)
    wall = time.monotonic() - t0
    l3 = results[-1]
    m = l3.metrics or {}
    lease = m.get("device_lease") or {}
    out = {"arm": arm_id, "wall_s": round(wall, 1), "passed": bool(l3.passed),
           "candidate_ms": m.get("candidate_median_time_ms") or m.get("median_time_ms"),
           "speedup": m.get("speedup_vs_baseline"),
           "gpu_util_pct": m.get("gpu_util_pct"),
           "harness_peers": m.get("gpu_harness_peers"),
           "peer_scans": m.get("gpu_harness_peer_scans"),
           "validity": m.get("measurement_validity"),
           "reasons": m.get("measurement_validity_reasons"),
           "lease_competition": lease.get("competition"),
           "lease_harness_peers": lease.get("harness_peers"),
           "selected_gpu": m.get("selected_gpu")}
    print("ARM_RESULT " + json.dumps(out, ensure_ascii=False), flush=True)
    return 0


def _spawn_and_collect(arm_id: str):
    """跑一个臂子进程, 返回 ARM_RESULT 那一行解析出的 dict。"""
    proc = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                           "--arm", arm_id],
                          capture_output=True, text=True, timeout=ARM_TIMEOUT_S)
    line = next((l for l in reversed(proc.stdout.splitlines())
                 if l.startswith("ARM_RESULT ")), None)
    if line is None:
        raise RuntimeError(f"臂 {arm_id} 没有 ARM_RESULT 行:\n{proc.stdout[-2000:]}\n"
                           f"{proc.stderr[-2000:]}")
    return json.loads(line[len("ARM_RESULT "):])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", help="内部用: 以子进程身份跑一个臂")
    ap.add_argument("--skip-serial", action="store_true",
                    help="跳过串行基线臂 (它同时是部署预热)")
    args = ap.parse_args()
    if args.arm:
        return run_arm(args.arm)

    baseline = None
    if not args.skip_serial:
        # 串行基线兼作**部署预热**: 先把新 run_kernel.py / gpu_claim.py 推上去,
        # 两臂并发时就不再有并发 scp 同一文件的竞争 (那个竞争本身值得另挂号,
        # 不属于本片的判据)。
        print("── 串行基线臂 (部署预热) ──", flush=True)
        baseline = _spawn_and_collect("S")

    print("── 并发两臂 (同一张卡, I68 的场景) ──", flush=True)
    procs = {}
    for arm in ("A", "B"):
        procs[arm] = subprocess.Popen(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--arm", arm],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def collect(arm):
        out, err = procs[arm].communicate(timeout=ARM_TIMEOUT_S)
        line = next((l for l in reversed(out.splitlines())
                     if l.startswith("ARM_RESULT ")), None)
        if line is None:
            raise RuntimeError(f"臂 {arm} 没有 ARM_RESULT 行:\n{out[-2000:]}\n{err[-2000:]}")
        return json.loads(line[len("ARM_RESULT "):])

    a, b = collect("A"), collect("B")
    for tag, r in (("serial", baseline), ("A", a), ("B", b)):
        if r:
            print(f"{tag:6s} gpu={r['selected_gpu']} ms={r['candidate_ms']} "
                  f"peers={r['harness_peers']} (scans={r['peer_scans']}) "
                  f"util={r['gpu_util_pct']}% validity={r['validity']} "
                  f"lease={r['lease_competition']}", flush=True)
    if baseline and a.get("candidate_ms") and baseline.get("candidate_ms"):
        ratio = a["candidate_ms"] / baseline["candidate_ms"]
        print(f"并发/串行 膨胀 = {ratio:.2f}× (I68 修复前实测 2.1×; 修复后仍会膨胀 —— "
              f"claim 治的是「看不见」, 不是「不发生」)", flush=True)

    checks = {
        "C1 两臂同一张卡": a.get("selected_gpu") is not None
                          and a.get("selected_gpu") == b.get("selected_gpu"),
        "C2 两臂都看见对手 (peers>=1)": all(
            isinstance(r.get("harness_peers"), int) and r["harness_peers"] >= 1
            for r in (a, b)),
        "C3 档位 degraded 且理由点名 harness (false-valid 消失)": all(
            r.get("validity") == "degraded"
            and any("harness" in x for x in (r.get("reasons") or []))
            for r in (a, b)),
        "C4 租约 contended": all(r.get("lease_competition") == "contended"
                                 for r in (a, b)),
    }
    ok = True
    for name, passed in checks.items():
        print(f"  {'✓' if passed else '✗'} {name}", flush=True)
        ok = ok and passed
    print(f"VERDICT {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
