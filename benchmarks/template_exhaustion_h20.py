"""§10.4/§13.5 停止出口的真机观测：穷尽 + 两条件停滞第一次 fire（H20）

I72 的边界明记：48 点空间 3 轮跑不满，§10.4「有希望的区域已穷尽」从未在真机上
fire 过；§13.5 收敛出口同理。这一跑回答三件事：

  O1  explore 相位里 stagnation 会不会先掐死循环？—— 预期**不会**：阈值到了但
      覆盖判据说 unexplored ⇒ declined 留痕继续跑（§10.4 的设计点）, 这将是它
      第一次在真机上被观测到「拒绝只因轮数停止」
  O2  全部点试完 + K 轮无改善（verdict 非 FAST, 带噪声带）⇒ 两条件停滞出口
      fire, 停机理由点名「已穷尽 (N/N)」
  O3  §13.5 天花板出口预计**不** fire（最优模板点 ~36% of 参考下界, 远低于阈值）——
      不 fire 也是读数, 如实记

做法: smem 上界压到 32 KiB 把空间缩到 **12 点**（1×1×2×3×2 —— 缩格是
`param_space()` 的真实机制, 不是旁路）; min_speedup=0.5（实测点位 0.32~0.37x
必不过 ⇒ 循环只能靠设计的出口停）; max_rounds=18（12 轮探完 + 余量）;
heartbeat_interval 关掉（隔离观测变量 —— 心跳自身的真机观测另开一片）。
不经 LLM（chat 被调即炸: 升级门评估点是纯判据, 若它调 LLM 那本身就是读数）。

跑法: python3 benchmarks/template_exhaustion_h20.py [--rounds 18]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("OPERATOR_AGENT_KERNEL_DB", "off")
os.environ.setdefault("OPERATOR_AGENT_FAILURE_DB", "off")
os.environ.setdefault("OPERATOR_AGENT_LOG_KEEP", "100")

import numpy as np  # noqa: E402

from operator_agent.conductor import Conductor  # noqa: E402
from operator_agent.remote_executor import RemoteExecutor  # noqa: E402
from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec,  # noqa: E402
                                    TaskSchema, TaskComplexity, TaskType, TargetSpec)
from operator_agent.triton_matmul_template import TritonTemplateSSHExecutor  # noqa: E402

#: 32 KiB ⇒ BM=BN=64 (128 的 tile 放不下), BK 32/64 都留 ⇒ 1×1×2×3×2 = 12 点
SMEM_LIMIT = 32 * 1024


class _DesignerOnlyLLM:
    """与 template_e2e_h20 同一款桩: 只供 Designer 一次 sketch; chat 被调即炸。"""

    def __init__(self):
        self.chat_calls = 0

    def chat(self, *a, **k):
        self.chat_calls += 1
        raise AssertionError("观测跑不许调 LLM chat (Coder/Fixer/Search/升级门未被跳过)")

    def chat_json(self, *a, **k):
        return {"algorithm": "matmul",
                "tiling": {"block_m": 64, "block_n": 64, "block_k": 32},
                "memory_plan": {}, "pipeline": [], "optimization_hints": []}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rounds", type=int, default=18)
    args = ap.parse_args()

    backend = TritonTemplateSSHExecutor(RemoteExecutor(mode="ssh"),
                                        smem_limit_bytes=SMEM_LIMIT, elem_bytes=4)
    space_size = backend.param_space().size()
    print(f"空间: {space_size} 点 (smem 上界 {SMEM_LIMIT // 1024} KiB 缩格)", flush=True)
    assert space_size == 12, f"缩格口径变了: 预期 12 点, 实得 {space_size} —— 先改断言再跑"

    llm = _DesignerOnlyLLM()
    task = TaskSchema(task_id="tmpl-exhaustion",
                      task_type=TaskType.OPERATOR_GENERATION,
                      source=SourceSpec(kernel_type="matmul",
                                        complexity=TaskComplexity.MEDIUM),
                      target=TargetSpec(hardware="nvidia_gpu", dsl="triton"),
                      constraints=ConstraintSpec(precision=Precision.FP32,
                                                 min_speedup=0.5, max_error=1e-2))
    rng = np.random.default_rng(11)
    inputs = {"A": rng.standard_normal((4096, 4096)).astype(np.float32),
              "B": rng.standard_normal((4096, 4096)).astype(np.float32)}

    conductor = Conductor(backend=backend, llm=llm, max_optimize_rounds=args.rounds,
                          heartbeat_interval=999)   # 隔离观测变量, 边界见 docstring
    t0 = time.monotonic()
    art = conductor.execute(task, inputs, target_gflops=0.0)
    wall = time.monotonic() - t0

    hist = art.iteration_history
    meta = art.metadata or {}
    stop = meta.get("stop") or {}
    coverage = meta.get("coverage") or {}
    rounds = [{
        "round": h.get("round", i + 1),
        "point": (h.get("template_selection") or {}).get("chosen"),
        "arm": json.dumps(h.get("bandit_arm"), default=str),
        "gflops": h.get("gflops"),
        "phase": h.get("phase"),
        "cov": (h.get("coverage") or {}).get("state"),
        "threshold": (h.get("coverage") or {}).get("threshold_reached"),
    } for i, h in enumerate(hist)]
    distinct = {r["arm"] for r in rounds}
    out = {"wall_s": round(wall, 1), "space": space_size, "rounds_run": len(rounds),
           "distinct_points": len(distinct), "stop": stop,
           "llm_chat": llm.chat_calls,
           "coverage_ledger": coverage, "rounds": rounds}
    print("RESULT " + json.dumps(out, ensure_ascii=False, default=str), flush=True)

    # ---- 判据: 每条独立报 ----
    checks = {
        f"C1 全部 {space_size} 点都被试过 (distinct==space)":
            len(distinct) == space_size,
        "C2 停在设计的出口 (stagnant/converged), 不是轮数用尽":
            stop.get("reason") in ("stagnant", "converged"),
        "C3 探索期出现过 declined 停滞 (阈值到但未穷尽 ⇒ 继续)":
            any(r["threshold"] and r["cov"] not in ("exhausted", None)
                for r in rounds),
        "C4 LLM chat 零调用 (升级门评估点是纯判据)":
            llm.chat_calls == 0,
    }
    if stop.get("reason") == "stagnant":
        checks["C2b 停机理由点名两半 (穷尽 + N/N)"] = (
            "穷尽" in (stop.get("detail") or "")
            and f"{space_size}/{space_size}" in (stop.get("detail") or ""))
    for k, v in checks.items():
        print(f"  {'✓' if v else '✗'} {k}", flush=True)
    # 报告不设门: 收敛出口与升级门的读数
    best = max((r["gflops"] or 0) for r in rounds) if rounds else 0
    print(f"  (读数) 最优 {best:.0f} GFLOPS ≈ {best / 70800 * 100:.0f}% of 参考下界"
          f" 70.8T —— §13.5 收敛出口预期不 fire", flush=True)
    ok = all(checks.values())
    print(f"VERDICT {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
