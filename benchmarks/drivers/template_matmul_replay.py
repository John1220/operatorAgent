"""模板路径的真机复跑（§12 真模板后端 → H20）

用法:
    python benchmarks/drivers/template_matmul_replay.py 128 128 32 2 8
    python benchmarks/drivers/template_matmul_replay.py 64 64 64 3 4
参数: BLOCK_M BLOCK_N BLOCK_K stages warps

这条脚本回答的问题是**参数的因果性**: 同一个模板、只换参数点, 真机读数会不会跟着变?
—— 若不会变, 那么整个 §12 参数空间 + bandit 搜索都是空转（搜索的对象不存在）。
它**不**经过 Conductor 的模板接线（那一步还没做）, 只把「点 → 源码 → 评测器」这一段
放在真机上; 因此产物里没有候选队列/覆盖判据那一半, 不许把它读成「整条 Stage 3 已端到端」。
"""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from operator_agent._harness_controls import enable_for_harness
enable_for_harness()
os.environ.setdefault("OPERATOR_AGENT_KERNEL_DB", "off")
os.environ.setdefault("OPERATOR_AGENT_FAILURE_DB", "off")

import numpy as np
from operator_agent.remote_executor import RemoteExecutor
from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec, TaskSchema,
                                    TaskType, TargetSpec)
from operator_agent.triton_matmul_template import instantiate
from operator_agent.verifier import Verifier

bm, bn, bk, stages, warps = (int(x) for x in sys.argv[1:6])
point = {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "stages": stages, "warps": warps}
inst = instantiate(point)
print("实例化:", inst.evidence, flush=True)

task = TaskSchema(task_id=f"tmpl-{bm}x{bn}x{bk}s{stages}w{warps}",
                  task_type=TaskType.CODE_TRANSLATION,
                  source=SourceSpec(kernel_type="matmul"),
                  target=TargetSpec(hardware="nvidia_gpu", dsl="triton"),
                  constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=0.1,
                                             max_error=1e-2))
rng = np.random.default_rng(11)
inputs = {"A": rng.standard_normal((4096, 4096)).astype(np.float32),
          "B": rng.standard_normal((4096, 4096)).astype(np.float32)}
t0 = time.monotonic()
results = Verifier(RemoteExecutor(mode="ssh")).verify(inst.source, task, inputs,
                                                      target_gflops=200.0)
out = {"point": point, "instantiator_ver": inst.instantiator_ver, "evidence": inst.evidence}
for r in results:
    print(f"-- {r.level.name} passed={r.passed} :: {r.message[:120]}", flush=True)
    for key in ("compile_status", "static_guards", "max_error", "speedup_vs_baseline",
                "measurement_validity", "gflops", "gpu_util_pct"):
        if key in (r.metrics or {}):
            out[f"{r.level.name}.{key}"] = r.metrics[key]
    if "num_stages" in (r.metrics or {}) or "num_warps" in (r.metrics or {}):
        out["launch"] = {"num_stages": r.metrics.get("num_stages"),
                         "num_warps": r.metrics.get("num_warps")}
print("RESULT " + json.dumps(out, ensure_ascii=False), flush=True)
print("wall_s", round(time.monotonic() - t0, 1), flush=True)
