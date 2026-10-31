"""真机复跑探针（CHECKLOG I67 / doc 79 的读数从这里可重建）

用法（本机无卡也能跑, 测量在远端）:
    python benchmarks/drivers/real_machine_replay.py

- 端点/卡位全部来自环境与 `~/.ssh/config`（仓库不存真实地址）:
  `REMOTE_SSH_HOST`（默认 ssh config 别名 algorithm-h20）+ `REMOTE_WORK_DIR`；
- **不经 LLM**（所以没有 `LLM_BASE_URL` 也能跑）: 核用仓里语料
  `kernel_db._TRITON_MATMUL_CODE`, 只把 `tl.dot` 改成 `input_precision="ieee"`
  （默认 TF32 在 fp32 容差 1e-4 下过不了 L2 —— 那本身是 doc 79 的观察①）；
- 输出: 逐级的 metrics / `rule_inputs` / 资格位 / 峰值标定, 以及
  `verify_l3_only(static_guards=False)` 那一路（规则 8 的真机等价物）；
- 复跑一次约 25s（SSH + uv + triton JIT + 基准）。上一次读数见 doc 79。
"""
import json, os, sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from operator_agent.app._harness_controls import enable_for_harness
enable_for_harness()
os.environ.setdefault("OPERATOR_AGENT_KERNEL_DB", "off")
os.environ.setdefault("OPERATOR_AGENT_FAILURE_DB", "off")


import numpy as np, torch
from operator_agent.core.kernel_db import _TRITON_MATMUL_CODE as _SEED
CODE = _SEED.replace('tl.dot(a, b, acc)', 'tl.dot(a, b, acc, input_precision="ieee")')
from operator_agent.execute.remote_executor import RemoteExecutor
from operator_agent.decide.verifier import Verifier
from operator_agent.core.schemas import (ConstraintSpec, Precision, SourceSpec, TargetSpec,
                                    TaskSchema, TaskType)

task = TaskSchema(task_id="real-smoke", task_type=TaskType.CODE_TRANSLATION,
                  source=SourceSpec(kernel_type="matmul"),
                  target=TargetSpec(hardware="nvidia_gpu", dsl="triton"),
                  constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=0.1,
                                             max_error=1e-2))
rng = np.random.default_rng(0)
inputs = {"A": rng.standard_normal((4096, 4096)).astype(np.float32),
          "B": rng.standard_normal((4096, 4096)).astype(np.float32)}

t0 = time.monotonic()
exc = RemoteExecutor(mode="ssh")
verifier = Verifier(exc)
print("== 后端:", exc, flush=True)
results = verifier.verify(CODE, task, inputs, target_gflops=200.0)
print("== L1/L2/L3 用时 %.1fs" % (time.monotonic() - t0), flush=True)
for r in results:
    print(f"-- {r.level.name} passed={r.passed} :: {r.message[:160]}", flush=True)
    keep = ("compile_status", "static_guards", "static_guards_lang", "max_error",
            "speedup_vs_baseline", "measurement_validity", "benchmark_contaminated",
            "device_lease", "arm_windows", "gpu_util_pct", "dispersion_iqr_over_median",
            "candidate_peak_memory_bytes", "baseline_peak_memory_bytes")
    print("   ", {k: r.metrics.get(k) for k in keep if k in r.metrics}, flush=True)
    if "observation" in r.metrics:
        obs = r.metrics["observation"]
        print("    rule_inputs:", json.dumps(obs["provenance"]["rule_inputs"], ensure_ascii=False), flush=True)
        print("    eligibility:", json.dumps(obs["eligibility"], ensure_ascii=False), flush=True)
        print("    hash_check:", obs.get("hash_check"), "| reasons:", obs["reasons"][:3], flush=True)
        print("    peak 标定:", (r.metrics.get("peak_probe") or {}).get("reason", r.metrics.get("peak_probe")), flush=True)
    if not r.passed:
        print("    error_log:", (r.error_log or "")[:400], flush=True)

print("\n== 真机上把适用性传成 False (I65 的生产路径) ==", flush=True)
l3 = verifier.verify_l3_only(CODE, inputs, 20.0, task, static_guards=False)[0]
print("speedup:", l3.metrics.get("speedup_vs_baseline"),
      "| validity:", l3.metrics.get("measurement_validity"), flush=True)
print("rule_inputs['static_guards']:",
      l3.metrics["observation"]["provenance"]["rule_inputs"]["static_guards"], flush=True)
print("eligibility:", json.dumps(l3.metrics["eligibility"], ensure_ascii=False), flush=True)
print("reasons:", [x for x in l3.metrics["observation"]["reasons"] if "守卫" in x], flush=True)
