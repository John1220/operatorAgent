"""真机探针: 评测器手里到底有哪些「编译产物元数据」(为 §10.3 条件 2 找真源)

回答的问题: `num_warps` / `n_regs` / `n_spills` 这些**关于被测量那份码**的事实,
今天有没有从远端 runner 走到控制端? 有 ⇒ 只改控制侧的发布; 没有 ⇒ 必须动 runner,
而且要动得对(编译产物的真源, 不是控制端解析源码文本)。

用法: python benchmarks/probes/probe_compile_metadata.py   (约 25s, 只跑一个点)
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from operator_agent.app._harness_controls import enable_for_harness
enable_for_harness()
os.environ.setdefault("OPERATOR_AGENT_KERNEL_DB", "off")
os.environ.setdefault("OPERATOR_AGENT_FAILURE_DB", "off")

import numpy as np
from operator_agent.execute.remote_executor import RemoteExecutor
from operator_agent.core.schemas import (ConstraintSpec, Precision, SourceSpec, TaskSchema,
                                    TaskType, TargetSpec)
from operator_agent.generate.triton_matmul_template import instantiate
from operator_agent.decide.verifier import Verifier

task = TaskSchema(task_id="probe-meta", task_type=TaskType.CODE_TRANSLATION,
                  source=SourceSpec(kernel_type="matmul"),
                  target=TargetSpec(hardware="nvidia_gpu", dsl="triton"),
                  constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=0.1,
                                             max_error=1e-2))
rng = np.random.default_rng(3)
inputs = {"A": rng.standard_normal((1024, 1024)).astype(np.float32),
          "B": rng.standard_normal((1024, 1024)).astype(np.float32)}
src = instantiate({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32,
                   "stages": 2, "warps": 8}).source
for r in Verifier(RemoteExecutor(mode="ssh")).verify(src, task, inputs, target_gflops=200.0):
    m = r.metrics or {}
    print(f"-- {r.level.name}: {len(m)} 个键")
    if r.level.name in ("L1_COMPILE", "L3_PERFORMANCE"):
        keys = sorted(m)
        print("   全部键:", ", ".join(keys))
        for k, v in sorted(m.items()):
            if any(s in k.lower() for s in ("warp", "reg", "spill", "compile", "thread", "smem")):
                print(f"   ★ {k} = {v!r}")
