"""端到端验证 L2 数值门: 真 Triton 核过, 缩放 0.97 的同一核被拦下

在控制端运行 (会触发 SSHExecutor.deploy(), 需要能连到 GPU 机器):
    python3 benchmarks/verify/verify_l2_gate_e2e.py

结论记录在 docs/research/30-L2数值门真机验证.md 第一节。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from operator_agent.remote_executor import RemoteExecutor
from operator_agent.verifier import Verifier
from operator_agent.schemas import (TaskSchema, SourceSpec, TargetSpec, ConstraintSpec,
                                    TaskType, TaskComplexity, Precision)

TRITON_MM = '''
import torch, triton, triton.language as tl

@triton.jit
def _mm(a_ptr, b_ptr, c_ptr, M, N, K,
        sam, sak, sbk, sbn, scm, scn,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    off_m = pid_m * BM + tl.arange(0, BM)
    off_n = pid_n * BN + tl.arange(0, BN)
    a_ptrs = a_ptr + off_m[:, None] * sam + tl.arange(0, BK)[None, :] * sak
    b_ptrs = b_ptr + tl.arange(0, BK)[:, None] * sbk + off_n[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K, BK):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    c = (acc * __SCALE__).to(tl.float16)
    tl.store(c_ptr + off_m[:, None] * scm + off_n[None, :] * scn, c)

def kernel(A, B):
    M, K = A.shape
    N = B.shape[1]
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    BM, BN, BK = 64, 64, 32
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _mm[grid](A, B, C, M, N, K,
              A.stride(0), A.stride(1), B.stride(0), B.stride(1),
              C.stride(0), C.stride(1), BM=BM, BN=BN, BK=BK)
    return C
'''

def task(max_error):
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.MEDIUM),
        target=TargetSpec(hardware="nvidia_gpu", dsl="triton", compute_units=78, warp_size=32),
        constraints=ConstraintSpec(precision=Precision.FP16, min_speedup=0.0, max_error=max_error))

backend = RemoteExecutor(mode="ssh")
ssh = backend.backend            # facade 不暴露 deploy(), 取里面的 SSHExecutor
print("=== 实际后端:", ssh.name, "===")
print("=== deploy (应推送 tolerance.py + run_kernel.py) ===")
ok = ssh.deploy()
print("deploy ->", ok)
if not ok:
    sys.exit(1)

v = Verifier(backend)
rng = np.random.default_rng(7)
n = 1024
inputs = {"A": (rng.standard_normal((n, n)) * 0.1).astype(np.float16),
          "B": (rng.standard_normal((n, n)) * 0.1).astype(np.float16)}

for label, scale, max_error in (
        ("正确核 (SCALE=1.0), driver 式 max_error=5e-1", 1.0, 5e-1),
        ("正确核 (SCALE=1.0), schema 默认 max_error=1e-5", 1.0, 1e-5),
        ("缩放 0.97 的错核, driver 式 max_error=5e-1", 0.97, 5e-1),
        ("缩放 0.999 的错核, driver 式 max_error=5e-1", 0.999, 5e-1)):
    code = TRITON_MM.replace("__SCALE__", repr(scale))
    r = v._verify_function(code, inputs, None, task(max_error))
    m = r.metrics
    print(f"\n{label}\n  passed={r.passed}  matched_ratio={m.get('matched_ratio')}  "
          f"bad={m.get('bad_elements')}/{m.get('compared_elements')}")
    print(f"  atol={m.get('atol')} rtol={m.get('rtol')} max_abs={m.get('max_error')} "
          f"max_rel={m.get('max_rel_error')}")
    if not r.passed:
        print(f"  诊断: {r.message}")
