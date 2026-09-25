"""端到端验证真机 runner 的运行时屏蔽 (docs/research/03 洞 1)

在控制端运行 (会触发 deploy(), 需要能连到 GPU 机器):
    python3 benchmarks/verify_runtime_shield.py

要证明三件事, 缺一不可:
  1. deploy() 把 anticheat.py 一起推上去 —— 否则 runner 一 import 就炸;
  2. 静态审计漏掉的作弊核, 在真机上被运行时屏蔽拦住 (第二道防线确实生效);
  3. 合法 Triton 核照过 L2, 且 L3 的四个 baseline 都还有数字 —— 屏蔽没有
     误伤参考实现, 也没有把逐次 setattr 的开销压进计时区间。

第 2 条用的向量是 `op = getattr(torch, "matmul")` 然后 `op(A, B)`:
static_audit 只看 Call 节点, 这条赋值不是 Call, 后面的 `op(...)` 又解析不到
来源, 所以审计放行; 但 getattr 是在屏蔽上下文**内**执行的, 拿到的是抛错桩。
这也正说明为什么运行时屏蔽不能被静态审计取代。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from operator_agent.anticheat import static_audit
from operator_agent.remote_executor import RemoteExecutor
from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec, TargetSpec,
                                    TaskComplexity, TaskSchema, TaskType)
from operator_agent.verifier import Verifier

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
        acc += tl.dot(a, b__IP__)
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    tl.store(c_ptr + off_m[:, None] * scm + off_n[None, :] * scn, acc)

def kernel(A, B):
    M, K = A.shape
    N = B.shape[1]
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    BM, BN, BK = 64, 64, 32
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _mm[grid](A, B, C, M, N, K,
              A.stride(0), A.stride(1), B.stride(0), B.stride(1),
              C.stride(0), C.stride(1), BM=BM, BN=BN, BK=BK)
    return C
'''

# 静态审计放行、运行时屏蔽必须拦下
GETATTR_CHEAT = '''
def kernel(A, B):
    import torch
    op = getattr(torch, "matmul")
    return op(A, B)
'''


def make_task():
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.MEDIUM),
        target=TargetSpec(hardware="nvidia_gpu", dsl="triton", compute_units=78, warp_size=32),
        constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=0.0, max_error=1e-2))


backend = RemoteExecutor(mode="ssh")
print("=== 1. deploy: 应推送 tolerance.py + anticheat.py + run_kernel.py ===")
ok = backend.deploy()
print("deploy ->", ok)
if not ok:
    sys.exit(1)
remote_dir = Path(backend.backend.remote_runner_path).parent
print(backend.backend._ssh_exec(f"ls -1 {remote_dir}/*.py"))

v = Verifier(backend)
rng = np.random.default_rng(7)
n = 1024
inputs = {"A": (rng.standard_normal((n, n)) * 0.1).astype(np.float32),
          "B": (rng.standard_normal((n, n)) * 0.1).astype(np.float32)}
task = make_task()

print("\n=== 2. 静态审计漏过的作弊核, 真机运行时屏蔽是否拦住 ===")
audit = static_audit(GETATTR_CHEAT, "triton")
print(f"static_audit -> {audit or '放行 (符合预期: 赋值不是 Call 节点)'}")
r = v._verify_function(GETATTR_CHEAT, inputs, None, task)
print(f"L2 passed={r.passed}")
print(f"L2 message={r.message}")

print("\n=== 3. 合法 Triton 核: L2 照过, L3 的 baseline 都要有数字 ===")
# 同一个核跑两种 tl.dot 精度。这不是屏蔽的问题, 而是顺手量出来的一件事:
# Triton 的 tl.dot 在 fp32 输入下**默认走 TF32**, 于是「声明 fp32 的 matmul 任务」
# 会被容差表的 fp32 行 (atol/rtol=1e-4) 判不过。要真 fp32 必须显式写
# input_precision="ieee"。下面两组用来把这两件事分开: ieee 组过了, 就证明屏蔽
# 没有误伤合法核; 默认组不过, 是 TF32 与 fp32 容差不匹配, 与屏蔽无关。
#
# 注: 本想再加一组「任务声明 tf32」, 但 schemas.Precision 只有
# FP32/FP16/BF16/INT8 —— tolerance.TOLERANCE_BY_PRECISION 里的 "tf32" 行在
# 任务侧根本无法表达。这个缺口单独记录, 不在本 probe 范围内。
for label, ip in (
        ('tl.dot 默认 (fp32 输入 → TF32)', ''),
        ('tl.dot input_precision="ieee"', ', input_precision="ieee"')):
    code = TRITON_MM.replace("__IP__", ip)
    task = make_task()
    r = v._verify_function(code, inputs, None, task)
    m = r.metrics
    print(f"\n{label}\n  L2 passed={r.passed}  matched_ratio={m.get('matched_ratio')}  "
          f"bad={m.get('bad_elements')}/{m.get('compared_elements')}")
    print(f"  atol={m.get('atol')} rtol={m.get('rtol')} max_abs={m.get('max_error')} "
          f"max_rel={m.get('max_rel_error')}")
    if not r.passed:
        print(f"  诊断: {r.message}")

task = make_task()
bench = backend.benchmark_with_baselines(
    TRITON_MM.replace("__IP__", ""), inputs, kernel_type="matmul",
    warmup=3, repeat=10, precision="fp32")
print(f"\nL3 success={bench.get('success')}  gflops={bench.get('gflops')}")
for name, stats in (bench.get("baselines") or {}).items():
    if isinstance(stats, dict):
        print(f"  {name:20s} median={stats.get('median_time_ms')} ms")
    else:
        print(f"  {name:20s} ERROR -> {stats}")
print("\n基线里若出现 *_error 或 CheatDetectedError, 说明屏蔽把参考实现也打死了 —— "
      "参考必须留在 shield 之外。")
