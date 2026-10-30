"""CSA 压缩算子试打 · 本地冒烟 (不需要 GPU / LLM)

目的: 验证 P1「可外挂参考」这条缝真的通了 ——
  任务带 `reference_source` → L2 能对上 → L3 能拿到 eager 基线。
真机试打 (Conductor + SSH) 是下一步; 这里只证明闭环存在。

    python3 benchmarks/experiments/misc/try_csa_compress.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from operator_agent.ops import csa_compress  # noqa: E402
from operator_agent.schemas import (  # noqa: E402
    ConstraintSpec, Precision, SourceSpec, TargetSpec, TaskComplexity, TaskSchema, TaskType,
)
from operator_agent.verifier import Verifier  # noqa: E402
from operator_agent.mock_gpu import MockGPU  # noqa: E402

# 一份「正确但不快」的候选: NumPy 实现同一公式。
# python DSL 下 NumPy 是实现介质 (anticheat 只禁 torch), 所以它能过静态审计。
# 它的价值是把 L2 对拍链路打穿; 性能侧是下一轮的事。
CANDIDATE = '''
import numpy as np

def kernel(C, Z, B_a, B_b):
    C = np.asarray(C, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    B_a = np.asarray(B_a, dtype=np.float64)
    B_b = np.asarray(B_b, dtype=np.float64)
    n, c = C.shape
    m = B_a.shape[0]
    n_out = n // m
    outs = np.empty((n_out, c), dtype=np.float64)
    neg = np.full((m, c), -np.inf)
    zeros = np.zeros((m, c))
    for i in range(n_out):
        ca = C[i * m:(i + 1) * m]
        za = Z[i * m:(i + 1) * m] + B_a
        if i == 0:
            cb, zb = zeros, neg
        else:
            cb = C[(i - 1) * m:i * m]
            zb = Z[(i - 1) * m:i * m] + B_b
        logits = np.concatenate([za, zb], axis=0)
        logits = logits - logits.max(axis=0, keepdims=True)
        ex = np.exp(logits)
        s = ex / ex.sum(axis=0, keepdims=True)
        outs[i] = (s[:m] * ca).sum(axis=0) + (s[m:] * cb).sum(axis=0)
    return outs.astype(np.float32)
'''

# 一份故意写错的候选 (权重归一错成按行): 用来确认 L2 真的在比, 不是空转
WRONG_CANDIDATE = '''
import numpy as np

def kernel(C, Z, B_a, B_b):
    return np.zeros((C.shape[0] // B_a.shape[0], C.shape[1]), dtype=np.float32)
'''


def make_task() -> TaskSchema:
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(
            kernel_type="csa_compress",
            complexity=TaskComplexity.HIGH,   # 融合/新算子走 AGENT_GENERATION
            reference_source=csa_compress.REFERENCE_SOURCE,
        ),
        target=TargetSpec(hardware="mock_gpu", dsl="python"),
        constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=1.0, max_error=1e-4),
    )


def main() -> int:
    xs = csa_compress.make_inputs(n=32, c=16, m=4, seed=0)
    task = make_task()
    verifier = Verifier(MockGPU())

    fn, kind = verifier._build_reference_fn(task, xs)
    print(f"[ref] resolve → {kind}")
    assert kind == "reference_source", f"外挂参考没被选中: {kind}"

    print("\n=== ① 正确候选应当 L2 过 ===")
    results = verifier._verify_function(CANDIDATE, xs, None, task)
    print(f"  passed={results.passed}  {results.message}")
    if not results.passed:
        print("  ❌ 正确候选没过 L2 —— 参考或对拍链路有问题")
        return 1

    print("\n=== ② 错误候选应当 L2 挂 ===")
    bad = verifier._verify_function(WRONG_CANDIDATE, xs, None, task)
    print(f"  passed={bad.passed}  {bad.message}")
    if bad.passed:
        print("  ❌ 错误候选也过了 —— L2 在空转")
        return 1

    print("\n=== ③ 本地 L3 基线 ===")
    baseline = verifier._measure_baseline(task, xs, warmup=1, repeat=3)
    print(f"  baseline={baseline}")
    if baseline is None:
        print("  ❌ 没拿到基线 —— 新算子 L3 闭环仍不存在")
        return 1

    print("\n✅ P1 缝已通: 外挂参考 → L2 对拍 → L3 基线")
    print("下一步: 把 REFERENCE_SOURCE 塞进 Conductor 的真机试打 (需要 ssh + LLM)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
