"""真机标定「出处门」与「漂移门」 (doc 49 §7.9 / §八 第 9 条)

在控制端运行 (会触发 deploy(), 需要能连到 GPU 机器):
    OPERATOR_AGENT_GPU=2 python3 benchmarks/verify/verify_provenance_gate.py

要还的债: 那两道门在提交时**只有单测证据** —— 78 例全离线, 而真机上执行侧如实声明
`REAL_DEVICE`、任务声明的 `required_caps` 与它一致 ⇒ shortfall 恒空、门永远不开。
这里注入一个**已知会被看见**的信号: 要求一个真机确实没声明的能力, 看它是否真的拒绝把
L3 记成通过; 再伪造一次漂移, 看它是否只降档而**不**拒绝。

臂之间只差一个变量 —— `TargetSpec.required_caps` 或 `environment_drift`:

| 臂 | 注入 | 期望 |
|---|---|---|
| A | 要求真机已有的 {REAL_DEVICE, SAME_DEVICE_BASELINE} | L3 通过 |
| B | A + FULL_OUTPUT_TENSOR (真机**没**声明) | L3 **不**通过, 报 shortfall, 拒绝位为真 |
| C | 不声明 (老行为) | L3 通过 |
| D | 只要求 {DEPLOY} (真机有) | L3 通过 —— 证明门看的是**差集**, 不是「required_caps 非空」 |
| E | 漂移注入一项, 能力齐全 | L3 **照旧通过**, 档位降到 degraded —— 漂移不许判核失败 |

`min_speedup=0.0` 不是放水, 是实验设计: 它让 `passed` 在基线臂上恒真, 于是 B 臂上的任何
False 都只能来自那道门。`verify_measurement_validity.py` 用的是同一个口径, 同样是直调
`_verify_performance` (绕开 L1/L2, 把被测量隔离在 L3 这一层)。

⚠️ 本脚本只打印**派生值** (能力名、档位、布尔), 不打印远端原文与路径 —— 项目约定:
远程错误原文必然带那台机器的工作目录与用户名, 不进仓库。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from operator_agent.core.backend_caps import Cap
from operator_agent.decide.measurement import Validity
from operator_agent.execute.remote_executor import RemoteExecutor
from operator_agent.core.schemas import (ConstraintSpec, Precision, SourceSpec, TargetSpec,
                                    TaskComplexity, TaskSchema, TaskType)
from operator_agent.decide.verifier import Verifier

TRITON_LN = '''
import torch, triton, triton.language as tl

@triton.jit
def _ln(x_ptr, w_ptr, b_ptr, y_ptr, n_rows, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / n_cols
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * n_cols + cols,
             (xc * rstd * w + b).to(y_ptr.dtype.element_ty), mask=mask)

def kernel(X, weight, bias):
    n_rows, n_cols = X.shape
    Y = torch.empty_like(X)
    _ln[(n_rows,)](X, weight, bias, Y, n_rows, n_cols, 1e-5,
                   BLOCK=triton.next_power_of_2(n_cols))
    return Y
'''

ROWS, COLS = 16384, 6144      # 脱离 launch 地板的那一档 (doc 31 实测 402.7 MB)
BASE = (Cap.REAL_DEVICE, Cap.SAME_DEVICE_BASELINE)

# (标签, required_caps, 注入的漂移列表, 期望 passed)
ARMS = (
    ("A 要求真机已有的两项", BASE, None, True),
    ("B 再加一项真机没声明的 FULL_OUTPUT_TENSOR", BASE + (Cap.FULL_OUTPUT_TENSOR,),
     None, False),
    ("C 不声明 (老行为)", (), None, True),
    ("D 只要求 DEPLOY (真机有)", (Cap.DEPLOY,), None, True),
    ("E 能力齐全但伪造一次环境漂移", BASE, ["torch_version: '2.14.0' → '2.15.0'"], True),
)


def make_task(required):
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type="layernorm", complexity=TaskComplexity.MEDIUM),
        target=TargetSpec(hardware="nvidia_gpu", dsl="triton", compute_units=78,
                          warp_size=32, required_caps=required),
        constraints=ConstraintSpec(precision=Precision.FP16, min_speedup=0.0,
                                   max_error=1e-1))


def inputs():
    rng = np.random.default_rng(11)
    return {"X": rng.standard_normal((ROWS, COLS)).astype(np.float16),
            "weight": np.ones(COLS, dtype=np.float16),
            "bias": np.zeros(COLS, dtype=np.float16)}


def main() -> int:
    backend = RemoteExecutor(mode="ssh", timeout=900)
    backend.require(*BASE)
    declared = sorted(c.value for c in backend.capabilities)
    print(f"执行侧: {backend.label}  (degraded={backend.resolution.degraded})")
    print(f"它声明的能力: {declared}")
    missing = sorted(c.value for c in (set(Cap) - set(backend.capabilities)))
    print(f"它**没**声明的能力 (可用来注入已知信号): {missing}")
    if Cap.FULL_OUTPUT_TENSOR.value not in missing:
        print("❌ 前提变了: FULL_OUTPUT_TENSOR 现在被真机声明了, B 臂不再是已知信号")
        return 1
    if not backend.deploy():
        print("❌ 部署失败")
        return 1

    verifier = Verifier(backend)
    data = inputs()
    results = []
    for label, required, drift, expect in ARMS:
        print(f"\n{'=' * 66}\n[{label}]\n{'=' * 66}")
        r = verifier._verify_performance(TRITON_LN, data, 0.0, make_task(required),
                                         environment_drift=drift)
        m = r.metrics or {}
        got = bool(r.passed)
        refused = m.get("measurement_l3_pass_refused")
        shortfall = m.get("measurement_capability_shortfall")
        print(f"  passed={got}  有效性={m.get('measurement_validity')}  "
              f"can_gate={m.get('measurement_can_gate')}")
        print(f"  shortfall={shortfall!r}  矛盾={m.get('measurement_provenance_conflict')!r}")
        print(f"  拒绝位={refused!r}  漂移={m.get('measurement_environment_drift')!r}")
        print(f"  backend_real_device={m.get('backend_real_device')!r}")
        ok = (got == expect)
        print(f"  {'✅' if ok else '❌'} 期望 passed={expect}, 实得 {got}"
              f"{'' if ok else '   ← 量具看不见, 或门的方向反了'}")
        results.append((label, got, expect, shortfall, refused,
                        m.get("measurement_validity")))

    print(f"\n{'=' * 66}\n标定判据 (方向都口算过)\n{'=' * 66}")
    b = next(x for x in results if x[0].startswith("B"))
    a, c, d = (next(x for x in results if x[0].startswith(t)) for t in "ACD")
    e = next(x for x in results if x[0].startswith("E"))
    checks = [
        ("B 被拒绝而 A/C/D 照旧通过 (门只对差集起作用)",
         b[1] is False and a[1] and c[1] and d[1]),
        ("B 的 shortfall 里就是那一项, 且拒绝位为真",
         b[3] == [Cap.FULL_OUTPUT_TENSOR.value] and b[4] is True),
        ("拒绝位在 A/C/D 上为假 (不是恒真)",
         a[4] is False and c[4] is False and d[4] is False),
        ("B 的档位降到 uninformative",
         b[5] == Validity.UNINFORMATIVE.value),
        ("E 漂移只降到 degraded 且**不**拒绝通过 (观测先于判定)",
         e[1] is True and e[5] == Validity.DEGRADED.value and e[4] is False),
    ]
    bad = 0
    for name, passed_check in checks:
        print(f"  {'✅' if passed_check else '❌'} {name}")
        bad += 0 if passed_check else 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
