"""真机 A/B: 处方式错误反馈 (docs/research/20 C1) 到底有没有让修复更容易

在控制端运行 (需要 GPU 机 + LLM 端点, 端点走环境变量不进仓库):
    python3 benchmarks/verify/verify_prescriptive_feedback.py

要验的是从 SPARC (arXiv:2607.09803) 吸纳来的那条结论, 在**本项目自建模型**上是否成立:
同一个真机实测到的失败, 只换"错误怎么摊给模型"这一件事 ——

    裸臂 (prescriptive=False): `[L2] 反作弊: ...` + 一段 traceback
    处方臂 (prescriptive=True): 外部归因段 + `Wait` + 要求 / 该改哪里

两臂同温度、同 max_tokens、同一份失败文本、一次生成不复述, 然后拿生成的代码回到
H20 上过真机 L1 (远程编译) + L2 (数值功能)。

单元格重复次数 REPS: 1 次不足以区分运气, 但每格都要一次 LLM 调用 + 一次远程执行,
所以默认只给 2 次 —— 这是方向性探针, 不是统计检验 (报告里要按这个口径读)。
"""
import os
import sys
import textwrap
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from operator_agent.remote_executor import RemoteExecutor
from operator_agent.verifier import Verifier
from operator_agent.fixer import FixerAgent, prescribe
from operator_agent.prompt_manager import PromptManager
from operator_agent.llm_client import LLMClient
from operator_agent.schemas import (
    ConstraintSpec, Precision, SourceSpec, TargetSpec,
    TaskComplexity, TaskSchema, TaskType,
)

REPS = int(os.environ.get("REPS", "2"))
COLLECT_ONLY = bool(os.environ.get("COLLECT_ONLY"))   # 只采失败原文, 不打 LLM

# ─── 候选核: 四个不同产出方的真实失败 ────────────────────────
# 每个都是"模型真会写出来"的错法, 不是为报错而报错的假代码。

LN_HEAD = '''
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
'''

LN_TAIL = '''
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

# 产出方 = 远程 Triton 编译器: 臆造的 tl.* API (doc 15 一手核实 tl.cache 不存在)
CASE_MISSING_API = LN_HEAD + "    rstd = tl.cache(1.0 / tl.sqrt(var + eps))\n" + LN_TAIL

# 产出方 = 数值检查器: 忘了开方, 拿 var 当 rstd 用
CASE_NUMERIC = LN_HEAD + "    rstd = var + eps\n" + LN_TAIL

# 产出方 = 反作弊屏蔽层: 直接用 torch 的高层算子
CASE_CHEAT = '''
import torch

def kernel(X, weight, bias):
    Y = torch.nn.functional.layer_norm(X, (X.shape[-1],), weight, bias, 1e-5)
    return Y
'''

# 产出方 = 本地 L1 预检: 截断的核 (括号不闭合)
CASE_TRUNCATED = LN_HEAD + "    rstd = 1.0 / tl.sqrt(var + eps"   # 少了两个右括号


def ln_ref(rows, cols, rng):
    X = rng.standard_normal((rows, cols)).astype(np.float16)
    w = np.ones(cols, dtype=np.float16)
    b = np.zeros(cols, dtype=np.float16)
    x32 = X.astype(np.float32)
    mean = x32.mean(axis=1, keepdims=True)
    var = x32.var(axis=1, keepdims=True)
    y = (x32 - mean) / np.sqrt(var + 1e-5) * w.astype(np.float32) + b.astype(np.float32)
    return {"X": X, "weight": w, "bias": b}, y.astype(np.float16)


CASES = []
for label, code in (("臆造 tl.* API", CASE_MISSING_API),
                    ("数值超容差", CASE_NUMERIC),
                    ("反作弊屏蔽", CASE_CHEAT),
                    ("输出截断", CASE_TRUNCATED)):
    inputs, expected = ln_ref(512, 768, np.random.default_rng(11))
    CASES.append((label, code, inputs, expected))


def task():
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type="layernorm", complexity=TaskComplexity.MEDIUM),
        target=TargetSpec(hardware="nvidia_gpu", dsl="triton",
                          compute_units=78, warp_size=32),
        constraints=ConstraintSpec(precision=Precision.FP16, min_speedup=0.0,
                                   max_error=5e-2),
    )


def run_pipeline(v, t, code, inputs, expected):
    """跑真机 L1+L2, 返回 (结果列表, 是否 L1 过, 是否 L2 过)"""
    rs = v.verify(code, t, inputs, expected, target_gflops=0.0)
    l1 = next((r for r in rs if r.level.value == "L1"), None)
    l2 = next((r for r in rs if r.level.value == "L2"), None)
    return rs, bool(l1 and l1.passed), bool(l2 and l2.passed)


def build_prompt(pm, t, failed, code, prescriptive):
    fixer = FixerAgent(llm=None)
    return pm.render(
        "fixer_repair",
        _agent_role="coder",
        _skill_name="triton_skill",
        kernel_type=t.source.kernel_type,
        hardware=t.target.hardware,
        dsl=t.target.dsl,
        precision=t.constraints.precision.value,
        error_details=fixer._format_failures(failed, prescriptive=prescriptive),
        history_cases="无",
        code=code,
    )


backend = RemoteExecutor(mode="ssh")
print("=== deploy ===")
if not backend.deploy():
    sys.exit(1)
v = Verifier(backend)

llm = None if COLLECT_ONLY else LLMClient(temperature=0.3, max_tokens=4096)
if llm is not None:
    assert not llm.fallback_model, (
        "本实验不许降级到备用端点: 换模型会让两臂差异不再只是消息形状, "
        "且备用端点是按量计费的第三方 API。请清空 HY_MODEL / LLM_FALLBACK_MODEL。")
pm = PromptManager()
t = task()

print("\n=== 第一步: 采集真机失败原文 (两臂共用同一份) ===")
collected = []
for label, code, inputs, expected in CASES:
    rs, l1_ok, l2_ok = run_pipeline(v, t, code, inputs, expected)
    failed = [r for r in rs if not r.passed]
    if not failed:
        print(f"\n❌ {label}: 这个样本本该失败却全过了, 换样本再测")
        continue
    joined = "\n".join(f"{r.message}\n{r.error_log or ''}" for r in failed)
    fx = prescribe(joined)
    print(f"\n--- {label}: L1={l1_ok} L2={l2_ok} "
          f"处方={'命中 ' + fx[0][:26] if fx else '无 (认不出这个失败形态)'}")
    print(textwrap.indent(joined.strip()[:700], "    "))
    collected.append((label, code, inputs, expected, failed, fx))

if os.environ.get("DUMP_PROMPT"):
    # 把两臂真正发给模型的 prompt 摊出来 (写验证文档要用原文, 不凭印象)
    i = int(os.environ["DUMP_PROMPT"])
    label, code, inputs, expected, failed, fx = collected[i]
    for arm in (False, True):
        print(f"\n@@@@@@@@ {label} / {'处方臂' if arm else '裸文本臂'} @@@@@@@@")
        print(build_prompt(pm, t, failed, code, arm))
    sys.exit(0)

print("\n\n=== 第二步: A/B 修复 (每臂 1 次生成, 然后回真机验证) ===")
rows = []
if not COLLECT_ONLY:
    for label, code, inputs, expected, failed, fx in collected:
        for arm in (False, True):
            ok1 = ok2 = 0
            for _ in range(REPS):
                prompt = build_prompt(pm, t, failed, code, arm)
                raw = llm.chat([{"role": "user", "content": prompt}],
                               temperature=0.1, max_tokens=4096, caller_tag="AB.C1")
                assert not llm._fallback_triggered, "不许降级到备用端点"
                fixed = FixerAgent._extract_code(raw)
                if not fixed:
                    continue
                _, l1_ok, l2_ok = run_pipeline(v, t, fixed, inputs, expected)
                ok1 += l1_ok
                ok2 += l2_ok
            rows.append((label, "处方" if arm else "裸文本", ok1, ok2))
            print(f"  {label:12s} {'处方' if arm else '裸文本'}: "
                  f"L1 {ok1}/{REPS}  L2 {ok2}/{REPS}")

    print(f"\n=== 汇总 (每格 {REPS} 次) ===")
    for arm_name in ("裸文本", "处方"):
        sel = [r for r in rows if r[1] == arm_name]
        n = REPS * len(sel)
        print(f"  {arm_name:4s}: L1 {sum(r[2] for r in sel)}/{n}   "
              f"L2 {sum(r[3] for r in sel)}/{n}")
    print(f"\n  用量: {llm.cost_tracker.summary()}")
