"""Versioned trusted defaults for NEW task entry points, not legacy migration.

The registry describes exact built-in semantics, never candidate code or guessed
operator families. Unsupported declarations remain unknown before any execution.
"""
from dataclasses import replace

import numpy as np

from ..decide.bench_cases import BenchCaseError
from ..decide.correctness import as_suite, CorrectnessCase, CorrectnessSuite
from ..decide.bench_cases import BenchmarkCase, TensorSpec
from ..decide.reference_contract import (PROFILE_ATTENTION, PROFILE_FUSED_LN_RESIDUAL,
                                         PROFILE_V2, as_reference, registered_reference,
                                         validate_gate, REGISTERED_ATOL)
#: attention 的 correctness profile 与它的参考 profile 是**同一个字符串** ——
#: 与 layernorm（PROFILE_V2 同时充当两者）保持同一种写法，免得两份名字各自漂。
ATTENTION_PROFILE_VERSION = PROFILE_ATTENTION
from ..decide.correctness_boundaries import (BoundaryAxis, BoundaryPolicy, TensorAxes,
                                    as_policy, lint_coverage)

PROFILE_VERSION = "builtin-l2-v1"
OPERATORS = ("relu", "matmul", "softmax", "layernorm")
#: 已注册的融合族（kernel_type 白名单）。规则解析把它们视为**已知语义**，
#: 不再打 `unresolved_fusion` 标记 —— 那标记是给"没注册的融合"用的。
REGISTERED_FUSED_KERNEL_TYPES = ("fused_layernorm_residual",)
SCOPE = "内建二维算子有限 L2 子域；layernorm 为无 affine、末轴归一化、eps=1e-5；不是全域证明"

#: attention 的显式 case 清单。它**不走** axis 交叉生成器：S 必须是真实规模
#: (1000/4096 不是边界对齐值), 而 causal 是 case 级标量而非形状轴。
#: 最后一条 `large` 是**必需**的 —— 2026-11-07 标定证明: 随机输入下 exp 不溢出,
#: "漏掉 max 平移" 的实现能骗过容差, 只有极值输入能抓到它。
#: 融合 residual+LayerNorm 的显式 case 清单（两个输入 X/R，pattern 同时作用于两者）。
#: 形状/值模式沿用 v2 已标定过的矩阵；strided 只放在 rows>1 且 cols>1 的 case 上
#: （单列/单行的跨步取样没有可比的元素步长，materialize 会拒绝）。
FUSED_LN_RESIDUAL_CASES = (
    ("r1c1", (1, 1), "random", "contiguous"),
    ("r1c33", (1, 33), "random", "contiguous"),
    ("r8c1", (8, 1), "random", "contiguous"),
    ("r8c31", (8, 31), "random", "contiguous"),
    ("r8c32", (8, 32), "random", "contiguous"),
    ("r8c33", (8, 33), "random", "contiguous"),
    ("r9c1", (9, 1), "random", "contiguous"),
    ("r9c33", (9, 33), "random", "contiguous"),
    ("r8c33-zeros", (8, 33), "zeros", "contiguous"),
    ("r8c33-constant", (8, 33), "constant", "contiguous"),
    ("r8c33-large", (8, 33), "large", "contiguous"),
    ("r8c33-small", (8, 33), "small", "contiguous"),
    *((f"single-column-{name}-{layout}", (8, 1), pattern, layout)
      for name, pattern in (("zero", "zeros"), ("one", "constant"), ("large", "large"),
                            ("small", "small"), ("nonbinary", "constant_nonbinary"))
      for layout in ("contiguous", "strided")),
    *((f"{pattern}-{layout}", (8, 33), pattern, layout)
      for pattern in ("constant_nonbinary", "near_constant", "large_varying",
                      "small_varying", "signed_large")
      for layout in ("contiguous", "strided")),
)

ATTENTION_CASES = (
    ("s1024-nocausal", (1, 16, 1024, 128), False, "random", "contiguous"),
    ("s1024-causal", (1, 16, 1024, 128), True, "random", "contiguous"),
    ("s2048-causal", (1, 16, 2048, 128), True, "random", "contiguous"),
    ("s4096-causal", (1, 16, 4096, 128), True, "random", "contiguous"),
    ("s1000-causal", (1, 16, 1000, 128), True, "random", "contiguous"),
    ("s4096-causal-strided", (1, 16, 4096, 128), True, "random", "strided"),
    ("s1024-causal-zeros", (1, 16, 1024, 128), True, "zeros", "contiguous"),
    ("s1024-causal-large", (1, 16, 1024, 128), True, "large", "contiguous"),
)


#: Scope A 的**几何边界**。它不是"建议"，是这套 profile 唯一标定过的范围：
#: 参考实现（SDPA）、8 条 case 与容差（3e-4）都在这个范围里量出来的。越界不是"没测过"，
#: 是"用一份标定去判另一件事"——所以入口直接拒绝，不静默补齐（与 attention + fp32
#: 那条拒绝同一条纪律）。
ATTENTION_SCOPE = dict(batch=1, heads=16, head_dim=128, sequence_lengths=(1024, 2048, 4096))

#: 能力出口（`capabilities()`）用的族清单：说清**支持什么**、**排除什么**。
#: `OPERATORS` 是二维 axis 生成器的词表，不是支持清单 —— 两者混用会让调用方
#: 以为 attention 不支持（2026-11-08 一次会话开场就这么误读了）。
ATTENTION_FAMILY = dict(
    profile=PROFILE_ATTENTION,
    dtype=["fp16"],
    shape="(B,H,S,D)",
    scope=("prefill MHA: B=1, H_q=H_kv=16, D=128, S in {1024,2048,4096}; "
           "causal 由标量按值直传"),
    excluded=["decode (S_q=1) — 结构上不可测",
              "GQA / MQA", "D != 128", "bf16 / fp8",
              "sliding window", "additive bias", "paged KV",
              "非 2 的幂的 S 作为**声明**形状（只在派生 case 里出现）"],
    correctness_cases=8,
    performance="只测主用例；roofline 是粗估模型 attention_coarse_v1，reference_lower_bound=false",
    gate="preflight --op attention --shape B,H,S,D [--causal]",
)


class AttentionScopeError(ValueError):
    """attention 声明越出 Scope A：明确拒绝，不猜语义、不静默降级。"""


def attention_scope_problem(shapes):
    """声明的形状是否落在 Scope A 里；返回可读的问题串，没问题返回 None。"""
    problems = []
    for shape in shapes or ():
        try:
            values = [int(value) for value in shape]
        except (TypeError, ValueError):
            problems.append(f"attention 的形状必须是整数元组，收到 {shape!r}")
            continue
        if len(values) != 4:
            problems.append(f"attention 的形状必须是 (B,H,S,D)，收到 {tuple(values)}")
            continue
        batch, heads, seq, dim = values
        if batch != ATTENTION_SCOPE["batch"]:
            problems.append(f"batch 只支持 {ATTENTION_SCOPE['batch']}，收到 B={batch}")
        if heads != ATTENTION_SCOPE["heads"]:
            problems.append(f"只支持 MHA H_q=H_kv={ATTENTION_SCOPE['heads']}，收到 H={heads}")
        if dim != ATTENTION_SCOPE["head_dim"]:
            problems.append(f"head 维只支持 {ATTENTION_SCOPE['head_dim']}，收到 D={dim}")
        if seq not in ATTENTION_SCOPE["sequence_lengths"]:
            problems.append(f"S 只支持 {list(ATTENTION_SCOPE['sequence_lengths'])}，收到 S={seq}"
                            "（其它规模的核没有被这套参考与容差标定过）")
    return "; ".join(dict.fromkeys(problems)) or None


def attention_suite(seed=23):
    """attention 的默认正确性套件：8 条显式 case（不是全域证明）。

    声明的是"这 8 条", 不是"attention 都一样"。causal 走 case 级标量按值直传;
    strided 与 large 各一条, 后者对应标定里那个容差抓不住的失败模式。
    """
    cases = []
    for case_id, (b, h, s_len, d), causal, pattern, layout in ATTENTION_CASES:
        inputs = BenchmarkCase(
            name=f"attention-b{b}h{h}s{s_len}d{d}",
            tensors=(TensorSpec("Q", (b, h, s_len, d), "fp16"),
                     TensorSpec("K", (b, h, s_len, d), "fp16"),
                     TensorSpec("V", (b, h, s_len, d), "fp16")),
            seed=seed, scalars={"causal": causal})
        cases.append(CorrectnessCase(case_id, inputs, pattern, layout))
    return as_suite({"cases": [case.content_dict() for case in cases]})


def fused_ln_residual_suite(seed=17):
    """融合 residual+LayerNorm 的默认套件：32 条显式 case（不是全域证明）。

    每条 case 有两个输入 X/R（同形、同 dtype）；pattern 同时作用于两者，与 attention
    的 Q/K/V 同一规则。输出元素数按 X 计。
    """
    cases = []
    for case_id, (rows, cols), pattern, layout in FUSED_LN_RESIDUAL_CASES:
        inputs = BenchmarkCase(
            name=f"fused-ln-residual-r{rows}c{cols}",
            tensors=(TensorSpec("X", (rows, cols), "fp16"),
                     TensorSpec("R", (rows, cols), "fp16")),
            seed=seed)
        cases.append(CorrectnessCase(
            f"fused-ln-residual-{rows}x{cols}-{pattern}-{layout}", inputs, pattern, layout))
    return as_suite({"cases": [case.content_dict() for case in cases]})


def default_policy(kernel_type, dtype, version=PROFILE_VERSION):
    if version not in (PROFILE_VERSION, PROFILE_V2):
        raise BenchCaseError(f"未知 correctness profile 版本: {version!r}")
    if kernel_type not in OPERATORS:
        raise BenchCaseError(f"没有可信默认正确性配置: {kernel_type!r}（不猜算子语义）")
    if dtype not in ("fp16", "fp32"):
        raise BenchCaseError(f"默认正确性配置不支持 dtype={dtype!r}，不伪造转换")
    if version == PROFILE_V2 and (kernel_type, dtype) != ("layernorm", "fp16"):
        raise BenchCaseError("v2 仅支持 LayerNorm fp16 CUDA 参考定义")
    if kernel_type == "layernorm" and dtype == "fp16" and version == PROFILE_VERSION:
        raise BenchCaseError("fp16 LayerNorm 标定发现 CPU/CUDA 参考存在门限级差异；"
                             "参考身份/精度语义尚未冻结，默认 profile 暂不支持"
                             "（见 docs/design/fp16-layernorm-calibration.md）")
    if kernel_type == "matmul":
        # 'large'=100 at K=4 => 40,000: finite in fp16. This is an explicit
        # sampling subdomain, NOT a promise to cover all reduction lengths.
        axes = (BoundaryAxis("m", 1, 33, 32, 32), BoundaryAxis("n", 1, 33, 32, 32),
                BoundaryAxis("k", 1, 5, 4, 4))
        tensors = (TensorAxes("A", ("m", "k")), TensorAxes("B", ("k", "n")))
    else:
        axes = (BoundaryAxis("rows", 1, 9, 8, 8), BoundaryAxis("cols", 1, 33, 32, 32))
        tensors = (TensorAxes("X", ("rows", "cols")),)
    policy = BoundaryPolicy(axes, tensors, dtypes=(dtype,), seed=17)
    if version == PROFILE_V2:
        policy = replace(policy, patterns=policy.patterns + (
            "constant_nonbinary", "near_constant", "large_varying", "small_varying", "signed_large"))
    return as_policy(policy)


def v2_suite():
    """All 32 calibrated fixtures, now contract-owned, not a passing subset."""
    policy = default_policy("layernorm", "fp16", PROFILE_V2)
    # Keep the original 12 calibrated boundary witnesses, then add the exact
    # diagnostic matrix; do not regenerate extra patterns at different shapes.
    base_policy = replace(policy, patterns=("random", "zeros", "constant", "large", "small"))
    base = as_suite({"coverage_policy": base_policy.content_dict()})
    extra = []
    for name, pattern in (("zero", "zeros"), ("one", "constant"), ("large", "large"),
                          ("small", "small"), ("nonbinary", "constant_nonbinary")):
        for layout in ("contiguous", "strided"):
            extra.append(CorrectnessCase(f"v2-single-column-{name}-{layout}", BenchmarkCase(
                tensors=(TensorSpec("X", (8, 1), "fp16"),), seed=17), pattern, layout))
    for pattern in ("constant_nonbinary", "near_constant", "large_varying", "small_varying", "signed_large"):
        for layout in ("contiguous", "strided"):
            extra.append(CorrectnessCase(f"v2-{pattern}-{layout}", BenchmarkCase(
                tensors=(TensorSpec("X", (8, 33), "fp16"),), seed=17), pattern, layout))
    return as_suite(CorrectnessSuite(base.cases + tuple(extra), coverage_policy=policy))


def expected_output_elements(spec, inputs):
    """这个受信任 profile 的**输出元素数**：由 profile 拥有，不靠输入键名猜。

    远端回报的 `compared_elements` 是"参考比对了多少个元素"，控制端要核它就必须
    知道**输出**有多大。旧写法取 `inputs["X"].size`，那只对"输出与唯一输入同形"的族
    成立（layernorm 恰好如此）；attention 的输入是 Q/K/V、输出与 Q 同形，没有 `X`。
    未知 profile 返回 `None` —— 由调用方判 unknown，不兜默认。
    """
    profile = getattr(spec, "profile", None)
    if profile in (PROFILE_V2, PROFILE_FUSED_LN_RESIDUAL):
        only = inputs.get("X")
    elif profile == PROFILE_ATTENTION:
        only = inputs.get("Q")
    else:
        return None
    if only is None or not hasattr(only, "size"):
        return None
    return int(np.size(only))


def declaration_check(kernel_type, dtype, version, suite, reference_contract=None):
    result = dict(status="not_declared", profile=version, scope=SCOPE)
    if version is None:
        return result
    try:
        if version == PROFILE_V2:
            spec = as_reference(reference_contract)
            if as_suite(suite).digest() != v2_suite().digest():
                raise BenchCaseError("v2 必须保留完整 32 组标定套件，不能删改低方差/单列反例")
            result.update(reference_hash=spec.digest(), scope="H20 CUDA 指定 native 构建；32 组有限 L2 标定域，不证明全域/内部算术")
        elif version == ATTENTION_PROFILE_VERSION:
            spec = as_reference(reference_contract)
            if spec.profile != ATTENTION_PROFILE_VERSION:
                raise BenchCaseError("attention profile 与参考身份不一致")
            if as_suite(suite).digest() != attention_suite().digest():
                raise BenchCaseError("attention 必须保留全部 8 条显式 case（含 strided 与 large）")
            result.update(status="passed", reference_hash=spec.digest(),
                          scope="H20 CUDA SDPA 参考；8 条有限 L2 case，不证明任意 S/D/mask",
                          policy_hash=attention_suite().digest())
            # 早返回：attention 不走下面那套"axis policy 必须与默认 policy 相等"的检查
            # （那套只对二维生成器成立）。
            return result
        elif version == PROFILE_FUSED_LN_RESIDUAL:
            spec = as_reference(reference_contract)
            if spec.profile != PROFILE_FUSED_LN_RESIDUAL:
                raise BenchCaseError("fused residual LayerNorm profile 与参考身份不一致")
            if as_suite(suite).digest() != fused_ln_residual_suite().digest():
                raise BenchCaseError(
                    "fused residual LayerNorm 必须保留全部 32 条显式 case（含 strided/近常量）")
            result.update(status="passed", reference_hash=spec.digest(),
                          scope="H20 CUDA eager x+r(native fp16 add)+native LayerNorm；"
                                "32 条有限 L2 case，不证明任意形状/值域",
                          policy_hash=fused_ln_residual_suite().digest())
            return result
        elif reference_contract is not None:
            raise BenchCaseError("参考身份声明只能与其可信 v2 profile 一同使用")
        policy = default_policy(kernel_type, dtype, version)
        if suite is None:
            raise BenchCaseError("声明了默认 correctness profile，但套件被移除/未生成")
        suite = as_suite(suite)
        if suite.coverage_policy is None or as_policy(suite.coverage_policy).digest() != policy.digest():
            raise BenchCaseError("默认 profile 的覆盖 policy 已漂移/缺失，不能删 dtype/值模式/布局/边界")
        if not suite.deterministic:
            raise BenchCaseError("默认 profile 必须保留确定性检查")
        coverage = lint_coverage(suite)
        if coverage["status"] != "passed":
            raise BenchCaseError("默认 profile 覆盖不足: " + "; ".join(coverage["missing"] + coverage["errors"]))
        result.update(status="passed", policy_hash=policy.digest())
    except (ValueError, TypeError) as exc:
        result.update(status="unknown", reason=str(exc))
    return result


def profile_check(task, inputs=None):
    c, src = task.constraints, task.source
    version = c.correctness_profile
    if version is None:
        if getattr(c, "reference_contract", None) is not None:
            return dict(status="unknown", profile=None, scope=SCOPE, reason="参考身份没有可信 profile，不忽略声明")
        return dict(status="not_declared", profile=None, scope=SCOPE)
    # Built-ins cannot stand in for custom reference/fusion semantics.
    if (src.reference_source or src.fused_ops or src.code_path or src.framework != "pytorch"
            or src.semantic_parameters):
        return dict(status="unknown", profile=version, scope=SCOPE,
                    reason="内建 profile 不适用于外挂参考/融合/源文件/非 PyTorch/自定义参数语义；需要独立可信套件")
    result = declaration_check(src.kernel_type, getattr(c.precision, "value", None),
                               version, c.correctness_suite, getattr(c, "reference_contract", None))
    if result["status"] == "passed" and version == PROFILE_V2:
        try:
            validate_gate(c.reference_contract, c.precision.value, c.max_error)
        except (ValueError, TypeError) as exc:
            result.update(status="unknown", reason=str(exc))
    if result["status"] == "passed" and version == ATTENTION_PROFILE_VERSION:
        # attention 没有 axis policy，主输入名直接按套件语义核：Q/K/V + 按值直传的 causal。
        # （`default_policy` 那套"主输入名必须与 policy.tensors 一致"只对二维生成器成立。）
        try:
            validate_gate(c.reference_contract, c.precision.value, c.max_error,
                          kernel_type="attention")
        except (ValueError, TypeError) as exc:
            result.update(status="unknown", reason=str(exc))
        if result["status"] == "passed" and inputs is not None:
            sets = inputs if isinstance(inputs, list) else [inputs]
            if not sets or any(not isinstance(v, dict) or set(v) != {"Q", "K", "V", "causal"}
                               for v in sets):
                result.update(status="unknown",
                              reason="attention 主输入必须是 Q/K/V + causal 标量")
        return result
    if result["status"] == "passed" and version == PROFILE_FUSED_LN_RESIDUAL:
        try:
            validate_gate(c.reference_contract, c.precision.value, c.max_error,
                          kernel_type="fused_layernorm_residual")
        except (ValueError, TypeError) as exc:
            result.update(status="unknown", reason=str(exc))
        if result["status"] == "passed" and inputs is not None:
            sets = inputs if isinstance(inputs, list) else [inputs]
            for values in sets:
                if (not isinstance(values, dict) or set(values) != {"X", "R"}
                        or not hasattr(values["X"], "shape")
                        or values["X"].shape != values["R"].shape
                        or str(getattr(values["X"], "dtype", "")).split(".")[-1] != "float16"
                        or str(getattr(values["R"], "dtype", "")).split(".")[-1] != "float16"):
                    result.update(status="unknown",
                                  reason="fused residual LayerNorm 主输入必须是同形 fp16 X/R")
                    break
        return result
    if result["status"] != "passed" or inputs is None:
        return result
    try:
        policy = default_policy(src.kernel_type, c.precision.value, version)
        sets = inputs if isinstance(inputs, list) else [inputs]
        if not sets:
            raise BenchCaseError("profile 主输入集不能为空")
        for values in sets:
            if not isinstance(values, dict) or set(values) != {t.name for t in policy.tensors}:
                raise BenchCaseError("主输入名/参数不符合内建 profile（不支持 affine/自定义 eps 等变体）")
            dims = {}
            for binding in policy.tensors:
                value = values[binding.name]
                if (not hasattr(value, "shape") or len(value.shape) != len(binding.axes)
                        or any(d <= 0 for d in value.shape)):
                    raise BenchCaseError("内建 profile 要求正整数二维输入")
                if str(getattr(value, "dtype", "")) != {"fp16": "float16", "fp32": "float32"}[c.precision.value]:
                    raise BenchCaseError("主输入 dtype 与 profile 精度不一致")
                for axis, size in zip(binding.axes, value.shape):
                    if axis in dims and dims[axis] != size:
                        raise BenchCaseError(f"主输入共享轴 {axis} 不一致")
                    dims[axis] = size
    except (ValueError, TypeError) as exc:
        result.update(status="unknown", reason=str(exc))
    return result


def with_default_correctness(task):
    """Formal new-task entry: returns a copy, never overwrites an explicit suite.

    Unsupported tasks retain a requirement marker with no fabricated suite. The
    production preflight reports unknown, rather than reverting to legacy L2.
    """
    if (task.constraints.correctness_suite is not None or task.constraints.correctness_profile is not None
            or getattr(task.constraints, "reference_contract", None) is not None):
        return task
    suite = None
    reference = None
    version = PROFILE_VERSION
    if (not task.source.reference_source and not task.source.fused_ops
            and not task.source.code_path and task.source.framework == "pytorch"
            and not task.source.semantic_parameters):
        try:
            if (task.source.kernel_type == "attention"
                    and getattr(task.constraints.precision, "value", None) == "fp16"):
                # 先判几何再挂 profile：挂上一份"标定过"的参考去判没标定过的形状，
                # 是这一层最容易犯的错（2026-11-08 实测 S=3072 / B=8H=8D=64 都能静默冻结）。
                problem = attention_scope_problem(getattr(task.constraints, "shapes", ()) or ())
                if problem is not None:
                    raise AttentionScopeError(
                        f"attention 声明超出 Scope A：{problem}。"
                        "要么换到受支持的形状，要么先为它标定参考与容差（新 profile），"
                        "不能借用现有 profile 的判据")
                # attention 用显式套件 + 自己的受信任参考（不在 OPERATORS 的
                # axis 生成器里 —— 那套只服务二维算子）。
                version = ATTENTION_PROFILE_VERSION
                reference = registered_reference(PROFILE_ATTENTION).content_dict()
                suite = attention_suite().content_dict()
            elif (task.source.kernel_type == "fused_layernorm_residual"
                    and getattr(task.constraints.precision, "value", None) == "fp16"):
                version = PROFILE_FUSED_LN_RESIDUAL
                reference = registered_reference(PROFILE_FUSED_LN_RESIDUAL).content_dict()
                suite = fused_ln_residual_suite().content_dict()
            elif task.source.kernel_type == "layernorm" and getattr(task.constraints.precision, "value", None) == "fp16":
                version = PROFILE_V2
                reference = registered_reference().content_dict()
                suite = v2_suite().content_dict()
            else:
                suite = {"coverage_policy": default_policy(task.source.kernel_type,
                    getattr(task.constraints.precision, "value", None)).content_dict()}
        except BenchCaseError:
            pass  # The requirement marker is consumed as unknown, not success.
    constraints = replace(task.constraints, correctness_profile=version,
                          correctness_suite=suite, reference_contract=reference)
    if reference is not None:
        # 容差与参考身份同源：挂哪个 profile 就用哪个 profile 标定的 atol。全局默认
        # 1e-5 是 layernorm 的标定值，attention 标定的是 3e-4 —— 不在这里绑定，就会有个
        # 严 30× 的阈值静默漏下去，而调用方无权纠正（hitl_tools.FORBIDDEN_KEYS）。
        constraints = replace(constraints, max_error=REGISTERED_ATOL[version])
    return replace(task, constraints=constraints)
