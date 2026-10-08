"""Versioned trusted defaults for NEW task entry points, not legacy migration.

The registry describes exact built-in semantics, never candidate code or guessed
operator families. Unsupported declarations remain unknown before any execution.
"""
from dataclasses import replace

from .bench_cases import BenchCaseError
from .correctness import as_suite
from .correctness_boundaries import (BoundaryAxis, BoundaryPolicy, TensorAxes,
                                    as_policy, lint_coverage)

PROFILE_VERSION = "builtin-l2-v1"
OPERATORS = ("relu", "matmul", "softmax", "layernorm")
SCOPE = "内建二维算子有限 L2 子域；layernorm 为无 affine、末轴归一化、eps=1e-5；不是全域证明"


def default_policy(kernel_type, dtype, version=PROFILE_VERSION):
    if version != PROFILE_VERSION:
        raise BenchCaseError(f"未知 correctness profile 版本: {version!r}")
    if kernel_type not in OPERATORS:
        raise BenchCaseError(f"没有可信默认正确性配置: {kernel_type!r}（不猜算子语义）")
    if dtype not in ("fp16", "fp32"):
        raise BenchCaseError(f"默认正确性配置不支持 dtype={dtype!r}，不伪造转换")
    if kernel_type == "layernorm" and dtype == "fp16":
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
    return as_policy(BoundaryPolicy(axes, tensors, dtypes=(dtype,), seed=17))


def declaration_check(kernel_type, dtype, version, suite):
    result = dict(status="not_declared", profile=version, scope=SCOPE)
    if version is None:
        return result
    try:
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
        return dict(status="not_declared", profile=None, scope=SCOPE)
    # Built-ins cannot stand in for custom reference/fusion semantics.
    if (src.reference_source or src.fused_ops or src.code_path or src.framework != "pytorch"
            or src.semantic_parameters):
        return dict(status="unknown", profile=version, scope=SCOPE,
                    reason="内建 profile 不适用于外挂参考/融合/源文件/非 PyTorch/自定义参数语义；需要独立可信套件")
    result = declaration_check(src.kernel_type, getattr(c.precision, "value", None),
                               version, c.correctness_suite)
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
    if task.constraints.correctness_suite is not None or task.constraints.correctness_profile is not None:
        return task
    suite = None
    if (not task.source.reference_source and not task.source.fused_ops
            and not task.source.code_path and task.source.framework == "pytorch"
            and not task.source.semantic_parameters):
        try:
            suite = {"coverage_policy": default_policy(task.source.kernel_type,
                getattr(task.constraints.precision, "value", None)).content_dict()}
        except BenchCaseError:
            pass  # The requirement marker is consumed as unknown, not success.
    return replace(task, constraints=replace(task.constraints,
                   correctness_profile=PROFILE_VERSION, correctness_suite=suite))
