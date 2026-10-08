"""Trusted, candidate-independent finite L2 boundary generation and coverage lint.

The harness must declare a legal positive-integer axis box and tensor bindings.
No operator/layout guessing, candidate BLOCK inspection, allocation in lint, or
silent budget clipping. This does NOT prove all shapes in the box are correct.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from typing import Mapping

from .bench_cases import BenchmarkCase, BenchCaseError, TensorSpec
from .correctness import CorrectnessCase, CorrectnessSuite, LAYOUTS, PATTERNS
from .input_fidelity import MAX_STORAGE_BYTES

RECIPE_VERSION = "axis-boundary-v1"
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_ITEMSIZE = {"fp16": 2, "fp32": 4}


def _keys(payload, allowed, label):
    if not isinstance(payload, Mapping) or set(payload) - set(allowed):
        raise BenchCaseError(f"{label} 必须是映射且不能包含未知字段")


def _sequence(payload, key):
    value = payload.get(key)
    if not isinstance(value, (list, tuple)):
        raise BenchCaseError(f"boundary {key} 必须是序列")
    return tuple(value)


@dataclass(frozen=True)
class BoundaryAxis:
    name: str
    minimum: int
    maximum: int
    anchor: int
    alignment: int = 8

    def validate(self):
        errors = []
        if not isinstance(self.name, str) or not _NAME.fullmatch(self.name):
            errors.append("boundary axis.name 必须是标识符")
        if any(type(v) is not int for v in (self.minimum, self.maximum, self.anchor, self.alignment)):
            errors.append("boundary axis 的范围/anchor/alignment 必须是 int（不接受 bool/截断）")
        elif not (1 <= self.minimum <= self.anchor <= self.maximum and self.alignment >= 2):
            errors.append("boundary axis 需要 1 <= minimum <= anchor <= maximum，alignment >= 2")
        return errors

    def content_dict(self):
        return dict(name=self.name, minimum=self.minimum, maximum=self.maximum,
                    anchor=self.anchor, alignment=self.alignment)

    @staticmethod
    def from_dict(payload):
        _keys(payload, ("name", "minimum", "maximum", "anchor", "alignment"), "BoundaryAxis")
        return BoundaryAxis(payload.get("name"), payload.get("minimum"), payload.get("maximum"),
                            payload.get("anchor"), payload.get("alignment", 8))


@dataclass(frozen=True)
class TensorAxes:
    name: str
    axes: tuple

    def content_dict(self):
        return {"name": self.name, "axes": list(self.axes)}

    @staticmethod
    def from_dict(payload):
        _keys(payload, ("name", "axes"), "TensorAxes")
        return TensorAxes(payload.get("name"), _sequence(payload, "axes"))


@dataclass(frozen=True)
class BoundaryPolicy:
    axes: tuple
    tensors: tuple
    dtypes: tuple = ("fp32",)
    patterns: tuple = ("random", "zeros", "constant", "large", "small")
    layouts: tuple = ("contiguous", "strided")
    seed: int = 0
    max_cases: int = 128
    max_input_bytes: int = 16 * 1024 * 1024
    recipe_version: str = RECIPE_VERSION

    def validate(self):
        errors = []
        if self.recipe_version != RECIPE_VERSION:
            errors.append("未知 boundary recipe_version，不自动升级生成规则")
        names = []
        if not isinstance(self.axes, tuple) or not 1 <= len(self.axes) <= 8:
            errors.append("boundary axes 必须是 1..8 个轴的 tuple")
        if isinstance(self.axes, (list, tuple)):
            for axis in self.axes:
                if not isinstance(axis, BoundaryAxis):
                    errors.append("boundary axes 必须包含 BoundaryAxis")
                else:
                    errors.extend(axis.validate())
                    names.append(axis.name)
        if any(type(n) is not str for n in names) or len(set(n for n in names if isinstance(n, str))) != len(names):
            errors.append("boundary axis 名称必须唯一")
        tensor_names, used = [], set()
        if not isinstance(self.tensors, tuple) or not 1 <= len(self.tensors) <= 16:
            errors.append("boundary tensors 必须是 1..16 个绑定的 tuple")
        if isinstance(self.tensors, (list, tuple)):
            for binding in self.tensors:
                if not isinstance(binding, TensorAxes):
                    errors.append("boundary tensors 必须包含 TensorAxes")
                    continue
                if not isinstance(binding.name, str) or not _NAME.fullmatch(binding.name):
                    errors.append("boundary tensor.name 必须是标识符")
                tensor_names.append(binding.name)
                if not isinstance(binding.axes, tuple) or not 1 <= len(binding.axes) <= 8:
                    errors.append("boundary tensor.axes 必须是非空 tuple，rank <= 8")
                elif any(type(a) is not str or a not in names for a in binding.axes):
                    errors.append("boundary tensor.axes 引用了未知轴")
                else:
                    used.update(binding.axes)
        if len(set(n for n in tensor_names if isinstance(n, str))) != len(tensor_names):
            errors.append("boundary tensor 名称必须唯一")
        if used != set(n for n in names if isinstance(n, str)):
            errors.append("boundary 每个声明轴必须被输入绑定使用")
        for label, values, allowed, mandatory in (
            ("dtypes", self.dtypes, tuple(_ITEMSIZE), None),
            ("patterns", self.patterns, PATTERNS, "random"),
            ("layouts", self.layouts, LAYOUTS, "contiguous"),
        ):
            if (not isinstance(values, tuple) or not values or any(type(v) is not str or v not in allowed for v in values)
                    or len(set(v for v in values if isinstance(v, str))) != len(values)):
                errors.append(f"boundary {label} 必须是词表内非空、无重复的 tuple")
            if mandatory and (not isinstance(values, (tuple, list)) or mandatory not in values):
                errors.append(f"boundary {label} 必须声明 {mandatory}，不能靠零值/布局样本替代随机连续基准")
        if type(self.seed) is not int or self.seed < 0:
            errors.append("boundary seed 必须是非负 int")
        if type(self.max_cases) is not int or not 1 <= self.max_cases <= 512:
            errors.append("boundary max_cases 必须在 1..512 内")
        if type(self.max_input_bytes) is not int or not 1 <= self.max_input_bytes <= MAX_STORAGE_BYTES:
            errors.append("boundary max_input_bytes 必须在输入保真协议上限内")
        return errors

    def content_dict(self):
        return dict(recipe_version=self.recipe_version,
            axes=[a.content_dict() for a in sorted(self.axes, key=lambda a: a.name)],
            tensors=[t.content_dict() for t in sorted(self.tensors, key=lambda t: t.name)],
            dtypes=sorted(self.dtypes), patterns=sorted(self.patterns), layouts=sorted(self.layouts),
            seed=self.seed, max_cases=self.max_cases, max_input_bytes=self.max_input_bytes)

    def digest(self):
        return hashlib.sha256(json.dumps(self.content_dict(), sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()

    @staticmethod
    def from_dict(payload):
        _keys(payload, ("axes", "tensors", "dtypes", "patterns", "layouts", "seed", "max_cases",
                        "max_input_bytes", "recipe_version"), "BoundaryPolicy")
        return BoundaryPolicy(
            axes=tuple(BoundaryAxis.from_dict(a) for a in _sequence(payload, "axes")),
            tensors=tuple(TensorAxes.from_dict(t) for t in _sequence(payload, "tensors")),
            **{k: _sequence(payload, k) for k in ("dtypes", "patterns", "layouts") if k in payload},
            **{k: payload[k] for k in ("seed", "max_cases", "max_input_bytes", "recipe_version") if k in payload})


def as_policy(value):
    policy = value if isinstance(value, BoundaryPolicy) else BoundaryPolicy.from_dict(value)
    errors = policy.validate()
    if errors:
        raise BenchCaseError("; ".join(errors))
    return replace(policy, axes=tuple(sorted(policy.axes, key=lambda a: a.name)),
        tensors=tuple(sorted(policy.tensors, key=lambda t: t.name)),
        dtypes=tuple(sorted(policy.dtypes)), patterns=tuple(sorted(policy.patterns)),
        layouts=tuple(sorted(policy.layouts)))


def _aligned(axis):
    first = ((axis.minimum + axis.alignment - 1) // axis.alignment) * axis.alignment
    last = (axis.maximum // axis.alignment) * axis.alignment
    if first > last:
        return None
    floor = (axis.anchor // axis.alignment) * axis.alignment
    return min({max(first, min(last, floor)), max(first, min(last, floor + axis.alignment))},
               key=lambda n: (abs(n - axis.anchor), n))


def _shape_rules(policy):
    """Closed v1 recipe: isolate each axis; never propose an illegal neighbor."""
    axes = sorted(policy.axes, key=lambda a: a.name)
    anchor = tuple(a.anchor for a in axes)
    required = {"shape:anchor": anchor}
    inapplicable = {}
    aligned = [_aligned(a) for a in axes]
    if all(v is not None for v in aligned):
        required["shape:aligned"] = tuple(aligned)
    else:
        inapplicable["shape:aligned"] = "声明的合法域不存在全轴对齐 shape"
    for i, axis in enumerate(axes):
        def point(value):
            return anchor[:i] + (value,) + anchor[i + 1:]
        required[f"shape:min:{axis.name}"] = point(axis.minimum)
        required[f"shape:max:{axis.name}"] = point(axis.maximum)
        if axis.minimum == 1:
            required[f"shape:one:{axis.name}"] = point(1)
        else:
            inapplicable[f"shape:one:{axis.name}"] = "合法域不含 1，不捏造 tiny/batch=1 输入"
        pivot = aligned[i]
        for label, delta in (("below", -1), ("above", 1)):
            key = f"shape:{label}:{axis.name}"
            if pivot is not None and axis.minimum <= pivot + delta <= axis.maximum:
                required[key] = point(pivot + delta)
            else:
                inapplicable[key] = "域内对齐点的相邻输入不在合法范围，不越界生成"
        tail = next((v for v in (axis.anchor, axis.minimum, axis.minimum + 1)
                     if v <= axis.maximum and v % axis.alignment), None)
        if tail is None:
            inapplicable[f"shape:tail:{axis.name}"] = "合法域不含非对齐输入"
        else:
            required[f"shape:tail:{axis.name}"] = point(tail)
    return required, inapplicable


def _inputs(policy, point, dtype):
    dims = dict(zip(sorted(a.name for a in policy.axes), point))
    return BenchmarkCase(tensors=tuple(TensorSpec(t.name, tuple(dims[a] for a in t.axes), dtype)
        for t in sorted(policy.tensors, key=lambda t: t.name)), seed=policy.seed)


def _budget(case, policy):
    # Check backing allocation too: the strided materializer doubles its last axis.
    factor = 2 if case.layout == "strided" else 1
    size = sum(math.prod(t.shape) * _ITEMSIZE[t.dtype] * factor for t in case.inputs.tensors)
    if size > policy.max_input_bytes:
        raise BenchCaseError(f"boundary 输入预算不足 ({size} > {policy.max_input_bytes} bytes)，不裁掉边界")
    if case.layout != "contiguous" and any(
            t.shape[-1] <= 1 and not (case.layout == "strided" and len(t.shape) == 2 and t.shape[0] > 1)
            for t in case.inputs.tensors):
        raise BenchCaseError("声明布局无法实际扰动输入（单列 strided 需要多行以建立真实行间隙）")


def generate_correctness_suite(policy):
    """Deterministic recipe, no ndarray allocation or candidate-dependent choices."""
    policy = as_policy(policy)
    rules, _ = _shape_rules(policy)
    shapes = sorted(set(rules.values()))
    anchor = rules["shape:anchor"]
    cases = {}
    def add(point, dtype, pattern="random", layout="contiguous"):
        dims = "-".join(f"{a}{v}" for a, v in zip(sorted(a.name for a in policy.axes), point))
        case = CorrectnessCase(f"auto-{dims}-{dtype}-{pattern}-{layout}", _inputs(policy, point, dtype), pattern, layout)
        _budget(case, policy)
        key = json.dumps(dict(inputs=case.inputs.content_dict(), pattern=pattern, layout=layout), sort_keys=True)
        cases[key] = case
        if len(cases) > policy.max_cases:
            raise BenchCaseError("boundary max_cases 预算不足，不截断生成套件/不裁掉 dtype 或边界")
    for dtype in sorted(policy.dtypes):
        for point in shapes:
            add(point, dtype)
        for pattern in sorted(set(policy.patterns) - {"random"}):
            add(anchor, dtype, pattern=pattern)
        for layout in sorted(set(policy.layouts) - {"contiguous"}):
            eligible = [p for p in shapes if all(t.shape[-1] > 1 for t in _inputs(policy, p, dtype).tensors)]
            if not eligible:
                raise BenchCaseError("合法域没有可保真构造所声明布局的 shape，不删掉布局要求")
            add(anchor if anchor in eligible else eligible[0], dtype, layout=layout)
    suite = CorrectnessSuite(tuple(sorted(cases.values(), key=lambda c: c.case_id)), coverage_policy=policy)
    lint = lint_coverage(suite)
    if lint["status"] != "passed":
        raise BenchCaseError("生成套件未满足 coverage lint: " + "; ".join(lint["errors"] + lint["missing"]))
    return suite


def _dimensions(case, policy):
    actual = {t.name: t for t in case.inputs.tensors}
    if set(actual) != {t.name for t in policy.tensors}:
        raise BenchCaseError("输入名称与 policy 的张量绑定不同")
    dims = {}
    for binding in policy.tensors:
        tensor = actual[binding.name]
        if len(tensor.shape) != len(binding.axes):
            raise BenchCaseError("输入 rank 与轴绑定不同")
        for axis, value in zip(binding.axes, tensor.shape):
            if axis in dims and dims[axis] != value:
                raise BenchCaseError(f"共享轴 {axis} 在不同输入/维度上不一致")
            dims[axis] = value
    if any(not a.minimum <= dims[a.name] <= a.maximum for a in policy.axes):
        raise BenchCaseError("测试 shape 越出 policy 的合法范围")
    dtypes = {t.dtype for t in case.inputs.tensors}
    if len(dtypes) != 1 or not dtypes <= set(policy.dtypes):
        raise BenchCaseError("case dtype 不在 policy 中或混合精度未支持")
    if case.pattern not in policy.patterns or case.layout not in policy.layouts:
        raise BenchCaseError("case 值模式/布局不在 policy 中")
    _budget(case, policy)
    return tuple(dims[a] for a in sorted(dims)), next(iter(dtypes))


def lint_coverage(suite):
    """Metadata coverage only. Name/count is never a witness; runtime stays in L2."""
    result = dict(status="not_declared", missing=[], errors=[], witnesses={}, inapplicable={},
                  scope="仅声明域内 v1 有限边界点；不证明组合全覆盖、GPU执行或数值正确")
    if suite.coverage_policy is None:
        return result
    try:
        policy = as_policy(suite.coverage_policy)
    except (ValueError, TypeError) as exc:
        result.update(status="invalid", errors=[str(exc)])
        return result
    rules, inapplicable = _shape_rules(policy)
    expected = {f"{key}/dtype:{dt}": (point, dt, "random", "contiguous")
                for key, point in rules.items() for dt in policy.dtypes}
    expected.update({f"pattern:{p}/dtype:{dt}": (None, dt, p, None)
                     for p in policy.patterns for dt in policy.dtypes})
    expected.update({f"layout:{layout}/dtype:{dt}": (None, dt, "random", layout)
                     for layout in policy.layouts for dt in policy.dtypes})
    witnesses = {key: [] for key in sorted(expected)}
    errors = list(suite.validate())
    if len(suite.cases) > policy.max_cases:
        errors.append("套件超过 boundary max_cases 预算")
    for case in suite.cases:
        if not isinstance(case, CorrectnessCase) or case.validate():
            continue  # Structural errors above; malformed cases cannot witness a rule.
        try:
            point, dtype = _dimensions(case, policy)
        except (BenchCaseError, KeyError, TypeError) as exc:
            errors.append(f"case {getattr(case, 'case_id', '?')}: {exc}")
            continue
        for key, (shape, dt, pattern, layout) in expected.items():
            if dtype == dt and case.pattern == pattern and (shape is None or shape == point) and (layout is None or layout == case.layout):
                witnesses[key].append(case.case_id)
    missing = [key for key, cases in witnesses.items() if not cases]
    result.update(status="invalid" if errors else "missing" if missing else "passed",
        policy_hash=policy.digest(), recipe_version=policy.recipe_version,
        case_count=len(suite.cases), required_rule_count=len(expected),
        witnessed_rule_count=len(expected) - len(missing), missing=missing, errors=errors,
        witnesses={k: sorted(v) for k, v in witnesses.items()}, inapplicable=inapplicable)
    return result


def has_boundary_policy(raw):
    return (isinstance(raw, CorrectnessSuite) and raw.coverage_policy is not None
            or isinstance(raw, Mapping) and "coverage_policy" in raw)
