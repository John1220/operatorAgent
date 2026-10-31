"""Contract-owned L2 inputs; correctness cases never enter the L3 workload.

Case identity includes shape, dtype, values, layout and seed (not just the
performance ledger's shape key). Unsupported checks remain unknown, not passed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional

from ..decide.bench_cases import BenchmarkCase, BenchCaseError

PATTERNS = ("random", "zeros", "constant", "large", "small", "nan", "inf",
            "constant_nonbinary", "near_constant", "large_varying", "small_varying", "signed_large")
LAYOUTS = ("contiguous", "strided", "broadcast")
EXTRA_CHECKS = ("backward", "sanitizer", "racecheck", "aliasing", "inplace",
                "accum_fp16", "accum_fp64")


@dataclass(frozen=True)
class CorrectnessCase:
    case_id: str
    inputs: BenchmarkCase
    pattern: str = "random"
    layout: str = "contiguous"

    def validate(self) -> list:
        errors = []
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            errors.append("correctness case_id 必须是非空字符串")
        if not isinstance(self.inputs, BenchmarkCase):
            errors.append("correctness inputs 必须是 BenchmarkCase")
        else:
            errors.extend(self.inputs.validate())
        if self.pattern not in PATTERNS:
            errors.append(f"未知 correctness pattern: {self.pattern!r}")
        if self.layout not in LAYOUTS:
            errors.append(f"未知 correctness layout: {self.layout!r}")
        return errors

    def content_dict(self) -> dict:
        return {"case_id": self.case_id, "inputs": self.inputs.content_dict(),
                "pattern": self.pattern, "layout": self.layout}

    def materialize(self) -> dict:
        import numpy as np

        errors = self.validate()
        if errors:
            raise BenchCaseError("; ".join(errors))
        arrays = self.inputs.materialize()
        for name, value in arrays.items():
            if self.pattern != "random":
                if self.pattern in ("nan", "inf") and value.dtype.kind != "f":
                    raise BenchCaseError("NaN/Inf 检查需要浮点输入")
                sequences = {"near_constant": (1, 1+2**-10, 1-2**-11),
                             "large_varying": (100, 100.0625, 99.9375),
                             "small_varying": (0, 1e-4, -1e-4), "signed_large": (100, -100, 0)}
                if self.pattern in sequences:
                    if not value.ndim:
                        raise BenchCaseError("行模式检查需要至少一维输入")
                    row = np.resize(np.array(sequences[self.pattern], dtype=value.dtype), value.shape[-1])
                    value = np.broadcast_to(row, value.shape).copy()
                else:
                    fill = {"zeros": 0, "constant": 1, "large": 100, "constant_nonbinary": .1,
                            "small": 1e-4, "nan": np.nan, "inf": np.inf}[self.pattern]
                    value = np.full(value.shape, fill, dtype=value.dtype)
            if self.layout != "contiguous":
                singleton_rows = self.layout == "strided" and value.ndim == 2 and value.shape[0] > 1
                if not value.ndim or not value.size or (value.shape[-1] <= 1 and not singleton_rows):
                    raise BenchCaseError("layout 扰动需要非空且最后一维 > 1 的张量")
                if self.layout == "strided":
                    backing = np.zeros(value.shape[:-1] + (value.shape[-1] * 2,),
                                       dtype=value.dtype)
                    backing[..., ::2] = value
                    value = backing[..., ::2]
                else:
                    value = np.broadcast_to(value[..., :1], value.shape)
            arrays[name] = value
        return arrays

    @staticmethod
    def from_dict(payload) -> "CorrectnessCase":
        if not isinstance(payload, Mapping):
            raise BenchCaseError("correctness case 必须是映射")
        if set(payload) - {"case_id", "inputs", "pattern", "layout"}:
            raise BenchCaseError("correctness case 有未知字段")
        raw = payload.get("inputs")
        # BenchmarkCase's legacy loader coerces numbers; new suite declarations
        # must not silently truncate fractional dimensions or accept bool seeds.
        if not isinstance(raw, Mapping):
            raise BenchCaseError("correctness inputs 必须是映射")
        seed = raw.get("seed", 0)
        if type(seed) is not int or seed < 0:
            raise BenchCaseError("correctness seed 必须是非负整数")
        tensors = raw.get("tensors")
        if not isinstance(tensors, (list, tuple)):
            raise BenchCaseError("correctness tensors 必须是序列")
        for t in tensors:
            if not isinstance(t, Mapping) or not isinstance(t.get("shape"), (list, tuple)):
                raise BenchCaseError("correctness tensor.shape 必须是序列")
            if any(type(d) is not int or d < 0 for d in t["shape"]):
                raise BenchCaseError("correctness shape 必须是非负整数")
        case = CorrectnessCase(payload.get("case_id", ""), BenchmarkCase.from_dict(raw),
                               payload.get("pattern", "random"),
                               payload.get("layout", "contiguous"))
        errors = case.validate()
        if errors:
            raise BenchCaseError("; ".join(errors))
        return case


@dataclass(frozen=True)
class CorrectnessSuite:
    cases: tuple
    deterministic: bool = True
    required_checks: tuple = ()
    # Optional trusted finite-domain recipe. Absent => legacy suite/hash unchanged.
    coverage_policy: Optional[Any] = None

    def validate(self) -> list:
        errors = []
        if not isinstance(self.cases, tuple) or not self.cases:
            errors.append("CorrectnessSuite.cases 必须是非空 tuple")
        ids = []
        for case in self.cases:
            if not isinstance(case, CorrectnessCase):
                errors.append("CorrectnessSuite.cases 必须包含 CorrectnessCase")
            else:
                errors.extend(case.validate())
                if isinstance(case.case_id, str):
                    ids.append(case.case_id)
        if len(set(ids)) != len(ids):
            errors.append("correctness case_id 必须唯一（同 shape 的不同值/dtype 不合并）")
        if type(self.deterministic) is not bool:
            errors.append("correctness deterministic 必须是 bool")
        if not isinstance(self.required_checks, tuple):
            errors.append("correctness required_checks 必须是 tuple")
        if any(c not in EXTRA_CHECKS for c in self.required_checks):
            errors.append("correctness required_checks 有未知检查项")
        if self.coverage_policy is not None:
            from ..decide.correctness_boundaries import as_policy
            try:
                as_policy(self.coverage_policy)
            except (ValueError, TypeError) as exc:
                errors.append(f"correctness coverage_policy: {exc}")
        return errors

    def content_dict(self) -> dict:
        result = {"cases": [c.content_dict() for c in sorted(self.cases, key=lambda c: c.case_id)],
                "deterministic": self.deterministic,
                "required_checks": sorted(set(self.required_checks))}
        if self.coverage_policy is not None:
            from ..decide.correctness_boundaries import as_policy
            result["coverage_policy"] = as_policy(self.coverage_policy).content_dict()
        return result

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.content_dict(), sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()

    @staticmethod
    def from_dict(payload) -> "CorrectnessSuite":
        if not isinstance(payload, Mapping):
            raise BenchCaseError("CorrectnessSuite 必须是映射")
        if set(payload) - {"cases", "deterministic", "required_checks", "coverage_policy"}:
            raise BenchCaseError("CorrectnessSuite 有未知字段（不忽略拼写错误）")
        cases = payload.get("cases")
        checks = payload.get("required_checks", ())
        policy = None
        generated = False
        if "coverage_policy" in payload:
            from ..decide.correctness_boundaries import as_policy, generate_correctness_suite
            policy = as_policy(payload["coverage_policy"])
            # Omitted cases = explicit automatic generation. Empty/null cases is
            # still an invalid manual declaration, never silently repaired.
            if "cases" not in payload:
                cases = generate_correctness_suite(policy).cases
                generated = True
        if not isinstance(cases, (list, tuple)) or not isinstance(checks, (list, tuple)):
            raise BenchCaseError("correctness cases/required_checks 必须是序列")
        parsed = tuple(cases) if generated else tuple(CorrectnessCase.from_dict(c) for c in cases)
        suite = CorrectnessSuite(parsed,
                                 payload.get("deterministic", True), tuple(checks), policy)
        errors = suite.validate()
        if errors:
            raise BenchCaseError("; ".join(errors))
        return suite


def as_suite(value) -> CorrectnessSuite:
    suite = value if isinstance(value, CorrectnessSuite) else CorrectnessSuite.from_dict(value)
    errors = suite.validate()
    if errors:
        raise BenchCaseError("; ".join(errors))
    if suite.coverage_policy is not None:
        from ..decide.correctness_boundaries import as_policy
        suite = replace(suite, coverage_policy=as_policy(suite.coverage_policy))
    return suite


def verdict_of(result) -> tuple:
    """Strict suite verdict from reported facts; a partial comparison is unknown."""
    m = result.metrics
    # Lazy import: schemas -> correctness -> measurement -> schemas is a cycle.
    from ..decide.measurement import TRANSPORT_KEY, TRANSPORT_UNKNOWN
    if m.get(TRANSPORT_KEY) == TRANSPORT_UNKNOWN:
        return "unknown", "传输结果未知; 未拿到可归因回答，不能判候选数值失败"
    if "input_fidelity_verified" in m and m["input_fidelity_verified"] is not True:
        return "unknown", m.get("input_fidelity_reason", "输入保真无法确立")
    if m.get("reference_error"):
        return "unknown", m["reference_error"]
    if m.get("correctness_execution_error"):
        return "unknown", m["correctness_execution_error"]
    if m.get("reference_nonfinite", 0):
        return "unknown", "参考含 NaN/Inf，不能确立全部输出正确"
    if "max_error" not in m and result.passed:
        return "unknown", "未提供数值比对证据（无参考或仅可执行）"
    if not result.passed:
        return "failed", result.message
    if m.get("coverage") != 1.0 or m.get("compared_elements", 0) <= 0:
        return "unknown", "输出比对覆盖不足或未报告覆盖证据"
    if m.get("and_form_passed") is False:
        return "failed", "采样纯绝对容差通过，但统一 AND 容差不通过"
    return "passed", result.message


def credential(results):
    """Preserve suite unknown through candidate promotion and corpus decisions."""
    values = []
    for result in results:
        state = (getattr(result, "metrics", {}) or {}).get("correctness", {}).get("verdict")
        values.append(None if state == "unknown" else bool(result.passed))
    if not values:
        return None
    if False in values:
        return False
    return None if None in values else True
