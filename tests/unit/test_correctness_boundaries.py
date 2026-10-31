"""Boundary recipe + independent metadata lint + production hard gates."""
import dataclasses
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from operator_agent import (BoundaryAxis, BoundaryPolicy, TensorAxes, CorrectnessSuite,
    ConstraintSpec, MockGPU, SourceSpec, TargetSpec, TaskSchema, Verifier,
    generate_correctness_suite, lint_coverage)
from operator_agent.decide.bench_cases import BenchCaseError, BenchmarkCase
from operator_agent.core.contract import ContractError, freeze_from_task, load_dict
from operator_agent.decide.correctness import as_suite, credential
from operator_agent.decide.failure_route import ROUTE_HOLD, classify
from operator_agent.core.report import build_report, collect_unable
from operator_agent.core.schemas import KernelArtifact, VerificationLevel, VerificationResult

GOOD = "import numpy as np\ndef kernel(A, B):\n    return A @ B\n"


def policy(**overrides):
    values = dict(axes=tuple(BoundaryAxis(a, 1, 9, 8) for a in ("m", "n", "k")),
        tensors=(TensorAxes("A", ("m", "k")), TensorAxes("B", ("k", "n"))),
        dtypes=("fp32", "fp16"), seed=17)
    values.update(overrides)
    return BoundaryPolicy(**values)


def task(suite):
    return TaskSchema(source=SourceSpec(kernel_type="matmul"), target=TargetSpec(dsl="python"),
                      constraints=ConstraintSpec(min_speedup=0.5, correctness_suite=suite))


def finite_policy():
    # Fixed 'large'=100 stresses fp16 too. Keep GEMM anchor K=4 (40,000)
    # below fp16 overflow; the default K=8 (80,000) is explicitly unknown below.
    return policy(axes=tuple(BoundaryAxis(a, 1, 5, 4, 4) for a in ("m", "n", "k")))


def primary():
    return {"A": np.arange(-32, 32, dtype=np.float32).reshape(8, 8),
            "B": np.eye(8, dtype=np.float32)}


def verifier():
    v = Verifier(MockGPU())
    v._verify_performance = Mock(return_value=VerificationResult(VerificationLevel.L3_PERFORMANCE, True))
    return v


def test_generator_covers_each_axis_each_dtype_and_real_value_layout_variations():
    p = policy()
    suite = generate_correctness_suite(p)
    assert lint_coverage(suite)["status"] == "passed"
    # Independent expected probes, not replaying generator's rule producer.
    for dt in ("fp16", "fp32"):
        base = [c for c in suite.cases if c.pattern == "random" and c.layout == "contiguous"
                and c.inputs.tensors[0].dtype == dt]
        shapes = {(c.inputs.tensors[0].shape, c.inputs.tensors[1].shape) for c in base}
        for axis in ("m", "n", "k"):
            for value in (1, 7, 8, 9):
                dims = dict(m=8, n=8, k=8)
                dims[axis] = value
                assert ((dims["m"], dims["k"]), (dims["k"], dims["n"])) in shapes
        assert {c.pattern for c in suite.cases if c.inputs.tensors[0].dtype == dt} == set(p.patterns)
        assert {c.layout for c in suite.cases if c.inputs.tensors[0].dtype == dt} == set(p.layouts)
    assert len({json.dumps(dict(inputs=c.inputs.content_dict(), pattern=c.pattern, layout=c.layout), sort_keys=True)
                for c in suite.cases}) == len(suite.cases)
    for c in suite.cases:
        arrays = c.materialize()
        assert arrays["A"].shape[1] == arrays["B"].shape[0]
        assert arrays["A"].dtype == arrays["B"].dtype
        if c.layout == "strided":
            assert all(not a.flags.c_contiguous and a.strides[-1] == 2 * a.itemsize for a in arrays.values())
        elif c.pattern != "random":
            assert all(np.unique(a).size == 1 for a in arrays.values())


def test_deferred_task_declaration_expands_before_freeze_and_roundtrips():
    p = policy()
    raw = {"coverage_policy": p.content_dict()}
    t = task(raw)
    generated = generate_correctness_suite(p)
    assert as_suite(raw) == generated
    frozen = freeze_from_task(t)
    assert frozen.correctness_suite == generated
    assert load_dict(json.loads(json.dumps(frozen.as_dict()))).hash == frozen.hash
    assert t.to_dict()["constraints"]["correctness_suite"] == generated.content_dict()
    assert t.constraints.correctness_suite is raw and "cases" not in raw
    assert as_suite(generated.content_dict()) == generated


def test_policy_order_is_canonical_but_input_axis_order_is_semantic():
    p = policy()
    reordered = dataclasses.replace(p, axes=p.axes[::-1], tensors=p.tensors[::-1],
        dtypes=p.dtypes[::-1], patterns=p.patterns[::-1], layouts=p.layouts[::-1])
    assert p.digest() == reordered.digest()
    assert generate_correctness_suite(p).digest() == generate_correctness_suite(reordered).digest()
    changed = dataclasses.replace(p, tensors=(TensorAxes("A", ("k", "m")), p.tensors[1]))
    assert p.digest() != changed.digest()


@pytest.mark.parametrize("field,value", [("seed", 18), ("max_cases", 127),
    ("max_input_bytes", 4096), ("layouts", ("contiguous",)), ("patterns", ("random",))])
def test_policy_changes_enter_suite_and_contract_hash(field, value):
    p = policy()
    a, b = generate_correctness_suite(p), generate_correctness_suite(dataclasses.replace(p, **{field: value}))
    assert a.digest() != b.digest()
    assert freeze_from_task(task(a)).hash != freeze_from_task(task(b)).hash


def test_axis_anchor_alignment_and_domain_are_hashed_even_when_same_cases_remain_legal():
    p = policy()
    s = generate_correctness_suite(p)
    changed = dataclasses.replace(p, axes=tuple(dataclasses.replace(a, alignment=4) for a in p.axes))
    s2 = dataclasses.replace(s, coverage_policy=changed)
    assert lint_coverage(s2)["status"] == "passed"  # Rules below/above anchor 8 are unchanged.
    assert freeze_from_task(task(s)).hash != freeze_from_task(task(s2)).hash
    payload = freeze_from_task(task(s)).as_dict()
    payload["correctness_suite"]["coverage_policy"]["seed"] += 1
    with pytest.raises(ContractError, match="哈希"):
        load_dict(payload)


@pytest.mark.parametrize("key,value", [("minimum", True), ("maximum", 9.5), ("anchor", "8"),
    ("alignment", 1), ("minimum", 0), ("maximum", 7), ("anchor", 10)])
def test_axis_strict_loader_never_coerces_invalid_boundaries(key, value):
    raw = policy().content_dict()
    raw["axes"][0][key] = value
    with pytest.raises(BenchCaseError):
        as_suite({"coverage_policy": raw})


@pytest.mark.parametrize("key,value", [("recipe_version", "future-v2"), ("dtypes", ["bf16"]),
    ("layouts", []), ("patterns", ["zeros"]), ("seed", True), ("max_cases", 0),
    ("max_input_bytes", float("inf")), ("dtypes", ["fp32", "fp32"]), ("unknown_typo", 1)])
def test_policy_strict_loading_and_vocabulary(key, value):
    raw = policy().content_dict()
    raw[key] = value
    with pytest.raises(BenchCaseError):
        as_suite({"coverage_policy": raw})


@pytest.mark.parametrize("bad", [
    {"axes": ()}, {"axes": (BoundaryAxis("k", 1, 9, 8),) * 2},
    {"tensors": (TensorAxes("A", ("missing",)),)},
    {"tensors": (TensorAxes("A", ("m",)),)}, {"patterns": None}, {"layouts": None}])
def test_invalid_python_declarations_are_rejected_without_allocations(bad):
    with pytest.raises(BenchCaseError):
        generate_correctness_suite(policy(**bad))


@pytest.mark.parametrize("cases", [None, []])
def test_explicit_empty_manual_cases_are_not_silently_regenerated(cases):
    with pytest.raises(BenchCaseError):
        as_suite({"cases": cases, "coverage_policy": policy().content_dict()})


def test_old_suite_content_hash_and_unchecked_coverage_remain_unchanged():
    from tests.unit.test_correctness_suite import case
    s = CorrectnessSuite((case(),))
    assert set(s.content_dict()) == {"cases", "deterministic", "required_checks"}
    assert lint_coverage(s)["status"] == "not_declared"
    assert "coverage_policy" not in freeze_from_task(task(s)).as_dict()["correctness_suite"]
    assert "未声明规则" in build_report(artifact=KernelArtifact(verification_results=[
        VerificationResult(VerificationLevel.L2_FUNCTION, True, metrics={"correctness": {"verdict": "passed"}})]))


def test_lint_uses_actual_dimensions_not_labels_number_or_a_favorable_dtype():
    s = generate_correctness_suite(policy())
    # Rename the aligned point 'tail' and duplicate its content under many names.
    aligned = next(c for c in s.cases if c.inputs.tensors[0].shape == (8, 8)
        and c.inputs.tensors[1].shape == (8, 8) and c.pattern == "random" and c.layout == "contiguous")
    fake = dataclasses.replace(s, cases=tuple(dataclasses.replace(aligned, case_id=f"tail-{i}") for i in range(40)))
    lint = lint_coverage(fake)
    assert lint["status"] == "missing"
    assert "shape:below:m/dtype:fp32" in lint["missing"]
    assert any(k.endswith("dtype:fp16") for k in lint["missing"])
    assert "pattern:zeros/dtype:fp32" in lint["missing"]
    assert "layout:strided/dtype:fp32" in lint["missing"]
    renamed = dataclasses.replace(s, cases=tuple(dataclasses.replace(c, case_id=f"opaque-{i}") for i, c in enumerate(s.cases)))
    assert lint_coverage(renamed)["status"] == "passed"


def test_zero_value_cases_cannot_substitute_random_tail_witnesses():
    s = generate_correctness_suite(policy())
    replaced = dataclasses.replace(s, cases=tuple(dataclasses.replace(c, pattern="zeros") for c in s.cases))
    assert any(k.startswith("shape:below:") for k in lint_coverage(replaced)["missing"])


@pytest.mark.parametrize("bad", ["domain", "shared_axis", "name", "rank", "dtype", "layout"])
def test_manual_cases_cannot_lie_about_domain_bindings_or_layout_feasibility(bad):
    s = generate_correctness_suite(policy())
    c = next(c for c in s.cases if c.pattern == "random" and c.layout == "contiguous")
    tensors = list(c.inputs.tensors)
    if bad == "domain":
        tensors[0] = dataclasses.replace(tensors[0], shape=(10, tensors[0].shape[1]))
    elif bad == "shared_axis":
        tensors[1] = dataclasses.replace(tensors[1], shape=(9 if tensors[0].shape[1] != 9 else 8, tensors[1].shape[1]))
    elif bad == "name":
        tensors[0] = dataclasses.replace(tensors[0], name="fakeA")
    elif bad == "rank":
        tensors[0] = dataclasses.replace(tensors[0], shape=(8,))
    elif bad == "dtype":
        tensors[0] = dataclasses.replace(tensors[0], dtype="fp64")
    else:
        tensors[0] = dataclasses.replace(tensors[0], shape=(1, 1))
        tensors[1] = dataclasses.replace(tensors[1], shape=(1, 1))
        c = dataclasses.replace(c, layout="strided")
    c = dataclasses.replace(c, inputs=dataclasses.replace(c.inputs, tensors=tuple(tensors)))
    manual = dataclasses.replace(s, cases=(c,) + tuple(x for x in s.cases if x.case_id != c.case_id))
    assert lint_coverage(manual)["status"] == "invalid"
    with pytest.raises(ContractError, match="覆盖"):
        freeze_from_task(task(manual))


def test_inapplicable_neighbors_are_explicit_and_never_generated_outside_legal_box():
    p = BoundaryPolicy((BoundaryAxis("n", 3, 5, 4),), (TensorAxes("X", ("n",)),), layouts=("contiguous",))
    s = generate_correctness_suite(p)
    lint = lint_coverage(s)
    assert lint["status"] == "passed"
    assert set(lint["inapplicable"]) == {"shape:aligned", "shape:one:n", "shape:below:n", "shape:above:n"}
    assert {c.inputs.tensors[0].shape for c in s.cases} == {(3,), (4,), (5,)}
    assert "shape:tail:n/dtype:fp32" in lint["witnesses"]


@pytest.mark.parametrize("overrides", [{"max_cases": 1}, {"max_input_bytes": 64},
    {"axes": tuple(BoundaryAxis(a, 1, 10**20, 8) for a in ("m", "n", "k"))}])
def test_budget_failure_never_allocates_or_silently_clips_cases(monkeypatch, overrides):
    materialize = Mock(side_effect=AssertionError("must not allocate"))
    monkeypatch.setattr(BenchmarkCase, "materialize", materialize)
    p = policy(**overrides)
    with pytest.raises(BenchCaseError, match="预算"):
        generate_correctness_suite(p)
    v = verifier()
    backend_compile = Mock()
    v.backend.compile = backend_compile
    r = v.verify(GOOD, task({"coverage_policy": p.content_dict()}), primary())
    assert r[0].metrics["correctness"]["verdict"] == "unknown"
    backend_compile.assert_not_called()
    v._verify_performance.assert_not_called()
    materialize.assert_not_called()


def test_unrepresentable_declared_layout_is_not_removed_by_generator():
    p = BoundaryPolicy((BoundaryAxis("n", 1, 1, 1),), (TensorAxes("X", ("n",)),))
    with pytest.raises(BenchCaseError, match="布局"):
        generate_correctness_suite(p)
    p = dataclasses.replace(p, layouts=("contiguous",))
    s = generate_correctness_suite(p)
    assert lint_coverage(s)["status"] == "passed"
    assert set(lint_coverage(s)["inapplicable"]) == {"shape:aligned", "shape:below:n", "shape:above:n"}


def test_layout_witness_uses_legal_non_tiny_point_when_anchor_cannot_be_strided():
    p = BoundaryPolicy((BoundaryAxis("rows", 1, 1, 1), BoundaryAxis("cols", 1, 9, 1)),
        (TensorAxes("X", ("rows", "cols")), TensorAxes("weight", ("cols",))))
    s = generate_correctness_suite(p)
    layout = next(c for c in s.cases if c.layout == "strided")
    assert all(t.shape[-1] > 1 for t in layout.inputs.tensors)
    assert all(v.strides[-1] == v.itemsize * 2 for v in layout.materialize().values())
    assert lint_coverage(s)["status"] == "passed"


def test_policy_binding_can_express_shared_square_axes_without_guessing_operator_type():
    p = BoundaryPolicy((BoundaryAxis("n", 1, 9, 8),), (TensorAxes("X", ("n", "n")),))
    s = generate_correctness_suite(p)
    assert lint_coverage(s)["status"] == "passed"
    assert all(c.inputs.tensors[0].shape[0] == c.inputs.tensors[0].shape[1] for c in s.cases)


def test_broadcast_policy_lint_cannot_hide_remote_readonly_execution_gap(monkeypatch):
    from tests.unit.test_input_wire_format import _runner
    from tests.unit.test_remote_input_fidelity import _offline_verifier, GOOD as TORCH_GOOD
    # CPU protocol fixture explicitly does NOT claim REAL_DEVICE.
    v, _ = _offline_verifier(_runner(), monkeypatch)
    # Focus on the readonly gap, not fp16 product-rounding vs fp32-accum GEMM.
    p = dataclasses.replace(finite_policy(), dtypes=("fp32",), layouts=("contiguous", "broadcast"))
    r = v.verify(TORCH_GOOD, task({"coverage_policy": p.content_dict()}), primary())
    c = r[-1].metrics["correctness"]
    assert c["coverage_lint"]["status"] == "passed" and c["verdict"] == "unknown"
    rows = [row for row in c["cases"] if row["layout"] == "broadcast"]
    assert len(rows) == 1 and all(row["verdict"] == "unknown" and "只读" in row["reason"] for row in rows)
    v._verify_performance.assert_not_called()


def test_missing_coverage_is_unknown_hold_no_candidate_execution_no_l3_and_reported():
    s = generate_correctness_suite(policy())
    s = dataclasses.replace(s, cases=tuple(c for c in s.cases if c.inputs.tensors[0].dtype == "fp32"))
    v = verifier()
    v.backend.compile = Mock(side_effect=AssertionError("must not compile"))
    v.backend.execute = Mock(side_effect=AssertionError("must not execute"))
    r = v.verify(GOOD, task(s), primary())
    assert len(r) == 1 and r[0].level == VerificationLevel.L2_FUNCTION
    assert r[0].metrics["correctness"]["verdict"] == "unknown"
    assert credential(r) is None and classify(r).route == ROUTE_HOLD
    v._verify_performance.assert_not_called()
    v.backend.compile.assert_not_called()
    v.backend.execute.assert_not_called()
    artifact = KernelArtifact(verification_results=r)
    report = build_report(artifact=artifact)
    assert "边界覆盖 lint: **missing**" in report and "dtype:fp16" in report
    assert "覆盖" in collect_unable(artifact=artifact).render()
    with pytest.raises(ContractError, match="覆盖"):
        freeze_from_task(task(s))


def test_generated_suite_runs_l2_and_keeps_the_original_performance_workload():
    raw = {"coverage_policy": finite_policy().content_dict()}
    v = verifier()
    inputs = primary()
    r = v.verify(GOOD, task(raw), inputs)
    assert r[1].passed, r[1].message
    c = r[1].metrics["correctness"]
    assert c["coverage_lint"]["status"] == "passed"
    assert all(row["verdict"] == "passed" and row["determinism"] == "passed" for row in c["cases"])
    assert all(row["metrics"]["input_fidelity_verified"] is True for row in c["cases"])
    v._verify_performance.assert_called_once()
    assert v._verify_performance.call_args.args[1] is inputs
    assert credential(r) is True


@pytest.mark.parametrize("body", [
    "return A @ B if A.shape == (8, 8) and B.shape == (8, 8) else np.zeros((A.shape[0], B.shape[1]), dtype=A.dtype)",
    "return A @ B if A.dtype == np.float32 else np.zeros((A.shape[0], B.shape[1]), dtype=A.dtype)",
    "a = np.ndarray(A.shape, dtype=A.dtype, buffer=A.base) if A.base is not None else A\n    return a @ B",
])
def test_actual_bad_candidates_never_enter_l3_despite_passing_coverage_lint(body):
    # Last negative is deliberately wrong storage interpretation on strided A.
    code = "import numpy as np\ndef kernel(A, B):\n    " + body + "\n"
    v = verifier()
    r = v.verify(code, task({"coverage_policy": finite_policy().content_dict()}), primary())
    assert not r[-1].passed
    assert r[-1].metrics["correctness"]["coverage_lint"]["status"] == "passed"
    assert r[-1].metrics["correctness"]["verdict"] == "failed"
    v._verify_performance.assert_not_called()


def test_generation_and_lint_do_not_turn_nonfinite_reference_into_pass():
    p = policy(patterns=("random", "nan", "inf"), layouts=("contiguous",))
    v = verifier()
    r = v.verify(GOOD, task({"coverage_policy": p.content_dict()}), primary())
    assert lint_coverage(as_suite(task({"coverage_policy": p.content_dict()}).constraints.correctness_suite))["status"] == "passed"
    assert r[-1].metrics["correctness"]["verdict"] == "unknown"
    v._verify_performance.assert_not_called()


def test_generated_fp16_extreme_gemm_overflow_remains_unknown_not_dropped():
    v = verifier()
    r = v.verify(GOOD, task({"coverage_policy": policy().content_dict()}), primary())
    c = r[-1].metrics["correctness"]
    assert c["coverage_lint"]["status"] == "passed" and c["verdict"] == "unknown"
    large = [row for row in c["cases"] if row["pattern"] == "large" and row["dtypes"]["A"] == "fp16"]
    assert len(large) == 1 and large[0]["verdict"] == "unknown"
    assert large[0]["metrics"]["reference_nonfinite"] > 0
    v._verify_performance.assert_not_called()


@pytest.mark.parametrize("missing", [False, True])
def test_conductor_defers_to_trusted_generation_or_holds_before_llm_and_saves_valid_runs(missing, tmp_path):
    from tests.golden_regression import _FixedCodeMockLLM
    from operator_agent import Conductor
    from operator_agent.core.artifact_store import ArtifactStore
    raw = {"coverage_policy": finite_policy().content_dict()}
    if missing:
        s = as_suite(raw)
        raw = dataclasses.replace(s, cases=tuple(c for c in s.cases if c.layout == "contiguous"))
    t = task(raw)
    conductor = Conductor(backend=MockGPU(), llm=_FixedCodeMockLLM(code=GOOD), max_optimize_rounds=1)
    conductor.store = ArtifactStore(str(tmp_path))
    conductor.verifier._verify_performance = Mock(return_value=VerificationResult(
        VerificationLevel.L3_PERFORMANCE, True, metrics={"gflops": 1, "measurement_validity": "valid"}))
    conductor._record_failure = Mock()
    if missing:
        conductor.router.route = Mock(side_effect=AssertionError("do not call LLM route"))
        conductor.designer.generate = Mock(side_effect=AssertionError("do not design"))
    artifact = conductor.execute(t, primary())
    c = next(r.metrics["correctness"] for r in artifact.verification_results if r.level == VerificationLevel.L2_FUNCTION)
    assert c["verdict"] == ("unknown" if missing else "passed")
    conductor._record_failure.assert_not_called()
    if missing:
        assert not artifact.all_passed and classify(artifact.verification_results).route == ROUTE_HOLD
        conductor.router.route.assert_not_called()
        conductor.verifier._verify_performance.assert_not_called()
    else:
        assert artifact.all_passed
        saved = Path(artifact.saved_dir)
        data = json.loads((saved / "result.json").read_text())
        assert data["task"]["constraints"]["correctness_suite"] == as_suite(raw).content_dict()
        assert "边界覆盖 lint: **passed**" in (saved / "report.md").read_text()
