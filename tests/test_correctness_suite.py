"""Suite identity, contract integrity, strict L2 gates and explicit unknowns."""
import dataclasses
from unittest.mock import Mock

import numpy as np
import pytest

from operator_agent import (CorrectnessCase, CorrectnessSuite, ConstraintSpec,
                            MockGPU, SourceSpec, TargetSpec, TaskSchema, Verifier)
from operator_agent.bench_cases import BenchmarkCase, TensorSpec, BenchCaseError
from operator_agent.contract import ContractError, freeze_from_task, load_dict
from operator_agent.conductor import Conductor, _correctness_credential
from operator_agent.correctness import as_suite, verdict_of
from operator_agent.failure_route import classify, ROUTE_HOLD
from operator_agent.report import build_report, collect_unable
from operator_agent.schemas import KernelArtifact, VerificationLevel, VerificationResult

GOOD = "import numpy as np\ndef kernel(A, B):\n    return A @ B\n"


def case(case_id="random", m=7, n=9, k=5, dtype="fp32", pattern="random",
         layout="contiguous", seed=1):
    return CorrectnessCase(case_id, BenchmarkCase(tensors=(
        TensorSpec("A", (m, k), dtype), TensorSpec("B", (k, n), dtype)), seed=seed),
        pattern=pattern, layout=layout)


def task(suite=None):
    return TaskSchema(source=SourceSpec(kernel_type="matmul"),
                      target=TargetSpec(dsl="python"),
                      constraints=ConstraintSpec(min_speedup=0.5,
                                                 correctness_suite=suite))


def verify(code=GOOD, suite=None, primary=None):
    verifier = Verifier(MockGPU())
    # A rejected suite must never consume the performance oracle, even though
    # the caller's original aligned shape can pass with the bad candidate.
    verifier._verify_performance = Mock(return_value=VerificationResult(
        level=VerificationLevel.L3_PERFORMANCE, passed=True))
    results = verifier.verify(code, task(suite), primary or case(m=8, n=8, k=8).materialize())
    return results, verifier


def test_suite_roundtrip_and_canonical_hash():
    suite = CorrectnessSuite((case("fp16", dtype="fp16"), case("fp32")))
    assert as_suite(suite.content_dict()) == suite
    assert dataclasses.replace(suite, cases=suite.cases[::-1]).digest() == suite.digest()
    frozen = freeze_from_task(task(suite))
    assert load_dict(frozen.as_dict()).hash == frozen.hash
    assert frozen.hash == freeze_from_task(task(suite.content_dict())).hash
    assert frozen.hash != freeze_from_task(task()).hash
    assert task(suite).to_dict()["constraints"]["correctness_suite"] == suite.content_dict()
    assert "correctness_suite" not in task().to_dict()["constraints"]


@pytest.mark.parametrize("field,value", [
    ("pattern", "zeros"), ("layout", "strided"), ("case_id", "renamed")])
def test_case_changes_are_hashed(field, value):
    base = CorrectnessSuite((case(),))
    changed = dataclasses.replace(base, cases=(dataclasses.replace(base.cases[0], **{field: value}),))
    assert freeze_from_task(task(base)).hash != freeze_from_task(task(changed)).hash


@pytest.mark.parametrize("changed", [case(seed=2), case(k=6), case(dtype="fp16")])
def test_input_shape_dtype_seed_are_hashed(changed):
    assert CorrectnessSuite((case(),)).digest() != CorrectnessSuite((changed,)).digest()


def test_suite_checks_and_determinism_are_hashed():
    suite = CorrectnessSuite((case(),))
    assert suite.digest() != dataclasses.replace(suite, deterministic=False).digest()
    assert suite.digest() != dataclasses.replace(suite, required_checks=("backward",)).digest()


def test_suite_determinism_policy_is_consistent_with_frozen_semantics():
    from operator_agent.contract import freeze
    suite = CorrectnessSuite((case(),), deterministic=False)
    c = freeze_from_task(task(suite))
    assert c.dtype_semantics.deterministic is False
    assert load_dict(c.as_dict()).hash == c.hash
    with pytest.raises(ContractError, match="确定性"):
        freeze(kernel_type=c.kernel_type,
               dtype_semantics=dataclasses.replace(c.dtype_semantics, deterministic=True),
               baseline=c.baseline, shapes=c.shapes, max_error=c.max_error,
               min_speedup=c.min_speedup, correctness_suite=suite)


def test_tampered_suite_payload_refused():
    payload = freeze_from_task(task(CorrectnessSuite((case(),)))).as_dict()
    payload["correctness_suite"]["cases"][0]["pattern"] = "zeros"
    with pytest.raises(ContractError, match="哈希"):
        load_dict(payload)


@pytest.mark.parametrize("bad", [
    {"cases": []}, {"cases": [case().content_dict()], "deterministic": 1},
    {"cases": [case().content_dict()], "required_checks": ["made_up"]},
    {"cases": [case().content_dict(), case().content_dict()]},
])
def test_invalid_suite_fails_closed(bad):
    assert task(bad).constraints.validate()
    with pytest.raises(ContractError):
        freeze_from_task(task(bad))
    results, verifier = verify(suite=bad)
    assert results[-1].metrics["correctness"]["verdict"] == "unknown"
    verifier._verify_performance.assert_not_called()


@pytest.mark.parametrize("bad_dim", [True, 2.5, "2", -1])
def test_new_suite_loader_does_not_coerce_bad_dimensions(bad_dim):
    payload = case().content_dict()
    payload["inputs"]["tensors"][0]["shape"][0] = bad_dim
    with pytest.raises(BenchCaseError):
        CorrectnessCase.from_dict(payload)


@pytest.mark.parametrize("pattern", ["random", "zeros", "constant", "large", "small"])
def test_materialization_is_reproducible_and_declared(pattern):
    c = case(pattern=pattern)
    first, second = c.materialize(), c.materialize()
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
        assert first[key].dtype == np.float32
    if pattern != "random":
        assert np.unique(first["A"]).size == 1


@pytest.mark.parametrize("layout", ["strided", "broadcast"])
def test_layout_materializer_really_perturbs_strides(layout):
    arrays = case(layout=layout).materialize()
    assert not arrays["A"].flags.c_contiguous
    if layout == "broadcast":
        assert arrays["A"].strides[-1] == 0
    else:
        assert arrays["A"].strides[-1] == arrays["A"].itemsize * 2


def test_correct_candidate_passes_all_values_shapes_and_dtypes():
    suite = CorrectnessSuite((
        case("tiny", m=1, n=1, k=1), case("tail"),
        case("fp16", dtype="fp16"), case("zeros", pattern="zeros"),
        case("constant", pattern="constant"), case("large", pattern="large"),
        case("small", pattern="small")))
    results, verifier = verify(suite=suite)
    c = results[1].metrics["correctness"]
    assert c["verdict"] == "passed"
    assert len(c["cases"]) == 7
    assert all(row["determinism"] == "passed" for row in c["cases"])
    assert _correctness_credential(results) is True
    verifier._verify_performance.assert_called_once()
    # Correctness inputs must not leak into the performance workload.
    perf_inputs = verifier._verify_performance.call_args.args[1]
    assert isinstance(perf_inputs, dict) and perf_inputs["A"].shape == (8, 8)


@pytest.mark.parametrize("body,suite_case", [
    ("return A @ B if A.shape[0] == 8 else np.zeros((A.shape[0], B.shape[1]))", case("tail")),
    ("out = A @ B\n    out[-1] = 0 if A.shape[0] % 8 else out[-1]\n    return out", case("tail")),
    ("return A @ B if A.dtype == np.float32 else np.zeros((A.shape[0], B.shape[1]))", case("dtype", dtype="fp16")),
    ("return A @ B if np.any(A) else np.ones((A.shape[0], B.shape[1]))", case("zeros", pattern="zeros")),
])
def test_shape_tail_dtype_value_specialization_cannot_reach_l3(body, suite_case):
    code = "import numpy as np\ndef kernel(A, B):\n    " + body + "\n"
    results, verifier = verify(code, CorrectnessSuite((suite_case,)))
    assert results[0].passed
    c = results[-1].metrics["correctness"]
    assert c["verdict"] == "failed"
    assert c["cases"][0]["case_id"] == suite_case.case_id
    assert "first_bad_index" in c["cases"][0]["metrics"]
    verifier._verify_performance.assert_not_called()
    assert _correctness_credential(results) is False


@pytest.mark.parametrize("layout", ["strided", "broadcast"])
def test_layout_fidelity_gap_is_unknown_not_success(layout, monkeypatch):
    # A provider declaration is not proof. Strip the runtime receipt to emulate
    # an old/unsupported transport, even though it returned correct numbers.
    original = MockGPU.execute
    def no_receipt(self, code, inputs):
        result = original(self, code, inputs)
        result.metrics.pop("input_fidelity", None)
        return result
    monkeypatch.setattr(MockGPU, "execute", no_receipt)
    results, verifier = verify(suite=CorrectnessSuite((case(layout=layout),)))
    assert not results[-1].passed
    assert results[-1].metrics["correctness"]["verdict"] == "unknown"
    assert "回读证据" in results[-1].metrics["correctness"]["reason"]
    assert _correctness_credential(results) is None
    assert Conductor._l2_verdict(KernelArtifact(verification_results=results)) is None
    assert classify(results).route == ROUTE_HOLD
    verifier._verify_performance.assert_not_called()


def test_missing_reference_is_unknown_not_execution_success(monkeypatch):
    monkeypatch.setattr(Verifier, "_generate_reference", lambda *args, **kwargs: None)
    results, verifier = verify(suite=CorrectnessSuite((case(),)))
    assert results[-1].metrics["correctness"]["verdict"] == "unknown"
    assert _correctness_credential(results) is None
    verifier._verify_performance.assert_not_called()


def test_partial_nonfinite_reference_is_unknown(monkeypatch):
    def bad_ref(self, task, inputs, **kwargs):
        out = (inputs["A"] @ inputs["B"]).astype(np.float32)
        out[0] = np.nan
        return out
    monkeypatch.setattr(Verifier, "_generate_reference", bad_ref)
    results, verifier = verify(suite=CorrectnessSuite((case(),)))
    assert results[-1].metrics["correctness"]["verdict"] == "unknown"
    verifier._verify_performance.assert_not_called()


def test_remote_input_regeneration_cannot_silently_claim_value_coverage():
    from operator_agent.remote_executor import SSHExecutor
    backend = SSHExecutor(host="offline")
    backend._deployed = True
    # A capability claim cannot substitute for runtime evidence from old runners.
    backend._ssh_exec_json = Mock(return_value={"success": True, "max_error": 0,
        "coverage": 1.0, "compared_elements": 64})
    assert backend.correctness_input_support(GOOD, "contiguous")[0] is True
    result = Verifier(backend)._verify_function(GOOD, case().materialize(), None,
                                               task(CorrectnessSuite((case(),))))
    assert result.metrics["correctness"]["verdict"] == "unknown"
    assert result.metrics["input_fidelity_verified"] is None


@pytest.mark.parametrize("check", ["backward", "sanitizer", "racecheck", "aliasing", "inplace", "accum_fp16"])
def test_unsupported_semantics_block_promotion(check):
    results, verifier = verify(suite=CorrectnessSuite((case(),), required_checks=(check,)))
    assert results[-1].metrics["correctness"]["verdict"] == "unknown"
    assert results[-1].metrics["correctness"]["cases"][-1]["case_id"] == f"check:{check}"
    verifier._verify_performance.assert_not_called()


def test_nondeterministic_candidate_below_tolerance_is_still_rejected():
    code = ("import numpy as np\ndef kernel(A, B):\n"
            "    return (A @ B).astype(np.float64) + np.random.default_rng().uniform(0, 1e-10)\n")
    results, verifier = verify(code, CorrectnessSuite((case(),)))
    row = results[-1].metrics["correctness"]["cases"][0]
    assert row["verdict"] == "failed" and "确定性失败" in row["reason"]
    verifier._verify_performance.assert_not_called()


def test_unknown_report_reads_l2_even_without_benchmark_data():
    results, _ = verify(suite=CorrectnessSuite((case(),), required_checks=("aliasing",)))
    artifact = KernelArtifact(verification_results=results)
    report = build_report(artifact=artifact)
    assert "CorrectnessSuite (§20)" in report
    assert "unknown" in report and "aliasing" in report
    assert "正确性检查 check:aliasing 无法确立" in collect_unable(artifact=artifact).render()


def test_suite_declared_checks_cannot_be_replaced_by_caller_inputs():
    suite = CorrectnessSuite((case("different_shape", m=13),))
    results, _ = verify(suite=suite, primary=case(m=8, n=8, k=8).materialize())
    row = results[1].metrics["correctness"]["cases"][0]
    assert row["shape_key"] == suite.cases[0].inputs.key()


def test_empty_and_partial_comparisons_are_unknown():
    for m in ({"max_error": 0, "coverage": 0.5, "compared_elements": 2},
              {"max_error": 0, "coverage": 1, "compared_elements": 0}):
        result = VerificationResult(VerificationLevel.L2_FUNCTION, True, metrics=m)
        assert verdict_of(result)[0] == "unknown"


@pytest.mark.parametrize("broken", [False, True])
def test_conductor_candidate_selection_and_saved_report_use_suite(broken, tmp_path):
    import json
    from pathlib import Path
    from tests.golden_regression import _FixedCodeMockLLM
    from operator_agent.artifact_store import ArtifactStore

    code = ("import numpy as np\ndef kernel(A, B):\n"
            "    return A @ B if A.shape[0] == 8 else np.zeros((A.shape[0], B.shape[1]))\n"
            if broken else GOOD)
    suite = CorrectnessSuite((case("tail"),))
    t = task(suite)
    conductor = Conductor(backend=MockGPU(), llm=_FixedCodeMockLLM(code=code),
                          max_optimize_rounds=1)
    conductor.store = ArtifactStore(str(tmp_path))
    perf = Mock(return_value=VerificationResult(
        VerificationLevel.L3_PERFORMANCE, True,
        metrics={"gflops": 1.0, "speedup_vs_baseline": 1.0, "measurement_validity": "valid"}))
    conductor.verifier._verify_performance = perf
    artifact = conductor.execute(t, test_inputs=case(m=8, n=8, k=8).materialize())
    if broken:
        assert not artifact.all_passed
        perf.assert_not_called()
    else:
        assert artifact.all_passed
        assert perf.called
    l2 = next(r for r in artifact.verification_results if r.level == VerificationLevel.L2_FUNCTION)
    assert l2.metrics["correctness"]["verdict"] == ("failed" if broken else "passed")
    saved = Path(artifact.saved_dir)
    rows = json.loads((saved / "verification.json").read_text())
    assert next(r for r in rows if r["level"] == "L2")["metrics"]["correctness"]["suite_hash"] == suite.digest()
    assert "CorrectnessSuite (§20)" in (saved / "report.md").read_text()
    payload = json.loads((saved / "result.json").read_text())
    assert payload["task"]["constraints"]["correctness_suite"] == suite.content_dict()
