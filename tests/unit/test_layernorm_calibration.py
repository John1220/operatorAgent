"""Calibration witnesses: incomplete evidence is not a calibrated/active profile."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from benchmarks import calibrate_layernorm as driver
from benchmarks import layernorm_numerics as numerics
from operator_agent.anticheat import static_audit
from operator_agent.correctness_profiles import with_default_correctness
from operator_agent.correctness_boundaries import lint_coverage
from operator_agent.input_fidelity import describe_inputs
from operator_agent.mock_gpu import MockGPU
from operator_agent.verifier import Verifier


def measured(inputs=None, reference=None):
    torch = pytest.importorskip("torch")
    inputs = inputs or {"X": np.arange(12, dtype=np.float16).reshape(3, 4)}
    if reference is None:
        reference, _ = Verifier(MockGPU())._build_reference_fn(driver.reference_task(), inputs)
    return inputs, numerics.measure(inputs, reference, "cpu",
        dict(factory="test_fixture", source_sha256="a" * 64))


def test_proposed_suite_keeps_original_boundaries_patterns_and_strides():
    suite = driver.proposed_suite()
    assert len(suite.cases) == 12
    assert suite.digest() == "0453674dc3138df396e7455a784e27a0f885a398fba55834d6073f0fbb706ecd"
    assert lint_coverage(suite)["status"] == "passed"
    assert all(c.inputs.tensors[0].dtype == "fp16" for c in suite.cases)
    rows = driver.fixtures()
    assert len(rows) == len({name for name, _, _ in rows}) == 32
    original = [(n, xs) for n, origin, xs in rows if origin == "original-boundary"]
    assert {n for n, _ in original} == {c.case_id for c in suite.cases}
    for name, xs in original:
        expected = next(c.materialize() for c in suite.cases if c.case_id == name)
        assert describe_inputs(xs) == describe_inputs(expected)
    for name, _, xs in rows:
        x = xs["X"]
        assert x.dtype == np.float16 and np.isfinite(x).all() and x.nbytes <= 1024
        if name.endswith("strided"):
            assert not x.flags.c_contiguous and x.strides[-1] == 2 * x.itemsize
    a, b = driver.fixtures(), driver.fixtures()
    assert [describe_inputs(xs) for _, _, xs in a] == [describe_inputs(xs) for _, _, xs in b]


def test_zero_reference_residual_is_not_hidden_by_relative_tolerance():
    zero = np.zeros((8, 1), np.float16)
    residual = np.full_like(zero, 2.384185791015625e-5)
    reading = numerics.compare(zero, residual)
    assert reading["passed"] is False and reading["bad_elements"] == 8
    assert (reading["atol"], reading["rtol"], reading["rho"]) == (1e-5, .01, 1.0)
    assert numerics.compare(zero, np.full_like(zero, 1e-6))["passed"] is True
    assert numerics.compare(zero, zero, max_error=1e-8)["atol"] == 1e-8


@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf))
def test_nonfinite_oracle_or_output_never_passes(value):
    a = np.ones((1, 1))
    bad = np.full_like(a, value)
    assert numerics.compare(a, bad)["passed"] is False
    assert numerics.compare(bad, a)["passed"] is False


def test_output_rounding_is_separate_from_accumulation_error():
    x = np.array([[0, 1, 4]], np.float16)
    truth = numerics.centered(x, np.float64)
    rounded = truth.astype(np.float16)
    assert np.abs(rounded.astype(np.float64)-truth).max() > 1e-5
    assert numerics.compare(rounded, rounded)["passed"] is True
    inputs, result = measured({"X": x})
    assert driver.assess_measurement(result, inputs, "cpu") == []
    assert result["oracle"]["quantization_max_abs"] > 1e-5
    native = result["arms"]["production_eager"]
    assert native["vs_rounded_oracle"]["max_abs"] == 0
    assert native["max_abs_to_unrounded"] > 1e-5
    assert result["reference_identity"]["semantics"]["eager_accum"] == "implementation_defined_not_attested"


def test_calibration_observes_supplied_reference_does_not_replace_it():
    torch = pytest.importorskip("torch")
    x = np.full((8, 1), 100, np.float16)
    residue = torch.full((8, 1), .0009765625, dtype=torch.float16)
    reference = Mock(return_value=residue)
    inputs, result = measured({"X": x}, reference)
    assert reference.call_count == 2
    assert result["arms"]["production_eager"]["vs_rounded_oracle"]["passed"] is False
    assert result["arms"]["centered_fp32"]["vs_production"]["passed"] is False
    assert result["arms"]["centered_fp32"]["vs_rounded_oracle"]["passed"] is True
    assert driver.assess_measurement(result, inputs, "cpu") == []


@pytest.mark.parametrize("mutation", ("cpu-as-cuda", "dtype", "repeat", "identity", "source", "semantics",
                                      "shared-source", "arm", "tolerance", "missing-gate", "self", "oracle", "oracle-disagree"))
def test_measurement_gate_refuses_missing_or_mutated_evidence(mutation):
    inputs, result = measured()
    if mutation == "cpu-as-cuda":
        assert driver.assess_measurement(result, inputs, "cuda:0")
        assert driver.assess_measurement(result, {"X": inputs["X"] + np.float16(1)}, "cpu")
        return
    if mutation == "dtype": result["reference_output_dtype"] = "float32"
    if mutation == "repeat": result["reference_repeat_exact"] = False
    if mutation == "identity": result["reference_identity_hash"] = "wrong"
    if mutation == "source": result["reference_identity"]["provenance"].pop("source_sha256")
    if mutation == "semantics": result["reference_identity"]["semantics"] = dict(numerics.SEMANTICS, epsilon=1e-3)
    if mutation == "shared-source": result["reference_identity"]["shared_source_hashes"]["tolerance.py"] = "b" * 64
    if mutation == "arm": result["arms"].pop("centered_fp32")
    if mutation == "tolerance": result["arms"]["centered_fp32"]["vs_production"]["atol"] = 1e-2
    if mutation == "missing-gate": result["arms"]["centered_fp32"].pop("vs_rounded_oracle")
    if mutation == "self": result["arms"]["production_eager"]["vs_production"]["passed"] = False
    if mutation == "oracle": result["oracle"].pop("quantization_max_abs")
    if mutation == "oracle-disagree": result["oracle"]["eager64_max_abs"] = .1
    if mutation != "identity":
        # Isolate semantic/source checks from the independent content hash check.
        result["reference_identity_hash"] = numerics.digest(result["reference_identity"])
    assert driver.assess_measurement(result, inputs, "cpu")


@pytest.mark.parametrize("passed", (True, False))
def test_complete_calibration_even_all_gpu_passes_cannot_enable_profile(passed):
    row = dict(case_id="one", measurement={"arms": {"centered_fp32": {"vs_production": {"passed": passed}}}}, errors=[])
    result = driver.summarize([row], ["one"])
    assert result["measurement_complete"] is True
    assert result["centered_fp32_vs_eager_failures"] == ([] if passed else ["one"])
    assert result["profile_enabled"] is False and result["default_profile"] == "unknown"
    for bad in ([], [row, row], [dict(row, case_id="other")], [dict(row, errors=["missing"])],
                [dict(row, measurement={})], [dict(row, errors=None)]):
        assert driver.summarize(bad, ["one"])["measurement_complete"] is False


def candidate_matrix():
    rows = [dict(case_id="case", variant=v, errors=[], response=dict(success=True, output_digest="same"),
                 repeat_response=dict(success=True, output_digest="same")) for v in driver.CANDIDATE_VARIANTS]
    rows += [dict(case_id=n, variant=v, errors=[], response=dict(success=False, bad_elements=1))
             for v, n in driver.CONTROLS]
    return rows


def test_candidate_matrix_requires_full_repeats_and_rejected_controls():
    rows = candidate_matrix()
    assert driver.assess_candidate_matrix(rows, ["case"]) is True
    for bad in (rows[:-1], rows + rows[:1], []):
        assert driver.assess_candidate_matrix(bad, ["case"]) is False
    for mutation in ("repeat", "control", "missing-errors", "no-bad-element"):
        bad = copy.deepcopy(rows)
        if mutation == "repeat": bad[0]["repeat_response"]["output_digest"] = "different"
        if mutation == "control": bad[-1]["response"]["success"] = True
        if mutation == "missing-errors": bad[0].pop("errors")
        if mutation == "no-bad-element": bad[-1]["response"]["bad_elements"] = 0
        assert driver.assess_candidate_matrix(bad, ["case"]) is False
    # Honest numeric failures are a valid calibration reading, not a fake pass.
    bad = copy.deepcopy(rows)
    bad[0]["response"]["success"] = bad[0]["repeat_response"]["success"] = False
    assert driver.assess_candidate_matrix(bad, ["case"]) is True


def test_fixed_candidates_remain_safe_and_compile_no_library_fallback():
    for variant in (*driver.CANDIDATE_VARIANTS, *(v for v, _ in driver.CONTROLS)):
        code = driver.candidate_source(variant)
        compile(code, "<calibration>", "exec")
        assert static_audit(code, "triton").violations == []
    assert "enable_fp_fusion=False" in driver.candidate_source("unfused-fp32")
    assert driver.candidate_source("centered-fp32") != driver.candidate_source("fp16-intermediate")
    server = driver.diagnostic_server()
    compile(server, "<calibration-peer>", "exec")
    assert "runner_identity(__file__)" in server and "remote_probe(req, run_kernel)" in server
    assert "calibrate_layernorm" not in (driver.ROOT / "remote_scripts/runner_server.py").read_text()
    assert "calibrate_layernorm" not in (driver.ROOT / "remote_scripts/run_kernel.py").read_text()


def test_calibrated_but_unfrozen_profile_still_stops_before_execution():
    task = with_default_correctness(driver.reference_task())
    verifier = Verifier(MockGPU())
    verifier._verify_compile = Mock(side_effect=AssertionError("compiled"))
    verifier._verify_performance = Mock(side_effect=AssertionError("benchmarked"))
    rs = verifier.verify(driver.candidate_source("unfused-fp32"), task, {"X": np.ones((8, 1), np.float16)})
    assert rs[0].metrics["correctness"]["verdict"] == "unknown"
    assert "CPU/CUDA" in rs[0].metrics["correctness"]["reason"]
    assert task.constraints.correctness_suite is None
    verifier._verify_compile.assert_not_called()
    verifier._verify_performance.assert_not_called()


def test_cleanup_precedes_fallible_report_on_prepare_failure(tmp_path, monkeypatch):
    events = []
    backend = Mock(forbidden_actions=[])
    backend.prepare.side_effect = RuntimeError("no idle GPU")
    backend.cleanup.side_effect = lambda: events.append("cleanup") or dict(quiescent=True, removed=True)
    monkeypatch.setattr(driver, "CalibrationSSH", lambda *a, **kw: backend)
    def write(role, path, data):
        if path == "measurements/reference.json":
            events.append("write")
            raise OSError("disk full")
    monkeypatch.setattr(driver, "run_layout", lambda _: SimpleNamespace(root=str(tmp_path), write_for=write))
    with pytest.raises(OSError, match="disk full"):
        driver.run(SimpleNamespace(output=str(tmp_path), host="offline", h20=True, gpu=None))
    assert events == ["cleanup", "write"]


def test_cpu_driver_reports_measurement_completion_not_gpu_or_profile_admission(tmp_path):
    pytest.importorskip("torch")
    assert driver.run(SimpleNamespace(output=str(tmp_path), h20=False)) == 0
    root = next(tmp_path.iterdir())
    summary = json.loads((root / "report/summary.json").read_text())
    assert summary["completed"] is True
    assert summary["candidate_matrix_complete"] is None and summary["candidate_failures"] is None
    assert summary["profile_enabled"] is False and summary["default_profile"] == "unknown"
