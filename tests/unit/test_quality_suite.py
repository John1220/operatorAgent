"""Fixed task inputs and accounting cannot turn partial/replay evidence into quality."""
import copy
from dataclasses import replace

import numpy as np
import pytest

from operator_agent.contract import freeze_from_task
from operator_agent.correctness import as_suite
from operator_agent.correctness_profiles import profile_check, with_default_correctness
from operator_agent.mock_gpu import MockGPU
from operator_agent.quality_suite import (cases, compare_reports, evaluation_inputs_for,
                                         get_case, input_cases_for, inputs_for,
                                         suite_manifest, summarize, task_for)
from operator_agent.run_contract import digest
from operator_agent.schemas import VerificationLevel
from operator_agent.verifier import Verifier


GOOD = {
    "matmul": "def kernel(A,B):\n    return A @ B\n",
    "layernorm": "import numpy as np\ndef kernel(X):\n    z=X.astype(np.float32)\n    m=z.mean(axis=-1,keepdims=True)\n    return ((z-m)/np.sqrt(((z-m)**2).mean(axis=-1,keepdims=True)+1e-5)).astype(X.dtype)\n",
    "softmax": "import numpy as np\ndef kernel(X):\n    z=X.astype(np.float32)\n    e=np.exp(z-z.max(axis=-1,keepdims=True))\n    return (e/e.sum(axis=-1,keepdims=True)).astype(X.dtype)\n",
}


def record(case_id=None, attempt=1, status="passed", **overrides):
    row = dict(case_id=case_id or cases()[0].case_id, attempt=attempt,
               status=status, full_l2_passed=True, performance_target_passed=True,
               measurement_validity="valid", execution_kind="real_device",
               model_calls=1, tokens=100, wall_seconds=2.5)
    row.update(overrides)
    return row


def controls(**overrides):
    value = dict(model="model-a", strategy="plain", budgets={"calls": 2, "tokens": 1024},
                 hardware="fixed-device", protocol="fixed-ruler", reference="frozen")
    value.update(overrides)
    return value


def test_manifest_is_deterministic_detached_and_names_every_fixed_input():
    first = suite_manifest()
    assert first == suite_manifest()
    claimed = first.pop("hash")
    assert claimed == digest(first)
    assert len(first["cases"]) == 10
    assert {row["kernel_type"] for row in first["cases"]} == {"matmul", "layernorm", "softmax"}
    for row in first["cases"]:
        declarations = row["evaluation_inputs"]
        assert len(declarations) == 3
        seeds = [entry["inputs"]["seed"] for entry in declarations]
        assert len(set(seeds)) == 3
    first["cases"][0]["shape"][0] = 999
    assert suite_manifest()["cases"][0]["shape"][0] != 999


@pytest.mark.parametrize("case", cases(), ids=lambda case: case.case_id)
def test_trusted_profile_is_preserved_and_all_inputs_match_it(case):
    task = task_for(case)
    assert task.task_id == case.case_id
    assert profile_check(task, list(evaluation_inputs_for(case)))["status"] == "passed"
    baseline = with_default_correctness(replace(task, constraints=replace(
        task.constraints, correctness_suite=None, correctness_profile=None)))
    assert as_suite(task.constraints.correctness_suite).digest() == as_suite(baseline.constraints.correctness_suite).digest()
    contract = freeze_from_task(task)
    assert contract.cases[0].key() in contract.required_cases
    assert set(inputs_for(case)) == ({"A", "B"} if case.kernel_type == "matmul" else {"X"})
    if case.layout == "strided":
        assert all(not value.flags.c_contiguous for value in inputs_for(case).values())


def test_independent_holdouts_reproduce_without_reusing_primary_values():
    case = get_case("layernorm-tail")
    first, second, third = evaluation_inputs_for(case)
    assert np.array_equal(first["X"], inputs_for(case)["X"])
    assert not np.array_equal(first["X"], second["X"])
    assert not np.array_equal(second["X"], third["X"])
    again = evaluation_inputs_for(case)
    assert all(np.array_equal(a["X"], b["X"]) for a, b in zip((first, second, third), again))


def test_challenge_patterns_cover_numerical_and_layout_failures():
    large_mean = inputs_for("layernorm-large-mean")["X"]
    assert abs(float(large_mean.mean())) > 90
    assert 0 < float(large_mean.var()) < .1
    constant = inputs_for("layernorm-constant-strided")["X"]
    assert np.all(constant == np.float32(.1))
    extreme = inputs_for("softmax-extreme")["X"]
    assert extreme.min() == -100 and extreme.max() == 100
    matmul = inputs_for("matmul-strided-fp32")
    assert matmul["A"].shape == (17, 13) and matmul["B"].shape == (13, 29)


@pytest.mark.parametrize("case", cases(), ids=lambda case: case.case_id)
def test_known_correct_implementations_reach_the_complete_formal_l2(case):
    task = task_for(case, hardware="mock_gpu", dsl="python")
    verifier = Verifier(MockGPU())
    for inputs in evaluation_inputs_for(case):
        results = verifier.verify(GOOD[case.kernel_type], task, inputs,
                                  stop_after=VerificationLevel.L2_FUNCTION)
        l2 = next(result for result in results if result.level is VerificationLevel.L2_FUNCTION)
        assert l2.passed, l2.message
        assert l2.metrics["correctness"]["verdict"] == "passed"
    # These CPU controls validate the harness; they are never native performance evidence.


def test_missing_and_failed_tasks_remain_in_the_task_denominator_and_costs():
    rows = [record(status="generation_failed", full_l2_passed=False, model_calls=1),
            record(attempt=2, model_calls=1)]
    report = summarize(rows, controls=controls())
    assert report["metrics"]["first_full_l2"] == {"count": 0, "denominator": 10, "rate": 0}
    assert report["metrics"]["within_budget_full_l2"]["count"] == 1
    assert report["metrics"]["final_full_l2"]["denominator"] == 10
    assert report["missing_tasks"] == 9
    assert report["costs"]["model_calls"]["per_correct_task"] == 2
    assert report["costs"]["tokens"]["known_total"] == 200
    assert report["attempts"] == rows
    rows[0]["tokens"] = 999
    assert report["attempts"][0]["tokens"] == 100


@pytest.mark.parametrize("change", [
    {"execution_kind": "mock"}, {"execution_kind": "replay"},
    {"measurement_validity": "degraded"}, {"measurement_validity": "uninformative"},
    {"full_l2_passed": None}, {"full_l2_passed": False}, {"status": "unknown"},
])
def test_partial_mock_or_untrusted_readings_cannot_earn_native_performance(change):
    report = summarize([record(**change)])
    assert report["metrics"]["native_performance_target"]["count"] == 0
    assert report["native_performance_observed"] is False


@pytest.mark.parametrize("rows", [
    [record(), record()], [record(case_id="not-declared")], [record(attempt=2)],
    [record(attempt=True)], [record(model_calls=-1)], [record(tokens=True)],
    [record(wall_seconds=float("inf"))], [record(full_l2_passed="true")],
    [record(execution_kind="gpu-ish")],
])
def test_invalid_attempts_are_rejected_instead_of_silently_dropped(rows):
    with pytest.raises(ValueError):
        summarize(rows)


def test_unknown_costs_are_not_reported_as_zero_cost_successes():
    report = summarize([record(tokens=None)])
    assert report["costs"]["tokens"]["missing_records"] == 1
    assert report["costs"]["tokens"]["per_correct_task"] is None


def test_unmeasured_native_performance_is_explicit_not_a_zero_success_rate():
    report = summarize([record(execution_kind="replay")])
    metric = report["metrics"]["native_performance_target"]
    assert metric == dict(count=0, denominator=10, rate=None,
                          measured_tasks=0, state="not_measured")
    assert compare_reports(report, report)["metrics"]["native_performance_target"]["rate_delta"] is None


def test_valid_native_performance_failure_is_still_an_observation():
    report = summarize([record(status="failed", performance_target_passed=False)])
    assert report["native_performance_observed"] is True
    assert report["metrics"]["native_performance_target"]["measured_tasks"] == 1


def test_performance_failure_does_not_erase_independent_full_l2_evidence():
    report = summarize([record(status="failed", performance_target_passed=False)])
    assert report["metrics"]["first_full_l2"]["count"] == 1
    assert report["metrics"]["within_budget_full_l2"]["count"] == 1
    assert report["metrics"]["native_performance_target"]["count"] == 0


def test_explicit_subsets_get_their_own_identity_and_denominator():
    selected = suite_manifest([cases()[0].case_id])
    report = summarize([record()], manifest=selected)
    assert report["metrics"]["first_full_l2"]["rate"] == 1
    assert selected["hash"] != suite_manifest()["hash"]
    with pytest.raises(ValueError):
        suite_manifest([])
    with pytest.raises(ValueError):
        suite_manifest([cases()[0].case_id] * 2)


def test_model_and_strategy_can_vary_while_fixed_experiment_controls_match():
    a = summarize([record()], controls=controls())
    b = summarize([record()], controls=controls(model="model-b", strategy="diagnosed"))
    comparison = compare_reports(a, b)
    assert comparison["comparable"] is True
    assert set(comparison["varied_controls"]) == {"model", "strategy"}
    assert comparison["metrics"]["first_full_l2"]["rate_delta"] == 0


@pytest.mark.parametrize("change", [{"budgets": {"calls": 4}}, {"hardware": "other"},
                                    {"protocol": "other-ruler"}, {"reference": "other"}])
def test_budget_hardware_reference_and_protocol_drift_refuse_comparison(change):
    with pytest.raises(ValueError, match="identical suite and fixed controls"):
        compare_reports(summarize([record()], controls=controls()),
                        summarize([record()], controls=controls(**change)))


def test_rehashed_manifest_and_edited_report_cannot_change_the_question():
    manifest = suite_manifest()
    manifest["cases"][0]["precision"] = "fp32"
    manifest["hash"] = digest({key: value for key, value in manifest.items() if key != "hash"})
    with pytest.raises(ValueError, match="definition drift"):
        summarize([], manifest=manifest)
    report = summarize([record()], controls=controls())
    changed = copy.deepcopy(report)
    changed["metrics"]["first_full_l2"]["count"] = 10
    with pytest.raises(ValueError, match="retained evidence"):
        compare_reports(report, changed)


def test_arbitrary_case_objects_cannot_replace_fixed_tasks():
    with pytest.raises(ValueError, match="declaration drift"):
        task_for(replace(cases()[0], seed=999))
    assert input_cases_for(cases()[0])[0].inputs.seed == cases()[0].seed
