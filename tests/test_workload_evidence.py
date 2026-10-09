"""Deterministic case evidence tests, not actual GPU performance claims."""
import contextlib
import copy
import io
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from operator_agent import Conductor, KernelDatabase
from operator_agent.backend_caps import Cap
from operator_agent.confirm import ConfirmRecord, confirm_metrics, confirm_persist
from operator_agent.contract import case_key
from operator_agent.eligibility import aggregate_speedup, decision_speedup, performance_verdict
from operator_agent.kernel_db import FailureDatabase
from operator_agent.oracle import admit_to_corpus
from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec, TargetSpec,
                                    TaskSchema, UnifiedSketch)
from operator_agent.verifier import Verifier
from operator_agent.workload import confirmation_evidence
from tests.backend_fakes import DeclaredFakeBackend

CODE = "def kernel(A, B):\n    return A @ B\n"
A, B, C = "A=16x16|B=16x16", "A=32x32|B=32x32", "A=64x64|B=64x64"


def inputs(n):
    return {k: np.ones((n, n), dtype=np.float32) for k in ("A", "B")}


def task(**overrides):
    cons = dict(precision=Precision.FP32, min_speedup=1.05, max_error=1e-3,
                required_cases=(A, B), case_weights={A: 1., B: 1.},
                aggregate="mean", max_regression={A: .1, B: .1},
                shapes=((16, 16, 16), (32, 32, 32)))
    cons.update(overrides)
    return TaskSchema(task_id="workload-local-fixture", source=SourceSpec(kernel_type="matmul"),
        target=TargetSpec(hardware="nvidia_gpu", dsl="python", required_caps=(Cap.REAL_DEVICE,)),
        constraints=ConstraintSpec(**cons))


def backend(change=None):
    calls = []
    def bench(code, x, *args, **kwargs):
        n = x["A"].shape[0]
        calls.append(n)
        window = "fixture-window-" + str(len(calls))
        identity = {"host": "fixture", "device": "cuda:0", "visible_index": 0, "name": "NVIDIA H20", "cuda_visible_devices": "0"}
        p = dict(success=True, mode="real", device="cuda:0", gflops=1., gpu_util_pct=0,
            candidate=dict(median_time_ms=.2, min_time_ms=.2, avg_time_ms=.2,
                max_time_ms=.2, runs=30, dispersion_iqr_over_median=0),
            baselines={"pytorch_eager": dict(median_time_ms=.4)}, baseline_selected="pytorch_eager",
            paired=dict(arm_identity={"candidate": identity, "baseline": identity},
                        arm_windows={"candidate": window, "baseline": window}))
        if change:
            change(p, n, len(calls))
        return p
    b = DeclaredFakeBackend(caps={Cap.REAL_DEVICE}, benchmark_with_baselines=bench,
        get_hardware_info=lambda: dict(name="NVIDIA H20", device="cuda:0", compute_capability="9.0"))
    return b, calls


def measure(change=None, t=None, ns=(16, 32)):
    b, calls = backend(change)
    with contextlib.redirect_stdout(io.StringIO()):
        r = Verifier(b)._verify_performance(CODE, [inputs(n) for n in ns], 0., t or task(),
                                           static_guards=True, correctness_verified=True)
    return r, calls


def run(tmp_path, change=None, t=None, ns=(16, 32)):
    b, calls = backend(change)
    c = Conductor(backend=b, kernel_db=KernelDatabase(str(tmp_path / "corpus.jsonl"), persist_retrievals=False),
        failure_db=FailureDatabase(str(tmp_path / "failures.jsonl")),
        max_optimize_rounds=1, use_coder_pipeline=False)
    c.store.output_dir = tmp_path / "artifacts"
    c.designer.run = lambda *a, **k: UnifiedSketch(algorithm="matmul", target_dsl="python")
    c.coder = SimpleNamespace(run=lambda *a, **k: CODE)
    with contextlib.redirect_stdout(io.StringIO()):
        art = c.execute(t or task(), [inputs(n) for n in ns], target_gflops=0.)
    return art, c, calls


def test_all_cases_use_full_oracle_and_only_two_measurements():
    r, calls = measure()
    assert r.passed and calls == [16, 32]
    for key, row in r.metrics["cases"].items():
        assert row["observation"]["eligibility"] == row["eligibility"]
        assert row["measurement_validity"] == "valid"
        assert row["eligibility"]["can_persist"] is True
        assert row["contract_hash"] == r.metrics["contract_hash"]
    assert r.metrics["observation"]["eligibility"] == r.metrics["eligibility"]


@pytest.mark.parametrize("bad", ("load", "noise", "floor", "cross_window", "cross_device", "missing", "contract"))
def test_bad_secondary_cannot_inherit_primary_qualification(bad):
    def change(p, n, count):
        if n != 32:
            return
        if bad == "load": p["gpu_util_pct"] = 80
        elif bad == "noise": p["candidate"]["dispersion_iqr_over_median"] = .2
        elif bad == "floor": p["candidate"]["median_time_ms"] = .001
        elif bad == "cross_window": p["paired"]["arm_windows"]["baseline"] = "other"
        elif bad == "cross_device": p["paired"]["arm_identity"]["baseline"] = dict(visible_index=0, name="NVIDIA H20", cuda_visible_devices="1")
        elif bad == "missing": p.update(success=False, error="missing fixture observation")
        else: p["contract_hash"] = "wrong-contract"
    r, calls = measure(change)
    m = r.metrics
    assert m["cases"][A]["eligibility"]["can_persist"] is True
    assert m["eligibility"]["can_persist"] is False
    if bad != "missing":
        assert m["cases"][B]["speedup_vs_baseline"] is not None  # raw diagnostics retained
    assert calls == [16, 32]
    if bad in ("floor", "cross_window", "cross_device", "missing", "contract"):
        assert not r.passed
        assert m["performance_verdict"]["value"] is None
    else:
        assert m["measurement_validity"] == "degraded"


def test_required_missing_case_blocks_before_any_corpus_write(tmp_path):
    art, c, calls = run(tmp_path, t=task(case_weights={A: 1.}, max_regression={A: .1}), ns=(16,))
    assert not art.all_passed
    assert calls == [16]
    assert not (tmp_path / "corpus.jsonl").exists()
    assert art.benchmark_data["implementation_decision"]["decision"] == "inconclusive"


def test_objective_can_pass_with_primary_below_threshold_and_persist(tmp_path):
    def change(p, n, count):
        if n == 16: p["baselines"]["pytorch_eager"]["median_time_ms"] = .18
    art, c, calls = run(tmp_path, change, task(case_weights={A: .1, B: 1.}, max_regression={A: .2, B: .1}))
    m = art.benchmark_data
    assert m["speedup_vs_baseline"] == .9
    assert decision_speedup(m) > 1.05
    assert m["implementation_decision"]["decision"] == "use_custom"
    assert art.metadata["corpus_admission"]["kept"] is True
    assert calls == [16, 32, 16, 32]
    event = json.loads((tmp_path / "corpus.jsonl").read_text().splitlines()[0])
    assert event["record"]["score"] == decision_speedup(m)
    assert event["admission"]["event_hash"]
    assert confirm_metrics(m, art.remeasure, 1.05)[0]
    before = (tmp_path / "corpus.jsonl").read_bytes()
    recovered = subprocess.run([sys.executable, "-m", "operator_agent.corpus_admission",
        "--artifact", art.saved_dir, "--corpus", str(tmp_path / "corpus.jsonl")],
        capture_output=True, text=True, timeout=30)
    assert recovered.returncode == 0, recovered.stdout + recovered.stderr
    assert json.loads(recovered.stdout)["status"] == "already_committed"
    assert (tmp_path / "corpus.jsonl").read_bytes() == before


def test_secondary_confirmation_regression_blocks_commit_even_if_primary_stays_fast(tmp_path):
    def change(p, n, count):
        if n == 32 and count > 2: p["baselines"]["pytorch_eager"]["median_time_ms"] = .02
    art, c, calls = run(tmp_path, change)
    assert art.all_passed  # initial full workload passed
    assert art.remeasure.speedup == 1.05
    assert not confirm_metrics(art.benchmark_data, art.remeasure, 1.05)[0]
    assert not (tmp_path / "corpus.jsonl").exists()
    assert art.benchmark_data["implementation_decision"]["decision"] == "inconclusive"


def test_measured_regression_gets_library_decision_on_failure_path(tmp_path):
    def change(p, n, count):
        if n == 32: p["baselines"]["pytorch_eager"]["median_time_ms"] = .1
    art, c, calls = run(tmp_path, change)
    m = art.benchmark_data
    assert not art.all_passed and calls == [16, 32]
    assert m["performance_verdict"]["value"] > 1.05
    assert m["implementation_decision"]["decision"] == "use_library"
    assert not (tmp_path / "corpus.jsonl").exists()
    saved = json.loads((Path(art.saved_dir) / "benchmark.json").read_text())
    assert saved["implementation_decision"] == m["implementation_decision"]


@pytest.mark.parametrize("mutation", ("window", "scope", "mask", "summary", "value", "basis", "missing_case",
                                     "row_value_vs_oracle", "oracle_ruler_broken", "missing_window_and_device"))
def test_confirmation_drift_and_reused_windows_hold(tmp_path, mutation):
    art, _, _ = run(tmp_path)
    m = art.benchmark_data
    rm = copy.deepcopy(art.remeasure)
    evidence = rm.workload_evidence
    if mutation == "window": evidence["cases"][B]["arm_windows"] = copy.deepcopy(m["cases"][B]["arm_windows"])
    elif mutation == "scope": evidence["workload_qualification"]["scope_cases"] = [A]
    elif mutation == "mask": evidence["cases"][B]["eligibility"]["can_verdict"] = False
    elif mutation == "summary": evidence["workload_qualification"]["validity"] = "degraded"
    elif mutation == "value": evidence["performance_verdict"]["value"] = 100.
    elif mutation == "basis": rm.performance_basis = "primary_case"
    elif mutation == "row_value_vs_oracle":
        evidence["cases"][B]["speedup_vs_baseline"] = 10.
        evidence["performance_verdict"]["value"] = 6.
        rm.speedup = 6.
    elif mutation == "oracle_ruler_broken": evidence["cases"][B]["observation"]["ruler_broken"] = True
    elif mutation == "missing_window_and_device":
        evidence["cases"][B].pop("arm_windows")
        evidence["cases"][B].pop("backend_real_device")
    else: del evidence["cases"][B]
    assert not confirm_metrics(m, rm, 1.05)[0]
    assert not admit_to_corpus(m, rm, min_speedup=1.05)[0]


def test_two_self_reported_narrow_criteria_do_not_override_frozen_contract(tmp_path):
    art, _, _ = run(tmp_path)
    m, rm = copy.deepcopy(art.benchmark_data), copy.deepcopy(art.remeasure)
    for metrics in (m, rm.workload_evidence):
        metrics["workload_qualification"]["scope_cases"] = [A]
        for k in ("case_weights", "max_regression"):
            del metrics["performance_verdict"][k][B]
        metrics["performance_verdict"]["required_cases"] = [A]
    assert not confirm_metrics(m, rm, 1.05)[0]


def test_non_last_real_case_cannot_borrow_last_case_window_exemption(tmp_path):
    t = task()
    t.target.required_caps = ()
    art, _, _ = run(tmp_path, t=t)
    m, rm = copy.deepcopy(art.benchmark_data), copy.deepcopy(art.remeasure)
    for evidence in (m, rm.workload_evidence):
        evidence["cases"][B]["backend_real_device"] = False
        evidence["cases"][B]["observation"]["provenance"]["rule_inputs"]["real_device"] = False
    rm.workload_evidence["cases"][A].pop("arm_windows")
    rm.workload_evidence["cases"][A].pop("backend_real_device")
    assert not confirm_metrics(m, rm, 1.05)[0]
    assert not admit_to_corpus(m, rm, min_speedup=1.05)[0]


@pytest.mark.parametrize("mode", ("fresh", "reused", "missing"))
def test_declared_single_case_requires_fresh_performance_windows(tmp_path, mode):
    t = task(required_cases=(A,), case_weights={A: 1.}, max_regression={A: .1}, shapes=((16, 16, 16),))
    art, _, _ = run(tmp_path, t=t, ns=(16,))
    m, rm = art.benchmark_data, copy.deepcopy(art.remeasure)
    if mode == "reused": rm.workload_evidence["cases"][A]["arm_windows"] = copy.deepcopy(m["cases"][A]["arm_windows"])
    elif mode == "missing": rm.workload_evidence["cases"][A].pop("arm_windows")
    assert confirm_metrics(m, rm, 1.05)[0] is (mode == "fresh")


def test_non_aggregate_failed_gate_never_suggests_use_custom(tmp_path):
    t = task(case_weights={}, aggregate="", min_speedup=2.5)
    art, _, _ = run(tmp_path, t=t)
    assert not art.all_passed
    assert art.benchmark_data["implementation_decision"]["decision"] == "inconclusive"
    assert not (tmp_path / "corpus.jsonl").exists()


def test_extra_diagnostic_case_cannot_change_declared_scope():
    def change(p, n, count):
        if n == 32: p["gpu_util_pct"] = 80
    t = task(required_cases=(A,), case_weights={A: 1.}, max_regression={A: .1}, shapes=())
    r, _ = measure(change, t)
    assert r.passed and r.metrics["eligibility"]["can_persist"] is True
    assert r.metrics["cases"][B]["eligibility"]["can_persist"] is False


@pytest.mark.parametrize("value", (math.nan, math.inf, -math.inf, True))
def test_nonfinite_cannot_be_aggregated_or_confirmed(value):
    assert aggregate_speedup({A: 1., B: 1.}, {A: 2., B: value}, "mean")[0] is None
    assert not confirm_persist(2., "valid", ConfirmRecord(True, value, "valid"), 1.05)[0]


def test_declared_hard_regression_is_not_ignored_without_aggregate():
    assert performance_verdict(1.05, primary_speedup=2., max_regression={B: .1},
                               regression_over_limit_value=True)[0] is False
    assert performance_verdict(1.05, primary_speedup=2., max_regression={B: .1},
                               regression_over_limit_value=None)[0] is False


def test_missing_baseline_does_not_turn_objective_into_gflops_success():
    b = DeclaredFakeBackend(caps={Cap.REAL_DEVICE},
        benchmark=lambda *a, **k: dict(avg_time_ms=.2, median_time_ms=.2, gflops=1., memory_bandwidth_gbs=1.),
        get_hardware_info=lambda: dict(name="NVIDIA H20", device="cuda:0"))
    v = Verifier(b)
    v._measure_baseline = lambda *a: None
    with contextlib.redirect_stdout(io.StringIO()):
        r = v._verify_performance(CODE, [inputs(16), inputs(32)], 0., task())
    assert not r.passed
    assert r.metrics["performance_verdict"]["basis"] == "objective"
    assert r.metrics["performance_verdict"]["value"] is None
