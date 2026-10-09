"""Offline protocol/driver fixtures ONLY: no H20, SSH, upload or GPU timings.

Fabricated real-device fields below test auditors, not hardware performance.
The weight-drift test uses the real Conductor's sealed resume preflight.
"""
import copy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import workload_decision_h20 as driver
from operator_agent import Conductor, KernelDatabase, MockGPU
from operator_agent.artifact_store import ArtifactStore
from operator_agent.confirm import ConfirmRecord, confirm_metrics
from operator_agent.contract import freeze_from_task
from operator_agent.corpus_journal import confirmed_key
from operator_agent.eligibility import ALL_FALSE, ALL_TRUE, DEGRADED, case_speedups, decision_speedup
from operator_agent.kernel_db import KernelRecord
from operator_agent.reference_contract import check_observed, registered_reference
from operator_agent import run_contract as rc
from operator_agent.schemas import VerificationLevel as Level, VerificationResult
from operator_agent.workload import apply_qualification, confirmation_evidence
from operator_agent.verifier import Verifier


@pytest.fixture
def declared():
    return driver.tasks()


def args(tmp_path, execute=False):
    argv = ["--output", str(tmp_path)]
    if execute:
        argv += ["--execute", "--gpu", "2"]
    return driver.parser().parse_args(argv)


def test_fixed_plan_freezes_all_identities_and_criteria_without_allocating(declared, tmp_path):
    code = driver.trial_code("unfused-fp32")
    p = driver.workload_plan(args(tmp_path), declared, code)
    assert p["hash"] == rc.digest({k:v for k,v in p.items() if k != "hash"})
    assert p["candidate"] == dict(variant="unfused-fp32", source=code, hash=rc.code_hash(code))
    assert p["reference"]["hash"] == registered_reference().digest()
    assert p["measurement_protocol"] == rc.protocol_identity()
    assert "operator_agent/workload.py" in p["measurement_protocol"]
    assert p["evaluator_sources"]["operator_agent/workload.py"] == rc.code_hash((driver.ROOT / "operator_agent/workload.py").read_text())
    assert [t["name"] for t in p["trials"]] == list(driver.TRIAL_NAMES)
    assert p["trials"][0]["supplied_cases"] == ["X=8192x15360", "X=8192x12288"]
    assert p["trials"][1]["supplied_cases"] == ["X=8192x15360", "X=8192x12288", "X=32x64"]
    assert p["trials"][2]["supplied_cases"] == ["X=8192x15360"]
    assert "X=8192x12288" in p["trials"][2]["contract"]["required_cases"]
    assert p["trials"][3]["task_id"] == p["trials"][0]["task_id"]
    first, drift = [p["trials"][i]["contract"] for i in (0, 3)]
    assert first["hash"] != drift["hash"]
    assert {k:v for k,v in first.items() if k not in {"hash", "case_weights"}} == {
        k:v for k,v in drift.items() if k not in {"hash", "case_weights"}}
    assert p["adaptive"] is False and p["retry_windows"] is False and p["sequential_cases"] is True
    assert [r["input_storage_bytes"] for r in p["input_bounds"]] == [240*1024**2, 192*1024**2, 4096]
    assert all(r["storage_plus_scalars_bytes"] < 256*1024**2 for r in p["input_bounds"])
    for row in p["trials"]:
        c = row["contract"]
        assert c["aggregate"] == "weighted_geomean" and c["min_speedup"] == 1.05
        assert set(dict(c["case_weights"])) == set(c["required_cases"]) == set(dict(c["max_regression"]))
        assert set(dict(c["max_regression"]).values()) == {.05}
    assert p["controls"]["rpc_timeout"] == 180 and p["controls"]["max_rpcs"] == 512
    assert p["controls"]["max_optimize_rounds"] == 1 and p["controls"]["repeat"] == 30


def test_default_local_plan_has_zero_transport_and_no_large_input_allocations(tmp_path, monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError("local plan attempted transport/allocation")
    monkeypatch.setattr(driver, "BoundedProfileSSH", forbidden)
    monkeypatch.setattr(driver, "primary", forbidden)
    assert driver.run(args(tmp_path)) == 0
    root = next(tmp_path.iterdir())
    assert root.stat().st_mode & 0o777 == 0o700
    assert (root / "plan.json").stat().st_mode & 0o777 == 0o444
    s = json.loads((root / "summary.json").read_text())
    assert s["mode"] == "plan-only" and not s["completed"]
    assert not s["functional_completed"] and not s["performance_completed"] and s["trials"] == []
    assert s["regression_fixture"]["source"] == "local_fixture_only"
    assert s["regression_fixture"]["machine_timings"] is None
    assert driver.run(args(tmp_path)) == 0
    assert len(list(tmp_path.iterdir())) == 2  # Random owned paths, no overwriting.


def test_execute_without_explicit_gpu_fails_before_any_side_effect(tmp_path):
    a = args(tmp_path); a.execute = True
    with pytest.raises(ValueError, match="--gpu"):
        driver.run(a)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("argv", [["--workload", "8192x16384"], ["--gpu", "8"],
    ["--min-speedup", ".01"], ["--repeat", "1"], ["--max-regression", "1"]])
def test_cli_cannot_adapt_workloads_or_measurement_gates(argv):
    with pytest.raises(SystemExit):
        driver.parser().parse_args(argv)


def test_task_factory_rejects_unbounded_or_different_selection():
    with pytest.raises(ValueError):
        driver.task("bad", ((8192, 16384),))
    with pytest.raises(ValueError):
        driver.task("bad", tuple(reversed(driver.LARGE)))


@pytest.mark.parametrize("variant", ["plan", "protocol", "extra-sources"])
def test_plan_drift_is_checked_before_dispatch(declared, tmp_path, monkeypatch, variant):
    p = driver.workload_plan(args(tmp_path), declared, driver.trial_code("unfused-fp32"))
    path = tmp_path / "plan.json"; rc.atomic_json(path, p)
    driver.check_plan(path, p)
    if variant == "plan":
        changed = copy.deepcopy(p); changed["controls"]["repeat"] = 1
        path.write_text(json.dumps(changed))
    elif variant == "protocol":
        monkeypatch.setattr(driver, "protocol_identity", lambda: {"wrong":"protocol"})
    else:
        monkeypatch.setattr(driver, "evaluator_sources", lambda: {"wrong":"sources"})
    with pytest.raises(ValueError, match="drift"):
        driver.check_plan(path, p)


def test_bounded_dispatch_and_independent_cleanup_allowance(monkeypatch):
    monkeypatch.setattr(driver.time, "monotonic", lambda: 100.0)
    b = driver.BoundedProfileSSH(host="offline-do-not-connect", timeout=180)
    assert b.deadline == 100 + driver.DISPATCH_SECONDS
    b.counts["rpc"] = driver.MAX_RPCS
    with pytest.raises(TimeoutError): b.guard("rpc")
    b.counts["command"] = driver.MAX_COMMANDS
    with pytest.raises(TimeoutError): b.guard("command")
    b.counts = dict(rpc=0, command=0); b.deadline = 99
    with pytest.raises(TimeoutError): b.guard("rpc")
    # Substitute only the parent cleanup; no SSH method is called in this test.
    monkeypatch.setattr(driver.ProfileSSH, "cleanup", lambda self: dict(quiescent=True, removed=True))
    assert b.cleanup()["quiescent"]
    assert b.cleaning and b.deadline == 160
    b.guard("rpc")
    assert b.counts["rpc"] == 1 and b.timeout == 60
    assert b.work_counts == dict(rpc=0, command=0)


def test_large_case_eviction_may_reupload_but_requires_exact_bounded_proof():
    packet = dict(protocol=driver.INPUT_PROTOCOL, hash="a"*64,
        storages=[dict(sha256="b"*64, bytes=240*1024**2)], tensors={}, scalars={})
    proof = dict(protocol=driver.INPUT_PROTOCOL, manifest_hash=packet["hash"],
        storage_bytes=240*1024**2, uploaded_bytes=240*1024**2, content_verified=True)
    trace = [dict(request=dict(action="benchmark_with_baselines", inputs=packet), response=dict(input_transfer=proof))]
    assert driver.transfer_errors(trace) == []
    proof["uploaded_bytes"] = 0
    assert driver.transfer_errors(trace) == []
    for field, value in (("uploaded_bytes", True), ("uploaded_bytes", -1), ("uploaded_bytes", 256*1024**2),
                         ("content_verified", False), ("manifest_hash", "wrong")):
        changed = copy.deepcopy(trace); changed[0]["response"]["input_transfer"][field] = value
        assert driver.transfer_errors(changed)
    packet["storages"][0]["bytes"] = 256*1024**2
    assert driver.transfer_errors(trace)


def _metrics(t, window, speeds=(2.0, 1.5), tiny_tier=None, missing=False):
    """FABRICATED OFFLINE Oracle/RPC-shaped data. Never passed to SSH."""
    contract = freeze_from_task(t)
    ordered = list(t.constraints.required_cases)
    if missing:
        ordered = ordered[:1]
    ledger = {}
    ident = dict(visible_index=0, name="offline-fixture-H20", count=1, cuda_visible_devices="2")
    receipt = check_observed(contract.reference_contract, contract.reference_contract.execution.content_dict())
    for index, key in enumerate(ordered):
        tier = tiny_tier if key == "X=32x64" and tiny_tier else "valid"
        mask = (ALL_FALSE if tier == "uninformative" else DEGRADED if tier == "degraded" else ALL_TRUE).as_dict()
        reasons = ["launch 地板占实测时间 80%"] if tier != "valid" else []
        w = window + "-" + str(index)
        value = speeds[index] if index < len(speeds) else 2.0
        ledger[key] = dict(speedup_vs_baseline=value, backend_real_device=True, correctness_verified=True,
            contract_hash=contract.hash, eligibility=mask, measurement_validity=tier,
            measurement_validity_reasons=reasons,
            arm_windows=dict(candidate=w, baseline=w), arm_identity=dict(candidate=ident, baseline=ident),
            reference_receipt=receipt, observation=dict(metric="speedup_vs_baseline", value=value, unit="x",
                validity=tier, eligibility=copy.deepcopy(mask), reasons=reasons,
                provenance=dict(contract_hash=contract.hash), refused="", ruler_broken=False))
    m = copy.deepcopy(ledger[ordered[0]])
    m["cases"] = ledger
    apply_qualification(m, contract, ordered[0])
    speeds_map = case_speedups(m)
    over, _ = driver.regression_over_limit(contract.max_regression, speeds_map)
    passed, basis, reason, value = driver.performance_verdict(contract.min_speedup, aggregate=contract.aggregate,
        case_weights=contract.case_weights, case_speedups=speeds_map,
        max_regression=contract.max_regression, regression_over_limit_value=over)
    m["performance_verdict"] = dict(basis=basis, passed=passed, value=value, why=reason,
        min_speedup=contract.min_speedup, aggregate=contract.aggregate, case_weights=dict(contract.case_weights),
        max_regression=dict(contract.max_regression), required_cases=list(contract.required_cases), regression_over_limit=over)
    return m


@pytest.fixture
def evidence(declared, tmp_path):
    t = declared[0]; contract = freeze_from_task(t)
    code = driver.trial_code("unfused-fp32")
    first = _metrics(t, "first", (1.04, 2.0))  # Primary below threshold; objective above!
    second = _metrics(t, "second", (1.03, 1.8))
    binding = dict(run_id="a"*32, manifest_hash="b"*64, contract_hash=contract.hash,
        workload_hash="d"*64, protocol_hash="e"*64, candidate_hash=rc.code_hash(code), verification_id="first-verification")
    first["run_binding"] = binding
    second["run_binding"] = dict(binding, verification_id="second-verification")
    record = ConfirmRecord(ok=True, speedup=decision_speedup(second), validity="valid", independent=True,
        contract_hash=contract.hash, performance_basis="objective", workload_evidence=confirmation_evidence(second))
    directory = tmp_path / "artifact"; directory.mkdir()
    artifact = SimpleNamespace(kernel_id="OFFLINE-fixture-not-H20-evidence", code=code,
        compile_info=dict(run_binding=copy.deepcopy(binding)), saved_dir=str(directory), remeasure=record)
    cf = dict(first_binding=copy.deepcopy(binding), second_binding=copy.deepcopy(second["run_binding"]),
              confirmation=asdict(record), second_metrics=second)
    (directory / "confirmation.json").write_text(json.dumps(cf))
    stored = KernelRecord(kernel_id=artifact.kernel_id, algorithm=t.source.kernel_type, hardware=t.target.hardware,
        dsl=t.target.dsl, code=code, compile_info=copy.deepcopy(artifact.compile_info),
        score=decision_speedup(first), measurement_validity="valid")
    return SimpleNamespace(t=t, contract=contract, supplied=list(t.constraints.required_cases),
        first=first, second=second, cf=cf, artifact=artifact, record=record, stored=stored, path=tmp_path / "corpus.jsonl")


def test_positive_full_case_evidence_uses_aggregate_not_primary(evidence):
    f = evidence
    assert f.first["speedup_vs_baseline"] < 1.05 < decision_speedup(f.first)
    assert driver.metrics_errors(f.first, f.contract, f.supplied, positive=True) == []
    assert math.isclose(decision_speedup(f.first), math.sqrt(1.04*2.0))
    second, errors = driver.confirmation_errors(f.artifact, f.first, f.contract, f.supplied)
    assert errors == [] and second == rc.json_value(f.second)
    assert confirm_metrics(f.first, f.record, min_speedup=1.05)[0]
    assert f.record.speedup != f.second["speedup_vs_baseline"]
    assert rc.json_value(f.record.workload_evidence["workload_contract"]) == rc.json_value(f.contract.as_dict())


@pytest.mark.parametrize("variant", ["no-observation", "no-provenance", "mask-missing", "mask-bool-like", "obs-mask",
    "wrong-validity", "wrong-contract", "not-real", "no-correctness", "bad-receipt", "no-windows",
    "cross-arm", "cross-device", "reused-case-window", "speedup-bool", "speedup-nan", "ledger-missing",
    "wrong-scope", "primary-only-mask", "basis", "aggregate", "weights", "required", "regression",
    "wrong-aggregate", "no-aggregate", "wrong-threshold", "wrong-verdict", "wrong-regression-verdict", "workload-contract"])
def test_every_positive_case_requires_complete_qualifications_not_a_primary_scalar(evidence, variant):
    f = evidence; m = copy.deepcopy(f.first); key = f.supplied[1]; row = m["cases"][key]
    if variant == "no-observation": row.pop("observation")
    elif variant == "no-provenance": row["observation"].pop("provenance")
    elif variant == "mask-missing": row["eligibility"].pop("can_report")
    elif variant == "mask-bool-like": row["eligibility"]["can_report"] = 1
    elif variant == "obs-mask": row["observation"]["eligibility"] = ALL_FALSE.as_dict()
    elif variant == "wrong-validity": row["measurement_validity"] = "uninformative"
    elif variant == "wrong-contract": row["contract_hash"] = "other"
    elif variant == "not-real": row["backend_real_device"] = False
    elif variant == "no-correctness": row.pop("correctness_verified")
    elif variant == "bad-receipt": row["reference_receipt"] = {"status":"passed"}
    elif variant == "no-windows": row.pop("arm_windows")
    elif variant == "cross-arm": row["arm_windows"]["baseline"] = "other-window"
    elif variant == "cross-device": row["arm_identity"]["baseline"] = dict(row["arm_identity"]["baseline"], visible_index=1)
    elif variant == "reused-case-window": row["arm_windows"] = copy.deepcopy(m["cases"][f.supplied[0]]["arm_windows"])
    elif variant == "speedup-bool": row["speedup_vs_baseline"] = True
    elif variant == "speedup-nan": row["speedup_vs_baseline"] = float("nan")
    elif variant == "ledger-missing": m["cases"].pop(key)
    elif variant == "wrong-scope": m["workload_qualification"]["scope_cases"] = [key]
    elif variant == "primary-only-mask": m["workload_qualification"]["eligibility"] = ALL_FALSE.as_dict()
    elif variant == "workload-contract": m.pop("workload_contract")
    else:
        field, value = {"basis":("basis","primary_case"), "aggregate":("aggregate","mean"),
            "weights":("case_weights", {key:2.0}), "required":("required_cases",[key]),
            "regression":("max_regression",{}), "wrong-aggregate":("value",9.0), "no-aggregate":("value",None),
            "wrong-threshold":("min_speedup",.01), "wrong-verdict":("passed",False),
            "wrong-regression-verdict":("regression_over_limit",True)}[variant]
        m["performance_verdict"][field] = value
    assert driver.metrics_errors(m, f.contract, f.supplied, positive=True)


@pytest.mark.parametrize("tier", ["uninformative", "degraded"])
def test_tiny_is_required_and_weighted_and_lowers_whole_workload_without_changing_ruler(declared, tier):
    t = declared[1]; c = freeze_from_task(t)
    m = _metrics(t, "tiny-hold", tiny_tier=tier)
    assert driver.metrics_errors(m, c, list(t.constraints.required_cases)) == []
    assert m["workload_qualification"]["validity"] == tier
    assert m["eligibility"]["can_persist"] is False
    assert m["cases"]["X=32x64"]["measurement_validity_reasons"]
    assert m["cases"]["X=32x64"]["speedup_vs_baseline"] == 2.0  # Raw diagnostics survive.
    if tier == "uninformative":
        assert case_speedups(m)["X=32x64"] is None and decision_speedup(m) is None
        assert m["performance_verdict"]["passed"] is False
        assert driver.metrics_errors(m, c, list(t.constraints.required_cases), positive=True)
    else:
        assert decision_speedup(m) > 1.05  # Can-verdict but cannot-persist is still HOLD.


def test_missing_declared_case_remains_unknown_not_numerical_failure(declared):
    t = declared[2]; c = freeze_from_task(t); supplied = list(t.constraints.required_cases[:1])
    m = _metrics(t, "missing", missing=True)
    assert driver.metrics_errors(m, c, supplied) == []
    assert decision_speedup(m) is None and m["performance_verdict"]["passed"] is False
    assert m["workload_qualification"]["eligibility"] == ALL_FALSE.as_dict()
    assert len(m["cases"]) == 1 and set(m["cases"]) == {"X=8192x15360"}
    assert any("X=8192x12288" in r for r in m["workload_qualification"]["reasons"])
    assert "X=8192x12288" not in m["cases"]  # No fabricated failed numeric observation.
    m["workload_qualification"]["reasons"] = []
    assert driver.metrics_errors(m, c, supplied)


def test_actual_producer_publishes_one_complete_declared_case_and_missing_scope(declared):
    t = declared[2]; c = freeze_from_task(t); key = t.constraints.required_cases[0]
    raw = copy.deepcopy(_metrics(t, "one-case", missing=True)["cases"][key])
    original = copy.deepcopy(raw)
    v = Verifier(MockGPU())
    inputs = [{"X": SimpleNamespace(shape=driver.LARGE[0])}]  # Inert shape for publisher, never numeric RPC.
    assert v._publish_case_ledger(raw, "offline", inputs, t) == key
    assert raw["cases"] == {key: original} and "cases" not in raw["cases"][key]
    assert raw["workload_qualification"]["scope_cases"] == sorted(t.constraints.required_cases)
    assert v._apply_performance_verdict(True, raw, t, original["speedup_vs_baseline"], primary_key=key) is False
    assert driver.metrics_errors(raw, c, [key]) == []
    legacy = driver.IntentParser().parse("layernorm fp16 triton nvidia")
    bare = copy.deepcopy(original)
    v._publish_case_ledger(bare, "offline", inputs, legacy)
    assert "cases" not in bare and "workload_qualification" not in bare


@pytest.mark.parametrize("variant", ["missing", "same-verification", "other-workload", "record-speedup-primary",
    "basis", "evidence-missing", "evidence-drift", "second-ledger-missing", "second-refused",
    "second-mask", "criterion-drift", "same-case-window", "reused-other-case-window", "not-independent",
    "contract", "uninformative", "nan", "bool", "below-threshold", "full-contract-missing", "full-contract-drift"])
def test_confirmation_audits_full_second_metrics_and_all_new_windows(evidence, variant):
    f = evidence; cf = copy.deepcopy(f.cf); rec = f.record
    path = Path(f.artifact.saved_dir) / "confirmation.json"
    if variant == "missing": path.unlink()
    else:
        if variant == "same-verification": cf["second_binding"]["verification_id"] = cf["first_binding"]["verification_id"]
        elif variant == "other-workload": cf["second_binding"]["workload_hash"] = "f"*64
        elif variant == "record-speedup-primary": rec.speedup = cf["second_metrics"]["speedup_vs_baseline"]
        elif variant == "basis": rec.performance_basis = "primary_case"
        elif variant == "evidence-missing": rec.workload_evidence = {}
        elif variant == "evidence-drift": rec.workload_evidence["eligibility"] = ALL_FALSE.as_dict()
        elif variant == "second-ledger-missing": cf["second_metrics"]["cases"].pop(f.supplied[1])
        elif variant == "second-refused": cf["second_metrics"]["cases"][f.supplied[1]]["observation"]["refused"] = "fixture refusal"
        elif variant == "second-mask": cf["second_metrics"]["cases"][f.supplied[1]]["eligibility"] = ALL_FALSE.as_dict()
        elif variant == "criterion-drift": cf["second_metrics"]["performance_verdict"]["case_weights"][f.supplied[1]] = 2
        elif variant == "full-contract-missing": cf["second_metrics"].pop("workload_contract")
        elif variant == "full-contract-drift": cf["second_metrics"]["workload_contract"]["min_speedup"] = .01
        elif variant in {"same-case-window", "reused-other-case-window"}:
            from_key = f.supplied[0] if variant == "reused-other-case-window" else f.supplied[1]
            cf["second_metrics"]["cases"][f.supplied[1]]["arm_windows"] = copy.deepcopy(f.first["cases"][from_key]["arm_windows"])
        else:
            field, value = {"not-independent":("independent",False), "contract":("contract_hash","f"*64),
                "uninformative":("validity","uninformative"), "nan":("speedup",float("nan")),
                "bool":("speedup",True), "below-threshold":("speedup",1.01)}[variant]
            setattr(rec, field, value)
        # Mutations of the second metrics also update the compact receipt: the
        # auditor must catch genuine bad evidence, not just receipt drift.
        if variant.startswith(("second-", "full-contract-")) or variant in {"criterion-drift", "same-case-window", "reused-other-case-window"}:
            rec.workload_evidence = confirmation_evidence(cf["second_metrics"])
        cf["confirmation"] = asdict(rec)
        path.write_text(json.dumps(cf))
    assert driver.confirmation_errors(f.artifact, f.first, f.contract, f.supplied)[1]


def test_degraded_usable_confirmation_preserves_production_semantics(evidence):
    f = evidence
    for row in f.second["cases"].values():
        row.update(measurement_validity="degraded", eligibility=DEGRADED.as_dict())
        row["observation"].update(validity="degraded", eligibility=DEGRADED.as_dict())
    apply_qualification(f.second, f.contract, f.supplied[0])
    f.record.validity = "degraded"; f.record.workload_evidence = confirmation_evidence(f.second)
    f.cf.update(second_metrics=f.second, confirmation=asdict(f.record))
    (Path(f.artifact.saved_dir) / "confirmation.json").write_text(json.dumps(f.cf))
    assert driver.confirmation_errors(f.artifact, f.first, f.contract, f.supplied)[1] == []


def _trace(f, metrics_list):
    trace = []
    for m in metrics_list:
        for key in f.supplied:
            row = m["cases"][key]
            shape = [int(x) for x in key.split("=")[1].split("x")]
            req = dict(action="benchmark_with_baselines", code=f.artifact.code,
                inputs=dict(tensors={"X":dict(shape=shape)}))
            res = dict(success=True, device="cuda", selected_gpu=2, paired=dict(
                arm_windows=row["arm_windows"], arm_identity=row["arm_identity"]),
                candidate=dict(median_time_ms=1.0), baseline_selected="pytorch_eager",
                baselines={"pytorch_eager":dict(median_time_ms=row["speedup_vs_baseline"])})
            trace.append(dict(request=req, response=res))
    return copy.deepcopy(trace)


@pytest.mark.parametrize("variant", ["ok", "extra-window", "omitted-rpc", "rpc-order", "wrong-shape", "wrong-code",
    "wrong-gpu", "wrong-timing", "wrong-window", "not-cuda", "unsuccessful", "wrong-baseline"])
def test_case_evidence_must_match_all_actual_rpc_windows_not_selected_subset(evidence, variant):
    f = evidence; trace = _trace(f, [f.first, f.second])
    if variant == "extra-window": trace.append(copy.deepcopy(trace[0]))
    elif variant == "omitted-rpc": trace.pop()
    elif variant == "rpc-order": trace[0], trace[1] = trace[1], trace[0]
    elif variant == "wrong-shape": trace[1]["request"]["inputs"]["tensors"]["X"]["shape"] = [32,64]
    elif variant == "wrong-code": trace[1]["request"]["code"] += "\n# drift"
    elif variant == "wrong-gpu": trace[1]["response"]["selected_gpu"] = 0
    elif variant == "wrong-timing": trace[1]["response"]["candidate"]["median_time_ms"] = 99
    elif variant == "wrong-window": trace[1]["response"]["paired"]["arm_windows"] = dict(candidate="old",baseline="old")
    elif variant == "not-cuda": trace[1]["response"]["device"] = "cpu"
    elif variant == "unsuccessful": trace[1]["response"]["success"] = False
    elif variant == "wrong-baseline": trace[1]["response"]["baseline_selected"] = "some-fallback"
    errors = driver.trace_metrics_errors([f.first, f.second], trace, f.supplied, f.artifact.code, 2)
    assert bool(errors) is (variant != "ok")


def test_actual_sealed_corpus_commit_replays_exact_aggregate_not_primary(evidence):
    f = evidence; db = KernelDatabase(str(f.path), persist_retrievals=False)
    key = confirmed_key(f.first["run_binding"], f.second["run_binding"])
    assert db.commit(f.stored, key).persisted
    proof, errors = driver.corpus_readback(f.path, b"", f.artifact, f.t, f.first)
    assert errors == [] and proof["verified"] and proof["sealed"] and proof["fresh_replay"]
    assert proof["score"] == decision_speedup(f.first) != f.first["speedup_vs_baseline"]


@pytest.mark.parametrize("variant", ["missing", "unchanged", "no-seal", "wrong-seal", "wrong-event-hash",
    "primary-score", "wrong-code", "wrong-task", "wrong-validity", "bool-score", "wrong-binding",
    "bad-prefix", "rewrite-prefix", "extra-event"])
def test_admission_flag_and_file_growth_are_not_a_sealed_corpus_proof(evidence, variant):
    f = evidence; db = KernelDatabase(str(f.path), persist_retrievals=False)
    assert db.commit(f.stored, confirmed_key(f.first["run_binding"], f.second["run_binding"])).persisted
    original = f.path.read_bytes(); event = json.loads(original); before = b""
    if variant == "missing": f.path.unlink()
    elif variant == "unchanged": before = original
    else:
        if variant == "no-seal": event.pop("admission")
        elif variant == "wrong-seal": event["admission"]["id"] = "f"*64
        elif variant == "wrong-event-hash": event["admission"]["event_hash"] = "f"*64
        elif variant == "primary-score": event["record"]["score"] = f.first["speedup_vs_baseline"]
        elif variant == "wrong-code": event["record"]["code"] += "\n# not this candidate"
        elif variant == "wrong-task": event["record"]["algorithm"] = "relu"
        elif variant == "wrong-validity": event["record"]["measurement_validity"] = "degraded"
        elif variant == "bool-score": event["record"]["score"] = True
        elif variant == "wrong-binding": event["record"]["compile_info"]["run_binding"]["workload_hash"] = "f"*64
        elif variant == "bad-prefix": before = b"invalid historical data\n"
        elif variant == "rewrite-prefix": before = b"unrelated old data\n"
        content = json.dumps(event).encode() + b"\n"
        if variant == "bad-prefix": content = before + content
        if variant == "extra-event": content += b"{}\n"
        f.path.write_bytes(content)
    proof, errors = driver.corpus_readback(f.path, before, f.artifact, f.t, f.first)
    assert proof["verified"] is False and errors


def test_regression_over_limit_is_only_a_local_fixture_even_when_aggregate_passes(declared):
    r = driver.regression_fixture(declared[0])
    assert r["source"] == "local_fixture_only" and r["real_device"] is False and r["machine_timings"] is None
    assert r["aggregate"] > 1.05 and r["regression_over_limit"] is True
    assert r["performance_passed"] is False and r["implementation_decision"]["decision"] == "use_library"
    assert r["errors"] == []


def test_weight_drift_resume_is_real_production_preflight_with_zero_rpc(declared, tmp_path):
    original, changed = declared[0], declared[3]
    inputs = [driver.primary((4,8)), driver.primary((4,12))]  # Tiny local arrays; no large allocation.
    controls = dict(target_gflops=0.0, heartbeat_interval=5, cuda_calibration_identity_hash="",
                    cuda_ruler_profile_hash="", expected_output=None)
    environment = {"local_fixture":True}
    binding = rc.create(original, inputs, environment, controls)
    saved = tmp_path / "first"; saved.mkdir()
    rc.write(saved, binding)
    snap = dict(schema_version=1, task_id=original.task_id, round_done=1,
        current_code=driver.trial_code("unfused-fp32"), sketch_dict={}, pending_fix=None,
        environment_fingerprint=environment)
    rc.atomic_json(saved / "execution_snapshot.json", rc.seal(snap, binding))
    class NoRPC(MockGPU):
        def __init__(self): super().__init__(); self.calls = []
        def gpu_info(self): self.calls.append("gpu_info"); raise AssertionError("resume did RPC")
        def compile(self, *a, **kw): self.calls.append("compile"); raise AssertionError("resume did compile")
        def reference_identity(self, *a, **kw): self.calls.append("reference_identity"); raise AssertionError("resume did RPC")
    backend = NoRPC()
    db_path = tmp_path / "resume-corpus.jsonl"
    c = Conductor(backend=backend, kernel_db=KernelDatabase(str(db_path), persist_retrievals=False),
                  use_coder_pipeline=False, max_optimize_rounds=2)
    c.store = ArtifactStore(str(tmp_path / "resume-artifact"))
    def no_generation(*a, **kw): raise AssertionError("identity HOLD reached generation")
    c.designer.run = no_generation; c.coder = SimpleNamespace(run=no_generation)
    before = (saved / "execution_snapshot.json").read_bytes()
    art = c.execute(changed, inputs, target_gflops=0, resume_from=str(saved))
    l2 = next(r for r in art.verification_results if r.level == Level.L2_FUNCTION)
    assert l2.metrics["run_identity_unknown"] is True and l2.metrics["correctness"]["verdict"] == "unknown"
    assert not art.all_passed and art.remeasure is None and backend.calls == []
    assert not db_path.exists() and (saved / "execution_snapshot.json").read_bytes() == before


@pytest.mark.parametrize("cleanup", [dict(quiescent=False, removed=True), dict(quiescent=True, removed=False), {}])
def test_cleanup_failure_never_counts_as_acceptance(cleanup):
    assert driver.cleanup_errors(cleanup, SimpleNamespace(forbidden_actions=[]))


def test_forbidden_action_is_failure_even_when_cleanup_succeeds():
    assert driver.cleanup_errors(dict(quiescent=True,removed=True), SimpleNamespace(forbidden_actions=["deploy"]))


def test_failed_preparation_keeps_frozen_plan_lifecycle_and_honest_summary(tmp_path, monkeypatch):
    observations = []
    class OfflineBackend:
        def __init__(self, **kw):
            self.trace = []; self.forbidden_actions = []
            p = next(tmp_path.glob("*/plan.json")); observations.append(json.loads(p.read_text())["hash"])
            assert p.stat().st_mode & 0o777 == 0o444
        def prepare(self, gpu): raise RuntimeError("LOCAL injected preparation failure")
        def cleanup(self): observations.append("cleanup"); raise RuntimeError("LOCAL injected cleanup failure")
    monkeypatch.setattr(driver, "BoundedProfileSSH", OfflineBackend)
    old = {k:os.environ.get(k) for k in ("OPERATOR_AGENT_LOG_DIR", "OPERATOR_AGENT_LOG_KEEP")}
    assert driver.run(args(tmp_path, execute=True)) == 1
    s = json.loads(next(tmp_path.glob("*/summary.json")).read_text())
    assert observations == [s["plan_hash"], "cleanup"]
    assert not s["functional_completed"] and not s["performance_completed"] and not s["completed"]
    assert "cleanup failure" in s["cleanup"]["error"] and s["errors"]
    assert next(tmp_path.glob("*/lifecycle.json")).exists()
    assert {k:os.environ.get(k) for k in old} == old
def test_dispatch_contract_uses_json_normal_form_and_still_rejects_real_drift():
    import copy
    from benchmarks import workload_decision_h20 as driver
    from operator_agent.contract import freeze_from_task
    from operator_agent.run_contract import json_value
    for t in driver.tasks():
        expected = json_value(freeze_from_task(t).as_dict())
        driver.check_trial_contract(t, expected)
        changed = copy.deepcopy(t)
        key = next(iter(changed.constraints.case_weights))
        changed.constraints.case_weights[key] += .25
        with pytest.raises(ValueError, match="contract drift"):
            driver.check_trial_contract(changed, expected)


def test_input_selection_matches_frozen_missing_case_plan():
    arrays = [object(), object(), object()]
    assert driver.trial_inputs(1, arrays) == arrays
    assert driver.trial_inputs(2, arrays) == arrays[:1]
    for index in (0, 3):
        assert driver.trial_inputs(index, arrays) == arrays[:2]


def test_rpc_budget_accounts_for_two_rpc_reference_preflight_per_numerical_check():
    suites_alone = 3 * 32 * 2 * 2
    assert suites_alone < driver.MAX_RPCS <= 512


def test_negative_continuation_cannot_start_without_matching_identity_context(tmp_path):
    a = args(tmp_path)
    a.continue_negatives_from = str(tmp_path / "missing-artifact")
    with pytest.raises(FileNotFoundError):
        driver.run(a)
    assert not list(tmp_path.glob("*/plan.json"))
