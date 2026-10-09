"""Offline coverage of bounded H20 workload selection; never connects to SSH."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import durable_profile_h20 as driver
from benchmarks.durable_profile_h20 import WORKLOADS, parser, receipt_errors, selected_workloads, transfer_errors
from operator_agent.confirm import ConfirmRecord
from operator_agent.kernel_db import KernelDatabase, KernelRecord
from operator_agent.reference_contract import check_observed, registered_reference
from operator_agent.remote_input_fidelity import json_input_parts
from operator_agent.run_contract import code_hash, digest
from operator_agent.session_inputs import PROTOCOL


def test_original_default_and_resume_only_workloads_are_unchanged():
    assert selected_workloads(parser().parse_args([])) == WORKLOADS
    assert selected_workloads(parser().parse_args(["--resume-only"])) == WORKLOADS[:1]


def test_explicit_workloads_are_fixed_in_declared_order():
    args = parser().parse_args(["--workload", "1024x1024", "--workload", "8192x8192"])
    assert selected_workloads(args) == ((1024, 1024), (8192, 8192))
    # A first workload other than the old default must also be the resume input.
    assert selected_workloads(parser().parse_args(["--workload", "8192x8192"]))[0] == (8192, 8192)


@pytest.mark.parametrize("argv", [["--workload", "1000000x1000000"],
                                  ["--resume-only", "--workload", "8192x8192"]])
def test_unbounded_or_ambiguous_workload_selection_is_rejected(argv):
    with pytest.raises(SystemExit) as exc:
        parser().parse_args(argv)
    assert exc.value.code == 2


def test_missing_receipt_is_a_failed_trial_not_an_exception_that_discards_evidence():
    trace = [{"request": {"action": "verify"}, "response": {
        "transport_outcome": "outcome_unknown", "session_state": "timeout", "session_phase": "send"}},
        {"request": {"action": "compile"}, "response": {"success": True}}]
    errors = receipt_errors(registered_reference(), trace)
    assert len(errors) == 1 and "verify reference identity unknown" in errors[0]
    assert trace[0]["response"]["transport_outcome"] == "outcome_unknown"


def test_valid_receipt_still_required_for_each_numeric_rpc():
    # Fabricated offline protocol fixture only, never CUDA/performance evidence.
    reference = registered_reference()
    receipt = check_observed(reference, reference.execution.content_dict())
    trace = [{"request": {"action": action}, "response": {"reference_receipt": receipt}}
             for action in ("verify", "benchmark_with_baselines")]
    assert receipt_errors(reference, trace) == []
    trace[1]["response"]["reference_receipt"] = {"status": "passed"}
    assert len(receipt_errors(reference, trace)) == 1


def test_cached_l3_requires_complete_input_proof_and_no_retransmission():
    packet = {"protocol": PROTOCOL, "hash": "a"*64, "storages": [{"sha256":"b"*64,"bytes":16}]}
    proof = {"protocol": PROTOCOL,"manifest_hash":packet["hash"],"storage_bytes":16,
             "content_verified":True,"uploaded_bytes":0}
    trace = [{"request":{"action":"benchmark_with_baselines","inputs":packet},"response":{"input_transfer":proof}}]
    assert transfer_errors(trace) == []
    proof["uploaded_bytes"] = 16
    assert "retransmitted" in transfer_errors(trace)[0]
    proof["uploaded_bytes"] = 0
    proof["content_verified"] = False
    assert "missing complete" in transfer_errors(trace)[0]


def test_larger_workload_plan_is_bounded_fixed_and_keeps_original_candidate_and_gates():
    shapes = selected_workloads(parser().parse_args([
        "--workload", "8192x15360", "--workload", "8192x12288"]))
    code = driver.trial_code("unfused-fp32")
    plan = driver.workload_plan(shapes, code)
    assert shapes == ((8192, 15360), (8192, 12288))
    assert [r["input_storage_bytes"] for r in plan["workloads"]] == [240*1024**2, 192*1024**2]
    assert all(r["input_storage_bytes"] + plan["scalar_wire_bytes"] < plan["input_storage_scalar_limit"]
               for r in plan["workloads"])
    assert plan["input_storage_scalar_limit"] == 256*1024**2
    assert plan["candidate_hash"] == code_hash(code)
    assert plan["min_speedup"] == driver.task("offline").constraints.min_speedup == 1.05
    assert plan["rpc_timeout"] == 180 and plan["adaptive_workload_selection"] is False
    assert plan["hash"] == digest({k:v for k,v in plan.items() if k != "hash"})
    metadata, storages = json_input_parts(driver.primary((8, 32)))
    assert sum(map(len, storages)) == 8*32*2 and metadata["scalars"] == {}
    assert len(json.dumps(metadata["scalars"]).encode()) == plan["scalar_wire_bytes"]


@pytest.mark.parametrize("spec", ["8192x16384", "16384x8192"])
def test_exactly_256_mib_shapes_are_not_accepted_by_workload_whitelist(spec):
    # Even 256 MiB of ndarray alone leaves no room for the scalar wire metadata.
    with pytest.raises(SystemExit): parser().parse_args(["--workload", spec])


def test_duplicate_workloads_fail_before_any_backend_or_output_side_effects(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError("backend created before workload validation")
    monkeypatch.setattr(driver, "ProfileSSH", forbidden)
    root = tmp_path / "not-created"
    args = parser().parse_args(["--workload", "8192x15360", "--workload", "8192x15360", "--output", str(root)])
    with pytest.raises(ValueError, match="distinct"): driver.run(args)
    assert not root.exists()


def test_programmatic_workload_selection_cannot_bypass_bounds(monkeypatch):
    args = SimpleNamespace(resume_only=False, workload=["8192x16384"])
    with pytest.raises(ValueError): selected_workloads(args)
    args.resume_only = True
    with pytest.raises(ValueError, match="mutually exclusive"): selected_workloads(args)
    # Exercise the byte check independently of argparse's fixed whitelist.
    monkeypatch.setattr(driver, "EXTENDED_WORKLOADS", ((8192, 16384),))
    args.resume_only = False
    with pytest.raises(ValueError, match="storage/scalar limit"): selected_workloads(args)


def test_plan_is_durably_written_before_remote_preparation_or_input_allocation(tmp_path, monkeypatch):
    observed = []
    class OfflineBackend:
        forbidden_actions = []
        def __init__(self, **kwargs): assert kwargs["timeout"] == 180
        def prepare(self, gpu):
            paths = list(tmp_path.glob("*/plan.json"))
            assert len(paths) == 1
            data = json.loads(paths[0].read_text())
            assert data["workloads"][0]["shape"] == [8192, 15360]
            assert paths[0].stat().st_mode & 0o777 == 0o444
            observed.append(data["hash"])
            raise RuntimeError("offline prepare refusal")
        def cleanup(self): return dict(quiescent=True, removed=True)
    def forbidden(*args): raise AssertionError("input allocated before remote preparation")
    monkeypatch.setattr(driver, "ProfileSSH", OfflineBackend)
    monkeypatch.setattr(driver, "primary", forbidden)
    assert driver.run(parser().parse_args(["--workload", "8192x15360", "--output", str(tmp_path)])) == 1
    summary = json.loads(next(tmp_path.glob("*/summary.json")).read_text())
    assert summary["plan_hash"] == observed[0] and summary["completed"] is False


@pytest.fixture
def persistence_fixture(tmp_path):
    # Fabricated offline protocol fixtures, NOT CUDA/performance evidence.
    code = driver.trial_code("unfused-fp32")
    binding = dict(run_id="a"*32, manifest_hash="b"*64, contract_hash="c"*64,
        workload_hash="d"*64, protocol_hash="e"*64, candidate_hash=code_hash(code),
        verification_id="first-verification")
    metrics = dict(run_binding=binding, speedup_vs_baseline=2.0,
        arm_windows={"candidate":"first-window", "baseline":"first-window"})
    second = dict(binding, verification_id="second-verification")
    record = ConfirmRecord(ok=True, speedup=1.9, validity="valid", independent=True,
                           contract_hash=binding["contract_hash"])
    saved = tmp_path / "artifact"; saved.mkdir()
    artifact = SimpleNamespace(kernel_id="offline-candidate", code=code,
        compile_info={"run_binding":copy.deepcopy(binding)}, saved_dir=str(saved), remeasure=record)
    confirmation = dict(first_binding=copy.deepcopy(binding), second_binding=second,
        confirmation=asdict(record), second_metrics=dict(run_binding=copy.deepcopy(second),
            arm_windows={"candidate":"second-window", "baseline":"second-window"},
            contract_hash=record.contract_hash, speedup_vs_baseline=record.speedup,
            measurement_validity=record.validity))
    (saved / "confirmation.json").write_text(json.dumps(confirmation))
    t = driver.task("offline")
    stored = KernelRecord(kernel_id=artifact.kernel_id, algorithm=t.source.kernel_type,
        hardware=t.target.hardware, dsl=t.target.dsl, code=artifact.code,
        compile_info=artifact.compile_info, measurement_validity="valid", score=2.0)
    path = tmp_path / "kernel_db.jsonl"
    return SimpleNamespace(artifact=artifact, metrics=metrics, confirmation=confirmation,
        task=t, stored=stored, path=path)


def test_actual_corpus_append_is_reopened_and_replayed_exactly(persistence_fixture):
    f = persistence_fixture
    db = KernelDatabase(str(f.path), persist_retrievals=False)
    assert db.promote(f.stored)
    before = f.path.read_bytes()
    next_record = copy.deepcopy(f.stored); next_record.score = 2.1
    next_metrics = dict(f.metrics, speedup_vs_baseline=2.1)
    assert db.promote(next_record)
    proof, errors = driver.corpus_readback(f.path, before, f.artifact, f.task, next_metrics)
    assert errors == [] and proof["verified"] is True and proof["fresh_replay"] is True
    assert proof["appended_events"] == 1 and proof["candidate_hash"] == code_hash(f.artifact.code)
    assert driver.confirmation_errors(f.artifact, f.metrics) == []


@pytest.mark.parametrize("variant", ["missing", "unchanged", "rewritten", "invalid-json", "wrong-event", "extra-event",
    "wrong-kernel", "wrong-code", "wrong-task", "wrong-validity", "wrong-score", "bool-score", "nan-score",
    "wrong-binding", "no-binding", "bad-replay-prefix"])
def test_file_growth_or_admission_flag_is_not_persisted_candidate_evidence(persistence_fixture, variant):
    f = persistence_fixture
    event = dict(event="kernel", record=f.stored.to_dict(), at=1)
    before, prefix, suffix = b"", b"", b""
    if variant == "missing": pass
    elif variant == "unchanged": f.path.write_bytes(b"")
    else:
        if variant == "rewritten": before = b"old-content\n"
        elif variant == "wrong-event": event["event"] = "retrieved"
        elif variant == "extra-event": suffix = b'{}\n'
        elif variant == "wrong-kernel": event["record"]["kernel_id"] = "someone-else"
        elif variant == "wrong-code": event["record"]["code"] += "\n# different candidate"
        elif variant == "wrong-task": event["record"]["algorithm"] = "relu"
        elif variant == "wrong-validity": event["record"]["measurement_validity"] = "degraded"
        elif variant == "wrong-score": event["record"]["score"] = 3.0
        elif variant in {"bool-score", "nan-score"}:
            # Matching artifact metrics cannot bless a malformed numeric field.
            f.metrics["speedup_vs_baseline"] = event["record"]["score"] = True if variant == "bool-score" else float("nan")
        elif variant == "wrong-binding": event["record"]["compile_info"]["run_binding"]["workload_hash"] = "wrong-workload"
        elif variant == "no-binding": event["record"]["compile_info"].pop("run_binding")
        elif variant == "bad-replay-prefix": before = prefix = b"invalid historical JSON\n"
        data = b"invalid JSON\n" if variant == "invalid-json" else json.dumps(event).encode()+b"\n"
        f.path.write_bytes(prefix+data+suffix)
    proof, errors = driver.corpus_readback(f.path, before, f.artifact, f.task, f.metrics)
    assert proof["verified"] is False and len(errors) == 1


@pytest.mark.parametrize("variant", ["missing", "invalid-json", "old-first", "other-workload", "same-verification",
    "same-window", "window-missing", "record-drift", "second-metrics-drift", "unusable", "not-independent",
    "uninformative", "below-threshold", "nan-speedup", "bool-speedup", "contract-drift"])
def test_confirmation_needs_durable_matching_identity_and_a_distinct_window(persistence_fixture, variant):
    f = persistence_fixture
    p = Path(f.artifact.saved_dir) / "confirmation.json"
    c = f.confirmation
    if variant == "missing": p.unlink()
    elif variant == "invalid-json": p.write_text("not JSON")
    else:
        if variant == "old-first": c["first_binding"]["verification_id"] = "old-result"
        elif variant == "other-workload": c["second_binding"]["workload_hash"] = "another-workload"
        elif variant == "same-verification": c["second_binding"]["verification_id"] = c["first_binding"]["verification_id"]
        elif variant == "same-window": c["second_metrics"]["arm_windows"]["candidate"] = "first-window"
        elif variant == "window-missing": c["second_metrics"].pop("arm_windows")
        elif variant == "record-drift": c["confirmation"]["speedup"] = 3.0
        elif variant == "second-metrics-drift": c["second_metrics"]["speedup_vs_baseline"] = 3.0
        else:
            changes = {"unusable":("ok",False), "not-independent":("independent",False),
                "uninformative":("validity","uninformative"), "below-threshold":("speedup",1.04),
                "nan-speedup":("speedup",float("nan")), "bool-speedup":("speedup",True),
                "contract-drift":("contract_hash","other-contract")}
            k, v = changes[variant]; setattr(f.artifact.remeasure, k, v)
            c["confirmation"] = asdict(f.artifact.remeasure)
        p.write_text(json.dumps(c))
    assert len(driver.confirmation_errors(f.artifact, f.metrics)) == 1


def test_usable_degraded_confirmation_is_not_silently_changed_to_a_stricter_rule(persistence_fixture):
    f = persistence_fixture
    f.artifact.remeasure.validity = "degraded"
    f.confirmation["confirmation"] = asdict(f.artifact.remeasure)
    f.confirmation["second_metrics"]["measurement_validity"] = "degraded"
    (Path(f.artifact.saved_dir) / "confirmation.json").write_text(json.dumps(f.confirmation))
    assert driver.confirmation_errors(f.artifact, f.metrics) == []
