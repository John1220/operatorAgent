"""Offline protocol/hold guards; fabricated CUDA receipts are NOT GPU evidence."""
import copy
import dataclasses
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from operator_agent import Conductor, IntentParser, MockGPU, Verifier
from operator_agent.core.contract import ContractError, freeze_from_task, load_dict
from operator_agent.decide.correctness import as_suite, credential
from operator_agent.decide.correctness_profiles import PROFILE_VERSION, PROFILE_V2, profile_check, v2_suite
from operator_agent.decide.reference_contract import (ReferenceIdentityError, as_reference, registered_reference,
    check_observed, execution_receipt, validate_receipt, validate_gate, _file_hash)
from operator_agent.execute.remote_executor import SSHExecutor, RemoteExecutor
from operator_agent.core.schemas import VerificationLevel as Level


def task():
    return IntentParser().parse("layernorm fp16 triton nvidia")


def receipt():
    spec = registered_reference()
    return check_observed(spec, spec.execution.content_dict())


class ProtocolSSH(SSHExecutor):
    """Deliberately fake peer for controller tests, never a numerical oracle."""
    def __init__(self):
        super().__init__(host="offline")
        self._deployed = True
        self.requests = []
        self.reply = None

    def _ssh_exec_json(self, payload):
        from operator_agent.decide.remote_input_fidelity import decode_json_inputs, to_torch_inputs, torch_input_readback
        self.requests.append(copy.deepcopy(payload))
        action = payload["action"]
        if action == "gpu_info":
            return dict(success=True, reference_receipt=receipt())
        if self.reply is not None:
            return copy.deepcopy(self.reply)
        values = decode_json_inputs(payload["inputs"])
        reading = torch_input_readback(to_torch_inputs(values, "cpu"))
        # Fabricated label for testing controller branches only.
        reading["tensor_devices"] = {k: "cuda:0" for k in values}
        return dict(success=True, reference_receipt=receipt(), output_dtype="torch.float16",
                    input_fidelity=reading, max_error=0.0, max_rel_error=0.0,
                    matched_ratio=1.0, bad_elements=0, compared_elements=values["X"].size,
                    coverage=1.0, atol=payload["max_error"], rtol=.01,
                    required_matched_ratio=1.0, reference_nonfinite=0,
                    output_digest="f" * 64, device="cuda", reference_type="pytorch_eager")


def test_new_task_exact_v2_and_old_payload_hash_compatible():
    t = task()
    assert t.constraints.correctness_profile == PROFILE_V2
    assert len(as_suite(t.constraints.correctness_suite).cases) == 32
    c = freeze_from_task(t)
    assert c.reference_contract == registered_reference()
    assert load_dict(json.loads(json.dumps(c.as_dict()))).hash == c.hash
    assert t.to_dict()["constraints"]["reference_contract"] == registered_reference().content_dict()
    old = IntentParser().parse("relu fp32 python mock_gpu")
    d = freeze_from_task(old).as_dict()
    assert "reference_contract" not in d and "reference_contract" not in old.to_dict()["constraints"]
    assert load_dict(d).hash == freeze_from_task(old).hash


@pytest.mark.parametrize("key,value", [
    ("epsilon", 1e-3), ("normalized_axis", 0), ("affine", True), ("input_dtype", "fp32"),
    ("output_dtype", "fp32"), ("reference_accum", "fp32"), ("fp_fusion", "required_disabled"),
    ("comparison", "absolute"), ("rtol", .1), ("required_matched_ratio", .99),
    ("required_matched_ratio", True), ("epsilon", True), ("normalized_axis", True)])
def test_semantics_not_candidate_customizable(key, value):
    raw = registered_reference().content_dict(); raw[key] = value
    with pytest.raises(ReferenceIdentityError): as_reference(raw)
    t = task(); t.constraints.reference_contract = raw
    assert profile_check(t)["status"] == "unknown"
    with pytest.raises(ContractError): freeze_from_task(t)


@pytest.mark.parametrize("key", list(registered_reference().execution.content_dict()))
def test_each_runtime_component_required_and_drift_is_unknown(key):
    raw = registered_reference().content_dict(); del raw["execution"][key]
    with pytest.raises(ReferenceIdentityError): as_reference(raw)
    observed = registered_reference().execution.content_dict(); observed[key] = "changed"
    with pytest.raises(ReferenceIdentityError): check_observed(registered_reference(), observed)


@pytest.mark.parametrize("change", ["missing", "hash", "cpu", "extra", "bool"])
def test_missing_or_forged_receipt_not_trusted(change):
    raw = receipt()
    if change == "missing": raw = None
    if change == "hash": raw["reference_hash"] = "0" * 64
    if change == "cpu": raw["observed"]["device"] = "cpu"
    if change == "extra": raw["cached_passed"] = True
    if change == "bool": raw["observed"]["deterministic_algorithms"] = 0
    with pytest.raises(ReferenceIdentityError): validate_receipt(registered_reference(), raw)


def test_pinned_primitive_and_cache_invalidation(tmp_path):
    from operator_agent.decide import builtin_layernorm
    assert hashlib.sha256(Path(builtin_layernorm.__file__).read_bytes()).hexdigest() == registered_reference().execution.primitive_sha256
    p = tmp_path / "reference.py"; p.write_bytes(b"first")   # 只是要一个文件，层目录与判据无关
    first = _file_hash(p)
    assert _file_hash(p) == first
    p.write_bytes(b"other")  # Same size, still re-hashed via stat.
    assert _file_hash(p) != first


@pytest.mark.parametrize("error", [0, -1, 1e-4, True, float("nan"), float("inf")])
def test_tolerance_not_loosened(error):
    with pytest.raises(ReferenceIdentityError): validate_gate(registered_reference(), "fp16", error)


def test_finite_suite_is_exact_calibration_not_passing_subset():
    from benchmarks.calibration.calibrate_layernorm import fixtures
    actual = v2_suite()
    assert len(actual.cases) == 32
    # Multiset of dtype/shape/value/layout matches all original measurements.
    def signature(xs):
        x = xs["X"]
        return (x.dtype.str, x.shape, x.strides, x.tobytes())
    assert sorted(signature(c.materialize()) for c in actual.cases) == sorted(signature(xs) for _, _, xs in fixtures())
    t = task(); raw = copy.deepcopy(t.constraints.correctness_suite); raw["cases"].pop()
    t.constraints.correctness_suite = raw
    assert profile_check(t)["status"] == "unknown"


def test_real_identity_is_required_before_generation_and_failure_learning(tmp_path):
    from operator_agent.core.kernel_db import FailureDatabase
    c = Conductor(backend=MockGPU(), failure_db=FailureDatabase(str(tmp_path / "failures.jsonl")))
    c.designer.run = Mock(side_effect=AssertionError("generated"))
    c.fixer.run = Mock(side_effect=AssertionError("fixed"))
    art = c.execute(task(), {"X": np.ones((8, 8), np.float16)})
    assert art.verification_results[0].metrics["reference_identity_unknown"]
    assert credential(art.verification_results) is None
    c.designer.run.assert_not_called(); c.fixer.run.assert_not_called()
    c._record_failure(task(), art.verification_results, "", False)
    assert not (tmp_path / "failures.jsonl").exists()


@pytest.mark.parametrize("change", ["remove", "tolerance", "suite", "source"])
def test_bound_contract_cannot_change_between_generation_l2_and_l3(change):
    t = task(); b = ProtocolSSH(); v = Verifier(b)
    assert v.reference_preflight(t, start_run=True) is None
    before = len(b.requests)
    if change == "remove":
        t.constraints.reference_contract = None
        t.constraints.correctness_profile = None
        t.constraints.correctness_suite = None
    if change == "tolerance": t.constraints.max_error = 1e-6
    if change == "suite": t.constraints.correctness_suite["cases"].pop()
    if change == "source": t.source.reference_source = "def reference(X): return X"
    held = v.reference_preflight(t)
    assert held.metrics["reference_identity_unknown"] and credential([held]) is None
    assert len(b.requests) == before
    result = v.verify_l3_only("def kernel(X): return X", {"X": np.ones((8, 8), np.float16)}, 0, t, correctness_verified=True)[0]
    assert not result.passed


def test_controller_transports_exact_contract_and_all_case_receipts():
    b = ProtocolSSH(); v = Verifier(b); t = task()
    r = v._verify_function("def kernel(X): return X", {"X": np.ones((8, 8), np.float16)}, None, t)
    assert r.passed
    assert r.metrics["reference_receipts"] == [receipt()]
    rows = r.metrics["correctness"]["cases"]
    assert len(rows) == 32 and all(x["determinism"] == "passed" for x in rows)
    assert all(x["metrics"]["reference_receipt"] == receipt() for x in rows)
    payloads = [x for x in b.requests if x["action"] == "verify"]
    assert len(payloads) == 65
    assert all(x["reference_contract"] == registered_reference().content_dict() for x in payloads)


@pytest.mark.parametrize("change", ["missing", "malformed", "cpu", "dtype", "gate",
                                  "matched", "partial", "nonfinite", "missingstats", "digest"])
def test_unattested_l2_response_is_unknown_and_not_a_candidate_failure(change):
    b = ProtocolSSH(); t = task(); v = Verifier(b)
    xs = {"X": np.ones((8, 8), np.float16)}
    b.reply = b.verify_function("fixture", xs, "layernorm", 1e-5, "fp16")
    if change == "missing": del b.reply["reference_receipt"]
    if change == "malformed": b.reply = ["invalid"]
    if change == "cpu": b.reply["reference_receipt"]["observed"]["device"] = "cpu"
    if change == "dtype": del b.reply["output_dtype"]
    if change == "gate": b.reply.update(max_error=0, atol=.1, rtol=.1, required_matched_ratio=1.0)
    if change == "matched": b.reply["matched_ratio"] = .99
    if change == "partial": b.reply["compared_elements"] = 63
    if change == "nonfinite": b.reply["max_rel_error"] = float("nan")
    if change == "missingstats": del b.reply["bad_elements"]
    if change == "digest": b.reply["output_digest"] = "short"
    r = v._verify_single_set("def kernel(X): return X", xs, None, t)
    assert r.metrics["reference_identity_unknown"] and r.metrics["correctness"]["verdict"] == "unknown"


def test_l3_cannot_measure_without_l2_credential_or_fallback_to_another_baseline():
    b = ProtocolSSH(); v = Verifier(b); t = task(); xs = {"X": np.ones((8, 8), np.float16)}
    r = v.verify_l3_only("def kernel(X): return X", xs, 0, t)[0]
    assert not r.passed and r.metrics["reference_identity_unknown"]
    assert not any(x["action"] == "benchmark_with_baselines" for x in b.requests)
    b.reply = b.verify_function("fixture", xs, "layernorm", 1e-5, "fp16")
    b.reply["baseline_selected"] = "torch_compile"
    bench, _ = v._bench_case("def kernel(X): return X", xs, t)
    assert bench["reference_identity_unknown"]
    payload = b.requests[-1]
    assert payload["baseline_priority"] == ["pytorch_eager"]
    from operator_agent.decide.remote_input_fidelity import JSON_PROTOCOL
    assert payload["inputs"]["protocol"] == JSON_PROTOCOL


def test_facade_forwards_identity_without_synthetic_success():
    inner = ProtocolSSH(); facade = RemoteExecutor(mode="ssh"); facade._backend = inner
    assert facade.reference_identity(registered_reference().content_dict())["reference_receipt"] == receipt()
    assert inner.requests[-1]["action"] == "gpu_info"


def test_cpu_runner_refuses_before_loading_candidate(monkeypatch):
    from tests.unit.test_input_wire_format import _runner
    runner = _runner()
    runner._load_module = Mock(side_effect=AssertionError("candidate loaded"))
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    r = runner.run_verify("invalid source", {}, "layernorm", 1e-5, "fp16",
                          reference_contract=registered_reference().content_dict())
    assert not r["success"] and r["reference_identity_unknown"]
    runner._load_module.assert_not_called()
    r = runner.run_benchmark_with_baselines("invalid source", {}, "layernorm", precision="fp16",
                                            reference_contract=registered_reference().content_dict())
    assert r["reference_identity_unknown"]
    runner._load_module.assert_not_called()


def test_v2_local_reference_is_never_an_oracle():
    with pytest.raises(ValueError, match="local/CPU"):
        Verifier(MockGPU())._build_reference_fn(task(), {"X": np.ones((8, 8), np.float16)})


@pytest.mark.parametrize("reply", [None, [], {"success": True}, {"success": False, "reference_error": "drift"}])
def test_absent_or_malformed_preflight_reply_is_unknown(reply):
    b = ProtocolSSH(); b.reference_identity = lambda raw: reply
    r = Verifier(b).reference_preflight(task())
    assert r.metrics["reference_identity_unknown"] and not r.passed


def test_static_reference_drift_is_hold_not_fixer_input():
    t = task(); t.constraints.reference_contract["epsilon"] = .1
    r = Verifier.correctness_preflight(t)
    assert r.metrics["reference_identity_unknown"]
    assert credential([r]) is None


@pytest.mark.parametrize("change", ["dtype", "accum", "tf32", "reduced", "nondeterministic", "baseline"])
def test_freeze_rejects_inconsistent_precision_or_baseline(change):
    from operator_agent.core.contract import freeze
    c = freeze_from_task(task())
    ds = c.dtype_semantics
    baseline = c.baseline
    if change == "dtype": ds = dataclasses.replace(ds, dtype="fp32")
    if change == "accum": ds = dataclasses.replace(ds, accum="fp16")
    if change == "tf32": ds = dataclasses.replace(ds, tf32=True)
    if change == "reduced": ds = dataclasses.replace(ds, allow_reduced_precision=True)
    if change == "nondeterministic": ds = dataclasses.replace(ds, deterministic=False)
    if change == "baseline": baseline = dataclasses.replace(baseline, name="torch_compile")
    with pytest.raises(ContractError):
        freeze(c.kernel_type, ds, baseline, c.shapes, max_error=c.max_error, min_speedup=c.min_speedup,
               correctness_suite=c.correctness_suite, correctness_profile=c.correctness_profile,
               reference_contract=c.reference_contract)


def test_xpu_explicitly_refuses_identity_contract_before_execution():
    import ast
    from types import SimpleNamespace
    path = Path(__file__).resolve().parents[2] / "remote_scripts" / "run_kernel_xpu.py"
    # Exercise actual function bodies without running platform bootstrap, which
    # modifies XPU cache paths and environment on import.
    names = {"_reference_refusal", "run_gpu_info", "run_verify", "run_benchmark", "run_benchmark_with_baselines"}
    definitions = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {}; exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    runner = SimpleNamespace(**namespace)
    raw = registered_reference().content_dict()
    runner._load_module = Mock(side_effect=AssertionError("candidate loaded"))
    results = [runner.run_gpu_info(raw),
               runner.run_verify("bad code", {}, "layernorm", 1e-5, reference_contract=raw),
               runner.run_benchmark("bad code", {}, reference_contract=raw),
               runner.run_benchmark_with_baselines("bad code", {}, "layernorm", reference_contract=raw)]
    assert all(r.get("reference_identity_unknown") and r["success"] is False for r in results)
    runner._load_module.assert_not_called()


def test_identity_hold_after_generation_stops_round_without_fix_search_retry_or_learning(tmp_path):
    from types import SimpleNamespace
    from operator_agent.core.backend_caps import Cap
    from operator_agent.core.kernel_db import KernelDatabase, FailureDatabase
    from operator_agent.core.schemas import UnifiedSketch
    class NoProvisionPeer(ProtocolSSH):
        @property
        def capabilities(self):
            return super().capabilities - {Cap.ENV_PROVISION, Cap.DEP_ANALYSIS, Cap.DEPLOY}
    b = NoProvisionPeer()
    b.get_hardware_info = lambda: {"name": "NVIDIA H20", "device": "cuda", "compute_capability": "9.0"}
    b.compile = Mock(side_effect=AssertionError("compiled"))
    b.reference_identity = Mock(side_effect=[dict(success=True, reference_receipt=receipt()),
                                             dict(success=False, reference_error="runtime drift")])
    c = Conductor(backend=b, kernel_db=KernelDatabase(str(tmp_path / "kernels.jsonl"), persist_retrievals=False),
                  failure_db=FailureDatabase(str(tmp_path / "failures.jsonl")),
                  max_optimize_rounds=3, use_coder_pipeline=False)
    from operator_agent.core.artifact_store import ArtifactStore
    c.store = ArtifactStore(str(tmp_path / "artifacts"))
    c.designer.run = Mock(return_value=UnifiedSketch(algorithm="layernorm", target_dsl="triton"))
    from benchmarks.calibration.calibrate_layernorm import candidate_source
    c.coder = SimpleNamespace(run=lambda *a, **k: candidate_source("unfused-fp32"))
    c.fixer.run = Mock(side_effect=AssertionError("fixed"))
    c.search.run = Mock(side_effect=AssertionError("searched"))
    art = c.execute(task(), {"X": np.ones((8, 8), np.float16)})
    assert art.metadata["stop"]["reason"] == "reference_identity_hold"
    assert len(art.iteration_history) == 1
    assert art.verification_results[-1].metrics["correctness"]["verdict"] == "unknown"
    c.designer.run.assert_called_once(); b.compile.assert_not_called()
    c.fixer.run.assert_not_called(); c.search.run.assert_not_called()
    assert not (tmp_path / "failures.jsonl").exists()
    assert not (tmp_path / "kernels.jsonl").exists()


def test_secondary_case_reference_unknown_stops_remaining_ledger_and_revokes_l3():
    from operator_agent.core.contract import case_key
    xs = [{"X": np.ones((8, n), np.float16)} for n in (8, 9, 10)]
    v = Verifier(ProtocolSSH())
    v._measure_case_speedup = Mock(return_value=dict(speedup_vs_baseline=None,
                                                   reference_identity_unknown=True, reason="changed"))
    metrics = dict(speedup_vs_baseline=2)
    v._publish_case_ledger(metrics, "fixture", xs, task())
    assert metrics["reference_identity_unknown"]
    assert metrics["cases"][case_key(xs[2])]["speedup_vs_baseline"] is None
    v._measure_case_speedup.assert_called_once()
    held = v._reference_unknown(metrics["reference_error"], Level.L3_PERFORMANCE, metrics)
    assert not held.passed and not any(held.metrics["eligibility"].values())
    assert "speedup_vs_baseline" not in held.metrics and "gflops" not in held.metrics
