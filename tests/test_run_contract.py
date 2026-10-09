"""Durable evidence/resume gates, using real Conductor + real artifact I/O."""
import copy
import contextlib
import io
import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from operator_agent import run_contract as rc
from operator_agent.artifact_store import ArtifactStore
from operator_agent.contract import freeze_from_task
from tests.test_resume_integrity import _conductor, _inputs, _run, _task, SNAP


def completed(tmp_path, task=None):
    task = task or _task("durable", min_speedup=0)
    art, _, _ = _run(str(tmp_path), task, rounds=1)
    assert art.saved_dir and not any(r.metrics.get("run_identity_unknown") for r in art.verification_results)
    return task, art, Path(art.saved_dir)


def hold(c, task, inputs, source, *, rounds=2):
    c.max_optimize_rounds = rounds
    # All file/contract/input negative controls must stop before routing/GPU.
    with mock.patch.object(c.router, "route", side_effect=AssertionError("routing on HOLD")), \
         mock.patch.object(c.backend, "get_hardware_info", side_effect=AssertionError("GPU on HOLD")), \
         mock.patch.object(c.designer, "run", side_effect=AssertionError("LLM on HOLD")), \
         contextlib.redirect_stdout(io.StringIO()):
        art = c.execute(task, inputs, target_gflops=0, resume_from=str(source))
    assert not art.all_passed
    assert len(art.verification_results) == 1
    assert art.verification_results[0].metrics["run_identity_unknown"] is True
    assert all(v is False for v in art.verification_results[0].metrics["eligibility"].values())
    result = json.loads((Path(art.saved_dir) / "result.json").read_text())
    assert result["resume"] == art.resume_status
    assert not (Path(art.saved_dir) / SNAP).exists()
    return art


def test_contract_is_durable_before_agents_and_final_copy_is_identical(tmp_path):
    c = _conductor(str(tmp_path), rounds=1)
    original = c.designer.run
    seen = []
    def design(*a, **k):
        folder = Path(c._run_binding_dir)
        binding = c._run_binding
        assert json.loads((folder / rc.CONTRACT).read_text()) == binding["contract"]
        assert json.loads((folder / rc.MANIFEST).read_text()) == binding["manifest"]
        assert (folder / rc.CONTRACT).stat().st_mode & 0o222 == 0
        seen.append(folder)
        return original(*a, **k)
    c.designer.run = design
    with contextlib.redirect_stdout(io.StringIO()):
        art = c.execute(_task("before-generation", 0), _inputs(), target_gflops=0)
    assert len(seen) == 1
    final = Path(art.saved_dir)
    for name in (rc.CONTRACT, rc.MANIFEST):
        assert (seen[0] / name).read_bytes() == (final / name).read_bytes()
    snap = json.loads((final / SNAP).read_text())
    assert snap["run_binding"]["candidate_hash"] == rc.code_hash(art.code)
    assert snap["run_binding"]["checkpoint_hash"] == rc.digest({k: v for k, v in snap.items() if k != "run_binding"})
    assert art.benchmark_data["run_binding"]["manifest_hash"] == c._run_binding["manifest"]["hash"]


@pytest.mark.parametrize("name", [rc.CONTRACT, rc.MANIFEST, SNAP])
@pytest.mark.parametrize("mutation", ["missing", "truncated", "symlink"])
def test_missing_torn_or_symlink_evidence_holds_without_side_effects(tmp_path, name, mutation):
    task, art, folder = completed(tmp_path)
    path = folder / name
    path.chmod(0o644)
    if mutation == "missing":
        path.unlink()
    elif mutation == "truncated":
        path.write_text(path.read_text()[:25])
    else:
        copy_path = tmp_path / "copy.json"
        copy_path.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(copy_path)
    hold(_conductor(str(tmp_path)), task, _inputs(), folder)


@pytest.mark.parametrize("field,value", [
    ("current_code", "def kernel(A, B): return A"),
    ("pending_fix", "def kernel(A, B): return B"),
    ("round_done", 1000), ("best_gflops", 1e20),
    ("verification_results", [{"level": "L3", "passed": True}]),
    ("sketch_dict", {}), ("environment_fingerprint", {}),
    ("iteration_history", []), ("run_binding", {}),
])
def test_candidate_and_checkpoint_state_tampering_holds(tmp_path, field, value):
    task, _, folder = completed(tmp_path)
    path = folder / SNAP
    snap = json.loads(path.read_text())
    snap[field] = value
    path.write_text(json.dumps(snap))
    hold(_conductor(str(tmp_path)), task, _inputs(), folder)


@pytest.mark.parametrize("change", ["values", "dtype", "strides", "offset", "scalar", "shape", "alias"])
def test_actual_workload_not_only_shape_is_bound(tmp_path, change):
    task, _, folder = completed(tmp_path)
    x = _inputs()
    if change == "values":
        x["A"][0, 0] += 1
    elif change == "dtype":
        x["A"] = x["A"].astype(np.float64)
    elif change == "strides":
        x["A"] = x["A"].T.copy().T  # same values/shape, different layout
    elif change == "offset":
        storage = np.zeros((33, 32), dtype=np.float32)
        storage[1:] = x["A"]
        x["A"] = storage[1:]
    elif change == "scalar":
        x["epsilon"] = 1e-5
    elif change == "shape":
        x = _inputs(64)
    else:
        x["B"] = x["A"]
    hold(_conductor(str(tmp_path)), task, x, folder)


@pytest.mark.parametrize("change", ["threshold", "reference", "semantic", "protocol", "controls"])
def test_contract_reference_protocol_and_controls_are_not_rebound_on_resume(tmp_path, change):
    task, _, folder = completed(tmp_path)
    c = _conductor(str(tmp_path))
    if change == "threshold":
        task.constraints.max_error *= 10
    elif change == "reference":
        task.source.reference_source = "def reference(A, B): return A @ B"
    elif change == "semantic":
        task.source.semantic_parameters["epsilon"] = .1
    elif change == "controls":
        c.heartbeat_interval += 1
    else:
        identity = rc.protocol_identity()
        identity["operator_agent/verifier.py"] = "0" * 64
        with mock.patch.object(rc, "protocol_identity", return_value=identity):
            hold(c, task, _inputs(), folder)
        return
    hold(c, task, _inputs(), folder)


def test_recomputing_sidecar_hash_cannot_bless_a_different_task_contract(tmp_path):
    task, _, folder = completed(tmp_path)
    altered = copy.deepcopy(task)
    altered.constraints.max_error *= 10
    contract = freeze_from_task(altered).as_dict()
    manifest = json.loads((folder / rc.MANIFEST).read_text())
    manifest["contract_hash"] = contract["hash"]
    manifest.pop("hash")
    manifest["hash"] = rc.digest(manifest)
    for name, data in ((rc.CONTRACT, contract), (rc.MANIFEST, manifest)):
        (folder / name).chmod(0o644)
        (folder / name).write_text(json.dumps(data))
    hold(_conductor(str(tmp_path)), task, _inputs(), folder)


def test_legacy_unbound_snapshot_and_exhausted_resume_do_not_promote(tmp_path):
    task, _, folder = completed(tmp_path)
    hold(_conductor(str(tmp_path)), task, _inputs(), folder, rounds=1)
    (folder / rc.MANIFEST).unlink()
    hold(_conductor(str(tmp_path)), task, _inputs(), folder)


def test_fresh_process_reverifies_saved_candidate_with_zero_generation(tmp_path):
    task, first, folder = completed(tmp_path)
    script = '''
import contextlib, io, json, sys
from pathlib import Path
from tests.test_resume_integrity import _conductor, _inputs, _task
c = _conductor(sys.argv[1], rounds=2)
def forbidden(*a, **k): raise AssertionError("generation on resume")
c.designer.run = forbidden
c.coder.run = forbidden
calls = []
verify = c.verifier.verify
def fresh(*a, **k):
    calls.append(k['kernel_code'])
    return verify(*a, **k)
c.verifier.verify = fresh
with contextlib.redirect_stdout(io.StringIO()):
    art = c.execute(_task('durable', 0), _inputs(), target_gflops=0, resume_from=sys.argv[2])
assert len(calls) == 1
assert c.llm.calls == 0
assert art.resume_status['status'] == 'resumed'
assert art.iteration_history[0]['round'] == 2
assert art.metadata['resume_audit']['evidence_scope'] == 'previous_process_audit_only'
assert art.metadata['resume_audit']['round_done'] == 1
assert all(r.metrics['run_binding']['candidate_hash'] for r in art.verification_results)
print(json.dumps({'code': art.code, 'run_id': c._run_binding['manifest']['run_id']}))
'''
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path), str(folder)],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(result.stdout.splitlines()[-1])
    assert data["code"] == first.code
    assert data["run_id"] == json.loads((folder / rc.MANIFEST).read_text())["run_id"]


def test_atomic_failure_preserves_old_snapshot_and_cleans_unique_temporary(tmp_path):
    path = tmp_path / SNAP
    rc.atomic_json(path, {"old": True})
    def torn(data, f, **kw):
        f.write('{"torn":')
        raise OSError("disk full")
    with mock.patch.object(rc.json, "dump", side_effect=torn), pytest.raises(OSError):
        rc.atomic_json(path, {"new": True})
    assert json.loads(path.read_text()) == {"old": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_sidecar_failure_stops_before_agents(tmp_path):
    c = _conductor(str(tmp_path), rounds=1)
    with mock.patch.object(rc, "atomic_json", side_effect=OSError("disk full")), \
         mock.patch.object(c.designer, "run", side_effect=AssertionError("generation")), \
         contextlib.redirect_stdout(io.StringIO()):
        art = c.execute(_task("disk", 0), _inputs(), target_gflops=0)
    assert art.verification_results[0].metrics["run_identity_unknown"]
    assert c.llm.calls == 0


def test_live_task_mutation_after_generation_stops_before_verification(tmp_path):
    c = _conductor(str(tmp_path), rounds=1)
    task = _task("live-drift", 0)
    original = c.coder.run
    def mutate(*a, **k):
        code = original(*a, **k)
        task.constraints.max_error *= 2
        return code
    c.coder.run = mutate
    with mock.patch.object(c.verifier, "verify", side_effect=AssertionError("verification on drift")), \
         mock.patch.object(c, "_save_to_db", side_effect=AssertionError("admission on drift")), \
         contextlib.redirect_stdout(io.StringIO()):
        art = c.execute(task, _inputs(), target_gflops=0)
    assert art.verification_results[0].metrics["run_identity_unknown"]


@pytest.mark.parametrize("filename", [SNAP, "confirmation.json"])
def test_critical_evidence_write_failure_cannot_continue_to_admission(tmp_path, filename):
    c = _conductor(str(tmp_path), rounds=1)
    atomic = rc.atomic_json
    def fail(path, data, **kw):
        if Path(path).name == filename:
            raise OSError("disk full after verification")
        return atomic(path, data, **kw)
    with mock.patch.object(rc, "atomic_json", side_effect=fail), \
         mock.patch.object(c, "_save_to_db", side_effect=AssertionError("admission after write failure")), \
         contextlib.redirect_stdout(io.StringIO()):
        art = c.execute(_task("critical-write", 0), _inputs(), target_gflops=0)
    assert art.verification_results[0].metrics["run_identity_unknown"]


def test_confirmation_binds_same_candidate_and_distinct_new_window(tmp_path):
    _, art, folder = completed(tmp_path)
    proof = json.loads((folder / "confirmation.json").read_text())
    first, second = proof["first_binding"], proof["second_binding"]
    assert first["verification_id"] != second["verification_id"]
    for key in ("candidate_hash", "manifest_hash", "contract_hash", "workload_hash", "protocol_hash"):
        assert first[key] == second[key]
    assert first["candidate_hash"] == rc.code_hash(art.code)


def test_nonfinite_diagnostics_are_explicit_not_zero_or_nonstandard_json(tmp_path):
    data = rc.audit_value({"bad_reference": float("inf"), "bad_error": float("nan")})
    path = tmp_path / "audit.json"
    rc.atomic_json(path, data)
    assert json.loads(path.read_text()) == {"bad_reference": {"$nonfinite": "+inf"},
                                           "bad_error": {"$nonfinite": "nan"}}
    assert "Infinity" not in path.read_text()
    with pytest.raises(ValueError):
        rc.digest({"atol": float("nan")})


def test_task_id_lookup_cannot_skip_unreadable_and_reuse_an_older_checkpoint(tmp_path):
    task, _, folder = completed(tmp_path)
    bad = tmp_path / "newer" / SNAP
    bad.parent.mkdir()
    bad.write_text('{"task_id":"durable",')
    art = hold(_conductor(str(tmp_path)), task, _inputs(), task.task_id)
    assert art.resume_status["status"] == "ambiguous_lookup"


def test_json_array_checkpoint_is_unreadable_not_an_unhandled_exception(tmp_path):
    task, _, folder = completed(tmp_path)
    (folder / SNAP).write_text("[]")
    art = hold(_conductor(str(tmp_path)), task, _inputs(), folder)
    assert art.resume_status["status"] == "unreadable_snapshot"


@pytest.mark.parametrize("filename", [rc.CONTRACT, rc.MANIFEST])
def test_live_sidecar_drift_is_not_silently_ignored_or_overwritten(tmp_path, filename):
    c = _conductor(str(tmp_path), rounds=1)
    original = c.designer.run
    altered = []
    def mutate(*a, **k):
        sketch = original(*a, **k)
        path = Path(c._run_binding_dir) / filename
        data = json.loads(path.read_text())
        data["hash"] = "0" * 64
        path.chmod(0o644)
        path.write_text(json.dumps(data))
        altered.append(path)
        return sketch
    c.designer.run = mutate
    with mock.patch.object(c.coder, "run", side_effect=AssertionError("generation after sidecar drift")), \
         mock.patch.object(c.verifier, "verify", side_effect=AssertionError("GPU after sidecar drift")), \
         contextlib.redirect_stdout(io.StringIO()):
        art = c.execute(_task("sidecar-live", 0), _inputs(), target_gflops=0)
    assert art.verification_results[0].metrics["run_identity_unknown"]
    assert json.loads(altered[0].read_text())["hash"] == "0" * 64


def test_protocol_includes_numeric_and_admission_gate_implementations():
    sources = rc.protocol_identity()
    for file in ("tolerance.py", "measurement.py", "eligibility.py", "oracle.py", "confirm.py"):
        assert len(sources["operator_agent/" + file]) == 64


def test_live_sidecar_boolean_to_integer_drift_is_not_hidden_by_python_equality(tmp_path):
    binding = rc.create(_task("bool-drift", 0), [_inputs()], {"name": "MockGPU"})
    rc.write(tmp_path, binding)
    path = tmp_path / rc.CONTRACT
    data = json.loads(path.read_text())
    assert data["dtype_semantics"]["tf32"] is False
    data["dtype_semantics"]["tf32"] = 0
    assert data == binding["contract"]  # This is exactly the Python-equality trap.
    path.chmod(0o644)
    path.write_text(json.dumps(data))
    with pytest.raises(rc.RunBindingError, match="immutable sidecar drift"):
        rc.write(tmp_path, binding)
