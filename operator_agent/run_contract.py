"""Durable run identity. Hashes detect drift, not hostile same-UID forgery.

Old FrozenContract payloads/hashes are unchanged. The separate manifest binds
actual caller inputs and the evaluator implementation, which that schema does
not describe. Checkpoints are sealed only after immutable sidecars are durable.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import uuid
from pathlib import Path

from .contract import freeze_from_task, load_dict
from .input_fidelity import describe_inputs

CONTRACT = "contract.json"
MANIFEST = "run_manifest.json"
SCHEMA = "frozen-run-v1"


class RunBindingError(ValueError):
    """Unknown/HOLD, never a candidate failure or a restart instruction."""


def digest(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def code_hash(code):
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def json_value(value):
    # FrozenContract uses tuples in memory; comparisons must use the same JSON
    # normal form in this process and after a round trip through disk.
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def audit_value(value):
    """Nonfinite diagnostics stay explicit in standard JSON, never become zero.

    Only checkpoint/confirmation audit data use this representation. Manifest
    declarations remain strict (NaN thresholds cannot acquire an identity).
    Old observations are never restored as live verification credentials.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return {"$nonfinite": "nan" if math.isnan(value) else
                ("+inf" if value > 0 else "-inf")}
    if isinstance(value, dict):
        return {k: audit_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [audit_value(v) for v in value]
    return value


def protocol_identity():
    """Version the *implementation*, not just a handwritten protocol label."""
    root = Path(__file__).resolve().parents[1]
    files = (
        "operator_agent/run_contract.py", "operator_agent/contract.py",
        "operator_agent/conductor.py", "operator_agent/verifier.py",
        "operator_agent/measurement.py", "operator_agent/correctness.py",
        "operator_agent/correctness_profiles.py", "operator_agent/reference_contract.py",
        "operator_agent/builtin_layernorm.py", "operator_agent/bench_cases.py",
        "operator_agent/input_fidelity.py", "operator_agent/anticheat.py",
        "operator_agent/reference.py", "operator_agent/remote_executor.py",
        "operator_agent/remote_input_fidelity.py", "operator_agent/confirm.py",
        "operator_agent/oracle.py", "operator_agent/backend_caps.py",
        "operator_agent/tolerance.py", "operator_agent/eligibility.py",
        "operator_agent/correctness_boundaries.py", "operator_agent/source_lang.py",
        "operator_agent/resource_check.py", "operator_agent/hardware_caps.py",
        "operator_agent/exec_guard.py", "operator_agent/budget.py", "operator_agent/calibration.py",
        "remote_scripts/run_kernel.py", "remote_scripts/runner_server.py",
        "remote_scripts/run_kernel_xpu.py",
    )
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in files}


def declaration(task, input_sets, controls=None):
    contract = freeze_from_task(task)
    task_dict = task.to_dict()
    task_dict.pop("created_at", None)
    task_dict.pop("task_id", None)
    # TaskSchema.to_dict deliberately omits source paths/reference source.
    # Reference text affects both L2 semantics and L3 baseline selection.
    task_dict["source"]["reference_source"] = task.source.reference_source
    task_dict["source"]["code_path"] = task.source.code_path
    if task.source.code_path:
        task_dict["source"]["code_hash"] = code_hash(
            Path(task.source.code_path).read_text(encoding="utf-8"))
    inputs = [describe_inputs(item) for item in input_sets]
    reference = contract.reference_contract
    suite = contract.correctness_suite
    return json_value(contract.as_dict()), json_value({
        "task_id": task.task_id,
        "contract_hash": contract.hash,
        "reference_hash": reference.digest() if reference is not None else None,
        "suite_hash": suite.digest() if suite is not None else None,
        "task": task_dict, "task_hash": digest(task_dict),
        "inputs": inputs, "workload_hash": digest(inputs),
        "measurement_protocol": protocol_identity(),
        "controls": controls or {},
    })


def create(task, input_sets, environment, controls=None):
    contract, identity = declaration(task, input_sets, controls)
    manifest = dict(identity, schema=SCHEMA, run_id=uuid.uuid4().hex,
                    environment=environment)
    manifest["hash"] = digest(manifest)
    return {"contract": contract, "manifest": manifest}


def atomic_json(path, data, *, readonly=False):
    """Same-directory unique temporary + fsync + replace + directory fsync."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RunBindingError("refuse symlink: " + path.name)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
            if readonly:
                os.fchmod(f.fileno(), 0o444)
        os.replace(tmp, path)
        directory = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _read(path):
    path = Path(path)
    if path.is_symlink() or path.parent.is_symlink():
        raise RunBindingError("symlink sidecar/checkpoint")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RunBindingError("expected JSON object: " + path.name)
    return data


def write(directory, binding):
    """Never bless an existing different/corrupt sidecar by overwriting it."""
    for name, key in ((CONTRACT, "contract"), (MANIFEST, "manifest")):
        path = Path(directory) / name
        if path.exists() or path.is_symlink():
            # Python equality conflates True/1 and False/0; canonical JSON
            # hashes must notice type changes as well as value changes.
            if digest(_read(path)) != digest(binding[key]):
                raise RunBindingError("immutable sidecar drift: " + name)
        else:
            atomic_json(path, binding[key], readonly=True)


def seal(snapshot, binding):
    m = binding["manifest"]
    state = dict(snapshot)
    state.pop("run_binding", None)
    snapshot["run_binding"] = {
        "schema": SCHEMA, "run_id": m["run_id"], "manifest_hash": m["hash"],
        "contract_hash": m["contract_hash"], "workload_hash": m["workload_hash"],
        "protocol_hash": digest(m["measurement_protocol"]),
        "candidate_hash": code_hash(snapshot["current_code"]),
        "pending_fix_hash": (code_hash(snapshot["pending_fix"])
                             if snapshot["pending_fix"] is not None else None),
        "checkpoint_hash": digest(state),
    }
    return snapshot


def load_checkpoint(path, task, input_sets, controls=None):
    """Validate all durable evidence BEFORE provisioning, routing or generation."""
    try:
        path = Path(path)
        snapshot = _read(path)
        binding = {"contract": _read(path.parent / CONTRACT),
                   "manifest": _read(path.parent / MANIFEST)}
        m = binding["manifest"]
        body = dict(m)
        claimed = body.pop("hash", None)
        if m.get("schema") != SCHEMA or claimed != digest(body):
            raise RunBindingError("manifest schema/hash mismatch")
        if not isinstance(m.get("run_id"), str) or len(m["run_id"]) != 32:
            raise RunBindingError("invalid run_id")
        contract, expected = declaration(task, input_sets, controls)
        loaded = load_dict(binding["contract"])
        if loaded.hash != m.get("contract_hash") or binding["contract"] != contract:
            raise RunBindingError("frozen contract mismatch")
        for key, value in expected.items():
            if m.get(key) != value:
                raise RunBindingError("resume identity mismatch: " + key)
        if (snapshot.get("schema_version") != 1
                or snapshot.get("task_id") != task.task_id
                or type(snapshot.get("round_done")) is not int
                or snapshot["round_done"] < 0
                or not isinstance(snapshot.get("current_code"), str)
                or not isinstance(snapshot.get("sketch_dict"), dict)
                or (snapshot.get("pending_fix") is not None
                    and not isinstance(snapshot["pending_fix"], str))):
            raise RunBindingError("invalid checkpoint state")
        original = snapshot.get("run_binding")
        copy = dict(snapshot)
        if original != seal(copy, binding)["run_binding"]:
            raise RunBindingError("checkpoint/candidate binding mismatch")
        if snapshot.get("environment_fingerprint") != m.get("environment"):
            raise RunBindingError("checkpoint environment mismatch")
        return snapshot, binding
    except RunBindingError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise RunBindingError("unreadable/unbound resume: " + type(exc).__name__) from exc


def check_current(binding, task, input_sets, environment, controls=None):
    _, expected = declaration(task, input_sets, controls)
    for key, value in dict(expected, environment=environment).items():
        if binding["manifest"].get(key) != value:
            raise RunBindingError("live run identity mismatch: " + key)
