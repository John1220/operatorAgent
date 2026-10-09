"""Bounded native-v2 workload/admission + durable resume acceptance on H20.

Only generation is stubbed. Original numerical/measurement/admission gates are
unchanged. Honest tiny/noisy windows may HOLD; at least one actual positive
admission is required to claim performance/admission acceptance complete.
No install/sync, shared-runtime edits or broad process cleanup.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from benchmarks.correctness_profiles_h20 import ProfileSSH, compact_trace
from benchmarks.input_fidelity_h20 import dumps
from benchmarks.reference_profile_h20 import trial_code
from operator_agent import Conductor, IntentParser, KernelDatabase
from operator_agent.artifact_store import ArtifactStore
from operator_agent.backend_caps import Cap
from operator_agent.kernel_db import FailureDatabase
from operator_agent.input_fidelity import MAX_STORAGE_BYTES
from operator_agent.reference_contract import ReferenceIdentityError, registered_reference, validate_receipt
from operator_agent.run_contract import atomic_json, code_hash, digest
from operator_agent.run_layout import new_run_root
from operator_agent.schemas import UnifiedSketch, VerificationLevel as Level
from operator_agent.session_inputs import PROTOCOL as INPUT_CACHE_PROTOCOL

WORKLOADS = ((1024, 1024), (4096, 4096), (4096, 8192))
EXTENDED_WORKLOADS = WORKLOADS + ((8192, 8192), (8192, 15360), (8192, 12288))
MIN_SPEEDUP = 1.05
RPC_TIMEOUT = 180


def selected_workloads(args):
    # Selection is fixed before backend preparation or any measurement. A larger
    # primary changes the workload identity, never the measurement ruler.
    if args.resume_only and args.workload:
        raise ValueError("resume-only and explicit workloads are mutually exclusive")
    workloads = (WORKLOADS[:1] if args.resume_only else
        tuple(tuple(map(int, spec.split("x"))) for spec in args.workload) if args.workload else WORKLOADS)
    if (not workloads or len(workloads) > len(EXTENDED_WORKLOADS)
            or len(set(workloads)) != len(workloads)
            or any(shape not in EXTENDED_WORKLOADS for shape in workloads)):
        raise ValueError("workloads must be distinct members of the bounded whitelist")
    # The exact codec counts scalar JSON bytes too (empty scalar map = 2).
    if any(rows * cols * np.dtype(np.float16).itemsize + 2 > MAX_STORAGE_BYTES
           for rows, cols in workloads):
        raise ValueError("workload exceeds exact input protocol storage/scalar limit")
    return workloads


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="algorithm-h20")
    p.add_argument("--gpu", type=int)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--resume-only", action="store_true", help="recheck durable identity/resume only; not performance acceptance")
    group.add_argument("--workload", action="append", choices=["x".join(map(str, shape)) for shape in EXTENDED_WORKLOADS],
                       help="explicit bounded workload (repeatable); default is the original three shapes")
    p.add_argument("--output", default=str(ROOT / "output" / "durable_profile_h20"))
    return p


def primary(shape):
    # Declared, deterministic workload; no specialization in the candidate.
    return {"X": np.tile(np.array([-1, 0, 1, 2], np.float16),
                         (shape[0], shape[1] // 4))}


def task(name):
    t = IntentParser().parse("layernorm fp16 triton nvidia")
    t.task_id = name
    t.target.required_caps = (Cap.REAL_DEVICE,)
    t.constraints.min_speedup = MIN_SPEEDUP  # Declared before ANY measurement.
    return t


def receipt_errors(reference, trace):
    """Keep an unsuccessful/unknown RPC as trial evidence, not a lost trial.

    A missing receipt never passes acceptance. Collecting the error instead of
    raising allows compact trace/results to be written before owned cleanup.
    """
    errors = []
    for row in trace:
        action = row["request"]["action"]
        if action in {"verify", "benchmark_with_baselines"}:
            try:
                validate_receipt(reference, row.get("response", {}).get("reference_receipt"))
            except ReferenceIdentityError as exc:
                errors.append(f"{action} reference identity unknown: {exc}")
    return errors


def transfer_errors(trace):
    errors = []
    for row in trace:
        request = row["request"]
        packet = request.get("inputs", {})
        if not isinstance(packet, dict) or packet.get("protocol") != INPUT_CACHE_PROTOCOL:
            continue
        proof = row.get("response", {}).get("input_transfer") or {}
        if (proof.get("content_verified") is not True or proof.get("manifest_hash") != packet["hash"]
                or proof.get("protocol") != INPUT_CACHE_PROTOCOL
                or proof.get("storage_bytes") != sum(r["bytes"] for r in packet["storages"])
                or type(proof.get("uploaded_bytes")) is not int or proof["uploaded_bytes"] < 0):
            errors.append(request["action"] + " missing complete session-input proof")
        elif request["action"] == "benchmark_with_baselines" and proof["uploaded_bytes"] != 0:
            errors.append("L3/confirmation retransmitted verified input bytes")
    return errors


def workload_plan(workloads, code):
    """Persist the complete, non-adaptive experiment before any remote work.

    Storage estimates bound input bytes, not host RSS/total device memory. The
    normal frozen run manifest will bind actual bytes and the evaluator as well.
    """
    body = dict(workloads=[dict(shape=[rows, cols], input_storage_bytes=rows * cols * 2)
                          for rows, cols in workloads],
        input_pattern=dict(values=[-1, 0, 1, 2], dtype="float16", layout="contiguous"),
        input_storage_scalar_limit=MAX_STORAGE_BYTES, scalar_wire_bytes=2,
        candidate_hash=code_hash(code), min_speedup=MIN_SPEEDUP, rpc_timeout=RPC_TIMEOUT,
        reference_hash=registered_reference().digest(),
        performance_scope="each declared primary separately; not an aggregate or all-shape claim",
        adaptive_workload_selection=False)
    return dict(body, hash=digest(body))


def confirmation_errors(artifact, metrics):
    """Audit the actual durable confirmation, without changing admission rules.

    Production permits a usable degraded second window; the FIRST must be
    valid. Never promote an uninformative/missing/NaN confirmation fixture.
    """
    try:
        record = artifact.remeasure
        saved = json.loads((Path(artifact.saved_dir) / "confirmation.json").read_text())
        first, second = saved["first_binding"], saved["second_binding"]
        if saved["confirmation"] != asdict(record) or first != metrics["run_binding"]:
            raise ValueError("confirmation record/first binding drift")
        identity_keys = ("run_id", "manifest_hash", "contract_hash", "workload_hash",
                         "protocol_hash", "candidate_hash")
        if (any(not first.get(k) or first[k] != second.get(k) for k in identity_keys)
                or first["candidate_hash"] != code_hash(artifact.code)
                or not first.get("verification_id") or not second.get("verification_id")
                or first["verification_id"] == second["verification_id"]
                or record.contract_hash != first["contract_hash"]):
            raise ValueError("confirmation identity/independence mismatch")
        speedup = record.speedup
        if (record.ok is not True or record.independent is not True
                or record.validity not in {"valid", "degraded"}
                or type(speedup) not in {int, float} or not math.isfinite(speedup)
                or speedup < MIN_SPEEDUP):
            raise ValueError("confirmation not usable/above threshold")
        second_metrics = saved["second_metrics"]
        first_window = metrics["arm_windows"]["candidate"]
        second_window = second_metrics["arm_windows"]["candidate"]
        if (not first_window or not second_window or first_window == second_window
                or second_metrics["run_binding"] != second
                or second_metrics["contract_hash"] != record.contract_hash
                or second_metrics["measurement_validity"] != record.validity
                or second_metrics["speedup_vs_baseline"] != speedup):
            raise ValueError("confirmation window/metrics drift")
    except (AttributeError, OSError, ValueError, KeyError, TypeError) as exc:
        return ["positive admission confirmation audit failed: " + str(exc)]
    return []


def corpus_readback(path, before, artifact, task, metrics):
    """An admission flag/file growth is not proof this candidate reached disk."""
    try:
        after = Path(path).read_bytes()
        if not after.startswith(before) or len(after) <= len(before):
            raise ValueError("corpus not durably appended")
        events = [json.loads(line) for line in after[len(before):].splitlines() if line.strip()]
        if len(events) != 1 or events[0].get("event") != "kernel":
            raise ValueError("expected one actual kernel event")
        stored = events[0]["record"]
        expected = dict(kernel_id=artifact.kernel_id, algorithm=task.source.kernel_type,
            hardware=task.target.hardware, dsl=task.target.dsl, code=artifact.code,
            measurement_validity="valid", score=metrics["speedup_vs_baseline"],
            compile_info=artifact.compile_info)
        if any(stored.get(k) != v for k, v in expected.items()):
            raise ValueError("persisted candidate/task/measurement mismatch")
        if (type(stored["score"]) not in {int, float} or not math.isfinite(stored["score"])
                or stored["score"] <= MIN_SPEEDUP
                or stored["compile_info"].get("run_binding") != metrics["run_binding"]
                or metrics["run_binding"]["candidate_hash"] != code_hash(stored["code"])):
            raise ValueError("persisted run/candidate binding mismatch")
        replay = KernelDatabase(str(path), persist_retrievals=False)
        loaded = replay.get(artifact.kernel_id)
        if replay.disk_skipped_lines or loaded is None or loaded.to_dict() != stored:
            raise ValueError("fresh database replay did not recover exact record")
        return dict(verified=True, kernel_id=loaded.kernel_id, candidate_hash=code_hash(loaded.code),
            run_binding=loaded.compile_info["run_binding"], measurement_validity=loaded.measurement_validity,
            score=loaded.score, appended_events=len(events), fresh_replay=True), []
    except (AttributeError, OSError, ValueError, KeyError, TypeError) as exc:
        return dict(verified=False), ["positive admission corpus readback failed: " + str(exc)]


def run(args):
    workloads = selected_workloads(args)
    root = Path(new_run_root(args.output, "durable-profile-h20"))
    root.mkdir(parents=True, exist_ok=False)
    os.environ["OPERATOR_AGENT_LOG_DIR"] = str(root / "logs")
    backend = ProfileSSH(host=args.host, timeout=RPC_TIMEOUT)
    records = []
    code = trial_code("unfused-fp32")
    plan = workload_plan(workloads, code)
    atomic_json(root / "plan.json", plan, readonly=True)
    print("PLAN " + dumps(plan) + "\nRUN_ROOT " + str(root), flush=True)
    db_path = root / "corpus" / "kernel_db.jsonl"

    def conductor(name, rounds, generation=True):
        c = Conductor(backend=backend,
            kernel_db=KernelDatabase(str(db_path), persist_retrievals=False),
            failure_db=FailureDatabase(str(root / "failures" / "failure_db.jsonl")),
            max_optimize_rounds=rounds, use_coder_pipeline=False)
        c.store = ArtifactStore(str(root / "artifacts" / name))
        generated = []
        def design(*a, **k):
            if not generation:
                raise AssertionError("resume/hold reached Designer")
            generated.append("designer")
            return UnifiedSketch(algorithm="layernorm", target_dsl="triton")
        def generate(*a, **k):
            if not generation:
                raise AssertionError("resume/hold reached Coder")
            generated.append("coder")
            return code
        c.designer.run = design
        c.coder = SimpleNamespace(run=generate)
        return c, generated

    def execute(name, t, inputs, *, resume=None, rounds=1, generation=True, expected="numeric"):
        c, generated = conductor(name, rounds, generation)
        start = len(backend.trace)
        before = db_path.read_bytes() if db_path.exists() else b""
        art = c.execute(t, inputs, target_gflops=0, resume_from=resume)
        trace = backend.trace[start:]
        results = [dict(level=r.level.value, passed=r.passed, message=r.message,
                        metrics=r.metrics) for r in art.verification_results]
        l2 = next((r for r in art.verification_results if r.level == Level.L2_FUNCTION), None)
        l3 = next((r for r in art.verification_results if r.level == Level.L3_PERFORMANCE), None)
        after = db_path.read_bytes() if db_path.exists() else b""
        errors = []
        persisted = None
        admission = (art.metadata or {}).get("corpus_admission", {})
        if expected == "hold":
            if not l2 or not l2.metrics.get("run_identity_unknown") or art.all_passed:
                errors.append("missing fail-closed identity hold")
            if trace or generated or after != before:
                errors.append("HOLD had RPC/generation/corpus side effects")
        else:
            if not l2 or not l2.passed or l2.metrics.get("correctness", {}).get("verdict") != "passed":
                errors.append("v2 primary + full suite not passed")
            if not l3 or not l3.metrics.get("run_binding"):
                errors.append("missing actual L3/binding evidence")
            errors.extend(receipt_errors(t.constraints.reference_contract, trace))
            errors.extend(transfer_errors(trace))
            if admission.get("kept") and l3 is not None:
                m = l3.metrics
                if (m.get("measurement_can_gate") is not True
                        or m.get("measurement_validity") != "valid"
                        or m.get("eligibility", {}).get("can_persist") is not True
                        or not art.remeasure or not art.remeasure.ok
                        or type(art.remeasure.speedup) not in {int, float}
                        or not math.isfinite(art.remeasure.speedup)
                        or art.remeasure.speedup < MIN_SPEEDUP or after == before):
                    errors.append("positive admission without valid paired confirmation")
                errors.extend(confirmation_errors(art, m))
                persisted, readback_errors = corpus_readback(db_path, before, art, t, m)
                errors.extend(readback_errors)
            elif admission.get("kept"):
                errors.append("positive admission missing L3 evidence")
            elif after != before:
                errors.append("denied admission nevertheless wrote corpus")
            if resume and (generated or (art.resume_status or {}).get("status") != "resumed"
                           or not any(r["request"]["action"] == "compile" for r in trace)):
                errors.append("resume did not freshly compile/verify without generation")
        record = dict(name=name, errors=errors, generated=generated, resume=art.resume_status,
                      saved_dir=art.saved_dir, admission=admission, corpus_readback=persisted,
                      confirmation=asdict(art.remeasure) if art.remeasure else None,
                      results=results, trace=compact_trace(trace))
        path = root / "evidence" / (name + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps(record))
        # Large primary wire buffers are already compacted in the durable log.
        del backend.trace[start:]
        records.append(dict(name=name, errors=errors, admission=admission,
            corpus_readback=persisted,
            speedup=(l3.metrics.get("speedup_vs_baseline") if l3 else None),
            validity=(l3.metrics.get("measurement_validity") if l3 else None),
            can_gate=(l3.metrics.get("measurement_can_gate") if l3 else None),
            remeasure_speedup=getattr(art.remeasure, "speedup", None), saved_dir=art.saved_dir))
        print("ACCEPTANCE " + name + ": " + dumps(records[-1]), flush=True)
        return art

    try:
        environment = backend.prepare(args.gpu)
        identity = backend.reference_identity(registered_reference().content_dict())
        validate_receipt(registered_reference(), identity.get("reference_receipt"))
        (root / "environment.json").write_text(dumps(dict(environment=environment, identity=identity)))
        first_task = None
        first = None
        for shape in workloads:
            name = "workload-" + "x".join(map(str, shape))
            t = task(name)
            art = execute(name, t, primary(shape))
            if first is None:
                first_task, first = t, art
        # Separate Conductor/Verifier instance: no compile cache or old results.
        execute("resume-positive", first_task, primary(workloads[0]),
                resume=first.saved_dir, rounds=2, generation=False)
        changed = primary(workloads[0])
        changed["X"][0, 0] += 1
        execute("resume-input-drift", first_task, changed,
                resume=first.saved_dir, rounds=3, generation=False, expected="hold")
        altered = copy.deepcopy(first_task)
        # A still-valid trusted v2 declaration, but not this resume's frozen
        # contract. Test durable consistency, not rejection of a looser profile.
        altered.constraints.max_error *= .5
        execute("resume-threshold-drift", altered, primary(workloads[0]),
                resume=first.saved_dir, rounds=3, generation=False, expected="hold")
    except (Exception, KeyboardInterrupt) as exc:
        records.append(dict(name="harness", errors=[f"{type(exc).__name__}: {exc}"]))
        print(records[-1], flush=True)
    finally:
        cleanup = backend.cleanup()
        print("CLEANUP " + dumps(cleanup), flush=True)
    positive = [r["name"] for r in records if r.get("admission", {}).get("kept")
                and (r.get("corpus_readback") or {}).get("verified") is True and not r["errors"]]
    summary = dict(functional_completed=len(records) == len(workloads) + 3 and all(not r["errors"] for r in records),
        positive_admissions=positive, workload_specs=workloads, min_speedup=MIN_SPEEDUP,
        plan_hash=plan["hash"], performance_scope=plan["performance_scope"],
        acceptance_mode="resume-only" if args.resume_only else "representative",
        reference_hash=registered_reference().digest(), trials=records,
        cleanup=cleanup, forbidden_actions=backend.forbidden_actions, run_root=str(root))
    summary["completed"] = bool(summary["functional_completed"] and positive
        and cleanup.get("quiescent") is True and cleanup.get("removed") is True
        and not backend.forbidden_actions)
    (root / "summary.json").write_text(dumps(summary))
    print(dumps(summary), flush=True)
    return 0 if summary["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
