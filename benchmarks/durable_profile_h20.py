"""Bounded native-v2 workload/admission + durable resume acceptance on H20.

Only generation is stubbed. Original numerical/measurement/admission gates are
unchanged. Honest tiny/noisy windows may HOLD; at least one actual positive
admission is required to claim performance/admission acceptance complete.
No install/sync, shared-runtime edits or broad process cleanup.
"""
from __future__ import annotations

import argparse
import copy
import json
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
from operator_agent.reference_contract import ReferenceIdentityError, registered_reference, validate_receipt
from operator_agent.run_layout import new_run_root
from operator_agent.schemas import UnifiedSketch, VerificationLevel as Level

WORKLOADS = ((1024, 1024), (4096, 4096), (4096, 8192))
EXTENDED_WORKLOADS = WORKLOADS + ((8192, 8192),)


def selected_workloads(args):
    # Selection is fixed before backend preparation or any measurement. A larger
    # primary changes the workload identity, never the measurement ruler.
    if args.resume_only:
        return WORKLOADS[:1]
    if args.workload:
        return tuple(tuple(map(int, spec.split("x"))) for spec in args.workload)
    return WORKLOADS


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
    t.constraints.min_speedup = 1.05  # Declared before ANY measurement.
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


def run(args):
    workloads = selected_workloads(args)
    root = Path(new_run_root(args.output, "durable-profile-h20"))
    root.mkdir(parents=True, exist_ok=False)
    os.environ["OPERATOR_AGENT_LOG_DIR"] = str(root / "logs")
    backend = ProfileSSH(host=args.host, timeout=180)
    records = []
    code = trial_code("unfused-fp32")
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
            if admission.get("kept"):
                m = l3.metrics
                if (m.get("measurement_can_gate") is not True
                        or m.get("measurement_validity") != "valid"
                        or m.get("eligibility", {}).get("can_persist") is not True
                        or not art.remeasure or not art.remeasure.ok
                        or art.remeasure.speedup < 1.05 or after == before):
                    errors.append("positive admission without valid paired confirmation")
            elif after != before:
                errors.append("denied admission nevertheless wrote corpus")
            if resume and (generated or art.resume_status.get("status") != "resumed"
                           or not any(r["request"]["action"] == "compile" for r in trace)):
                errors.append("resume did not freshly compile/verify without generation")
        record = dict(name=name, errors=errors, generated=generated, resume=art.resume_status,
                      saved_dir=art.saved_dir, admission=admission,
                      results=results, trace=compact_trace(trace))
        path = root / "evidence" / (name + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps(record))
        # Large primary wire buffers are already compacted in the durable log.
        del backend.trace[start:]
        records.append(dict(name=name, errors=errors, admission=admission,
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
    positive = [r["name"] for r in records if r.get("admission", {}).get("kept")]
    summary = dict(functional_completed=len(records) == len(workloads) + 3 and all(not r["errors"] for r in records),
        positive_admissions=positive, workload_specs=workloads, min_speedup=1.05,
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
