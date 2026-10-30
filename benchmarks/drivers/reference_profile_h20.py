"""Bounded production Conductor acceptance of native fp16 LayerNorm v2.

Uses owned private scratch and existing H20 runtime; no dependency installation,
no broad process cleanup. Only candidate generation is stubbed. Small L3 windows
may honestly hold: this is NOT a speedup/corpus-admission claim.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.calibration.calibrate_layernorm import candidate_source
from benchmarks.drivers.correctness_profiles_h20 import ProfileSSH, assess, compact_trace, primary
from benchmarks.drivers.input_fidelity_h20 import dumps, validate_scratch
from operator_agent import Conductor, IntentParser, KernelDatabase
from operator_agent.artifact_store import ArtifactStore
from operator_agent.backend_caps import Cap
from operator_agent.contract import freeze_from_task
from operator_agent.correctness import as_suite
from operator_agent.kernel_db import FailureDatabase
from operator_agent.reference_contract import registered_reference, validate_receipt
from operator_agent.run_layout import (new_run_root, run_layout, ROLE_HARNESS, ROLE_ORACLE,
                                        ROLE_REPORT_WRITER)
from operator_agent.schemas import UnifiedSketch, VerificationLevel as Level


def trial_code(variant):
    if variant == "output-fp32":
        return candidate_source("unfused-fp32").replace("device=X.device,dtype=X.dtype", "device=X.device,dtype=torch.float32")
    return candidate_source(variant)


def run(args):
    root = Path(new_run_root(args.output, "reference-profile-h20"))
    os.environ["OPERATOR_AGENT_LOG_DIR"] = str(root / "logs")
    layout = run_layout(str(root))
    backend = ProfileSSH(host=args.host, timeout=120)
    records, cleanup, environment = [], None, None

    def execute(name, code, expected):
        t = IntentParser().parse("layernorm fp16 triton nvidia")
        t.task_id = name
        t.target.required_caps = (Cap.REAL_DEVICE,)
        t.constraints.min_speedup = .01  # Fixture threshold only, not a speed claim.
        suite = as_suite(t.constraints.correctness_suite)
        contract = freeze_from_task(t)
        layout.write_for(ROLE_HARNESS, f"contract/{name}.json", dumps(contract.as_dict()))
        out = root / "artifacts" / name
        db = out / "kernel_db.jsonl"
        c = Conductor(backend=backend, kernel_db=KernelDatabase(str(db), persist_retrievals=False),
                      failure_db=FailureDatabase(str(out / "failure_db.jsonl")),
                      max_optimize_rounds=1, use_coder_pipeline=False)
        c.store = ArtifactStore(str(out))
        generated = []
        def design(*a, **k):
            generated.append("designer")
            return UnifiedSketch(algorithm="layernorm", target_dsl="triton")
        c.designer.run = design
        c.coder = SimpleNamespace(run=lambda *a, **k: code)
        start = len(backend.trace)
        art = c.execute(t, primary("layernorm", "fp16"), target_gflops=0)
        trace = backend.trace[start:]
        persisted = db.exists() and bool(db.stat().st_size)
        l2 = next((r for r in art.verification_results if r.level == Level.L2_FUNCTION), None)
        if name == "output-fp32":
            errors = []
            if (l2 is None or l2.passed or not l2.metrics.get("candidate_semantics_error")
                    or l2.metrics.get("correctness", {}).get("verdict") != "failed"):
                errors.append("wrong output dtype not rejected as candidate failure")
            if any(x["request"]["action"] == "benchmark_with_baselines" for x in trace) or persisted:
                errors.append("wrong output dtype reached L3/corpus")
        else:
            errors = assess(art, expected, suite, trace, persisted)
        if expected == "unknown" and generated:
            errors.append("identity drift reached generation")
        if expected != "unknown":
            for row in trace:
                res = row.get("response", {})
                if row["request"]["action"] in {"verify", "benchmark_with_baselines"}:
                    try:
                        validate_receipt(t.constraints.reference_contract, res.get("reference_receipt"))
                    except ValueError as exc:
                        errors.append(str(exc))
            l3 = next((r for r in art.verification_results if r.level == Level.L3_PERFORMANCE), None)
            if expected == "passed" and (l3 is None or not l3.metrics.get("reference_receipt")
                                          or l3.metrics.get("frozen_contract_hash") != contract.hash):
                errors.append("L3 reference/frozen identity not preserved")
        record = dict(name=name, expected=expected, errors=errors, generated=generated,
                      persisted=persisted, results=[dict(level=r.level.value, passed=r.passed,
                          message=r.message, metrics=r.metrics) for r in art.verification_results],
                      trace=compact_trace(trace))
        records.append(record)
        layout.write_for(ROLE_ORACLE, f"measurements/{name}.json", dumps(record))
        print(f"ACCEPTANCE {name}: {'PASS' if not errors else 'FAIL'} {errors}", flush=True)

    try:
        environment = backend.prepare(args.gpu)
        identity = backend.reference_identity(registered_reference().content_dict())
        validate_receipt(registered_reference(), identity.get("reference_receipt"))
        layout.write_for(ROLE_HARNESS, "harness/environment.json", dumps(dict(
            **environment, reference_identity=identity)))
        for variant in ("unfused-fp32", "centered-fp32", "dense", "tail", "fp16-intermediate", "output-fp32"):
            execute(variant, trial_code(variant), "passed" if variant == "unfused-fp32" else "failed")

        # Honest drift witness: append a comment to OUR deployed primitive only.
        # Shut down the owned peer before mutation; never edit the shared runtime.
        closed = backend.close()
        if closed.get("quiescent") is not True:
            raise RuntimeError("owned peer not quiescent; no drift mutation attempted")
        primitive = validate_scratch(backend.scratch) + "/builtin_layernorm.py"
        original = (ROOT / "operator_agent" / "builtin_layernorm.py").read_bytes()
        backend.remote_python("import json; from pathlib import Path; p=Path(" + repr(primitive) + "); "
                              "p.write_bytes(p.read_bytes()+b'\\n# acceptance drift\\n'); print(json.dumps({'changed':True}))")
        try:
            execute("runtime-drift", trial_code("unfused-fp32"), "unknown")
        finally:
            closed = backend.close()
            if closed.get("quiescent") is not True:
                raise RuntimeError("drift peer not quiescent; scratch retained")
            backend.remote_python("import json; from pathlib import Path; Path(" + repr(primitive) + ").write_bytes(" + repr(original) + "); print(json.dumps({'restored':True}))")
        execute("good-after-drift", trial_code("unfused-fp32"), "passed")
    except (Exception, KeyboardInterrupt) as exc:
        records.append(dict(name="harness", errors=[f"{type(exc).__name__}: {exc}"]))
        print(records[-1], flush=True)
    finally:
        # Cleanup before any fallible report write.
        cleanup = backend.cleanup()
    summary = dict(completed=len(records) == 8 and all(not r["errors"] for r in records),
                   profile=registered_reference().profile, reference_hash=registered_reference().digest(),
                   trials=[dict(name=r["name"], errors=r["errors"]) for r in records],
                   cleanup=cleanup, forbidden_actions=backend.forbidden_actions, run_root=str(root))
    layout.write_for(ROLE_REPORT_WRITER, "report/summary.json", dumps(summary))
    print(dumps(summary), flush=True)
    return 0 if summary["completed"] and not backend.forbidden_actions else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="algorithm-h20")
    p.add_argument("--gpu", type=int)
    p.add_argument("--output", default=str(ROOT / "output" / "reference_profile_h20"))
    raise SystemExit(run(p.parse_args()))
