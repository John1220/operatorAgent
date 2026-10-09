"""Bounded real Conductor acceptance for trusted default L2 profiles.

Uses the existing isolated runner/runtime; no install/sync/LLM credentials.
Good candidates run real L3 + normal qualification/reporting; wrong candidates
must stop before L3. Not a performance comparison or a claim of corpus admission.
Run: python3 benchmarks/correctness_profiles_h20.py --gpu 2
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from benchmarks.input_fidelity_h20 import AcceptanceSSH, dumps, gpu_evidence  # noqa: E402
from operator_agent import Conductor, IntentParser, KernelDatabase  # noqa: E402
from operator_agent.artifact_store import ArtifactStore  # noqa: E402
from operator_agent.backend_caps import Cap  # noqa: E402
from operator_agent.correctness import as_suite  # noqa: E402
from operator_agent.contract import freeze_from_task  # noqa: E402
from operator_agent.input_fidelity import describe_inputs  # noqa: E402
from operator_agent.kernel_db import FailureDatabase  # noqa: E402
from operator_agent.run_layout import (new_run_root, run_layout, ROLE_HARNESS, ROLE_ORACLE, ROLE_REPORT_WRITER)  # noqa: E402
from operator_agent.schemas import UnifiedSketch, VerificationLevel as Level  # noqa: E402

RELU = '''import torch
import triton
import triton.language as tl
@triton.jit
def _op(X,Y,N:tl.constexpr,C:tl.constexpr,SR:tl.constexpr,SC:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    x=tl.load(X+(i//C)*SR+(i%C)*SC,i<N,0).to(tl.float32)
    y=tl.maximum(x,0)
    tl.store(Y+i,y,i<N)
def kernel(X):
    Y=torch.empty(X.shape,device=X.device,dtype=X.dtype)
    _op[(triton.cdiv(X.numel(),256),)](X,Y,X.numel(),X.shape[1],X.stride(0),X.stride(1),256)
    return Y
'''
REDUCE = '''import torch
import triton
import triton.language as tl
@triton.jit
def _op(X,Y,C:tl.constexpr,SR:tl.constexpr,SC:tl.constexpr,B:tl.constexpr):
    r=tl.program_id(0)
    j=tl.arange(0,B)
    x=tl.load(X+r*SR+j*SC,j<C,0).to(tl.float32)
    COMPUTE
    tl.store(Y+r*C+j,y,j<C)
def kernel(X):
    Y=torch.empty(X.shape,device=X.device,dtype=X.dtype)
    _op[(X.shape[0],)](X,Y,X.shape[1],X.stride(0),X.stride(1),triton.next_power_of_2(X.shape[1]))
    return Y
'''
SOFTMAX = REDUCE.replace("    COMPUTE", "    z=tl.where(j<C,x,float('-inf'))\n    e=tl.exp(z-tl.max(z,0))\n    y=e/tl.sum(e,0)")
LAYERNORM = REDUCE.replace("    COMPUTE", "    mu=tl.sum(x,0)/C\n    z=tl.where(j<C,x-mu,0)\n    var=tl.sum(z*z,0)/C\n    y=(x-mu)*tl.rsqrt(var+1e-5)")
MATMUL = '''import torch
import triton
import triton.language as tl
@triton.jit
def _op(A,B,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
        AM:tl.constexpr,AK:tl.constexpr,BK:tl.constexpr,BN:tl.constexpr):
    m=tl.program_id(0)*32+tl.arange(0,32)
    n=tl.program_id(1)*32+tl.arange(0,32)
    k=tl.arange(0,32)
    acc=tl.full((32,32),0,tl.float32)
    for base in range(triton.cdiv(K,32)):
        kk=base*32+k
        a=tl.load(A+m[:,None]*AM+kk[None,:]*AK,(m[:,None]<M)&(kk[None,:]<K),0)
        b=tl.load(B+kk[:,None]*BK+n[None,:]*BN,(kk[:,None]<K)&(n[None,:]<N),0)
        acc+=tl.dot(a,b,input_precision="ieee")
    tl.store(Y+m[:,None]*N+n[None,:],acc,(m[:,None]<M)&(n[None,:]<N))
def kernel(A,B):
    M,K=A.shape
    N=B.shape[1]
    Y=torch.empty((M,N),device=A.device,dtype=A.dtype)
    _op[(triton.cdiv(M,32),triton.cdiv(N,32))](A,B,Y,M,N,K,A.stride(0),A.stride(1),B.stride(0),B.stride(1))
    return Y
'''
GOOD = dict(relu=RELU, matmul=MATMUL, softmax=SOFTMAX, layernorm=LAYERNORM)


def candidate(op, variant):
    code = GOOD[op]
    if variant == "good": return code
    if variant == "dense":
        if op == "matmul": return code.replace("m[:,None]*AM+kk[None,:]*AK", "m[:,None]*K+kk[None,:]").replace("kk[:,None]*BK+n[None,:]*BN", "kk[:,None]*N+n[None,:]")
        return code.replace("(i//C)*SR+(i%C)*SC", "i").replace("r*SR+j*SC", "r*C+j")
    if variant == "tail":
        if op == "matmul": return code.replace("],acc,(m", "],tl.where(n[None,:]<(N//32)*32,acc,0),(m")
        if op == "relu": return code.replace("Y+i,y,i<N", "Y+i,tl.where(i<(N//B)*B,y,0),i<N")
        return code.replace("Y+r*C+j,y,j<C", "Y+r*C+j,tl.where(j<(C//32)*32,y,0),j<C")
    if variant == "precision" and op == "relu":
        return code.replace("y=tl.maximum(x,0)", "y=tl.maximum(x.to(tl.float16).to(tl.float32),0)")
    if variant == "precision" and op == "matmul":
        return code.replace("tl.dot(a,b,input_precision", "tl.dot(a.to(tl.float16),b.to(tl.float16),input_precision")
    raise ValueError("unsupported acceptance mutation")


def trial_specs():
    specs = [(op, dt, v) for op in GOOD for dt in ("fp32", "fp16")
             if (op, dt) != ("layernorm", "fp16") for v in ("good", "dense", "tail")]
    return specs + [(op, "fp32", "precision") for op in ("relu", "matmul")]


def primary(op, dtype):
    # Integral, aligned primary also passes lossy-precision controls. The random
    # fractional suite must kill them. Small fixtures exercise the production L3
    # gate, not throughput; qualification can honestly refuse tiny measurements.
    dt = np.float16 if dtype == "fp16" else np.float32
    if op == "matmul":
        return {"A": np.tile(np.array([-1,0,1,0],dt),(64,16)),
                "B": np.tile(np.array([-1,0,1,0],dt),(64,16))}
    return {"X": np.tile(np.array([-1,0,1,2],dt),(32,16))}


class ProfileSSH(AcceptanceSSH):
    @property
    def capabilities(self):
        # Already probed/deployed: this harness is not permitted to provision or
        # analyze/install dependencies. Numeric and L3 methods remain production.
        return super().capabilities - {Cap.ENV_PROVISION, Cap.DEP_ANALYSIS, Cap.DEPLOY}

    def _ssh_exec_json(self, payload):
        if payload.get("action") not in {"compile","verify","gpu_info","ping","shutdown","benchmark_with_baselines"}:
            self.forbidden_actions.append(payload.get("action"))
            raise RuntimeError("profile acceptance forbids this action")
        row = {"request": payload}; self.trace.append(row)
        from operator_agent.remote_executor import SSHExecutor
        try:
            row["response"] = SSHExecutor._ssh_exec_json(self, payload)
            return row["response"]
        except Exception as exc:
            row["exception"] = repr(exc)
            raise


def compact_trace(trace):
    """Keep hashes and metadata, not hundreds of MiB of primary wire buffers."""
    rows = []
    for item in trace:
        request = dict(item["request"])
        inputs = request.get("inputs")
        if isinstance(inputs, dict) and "storages" in inputs and all(isinstance(s, str) for s in inputs["storages"]):
            request["inputs"] = {**inputs, "storages": [dict(encoded_bytes=len(s),
                encoded_sha256=hashlib.sha256(s.encode("ascii")).hexdigest()) for s in inputs["storages"]]}
        rows.append({**item, "request": request})
    return rows


def assess(artifact, expected, suite, trace, persisted):
    errors = []
    l2 = next((r for r in artifact.verification_results if r.level == Level.L2_FUNCTION), None)
    if l2 is None: return ["no L2 result"]
    c = l2.metrics.get("correctness", {})
    if c.get("verdict") != expected or l2.passed != (expected == "passed"):
        errors.append("unexpected L2 verdict")
    benches = [r for r in trace if r["request"]["action"] == "benchmark_with_baselines"]
    if expected != "passed" and (benches or any(r.level == Level.L3_PERFORMANCE for r in artifact.verification_results) or persisted):
        errors.append("bad/unknown candidate reached L3 or persistent corpus")
    if expected == "unknown":
        if any(r["request"]["action"] in {"verify","compile"} for r in trace): errors.append("unsupported semantics executed")
        return errors
    if c.get("suite_hash") != suite.digest() or c.get("coverage_lint",{}).get("status") != "passed":
        errors.append("suite identity/coverage missing")
    rows = c.get("cases", [])
    if len(rows) != len(suite.cases) or {r["case_id"] for r in rows} != {x.case_id for x in suite.cases}:
        errors.append("skipped/duplicated suite cases")
    if not gpu_evidence(l2.metrics) or l2.metrics.get("max_error",1) > 1e-5:
        errors.append("aligned primary did not pass actual CUDA numeric/fidelity check")
    if expected == "passed":
        for row in rows:
            m, again = row.get("metrics",{}), row.get("repeat_metrics",{})
            if (row.get("verdict") != "passed" or row.get("determinism") != "passed"
                    or not gpu_evidence(m) or not gpu_evidence(again)
                    or not m.get("output_digest") or m["output_digest"] != again.get("output_digest")):
                errors.append("incomplete CUDA case/repeat: " + row["case_id"])
        l3 = next((r for r in artifact.verification_results if r.level == Level.L3_PERFORMANCE), None)
        if not benches or l3 is None or not l3.metrics.get("backend_real_device"):
            errors.append("good candidate never reached real production L3")
        if not any(r.get("response", {}).get("success") is True and r["response"].get("device") == "cuda" for r in benches):
            errors.append("no successful CUDA L3 response")
        if l3 is not None and l3.metrics.get("measurement_can_gate") is False and persisted:
            errors.append("uninformative tiny measurement was persisted")
        # Performance can honestly HOLD/refuse admission; do not alter thresholds
        # to force a corpus success out of a correctness acceptance run.
    elif not any(r.get("verdict") == "failed" and gpu_evidence(r.get("metrics",{})) for r in rows):
        errors.append("mutation not killed numerically on actual CUDA")
    return errors


def run(args):
    root = Path(new_run_root(args.output, "correctness-profiles-h20"))
    os.environ["OPERATOR_AGENT_LOG_DIR"] = str(root / "logs")
    os.environ["OPERATOR_AGENT_LOG_KEEP"] = "100"
    layout = run_layout(str(root))
    backend = ProfileSSH(host=args.host, timeout=120)
    records, cleanup = [], None
    try:
        environment = backend.prepare(args.gpu)
        layout.write_for(ROLE_HARNESS, "harness/environment.json", dumps(environment))
        specs = trial_specs() + [("attention","fp32","unknown"),("relu","bf16","unknown"),("layernorm","fp16","unknown")]
        for op, dtype, variant in specs:
            name = f"{op}-{dtype}-{variant}"
            t = IntentParser().parse(f"{op} {dtype} triton nvidia")
            if (op, dtype, variant) == ("layernorm", "fp16", "unknown"):
                # This historical acceptance covers v1; v2 has its own driver.
                t.constraints.correctness_profile = "builtin-l2-v1"
                t.constraints.correctness_suite = None
                t.constraints.reference_contract = None
            t.task_id = name
            t.target.required_caps = (Cap.REAL_DEVICE,)
            t.constraints.min_speedup = 0.01  # Test fixture threshold, not a speed claim.
            inputs = primary(op if op in GOOD else "relu", dtype)
            suite = as_suite(t.constraints.correctness_suite) if t.constraints.correctness_suite is not None else None
            if suite: layout.write_for(ROLE_HARNESS, f"contract/{name}.json", dumps(freeze_from_task(t).as_dict()))
            before = describe_inputs(inputs)
            out = root / "artifacts" / name
            db = out / "kernel_db.jsonl"
            c = Conductor(backend=backend, kernel_db=KernelDatabase(str(db), persist_retrievals=False),
                failure_db=FailureDatabase(str(out / "failure_db.jsonl")), max_optimize_rounds=1, use_coder_pipeline=False)
            c.store = ArtifactStore(str(out))
            # Only candidate production is stubbed, not verification or promotion.
            c.designer.run = lambda *a: UnifiedSketch(algorithm=op, target_dsl="triton")
            code = candidate(op, variant) if variant != "unknown" else RELU
            c.coder = SimpleNamespace(run=lambda *a, **k: code)
            start = len(backend.trace)
            art = c.execute(t, inputs, target_gflops=0)
            trace = backend.trace[start:]
            expected = "unknown" if variant == "unknown" else "passed" if variant == "good" else "failed"
            errors = assess(art, expected, suite, trace, db.exists() and bool(db.stat().st_size))
            if describe_inputs(inputs) != before: errors.append("controller inputs mutated")
            if suite:
                saved = list(out.rglob("result.json"))
                if len(saved) != 1: errors.append("production artifact not saved")
                else:
                    stored = json.loads(saved[0].read_text())
                    if as_suite(stored["task"]["constraints"]["correctness_suite"]).digest() != suite.digest(): errors.append("saved suite differs")
                    report = saved[0].with_name("report.md")
                    if not report.exists() or "builtin-l2-v1" not in report.read_text(): errors.append("profile missing from production report")
            record = dict(name=name, expected=expected, accepted=not errors, errors=errors,
                results=[dict(dataclasses.asdict(r), level=r.level.value) for r in art.verification_results],
                l3_requests=sum(r["request"]["action"] == "benchmark_with_baselines" for r in trace),
                corpus_admission=(art.metadata or {}).get("corpus_admission"),
                all_passed=art.all_passed)
            records.append(record)
            layout.write_for(ROLE_ORACLE, f"measurements/{name}.json", dumps(dict(record=record,trace=compact_trace(trace))))
            backend.trace.clear()  # Bound controller memory across trials.
            print("TRIAL " + dumps({k:record[k] for k in ("name","accepted","errors","l3_requests","all_passed")}), flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        records.append(dict(name="driver-error", accepted=False, errors=[repr(exc)]))
    finally:
        try: cleanup = backend.cleanup()
        except Exception as exc: cleanup = {"error": repr(exc)}
        ok = len(records) == len(trial_specs()) + 3 and all(r["accepted"] for r in records) and cleanup.get("quiescent") is True and cleanup.get("removed") is True and not backend.forbidden_actions
        summary = dict(accepted=ok, trials=records, cleanup=cleanup, forbidden_actions=backend.forbidden_actions,
            scope="Real CUDA L2 + production Conductor/L3/report; stub candidate producer, not LLM evaluation or speed/admission claim")
        layout.write_for(ROLE_REPORT_WRITER, "report/summary.json", dumps(summary))
        print("SUMMARY " + dumps(dict(accepted=ok, trials=len(records), cleanup=cleanup, run_root=str(root))), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default=None)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--output", default=str(ROOT / "output" / "correctness_profiles_h20"))
    sys.exit(run(p.parse_args()))
