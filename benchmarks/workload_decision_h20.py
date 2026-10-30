"""Fixed, bounded workload-decision acceptance; generation alone is stubbed.

Safe default (local only, no input allocation, SSH or upload)::
    python3 -B benchmarks/workload_decision_h20.py
    python3 -B -m pytest -q -p no:cacheprovider tests/unit/test_workload_decision_driver.py

ONLY after separate authorization to use the existing remote runtime::
    python3 -B benchmarks/workload_decision_h20.py --execute --host algorithm-h20 \
        --gpu 2

Optional --cuda-ruler-profile / --cuda-calibration are frozen before preparation.
ProfileSSH then deploys ONLY into its owned random scratch; it never installs,
syncs dependencies, edits the shared runtime, or kills shared processes. Large
inputs are 240/192 MiB per transfer, measured sequentially, NOT one 432 MiB RPC.
The input cache is only 256 MiB: eviction/re-upload is legitimate, not a failed
zero-upload assertion. This is an input-storage bound, NOT a total RSS/VRAM bound.

One candidate, one main window per case and at most one production confirmation;
no window selection, retries, threshold changes, or adaptive shape selection.
HOLD/failed performance is evidence, not H20 acceptance. Regression-over-limit
is an explicitly LOCAL policy fixture: no fabricated machine timings. The
dispatch deadline, RPC/command caps and existing per-call timeouts bound remote
work; cleanup has a separate allowance and must prove owned-peer quiescence.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, is_dataclass
import inspect
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.correctness_profiles_h20 import ProfileSSH, assess, compact_trace
from benchmarks.durable_profile_h20 import primary, receipt_errors
from benchmarks.input_fidelity_h20 import dumps, validate_scratch
from benchmarks.reference_profile_h20 import trial_code
from operator_agent import Conductor, IntentParser, KernelDatabase
from operator_agent.artifact_store import ArtifactStore
from operator_agent.backend_caps import Cap
from operator_agent.bench_cases import BenchmarkCase, TensorSpec
from operator_agent.contract import freeze_from_task, case_key
from operator_agent.correctness import as_suite
from operator_agent.confirm import confirm_metrics
from operator_agent.corpus_journal import confirmed_key, admission_id, record_hash, scan
from operator_agent.eligibility import (ALL_FALSE, ALL_TRUE, aggregate_speedup,
    case_speedups, decision_speedup, implementation_decision, performance_verdict, regression_over_limit)
from operator_agent.input_fidelity import MAX_STORAGE_BYTES
from operator_agent.kernel_db import FailureDatabase
from operator_agent.placement import observed_pairing, SCOPE_SAME
from operator_agent.reference_contract import registered_reference, validate_receipt
from operator_agent.run_contract import atomic_json, audit_value, code_hash, digest, json_value, protocol_identity
from operator_agent.schemas import UnifiedSketch, VerificationLevel as Level
from operator_agent.session_inputs import PROTOCOL as INPUT_PROTOCOL
from operator_agent.workload import confirmation_evidence

LARGE = ((8192, 15360), (8192, 12288))
TINY = (32, 64)
MIN_SPEEDUP = 1.05
MAX_REGRESSION = .05
RPC_TIMEOUT = 180
# Three v2 suites alone need 3 * 32 * 2 repeats * 2 RPCs = 384 calls:
# reference identity preflight AND verification. Reserve bounded headroom for
# primaries, compile, paired windows, native heartbeat and telemetry.
# This harness request budget is NOT the exact-input protocol size limit.
MAX_RPCS = 512
MAX_COMMANDS = 512
DISPATCH_SECONDS = 1800
CLEANUP_SECONDS = 60
TRIAL_NAMES = ("two-large", "tiny-required-weighted", "missing-declared-case", "resume-weight-drift")
MASK_KEYS = tuple(ALL_FALSE.as_dict())


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--execute", action="store_true", help="explicitly permit remote preparation/execution (default: local plan only)")
    p.add_argument("--host", default="algorithm-h20")
    p.add_argument("--gpu", type=int, choices=range(8), help="explicit physical GPU required with --execute")
    p.add_argument("--output", default=str(ROOT / "output" / "workload_decision_h20"))
    p.add_argument("--continue-negatives-from", help="run ONLY unfinished negatives; existing positive is read-only identity context, never a live credential")
    p.add_argument("--negative-trial", action="append", choices=TRIAL_NAMES[1:], help="explicit frozen unfinished negative subset; requires --continue-negatives-from")
    ruler = p.add_mutually_exclusive_group()
    ruler.add_argument("--cuda-calibration")
    ruler.add_argument("--cuda-ruler-profile")
    return p


def task(name, shapes=LARGE):
    if tuple(shapes) not in (LARGE, LARGE + (TINY,)):
        raise ValueError("only the fixed bounded acceptance cases are permitted")
    t = IntentParser().parse("layernorm fp16 triton nvidia")
    t.task_id = name
    t.target.required_caps = (Cap.REAL_DEVICE,)
    cases = tuple(BenchmarkCase(name="x".join(map(str, s)),
        tensors=(TensorSpec("X", s, "fp16"),), seed=0) for s in shapes)
    t.constraints.cases = tuple(c.to_dict() for c in cases)
    t.constraints.required_cases = tuple(c.key() for c in cases)
    t.constraints.case_weights = {c.key(): 1.0 for c in cases}
    t.constraints.max_regression = {c.key(): MAX_REGRESSION for c in cases}
    t.constraints.aggregate = "weighted_geomean"
    t.constraints.min_speedup = MIN_SPEEDUP
    return t


def tasks():
    positive = task(TRIAL_NAMES[0])
    drift = copy.deepcopy(positive)  # Same task_id and inputs; ONLY weights drift.
    drift.constraints.case_weights[positive.constraints.required_cases[1]] = 2.0
    return (positive, task(TRIAL_NAMES[1], LARGE + (TINY,)), task(TRIAL_NAMES[2]), drift)


def ruler(args):
    from operator_agent.cuda_calibration import read_calibration
    from operator_agent.cuda_ruler_profile import read_profile
    profile = read_profile(args.cuda_ruler_profile) if args.cuda_ruler_profile else None
    calibration = profile["calibration"] if profile else (
        read_calibration(args.cuda_calibration) if args.cuda_calibration else None)
    if calibration and args.gpu != calibration["identity"]["selected_gpu"]:
        raise ValueError("ruler requires its frozen physical GPU")
    return profile, calibration


def evaluator_sources():
    # Keep the complete production manifest protocol, AND files omitted by older
    # protocol_identity implementations (notably the new workload composition).
    paths = list((ROOT / "operator_agent").glob("*.py")) + list((ROOT / "remote_scripts").glob("*.py"))
    paths += [ROOT / "benchmarks" / n for n in ("workload_decision_h20.py", "durable_profile_h20.py",
        "correctness_profiles_h20.py", "input_fidelity_h20.py", "reference_profile_h20.py",
        "calibrate_layernorm.py", "_scratch.py")]
    return {str(p.relative_to(ROOT)): code_hash(p.read_text()) for p in sorted(paths)}


def workload_plan(args, declared_tasks, code, profile=None, calibration=None):
    rows = []
    for name, t in zip(TRIAL_NAMES, declared_tasks):
        contract = freeze_from_task(t)
        # FrozenContract sorts its keys; the RPC order is deliberately the caller
        # order (15360 then 12288), not that canonical serialization order.
        supplied = list(t.constraints.required_cases)
        if name == TRIAL_NAMES[2]:
            supplied = supplied[:1]  # Omit the second large case, NOT a broken kernel.
        rows.append(dict(name=name, task_id=t.task_id, contract=contract.as_dict(), supplied_cases=supplied,
            resume_from_trial=TRIAL_NAMES[0] if name == TRIAL_NAMES[3] else None,
            expected=("positive-or-honest-hold", "launch-limited-hold", "missing-evidence", "identity-hold")[len(rows)]))
    # No big arrays are allocated while freezing. Seed is retained in the case
    # declaration; caller-supplied aligned bytes, NOT materialize(seed), are used.
    bounds = [dict(shape=list(s), input_storage_bytes=s[0] * s[1] * 2,
                   storage_plus_scalars_bytes=s[0] * s[1] * 2 + 2) for s in LARGE + (TINY,)]
    if any(r["storage_plus_scalars_bytes"] >= MAX_STORAGE_BYTES for r in bounds):
        raise ValueError("each exact input packet must be strictly below 256 MiB")
    signature = inspect.signature(ProfileSSH.benchmark_with_baselines)
    body = dict(schema="workload-decision-h20-v1", candidate=dict(variant="unfused-fp32", source=code, hash=code_hash(code)),
        reference=dict(content=registered_reference().content_dict(), hash=registered_reference().digest()),
        measurement_protocol=protocol_identity(), evaluator_sources=evaluator_sources(),
        controls=dict(host=args.host, selected_gpu=args.gpu, target_gflops=0,
            heartbeat_interval=1 if calibration else 5,
            cuda_calibration=calibration, cuda_ruler_profile=profile,
            baseline_priority=["pytorch_eager"],
            warmup=signature.parameters["warmup"].default, repeat=signature.parameters["repeat"].default,
            max_optimize_rounds=1, resume_max_optimize_rounds=2,
            rpc_timeout=RPC_TIMEOUT, max_rpcs=MAX_RPCS, max_commands=MAX_COMMANDS,
            dispatch_seconds=DISPATCH_SECONDS, cleanup_seconds=CLEANUP_SECONDS),
        trials=rows, input_bounds=bounds, input_storage_scalar_limit=MAX_STORAGE_BYTES,
        input_pattern=dict(source="caller", values=[-1, 0, 1, 2], dtype="float16", layout="contiguous"),
        regression_fixture=dict(source="local_fixture_only", case_speedups=[2.0, .90],
            max_regression=MAX_REGRESSION, min_speedup=MIN_SPEEDUP),
        sequential_cases=True, adaptive=False, retry_windows=False,
        performance_scope="two declared large cases, equal-weight geometric mean; not all-shape/general H20 performance")
    continuation = getattr(args, "continue_negatives_from", None)
    selected = getattr(args, "negative_trial", None)
    if selected and not continuation:
        raise ValueError("negative subset requires positive identity context")
    if continuation:
        indexes = [TRIAL_NAMES.index(name) for name in selected] if selected else [1, 2, 3]
        if indexes != sorted(set(indexes)):
            raise ValueError("negative subset must be distinct and in frozen trial order")
        body["execution_indexes"] = indexes
        body["positive_context"] = positive_context(continuation, declared_tasks[0], code)
    return json_value(dict(body, hash=digest(body)))


def check_plan(path, plan):
    if json.loads(Path(path).read_text()) != plan or plan["hash"] != digest({k:v for k,v in plan.items() if k != "hash"}):
        raise ValueError("frozen plan drift")
    if protocol_identity() != plan["measurement_protocol"] or evaluator_sources() != plan["evaluator_sources"]:
        raise ValueError("frozen evaluator/protocol drift")


class BoundedProfileSSH(ProfileSSH):
    """Transport-only safety limits; never modifies measurement/decision policy."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.deadline = time.monotonic() + DISPATCH_SECONDS
        self.counts = dict(rpc=0, command=0)
        self.cleaning = False

    def guard(self, kind):
        limit = 64 if self.cleaning else (MAX_RPCS if kind == "rpc" else MAX_COMMANDS)
        if self.counts[kind] >= limit or time.monotonic() >= self.deadline:
            raise TimeoutError("fixed transport/deadline bound exhausted: " + kind)
        self.counts[kind] += 1
        self.timeout = min(CLEANUP_SECONDS if self.cleaning else RPC_TIMEOUT,
                           max(.001, self.deadline - time.monotonic()))

    def _ssh_exec_json(self, payload):
        self.guard("rpc")
        return super()._ssh_exec_json(payload)

    def _ssh_exec(self, remote_cmd, stdin_data=None):
        self.guard("command")
        return super()._ssh_exec(remote_cmd, stdin_data)

    def _scp(self, local, remote_path):
        self.guard("command")
        return super()._scp(local, remote_path)  # Existing scp has a 60s timeout.

    def cleanup(self):
        self.work_counts = dict(self.counts)
        self.cleaning = True
        self.counts = dict(rpc=0, command=0)
        self.deadline = time.monotonic() + CLEANUP_SECONDS
        return super().cleanup()


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def transfer_errors(trace):
    errors = []
    for row in trace:
        packet = row["request"].get("inputs") or {}
        if packet.get("protocol") != INPUT_PROTOCOL:
            continue
        proof = row.get("response", {}).get("input_transfer") or {}
        size = sum(r["bytes"] for r in packet["storages"])
        # Alternating the two large cases evicts the previous storage. Real
        # verified re-uploads must not be confused with reused timing windows.
        if (size + len(json.dumps(packet["scalars"]).encode()) >= MAX_STORAGE_BYTES
                or proof.get("protocol") != INPUT_PROTOCOL or proof.get("manifest_hash") != packet["hash"]
                or proof.get("storage_bytes") != size or proof.get("content_verified") is not True
                or type(proof.get("uploaded_bytes")) is not int or not 0 <= proof["uploaded_bytes"] <= size):
            errors.append("missing/bounded session input proof: " + row["request"]["action"])
    return errors


def metrics_errors(metrics, contract, supplied, *, positive=False):
    """Read-only acceptance assertions; do not replace the production Oracle."""
    errors, windows = [], []
    ledger = metrics.get("cases") or {}
    scope = sorted(set(contract.required_cases) | {k for k,_ in contract.case_weights}
                   | {k for k,_ in contract.max_regression})
    if set(ledger) != set(supplied):
        errors.append("missing/extra measured case ledger")
    masks, tiers = [], []
    for key in scope:
        m = ledger.get(key) or {}
        if key not in supplied:
            masks.append(ALL_FALSE.as_dict()); tiers.append("uninformative")
            continue
        mask, obs = m.get("eligibility") or {}, m.get("observation") or {}
        tier = m.get("measurement_validity")
        if (not all(type(mask.get(k)) is bool for k in MASK_KEYS)
                or obs.get("eligibility") != mask or obs.get("validity") != tier
                or obs.get("value") != m.get("speedup_vs_baseline")
                or tier not in {"valid", "degraded", "uninformative"}
                or not isinstance(obs.get("reasons"), list) or not obs.get("provenance")
                or obs["provenance"].get("contract_hash") != contract.hash
                or m.get("contract_hash") != contract.hash or m.get("backend_real_device") is not True
                or m.get("correctness_verified") is not True or not _number(m.get("speedup_vs_baseline"))):
            errors.append("incomplete real case qualification: " + key)
        masks.append(mask); tiers.append(tier)
        arms = m.get("arm_windows") or {}
        window = arms.get("candidate")
        identity = m.get("arm_identity") or {}
        if (type(window) is not str or not window or arms.get("baseline") != window
                or observed_pairing(identity.get("candidate"), identity.get("baseline"))[2] != SCOPE_SAME):
            errors.append("missing/cross-arm pairing: " + key)
        else:
            windows.append(window)
        try:
            validate_receipt(contract.reference_contract, m.get("reference_receipt"))
        except ValueError as exc:
            errors.append(key + " reference receipt: " + str(exc))
        if positive and (tier not in {"valid", "degraded"} or mask.get("can_verdict") is not True):
            errors.append("positive case unusable: " + key)
    if len(windows) != len(set(windows)):
        errors.append("reused case timing window")
    q = metrics.get("workload_qualification") or {}
    if json_value(metrics.get("workload_contract")) != json_value(contract.as_dict()):
        errors.append("missing/changed full frozen workload contract")
    intersection = {k: bool(masks) and all(m.get(k) is True for m in masks) for k in MASK_KEYS}
    tier = "uninformative" if "uninformative" in tiers else "degraded" if "degraded" in tiers else "valid"
    if (q.get("scope_cases") != scope or q.get("eligibility") != intersection
            or q.get("validity") != tier or not isinstance(q.get("reasons"), list)
            or metrics.get("eligibility") != intersection or metrics.get("measurement_validity") != tier
            or metrics.get("measurement_can_gate") is not intersection["can_verdict"]):
        errors.append("workload qualification not full-scope intersection")
    missing = set(scope) - set(supplied)
    if missing and not all(any(key in str(r) for r in q.get("reasons", [])) for key in missing):
        errors.append("missing declared case not named as missing evidence")
    # Raw numbers are retained above for diagnostic/RPC readback. Only the
    # production qualification-aware ledger may feed decision arithmetic.
    speeds = case_speedups(metrics)
    value, _ = aggregate_speedup(contract.case_weights, speeds, contract.aggregate)
    if value is not None and all(_number(speeds.get(k)) for k,_ in contract.case_weights):
        # Independent arithmetic audit, not another admission policy.
        recomputed = math.exp(math.fsum(w * math.log(speeds[k]) for k,w in contract.case_weights)
                              / math.fsum(w for _,w in contract.case_weights))
        if not math.isclose(value, recomputed, rel_tol=1e-12):
            errors.append("aggregate arithmetic mismatch")
    verdict = metrics.get("performance_verdict") or {}
    over, _ = regression_over_limit(contract.max_regression, speeds)
    passed, _, _, _ = performance_verdict(contract.min_speedup, aggregate=contract.aggregate,
        case_weights=contract.case_weights, case_speedups=speeds,
        max_regression=contract.max_regression, regression_over_limit_value=over)
    if (verdict.get("basis") != "objective" or verdict.get("min_speedup") != contract.min_speedup
            or verdict.get("aggregate") != contract.aggregate
            or verdict.get("case_weights") != dict(contract.case_weights)
            or verdict.get("max_regression") != dict(contract.max_regression)
            or verdict.get("required_cases") != list(contract.required_cases)
            or verdict.get("regression_over_limit") is not over or verdict.get("passed") is not passed
            or decision_speedup(metrics) != verdict.get("value")
            or (value is None and verdict.get("value") is not None)
            or (value is not None and (not _number(verdict.get("value"))
                or not math.isclose(value, verdict["value"], rel_tol=1e-7, abs_tol=1e-7)))):
        errors.append("declared aggregate not recomputable (primary fallback forbidden)")
    return errors


def trace_metrics_errors(metrics_list, trace, supplied, code, gpu):
    """Match ALL case windows to actual RPCs; duplicate/extra windows cannot hide."""
    errors = []
    benches = [r for r in trace if r["request"]["action"] == "benchmark_with_baselines"]
    expected = [(key, (m.get("cases") or {}).get(key) or {}) for m in metrics_list for key in supplied]
    if len(benches) != len(expected):
        errors.append("wrong number of main/confirmation case RPCs (no retry windows)")
    for rpc, (key, m) in zip(benches, expected):
        req, res = rpc["request"], rpc.get("response") or {}
        tensors = (req.get("inputs") or {}).get("tensors") or {}
        observed = "|".join(k + "=" + "x".join(map(str, v.get("shape", []))) for k,v in sorted(tensors.items()))
        pair = res.get("paired") or {}
        cand = res.get("candidate") or {}
        base = (res.get("baselines") or {}).get(res.get("baseline_selected")) or {}
        ctime, btime = cand.get("median_time_ms"), base.get("median_time_ms")
        if (observed != key or req.get("code") != code or res.get("success") is not True
                or res.get("device") != "cuda" or res.get("selected_gpu") != gpu
                or res.get("baseline_selected") != "pytorch_eager"
                or pair.get("arm_windows") != m.get("arm_windows")
                or pair.get("arm_identity") != m.get("arm_identity")
                or not _number(ctime) or not _number(btime)
                or round(btime / ctime, 4) != m.get("speedup_vs_baseline")):
            errors.append("case metrics lack matching actual CUDA RPC: " + key)
    return errors


def confirmation_errors(artifact, metrics, contract, supplied):
    m2 = None
    try:
        cf = json.loads((Path(artifact.saved_dir) / "confirmation.json").read_text())
        record = artifact.remeasure
        raw = asdict(record) if is_dataclass(record) else vars(record)
        first, second = cf["first_binding"], cf["second_binding"]
        confirmed_key(first, second)
        m2 = cf["second_metrics"]  # Full second metrics, never a slim record substitute.
        if (first != metrics["run_binding"] or second != m2["run_binding"]
                or first["candidate_hash"] != code_hash(artifact.code) or first["contract_hash"] != contract.hash
                or cf["confirmation"] != json_value(audit_value(raw)) or record.contract_hash != contract.hash
                or record.ok is not True or record.independent is not True
                or record.validity not in {"valid", "degraded"}
                or record.validity != m2.get("measurement_validity")
                or record.speedup != decision_speedup(m2) or not _number(record.speedup)
                or record.speedup < contract.min_speedup or record.performance_basis != "objective"
                or json_value(audit_value(record.workload_evidence)) != confirmation_evidence(m2)
                or m2.get("performance_verdict", {}).get("passed") is not True):
            raise ValueError("confirmation identity/declared objective/evidence drift")
        errors = metrics_errors(m2, contract, supplied, positive=True)
        ok, detail = confirm_metrics(metrics, record, min_speedup=contract.min_speedup)
        if not ok:
            errors.append("production confirm_metrics refused: " + detail)
        windows = [(m["cases"][k].get("arm_windows") or {}).get("candidate")
                   for m in (metrics, m2) for k in supplied]
        if len(set(windows)) != len(windows):
            errors.append("main/confirmation reused case timing windows")
        return m2, errors
    except (AttributeError, OSError, ValueError, KeyError, TypeError) as exc:
        # An unusable second window remains evidence. Do not lose its RPCs or
        # count an honest below-threshold confirmation as a picked/retried window.
        return m2, ["durable workload confirmation audit: " + str(exc)]


def binding_errors(artifact, metrics, contract, plan):
    try:
        directory = Path(artifact.saved_dir)
        manifest = json.loads((directory / "run_manifest.json").read_text())
        raw = json.loads((directory / "contract.json").read_text())
        binding = metrics["run_binding"]
        expected = dict(run_id=manifest["run_id"], manifest_hash=manifest["hash"],
            contract_hash=contract.hash, workload_hash=manifest["workload_hash"],
            protocol_hash=digest(plan["measurement_protocol"]), candidate_hash=plan["candidate"]["hash"])
        if (raw != json_value(contract.as_dict())
                or manifest["hash"] != digest({k:v for k,v in manifest.items() if k != "hash"})
                or manifest["contract_hash"] != contract.hash or manifest["workload_hash"] != digest(manifest["inputs"])
                or manifest["measurement_protocol"] != plan["measurement_protocol"]
                or any(binding.get(k) != v for k,v in expected.items())
                or not binding.get("verification_id") or code_hash(artifact.code) != expected["candidate_hash"]):
            raise ValueError("durable candidate/contract/full protocol binding mismatch")
    except (AttributeError, OSError, ValueError, KeyError, TypeError) as exc:
        return ["run sidecar audit: " + str(exc)]
    return []


def ruler_errors(metrics_list, trace, calibration, profile, gpu):
    if not calibration:
        return []
    from operator_agent.cuda_calibration import validate_probe
    from operator_agent.cuda_ruler_profile import anchor_reading
    probes = [r.get("response") or {} for r in trace if r["request"]["action"] == "cuda_calibration_probe"]
    errors = [] if len(probes) == len(metrics_list) else ["main/confirmation missing independent native heartbeat"]
    for probe in probes:
        try:
            if validate_probe(probe, mode="heartbeat", selected_gpu=gpu) < 2 or probe["identity_hash"] != calibration["identity_hash"]:
                errors.append("native heartbeat frozen identity mismatch")
            if profile and not anchor_reading(probe, profile)["ok"]:
                errors.append("native absolute anchor rejected")
        except (ValueError, TypeError, KeyError) as exc:
            errors.append("native heartbeat audit: " + str(exc))
    if profile:
        for metrics in metrics_list:
            for key, m in (metrics.get("cases") or {}).items():
                if (m.get("cuda_ruler_profile", {}).get("hash") != profile["hash"]
                        or m.get("launch_floor_us") != profile["launch_floor_us"]
                        or not m.get("heartbeat", {}).get("heartbeat_absolute_anchor_checked")):
                    errors.append("case did not use frozen native ruler: " + key)
    return errors


def corpus_readback(path, before, artifact, t, metrics):
    try:
        after = Path(path).read_bytes()
        if not after.startswith(before) or len(after) <= len(before):
            raise ValueError("corpus not actually appended")
        events = [json.loads(x) for x in after[len(before):].splitlines() if x.strip()]
        if len(events) != 1 or events[0].get("event") != "kernel":
            raise ValueError("expected exactly one kernel event")
        event, stored = events[0], events[0]["record"]
        cf = json.loads((Path(artifact.saved_dir) / "confirmation.json").read_text())
        key = confirmed_key(metrics["run_binding"], cf["second_binding"])
        seal = event.get("admission") or {}
        if (seal.get("key") != key or seal.get("id") != admission_id(key)
                or seal.get("record_hash") != record_hash(stored)):
            raise ValueError("missing/incorrect confirmed-observation seal")
        expected = dict(kernel_id=artifact.kernel_id, algorithm=t.source.kernel_type, hardware=t.target.hardware,
            dsl=t.target.dsl, code=artifact.code, compile_info=artifact.compile_info,
            score=decision_speedup(metrics), measurement_validity="valid")
        if (any(stored.get(k) != v for k,v in expected.items()) or not _number(stored.get("score"))
                or stored["score"] <= t.constraints.min_speedup
                or stored["compile_info"].get("run_binding") != metrics["run_binding"]
                or code_hash(stored["code"]) != key["candidate_hash"]):
            raise ValueError("stored candidate/task/decision aggregate drift")
        with Path(path).open("rb") as f:
            journal = scan(f)
        replay = KernelDatabase(str(path), persist_retrievals=False)
        loaded = replay.get(artifact.kernel_id)
        if (journal.skipped or journal.errors or journal.duplicates or journal.admissions.get(seal["id"]) != event
                or replay.disk_skipped_lines or loaded is None or loaded.to_dict() != stored):
            raise ValueError("sealed journal/fresh replay mismatch")
        return dict(verified=True, sealed=True, fresh_replay=True, admission_id=seal["id"], score=loaded.score), []
    except (AttributeError, OSError, ValueError, KeyError, TypeError) as exc:
        return dict(verified=False), ["corpus readback audit: " + str(exc)]


def regression_fixture(t):
    """Exercise existing production pure functions ONLY; this is not L3 evidence."""
    c = freeze_from_task(t)
    speeds = dict(zip(c.required_cases, (2.0, .90)))
    over, why = regression_over_limit(c.max_regression, speeds)
    passed, basis, reason, value = performance_verdict(c.min_speedup, aggregate=c.aggregate,
        case_weights=c.case_weights, case_speedups=speeds, max_regression=c.max_regression,
        regression_over_limit_value=over)
    decision = implementation_decision(eligibility=ALL_TRUE,
        claim_record={"level":"validated", "ceiling":"persisted"}, required_cases_ok=True,
        regression_over_limit=over, evidence_notes=why).as_dict()
    errors = [] if (value > c.min_speedup and over is True and passed is False
                   and basis == "objective" and decision["decision"] == "use_library") else ["local regression fixture failed"]
    return dict(source="local_fixture_only", real_device=False, machine_timings=None,
        case_speedups=speeds, aggregate=value, performance_passed=passed,
        regression_over_limit=over, implementation_decision=decision, reason=reason, errors=errors)


def cleanup_errors(cleanup, backend):
    errors = []
    if cleanup.get("quiescent") is not True or cleanup.get("removed") is not True:
        errors.append("owned runner/scratch cleanup not confirmed")
    if backend.forbidden_actions:
        errors.append("forbidden remote actions")
    return errors


def check_trial_contract(t, expected):
    # The plan is a JSON-normal-form frozen artifact; tuples in the in-memory
    # contract are not semantic drift. Still compare every field and hash.
    if json_value(freeze_from_task(t).as_dict()) != expected:
        raise ValueError("trial contract drift before dispatch")


def positive_context(path, t, code):
    from operator_agent.run_contract import MANIFEST, CONTRACT
    from operator_agent.contract import load_dict
    path = Path(path).expanduser().resolve()
    manifest = json.loads((path / MANIFEST).read_text())
    contract = freeze_from_task(t)
    metrics = json.loads((path / "benchmark.json").read_text())
    if (manifest.get("measurement_protocol") != protocol_identity()
            or manifest.get("contract_hash") != contract.hash
            or load_dict(json.loads((path / CONTRACT).read_text())).hash != contract.hash
            or metrics.get("run_binding", {}).get("candidate_hash") != code_hash(code)):
        raise ValueError("positive continuation identity mismatch")
    # These hashes bind historical audit context, not restored qualifications.
    files = (MANIFEST, CONTRACT, "benchmark.json", "confirmation.json", "admission.json")
    return dict(artifact=str(path), manifest_hash=manifest["hash"],
                evidence_hashes={f: code_hash((path / f).read_text()) for f in files},
                evidence_scope="previous_positive_audit_only")


def trial_inputs(index, arrays):
    return arrays if index == 1 else arrays[:1] if index == 2 else arrays[:2]


def run(args):
    if args.execute and args.gpu is None:
        raise ValueError("--execute requires an explicitly authorized --gpu")
    profile, calibration = ruler(args)
    declared_tasks, code = tasks(), trial_code("unfused-fp32")
    plan = workload_plan(args, declared_tasks, code, profile, calibration)
    parent = Path(args.output).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="workload-decision-", dir=parent))  # Owned 0700; retained evidence.
    atomic_json(root / "plan.json", plan, readonly=True)
    print("PLAN " + str(root / "plan.json") + " HASH " + plan["hash"], flush=True)
    fixture = regression_fixture(declared_tasks[0])
    summary = dict(run_root=str(root), plan_hash=plan["hash"], mode="execute" if args.execute else "plan-only",
        functional_completed=False, performance_completed=False, completed=False,
        regression_fixture=fixture, trials=[], cleanup=None, errors=[])
    continuation = plan.get("positive_context")
    indexes = plan.get("execution_indexes", [0, 1, 2, 3])
    if continuation:
        summary["mode"] = "negative-followup" if args.execute else "negative-followup-plan-only"
    if not args.execute:
        atomic_json(root / "summary.json", summary)
        print(dumps(summary), flush=True)
        return 0 if not fixture["errors"] else 1

    backend, arrays = None, None
    first = SimpleNamespace(saved_dir=continuation["artifact"]) if continuation else None
    old_log = os.environ.get("OPERATOR_AGENT_LOG_DIR")
    old_keep = os.environ.get("OPERATOR_AGENT_LOG_KEEP")
    os.environ["OPERATOR_AGENT_LOG_DIR"] = str(root / "logs")
    os.environ["OPERATOR_AGENT_LOG_KEEP"] = "10000"  # Preserve every trial, including failures/HOLD.
    try:
        check_plan(root / "plan.json", plan)
        backend = BoundedProfileSSH(host=args.host, timeout=RPC_TIMEOUT)
        environment = backend.prepare(args.gpu)
        validate_scratch(backend.scratch)
        if "H20" not in environment.get("selected", {}).get("name", ""):
            raise ValueError("selected device is not H20")
        identity = backend.reference_identity(registered_reference().content_dict())
        validate_receipt(registered_reference(), identity.get("reference_receipt"))
        atomic_json(root / "environment.json", dict(environment=environment, reference_identity=identity))
        arrays = [primary(s) for s in LARGE + (TINY,)]  # No float64 random materialization.
        for index, (name, t) in enumerate(zip(TRIAL_NAMES, declared_tasks)):
            if index not in indexes:
                continue
            supplied = plan["trials"][index]["supplied_cases"]
            inputs = trial_inputs(index, arrays)
            out = root / "artifacts" / name
            db_path = root / "corpus" / name / "kernel_db.jsonl"  # No positive contamination of negatives.
            start = len(backend.trace)
            before = db_path.read_bytes() if db_path.exists() else b""
            generated, art = [], None
            record = dict(name=name, source="real_h20_execution", errors=[], performance_errors=[],
                          performance_completed=False, outcome="not_run")
            try:
                check_plan(root / "plan.json", plan)
                check_trial_contract(t, plan["trials"][index]["contract"])
                if [case_key(x) for x in inputs] != supplied:
                    raise ValueError("actual supplied cases differ from frozen dispatch plan")
                c = Conductor(backend=backend, kernel_db=KernelDatabase(str(db_path), persist_retrievals=False),
                    failure_db=FailureDatabase(str(root / "failures" / name / "failure_db.jsonl")),
                    max_optimize_rounds=2 if index == 3 else 1, use_coder_pipeline=False,
                    heartbeat_interval=plan["controls"]["heartbeat_interval"],
                    cuda_calibration_identity_hash=calibration["identity_hash"] if calibration else "",
                    cuda_ruler_profile_path=args.cuda_ruler_profile if profile else None)
                c.store = ArtifactStore(str(out))
                def design(*a, **kw):
                    if index == 3 or "designer" in generated:
                        raise AssertionError("fixed trial cannot regenerate/resume Designer")
                    generated.append("designer")
                    return UnifiedSketch(algorithm="layernorm", target_dsl="triton")
                def generate(*a, **kw):
                    if index == 3 or "coder" in generated:
                        raise AssertionError("fixed trial cannot regenerate/resume Coder")
                    generated.append("coder")
                    return code
                c.designer.run, c.coder = design, SimpleNamespace(run=generate)
                if index == 3 and (first is None or not first.saved_dir):
                    raise ValueError("no first-trial checkpoint; weight-drift resume not tested")
                art = c.execute(t, inputs, target_gflops=0, resume_from=first.saved_dir if index == 3 else None)
                if index == 0:
                    first = art
                trace = backend.trace[start:]
                after = db_path.read_bytes() if db_path.exists() else b""
                admission = (art.metadata or {}).get("corpus_admission") or {}
                l2 = next((r for r in art.verification_results if r.level == Level.L2_FUNCTION), None)
                l3 = next((r for r in art.verification_results if r.level == Level.L3_PERFORMANCE), None)
                record.update(saved_dir=art.saved_dir, admission=admission, resume=art.resume_status,
                    results=[dict(level=r.level.value, passed=r.passed, message=r.message, metrics=r.metrics)
                             for r in art.verification_results])
                if index == 3:
                    if (not l2 or l2.passed or l2.metrics.get("run_identity_unknown") is not True
                            or l2.metrics.get("correctness", {}).get("verdict") != "unknown"
                            or l3 or art.all_passed or art.remeasure or trace or generated or after != before):
                        record["errors"].append("weight drift not a zero-RPC/generation/corpus identity HOLD")
                    record["outcome"] = "identity_hold"
                else:
                    record["errors"] += assess(art, "passed", as_suite(t.constraints.correctness_suite), trace, after != before)
                    record["errors"] += receipt_errors(t.constraints.reference_contract, trace) + transfer_errors(trace)
                    if not l3:
                        raise ValueError("no actual L3 workload evidence")
                    m, contract = l3.metrics, freeze_from_task(t)
                    record["errors"] += metrics_errors(m, contract, supplied)
                    record["errors"] += binding_errors(art, m, contract, plan)
                    decision = m.get("implementation_decision") or {}
                    if (not admission or type(admission.get("kept")) is not bool
                            or not decision.get("decision")):
                        record["errors"].append("missing production decision/non-admission projection")
                    if index == 0:
                        record["outcome"] = "admitted" if admission.get("kept") else "hold_or_performance_failed"
                        second, errs = confirmation_errors(art, m, contract, supplied)
                        record["performance_errors"] += errs + metrics_errors(m, contract, supplied, positive=True)
                        if (not l3.passed or m.get("measurement_validity") != "valid"
                                or m.get("eligibility", {}).get("can_persist") is not True
                                or m.get("performance_verdict", {}).get("passed") is not True
                                or not _number(decision_speedup(m)) or decision_speedup(m) <= MIN_SPEEDUP
                                or decision.get("decision") != "use_custom" or admission.get("kept") is not True):
                            record["performance_errors"].append("no confirmed full-workload use_custom admission")
                        if admission.get("kept"):
                            proof, errs = corpus_readback(db_path, before, art, t, m)
                            record["corpus_readback"] = proof
                            record["performance_errors"] += errs
                        elif after != before:
                            record["errors"].append("denied candidate wrote corpus")
                        record["errors"] += trace_metrics_errors([m] + ([second] if second else []), trace, supplied, code, args.gpu)
                        record["errors"] += ruler_errors([m] + ([second] if second else []), trace, calibration, profile, args.gpu)
                        record["performance_completed"] = not record["errors"] and not record["performance_errors"]
                    else:
                        negative_metrics = [m]
                        # Production may attempt its one confirmation for a
                        # degraded (not uninformative) tiny window. Retain it;
                        # never retry it or present it as a successful admission.
                        if index == 1 and art.remeasure:
                            cf = json.loads((Path(art.saved_dir) / "confirmation.json").read_text())
                            raw = asdict(art.remeasure)
                            if (cf["confirmation"] != json_value(audit_value(raw))
                                    or json_value(audit_value(raw["workload_evidence"])) != confirmation_evidence(cf["second_metrics"])):
                                record["errors"].append("tiny confirmation evidence drift")
                            negative_metrics.append(cf["second_metrics"])
                            record["errors"] += metrics_errors(cf["second_metrics"], contract, supplied)
                            wins = [n["cases"][k]["arm_windows"]["candidate"] for n in negative_metrics for k in supplied]
                            if len(set(wins)) != len(wins):
                                record["errors"].append("tiny confirmation reused case windows")
                        record["errors"] += trace_metrics_errors(negative_metrics, trace, supplied, code, args.gpu)
                        record["errors"] += ruler_errors(negative_metrics, trace, calibration, profile, args.gpu)
                        if l2 is None or not l2.passed or after != before or admission.get("kept"):
                            record["errors"].append("negative case mislabeled correctness failure or persisted")
                        if index == 1:
                            tiny = m["cases"].get("X=32x64") or {}
                            reasons = tiny.get("measurement_validity_reasons", []) + (tiny.get("observation") or {}).get("reasons", [])
                            if (tiny.get("eligibility", {}).get("can_persist") is not False
                                    or not any("launch" in str(r).lower() or "地板" in str(r) for r in reasons)
                                    or m.get("eligibility", {}).get("can_persist") is not False
                                    or decision.get("decision") != "inconclusive"):
                                record["errors"].append("tiny required/weighted launch limitation did not honestly HOLD")
                            record["outcome"] = "launch_limited_hold"
                        else:
                            if (l3.passed or decision_speedup(m) is not None
                                    or m.get("eligibility", {}).get("can_persist") is not False
                                    or decision.get("decision") != "inconclusive" or art.remeasure
                                    or decision.get("required_cases_ok") is not None):
                                record["errors"].append("missing declared case did not fail closed as missing evidence")
                            record["outcome"] = "missing_evidence"
            except (Exception, KeyboardInterrupt) as exc:
                record["errors"].append(f"{type(exc).__name__}: {exc}")
            finally:
                record["generated"] = generated
                record["trace"] = compact_trace(backend.trace[start:])
                atomic_json(root / "evidence" / (name + ".json"), audit_value(record))
                del backend.trace[start:]  # Evidence retained, large wire buffers released.
                summary["trials"].append({k:v for k,v in record.items() if k not in {"trace", "results"}})
                print("TRIAL " + dumps(summary["trials"][-1]), flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        summary["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        if backend is not None:
            try:
                summary["cleanup"] = backend.cleanup()
            except (Exception, KeyboardInterrupt) as exc:
                summary["cleanup"] = dict(quiescent=False, removed=False, error=f"{type(exc).__name__}: {exc}")
            summary["errors"] += cleanup_errors(summary["cleanup"], backend)
            summary["forbidden_actions"] = backend.forbidden_actions
            summary["errors"] += receipt_errors(registered_reference(), backend.trace)
            atomic_json(root / "lifecycle.json", audit_value(dict(trace=compact_trace(backend.trace),
                cleanup=summary["cleanup"], work_counts=getattr(backend, "work_counts", {}))))
        for key, old in (("OPERATOR_AGENT_LOG_DIR", old_log), ("OPERATOR_AGENT_LOG_KEEP", old_keep)):
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
        arrays = None
    summary["functional_completed"] = (len(summary["trials"]) == len(indexes)
        and not summary["errors"] and not fixture["errors"] and all(not r["errors"] for r in summary["trials"]))
    summary["performance_completed"] = bool(not continuation and summary["functional_completed"] and summary["trials"][0]["performance_completed"])
    summary["completed"] = bool(summary["functional_completed"] and (continuation or summary["performance_completed"]))
    atomic_json(root / "summary.json", audit_value(summary))
    print(dumps(summary), flush=True)
    return 0 if summary["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
