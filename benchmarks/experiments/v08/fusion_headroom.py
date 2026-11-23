"""Fused residual+LayerNorm headroom gate (pre-registered, zero model calls).

Question: on GPU7, with the frozen task ``Y = LayerNorm(X + R)`` and a frozen
hand-written fused kernel, is the fused median at least 2.5% below the eager
``X + R`` -> ``F.layer_norm`` baseline and outside the current-window band?

This is a *headroom gate*, not an A/C comparison and not a plugin claim: the
fused kernel is hand-written here. The eager baseline is not a second candidate;
the L3 receipt measures it in the same window (``baseline_type=pytorch_eager``)
and L2 already gates the fused source against it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

from operator_agent.app.hitl_tools import build_task
from operator_agent.core.contract import freeze_from_task
from operator_agent.core.run_contract import digest
from operator_agent.decide import reference_contract as rc
from operator_agent.decide.noise_preflight import PROTOCOL

SCHEMA = "v08-fusion-headroom-plan-v1"
SHAPE = (32768, 6144)
DECLARATION = dict(operator='fused_layernorm_residual', request="fp16 32768x6144 residual add layernorm",
                   shapes=[list(SHAPE)], precision="fp16", dsl="triton",
                   hardware="nvidia_gpu", min_speedup=1.0)
KERNEL_PATH = Path(__file__).resolve().parent / "kernels" / "fused_ln_residual.py"
PROBE_CHANNEL = "fused-layernorm-residual-32768x6144-fp16-cuda-event"
PROBE_OP = "fused_layernorm_residual"
EFFECT_THRESHOLD_PCT = 2.5
GPU = dict(index=7, uuid="GPU-52ca6155-d8bc-a416-847a-6f0c71db1552", name="NVIDIA H20")
REQUIRED_CASES = 32
BUDGET = dict(max_verifications=3, max_model_calls=1, wall_clock_s=2400)


def kernel_source():
    code = KERNEL_PATH.read_text(encoding="utf-8")
    return code, hashlib.sha256(code.encode("utf-8")).hexdigest()


def contract_hash():
    return freeze_from_task(build_task(DECLARATION)).hash


def _sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _positive(value):
    return type(value) in (int, float) and value > 0


def gate_band(gate):
    """The window receipt must be a passing fused probe on the frozen channel."""
    identity = gate.get("identity") or {}
    if (not isinstance(gate, dict) or gate.get("schema") != PROTOCOL
            or gate.get("channel") != PROBE_CHANNEL
            or identity.get("shape") != list(SHAPE)
            or identity.get("backend") != "ssh"
            or gate.get("test_only") is True
            or gate.get("planned_effect_pct") != EFFECT_THRESHOLD_PCT):
        raise ValueError("fusion headroom requires the frozen fused noise probe")
    if gate.get("ok") is not True or gate.get("action") != "start":
        raise ValueError("fusion headroom window did not permit starting")
    band = gate.get("effective_band_pct")
    if not _positive(band) or band >= EFFECT_THRESHOLD_PCT:
        raise ValueError("fusion headroom band must be positive and below 2.5%")
    return float(band)


def freeze(output):
    code, kernel_sha = kernel_source()
    plan = dict(schema=SCHEMA, declaration=DECLARATION, shape=list(SHAPE),
                contract_hash=contract_hash(),
                profile=rc.PROFILE_FUSED_LN_RESIDUAL,
                atol=rc.REGISTERED_ATOL[rc.PROFILE_FUSED_LN_RESIDUAL],
                reference_hash=rc.registered_reference(rc.PROFILE_FUSED_LN_RESIDUAL).digest(),
                kernel_sha256=kernel_sha, probe_channel=PROBE_CHANNEL,
                effect_threshold_pct=EFFECT_THRESHOLD_PCT, gpu=GPU, budget=BUDGET,
                formal_gain_claim=False, frozen_at=time.time())
    plan["hash"] = digest(plan)
    from benchmarks.experiments.phase1.plan import write_once
    write_once(Path(output), plan)
    return plan


def validate(plan):
    issues = []
    if not isinstance(plan, dict) or plan.get("schema") != SCHEMA:
        return ["expected a fusion headroom plan"]
    fixed = dict(declaration=DECLARATION, shape=list(SHAPE), contract_hash=contract_hash(),
                 profile=rc.PROFILE_FUSED_LN_RESIDUAL,
                 atol=rc.REGISTERED_ATOL[rc.PROFILE_FUSED_LN_RESIDUAL],
                 reference_hash=rc.registered_reference(rc.PROFILE_FUSED_LN_RESIDUAL).digest(),
                 kernel_sha256=kernel_source()[1], probe_channel=PROBE_CHANNEL,
                 effect_threshold_pct=EFFECT_THRESHOLD_PCT, gpu=GPU, budget=BUDGET)
    issues = [f"plan field differs: {key}" for key, value in fixed.items()
              if plan.get(key) != value]
    if plan.get("formal_gain_claim") is not False or not _positive(plan.get("frozen_at")):
        issues.append("plan must be exploratory and frozen")
    if plan.get("hash") != digest({k: v for k, v in plan.items() if k != "hash"}):
        issues.append("plan hash differs")
    return issues


def _l3(receipt):
    for verdict in receipt.get("verdicts") or []:
        if verdict.get("level") == "L3":
            return verdict.get("metrics") or {}
    return {}


def _l2(receipt):
    for verdict in receipt.get("verdicts") or []:
        if verdict.get("level") == "L2":
            return verdict.get("metrics") or {}
    return {}


def judge(*, gate, receipt):
    """Recompute the headroom verdict from a final L3 receipt; no hand-entered numbers."""
    band = gate_band(gate)
    l3 = _l3(receipt)
    l2 = _l2(receipt)
    reasons = []
    cases = ((l2.get("correctness") or {}).get("cases") or [])
    passed_cases = sum(1 for case in cases
                       if case.get("verdict") == "passed" or
                       (case.get("metrics") or {}).get("bad_elements") == 0)
    if len(cases) != REQUIRED_CASES or passed_cases != REQUIRED_CASES:
        reasons.append(f"L2 suite incomplete: {passed_cases}/{len(cases)}")
    if l3.get("measurement_validity") != "valid":
        reasons.append("final L3 validity is not valid")
    if l3.get("baseline_type") != "pytorch_eager":
        reasons.append("final L3 did not measure the eager baseline")
    candidate = l3.get("median_time_ms")
    baseline = l3.get("baseline_time_ms")
    if not _positive(candidate) or not _positive(baseline):
        reasons.append("final L3 missing candidate/baseline medians")
    decidable = not reasons
    diff = ((baseline - candidate) / baseline * 100.0) if decidable else None
    if not decidable:
        verdict = "unknown"
    elif diff >= EFFECT_THRESHOLD_PCT and diff > band:
        verdict = "headroom_confirmed"
    elif diff < -band:
        verdict = "fused_hurts"
    else:
        verdict = "no_increment"
    return dict(decidable=decidable, verdict=verdict, formal_gain_claim=False,
                band_pct=band, difference_pct=diff,
                effect_threshold_pct=EFFECT_THRESHOLD_PCT,
                candidate_median_ms=candidate, baseline_median_ms=baseline,
                candidate_speedup=l3.get("speedup_vs_baseline"),
                measurement_validity=l3.get("measurement_validity"),
                passed_cases=passed_cases, reasons=reasons)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def live_preflight(args, gpu_env):
    from operator_agent.app import hitl_tools
    previous = {k: os.environ.get(k) for k in gpu_env}
    os.environ.update(gpu_env)
    try:
        return hitl_tools.noise_preflight(
            backend="ssh", reps=5, repeat=8, shape=list(SHAPE), op=PROBE_OP,
            dtype="fp16", planned_effect_pct=EFFECT_THRESHOLD_PCT,
            host=args.host, work_dir=args.work_dir, timeout=args.timeout)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run(args):
    from operator_agent.app import hitl_tools
    from operator_agent.execute.backend_ownership import owned_worker_scope
    from benchmarks.experiments.phase1.plan import write_once

    args.root = Path(args.root).resolve()
    args.root.mkdir(parents=True, exist_ok=False)
    gpu_env = dict(OPERATOR_AGENT_REQUIRED_GPU=str(GPU["index"]),
                   OPERATOR_AGENT_REQUIRED_GPU_UUID=GPU["uuid"],
                   OPERATOR_AGENT_REQUIRED_GPU_NAME=GPU["name"])
    os.environ.update(gpu_env)
    code, _ = kernel_source()
    run_dir = args.root / "run"
    hitl_tools.prepare(DECLARATION, run_dir, host_budget=dict(
        wall_clock_s=BUDGET["wall_clock_s"], max_verifications=BUDGET["max_verifications"],
        max_model_calls=1, max_output_tokens_per_call=8192,
        total_output_token_reservations=8192))

    def verify(purpose, level):
        worker = run_dir / (purpose + "-worker")
        worker.mkdir()
        with owned_worker_scope(worker, uuid.uuid4().hex):
            return hitl_tools.verify(
                run_dir, code, backend="ssh", host=args.host, work_dir=args.work_dir,
                timeout=args.timeout, persistent=True, stop_after=level,
                verification_purpose=purpose)

    gate_start = live_preflight(args, gpu_env)
    write_once(args.root / "preflight-start.json", gate_start)
    gate_band(gate_start)
    explore = verify("explore", "L2")
    if not (explore.get("levels") or {}).get("L2"):
        raise SystemExit("fused candidate did not pass the exploratory L2")
    gate_final = live_preflight(args, gpu_env)
    write_once(args.root / "preflight-final.json", gate_final)
    final = verify("final", "L3")
    host_task = final.get("host_task") or {}
    if host_task.get("phase") != "final_complete":
        raise SystemExit(f"final host task stopped: {host_task.get('stop_reason')}")
    receipt = _read(final["evidence"])
    write_once(args.root / "receipt.json", receipt)
    judgement = judge(gate=gate_final, receipt=receipt)
    write_once(args.root / "judgement.json", judgement)
    print(json.dumps(judgement, ensure_ascii=False, indent=2))
    return 0 if judgement["decidable"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_freeze = sub.add_parser("freeze")
    p_freeze.add_argument("--output", type=Path, required=True)
    p_judge = sub.add_parser("judge")
    p_judge.add_argument("--plan", type=Path, required=True)
    p_judge.add_argument("--gate", type=Path, required=True)
    p_judge.add_argument("--receipt", type=Path, required=True)
    p_judge.add_argument("--output", type=Path, required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--root", type=Path, required=True)
    p_run.add_argument("--host", default="h20")
    p_run.add_argument("--work-dir", default="/root/operator-p2-seed-20261116")
    p_run.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        plan = freeze(args.output)
        print(plan["hash"])
        return 0
    if args.command == "judge":
        plan = _read(args.plan)
        issues = validate(plan)
        if issues:
            raise SystemExit("invalid plan: " + "; ".join(issues))
        result = judge(gate=_read(args.gate), receipt=_read(args.receipt))
        from benchmarks.experiments.phase1.plan import write_once
        write_once(args.output, result)
        print(result["verdict"])
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
