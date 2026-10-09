"""Two fixed native CUDA audits in one private, owned H20 runner; no floor edits."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.correctness_profiles_h20 import ProfileSSH, compact_trace
from benchmarks.input_fidelity_h20 import dumps
from operator_agent.cuda_calibration import (CONFIG, PROTOCOL, freeze_audits,
    source_hashes, digest, validate_probe)
from operator_agent.run_contract import atomic_json
from operator_agent.run_layout import new_run_root
from operator_agent.cuda_ruler_profile import build_profile, POLICY as RULER_POLICY


def run(args):
    root = Path(new_run_root(args.output, "cuda-calibration-h20"))
    root.mkdir(parents=True, exist_ok=False)
    backend = ProfileSSH(host=args.host, timeout=180)
    plan = dict(protocol=PROTOCOL, mode="audit", windows=2, selected_gpu=args.gpu,
        config=CONFIG, sources=source_hashes(), floor_update_authorized=False,
        absolute_anchor_checked=False, acceptance="both windows, no adaptive retry")
    if getattr(args, "ruler_profile", False):
        plan["ruler_policy"] = RULER_POLICY
    atomic_json(root / "plan.json", dict(plan, hash=digest(plan)), readonly=True)
    summary = dict(run_root=str(root), completed=False, probes=[])
    receipts = []
    print("RUN_ROOT " + str(root), flush=True)
    try:
        environment = backend.prepare(args.gpu)
        atomic_json(root / "environment.json", environment)
        for window in range(2):
            payload = backend.calibration_probe(mode="audit")
            atomic_json(root / f"audit-{window + 1}.json", payload, readonly=True)
            ratio = validate_probe(payload, mode="audit", selected_gpu=args.gpu)
            receipts.append(payload)
            row = dict(window=window + 1, ratio=ratio, identity_hash=payload["identity_hash"],
                median_us={k: v * 1000 for k, v in payload["sample_medians_ms"].items()},
                telemetry=payload["telemetry"])
            summary["probes"].append(row)
            print("AUDIT " + dumps(row), flush=True)
        frozen = freeze_audits(receipts, args.gpu)
        profile = build_profile(frozen) if getattr(args, "ruler_profile", False) else None
        if profile:
            summary["ruler_profile"] = dict(hash=profile["hash"],
                launch_floor_us=profile["launch_floor_us"], expected_good_ms=profile["expected_good_ms"])
        summary["calibration_hash"] = frozen["hash"]
    except (Exception, KeyboardInterrupt) as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            summary["cleanup"] = backend.cleanup()
        except Exception as exc:
            summary["cleanup"] = dict(quiescent=False, error=repr(exc))
        summary["forbidden_actions"] = backend.forbidden_actions
        atomic_json(root / "transport.json", compact_trace(backend.trace))
    cleanup = summary["cleanup"]
    summary["completed"] = bool(summary.get("calibration_hash")
        and cleanup.get("quiescent") is True and cleanup.get("removed") is True
        and not backend.forbidden_actions)
    if summary["completed"]:
        atomic_json(root / "calibration.json", frozen, readonly=True)
        if profile:
            atomic_json(root / "ruler-profile.json", profile, readonly=True)
    atomic_json(root / "summary.json", summary)
    print(dumps(summary), flush=True)
    return 0 if summary["completed"] else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="algorithm-h20")
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--ruler-profile", action="store_true", help="predeclare and derive a versioned flush ruler for NEW runs only")
    p.add_argument("--output", default=str(ROOT / "output" / "cuda_calibration_h20"))
    raise SystemExit(run(p.parse_args()))
