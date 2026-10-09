"""Explicit, versioned channel-specific ruler, derived from independent audits.

No configurable floor/anchor numbers and no online self-normalization. The
profile must precede the task in a different measurement window. Preflight
crosschecks a long production event batch against the host monotonic clock.
This does not defend against a hostile owner fabricating all receipts or a
compromise of both clocks.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

try:
    from . import cuda_calibration as cc
except ImportError:
    import cuda_calibration as cc

PROTOCOL = "cuda-flush-ruler-profile-v1"
POLICY = dict(floor_statistic="max-of-two-empty-flush-p95-nearest-rank",
    anchor_statistic="median-of-two-production-good-medians-ms",
    good_ratio_band=[0.5, 2.0], min_slow_ratio=2.0,
    max_audit_iqr_over_median=0.10, max_audit_window_ratio=1.10,
    cuda_to_host_clock_ratio_band=[0.5, 2.0],
    heartbeat_interval=1, independent_window_required=True)


def _quantile(xs, q):
    import math
    return sorted(xs)[max(0, math.ceil(q * len(xs)) - 1)]


def build_profile(calibration):
    """Authority is the full verified preflight evidence, never a supplied scalar."""
    if not isinstance(calibration, dict):
        raise ValueError("ruler requires a complete frozen calibration")
    identity = calibration.get("identity", {})
    frozen = cc.freeze_audits(calibration.get("audit_receipts"), identity.get("selected_gpu"))
    if frozen != calibration:
        raise ValueError("ruler calibration content/hash mismatch")
    receipts = frozen["audit_receipts"]
    for r in receipts:
        clock = r.get("clock_crosscheck", {})
        if (clock.get("protocol") != "cuda-batch-vs-host-monotonic-v1"
                or clock.get("batch_repeats") != cc.CONFIG["clock_batch_repeats"]
                or type(clock.get("samples")) is not list
                or len(clock["samples"]) != cc.CONFIG["clock_samples"]):
            raise ValueError("ruler independent clock evidence missing")
        for sample in clock["samples"]:
            if (not isinstance(sample, dict) or not cc._positive(sample.get("cuda_ms"))
                    or not cc._positive(sample.get("host_ms"))):
                raise ValueError("ruler independent clock samples invalid")
            ratio = sample["cuda_ms"] / sample["host_ms"]
            if not POLICY["cuda_to_host_clock_ratio_band"][0] <= ratio <= POLICY["cuda_to_host_clock_ratio_band"][1]:
                raise ValueError("ruler CUDA event/independent host clock scale conflict")
    good = [r["good"]["median_time_ms"] for r in receipts]
    empty = [r["samples"]["empty-flush"] for r in receipts]
    empty_medians = [statistics.median(xs) for xs in empty]
    for xs in empty:
        if ((_quantile(xs, .75) - _quantile(xs, .25)) / statistics.median(xs)
                > POLICY["max_audit_iqr_over_median"]):
            raise ValueError("ruler empty-flush preflight dispersion too high")
    for r in receipts:
        dispersion = r["good"].get("dispersion_iqr_over_median")
        if (type(dispersion) not in (int, float) or not math.isfinite(dispersion)
                or not 0 <= dispersion <= POLICY["max_audit_iqr_over_median"]):
            raise ValueError("ruler good preflight dispersion missing/high")
    if any(max(xs) / min(xs) > POLICY["max_audit_window_ratio"]
           for xs in (good, empty_medians)):
        raise ValueError("ruler preflight windows unstable")
    content = dict(protocol=PROTOCOL, policy=json.loads(json.dumps(POLICY)), channel=cc.CHANNEL,
        calibration=calibration, calibration_hash=frozen["hash"],
        identity_hash=frozen["identity_hash"],
        launch_floor_us=max(_quantile(xs, .95) for xs in empty) * 1000,
        expected_good_ms=statistics.median(good),
        audit_windows=[r["paired"]["arm_windows"]["candidate"] for r in receipts],
        scope="explicit preflight ruler; new observations only; no retrospective promotion")
    return dict(content, hash=cc.digest(content))


def validate_profile(profile):
    if not isinstance(profile, dict) or profile != build_profile(profile.get("calibration")):
        raise ValueError("CUDA ruler profile content/policy/hash mismatch")
    # Isolate mutable caller objects; the frozen controls carry the content hash.
    return json.loads(json.dumps(profile, allow_nan=False))


def read_profile(path):
    path = Path(path)
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("CUDA ruler profile exceeds bounded receipt size")
    return validate_profile(json.loads(path.read_text()))


def anchor_reading(probe, profile):
    profile = validate_profile(profile)
    gpu = profile["calibration"]["identity"]["selected_gpu"]
    slow_ratio = cc.validate_probe(probe, mode="heartbeat", selected_gpu=gpu)
    if probe["identity_hash"] != profile["identity_hash"]:
        raise ValueError("CUDA absolute anchor identity drift")
    if probe["paired"]["arm_windows"]["candidate"] in profile["audit_windows"]:
        raise ValueError("CUDA absolute anchor cannot reuse its own preflight window")
    good_ratio = probe["good"]["median_time_ms"] / profile["expected_good_ms"]
    low, high = POLICY["good_ratio_band"]
    ok = slow_ratio >= POLICY["min_slow_ratio"] and low <= good_ratio <= high
    return dict(ok=ok, checked=True, profile_hash=profile["hash"],
        expected_good_ms=profile["expected_good_ms"], good_ratio=good_ratio,
        slow_ratio=slow_ratio, good_ratio_band=POLICY["good_ratio_band"],
        source="independent frozen preflight; not current heartbeat self-normalization")


def measurement_floor(profile, heartbeat, receipt):
    """Resolve only for an attested matching production timing window."""
    profile = validate_profile(profile)
    if (heartbeat is None or heartbeat.ok is not True
            or heartbeat.channel != cc.CHANNEL or heartbeat.absolute_anchor_checked is not True
            or heartbeat.ruler_profile_hash != profile["hash"]):
        raise ValueError("CUDA ruler requires a passing independently anchored native heartbeat")
    check = anchor_reading(heartbeat.evidence, profile)
    if (not check["ok"] or heartbeat.evidence.get("anchor_check") != check
            or heartbeat.identity_hash != profile["identity_hash"]
            or heartbeat.known_good_ratio != check["good_ratio"]
            or heartbeat.known_slow_ratio != check["slow_ratio"]):
        raise ValueError("CUDA ruler heartbeat anchor evidence mismatch")
    if not isinstance(receipt, dict):
        raise ValueError("CUDA ruler production timing receipt missing")
    identity = profile["calibration"]["identity"]
    if (receipt.get("calibration_identity") != identity
            or receipt.get("identity_hash") != profile["identity_hash"]
            or receipt.get("channel") != cc.CHANNEL
            or receipt.get("requested_counts") != {"warmup": 10, "repeat": 30}
            or receipt.get("default_stream_after") is not True):
        raise ValueError("CUDA ruler production timing channel/device/runtime/source drift")
    pair = receipt.get("paired", {})
    n = receipt.get("candidate_runs")
    block = pair.get("paired_block_samples")
    if (type(n) is not int or not 30 <= n <= cc.CONFIG["max_repeats"]
            or receipt.get("baseline_runs") != n or pair.get("paired_runs") != n
            or type(block) is not int or not 1 <= block <= cc.CONFIG["block_max"]):
        raise ValueError("CUDA ruler production paired sample counts invalid")
    if pair.get("arm_identity") != {k: identity["device"] for k in ("candidate", "baseline", "after")}:
        raise ValueError("CUDA ruler production per-arm device drift")
    windows = pair.get("arm_windows", {})
    wid = windows.get("candidate")
    if (type(wid) is not str or not wid or windows.get("baseline") != wid
            or wid in profile["audit_windows"]
            or wid == heartbeat.evidence["paired"]["arm_windows"]["candidate"]):
        raise ValueError("CUDA ruler production timing window missing/reused/drifted")
    return profile["launch_floor_us"]
