"""Trusted preflight/time-ruler tests. Offline receipts are not CUDA evidence."""
import copy
import json
from unittest.mock import Mock

import pytest

from operator_agent import cuda_calibration as cc, cuda_ruler_profile as rp
from operator_agent.backend_caps import Cap
from operator_agent.conductor import Conductor
from operator_agent.verifier import Verifier
from operator_agent.measurement import assess, Validity
from operator_agent.run_contract import atomic_json, protocol_identity
from tests.test_cuda_calibration import receipt, NativeBackend, change
from tests.test_oracle import _eval
from benchmarks import durable_profile_h20 as driver


def profile():
    a, b = receipt("audit", "anchor-a"), receipt("audit", "anchor-b")
    for r, base, tail in ((a, .0045, .0048), (b, .0046, .0050)):
        r["good"]["dispersion_iqr_over_median"] = .01
        r["samples"]["empty-flush"] = [base] * 180 + [tail] * 20
        r["clock_crosscheck"] = dict(protocol="cuda-batch-vs-host-monotonic-v1",
            batch_repeats=1024, samples=[dict(cuda_ms=10, host_ms=10.5) for _ in range(3)])
    return rp.build_profile(cc.freeze_audits([a, b], 2))


def heartbeat(p=None, probe=None):
    return cc.native_heartbeat(NativeBackend(calibration_probe=lambda **k: probe or receipt()),
        epoch=1, window_id="hb-1", ruler_profile=p if p is not None else profile())


def timing(p=None):
    p = p if p is not None else profile()
    return dict(channel=cc.CHANNEL, calibration_identity=copy.deepcopy(p["calibration"]["identity"]),
        identity_hash=p["identity_hash"], requested_counts=dict(warmup=10, repeat=30),
        default_stream_after=True, paired=receipt(window="l3-window")["paired"],
        candidate_runs=30, baseline_runs=30)


def test_floor_is_conservative_predeclared_p95_not_a_handwritten_lower_number(tmp_path):
    p = profile()
    assert p["launch_floor_us"] == 5.0
    assert p["expected_good_ms"] == .01
    assert p["policy"] == rp.POLICY and p["protocol"] == rp.PROTOCOL
    path = tmp_path / "profile.json"
    atomic_json(path, p, readonly=True)
    assert rp.read_profile(path) == p
    validated = rp.validate_profile(p)
    validated["policy"]["good_ratio_band"][0] = 0
    assert p["policy"]["good_ratio_band"][0] == .5


@pytest.mark.parametrize("field,value", [("launch_floor_us", .01), ("expected_good_ms", 1.0),
    ("hash", "0" * 64), ("protocol", "v2"), ("channel", "no-flush"),
    ("audit_windows", ["new"]), ("identity_hash", "0" * 64), ("policy", {})])
def test_rehashing_does_not_authorize_arbitrary_floor_anchor_or_policy(field, value):
    p = profile(); p[field] = value
    if field != "hash":
        p["hash"] = cc.digest({k:v for k,v in p.items() if k != "hash"})
    with pytest.raises(ValueError): rp.validate_profile(p)


@pytest.mark.parametrize("bad", ["dispersion", "missing", "window-drift", "empty-drift", "empty-noise"])
def test_unstable_or_unknown_preflight_cannot_create_a_profile(bad):
    frozen = profile()["calibration"]
    rows = frozen["audit_receipts"]
    if bad == "dispersion": rows[0]["good"]["dispersion_iqr_over_median"] = .11
    if bad == "missing": rows[0]["good"].pop("dispersion_iqr_over_median")
    if bad == "window-drift":
        rows[1]["good"] = {**rows[1]["good"], "min_time_ms": .012, "median_time_ms": .012, "max_time_ms": .012}
    if bad == "empty-drift": rows[1]["samples"]["empty-flush"] = [.006] * 200
    if bad == "empty-noise": rows[0]["samples"]["empty-flush"] = [.004] * 100 + [.006] * 100
    frozen = cc.freeze_audits(rows, 2)
    with pytest.raises(ValueError): rp.build_profile(frozen)


@pytest.mark.parametrize("bad", ["missing", "zero", "nonfinite", "samples", "fast", "slow"])
def test_preflight_requires_independent_clock_not_just_a_self_measured_anchor(bad):
    frozen = profile()["calibration"]
    clock = frozen["audit_receipts"][0]["clock_crosscheck"]
    if bad == "missing": frozen["audit_receipts"][0].pop("clock_crosscheck")
    if bad == "zero": clock["samples"][0]["host_ms"] = 0
    if bad == "nonfinite": clock["samples"][0]["cuda_ms"] = float("inf")
    if bad == "samples": clock["samples"].pop()
    if bad in ("fast", "slow"):
        for r in frozen["audit_receipts"]:
            scale = 10 if bad == "slow" else .1
            for s in (r["good"], r["slow"]):
                for k in ("min_time_ms", "median_time_ms", "max_time_ms"): s[k] *= scale
            for xs in r["samples"].values():
                for i in range(len(xs)): xs[i] *= scale
            for sample in r["clock_crosscheck"]["samples"]: sample["cuda_ms"] *= scale
    # freeze_audits' JSON hash also correctly refuses NaN/Inf; do not invent an anchor.
    with pytest.raises(ValueError): rp.build_profile(cc.freeze_audits(frozen["audit_receipts"], 2))


@pytest.mark.parametrize("scale,ok", [(1, True), (.5, True), (2, True), (.1, False), (10, False)])
def test_absolute_anchor_catches_uniform_scaling_even_when_separation_stays_eight(scale, ok):
    probe = receipt()
    for stats in (probe["good"], probe["slow"]):
        for k in ("min_time_ms", "median_time_ms", "max_time_ms"): stats[k] *= scale
    hb = heartbeat(probe=probe)
    assert hb.ok is ok
    assert hb.known_slow_ratio == 8
    assert hb.known_good_ratio == pytest.approx(scale)
    assert hb.absolute_anchor_checked and hb.ruler_profile_hash == profile()["hash"]
    assert hb.evidence["anchor_check"]["expected_good_ms"] == .01
    if not ok:
        obs = _eval(heartbeat=hb, heartbeat_expected=True, candidate_window=hb.window_id)
        assert obs.ruler_broken and not obs.eligibility.can_persist


def test_absolute_anchor_cannot_be_its_own_calibration_window_or_another_device():
    p = profile(); probe = receipt(window="anchor-a")
    with pytest.raises(ValueError, match="reuse"): rp.anchor_reading(probe, p)
    probe = receipt()
    change(probe, ("identity", "runtime", "gpu_uuid"), "other-gpu")
    with pytest.raises(ValueError, match="identity drift"): rp.anchor_reading(probe, p)


@pytest.mark.parametrize("path,value", [
    (("channel",), "no-flush"), (("identity_hash",), "0" * 64),
    (("calibration_identity", "runtime", "driver"), "other-driver"),
    (("calibration_identity", "sources", "run_kernel.py"), "0" * 64),
    (("requested_counts", "repeat"), 10), (("default_stream_after",), False),
    (("paired", "arm_windows", "baseline"), "different-window"),
    (("paired", "arm_windows", "candidate"), ""),
    (("paired", "arm_identity", "after", "visible_index"), 1),
    (("candidate_runs",), 0), (("baseline_runs",), 31),
    (("paired", "paired_runs"), 29), (("paired", "paired_block_samples"), 11),
])
def test_matching_profile_requires_actual_production_timing_attestation(path, value):
    p = profile(); raw = timing(p)
    target = raw
    for k in path[:-1]: target = target[k]
    target[path[-1]] = value
    with pytest.raises(ValueError): rp.measurement_floor(p, heartbeat(p), raw)


def test_audit_and_heartbeat_windows_are_not_reusable_as_candidate_measurements():
    p = profile(); hb = heartbeat(p)
    for wid in ("anchor-a", "anchor-b", hb.evidence["paired"]["arm_windows"]["candidate"]):
        raw = timing(p); raw["paired"]["arm_windows"] = dict(candidate=wid, baseline=wid)
        with pytest.raises(ValueError, match="window"): rp.measurement_floor(p, hb, raw)
    assert rp.measurement_floor(p, hb, timing(p)) == 5


def test_missing_or_separation_only_heartbeat_never_enables_new_floor():
    p = profile()
    raw_hb = cc.native_heartbeat(NativeBackend(calibration_probe=lambda **k: receipt()), epoch=1, window_id="h")
    for hb in (None, raw_hb):
        with pytest.raises(ValueError, match="anchored"): rp.measurement_floor(p, hb, timing(p))
    with pytest.raises(ValueError, match="missing"): rp.measurement_floor(p, heartbeat(p), None)


def test_original_launch_validity_thresholds_are_unchanged_and_floor_is_channel_specific():
    assert assess(median_time_ms=.149, launch_floor_us=20.24).validity == Validity.DEGRADED
    assert assess(median_time_ms=.149, launch_floor_us=5).validity == Validity.VALID
    assert assess(median_time_ms=.001, launch_floor_us=5).validity == Validity.UNINFORMATIVE
    # Selecting a new ruler cannot override contamination or high dispersion.
    assert assess(median_time_ms=.149, launch_floor_us=5, gpu_util_pct=50).validity != Validity.VALID
    assert assess(median_time_ms=.149, launch_floor_us=5, dispersion=.4).validity == Validity.UNINFORMATIVE


def test_verifier_uses_attested_profile_and_fail_closed_errors_reach_oracle():
    p = profile()
    b = NativeBackend(caps={Cap.REAL_DEVICE}, calibration_probe=lambda **k: receipt(),
        get_hardware_info=lambda: dict(name="NVIDIA H20", device="cuda"))
    v = Verifier(b, cuda_ruler_profile=p)
    m = dict(speedup_vs_baseline=2, timing_receipt=timing(p), static_guards=True, correctness_verified=True)
    _, assessment = v._assess_measurement(m, "", .149, task=driver.task("ruler"), heartbeat=heartbeat(p), heartbeat_expected=True)
    assert assessment.validity == Validity.VALID and m["launch_floor_us"] == 5
    assert m["eligibility"]["can_persist"] and m["heartbeat"]["heartbeat_absolute_anchor_checked"]
    assert m["observation"]["provenance"]["cuda_ruler_profile"]["hash"] == p["hash"]
    m = dict(speedup_vs_baseline=2, static_guards=True, correctness_verified=True)
    _, assessment = v._assess_measurement(m, "", .149, task=driver.task("ruler-bad"), heartbeat=heartbeat(p), heartbeat_expected=True)
    assert assessment.l3_pass_is_refused and assessment.validity == Validity.UNINFORMATIVE
    assert m["ruler_broken"] and not m["eligibility"]["can_persist"]
    assert "launch_floor_us" not in m  # never silently fall back to legacy hardware table


def test_conductor_requires_every_window_heartbeat_and_freezes_ruler_choice(tmp_path):
    p = profile(); path = tmp_path / "ruler.json"; atomic_json(path, p)
    b = NativeBackend(calibration_probe=Mock(return_value=receipt()))
    with pytest.raises(ValueError, match="interval"): Conductor(backend=b, cuda_ruler_profile_path=str(path))
    with pytest.raises(ValueError, match="mismatch"):
        Conductor(backend=b, heartbeat_interval=1, cuda_ruler_profile_path=str(path), cuda_calibration_identity_hash="0" * 64)
    c = Conductor(backend=b, heartbeat_interval=1, cuda_ruler_profile_path=str(path))
    hb, due = c._heartbeat_for_next_measurement()
    assert due and hb.ok and hb.absolute_anchor_checked
    assert c._heartbeat_for_next_measurement() == (hb, True)
    assert c.cuda_calibration_identity_hash == p["identity_hash"]
    assert c.verifier._cuda_ruler_profile == p
    assert "operator_agent/cuda_ruler_profile.py" in protocol_identity()


def test_invalid_profile_is_rejected_before_h20_deployment_or_allocation(tmp_path, monkeypatch):
    p = profile(); path = tmp_path / "ruler.json"; atomic_json(path, p)
    monkeypatch.setattr(driver, "ProfileSSH", lambda *a, **k: pytest.fail("backend constructed"))
    args = driver.parser().parse_args(["--gpu", "3", "--cuda-ruler-profile", str(path), "--output", str(tmp_path / "output")])
    with pytest.raises(ValueError, match="physical GPU"): driver.run(args)
    args.gpu = 2; p["launch_floor_us"] = 1; atomic_json(path, p)
    with pytest.raises(ValueError, match="content/policy/hash"): driver.run(args)
    assert not (tmp_path / "output").exists()


def test_oracle_identity_failure_revokes_even_an_otherwise_valid_reading():
    hb = heartbeat()
    obs = _eval(measure=lambda: dict(value=2, validity="valid",
        ruler_identity_error="production source drift"), heartbeat=hb,
        heartbeat_expected=True, candidate_window=hb.window_id)
    assert obs.value == 2 and obs.ruler_broken
    assert not any(vars(obs.eligibility).values())
    assert any("production source drift" in r for r in obs.reasons)


def test_changing_only_ruler_choice_cannot_resume_an_old_snapshot(tmp_path):
    from tests.test_run_contract import completed, hold
    from tests.test_resume_integrity import _conductor, _inputs
    task, _, folder = completed(tmp_path / "old")
    c = _conductor(str(tmp_path / "new"))
    # Change only this frozen control. This is an offline HOLD test, not a
    # declaration that MockGPU can measure with a CUDA ruler.
    c.cuda_ruler_profile = profile()
    art = hold(c, task, _inputs(), folder)
    assert c._run_controls["cuda_ruler_profile_hash"] == profile()["hash"]
    assert art.resume_status["status"] == "identity_mismatch"
    assert c.llm.calls == 0
