"""Fusion headroom judge and plan freeze rules (no GPU)."""
import json

import pytest

from benchmarks.experiments.v08 import fusion_headroom as fh


def _gate(**over):
    gate = dict(schema="noise-preflight-v1", channel=fh.PROBE_CHANNEL,
                identity={"shape": list(fh.SHAPE), "backend": "ssh"},
                test_only=False, ok=True, action="start",
                effective_band_pct=0.4, planned_effect_pct=2.5)
    gate.update(over)
    return gate


def _receipt(*, cases=32, passed_cases=32, validity="valid", baseline=1.0, candidate=0.5,
             baseline_type="pytorch_eager"):
    l2_cases = [{"case_id": f"c{i}", "verdict": "passed"} for i in range(passed_cases)]
    l2_cases += [{"case_id": f"bad{i}", "verdict": "failed"} for i in range(cases - passed_cases)]
    return {"verdicts": [
        dict(level="L2", passed=passed_cases == cases, metrics=dict(
            correctness=dict(cases=l2_cases))),
        dict(level="L3", passed=True, metrics=dict(
            measurement_validity=validity, baseline_type=baseline_type,
            baseline_time_ms=baseline, median_time_ms=candidate,
            speedup_vs_baseline=baseline / candidate)),
    ]}


def test_judge_confirms_large_headroom():
    verdict = fh.judge(gate=_gate(), receipt=_receipt())
    assert verdict["decidable"] is True
    assert verdict["verdict"] == "headroom_confirmed"
    assert verdict["difference_pct"] == pytest.approx(50.0)
    assert verdict["band_pct"] == pytest.approx(0.4)
    assert verdict["formal_gain_claim"] is False


def test_judge_rejects_inside_threshold():
    assert fh.judge(gate=_gate(), receipt=_receipt(baseline=1.0, candidate=0.99))["verdict"] \
        == "no_increment"          # 1% < 2.5% threshold
    assert fh.judge(gate=_gate(), receipt=_receipt(baseline=1.0, candidate=0.98))["verdict"] \
        == "no_increment"          # 2% < 2.5% threshold


def test_judge_flags_regression_beyond_band():
    verdict = fh.judge(gate=_gate(), receipt=_receipt(baseline=1.0, candidate=1.1))
    assert verdict["verdict"] == "fused_hurts"


def test_judge_is_unknown_when_validity_or_suite_or_baseline_is_missing():
    for receipt in (_receipt(validity="degraded"),
                    _receipt(cases=32, passed_cases=31),
                    _receipt(baseline_type="cublas"),
                    _receipt(baseline=0.0)):
        verdict = fh.judge(gate=_gate(), receipt=receipt)
        assert verdict["decidable"] is False
        assert verdict["verdict"] == "unknown"
        assert verdict["reasons"]


def test_gate_must_be_a_passing_window_receipt():
    for gate in (_gate(action="hold"), _gate(effective_band_pct=3.0),
                 _gate(channel="layernorm-32768x6144-fp16-cuda-event"),
                 _gate(test_only=True), {**_gate(), "schema": "other"}):
        with pytest.raises(ValueError):
            fh.gate_band(gate)


def test_freeze_and_validate_round_trip(tmp_path):
    plan = fh.freeze(tmp_path / "plan.json")
    assert fh.validate(plan) == []
    tampered = dict(plan, gpu=dict(plan["gpu"], uuid="GPU-deadbeef"))
    assert any("gpu" in issue for issue in fh.validate(tampered))
    tampered = dict(plan, kernel_sha256="0" * 64)
    assert any("kernel_sha256" in issue for issue in fh.validate(tampered))
    tampered = dict(plan, hash="0" * 64)
    assert any("hash" in issue for issue in fh.validate(tampered))
