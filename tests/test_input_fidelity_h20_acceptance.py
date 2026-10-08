"""Offline tests of the acceptance driver, NOT real-device acceptance evidence."""
from unittest.mock import Mock
from types import SimpleNamespace

import pytest

from benchmarks.input_fidelity_h20 import (AcceptanceSSH, GOOD, L2OnlyVerifier,
    assess, case, gpu_evidence, select_gpu, shell_path, trials, validate_scratch)
from operator_agent.anticheat import static_audit
from operator_agent.schemas import VerificationLevel as Level, VerificationResult as Result


def evidence():
    # Synthetic receipts only for checker calibration.
    return {"input_fidelity_verified": True, "output_digest": "synthetic-digest",
            "input_fidelity": {"representation": "torch-storage-v1",
                "observed": {"X": {}}, "tensor_devices": {"X": "cuda:0"}}}


def results():
    row = {"case_id": "tail", "verdict": "passed", "determinism": "passed",
           "metrics": evidence(), "repeat_metrics": evidence()}
    return [Result(Level.L1_COMPILE, True), Result(Level.L2_FUNCTION, True, metrics={
        "max_error": 0, "matched_ratio": 1.0, "input_fidelity_readings": [evidence()], "correctness": {
            "verdict": "passed", "cases": [row]}})]


def test_positive_checker_requires_every_declared_case_and_actual_repeat_evidence():
    assert assess(results(), "passed", "", ["tail"], 0) == []
    assert assess(results(), "passed", "", ["tail", "missing"], 0)
    assert assess(results(), "passed", "", ["tail"], 1)
    bad = results()
    bad.append(Result(Level.L3_PERFORMANCE, False))
    assert assess(bad, "passed", "", ["tail"], 0)


@pytest.mark.parametrize("change", ["cpu", "missing_device", "missing_receipt", "missing_digest",
                                   "changed_digest", "missing_repeat", "skipped", "not_torch"])
def test_positive_checker_does_not_accept_incomplete_or_fake_gpu_evidence(change):
    r = results()
    row = r[-1].metrics["correctness"]["cases"][0]
    m = row["metrics"]
    if change == "cpu":
        m["input_fidelity"]["tensor_devices"]["X"] = "cpu"
    elif change == "missing_device":
        m["input_fidelity"]["tensor_devices"] = {}
    elif change == "missing_receipt":
        m.pop("input_fidelity")
    elif change == "missing_digest":
        m.pop("output_digest")
    elif change == "changed_digest":
        row["repeat_metrics"]["output_digest"] = "different"
    elif change == "missing_repeat":
        row.pop("repeat_metrics")
    elif change == "skipped":
        row["verdict"] = "not_run"
    else:
        m["input_fidelity"]["representation"] = "numpy"
    assert assess(r, "passed", "", ["tail"], 0)


def test_negative_checker_requires_aligned_primary_pass_then_target_failure():
    r = results()
    r[-1].passed = False
    c = r[-1].metrics["correctness"]
    c["verdict"] = c["cases"][0]["verdict"] = "failed"
    assert assess(r, "failed", "tail", ["tail"], 0) == []
    c["cases"][0]["metrics"]["input_fidelity_verified"] = None
    assert assess(r, "failed", "tail", ["tail"], 0)


def test_determinism_negative_primary_can_numerically_pass_with_sub_tolerance_error():
    r = results()
    r[-1].passed = False
    r[-1].metrics["max_error"] = 1e-10
    c = r[-1].metrics["correctness"]
    c["verdict"] = c["cases"][0]["verdict"] = "failed"
    assert assess(r, "failed", "tail", ["tail"], 0) == []
    r[-1].metrics["matched_ratio"] = 0.9
    assert assess(r, "failed", "tail", ["tail"], 0)


def test_unknown_must_be_specific_unsupported_layout_not_unrelated_transport_failure():
    r = results()
    r[-1].passed = False
    r[-1].metrics["correctness"].update(verdict="unknown", reason="SSH timed out")
    assert assess(r, "unknown", "只读", ["tail"], 0)
    r[-1].metrics["correctness"].update(reason="Torch 无只读属性")
    assert assess(r, "unknown", "只读", ["tail"], 0) == []


def test_checker_handles_aggregated_primary_receipts_but_rejects_missing_one():
    m = {"input_fidelity_readings": [evidence(), evidence()]}
    assert gpu_evidence(m)
    m["input_fidelity_readings"][1] = {}
    assert not gpu_evidence(m)


def test_trial_inputs_really_include_tail_offset_zero_stride_fp16_and_readonly():
    specs = {t[0]: t for t in trials()}
    assert len(specs) == 11
    assert {c.dtype for c in specs["good-core"][2].cases[3].inputs.tensors} == {"fp16"}
    x = specs["good-offset"][3]["X"]
    assert x.strides == (64, 8) and x.ctypes.data != x.base.ctypes.data
    x = specs["good-writable-zero-stride"][3]["X"]
    assert x.strides[-1] == 0 and x.flags.writeable
    x = specs["readonly-refused"][2].cases[0].materialize()["X"]
    assert x.strides[-1] == 0 and not x.flags.writeable
    assert specs["negative-stride-refused"][3]["X"].strides[0] < 0
    for _, source, _, _, _, _ in specs.values():
        compile(source, "acceptance", "exec")
        assert not static_audit(source, dsl="triton")


@pytest.mark.parametrize("path", ["/tmp", "/tmp/other-random123", "/tmp/opag-fidelity-fixed",
    "/tmp/opag-fidelity-abcdefgh/../danger", "/tmp/opag-fidelity-abcdefgh/",
    "/tmp/opag-fidelity-abcdefgh;bad", "/home/opag-fidelity-abcdefgh"])
def test_scratch_validation_refuses_predictable_wrong_root_or_shell_paths(path):
    with pytest.raises(ValueError):
        validate_scratch(path)


def test_selection_refuses_busy_cards_and_pins_an_idle_physical_id():
    csv = "0, H20, 40000, 100\n2, H20, 10000, 0\n3, H20, 20000, 0"
    assert select_gpu(csv)["index"] == 3
    assert select_gpu(csv, 2)["index"] == 2
    for gpu in (0, 8):
        with pytest.raises(RuntimeError):
            select_gpu(csv, gpu)
    with pytest.raises(RuntimeError):
        select_gpu("0, H20, 100, 0")


def test_runtime_adapter_cannot_sync_or_install_dependencies():
    backend = AcceptanceSSH(host="offline")
    backend.runtime = "/existing runtime/bin/python"
    assert backend._existing_runtime("~/.local/bin/uv run python -u script.py") == "'/existing runtime/bin/python' -u script.py"
    assert shell_path("~/existing env") == '"$HOME"/\'existing env\''
    with pytest.raises(RuntimeError):
        backend.deploy()
    with pytest.raises(RuntimeError):
        backend.ensure_environment(["torch"])
    for action in ("execute", "benchmark", "benchmark_with_baselines"):
        with pytest.raises(RuntimeError):
            backend._ssh_exec_json({"action": action})
    assert backend.forbidden_actions == ["execute", "benchmark", "benchmark_with_baselines"]
    assert backend.trace == []


def test_cleanup_refuses_live_server_and_only_removes_validated_scratch(monkeypatch):
    backend = AcceptanceSSH(host="offline")
    backend.scratch = "/tmp/opag-fidelity-abcdefgh"
    commands = []
    monkeypatch.setattr(backend, "close", Mock())
    monkeypatch.setattr(backend, "_ssh_exec", lambda cmd: commands.append(cmd) or "12345")
    remote = Mock(return_value={"quiescent": True, "removed": True})
    monkeypatch.setattr(backend, "remote_python", remote)
    with pytest.raises(RuntimeError, match="retained"):
        backend.cleanup()
    remote.assert_not_called()
    assert "abc def" not in commands[0] and "abcdefg[h]" in commands[0]
    monkeypatch.setattr(backend, "_ssh_exec", lambda cmd: "")
    assert backend.cleanup() == {"quiescent": True, "removed": True}
    code = remote.call_args.args[0]
    assert "os.lstat" in code and "st_uid==os.getuid()" in code and "0o700" in code
    assert repr(backend.scratch) in code


def test_l3_regression_is_counted_and_never_runs_real_performance():
    backend = AcceptanceSSH(host="offline")
    verifier = L2OnlyVerifier(backend)
    result = verifier._verify_performance(GOOD, case("tail").materialize())
    assert verifier.l3_attempts == 1 and not result.passed
    assert result.level == Level.L3_PERFORMANCE and backend.trace == []


def test_remote_cleanup_precedes_fallible_local_transport_write(monkeypatch, tmp_path):
    import benchmarks.input_fidelity_h20 as driver
    events = []
    backend = Mock(trace=[], forbidden_actions=[])
    backend.prepare.return_value = {}
    backend.cleanup.side_effect = lambda: events.append("cleanup") or {"quiescent": True}
    def write(role, path, data):
        if path == "measurements/transport.json":
            events.append("transport-write")
            raise OSError("disk full")
    monkeypatch.setattr(driver, "AcceptanceSSH", lambda **kw: backend)
    monkeypatch.setattr(driver, "run_layout", lambda _: SimpleNamespace(root=str(tmp_path), write_for=write))
    monkeypatch.setattr(driver, "trials", lambda: [])
    with pytest.raises(OSError, match="disk full"):
        driver.run(SimpleNamespace(host="offline", gpu=None, output=str(tmp_path)))
    assert events == ["cleanup", "transport-write"]
