"""Offline protocol/control coverage; fake CUDA timings are NOT GPU evidence."""
import copy
import hashlib
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from operator_agent import cuda_calibration as cc
from operator_agent.backend_caps import Cap
from operator_agent.conductor import Conductor
from operator_agent.kernel_db import KernelDatabase
from operator_agent.remote_executor import SSHExecutor, RemoteExecutor
from operator_agent.run_contract import atomic_json, protocol_identity
from tests.backend_fakes import DeclaredFakeBackend
from tests.unit.test_session_inputs import real_session  # actual deployed CPU server fixture
from tests.unit.test_oracle import _eval


def receipt(mode="heartbeat", window="native-1"):
    device = dict(visible_index=0, name="H20", count=1, cuda_visible_devices="2")
    ident = dict(protocol=cc.PROTOCOL, channel=cc.CHANNEL, config=cc.CONFIG,
        sources=cc.source_hashes(), selected_gpu=2, device=device, default_stream=True,
        reference_sha256=hashlib.sha256(cc.REFERENCE_SOURCE.encode()).hexdigest(),
        inputs=cc.describe_inputs(cc.fixed_inputs()), runtime=dict(torch="2.8", cuda="12.8",
        triton="3.4", driver="570", gpu_uuid="GPU-test", pci_bus_id="0000:00:00.0"))
    def stats(t): return dict(runs=30, min_time_ms=t, median_time_ms=t, max_time_ms=t)
    p = dict(success=True, mode=mode, identity=ident, identity_hash=cc.digest(ident),
        absolute_anchor_checked=False, outputs_exact=dict(good=True, slow=True),
        input_fidelity=dict(observed=copy.deepcopy(ident["inputs"]), tensor_devices={"X": "cuda:0"},
                            representation="torch-storage-v1"),
        good=stats(.01), slow=stats(.08), paired=dict(paired_runs=30, paired_block_samples=1,
        speedup_paired_median=8, arm_windows=dict(candidate=window, baseline=window),
        arm_identity={k: device.copy() for k in ("candidate", "baseline", "after")}))
    if mode == "audit":
        p["samples"] = {f"{k}-{c}": [.01] * cc.AUDIT_SAMPLES
                        for k in ("empty", "good") for c in ("flush", "no-flush", "host-e2e")}
        p["telemetry"] = dict(before={}, after={})
    return copy.deepcopy(p)


def change(p, path, value):
    target = p
    for k in path[:-1]: target = target[k]
    target[path[-1]] = value
    p["identity_hash"] = cc.digest(p["identity"])


@pytest.mark.parametrize("path,value", [
    (("identity", "protocol"), "old"), (("identity", "channel"), "cpu"),
    (("identity", "reference_sha256"), "0" * 64),
    (("identity", "sources", "run_kernel.py"), "0" * 64),
    (("identity", "config", "l2_flush_bytes"), 0),
    (("identity", "inputs", "X", "value_digest"), "0" * 64),
    (("identity", "selected_gpu"), 3), (("identity", "default_stream"), False),
    (("identity", "device", "cuda_visible_devices"), "2,3"),
    (("identity", "device", "name"), "cpu"),
    (("identity", "runtime", "driver"), ""),
    (("input_fidelity", "tensor_devices", "X"), "cpu"),
    (("input_fidelity", "observed", "X", "strides"), [4, 4]),
    (("outputs_exact", "slow"), False), (("mode",), "audit"),
    (("absolute_anchor_checked",), True), (("good", "runs"), 29),
    (("slow", "runs"), 31), (("paired", "paired_runs"), 28),
    (("paired", "arm_windows", "baseline"), "other-window"),
    (("paired", "arm_identity", "after", "visible_index"), 1),
    (("paired", "paired_block_samples"), 11),
    (("paired", "speedup_paired_median"), float("nan")),
    (("good", "median_time_ms"), 0), (("slow", "max_time_ms"), float("inf")),
    (("good", "min_time_ms"), True),
])
def test_receipt_drift_or_missing_evidence_fails_closed(path, value):
    p = receipt()
    change(p, path, value)
    with pytest.raises(ValueError):
        cc.validate_probe(p, mode="heartbeat", selected_gpu=2)


def test_sample_counts_and_physical_gpu_must_be_known():
    p = receipt()
    for gpu in (None, True, -1, 3):
        with pytest.raises(ValueError): cc.validate_probe(p, mode="heartbeat", selected_gpu=gpu)
    p["identity_hash"] = "0" * 64
    with pytest.raises(ValueError, match="hash"): cc.validate_probe(p, mode="heartbeat", selected_gpu=2)


class NativeBackend(DeclaredFakeBackend):
    @property
    def selected_gpu(self): return 2


def conductor(probe):
    b = NativeBackend(calibration_probe=probe,
                      benchmark=lambda *a, **k: pytest.fail("CPU proxy cannot certify CUDA"))
    c = Conductor(backend=b, kernel_db=KernelDatabase(), heartbeat_interval=1)
    c._environment["fingerprint"] = {"device": "cuda"}
    return c


def test_conductor_runs_native_once_per_due_measurement_and_binds_provenance():
    probe = Mock(return_value=receipt())
    c = conductor(probe)
    hb, expected = c._heartbeat_for_next_measurement()
    assert expected and hb.ok and hb.channel == cc.CHANNEL
    assert c._heartbeat_for_next_measurement() == (hb, True)
    probe.assert_called_once_with(mode="heartbeat")
    obs = _eval(heartbeat=hb, heartbeat_expected=True, candidate_window=hb.window_id,
        measure=lambda: dict(value=1.5, validity="valid", provenance=dict(
            heartbeat_channel="cpu-forged", heartbeat_identity_hash="forged", heartbeat_evidence={})))
    assert obs.provenance["heartbeat_channel"] == cc.CHANNEL
    assert obs.provenance["heartbeat_identity_hash"] == hb.identity_hash
    assert obs.provenance["heartbeat_evidence"] == hb.evidence


@pytest.mark.parametrize("bad", ["flat", "transport", "drift", "nonfinite"])
def test_native_failure_revokes_window_eligibility_without_cpu_fallback(bad):
    p = receipt()
    if bad == "flat": p["slow"] = p["good"]; p["paired"]["speedup_paired_median"] = 1
    if bad == "transport": p = dict(success=False, error="deadline expired")
    if bad == "drift": p["identity_hash"] = "0" * 64
    if bad == "nonfinite": p["good"]["median_time_ms"] = float("inf")
    hb, due = conductor(lambda **k: p)._heartbeat_for_next_measurement()
    assert due and not hb.ok
    assert "RULER_BROKEN" in str(hb.reasons)
    obs = _eval(heartbeat=hb, heartbeat_expected=True, candidate_window=hb.window_id)
    assert not obs.eligibility.can_persist and not obs.eligibility.can_verdict


def test_cuda_missing_capability_is_due_and_unchecked_not_cpu_green():
    c = conductor(lambda **k: receipt())
    c.backend = DeclaredFakeBackend(benchmark=lambda *a, **k: pytest.fail("CPU fallback"))
    assert c._heartbeat_for_next_measurement() == (None, True)


def test_independent_frozen_identity_drift_fails_heartbeat():
    c = conductor(lambda **k: receipt())
    c.cuda_calibration_identity_hash = "0" * 64
    hb, due = c._heartbeat_for_next_measurement()
    assert due and not hb.ok and "frozen calibration identity drift" in str(hb.reasons)


@pytest.mark.parametrize("bad", [None, True, 0, "short", "A" * 64, "g" * 64])
def test_conductor_rejects_ambiguous_frozen_identity_at_construction(bad):
    with pytest.raises(ValueError, match="exact SHA256"):
        Conductor(cuda_calibration_identity_hash=bad)


def test_calibrated_workload_rejects_tamper_or_gpu_drift_before_backend(tmp_path, monkeypatch):
    from benchmarks.drivers import durable_profile_h20 as driver
    frozen = cc.freeze_audits([receipt("audit", "a"), receipt("audit", "b")], 2)
    path = tmp_path / "frozen.json"
    atomic_json(path, frozen)
    def forbidden(*a, **k): pytest.fail("backend constructed before calibration validation")
    monkeypatch.setattr(driver, "ProfileSSH", forbidden)
    output = tmp_path / "not-created"
    args = driver.parser().parse_args(["--gpu", "3", "--cuda-calibration", str(path), "--output", str(output)])
    with pytest.raises(ValueError, match="physical GPU"): driver.run(args)
    args.gpu = 2
    frozen["hash"] = "0" * 64
    atomic_json(path, frozen)
    with pytest.raises(ValueError, match="content/hash mismatch"): driver.run(args)
    assert not output.exists()


def test_uniform_clock_scaling_is_explicitly_not_absolute_calibration():
    p = receipt()
    for stats in (p["good"], p["slow"]):
        for key in ("min_time_ms", "median_time_ms", "max_time_ms"): stats[key] *= 10
    hb = cc.native_heartbeat(NativeBackend(calibration_probe=lambda **k: p), epoch=1, window_id="w")
    assert hb.ok and hb.known_good_ratio == 1
    assert hb.evidence["absolute_anchor_checked"] is False
    assert "separation-only" in str(hb.reasons)


@pytest.mark.parametrize("path,value", [(("samples", "empty-flush"), [1] * 199),
    (("samples", "good-host-e2e"), [float("nan")] * 200), (("telemetry",), None)])
def test_audit_requires_every_bounded_raw_channel(path, value):
    p = receipt("audit")
    change(p, path, value)
    with pytest.raises(ValueError): cc.validate_probe(p, mode="audit", selected_gpu=2)


def test_freeze_and_readback_are_exact_and_require_independent_windows(tmp_path):
    a, b = receipt("audit", "a"), receipt("audit", "b")
    f = cc.freeze_audits([a, b], 2)
    path = tmp_path / "calibration.json"
    atomic_json(path, f, readonly=True)
    assert cc.read_calibration(path) == f
    assert f["floor_update_authorized"] is False
    for receipts in ([a], [a, a], [a, b, a]):
        with pytest.raises(ValueError): cc.freeze_audits(receipts, 2)
    path.chmod(0o600)
    f["floor_update_authorized"] = True
    atomic_json(path, f)
    with pytest.raises(ValueError): cc.read_calibration(path)


def test_identity_and_good_slow_must_agree_in_both_audit_windows():
    a, b = receipt("audit", "a"), receipt("audit", "b")
    change(b, ("identity", "runtime", "driver"), "other-driver")
    with pytest.raises(ValueError, match="identities differ"): cc.freeze_audits([a, b], 2)
    b = receipt("audit", "b")
    b["paired"]["speedup_paired_median"] = 1
    with pytest.raises(ValueError, match="distinguish"): cc.freeze_audits([a, b], 2)


def test_native_method_is_persistent_only_and_facade_only_forwards():
    assert Cap.NATIVE_CUDA_CALIBRATION not in SSHExecutor(host="h").capabilities
    ssh = SSHExecutor(host="h", persistent=True)
    assert Cap.NATIVE_CUDA_CALIBRATION in ssh.capabilities
    assert Cap.NATIVE_CUDA_CALIBRATION not in SSHExecutor(host="h", platform="xpu").capabilities
    with pytest.raises(ValueError): SSHExecutor(host="h", platform="xpu", persistent=True)
    p = receipt()
    ssh._gpu_pin = 2
    ssh._deployed = True
    ssh._ssh_exec_json = Mock(return_value=p)
    assert ssh.calibration_probe() == p
    facade = RemoteExecutor(mode="ssh")
    facade._backend = ssh
    assert facade.calibration_probe(mode="audit")["success"] is False  # missing audit samples


def test_production_protocol_and_deployment_cover_shared_native_source():
    assert "operator_agent/cuda_calibration.py" in protocol_identity()
    files = [p.name for p, _ in SSHExecutor(host="h")._remote_files()]
    assert files.index("cuda_calibration.py") < files.index("runner_server.py")


@pytest.mark.parametrize("req", [{"action": "cuda_calibration_probe", "mode": "other"},
    {"action": "cuda_calibration_probe", "mode": "heartbeat", "code": "raise Exception()"},
    {"action": "cuda_calibration_probe", "mode": "audit", "repeat": 100000000}])
def test_real_server_rejects_arbitrary_native_requests_before_execution(real_session, req):
    p = real_session.call(req)
    assert p["success"] is False and "native CUDA probe" in p["error"]
    assert real_session.call({"action": "ping"})["success"] is True


def test_real_cpu_server_cannot_mint_native_cuda_receipt(real_session):
    p = real_session.call(dict(action="cuda_calibration_probe", mode="heartbeat"))
    assert p["success"] is False and "CPU fallback forbidden" in p["error"]


def test_remote_probe_calls_production_helpers_not_its_own_event_loop(monkeypatch):
    p = receipt("audit")
    cuda = SimpleNamespace(is_available=lambda: True, current_stream=lambda: 0,
        default_stream=lambda: 0, synchronize=lambda: None)
    torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="12.8"),
        __version__="2.8", device=lambda x: x, equal=lambda a,b: a == b)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace(__version__="3.4"))
    monkeypatch.setattr("subprocess.check_output", lambda *a, **k: "570,GPU-test,0000:00:00.0")
    monkeypatch.setattr(cc, "to_torch_inputs", lambda *a: {"X": 1})
    monkeypatch.setattr(cc, "torch_input_readback", lambda *a: p["input_fidelity"])
    runner = SimpleNamespace(_SELECTED_GPU_ID=2, _L2_FLUSH_BYTES=cc.CONFIG["l2_flush_bytes"],
        WARMUP_MS_BUDGET=25, REPEAT_MS_BUDGET=50, MAX_REPEATS=5000,
        _arm_identity=lambda *a: p["identity"]["device"], read_gpu_state=lambda: {},
        _load_module=Mock(return_value=SimpleNamespace(kernel=lambda **k: 1,
                         slow=lambda **k: 1, empty=lambda **k: None)),
        _bench_pair_interleaved=Mock(return_value=(p["good"], p["slow"], p["paired"])),
        _timed_runs=Mock(return_value=[.01] * cc.AUDIT_SAMPLES))
    out = cc.run_probe("audit", {"mode": "audit"}, runner)
    assert out["identity"] == p["identity"]
    assert runner._bench_pair_interleaved.call_args.args[2:] == (10, 30, True)
    assert runner._timed_runs.call_count == 7
    assert [c.kwargs["flush_l2"] for c in runner._timed_runs.call_args_list] == [False, True, False, True, True, True, True]
    assert all(c.args[1] == 1 for c in runner._timed_runs.call_args_list[4:])
    runner._load_module.assert_called_once_with(cc.REFERENCE_SOURCE)
