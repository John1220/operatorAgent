"""Offline acceptance-gate guards; never represented as GPU evidence."""
import copy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from benchmarks.drivers.rpc_deadline_h20 import assess_stall, assess_numeric
from benchmarks.drivers.input_fidelity_h20 import AcceptanceSSH


def stall(mode="stopped"):
    return dict(session_state="timeout", session_phase="response" if mode == "late" else "send",
                elapsed=.51, bytes_sent=8192, payload_bytes=16*1024*1024, restart_refused=True,
                cleanup_elapsed=.3, cleanup=dict(quiescent=True, remote=dict(quiescent=True)))


@pytest.mark.parametrize("mode", ["stopped", "partial", "late"])
def test_stall_requires_both_actual_deadline_and_owned_quiescence(mode):
    row = stall(mode)
    assert assess_stall(row, mode, .5) == []
    for key in ("session_state", "session_phase", "elapsed", "restart_refused", "cleanup", "cleanup_elapsed"):
        bad = copy.deepcopy(row)
        bad.pop(key)
        assert assess_stall(bad, mode, .5), key
    for key, value in (("elapsed", .001), ("elapsed", 4), ("cleanup_elapsed", 25),
                       ("session_state", "eof"), ("restart_refused", False)):
        bad = dict(row, **{key: value})
        assert assess_stall(bad, mode, .5)
    bad = copy.deepcopy(row)
    bad["cleanup"]["remote"]["quiescent"] = False
    assert assess_stall(bad, mode, .5)


@pytest.mark.parametrize("sent", [0, 16*1024*1024, 32*1024*1024])
def test_stopped_reader_requires_witness_of_partial_send(sent):
    assert assess_stall(dict(stall(), bytes_sent=sent), "stopped", .5)


def test_numeric_gate_cannot_be_satisfied_by_no_gpu_or_partial_evidence():
    from operator_agent.decide.input_fidelity import describe_inputs
    expected = describe_inputs({"X": np.ones((2,3),np.float32)})
    for response in ({}, {"success": True, "max_error": 0, "coverage": 1.0, "output_digest": "fake"}):
        assert assess_numeric(response, expected)


def test_failed_owned_cleanup_retains_scratch_before_directory_probe(monkeypatch):
    backend = AcceptanceSSH(host="offline")
    backend.scratch = "/tmp/opag-fidelity-abcdefgh"
    monkeypatch.setattr(backend, "close", lambda: {"quiescent": False, "remote": {"reason": "token mismatch"}})
    ssh = Mock(side_effect=AssertionError("must not remove/probe scratch after failed ownership cleanup"))
    monkeypatch.setattr(backend, "_ssh_exec", ssh)
    monkeypatch.setattr(backend, "remote_python", ssh)
    with pytest.raises(RuntimeError, match="retained"):
        backend.cleanup()
    ssh.assert_not_called()


def test_driver_cleanup_precedes_evidence_write_on_prepare_failure(tmp_path, monkeypatch):
    import benchmarks.drivers.rpc_deadline_h20 as driver
    events = []
    backend = Mock(forbidden_actions=[])
    backend.prepare.side_effect = RuntimeError("probe failed")
    backend.cleanup.side_effect = lambda: events.append("cleanup") or dict(quiescent=True, removed=True)
    monkeypatch.setattr(driver, "AcceptanceSSH", lambda **kw: backend)
    def write(*args):
        events.append("write")
        raise OSError("disk full")
    monkeypatch.setattr(driver, "run_layout", lambda _: SimpleNamespace(write_for=write))
    with pytest.raises(OSError, match="disk full"):
        driver.run(SimpleNamespace(output=str(tmp_path), host="offline", gpu=None))
    assert events == ["cleanup", "write"]
