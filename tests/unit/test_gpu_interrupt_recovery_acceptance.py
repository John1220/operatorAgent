"""Acceptance cannot credit handshakes, other CUDA owners or local-only cleanup."""
import copy

import pytest

from benchmarks.gpu_interrupt_recovery import active_witness, terminal_errors, SLOW
from operator_agent.anticheat import static_audit
from operator_agent.measurement import TRANSPORT_UNKNOWN


def samples():
    return [dict(gpu_uuid="gpu-fixture", observed_monotonic=t,
                 owned=[dict(pid=123, start_ticks="45", gpu_uuid="gpu-fixture", sm_pct=20)])
            for t in (1.0, 2.0)]


@pytest.mark.parametrize("change", ["one", "zero_sm", "pid", "birth", "uuid", "too_close"])
def test_active_cuda_witness_needs_two_samples_of_same_owned_process(change):
    rows = samples()
    assert active_witness(rows)
    if change == "one": rows.pop()
    elif change == "zero_sm": rows[1]["owned"][0]["sm_pct"] = 0
    elif change == "pid": rows[1]["owned"][0]["pid"] = 124
    elif change == "birth": rows[1]["owned"][0]["start_ticks"] = "46"
    elif change == "uuid": rows[1]["owned"][0]["gpu_uuid"] = "another"
    else: rows[1]["observed_monotonic"] = 1.1
    assert not active_witness(rows)


def test_local_exit_or_memory_residency_cannot_certify_remote_cleanup():
    status = dict(phase="cancelled", cleanup=dict(worker_reaped=True, remote_execution="quiescent"))
    body = dict(error="task_cancelled", passed=False)
    assert terminal_errors("cancel", status, body, None, samples(), dict(owned=[])) == []
    bad = copy.deepcopy(status)
    bad["cleanup"]["remote_execution"] = "unknown"
    assert terminal_errors("cancel", bad, body, None, samples(), dict(owned=[]))
    assert terminal_errors("cancel", status, body, None, samples(), dict(owned=[dict(pid=123)]))
    assert terminal_errors("cancel", status, dict(body, passed=True), None, samples(), dict(owned=[]))


def test_timeout_must_be_actual_response_unknown_and_never_performance():
    status = dict(cleanup=dict(worker_reaped=True, remote_execution="quiescent"))
    result = dict(verdicts=[dict(level="L2", passed=False, metrics=dict(
        transport_outcome=TRANSPORT_UNKNOWN, session_state="timeout", session_phase="response"))])
    assert terminal_errors("timeout", status, dict(passed=False), result, samples(), dict(owned=[])) == []
    wrong = copy.deepcopy(result)
    wrong["verdicts"][0]["metrics"]["session_phase"] = "startup"
    assert terminal_errors("timeout", status, {}, wrong, samples(), dict(owned=[]))
    result["verdicts"].append(dict(level="L3"))
    assert terminal_errors("timeout", status, {}, result, samples(), dict(owned=[]))


def test_fixed_cuda_probe_is_finite_and_retains_existing_candidate_guards():
    assert static_audit(SLOW, dsl="triton") == []
    assert "torch.cuda.synchronize()" in SLOW
    assert "torch.cuda._sleep(40000000000)" in SLOW
    assert "while" not in SLOW
