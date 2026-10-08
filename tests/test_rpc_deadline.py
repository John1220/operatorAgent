"""Real pipe backpressure, shared deadlines and fail-closed resident recovery."""
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from operator_agent import remote_executor as rx
from operator_agent.failure_route import classify, ROUTE_HOLD
from operator_agent.measurement import TRANSPORT_UNKNOWN, transport_outcome
from operator_agent.mock_gpu import MockGPU
from operator_agent.schemas import VerificationLevel as Level, VerificationResult
from operator_agent.verifier import Verifier
from remote_scripts import session_control as control

ROOT = Path(__file__).resolve().parents[1]
ECHO = """import sys,json
print('ready',flush=True)
for line in sys.stdin:
 r=json.loads(line)
 print(json.dumps(dict(id=r['id'],result=dict(n=len(r.get('blob','')),shutdown=r.get('action')=='shutdown'))),flush=True)
 if r.get('action')=='shutdown': break
"""


def session(body, timeout=1):
    s = rx._RemoteSession([sys.executable, "-u", "-c", body], timeout, "offline")
    s.start()
    return s


def controller(code):
    """A regression must fail, not hang pytest; also reap its own child group."""
    p = subprocess.Popen([sys.executable, "-c", textwrap.dedent(code)],
                         cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, start_new_session=True)
    try:
        out, err = p.communicate(timeout=12)
        assert p.returncode == 0, out + err
        return json.loads(out.strip().splitlines()[-1])
    finally:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait(timeout=2)


@pytest.mark.parametrize("partial", [False, True])
def test_stopped_or_partial_reader_bounds_send_and_broken_close(partial):
    peer = ("import sys,time; print('ready',flush=True); "
            + ("sys.stdin.buffer.read(1024); " if partial else "") + "time.sleep(60)")
    row = controller(f"""
        import json,sys,time
        from operator_agent.remote_executor import _RemoteSession,SessionError
        s=_RemoteSession([sys.executable,'-u','-c',{peer!r}],.2,'fake')
        s.start()
        until=time.monotonic()+2
        while not s._noise_tail and time.monotonic()<until: time.sleep(.01)
        started=time.monotonic()
        try:
            s.call({{'action':'verify','blob':'x'*(16*1024*1024)}})
            raise AssertionError('accepted a stalled pipe')
        except SessionError as e:
            assert e.session_state=='timeout' and e.session_phase=='send'
            sent=e.bytes_sent
        elapsed=time.monotonic()-started
        assert .1 < elapsed < .65 and 0 < sent < 16*1024*1024
        assert not s.alive
        try: s.call({{'action':'ping'}})
        except SessionError: pass
        else: raise AssertionError('reused partial JSON stream')
        assert s._next_id==1
        started=time.monotonic()
        cleanup=s.close(grace=.3)
        assert time.monotonic()-started < .65
        assert cleanup['local_quiescent'] and not cleanup['shutdown_ack']
        assert s._next_id==1, 'shutdown appended to partial JSON'
        print(json.dumps(dict(sent=sent,elapsed=elapsed)))
    """)
    assert row["sent"] > 0


def test_large_unicode_request_is_one_intact_line_and_followup_ids_match():
    s = session(ECHO, 5)
    try:
        blob = "汉字🙂" * (1024 * 1024)
        assert s.call({"action": "verify", "blob": blob})["n"] == len(blob)
        assert s.call({"action": "ping"})["n"] == 0
        assert s._next_id == 2 and s.alive
        assert s.close(grace=1)["shutdown_ack"]
    finally:
        s.close(grace=.2)


def test_serialization_wait_is_bounded_and_late_encoder_cannot_send(monkeypatch):
    s = session(ECHO)
    original = rx.json.dumps
    def slow(*args, **kw):
        time.sleep(.35)
        return original(*args, **kw)
    try:
        monkeypatch.setattr(rx.json, "dumps", slow)
        started = time.monotonic()
        with pytest.raises(rx.SessionError) as exc:
            s.call({"action": "verify"}, timeout=.1)
        assert time.monotonic() - started < .3
        assert exc.value.session_phase == "serialize" and exc.value.bytes_sent == 0
        time.sleep(.4)
        assert s._queue.empty() and not s.alive
    finally:
        s.close(grace=.2)


def test_send_and_response_use_the_same_budget(monkeypatch):
    s = session(ECHO.replace("r=json.loads(line)", "r=json.loads(line); __import__('time').sleep(.5)"))
    original = s._send
    def slow(data, deadline):
        time.sleep(.18)
        return original(data, deadline)
    monkeypatch.setattr(s, "_send", slow)
    try:
        started = time.monotonic()
        with pytest.raises(rx.SessionError) as exc:
            s.call({"action": "verify"}, timeout=.3)
        assert exc.value.session_phase == "response"
        assert .25 < time.monotonic() - started < .5
    finally:
        s.close(grace=.2)


def test_lock_deadline_does_not_poison_another_request():
    s = session(ECHO)
    s._lock.acquire()
    try:
        with pytest.raises(rx.SessionError) as exc:
            s.call({"action": "ping"}, timeout=.05)
        assert exc.value.session_phase == "lock" and exc.value.bytes_sent == 0
        assert s.alive and s._next_id == 0
    finally:
        s._lock.release()
    assert s.call({"action": "ping"})["n"] == 0
    s.close(grace=.2)


def test_close_cancels_a_waiting_call_without_waiting_for_request_timeout():
    s = session("import sys,time; print('ready',flush=True); sys.stdin.readline(); time.sleep(60)", 30)
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(s.call, {"action": "verify"})
        until = time.monotonic()+2
        while not s._bytes_sent and time.monotonic()<until: time.sleep(.005)
        assert s._bytes_sent > 0
        started = time.monotonic()
        assert s.close(grace=.3)["local_quiescent"]
        with pytest.raises(rx.SessionError):
            future.result(timeout=.5)
        assert time.monotonic()-started < .8
    finally:
        s.close(grace=.2)
        pool.shutdown(wait=False)


def test_failed_startup_keeps_early_identity_for_owned_cleanup(monkeypatch):
    identity = dict(pid=987654, uid=os.getuid(), start_ticks="1", script="/private/runner_server.py")
    body = f"import time; print({json.dumps({'session_identity':identity})!r},flush=True); time.sleep(60)"
    ex = rx.SSHExecutor(host="offline", persistent=True, timeout=.2)
    ex._deployed, ex._gpu_pin = True, -1
    monkeypatch.setattr(ex, "_session_argv", lambda _: [sys.executable,"-u","-c",body])
    try:
        with pytest.raises(rx.SessionError):
            ex._session_call({"action": "verify"})
        assert ex._session is None and ex._pending_session.identity == identity
        cleanup = Mock(return_value=dict(quiescent=True))
        monkeypatch.setattr(ex, "_cleanup_session_owner", cleanup)
        assert ex.close(grace=.3)["quiescent"]
        assert cleanup.call_args.args[0].identity == identity
    finally:
        ex.close(grace=.2)


@pytest.mark.parametrize("budget", [0, -1, float("inf"), float("nan")])
def test_invalid_deadline_does_not_send(budget):
    s = session(ECHO)
    try:
        with pytest.raises(ValueError):
            s.call({"action": "verify"}, budget)
        assert s.alive and s._next_id == 0
    finally:
        s.close(grace=.2)


def test_failed_encoding_does_not_corrupt_stream():
    s = session(ECHO)
    try:
        with pytest.raises(TypeError):
            s.call({"action": "verify", "bad": object()})
        assert s.alive and s.call({"action": "ping"})["n"] == 0
    finally:
        s.close(grace=.2)


@pytest.mark.parametrize("reply", [dict(id=True, result={}), dict(id=1, result=["bad"])])
def test_invalid_envelope_breaks_stream(reply):
    s = session("import sys; sys.stdin.readline(); print(" + repr(json.dumps(reply)) + ",flush=True)")
    try:
        with pytest.raises(rx.SessionError) as exc:
            s.call({"action": "ping"})
        assert exc.value.session_state == "desync" and not s.alive
    finally:
        s.close(grace=.2)


@pytest.mark.parametrize("state", ["timeout", "eof", "desync", "broken"])
def test_session_errors_are_unknown_hold_not_numeric_or_retry(state):
    error = rx.SessionError(state, "transport test", phase="send")
    assert transport_outcome(error) == TRANSPORT_UNKNOWN
    ex = rx.SSHExecutor(host="offline", persistent=True)
    ex._deployed = True
    ex._gpu_pin = -1
    ex._ssh_exec_json = Mock(side_effect=error)
    inputs = {"X": np.ones((2, 3), np.float32)}
    assert ex.execute("code", inputs).transport_outcome == TRANSPORT_UNKNOWN
    for result in (ex.verify_function("code", inputs, "relu", 1e-5),
                   ex.benchmark("code", inputs),
                   ex.benchmark_with_baselines("code", inputs, "relu")):
        assert result["transport_outcome"] == TRANSPORT_UNKNOWN
    with pytest.raises(rx.SessionError):
        ex.compile("code")
    verifier = Verifier(ex)
    verifier._l1_caps = None
    result = verifier._verify_compile("def kernel(X): return X")
    assert result.metrics["compile_status"] == "transport_unknown"
    route = classify([result])
    assert route.route == ROUTE_HOLD and route.kind == "infra"
    assert not route.counts_as_iteration and not route.pollutes_learning


def test_broken_executor_never_auto_restarts_and_unconfirmed_cleanup_latches(monkeypatch):
    ex = rx.SSHExecutor(host="offline", persistent=True)
    s = session(ECHO)
    s.state = "broken"
    ex._session = s
    ex._deployed = True
    argv = Mock(side_effect=AssertionError("must not spawn"))
    monkeypatch.setattr(ex, "_session_argv", argv)
    try:
        for _ in range(2):
            with pytest.raises(rx.SessionError, match="禁止自动重试"):
                ex._session_call({"action": "verify"})
        monkeypatch.setattr(ex, "_cleanup_session_owner", lambda *a: {"quiescent": False})
        assert ex.close(grace=.3)["quiescent"] is False
        with pytest.raises(rx.SessionError, match="禁止新建"):
            ex._session_call({"action": "verify"})
        monkeypatch.setattr(ex, "_cleanup_session_owner", lambda *a: {"quiescent": True})
        assert ex.close(grace=.3)["quiescent"] is True
        assert ex._pending_session is None
        argv.assert_not_called()
    finally:
        s.close(grace=.2)


def test_cleanup_control_timeout_is_bounded_and_redacted(monkeypatch):
    ex = rx.SSHExecutor(host="offline", persistent=True)
    s = session(ECHO)
    s.identity = dict(pid=987654, uid=os.getuid(), start_ticks="123", script="/tmp/runner_server.py")
    s.owner_token = "a" * 32
    monkeypatch.setattr(rx.subprocess, "run", Mock(side_effect=subprocess.TimeoutExpired(
        ["ssh", "-i", "/secret/key", "user@host", s.owner_token], .1)))
    result = ex._cleanup_session_owner(s, .1)
    assert result["quiescent"] is False
    assert "secret" not in str(result) and s.owner_token not in str(result)
    assert rx.subprocess.run.call_args.kwargs["timeout"] == .1
    s.close(grace=.2)


def test_session_error_text_is_redacted_before_compile_persistence():
    exc = rx.SessionError("eof", "ssh -i /secret/key -p 2222 unit@203.0.113.9: reset")
    ex = rx.SSHExecutor(host="offline")
    ex.compile = Mock(side_effect=exc)
    verifier = Verifier(ex)
    verifier._l1_caps = None
    result = verifier._verify_compile("def kernel(X): return X")
    text = result.error_log + result.message
    for secret in ("/secret/key", "2222", "unit@", "203.0.113.9"):
        assert secret not in text
    assert result.metrics["transport_outcome"] == TRANSPORT_UNKNOWN


def test_owner_identity_is_emitted_before_heavy_imports():
    source = (ROOT / "remote_scripts/runner_server.py").read_text()
    assert source.index('"session_identity"') < source.index("import run_kernel")
    names = [p.name for p, _ in rx.SSHExecutor(host="offline")._remote_files()]
    assert names.index("session_control.py") < names.index("runner_server.py")


def test_missing_remote_identity_has_no_cleanup_authority(monkeypatch):
    ex = rx.SSHExecutor(host="offline", persistent=True)
    s = session(ECHO)
    run = Mock(side_effect=AssertionError("missing identity must not invoke remote cleanup"))
    monkeypatch.setattr(rx.subprocess, "run", run)
    try:
        result = ex._cleanup_session_owner(s, .2)
        assert result["quiescent"] is False
        run.assert_not_called()
    finally:
        s.close(grace=.2)


@pytest.mark.parametrize("where", ["primary", "case", "repeat"])
def test_suite_transport_unknown_stops_remaining_requests_and_preserves_hold(monkeypatch, where):
    from operator_agent import ConstraintSpec, SourceSpec, TargetSpec, TaskSchema
    from operator_agent.bench_cases import BenchmarkCase, TensorSpec
    from operator_agent.correctness import CorrectnessCase, CorrectnessSuite
    from operator_agent.mock_gpu import MockGPU
    suite = CorrectnessSuite(tuple(CorrectnessCase(str(i), BenchmarkCase(
        tensors=(TensorSpec("X", (2, 3), "fp32"),))) for i in range(3)))
    task = TaskSchema(source=SourceSpec(kernel_type="relu"), target=TargetSpec(dsl="python"),
                      constraints=ConstraintSpec(correctness_suite=suite))
    verifier = Verifier(MockGPU())
    passed = VerificationResult(Level.L2_FUNCTION, True, metrics=dict(
        max_error=0, coverage=1.0, compared_elements=6, output_digest="same"))
    unknown = VerificationResult(Level.L2_FUNCTION, False, metrics=dict(
        transport_outcome=TRANSPORT_UNKNOWN, session_state="timeout", session_phase="send"))
    monkeypatch.setattr(verifier, "_verify_primary_function", Mock(return_value=unknown if where == "primary" else passed))
    singles = Mock(side_effect=[passed, unknown] if where == "repeat" else [unknown])
    monkeypatch.setattr(verifier, "_verify_single_set", singles)
    result = verifier._verify_function("import numpy as np\ndef kernel(X): return X", {"X": np.ones((2,3))}, None, task)
    assert result.metrics["correctness"]["verdict"] == "unknown"
    assert result.metrics["transport_outcome"] == TRANSPORT_UNKNOWN
    assert singles.call_count == {"primary": 0, "case": 1, "repeat": 2}[where]
    assert any(r["verdict"] == "not_run" for r in result.metrics["correctness"]["cases"])
    route = classify([result])
    assert route.route == ROUTE_HOLD and route.kind == "infra"


@pytest.mark.parametrize("level", [Level.L1_COMPILE, Level.L2_FUNCTION, Level.L3_PERFORMANCE])
def test_conductor_unknown_transport_never_fixes_retries_or_enters_l3(tmp_path, monkeypatch, level):
    from operator_agent import ConstraintSpec, SourceSpec, TargetSpec, TaskSchema
    from operator_agent.artifact_store import ArtifactStore
    from operator_agent.conductor import Conductor
    from operator_agent.mock_gpu import MockGPU
    from tests.golden_regression import _FixedCodeMockLLM
    code = "import numpy as np\ndef kernel(A, B): return A @ B"
    conductor = Conductor(backend=MockGPU(), llm=_FixedCodeMockLLM(code=code), max_optimize_rounds=3)
    conductor.store = ArtifactStore(str(tmp_path))
    results = ([] if level == Level.L1_COMPILE else [VerificationResult(Level.L1_COMPILE, True)])
    if level == Level.L3_PERFORMANCE:
        results.append(VerificationResult(Level.L2_FUNCTION, True))
    results.append(VerificationResult(level, False, metrics={"transport_outcome": TRANSPORT_UNKNOWN}))
    verify = Mock(return_value=results)
    monkeypatch.setattr(conductor.verifier, "verify", verify)
    # CoderPipeline finalist selection only applies to L1/L2. This fixture tests
    # the normal three-level path, including an unknown L3 result.
    monkeypatch.setattr(conductor, "_coder_pipeline", None)
    fixer = Mock(side_effect=AssertionError("must not repair transport uncertainty"))
    monkeypatch.setattr(conductor.fixer, "run_with_action", fixer)
    perf = Mock(side_effect=AssertionError("must not benchmark transport uncertainty"))
    monkeypatch.setattr(conductor.verifier, "_verify_performance", perf)
    t = TaskSchema(source=SourceSpec(kernel_type="matmul"), target=TargetSpec(dsl="python"),
                   constraints=ConstraintSpec())
    artifact = conductor.execute(t, test_inputs={"A": np.ones((2,3),np.float32), "B": np.ones((3,2),np.float32)})
    assert not artifact.all_passed and verify.call_count == 1
    assert artifact.metadata["stop"]["reason"] == "transport_unknown_hold"
    fixer.assert_not_called()
    perf.assert_not_called()
    assert len(conductor.failure_db) == 0


def test_l3_primary_transport_unknown_does_not_execute_fallback_or_cases(monkeypatch):
    from tests.test_correctness_suite import task
    verifier = Verifier(MockGPU())
    payload = dict(success=False, transport_outcome=TRANSPORT_UNKNOWN, session_phase="response")
    monkeypatch.setattr(verifier, "_bench_case", Mock(return_value=(payload, payload)))
    further = Mock(side_effect=AssertionError("unknown primary must not trigger any more measurement"))
    monkeypatch.setattr(verifier, "_publish_case_ledger", further)
    monkeypatch.setattr(verifier.backend, "execute", further)
    result = verifier._verify_performance("code", {"A":np.ones((2,2))}, 1, task())
    assert result.metrics["transport_outcome"] == TRANSPORT_UNKNOWN and not result.passed
    further.assert_not_called()


def test_l3_case_transport_unknown_stops_ledger_and_survives_to_metrics(monkeypatch):
    from tests.test_correctness_suite import task
    verifier = Verifier(MockGPU())
    payload = dict(success=False, transport_outcome=TRANSPORT_UNKNOWN, session_phase="send")
    benchmark = Mock(return_value=(payload,payload))
    monkeypatch.setattr(verifier, "_bench_case", benchmark)
    metrics = dict(speedup_vs_baseline=2)
    inputs = [{"A":np.ones((n,2)), "B":np.ones((2,n))} for n in (2,3,4)]
    verifier._publish_case_ledger(metrics, "code", inputs, task())
    assert benchmark.call_count == 1
    assert metrics["transport_outcome"] == TRANSPORT_UNKNOWN
    assert any("未继续测量" in v.get("reason","") for v in metrics["cases"].values())


def owner_fixture(tmp_path):
    identity = dict(pid=12345, uid=os.getuid(), start_ticks="98765", script="/private/runner_server.py")
    proc = tmp_path / str(identity["pid"])
    proc.mkdir()
    (proc / "stat").write_text("12345 (runner (weird)) S " + "0 " * 18 + "98765 0")
    (proc / "cmdline").write_bytes(b"python\0/private/runner_server.py\0")
    (proc / "environ").write_bytes((control.TOKEN_ENV + "=" + "a" * 32).encode() + b"\0")
    return identity, proc


@pytest.mark.parametrize("bad", ["uid", "birth", "argv", "token"])
def test_owner_mismatch_never_signals(tmp_path, monkeypatch, bad):
    identity, proc = owner_fixture(tmp_path)
    if bad == "uid": identity["uid"] += 1
    if bad == "birth": identity["start_ticks"] = "1"
    if bad == "argv": (proc / "cmdline").write_bytes(b"python\0/other/runner_server.py\0")
    if bad == "token": (proc / "environ").write_bytes(b"OPERATOR_AGENT_SESSION_TOKEN=wrong\0")
    validate = control._validate_owner
    monkeypatch.setattr(control, "_validate_owner", lambda i, t: validate(i, t, tmp_path))
    fd = os.open(os.devnull, os.O_RDONLY)
    monkeypatch.setattr(control.os, "pidfd_open", lambda pid: fd, raising=False)
    send = Mock()
    monkeypatch.setattr(control.signal, "pidfd_send_signal", send, raising=False)
    monkeypatch.setattr(control.select, "select", lambda *a: ([], [], []))
    result = control.cleanup_owned(identity, "a" * 32, .1)
    assert result["quiescent"] is False and result["signals"] == []
    send.assert_not_called()


def test_verified_pidfd_cleanup_waits_for_exit(tmp_path, monkeypatch):
    identity, _ = owner_fixture(tmp_path)
    validate = control._validate_owner
    monkeypatch.setattr(control, "_validate_owner", lambda i, t: validate(i, t, tmp_path))
    fd = os.open(os.devnull, os.O_RDONLY)
    monkeypatch.setattr(control.os, "pidfd_open", lambda pid: fd, raising=False)
    send = Mock()
    monkeypatch.setattr(control.signal, "pidfd_send_signal", send, raising=False)
    monkeypatch.setattr(control.select, "select", Mock(side_effect=[([], [], []), ([fd], [], [])]))
    result = control.cleanup_owned(identity, "a" * 32, .1)
    assert result["quiescent"] and result["identity_verified"]
    assert result["signals"] == ["SIGTERM"]
    send.assert_called_once_with(fd, signal.SIGTERM)


def test_requesting_termination_is_not_evidence_of_exit(tmp_path, monkeypatch):
    identity, _ = owner_fixture(tmp_path)
    validate = control._validate_owner
    monkeypatch.setattr(control, "_validate_owner", lambda i, t: validate(i, t, tmp_path))
    fd = os.open(os.devnull, os.O_RDONLY)
    monkeypatch.setattr(control.os, "pidfd_open", lambda pid: fd, raising=False)
    send = Mock()
    monkeypatch.setattr(control.signal, "pidfd_send_signal", send, raising=False)
    monkeypatch.setattr(control.select, "select", lambda *args: ([], [], []))
    result = control.cleanup_owned(identity, "a" * 32, .1)
    assert result["quiescent"] is False and result["signals"] == ["SIGTERM", "SIGKILL"]
    assert send.call_count == 2 and "deadline" in result["reason"]


def test_missing_pidfd_authority_never_claims_remote_quiescence(monkeypatch):
    monkeypatch.delattr(control.os, "pidfd_open", raising=False)
    result = control.cleanup_owned(dict(pid=123, uid=os.getuid(), start_ticks="1",
                                       script="/private/runner_server.py"), "a" * 32)
    assert result["quiescent"] is False and result["signals"] == []
