"""Production cleanup defaults with real local workers; no SSH or GPU execution."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest

from operator_agent import autonomous, backend_ownership as ownership, exec_guard, tool_calls
from operator_agent.task_cancellation import launch_identity, request_cancel

ROOT = Path(__file__).resolve().parents[2]
INVOCATION = 'a' * 32

# Runs inside the actual production worker entry point. Only the trusted work
# dispatcher and owned backend are doubles; process signalling and receipts are real.
WORKER = '''
import signal,sys,time
from pathlib import Path
from operator_agent import backend_ownership as ownership

def block(directory):
    class Backend:
        def close(self, grace=10):
            assert grace <= 10
            assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
            Path(directory, 'backend-closed').touch()
            return {'quiescent': True}
    ownership.register_backend(Backend(), persistent=True)
    ownership.session_starting()
    ownership.session_ready({'pid': 123, 'uid': 1000, 'start_ticks': '123',
                             'script': '/tmp/runner_server.py'})
    Path(directory, 'ready').touch()
    while True:
        time.sleep(.02)
'''


def wait_for(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(.02)
    pytest.fail('owned local worker did not reach the expected state')


def start(script, *args, log_path=None):
    return exec_guard.start_task_worker([sys.executable, '-c', script, *map(str, args)],
                                      cwd=str(ROOT), log_path=log_path)


@pytest.mark.parametrize('name, persistent, event', [
    ('operator_verify', True, 'cancel'),
    ('operator_bench', None, 'cancel'),  # bench's production default is persistent
    ('operator_repair', True, 'deadline'),
])
def test_foreground_defaults_close_owned_sessions_before_terminal_result(
        tmp_path, monkeypatch, name, persistent, event):
    monkeypatch.delenv('OPERATOR_AGENT_S4_GRACEFUL_CLEANUP', raising=False)
    (tmp_path / 'declaration.json').write_text('{}')
    script = WORKER + '''
from operator_agent import mcp_server, tool_calls
mcp_server.handle_message = lambda request: block(Path(sys.argv[1]))
sys.exit(tool_calls.main([sys.argv[1]]))
'''
    monkeypatch.setattr(tool_calls, '_start_call_worker', lambda directory:
        start(script, directory, log_path=directory / 'execution.log'))
    args = dict(run_dir=str(tmp_path), backend='ssh', host='local-fixture')
    if persistent is not None:
        args['persistent'] = persistent
    if event == 'deadline':
        args['wall_clock_s'] = 3
    call = tool_calls.ToolCall(name, args)
    try:
        wait_for((call.directory / 'ready').exists)
        assert call.graceful_remote is True
        if event == 'cancel':
            assert call.cancel()['accepted']
        response = wait_for(call.poll)
        body = json.loads(response['content'][0]['text'])
        status = json.loads((call.directory / 'run-status.json').read_text())
        expected = 'user_cancelled' if event == 'cancel' else 'budget_wall_clock'
        assert status['stopped_reason'] == expected
        assert status['cleanup']['worker_reaped'] is True
        assert status['cleanup']['remote_execution'] == 'quiescent'
        assert status['cleanup']['provider_call'] == 'unknown'
        assert status['cleanup']['all_descendants_proven'] is False
        assert (call.directory / 'backend-closed').is_file()
        assert not (call.directory / 'worker-result.json').exists()
        assert body['error'] == ('task_cancelled' if event == 'cancel' else 'task_budget_exhausted')
    finally:
        call.close()
        if call.process.poll() is None:
            call.process.kill()
            call.process.wait()


@pytest.mark.parametrize('event', ['cancel', 'deadline'])
def test_autonomous_defaults_close_owned_sessions_on_cancel_and_deadline(
        tmp_path, monkeypatch, event):
    monkeypatch.delenv('OPERATOR_AGENT_S4_GRACEFUL_CLEANUP', raising=False)
    launch_identity(tmp_path, INVOCATION, kind='autonomous')
    (tmp_path / 'autonomous.json').write_text(json.dumps(dict(
        invocation_id=INVOCATION, backend='ssh', persistent=True,
        task_budget=dict(wall_clock_s=3))))
    script = WORKER + '''
from operator_agent import autonomous
autonomous._worker_main = lambda run, *args: block(run)
sys.exit(autonomous.main(['--worker', *sys.argv[1:]]))
'''
    monkeypatch.setattr(autonomous, '_start_worker', lambda run, deadline, started, invocation:
        start(script, run, deadline, started, invocation, log_path=run / 'worker.log'))
    cancellation = None
    if event == 'cancel':
        def cancel_when_ready():
            wait_for((tmp_path / 'ready').exists)
            request_cancel(tmp_path, INVOCATION)
        cancellation = threading.Thread(target=cancel_when_ready)
        cancellation.start()
    try:
        assert autonomous.main([str(tmp_path)]) == (1 if event == 'cancel' else 0)
        status = json.loads((tmp_path / 'run-status.json').read_text())
        expected = 'user_cancelled' if event == 'cancel' else 'budget_wall_clock'
        assert status['stopped_reason'] == expected
        assert status['cleanup']['worker_reaped'] is True
        assert status['cleanup']['remote_execution'] == 'quiescent'
        assert status['cleanup']['provider_call'] == 'unknown'
        assert status['cleanup']['all_descendants_proven'] is False
        assert (tmp_path / 'backend-closed').is_file()
    finally:
        if cancellation:
            cancellation.join(timeout=10)


@pytest.mark.parametrize('options', [
    dict(backend='ssh', persistent=False),
    dict(backend='ssh'),
    dict(backend='mock', persistent=True),
    dict(backend='local', persistent=True),
])
def test_nonpersistent_or_other_backends_do_not_claim_owned_ssh_cleanup(options):
    assert ownership.owned_persistent_ssh(options) is False


def test_cleanup_failure_does_not_abandon_other_owned_sessions(tmp_path):
    closed = []
    class Interrupted:
        def close(self, grace=10):
            raise SystemExit(7)
    class Other:
        def close(self, grace=10):
            closed.append(True)
            return {'quiescent': True}
    ownership.begin_scope(tmp_path, INVOCATION)
    ownership.register_backend(Interrupted(), persistent=True)
    ownership.register_backend(Other(), persistent=True)
    ownership.session_starting()
    ownership.session_ready(dict(pid=123, uid=1000, start_ticks='123', script='/tmp/runner_server.py'))
    receipt = ownership.finish_scope()
    assert closed == [True]
    assert receipt['remote_execution'] == 'unknown'
    assert receipt['backends'] == [dict(persistent=True, quiescent=None),
                                  dict(persistent=True, quiescent=True)]


def test_worker_body_exit_survives_cleanup_and_repeated_signals(tmp_path):
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    class Backend:
        def close(self, grace=10):
            os.kill(os.getpid(), signal.SIGINT)
            os.kill(os.getpid(), signal.SIGTERM)
            return {'quiescent': True}
    with pytest.raises(SystemExit) as error:
        with ownership.owned_worker_scope(tmp_path, INVOCATION):
            ownership.register_backend(Backend(), persistent=True)
            ownership.session_starting()
            ownership.session_ready(dict(pid=123, uid=1000, start_ticks='123',
                                         script='/tmp/runner_server.py'))
            raise SystemExit(7)
    assert error.value.code == 7
    assert ownership.cleanup_state(tmp_path, INVOCATION) == 'quiescent'
    assert {sig: signal.getsignal(sig) for sig in previous} == previous
