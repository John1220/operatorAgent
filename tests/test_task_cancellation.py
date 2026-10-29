"""S2: local workers/fake HTTP only; no SSH, provider or device claims."""
import io
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from operator_agent.task_clock import monotonic
from operator_agent import autonomous, exec_guard, hitl_tools as ht
from operator_agent.task_cancellation import (cancellation, launch_identity, publish_json,
                                             request_cancel, stop_event)

DECL = dict(request='matmul fp32', dsl='python', shapes=[[8, 8, 8]], min_speedup=0)
REPO = Path(__file__).resolve().parents[1]


def test_task_clock_has_the_same_epoch_in_independent_processes():
    before = monotonic()
    child = subprocess.run([sys.executable, '-c',
        'from operator_agent.task_clock import monotonic;print(monotonic())'],
        cwd=str(REPO), check=True, capture_output=True, text=True, timeout=10)
    assert before <= float(child.stdout) <= monotonic()


def wait_for(predicate, timeout=15):
    end = time.monotonic()+timeout
    while time.monotonic() < end:
        value = predicate()
        if value:
            return value
        time.sleep(.02)
    pytest.fail('local fixture did not become ready')


def run_identity(tmp_path, **budget):
    run = tmp_path/'run'
    run.mkdir()
    identity = 'a'*32
    launch_identity(run, identity, kind='autonomous')
    (run/'autonomous.json').write_text(json.dumps(dict(invocation_id=identity,
        task_budget=dict(wall_clock_s=10, **budget))))
    return run, identity


def test_cancel_identity_idempotence_and_terminal_no_signal(tmp_path, monkeypatch):
    run, identity = run_identity(tmp_path)
    monkeypatch.setattr(os, 'kill', lambda *a: pytest.fail('cancel must not signal a PID'))
    monkeypatch.setattr(os, 'killpg', lambda *a: pytest.fail('cancel must not signal a PGID'))
    with pytest.raises(ht.ToolError, match='cancellation_refused'):
        ht.cancel_task(run, 'b'*32)
    with pytest.raises(ValueError):
        request_cancel(run, '../../other')
    first = request_cancel(run, identity)
    original = (run/'cancel-request.json').read_bytes()
    assert request_cancel(run, identity)['requested_at'] == first['requested_at']
    assert (run/'cancel-request.json').read_bytes() == original
    (run/'run-status.json').write_text(json.dumps(dict(phase='done', invocation_id=identity)))
    assert request_cancel(run, identity) == dict(ok=True, accepted=False, phase='done',
        invocation_id=identity, run_dir=str(run), reason='already_terminal', cleanup=None,
        stopped_reason=None)
    legacy = tmp_path/'legacy'
    legacy.mkdir()
    with pytest.raises(ValueError, match='legacy'):
        request_cancel(legacy, identity)


def test_immediate_cancel_uses_returned_launch_identity_and_starts_no_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(ht, '_llm_problem', lambda: None)
    monkeypatch.setattr(exec_guard, 'start_background_worker', lambda *a, **k: SimpleNamespace(pid=os.getpid()))
    run = tmp_path/'run'
    launched = ht.run_autonomous(DECL, run)
    request_cancel(run, launched['invocation_id'])
    monkeypatch.setattr(autonomous, '_start_worker', lambda *a: pytest.fail('cancelled before start'))
    assert autonomous.main([str(run)]) == 1
    status = ht.autonomous_status(run)
    assert status['phase'] == 'cancelled'
    assert status['cleanup']['status'] == 'no_worker_started'
    assert status['invocation_id'] == launched['invocation_id']
    assert status['budget']['model_calls']['reserved'] is None
    with pytest.raises(ht.ToolError, match='run_already_started'):
        ht.run_autonomous(DECL, run)


def test_cancel_cleans_owned_sessions_preserves_partial_and_unrelated_process(tmp_path, monkeypatch):
    run, identity = run_identity(tmp_path)
    ready = run/'ready'
    marker = run/'should-not-survive'
    child = 'import time;from pathlib import Path;time.sleep(2);Path('+repr(str(marker))+').touch()'
    script = ('import subprocess,sys,time;from pathlib import Path;'
        'subprocess.Popen([sys.executable,"-c",'+repr(child)+'],start_new_session=True);'
        'Path('+repr(str(ready))+').touch();time.sleep(60)')
    (run/'candidate').mkdir()
    partial = run/'candidate/source.py'
    partial.write_text('partial candidate remains')
    monkeypatch.setattr(autonomous, '_start_worker', lambda *a: subprocess.Popen(
        [sys.executable, '-c', script], start_new_session=True))
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'], start_new_session=True)
    def cancel_when_ready():
        wait_for(ready.exists)
        time.sleep(.3)  # allow the separate session to be observed
        request_cancel(run, identity)
    thread = threading.Thread(target=cancel_when_ready)
    thread.start()
    try:
        assert autonomous.main([str(run)]) == 1
        thread.join(timeout=3)
        status = ht.autonomous_status(run)
        assert status['phase'] == 'cancelled' and status['stopped_reason'] == 'user_cancelled'
        assert status['cleanup']['worker_reaped']
        assert status['cleanup']['remote_execution'] == 'unknown'
        assert status['cleanup']['provider_call'] == 'unknown'
        assert not status['summary']['all_passed']
        assert 'candidate' in status['partial_artifacts']
        assert partial.read_text() == 'partial candidate remains'
        assert unrelated.poll() is None
        time.sleep(2)
        assert not marker.exists()
        assert not request_cancel(run, identity)['accepted']
    finally:
        unrelated.kill()
        unrelated.wait()


@pytest.mark.parametrize('finish_before_cancel', [True, False])
@pytest.mark.parametrize('phase', ['done', 'failed'])
def test_terminal_result_vs_cancel_uses_event_order(tmp_path, monkeypatch, finish_before_cancel, phase):
    run, identity = run_identity(tmp_path)
    class Finished:
        pid = 123456789
        def poll(self): return 0
        def wait(self, **kw): return 0
    def start(run, deadline, started, invocation):
        if not finish_before_cancel:
            request_cancel(run, invocation)
        finished = monotonic()
        publish_json(run/autonomous.WORKER_RESULT, dict(phase=phase, invocation_id=invocation,
            finished_monotonic=finished, error='OriginalFailure' if phase == 'failed' else None))
        if finish_before_cancel:
            request_cancel(run, invocation)
        return Finished()
    monkeypatch.setattr(autonomous, '_start_worker', start)
    monkeypatch.setattr(autonomous, '_stop_worker', lambda *a: dict(status='fixture'))
    autonomous.main([str(run)])
    status = ht.autonomous_status(run)
    assert status['phase'] == (phase if finish_before_cancel else 'cancelled')
    if phase == 'failed' and finish_before_cancel:
        assert status['error'] == 'OriginalFailure'


def test_deadline_cancel_order_and_bad_marker(tmp_path):
    run, identity = run_identity(tmp_path)
    requested = request_cancel(run, identity)
    stamp = cancellation(run, identity)['requested_monotonic']
    assert requested['accepted']
    assert stop_event(run, identity, stamp+.1) == ('user_cancelled', stamp)
    assert stop_event(run, identity, stamp-.1) == ('budget_wall_clock', stamp-.1)
    assert cancellation(run, 'b'*32) is None


def test_stale_pgid_identity_is_not_signalled(monkeypatch):
    observed = exec_guard.OwnedGroups([11, 22])
    observed.identities = {11: 'worker-start', 22: 'old-start'}
    current = exec_guard.OwnedGroups([11])
    current.identities = {11: 'worker-start'}
    monkeypatch.setattr(exec_guard, '_process_rows', lambda: [
        (11, 1, 11, 'worker-start'), (22, 1, 22, 'unrelated-new-start')])
    killed = []
    monkeypatch.setattr(os, 'killpg', lambda group, sig: killed.append(group))
    class Process:
        pid = 11
        exited = False
        def poll(self): return 0 if self.exited else None
        def wait(self, **kw): self.exited = True
    result = exec_guard.stop_task_worker(Process(), observed, discover=lambda pid: (current, True))
    assert killed == [11]
    assert result['stale_groups_skipped'] == 1 and result['worker_reaped']


def test_lost_owner_and_reused_status_pid_are_explicit_unknown(tmp_path, monkeypatch):
    run, identity = run_identity(tmp_path)
    (run/'run-status.json').write_text(json.dumps(dict(phase='running', invocation_id=identity,
        pid=os.getpid(), pid_identity='old-owner-start')))
    monkeypatch.setattr(exec_guard, 'process_identity', lambda pid: 'different-process-start')
    monkeypatch.setattr(os, 'killpg', lambda *a: pytest.fail('unrelated process must survive'))
    response = request_cancel(run, identity)
    assert response['phase'] == 'lost' and not response['accepted']
    assert response['reason'] == 'owner_unavailable'
    status = ht.autonomous_status(run, identity)
    assert status['phase'] == 'lost' and status['cleanup']['status'] == 'cleanup_unknown'
    assert not (run/'cancel-request.json').exists()


def test_cleanup_signal_failure_and_failed_discovery_remain_unknown(monkeypatch):
    class Process:
        pid = 1234
        def poll(self): return None
        def wait(self, **kw): raise subprocess.TimeoutExpired('fixture', 2)
    groups = exec_guard.OwnedGroups([1234])
    groups.discovery_complete = False
    groups.identities = {1234: 'owned-start'}
    monkeypatch.setattr(exec_guard, '_process_rows', lambda: [(1234, 1, 1234, 'owned-start')])
    def refuse(*args):
        raise PermissionError('fixture cannot signal')
    monkeypatch.setattr(os, 'killpg', refuse)
    result = exec_guard.stop_task_worker(Process(), groups, discover=lambda pid: (groups, True))
    assert result['status'] == 'cleanup_unknown' and not result['worker_reaped']
    assert result['all_descendants_proven'] is False


# A trusted deliberately blocked dispatcher: isolates protocol/process lifecycle
# from candidate safety gates. Real verify dispatch is covered by test_mcp_server.
def controlled_worker_patch():
    return '''
import sys
from operator_agent import tool_calls, exec_guard
def start(directory):
    code = "from pathlib import Path;import time;Path("+repr(str(directory/'ready'))+").touch();print('worker noise',flush=True);time.sleep(60)"
    return exec_guard.start_task_worker([sys.executable,'-c',code],cwd=str(directory),log_path=directory/'execution.log')
tool_calls._start_call_worker = start
'''


@pytest.mark.parametrize('tool', ['operator_verify', 'operator_repair', 'operator_bench'])
def test_mcp_cancellation_notification_is_live_and_correlated(tmp_path, tool):
    run = tmp_path/'run'
    ht.prepare(DECL, run)
    script = controlled_worker_patch()+'\nfrom operator_agent.mcp_server import main\nmain()\n'
    proc = subprocess.Popen([sys.executable, '-c', script], cwd=str(REPO),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    replies = queue.Queue()
    threading.Thread(target=lambda: [replies.put(json.loads(line)) for line in proc.stdout], daemon=True).start()
    def send(value):
        proc.stdin.write(json.dumps(value)+'\n')
        proc.stdin.flush()
    try:
        send(dict(jsonrpc='2.0', id='active', method='tools/call',
            params=dict(name=tool, arguments=dict(run_dir=str(run), code='def kernel(A,B): return A@B'))))
        ready = wait_for(lambda: list((run/'tool-calls').glob('*/ready')))
        invocation = ready[0].parent.name
        send(dict(jsonrpc='2.0', id='ping', method='ping'))
        assert replies.get(timeout=5)['id'] == 'ping'  # worker still blocked
        status = ht.autonomous_status(run)
        assert status['calls'][0]['invocation_id'] == invocation
        send(dict(jsonrpc='2.0', method='notifications/cancelled', params=dict(requestId='active')))
        reply = replies.get(timeout=5)
        assert reply['id'] == 'active' and reply['result']['isError']
        body = json.loads(reply['result']['content'][0]['text'])
        assert body['phase'] == 'cancelled' and body['invocation_id'] == invocation
        assert body['cleanup']['worker_reaped']
        assert ht.autonomous_status(run, invocation)['phase'] == 'cancelled'
        assert 'worker noise' in (ready[0].parent/'execution.log').read_text()
        proc.stdin.close()
        assert proc.wait(timeout=5) == 0
        assert replies.empty()  # no response to the notification
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)


@pytest.mark.parametrize('sig', [signal.SIGINT, signal.SIGTERM])
def test_cli_interrupt_is_acknowledged_after_cleanup(tmp_path, sig):
    run = tmp_path/'run'
    ht.prepare(DECL, run)
    script = controlled_worker_patch()+'\nfrom operator_agent.cli import main\nsys.exit(main())\n'
    proc = subprocess.Popen([sys.executable, '-c', script, 'verify', '--run-dir', str(run),
        '--code', 'def kernel(A,B): return A@B'], cwd=str(REPO),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        ready = wait_for(lambda: list((run/'tool-calls').glob('*/ready')))
        proc.send_signal(sig)
        out, err = proc.communicate(timeout=5)
        body = json.loads(out)
        assert proc.returncode == 130, err
        assert body['phase'] == 'cancelled' and body['cleanup']['worker_reaped']
        assert (ready[0].parent/'result.json').is_file()
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)


def test_mcp_transport_termination_cleans_foreground_call(tmp_path):
    run = tmp_path/'run'
    ht.prepare(DECL, run)
    script = controlled_worker_patch()+'\nfrom operator_agent.mcp_server import main\nsys.exit(main())\n'
    proc = subprocess.Popen([sys.executable, '-c', script], cwd=str(REPO),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        proc.stdin.write(json.dumps(dict(jsonrpc='2.0', id=1, method='tools/call', params=dict(
            name='operator_verify', arguments=dict(run_dir=str(run), code='def kernel(A,B): return A@B'))))+'\n')
        proc.stdin.flush()
        ready = wait_for(lambda: list((run/'tool-calls').glob('*/ready')))
        proc.terminate()
        out, err = proc.communicate(timeout=5)
        assert proc.returncode == 130, err
        status = ht.autonomous_status(run, ready[0].parent.name)
        assert status['phase'] == 'cancelled' and status['cleanup']['worker_reaped']
        assert not out.strip()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_explicit_mcp_cancel_and_status_use_the_same_invocation(tmp_path):
    from operator_agent import mcp_server
    run, identity = run_identity(tmp_path)
    result = mcp_server._call_tool('operator_cancel', dict(run_dir=str(run), invocation_id=identity))
    assert json.loads(result['content'][0]['text'])['accepted']
    (run/'run-status.json').write_text(json.dumps(dict(phase='cancelled', invocation_id=identity)))
    status = mcp_server._call_tool('operator_status', dict(run_dir=str(run), invocation_id=identity))
    assert json.loads(status['content'][0]['text'])['phase'] == 'cancelled'
    with pytest.raises(ht.ToolError, match='cancellation_refused'):
        mcp_server._call_tool('operator_cancel', dict(run_dir=str(run)))


def test_real_blocked_model_cancel_keeps_reservation(tmp_path, monkeypatch):
    from tests.test_task_budget_runtime import Server
    from tests.test_autonomous import SKETCH
    with Server([json.dumps(SKETCH)], delay=3) as server:
        monkeypatch.setenv('LLM_BASE_URL', f'http://127.0.0.1:{server.server.server_port}/v1')
        monkeypatch.setenv('LLM_API_KEY', 'synthetic-cancel-fixture-key')
        monkeypatch.setenv('LLM_MODEL', 'local-cancel-fixture')
        run = tmp_path/'run'
        launched = ht.run_autonomous(DECL, run, wall_clock_s=20, max_llm_calls=2)
        wait_for(lambda: server.calls)
        request_cancel(run, launched['invocation_id'])
        status = wait_for(lambda: (s if (s := ht.autonomous_status(run))['phase'] != 'running' else None))
    assert status['phase'] == 'cancelled'
    assert status['budget']['model_calls']['reserved'] == 1
    assert status['budget']['model_calls']['remaining'] == 1
    assert status['summary']['all_passed'] is False
    assert status['budget']['remote_cancellation_proven'] is False
    assert (run/'model-journal/call-001.json').is_file()


def test_real_mcp_repair_cancel_preserves_completed_verification(tmp_path, monkeypatch):
    from tests.test_task_budget_runtime import Server
    from tests.test_autonomous import CODE
    run = tmp_path/'run'
    ht.prepare(DECL, run)
    with Server([CODE], delay=4) as server:
        monkeypatch.setenv('LLM_BASE_URL', f'http://127.0.0.1:{server.server.server_port}/v1')
        monkeypatch.setenv('LLM_MODEL', 'local-repair-cancel-fixture')
        monkeypatch.setenv('LLM_API_KEY', 'synthetic-repair-cancel-key')
        proc = subprocess.Popen([sys.executable, '-m', 'operator_agent.mcp_server'], cwd=str(REPO),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            request = dict(jsonrpc='2.0', id=7, method='tools/call', params=dict(name='operator_repair',
                arguments=dict(run_dir=str(run), code='def kernel(A,B): return A-B',
                               backend='mock', stop_after='L2', max_rounds=2)))
            proc.stdin.write(json.dumps(request)+'\n')
            proc.stdin.flush()
            wait_for(lambda: server.calls)
            proc.stdin.write(json.dumps(dict(jsonrpc='2.0', method='notifications/cancelled',
                                             params=dict(requestId=7)))+'\n')
            proc.stdin.flush()
            proc.stdin.close()
            proc.stdin = None
            out, err = proc.communicate(timeout=8)
            assert proc.returncode == 0, err
            replies = [json.loads(line) for line in out.splitlines()]
            assert len(replies) == 1 and replies[0]['id'] == 7
            body = json.loads(replies[0]['result']['content'][0]['text'])
            assert body['phase'] == 'cancelled' and body['cleanup']['worker_reaped']
            assert (run/'repair-1/candidate.py').read_text() == 'def kernel(A,B): return A-B'
            assert not (run/'repair-1/complete.json').exists()
            result = json.loads((run/'repair-1/step-1/result.json').read_text())
            assert result['levels']['L2'] is False
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)
