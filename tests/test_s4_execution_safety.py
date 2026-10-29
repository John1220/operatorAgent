"""Local doubles for frozen GPU selection, existing remote files and owned cleanup."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from operator_agent import exec_guard
from operator_agent.backend_ownership import cleanup_state
from operator_agent.remote_executor import HardwareProbeFailed, SSHExecutor
from operator_agent import remote_executor

UUID = 'GPU-52ca6155-d8bc-a416-847a-6f0c71db1552'
INVOCATION = 'a'*32


def strict_env(monkeypatch):
    monkeypatch.setenv('OPERATOR_AGENT_REQUIRED_GPU', '7')
    monkeypatch.setenv('OPERATOR_AGENT_REQUIRED_GPU_UUID', UUID)
    monkeypatch.setenv('OPERATOR_AGENT_REQUIRED_GPU_NAME', 'NVIDIA H20')
    monkeypatch.setenv('OPERATOR_AGENT_EXISTING_REMOTE_ONLY', '1')


def test_required_gpu_uses_exact_identity_and_sufficient_idle_capacity(monkeypatch):
    strict_env(monkeypatch)
    executor = SSHExecutor(host='fixture', persistent=True)
    monkeypatch.setattr(executor, '_ssh_exec', lambda command, **kw:
        f'0, 99999, 0, GPU-other, NVIDIA H20\n7, 4096, 0, {UUID}, NVIDIA H20\n')
    assert executor._pinned_gpu() == 7


@pytest.mark.parametrize('entry', [
    f'7, 4096, 6, {UUID}, NVIDIA H20',
    f'7, 1000, 0, {UUID}, NVIDIA H20',
    '7, 4096, 0, GPU-other, NVIDIA H20',
    f'7, 4096, 0, {UUID}, NVIDIA A100',
    f'6, 4096, 0, {UUID}, NVIDIA H20',
    'unparseable',
])
def test_required_gpu_failure_never_falls_back_to_another_card(monkeypatch, entry):
    strict_env(monkeypatch)
    executor = SSHExecutor(host='fixture', persistent=True)
    monkeypatch.setattr(executor, '_ssh_exec', lambda command, **kw: entry)
    with pytest.raises(HardwareProbeFailed):
        executor._pinned_gpu()
    with pytest.raises(HardwareProbeFailed):
        executor._pinned_gpu()


def test_existing_remote_mode_never_deploys_or_installs(monkeypatch):
    strict_env(monkeypatch)
    executor = SSHExecutor(host='fixture', persistent=True)
    monkeypatch.setattr(executor, '_pinned_gpu', lambda: 7)
    def forbidden(*args, **kwargs):
        pytest.fail('existing remote mode must not install or deploy')
    monkeypatch.setattr(executor, '_setup_uv_env', forbidden)
    monkeypatch.setattr(executor, '_scp', forbidden)
    def readonly(command, **kwargs):
        if command.startswith('test -x '):
            return ''
        assert command.startswith('python3 -c ')
        return '\n'.join(str(remote)+' '+hashlib.sha256(Path(local).read_bytes()).hexdigest()
                         for local, remote in executor._remote_files())
    monkeypatch.setattr(executor, '_ssh_exec', readonly)
    assert executor.deploy()
    assert executor._verified_python == '.venv/bin/python'


def test_existing_remote_file_mismatch_stops_without_sync(monkeypatch):
    strict_env(monkeypatch)
    executor = SSHExecutor(host='fixture', persistent=True)
    monkeypatch.setattr(executor, '_pinned_gpu', lambda: 7)
    monkeypatch.setattr(executor, '_ssh_exec', lambda command, **kw: '')
    monkeypatch.setattr(executor, '_scp', lambda *a, **kw: pytest.fail('must not deploy'))
    with pytest.raises(HardwareProbeFailed):
        executor.deploy()
    assert executor._deployed is False


@pytest.mark.parametrize('selected,expected', [(7, True), (6, False), (None, False), (True, False)])
def test_required_gpu_must_match_remote_handshake(monkeypatch, selected, expected):
    strict_env(monkeypatch)
    executor = SSHExecutor(host='fixture', persistent=True)
    executor._deployed = True
    monkeypatch.setattr(executor, '_pinned_gpu', lambda: 7)
    monkeypatch.setattr(executor, '_session_argv', lambda command: [])
    class Session:
        identity = {'pid': 123, 'uid': 1000, 'start_ticks': '123',
                    'script': '/tmp/runner_server.py'}
        alive = True
        def __init__(self, *args, **kwargs): pass
        def start(self): pass
        def call(self, request): return {'pong': True, 'selected_gpu': selected}
        def close(self, grace=5): return {'local_quiescent': True}
    monkeypatch.setattr(remote_executor, '_RemoteSession', Session)
    monkeypatch.setattr(remote_executor.atexit, 'register', lambda *args: None)
    if expected:
        assert executor._start_session().identity == Session.identity
        assert executor.session_gpu == 7
    else:
        with pytest.raises(remote_executor.SessionError):
            executor._start_session()
        assert executor._pending_session is not None


def wait_for(path):
    end = time.monotonic()+5
    while time.monotonic() < end:
        if list(path.glob('owned-session-ready-*.json')):
            return
        time.sleep(.02)
    pytest.fail('controlled worker did not become ready')


def test_graceful_owned_worker_publishes_cleanup_before_parent_reaps(tmp_path):
    script = '''
import signal,sys,time
from operator_agent.backend_ownership import begin_scope,register_backend,session_starting,session_ready,finish_scope
begin_scope(sys.argv[1],sys.argv[2])
class Backend:
 def close(self,grace=10): return {'quiescent':True}
register_backend(Backend(),persistent=True)
session_starting();session_ready({'pid':123,'uid':1000,'start_ticks':'123','script':'/tmp/runner_server.py'})
signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
try:
 while True: time.sleep(.1)
finally:
 signal.signal(signal.SIGTERM,signal.SIG_IGN)
 finish_scope()
'''
    proc = exec_guard.start_task_worker([sys.executable, '-c', script, str(tmp_path), INVOCATION],
                                        cwd=str(Path(__file__).resolve().parents[1]))
    try:
        wait_for(tmp_path)
        cleanup = exec_guard.stop_task_worker(proc, graceful_s=3)
        assert cleanup['worker_reaped']
        assert cleanup_state(tmp_path, INVOCATION) == 'quiescent'
        receipt = json.loads((tmp_path/'owned-cleanup.json').read_text())
        assert receipt['sessions_ready'] == 1
        assert 'token' not in json.dumps(receipt)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_missing_mismatched_or_inconsistent_receipt_is_unknown(tmp_path):
    path = tmp_path/'owned-cleanup.json'
    assert cleanup_state(tmp_path, INVOCATION) == 'unknown'
    path.write_text(json.dumps(dict(schema='operator-owned-cleanup-v1', invocation_id=INVOCATION,
        remote_execution='quiescent', sessions_started=0, sessions_ready=0,
        backend_count=0, backends=[])))
    assert cleanup_state(tmp_path, INVOCATION) == 'unknown'
    assert cleanup_state(tmp_path, 'b'*32) == 'unknown'


def test_cleanup_receipt_rejects_non_boolean_proof(tmp_path):
    path = tmp_path/'owned-cleanup.json'
    path.write_text(json.dumps(dict(schema='operator-owned-cleanup-v1', invocation_id=INVOCATION,
        completed_at=1, remote_execution='quiescent', sessions_started=1, sessions_ready=1,
        backend_count=1, backends=[dict(persistent=True, quiescent=1)])))
    assert cleanup_state(tmp_path, INVOCATION) == 'unknown'


def test_invalid_runner_identity_cannot_publish_ready(tmp_path):
    from operator_agent.backend_ownership import begin_scope, session_starting, session_ready, finish_scope
    begin_scope(tmp_path, INVOCATION)
    try:
        session_starting()
        with pytest.raises(ValueError):
            session_ready({'pid': 123})
        assert not list(tmp_path.glob('owned-session-ready-*.json'))
    finally:
        finish_scope()


def test_interruption_before_handshake_cannot_claim_remote_quiescence(tmp_path):
    from operator_agent.backend_ownership import begin_scope, register_backend, session_starting, finish_scope
    class Backend:
        def close(self, grace=10):
            return {'quiescent': True}
    begin_scope(tmp_path, INVOCATION)
    register_backend(Backend(), persistent=True)
    session_starting()
    receipt = finish_scope()
    assert receipt['remote_execution'] == 'unknown'
    assert cleanup_state(tmp_path, INVOCATION) == 'unknown'


def test_forged_quiescence_with_incomplete_handshake_is_unknown(tmp_path):
    path = tmp_path/'owned-cleanup.json'
    path.write_text(json.dumps(dict(schema='operator-owned-cleanup-v1', invocation_id=INVOCATION,
        completed_at=1, remote_execution='quiescent', sessions_started=2, sessions_ready=1,
        backend_count=1, backends=[dict(persistent=True, quiescent=True)])))
    assert cleanup_state(tmp_path, INVOCATION) == 'unknown'
