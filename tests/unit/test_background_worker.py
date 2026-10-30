"""Trusted worker launch stays detached without weakening candidate guards."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from operator_agent import exec_guard, hitl_tools


REPO = Path(__file__).resolve().parents[2]


def _wait_for(path, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(.01)
    pytest.fail('worker did not publish its local test artifact')


def _cleanup(proc):
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=5)


def test_worker_gets_new_session_closed_stdin_fds_and_appended_log(tmp_path, monkeypatch):
    script = tmp_path/'trusted-worker.py'
    ready, release, result = [tmp_path/name for name in ('ready','release','result.json')]
    script.write_text('''import json, os, sys, time
from pathlib import Path
ready, release, result = map(Path, sys.argv[1:4])
try:
    os.fstat(int(sys.argv[4]))
    descriptor_leaked = True
except OSError:
    descriptor_leaked = False
data = dict(pid=os.getpid(), sid=os.getsid(0), pgid=os.getpgrp(), cwd=os.getcwd(),
            stdin=sys.stdin.read(), descriptor_leaked=descriptor_leaked,
            environment=os.environ.get('OPERATOR_WORKER_TEST_MARKER'))
ready.write_text('ready')
deadline = time.monotonic() + 5
while not release.exists() and time.monotonic() < deadline:
    time.sleep(.01)
result.write_text(json.dumps(data))
print('worker stdout', flush=True)
print('worker stderr', file=sys.stderr, flush=True)
''')
    log = tmp_path/'worker.log'
    log.write_text('previous log\n')
    marker = 'inherited worker environment'
    monkeypatch.setenv('OPERATOR_WORKER_TEST_MARKER', marker)
    source_fd = os.open(str(tmp_path/'unrelated-fd'), os.O_CREAT | os.O_RDWR, 0o600)
    leaked_fd = fcntl.fcntl(source_fd, fcntl.F_DUPFD, 40)
    os.set_inheritable(leaked_fd, True)
    proc = None
    try:
        proc = exec_guard.start_background_worker(
            [sys.executable, str(script), str(ready), str(release), str(result), str(leaked_fd)],
            cwd=tmp_path, log_path=log)
        _wait_for(ready)
        assert proc.poll() is None and not result.exists()
        assert proc.pid > 0
        release.write_text('continue')
        assert proc.wait(timeout=5) == 0
        data = json.loads(result.read_text())
        assert data['pid'] == data['sid'] == data['pgid'] == proc.pid
        assert data['cwd'] == str(tmp_path.resolve())
        assert data['stdin'] == '' and data['descriptor_leaked'] is False
        assert data['environment'] == marker
        assert log.read_text() == 'previous log\nworker stdout\nworker stderr\n'
    finally:
        if proc is not None:
            _cleanup(proc)
        os.close(leaked_fd)
        os.close(source_fd)


def test_worker_completes_after_its_launching_parent_exits(tmp_path):
    ready, release, result = [tmp_path/name for name in ('ready','release','result.json')]
    worker_script = tmp_path/'trusted-worker.py'
    worker_script.write_text('''import json, os, sys, time
from pathlib import Path
ready, release, result = map(Path, sys.argv[1:])
ready.write_text(json.dumps(dict(pid=os.getpid(), sid=os.getsid(0), pgid=os.getpgrp())))
deadline = time.monotonic() + 8
while not release.exists() and time.monotonic() < deadline:
    time.sleep(.01)
result.write_text('finished after launching parent exited')
print('detached worker finished', flush=True)
''')
    wrapper_script = tmp_path/'launcher.py'
    wrapper_script.write_text('''import json, sys
from operator_agent.exec_guard import start_background_worker
proc = start_background_worker(sys.argv[3:], cwd=sys.argv[1], log_path=sys.argv[2])
print(json.dumps(dict(pid=proc.pid)), flush=True)
''')
    log = tmp_path/'detached.log'
    env = dict(os.environ)
    env['PYTHONPATH'] = str(REPO) + os.pathsep + env.get('PYTHONPATH','')
    launcher = subprocess.Popen(
        [sys.executable, str(wrapper_script), str(tmp_path), str(log), sys.executable,
         str(worker_script), str(ready), str(release), str(result)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    pid = None
    completed = False
    try:
        output, error = launcher.communicate(timeout=5)
        assert launcher.returncode == 0, error
        pid = json.loads(output)['pid']
        _wait_for(ready)
        identity = json.loads(ready.read_text())
        assert identity == dict(pid=pid, sid=pid, pgid=pid)
        assert not result.exists()
        # The spawning interpreter has exited; the worker still reaches this
        # signal and publishes its result through descriptors it owns.
        release.write_text('parent already exited')
        _wait_for(result)
        assert result.read_text() == 'finished after launching parent exited'
        _wait_for(log)
        deadline = time.monotonic() + 2
        while 'detached worker finished' not in log.read_text() and time.monotonic() < deadline:
            time.sleep(.01)
        assert 'detached worker finished' in log.read_text()
        completed = True
    finally:
        if launcher.poll() is None:
            launcher.kill()
        launcher.wait(timeout=5)
        # The worker has an independent session, so cleanup must address only
        # that recorded child. Never signal the pytest parent's process group.
        if pid is not None and not completed:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize('argv', ['python -c anything', [], [''], ['python', 2],
                                  ['python', 'bad\x00argument']])
def test_worker_rejects_implicit_shell_or_invalid_vectors_before_opening_log(tmp_path, argv):
    log = tmp_path/'worker.log'
    with pytest.raises(ValueError, match='argument vector'):
        exec_guard.start_background_worker(argv, cwd=tmp_path, log_path=log)
    assert not log.exists()


def test_launch_failure_closes_the_parent_log_handle(tmp_path, monkeypatch):
    seen = []
    def cannot_start(*args, **kwargs):
        seen.append(kwargs['stdout'])
        assert not seen[0].closed
        raise OSError('synthetic launch failure')
    monkeypatch.setattr(exec_guard.subprocess, 'Popen', cannot_start)
    with pytest.raises(OSError, match='synthetic launch failure'):
        exec_guard.start_background_worker(['synthetic-worker'], cwd=tmp_path,
                                            log_path=tmp_path/'worker.log')
    assert seen[0].closed


def test_public_autonomous_entry_uses_central_trusted_worker_launcher(tmp_path, monkeypatch):
    run = tmp_path/'run'
    run.mkdir()
    monkeypatch.setattr(hitl_tools, '_llm_problem', lambda: None)
    monkeypatch.setattr(hitl_tools, 'prepare', lambda *a, **k: {'contract_hash':'synthetic-contract'})
    calls = []
    def launch(argv, **kwargs):
        calls.append((argv,kwargs))
        return SimpleNamespace(pid=12345)
    monkeypatch.setattr(exec_guard, 'start_background_worker', launch)
    result = hitl_tools.run_autonomous({'request':'synthetic'}, run, backend='mock', max_rounds=2)
    assert result['pid'] == 12345 and result['phase'] == 'running'
    assert result['contract_hash'] == 'synthetic-contract'
    argv, options = calls[0]
    assert argv == [sys.executable, '-m', 'operator_agent.autonomous', str(run)]
    assert Path(options['cwd']) == REPO
    assert options['log_path'] == run/'autonomous.log'
    config = json.loads((run/hitl_tools.CONFIG_FILE).read_text())
    assert config['max_rounds'] == 2 and config['backend'] == 'mock'
