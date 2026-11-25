import json
import shlex
import subprocess
import time
from pathlib import Path

import pytest

from operator_agent.execute.remote_executor import (
    SSHExecutor, cleanup_profile_transport)


def test_cleanup_profile_terminates_the_remote_group(monkeypatch):
    executor = SSHExecutor(host='h20', work_dir='/root/x')
    calls = []
    monkeypatch.setattr(executor, '_ssh_exec', lambda cmd, **kwargs: calls.append(cmd) or '')
    assert executor._cleanup_profile('/root/x/profile-1-abc') == 'attempted'
    assert calls and 'kill -TERM' in calls[0] and 'kill -KILL' in calls[0]
    assert '/root/x/profile-1-abc/pid' in calls[0]


def test_profile_timeout_settles_with_cleanup(monkeypatch, tmp_path):
    executor = SSHExecutor(host='h20', work_dir='/root/x')
    executor._deployed = True
    monkeypatch.setattr(executor, '_pinned_gpu', lambda: 7)
    monkeypatch.setattr(executor, '_ssh_exec', lambda cmd, **kwargs: '')

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired('ncu', 1)

    cleaned = []
    monkeypatch.setattr(executor, '_ncu_exec', timeout)
    monkeypatch.setattr(executor, '_cleanup_profile',
                        lambda directory: cleaned.append(directory) or 'attempted')
    result = executor.profile('code', {'protocol': 'numpy-seed-v1'},
                              artifact_dir=tmp_path / 'profile-1', timeout=60)
    assert result['ok'] is False and result['error'] == 'profile_timeout'
    assert result['cleanup'] == 'attempted' and cleaned


def test_profile_success_fetches_artifacts_and_publishes_transport(monkeypatch, tmp_path):
    executor = SSHExecutor(host='h20', work_dir='/root/x')
    executor._deployed = True
    monkeypatch.setattr(executor, '_pinned_gpu', lambda: 7)
    monkeypatch.setattr(executor, '_ssh_exec', lambda cmd, **kwargs: '')
    captured = {}

    def capture(payload, command, *, timeout):
        captured.update(payload=payload, command=command, timeout=timeout)

    def fetch(remote, local, *, timeout):
        name = Path(remote).name
        content = {'raw.csv': b'ID\n', 'report.ncu-rep': b'report',
                   'reexport.csv': b'ID\n', 'result.json': b'{"success":true}',
                   'rc': b'0\n', 'ncu.log': b'', 'ncu-version.txt': b'Version 2025.3\n'}[name]
        Path(local).write_bytes(content)

    monkeypatch.setattr(executor, '_ncu_exec', capture)
    monkeypatch.setattr(executor, '_scp_fetch', fetch)
    result = executor.profile('code', {'protocol': 'numpy-seed-v1'},
                              artifact_dir=tmp_path / 'profile-1', timeout=60)

    assert result['ok'] is True and result['rc'] == '0'
    assert result['gpu'] == 7 and result['ncu_version'] == 'Version 2025.3'
    assert captured['payload']['action'] == 'profile'
    assert captured['payload']['result_path'].endswith('/result.json')
    assert '--nvtx-include candidate/' in captured['command']
    assert '--cache-control none --clock-control none' in captured['command']
    assert 'export TMPDIR=' in captured['command'] and '/tmp;' in captured['command']
    transport = json.loads((tmp_path / 'profile-1' / 'transport.json').read_text())
    assert transport['remote_dir'] == result['remote_dir']
    assert transport['pid_path'] == result['remote_dir'] + '/pid'


def test_ssh_exec_caps_timeout_by_the_profile_deadline(monkeypatch):
    executor = SSHExecutor(host='h20', work_dir='/root/x')
    captured = {}

    def fake_run(cmd, **kwargs):
        captured['timeout'] = kwargs['timeout']
        return subprocess.CompletedProcess(cmd, 0, 'ok', '')

    monkeypatch.setattr('operator_agent.execute.remote_executor.subprocess.run', fake_run)
    executor._profile_deadline = time.monotonic() + 5
    executor._ssh_exec('true', timeout=300)
    assert captured['timeout'] <= 5


def test_scp_fetch_does_not_publish_a_truncated_artifact(tmp_path, monkeypatch):
    executor = SSHExecutor(host='h20', work_dir='/root/x')
    monkeypatch.setattr(executor, '_remote_size', lambda path, timeout=30: 100)

    def fake_run(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b'123456789')
        return subprocess.CompletedProcess(cmd, 0, '', '')

    monkeypatch.setattr('operator_agent.execute.remote_executor.subprocess.run', fake_run)
    target = tmp_path / 'report.ncu-rep'
    with pytest.raises(OSError):
        executor._scp_fetch('/root/x/report.ncu-rep', target)
    assert not target.exists()
    assert not (tmp_path / 'report.ncu-rep.part').exists()


def test_cleanup_profile_transport_kills_the_remote_group(tmp_path, monkeypatch):
    transport = tmp_path / 'transport.json'
    transport.write_text(json.dumps(dict(host='h20', user='root', port=2222,
                                         key_path=None, remote_dir='/root/x profile')))
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, '', '')

    monkeypatch.setattr('operator_agent.execute.remote_executor.subprocess.run', fake_run)
    assert cleanup_profile_transport(transport) == 'attempted'
    assert calls and 'kill -TERM' in calls[0][-1]
    assert shlex.quote('/root/x profile/pid') in calls[0][-1]
    assert json.loads(transport.read_text())['cleanup'] == 'attempted'


def test_pinned_gpu_reports_its_observed_uuid(monkeypatch):
    executor = SSHExecutor(host='fixture', work_dir='/root/x')
    queries = []

    def reply(command, **kwargs):
        queries.append(command)
        return '3, 57467, 0, GPU-634bbc76-9ab6-220a-48a8-27247d7acbe2, NVIDIA H20\n'

    monkeypatch.setattr(executor, '_ssh_exec', reply)
    monkeypatch.delenv('OPERATOR_AGENT_REQUIRED_GPU', raising=False)
    assert executor._pinned_gpu() == 3
    assert executor.selected_gpu_uuid == 'GPU-634bbc76-9ab6-220a-48a8-27247d7acbe2'
    assert 'uuid,name' in queries[0]


def test_profile_refuses_non_nvidia_runner(tmp_path):
    executor = SSHExecutor(host='fixture', work_dir='/root/x', platform='kunlunxin')
    result = executor.profile('code', {}, artifact_dir=tmp_path / 'profile')
    assert result['ok'] is False and result['error'] == 'profile_unsupported'
