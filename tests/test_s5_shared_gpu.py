"""Frozen shared-device checks use mocked read-only SSH observations only."""
import copy
import subprocess
from unittest.mock import patch

import pytest

from benchmarks import s5_shared_gpu as shared


GPU = 'GPU-13c6c4b4-45ce-8963-1122-df2b3d72fb76'
OTHER = 'GPU-52ca6155-d8bc-a416-847a-6f0c71db1552'
PIN = dict(host='algorithm-h20', gpu_index=2, gpu_uuid=GPU, gpu_name='NVIDIA H20')
POLICY = dict(mode='shared-known-processes', min_free_mib=8192, max_util_pct=5,
              allowed_compute_pids=[123, 456])


def response(text):
    return subprocess.CompletedProcess([], 0, stdout=text, stderr='')


def observe(apps='', gpu=None, **kwargs):
    gpu = gpu if gpu is not None else f'2, {GPU}, NVIDIA H20, 22246, 0\n'
    with patch.object(shared.subprocess, 'run', side_effect=[response(gpu), response(apps)]) as run:
        result = shared.snapshot(kwargs.get('pin', PIN), kwargs.get('policy', POLICY), env={})
    assert len(run.call_args_list) == 2
    for call in run.call_args_list:
        assert call.args[0][:6] == ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', 'algorithm-h20']
        assert call.kwargs['check'] and call.kwargs['timeout'] == 15
        assert '--format=csv,noheader,nounits' in call.args[0][-1]
    return result


def test_known_processes_pass_and_snapshot_keeps_full_target_observation():
    r = observe(f'{GPU}, 456, python, 5000\n{GPU}, 123, VLLM::EngineCore, 20000\n'
                f'{OTHER}, 999, other-python, 60000\n')
    assert r['gpu_index'] == 2 and r['uuid'] == GPU and r['name'] == 'NVIDIA H20'
    assert r['free_mib'] == 22246 and r['util_pct'] == 0 and r['other_compute_processes'] == 2
    assert r['exclusive_lease'] is False and r['resource_mode'] == 'shared-known-processes'
    assert r['observed_at'] > 0
    assert r['compute_apps'] == [dict(pid=123, process_name='VLLM::EngineCore', used_gpu_memory_mib=20000),
                                 dict(pid=456, process_name='python', used_gpu_memory_mib=5000)]


def test_known_process_may_exit_and_other_gpu_process_does_not_need_allowlisting():
    assert observe(f'{GPU}, 123, python, 10\n')['other_compute_processes'] == 1
    assert observe(f'{OTHER}, 999, python, 10\n')['other_compute_processes'] == 0
    assert observe()['compute_apps'] == []


def test_new_target_process_refuses():
    with pytest.raises(ValueError, match='unapproved compute process'):
        observe(f'{GPU}, 789, python, 10\n')


@pytest.mark.parametrize('row', [
    f'2, {GPU}, NVIDIA H20, 8191, 0\n',
    f'2, {GPU}, NVIDIA H20, 22246, 6\n',
    f'7, {GPU}, NVIDIA H20, 22246, 0\n',
    f'2, {GPU}, NVIDIA A100, 22246, 0\n',
    f'2, {OTHER}, NVIDIA H20, 22246, 0\n',
    f'2, {GPU}, NVIDIA H20, 22246, 0\n2, {GPU}, NVIDIA H20, 22246, 0\n',
])
def test_capacity_utilization_and_identity_are_still_required(row):
    with pytest.raises(ValueError):
        observe(gpu=row)


@pytest.mark.parametrize('apps', [
    'bad\n', f'{GPU}, N/A, python, 10\n', f'{GPU}, 123, python, N/A\n',
    f'{GPU}, 0, python, 10\n', f'{GPU}, 123, , 10\n',
    f'{GPU}, 123, python, 10\n{GPU}, 123, python, 10\n',
])
def test_malformed_process_observations_never_pass(apps):
    with pytest.raises(ValueError):
        observe(apps)


@pytest.mark.parametrize('changes', [
    dict(mode='ignore-processes'), dict(min_free_mib=4096), dict(max_util_pct=100),
    dict(min_free_mib=True), dict(max_util_pct=True), dict(allowed_compute_pids=[True]),
    dict(allowed_compute_pids=[-1]), dict(allowed_compute_pids=[123, 123]),
    dict(allowed_compute_pids=[456, 123]), dict(allowed_compute_pids=['123']),
    dict(extra='not-frozen'),
])
def test_policy_shape_and_values_are_strict_before_ssh(changes):
    p = dict(POLICY, **changes)
    with patch.object(shared.subprocess, 'run') as run:
        with pytest.raises(ValueError, match='policy'):
            shared.snapshot(PIN, p)
        run.assert_not_called()


def test_empty_allowlist_can_only_observe_no_target_processes():
    policy = dict(POLICY, allowed_compute_pids=[])
    assert observe(policy=policy)['other_compute_processes'] == 0
    with pytest.raises(ValueError, match='unapproved'):
        observe(f'{GPU}, 123, python, 10\n', policy=policy)


@pytest.mark.parametrize('changes', [dict(host='-oProxyCommand=oops'), dict(gpu_index=True),
    dict(gpu_uuid='GPU-bad'), dict(gpu_name=''), dict(extra=1)])
def test_bad_pin_refuses_before_ssh(changes):
    with patch.object(shared.subprocess, 'run') as run:
        with pytest.raises(ValueError, match='pin'):
            shared.snapshot(dict(PIN, **changes), POLICY)
        run.assert_not_called()


def test_policy_validation_never_mutates_the_frozen_allowlist():
    policy = copy.deepcopy(POLICY)
    assert shared.validate_policy(policy) == POLICY
    assert policy == POLICY
