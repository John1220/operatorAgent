"""Offline checks for the supervisor-side S5 verification service."""
import json
import subprocess
from pathlib import Path

import pytest

from benchmarks import s5_verify_remote as service
from benchmarks.s5_vertical_validation import BAD_FIXTURE, GOOD_FIXTURE

DECLARATION = dict(request='layernorm fp32', shapes=[[7, 33]], hardware='mock_gpu',
                   dsl='python', precision='fp32')


def _service_plan(tmp_path, backend='mock', **kwargs):
    root = tmp_path / 'service'
    sha = service.freeze(root, declaration=DECLARATION, backend=backend, **kwargs)
    return service.load_plan(root / 'plan.json', sha), sha, root


def test_plan_binds_the_prepared_contract_and_refuses_a_second_freeze(tmp_path):
    plan, sha, root = _service_plan(tmp_path)
    assert plan['stop_after'] == 'L2' and plan['mode'] == 'mock'
    assert len(plan['contract_hash']) == 64 and plan['local_sha256']
    with pytest.raises(FileExistsError):
        service.freeze(root, declaration=DECLARATION, backend='mock')
    assert (root / 'task' / 'contract.json').is_file()


def test_plan_load_refuses_a_foreign_hash_or_drifted_identity(tmp_path):
    plan, sha, root = _service_plan(tmp_path)
    with pytest.raises(ValueError):
        service.load_plan(root / 'plan.json', 'f' * 64)
    stored = json.loads((root / 'plan.json').read_text())
    stored['local_sha256'] = {'fake': '0' * 64}
    (root / 'plan.json').write_text(json.dumps(stored))
    with pytest.raises(ValueError):
        service.load_plan(root / 'plan.json', sha)


def test_mock_pass_and_fail_carry_levels_and_evidence(tmp_path):
    plan, _, root = _service_plan(tmp_path)
    passed = service.submit(plan, arm='A', candidate_text=GOOD_FIXTURE, output_dir=root / 'A')
    assert passed['status'] == 'passed' and passed['full_l2_passed'] is True
    assert passed['levels'] == {'L1': True, 'L2': True}
    assert passed['mock_only'] is True and passed['remote_backend_executed'] is False
    assert passed['formal_s5_credit'] is False and passed['gpu_lease'] is False
    assert passed['evidence']['evidence_sha256'] and passed['evidence']['receipt_sha256']
    failed = service.submit(plan, arm='B', candidate_text=BAD_FIXTURE, output_dir=root / 'B')
    assert failed['status'] == 'failed' and failed['full_l2_passed'] is False
    assert failed['levels'] == {'L1': True, 'L2': False}


def test_duplicate_submission_is_refused(tmp_path):
    plan, _, root = _service_plan(tmp_path)
    service.submit(plan, arm='A', candidate_text=GOOD_FIXTURE, output_dir=root / 'A')
    with pytest.raises(service.VerificationRefused):
        service.submit(plan, arm='A', candidate_text=GOOD_FIXTURE, output_dir=root / 'A')
    with pytest.raises(service.VerificationRefused):
        service.submit(plan, arm='C', candidate_text='   ', output_dir=root / 'C')


def test_timeout_is_unknown_and_names_the_cleanup_gap(tmp_path, monkeypatch):
    plan, _, root = _service_plan(tmp_path, timeout=5)

    def boom(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd='verify', timeout=5)

    monkeypatch.setattr(service.subprocess, 'run', boom)
    result = service.submit(plan, arm='A', candidate_text=GOOD_FIXTURE, output_dir=root / 'T')
    assert result['status'] == 'unknown' and result['full_l2_passed'] is None
    assert result['timed_out'] is True and 'cleanup unknown' in result['reason']


def test_unparsable_output_is_unknown(tmp_path, monkeypatch):
    plan, _, root = _service_plan(tmp_path)
    monkeypatch.setattr(service, '_command', lambda *args: ['/bin/echo', 'not-json'])
    result = service.submit(plan, arm='B', candidate_text=GOOD_FIXTURE, output_dir=root / 'N')
    assert result['status'] == 'unknown' and result['full_l2_passed'] is None
    assert 'not JSON' in result['reason']


def test_tampered_round_evidence_is_rejected(tmp_path):
    plan, _, root = _service_plan(tmp_path)
    result = service.submit(plan, arm='C', candidate_text=GOOD_FIXTURE, output_dir=root / 'C')
    assert result['status'] == 'passed'
    round_dir = next(path for path in (root / 'C' / 'task').glob('verify-*') if path.is_dir())
    target = round_dir / 'candidate.py'
    target.chmod(0o600)
    target.write_text(BAD_FIXTURE)
    evidence_path = json.loads((root / 'C' / 'result.json').read_text())['payload_sha256']
    assert evidence_path
    evidence, problems = service._evidence(plan, root / 'C' / 'task', round_dir / 'result.json')
    assert evidence is None and any('receipt member' in problem for problem in problems)


def test_gpu_pin_reaches_the_verification_process(tmp_path, monkeypatch):
    plan, _, root = _service_plan(tmp_path, backend='h20', gpu_index=7,
                                  gpu_uuid='GPU-1234abcd-0000', gpu_name='NVIDIA H20')
    seen = {}

    def capture(command, **kwargs):
        seen['env'] = kwargs.get('env', {})
        raise subprocess.TimeoutExpired(cmd='verify', timeout=5)

    monkeypatch.setattr(service.subprocess, 'run', capture)
    result = service.submit(plan, arm='A', candidate_text=GOOD_FIXTURE, output_dir=root / 'G')
    assert result['status'] == 'unknown' and result['timed_out'] is True
    assert seen['env'].get('OPERATOR_AGENT_REQUIRED_GPU') == '7'
    assert seen['env'].get('OPERATOR_AGENT_REQUIRED_GPU_UUID') == 'GPU-1234abcd-0000'
    assert seen['env'].get('OPERATOR_AGENT_REQUIRED_GPU_NAME') == 'NVIDIA H20'
    assert seen['env'].get('PYTHONNOUSERSITE') == '1'


def test_a_gpu_pin_requires_the_ssh_backend_and_full_identity(tmp_path):
    with pytest.raises(ValueError):
        service.freeze(tmp_path / 'bad', declaration=DECLARATION, backend='mock', gpu_index=7)
    with pytest.raises(ValueError):
        service.freeze(tmp_path / 'index-only', declaration=DECLARATION,
                       backend='h20', gpu_index=7)


def test_ssh_backend_plan_never_dispatches_during_freeze(tmp_path):
    plan, _, root = _service_plan(tmp_path, backend='h20')
    assert plan['mode'] == 'ssh' and plan['host'] == 'algorithm-h20'
    command = service._command(plan, root / 'task', root / 'task' / 'candidate.py')
    assert '--backend' in command and command[command.index('--backend') + 1] == 'ssh'
    assert command[command.index('--host') + 1] == 'algorithm-h20'
    assert '--stop-after' in command and command[command.index('--stop-after') + 1] == 'L2'


def test_decisive_l1_syntax_rejection_is_not_infrastructure_unknown(tmp_path):
    plan,_,root=_service_plan(tmp_path)
    result=service.submit(plan,arm='A',candidate_text='def kernel( broken',output_dir=root/'syntax')
    assert result['status']=='failed',result
    assert result['levels']['L1']is False
