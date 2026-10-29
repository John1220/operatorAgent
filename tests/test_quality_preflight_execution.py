"""Readiness is an ordering gate, not a model/GPU correctness credential."""
import json
from unittest.mock import Mock

import pytest

from operator_agent import quality_evaluation as q, quality_preflight as pf
from operator_agent.remote_executor import SSHExecutor
from operator_agent.run_contract import code_hash
from operator_agent.schemas import VerificationLevel as Level, VerificationResult


CASE = 'layernorm-tail'


def real_plan(monkeypatch, cases=None):
    monkeypatch.setenv('LLM_MODEL', 'synthetic-model')
    monkeypatch.setenv('LLM_BASE_URL', 'https://model.invalid/v1')
    return q.make_plan(model='synthetic-model', endpoint='https://model.invalid/v1',
                       backend='ssh', host='synthetic-host', case_ids=cases or [CASE])


def test_failed_preflight_prevents_client_creation_and_preserves_task_costs(tmp_path, monkeypatch):
    plan = real_plan(monkeypatch)
    monkeypatch.setattr(pf, 'preflight_backend', lambda _: {'status':'unknown', 'identity':{}})
    client = Mock(side_effect=AssertionError('must not construct model'))
    report = q.run_plan(plan, tmp_path/'run', host='synthetic-host', client_factory=client,
                        backend_factory=lambda _: object())
    client.assert_not_called()
    assert report['attempts'][0]['status'] == 'preflight_failed'
    assert report['costs']['model_calls']['known_total'] == 0
    assert report['metrics']['first_full_l2']['denominator'] == 1
    assert not (tmp_path/'run'/CASE/'model-journal').exists()
    assert json.loads((tmp_path/'run'/CASE/'preflight.json').read_text())['status'] == 'unknown'


def test_order_is_backend_preflight_generation_candidate_verification_then_cleanup(tmp_path, monkeypatch):
    plan, order = real_plan(monkeypatch), []
    backend = Mock()
    backend.close.side_effect = lambda: order.append('cleanup') or {'quiescent':True}
    def create_backend(_):
        order.append('backend')
        return backend
    def preflight(_):
        order.append('preflight')
        return {'status':'passed', 'identity':{'device':'synthetic'}}
    def create_client(key, ledger):
        order.append('client')
        return q.ReplayClient([json.dumps(dict(code='def kernel(X): return X', rationale='fake'))], ledger)
    def verify(*args, **kwargs):
        order.append('verify')
        return [VerificationResult(Level.L1_COMPILE, True), VerificationResult(
            Level.L2_FUNCTION, True, metrics={'correctness':{'verdict':'passed'}})]
    monkeypatch.setattr(pf, 'preflight_backend', preflight)
    monkeypatch.setattr(q, 'precheck', lambda *args: None)
    monkeypatch.setattr(q, 'Verifier', lambda _: Mock(verify=verify))
    report = q.run_plan(plan, tmp_path/'run', host='synthetic-host',
                        client_factory=create_client, backend_factory=create_backend)
    assert order == ['backend', 'preflight', 'client', 'verify', 'cleanup']
    assert report['costs']['model_calls']['known_total'] == 1
    assert report['metrics']['native_performance_target']['rate'] is None


def test_standalone_preflight_needs_no_model_configuration_or_input_materialization(tmp_path, monkeypatch):
    plan = real_plan(monkeypatch, [CASE, 'matmul-strided-fp32'])
    monkeypatch.delenv('LLM_MODEL')
    def forbidden(*args, **kwargs):
        pytest.fail('preflight must not construct models or task tensors')
    monkeypatch.setattr(q, 'LLMClient', forbidden)
    monkeypatch.setattr(q.quality_suite, 'evaluation_inputs_for', forbidden)
    results = iter([{'status':'passed', 'identity':{}}, {'status':'failed', 'identity':{}}])
    monkeypatch.setattr(pf, 'preflight_backend', lambda _: next(results))
    report = q.run_preflight(plan, tmp_path/'preflight', host='synthetic-host',
                            backend_factory=lambda _: object())
    assert report['model_calls'] == 0 and report['passed'] is False
    assert len(report['cases']) == 2
    assert report['cases'][0]['status'] == 'passed'


def test_verified_ssh_runner_reuses_existing_python_without_uv_or_deployment(monkeypatch):
    backend = SSHExecutor(host='synthetic-host', work_dir='~/dir with spaces',
                          remote_runner_path='~/dir with spaces/run_kernel.py')
    backend._gpu_pin, backend._deployed = 2, True
    backend._verified_python = '.venv/bin/python'
    observed = []
    backend._ssh_exec = lambda command, **kwargs: observed.append(command) or '{"success":true}'
    backend.deploy = Mock(side_effect=AssertionError('must not provision'))
    assert backend._ssh_exec_json({'request':'synthetic'})['success'] is True
    command = observed[0]
    assert 'cd "$HOME"/\'dir with spaces\'' in command
    assert "OPERATOR_AGENT_GPU=2 .venv/bin/python" in command
    assert 'CUDA_VISIBLE_DEVICES=2 OPERATOR_AGENT_GPU=2' in command
    assert 'uv run' not in command and 'mkdir' not in command
    backend.deploy.assert_not_called()


def test_unrecognized_verified_python_cannot_become_a_remote_shell_fragment():
    backend = SSHExecutor(host='synthetic-host')
    backend._gpu_pin, backend._verified_python = 0, 'python; unexpected'
    backend._ssh_exec = Mock(side_effect=AssertionError('must not dispatch'))
    with pytest.raises(ValueError, match='unrecognized'):
        backend._ssh_exec_json({'request':'synthetic'})
    backend._ssh_exec.assert_not_called()


def test_verified_environment_cannot_continue_after_losing_the_observed_gpu_pin():
    backend = SSHExecutor(host='synthetic-host')
    backend._verified_python = '.venv/bin/python'
    backend._pinned_gpu = Mock(return_value=None)
    backend._ssh_exec = Mock(side_effect=AssertionError('must not dispatch'))
    with pytest.raises(ValueError, match='GPU pin'):
        backend._ssh_exec_json({'request':'synthetic'})
    backend._ssh_exec.assert_not_called()
