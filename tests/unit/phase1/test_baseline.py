"""The rehearsal baseline owns generation and retains ordinary verifier facts."""
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from benchmarks.experiments.phase1 import baseline
from operator_agent.app import hitl_tools as ht
from operator_agent.execute import s5_shared_budget as shared
from operator_agent.execute.task_clock import monotonic


def config(tmp_path, *, scenario='success', max_calls=4, oracle_total=20, max_rounds=2):
    ledger = tmp_path/'budget.sqlite'
    task_id = 'phase1-baseline-test'
    shared.create(ledger, task_id=task_id, max_calls=max_calls, max_output_tokens=40960,
                  total_output_tokens=max_calls*40960, wall_clock_s=30)
    started = monotonic()
    return dict(schema='phase1-cell-config-v1', cell_id='task-01-A', arm='A',
        plan_hash='a'*64, run_dir=str(tmp_path/'run'), declaration=dict(
            request='relu', shapes=[[4, 8]], dsl='python'), backend='fake',
        fake=dict(scenario=scenario, profile='small', llm='fake'), connection={},
        budget=dict(wall_clock_s=30, max_llm_calls=max_calls, oracle_total=oracle_total),
        shared_ledger=str(ledger), shared_task_id=task_id, started_monotonic=started,
        deadline_monotonic=started+30, max_rounds=max_rounds)


def read(path):
    return json.loads(Path(path).read_text())


def test_baseline_is_independent_and_offline_with_full_evidence(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    from operator_agent.execute import llm_client, llm_transport, mock_gpu
    forbidden = Mock(side_effect=AssertionError('baseline used GPU, credentials, transport or delegation'))
    monkeypatch.setattr(ht, 'run_autonomous', forbidden)
    monkeypatch.setattr(ht, 'repair', forbidden)
    monkeypatch.setattr(llm_client, '_resolve_api_key', forbidden)
    monkeypatch.setattr(llm_transport, 'dispatch', forbidden)
    monkeypatch.setattr(mock_gpu, '_has_cuda', forbidden)
    monkeypatch.setenv('LLM_API_KEY_FILE', '/no/offline/credentials')
    original = ht._backend
    result = baseline.run(cfg)
    assert ht._backend is original
    forbidden.assert_not_called()
    assert result['phase'] == 'done' and result['reason'] == 'offline_correctness_complete'
    assert result['levels'] == {'L1': True, 'L2': True, 'L3': False}
    assert result['all_passed'] is False
    assert result['execution_context']['test_only'] is True
    assert result['budget']['model_calls']['reserved'] == 1
    assert result['budget']['verification']['spent']['confirm'] == 0
    run = Path(cfg['run_dir'])
    assert result == read(run/'baseline-result.json')
    assert result['source_hash'] == hashlib.sha256(Path(result['selected_source']).read_bytes()).hexdigest()
    assert read(run/'verify-1.json')['candidate_origin'] == 'outer'
    retained = read(run/'verify-1'/'result.json')
    assert retained['evidence_complete'] is True
    assert retained['implementation_stable'] is True
    for event in read(run/'events.json'):
        source = event['source']
        assert source['sha256'] == hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()
        assert event['actor'] == 'agent'
    shared_audit = shared.snapshot(cfg['shared_ledger'], expected_task_id=cfg['shared_task_id'])
    assert shared_audit['reserved_calls'] == 1
    assert shared_audit['calls'][0]['dispatched_monotonic'] is None
    local = read(next((run/'model-journal').glob('call-*.json')))
    assert local['usage_source'] == 'test_fixture' and local['api_usage'] is None


@pytest.mark.parametrize('scenario,failed_level', [
    ('compile-repair', 'L1'), ('correctness-repair', 'L2')])
def test_fault_requires_new_source_and_a_real_reverification(tmp_path, scenario, failed_level):
    cfg = config(tmp_path, scenario=scenario)
    result = baseline.run(cfg)
    run = Path(cfg['run_dir'])
    first, second = read(run/'verify-1.json'), read(run/'verify-2.json')
    assert first['levels'][failed_level] is False
    assert second['levels'] == {'L1': True, 'L2': True, 'L3': False}
    assert first['source_hash'] != second['source_hash']
    assert result['verification_count'] == 2
    assert result['budget']['model_calls']['reserved'] == 2
    assert any(event['kind'] == 'retry' for event in read(run/'events.json'))


def test_shared_model_budget_stops_repair_without_poisoning_ledger(tmp_path):
    cfg = config(tmp_path, scenario='correctness-repair', max_calls=1)
    result = baseline.run(cfg)
    assert result['phase'] == 'done' and result['reason'] == 'budget_llm_calls'
    assert result['verification_count'] == 1
    audit = shared.snapshot(cfg['shared_ledger'], expected_task_id=cfg['shared_task_id'])
    assert audit['reserved_calls'] == 1 and audit['aborted'] is False


def test_confirm_reserve_is_not_spent_on_development_or_model_calls(tmp_path):
    cfg = config(tmp_path, oracle_total=1)
    result = baseline.run(cfg)
    assert result['phase'] == 'done' and result['reason'] == 'verification_budget_exhausted'
    assert result['verification_count'] == 0 and result['selected_source'] is None
    audit = shared.snapshot(cfg['shared_ledger'], expected_task_id=cfg['shared_task_id'])
    assert audit['reserved_calls'] == 0
    verification = result['budget']['verification']
    assert verification['spent_all'] == 0
    assert verification['remaining'] == {'non_confirm': 0, 'confirm': 1}


def test_wrong_generated_candidate_is_not_replaced_with_reference(tmp_path, monkeypatch):
    cfg = config(tmp_path, max_rounds=1)
    original = baseline.FakeLLM
    class WrongFixture(original):
        def chat(self, *args, **kwargs):
            super().chat(*args, **kwargs)
            return 'import numpy as np\ndef kernel(X):\n    return np.minimum(X, 0)\n'
    monkeypatch.setattr(baseline, 'FakeLLM', WrongFixture)
    result = baseline.run(cfg)
    assert result['levels'] == {'L1': True, 'L2': False}
    assert result['reason'] == 'round_limit' and result['all_passed'] is False
    assert read(Path(cfg['run_dir'])/'verify-1.json')['levels']['L2'] is False


def test_backend_scope_restores_even_when_verification_raises(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    original = ht._backend
    monkeypatch.setattr(ht, 'verify', Mock(side_effect=RuntimeError('fixture interruption')))
    result = baseline.run(cfg)
    assert ht._backend is original
    assert result['phase'] == 'failed' and result['error']['type'] == 'RuntimeError'
    assert Path(cfg['run_dir'], 'baseline-result.json').is_file()
    assert read(Path(cfg['run_dir'])/'verification-budget.json')['spent_all'] == 1


@pytest.mark.parametrize('override', [dict(backend='ssh'), dict(fake={'llm': 'live'})])
def test_rehearsal_rejects_live_configuration_with_a_terminal_record(tmp_path, override):
    cfg = dict(config(tmp_path), **override)
    result = baseline.run(cfg)
    assert result['phase'] == 'failed'
    assert result['error']['type'] == 'ValueError'
    assert result['verification_count'] == 0
    assert not Path(cfg['run_dir'], 'model-journal').exists()


def test_run_directory_cannot_be_reused(tmp_path):
    cfg = config(tmp_path)
    baseline.run(cfg)
    original = Path(cfg['run_dir'], 'baseline-result.json').read_bytes()
    with pytest.raises(FileExistsError):
        baseline.run(cfg)
    assert Path(cfg['run_dir'], 'baseline-result.json').read_bytes() == original


def test_inherited_verification_delegation_is_refused_before_dispatch(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    from operator_agent.execute import s5_verify_client
    forbidden = Mock(side_effect=AssertionError('offline candidate reached a verification service'))
    monkeypatch.setattr(s5_verify_client, 'delegated_verify', forbidden)
    monkeypatch.setenv('OPERATOR_AGENT_S5_DEV_URL', 'http://127.0.0.1:1/development/verify')
    result = baseline.run(cfg)
    assert result['phase'] == 'failed' and result['error']['type'] == 'ValueError'
    assert 'inherited S5' in result['error']['detail']
    forbidden.assert_not_called()
