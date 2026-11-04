"""Missing evidence stays unknown; a summary cannot manufacture acceptance."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.experiments.phase1 import baseline, finalize, plan as plans, summary
from operator_agent.execute import s5_shared_budget as shared
from operator_agent.execute.budget import BudgetLedger, BudgetPlan
from operator_agent.execute.task_clock import monotonic

CODE = 'import numpy as np\ndef kernel(A, B):\n    return np.matmul(A, B)\n'


@pytest.fixture(autouse=True)
def stable_implementation(monkeypatch):
    identity = dict(protocol={'fixture.py': 'a'*64}, harness={'summary.py': 'b'*64})
    monkeypatch.setattr(plans, 'implementation_identity', lambda: deepcopy(identity))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    replacement = path.with_name(path.name+'.fixture-replacement')
    replacement.write_text(json.dumps(value))
    replacement.replace(path)


def execution(tmp_path):
    root = tmp_path/'execution'
    plan = plans.freeze(root)
    write(root/'execution.json', dict(schema='phase1-execution-v1', plan_hash=plan['hash'],
                                    phase='done', invocation_id='a'*32))
    return root, plan


def cell_fixture(root, plan):
    slot = plan['schedule'][0]
    task = plan['tasks'][0]
    cell = root/'cells'/slot['cell_id']
    work = cell/'work'
    work.mkdir(parents=True)
    config = dict(schema='phase1-cell-config-v1', **slot, plan_hash=plan['hash'],
        budget=plan['budget'],
        shared_task_id='p1-'+plans.digest([plan['hash'], slot['cell_id']])[:24], run_dir=str(work),
        declaration=task['declaration'], backend='fake', fake=plan['fake'], connection={},
        shared_ledger=str(cell/'budget.sqlite'), started_monotonic=monotonic(),
        max_rounds=plan['max_rounds'])
    config['deadline_monotonic'] = config['started_monotonic']+plan['budget']['wall_clock_s']
    record = dict(schema='phase1-cell-record-v1', **slot, plan_hash=plan['hash'],
        phase='done', reason='fixture_complete', worker_result=str(work/'baseline-result.json'),
        development=dict(returncode=0, stopped_reason=None, elapsed_s=.01),
        final_evaluation=None, selected_source_hash=None)
    write(cell/'cell-config.json', config)
    write(work/'baseline-result.json', dict(phase='done'))
    write(cell/'record.json', record)
    return cell, task, config, record


def fake_budgets(monkeypatch):
    monkeypatch.setattr(summary, '_model_budget', lambda cell, config, problems:
        dict(complete=config is not None, reserved_calls=1))
    monkeypatch.setattr(summary, '_oracle_budget', lambda cell, plan, problems:
        dict(complete=(cell/'cell-config.json').is_file()))


def first_row(root):
    return summary.summarize(root)['cells'][0]


def test_missing_records_still_retain_all_eight_cells_and_unknown_human_effort(tmp_path):
    root, _ = execution(tmp_path)
    result = summary.summarize(root)
    assert result['planned_cells'] == result['retained_cells'] == 8
    assert len({row['cell_id'] for row in result['cells']}) == 8
    assert all(row['phase'] == 'unknown' and row['problems'] for row in result['cells'])
    assert all(row['events']['count'] is None for row in result['cells'])
    assert all(row['budget_auditable'] is False for row in result['cells'])
    assert all(arm['planned'] == 4 and arm['correct_candidates'] == 0 for arm in result['arms'].values())
    assert result['human_minutes'] is None
    assert result['formal_comparison_credit'] is False
    assert result['gpu_acceptance'] is False


@pytest.mark.parametrize('tamper', ['changed_bytes', 'missing_file', 'outside_cell'])
def test_event_sources_require_the_actual_retained_bytes(tmp_path, monkeypatch, tamper):
    fake_budgets(monkeypatch)
    root, plan = execution(tmp_path)
    cell, _, _, _ = cell_fixture(root, plan)
    path = cell/'work'/'capture.json'
    path.write_text('captured user takeover')
    event = dict(id='takeover-1', timestamp=1., stage='modify', actor='human', kind='takeover',
        source=dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), record='1'))
    write(cell/'work'/'events.json', [event])
    if tamper == 'changed_bytes':
        path.write_text('replaced capture')
    elif tamper == 'missing_file':
        path.unlink()
    else:
        outside = root/'outside.json'
        outside.write_text('captured user takeover')
        event['source']['path'] = str(outside)
        write(cell/'work'/'events.json', [event])
    row = first_row(root)
    assert row['events']['count'] is None
    assert row['events']['observed_count'] == 0
    assert row['events']['tools_complete'] is False
    assert any('unbound event source' in problem for problem in row['problems'])


def test_failures_retries_and_missing_human_sources_are_all_retained(tmp_path, monkeypatch):
    fake_budgets(monkeypatch)
    root, plan = execution(tmp_path)
    cell, _, _, _ = cell_fixture(root, plan)
    source = cell/'work'/'capture.json'
    source.write_text('original failed verification and retry capture')
    ref = dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(), record='fixture')
    events = [dict(id='failure', timestamp=1., stage='check', actor='agent', kind='tool_result',
        action='candidate_verify', tool='bash', ok=False, source=ref),
        dict(id='retry', timestamp=2., stage='modify', actor='agent', kind='retry',
             reason='verification_failed', source=ref)]
    write(cell/'work'/'events.json', events)
    row = first_row(root)
    assert row['events']['raw_events'] == events
    assert row['events']['observed_failure_count'] == row['events']['failure_count'] == 1
    assert row['events']['observed_retry_count'] == row['events']['retry_count'] == 1
    assert row['events']['observed_count'] == 0
    assert row['events']['count'] is None  # Unobserved human sources are not zero.
    assert row['events']['bypass_count'] == 0  # Arm A may use its shell.


@pytest.mark.parametrize('tamper', [None, 'missing_local', 'missing_shared',
    'changed_local', 'orphan_shared', 'wider_shared_limit', 'wider_local_limit'])
def test_model_budget_is_auditable_only_with_matching_local_and_shared_evidence(
        tmp_path, monkeypatch, tamper):
    root, plan = execution(tmp_path)
    cell, _, config, _ = cell_fixture(root, plan)
    shared.create(config['shared_ledger'], task_id=config['shared_task_id'],
        max_calls=plan['budget']['max_llm_calls'], max_output_tokens=40960,
        total_output_tokens=40960*plan['budget']['max_llm_calls'],
        wall_clock_s=plan['budget']['wall_clock_s'])
    # cell_fixture's placeholder is not the worker's write-once start receipt.
    Path(config['run_dir'], 'baseline-result.json').unlink()
    result = baseline.run(config)
    assert result['phase'] == 'done'
    journal = cell/'work'/'model-journal'
    local_call = next(journal.glob('call-*.json'))
    if tamper == 'missing_local':
        local_call.unlink()
    elif tamper == 'missing_shared':
        Path(config['shared_ledger']).unlink()
    elif tamper == 'changed_local':
        value = plans.read(local_call)
        value['request_sha256'] = 'f'*64
        write(local_call, value)
    elif tamper == 'orphan_shared':
        shared.reserve(config['shared_ledger'], participant='outer', caller='Coder',
            max_output_tokens=4096, request_sha256='f'*64, expected_task_id=config['shared_task_id'])
    elif tamper == 'wider_shared_limit':
        original = shared.snapshot
        def larger(path, **kwargs):
            value = original(path, **kwargs)
            value['max_calls'] += 1
            return value
        monkeypatch.setattr(shared, 'snapshot', larger)
    elif tamper == 'wider_local_limit':
        limits = plans.read(journal/'limits.json')
        limits['max_calls'] += 1
        local = plans.read(journal/'summary.json')
        local['limits'] = limits
        write(journal/'limits.json', limits)
        write(journal/'summary.json', local)
    row = first_row(root)
    assert row['budget_auditable'] is (tamper is None)
    assert row['model_budget']['complete'] is (tamper is None)
    if tamper is None:
        assert row['model_budget']['fixture_completed_calls'] == 1
        assert row['model_budget']['provider_dispatches'] == 0
        assert row['model_budget']['api_usage'] is None
        assert row['model_budget']['shared']['completed_calls'] == 0


@pytest.mark.parametrize('tamper', [None, 'source', 'contract', 'verified_contract',
    'selected_record', 'selected_bytes', 'l3_verdict', 'l3_gate', 'l3_levels'])
def test_independent_final_credit_requires_matching_source_contract_and_fake_qualification(
        tmp_path, monkeypatch, tamper):
    fake_budgets(monkeypatch)
    root, plan = execution(tmp_path)
    cell, task, _, record = cell_fixture(root, plan)
    selected = cell/'selected.py'
    selected.write_text(CODE)
    digest = hashlib.sha256(selected.read_bytes()).hexdigest()
    value = finalize.run(dict(cell_id=record['cell_id'], plan_hash=plan['hash'],
        run_dir=str(cell/'final'), source=str(selected), source_hash=digest,
        declaration=task['declaration'], contract_hash=task['contract_hash'], profile='small'))
    record.update(selected_source_hash=digest,
        final_evaluation=dict(returncode=0, stopped_reason=None, elapsed_s=.01))
    if tamper == 'source':
        value['source_hash'] = 'f'*64
    elif tamper == 'contract':
        value['contract_hash'] = 'f'*64
    elif tamper == 'verified_contract':
        value['result']['contract_hash'] = 'f'*64
    elif tamper == 'selected_record':
        record['selected_source_hash'] = 'f'*64
    elif tamper == 'selected_bytes':
        selected.write_text(CODE+'# replaced after final evaluation\n')
    elif tamper in ('l3_verdict', 'l3_gate'):
        verdict = next(v for v in value['result']['verdicts'] if v['level'] == 'L3')
        if tamper == 'l3_verdict':
            verdict['passed'] = True
        else:
            verdict['metrics']['measurement_can_gate'] = True
    elif tamper == 'l3_levels':
        value['result']['levels']['L3'] = True
    write(cell/'record.json', record)
    write(cell/'final'/'final-result.json', value)
    result = summary.summarize(root)
    row = result['cells'][0]
    assert row['final_evaluation']['complete'] is (tamper is None)
    assert row['final_evaluation']['full_l2_passed'] is (True if tamper is None else None)
    assert result['arms']['A']['correct_candidates'] == (1 if tamper is None else 0)
    assert row['final_evaluation']['gpu_acceptance'] is False


def test_missing_oracle_ledger_cannot_be_hidden_by_a_complete_model_audit(tmp_path, monkeypatch):
    root, plan = execution(tmp_path)
    cell_fixture(root, plan)
    monkeypatch.setattr(summary, '_model_budget', lambda *args:
        dict(complete=True, reserved_calls=1))
    row = first_row(root)
    assert row['verification_budget']['complete'] is False
    assert row['budget_auditable'] is False


def test_oracle_metadata_cannot_remove_the_reserved_confirmation_pool(tmp_path, monkeypatch):
    root, plan = execution(tmp_path)
    cell, _, _, _ = cell_fixture(root, plan)
    monkeypatch.setattr(summary, '_model_budget', lambda *args:
        dict(complete=True, reserved_calls=0))
    budget = BudgetLedger(BudgetPlan(oracle_total=plan['budget']['oracle_total'])).as_dict()
    budget['reservation_policy'] = 'pre_dispatch_non_refundable'
    write(cell/'work'/'verification-budget.json', budget)
    assert first_row(root)['budget_auditable'] is True
    budget['plan']['phase_caps']['confirm'] = 0
    write(cell/'work'/'verification-budget.json', budget)
    row = first_row(root)
    assert row['verification_budget']['complete'] is False
    assert row['budget_auditable'] is False


@pytest.mark.parametrize('tamper', [None, 'missing_timing', 'invalid_timing',
    'l1_failure', 'l2_failure', 'unknown_suite'])
def test_first_full_l2_needs_consistent_stage_facts_and_observed_completion_time(
        tmp_path, monkeypatch, tamper):
    fake_budgets(monkeypatch)
    root, plan = execution(tmp_path)
    cell, task, _, _ = cell_fixture(root, plan)
    candidate = cell/'work'/'candidate.py'
    candidate.write_text(CODE)
    event = dict(schema='operator-verification-completion-v1', event_id='a'*32,
        contract_hash=task['contract_hash'], source_hash=hashlib.sha256(candidate.read_bytes()).hexdigest(),
        level='L2', passed=True, full_l2_passed=True,
        timing=dict(started_at=1., completed_at=2., started_monotonic=100., completed_monotonic=101.),
        verdicts=[dict(level='L1', passed=True, metrics={}),
                  dict(level='L2', passed=True,
                       metrics=dict(correctness={'verdict': 'passed'}, input_fidelity_verified=True))])
    if tamper == 'missing_timing':
        event.pop('timing')
    elif tamper == 'invalid_timing':
        event['timing']['completed_at'] = 'not-a-timestamp'
    elif tamper == 'l1_failure':
        event['verdicts'][0]['passed'] = False
    elif tamper == 'l2_failure':
        event['verdicts'][1]['passed'] = False
    elif tamper == 'unknown_suite':
        event['verdicts'][1]['metrics']['correctness']['verdict'] = 'unknown'
    write(cell/'work'/'phase-events'/'verification-event-fixture.json', event)
    row = first_row(root)
    if tamper is None:
        assert row['first_full_l2']['source_hash'] == event['source_hash']
        assert row['first_full_l2']['completed_at'] == 2.
    else:
        assert row['first_full_l2'] is None
