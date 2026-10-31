"""Offline study tables retain fixed denominator and distinguish success-pair cost."""
import copy
import json

import pytest

from benchmarks.experiments.s5 import s5_minimal_tasks, s5_plugin_summary
from benchmarks.experiments.s5 import s5_plugin_study_summary as summary
from operator_agent.core.run_contract import digest


def fixture():
    manifest = s5_minimal_tasks.manifest()
    plan = dict(schema=summary.PLAN_SCHEMA, mode='study', hash='study', arms=['A', 'C'],
        manifest_hash=manifest['hash'], cases=manifest['tasks'], repeats=2, planned_runs=24,
        schedule=s5_plugin_summary.schedule(manifest['tasks'], 2))
    records = [dict(slot, plan_hash='study', status='passed', execution_state='finished',
        outcome_class='completed', model_calls=dict(reserved=4 if slot['arm'] == 'A' else 3, completed=3),
        wall_seconds=10 if slot['arm'] == 'A' else 8,
        development_audit=dict(plugin_verification_observed=False)) for slot in plan['schedule']]
    return plan, records


def test_study_tables_report_paired_success_costs_without_supposing_plugin_use():
    plan, records = fixture()
    result = summary.aggregate(plan, records, bootstrap_samples=100)
    assert result['groups']['A']['planned'] == result['groups']['C']['planned'] == 12
    assert result['pair_outcomes'] == dict(win=0, tie=12, loss=0, unknown=0)
    assert all(p['success_pair_cost_difference'] == dict(calls=-1, wall_seconds=-2)
               for p in result['attempt_pairs'])
    assert result['groups']['C']['mcp_used'] == 0
    assert result['l3_performance']['qualified_successes'] is None
    assert result['formal_s5_credit'] is False


def test_study_costs_do_not_compare_fast_failures_as_success_cost_reductions():
    plan, records = fixture()
    records[1].update(status='failed', outcome_class='candidate_rejected', wall_seconds=1)
    result = summary.aggregate(plan, records, bootstrap_samples=100)
    first = result['attempt_pairs'][0]
    assert first['outcome'] == 'loss' and first['both_passed'] is False
    assert first['success_pair_cost_difference'] == dict(calls=None, wall_seconds=None)
    assert result['groups']['C']['costs']['wall_seconds']['observed_sum'] == 89


def test_study_missing_unknown_and_budget_noncompletion_remain_distinct():
    plan, records = fixture()
    records[1].update(status='unknown', execution_state='interrupted', outcome_class='call_budget_exhausted')
    records[3].update(status='unknown', execution_state='interrupted', outcome_class='infrastructure_error')
    result = summary.aggregate(plan, records[:-1], bootstrap_samples=100)
    assert result['planned'] == 24 and result['received'] == 23
    assert result['pair_outcomes'] == dict(win=0, tie=9, loss=1, unknown=2)
    assert result['comparisons']['C-minus-A']['inference_allowed'] is False
    assert result['groups']['C']['costs']['calls']['unknown_rows'] == 1


@pytest.mark.parametrize('change', ['task', 'repeats', 'schedule'])
def test_study_summary_rejects_other_grids(change):
    plan, records = fixture()
    plan = copy.deepcopy(plan)
    if change == 'task': plan['cases'][0]['case_id'] = 'historical'
    if change == 'repeats': plan['repeats'] = 1
    if change == 'schedule': plan['schedule'] = plan['schedule'][:-1]
    with pytest.raises(ValueError, match='24-slot'):
        summary.aggregate(plan, records)


def test_study_cli_recomputes_from_frozen_grid_without_runtime_gate(tmp_path):
    plan, records = fixture()
    plan.pop('hash')
    plan['hash'] = digest(plan)
    for record in records: record['plan_hash'] = plan['hash']
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text(json.dumps(plan))
    record_dir = tmp_path / 'execution/records'
    record_dir.mkdir(parents=True)
    for i, record in enumerate(records):
        (record_dir / ('%02d.json' % i)).write_text(json.dumps(record))
    output = tmp_path / 'recomputed.json'
    summary.main(['--plan', str(plan_path), '--execution', str(record_dir.parent), '--output', str(output)])
    result = json.loads(output.read_text())
    assert result['hash'] == summary.aggregate(plan, records)['hash']
    with pytest.raises(FileExistsError):
        summary.main(['--plan', str(plan_path), '--execution', str(record_dir.parent), '--output', str(output)])
