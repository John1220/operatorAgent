"""A/C-only reports retain failed and unknown attempts without fake precision."""
import copy

import pytest

from benchmarks import s5_plugin_summary as stats


def grid():
    plan = dict(hash='frozen-plan', arms=['A', 'C'],
                cases=[dict(case_id='one'), dict(case_id='two')], repeats=2, planned_runs=8)
    rows = [dict(slot, plan_hash=plan['hash'], status='passed' if slot['arm'] == 'C' else 'failed',
                 execution_state='finished', model_calls=dict(reserved=2), wall_seconds=3,
                 development_audit=dict(plugin_verification_observed=False))
            for slot in stats.schedule(plan['cases'], plan['repeats'])]
    return plan, rows


def test_schedule_has_complete_grid_and_alternates_first_arm():
    cases = [dict(case_id='one'), dict(case_id='two'), dict(case_id='three')]
    rows = stats.schedule(cases, 2)
    assert len(rows) == len({(r['case_id'], r['repeat_id'], r['arm']) for r in rows}) == 12
    assert [r['arm'] for r in rows] == list('ACCAACCAACCA')
    assert all(r['arm'] in ('A', 'C') for r in rows)


@pytest.mark.parametrize('cases,repeats', [([], 1), ([dict(case_id='one')]*2, 1),
    ([dict(case_id='')], 1), ([dict(case_id='one')], True), ([dict(case_id='one')], 0)])
def test_bad_schedule_identity_refuses(cases, repeats):
    with pytest.raises(ValueError):
        stats.schedule(cases, repeats)


def test_complete_paired_summary_and_deterministic_bootstrap():
    plan, rows = grid()
    for row in rows:
        row['workflow_progress'] = dict(model_steps_observed=8, tool_calls=5,
            exploration_tool_calls=1, first_candidate_write_step=4,
            first_verification_step=6, development_verifications=1,
            development_feedback_observed=1)
    result = stats.aggregate(plan, rows, bootstrap_samples=100)
    pair = result['comparisons']['C-minus-A']
    assert pair['success_rate_difference'] == 1
    assert pair['paired_task_differences'] == [1, 1]
    assert pair['task_cluster_percentile_95'] == [1, 1] and pair['inference_allowed']
    assert result['primary_comparison'] == 'C-minus-A'
    assert set(result['groups']) == {'A', 'C'}
    assert result['planned'] == result['received'] == result['dispatched_runs'] == 8
    assert result['counts'] == dict(passed=4, failed=4, unknown=0)
    assert result['groups']['A']['costs']['calls']['observed_sum'] == 8
    assert result['groups']['C']['mcp_used'] == 0
    assert result['groups']['A']['workflow_progress']['values'][0]['first_verification_step'] == 6
    assert result['groups']['C']['workflow_progress']['values'][0]['exploration_tool_calls'] == 1
    assert not result['formal_s5_credit']
    assert stats.aggregate(plan, rows, bootstrap_samples=100) == result


def test_repeats_are_averaged_inside_task_before_bootstrap():
    plan, rows = grid()
    for row in rows:
        passed = ((row['case_id'] == 'one' and row['arm'] == 'C' and row['repeat_id'] == 1)
                  or (row['case_id'] == 'two' and (row['arm'] == 'A' or row['repeat_id'] == 2)))
        row['status'] = 'passed' if passed else 'failed'
    pair = stats.aggregate(plan, rows, bootstrap_samples=100)['comparisons']['C-minus-A']
    assert pair['paired_task_differences'] == [.5, -.5]
    assert pair['success_rate_difference'] == 0
    assert pair['task_cluster_percentile_95'] == [-.5, .5]


def test_missing_rows_preserve_denominator_and_suppress_inference():
    plan, rows = grid()
    missing_arm = rows[-1]['arm']
    result = stats.aggregate(plan, rows[:-1], bootstrap_samples=100)
    assert result['planned'] == 8 and result['received'] == result['dispatched_runs'] == 7
    assert result['groups'][missing_arm]['not_executed'] == 1
    assert result['groups'][missing_arm]['costs']['calls']['unknown_rows'] == 1
    pair = result['comparisons']['C-minus-A']
    assert not pair['inference_allowed']
    assert pair['success_rate_difference'] is pair['task_cluster_percentile_95'] is None
    assert pair['cost_per_attempt_difference']['calls'] is None


def test_empty_records_never_report_unknown_cost_or_usage_as_zero():
    plan, _ = grid()
    result = stats.aggregate(plan, [], bootstrap_samples=100)
    assert result['planned'] == 8 and result['received'] == result['dispatched_runs'] == 0
    for group in result['groups'].values():
        assert group['not_executed'] == group['unknown'] == 4
        assert group['costs']['calls']['observed_sum'] is None
        assert group['dsh_usage']['outputTokens']['observed_sum'] is None
        assert group['costs']['calls']['unknown_rows'] == 4


def test_explicit_zero_cost_and_usage_are_distinct_from_absent_fields():
    plan, rows = grid()
    for row in rows:
        row.update(model_calls=dict(reserved=0), wall_seconds=0, dsh_usage=dict(outputTokens=0))
    result = stats.aggregate(plan, rows, bootstrap_samples=100)
    for group in result['groups'].values():
        assert group['costs']['calls']['observed_sum'] == 0
        assert group['costs']['calls']['complete']
        assert group['dsh_usage']['outputTokens']['observed_sum'] == 0
        assert group['dsh_usage']['outputTokens']['complete']
        assert group['dsh_usage']['inputTokens']['observed_sum'] is None


@pytest.mark.parametrize('outcome', sorted(stats.ORDINARY_NONCOMPLETION))
def test_known_noncompletion_is_non_success_and_all_attempt_costs_count(outcome):
    plan, rows = grid()
    for row in rows:
        if row['arm'] == 'C':
            row.update(status='unknown', execution_state='interrupted', outcome_class=outcome,
                       model_calls=dict(reserved=24))
    result = stats.aggregate(plan, rows, bootstrap_samples=100)
    assert result['comparisons']['C-minus-A']['inference_allowed']
    assert result['comparisons']['C-minus-A']['success_rate_difference'] == 0
    assert result['groups']['C']['unknown'] == 4
    assert result['groups']['C']['costs']['calls']['observed_sum'] == 96
    assert result['comparisons']['C-minus-A']['cost_per_attempt_difference']['calls'] == 22


def test_infrastructure_unknown_is_not_interpreted_as_model_failure():
    plan, rows = grid()
    rows[0].update(status='unknown', execution_state='interrupted', outcome_class='infrastructure_error')
    result = stats.aggregate(plan, rows, bootstrap_samples=100)
    pair = result['comparisons']['C-minus-A']
    assert result['completed_denominator']
    assert not pair['inference_allowed'] and pair['paired_task_differences'][0] is None
    assert pair['success_rate_difference'] is pair['task_cluster_percentile_95'] is None
    assert result['groups']['A']['costs']['calls']['observed_sum'] == 8


@pytest.mark.parametrize('change', [dict(arm='B'), dict(case_id='foreign'), dict(repeat_id=3),
    dict(repeat_id=True), dict(plan_hash='another-plan'), dict(status='success'),
    dict(status='passed', execution_state='interrupted')])
def test_foreign_or_inconsistent_record_refuses(change):
    plan, rows = grid()
    rows[0].update(change)
    with pytest.raises(ValueError):
        stats.aggregate(plan, rows, bootstrap_samples=100)


def test_duplicate_record_refuses():
    plan, rows = grid()
    with pytest.raises(ValueError, match='duplicate'):
        stats.aggregate(plan, rows + [copy.deepcopy(rows[0])], bootstrap_samples=100)


@pytest.mark.parametrize('change', [dict(arms=['A', 'B', 'C']), dict(planned_runs=9),
    dict(repeats=True), dict(hash='')])
def test_plan_must_be_new_ac_grid(change):
    plan, rows = grid()
    plan.update(change)
    with pytest.raises(ValueError):
        stats.aggregate(plan, rows, bootstrap_samples=100)


@pytest.mark.parametrize('change', [dict(model_calls=dict(reserved=True)),
    dict(model_calls=dict(reserved=1.5)), dict(wall_seconds=float('nan')),
    dict(dsh_usage=dict(outputTokens=-1)), dict(dsh_usage='unknown')])
def test_invalid_cost_or_usage_is_never_silently_summed(change):
    plan, rows = grid()
    rows[0].update(change)
    with pytest.raises(ValueError):
        stats.aggregate(plan, rows, bootstrap_samples=100)
