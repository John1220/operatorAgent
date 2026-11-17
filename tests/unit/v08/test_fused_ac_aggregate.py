"""v7 fused A/C aggregation: qualified delivery, not L2-only delivery."""
import pytest

from benchmarks.experiments.v08 import ac_plan, fused_ac_aggregate as fa


@pytest.fixture()
def plan(tmp_path):
    return ac_plan.freeze(dict(cells=4), tmp_path / 'v7.json', version='v7')


def cell(arm, delivery_class, *, task_id=None, qualified=None):
    return dict(task_id=task_id or fa.ac_plan.FUSED_TASK['task_id'], arm=arm,
                state='completed', final_receipt={'verification_purpose': 'final'},
                delivery_class=delivery_class,
                qualified=bool(qualified if qualified is not None
                               else delivery_class == 'qualified'),
                final_speedup=2.0 if delivery_class == 'qualified' else 1.1,
                final_validity='valid', fusion_static={'passed': True},
                model_dispatches=6, verifications=8, wall_clock_total_s=1200)


def pair(plan, a, c):
    return fa.pair_verdict(plan, [a, c])


def test_pair_plus_one_requires_c_qualified_and_a_not(plan):
    result = pair(plan, cell('A', 'correct_only'), cell('C', 'qualified'))
    assert result['paired_difference_C_minus_A'] == 1
    assert result['round_complete'] is True
    assert result['per_arm']['A']['delivery_class'] == 'correct_only'


def test_pair_zero_keeps_the_subtype(plan):
    both_qualified = pair(plan, cell('A', 'qualified'), cell('C', 'qualified'))
    assert both_qualified['paired_difference_C_minus_A'] == 0
    assert both_qualified['subtype'] == 'both_qualified'
    both_correct = pair(plan, cell('A', 'correct_only'), cell('C', 'correct_only'))
    assert both_correct['subtype'] == 'both_correct_only'
    both_failed = pair(plan, cell('A', 'failed'), cell('C', 'failed'))
    assert both_failed['subtype'] == 'both_failed'
    mixed = pair(plan, cell('A', 'correct_only'), cell('C', 'failed'))
    assert mixed['subtype'].startswith('mixed_')


def test_pair_minus_one_is_a_contradiction(plan):
    result = pair(plan, cell('A', 'qualified'), cell('C', 'failed'))
    assert result['paired_difference_C_minus_A'] == -1


def test_aggregate_reproduced_only_when_both_pairs_are_plus_one(plan):
    plus = pair(plan, cell('A', 'correct_only'), cell('C', 'qualified'))
    result = fa.aggregate_fused(plan, [plus, plus])
    assert result['verdict'] == 'reproduced'
    assert result['formal_gain_claim'] is True


def test_aggregate_one_plus_one_zero_is_not_established(plan):
    plus = pair(plan, cell('A', 'correct_only'), cell('C', 'qualified'))
    zero = pair(plan, cell('A', 'correct_only'), cell('C', 'correct_only'))
    result = fa.aggregate_fused(plan, [plus, zero])
    assert result['verdict'] == 'not_established'
    assert result['formal_gain_claim'] is False
    assert result['per_pair'][1]['subtype'] == 'both_correct_only'


def test_aggregate_both_zero_is_delivery_parity(plan):
    zero = pair(plan, cell('A', 'correct_only'), cell('C', 'correct_only'))
    assert fa.aggregate_fused(plan, [zero, zero])['verdict'] == 'delivery_parity'


def test_aggregate_negative_pair_is_contrary(plan):
    minus = pair(plan, cell('A', 'qualified'), cell('C', 'failed'))
    assert fa.aggregate_fused(plan, [minus, minus])['verdict'] == 'contrary'


def test_unknown_cell_makes_the_round_unknown_and_never_adds_a_third_pair(plan):
    bad = dict(task_id=fa.ac_plan.FUSED_TASK['task_id'], arm='C', state='crashed',
               outcome='unknown', final_receipt=None)
    good = pair(plan, cell('A', 'correct_only'), cell('C', 'qualified'))
    result = fa.aggregate_fused(plan, [good, fa.pair_verdict(plan, [cell('A', 'failed'), bad])])
    assert result['verdict'] == 'unknown'
    assert result['pairs_expected'] == 2 and result['pairs_observed'] == 2
