"""The R2 preregistration must be freezable, checkable, and unable to claim results."""
import json

import pytest

from benchmarks.experiments.v08 import r2_plan

COMPLETE = dict(pairs=2, max_model_calls=12, max_output_tokens_per_call=8192,
                total_output_token_reservations=98304, max_verifications=6,
                wall_clock_s=2400, gpu_minutes=60)


def frozen(tmp_path, **budget):
    return r2_plan.freeze(dict(COMPLETE, **budget), tmp_path/'preregister.json')


def test_a_draft_carries_no_outcome_and_no_credit():
    plan = r2_plan.design()
    assert plan['status'] == 'draft_awaiting_budget'
    assert plan['outcome'] is None
    assert plan['formal_gain_claim'] is False and plan['formal_comparison_credit'] is False
    assert plan['budget'] == {name: None for name in r2_plan.BUDGET_FIELDS}
    assert r2_plan.validate(plan) == []


def test_the_design_hash_is_stable_and_covers_the_task_list():
    first, second = r2_plan.design(), r2_plan.design()
    assert first['design_hash'] == second['design_hash']
    tampered = r2_plan.design()
    tampered['tasks'][0]['declaration']['shapes'] = [[1024, 1024, 1024]]
    assert r2_plan.validate(tampered) == []          # 设计变了但 hash 没变 => 由冻结时重算拦住
    assert tampered['tasks'][0]['declaration']['shapes'] != r2_plan.design()['tasks'][0]['declaration']['shapes']


def test_a_registered_effect_must_clear_the_measured_band_with_margin():
    plan = r2_plan.design()
    plan['detectability']['min_detectable_effect_pct'] = 0.5
    problems = r2_plan.validate(plan)
    assert any('does not clear' in row for row in problems)
    # 2.0% 也不够: 0.203% × 10 = 2.03%, 这正是写这份文件时被自己的规则拦下的数
    plan['detectability']['min_detectable_effect_pct'] = 2.0
    assert any('does not clear' in row for row in r2_plan.validate(plan))
    plan['detectability']['min_detectable_effect_pct'] = 2.5
    assert r2_plan.validate(plan) == []


def test_the_series_stops_after_two_pairs_without_a_valid_reading():
    plan = r2_plan.design()
    assert plan['stop_rules']['per_series']['no_valid_pairs_in_a_row'] == 2
    plan['stop_rules']['per_series']['no_valid_pairs_in_a_row'] = 3
    assert any('two consecutive pairs' in row for row in r2_plan.validate(plan))
    assert 'a held window' in plan['measurement_gate']['on_hold']
    assert plan['measurement_gate']['start_only_if'] == 'action == "start"'


def test_freezing_requires_every_budget_and_writes_once(tmp_path):
    with pytest.raises(ValueError, match='missing'):
        r2_plan.freeze(dict(COMPLETE, max_model_calls=None), tmp_path/'a.json')
    assert not (tmp_path/'a.json').exists()
    plan = frozen(tmp_path)
    assert plan['status'] == 'frozen' and plan['budget']['pairs'] == 2
    stored = json.loads((tmp_path/'preregister.json').read_text(encoding='utf-8'))
    assert stored == plan and stored['hash'] == plan['hash']
    assert r2_plan.validate(stored) == []        # 冻结后的计划必须自己就能过校验
    with pytest.raises(FileExistsError):
        r2_plan.freeze(COMPLETE, tmp_path/'preregister.json')      # 不允许覆盖已冻结的计划


def test_a_frozen_plan_refuses_a_gain_claim_or_an_outcome(tmp_path):
    plan = frozen(tmp_path)
    for field, value in (('formal_gain_claim', True), ('outcome', {'speedup': 1.4})):
        broken = dict(plan, **{field: value})
        assert any('cannot' in row for row in r2_plan.validate(broken))
    broken = dict(plan)
    broken['budget'] = dict(broken['budget'], pairs=3)
    assert any('hash does not match' in row for row in r2_plan.validate(broken))


def test_boundaries_name_what_this_series_cannot_conclude():
    plan = r2_plan.design()
    text = ' '.join(plan['boundaries'])
    assert 'sampling_comparability' in text
    assert 'no A/B workflow claim' in text
    assert 'unknown' in text