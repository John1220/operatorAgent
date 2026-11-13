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
    # 2026-11-08 起，门控的**形状**也绑到声明（与 dtype 同一条纪律）：改声明不改门控
    # 会被校验器当场拦住，不必等到冻结时靠 hash 重算。
    assert any('declares [[1024, 1024, 1024]]' in row for row in r2_plan.validate(tampered))
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


def test_each_task_carries_its_own_probe_not_the_other_precision():
    fp16 = r2_plan.design('matmul-4096')
    fp32 = r2_plan.design('matmul-4096-fp32')
    assert fp16['measurement_gate']['channel'] != fp32['measurement_gate']['channel']
    assert fp16['measurement_gate']['observed_band_pct'] == 0.203
    # 0.1% 来自 gate-fp32.json 的 effective_band_pct；25× 才是这份探针的余量。
    assert fp32['measurement_gate']['observed_band_pct'] == 0.1
    assert fp32['detectability']['measured_band_pct'] == 0.1
    assert fp32['detectability']['margin_x'] == 25.0
    assert fp32['measurement_gate']['channel'] == 'matmul-4096x4096x4096-fp32-cuda-event'
    assert r2_plan.validate(fp32) == []


def test_a_gate_probing_another_dtype_is_refused():
    plan = r2_plan.design('matmul-4096-fp32')
    # 被撤回的 8d460f94 就是这个形状：任务声明 fp32，门控命令却探 fp16（--dtype 缺省）。
    plan['measurement_gate']['command'] = r2_plan.design('matmul-4096')['measurement_gate']['command']
    assert any('probes fp16' in row for row in r2_plan.validate(plan))


def test_an_explicit_dtype_is_bound_to_the_task_too():
    plan = r2_plan.design('matmul-4096')
    for spelling in (' --dtype fp32', ' --dtype=fp32'):
        broken = dict(plan, measurement_gate=dict(plan['measurement_gate'],
                      command=plan['measurement_gate']['command'] + spelling))
        assert any('probes fp32' in row for row in r2_plan.validate(broken))


def test_the_withdrawn_plan_id_is_refused_by_name():
    plan = r2_plan.design('matmul-4096-fp32')
    plan['plan_id'] = 'operator-v08-increment-r2-fp32'
    assert any('was withdrawn' in row for row in r2_plan.validate(plan))


def test_the_two_copies_of_the_band_must_agree():
    plan = r2_plan.design('matmul-4096-fp32')
    plan['detectability']['measured_band_pct'] = 0.203
    assert any('must equal' in row for row in r2_plan.validate(plan))


def test_the_margin_is_derived_from_the_registered_band():
    for task_id in r2_plan.PLAN_IDS:
        detect = r2_plan.design(task_id)['detectability']
        assert detect['margin_x'] == round(
            detect['min_detectable_effect_pct'] / detect['measured_band_pct'], 2)

# ---------------------------------------------------------------------------
# A3 (2026-11-08): attention 进 R2 —— 门控必须量**登记的那个任务**
# ---------------------------------------------------------------------------

def test_the_registered_attention_task_is_the_one_with_a_valid_L3():
    """S=1024 的核只有 96 µs，地板占 17.3% ⇒ 每次配对结构性不可判，已撤回；
    登记的是 S=4096（实测 valid）。这条用例同时钉住"撤回按名字生效"。"""
    plan = r2_plan.design('attention-prefill-4096')
    assert plan['plan_id'] == 'operator-v08-increment-r2-attention-v2'
    gate, task = plan['measurement_gate'], plan['tasks'][0]
    assert gate['channel'] == 'attention-1x16x4096x128-fp16-nocausal-cuda-event'
    assert '--op attention' in gate['command'] and '--shape 1,16,4096,128' in gate['command']
    assert '--causal' not in gate['command']          # 主用例是 nocausal
    assert task['mask'] == 'nocausal'
    assert task['declaration']['precision'] == 'fp16'
    assert plan['detectability']['margin_x'] == 25.0
    assert r2_plan.validate(plan) == []


def test_the_superseded_attention_plan_is_refused_by_name():
    old = r2_plan.design('attention-prefill-1024')
    problems = ' '.join(r2_plan.validate(old))
    assert 'was withdrawn' in problems and 'attention-v2' in problems


def test_a_gate_borrowed_from_another_operator_is_refused():
    """fp32 计划当年继承了 fp16 的探针；同一个坑在算子层面也一样踩得进去。"""
    plan = r2_plan.design('attention-prefill-4096')
    plan['measurement_gate']['command'] = (
        'python -m operator_agent preflight --shape 4096,4096,4096 --dtype fp16')
    problems = ' '.join(r2_plan.validate(plan))
    assert 'probes \'matmul\'' in problems
    assert 'declares [[1, 16, 4096, 128]]' in problems


def test_the_mask_registration_must_match_the_gate_and_the_channel():
    plan = r2_plan.design('attention-prefill-4096')
    plan['tasks'][0]['mask'] = 'causal'
    assert any('disagrees about --causal' in p for p in r2_plan.validate(plan))

    plan = r2_plan.design('attention-prefill-4096')
    plan['measurement_gate']['channel'] = 'attention-1x16x4096x128-fp16-cuda-event'
    assert any('does not name mask=nocausal' in p for p in r2_plan.validate(plan))


def test_the_two_existing_plans_are_untouched_by_the_new_binding():
    """老计划（已冻结、已执行）必须继续通过：新增校验只收紧，不改它们的读数。"""
    for task_id in ('matmul-4096', 'matmul-4096-fp32'):
        assert r2_plan.validate(r2_plan.design(task_id)) == [], task_id
