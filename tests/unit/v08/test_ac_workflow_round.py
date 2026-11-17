"""A/C 工作流轮：预登记不能事后挑判据，比较器必须把"没交付"当数据而不是坏输入。

这一轮与 B/C 增量轮的区别不是措辞：`r2_compare.arm_summary` 在任一臂终评 L2 没过时抛错，
而这里 A 臂没过正是待测的信号。下面钉住三件容易被悄悄改掉的事：
  * 失败/未知/未执行三态不许互相转化，分母不许缩小；
  * 源码哈希由比较器按文件重算，回执自称的不算；
  * 即使 C 多交付一个任务，`formal_gain_claim` 仍是 False（采样可比性未建立）。
"""
import hashlib
import json

import pytest

from benchmarks.experiments.v08 import ac_plan
from benchmarks.experiments.v08.ac_compare import (RoundInputError, cell_summary,
                                                   compare_round)
from benchmarks.experiments.v08.arm_gate import ArmGate


BUDGET = dict(cells=8, gpu_minutes=45)


def frozen(tmp_path, name='plan.json', **overrides):
    return ac_plan.freeze(dict(BUDGET, **overrides), tmp_path / name)


def write_source(tmp_path, name, text='def kernel():\n    pass\n'):
    path = tmp_path / name
    path.write_text(text, encoding='utf-8')
    return path


def receipt(source, *, l2=True, validity='valid', speedup=0.9, purpose='final',
            contract='contract-1', source_hash=None):
    return dict(verification_purpose=purpose, source=str(source),
                source_hash=source_hash or hashlib.sha256(source.read_bytes()).hexdigest(),
                contract_hash=contract, validity=validity,
                levels=dict(L1=True, L2=l2, L3=validity == 'valid'),
                verdicts=[dict(level='L3', passed=validity == 'valid',
                               metrics=dict(speedup_vs_baseline=speedup,
                                            median_time_ms=5.0,
                                            measurement_validity=validity))],
                host_task=dict(used=8))


def cell(tmp_path, arm, task_id, *, l2=True, state='completed', **kwargs):
    source = write_source(tmp_path, f'{arm}-{task_id}.py', f'# {arm} {task_id}\n')
    return dict(arm=arm, task_id=task_id, state=state, source=str(source),
                final_receipt=receipt(source, l2=l2, **kwargs),
                model_dispatches=11, verifications=8, wall_clock_to_first_full_L2_s=17.4)


def full_round(tmp_path, *, a_l2=False, c_l2=True):
    return [cell(tmp_path, arm, task_id, l2=(a_l2 if arm == 'A' else c_l2))
            for task_id in ac_plan.TASK_IDS for arm in ac_plan.ARMS]


# ── 预登记 ──────────────────────────────────────────────────────────────────

def test_the_design_is_executable_before_any_budget_exists():
    assert ac_plan.validate(ac_plan.design()) == []


def test_freeze_refuses_a_budget_decided_later(tmp_path):
    with pytest.raises(ValueError) as caught:
        ac_plan.freeze(dict(cells=8), tmp_path / 'plan.json')      # gpu_minutes 仍 None
    assert 'gpu_minutes' in str(caught.value)
    assert not (tmp_path / 'plan.json').exists()


def test_freeze_writes_once_and_the_hash_binds_the_content(tmp_path):
    plan = frozen(tmp_path)
    assert plan['status'] == 'frozen'
    assert ac_plan.validate(plan) == []
    with pytest.raises(FileExistsError):
        frozen(tmp_path)
    drifted = json.loads(json.dumps(plan))
    drifted['metric']['minimum_detectable_unit'] = '+2 tasks'
    assert any('hash' in problem for problem in ac_plan.validate(drifted))


def test_the_frozen_budget_is_the_one_that_overrides_the_cli_defaults(tmp_path):
    plan = frozen(tmp_path)
    # CLI 缺省 24 调用 / 1800 s；不显式覆盖就等于两臂预算不同。
    assert plan['budget']['max_model_calls'] == 30
    assert plan['budget']['wall_clock_s'] == 2400
    assert plan['budget']['cells'] == len(ac_plan.TASKS) * 2
    with pytest.raises(ValueError) as caught:
        frozen(tmp_path, name='bad.json', cells=6)
    assert 'cells must equal' in str(caught.value)


def test_the_arm_gate_mapping_uses_the_names_the_gate_enforces(tmp_path):
    plan = frozen(tmp_path)
    gate = ArmGate(tmp_path / 'arm-A.json', budget=ac_plan.arm_gate_budget(plan['budget']),
                   clock=lambda: 0.0)
    # 断言落在 arm log 这份公开产物上，不是 ArmGate 的私有属性。
    document = json.loads(gate.path.read_text())
    assert document['budget']['max_calls'] == 30
    assert document['budget']['wall_clock_s'] == 2400
    # 这套名字确实会被硬门消费：30 次用完后第 31 次派发根本发不出去。
    for _ in range(30):
        gate.finish(gate.reserve(provider='aliyun', model='m', max_output_tokens=1))
    with pytest.raises(Exception) as caught:
        gate.reserve(provider='aliyun', model='m', max_output_tokens=1)
    assert caught.value.code == 'arm_call_budget'


def test_a_plan_cannot_carry_a_gain_or_an_outcome(tmp_path):
    plan = frozen(tmp_path)
    for field, value in (('formal_gain_claim', True), ('formal_comparison_credit', True),
                         ('outcome', dict(delivered=4))):
        drifted = json.loads(json.dumps(plan))
        drifted[field] = value
        drifted['hash'] = ac_plan.digest({k: v for k, v in drifted.items() if k != 'hash'})
        assert ac_plan.validate(drifted), field


def test_every_registered_declaration_actually_builds_a_task(tmp_path):
    plan = frozen(tmp_path)
    kinds = {}
    for task in plan['tasks']:
        kinds[task['task_id']] = ac_plan._kernel_type(task['declaration'])
    assert kinds == dict({'matmul-4096-fp32': 'matmul',
                          'attention-prefill-4096': 'attention',
                          'softmax-16384x6144-fp16': 'softmax',
                          'matmul-2048x16384x2048-fp16': 'matmul'})


def test_a_declaration_that_cannot_be_prepared_cannot_be_frozen(tmp_path):
    plan = frozen(tmp_path)
    # Scope A 之外：S=1000 在 build_task 就被 attention_scope 具名拒绝。
    broken = json.loads(json.dumps(plan))
    broken['tasks'][1]['declaration']['shapes'] = [[1, 16, 1000, 128]]
    broken['hash'] = ac_plan.digest({k: v for k, v in broken.items() if k != 'hash'})
    assert any('does not build a task' in problem for problem in ac_plan.validate(broken))


def test_a_typo_operator_name_is_not_allowed_to_become_unknown(tmp_path):
    plan = frozen(tmp_path)
    drifted = json.loads(json.dumps(plan))
    drifted['tasks'][2]['declaration']['request'] = 'fp16 not_an_operator 16384x6144'
    drifted['hash'] = ac_plan.digest({k: v for k, v in drifted.items() if k != 'hash'})
    # 拼错算子时 build_task 不报错，而是给出 kernel_type='unknown'，没有 profile。
    problems = ac_plan.validate(drifted)
    assert any("kernel_type 'unknown'" in problem for problem in problems)


def test_only_one_slot_is_a_held_out_family(tmp_path):
    plan = frozen(tmp_path)
    labels = {row['task_id']: row['held_out'] for row in plan['tasks']}
    assert sum(1 for value in labels.values() if value == 'family') == 1
    assert labels['softmax-16384x6144-fp16'] == 'family'
    # 高 K matmul 的族参与过规则开发，只能标 configuration。
    assert labels['matmul-2048x16384x2048-fp16'] == 'configuration'
    assert any('held-out CONFIGURATION' in line for line in plan['boundaries'])


def test_the_two_surfaces_are_not_the_same_surface(tmp_path):
    plan = frozen(tmp_path)
    assert plan['arm_surfaces']['A']['ordinary_tools'] is True
    assert plan['arm_surfaces']['A']['plugin'] is False
    assert plan['arm_surfaces']['C']['plugin'] is True
    assert plan['arm_surfaces']['C']['ordinary_tools'] is False


# ── 比较器：失败是数据 ────────────────────────────────────────────────────────

def test_an_arm_that_fails_l2_is_counted_not_raised(tmp_path):
    plan = frozen(tmp_path)
    verdict = compare_round(plan, full_round(tmp_path, a_l2=False, c_l2=True))
    delivery = verdict['delivery']
    assert delivery['per_arm']['A'] == dict(dispatched=4, delivered=0, failed=4,
                                            unknown=0, not_executed=0)
    assert delivery['per_arm']['C']['delivered'] == 4
    assert delivery['paired_difference_C_minus_A'] == 4
    assert delivery['clears_minimum_detectable_unit'] is True
    assert delivery['round_complete'] is True


def test_a_null_is_reported_as_a_null(tmp_path):
    plan = frozen(tmp_path)
    verdict = compare_round(plan, full_round(tmp_path, a_l2=True, c_l2=True))
    assert verdict['delivery']['paired_difference_C_minus_A'] == 0
    assert verdict['delivery']['clears_minimum_detectable_unit'] is False
    assert verdict['delivery']['round_complete'] is True


def test_one_extra_delivery_is_the_registered_unit(tmp_path):
    plan = frozen(tmp_path)
    cells = full_round(tmp_path, a_l2=True, c_l2=True)
    # 只让 A 的第一格没交付：整轮差值恰好等于登记的最小可检出单位 +1。
    cells[0] = cell(tmp_path, 'A', ac_plan.TASK_IDS[0], l2=False)
    verdict = compare_round(plan, cells)
    assert verdict['delivery']['paired_difference_C_minus_A'] == 1
    assert verdict['delivery']['clears_minimum_detectable_unit'] is True
    # 即使达标，采样可比性未建立 ⇒ 仍然只是观测。
    assert verdict['formal_gain_claim'] is False
    assert verdict['formal_comparison_credit'] is False
    assert verdict['sampling_comparability'] == 'not_established'


def test_a_drifted_source_hash_stops_the_round(tmp_path):
    plan = frozen(tmp_path)
    cells = full_round(tmp_path)
    cells[0]['final_receipt']['source_hash'] = '0' * 64
    with pytest.raises(RoundInputError) as caught:
        compare_round(plan, cells)
    assert '判的不是提交的那份代码' in str(caught.value)


def test_an_explore_receipt_cannot_stand_in_for_the_final_one(tmp_path):
    plan = frozen(tmp_path)
    cells = full_round(tmp_path)
    cells[0]['final_receipt']['verification_purpose'] = 'explore'
    with pytest.raises(RoundInputError) as caught:
        compare_round(plan, cells)
    assert 'verification_purpose=final' in str(caught.value)


def test_a_terminal_cell_without_a_receipt_is_failed_not_unknown(tmp_path):
    plan = frozen(tmp_path)
    cells = full_round(tmp_path)
    for state in ('budget_exhausted', 'deadline'):
        cells[0]['state'] = state
        cells[0]['final_receipt'] = None
        summary = cell_summary(cells[0], plan_tasks=ac_plan.TASK_IDS)
        assert summary['outcome'] == 'failed'
        assert summary['delivered'] is False


def test_a_crash_is_unknown_and_blocks_the_round(tmp_path):
    plan = frozen(tmp_path)
    cells = full_round(tmp_path)
    cells[0].update(state='crashed', final_receipt=None)
    verdict = compare_round(plan, cells)
    assert verdict['delivery']['per_arm']['A']['unknown'] == 1
    assert verdict['delivery']['round_complete'] is False
    assert 'A/matmul-4096-fp32' in verdict['delivery']['incomplete_cells']


def test_an_unexecuted_cell_is_listed_and_keeps_the_denominator_honest(tmp_path):
    plan = frozen(tmp_path)
    cells = full_round(tmp_path)
    cells[0].update(state='not_executed', final_receipt=None)
    verdict = compare_round(plan, cells)
    counts = verdict['delivery']['per_arm']['A']
    assert counts['not_executed'] == 1
    assert counts['dispatched'] == 3
    assert verdict['delivery']['round_complete'] is False


def test_a_missing_cell_also_makes_the_round_incomplete(tmp_path):
    plan = frozen(tmp_path)
    cells = full_round(tmp_path)[:-1]
    verdict = compare_round(plan, cells)
    assert verdict['delivery']['round_complete'] is False
    assert verdict['delivery']['incomplete_cells']


def test_duplicate_cells_and_unknown_identities_are_rejected(tmp_path):
    plan = frozen(tmp_path)
    cells = full_round(tmp_path)
    with pytest.raises(RoundInputError):
        compare_round(plan, cells + [cells[0]])
    with pytest.raises(RoundInputError):
        compare_round(plan, cells + [dict(cells[0], task_id='not-registered')])
    with pytest.raises(RoundInputError):
        compare_round(plan, cells + [dict(cells[0], arm='B')])
    with pytest.raises(RoundInputError):
        compare_round(plan, cells + [dict(cells[0], state='probably_fine')])


def test_an_unfrozen_plan_cannot_produce_a_verdict(tmp_path):
    with pytest.raises(RoundInputError):
        compare_round(ac_plan.design(), full_round(tmp_path))


# ── 性能只是次要观测 ──────────────────────────────────────────────────────────

def test_performance_needs_both_delivered_valid_and_one_contract(tmp_path):
    plan = frozen(tmp_path)
    verdict = compare_round(plan, full_round(tmp_path, a_l2=True, c_l2=True))
    first = verdict['performance'][0]
    assert first['comparable'] is True
    assert first['speedup_difference_C_minus_A'] == 0.0

    cells = full_round(tmp_path, a_l2=True, c_l2=True)
    cells[1]['final_receipt']['validity'] = 'degraded'
    degraded = compare_round(plan, cells)['performance'][0]
    assert degraded['comparable'] is False
    assert degraded['speedup_difference_C_minus_A'] is None
    assert any('degraded' in reason for reason in degraded['reasons'])


def test_a_failed_arm_contributes_no_performance_number(tmp_path):
    plan = frozen(tmp_path)
    verdict = compare_round(plan, full_round(tmp_path, a_l2=False, c_l2=True))
    assert all(row['comparable'] is False for row in verdict['performance'])
    assert all(row['relative_speedup_difference_pct'] is None for row in verdict['performance'])
    assert any('未交付完整 L2' in reason for row in verdict['performance'] for reason in row['reasons'])


def test_different_contracts_are_not_compared(tmp_path):
    plan = frozen(tmp_path)
    cells = full_round(tmp_path, a_l2=True, c_l2=True)
    cells[1]['final_receipt']['contract_hash'] = 'contract-2'
    observation = compare_round(plan, cells)['performance'][0]
    assert observation['comparable'] is False
    assert '两臂契约哈希不同' in observation['reasons']

# ---------------------------------------------------------------------------
# 2026-11-09：预算口径修正（线上 32768 / 每次预留 40960），v1 按名字撤回
# ---------------------------------------------------------------------------

def test_the_output_budget_is_the_harness_wire_policy_not_the_plugin_default():
    """v1 抄了插件 host budget 的 8192/245760，把每一步的输出截在 8192。"""
    from benchmarks.experiments.phase1 import sampling

    assert ac_plan.WIRE_MAX_TOKENS == sampling.POLICY['max_tokens'] == 32768
    assert ac_plan.OUTPUT_RESERVATION_PER_CALL >= ac_plan.WIRE_MAX_TOKENS
    assert ac_plan.FROZEN_PER_CELL['max_output_tokens_per_call'] == 40960
    assert (ac_plan.FROZEN_PER_CELL['total_output_token_reservations']
            == ac_plan.FROZEN_PER_CELL['max_model_calls'] * 40960)


def test_the_v1_plan_is_withdrawn_by_name():
    problems = ' '.join(ac_plan.validate(ac_plan.design('v1')))
    assert 'was withdrawn' in problems and 'ac-v2' in problems
    assert ac_plan.design()['plan_id'] == 'operator-v08-workflow-ac-v2'


def test_v7_is_the_fused_qualified_delivery_plan():
    plan = ac_plan.design('v7')
    assert ac_plan.validate(plan) == []
    assert plan['plan_id'] == 'operator-v08-workflow-ac-v7'
    assert [task['task_id'] for task in plan['tasks']] == [
        'fused-layernorm-residual-32768x6144-fp16']
    task = plan['tasks'][0]
    assert task['declaration']['min_speedup'] == 1.5
    assert task['acceptance']['fusion_check'] == 'v08-fusion-static-v1'
    assert plan['pairs'] == 2
    assert plan['execution']['gpu']['uuid'].startswith('GPU-52ca6155')
    assert plan['execution']['wire_max_tokens'] == 32768
    assert plan['execution']['sampling']['reasoning_effort'] == 'xhigh'
    assert 'qualified' in plan['metric']['primary'].lower()


def test_v7_freezes_the_fused_budget_and_pins_the_gpu(tmp_path):
    plan = ac_plan.freeze(dict(cells=4), tmp_path / 'v7.json', version='v7')
    assert plan['budget']['max_model_calls'] == 20
    assert plan['budget']['max_verifications'] == 8
    assert plan['budget']['final_reserve'] == 1  # HostTaskLimits 只允许 1 个 final 预留
    assert plan['budget']['wall_clock_s'] == 2400
    assert plan['budget']['cells'] == 4
    assert plan['hash']


def test_v7_rejects_a_cell_count_that_is_not_tasks_arms_pairs(tmp_path):
    with pytest.raises(ValueError):
        ac_plan.freeze(dict(cells=2), tmp_path / 'bad.json', version='v7')
    plan = ac_plan.design('v7')
    plan['budget'] = dict(ac_plan.FROZEN_FUSED_PER_CELL, cells=2)
    plan['status'] = 'frozen'
    problems = ' '.join(ac_plan.validate(plan))
    assert 'tasks x arms x pairs' in problems


def test_v7_rejects_drifted_execution_identity():
    plan = ac_plan.design('v7')
    plan['execution']['gpu']['uuid'] = 'GPU-deadbeef'
    assert any('execution.gpu' in issue for issue in ac_plan.validate(plan))
    plan = ac_plan.design('v7')
    plan['execution']['wire_max_tokens'] = 40960
    assert any('wire_max_tokens' in issue for issue in ac_plan.validate(plan))
    plan = ac_plan.design('v7')
    plan['execution']['sampling'].pop('reasoning_effort')
    assert any('execution.sampling' in issue for issue in ac_plan.validate(plan))
