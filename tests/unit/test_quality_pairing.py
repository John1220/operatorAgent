"""Paired experiments share candidates, not billing or mutable input state."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from operator_agent.decide import quality_evaluation as q
from operator_agent.decide import quality_pairing as paired
from operator_agent.decide import quality_suite
from operator_agent.core.run_contract import code_hash, digest
from operator_agent.core.schemas import VerificationLevel as Level, VerificationResult


CASE = 'layernorm-tail'
STRIDED = 'matmul-strided-fp32'
GOOD = ('import numpy as np\ndef kernel(X):\n'
        '    z = X.astype(np.float32)\n'
        '    z = z - z.mean(axis=-1, keepdims=True)\n'
        '    return (z / np.sqrt((z*z).mean(axis=-1, keepdims=True)+1e-5)).astype(X.dtype)\n')
BAD = 'def kernel(X):\n    return X * 0\n'


def envelope(code):
    return json.dumps(dict(code=code, rationale='Authored offline paired control.'))


def replies(code=BAD, baseline=GOOD, diagnosis=GOOD):
    fence = lambda source: '```python\n'+source+'\n```'
    return {CASE: {'generation': [envelope(code)],
                   'baseline': [] if baseline is None else [fence(baseline)],
                   'diagnosis': [] if diagnosis is None else [fence(diagnosis)]}}


def passed():
    return [VerificationResult(Level.L1_COMPILE, True),
            VerificationResult(Level.L2_FUNCTION, True,
                metrics={'correctness': {'verdict': 'passed'}})]


def failed():
    return [VerificationResult(Level.L1_COMPILE, True),
            VerificationResult(Level.L2_FUNCTION, False, 'numerical mismatch', metrics={
                'max_error': 1., 'max_rel_error': 1.,
                'correctness_primary_verdict': 'failed',
                'correctness': {'verdict': 'failed'}})]


def fake_verification(monkeypatch, outcomes):
    verify = Mock(side_effect=outcomes)
    monkeypatch.setattr(paired.q, 'Verifier', lambda _: SimpleNamespace(verify=verify))
    return verify


def test_paired_plan_freezes_one_generation_and_equal_repair_budgets_without_work(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('planning must not construct models, devices or concrete inputs')
    monkeypatch.setattr(q, 'LLMClient', forbidden)
    monkeypatch.setattr(quality_suite, 'evaluation_inputs_for', forbidden)
    plan = paired.make_paired_plan(case_ids=[CASE])
    assert plan['schema'] == 'operator-paired-quality-plan-v1'
    assert plan['base_plan']['controls']['strategy'] == 'baseline'
    assert plan['pairing'] == dict(generation_calls=1, repair_calls_per_arm=1,
        arm_order='alternating_by_case',
        cost_scope='shared_generation_once_plus_repair_only_arms')
    assert plan['hash'] == digest({k: v for k, v in plan.items() if k != 'hash'})


def test_single_call_budget_is_rejected_instead_of_granting_an_extra_repair():
    with pytest.raises(ValueError):
        paired.make_paired_plan(case_ids=[CASE], budget=q.QualityBudget(max_calls=1))


def test_rehashed_unequal_pairing_budget_fails_before_any_run_directory(tmp_path):
    plan = paired.make_paired_plan(case_ids=[CASE])
    plan['pairing']['repair_calls_per_arm'] = 2
    plan['hash'] = digest({k: v for k, v in plan.items() if k != 'hash'})
    with pytest.raises(ValueError):
        paired.run_paired_plan(plan, tmp_path/'run', responses={})
    assert not (tmp_path/'run').exists()


def test_one_shared_generation_both_arms_repair_same_source_without_double_billing(tmp_path, monkeypatch):
    verification = fake_verification(monkeypatch, [failed(), passed(), failed(), passed()])
    report = paired.run_paired_plan(paired.make_paired_plan(case_ids=[CASE]), tmp_path/'run',
        responses=replies(), backend_factory=lambda _: object())
    assert report['schema'] == 'operator-paired-quality-report-v1'
    assert verification.call_count == 4
    assert [call.args[0] for call in verification.call_args_list][::2] == [BAD, BAD]
    for call in verification.call_args_list:
        assert len(call.args[2]) == 3
        assert call.args[1].constraints.correctness_suite is not None
        assert call.args[-1] is Level.L2_FUNCTION
    shared = tmp_path/'run'/'shared'/CASE/'candidate.py'
    assert shared.read_text() == BAD
    assert shared.stat().st_mode & 0o222 == 0
    for arm in ('baseline', 'diagnosis'):
        source = (tmp_path/'run'/'arms'/arm/CASE/'candidate-1.py').read_text()
        assert code_hash(source) == code_hash(shared.read_text())
        assert report['arms'][arm]['costs']['model_calls']['known_total'] == 1
        assert report['arms'][arm]['metrics']['final_full_l2']['count'] == 1
        assert report['arms'][arm]['metrics']['native_performance_target']['rate'] is None
    assert report['shared']['costs']['model_calls']['known_total'] == 1
    assert report['total_actual_costs']['model_calls']['known_total'] == 3
    assert report['pairs'][0]['initial_source_hash'] == code_hash(BAD)
    assert report['metrics']['eligible_failed_pairs'] == 1
    assert report['metrics']['baseline_repairs'] == report['metrics']['diagnosis_repairs'] == 1


def test_initial_success_spends_generation_once_and_never_requests_a_repair(tmp_path, monkeypatch):
    fake_verification(monkeypatch, [passed(), passed()])
    made = {}
    def client_factory(case_id, phase, ledger):
        made[phase] = ledger
        return q.ReplayClient([envelope(GOOD)] if phase == 'generation' else [], ledger)
    report = paired.run_paired_plan(paired.make_paired_plan(case_ids=[CASE]), tmp_path/'run',
        responses={}, client_factory=client_factory, backend_factory=lambda _: object())
    assert made['generation'].summary()['reserved_calls'] == 1
    assert all(ledger.summary()['reserved_calls'] == 0
               for phase, ledger in made.items() if phase != 'generation')
    assert report['total_actual_costs']['model_calls']['known_total'] == 1
    assert report['metrics']['eligible_failed_pairs'] == 0
    for arm in ('baseline', 'diagnosis'):
        assert report['arms'][arm]['costs']['model_calls']['known_total'] == 0
        assert report['arms'][arm]['metrics']['first_full_l2']['count'] == 1


def test_shared_generation_elapsed_time_is_not_counted_again_inside_each_arm(tmp_path, monkeypatch):
    clock = [1000.]
    monkeypatch.setattr(q.time, 'monotonic', lambda: clock[0])
    def verify(*args, **kwargs):
        clock[0] += 1.
        return passed()
    monkeypatch.setattr(paired.q, 'Verifier', lambda _: SimpleNamespace(verify=verify))
    class TimedReplay(q.ReplayClient):
        def chat(self, *args, **kwargs):
            clock[0] += 10.
            return super().chat(*args, **kwargs)
    def client_factory(case_id, phase, ledger):
        return TimedReplay([envelope(GOOD)] if phase == 'generation' else [], ledger)
    report = paired.run_paired_plan(paired.make_paired_plan(case_ids=[CASE]), tmp_path/'run',
        responses={}, client_factory=client_factory, backend_factory=lambda _: object())
    assert report['shared']['costs']['wall_seconds']['known_total'] == pytest.approx(10.)
    for arm in ('baseline', 'diagnosis'):
        assert report['arms'][arm]['costs']['wall_seconds']['known_total'] == pytest.approx(1.)
    assert report['total_actual_costs']['wall_seconds']['known_total'] == pytest.approx(12.)


def test_generation_failure_preserves_both_denominators_and_shared_spend(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('invalid generation must not execute a candidate')
    monkeypatch.setattr(paired.q, 'Verifier', forbidden)
    report = paired.run_paired_plan(paired.make_paired_plan(case_ids=[CASE]), tmp_path/'run',
        responses={CASE: {'generation': ['not JSON'], 'baseline': [], 'diagnosis': []}},
        backend_factory=lambda _: object())
    assert report['shared']['costs']['model_calls']['known_total'] == 1
    assert report['total_actual_costs']['model_calls']['known_total'] == 1
    assert report['metrics']['eligible_failed_pairs'] == 0
    for arm in ('baseline', 'diagnosis'):
        metric = report['arms'][arm]['metrics']['final_full_l2']
        assert metric == dict(count=0, denominator=1, rate=0.)
        assert report['arms'][arm]['missing_tasks'] == 0
        assert report['arms'][arm]['costs']['model_calls']['known_total'] == 0


def test_real_preflight_failure_preserves_pair_without_constructing_a_model(tmp_path, monkeypatch):
    monkeypatch.setenv('LLM_MODEL', 'synthetic-model')
    monkeypatch.setenv('LLM_BASE_URL', 'https://model.invalid/v1')
    plan = paired.make_paired_plan(case_ids=[CASE], backend='ssh', host='synthetic-host',
        model='synthetic-model', endpoint='https://model.invalid/v1')
    seen = []
    def backend_factory(case_id):
        seen.append('backend')
        return object()
    def preflight(backend, root):
        seen.append('preflight')
        raise q.LLMCallRefused('synthetic failed control kernel')
    def forbidden(*args, **kwargs):
        pytest.fail('failed real preflight must stop before constructing a model')
    monkeypatch.setattr(paired.q, '_preflight', preflight)
    report = paired.run_paired_plan(plan, tmp_path/'run', host='synthetic-host',
        client_factory=forbidden, backend_factory=backend_factory)
    assert seen == ['backend', 'preflight']
    assert report['shared']['tasks'][0]['status'] == 'preflight_failed'
    assert report['total_actual_costs']['model_calls']['known_total'] == 0
    assert report['metrics']['eligible_failed_pairs'] == 0
    for arm in ('baseline', 'diagnosis'):
        assert report['arms'][arm]['metrics']['final_full_l2']['denominator'] == 1
        assert report['arms'][arm]['metrics']['final_full_l2']['count'] == 0
        assert report['arms'][arm]['missing_tasks'] == 0


def test_poisoned_baseline_budget_does_not_block_diagnosis_and_unknown_usage_stays_unknown(tmp_path, monkeypatch):
    fake_verification(monkeypatch, [failed(), failed(), passed()])
    made = {}
    available = replies(baseline=None)[CASE]
    def client_factory(case_id, phase, ledger):
        assert phase not in made
        made[phase] = ledger
        return q.ReplayClient(available[phase], ledger)
    report = paired.run_paired_plan(paired.make_paired_plan(case_ids=[CASE]), tmp_path/'run',
        responses={}, client_factory=client_factory, backend_factory=lambda _: object())
    assert set(made) == {'generation', 'baseline', 'diagnosis'}
    assert len({id(ledger) for ledger in made.values()}) == 3
    assert made['baseline'].summary()['aborted'] is not None
    assert made['diagnosis'].summary()['reserved_calls'] == 1
    assert report['total_actual_costs']['model_calls']['known_total'] == 3
    assert report['arms']['baseline']['costs']['tokens']['missing_records'] >= 1
    assert report['arms']['baseline']['metrics']['final_full_l2']['count'] == 0
    assert report['arms']['diagnosis']['metrics']['final_full_l2']['count'] == 1
    assert report['metrics']['diagnosis_only_repairs'] == 1


def test_no_structural_change_spends_repair_call_without_second_verification(tmp_path, monkeypatch):
    verification = fake_verification(monkeypatch, [failed(), failed(), passed()])
    report = paired.run_paired_plan(paired.make_paired_plan(case_ids=[CASE]), tmp_path/'run',
        responses=replies(baseline=BAD), backend_factory=lambda _: object())
    assert verification.call_count == 3
    assert report['total_actual_costs']['model_calls']['known_total'] == 3
    assert report['metrics']['baseline_repairs'] == 0
    assert report['metrics']['diagnosis_only_repairs'] == 1
    assert not (tmp_path/'run'/'arms'/'baseline'/CASE/'candidate-2.py').exists()


def test_conflicting_initial_observations_do_not_claim_a_repair_opportunity(tmp_path, monkeypatch):
    fake_verification(monkeypatch, [passed(), failed(), passed()])
    report = paired.run_paired_plan(paired.make_paired_plan(case_ids=[CASE]), tmp_path/'run',
        responses=replies(), backend_factory=lambda _: object())
    assert report['pairs'][0]['initial_observation_conflict'] is True
    assert report['metrics']['initial_observation_conflicts'] == 1
    assert report['metrics']['eligible_failed_pairs'] == 0
    assert report['metrics']['diagnosis_only_repairs'] == 0
    assert report['arms']['diagnosis']['metrics']['final_full_l2']['count'] == 1


def test_each_arm_gets_fresh_original_inputs_without_losing_noncontiguous_layout(tmp_path, monkeypatch):
    expected = quality_suite.evaluation_inputs_for(STRIDED)
    observed = []
    def inspect_and_mutate(code, task, inputs, *args, **kwargs):
        assert len(inputs) == 3
        for actual_set, expected_set in zip(inputs, expected):
            for name, value in actual_set.items():
                assert np.array_equal(value, expected_set[name])
                assert value.strides == expected_set[name].strides
                assert not value.flags.c_contiguous
        observed.append(inputs)
        for value_set in inputs:
            for value in value_set.values():
                value[...] = 0
        return passed()
    monkeypatch.setattr(paired.q, 'Verifier', lambda _: SimpleNamespace(verify=inspect_and_mutate))
    report = paired.run_paired_plan(paired.make_paired_plan(case_ids=[STRIDED]), tmp_path/'run',
        responses={STRIDED: {'generation': [envelope('def kernel(A,B): return A @ B')],
                            'baseline': [], 'diagnosis': []}}, backend_factory=lambda _: object())
    assert len(observed) == 2
    for name in observed[0][0]:
        assert not np.shares_memory(observed[0][0][name], observed[1][0][name])
    assert report['metrics']['initial_observation_conflicts'] == 0


def test_manifest_order_alternates_arm_order_and_shared_backend_is_built_once_per_task(tmp_path, monkeypatch):
    fake_verification(monkeypatch, [passed(), passed(), passed(), passed()])
    backend_factory = Mock(side_effect=lambda _: object())
    report = paired.run_paired_plan(paired.make_paired_plan(case_ids=[CASE, STRIDED]), tmp_path/'run',
        responses={CASE: {'generation': [envelope(GOOD)]},
                   STRIDED: {'generation': [envelope('def kernel(A,B): return A @ B')]}},
        backend_factory=backend_factory)
    assert backend_factory.call_count == 2
    assert report['pairs'][0]['arm_order'] == ['baseline', 'diagnosis']
    assert report['pairs'][1]['arm_order'] == ['diagnosis', 'baseline']


def test_real_cpu_pairing_revalidates_repaired_source_and_cannot_overwrite_evidence(tmp_path):
    plan = paired.make_paired_plan(case_ids=[CASE])
    report = paired.run_paired_plan(plan, tmp_path/'run', responses=replies())
    assert report['metrics']['eligible_failed_pairs'] == 1
    assert report['metrics']['baseline_repairs'] == report['metrics']['diagnosis_repairs'] == 1
    assert report['total_actual_costs']['model_calls']['known_total'] == 3
    for arm in ('baseline', 'diagnosis'):
        assert report['arms'][arm]['metrics']['final_full_l2']['count'] == 1
        path = tmp_path/'run'/'arms'/arm/CASE/'verification-2.json'
        results = json.loads(path.read_text())['results']
        l2 = next(row for row in results if row['level'] == 'L2')
        assert l2['metrics']['num_input_sets'] == 3
        assert l2['metrics']['correctness']['verdict'] == 'passed'
    with pytest.raises(FileExistsError):
        paired.run_paired_plan(plan, tmp_path/'run', responses=replies())


@pytest.mark.parametrize('field', ['metrics', 'arm_costs', 'shared_costs', 'pair_binding'])
def test_paired_report_recomputes_aggregates_and_rejects_edits(tmp_path, monkeypatch, field):
    fake_verification(monkeypatch, [passed(), passed()])
    report = paired.run_paired_plan(paired.make_paired_plan(case_ids=[CASE]), tmp_path/'run',
        responses=replies(code=GOOD), backend_factory=lambda _: object())
    assert paired.validate_paired_report(report) is report
    changed = copy.deepcopy(report)
    if field == 'metrics':
        changed['metrics']['diagnosis_only_repairs'] = 99
    elif field == 'arm_costs':
        changed['arms']['diagnosis']['costs']['model_calls']['known_total'] = 99
    elif field == 'shared_costs':
        changed['shared']['costs']['model_calls']['known_total'] = 99
    else:
        changed['pairs'][0]['initial_source_hash'] = '0' * 64
    with pytest.raises(ValueError):
        paired.validate_paired_report(changed)


def test_shared_generation_time_is_not_charged_again_to_either_arm(tmp_path, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(q.time, 'monotonic', lambda: clock[0])
    def client_factory(case_id, phase, ledger):
        client = q.ReplayClient([envelope(GOOD)] if phase == 'generation' else [], ledger)
        chat = client.chat
        def timed_chat(*args, **kwargs):
            result = chat(*args, **kwargs)
            clock[0] += 10
            return result
        client.chat = timed_chat
        return client
    def verification(*args, **kwargs):
        clock[0] += 1
        return passed()
    monkeypatch.setattr(q, 'Verifier', lambda _: SimpleNamespace(verify=verification))
    report = paired.run_paired_plan(paired.make_paired_plan(case_ids=[CASE]), tmp_path/'run',
        responses={}, client_factory=client_factory, backend_factory=lambda _: object())
    assert report['shared']['costs']['wall_seconds']['known_total'] == 10
    assert all(report['arms'][arm]['costs']['wall_seconds']['known_total'] == 1
               for arm in ('baseline', 'diagnosis'))
    assert report['total_actual_costs']['wall_seconds']['known_total'] == 12


def test_smaller_output_budget_still_allows_a_bounded_repair_in_both_arms(tmp_path, monkeypatch):
    fake_verification(monkeypatch, [failed(), passed(), failed(), passed()])
    plan = paired.make_paired_plan(case_ids=[CASE], budget=q.QualityBudget(max_output_tokens=1024))
    report = paired.run_paired_plan(plan, tmp_path/'run', responses=replies(),
                                    backend_factory=lambda _: object())
    assert report['metrics']['baseline_repairs'] == report['metrics']['diagnosis_repairs'] == 1
    assert report['total_actual_costs']['model_calls']['known_total'] == 3
    for arm in ('baseline', 'diagnosis'):
        request = json.loads((tmp_path/'run'/'arms'/arm/CASE/'model-journal'/'call-001.json').read_text())
        assert request['request']['max_tokens'] == 1024


@pytest.mark.parametrize('change', ['negative_cost', 'input_binding'])
def test_shared_audit_records_cannot_supply_negative_cost_or_another_input_domain(tmp_path, monkeypatch, change):
    fake_verification(monkeypatch, [passed(), passed()])
    report = paired.run_paired_plan(paired.make_paired_plan(case_ids=[CASE]), tmp_path/'run',
        responses=replies(code=GOOD), backend_factory=lambda _: object())
    changed = copy.deepcopy(report)
    shared = changed['shared']['tasks'][0]
    if change == 'negative_cost':
        shared['model_calls'] = -1
    else:
        shared['input_manifest_hash'] = '0' * 64
    with pytest.raises(ValueError):
        paired.validate_paired_report(changed)
