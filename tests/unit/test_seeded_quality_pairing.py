"""Retained failures start paired repair trials without another generation call."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from operator_agent import quality_evaluation as q
from operator_agent import quality_pairing as paired
from operator_agent import quality_suite
from operator_agent.run_contract import code_hash, digest
from operator_agent.schemas import VerificationLevel as Level, VerificationResult


CASE = 'layernorm-tail'
STRIDED = 'matmul-strided-fp32'
BAD = 'def kernel(X):\n    return X * 0\n'
GOOD = ('import numpy as np\ndef kernel(X):\n'
        '    z = X.astype(np.float32)\n'
        '    z = z - z.mean(axis=-1, keepdims=True)\n'
        '    return (z / np.sqrt((z*z).mean(axis=-1, keepdims=True)+1e-5)).astype(X.dtype)\n')


def seeded(source=BAD, **kwargs):
    return paired.make_seeded_paired_plan(initial_sources={CASE: source},
        case_ids=[CASE], budget=q.QualityBudget(max_calls=1, max_verifications=2), **kwargs)


def replies(baseline=GOOD, diagnosis=GOOD):
    return {CASE: {arm: [] if source is None else ['```python\n'+source+'\n```']
                   for arm, source in [('baseline', baseline), ('diagnosis', diagnosis)]}}


def result(verdict):
    return [VerificationResult(Level.L1_COMPILE, True),
            VerificationResult(Level.L2_FUNCTION, verdict is True, 'offline observation',
                metrics={'correctness_primary_verdict':
                         'unknown' if verdict is None else ('passed' if verdict else 'failed'),
                         'correctness': {'verdict':
                         'unknown' if verdict is None else ('passed' if verdict else 'failed')}})]


def verifier(monkeypatch, verdicts):
    verify = Mock(side_effect=[result(v) for v in verdicts])
    monkeypatch.setattr(q, 'Verifier', lambda _: SimpleNamespace(verify=verify))
    return verify


def rehash(plan):
    plan['hash'] = digest({key: value for key, value in plan.items() if key != 'hash'})
    return plan


def test_seeded_planning_is_pure_and_freezes_the_exact_retained_bytes(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('freezing retained source must not construct clients or evaluation inputs')
    monkeypatch.setattr(q, 'LLMClient', forbidden)
    monkeypatch.setattr(quality_suite, 'evaluation_inputs_for', forbidden)
    plan = seeded()
    assert plan['schema'] == 'operator-seeded-paired-quality-plan-v1'
    assert plan['initial_sources'] == {CASE: {'source': BAD, 'sha256': code_hash(BAD)}}
    assert plan['base_plan']['controls']['strategy'] == 'baseline'
    assert plan['pairing'] == dict(generation_calls=0, repair_calls_per_arm=1,
        arm_order='alternating_by_case', cost_scope='seeded_initial_source_plus_repair_only_arms')
    assert plan['hash'] == digest({key: value for key, value in plan.items() if key != 'hash'})
    assert paired.validate_paired_plan(plan) is plan


def test_seed_mapping_must_cover_exactly_the_frozen_cases():
    for sources in ({}, {CASE: BAD, STRIDED: 'def kernel(A,B): return A @ B'},
                    {STRIDED: 'def kernel(A,B): return A @ B'}):
        with pytest.raises(ValueError):
            paired.make_seeded_paired_plan(initial_sources=sources, case_ids=[CASE])


def test_seeded_default_budget_is_one_repair_with_two_candidate_verifications():
    plan = paired.make_seeded_paired_plan(initial_sources={CASE: BAD}, case_ids=[CASE])
    assert plan['base_plan']['controls']['budget']['max_calls'] == 1
    assert plan['base_plan']['controls']['budget']['max_verifications'] == 2
    assert plan['pairing']['repair_calls_per_arm'] == 1


def test_utf8_source_at_the_byte_limit_is_frozen_without_truncation():
    source = '#' + '中'*10922 + '\n'
    assert len(source.encode('utf-8')) == 32768
    plan = seeded(source)
    assert plan['initial_sources'][CASE]['source'] == source
    assert plan['initial_sources'][CASE]['sha256'] == code_hash(source)


@pytest.mark.parametrize('source', [None, b'def kernel(X): return X', 17, '', ' \n\t',
                                  '#' + 'x'*32768, '#' + '中'*10923])
def test_seeded_source_rejects_invalid_types_blank_and_oversized_utf8(source):
    with pytest.raises(ValueError):
        seeded(source)


@pytest.mark.parametrize('change', ['source', 'source_hash', 'extra_seed_field',
                                    'extra_plan_field', 'repair_budget', 'generation_budget'])
def test_rehashed_seed_or_control_drift_fails_before_run_directory(tmp_path, change):
    plan = seeded()
    if change == 'source':
        plan['initial_sources'][CASE]['source'] = GOOD
    elif change == 'source_hash':
        plan['initial_sources'][CASE]['sha256'] = '0'*64
    elif change == 'extra_seed_field':
        plan['initial_sources'][CASE]['comment'] = 'unfrozen control'
    elif change == 'extra_plan_field':
        plan['generation'] = True
    elif change == 'repair_budget':
        plan['pairing']['repair_calls_per_arm'] = 2
    else:
        plan['pairing']['generation_calls'] = 1
    rehash(plan)
    with pytest.raises(ValueError):
        paired.run_paired_plan(plan, tmp_path/'run', responses={})
    assert not (tmp_path/'run').exists()


def test_evaluator_source_drift_is_rejected_before_reusing_a_seed(tmp_path, monkeypatch):
    plan = seeded()
    monkeypatch.setattr(q, 'evaluator_identity', lambda: '0'*64)
    with pytest.raises(ValueError):
        paired.run_paired_plan(plan, tmp_path/'run', responses={})
    assert not (tmp_path/'run').exists()


def test_two_fresh_verifications_and_repairs_do_not_generate_or_double_bill(tmp_path, monkeypatch):
    verify = verifier(monkeypatch, [False, True, False, True])
    made = []
    responses = replies()
    def client_factory(case_id, phase, ledger):
        assert phase in ('baseline', 'diagnosis')
        made.append((case_id, phase, ledger))
        return q.ReplayClient(responses[case_id][phase], ledger)
    report = paired.run_paired_plan(seeded(), tmp_path/'run', responses=responses,
        client_factory=client_factory, backend_factory=lambda _: object())
    assert report['schema'] == 'operator-seeded-paired-quality-report-v1'
    assert verify.call_count == 4
    assert [call.args[0] for call in verify.call_args_list] == [BAD, GOOD.rstrip(), BAD, GOOD.rstrip()]
    assert [(case_id, phase) for case_id, phase, _ in made] == [
        (CASE, 'baseline'), (CASE, 'diagnosis')]
    assert len({id(ledger) for _, _, ledger in made}) == 2
    assert all(ledger.summary()['reserved_calls'] == 1 for _, _, ledger in made)
    parent = tmp_path/'run'/'shared'/CASE/'candidate.py'
    assert parent.read_text() == BAD
    assert parent.stat().st_mode & 0o222 == 0
    assert not (parent.parent/'model-journal').exists()
    assert report['shared']['tasks'][0]['status'] == 'seeded'
    assert report['shared']['costs']['model_calls']['known_total'] == 0
    assert report['shared']['costs']['tokens']['known_total'] == 0
    assert report['total_actual_costs']['model_calls']['known_total'] == 2
    assert report['metrics']['eligible_failed_pairs'] == 1
    assert report['metrics']['baseline_repairs'] == report['metrics']['diagnosis_repairs'] == 1
    for arm in ('baseline', 'diagnosis'):
        assert (tmp_path/'run'/'arms'/arm/CASE/'candidate-1.py').read_text() == BAD
        assert report['arms'][arm]['costs']['model_calls']['known_total'] == 1
        assert report['arms'][arm]['metrics']['first_full_l2']['count'] == 0
        assert report['arms'][arm]['metrics']['final_full_l2']['count'] == 1
        assert report['arms'][arm]['metrics']['native_performance_target']['rate'] is None
    for call in verify.call_args_list:
        assert len(call.args[2]) == 3
        assert call.args[1].constraints.correctness_suite is not None
        assert call.args[-1] is Level.L2_FUNCTION
    assert paired.validate_paired_report(report) is report


def test_retained_source_preserves_crlf_and_unicode_bytes_through_each_arm(tmp_path, monkeypatch):
    source = '# 保留原始字节\r\ndef kernel(X):\r\n    return X * 0\r\n'
    verify = verifier(monkeypatch, [True, True])
    report = paired.run_paired_plan(seeded(source), tmp_path/'run', responses=replies(None, None),
        backend_factory=lambda _: object())
    assert [call.args[0] for call in verify.call_args_list] == [source, source]
    assert (tmp_path/'run'/'shared'/CASE/'candidate.py').read_bytes() == source.encode('utf-8')
    for arm in ('baseline', 'diagnosis'):
        assert (tmp_path/'run'/'arms'/arm/CASE/'candidate-1.py').read_bytes() == source.encode('utf-8')
    assert report['pairs'][0]['initial_source_hash'] == code_hash(source)
    assert report['pairs'][0]['source_hashes_match'] is True


def test_entire_declared_call_budget_remains_available_to_each_seeded_arm(tmp_path, monkeypatch):
    second_bad = 'def kernel(X):\n    return X + 1\n'
    verifier(monkeypatch, [False, False, True, False, False, True])
    plan = paired.make_seeded_paired_plan(initial_sources={CASE: BAD}, case_ids=[CASE],
        budget=q.QualityBudget(max_calls=2, max_verifications=3))
    responses = {CASE: {arm: ['```python\n'+source+'\n```' for source in (second_bad, GOOD)]
                         for arm in ('baseline', 'diagnosis')}}
    report = paired.run_paired_plan(plan, tmp_path/'run', responses=responses,
        backend_factory=lambda _: object())
    assert plan['pairing']['generation_calls'] == 0
    assert plan['pairing']['repair_calls_per_arm'] == 2
    assert report['total_actual_costs']['model_calls']['known_total'] == 4
    assert report['shared']['costs']['model_calls']['known_total'] == 0
    for arm in ('baseline', 'diagnosis'):
        assert report['arms'][arm]['costs']['model_calls']['known_total'] == 2
        assert report['arms'][arm]['metrics']['final_full_l2']['count'] == 1


def test_seeded_arms_recreate_actual_strided_inputs_after_other_arm_mutates_them(tmp_path, monkeypatch):
    expected = quality_suite.evaluation_inputs_for(STRIDED)
    observed = []
    def inspect(code, task, inputs, *args, **kwargs):
        assert len(inputs) == 3
        for actual_set, expected_set in zip(inputs, expected):
            for key, value in actual_set.items():
                assert np.array_equal(value, expected_set[key])
                assert value.strides == expected_set[key].strides
                assert not value.flags.c_contiguous
        observed.append(inputs)
        for actual_set in inputs:
            for value in actual_set.values():
                value[...] = 0
        return result(True)
    monkeypatch.setattr(q, 'Verifier', lambda _: SimpleNamespace(verify=inspect))
    source = 'def kernel(A,B): return A @ B'
    plan = paired.make_seeded_paired_plan(initial_sources={STRIDED: source}, case_ids=[STRIDED],
        budget=q.QualityBudget(max_calls=1))
    report = paired.run_paired_plan(plan, tmp_path/'run', responses={STRIDED: {}},
        backend_factory=lambda _: object())
    assert len(observed) == 2
    for key in observed[0][0]:
        assert not np.shares_memory(observed[0][0][key], observed[1][0][key])
    assert report['total_actual_costs']['model_calls']['known_total'] == 0
    assert report['metrics']['eligible_failed_pairs'] == 0


def test_preflight_failure_never_constructs_a_generation_or_repair_client(tmp_path, monkeypatch):
    monkeypatch.setenv('LLM_MODEL', 'synthetic-model')
    monkeypatch.setenv('LLM_BASE_URL', 'https://model.invalid/v1')
    plan = seeded(backend='ssh', host='synthetic-host', model='synthetic-model',
                  endpoint='https://model.invalid/v1')
    seen = []
    def backend_factory(case_id):
        seen.append('backend')
        return object()
    def preflight(backend, root):
        seen.append('preflight')
        raise q.LLMCallRefused('synthetic failed control kernel')
    def forbidden(*args, **kwargs):
        pytest.fail('failed preflight must prevent every model construction')
    monkeypatch.setattr(q, '_preflight', preflight)
    report = paired.run_paired_plan(plan, tmp_path/'run', host='synthetic-host',
        client_factory=forbidden, backend_factory=backend_factory)
    assert seen == ['backend', 'preflight']
    assert report['shared']['tasks'][0]['status'] == 'preflight_failed'
    assert report['total_actual_costs']['model_calls']['known_total'] == 0
    assert report['metrics']['eligible_failed_pairs'] == 0
    for arm in ('baseline', 'diagnosis'):
        assert report['arms'][arm]['missing_tasks'] == 0
        assert report['arms'][arm]['metrics']['final_full_l2'] == dict(count=0, denominator=1, rate=0.)


@pytest.mark.parametrize('observations,expected', [([True, True], True), ([None, None], None)])
def test_passed_or_unknown_seeds_are_reverified_without_inventing_repair_opportunities(
        tmp_path, monkeypatch, observations, expected):
    verify = verifier(monkeypatch, observations)
    report = paired.run_paired_plan(seeded(), tmp_path/'run', responses=replies(None, None),
        backend_factory=lambda _: object())
    assert verify.call_count == 2
    assert report['pairs'][0]['initial_full_l2'] == {'baseline': expected, 'diagnosis': expected}
    assert report['pairs'][0]['repair_opportunity'] is False
    assert report['metrics']['eligible_failed_pairs'] == 0
    assert report['metrics']['baseline_repairs'] == report['metrics']['diagnosis_repairs'] == 0
    assert report['total_actual_costs']['model_calls']['known_total'] == 0
    for arm in ('baseline', 'diagnosis'):
        assert report['arms'][arm]['attempts'][0]['full_l2_passed'] is expected
        assert report['arms'][arm]['metrics']['native_performance_target']['rate'] is None


def test_conflicting_fresh_observations_exclude_a_diagnosis_only_repair_claim(tmp_path, monkeypatch):
    verifier(monkeypatch, [True, False, True])
    report = paired.run_paired_plan(seeded(), tmp_path/'run', responses=replies(),
        backend_factory=lambda _: object())
    assert report['pairs'][0]['initial_observation_conflict'] is True
    assert report['metrics']['initial_observation_conflicts'] == 1
    assert report['metrics']['eligible_failed_pairs'] == 0
    assert report['metrics']['diagnosis_only_repairs'] == 0
    assert report['arms']['diagnosis']['metrics']['final_full_l2']['count'] == 1
    assert report['total_actual_costs']['model_calls']['known_total'] == 1


def test_missing_baseline_usage_is_not_hidden_by_zero_shared_generation(tmp_path, monkeypatch):
    verifier(monkeypatch, [False, False, True])
    report = paired.run_paired_plan(seeded(), tmp_path/'run', responses=replies(None, GOOD),
        backend_factory=lambda _: object())
    assert report['shared']['costs']['tokens'] == dict(known_total=0, missing_records=0)
    assert report['total_actual_costs']['model_calls']['known_total'] == 2
    assert report['arms']['baseline']['costs']['tokens']['missing_records'] >= 1
    assert report['total_actual_costs']['tokens']['missing_records'] >= 1
    assert report['metrics']['diagnosis_only_repairs'] == 1


@pytest.mark.parametrize('change', ['schema', 'seed_binding', 'shared_status', 'shared_calls',
                                   'shared_tokens', 'arm_calls', 'pair_binding', 'metrics'])
def test_seeded_report_rejects_relabeling_and_tampered_evidence(tmp_path, monkeypatch, change):
    verifier(monkeypatch, [True, True])
    report = paired.run_paired_plan(seeded(), tmp_path/'run', responses=replies(None, None),
        backend_factory=lambda _: object())
    changed = copy.deepcopy(report)
    if change == 'schema':
        changed['schema'] = 'operator-paired-quality-report-v1'
    elif change == 'seed_binding':
        changed['plan']['initial_sources'][CASE] = {'source': GOOD, 'sha256': code_hash(GOOD)}
        rehash(changed['plan'])
        changed['plan_hash'] = changed['plan']['hash']
    elif change == 'shared_status':
        changed['shared']['tasks'][0]['status'] = 'generated'
    elif change == 'shared_calls':
        changed['shared']['tasks'][0]['model_calls'] = 1
    elif change == 'shared_tokens':
        changed['shared']['tasks'][0]['tokens'] = 15
    elif change == 'arm_calls':
        changed['arms']['baseline']['costs']['model_calls']['known_total'] = 1
    elif change == 'pair_binding':
        changed['pairs'][0]['initial_source_hash'] = '0'*64
    else:
        changed['metrics']['eligible_failed_pairs'] = 1
    with pytest.raises(ValueError):
        paired.validate_paired_report(changed)


def test_actual_cpu_seeded_repairs_rerun_full_l2_and_do_not_overwrite_evidence(tmp_path):
    plan = seeded()
    report = paired.run_paired_plan(plan, tmp_path/'run', responses=replies())
    assert report['metrics']['eligible_failed_pairs'] == 1
    assert report['metrics']['baseline_repairs'] == report['metrics']['diagnosis_repairs'] == 1
    assert report['total_actual_costs']['model_calls']['known_total'] == 2
    for arm in ('baseline', 'diagnosis'):
        evidence = json.loads((tmp_path/'run'/'arms'/arm/CASE/'verification-2.json').read_text())
        l2 = next(row for row in evidence['results'] if row['level'] == 'L2')
        assert l2['metrics']['num_input_sets'] == 3
        assert l2['metrics']['correctness']['verdict'] == 'passed'
    assert paired.validate_paired_report(report) is report
    with pytest.raises(FileExistsError):
        paired.run_paired_plan(plan, tmp_path/'run', responses=replies())


def test_report_builder_rejects_an_arm_initial_source_outside_the_seed(tmp_path, monkeypatch):
    verifier(monkeypatch, [True, True])
    report = paired.run_paired_plan(seeded(), tmp_path/'run', responses=replies(None, None),
        backend_factory=lambda _: object())
    records = {arm: copy.deepcopy(report['arms'][arm]['attempts'])
               for arm in ('baseline', 'diagnosis')}
    records['diagnosis'][0]['source_hash'] = code_hash(GOOD)
    with pytest.raises(ValueError, match='seeded arm initial source drift'):
        paired._build_report(report['plan'], report['shared']['tasks'], records)


def test_report_builder_rejects_old_passed_results_after_seed_initialization_failed(tmp_path, monkeypatch):
    verifier(monkeypatch, [True, True])
    report = paired.run_paired_plan(seeded(), tmp_path/'run', responses=replies(None, None),
        backend_factory=lambda _: object())
    shared = copy.deepcopy(report['shared']['tasks'])
    shared[0].update(status='initialization_failed', source_hash=None)
    records = {arm: copy.deepcopy(report['arms'][arm]['attempts'])
               for arm in ('baseline', 'diagnosis')}
    for rows in records.values():
        rows[0]['source_hash'] = None
    with pytest.raises(ValueError, match='seeded shared failure cannot inherit arm results'):
        paired._build_report(report['plan'], shared, records)
