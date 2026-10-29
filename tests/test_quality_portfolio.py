"""A fixed three-candidate portfolio is an explicit experiment, not retries.

The first slot is the nested single-generation baseline. Every slot receives
the full trusted L2 gate, with independent ledgers and fresh evaluation inputs.
Offline fixtures test accounting and selection rather than model/GPU quality.
"""
import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from operator_agent import quality_evaluation as q
from operator_agent import quality_portfolio as portfolio
from operator_agent import quality_suite as suite
from operator_agent.run_contract import code_hash, digest
from operator_agent.schemas import VerificationLevel as Level, VerificationResult


CASE = 'layernorm-tail'
STRIDED = 'matmul-strided-fp32'
GOOD = ('import numpy as np\ndef kernel(X):\n'
        '    z = X.astype(np.float32)\n'
        '    z = z - z.mean(axis=-1, keepdims=True)\n'
        '    return (z / np.sqrt((z*z).mean(axis=-1, keepdims=True)+1e-5)).astype(X.dtype)\n')
BAD = 'def kernel(X):\n    return X * 0\n'
THIRD = GOOD + '\n# Independently authored third fixture.\n'


def envelope(source):
    return json.dumps(dict(code=source, rationale='Authored offline portfolio control.'))


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


def unknown():
    return [VerificationResult(Level.L1_COMPILE, True),
            VerificationResult(Level.L2_FUNCTION, False, 'unqualified observation', metrics={
                'correctness_primary_verdict': 'unknown',
                'correctness': {'verdict': 'unknown'}})]


def fake_verification(monkeypatch, outcomes):
    verify = Mock(side_effect=outcomes)
    monkeypatch.setattr(q, 'Verifier', lambda _: SimpleNamespace(verify=verify))
    return verify


def test_portfolio_planning_never_constructs_models_or_materializes_inputs(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('planning must not construct models, devices or concrete inputs')
    monkeypatch.setattr(q, 'LLMClient', forbidden)
    monkeypatch.setattr(suite, 'evaluation_inputs_for', forbidden)
    monkeypatch.setattr(suite, 'inputs_for', forbidden)
    plan = portfolio.make_portfolio_plan(case_ids=[CASE])
    assert plan['schema'] == 'operator-portfolio-quality-plan-v1'
    assert plan['base_plan']['controls']['strategy'] == 'baseline'
    assert plan['base_plan']['controls']['budget']['max_calls'] == 1
    assert plan['base_plan']['controls']['budget']['max_verifications'] == 1
    assert plan['portfolio'] == dict(
        slots=[dict(slot=1, strategy='layout_first'),
               dict(slot=2, strategy='numerics_first'),
               dict(slot=3, strategy='tiled')],
        generation_calls_per_slot=1, verifications_per_slot=1, repair_calls=0,
        selection='earliest_full_l2_pass', baseline_slot=1,
        cost_scope='all_slots_plus_readiness_once')
    assert plan['hash'] == digest({k: v for k, v in plan.items() if k != 'hash'})
    assert portfolio.validate_portfolio_plan(plan) is plan


def test_portfolio_rejects_user_supplied_repair_strategy():
    with pytest.raises(ValueError):
        portfolio.make_portfolio_plan(case_ids=[CASE], strategy='diagnosis')


@pytest.mark.parametrize('case_ids', [[], [CASE, CASE], ['undeclared-case']])
def test_portfolio_keeps_the_frozen_nonempty_unique_task_domain(case_ids):
    with pytest.raises(ValueError):
        portfolio.make_portfolio_plan(case_ids=case_ids)


@pytest.mark.parametrize('budget', [q.QualityBudget(), q.QualityBudget(max_calls=3),
                                   q.QualityBudget(max_calls=1, max_verifications=2)])
def test_portfolio_requires_one_generation_and_one_verification_per_slot(budget):
    with pytest.raises(ValueError):
        portfolio.make_portfolio_plan(case_ids=[CASE], budget=budget)


def test_plan_hash_drift_rejected_before_constructing_a_run_directory(tmp_path):
    plan = portfolio.make_portfolio_plan(case_ids=[CASE])
    plan['hash'] = '0' * 64
    with pytest.raises(ValueError):
        portfolio.run_portfolio_plan(plan, tmp_path/'run', responses={})
    assert not (tmp_path/'run').exists()


def test_model_or_hardware_drift_rejected_before_run_directory(tmp_path, monkeypatch):
    monkeypatch.setenv('LLM_MODEL', 'other-model')
    monkeypatch.setenv('LLM_BASE_URL', 'https://model.invalid/v1')
    plan = portfolio.make_portfolio_plan(case_ids=[CASE], backend='ssh', host='frozen-host',
        model='synthetic-model', endpoint='https://model.invalid/v1')
    with pytest.raises(ValueError):
        portfolio.run_portfolio_plan(plan, tmp_path/'run', host='frozen-host')
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('change', ['unexpected_key', 'schema', 'base_strategy',
                                   'source_identity', 'manifest'])
def test_rehashed_control_drift_is_not_accepted(tmp_path, change):
    plan = portfolio.make_portfolio_plan(case_ids=[CASE])
    if change == 'unexpected_key':
        plan['implicit_retries'] = 9
    elif change == 'schema':
        plan['schema'] = 'operator-portfolio-quality-plan-v999'
    else:
        base = plan['base_plan']
        if change == 'base_strategy':
            base['controls']['strategy'] = 'diagnosis'
        elif change == 'source_identity':
            base['controls']['evaluator'] = '0' * 64
        else:
            base['manifest']['cases'][0]['evaluation_inputs'][0]['seed'] = 123456
            base['manifest']['hash'] = digest({k: v for k, v in base['manifest'].items() if k != 'hash'})
        base['hash'] = digest({k: v for k, v in base.items() if k != 'hash'})
    plan['hash'] = digest({k: v for k, v in plan.items() if k != 'hash'})
    with pytest.raises(ValueError):
        portfolio.run_portfolio_plan(plan, tmp_path/'run', responses={})
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('field,value', [('generation_calls_per_slot', 2),
    ('verifications_per_slot', 2), ('repair_calls', 1), ('baseline_slot', 2),
    ('selection', 'smallest_error'), ('cost_scope', 'selected_slot_only')])
def test_rehashed_portfolio_protocol_cannot_enable_retries_or_different_selection(tmp_path, field, value):
    plan = portfolio.make_portfolio_plan(case_ids=[CASE])
    plan['portfolio'][field] = value
    plan['hash'] = digest({k: v for k, v in plan.items() if k != 'hash'})
    with pytest.raises(ValueError):
        portfolio.run_portfolio_plan(plan, tmp_path/'run', responses={})
    assert not (tmp_path/'run').exists()


def test_every_slot_is_generated_and_verified_even_when_baseline_already_passes(tmp_path, monkeypatch):
    verification = fake_verification(monkeypatch, [passed(), passed(), passed()])
    made = {}
    codes = [GOOD, BAD, THIRD]
    def client_factory(case_id, slot, ledger):
        made[slot] = ledger
        return q.ReplayClient([envelope(codes[slot-1])], ledger)
    backend_factory = Mock(side_effect=lambda _: object())
    report = portfolio.run_portfolio_plan(portfolio.make_portfolio_plan(case_ids=[CASE]),
        tmp_path/'run', responses={}, client_factory=client_factory, backend_factory=backend_factory)
    assert verification.call_count == 3
    assert backend_factory.call_count == 1
    assert [call.args[0] for call in verification.call_args_list] == codes
    assert sorted(made) == [1, 2, 3]
    assert len({id(ledger) for ledger in made.values()}) == 3
    for ledger in made.values():
        assert ledger.summary()['reserved_calls'] == 1
        assert ledger.limits.max_calls == 1
        assert ledger.limits.allowed_callers == ('QualityGenerate',)
    for call in verification.call_args_list:
        assert len(call.args[2]) == 3
        assert call.args[1].constraints.correctness_suite is not None
        assert call.kwargs['stop_after'] is Level.L2_FUNCTION
    assert report['tasks'][0]['selected_slot'] == 1
    assert report['tasks'][0]['selected_source_hash'] == code_hash(GOOD)
    assert report['costs']['model_calls']['known_total'] == 3
    assert report['performance'] == dict(state='not_measured', rate=None)
    assert report['metrics']['baseline_full_l2']['count'] == 1
    assert report['metrics']['portfolio_full_l2']['count'] == 1
    for slot in (1, 2, 3):
        root = tmp_path/'run'/'cases'/CASE/f'slot-{slot}'
        assert (root/'candidate-1.py').read_text() == codes[slot-1]
        assert not (root/'candidate-2.py').exists()


def test_first_full_l2_pass_rescues_failed_baseline_without_using_later_candidates(tmp_path, monkeypatch):
    fake_verification(monkeypatch, [failed(), passed(), passed()])
    report = portfolio.run_portfolio_plan(portfolio.make_portfolio_plan(case_ids=[CASE]),
        tmp_path/'run', responses={CASE: [envelope(BAD), envelope(GOOD), envelope(THIRD)]},
        backend_factory=lambda _: object())
    task = report['tasks'][0]
    assert task['baseline_full_l2'] is False
    assert task['portfolio_full_l2'] is True
    assert task['selected_slot'] == 2
    assert task['selected_source_hash'] == code_hash(GOOD)
    assert report['metrics']['rescued_failed_baselines'] == 1
    assert report['metrics']['resolved_unknown_baselines'] == 0
    assert report['costs']['model_calls']['known_total'] == 3


@pytest.mark.parametrize('outcomes,expected,unknown_count', [
    ([failed(), failed(), failed()], False, 0),
    ([failed(), unknown(), failed()], None, 1),
    ([unknown(), unknown(), unknown()], None, 1)])
def test_no_pass_keeps_explicit_failed_or_unknown_task_and_no_selected_candidate(
        tmp_path, monkeypatch, outcomes, expected, unknown_count):
    fake_verification(monkeypatch, outcomes)
    report = portfolio.run_portfolio_plan(portfolio.make_portfolio_plan(case_ids=[CASE]),
        tmp_path/'run', responses={CASE: [envelope(BAD)]*3}, backend_factory=lambda _: object())
    task = report['tasks'][0]
    assert task['portfolio_full_l2'] is expected
    assert task['selected_slot'] is None
    assert task['selected_source_hash'] is None
    assert report['metrics']['portfolio_full_l2'] == dict(count=0, denominator=1, rate=0.,
                                                        unknown_tasks=unknown_count)
    assert len(report['candidates']) == 3
    assert report['costs']['model_calls']['known_total'] == 3


def test_unknown_baseline_resolved_by_later_pass_is_not_claimed_as_failed_repair(tmp_path, monkeypatch):
    fake_verification(monkeypatch, [unknown(), passed(), failed()])
    report = portfolio.run_portfolio_plan(portfolio.make_portfolio_plan(case_ids=[CASE]),
        tmp_path/'run', responses={CASE: [envelope(BAD), envelope(GOOD), envelope(THIRD)]},
        backend_factory=lambda _: object())
    assert report['tasks'][0]['baseline_full_l2'] is None
    assert report['tasks'][0]['selected_slot'] == 2
    assert report['metrics']['rescued_failed_baselines'] == 0
    assert report['metrics']['resolved_unknown_baselines'] == 1


def test_invalid_or_missing_generation_never_blocks_other_independent_slots(tmp_path, monkeypatch):
    verification = fake_verification(monkeypatch, [passed()])
    report = portfolio.run_portfolio_plan(portfolio.make_portfolio_plan(case_ids=[CASE]),
        tmp_path/'run', responses={CASE: ['not JSON', envelope(GOOD)]},
        backend_factory=lambda _: object())
    assert verification.call_count == 1
    assert [row['slot'] for row in report['candidates']] == [1, 2, 3]
    assert [row['full_l2_passed'] for row in report['candidates']] == [None, True, None]
    assert report['tasks'][0]['selected_slot'] == 2
    assert report['metrics']['baseline_full_l2']['unknown_tasks'] == 1
    assert report['costs']['model_calls']['known_total'] == 3
    assert report['costs']['tokens']['missing_records'] >= 1


def test_failure_in_verifier_does_not_spend_a_hidden_regeneration(tmp_path, monkeypatch):
    verification = fake_verification(monkeypatch, [RuntimeError('synthetic verification error'),
                                                  passed(), failed()])
    report = portfolio.run_portfolio_plan(portfolio.make_portfolio_plan(case_ids=[CASE]),
        tmp_path/'run', responses={CASE: [envelope(BAD), envelope(GOOD), envelope(THIRD)]},
        backend_factory=lambda _: object())
    assert verification.call_count == 3
    assert report['candidates'][0]['full_l2_passed'] is None
    assert report['tasks'][0]['selected_slot'] == 2
    assert report['costs']['model_calls']['known_total'] == 3


def test_identical_sources_are_not_counted_as_candidate_diversity(tmp_path, monkeypatch):
    fake_verification(monkeypatch, [passed(), passed(), passed()])
    report = portfolio.run_portfolio_plan(portfolio.make_portfolio_plan(case_ids=[CASE]),
        tmp_path/'run', responses={CASE: [envelope(GOOD)]*3}, backend_factory=lambda _: object())
    assert report['tasks'][0]['distinct_sources'] == 1
    assert report['tasks'][0]['distinct_asts'] == 1
    assert report['tasks'][0]['selected_slot'] == 1
    assert report['costs']['model_calls']['known_total'] == 3


def test_comment_only_variants_increase_sources_but_not_distinct_asts(tmp_path, monkeypatch):
    fake_verification(monkeypatch, [passed(), passed(), passed()])
    sources = [GOOD, GOOD+'\n# Variant 2\n', GOOD+'\n# Variant 3\n']
    report = portfolio.run_portfolio_plan(portfolio.make_portfolio_plan(case_ids=[CASE]),
        tmp_path/'run', responses={CASE: list(map(envelope, sources))}, backend_factory=lambda _: object())
    assert report['tasks'][0]['distinct_sources'] == 3
    assert report['tasks'][0]['distinct_asts'] == 1


def test_all_frozen_tasks_remain_in_denominator_when_every_generation_is_missing(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('missing generation must never execute a candidate')
    monkeypatch.setattr(q, 'Verifier', forbidden)
    plan = portfolio.make_portfolio_plan()
    report = portfolio.run_portfolio_plan(plan, tmp_path/'run', responses={},
        backend_factory=lambda _: object())
    assert len(report['tasks']) == 10
    assert len(report['readiness']) == 10
    assert len(report['candidates']) == 30
    assert report['metrics']['baseline_full_l2'] == dict(count=0, denominator=10, rate=0., unknown_tasks=10)
    assert report['metrics']['portfolio_full_l2'] == dict(count=0, denominator=10, rate=0., unknown_tasks=10)
    assert report['costs']['model_calls']['known_total'] == 30
    assert report['costs']['tokens']['missing_records'] == 30


def test_readiness_and_slot_elapsed_time_are_counted_once_each(tmp_path, monkeypatch):
    clock = [1000.]
    monkeypatch.setattr(q.time, 'monotonic', lambda: clock[0])
    def verification(*args, **kwargs):
        clock[0] += 1.
        return passed()
    monkeypatch.setattr(q, 'Verifier', lambda _: SimpleNamespace(verify=verification))
    class TimedReplay(q.ReplayClient):
        def chat(self, *args, **kwargs):
            clock[0] += 10.
            return super().chat(*args, **kwargs)
    def factory(case_id, slot, ledger):
        return TimedReplay([envelope(GOOD)], ledger)
    def backend_factory(case_id):
        clock[0] += 5.
        return object()
    report = portfolio.run_portfolio_plan(portfolio.make_portfolio_plan(case_ids=[CASE]),
        tmp_path/'run', responses={}, client_factory=factory, backend_factory=backend_factory)
    assert [row['wall_seconds'] for row in report['candidates']] == [11.]*3
    assert report['readiness'][0]['wall_seconds'] == 5.
    assert report['costs']['wall_seconds']['known_total'] == 38.


def test_each_slot_gets_fresh_original_noncontiguous_inputs_and_one_task_backend(tmp_path, monkeypatch):
    expected = suite.evaluation_inputs_for(STRIDED)
    observed, backends = [], []
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
    def verifier(backend):
        backends.append(backend)
        return SimpleNamespace(verify=inspect_and_mutate)
    monkeypatch.setattr(q, 'Verifier', verifier)
    backend = object()
    factory = Mock(return_value=backend)
    report = portfolio.run_portfolio_plan(portfolio.make_portfolio_plan(case_ids=[STRIDED]),
        tmp_path/'run', responses={STRIDED: [envelope('def kernel(A,B): return A @ B')]*3},
        backend_factory=factory)
    assert factory.call_count == 1
    assert backends == [backend]*3
    assert len(observed) == 3
    for previous in range(3):
        for later in range(previous+1, 3):
            for name in observed[previous][0]:
                assert not np.shares_memory(observed[previous][0][name], observed[later][0][name])
    assert report['metrics']['portfolio_full_l2']['count'] == 1


def test_real_preflight_failure_preserves_all_slots_with_no_model_calls(tmp_path, monkeypatch):
    monkeypatch.setenv('LLM_MODEL', 'synthetic-model')
    monkeypatch.setenv('LLM_BASE_URL', 'https://model.invalid/v1')
    plan = portfolio.make_portfolio_plan(case_ids=[CASE], backend='ssh', host='synthetic-host',
        model='synthetic-model', endpoint='https://model.invalid/v1')
    seen = []
    def backend_factory(case_id):
        seen.append('backend')
        return object()
    def preflight(backend, root):
        seen.append('preflight')
        raise q.LLMCallRefused('synthetic failed control kernel')
    def forbidden(*args, **kwargs):
        pytest.fail('failed preflight must stop before model or candidate construction')
    monkeypatch.setattr(q, '_preflight', preflight)
    monkeypatch.setattr(q, 'Verifier', forbidden)
    report = portfolio.run_portfolio_plan(plan, tmp_path/'run', host='synthetic-host',
        client_factory=forbidden, backend_factory=backend_factory)
    assert seen == ['backend', 'preflight']
    assert len(report['readiness']) == 1
    assert len(report['candidates']) == 3
    assert all(row['status'] == 'preflight_failed' for row in report['candidates'])
    assert report['costs']['model_calls']['known_total'] == 0
    assert report['metrics']['portfolio_full_l2']['denominator'] == 1
    assert report['metrics']['portfolio_full_l2']['count'] == 0


def test_slot_generation_requests_have_distinct_fixed_strategy_without_prior_source(tmp_path, monkeypatch):
    fake_verification(monkeypatch, [passed(), passed(), passed()])
    captured = []
    class InspectingReplay(q.ReplayClient):
        def chat(self, messages, **kwargs):
            captured.append(copy.deepcopy(messages))
            return super().chat(messages, **kwargs)
    def factory(case_id, slot, ledger):
        return InspectingReplay([envelope([BAD, GOOD, THIRD][slot-1])], ledger)
    portfolio.run_portfolio_plan(portfolio.make_portfolio_plan(case_ids=[CASE]),
        tmp_path/'run', responses={}, client_factory=factory, backend_factory=lambda _: object())
    assert len(captured) == 3
    assert len({json.dumps(messages, sort_keys=True) for messages in captured}) == 3
    for messages in captured:
        combined = json.dumps(messages)
        assert 'return X * 0' not in combined
        assert 'z = X.astype' not in combined
        assert 'Independently authored third fixture' not in combined
        assert q.GENERATION_SPEC in messages[0]['content']


@pytest.mark.parametrize('field', ['metrics', 'costs', 'selection', 'source_count',
    'input_binding', 'strategy', 'source_hash', 'ast_hash', 'missing_slot', 'duplicate_slot',
    'device_identity', 'execution_kind'])
def test_report_recomputes_aggregates_bindings_and_complete_slot_history(tmp_path, monkeypatch, field):
    fake_verification(monkeypatch, [failed(), passed(), passed()])
    report = portfolio.run_portfolio_plan(portfolio.make_portfolio_plan(case_ids=[CASE]),
        tmp_path/'run', responses={CASE: [envelope(BAD), envelope(GOOD), envelope(THIRD)]},
        backend_factory=lambda _: object())
    assert portfolio.validate_portfolio_report(report) is report
    changed = copy.deepcopy(report)
    if field == 'metrics':
        changed['metrics']['rescued_failed_baselines'] = 99
    elif field == 'costs':
        changed['costs']['model_calls']['known_total'] = 1
    elif field == 'selection':
        changed['tasks'][0]['selected_slot'] = 3
    elif field == 'source_count':
        changed['tasks'][0]['distinct_sources'] = 99
    elif field == 'input_binding':
        changed['candidates'][0]['input_manifest_hash'] = '0'*64
    elif field == 'strategy':
        changed['candidates'][0]['strategy'] = 'implicit_retry'
    elif field in ('source_hash', 'ast_hash'):
        changed['candidates'][0][field] = 'invalid hash'
    elif field == 'missing_slot':
        changed['candidates'].pop()
    elif field == 'duplicate_slot':
        changed['candidates'][1]['slot'] = 1
    elif field == 'device_identity':
        changed['readiness'][0]['identity'] = dict(device='a different device')
    else:
        changed['candidates'][0]['execution_kind'] = 'real_device'
    with pytest.raises(ValueError):
        portfolio.validate_portfolio_report(changed)


@pytest.mark.parametrize('field,value', [('slot', True), ('slot', 1.0),
    ('model_calls', -1), ('model_calls', True), ('tokens', -1),
    ('wall_seconds', -1.), ('wall_seconds', float('nan')),
    ('full_l2_passed', 'passed')])
def test_invalid_retained_candidate_facts_cannot_be_normalized_into_valid_evidence(
        tmp_path, monkeypatch, field, value):
    fake_verification(monkeypatch, [failed(), passed(), passed()])
    report = portfolio.run_portfolio_plan(portfolio.make_portfolio_plan(case_ids=[CASE]),
        tmp_path/'run', responses={CASE: [envelope(BAD), envelope(GOOD), envelope(THIRD)]},
        backend_factory=lambda _: object())
    changed = copy.deepcopy(report)
    changed['candidates'][0][field] = value
    with pytest.raises(ValueError):
        portfolio.validate_portfolio_report(changed)


def test_real_cpu_replay_uses_full_profile_and_never_overwrites_existing_evidence(tmp_path):
    plan = portfolio.make_portfolio_plan(case_ids=[CASE])
    report = portfolio.run_portfolio_plan(plan, tmp_path/'run',
        responses={CASE: [envelope(BAD), envelope(GOOD), envelope(THIRD)]})
    assert report['tasks'][0]['selected_slot'] == 2
    assert report['metrics']['rescued_failed_baselines'] == 1
    assert report['costs']['model_calls']['known_total'] == 3
    assert portfolio.validate_portfolio_report(report) is report
    for slot in (1, 2, 3):
        evidence = json.loads((tmp_path/'run'/'cases'/CASE/f'slot-{slot}'/'verification-1.json').read_text())
        l2 = next(row for row in evidence['results'] if row['level'] == 'L2')
        if slot != 1:
            assert l2['metrics']['num_input_sets'] == 3
        assert l2['metrics']['correctness']['verdict'] == ('failed' if slot == 1 else 'passed')
    with pytest.raises(FileExistsError):
        portfolio.run_portfolio_plan(plan, tmp_path/'run', responses={CASE: [envelope(GOOD)]*3})


@pytest.mark.parametrize('change', ['device_identity', 'execution_kind', 'free_source', 'readiness_mode'])
def test_report_rejects_identity_execution_and_call_binding_drift(tmp_path, monkeypatch, change):
    fake_verification(monkeypatch, [passed()]*3)
    report = portfolio.run_portfolio_plan(portfolio.make_portfolio_plan(case_ids=[CASE]),
        tmp_path/'run', responses={CASE: [envelope(GOOD)]*3}, backend_factory=lambda _: object())
    changed = copy.deepcopy(report)
    if change == 'device_identity':
        changed['readiness'][0]['identity']['uuid'] = 'fabricated'
    elif change == 'execution_kind':
        changed['candidates'][0]['execution_kind'] = 'real_device'
    elif change == 'free_source':
        changed['candidates'][0]['model_calls'] = 0
    else:
        changed['readiness'][0]['status'] = 'passed'
    with pytest.raises(ValueError):
        portfolio.validate_portfolio_report(changed)


@pytest.mark.parametrize('responses', [{CASE: ['x']*4}, {'undeclared': []}, {CASE: 'x'}])
def test_replay_rejects_hidden_slots_or_undeclared_tasks_before_execution(tmp_path, responses):
    with pytest.raises(ValueError):
        portfolio.run_portfolio_plan(portfolio.make_portfolio_plan(case_ids=[CASE]),
                                     tmp_path/'run', responses=responses)
    assert not (tmp_path/'run').exists()


def test_subset_merge_requires_disjoint_complete_coverage_and_identical_controls(tmp_path, monkeypatch):
    fake_verification(monkeypatch, [passed()]*9)
    other = 'softmax-tail'
    master = portfolio.make_portfolio_plan(case_ids=[CASE, other])
    left = portfolio.run_portfolio_plan(portfolio.make_portfolio_plan(case_ids=[CASE]),
        tmp_path/'left', responses={CASE: [envelope(GOOD)]*3}, backend_factory=lambda _: object())
    right = portfolio.run_portfolio_plan(portfolio.make_portfolio_plan(case_ids=[other]),
        tmp_path/'right', responses={other: [envelope(GOOD)]*3}, backend_factory=lambda _: object())
    merged = portfolio.merge_portfolio_reports(master, [right, left])
    assert [row['case_id'] for row in merged['tasks']] == [CASE, other]
    assert merged['costs']['model_calls']['known_total'] == 6
    assert merged['metrics']['portfolio_full_l2']['denominator'] == 2
    for invalid in ([left], [left, left, right]):
        with pytest.raises(ValueError):
            portfolio.merge_portfolio_reports(master, invalid)
    with pytest.raises(ValueError):
        portfolio.merge_portfolio_reports(portfolio.make_portfolio_plan(case_ids=[CASE]), [left, right])
    changed_budget = q.QualityBudget(max_calls=1, max_verifications=1, max_output_tokens=2048)
    incompatible = portfolio.run_portfolio_plan(portfolio.make_portfolio_plan(case_ids=[CASE], budget=changed_budget),
        tmp_path/'different', responses={CASE: [envelope(GOOD)]*3}, backend_factory=lambda _: object())
    with pytest.raises(ValueError):
        portfolio.merge_portfolio_reports(master, [incompatible, right])


def test_precheck_rejected_paid_source_is_retained_and_counted(tmp_path, monkeypatch):
    from operator_agent.single_candidate_coder import PrecheckRefused
    monkeypatch.setenv('LLM_MODEL', 'synthetic-model')
    monkeypatch.setenv('LLM_BASE_URL', 'https://model.invalid/v1')
    monkeypatch.setattr(q, '_preflight', lambda *args: dict(identity=dict(device='cuda', uuid='fake-offline')))
    monkeypatch.setattr(q, 'precheck', Mock(side_effect=PrecheckRefused('jit_helper_missing')))
    plan = portfolio.make_portfolio_plan(case_ids=[CASE], backend='ssh', host='synthetic-host',
        model='synthetic-model', endpoint='https://model.invalid/v1')
    report = portfolio.run_portfolio_plan(plan, tmp_path/'run', host='synthetic-host',
        client_factory=lambda case, slot, ledger: q.ReplayClient([envelope(GOOD)], ledger),
        backend_factory=lambda _: object())
    assert report['tasks'][0]['distinct_sources'] == 1
    assert report['costs']['model_calls']['known_total'] == 3
    assert report['metrics']['portfolio_full_l2']['unknown_tasks'] == 1
    for slot in (1, 2, 3):
        root = tmp_path/'run'/'cases'/CASE/f'slot-{slot}'
        assert (root/'candidate-1.py').read_text() == GOOD
        assert json.loads((root/'precheck-failure.json').read_text())['reason'] == 'jit_helper_missing'
        assert not (root/'verification-1.json').exists()


@pytest.mark.parametrize('responses', [
    {CASE: [envelope(GOOD)]*4},
    {'undeclared-case': []},
    {CASE: 'not-a-slot-list'},
    {CASE: dict(slot_1=envelope(GOOD))},
    {CASE: (envelope(GOOD),)*3},
    {CASE: [envelope(GOOD), None]},
    []])
def test_replay_cannot_add_implicit_fourth_slots_or_undeclared_task_data(tmp_path, responses):
    plan = portfolio.make_portfolio_plan(case_ids=[CASE])
    with pytest.raises(ValueError):
        portfolio.run_portfolio_plan(plan, tmp_path/'run', responses=responses,
            backend_factory=lambda _: object())
    assert not (tmp_path/'run').exists()


def test_merge_disjoint_case_reports_restores_master_order_and_counts_all_costs_once(tmp_path, monkeypatch):
    fake_verification(monkeypatch, [failed(), passed(), failed(), passed(), failed(), unknown()])
    master = portfolio.make_portfolio_plan(case_ids=[CASE, STRIDED])
    reports = []
    for index, case_id in enumerate((CASE, STRIDED)):
        report = portfolio.run_portfolio_plan(portfolio.make_portfolio_plan(case_ids=[case_id]),
            tmp_path/f'part-{index}', responses={case_id: [envelope(GOOD)]*3},
            backend_factory=lambda _: object())
        reports.append(report)
    merged = portfolio.merge_portfolio_reports(master, list(reversed(reports)))
    assert portfolio.validate_portfolio_report(merged) is merged
    assert merged['plan'] == master
    assert [row['case_id'] for row in merged['tasks']] == [CASE, STRIDED]
    assert [row['case_id'] for row in merged['readiness']] == [CASE, STRIDED]
    assert [row['case_id'] for row in merged['candidates']] == [CASE]*3 + [STRIDED]*3
    assert [row['selected_slot'] for row in merged['tasks']] == [2, 1]
    assert merged['metrics']['baseline_full_l2']['count'] == 1
    assert merged['metrics']['portfolio_full_l2']['count'] == 2
    assert merged['metrics']['portfolio_full_l2']['denominator'] == 2
    assert merged['costs']['model_calls']['known_total'] == 6
    assert len(merged['readiness']) == 2
    assert merged['costs']['wall_seconds']['known_total'] == pytest.approx(
        sum(report['costs']['wall_seconds']['known_total'] for report in reports))


@pytest.mark.parametrize('change', ['overlap', 'missing', 'undeclared', 'model', 'budget'])
def test_merge_rejects_incomplete_overlapping_or_differently_controlled_parts(tmp_path, monkeypatch, change):
    fake_verification(monkeypatch, [passed()]*6)
    master = portfolio.make_portfolio_plan(case_ids=[CASE, STRIDED])
    first = portfolio.run_portfolio_plan(portfolio.make_portfolio_plan(case_ids=[CASE]),
        tmp_path/'first', responses={CASE: [envelope(GOOD)]*3}, backend_factory=lambda _: object())
    options = dict(case_ids=[STRIDED])
    if change == 'model':
        options['model'] = 'another-model'
    elif change == 'budget':
        options['budget'] = q.QualityBudget(max_calls=1, max_verifications=1, max_output_tokens=1024)
    second = portfolio.run_portfolio_plan(portfolio.make_portfolio_plan(**options),
        tmp_path/'second', responses={STRIDED: [envelope(GOOD)]*3}, backend_factory=lambda _: object())
    reports = [first, second]
    if change == 'overlap':
        reports = [first, first, second]
    elif change == 'missing':
        reports = [first]
    elif change == 'undeclared':
        master = portfolio.make_portfolio_plan(case_ids=[CASE])
    with pytest.raises(ValueError):
        portfolio.merge_portfolio_reports(master, reports)

