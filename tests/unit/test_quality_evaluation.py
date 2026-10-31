"""Quality plans and replay accounting; no external model or device calls."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from operator_agent.decide import quality_evaluation as q
from operator_agent.decide import quality_suite
from operator_agent.execute.llm_budget import LLMCallBudget, LLMCallLimits, LLMCallRefused
from operator_agent.core.schemas import VerificationLevel as Level, VerificationResult
from operator_agent.core.run_contract import digest


CASE = 'layernorm-tail'
GOOD = ('import numpy as np\ndef kernel(X):\n'
        '    x = X.astype(np.float32)\n'
        '    z = x - x.mean(axis=-1, keepdims=True)\n'
        '    return (z / np.sqrt((z*z).mean(axis=-1, keepdims=True) + 1e-5)).astype(X.dtype)\n')
BAD = 'def kernel(X):\n    return X * 0\n'


def envelope(code):
    return json.dumps(dict(code=code, rationale='Synthetic replay fixture.'))


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
    verifier = SimpleNamespace(verify=Mock(side_effect=outcomes))
    monkeypatch.setattr(q, 'Verifier', lambda backend: verifier)
    return verifier


def test_default_planning_has_no_model_device_or_materialization(monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail('plan must not execute external work or materialize inputs')
    monkeypatch.setattr(q, 'LLMClient', forbidden)
    monkeypatch.setattr(quality_suite, 'inputs_for', forbidden)
    monkeypatch.setattr(quality_suite, 'evaluation_inputs_for', forbidden)
    plan = q.make_plan(model='requested-model')
    assert len(plan['manifest']['cases']) == 10
    assert q.validate_plan(plan) is plan
    assert 'api_key' not in json.dumps(plan).lower()
    assert plan['controls']['scope'] == 'full_L2_only_no_performance_or_deployment_claim'


def test_generation_payload_uses_shared_rules_actual_inputs_and_hides_acceptance_seeds():
    task = quality_suite.task_for(CASE, hardware='mock_gpu', dsl='python')
    messages = q.generation_messages(CASE, task)
    assert q.GENERATION_SPEC in messages[0]['content']
    payload = json.loads(messages[1]['content'])
    assert payload['calling_convention']['parameter_names'] == ['X']
    assert payload['inputs'][0]['X']['shape'] == [7, 33]
    assert 'seed' not in json.dumps(payload).replace('independent seeds are withheld', '')
    assert 'illustrative_values' not in payload
    assert payload['semantics']['layernorm_epsilon'] == 1e-5


@pytest.mark.parametrize('field,value', [('max_calls', 0), ('max_calls', True),
                                      ('call_seconds', float('nan'))])
def test_invalid_budgets_fail_before_dispatch(field, value):
    with pytest.raises(ValueError):
        q.QualityBudget(**{field: value})


def test_plan_and_source_drift_fail_before_creating_run(tmp_path, monkeypatch):
    plan = q.make_plan(case_ids=[CASE])
    plan['controls']['budget']['max_calls'] += 1
    with pytest.raises(ValueError, match='hash drift'):
        q.run_plan(plan, tmp_path/'run', responses={})
    assert not (tmp_path/'run').exists()
    plan = q.make_plan(case_ids=[CASE])
    monkeypatch.setattr(q, 'evaluator_identity', lambda: 'changed')
    with pytest.raises(ValueError, match='source drift'):
        q.run_plan(plan, tmp_path/'run', responses={})
    assert not (tmp_path/'run').exists()


def test_real_execution_requires_frozen_ssh_and_model_identity(tmp_path, monkeypatch):
    mock_plan = q.make_plan(case_ids=[CASE])
    with pytest.raises(ValueError, match='frozen SSH'):
        q.run_plan(mock_plan, tmp_path/'run')
    plan = q.make_plan(backend='ssh', host='synthetic-host', model='synthetic-model',
                       endpoint='https://model.invalid/v1', case_ids=[CASE])
    monkeypatch.setenv('LLM_MODEL', 'changed-model')
    with pytest.raises(ValueError, match='model configuration'):
        q.run_plan(plan, tmp_path/'run', host='synthetic-host')
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('change', ['scope', 'thinking', 'hardware'])
def test_rehashed_unsupported_controls_cannot_claim_another_execution_scope(tmp_path, change):
    plan = q.make_plan(case_ids=[CASE])
    if change == 'scope':
        plan['controls']['scope'] = 'production_and_performance'
    elif change == 'thinking':
        plan['controls']['model']['enable_thinking'] = True
    else:
        plan['controls']['hardware']['real_device_required'] = True
    plan['hash'] = digest({k: v for k, v in plan.items() if k != 'hash'})
    with pytest.raises(ValueError, match='control drift'):
        q.run_plan(plan, tmp_path/'run', responses={})
    assert not (tmp_path/'run').exists()


def test_nonfinite_failure_evidence_remains_explicit_json(tmp_path, monkeypatch):
    outcomes = failed()
    outcomes[1].metrics['max_error'] = float('inf')
    fake_verification(monkeypatch, [outcomes])
    plan = q.make_plan(case_ids=[CASE], budget=q.QualityBudget(max_calls=1, max_verifications=1))
    report = q.run_plan(plan, tmp_path/'run', responses={CASE: [envelope(BAD)]},
                        backend_factory=lambda _: object())
    evidence = json.loads((tmp_path/'run'/CASE/'verification-1.json').read_text())
    assert evidence['results'][1]['metrics']['max_error'] == {'$nonfinite': '+inf'}
    assert report['metrics']['within_budget_full_l2']['count'] == 0


def test_replay_pass_verifies_all_holdout_inputs_and_never_earns_native_credit(tmp_path, monkeypatch):
    verifier = fake_verification(monkeypatch, [passed()])
    plan = q.make_plan(case_ids=[CASE])
    report = q.run_plan(plan, tmp_path/'run', responses={CASE: [envelope(GOOD)]},
                        backend_factory=lambda _: object())
    args = verifier.verify.call_args.args
    assert len(args[2]) == 3
    assert args[1].constraints.correctness_suite is not None
    assert report['metrics']['first_full_l2']['count'] == 1
    assert report['metrics']['native_performance_target']['count'] == 0
    assert report['costs']['model_calls']['known_total'] == 1
    assert report['attempts'][0]['execution_kind'] == 'replay'
    assert (tmp_path/'run'/CASE/'candidate-1.py').read_text() == GOOD
    with pytest.raises(FileExistsError):
        q.run_plan(plan, tmp_path/'run', responses={})


def test_only_l1_does_not_become_full_correctness(tmp_path, monkeypatch):
    fake_verification(monkeypatch, [[VerificationResult(Level.L1_COMPILE, True)]])
    report = q.run_plan(q.make_plan(case_ids=[CASE]), tmp_path/'run',
                        responses={CASE: [envelope(GOOD)]}, backend_factory=lambda _: object())
    assert report['metrics']['within_budget_full_l2']['count'] == 0


def test_generation_errors_still_spend_calls_and_stay_in_task_denominator(tmp_path):
    report = q.run_plan(q.make_plan(case_ids=[CASE]), tmp_path/'run',
                        responses={CASE: ['not JSON']})
    assert report['metrics']['first_full_l2'] == dict(count=0, denominator=1, rate=0.)
    assert report['attempts'][0]['status'] == 'generation_failed'
    assert report['costs']['model_calls']['known_total'] == 1
    assert (tmp_path/'run'/CASE/'failure.json').is_file()


def test_reserved_call_without_usage_keeps_unknown_tokens_instead_of_zero(tmp_path):
    report = q.run_plan(q.make_plan(case_ids=[CASE]), tmp_path/'run', responses={CASE: []})
    assert report['costs']['model_calls']['known_total'] == 1
    assert report['attempts'][0]['tokens'] is None
    assert report['costs']['tokens']['missing_records'] == 1
    assert report['costs']['tokens']['per_correct_task'] is None
    assert (tmp_path/'run'/CASE/'usage.json').exists()


def test_noop_repair_keeps_failed_task_and_accounts_for_second_call(tmp_path, monkeypatch):
    verifier = fake_verification(monkeypatch, [failed()])
    report = q.run_plan(q.make_plan(case_ids=[CASE]), tmp_path/'run',
                        responses={CASE: [envelope(BAD), '```python\n'+BAD+'```']},
                        backend_factory=lambda _: object())
    assert verifier.verify.call_count == 1
    assert report['costs']['model_calls']['known_total'] == 2
    assert report['metrics']['final_full_l2']['count'] == 0
    # Fixer's extraction strips the final newline, so the text differs while
    # the executable AST is unchanged. Both forms must stop without a recheck.
    assert report['attempts'][0]['repair_action'] == 'repair_no_structural_change'
    assert not (tmp_path/'run'/CASE/'candidate-2.py').exists()


def test_baseline_and_diagnosis_are_comparable_with_same_budget(tmp_path, monkeypatch):
    reports = []
    for arm in ('baseline', 'diagnosis'):
        fake_verification(monkeypatch, [passed()])
        reports.append(q.run_plan(q.make_plan(strategy=arm, case_ids=[CASE]), tmp_path/arm,
                                 responses={CASE: [envelope(GOOD)]}, backend_factory=lambda _: object()))
    comparison = quality_suite.compare_reports(*reports)
    assert comparison['varied_controls'] == {'strategy': {'left': 'baseline', 'right': 'diagnosis'}}
    tampered = copy.deepcopy(reports[1])
    tampered['controls']['budget']['max_calls'] = 99
    with pytest.raises(ValueError):
        quality_suite.compare_reports(reports[0], tampered)


def test_replay_honors_hard_call_limit_instead_of_requesting_another_response(tmp_path):
    limits = LLMCallLimits(max_calls=1, max_output_tokens=4096,
                          total_output_token_reservations=4096, allowed_callers=('QualityGenerate',))
    client = q.ReplayClient(['first', 'second'], LLMCallBudget(limits, tmp_path/'journal'))
    assert client.chat([{'role':'user','content':'x'}], max_tokens=4096,
                       caller_tag='QualityGenerate') == 'first'
    with pytest.raises(LLMCallRefused):
        client.chat([{'role':'user','content':'x'}], max_tokens=4096,
                    caller_tag='QualityGenerate')
    assert client.call_budget.summary()['reserved_calls'] == 1


def test_real_mock_verifier_runs_complete_profile_and_independent_inputs(tmp_path):
    # This proves CPU replay plumbing, not a model's authored kernel or native performance.
    report = q.run_plan(q.make_plan(case_ids=[CASE]), tmp_path/'run',
                        responses={CASE: [envelope(GOOD)]})
    assert report['metrics']['first_full_l2']['count'] == 1
    assert report['metrics']['native_performance_target']['count'] == 0
    evidence = json.loads((tmp_path/'run'/CASE/'verification-1.json').read_text())
    l2 = next(r for r in evidence['results'] if r['level'] == 'L2')
    assert l2['metrics']['num_input_sets'] == 3
    assert l2['metrics']['correctness']['verdict'] == 'passed'


def test_real_cpu_replay_repairs_failure_then_rechecks_full_profile_with_both_call_costs(tmp_path):
    report = q.run_plan(q.make_plan(case_ids=[CASE]), tmp_path/'run',
        responses={CASE: [envelope(BAD), '```python\n'+GOOD+'\n```']})
    assert report['metrics']['first_full_l2']['count'] == 0
    assert report['metrics']['final_full_l2']['count'] == 1
    assert report['costs']['model_calls']['known_total'] == 2
    assert len(report['attempts']) == 2
    assert report['attempts'][0]['repair_action'] == 'fixed'
    assert report['attempts'][0]['diagnosis_state'] == 'unknown'
    assert (tmp_path/'run'/CASE/'candidate-2.py').exists()
