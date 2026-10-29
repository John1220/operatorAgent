"""Offline vertical validator executes exact trusted fixtures only."""
from concurrent.futures import ThreadPoolExecutor
import pytest

from benchmarks.s5_vertical_validation import VerticalValidator, GOOD_FIXTURE, BAD_FIXTURE
from operator_agent.run_contract import code_hash


def test_good_fixture_keeps_source_contract_and_three_input_checks():
    validator = VerticalValidator()
    result = validator.validate(GOOD_FIXTURE)
    assert result['status'] == 'passed'
    assert result['candidate_sha256'] == code_hash(GOOD_FIXTURE)
    assert result['verification_attempts'] == result['max_verification_attempts'] == 1
    assert result['mock_only'] and not result['formal_s5_credit'] and not result['gpu_used']
    evidence = result['mock_evidence']
    assert evidence['draft_contract_hash'] == result['contract_hash']
    assert evidence['suite_hash'] == result['suite_hash']
    assert len(evidence['inputs']) == 3 and evidence['full_l2_passed'] is True


def test_known_bad_fixture_reports_failed_and_is_still_mock_only():
    result = VerticalValidator().validate(BAD_FIXTURE, arm='C')
    assert result['status'] == 'failed'
    assert result['mock_evidence']['full_l2_passed'] is False
    assert result['arm'] == 'C' and not result['arm_origin_verified']


def test_untrusted_source_never_reaches_evaluator_and_spends_budget(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('arbitrary model text must never be executed')
    monkeypatch.setattr('benchmarks.s5_l2_evaluator.evaluate_mock_fixture', forbidden)
    validator = VerticalValidator()
    result = validator.validate('import os\nos.system("echo unsafe")\n')
    assert result['status'] == 'unknown' and result['mock_evidence'] is None
    assert result['verification_attempts'] == 1 and not result['untrusted_source_executed']
    assert validator.validate(GOOD_FIXTURE)['status'] == 'budget_exhausted'


@pytest.mark.parametrize('value', [None, '', 'x' * 65537, '\ud800'])
def test_missing_invalid_or_large_candidate_is_unknown_and_spent(value):
    result = VerticalValidator().validate(value)
    assert result['status'] == 'unknown' and result['verification_attempts'] == 1
    assert result['mock_evidence'] is None


def test_evaluator_failure_remains_unknown_and_is_not_refunded(monkeypatch):
    def failure(*args, **kwargs): raise RuntimeError('do not disclose fake details')
    monkeypatch.setattr('benchmarks.s5_l2_evaluator.evaluate_mock_fixture', failure)
    validator = VerticalValidator()
    result = validator.validate(GOOD_FIXTURE)
    assert result['status'] == 'unknown' and result['reason'] == 'mock evaluator unavailable: RuntimeError'
    assert validator.validate(GOOD_FIXTURE)['status'] == 'budget_exhausted'


def test_parallel_attempts_cannot_exceed_shared_instance_limit(monkeypatch):
    seen = []
    def harmless(*args, **kwargs):
        seen.append(True)
        raise RuntimeError('fixture')
    monkeypatch.setattr('benchmarks.s5_l2_evaluator.evaluate_mock_fixture', harmless)
    validator = VerticalValidator(max_attempts=1)
    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(lambda _: validator.validate(GOOD_FIXTURE), range(8)))
    assert len(seen) == 1
    assert sum(x['status'] == 'budget_exhausted' for x in values) == 7


def test_identity_mismatch_never_promotes_a_mock_result(monkeypatch):
    monkeypatch.setattr('benchmarks.s5_l2_evaluator.evaluate_mock_fixture', lambda *a, **kw:
                        {'status':'passed', 'source_hash':'0'*64})
    result = VerticalValidator().validate(GOOD_FIXTURE)
    assert result['status'] == 'unknown' and result['mock_evidence'] is None


@pytest.mark.parametrize('limit', [0, 13, True, 1.5])
def test_invalid_budget_limits_are_rejected(limit):
    with pytest.raises(ValueError): VerticalValidator(max_attempts=limit)
