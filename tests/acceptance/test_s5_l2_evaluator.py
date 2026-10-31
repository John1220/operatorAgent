"""S5 mock-only terminal L2 evaluator, no model, SSH or GPU dispatch."""
import contextlib
import io

import pytest

from benchmarks.experiments.s5 import s5_l2_evaluator as evaluator
from operator_agent.decide import quality_suite
from operator_agent.core.run_contract import code_hash, digest

GOOD = ('import numpy as np\ndef kernel(X):\n'
        '    x = X.astype(np.float32)\n'
        '    z = x - x.mean(axis=-1, keepdims=True)\n'
        '    return (z / np.sqrt((z*z).mean(axis=-1, keepdims=True) + 1e-5)).astype(X.dtype)\n')
BAD = 'def kernel(X):\n    return X * 0\n'


def run(code, name='layernorm-tail', arm='A'):
    with contextlib.redirect_stdout(io.StringIO()):
        return evaluator.evaluate_mock_fixture(code, name, arm)


def test_mock_evaluates_original_primary_both_holdouts_and_complete_suite():
    result = run(GOOD)
    cases = quality_suite.input_cases_for('layernorm-tail')
    assert result['full_l2_passed'] is True and result['status'] == 'passed'
    assert result['mock_only'] is True and result['formal_s5_credit'] is False
    assert result['mock_contract_hash'] == result['draft_contract_hash']
    assert result['execution_target'] == dict(backend='mock', hardware='mock_gpu', dsl='python')
    assert [r['case_id'] for r in result['inputs']] == [c.case_id for c in cases]
    for row, case in zip(result['inputs'], cases):
        assert row['input_spec_hash'] == digest(case.content_dict())
        assert row['source_hash'] == code_hash(GOOD)
        assert row['contract_hash'] == result['mock_contract_hash']
        assert row['suite_hash'] == result['suite_hash']
        assert row['full_l2_passed'] is True
        l2 = next(r for r in row['verification'] if r['level'] == 'L2')
        assert l2['metrics']['correctness']['verdict'] == 'passed'
        assert len(l2['metrics']['correctness']['cases']) == 12


def test_real_failed_fixture_cannot_claim_complete_correctness():
    result = run(BAD)
    assert result['status'] == 'failed' and result['full_l2_passed'] is False
    assert len(result['inputs']) == 1
    assert result['inputs'][0]['full_l2_passed'] is False


def test_interface_mismatch_is_failed_but_transport_failure_is_unknown():
    from operator_agent.core.schemas import VerificationLevel, VerificationResult
    from operator_agent.decide.measurement import TRANSPORT_KEY, TRANSPORT_UNKNOWN
    result=run('def kernel(Y):\n    return Y\n')
    assert result['status']=='failed' and result['inputs'][0]['reason']=='verified local interface mismatch'
    l1=VerificationResult(VerificationLevel.L1_COMPILE,False,'transport failed',
                          metrics={TRANSPORT_KEY:TRANSPORT_UNKNOWN})
    assert evaluator._qualified([l1],None,None)[0] is None


def test_big_candidates_remain_unknown_before_any_materialization(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('large test candidate must not be allocated by mock prototype')
    monkeypatch.setattr(evaluator.CorrectnessCase, 'materialize', forbidden)
    for name in ('matmul-1024-fp16','matmul-tail-large-fp16',
                 'softmax-large-fp32','layernorm-large-fp32'):
        out=run(GOOD,name)
        assert out['status']=='unknown' and out['full_l2_passed'] is None
        assert not out['inputs']


@pytest.mark.parametrize('code', [
    'import triton\ndef kernel(X): return X\n',
    'import torch\ndef kernel(X): return torch.cuda.FloatTensor(X)\n',
    'import os\ndef kernel(X): return X\n',
    'def kernel(X): return __import__("torch").cuda.FloatTensor(X)\n',
])
def test_external_or_gpu_fixture_rejected_before_backend(code, monkeypatch):
    import operator_agent.execute.mock_gpu as mock
    monkeypatch.setattr(mock, 'MockGPU', lambda: pytest.fail('no backend must be made'))
    assert run(code)['status']=='unknown'


def test_missing_task_and_changed_manifest_are_unknown():
    assert run(GOOD,'not-a-task')['full_l2_passed'] is None
    assert run(GOOD,'layernorm-tail','D')['full_l2_passed'] is None
    result=evaluator.evaluate_mock_fixture(GOOD,'layernorm-tail','A',expected_manifest_hash='0'*64)
    assert result['status']=='unknown' and result['inputs']==[]


def test_one_missing_or_wrong_suite_case_cannot_become_pass(monkeypatch):
    from operator_agent.decide.verifier import Verifier
    original=Verifier.verify
    def drop_a_case(self,*a,**kw):
        results=original(self,*a,**kw)
        results[-1].metrics['correctness']['cases'].pop()
        return results
    monkeypatch.setattr(Verifier,'verify',drop_a_case)
    result=run(GOOD)
    assert result['status']=='unknown' and result['full_l2_passed'] is None
    assert result['inputs'][0]['reason']=='declared suite is missing or incomplete'


def test_missing_holdout_cannot_become_pass(monkeypatch):
    from operator_agent.decide.verifier import Verifier
    original=Verifier.verify
    count=[0]
    def break_second(self,*a,**kw):
        count[0]+=1
        if count[0]==2:
            return []
        return original(self,*a,**kw)
    monkeypatch.setattr(Verifier,'verify',break_second)
    result=run(GOOD,arm='C')
    assert result['status']=='unknown' and result['full_l2_passed'] is None
    assert len(result['inputs'])==2 and result['inputs'][0]['full_l2_passed'] is True
    assert result['inputs'][1]['full_l2_passed'] is None


@pytest.mark.parametrize('target', ['primary', 'suite', 'suite_repeat'])
def test_missing_fidelity_or_wrong_input_identity_is_unknown(monkeypatch, target):
    from operator_agent.decide.verifier import Verifier
    original=Verifier.verify
    def break_fidelity(self,*a,**kw):
        results=original(self,*a,**kw)
        metrics=results[-1].metrics
        if target == 'primary':
            metrics['input_fidelity_readings'][0]['input_fidelity_expected']['X']['shape']=[1,1]
        else:
            key='metrics' if target=='suite' else 'repeat_metrics'
            metrics['correctness']['cases'][0][key]['input_fidelity_verified']=None
        return results
    monkeypatch.setattr(Verifier,'verify',break_fidelity)
    result=run(GOOD)
    assert result['status']=='unknown' and result['full_l2_passed'] is None
