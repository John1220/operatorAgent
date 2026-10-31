import json
import hashlib

import pytest

from operator_agent.core import execution_audit
from operator_agent.core.contract import freeze_from_task
from operator_agent.app.hitl_tools import build_task
from operator_agent.execute.mock_gpu import MockGPU
from operator_agent.core.schemas import VerificationLevel
from operator_agent.execute.task_clock import monotonic
from operator_agent.decide.verifier import Verifier


def setup(root):
    task = build_task(dict(request='matmul fp32', shapes=[[8, 8, 8]],
                           dsl='python', min_speedup=0))
    inputs = freeze_from_task(task).cases[0].materialize()
    verifier = Verifier(MockGPU())
    execution_audit.install_level_recorder(verifier, root)
    return verifier, task, inputs


def events(root):
    return [json.loads(path.read_text()) for path in sorted((root/'phase-events').glob('*.json'))]


def test_full_l2_completion_is_durable_before_l3_can_fail(tmp_path, monkeypatch):
    verifier, task, inputs = setup(tmp_path)
    def l3_failure(*args, **kwargs):
        raise RuntimeError('controlled L3 failure after L2')
    monkeypatch.setattr(verifier, '_verify_performance', l3_failure)
    started = monotonic()
    with pytest.raises(RuntimeError, match='controlled L3'):
        verifier.verify('def kernel(A,B): return A@B', task, inputs)
    rows = events(tmp_path)
    assert {row['level'] for row in rows} == {'L1', 'L2'}
    l2 = next(row for row in rows if row['level'] == 'L2')
    assert l2['full_l2_passed'] is True
    assert started <= l2['timing']['completed_monotonic'] <= monotonic()
    assert l2['model_watermark'] is None
    assert l2['source_hash'] == hashlib.sha256(b'def kernel(A,B): return A@B').hexdigest()


def test_failed_l2_and_successful_l2_are_separate_immutable_events(tmp_path):
    verifier, task, inputs = setup(tmp_path)
    verifier.verify('def kernel(A,B): return A-B', task, inputs)
    old = {p: p.read_bytes() for p in (tmp_path/'phase-events').glob('*.json')}
    verifier.verify('def kernel(A,B): return A@B', task, inputs,
                    stop_after=VerificationLevel.L2_FUNCTION)
    l2 = [row for row in events(tmp_path) if row['level'] == 'L2']
    assert {row['full_l2_passed'] for row in l2} == {True, False}
    assert len({row['source_hash'] for row in l2}) == 2
    assert all(path.read_bytes() == body for path, body in old.items())


def test_missing_or_damaged_model_usage_stays_unknown(tmp_path):
    assert execution_audit.model_watermark(tmp_path) is None
    (tmp_path/'model-journal').mkdir()
    (tmp_path/'model-journal/summary.json').write_text('{broken')
    assert execution_audit.model_watermark(tmp_path) is None


def test_level_event_captures_only_a_consistent_api_usage_watermark(tmp_path):
    from operator_agent.execute.llm_budget import LLMCallLimits, ProductionLLMCallBudget
    from operator_agent.execute.task_clock import monotonic
    limits = LLMCallLimits(max_calls=1, max_output_tokens=16,
        total_output_token_reservations=16, call_seconds=5, campaign_seconds=60,
        allowed_callers=('Coder',))
    ledger = ProductionLLMCallBudget(limits, tmp_path/'model-journal')
    identity = dict(model='synthetic')
    ledger.bind(identity)
    call, _ = ledger.reserve(identity, json.dumps(dict(max_tokens=8)), 'Coder', 5)
    ledger.mark_dispatched(call)
    ledger.finish(call, response='synthetic', usage_source='api_usage',
                  usage=dict(prompt_tokens=7, completion_tokens=3, total_tokens=10))
    verifier, task, inputs = setup(tmp_path)
    verifier.verify('def kernel(A,B): return A@B', task, inputs,
                    stop_after=VerificationLevel.L2_FUNCTION)
    l2 = next(row for row in events(tmp_path) if row['level'] == 'L2')
    watermark = l2['model_watermark']
    assert watermark['reserved_calls'] == watermark['dispatched_calls'] == 1
    assert watermark['completed_calls'] == 1
    assert watermark['api_usage']['totals']['total_tokens'] == 10
    assert l2['timing']['completed_monotonic'] <= monotonic()
    (tmp_path/'model-journal/call-001.json').unlink()
    assert execution_audit.model_watermark(tmp_path) is None


def test_audit_write_failure_does_not_release_an_unaudited_success(tmp_path, monkeypatch):
    verifier, task, inputs = setup(tmp_path)
    def refuse(*args, **kwargs):
        raise OSError('controlled audit failure')
    monkeypatch.setattr(execution_audit, 'publish_json', refuse)
    with pytest.raises(OSError, match='controlled audit'):
        verifier.verify('def kernel(A,B): return A@B', task, inputs)


def test_library_verification_without_sink_has_no_implicit_phase_credit(tmp_path):
    task = build_task(dict(request='matmul fp32', shapes=[[8, 8, 8]], dsl='python'))
    verifier = Verifier(MockGPU())
    result = verifier.verify('def kernel(A,B): return A@B', task,
        freeze_from_task(task).cases[0].materialize(), stop_after=VerificationLevel.L1_COMPILE)
    assert 'execution_timing' not in result[0].metrics
    assert not (tmp_path/'phase-events').exists()
