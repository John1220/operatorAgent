"""Production reservations use local transport doubles, never real model calls."""
import concurrent.futures
import json
from pathlib import Path
import threading

import pytest

from operator_agent.execute import llm_budget as lb, llm_client as lc


def limits(**overrides):
    values = dict(max_calls=2, max_output_tokens=64,
        total_output_token_reservations=128, call_seconds=60.,
        campaign_seconds=300., allowed_callers=('Designer', 'Coder', 'Fixer'))
    values.update(overrides)
    return lb.LLMCallLimits(**values)


def client(tmp_path, *, config=None, deadline=None):
    budget = lb.ProductionLLMCallBudget(config or limits(), tmp_path/'journal', deadline=deadline)
    model = lc.LLMClient(base_url='https://model.invalid/v1', api_key='SYNTHETIC-KEY',
        model='fixture-model', timeout=60, max_tokens=64, call_budget=budget,
        fallback_model='must-not-be-used')
    return model, budget


def answer(content='def kernel(X):\n    return X', **overrides):
    value = dict(choices=[dict(finish_reason='stop', message=dict(content=content))],
                 usage=dict(prompt_tokens=4, completion_tokens=8, total_tokens=12))
    value.update(overrides)
    return json.dumps(value)


def messages():
    return [dict(role='user', content='PRIVATE-PROMPT: import torch # refused source')]


def test_explicit_retry_uses_remaining_credit_without_hidden_fallback(tmp_path, monkeypatch):
    model, budget = client(tmp_path)
    dispatches = []
    def dispatch(request, timeout):
        dispatches.append(request)
        if len(dispatches) == 1:
            raise TimeoutError('private provider failure')
        return answer()
    monkeypatch.setattr(lc, 'bounded_dispatch', dispatch)
    with pytest.raises(lb.LLMCallRefused):
        model.chat(messages(), caller_tag='Coder')
    assert model.chat(messages(), caller_tag='Fixer').startswith('def kernel')
    with pytest.raises(lb.ProductionBudgetExhausted) as caught:
        model.chat(messages(), caller_tag='Coder')
    assert caught.value.reason == 'llm_call_budget'
    summary = budget.summary()
    assert len(dispatches) == summary['reserved_calls'] == 2
    assert [r['status'] for r in summary['attempts']] == ['failed', 'answered']
    assert summary['aborted'] is None and summary['remaining_calls'] == 0
    assert summary['output_tokens_reserved'] == 128
    assert summary['stop_reason'] == 'llm_call_budget'
    assert not summary['fallback'] and summary['explicit_retries']
    assert json.loads((budget.root/'refusal.json').read_text())['reason'] == 'llm_call_budget'
    assert all(request['url'] == 'https://model.invalid/v1/chat/completions' for request in dispatches)


def test_exhausted_credit_is_not_a_stop_until_an_actual_refusal(tmp_path, monkeypatch):
    model, budget = client(tmp_path, config=limits(max_calls=1))
    monkeypatch.setattr(lc, 'bounded_dispatch', lambda *a: answer())
    model.chat(messages(), caller_tag='Coder')
    assert budget.summary()['remaining_calls'] == 0
    assert budget.summary()['stop_reason'] is None
    assert not (budget.root/'refusal.json').exists()


def test_invalid_json_is_attributed_and_next_explicit_call_is_allowed(tmp_path, monkeypatch):
    model, budget = client(tmp_path)
    replies = iter([answer('no JSON here'), answer('```json\n{"algorithm":"relu"}\n```')])
    monkeypatch.setattr(lc, 'bounded_dispatch', lambda *a: next(replies))
    with pytest.raises(lb.LLMCallRefused, match='reservation retained'):
        model.chat_json(messages(), caller_tag='Designer')
    assert model.chat_json(messages(), caller_tag='Designer') == dict(algorithm='relu')
    assert [r['status'] for r in budget.summary()['attempts']] == ['invalid_json', 'answered']
    assert json.loads((budget.root/'call-001.json').read_text())['status'] == 'invalid_json'
    assert budget.summary()['aborted'] is None


def test_request_and_response_source_and_credentials_are_not_persisted(tmp_path, monkeypatch):
    model, budget = client(tmp_path)
    monkeypatch.setattr(lc, 'bounded_dispatch', lambda *a: answer('PRIVATE-RESPONSE: import torch'))
    assert model.chat(messages(), caller_tag='Coder').startswith('PRIVATE-RESPONSE')
    stored = '\n'.join(path.read_text() for path in budget.root.glob('*.json'))
    for secret in ('PRIVATE-PROMPT', 'PRIVATE-RESPONSE', 'import torch', 'SYNTHETIC-KEY', 'https://model.invalid'):
        assert secret not in stored
    record = budget.summary()['attempts'][0]
    assert 'messages' not in record['request'] and 'response_text' not in record
    assert record['request']['model'] == 'fixture-model'
    assert len(record['request_sha256']) == len(record['response_sha256']) == 64
    assert record['request_bytes'] > 0 and record['response_bytes'] > 0


def test_non_object_json_is_recorded_as_invalid_output(tmp_path, monkeypatch):
    model, budget = client(tmp_path)
    monkeypatch.setattr(lc, 'bounded_dispatch', lambda *a: answer('```json\n[1,2]\n```'))
    with pytest.raises(lb.LLMCallRefused, match='reservation retained'):
        model.chat_json(messages(), caller_tag='Designer')
    assert budget.summary()['attempts'][0]['status'] == 'invalid_json'
    assert budget.summary()['aborted'] is None


def test_timeout_is_capped_by_shared_remaining_deadline(tmp_path, monkeypatch):
    now = [100.]
    monkeypatch.setattr(lb, 'monotonic', lambda: now[0])
    model, budget = client(tmp_path, deadline=103.)
    timeouts = []
    def dispatch(request, timeout):
        timeouts.append((request['timeout'], timeout))
        now[0] += .5
        return answer()
    monkeypatch.setattr(lc, 'bounded_dispatch', dispatch)
    model.chat(messages(), caller_tag='Coder')
    model.chat(messages(), caller_tag='Fixer')
    assert timeouts == [(3., 3.), (2.5, 2.5)]
    assert budget.remaining_seconds() == 2.


def test_deadline_before_dispatch_has_stable_evidence_and_no_reservation(tmp_path, monkeypatch):
    now = [100.]
    monkeypatch.setattr(lb, 'monotonic', lambda: now[0])
    model, budget = client(tmp_path, deadline=101.)
    now[0] = 101.
    monkeypatch.setattr(lc, 'bounded_dispatch', lambda *a: pytest.fail('expired task dispatched'))
    with pytest.raises(lb.ProductionBudgetExhausted) as caught:
        model.chat(messages(), caller_tag='Coder')
    assert caught.value.reason == 'task_wall_clock'
    assert budget.summary()['reserved_calls'] == 0
    assert json.loads((budget.root/'refusal.json').read_text())['before_dispatch'] is True


def test_expired_during_request_keeps_failed_call_and_stable_stop(tmp_path, monkeypatch):
    now = [100.]
    monkeypatch.setattr(lb, 'monotonic', lambda: now[0])
    model, budget = client(tmp_path, deadline=101.)
    def delayed(*args):
        now[0] = 102.
        raise TimeoutError('blocked transport')
    monkeypatch.setattr(lc, 'bounded_dispatch', delayed)
    with pytest.raises(lb.ProductionBudgetExhausted) as caught:
        model.chat(messages(), caller_tag='Coder')
    assert caught.value.reason == 'task_wall_clock'
    assert budget.summary()['attempts'][0]['status'] == 'failed'
    assert budget.summary()['reserved_calls'] == 1
    assert json.loads((budget.root/'refusal.json').read_text())['before_dispatch'] is False


def test_total_output_reservations_have_independent_limit(tmp_path, monkeypatch):
    model, budget = client(tmp_path, config=limits(max_calls=5, total_output_token_reservations=64))
    monkeypatch.setattr(lc, 'bounded_dispatch', lambda *a: answer())
    model.chat(messages(), caller_tag='Coder')
    with pytest.raises(lb.ProductionBudgetExhausted) as caught:
        model.chat(messages(), caller_tag='Fixer')
    assert caught.value.reason == 'llm_output_budget'
    assert budget.summary()['reserved_calls'] == 1


def test_journal_drift_still_poisoned_and_never_dispatched(tmp_path, monkeypatch):
    model, budget = client(tmp_path)
    path = budget.root/'limits.json'
    path.chmod(0o600)
    path.write_text('{}')
    monkeypatch.setattr(lc, 'bounded_dispatch', lambda *a: pytest.fail('drift dispatched'))
    with pytest.raises(lb.LLMCallRefused, match='journal drift'):
        model.chat(messages(), caller_tag='Coder')
    assert budget.summary()['aborted'] == 'journal_drift'


def test_result_journal_failure_still_poisoned(tmp_path, monkeypatch):
    model, budget = client(tmp_path)
    monkeypatch.setattr(lc, 'bounded_dispatch', lambda *a: answer())
    write = lb.atomic_json
    def failed_write(path, *a, **kw):
        if Path(path).name == 'call-001.json':
            raise OSError('disk full')
        return write(path, *a, **kw)
    monkeypatch.setattr(lb, 'atomic_json', failed_write)
    with pytest.raises(lb.LLMCallRefused, match='result journal failed'):
        model.chat(messages(), caller_tag='Coder')
    assert budget.summary()['aborted'] == 'journal_failure'
    with pytest.raises(lb.LLMCallRefused, match='campaign stopped'):
        model.chat(messages(), caller_tag='Fixer')


@pytest.mark.parametrize('reported_model', [None, 'provider-alias'])
def test_production_keeps_legacy_reported_model_semantics(tmp_path, monkeypatch, reported_model):
    model, budget = client(tmp_path)
    extra = {} if reported_model is None else dict(model=reported_model)
    monkeypatch.setattr(lc, 'bounded_dispatch', lambda *a: answer(**extra))
    assert model.chat(messages(), caller_tag='Coder').startswith('def kernel')
    assert model.cost_tracker.summary()['n_reported_model_absent'] == (reported_model is None)


def test_truncated_source_is_returned_for_existing_precheck_retry_policy(tmp_path, monkeypatch):
    model, budget = client(tmp_path)
    truncated = answer(choices=[dict(finish_reason='length', message=dict(content='def kernel(X):'))])
    monkeypatch.setattr(lc, 'bounded_dispatch', lambda *a: truncated)
    assert model.chat(messages(), caller_tag='Coder') == 'def kernel(X):'
    assert budget.summary()['attempts'][0]['status'] == 'answered'


def test_parallel_agents_cannot_overspend_the_same_task(tmp_path):
    budget = lb.ProductionLLMCallBudget(limits(), tmp_path/'journal')
    identity = dict(model='fixture')
    budget.bind(identity)
    body = json.dumps(dict(max_tokens=64, messages=messages()))
    def attempt(index):
        try:
            call, _ = budget.reserve(identity, body, 'Coder', 60.)
        except lb.ProductionBudgetExhausted:
            return 'refused'
        budget.finish(call, outcome='failed' if index == 0 else 'answered')
        return 'reserved'
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(attempt, range(4)))
    assert results.count('reserved') == 2 and results.count('refused') == 2
    assert budget.summary()['reserved_calls'] == 2
    assert budget.summary()['output_tokens_reserved'] == 128


def test_parallel_json_failure_does_not_relabel_another_threads_call(tmp_path):
    budget = lb.ProductionLLMCallBudget(limits(), tmp_path/'journal')
    identity = dict(model='fixture')
    budget.bind(identity)
    barrier = threading.Barrier(2)
    body = json.dumps(dict(max_tokens=64))
    def attempt(bad):
        call, _ = budget.reserve(identity, body, 'Designer', 60.)
        budget.finish(call)
        barrier.wait(5)
        if bad:
            budget.abort('invalid_json')
        return call, bad
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        calls = list(pool.map(attempt, [True, False]))
    records = budget.summary()['attempts']
    for call, bad in calls:
        assert records[call-1]['status'] == ('invalid_json' if bad else 'answered')


def test_strict_single_candidate_default_still_poisons_failure(tmp_path):
    budget = lb.LLMCallBudget(limits(), tmp_path/'journal')
    identity = dict(model='fixture')
    budget.bind(identity)
    call, _ = budget.reserve(identity, json.dumps(dict(max_tokens=64)), 'Coder', 60.)
    budget.finish(call, outcome='failed', response='STRICT-RETAINED-RESPONSE')
    with pytest.raises(lb.LLMCallRefused, match='campaign stopped'):
        budget.reserve(identity, json.dumps(dict(max_tokens=64)), 'Fixer', 60.)
    assert budget.summary()['attempts'][0]['response_text'] == 'STRICT-RETAINED-RESPONSE'
    assert budget.strict_response and budget.poison_dispatched_failure


def test_summary_tracks_bind_reserve_and_finish_before_and_after_transport(tmp_path, monkeypatch):
    model, budget = client(tmp_path)
    path = budget.root/'summary.json'
    bound = json.loads(path.read_text())
    assert bound['reserved_calls'] == 0 and bound['attempts'] == []
    assert bound['identity'] == budget.summary()['identity']
    assert bound['limits'] == json.loads((budget.root/'limits.json').read_text())
    def dispatched(*args):
        reserved = json.loads(path.read_text())
        assert reserved['reserved_calls'] == 1
        assert reserved['attempts'] == [json.loads((budget.root/'call-001.json').read_text())]
        assert reserved['attempts'][0]['status'] == 'reserved'
        return answer()
    monkeypatch.setattr(lc, 'bounded_dispatch', dispatched)
    model.chat(messages(), caller_tag='Coder')
    completed = json.loads(path.read_text())
    assert completed['attempts'][0]['status'] == 'answered'
    assert completed['attempts'] == [json.loads((budget.root/'call-001.json').read_text())]
    assert completed['aborted'] is None
    assert path.stat().st_mode & 0o777 == 0o444


def test_invalid_json_and_actual_refusal_are_durable_in_summary(tmp_path, monkeypatch):
    model, budget = client(tmp_path, config=limits(max_calls=1))
    monkeypatch.setattr(lc, 'bounded_dispatch', lambda *args: answer('invalid JSON'))
    with pytest.raises(lb.LLMCallRefused):
        model.chat_json(messages(), caller_tag='Designer')
    saved = json.loads((budget.root/'summary.json').read_text())
    assert saved['attempts'][0]['status'] == 'invalid_json'
    assert saved['stop_reason'] is None and saved['aborted'] is None
    with pytest.raises(lb.ProductionBudgetExhausted):
        model.chat(messages(), caller_tag='Coder')
    saved = json.loads((budget.root/'summary.json').read_text())
    assert saved['stop_reason'] == 'llm_call_budget' and saved['reserved_calls'] == 1


def test_summary_write_failure_blocks_dispatch_without_refunding_reservation(tmp_path, monkeypatch):
    model, budget = client(tmp_path)
    write = lb.atomic_json
    def failed_summary(path, *args, **kwargs):
        if Path(path).name == 'summary.json':
            raise OSError('summary unavailable')
        return write(path, *args, **kwargs)
    monkeypatch.setattr(lb, 'atomic_json', failed_summary)
    monkeypatch.setattr(lc, 'bounded_dispatch', lambda *args: pytest.fail('summary failure dispatched'))
    with pytest.raises(lb.LLMCallRefused, match='summary journal failed'):
        model.chat(messages(), caller_tag='Coder')
    assert budget.summary()['reserved_calls'] == 1
    assert budget.summary()['aborted'] == 'journal_failure'
    assert not (budget.root/'summary.json').exists()
    assert json.loads((budget.root/'abort.json').read_text())['reason'] == 'journal_failure'
    assert json.loads((budget.root/'call-001.json').read_text())['status'] == 'reserved'


def test_summary_failure_after_answer_does_not_leave_stale_complete_counters(tmp_path, monkeypatch):
    model, budget = client(tmp_path)
    write = lb.atomic_json
    answered = [False]
    def dispatch(*args):
        answered[0] = True
        return answer()
    def failed_summary(path, *args, **kwargs):
        if answered[0] and Path(path).name == 'summary.json':
            raise OSError('summary unavailable')
        return write(path, *args, **kwargs)
    monkeypatch.setattr(lc, 'bounded_dispatch', dispatch)
    monkeypatch.setattr(lb, 'atomic_json', failed_summary)
    with pytest.raises(lb.LLMCallRefused, match='summary journal failed'):
        model.chat(messages(), caller_tag='Coder')
    assert budget.summary()['reserved_calls'] == 1 and budget.summary()['aborted'] == 'journal_failure'
    assert json.loads((budget.root/'call-001.json').read_text())['status'] == 'answered'
    assert not (budget.root/'summary.json').exists()


def test_missing_call_journal_persists_aborted_summary_instead_of_zero(tmp_path, monkeypatch):
    model, budget = client(tmp_path)
    monkeypatch.setattr(lc, 'bounded_dispatch', lambda *args: answer())
    model.chat(messages(), caller_tag='Coder')
    (budget.root/'call-001.json').unlink()
    monkeypatch.setattr(lc, 'bounded_dispatch', lambda *args: pytest.fail('drift dispatched'))
    with pytest.raises(lb.LLMCallRefused, match='journal drift'):
        model.chat(messages(), caller_tag='Fixer')
    saved = json.loads((budget.root/'summary.json').read_text())
    assert saved['aborted'] == 'journal_drift'
    assert saved['reserved_calls'] == 1 and len(saved['attempts']) == 1


def test_explicit_abort_is_published_in_summary(tmp_path):
    _, budget = client(tmp_path)
    budget.abort('controller_stop')
    assert json.loads((budget.root/'summary.json').read_text())['aborted'] == 'controller_stop'


def test_bind_summary_failure_is_closed_before_any_call(tmp_path, monkeypatch):
    write = lb.atomic_json
    def failed_summary(path, *args, **kwargs):
        if Path(path).name == 'summary.json':
            raise OSError('summary unavailable')
        return write(path, *args, **kwargs)
    monkeypatch.setattr(lb, 'atomic_json', failed_summary)
    with pytest.raises(lb.LLMCallRefused, match='summary journal failed'):
        client(tmp_path)
    root = tmp_path/'journal'
    assert not list(root.glob('call-*.json')) and not (root/'summary.json').exists()
    assert json.loads((root/'abort.json').read_text())['reason'] == 'journal_failure'
