"""Production timing and API-usage evidence; all transports are local doubles."""
from datetime import datetime
import json
import urllib.error

import pytest

from operator_agent import llm_budget as lb, llm_client as lc


def _limits(**overrides):
    fields = dict(max_calls=4, max_output_tokens=32,
                  total_output_token_reservations=128, call_seconds=5.,
                  campaign_seconds=60., allowed_callers=('Coder', 'Designer', 'Fixer'))
    fields.update(overrides)
    return lb.LLMCallLimits(**fields)


def _client(tmp_path, *, strict=False):
    budget_type = lb.LLMCallBudget if strict else lb.ProductionLLMCallBudget
    budget = budget_type(_limits(), tmp_path / 'journal')
    model = lc.LLMClient(base_url='https://usage-fixture.invalid/v1',
        api_key='PRIVATE-USAGE-KEY', model='usage-fixture-model',
        timeout=5, max_tokens=32, call_budget=budget,
        fallback_model='must-not-be-used')
    return model, budget


def _reply(content='usable response', *, usage=True, **overrides):
    payload = dict(model='usage-fixture-model', choices=[dict(
        finish_reason='stop', message=dict(content=content))])
    if usage is True:
        payload['usage'] = dict(prompt_tokens=7, completion_tokens=3, total_tokens=10)
    elif usage is not None:
        payload['usage'] = usage
    payload.update(overrides)
    return json.dumps(payload)


def _messages():
    return [dict(role='user', content='PRIVATE-USAGE-PROMPT: candidate source')]


def _saved(budget, call=1):
    row = json.loads((budget.root / f'call-{call:03}.json').read_text())
    summary = json.loads((budget.root / 'summary.json').read_text())
    assert summary['attempts'][call - 1] == row
    expected_mode = (0o600 if row['status'] == 'reserved'
                     and row['dispatched_monotonic'] is None else 0o444)
    assert (budget.root / f'call-{call:03}.json').stat().st_mode & 0o777 == expected_mode
    return row, summary


def _assert_utc(value):
    assert value.endswith('Z')
    assert datetime.fromisoformat(value.replace('Z', '+00:00')).utcoffset().total_seconds() == 0


def test_answered_call_publishes_dispatch_and_completion_times_with_validated_usage(tmp_path, monkeypatch):
    now = [100.]
    utc = ['2026-10-27T00:00:00.000000Z']
    monkeypatch.setattr(lb, 'monotonic', lambda: now[0])
    monkeypatch.setattr(lb, '_utc_now', lambda: utc[0])
    model, budget = _client(tmp_path)

    def dispatch(*args):
        row, summary = _saved(budget)
        assert row['status'] == 'reserved'
        assert row['reserved_monotonic'] == row['dispatched_monotonic'] == 100.
        assert row['reserved_at_utc'] == row['dispatched_at_utc'] == utc[0]
        assert row['completed_monotonic'] is row['completed_at_utc'] is None
        assert row['api_usage'] is None
        assert summary['reserved_calls'] == summary['dispatched_calls'] == 1
        assert summary['completed_calls'] == 0
        now[0] = 101.25
        utc[0] = '2026-10-27T00:00:01.250000Z'
        return _reply(usage=dict(prompt_tokens=7, completion_tokens=3, total_tokens=10,
                                 prompt_tokens_details=dict(cached_tokens=2)))

    monkeypatch.setattr(lc, 'bounded_dispatch', dispatch)
    assert model.chat(_messages(), caller_tag='Coder') == 'usable response'
    row, summary = _saved(budget)
    assert row['status'] == 'answered'
    assert row['completed_monotonic'] == 101.25
    assert row['completed_at_utc'] == utc[0]
    for field in ('reserved_at_utc', 'dispatched_at_utc', 'completed_at_utc'):
        _assert_utc(row[field])
    assert row['api_usage'] == dict(prompt_tokens=7, completion_tokens=3, total_tokens=10,
                                    prompt_tokens_details=dict(cached_tokens=2))
    assert row['usage_source'] == 'api_usage'
    assert summary['completed_calls'] == summary['dispatched_calls'] == 1
    assert summary['unfinished_calls'] == summary['undispatched_reservations'] == 0
    assert summary['api_usage']['known_calls'] == 1
    assert summary['api_usage']['unknown_calls'] == 0
    assert summary['api_usage']['totals']['total_tokens'] == 10
    assert summary['api_usage']['totals']['cache_read_input_tokens'] is None


@pytest.mark.parametrize('usage', [None, {}, {'prompt_tokens': 7}, {'total_tokens': 10}])
def test_missing_or_estimated_usage_is_unknown_instead_of_zero(tmp_path, monkeypatch, usage):
    model, budget = _client(tmp_path)
    monkeypatch.setattr(lc, 'bounded_dispatch', lambda *args: _reply(usage=usage))
    assert model.chat(_messages(), caller_tag='Coder') == 'usable response'
    row, summary = _saved(budget)
    assert row['status'] == 'answered'
    assert row['api_usage'] is None and row['usage_source'] == 'estimated'
    assert summary['api_usage']['known_calls'] == 0
    assert summary['api_usage']['unknown_calls'] == 1
    assert all(value is None for value in summary['api_usage']['known_totals'].values())
    assert all(value is None for value in summary['api_usage']['totals'].values())


@pytest.mark.parametrize('usage', [
    [], 'private raw usage',
    {'prompt_tokens': True, 'completion_tokens': 3},
    {'prompt_tokens': -1, 'completion_tokens': 3},
    {'prompt_tokens': 7.5, 'completion_tokens': 3},
    {'prompt_tokens': '7', 'completion_tokens': 3},
    {'prompt_tokens': 7, 'completion_tokens': 3,
     'prompt_tokens_details': {'cached_tokens': False}},
])
def test_invalid_api_usage_is_retained_as_unknown_after_failed_call(tmp_path, monkeypatch, usage):
    model, budget = _client(tmp_path)
    monkeypatch.setattr(lc, 'bounded_dispatch', lambda *args: _reply(usage=usage))
    with pytest.raises(lb.LLMCallRefused):
        model.chat(_messages(), caller_tag='Coder')
    row, summary = _saved(budget)
    assert row['status'] == 'failed'
    assert row['api_usage'] is None and row['usage_source'] == 'invalid_api_usage'
    assert row['completed_at_utc'] is not None
    assert summary['api_usage']['unknown_calls'] == 1
    assert summary['remaining_calls'] == 3 and summary['reserved_calls'] == 1


def test_http_failure_has_unknown_usage_and_keeps_its_reservation(tmp_path, monkeypatch):
    model, budget = _client(tmp_path)

    def fail(*args):
        raise urllib.error.HTTPError('https://private-provider.invalid', 503,
                                     'PRIVATE-PROVIDER-FAILURE', {}, None)

    monkeypatch.setattr(lc, 'bounded_dispatch', fail)
    with pytest.raises(lb.LLMCallRefused):
        model.chat(_messages(), caller_tag='Coder')
    row, summary = _saved(budget)
    assert row['status'] == 'failed'
    assert row['dispatched_monotonic'] is not None
    assert row['completed_monotonic'] >= row['dispatched_monotonic']
    assert row['api_usage'] is None and row['usage_source'] == 'not_observed'
    assert summary['reserved_calls'] == summary['dispatched_calls'] == summary['completed_calls'] == 1
    assert summary['remaining_calls'] == 3 and summary['api_usage']['unknown_calls'] == 1
    assert 'PRIVATE-PROVIDER-FAILURE' not in '\n'.join(p.read_text() for p in budget.root.glob('*.json'))


def test_invalid_json_after_answer_keeps_api_usage_and_original_completion_time(tmp_path, monkeypatch):
    model, budget = _client(tmp_path)
    monkeypatch.setattr(lc, 'bounded_dispatch', lambda *args: _reply('not JSON'))
    with pytest.raises(lb.LLMCallRefused, match='reservation retained'):
        model.chat_json(_messages(), caller_tag='Designer')
    row, summary = _saved(budget)
    assert row['status'] == 'invalid_json'
    assert row['api_usage']['total_tokens'] == 10
    original_completion = row['completed_monotonic'], row['completed_at_utc']
    budget.abort('invalid_json')
    again, summary = _saved(budget)
    assert (again['completed_monotonic'], again['completed_at_utc']) == original_completion
    assert summary['completed_calls'] == summary['dispatched_calls'] == summary['reserved_calls'] == 1
    assert summary['api_usage']['known_calls'] == 1 and summary['api_usage']['unknown_calls'] == 0
    assert summary['remaining_calls'] == 3 and summary['aborted'] is None


def test_valid_usage_survives_a_bad_response_choice(tmp_path, monkeypatch):
    model, budget = _client(tmp_path)
    monkeypatch.setattr(lc, 'bounded_dispatch', lambda *args: _reply(choices=[]))
    with pytest.raises(lb.LLMCallRefused):
        model.chat(_messages(), caller_tag='Coder')
    row, summary = _saved(budget)
    assert row['status'] == 'failed' and row['api_usage']['total_tokens'] == 10
    assert summary['api_usage']['known_calls'] == 1


def test_unfinished_reservation_is_not_reported_as_dispatch_or_free_completed_call(tmp_path):
    model, budget = _client(tmp_path)
    budget.reserve(model._bounded_identity(), json.dumps(dict(max_tokens=32)), 'Coder', 5)
    row, summary = _saved(budget)
    assert row['status'] == 'reserved' and row['reserved_monotonic'] is not None
    assert row['dispatched_monotonic'] is row['completed_monotonic'] is None
    assert row['dispatched_at_utc'] is row['completed_at_utc'] is None
    assert row['api_usage'] is None
    assert summary['reserved_calls'] == summary['unfinished_calls'] == summary['undispatched_reservations'] == 1
    assert summary['dispatched_calls'] == summary['completed_calls'] == 0
    assert summary['remaining_calls'] == 3 and summary['api_usage']['unknown_calls'] == 1
    assert summary['api_usage']['totals']['total_tokens'] is None


def test_cancelled_call_does_not_refund_or_invent_api_usage(tmp_path):
    model, budget = _client(tmp_path)
    call, _ = budget.reserve(model._bounded_identity(), json.dumps(dict(max_tokens=32)), 'Coder', 5)
    budget.mark_dispatched(call)
    budget.finish(call, outcome='cancelled')
    row, summary = _saved(budget)
    assert row['status'] == 'cancelled' and row['api_usage'] is None
    assert row['completed_at_utc'] is not None
    assert summary['remaining_calls'] == 3 and summary['reserved_calls'] == 1


def test_mixed_known_and_unknown_usage_keeps_only_a_known_subtotal(tmp_path, monkeypatch):
    model, budget = _client(tmp_path)
    responses = iter([_reply(), _reply(usage=None)])
    monkeypatch.setattr(lc, 'bounded_dispatch', lambda *args: next(responses))
    model.chat(_messages(), caller_tag='Coder')
    model.chat(_messages(), caller_tag='Fixer')
    _, summary = _saved(budget, call=2)
    usage = summary['api_usage']
    assert usage['known_calls'] == usage['unknown_calls'] == 1
    assert usage['known_totals']['total_tokens'] == 10
    assert usage['totals']['total_tokens'] is None
    assert usage['field_unknown_calls']['total_tokens'] == 1


def test_absent_api_total_is_not_synthesized_and_explicit_zero_is_preserved(tmp_path, monkeypatch):
    model, budget = _client(tmp_path)
    responses = iter([_reply(usage=dict(prompt_tokens=7, completion_tokens=3)),
                      _reply(usage=dict(prompt_tokens=0, completion_tokens=0, total_tokens=0))])
    monkeypatch.setattr(lc, 'bounded_dispatch', lambda *args: next(responses))
    model.chat(_messages(), caller_tag='Coder')
    model.chat(_messages(), caller_tag='Fixer')
    row, summary = _saved(budget, call=2)
    assert row['api_usage']['total_tokens'] == 0
    assert 'total_tokens' not in summary['attempts'][0]['api_usage']
    assert summary['api_usage']['known_calls'] == 2
    assert summary['api_usage']['known_totals']['total_tokens'] == 0
    assert summary['api_usage']['totals']['total_tokens'] is None


def test_provider_extensions_source_and_credentials_are_never_persisted(tmp_path, monkeypatch):
    model, budget = _client(tmp_path)
    usage = dict(prompt_tokens=7, completion_tokens=3, total_tokens=10,
                 private_extension='PRIVATE-USAGE-PROVIDER-TEXT',
                 prompt_tokens_details=dict(cached_tokens=2, private='PRIVATE-USAGE-DETAILS'))
    monkeypatch.setattr(lc, 'bounded_dispatch', lambda *args: _reply('PRIVATE-USAGE-RESPONSE', usage=usage))
    assert model.chat(_messages(), caller_tag='Coder') == 'PRIVATE-USAGE-RESPONSE'
    stored = '\n'.join(path.read_text() for path in budget.root.glob('*.json'))
    for secret in ('PRIVATE-USAGE-PROMPT', 'PRIVATE-USAGE-RESPONSE', 'PRIVATE-USAGE-KEY',
                   'PRIVATE-USAGE-PROVIDER-TEXT', 'PRIVATE-USAGE-DETAILS',
                   'https://usage-fixture.invalid'):
        assert secret not in stored
    row, _ = _saved(budget)
    assert row['api_usage']['prompt_tokens_details'] == dict(cached_tokens=2)
    assert 'messages' not in row['request'] and 'response_text' not in row


def test_strict_budget_finish_keeps_its_original_record_format_even_with_optional_usage(tmp_path):
    budget = lb.LLMCallBudget(_limits(), tmp_path / 'strict')
    identity = dict(model='strict')
    budget.bind(identity)
    body = dict(max_tokens=32, messages=_messages())
    call, _ = budget.reserve(identity, json.dumps(body), 'Coder', 5.)
    reservation_keys = {'call', 'caller', 'status', 'request', 'request_bytes',
                        'request_sha256', 'output_tokens_reserved', 'timeout_seconds'}
    assert set(budget.summary()['attempts'][0]) == reservation_keys
    budget.finish(call, response='STRICT-ORIGINAL-RESPONSE', usage_source='api_usage',
                  usage=dict(prompt_tokens=7, completion_tokens=3), wall_seconds=1.)
    row = budget.summary()['attempts'][0]
    assert set(row) == reservation_keys | {'wall_seconds', 'usage_source', 'response_text',
                                         'response_sha256', 'response_bytes'}
    assert row['request'] == body and row['response_text'] == 'STRICT-ORIGINAL-RESPONSE'
    assert 'api_usage' not in budget.summary() and 'dispatched_calls' not in budget.summary()


def test_strict_client_uses_the_legacy_finish_signature(tmp_path, monkeypatch):
    model, budget = _client(tmp_path, strict=True)
    finish = budget.finish
    calls = []

    def legacy_finish(call, *, response=None, outcome='answered', wall_seconds=0., usage_source=None):
        calls.append(call)
        return finish(call, response=response, outcome=outcome,
                      wall_seconds=wall_seconds, usage_source=usage_source)

    monkeypatch.setattr(budget, 'finish', legacy_finish)
    monkeypatch.setattr(lc, 'bounded_dispatch', lambda *args: _reply())
    assert model.chat(_messages(), caller_tag='Coder') == 'usable response'
    assert calls == [1]
    assert 'reserved_at_utc' not in budget.summary()['attempts'][0]
