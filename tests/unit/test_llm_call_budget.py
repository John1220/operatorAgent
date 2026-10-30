"""CPU/mocked HTTP only: these guards are not real model or GPU acceptance."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
from unittest.mock import Mock

import pytest

from operator_agent import llm_client as lc, llm_budget as lb, llm_transport as transport
from operator_agent.llm_budget import LLMCallBudget, LLMCallLimits, LLMCallRefused
from operator_agent.llm_client import LLMClient


def client(tmp_path, **changes):
    limits = dict(max_calls=2, max_output_tokens=32, total_output_token_reservations=64,
                  allowed_callers=('Coder', 'Search'), call_seconds=1., campaign_seconds=30.)
    limits.update(changes)
    budget = LLMCallBudget(LLMCallLimits(**limits), tmp_path / 'calls')
    c = LLMClient(base_url='https://fixture.invalid/v1', model='fixture', api_key='PRIVATE_KEY',
                  max_tokens=32, timeout=5, call_budget=budget,
                  fallback_model='forbidden', fallback_base_url='https://elsewhere.invalid/v1')
    return c


def response(content='OK', **changes):
    data = dict(model='fixture', choices=[dict(message=dict(content=content), finish_reason='stop')],
                usage=dict(prompt_tokens=100, completion_tokens=5))
    data.update(changes)
    return json.dumps(data)


def chat(c, **kwargs):
    return c.chat([dict(role='user', content='bounded')], caller_tag='Coder', **kwargs)


def test_reserve_before_dispatch_and_lifetime_cap(tmp_path, monkeypatch):
    c = client(tmp_path)
    sent = []
    def send(payload, timeout):
        retained = json.loads((c.call_budget.root / ('call-%03d.json' % (len(sent) + 1))).read_text())
        assert retained['status'] == 'reserved'
        assert 'PRIVATE_KEY' not in json.dumps(retained)
        assert payload['key'] == 'PRIVATE_KEY'
        assert timeout == 1. and retained['timeout_seconds'] == 1.
        sent.append(payload)
        return response()
    monkeypatch.setattr(lc, 'bounded_dispatch', send)
    assert chat(c, temperature=0.) == 'OK'
    c.cost_tracker.reset()  # Observed usage is NOT the pre-dispatch currency.
    assert c.chat([], caller_tag='Search') == 'OK'
    with pytest.raises(LLMCallRefused, match='call budget exhausted'):
        chat(c)
    assert len(sent) == 2
    summary = c.call_budget.summary()
    assert summary['reserved_calls'] == 2 and summary['output_tokens_reserved'] == 64
    assert json.loads(sent[0]['body_text'])['temperature'] == 0.
    assert c.cost_tracker.records[0].caller_tag == 'Search'
    assert not c._fallback_triggered
    assert all((c.call_budget.root / ('call-%03d.json' % i)).stat().st_mode & 0o777 == 0o444 for i in (1, 2))
    assert 'PRIVATE_KEY' not in json.dumps(summary) + repr(c)
    assert 'fixture.invalid' not in json.dumps(summary) + repr(c)
    summary['attempts'].clear()
    assert c.call_budget.summary()['reserved_calls'] == 2


@pytest.mark.parametrize('kind', ('network', 'malformed', 'truncated', 'empty', 'wrong_model',
                                'over_output', 'negative_usage', 'fractional_usage', 'boolean_usage',
                                'non_object', 'duplicate_key', 'nonfinite', 'multiple_choices'))
def test_failure_poison_blocks_hidden_retry_and_fallback(tmp_path, monkeypatch, kind):
    c = client(tmp_path)
    data = json.loads(response())
    raw = None
    if kind == 'truncated': data['choices'][0]['finish_reason'] = 'length'
    if kind == 'empty': data['choices'][0]['message']['content'] = None
    if kind == 'wrong_model': data['model'] = 'fallback'
    if kind == 'over_output': data['usage']['completion_tokens'] = 33
    if kind == 'negative_usage': data['usage']['prompt_tokens'] = -1
    if kind == 'fractional_usage': data['usage']['prompt_tokens'] = 1.5
    if kind == 'boolean_usage': data['usage']['prompt_tokens'] = True
    if kind == 'multiple_choices': data['choices'] *= 2
    if kind == 'malformed': raw = 'bad JSON'
    if kind == 'non_object': raw = '[]'
    if kind == 'duplicate_key': raw = '{"model":"fixture","model":"fallback"}'
    if kind == 'nonfinite': raw = '{"temperature":NaN}'
    send = Mock(return_value=raw if raw is not None else json.dumps(data))
    if kind == 'network': send.side_effect = RuntimeError('PRIVATE_KEY https://secret.invalid')
    monkeypatch.setattr(lc, 'bounded_dispatch', send)
    monkeypatch.setattr(lc.urllib.request, 'urlopen', lambda *a, **k: pytest.fail('legacy/fallback transport'))
    with pytest.raises(LLMCallRefused, match='no fallback/retry') as err:
        chat(c)
    assert 'PRIVATE_KEY' not in str(err.value)
    with pytest.raises(LLMCallRefused, match='campaign stopped'):
        chat(c)
    assert send.call_count == 1
    summary = c.call_budget.summary()
    assert summary['attempts'][0]['status'] == 'failed'
    assert summary['reserved_calls'] == 1 and summary['output_tokens_reserved'] == 32
    if kind != 'network': assert summary['attempts'][0]['response_text'] == send.return_value
    if kind in ('truncated', 'wrong_model', 'over_output'):
        assert c.cost_tracker.total_calls == 1  # Billable response still observed.


@pytest.mark.parametrize('output', [0, -1, True, 1.5, 33])
def test_bad_output_reservation_refused_before_network(tmp_path, monkeypatch, output):
    c = client(tmp_path)
    monkeypatch.setattr(lc, 'bounded_dispatch', lambda *a: pytest.fail('bad output sent'))
    with pytest.raises(LLMCallRefused): chat(c, max_tokens=output)
    assert c.call_budget.summary()['reserved_calls'] == 0


def test_total_output_reservations_never_refund_unused_tokens(tmp_path, monkeypatch):
    c = client(tmp_path, total_output_token_reservations=40)
    send = Mock(return_value=response())
    monkeypatch.setattr(lc, 'bounded_dispatch', send)
    assert chat(c) == 'OK'  # 32 reserved, only 5 observed; no refund.
    with pytest.raises(LLMCallRefused, match='total output reservations'):
        chat(c, max_tokens=9)
    assert chat(c, max_tokens=8) == 'OK'
    assert c.call_budget.summary()['output_tokens_reserved'] == 40 and send.call_count == 2


def test_utf8_full_body_bound_and_unknown_caller_before_reservation(tmp_path, monkeypatch):
    c = client(tmp_path, max_request_bytes=1000)
    monkeypatch.setattr(lc, 'bounded_dispatch', lambda *a: pytest.fail('bad prompt sent'))
    with pytest.raises(LLMCallRefused, match='request byte'):
        c.chat([dict(role='user', content='中' * 350)], caller_tag='Coder')
    with pytest.raises(LLMCallRefused, match='undeclared'):
        c.chat([], caller_tag='Fixer')
    assert c.call_budget.summary()['reserved_calls'] == 0


@pytest.mark.parametrize('field,value', [('base_url', 'https://other.invalid/v1'), ('model', 'other'),
    ('api_key', 'OTHER_KEY'), ('enable_thinking', True), ('timeout', 10), ('max_tokens', 16),
    ('temperature', 0.7)])
def test_client_configuration_frozen(tmp_path, monkeypatch, field, value):
    c = client(tmp_path)
    setattr(c, field, value)
    monkeypatch.setattr(lc, 'bounded_dispatch', lambda *a: pytest.fail('drift sent'))
    with pytest.raises(LLMCallRefused, match='drift'): chat(c)
    assert c.call_budget.summary()['reserved_calls'] == 0


def test_campaign_deadline_before_dispatch_and_wall_overrun_after(tmp_path, monkeypatch):
    c = client(tmp_path)
    c.call_budget._deadline = 0
    send = Mock(return_value=response())
    monkeypatch.setattr(lc, 'bounded_dispatch', send)
    with pytest.raises(LLMCallRefused, match='insufficient campaign time'): chat(c)
    assert send.call_count == 0
    c.call_budget._deadline = float('inf')
    times = iter((0., 0., 2., 2.))  # reserve, start, check, finish
    monkeypatch.setattr(lc.time, 'monotonic', lambda: next(times))
    with pytest.raises(LLMCallRefused): chat(c)
    assert c.call_budget.summary()['attempts'][0]['wall_seconds'] == 2.


def test_reservation_disk_error_is_fail_closed_before_send(tmp_path, monkeypatch):
    c = client(tmp_path)
    monkeypatch.setattr(lb.os, 'fsync', Mock(side_effect=OSError('disk full PRIVATE_KEY')))
    monkeypatch.setattr(lc, 'bounded_dispatch', lambda *a: pytest.fail('unrecorded request'))
    with pytest.raises(LLMCallRefused, match='reservation journal failed'): chat(c)
    with pytest.raises(LLMCallRefused, match='campaign stopped'): chat(c)
    assert c.call_budget.summary()['reserved_calls'] == 1


def test_old_attempt_or_symlink_is_not_overwritten(tmp_path, monkeypatch):
    c = client(tmp_path)
    target = tmp_path / 'old.json'
    target.write_text('historical')
    (c.call_budget.root / 'call-001.json').symlink_to(target)
    monkeypatch.setattr(lc, 'bounded_dispatch', lambda *a: pytest.fail('old evidence overwrite'))
    with pytest.raises(LLMCallRefused, match='journal'): chat(c)
    assert target.read_text() == 'historical'


@pytest.mark.parametrize('name', ('limits.json', 'identity.json'))
def test_frozen_budget_journal_drift_blocks_before_request(tmp_path, monkeypatch, name):
    c = client(tmp_path)
    path = c.call_budget.root / name
    path.chmod(0o600); path.write_text('{}')
    monkeypatch.setattr(lc, 'bounded_dispatch', lambda *a: pytest.fail('journal drift dispatched'))
    with pytest.raises(LLMCallRefused, match='journal drift'): chat(c)
    assert c.call_budget.summary()['reserved_calls'] == 0


def test_previous_response_journal_cannot_be_changed_before_next_request(tmp_path, monkeypatch):
    c = client(tmp_path)
    send = Mock(return_value=response())
    monkeypatch.setattr(lc, 'bounded_dispatch', send)
    chat(c)
    path = c.call_budget.root / 'call-001.json'
    path.chmod(0o600); path.write_text('{}')
    with pytest.raises(LLMCallRefused, match='journal drift'): chat(c)
    assert send.call_count == 1 and c.call_budget.summary()['reserved_calls'] == 1


def test_failed_result_journal_does_not_return_success_or_allow_retry(tmp_path, monkeypatch):
    c = client(tmp_path)
    send = Mock(return_value=response())
    monkeypatch.setattr(lc, 'bounded_dispatch', send)
    monkeypatch.setattr(lb, 'atomic_json', Mock(side_effect=OSError('disk full')))
    with pytest.raises(LLMCallRefused, match='result journal failed'): chat(c)
    with pytest.raises(LLMCallRefused, match='campaign stopped'): chat(c)
    assert send.call_count == 1


def test_strict_json_mode_never_extracts_fences_or_repairs(tmp_path, monkeypatch):
    c = client(tmp_path)
    send = Mock(return_value=response('```json\n{"ok":true}\n```'))
    monkeypatch.setattr(lc, 'bounded_dispatch', send)
    with pytest.raises(LLMCallRefused, match='strict JSON'):
        c.chat_json([], caller_tag='Coder')
    with pytest.raises(LLMCallRefused, match='campaign stopped'):
        c.chat_json([], caller_tag='Coder')
    assert send.call_count == 1


def test_no_usage_is_estimated_not_hard_input_token_cap(tmp_path, monkeypatch):
    c = client(tmp_path)
    monkeypatch.setattr(lc, 'bounded_dispatch', lambda *a: response('no usage', usage=None))
    assert chat(c) == 'no usage'
    assert c.cost_tracker.records[0].usage_source == 'estimated'
    assert c.call_budget.summary()['attempts'][0]['usage_source'] == 'estimated'
    assert c.call_budget.summary()['exact_input_token_cap'] is False


@pytest.mark.parametrize('usage', ({'total_tokens': 100}, {'prompt_tokens': 100}, {'completion_tokens': 5}))
def test_partial_api_usage_is_not_presented_as_complete_token_count(tmp_path, monkeypatch, usage):
    c = client(tmp_path)
    monkeypatch.setattr(lc, 'bounded_dispatch', lambda *a: response('partial usage', usage=usage))
    assert chat(c) == 'partial usage'
    assert c.cost_tracker.records[0].usage_source == 'estimated'


def test_response_byte_limit_defense_even_if_transport_is_replaced(tmp_path, monkeypatch):
    c = client(tmp_path, max_response_bytes=100)
    monkeypatch.setattr(lc, 'bounded_dispatch', lambda *a: '中' * 101)
    with pytest.raises(LLMCallRefused): chat(c)
    assert 'response_text' not in c.call_budget.summary()['attempts'][0]


def test_parallel_reservations_cannot_overspend(tmp_path, monkeypatch):
    c = client(tmp_path)
    send = Mock(return_value=response())
    monkeypatch.setattr(lc, 'bounded_dispatch', send)
    def ask(_):
        try: return chat(c)
        except LLMCallRefused: return 'refused'
    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = list(pool.map(ask, range(8)))
    assert answers.count('OK') == send.call_count == 2
    assert c.call_budget.summary()['output_tokens_reserved'] == 64


def test_journal_cannot_be_resumed_or_rebound(tmp_path):
    c = client(tmp_path)
    with pytest.raises(FileExistsError):
        LLMCallBudget(c.call_budget.limits, c.call_budget.root)
    with pytest.raises(LLMCallRefused, match='already bound'):
        LLMClient(base_url='https://fixture.invalid/v1', call_budget=c.call_budget)


@pytest.mark.parametrize('field,value', [('max_calls', True), ('max_output_tokens', 0),
    ('total_output_token_reservations', -1), ('max_request_bytes', 1.5),
    ('call_seconds', float('nan')), ('campaign_seconds', float('inf')),
    ('allowed_callers', ['Coder']), ('allowed_callers', ('Coder', 'Coder'))])
def test_invalid_limits_rejected(field, value):
    limits = dict(max_calls=1, max_output_tokens=32, total_output_token_reservations=32)
    limits[field] = value
    with pytest.raises(ValueError): LLMCallLimits(**limits)


def test_worker_deadline_no_key_in_argv_no_retry(monkeypatch):
    send = Mock(side_effect=subprocess.TimeoutExpired('worker', .01))
    monkeypatch.setattr(transport.subprocess, 'run', send)
    with pytest.raises(RuntimeError, match='killed and reaped'):
        transport.dispatch(dict(key='PRIVATE_KEY'), .01)
    args, kwargs = send.call_args
    assert send.call_count == 1 and kwargs['timeout'] == .01
    assert 'PRIVATE_KEY' not in ' '.join(args[0])
    assert 'PRIVATE_KEY' in kwargs['input']


def test_worker_disables_proxy_redirect_and_caps_read(monkeypatch):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self, n):
            assert n == 11
            return b'x' * 11
    open_mock = Mock(return_value=Response())
    def opener(*handlers):
        assert any(isinstance(h, transport.NoRedirect) for h in handlers)
        assert any(isinstance(h, transport.urllib.request.ProxyHandler) and h.proxies == {} for h in handlers)
        return type('Opener', (), {'open': open_mock})()
    monkeypatch.setattr(transport.urllib.request, 'build_opener', opener)
    reply = transport.worker(dict(url='https://fixture.invalid/v1', key='', body_text='{}',
                                  timeout=1., max_response_bytes=10))
    assert reply == {'error': 'response_size_limit'} and open_mock.call_count == 1
