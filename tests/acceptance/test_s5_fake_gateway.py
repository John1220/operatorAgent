"""In-process fake gateway exercises; no HTTP, model provider or GPU."""
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from benchmarks.s5_fake_gateway import FakeGateway, GatewayFakeFailure, GatewayRefused
from operator_agent import s5_shared_budget as shared


def setup(tmp_path, adapter, *, max_calls=4, **options):
    path = tmp_path / 'budget.sqlite'
    shared.create(path, task_id='task-01', max_calls=max_calls, max_output_tokens=40,
                  total_output_tokens=max_calls * 40, wall_clock_s=120)
    gateway = FakeGateway(path, task_id='task-01', model='fake-model', adapter=adapter, **options)
    return path, gateway


def request(*, stream=False):
    return {'model':'fake-model', 'messages':[{'role':'user', 'content':'private-prompt'}],
            'max_tokens':20, 'stream':stream}


def test_fake_json_and_sse_pass_through_with_unknown_usage(tmp_path):
    def adapter(body):
        return ([b'data: {"choices":[]}\n\n', b'data: [DONE]\n\n'] if body['stream']
                else [b'{"choices":[],"usage":{"prompt_tokens":0}}'])
    path, gateway = setup(tmp_path, adapter)
    assert b''.join(gateway.stream(request(), participant='outer')).startswith(b'{')
    assert b''.join(gateway.stream(request(stream=True), participant='inner')).endswith(b'[DONE]\n\n')
    data = shared.snapshot(path)
    assert data['reserved_calls'] == 2 and data['completed_calls'] == 2
    assert {row['participant'] for row in data['calls']} == {'outer', 'inner'}
    assert all(row['outcome'] == 'answered' and row['api_usage'] is None for row in data['calls'])
    assert data['network_interception_proven'] is False
    assert b'private-prompt' not in path.read_bytes()


def test_concurrent_fake_outer_and_inner_share_one_exact_cap(tmp_path):
    called=[]
    lock=threading.Lock()
    def adapter(body):
        with lock: called.append(True)
        return [b'ok']
    path,gateway=setup(tmp_path,adapter,max_calls=5)
    def attempt(i):
        try:
            return b''.join(gateway.stream(request(), participant='outer' if i%2 else 'inner'))
        except GatewayRefused:
            return None
    with ThreadPoolExecutor(max_workers=10) as executor:
        results=list(executor.map(attempt,range(20)))
    assert results.count(b'ok') == 5 and len(called) == 5
    snapshot=shared.snapshot(path)
    assert snapshot['reserved_calls']==5 and snapshot['completed_calls']==5


@pytest.mark.parametrize('change', [
    {'model':'wrong'}, {'max_tokens':True}, {'max_tokens':0}, {'n':2},
    {'stream':1}, {'temperature':float('nan')}, {'messages':[]}, {'api_key':'secret'},
])
def test_invalid_request_never_calls_adapter_or_uses_budget(tmp_path,change):
    path,gateway=setup(tmp_path,lambda body: pytest.fail('adapter called'))
    with pytest.raises(GatewayRefused):
        gateway.stream(dict(request(),**change),participant='outer')
    assert shared.snapshot(path)['reserved_calls']==0


def test_response_limits_and_adapter_failure_do_not_refund(tmp_path):
    def adapter(body):
        yield b'12345'
        yield b'67890'
    path,gateway=setup(tmp_path,adapter,max_response_bytes=6)
    with pytest.raises(GatewayFakeFailure):
        list(gateway.stream(request(),participant='inner'))
    row=shared.snapshot(path)['calls'][0]
    assert row['outcome']=='failed' and row['api_usage'] is None
    assert shared.snapshot(path)['reserved_calls']==1


def test_consumer_close_records_cancel_and_closes_fake_source(tmp_path):
    closed=[]
    def adapter(body):
        try:
            yield b'first'
            yield b'second'
        finally:
            closed.append(True)
    path,gateway=setup(tmp_path,adapter)
    result=gateway.stream(request(stream=True),participant='outer')
    assert next(result)==b'first'
    result.close()
    assert closed==[True]
    assert shared.snapshot(path)['calls'][0]['outcome']=='cancelled'


def test_finish_audit_failure_stops_both_participants(tmp_path,monkeypatch):
    path,gateway=setup(tmp_path,lambda body:[b'ok'])
    def bad_finish(*args,**kwargs): raise shared.BudgetError('ledger_failure')
    monkeypatch.setattr(shared,'finish',bad_finish)
    with pytest.raises(GatewayRefused,match='campaign stop recorded'):
        list(gateway.stream(request(),participant='outer'))
    assert shared.snapshot(path)['aborted']
    with pytest.raises(GatewayRefused,match='gateway stopped'):
        list(gateway.stream(request(),participant='inner'))


def test_replaced_ledger_is_never_aborted_or_written_as_original(tmp_path):
    path,gateway=setup(tmp_path,lambda body:[b'ok'])
    path.rename(tmp_path/'old.sqlite')
    shared.create(path,task_id='other-task',max_calls=4,max_output_tokens=40,
                  total_output_tokens=160,wall_clock_s=120)
    with pytest.raises(GatewayRefused,match='global stop unproven'):
        list(gateway.stream(request(),participant='outer'))
    other=shared.snapshot(path)
    assert other['task_id']=='other-task' and not other['aborted'] and other['reserved_calls']==0


def test_external_abort_blocks_previously_reserved_request(tmp_path,monkeypatch):
    called=[]
    path,gateway=setup(tmp_path,lambda body:called.append(True) or [b'ok'])
    original=shared.reserve
    def reserve_then_stop(*args,**kwargs):
        result=original(*args,**kwargs)
        shared.abort(path,reason='manual_stop',expected_task_id='task-01')
        return result
    monkeypatch.setattr(shared,'reserve',reserve_then_stop)
    with pytest.raises(GatewayRefused,match='campaign_aborted'):
        list(gateway.stream(request(),participant='outer'))
    snapshot=shared.snapshot(path)
    assert not called and snapshot['aborted'] and snapshot['reserved_calls']==1
    assert snapshot['calls'][0]['dispatched_monotonic'] is None
