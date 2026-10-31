"""Model gateway reservation policy; all adapters are in-process fakes."""
from concurrent.futures import ThreadPoolExecutor
import json
import threading

import pytest
from benchmarks.experiments.s5.s5_gateway_policy import GatewayPolicy, PolicyRefused, PolicyUpstreamFailure
from operator_agent.execute import s5_shared_budget as shared


def make(tmp_path, adapter, max_calls=2, **options):
    path=tmp_path/'ledger.sqlite'
    shared.create(path,task_id='policy-fixture',max_calls=max_calls,max_output_tokens=100,
                  total_output_tokens=max_calls*100,wall_clock_s=120)
    return path,GatewayPolicy(path,task_id='policy-fixture',model='model',adapter=adapter,**options)


def req(**kw):
    return dict(model='model',messages=[{'role':'user','content':'private prompt'}],
                max_tokens=30,stream=True,**kw)


def test_outer_and_inner_compete_for_two_total_reservations(tmp_path):
    count=[]; lock=threading.Lock()
    def adapter(request):
        with lock: count.append(1)
        return [b'data: test\n\n',b'data: [DONE]\n\n']
    path,policy=make(tmp_path,adapter)
    def call(i):
        try: return b''.join(policy.stream(req(),participant='outer' if i%2 else 'inner'))
        except PolicyRefused: return None
    with ThreadPoolExecutor(max_workers=10) as pool: responses=list(pool.map(call,range(20)))
    assert sum(x is not None for x in responses)==len(count)==2
    state=shared.snapshot(path)
    assert state['reserved_calls']==state['completed_calls']==2
    assert all(row['outcome']=='unknown' and row['api_usage'] is None for row in state['calls'])
    assert b'private prompt' not in path.read_bytes()


@pytest.mark.parametrize('chunks',[[],[b'data: [DONE]\n\n'],[b'{"ok":true}']])
def test_full_or_empty_iterator_never_implies_provider_answer_or_usage(tmp_path,chunks):
    path,policy=make(tmp_path,lambda request:chunks)
    assert list(policy.stream(req(),participant='outer'))==chunks
    row=shared.snapshot(path)['calls'][0]
    assert row['outcome']=='unknown' and row['usage_evidence']=='missing' and row['api_usage'] is None


@pytest.mark.parametrize('bad',[
    {'base_url':'http://remote'}, {'api_key':'secret'}, {'model':'other'},
    {'max_tokens':0}, {'max_tokens':True}, {'messages':[]}, {'n':2},
    {'stream':1}, {'store':True}, {'temperature':float('nan')},
    {'max_completion_tokens':30}, {'chat_template_kwargs':{'enable_thinking':'yes'}},
])
def test_invalid_input_rejected_before_reserving(tmp_path,bad):
    path,policy=make(tmp_path,lambda _: pytest.fail('must not call adapter'))
    request=req(); request.update(bad)
    with pytest.raises(PolicyRefused): policy.stream(request,participant='outer')
    assert shared.snapshot(path)['reserved_calls']==0


def test_client_completion_token_field_passes_unchanged(tmp_path):
    seen=[]; path,policy=make(tmp_path,lambda request:seen.append(request) or [b'ok'])
    request=req(); del request['max_tokens']; request['max_completion_tokens']=40
    assert b''.join(policy.stream(request,participant='inner'))==b'ok'
    assert seen[0]['max_completion_tokens']==40 and 'max_tokens' not in seen[0]
    assert shared.snapshot(path)['calls'][0]['output_tokens_reserved']==40


def test_cancel_closes_adapter_without_refund_or_provider_claim(tmp_path):
    closed=[]
    def adapter(request):
        try:
            yield b'first'
            yield b'second'
        finally: closed.append(True)
    path,policy=make(tmp_path,adapter)
    stream=policy.stream(req(),participant='outer')
    assert next(stream)==b'first'
    stream.close()
    row=shared.snapshot(path)['calls'][0]
    assert closed==[True] and row['outcome']=='cancelled' and row['api_usage'] is None
    assert shared.snapshot(path)['reserved_calls']==1


def test_upstream_exception_or_size_limit_has_unknown_provider_outcome(tmp_path):
    path,policy=make(tmp_path,lambda _: [b'12345',b'67890'],max_response_bytes=6)
    with pytest.raises(PolicyUpstreamFailure): list(policy.stream(req(),participant='outer'))
    row=shared.snapshot(path)['calls'][0]
    assert row['outcome']=='unknown' and row['api_usage'] is None


def test_audit_failure_globally_stops_and_is_not_retriable(tmp_path,monkeypatch):
    path,policy=make(tmp_path,lambda _: [b'ok'])
    def fail(*a,**kw): raise OSError('injected failure')
    monkeypatch.setattr(shared,'dispatch',fail)
    with pytest.raises(PolicyRefused,match='campaign stop recorded'):
        list(policy.stream(req(),participant='outer'))
    assert shared.snapshot(path)['aborted']
    with pytest.raises(PolicyRefused,match='gateway stopped'):
        list(policy.stream(req(),participant='inner'))


def test_replaced_ledger_is_not_modified_or_aborted(tmp_path):
    path,policy=make(tmp_path,lambda _: [b'ok'])
    path.rename(tmp_path/'original.sqlite')
    shared.create(path,task_id='other',max_calls=2,max_output_tokens=100,
                  total_output_tokens=200,wall_clock_s=120)
    with pytest.raises(PolicyRefused,match='global stop unproven'):
        list(policy.stream(req(),participant='outer'))
    other=shared.snapshot(path)
    assert other['task_id']=='other' and other['reserved_calls']==0 and not other['aborted']


def test_excessive_ledger_budget_rejected_before_gateway_instantiation(tmp_path):
    path=tmp_path/'ledger.sqlite'
    shared.create(path,task_id='fixture',max_calls=3,max_output_tokens=100,
                  total_output_tokens=300,wall_clock_s=120)
    with pytest.raises(PolicyRefused,match='unsuitable'):
        GatewayPolicy(path,task_id='fixture',model='model',adapter=lambda _:[])


def test_reasoning_effort_is_capped_for_every_arm(tmp_path):
    seen={}
    def adapter(request):
        seen.update(request)
        return [b'ok']
    path=tmp_path/'responses.sqlite'
    shared.create(path,task_id='responses-fixture',max_calls=2,max_output_tokens=256,
                  total_output_tokens=512,wall_clock_s=120)
    policy=GatewayPolicy(path,task_id='responses-fixture',model='model',adapter=adapter,
                         protocol='responses',reasoning_effort='none')
    request=dict(model='model',stream=True,store=False,max_output_tokens=128,
                 input=[{'role':'user','content':'hi'}])
    list(policy.stream(request,participant='outer'))
    assert seen.get('reasoning')=={'effort':'none'}
    assert 'reasoning' not in request
