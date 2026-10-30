"""Loopback-only fake HTTP gateway tests. No real model, GPU, or upstream."""
import http.client
import json
import urllib.error
import urllib.request

import pytest

from benchmarks.experiments.s5.s5_fake_http import FakeHTTPServer
from operator_agent import s5_shared_budget as shared
from operator_agent.llm_budget import LLMCallLimits, ProductionLLMCallBudget
from operator_agent.llm_client import LLMClient


def ledger(tmp_path, max_calls=2):
    path = tmp_path / 'shared.sqlite'
    shared.create(path, task_id='fixture', max_calls=max_calls, max_output_tokens=40,
                  total_output_tokens=40*max_calls, wall_clock_s=120)
    return path


def body(**changes):
    return dict({'model':'fake-model', 'messages':[{'role':'user','content':'fixture'}],
                 'max_tokens':20, 'stream':False}, **changes)


def post(server, obj):
    req=urllib.request.Request(server.base_url+'/chat/completions',
        data=json.dumps(obj).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=5) as response:
        return response.status, response.read(), response.headers


def test_real_python_bounded_client_uses_loopback_and_shared_reservation(tmp_path,monkeypatch):
    path=ledger(tmp_path)
    monkeypatch.setenv('LLM_FALLBACK_MODEL','')
    seen=[]
    def adapter(request):
        seen.append(request)
        return [json.dumps({'model':'fake-model','choices':[{'message':{'content':'ok'},
                  'finish_reason':'stop'}],'usage':{'prompt_tokens':7,'completion_tokens':1,
                                                   'total_tokens':8}}).encode()]
    limits=LLMCallLimits(max_calls=2,max_output_tokens=40,total_output_token_reservations=80,
                         allowed_callers=('Coder',),call_seconds=10,campaign_seconds=60)
    local=ProductionLLMCallBudget(limits,tmp_path/'local')
    with FakeHTTPServer(path,task_id='fixture',model='fake-model',participant='inner',adapter=adapter) as server:
        client=LLMClient(base_url=server.base_url,api_key='placeholder',model='fake-model',
                         max_tokens=20,timeout=10,call_budget=local)
        assert client.chat([{'role':'user','content':'private-fixture-prompt'}],caller_tag='Coder')=='ok'
    snapshot=shared.snapshot(path)
    assert snapshot['reserved_calls']==snapshot['completed_calls']==1
    assert snapshot['calls'][0]['participant']=='inner'
    assert snapshot['calls'][0]['api_usage'] is None
    assert seen[0]['chat_template_kwargs']=={'enable_thinking':False}
    assert b'private-fixture-prompt' not in path.read_bytes()


def test_pi_ai_fields_sse_and_tool_calls_reach_fake_adapter_without_silent_loss(tmp_path):
    path=ledger(tmp_path); seen=[]
    chunks=[b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"x",'
            b'"type":"function","function":{"name":"f","arguments":"{}"}}]},'
            b'"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n',
            b'data: [DONE]\n\n']
    def adapter(request): seen.append(request); return chunks
    request=body(stream=True,stream_options={'include_usage':True},store=False,
                 tools=[{'type':'function','function':{'name':'f','parameters':{'type':'object'}}}])
    request['max_completion_tokens']=request.pop('max_tokens')
    with FakeHTTPServer(path,task_id='fixture',model='fake-model',participant='outer',adapter=adapter) as server:
        status,data,headers=post(server,request)
    assert status==200 and data==b''.join(chunks) and headers.get_content_type()=='text/event-stream'
    assert seen[0]['max_tokens']==20 and seen[0]['stream_options']=={'include_usage':True}
    assert seen[0]['store'] is False and seen[0]['tools']==request['tools']
    assert shared.snapshot(path)['calls'][0]['api_usage'] is None


@pytest.mark.parametrize('change',[
    {'max_completion_tokens':20}, {'chat_template_kwargs':{'enable_thinking':'false'}},
    {'store':True}, {'stream_options':{'include_usage':True}}, {'enable_thinking':0},
    {'unexpected':'x'}, {'model':'wrong'},
])
def test_malformed_or_conflicting_options_do_not_use_pool(tmp_path,change):
    path=ledger(tmp_path)
    with FakeHTTPServer(path,task_id='fixture',model='fake-model',participant='outer',
                        adapter=lambda _:pytest.fail('must not dispatch')) as server:
        with pytest.raises(urllib.error.HTTPError): post(server,body(**change))
    assert shared.snapshot(path)['reserved_calls']==0


def test_two_servers_share_cap_and_unknown_usage(tmp_path):
    path=ledger(tmp_path,max_calls=1)
    adapter=lambda _: [b'{"choices":[]}']
    with FakeHTTPServer(path,task_id='fixture',model='fake-model',participant='outer',adapter=adapter) as outer, \
         FakeHTTPServer(path,task_id='fixture',model='fake-model',participant='inner',adapter=adapter) as inner:
        assert post(outer,body())[0]==200
        with pytest.raises(urllib.error.HTTPError) as err: post(inner,body())
        assert err.value.code==429
    data=shared.snapshot(path)
    assert data['reserved_calls']==1 and data['usage_unknown_calls']==1


def test_truncated_response_records_failure_without_payload_leak(tmp_path):
    path=ledger(tmp_path)
    def adapter(_):
        yield b'first'
        raise RuntimeError('private fixture error')
    with FakeHTTPServer(path,task_id='fixture',model='fake-model',participant='outer',adapter=adapter) as server:
        with pytest.raises(urllib.error.HTTPError) as err: post(server,body())
        assert err.value.code==502 and b'private' not in err.value.read()
    assert shared.snapshot(path)['calls'][0]['outcome']=='failed'


def test_http_rejects_bad_path_and_oversized_length_without_dispatch(tmp_path):
    path=ledger(tmp_path)
    with FakeHTTPServer(path,task_id='fixture',model='fake-model',participant='outer',adapter=lambda _:[b'ok'],
                        max_request_bytes=100) as server:
        port=int(server.base_url.split(':')[-1].split('/')[0])
        conn=http.client.HTTPConnection('127.0.0.1',port,timeout=5)
        conn.request('POST','/wrong',body=b'{}',headers={'Content-Type':'application/json'})
        assert conn.getresponse().status==404
        conn.close()
        conn=http.client.HTTPConnection('127.0.0.1',port,timeout=5)
        conn.request('POST','/v1/chat/completions',body=b'x'*101,headers={'Content-Type':'application/json'})
        assert conn.getresponse().status==413
        conn.close()
    assert shared.snapshot(path)['reserved_calls']==0
