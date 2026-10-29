"""Loopback-only integration of budget gateway, clients and fake upstream."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from benchmarks.s5_gateway_http import GatewayServer
from benchmarks.s5_gateway_supervisor import SupervisedTransport
from operator_agent import s5_shared_budget as budget
from operator_agent.llm_budget import LLMCallLimits, ProductionLLMCallBudget
from operator_agent.llm_client import LLMClient

ROOT=Path('/Applications/DSH Desktop.app/Contents/Resources/app/node_modules/@deepseek-ai')
FIXTURE=Path(__file__).parent/'fixtures'/'s5_dsh_pi_ai_client.mjs'
NODE=shutil.which('node')


class FakeUpstream:
    def __init__(self):
        calls=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_POST(self):
                size=int(self.headers['Content-Length'])
                req=json.loads(self.rfile.read(size));calls.append((req,self.headers.get('Authorization')))
                if req.get('stream'):
                    payload=(b'data: {"id":"fake-1","model":"fake-model","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
                             b'data: {"id":"fake-1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
                             b'data: [DONE]\n\n')
                    content_type='text/event-stream'
                else:
                    payload=json.dumps({'model':'fake-model','choices':[{'message':{'content':'ok'},
                                       'finish_reason':'stop'}]}).encode()
                    content_type='application/json'
                self.send_response(200);self.send_header('Content-Type',content_type)
                self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.calls=calls
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
    @property
    def endpoint(self):
        return 'http://127.0.0.1:%d/v1/chat/completions' % self.server.server_address[1]
    def __enter__(self): self.thread.start();return self
    def __exit__(self,*exc): self.server.shutdown();self.server.server_close();self.thread.join()


@pytest.mark.skipif(not NODE or any(not (ROOT/p/'lib/index.js').exists()
    for p in ('cordis','dsh-llm','dsh-llm-pi-ai')),reason='installed DSH pi-ai unavailable')
def test_dsh_and_python_share_gateway_and_forward_to_fake_upstream(tmp_path,monkeypatch,capsys):
    ledger=tmp_path/'ledger.sqlite'
    budget.create(ledger,task_id='gateway-fixture',max_calls=2,max_output_tokens=40,
                  total_output_tokens=80,wall_clock_s=120)
    limits=LLMCallLimits(max_calls=1,max_output_tokens=40,total_output_token_reservations=40,
                         allowed_callers=('Coder',),call_seconds=10,campaign_seconds=60)
    local=ProductionLLMCallBudget(limits,tmp_path/'local')
    monkeypatch.setenv('LLM_FALLBACK_MODEL','')
    with FakeUpstream() as upstream:
        transport=SupervisedTransport(upstream.endpoint,'fixture-upstream-secret',wall_seconds=10)
        with GatewayServer(ledger,task_id='gateway-fixture',model='fake-model',participant='inner',adapter=transport) as inner, \
             GatewayServer(ledger,task_id='gateway-fixture',model='fake-model',participant='outer',adapter=transport) as outer:
            client=LLMClient(base_url=inner.base_url,api_key='client-placeholder',model='fake-model',
                             max_tokens=20,timeout=10,call_budget=local)
            assert client.chat([{'role':'user','content':'fixture'}],caller_tag='Coder')=='ok'
            env={'PATH':os.environ.get('PATH','/usr/bin:/bin'),'HOME':str(tmp_path),
                 'DSH_HOME':str(tmp_path/'dsh'),'S5_FAKE_API_KEY':'client-placeholder',
                 'NO_PROXY':'127.0.0.1','no_proxy':'127.0.0.1'}
            p=subprocess.run([NODE,str(FIXTURE),outer.base_url,str(ROOT)],env=env,
                             capture_output=True,text=True,timeout=30,check=True)
            data=json.loads(p.stdout.strip().splitlines()[-1])
            assert data['first']=={'text':'ok','reason':'stop'} and data['second']['reason']=='error'
        assert len(upstream.calls)==2
        assert all(auth=='Bearer fixture-upstream-secret' for _,auth in upstream.calls)
        assert [item[0]['stream'] for item in upstream.calls]==[False,True]
    snapshot=budget.snapshot(ledger)
    assert snapshot['reserved_calls']==snapshot['completed_calls']==2
    assert [c['participant'] for c in snapshot['calls']]==['inner','outer']
    assert all(c['outcome']=='unknown' and c['api_usage'] is None for c in snapshot['calls'])
    assert b'fixture-upstream-secret' not in ledger.read_bytes()
    captured=capsys.readouterr()
    assert 'fixture-upstream-secret' not in captured.out+captured.err+p.stdout+p.stderr


def test_invalid_stream_never_delivers_success_terminal_to_client(tmp_path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            payload=(b'data: {"choices":[{"delta":{"content":"x"},"finish_reason":null}]}\n\n'
                     b'data: [DONE]\n\n')
            self.send_response(200);self.send_header('Content-Type','text/event-stream')
            self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
    upstream=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=upstream.serve_forever,daemon=True);thread.start()
    ledger=tmp_path/'ledger.sqlite'
    budget.create(ledger,task_id='fixture',max_calls=1,max_output_tokens=40,total_output_tokens=40,wall_clock_s=60)
    try:
        endpoint='http://127.0.0.1:%d/v1/chat/completions' % upstream.server_address[1]
        transport=SupervisedTransport(endpoint,'fixture-secret',wall_seconds=5)
        with GatewayServer(ledger,task_id='fixture',model='fake-model',participant='outer',adapter=transport) as gateway:
            body=json.dumps({'model':'fake-model','messages':[{'role':'user','content':'x'}],
                             'max_tokens':20,'stream':True}).encode()
            req=urllib.request.Request(gateway.base_url+'/chat/completions',data=body,
                                       headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=10) as response:
                result=response.read()
        assert b'[DONE]' not in result
        snapshot=budget.snapshot(ledger)
        assert snapshot['reserved_calls']==1 and snapshot['calls'][0]['outcome']=='unknown'
    finally:
        upstream.shutdown();upstream.server_close();thread.join()


@pytest.mark.skipif(not NODE or any(not (ROOT/p/'lib/index.js').exists()
    for p in ('cordis','dsh-llm','dsh-llm-pi-ai')),reason='installed DSH pi-ai unavailable')
def test_truncated_stream_is_not_a_successful_dsh_finish(tmp_path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            # A finish_reason followed by an incomplete stream must not be
            # delivered as a successful completion to the actual DSH adapter.
            payload=(b'data: {"choices":[{"index":0,"delta":{"content":"x"},"finish_reason":null}]}\n\n'
                     b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n')
            self.send_response(200);self.send_header('Content-Type','text/event-stream')
            self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
    upstream=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=upstream.serve_forever,daemon=True);thread.start()
    ledger=tmp_path/'ledger.sqlite'
    budget.create(ledger,task_id='fixture',max_calls=1,max_output_tokens=40,total_output_tokens=40,wall_clock_s=60)
    try:
        endpoint='http://127.0.0.1:%d/v1/chat/completions' % upstream.server_address[1]
        transport=SupervisedTransport(endpoint,'fixture-secret',wall_seconds=5)
        with GatewayServer(ledger,task_id='fixture',model='fake-model',participant='outer',adapter=transport) as gateway:
            env={'PATH':os.environ.get('PATH','/usr/bin:/bin'),'HOME':str(tmp_path),
                 'DSH_HOME':str(tmp_path/'dsh'),'S5_FAKE_API_KEY':'placeholder',
                 'NO_PROXY':'127.0.0.1','no_proxy':'127.0.0.1'}
            p=subprocess.run([NODE,str(FIXTURE),gateway.base_url,str(ROOT)],env=env,
                             capture_output=True,text=True,timeout=30,check=True)
            data=json.loads(p.stdout.strip().splitlines()[-1])
            assert data['first']['reason']=='error'
        assert budget.snapshot(ledger)['calls'][0]['outcome']=='unknown'
    finally:
        upstream.shutdown();upstream.server_close();thread.join()


def test_slow_upstream_is_stopped_and_charged_unknown(tmp_path):
    release=threading.Event()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            release.wait(5)
    class Server(ThreadingHTTPServer):
        daemon_threads=True
    upstream=Server(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=upstream.serve_forever,daemon=True);thread.start()
    ledger=tmp_path/'ledger.sqlite'
    budget.create(ledger,task_id='fixture',max_calls=1,max_output_tokens=40,total_output_tokens=40,wall_clock_s=60)
    try:
        endpoint='http://127.0.0.1:%d/v1/chat/completions' % upstream.server_address[1]
        transport=SupervisedTransport(endpoint,'fixture-secret',wall_seconds=.7)
        with GatewayServer(ledger,task_id='fixture',model='fake-model',participant='outer',adapter=transport) as gateway:
            body=json.dumps({'model':'fake-model','messages':[{'role':'user','content':'x'}],
                             'max_tokens':20,'stream':True}).encode()
            req=urllib.request.Request(gateway.base_url+'/chat/completions',data=body,
                                       headers={'Content-Type':'application/json'})
            started=time.monotonic()
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(req,timeout=5)
            assert error.value.code==502 and time.monotonic()-started<4
        snapshot=budget.snapshot(ledger)
        assert snapshot['reserved_calls']==snapshot['completed_calls']==1
        assert snapshot['calls'][0]['outcome']=='unknown' and snapshot['calls'][0]['api_usage'] is None
    finally:
        release.set();upstream.shutdown();upstream.server_close();thread.join()
