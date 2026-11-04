"""Real model transport, CPU verification and supervised cells, using loopback only."""
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from benchmarks.experiments.phase1 import connected, connected_plan, connected_summary, runner
from benchmarks.experiments.phase1.plan import read
from tests.unit.test_autonomous import SKETCH, CODE


class Provider:
    def __init__(self,delay=0,invalid=False,usage=True):
        self.calls=[]
        self.keys=[]
        owner=self
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                owner.calls.append(body)
                owner.keys.append(self.headers.get('Authorization'))
                time.sleep(delay)
                prompt=''.join(m.get('content') or '' for m in body.get('messages',[]))
                text=json.dumps(SKETCH) if '设计专家' in prompt else CODE
                value=dict(choices=[dict(message=dict(content=text),finish_reason='stop')])
                if usage:value['usage']=dict(prompt_tokens=10,completion_tokens=5,total_tokens=15)
                data=b'bad response' if invalid else json.dumps(value).encode()
                try:
                    self.send_response(200);self.send_header('Content-Type','application/json')
                    self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
                except (BrokenPipeError,ConnectionResetError):pass
            def log_message(self,*a):pass
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.server.daemon_threads=True
    def __enter__(self):
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start();return self
    def __exit__(self,*a):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=2)
    @property
    def url(self):return 'http://127.0.0.1:%d/v1'%self.server.server_port


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for key in tuple(os.environ):
        if key.startswith(('LLM_','OPERATOR_AGENT_S5_DEV_','OPERATOR_AGENT_S5_SHARED_')):
            monkeypatch.delenv(key,raising=False)
    monkeypatch.delenv('OPERATOR_AGENT_S5_TASK_ID',raising=False)
    monkeypatch.setenv('PHASE1_FIXTURE_KEY','local-synthetic-only')


def freeze(tmp_path,server,**budget):
    config=dict(backend='fake',entry_b='core',
        model=dict(base_url=server.url,id='phase1-local',transport='local_fixture',
                   api_key_env='PHASE1_FIXTURE_KEY',temperature=.2),
        tasks=[dict(task_id='matmul',declaration=dict(request='matmul fp32',precision='fp32',
                    shapes=[[8,8,8]],dsl='python'))],max_rounds=1,
        final_wall_clock_s=30,budget=dict(wall_clock_s=30,max_llm_calls=8,oracle_total=20,**budget))
    connected_plan.freeze(tmp_path/'frozen',config)
    return tmp_path/'frozen/plan.json'


def test_real_transport_and_independent_finals_replay_without_credentials(tmp_path,monkeypatch):
    with Provider() as server:
        path=freeze(tmp_path,server)
        result=connected.run(path,tmp_path/'execution')
        assert server.calls
        assert set(server.keys)=={'Bearer local-synthetic-only'}
        assert all(c['model']=='phase1-local' for c in server.calls)
    assert result['phase']=='done'
    assert [r['arm'] for r in result['cells']]==['A','B']
    for row in result['cells']:
        assert row['phase']=='done',row
        assert row['budget_auditable'] is True,row['problems']
        assert row['final_evaluation']['complete'] is True,row['problems']
        assert row['final_evaluation']['full_l2_passed'] is True,row
        assert row['events']['count'] is None
        assert row['model_budget']['provider_dispatches']>0
        assert row['model_budget']['api_usage']['unknown_calls']==0
        assert row['model_budget']['cost_usd'] is None
    root=tmp_path/'execution'
    for path in root.rglob('*.json'):
        assert 'local-synthetic-only' not in path.read_text(),path
    monkeypatch.delenv('PHASE1_FIXTURE_KEY')
    assert connected_summary.summarize(root)==result
    assert result['execution_context']['test_only'] is True
    assert result['formal_comparison_credit'] is False
    assert result['gpu_acceptance'] is False
    cell=root/'cells/matmul-A'
    (cell/'selected.py').write_text('tampered')
    damaged=connected_summary.summarize(root)
    assert damaged['cells'][0]['final_evaluation']['complete'] is False
    assert damaged['cells'][0]['problems']


def test_cancel_blocked_live_transport_retains_reservation_and_unstarted_slot(tmp_path):
    with Provider(delay=8) as server:
        path=freeze(tmp_path,server)
        root=tmp_path/'execution'
        failures=[]
        def cancel():
            end=time.monotonic()+10
            while time.monotonic()<end:
                if server.calls:
                    runner.cancel(root);return
                time.sleep(.02)
            failures.append('provider never received first request')
        thread=threading.Thread(target=cancel,daemon=True);thread.start()
        started=time.monotonic()
        result=connected.run(path,root)
        elapsed=time.monotonic()-started
        thread.join(timeout=1)
    assert not failures and elapsed<6
    assert result['phase']=='cancelled'
    assert result['cells'][0]['phase']=='cancelled'
    assert result['cells'][1]['phase']=='not_started'
    assert result['cells'][0]['model_budget']['reserved_calls']==1
    assert result['cells'][0]['model_budget']['api_usage']['unknown_calls']==1
    assert result['cells'][0]['final_evaluation']['complete'] is False


def test_bad_provider_reply_is_failed_without_false_final_success(tmp_path):
    with Provider(invalid=True) as server:
        path=freeze(tmp_path,server)
        result=connected.run(path,tmp_path/'execution')
    assert all(row['phase']=='failed' or not row['final_evaluation']['complete'] for row in result['cells'])
    assert result['cells'][0]['phase']=='failed'
    assert result['cells'][0]['model_budget']['reserved_calls']==1
    assert result['cells'][0]['model_budget']['api_usage']['unknown_calls']==1
