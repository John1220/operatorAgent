"""Installed DSH + source plugin + pinned Relay, with loopback model fixtures only."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import sys
import threading
import time

import pytest

from benchmarks.experiments.phase1 import connected, connected_plan
from benchmarks.experiments.phase1.host import entry_observed
from benchmarks.experiments.phase1.plan import read
from benchmarks.experiments.s5 import s5_arm_env, s5_seatbelt
from operator_agent.execute import s5_shared_budget as shared

CODE='def kernel(A, B):\n    return A @ B\n'
SKETCH=dict(algorithm='matmul',tiling=dict(block_m=64,block_n=64,block_k=32),
            memory_plan={},pipeline=[],optimization_hints=[])
READY=(sys.platform=='darwin' and bool(shutil.which('node')) and
       (s5_arm_env.DSH_ROOT/'dsh/lib/bin.js').is_file() and
       s5_arm_env.MCP_CLIENT.is_file() and Path(s5_seatbelt.SANDBOX).is_file())


def _terminal(value):
    if isinstance(value,str):
        try:return _terminal(json.loads(value))
        except (ValueError,TypeError):return False
    if isinstance(value,dict):
        # The status embeds finished candidate/verification records while its
        # owning supervisor can still be running. Only its own phase is terminal.
        if 'phase' in value:return value['phase'] in ('done','failed','cancelled')
        return _terminal(value.get('content',value.get('text')))
    return isinstance(value,list) and any(_terminal(v) for v in value)


class Fixture:
    """Serve the actual SDK protocols; scripted replies still require real MCP calls."""
    def __init__(self):
        self.requests=[]
        self.errors=[]
        self.outer=0
        self.lock=threading.Lock()
        owner=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                try:
                    assert self.path=='/v1/chat/completions'
                    assert self.headers.get('Authorization')=='Bearer phase1-dsh-fixture-key'
                    request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    with owner.lock:owner.requests.append(request)
                    assert request['model']=='phase1-dsh-fixture'
                    if request.get('stream') is True:
                        payload=owner.streaming(request)
                        content_type='text/event-stream'
                    else:
                        prompt=''.join(str(m.get('content','')) for m in request['messages'])
                        content=json.dumps(SKETCH) if '设计专家' in prompt else CODE
                        payload=json.dumps(dict(id='inner-fixture',model=request['model'],
                            choices=[dict(index=0,message=dict(role='assistant',content=content),finish_reason='stop')],
                            usage=dict(prompt_tokens=10,completion_tokens=10,total_tokens=20))).encode()
                        content_type='application/json'
                    self.send_response(200)
                    self.send_header('Content-Type',content_type)
                    self.send_header('Content-Length',str(len(payload)))
                    self.end_headers();self.wfile.write(payload)
                except Exception as exc:
                    owner.errors.append(type(exc).__name__+': '+str(exc))
                    self.send_response(500);self.end_headers()
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.server.daemon_threads=True

    @property
    def base_url(self):return 'http://127.0.0.1:%d/v1'%self.server.server_address[1]

    def streaming(self,request):
        self.outer+=1
        names={t['function']['name'] for t in request.get('tools',[])}
        run='mcp__operator__operator_run'
        status='mcp__operator__operator_status'
        assert {run,status} <= names, sorted(names)
        tool_messages=[m for m in request['messages'] if m.get('role')=='tool']
        if self.outer>1 and tool_messages and _terminal(tool_messages[-1].get('content')):
            delta=dict(content='Observed the delegated terminal status through the operator plugin.')
            finish='stop'
        else:
            assert self.outer<=24,'DSH did not observe a terminal delegated status'
            if self.outer>1:time.sleep(.35)
            name=run if self.outer==1 else status
            delta=dict(tool_calls=[dict(index=0,id='phase1-call-%d'%self.outer,type='function',
                function=dict(name=name,arguments='{}'))])
            finish='tool_calls'
        chunks=[dict(id='outer-%d'%self.outer,model=request['model'],
                     choices=[dict(index=0,delta=delta,finish_reason=None)]),
                dict(id='outer-%d'%self.outer,model=request['model'],
                     choices=[dict(index=0,delta={},finish_reason=finish)],
                     usage=dict(prompt_tokens=10,completion_tokens=10,total_tokens=20))]
        return (''.join('data: '+json.dumps(chunk)+'\n\n' for chunk in chunks)+'data: [DONE]\n\n').encode()

    def __enter__(self):
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start();return self

    def __exit__(self,*args):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=2)


@pytest.mark.skipif(not READY,reason='installed macOS DSH/node/Seatbelt unavailable')
def test_installed_dsh_runs_frozen_operator_and_polls_terminal_through_real_mcp(tmp_path,monkeypatch):
    monkeypatch.setenv('PHASE1_DSH_FIXTURE_KEY','phase1-dsh-fixture-key')
    # Avoid any inherited provider credential or delegation from this workstation.
    for key in ('LLM_API_KEY','LLM_API_KEY_FILE','OPERATOR_AGENT_S5_DEV_CONFIG',
                'OPERATOR_AGENT_S5_DEV_URL','OPERATOR_AGENT_S5_DEV_TOKEN'):
        monkeypatch.delenv(key,raising=False)
    with Fixture() as model:
        config=dict(backend='fake',entry_b='dsh',model=dict(base_url=model.base_url,
            id='phase1-dsh-fixture',api_key_env='PHASE1_DSH_FIXTURE_KEY',transport='local_fixture'),
            tasks=[dict(task_id='matmul',declaration=dict(request='matmul fp32',dsl='python',
                precision='fp32',shapes=[[8,8,8]],min_speedup=0))],
            budget=dict(wall_clock_s=90.,max_llm_calls=30,oracle_total=20),
            max_rounds=1,final_wall_clock_s=30.)
        frozen=tmp_path/'frozen'
        connected_plan.freeze(frozen,config)
        root=tmp_path/'execution'
        summary=connected.run(frozen/'plan.json',root)
    assert not model.errors,model.errors
    cell=root/'cells/matmul-B'
    record=read(cell/'record.json')
    assert record['phase']=='done',dict(record=record,log=(cell/'execution.log').read_text())
    rows=[read(path) for path in sorted((cell/'mcp').glob('rpc-*.json'))]
    assert entry_observed(rows),rows
    requests=[row['request'] for row in rows]
    assert any(r['method']=='initialize' for r in requests)
    assert any(r['method']=='tools/list' for r in requests)
    calls=[row for row in rows if row['request'].get('method')=='tools/call']
    assert sum(row['request']['params']['name']=='operator_run' for row in calls)==1
    assert all(row['request']['params'].get('arguments',{})=={} for row in calls)
    receipt=read(cell/'work/connected-result.json')
    assert receipt['dsh_entry_verified'] is True
    assert receipt['execution_context']['test_only'] is True
    assert receipt['execution_context']['gpu_acceptance'] is False
    final=read(cell/'final/final-result.json')
    assert final['verification_count']==1 and final['result']['levels']['L2'] is True
    assert final['result']['levels']['L3'] is False
    budget=shared.snapshot(cell/'budget.sqlite')
    assert {call['participant'] for call in budget['calls']}=={'outer','inner'}
    assert budget['reserved_calls']==len([r for r in model.requests if r.get('stream')])+2
    assert budget['reserved_calls']<=30 and not budget['aborted']
    assert summary['formal_comparison_credit'] is False
    assert summary['dsh_entry_verified'] is True
    assert next(row for row in summary['cells'] if row['arm']=='B')['dsh_entry_verified'] is True
    assert next(row for row in summary['cells'] if row['arm']=='A')['dsh_entry_verified'] is False
    assert model.outer>=3
    # Both A and B used the real LLMClient; B additionally used the installed DSH.
    assert len([r for r in model.requests if not r.get('stream')])==3
    assert 'phase1-dsh-fixture-key' not in (cell/'dsh/patch.json').read_text()
