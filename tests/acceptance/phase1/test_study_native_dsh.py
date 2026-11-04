"""Actual native DSH A and delegated DSH B share frozen wire controls.

All HTTP is IPv4 loopback and all execution uses CPU FakeBackend. Scripted
model output still has to drive real bash/check.py and MCP production tools.
"""
from collections import Counter
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import threading
import time

import pytest

from benchmarks.experiments.phase1 import connected, connected_plan, sampling
from benchmarks.experiments.phase1.plan import read
from benchmarks.experiments.s5 import s5_arm_env, s5_seatbelt
from operator_agent.execute import s5_shared_budget as shared

CODE='def kernel(A, B):\n    return A @ B\n'
EXPECTED_WIRE=dict(temperature=.1,max_tokens=32768,n=1,
                   enable_thinking=True,reasoning_effort='medium')
CELL_WALL_CLOCK_S=180.0
SKETCH=dict(algorithm='matmul',tiling=dict(block_m=64,block_n=64,block_k=32),
            memory_plan={},pipeline=[],optimization_hints=[])
READY=(sys.platform=='darwin' and bool(shutil.which('node')) and
       (s5_arm_env.DSH_ROOT/'dsh/lib/bin.js').is_file() and
       s5_arm_env.MCP_CLIENT.is_file() and Path(s5_seatbelt.SANDBOX).is_file())


def terminal(value):
    if isinstance(value,str):
        try:return terminal(json.loads(value))
        except (ValueError,TypeError):return False
    if isinstance(value,dict):
        if 'phase' in value:return value['phase'] in ('done','failed','cancelled')
        return terminal(value.get('content',value.get('text')))
    return isinstance(value,list) and any(terminal(v) for v in value)


class Provider:
    def __init__(self,execution_dir):
        self.requests=[];self.request_hashes=[];self.errors=[];self.native=0;self.delegated=0
        self.execution_dir=execution_dir
        self.host_idle_timeouts={}
        owner=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                try:
                    assert self.path=='/compatible-mode/v1/chat/completions'
                    assert self.headers.get('Authorization')=='Bearer study-synthetic-key'
                    body=self.rfile.read(int(self.headers['Content-Length']))
                    request=json.loads(body)
                    owner.requests.append(request)
                    owner.request_hashes.append(hashlib.sha256(body).hexdigest())
                    assert request['model']=='local-study'
                    assert sampling.matches(request,sampling.policy()),request
                    for key,value in EXPECTED_WIRE.items():
                        assert type(request[key]) is type(value) and request[key]==value,request
                    if request.get('stream'):
                        payload=owner.stream(request);kind='text/event-stream'
                    else:
                        prompt=''.join(str(m.get('content','')) for m in request['messages'])
                        text=json.dumps(SKETCH) if '设计专家' in prompt else CODE
                        payload=json.dumps(dict(model=request['model'],choices=[dict(index=0,
                            message=dict(role='assistant',content=text),finish_reason='stop')],
                            usage=dict(prompt_tokens=10,completion_tokens=10,total_tokens=20,
                                       prompt_tokens_details=dict(cached_tokens=1)))).encode()
                        kind='application/json'
                    self.send_response(200);self.send_header('Content-Type',kind)
                    self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
                except Exception as exc:
                    owner.errors.append(type(exc).__name__+': '+str(exc))
                    self.send_response(500);self.end_headers()
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.server.daemon_threads=True

    @property
    def base_url(self):return 'http://127.0.0.1:%d/compatible-mode/v1'%self.server.server_port

    def stream(self,request):
        names={t['function']['name'] for t in request.get('tools',[])}
        tools=[m for m in request['messages'] if m.get('role')=='tool']
        delegated='mcp__operator__operator_run' in names
        arm='B' if delegated else 'A'
        if arm not in self.host_idle_timeouts:
            # The native verifier removes its private DSH patch on exit.
            # Inspect the exact launch patch while the host is awaiting a reply.
            patch=read(self.execution_dir/'cells'/f'matmul-{arm}'/'dsh/patch.json')
            llm=next(entry for entry in patch if entry.get('id')=='llm-pi-ai')
            provider=llm['config']['providers']['fixture']
            model=next(model for model in provider['models'] if model['id']=='local-study')
            assert model['maxTokens']==EXPECTED_WIRE['max_tokens']
            idle_timeout=provider['streamIdleTimeoutMs']
            assert type(idle_timeout) is int and idle_timeout>0
            self.host_idle_timeouts[arm]=idle_timeout
        if delegated:
            self.delegated+=1;sequence=self.delegated
            assert 'mcp__operator__operator_status' in names
            if tools and terminal(tools[-1].get('content')):
                delta=dict(content='Observed the delegated terminal state.');finish='stop'
            else:
                assert sequence<=24,'delegated task did not terminate'
                if sequence>1:time.sleep(.3)
                tool='mcp__operator__operator_run' if sequence==1 else 'mcp__operator__operator_status'
                delta=dict(tool_calls=[dict(index=0,id='study-b-%d'%sequence,type='function',
                    function=dict(name=tool,arguments='{}'))]);finish='tool_calls'
        else:
            self.native+=1;sequence=self.native
            assert 'bash' in names
            assert not any(name.startswith('mcp__operator__') for name in names)
            if sequence==1:
                script=('from pathlib import Path;import subprocess,sys;'
                        'Path("candidate.py").write_text('+repr(CODE)+');'
                        'subprocess.run([sys.executable,"check.py","--code-file","candidate.py"],check=True)')
                command=shlex.quote(str(Path(sys.executable).resolve()))+' -c '+shlex.quote(script)
                args=dict(command=command,description='Write the native candidate and invoke the public verification client')
                delta=dict(tool_calls=[dict(index=0,id='study-a-1',type='function',
                    function=dict(name='bash',arguments=json.dumps(args)))]);finish='tool_calls'
            else:
                assert sequence==2 and tools,'native source must be verified through actual bash'
                assert 'full_l2_passed' in str(tools[-1]['content']),tools[-1]
                delta=dict(content='Native candidate submitted to the full verifier.');finish='stop'
        chunks=[dict(id='study-stream',model=request['model'],choices=[dict(index=0,delta=delta,finish_reason=None)]),
                dict(id='study-stream',model=request['model'],choices=[dict(index=0,delta={},finish_reason=finish)])]
        return (''.join('data: '+json.dumps(c)+'\n\n' for c in chunks)+'data: [DONE]\n\n').encode()

    def __enter__(self):
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start();return self

    def __exit__(self,*args):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=2)


@pytest.mark.skipif(not READY,reason='installed macOS DSH/node/Seatbelt unavailable')
def test_real_native_and_delegated_dsh_use_the_same_pre_reservation_wire_policy(tmp_path,monkeypatch):
    for key in tuple(os.environ):
        if key.startswith(('LLM_','OPERATOR_AGENT_S5_DEV_','OPERATOR_AGENT_PHASE1_')):
            monkeypatch.delenv(key,raising=False)
    monkeypatch.setenv('STUDY_FIXTURE_KEY','study-synthetic-key')
    with Provider(tmp_path/'execution') as model:
        cfg=dict(backend='fake',entry_a='dsh_native',entry_b='dsh',sampling=sampling.policy(),
            model=dict(base_url=model.base_url,id='local-study',api_key_env='STUDY_FIXTURE_KEY',
                transport='local_fixture',temperature=.1),
            tasks=[dict(task_id='matmul',declaration=dict(request='matmul fp32',precision='fp32',
                dsl='python',shapes=[[8,8,8]],min_speedup=0))],max_rounds=1,
            budget=dict(wall_clock_s=CELL_WALL_CLOCK_S,max_llm_calls=30,oracle_total=20),final_wall_clock_s=30.)
        connected_plan.freeze(tmp_path/'frozen',cfg)
        result=connected.run(tmp_path/'frozen/plan.json',tmp_path/'execution')
    assert not model.errors,model.errors
    assert model.native==2 and model.delegated>=3
    assert model.host_idle_timeouts==dict(A=180000,B=180000)
    assert len([r for r in model.requests if not r.get('stream')])==2
    assert all(sampling.matches(r,sampling.policy()) for r in model.requests)
    assert result['phase']=='done' and len(result['cells'])==2
    dispatched_hashes=[]
    for row in result['cells']:
        assert row['phase']=='done',row
        assert row['budget_auditable'] is True,row['problems']
        assert row['model_budget']['sampling_aligned'] is True,row
        assert row['final_evaluation']['complete'] is True,row['problems']
        assert row['final_evaluation']['full_l2_passed'] is True,row
        assert row['final_evaluation']['levels']['L3'] is False
        assert row['events']['count'] is None
        cell=tmp_path/'execution/cells'/row['cell_id']
        audit=shared.snapshot(cell/'budget.sqlite')
        assert audit['reserved_calls']<=30 and not audit['aborted']
        assert all(c['output_tokens_reserved']==EXPECTED_WIRE['max_tokens'] for c in audit['calls'])
        assert all(c['dispatched_monotonic'] is not None for c in audit['calls'])
        participants={'outer'} if row['cell_id']=='matmul-A' else {'outer','inner'}
        assert {c['participant'] for c in audit['calls']}==participants
        for call in audit['calls']:
            if call['participant']=='inner':
                assert call['usage_evidence']=='provider'
                assert call['api_usage']['cache_read_tokens']==1
        dispatched_hashes.extend(c['request_sha256'] for c in audit['calls'])
        source_hash=hashlib.sha256((cell/'selected.py').read_bytes()).hexdigest()
        assert row['final_evaluation']['source_hash']==source_hash
        assert read(cell/'final/final-result.json')['source_hash']==source_hash
        assert 'study-synthetic-key' not in (cell/'host-surface.json').read_text()
    assert Counter(dispatched_hashes)==Counter(model.request_hashes)
    native=tmp_path/'execution/cells/matmul-A'
    assert not (native/'mcp').exists()
    assert read(native/'host-surface.json')['ordinary_shell_edit'] is True
    assert read(native/'host-surface.json')['source_plugin'] is False
    assert read(native/'work/connected-result.json')['entry_a']=='dsh_native'
    assert len(list((native/'work/native-verification').glob('submission-*.json')))==1
    delegated=tmp_path/'execution/cells/matmul-B/work'
    autonomous_config=read(delegated/'autonomous.json')
    assert autonomous_config['llm_timeout']==CELL_WALL_CLOCK_S
    inner_identity=read(delegated/'model-journal/identity.json')
    expected_inner_timeout=min(autonomous_config['llm_timeout'],
                               autonomous_config['task_budget']['wall_clock_s'])
    assert inner_identity['timeout']==expected_inner_timeout>120
    assert result['execution_context']['test_only'] is True and result['gpu_acceptance'] is False
    assert result['formal_comparison_credit'] is False
