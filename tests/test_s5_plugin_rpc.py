"""The public bridge forwards the actual core server without exposing its code."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks import s5_plugin_proxy as proxy, s5_runtime, s5_verify_remote
from benchmarks.s5_plugin_rpc import MCPRelay
from benchmarks.s5_development_service import DevelopmentService
from benchmarks.s5_vertical_validation import BAD_FIXTURE, GOOD_FIXTURE
from operator_agent.hitl_tools import prepare

DECL = dict(request='layernorm fp32', shapes=[[7,33]], hardware='mock_gpu',
            dsl='python', precision='fp32')


def rpc(identity, method, **params):
    return dict(jsonrpc='2.0', id=identity, method=method, params=params)


def tool(identity, name, **arguments):
    return rpc(identity, 'tools/call', name=name, arguments=arguments)


@pytest.fixture
def relay(tmp_path):
    if not s5_runtime.available():pytest.skip('frozen core dependencies unavailable')
    sha=s5_verify_remote.freeze(tmp_path/'plan',declaration=DECL,backend='mock')
    task=tmp_path/'participant/work/task';prepare(DECL,task)
    with DevelopmentService(tmp_path/'plan/plan.json',sha,tmp_path/'verification',arm='C')as svc:
        conf=dict(url=svc.url,token=svc.token,contract=svc.plan['contract_hash'],backend='mock',timeout=70)
        r=MCPRelay(conf,task,DECL,tmp_path/'relay',timeout=80)
        with r:
            config=tmp_path/'proxy.json';config.write_text(json.dumps(r.client_config()))
            yield r,svc,config
        assert r.proc.poll() is not None
        assert r.snapshot()['closed'] and r.snapshot()['cleanup']=='direct_child_reaped_descendants_unknown'
        assert not(tmp_path/'relay/development-client.json').exists()


def test_actual_schema_instructions_and_immutable_feedback(relay):
    r,svc,config=relay
    init=proxy.forward(config,rpc(1,'initialize',protocolVersion='2025-11-25',capabilities={},clientInfo=dict(name='fixture',version='1')))
    assert init['result']['serverInfo']['name']=='operator'
    assert 'S5 supervised development experiment' in init['result']['instructions']
    assert proxy.forward(config,dict(jsonrpc='2.0',method='notifications/initialized'))is None
    definitions=proxy.forward(config,rpc(2,'tools/list'))['result']['tools']
    assert {t['name']for t in definitions}=={'operator_prepare','operator_verify','operator_capabilities'}
    for number,source in enumerate([BAD_FIXTURE,GOOD_FIXTURE],3):
        candidate=r.workspace/'candidate.py';candidate.write_text(source)
        reply=proxy.forward(config,tool(number,'operator_verify',run_dir=str(r.task_dir),
                                       code_file='candidate.py',stop_after='L2',backend='mock'))
        result=json.loads(reply['result']['content'][0]['text'])
        assert result['status']==('failed'if number==3 else'passed')
        assert all(result.get(k)==v for k,v in svc.snapshot()['records'][number-3]['result'].items())
    assert svc.snapshot()['development_used']==2 and svc.snapshot()['final_used']==0
    denied=proxy.forward(config,tool(5,'operator_verify',run_dir=str(r.task_dir),code=GOOD_FIXTURE))
    assert json.loads(denied['result']['content'][0]['text'])['error']=='development_budget_exhausted'
    assert svc.final(GOOD_FIXTURE)['status']=='passed'
    # Per-run snapshots contain hashes and routing labels, not capabilities or source.
    snap=json.dumps(r.snapshot())
    assert r.token not in snap and svc.token not in snap and GOOD_FIXTURE not in snap
    assert r.snapshot()['real_core_server']is True


@pytest.mark.parametrize('name,args',[
    ('operator_run',{}),('operator_repair',{}),('operator_bench',{}),
    ('operator_verify',dict(run_dir='/private',code='x')),
    ('operator_verify',dict(run_dir='task',code='x',host='host')),
    ('operator_verify',dict(run_dir='task',code='x',backend='ssh')),
    ('operator_verify',dict(run_dir='task',code='x',stop_after='L3')),
    ('operator_prepare',dict(run_dir='task',request='changed',shapes=[[1,2]])),
])
def test_refused_calls_never_use_development_quota(relay,name,args):
    r,svc,config=relay
    reply=proxy.forward(config,tool(1,name,**args))
    assert reply['error']['code']==-32602
    assert svc.snapshot()['development_used']==0


def test_candidate_paths_cannot_read_private_or_linked_files(relay,tmp_path):
    r,svc,config=relay
    private=tmp_path/'private.py';private.write_text(GOOD_FIXTURE)
    alias=r.workspace/'alias.py';alias.symlink_to(private)
    local=r.workspace/'source.py';local.write_text(GOOD_FIXTURE)
    local_alias=r.workspace/'local-alias.py';local_alias.symlink_to(local)
    hard=r.workspace/'hard.py';os.link(private,hard)
    for name in (str(private),'../private.py','alias.py','local-alias.py','hard.py'):
        reply=proxy.forward(config,tool(name,'operator_verify',run_dir='task',code_file=name))
        assert reply['error']['code']==-32602
    assert svc.snapshot()['development_used']==0


def test_proxy_stdio_and_real_op_capabilities(relay):
    r,svc,config=relay
    commands=[rpc(1,'initialize',protocolVersion='2025-11-25',capabilities={},clientInfo=dict(name='fixture',version='1')),
              dict(jsonrpc='2.0',method='notifications/initialized'),rpc(2,'tools/list')]
    proc=subprocess.run([sys.executable,str(Path(proxy.__file__)),'--config',str(config),'--stdio'],
        input=''.join(json.dumps(x)+'\n'for x in commands),capture_output=True,text=True,timeout=30)
    assert proc.returncode==0 and not proc.stderr
    replies=[json.loads(x)for x in proc.stdout.splitlines()]
    assert [x['id']for x in replies]==[1,2]
    cap=subprocess.run([sys.executable,str(Path(proxy.__file__)),'--config',str(config),'capabilities'],
        capture_output=True,text=True,timeout=30)
    assert cap.returncode==0 and json.loads(cap.stdout)['ok']is True
    assert svc.snapshot()['development_used']==0


@pytest.mark.parametrize('url',['https://example.com/mcp','http://localhost:1/mcp',
    'http://127.0.0.1:1/final','http://a:b@127.0.0.1:1/mcp','http://127.0.0.1:1/mcp?token=x'])
def test_configuration_cannot_change_target(tmp_path,url):
    path=tmp_path/'config.json';path.write_text(json.dumps(dict(url=url,token='synthetic-only-token',timeout=1)))
    with pytest.raises(ValueError):proxy.configuration(path)


def test_prepared_declaration_is_forwarded_to_real_core(relay):
    r,svc,config=relay
    reply=proxy.forward(config,tool(1,'operator_prepare',run_dir='task',**DECL))
    result=json.loads(reply['result']['content'][0]['text'])
    assert result['ok'] and result['contract_hash']==svc.plan['contract_hash']
    assert svc.snapshot()['development_used']==0
