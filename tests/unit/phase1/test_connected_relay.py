"""Frozen MCP ingress and credential routing without a provider, SSH or DSH run."""
import copy
import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from benchmarks.experiments.phase1 import connection, relay
from benchmarks.experiments.s5 import s5_plugin_env
from operator_agent.app import mcp_server


def config(tmp_path, backend='fake'):
    return dict(run_dir=str(tmp_path/'cell/work'),backend=backend,
        declaration=dict(request='relu fp32',shapes=[[4,8]],dsl='python' if backend=='fake' else 'triton'),
        model=dict(base_url='http://127.0.0.1:17123/v1',id='frozen-model',api_key_env='PHASE1_TEST_KEY'),
        gpu=dict(index=2,uuid='GPU-frozen',name='Frozen GPU'),
        connection={} if backend=='fake' else dict(host='frozen-host',port=2222,user='worker',
            key_path='/tmp/frozen-key',work_dir='/srv/phase1/task-A',timeout=20),
        budget=dict(wall_clock_s=30,max_llm_calls=4,oracle_total=20),max_rounds=2,
        shared_ledger=str(tmp_path/'cell/budget.sqlite'),shared_task_id='p1c-test',
        deadline_monotonic=130.)


def request(identity, name, arguments=None):
    return dict(jsonrpc='2.0',id=identity,method='tools/call',
        params=dict(name=name,arguments={} if arguments is None else arguments))


def unpack(reply):
    return json.loads(reply['result']['content'][0]['text'])


def test_discovery_is_real_production_handshake_with_only_frozen_tools(tmp_path):
    owner=relay.Relay(config(tmp_path))
    hello=owner.request(dict(jsonrpc='2.0',id=1,method='initialize',params=dict(protocolVersion='2024-11-05')))
    assert hello['result']['serverInfo']['name']==mcp_server.SERVER_NAME
    tools=owner.request(dict(jsonrpc='2.0',id=2,method='tools/list'))['result']['tools']
    assert {tool['name'] for tool in tools}==set(relay.TOOLS)
    for tool in tools:
        if tool['name']=='operator_capabilities':continue
        assert tool['inputSchema']==dict(type='object',properties={},additionalProperties=False)
    retained=sorted((tmp_path/'cell/mcp').glob('rpc-*.json'))
    assert len(retained)==2
    assert json.loads(retained[0].read_text())['reply']==hello


@pytest.mark.parametrize('backend',['fake','ssh'])
def test_real_handler_receives_pinned_arguments_once_and_owned_status(tmp_path,monkeypatch,backend):
    cfg=config(tmp_path,backend)
    original=copy.deepcopy(cfg)
    owner=relay.Relay(cfg)
    monkeypatch.setattr(relay,'monotonic',lambda:105.)
    start=Mock(return_value=dict(ok=True,phase='running',invocation_id='owned-invocation'))
    status=Mock(return_value=dict(ok=True,phase='done'))
    cancel=Mock(return_value=dict(accepted=True))
    monkeypatch.setattr(mcp_server,'run_autonomous',start)
    monkeypatch.setattr(mcp_server,'autonomous_status',status)
    monkeypatch.setattr(mcp_server,'cancel_task',cancel)
    assert unpack(owner.request(request(1,'operator_run')))['invocation_id']=='owned-invocation'
    args,kw=start.call_args
    assert args==(cfg['declaration'],cfg['run_dir'])
    assert kw==dict(backend=backend,max_rounds=2,wall_clock_s=25.,max_llm_calls=4,oracle_total=20,
        **(dict(fake=dict(llm='live')) if backend=='fake' else dict(cfg['connection'],persistent=True)))
    assert 'error' in owner.request(request(2,'operator_run'))
    assert start.call_count==1
    assert unpack(owner.request(request(3,'operator_status')))['phase']=='done'
    assert unpack(owner.request(request(4,'operator_cancel')))['accepted'] is True
    status.assert_called_once_with(run_dir=cfg['run_dir'],invocation_id='owned-invocation')
    cancel.assert_called_once_with(run_dir=cfg['run_dir'],invocation_id='owned-invocation')
    assert cfg==original


@pytest.mark.parametrize('arguments',[
    dict(backend='ssh'),dict(max_llm_calls=30),dict(oracle_total=999),dict(wall_clock_s=9999),
    dict(run_dir='/someone/else'),dict(invocation_id='other'),dict(request='matmul'),
])
def test_agent_cannot_change_any_frozen_run_argument(tmp_path,monkeypatch,arguments):
    owner=relay.Relay(config(tmp_path))
    start=Mock(side_effect=AssertionError('untrusted override reached production'))
    monkeypatch.setattr(mcp_server,'run_autonomous',start)
    result=owner.request(request(1,'operator_run',arguments))
    assert result['error']['code']==-32602
    assert owner.run_requested is False
    start.assert_not_called()
    assert len(owner.records)==1 and owner.records[0]['request']['params']['arguments']==arguments


@pytest.mark.parametrize('name',['operator_status','operator_cancel','operator_verify','operator_repair'])
def test_no_unowned_invocation_or_direct_verification_surface(tmp_path,name):
    result=relay.Relay(config(tmp_path)).request(request(1,name))
    assert result['error']['code']==-32602


def test_expired_run_and_repeated_id_cannot_reset_delegation_budget(tmp_path,monkeypatch):
    owner=relay.Relay(config(tmp_path))
    monkeypatch.setattr(relay,'monotonic',lambda:131.)
    start=Mock(side_effect=AssertionError('expired request reached production'))
    monkeypatch.setattr(mcp_server,'run_autonomous',start)
    assert 'deadline' in owner.request(request(1,'operator_run'))['error']['message']
    assert 'one operator_run' in owner.request(request(1,'operator_run'))['error']['message']
    start.assert_not_called()


@pytest.mark.parametrize('params',[[1],'invalid',42,None])
def test_malformed_initialize_params_return_rpc_refusal_without_crashing(tmp_path,params):
    owner=relay.Relay(config(tmp_path))
    reply=owner.request(dict(jsonrpc='2.0',id=1,method='initialize',params=params))
    assert reply['error']['code']==-32602
    assert 'params' in reply['error']['message']
    assert len(owner.records)==1 and owner.records[0]['reply']==reply
    # Refusing a malformed envelope does not poison the next legitimate handshake.
    assert 'result' in owner.request(dict(jsonrpc='2.0',id=2,method='initialize',params={}))


def test_completed_capabilities_request_id_may_be_reused_by_stdio_proxy(tmp_path):
    owner=relay.Relay(config(tmp_path))
    message=request('capabilities','operator_capabilities')
    first=owner.request(message)
    second=owner.request(message)
    assert unpack(first)['ok'] is True
    assert unpack(second)==unpack(first)
    assert len(owner.records)==2
    assert len(list((tmp_path/'cell/mcp').glob('rpc-*.json')))==2


@pytest.mark.parametrize('backend',['fake','ssh'])
def test_environment_pins_shared_model_and_gpu_then_restores_after_error(tmp_path,monkeypatch,backend):
    cfg=config(tmp_path,backend)
    monkeypatch.setenv('PHASE1_TEST_KEY','fixture-secret-only')
    keys=('LLM_BASE_URL','LLM_MODEL','LLM_API_KEY','OPERATOR_AGENT_EXISTING_REMOTE_ONLY',
        'OPERATOR_AGENT_S5_SHARED_LEDGER','OPERATOR_AGENT_S5_TASK_ID','OPERATOR_AGENT_REQUIRED_GPU',
        'OPERATOR_AGENT_REQUIRED_GPU_UUID','OPERATOR_AGENT_REQUIRED_GPU_NAME')
    for key in keys:monkeypatch.setenv(key,'previous-'+key)
    prior={key:os.environ[key] for key in keys}
    with pytest.raises(RuntimeError,match='controlled failure'):
        with connection.environment(cfg):
            assert os.environ['LLM_API_KEY']=='fixture-secret-only'
            assert os.environ['LLM_BASE_URL']==cfg['model']['base_url']
            assert os.environ['LLM_MODEL']=='frozen-model'
            assert os.environ['OPERATOR_AGENT_S5_SHARED_LEDGER']==cfg['shared_ledger']
            assert os.environ['OPERATOR_AGENT_S5_TASK_ID']==cfg['shared_task_id']
            assert os.environ['OPERATOR_AGENT_EXISTING_REMOTE_ONLY']=='1'
            if backend=='ssh':
                assert os.environ['OPERATOR_AGENT_REQUIRED_GPU']=='2'
                assert os.environ['OPERATOR_AGENT_REQUIRED_GPU_UUID']=='GPU-frozen'
                assert os.environ['OPERATOR_AGENT_REQUIRED_GPU_NAME']=='Frozen GPU'
            else:
                assert not any(key in os.environ for key in keys[-3:])
            raise RuntimeError('controlled failure')
    assert {key:os.environ[key] for key in keys}==prior


def test_environment_refuses_inherited_delegation_and_missing_credentials(tmp_path,monkeypatch):
    cfg=config(tmp_path)
    monkeypatch.delenv('PHASE1_TEST_KEY',raising=False)
    with pytest.raises(ValueError,match='credential'):
        with connection.environment(cfg):pass
    monkeypatch.setenv('PHASE1_TEST_KEY','fixture-only')
    monkeypatch.setenv('OPERATOR_AGENT_S5_DEV_CONFIG','/unread/delegation.json')
    with pytest.raises(ValueError,match='inherited'):
        with connection.environment(cfg):pass


def test_real_dsh_plugin_builder_routes_only_gateway_placeholder_to_child(tmp_path,monkeypatch):
    # Inspect the real builder; skip only the OS sandbox profile generation and execution.
    monkeypatch.setattr(s5_plugin_env,'configure',lambda built,runtime:None)
    monkeypatch.setenv('LLM_API_KEY','private-provider-secret')
    monkeypatch.setenv('PHASE1_TEST_KEY','private-provider-secret')
    runtime=s5_plugin_env.stage_runtime(tmp_path/'runtime')
    capability=Mock()
    capability.client_config.return_value=dict(url='http://127.0.0.1:1234/mcp',token='relay-only',timeout=30)
    built=s5_plugin_env.build('C',runtime=runtime,relay=capability,
        base_url='http://127.0.0.1:1235/v1',arm_dir=tmp_path/'dsh',model='frozen-model',
        max_tokens=4096,context_window=65536,plugin_tool_timeout_ms=30000)
    assert built['surface']['plugin'] and built['surface']['mcp']
    assert not built['surface']['core_cli'] and not built['surface']['core_source']
    assert built['env']['S5_FAKE_API_KEY']=='placeholder'
    assert 'LLM_API_KEY' not in built['env'] and 'PHASE1_TEST_KEY' not in built['env']
    serialized=json.dumps(dict(env=built['env'],patch=built['patch']))
    assert 'private-provider-secret' not in serialized
    assert 'http://127.0.0.1:1235/v1' in serialized
    assert 'dsh-operator' not in built['env'].get('PYTHONPATH','')
    client=json.loads(Path(built['mcp_client_config']).read_text())
    assert client==capability.client_config.return_value
