"""Supervised discovery must describe the delegated contract, not local SSH."""
import json

import pytest

from operator_agent.app import mcp_server as mcp
from operator_agent.execute import s5_verify_client as client


@pytest.mark.parametrize('content', ['Backend="ssh"; use existing task.',
    [{'type':'text','text':'Backend="ssh"; use existing task.'}]])
def test_instruction_observation_does_not_confuse_json_escaping_with_missing_context(content):
    from benchmarks.experiments.s5.s5_plugin_probe import contains_message_text
    messages=[dict(role='system',content=content)]
    assert contains_message_text(messages,'Backend="ssh"; use existing task.')
    assert not contains_message_text(messages,'Backend="mock"')


@pytest.fixture
def configured(monkeypatch):
    for name in tuple(client.KEYS) + ('CONFIG',):
        monkeypatch.delenv(client.PREFIX + name, raising=False)
    for name, value in dict(URL='http://127.0.0.1:9/development/verify', TOKEN='SYNTHETIC_ONLY',
        CONTRACT='a'*64, BACKEND='ssh', TIMEOUT='610').items():
        monkeypatch.setenv(client.PREFIX + name, value)


def test_supervised_schema_matches_fixed_backend_and_relay_allowed_options(configured):
    definitions={d['name']:d for d in mcp._tool_definitions()}
    assert set(definitions)=={'operator_prepare','operator_verify','operator_capabilities'}
    verify=definitions['operator_verify']; properties=verify['inputSchema']['properties']
    assert set(properties)=={'run_dir','code','code_file','candidate_origin','backend','stop_after','timeout'}
    assert properties['backend']['enum']==['ssh'] and properties['backend']['default']=='ssh'
    assert properties['stop_after']['enum']==['L2']
    assert properties['timeout']['maximum']==610
    assert 'already prepared' in verify['description']


def test_capabilities_distinguish_configured_delegation_from_local_environment(configured):
    body=json.loads(mcp._call_tool('operator_capabilities', {})['content'][0]['text'])
    assert body['task_prepared'] is True and body['mode']=='supervised_development'
    assert set(body['backends'])=={'ssh'}
    backend=body['backends']['ssh']
    assert backend['configured'] is True and backend['delegated'] is True
    assert backend['requires']==[] and backend['connectivity_verified'] is False
    assert 'env_present' not in backend
    assert 'local_runtime' in body and 'runtime' not in body
    assert client.configuration()[0]['TOKEN'] not in json.dumps(body)
    assert client.configuration()[0]['URL'] not in json.dumps(body)


def test_initialize_and_schema_give_same_connection_rules(configured):
    reply=mcp.handle_message(dict(jsonrpc='2.0',id=1,method='initialize',params={}))
    instructions=reply['result']['instructions']
    assert 'already prepared' in instructions and 'Do not call operator_prepare again' in instructions
    assert 'backend="ssh"' in instructions and 'Do not provide host' in instructions
    assert 'not a connectivity probe' in instructions


def test_supervised_verify_omission_uses_advertised_backend_and_level(configured, monkeypatch):
    observed={}
    def verify(run_dir, code, **options):
        observed.update(options)
        return dict(ok=True)
    monkeypatch.setattr(mcp, 'verify', verify)
    mcp._call_tool('operator_verify',dict(run_dir='/prepared/task',code='def kernel(X): return X'))
    assert observed['backend']=='ssh' and observed['stop_after']=='L2'
    assert 'host' not in observed


def test_direct_mode_keeps_original_ssh_contract(monkeypatch):
    for name in tuple(client.KEYS)+('CONFIG',):monkeypatch.delenv(client.PREFIX+name,raising=False)
    definition=next(d for d in mcp._tool_definitions() if d['name']=='operator_verify')
    properties=definition['inputSchema']['properties']
    assert properties['backend']['default']=='mock' and 'host' in properties
    assert properties['stop_after']['enum']==['L1','L2','L3']
    body=json.loads(mcp._call_tool('operator_capabilities',{})['content'][0]['text'])
    assert body['backends']['ssh']['requires'] and 'runtime' in body


def test_partial_supervised_config_refuses_instead_of_advertising_direct_ssh(monkeypatch):
    for name in tuple(client.KEYS)+('CONFIG',):monkeypatch.delenv(client.PREFIX+name,raising=False)
    monkeypatch.setenv(client.PREFIX+'URL','http://127.0.0.1:9/development/verify')
    with pytest.raises(ValueError):mcp._tool_definitions()
    with pytest.raises(ValueError):mcp._supervised_capabilities()
