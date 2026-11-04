"""Freeze live integration inputs without executing a model, host or GPU."""
from copy import deepcopy
import json
from unittest.mock import Mock

import pytest

from benchmarks.experiments.phase1 import connected_plan as plans
from operator_agent.core.run_contract import digest


@pytest.fixture(autouse=True)
def stable_implementation(monkeypatch):
    # Other workers may be finishing the adapter while these tests execute.
    identity = dict(protocol={'fixture.py': 'a'*64}, harness={'plan.py': 'b'*64})
    monkeypatch.setattr(plans, 'implementation_identity', lambda: deepcopy(identity))


def config():
    return dict(backend='fake', entry_b='core', model=dict(
        base_url='http://127.0.0.1:43210/v1', id='local-model',
        api_key_env='PHASE1_TEST_KEY', temperature=.2, transport='local_fixture'),
        tasks=[dict(task_id='relu', declaration=dict(
            request='relu fp32', shapes=[[4, 8]], precision='fp32', dsl='python'))],
        budget=dict(wall_clock_s=30., max_llm_calls=4, oracle_total=20),
        max_rounds=2, final_wall_clock_s=20.)


def ssh_config(tmp_path):
    value = config()
    value.update(backend='ssh', connection=dict(host='gpu-host', port=2222,
        user='worker', work_dir='/tmp/phase1', timeout=120),
        gpu=dict(index=0, uuid='GPU-fixture', name='Fixture GPU'),
        probe=str(tmp_path/'card.json'))
    value['model'].update(base_url='https://model.example/v1', transport='live')
    value['tasks'][0]['declaration']['dsl'] = 'triton'
    return value


def resign(value):
    value['hash'] = digest({key: item for key, item in value.items() if key != 'hash'})
    return value


def test_freezing_live_configuration_is_offline_and_does_not_read_credentials(tmp_path, monkeypatch):
    from benchmarks.experiments.phase1 import connection
    from operator_agent.execute import llm_client
    import socket
    forbidden = Mock(side_effect=AssertionError('freeze used runtime credentials or network'))
    monkeypatch.setattr(connection, 'credential', forbidden)
    monkeypatch.setattr(llm_client, '_resolve_api_key', forbidden)
    monkeypatch.setattr(socket, 'socket', forbidden)
    monkeypatch.setenv('PHASE1_TEST_KEY', 'secret-that-must-never-be-serialized')
    cfg = config()
    cfg['model'].update(base_url='https://model.example/v1', transport='live')
    frozen = plans.freeze(tmp_path/'frozen', cfg)
    assert plans.load(tmp_path/'frozen/plan.json') == frozen
    forbidden.assert_not_called()
    serialized = (tmp_path/'frozen/plan.json').read_text()
    assert 'secret-that-must-never-be-serialized' not in serialized
    assert frozen['model']['api_key_env'] == 'PHASE1_TEST_KEY'
    assert frozen['execution_context']['test_only'] is True  # fake GPU still disqualifies it
    assert frozen['formal_comparison_credit'] is False
    assert frozen['dsh_entry_verified'] is False


def test_frozen_plan_is_deterministic_and_retains_derived_contract_and_interleaved_slots(tmp_path):
    cfg = config()
    second = deepcopy(cfg['tasks'][0])
    second['task_id'] = 'relu-small'
    second['declaration']['shapes'] = [[2, 4]]
    cfg['tasks'].append(second)
    first = plans.freeze(tmp_path/'one', cfg)
    assert first == plans.freeze(tmp_path/'two', cfg)
    assert [slot['arm'] for slot in first['schedule']] == ['A', 'B', 'B', 'A']
    assert len({row['cell_id'] for row in first['schedule']}) == 4
    assert all(len(row['contract_hash']) == 64 for row in first['tasks'])
    assert first['model']['protocol'] == 'chat_completions'
    assert len(first['model']['endpoint_sha256']) == 64
    assert plans.validate(first) == first


@pytest.mark.parametrize('update', [
    {'connection': {'host': 'remote'}}, {'gpu': {'index': 0}}, {'probe': '/tmp/probe.json'}])
def test_fake_backend_cannot_hide_remote_gpu_options(tmp_path, update):
    cfg = config(); cfg.update(update)
    with pytest.raises(ValueError, match='fake backend refuses'):
        plans.freeze(tmp_path/'invalid', cfg)
    assert not (tmp_path/'invalid').exists()


@pytest.mark.parametrize('update', [
    {'api_key': 'do-not-save'}, {'base_url': 'https://key@model.example/v1'},
    {'base_url': 'https://model.example/v1?key=value'},
    {'base_url': 'https://model.example/v1#fragment'},
    {'base_url': 'https://model.example/other'}, {'base_url': 'file:///v1'},
    {'base_url': 'http://127.0.0.1:99999/v1'}, {'temperature': float('nan')},
    {'temperature': float('inf')}, {'temperature': True}, {'temperature': -1},
    {'temperature': 2.1}, {'api_key_env': 'key with spaces'}, {'api_key_env': 'LLM_API_KEY\n'},
    {'transport': 'silently-fake'}, {'id': 'model\nother'}])
def test_invalid_model_bindings_are_refused_before_creating_a_plan(tmp_path, update):
    cfg = config(); cfg['model'].update(update)
    with pytest.raises(ValueError):
        plans.freeze(tmp_path/'invalid', cfg)
    assert not (tmp_path/'invalid').exists()


@pytest.mark.parametrize('endpoint', ['https://127.0.0.1/v1', 'http://localhost/v1',
    'http://[::1]/v1', 'https://model.example/v1'])
def test_local_fixture_requires_explicit_ipv4_loopback_http(tmp_path, endpoint):
    cfg=config(); cfg['model']['base_url']=endpoint
    with pytest.raises(ValueError, match='IPv4 loopback'):
        plans.freeze(tmp_path/'invalid',cfg)


@pytest.mark.parametrize('field,value', [
    ('wall_clock_s', 2401), ('max_llm_calls', 31), ('oracle_total', 21),
    ('wall_clock_s', True), ('max_llm_calls', True), ('oracle_total', 0)])
def test_common_budget_cannot_exceed_phase1_or_use_ambiguous_types(tmp_path, field, value):
    cfg=config(); cfg['budget'][field]=value
    with pytest.raises(ValueError):
        plans.freeze(tmp_path/'invalid',cfg)


@pytest.mark.parametrize('field,value', [
    ('max_rounds', 0), ('max_rounds', 9), ('max_rounds', True),
    ('final_wall_clock_s', 661), ('final_wall_clock_s', True)])
def test_round_and_final_evaluation_bounds_are_frozen(tmp_path, field, value):
    cfg=config(); cfg[field]=value
    with pytest.raises(ValueError):
        plans.freeze(tmp_path/'invalid',cfg)


def test_digest_detects_changed_model_without_resigning(tmp_path):
    frozen=plans.freeze(tmp_path/'frozen',config())
    frozen['model']['id']='different-model'
    with pytest.raises(ValueError,match='hash mismatch'):
        plans.validate(frozen)


@pytest.mark.parametrize('tamper', ['model_protocol', 'model_endpoint_hash', 'model_secret',
    'context_test_only', 'context_gpu', 'context_performance', 'formal_credit', 'dsh_credit',
    'sampling', 'plan_id', 'target_plan_id', 'entry', 'contract', 'schedule', 'task_extra', 'support'])
def test_resigning_cannot_promote_or_weaken_frozen_integration_claims(tmp_path, tamper):
    frozen=plans.freeze(tmp_path/'frozen',config())
    if tamper=='model_protocol': frozen['model']['protocol']='responses'
    elif tamper=='model_endpoint_hash': frozen['model']['endpoint_sha256']='f'*64
    elif tamper=='model_secret': frozen['model']['api_key']='secret'
    elif tamper=='context_test_only': frozen['execution_context']['test_only']=False
    elif tamper=='context_gpu': frozen['execution_context']['gpu_acceptance']=True
    elif tamper=='context_performance': frozen['execution_context']['performance_evidence']=True
    elif tamper=='formal_credit': frozen['formal_comparison_credit']=True
    elif tamper=='dsh_credit': frozen['dsh_entry_verified']=True
    elif tamper=='sampling': frozen['sampling_comparability']='established'
    elif tamper=='plan_id': frozen['plan_id']='phase1-agent-comparison-v2'
    elif tamper=='target_plan_id': frozen['target_plan_id']='unreviewed-study'
    elif tamper=='entry': frozen['entries']['B']='operator_agent.app.cli'
    elif tamper=='contract': frozen['tasks'][0]['contract_hash']='f'*64
    elif tamper=='schedule': frozen['schedule'].pop()
    elif tamper=='task_extra': frozen['tasks'][0]['max_llm_calls']=30
    elif tamper=='support': frozen['support']['released']=True
    with pytest.raises(ValueError):
        plans.validate(resign(frozen))


def test_implementation_drift_blocks_dispatch_but_allows_read_only_replay(tmp_path,monkeypatch):
    frozen=plans.freeze(tmp_path/'frozen',config())
    monkeypatch.setattr(plans,'implementation_identity',lambda: {'changed':True})
    with pytest.raises(ValueError,match='implementation drift'):
        plans.validate(frozen)
    assert plans.validate(frozen,check_implementation=False)==frozen


def test_dsh_runtime_identity_is_frozen_and_checked_without_starting_host(tmp_path,monkeypatch):
    cfg=config(); cfg['entry_b']='dsh'
    monkeypatch.setattr(plans,'host_identity',lambda entry: {'dsh.js':'a'*64})
    frozen=plans.freeze(tmp_path/'frozen',cfg)
    assert frozen['host_runtime']=={'dsh.js':'a'*64}
    monkeypatch.setattr(plans,'host_identity',lambda entry: {'dsh.js':'b'*64})
    with pytest.raises(ValueError,match='DSH runtime drift'):
        plans.validate(frozen)


@pytest.mark.parametrize('part,field,value', [
    ('connection','host','-oProxyCommand=command'), ('connection','host','gpu; command'),
    ('connection','user','-option'), ('connection','port',0), ('connection','port',65536),
    ('connection','port',True), ('connection','work_dir','/'),
    ('connection','work_dir','relative/path'), ('connection','work_dir','/tmp/../run'),
    ('connection','work_dir','/tmp/run\nother'), ('connection','timeout',0),
    ('connection','timeout',661), ('connection','timeout',float('nan')),
    ('connection','timeout',True), ('connection','key_path','bad\nkey'),
    ('gpu','index',True), ('gpu','index',-1), ('gpu','uuid',''), ('gpu','name','GPU\nother')])
def test_ssh_requires_explicit_safe_connection_and_gpu_identity(tmp_path,part,field,value):
    cfg=ssh_config(tmp_path); cfg[part][field]=value
    with pytest.raises(ValueError):
        plans.freeze(tmp_path/'invalid',cfg)


def test_ssh_requires_a_measured_probe_from_the_pinned_card(tmp_path):
    cfg=ssh_config(tmp_path)
    cfg.pop('probe')
    with pytest.raises(ValueError,match='measured card probe'):
        plans.freeze(tmp_path/'missing',cfg)
    cfg=ssh_config(tmp_path)
    (tmp_path/'card.json').write_text(json.dumps({'device': {'uuid':'GPU-other'}}))
    with pytest.raises(ValueError,match='UUID differs'):
        plans.freeze(tmp_path/'wrong-card',cfg)
    (tmp_path/'card.json').write_text(json.dumps({'device': {'uuid':'GPU-fixture'}}))
    with pytest.raises(ValueError,match='five card gates'):
        plans.freeze(tmp_path/'unmeasured',cfg)


def measured_probe():
    # The values model outputs already collected on one card. No probe is run here.
    uuid='GPU-fixture'
    return dict(device=dict(uuid=uuid,name='Fixture GPU',cc='8.9',total_memory_bytes=12*2**30),
        launch_floor=dict(median_us=11.4,samples=200,source_uuid=uuid),
        shared_memory=dict(per_block_optin_bytes=99*1024,per_sm_kb=100,source_uuid=uuid),
        peak=dict(mem_bw_gbs=504.,peak_by_dtype=dict(fp16=94.2),trusted=True,source_uuid=uuid),
        noise=dict(band=dict(valid=.0006,degraded=.0021,uninformative=.0075),
            contamination_util_pct=1.,dispersion_degraded=.021,source_uuid=uuid,
            peers_observed=0,sharing_pattern='exclusive'),
        shapes=[dict(label='4x8 fp32',bytes=128,validated_on_uuid=uuid)])


def test_real_gate_assessment_accepts_all_five_measured_sections(tmp_path):
    from benchmarks.probes.unknown_card_gates import assess
    cfg=ssh_config(tmp_path)
    value=measured_probe()
    actual=assess(value)
    assert actual['all_green'] is True
    assert len(actual['gates'])==5
    (tmp_path/'card.json').write_text(json.dumps(value))
    frozen=plans.freeze(tmp_path/'frozen',cfg)
    assert frozen['card_probe']['gates']==actual['gates']
    assert plans.validate(frozen)==frozen


@pytest.mark.parametrize('missing',['launch_floor','shared_memory','peak','noise','shapes'])
def test_real_gate_assessment_refuses_one_unmeasured_section(tmp_path,missing):
    from benchmarks.probes.unknown_card_gates import assess
    value=measured_probe();value.pop(missing)
    actual=assess(value)
    assert actual['all_green'] is False
    assert any(row['verdict']=='unknown' for row in actual['gates'].values())
    (tmp_path/'card.json').write_text(json.dumps(value))
    with pytest.raises(ValueError,match='five card gates'):
        plans.freeze(tmp_path/'frozen',ssh_config(tmp_path))


def test_ssh_probe_is_resolved_and_rechecked_in_read_only_replay(tmp_path,monkeypatch):
    cfg=ssh_config(tmp_path)
    probe=tmp_path/'card.json'; probe.write_text('{"measured": true}')
    monkeypatch.chdir(tmp_path)
    cfg['probe']='card.json'
    receipt={'path':str(probe),'sha256':'a'*64,'data':{'measured':True},'gates':{'fixture':'pass'}}
    mocked=Mock(return_value=receipt)
    monkeypatch.setattr(plans,'_probe',mocked)
    frozen=plans.freeze(tmp_path/'frozen',cfg)
    assert frozen['probe']==str(probe)
    assert frozen['execution_context']['test_only'] is False
    assert frozen['execution_context']['gpu_acceptance'] is False
    monkeypatch.chdir(tmp_path.parent)
    assert plans.validate(frozen)==frozen
    assert mocked.call_args.args==(str(probe),cfg['gpu'])
    mocked.return_value=dict(receipt,sha256='b'*64)
    with pytest.raises(ValueError,match='card probe changed'):
        plans.validate(frozen,check_implementation=False)


def test_cannot_replace_an_existing_frozen_plan(tmp_path):
    path=tmp_path/'frozen'
    plans.freeze(path,config())
    before=(path/'plan.json').read_bytes()
    with pytest.raises(FileExistsError):plans.freeze(path,config())
    assert (path/'plan.json').read_bytes()==before
