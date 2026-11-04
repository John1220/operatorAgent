"""Closed preregistration and measured-instance binding, without a provider/GPU."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from benchmarks.experiments.phase1 import study,study_instance,connected_plan,sampling
from operator_agent.core.run_contract import digest


MODEL=dict(base_url='https://provider.example/compatible-mode/v1',id='deepseek-fixture',
           temperature=.1,api_key_env='PHASE1_TEST_KEY',transport='live')


@pytest.fixture(autouse=True)
def frozen_identity(monkeypatch):
    identity=study.materials.identity()
    monkeypatch.setattr(study.materials,'identity',lambda:deepcopy(identity))
    implementation=connected_plan.implementation_identity()
    monkeypatch.setattr(connected_plan,'implementation_identity',lambda:deepcopy(implementation))


def resign(value):
    value['hash']=digest({k:v for k,v in value.items() if k!='hash'})
    return value


def registered(tmp_path):
    value=study.prepare(tmp_path/'pre',MODEL)
    return value,tmp_path/'pre/preregister.json'


def instance_config(tmp_path):
    return dict(connection=dict(host='fake-ssh',work_dir='/opt/prepared'),
        gpu=dict(index=1,uuid='GPU-test',name='Test GPU'),probe=str(tmp_path/'probe.json'),
        instance_receipt=str(tmp_path/'instance.json'))


def observations(monkeypatch,pre,cfg):
    raw=dict(runtime=dict(python='3.12.16',torch='2.8.0',triton='3.4.0',cuda='12.8'),
        gpu=dict(cfg['gpu'],driver_version='580.1'),
        runner_files={k:v['sha256'] for k,v in pre['identity']['expected_remote_payload'].items()})
    run=Mock(return_value=SimpleNamespace(returncode=0,stdout=json.dumps(raw),stderr=''))
    import subprocess
    monkeypatch.setattr(study_instance,'subprocess',SimpleNamespace(run=run,SubprocessError=subprocess.SubprocessError))
    monkeypatch.setattr(connected_plan,'_probe',lambda path,gpu:dict(path=path,sha256='a'*64,
        data=dict(device=dict(uuid=gpu['uuid'])),gates={str(i):dict(verdict='pass') for i in range(5)}))
    return run


def test_model_and_native_design_are_registered_without_credentials_or_network(tmp_path,monkeypatch):
    import socket
    monkeypatch.setattr(socket,'create_connection',Mock(side_effect=AssertionError('no network')))
    monkeypatch.setenv('LLM_API_KEY_FILE','/absent/secret-not-read')
    value,path=registered(tmp_path)
    assert study.load(path)==value
    assert [t['kernel_type'] for t in value['tasks']]==['matmul','relu','softmax','relu']
    assert [r['arm'] for r in value['schedule']]==list('ABBAABBA')
    assert value['model']['base_url']==MODEL['base_url']
    assert value['sampling']==sampling.policy()
    assert value['bindings']['model_configured'] is True
    assert value['bindings']['gpu'] is False
    assert value['formal_execution_ready'] is value['formal_comparison_credit'] is False
    assert value['observation']['human_count'].startswith('unknown')
    assert 'native' in value['design']['A']


def test_unbound_model_registration_is_truthful_and_cannot_bind_instance(tmp_path):
    value=study.prepare(tmp_path/'pre')
    assert value['model'] is None and value['bindings']['model_configured'] is False
    with pytest.raises(ValueError,match='model is unbound'):
        study.bind(tmp_path/'pre/preregister.json',instance_config(tmp_path),tmp_path/'bound')


@pytest.mark.parametrize('change',[
    lambda v:v.update(formal_execution_ready=True),lambda v:v.update(gpu_acceptance=True),
    lambda v:v['budget'].update(max_llm_calls=31),lambda v:v['sampling'].update(n=True),
    lambda v:v['tasks'][0]['declaration'].update(shapes=[[16,16,16]]),
    lambda v:v['schedule'].reverse(),lambda v:v['design'].update(A='CLI baseline'),
    lambda v:v['observation'].update(human_count=0),lambda v:v.update(extra='unregistered')])
def test_resigned_material_or_claim_changes_are_rejected(tmp_path,change):
    value,_=registered(tmp_path);change(value);resign(value)
    with pytest.raises(ValueError):study.validate(value)


def test_implementation_drift_refuses_binding_but_historical_inspection_is_possible(tmp_path,monkeypatch):
    value,path=registered(tmp_path)
    changed=deepcopy(value['identity']);changed['package_versions']['python']='other'
    monkeypatch.setattr(study.materials,'identity',lambda:changed)
    with pytest.raises(ValueError,match='drift'):study.load(path)
    assert study.load(path,check_identity=False)==value


def test_collect_and_bind_preserve_receipt_schema_and_canonical_connection(tmp_path,monkeypatch):
    pre,path=registered(tmp_path);cfg=instance_config(tmp_path)
    run=observations(monkeypatch,pre,cfg)
    envelope=study.collect(path,cfg,cfg['instance_receipt'])
    assert run.call_count==1
    assert envelope['preregistration_hash']==pre['hash']
    assert envelope['observation']['connection']['port']==22
    assert envelope['observation']['gpu_acceptance'] is False
    value=study.bind(path,cfg,tmp_path/'bound')
    assert connected_plan.load(tmp_path/'bound/plan.json')==value
    assert value['entry_a']=='dsh_native' and value['entry_b']=='dsh'
    assert value['study']['collection_preregistration_hash']==pre['hash']
    assert value['connection']['port']==22 and 'key_path' not in value['connection']
    assert value['formal_comparison_credit'] is False
    assert run.call_count==1,'bind must read the retained receipt, never recontact SSH'


@pytest.mark.parametrize('change',[
    lambda v:v.update(collection_preregistration_hash='f'*64),
    lambda v:v['instance_receipt']['runtime'].update(torch='9.9'),
    lambda v:v['expected_runner_files'].pop('run_kernel.py')])
def test_bound_receipt_or_identity_tampering_is_rejected(tmp_path,monkeypatch,change):
    pre,path=registered(tmp_path);cfg=instance_config(tmp_path);observations(monkeypatch,pre,cfg)
    study.collect(path,cfg,cfg['instance_receipt'])
    value=study.bind(path,cfg,tmp_path/'bound');change(value['study']);resign(value)
    with pytest.raises(ValueError):connected_plan.validate(value)
