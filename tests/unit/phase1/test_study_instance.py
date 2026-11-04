"""SSH instance binding uses actual captured facts; mocks never touch a backend."""
from copy import deepcopy
import json
import shlex
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from benchmarks.experiments.phase1 import study_instance as instance
from operator_agent.core.run_contract import digest


CONNECTION=dict(host='gpu-host',port=2222,user='worker',work_dir='/opt/runner',timeout=120)
GPU=dict(index=1,uuid='GPU-fixture',name='Fixture NVIDIA GPU')
EXPECTED={'run_kernel.py':'a'*64,'tolerance.py':'b'*64}


def response():
    return dict(runtime=dict(python='3.12.16',torch='2.9.1+cu128',triton='3.5.0',cuda='12.8'),
        gpu=dict(GPU,driver_version='580.82.09'),runner_files=dict(EXPECTED))


def capture(monkeypatch,*,raw=None,connection=None,expected=None):
    run=Mock(return_value=SimpleNamespace(returncode=0,stdout=json.dumps(raw or response()),stderr=''))
    monkeypatch.setattr(instance.subprocess,'run',run)
    receipt=instance.capture(connection or CONNECTION,GPU,expected or EXPECTED)
    return receipt,run


def resign(receipt):
    receipt['identity_sha256']=digest(instance.fixed_identity(receipt))
    return receipt


def test_capture_runs_fixed_read_only_ssh_and_records_complete_facts(monkeypatch):
    receipt,run=capture(monkeypatch)
    argv=run.call_args.args[0]
    assert argv[:7]==['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-p','2222']
    assert argv[-2]=='worker@gpu-host' and argv[-3]=='--'
    command=shlex.split(argv[-1])
    assert command[:4]==['/opt/runner/.venv/bin/python','-I','-B','-c']
    assert command[4]==instance.REMOTE_SCRIPT
    payload=json.loads(command[5])
    assert payload==dict(work_dir='/opt/runner',gpu_index=1,runner_names=sorted(EXPECTED))
    assert run.call_args.kwargs==dict(capture_output=True,text=True,timeout=120,check=False)
    assert receipt['runtime']==response()['runtime']
    assert receipt['gpu']==response()['gpu']
    assert receipt['runner_files']==EXPECTED
    assert receipt['provenance']=='ssh_read_only' and receipt['collected_at']
    assert receipt['gpu_acceptance'] is receipt['runtime_execution_verified'] is receipt['kernel_executed'] is False
    assert instance.validate(receipt,GPU,EXPECTED)==receipt


def test_materials_expected_payload_map_is_accepted_without_reading_key_contents(monkeypatch,tmp_path):
    key=tmp_path/'ssh-key-not-opened'
    entry=dict(CONNECTION,key_path=str(key))
    expected={name:dict(source='source/'+name,sha256=sha) for name,sha in EXPECTED.items()}
    receipt,run=capture(monkeypatch,connection=entry,expected=expected)
    assert not key.exists()
    argv=run.call_args.args[0]
    assert argv[7:9]==['-i',str(key)]
    assert receipt['expected_runner_files']==EXPECTED


@pytest.mark.parametrize('mutate',[
    lambda v:v['gpu'].update(index=2),lambda v:v['gpu'].update(index=True),
    lambda v:v['gpu'].update(uuid='GPU-other'),lambda v:v['gpu'].update(name='Another GPU'),
    lambda v:v['gpu'].pop('driver_version'),lambda v:v['gpu'].update(driver_version='unknown'),
    lambda v:v['runtime'].pop('cuda'),lambda v:v['runtime'].update(cuda=None),
    lambda v:v['runtime'].update(torch='unknown'),lambda v:v['runtime'].update(triton=''),
    lambda v:v['runtime'].update(python=True),lambda v:v['runtime'].update(extra='not frozen'),
    lambda v:v['runner_files'].update({'run_kernel.py':'c'*64}),
    lambda v:v['runner_files'].pop('tolerance.py'),lambda v:v['runner_files'].update({'unplanned.py':'d'*64}),
])
def test_capture_refuses_unknown_or_different_remote_facts(monkeypatch,mutate):
    raw=response();mutate(raw)
    with pytest.raises(ValueError):capture(monkeypatch,raw=raw)


@pytest.mark.parametrize('mutate',[
    lambda v:v.update(provenance='mock'),lambda v:v.update(kernel_executed=True),
    lambda v:v.update(gpu_acceptance=True),lambda v:v.update(runtime_execution_verified=True),
    lambda v:v.update(collected_at=None),lambda v:v.update(collected_at='2026-11-04T12:00:00'),
    lambda v:v.update(collected_at='2026-11-04T12:00:00+08:00'),
    lambda v:v.update(identity_sha256='0'*64),lambda v:v.update(extra='not a closed receipt'),
    lambda v:v['expected_runner_files'].update({'run_kernel.py':'f'*64}),
])
def test_validate_is_pure_and_refuses_promoted_or_unbound_receipts(monkeypatch,mutate):
    receipt,_=capture(monkeypatch)
    mutate(receipt)
    forbidden=Mock(side_effect=AssertionError('pure validation performed SSH'))
    monkeypatch.setattr(instance.subprocess,'run',forbidden)
    with pytest.raises(ValueError):instance.validate(receipt,GPU,EXPECTED)
    forbidden.assert_not_called()


@pytest.mark.parametrize('connection',[
    dict(CONNECTION,host='-oProxyCommand=bad'),dict(CONNECTION,host='bad host'),
    dict(CONNECTION,user='-root'),dict(CONNECTION,user='two\nlines'),
    dict(CONNECTION,host='already@gpu-host'),dict(CONNECTION,port=True),
    dict(CONNECTION,port=0),dict(CONNECTION,work_dir='/'),
    dict(CONNECTION,work_dir='~/runner'),dict(CONNECTION,work_dir='/opt/../private'),
    dict(CONNECTION,timeout=0),dict(CONNECTION,timeout=float('inf')),
    dict(CONNECTION,proxy='unplanned'),
])
def test_invalid_connection_is_refused_before_ssh(monkeypatch,connection):
    run=Mock(side_effect=AssertionError('invalid config dispatched'))
    monkeypatch.setattr(instance.subprocess,'run',run)
    with pytest.raises(ValueError):instance.capture(connection,GPU,EXPECTED)
    run.assert_not_called()


@pytest.mark.parametrize('expected',[{}, {'../secret.py':'a'*64},{'/absolute.py':'a'*64},
    {'with space.py':'a'*64},{'run_kernel.py':'not-sha'},{'run_kernel.py':'A'*64}])
def test_runner_inventory_cannot_request_another_path_or_unknown_digest(monkeypatch,expected):
    run=Mock(side_effect=AssertionError('invalid files dispatched'))
    monkeypatch.setattr(instance.subprocess,'run',run)
    with pytest.raises(ValueError):instance.capture(CONNECTION,GPU,expected)
    run.assert_not_called()


def test_work_directory_is_shell_quoted_as_one_argument(monkeypatch):
    receipt,run=capture(monkeypatch,connection=dict(CONNECTION,work_dir="/opt/runner $value's space"))
    command=shlex.split(run.call_args.args[0][-1])
    assert command[0]=="/opt/runner $value's space/.venv/bin/python"
    assert json.loads(command[-1])['work_dir']==receipt['connection']['work_dir']


@pytest.mark.parametrize('completed',[
    SimpleNamespace(returncode=255,stdout='',stderr='private SSH diagnostics'),
    SimpleNamespace(returncode=0,stdout='not-json',stderr=''),
    SimpleNamespace(returncode=0,stdout='x'*65537,stderr=''),
    SimpleNamespace(returncode=0,stdout='{}',stderr=''),
])
def test_unsuccessful_collection_never_creates_a_receipt(monkeypatch,completed):
    monkeypatch.setattr(instance.subprocess,'run',Mock(return_value=completed))
    with pytest.raises(ValueError) as exc:instance.capture(CONNECTION,GPU,EXPECTED)
    assert 'private SSH diagnostics' not in str(exc.value)


@pytest.mark.parametrize('error',[OSError('key path detail'),subprocess.TimeoutExpired(['ssh'],1)])
def test_transport_failure_is_sanitized(monkeypatch,error):
    monkeypatch.setattr(instance.subprocess,'run',Mock(side_effect=error))
    with pytest.raises(ValueError,match='read-only SSH instance observation failed'):
        instance.capture(CONNECTION,GPU,EXPECTED)


def test_check_recaptures_and_ignores_only_collection_timestamp(monkeypatch):
    retained,_=capture(monkeypatch)
    fresh=deepcopy(retained);fresh['collected_at']='2026-11-05T01:00:00+00:00'
    recapture=Mock(return_value=fresh);monkeypatch.setattr(instance,'capture',recapture)
    cfg=dict(backend='ssh',connection=CONNECTION,gpu=GPU,
             study=dict(instance_receipt=retained,expected_runner_files=EXPECTED))
    assert instance.check(cfg)==fresh
    recapture.assert_called_once_with(retained['connection'],GPU,EXPECTED)


@pytest.mark.parametrize('field,value',[
    ('runtime',dict(response()['runtime'],torch='2.10.0')),
    ('gpu',dict(response()['gpu'],driver_version='590.1')),
    ('runner_files',dict(EXPECTED,**{'run_kernel.py':'c'*64})),
])
def test_check_stops_when_fixed_identity_changes(monkeypatch,field,value):
    retained,_=capture(monkeypatch);fresh=deepcopy(retained);fresh[field]=value;resign(fresh)
    monkeypatch.setattr(instance,'capture',Mock(return_value=fresh))
    with pytest.raises(ValueError,match='identity drift'):
        instance.check(dict(backend='ssh',connection=CONNECTION,gpu=GPU,study=dict(instance_receipt=retained)))


def test_check_rejects_changed_connection_before_recapture(monkeypatch):
    retained,_=capture(monkeypatch);recapture=Mock();monkeypatch.setattr(instance,'capture',recapture)
    with pytest.raises(ValueError,match='SSH connection differs'):
        instance.check(dict(backend='ssh',connection=dict(CONNECTION,host='different'),gpu=GPU,
                            study=dict(instance_receipt=retained)))
    recapture.assert_not_called()


def test_fake_check_never_collects_or_claims_remote_facts(monkeypatch):
    run=Mock(side_effect=AssertionError('fake collected SSH'))
    monkeypatch.setattr(instance.subprocess,'run',run)
    result=instance.check(dict(backend='fake'))
    assert result['test_only'] is True and result['provenance']=='not_collected'
    assert result['gpu_acceptance'] is False
    run.assert_not_called()


def test_real_check_requires_retained_observation(monkeypatch):
    run=Mock(side_effect=AssertionError('missing receipt collected SSH'))
    monkeypatch.setattr(instance.subprocess,'run',run)
    with pytest.raises(ValueError,match='receipt required'):
        instance.check(dict(backend='ssh',connection=CONNECTION,gpu=GPU))
    run.assert_not_called()
