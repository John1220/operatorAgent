"""Fixed study inputs and local identity never claim a remote/model execution."""
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from benchmarks.experiments.phase1 import study_materials as materials
from operator_agent.core.contract import load_dict
from operator_agent.core.run_contract import digest
from operator_agent.decide.bench_cases import BenchmarkCase
from operator_agent.execute.remote_executor import SSHExecutor


def test_four_fixed_tasks_and_all_eight_slots_have_the_planned_order():
    rows=materials.tasks()
    assert [(r['kernel_type'],r['declaration']['shapes']) for r in rows]==[
        ('matmul',[[4096,4096,4096]]),('relu',[[16384,6144]]),
        ('softmax',[[4096,4096]]),('relu',[[1024,768]])]
    assert all(r['declaration']['precision']=='fp16' and r['declaration']['dsl']=='triton' for r in rows)
    schedule=materials.schedule()
    assert [r['arm'] for r in schedule]==list('ABBAABBA')
    assert len({r['cell_id'] for r in schedule})==8
    assert [r['task_id'] for r in schedule]==[r['task_id'] for r in rows for _ in range(2)]


def test_freezing_large_performance_inputs_never_materializes_them(monkeypatch):
    original=BenchmarkCase.materialize
    called=[]
    def bounded(case):
        elements=sum(math.prod(t.shape) for t in case.tensors)
        assert elements<=65536,'preregistration allocated a primary performance input'
        called.append(elements)
        return original(case)
    monkeypatch.setattr(BenchmarkCase,'materialize',bounded)
    result=materials.tasks()
    assert called and sum(row['input_manifest']['auxiliary_l2_count'] for row in result)==len(called)
    primary=result[0]['input_manifest']['primary_cases'][0]
    assert primary['declaration']['seed']==0
    assert primary['seed_spec']['seed']==0
    assert all(t['strides_bytes']==[8192,2] for t in primary['tensors'])


def test_all_contract_owned_l2_cases_and_actual_strides_are_retained():
    for row in materials.tasks():
        contract=load_dict(row['contract'])
        assert contract.hash==row['contract_hash']
        frozen=row['input_manifest']
        assert digest({k:v for k,v in frozen.items() if k!='hash'})==frozen['hash']
        assert frozen['primary_count']==len(contract.cases)==1
        declared={c.case_id:c for c in contract.correctness_suite.cases}
        assert frozen['auxiliary_l2_count']==len(declared)
        assert {c['case_id'] for c in frozen['auxiliary_l2_cases']}==set(declared)
        layouts=set()
        for case in frozen['auxiliary_l2_cases']:
            spec=declared[case['case_id']]
            assert case['declaration']==spec.content_dict()
            assert case['declaration_hash']==digest(spec.content_dict())
            assert case['declaration']['inputs']['seed']>=0
            arrays=spec.materialize()
            for tensor in case['tensors']:
                value=arrays[tensor['name']]
                assert tensor['shape']==list(value.shape)
                assert tensor['strides_bytes']==list(value.strides)
                assert tensor['dtype']=='fp16' and tensor['itemsize']==2
                layouts.add(tensor['layout'])
        assert 'contiguous' in layouts and 'strided' in layouts
        assert frozen['deterministic'] is True
        assert row['reference']['runtime_version'] is None
        assert row['reference']['expected_remote_baseline']=='pytorch_eager'


def test_materials_return_independent_json_values_and_hash_changes_track_inputs():
    first=materials.tasks();other=materials.tasks()
    assert first==other
    original=other[0]['input_manifest']['hash']
    first[0]['input_manifest']['primary_cases'][0]['declaration']['seed']=123
    assert first!=other
    assert digest({k:v for k,v in first[0]['input_manifest'].items() if k!='hash'})!=original
    assert materials.tasks()==other
    assert json.loads(json.dumps(other))==other


def test_identity_hashes_runtime_prompts_remote_payload_and_evaluators_without_backend(monkeypatch):
    constructor=Mock(side_effect=AssertionError('identity opened SSH'))
    monkeypatch.setattr(SSHExecutor,'__init__',constructor)
    value=materials.identity()
    constructor.assert_not_called()
    assert value['scope']=='expected_local_identity_not_remote_observed'
    assert value['model_bound'] is value['gpu_bound'] is value['remote_runner_observed'] is value['formal_execution_ready'] is False
    assert value['implementation_hash']==digest(value['implementation'])
    assert 'remote_scripts/run_kernel.py' in value['runtime_sources']
    assert any(name.startswith('operator_agent/prompts/') for name in value['runtime_sources'])
    assert {'run_kernel.py','runner_server.py','tolerance.py','session_control.py'}<=set(value['expected_remote_payload'])
    for name,data in value['expected_remote_payload'].items():
        assert Path(data['source']).name==name
        assert data['sha256']==hashlib.sha256((materials.REPO/data['source']).read_bytes()).hexdigest()
    for group in ('runtime_sources','measurement_sources','evaluator_sources'):
        for name,sha in value[group].items():
            assert not name.startswith('output/')
            assert sha==hashlib.sha256((materials.REPO/name).read_bytes()).hexdigest()
    assert 'benchmarks/measurement/measure_launch_floor.py' in value['measurement_sources']
    assert 'benchmarks/experiments/phase1/connected_final.py' in value['evaluator_sources']
    assert value['package_versions']['python'] and value['package_versions']['plugin']
    assert type(value['git']['dirty']) is bool


def test_reference_symbols_bind_actual_source_instead_of_external_reference_claims():
    for row in materials.tasks():
        for reference in row['reference']['sources']:
            assert reference['file_sha256']==hashlib.sha256((materials.REPO/reference['path']).read_bytes()).hexdigest()
            assert len(reference['symbol_sha256'])==64
            assert reference['symbol'] in ('_reference_torch','_build_reference_fn')
        assert row['reference']['kind']=='builtin_reference_expected_not_runtime_observed'


@pytest.mark.parametrize('raw,dirty',[('',False),(' M operator_agent/app/cli.py\n',True)])
def test_git_cleanliness_is_observed_not_assumed(monkeypatch,raw,dirty):
    def run(argv,**kwargs):
        return SimpleNamespace(returncode=0,stdout=(raw if 'status' in argv else 'observed\n'))
    monkeypatch.setattr(materials.subprocess,'run',run)
    assert materials._git()['dirty'] is dirty


def test_unavailable_git_state_is_unknown(monkeypatch):
    monkeypatch.setattr(materials.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=128,stdout=''))
    assert materials._git()==dict(head=None,branch=None,status_porcelain=None,dirty=None)
