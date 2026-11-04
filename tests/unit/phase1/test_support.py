"""v0.7 candidate scope and the calibrated LayerNorm input contract."""
import pytest

from benchmarks.experiments.phase1 import support
from operator_agent.app.hitl_tools import build_task
from operator_agent.core.contract import ContractError, freeze_from_task
from operator_agent.core.schemas import ConstraintSpec, Precision, SourceSpec, TargetSpec, TaskSchema
from operator_agent.decide.correctness_profiles import PROFILE_VERSION, PROFILE_V2, profile_check
from operator_agent import TestInputGenerator


def declaration(op='layernorm',precision='fp16',dsl='python'):
    return dict(request=op+' '+precision,precision=precision,dsl=dsl,
        shapes=[[4,8,4]] if op=='matmul' else [[4,8]])


@pytest.mark.parametrize('op',support.OPERATORS)
@pytest.mark.parametrize('precision',support.PRECISIONS)
@pytest.mark.parametrize('backend,dsl',[('fake','python'),('ssh','triton')])
def test_candidate_scope_freezes_each_declared_family_and_precision(op,precision,backend,dsl):
    task,contract=support.check(declaration(op,precision,dsl),backend)
    assert task.source.kernel_type==op
    assert task.constraints.precision.value==precision
    assert len(contract.cases)==1
    assert len(contract.hash)==64
    if op=='layernorm':
        assert tuple(t.name for t in contract.cases[0].tensors)==('X',)


def test_fp16_layernorm_v2_freezes_one_input_and_materializes_matching_profile_inputs():
    task,contract=support.check(declaration(),'fake')
    assert task.constraints.correctness_profile==PROFILE_V2
    assert profile_check(task)['status']=='passed'
    assert tuple(t.name for t in contract.cases[0].tensors)==('X',)
    assert contract.cases[0].tensors[0].shape==(4,8)
    assert contract.cases[0].tensors[0].dtype=='fp16'
    inputs=TestInputGenerator.generate(task,matrix_dim=(4,8),seed=11)
    assert set(inputs)=={'X'}
    assert str(inputs['X'].dtype)=='float16'
    assert profile_check(task,inputs)['status']=='passed'


def test_fp32_layernorm_v1_remains_non_affine_under_the_trusted_profile():
    task,contract=support.check(declaration(precision='fp32'),'fake')
    assert task.constraints.correctness_profile==PROFILE_VERSION
    assert tuple(t.name for t in contract.cases[0].tensors)==('X',)


@pytest.mark.parametrize('prompt,precision,profile',[
    ('layernorm','fp16',PROFILE_V2), ('layernorm fp16','fp32',PROFILE_VERSION)])
def test_declaration_precision_override_rebuilds_profile_for_final_precision(prompt,precision,profile):
    cfg=declaration(precision=precision)
    cfg['request']=prompt
    task,contract=support.check(cfg,'fake')
    assert task.constraints.precision.value==precision
    assert task.constraints.correctness_profile==profile
    assert profile_check(task)['status']=='passed'
    assert tuple(t.name for t in contract.cases[0].tensors)==('X',)


@pytest.mark.parametrize('update',[
    {'request':'layernorm affine=True','precision':'fp16'},
    {'request':'layernorm eps=1e-3','precision':'fp16'},
    {'request':'layernorm fp16','precision':'fp32','framework':'tensorflow'}])
def test_rebuilding_profile_preserves_unsupported_semantics_and_framework(update):
    cfg=declaration();cfg.update(update)
    task=build_task(cfg)
    assert profile_check(task)['status']=='unknown'
    assert task.constraints.correctness_suite is None
    with pytest.raises((ValueError,ContractError)):
        support.check(cfg,'fake')


def test_legacy_layernorm_without_profile_keeps_affine_inputs_and_distinct_contract():
    task=TaskSchema(source=SourceSpec(kernel_type='layernorm'),target=TargetSpec(dsl='python'),
        constraints=ConstraintSpec(precision=Precision.FP16,shapes=[[4,8]]))
    contract=freeze_from_task(task)
    assert task.constraints.correctness_profile is None
    assert tuple(t.name for t in contract.cases[0].tensors)==('X','weight','bias')
    _,calibrated=support.check(declaration(),'fake')
    assert contract.hash!=calibrated.hash


@pytest.mark.parametrize('backend,dsl',[('ssh','python'),('fake','triton')])
def test_backend_and_source_language_cannot_be_silently_changed(backend,dsl):
    with pytest.raises(ValueError,match='Triton|CPU Python'):
        support.check(declaration(dsl=dsl),backend)


def test_connected_cell_must_have_exactly_one_declared_shape():
    cfg=declaration('relu','fp32')
    cfg['shapes']=[[4,8],[8,8]]
    with pytest.raises(ValueError,match='one declared shape'):
        support.check(cfg,'fake')


def test_explicit_layernorm_variants_outside_non_affine_scope_are_rejected():
    cfg=declaration()
    cfg['request']='layernorm fp16 affine=True'
    with pytest.raises((ValueError,ContractError)):
        support.check(cfg,'fake')


def test_scope_matrix_does_not_claim_release_or_gpu_acceptance():
    value=support.matrix()
    assert value['released'] is False
    assert value['status']=='candidate_scope_not_release_acceptance'
    assert value['layernorm']==dict(affine=False,axis='last',eps=1e-5)
    value['operators'].append('invented')
    assert 'invented' not in support.matrix()['operators']
