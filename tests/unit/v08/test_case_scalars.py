"""Scalars on a case must survive by value — hashing, materializing, and the ABI."""
import json

import pytest

from operator_agent.decide.bench_cases import BenchmarkCase, BenchCaseError, TensorSpec
from operator_agent.decide.remote_input_fidelity import (InputFidelityError,
                                                        materialize_seed_inputs,
                                                        seed_input_digest)


def case(**scalars):
    return BenchmarkCase(name='attn', seed=3,
                         tensors=(TensorSpec('Q', (1, 2, 8, 4), 'fp16'),
                                  TensorSpec('K', (1, 2, 8, 4), 'fp16')),
                         scalars=dict(scalars))


def test_a_scalar_is_materialized_by_value_and_never_sampled():
    c = case(causal=True, scale=0.125)
    values = c.materialize()
    assert values['causal'] is True and type(values['causal']) is bool
    assert values['scale'] == 0.125
    # 同一 seed、加不加标量: 数组部分逐位相同 (标量不消耗 rng)
    plain = BenchmarkCase(name='attn', seed=3, tensors=c.tensors)
    assert (values['Q'] == plain.materialize()['Q']).all()


def test_scalars_enter_the_case_digest_so_fidelity_covers_them():
    on = seed_input_digest(case(causal=True).materialize())
    off = seed_input_digest(case(causal=False).materialize())
    none = seed_input_digest(BenchmarkCase(name='attn', seed=3,
                                           tensors=case().tensors).materialize())
    assert len({on, off, none}) == 3


def test_legacy_cases_keep_their_serialized_shape_and_hash():
    legacy = BenchmarkCase(name='x', seed=1,
                           tensors=(TensorSpec('X', (4, 4), 'fp16'),))
    for payload in (legacy.content_dict(), legacy.to_dict(), legacy.seed_spec()):
        assert 'scalars' not in payload
    assert legacy.scalars == {} and legacy.validate() == []


def test_scalars_take_part_in_the_contract_content_hash():
    a, b = case(causal=True), case(causal=False)
    assert a.content_dict() != b.content_dict()
    assert json.dumps(a.content_dict(), sort_keys=True) != json.dumps(b.content_dict(), sort_keys=True)


def test_round_trip_preserves_scalar_types():
    original = case(causal=True, eps=1e-5, label='causal', steps=4)
    restored = BenchmarkCase.from_dict(original.to_dict())
    assert restored.scalars == original.scalars
    assert type(restored.scalars['causal']) is bool
    assert type(restored.scalars['eps']) is float
    assert type(restored.scalars['steps']) is int


@pytest.mark.parametrize('scalars', [
    {'causal': None}, {'causal': [1, 2]}, {'causal': {'a': 1}},
    {'causal': float('nan')}, {'bad name': True}, {'Q': True},
    {name: 1 for name in 'abcdefghi'},
    {'label': 'x' * 300},
])
def test_invalid_scalars_are_refused_with_reasons(scalars):
    assert case(**scalars).validate()


def test_the_remote_materializer_refuses_scalars_it_cannot_pass_by_value():
    spec = dict(protocol='numpy-seed-v1', seed=1, tensors=[{'name': 'Q', 'shape': [2], 'dtype': 'fp16'}],
                scalars={'causal': [True]})
    with pytest.raises(InputFidelityError):
        materialize_seed_inputs(spec)
    spec['scalars'] = {'Q': True}
    with pytest.raises(InputFidelityError):
        materialize_seed_inputs(spec)


def test_the_host_abi_names_the_scalar_and_marks_its_provenance():
    from operator_agent.generate.generation_context import wrapper_calling_convention
    from operator_agent.generate.host_context import _case_metadata

    meta = _case_metadata(case(causal=True), layout='contiguous')
    assert wrapper_calling_convention([meta])['signature'] == 'def kernel(K, Q, causal):'
    assert meta['causal']['kind'] == 'scalar_by_value'
    assert meta['causal']['evidence'].endswith('not_sampled')
    assert meta['causal']['value'] is True


def test_a_case_without_scalars_has_no_scalar_rows_in_the_abi():
    from operator_agent.generate.generation_context import wrapper_calling_convention
    from operator_agent.generate.host_context import _case_metadata

    meta = _case_metadata(BenchmarkCase(name='x', seed=1,
                                        tensors=(TensorSpec('X', (4, 4), 'fp16'),)),
                          layout='contiguous')
    assert wrapper_calling_convention([meta])['signature'] == 'def kernel(X):'
    assert set(meta) == {'X'}

def test_attention_layout_is_four_dimensional_and_carries_causal():
    from operator_agent.decide.bench_cases import cases_from_shapes, known_layouts

    assert 'attention' in known_layouts()
    case = cases_from_shapes('attention', [((1, 16, 1024, 128), 1.0)],
                             dtype='fp16', seed=5)[0]
    assert case.name == 'attention-b1h16s1024d128'
    assert [(t.name, t.shape, t.dtype) for t in case.tensors] == [
        ('Q', (1, 16, 1024, 128), 'fp16'),
        ('K', (1, 16, 1024, 128), 'fp16'),
        ('V', (1, 16, 1024, 128), 'fp16')]
    assert case.scalars == {'causal': False}
    assert type(case.scalars['causal']) is bool
    assert case.validate() == []
    # 声明成三维就是错的（不猜 head 维）
    with pytest.raises(BenchCaseError):
        cases_from_shapes('attention', [((16, 1024, 128), 1.0)], dtype='fp16')


def test_attention_case_materializes_causal_by_value_not_by_seed():
    from operator_agent.decide.bench_cases import cases_from_shapes

    case = cases_from_shapes('attention', [((1, 2, 8, 4), 1.0)], dtype='fp16', seed=5)[0]
    values = case.materialize()
    assert values['causal'] is False and type(values['causal']) is bool
    # 第二个 seed 也一样：标量不随 seed 变
    other = cases_from_shapes('attention', [((1, 2, 8, 4), 1.0)], dtype='fp16', seed=9)[0]
    assert other.materialize()['causal'] is False


def test_the_case_identity_is_the_same_string_on_both_paths():
    """声明侧的 `key()` 与运行时侧 `case_key(materialize())` 必须逐字相同。

    C5 首次 L3（2026-11-08）就是这里分家：声明侧只喂张量得到
    `K=…|Q=…|V=…`，运行时那侧带着按值直传的标量、得到 `…|causal=?`，
    于是契约的 `required_cases` 与账本 key 对不上，workload 资格判 uninformative。
    """
    from operator_agent.decide.bench_cases import case_key
    c = case(causal=True)
    assert c.key() == case_key(c.materialize())
    assert c.key().endswith('|causal=?')
