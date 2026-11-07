"""The attention reference is registered on its own profile, with calibrated tolerance."""
import pytest

from operator_agent.decide.reference_contract import (PROFILE_ATTENTION, PROFILE_V2,
                                                      ReferenceIdentityError,
                                                      as_reference, registered_reference)


def test_attention_has_its_own_contract_and_layernorm_is_untouched():
    layer = registered_reference(PROFILE_V2)
    attn = registered_reference(PROFILE_ATTENTION)
    assert layer.profile == PROFILE_V2 and attn.profile == PROFILE_ATTENTION
    assert layer.definition.startswith('native-pytorch-layernorm')
    assert attn.definition == 'native-pytorch-sdpa-mha-prefill-fp16-v1'
    assert layer.digest() != attn.digest()
    # layernorm 的 digest 是冻结锚点：改它等于作废所有已归档回执
    assert layer.digest().startswith('8f779954d3d2b3af')


def test_attention_semantics_say_what_it_is_and_is_not():
    attn = registered_reference(PROFILE_ATTENTION)
    assert attn.epsilon == 0.0            # 没有 eps 归一化
    assert attn.normalized_axis == -1     # softmax 沿末轴
    assert attn.affine is False
    assert attn.fp_fusion == 'forbid_native_sdpa'
    assert attn.deterministic_outputs is True
    assert attn.rtol == 1e-2 and attn.required_matched_ratio == 1.0


def test_an_unregistered_profile_fails_closed():
    with pytest.raises(ReferenceIdentityError, match='no registered reference'):
        registered_reference('builtin-attention-fp16-cuda-v99')
    broken = registered_reference(PROFILE_ATTENTION).content_dict()
    broken['profile'] = 'builtin-attention-fp16-cuda-v99'
    with pytest.raises(ReferenceIdentityError):
        as_reference(broken)


@pytest.mark.parametrize('field,value', [('rtol', 0.5), ('required_matched_ratio', 0.9),
                                         ('fp_fusion', 'allowed'), ('definition', 'invented')])
def test_a_relaxed_or_renamed_attention_contract_is_refused(field, value):
    tampered = registered_reference(PROFILE_ATTENTION).content_dict()
    tampered[field] = value
    with pytest.raises(ReferenceIdentityError):
        as_reference(tampered)


def test_the_registry_round_trips_every_entry():
    for profile in (PROFILE_V2, PROFILE_ATTENTION):
        spec = registered_reference(profile)
        assert as_reference(spec.content_dict()) == spec
