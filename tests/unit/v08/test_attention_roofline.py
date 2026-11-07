"""The attention roofline is a labelled coarse estimate, not a measured peak."""
import numpy as np
import pytest

from operator_agent.decide.verifier import Verifier


class _T:  # 只带 _ai_numer_denom 需要的两层字段
    source = type('S', (), {'kernel_type': 'attention'})()
    constraints = type('C', (), {'precision': type('P', (), {'itemsize': 2})()})()


def ai(inputs):
    flops, nbytes = Verifier._ai_numer_denom(_T, inputs)
    return flops, nbytes


def test_attention_flops_follow_the_two_gemms_and_causal_halves_them():
    q = np.zeros((1, 16, 1024, 128), np.float16)
    base = {'Q': q, 'K': q, 'V': q}
    full, nbytes = ai(dict(base, causal=False))
    causal, _ = ai(dict(base, causal=True))
    assert full == 4.0 * 1 * 16 * 1024 * 1024 * 128
    assert causal == full / 2
    # Q/K/V 各读一次 + O 写一次
    assert nbytes == 4 * 1 * 16 * 1024 * 128 * 2


def test_arithmetic_intensity_is_sequence_over_dtype_width():
    q = np.zeros((1, 16, 1024, 128), np.float16)
    for causal, expected in ((False, 512.0), (True, 256.0)):
        flops, nbytes = ai({'Q': q, 'K': q, 'V': q, 'causal': causal})
        assert flops / nbytes == expected          # S / 2 bytes, causal 时再减半


def test_unreadable_shapes_yield_no_estimate_rather_than_a_guess():
    for inputs in ({'Q': np.zeros((16, 1024, 128), np.float16)},
                   {'Q': np.zeros((1, 0, 8, 4), np.float16), 'K': np.zeros((1, 0, 8, 4), np.float16)}):
        assert ai(inputs) == (None, None)


def test_the_attention_gate_only_accepts_the_calibrated_absolute_tolerance():
    from operator_agent.decide.reference_contract import (PROFILE_ATTENTION, ReferenceIdentityError,
                                                          registered_reference, validate_gate)
    spec = registered_reference(PROFILE_ATTENTION)
    assert validate_gate(spec.content_dict(), 'fp16', 3e-4, kernel_type='attention') == spec
    for bad in (3e-3, 1e-3, 5e-4, 0.0):
        with pytest.raises(ReferenceIdentityError):
            validate_gate(spec.content_dict(), 'fp16', bad, kernel_type='attention')
    # 算子名/精度/外挂参考三者任一不符都拒绝
    with pytest.raises(ReferenceIdentityError):
        validate_gate(spec.content_dict(), 'fp16', 3e-4, kernel_type='layernorm')
    with pytest.raises(ReferenceIdentityError):
        validate_gate(spec.content_dict(), 'fp32', 3e-4, kernel_type='attention')


def test_prepare_freezes_an_attention_contract_with_scalars(tmp_path):
    import json

    from operator_agent.app.hitl_tools import prepare

    run = tmp_path/'run'
    payload = prepare(dict(request='fp16 attention prefill 1x16x1024x128',
                           shapes=[[1, 16, 1024, 128]], hardware='nvidia_gpu',
                           dsl='triton', precision='fp16'), run)
    assert payload['ok'] is True
    assert payload['correctness_profile'] == 'builtin-attention-fp16-cuda-v1'
    assert sorted(payload['inputs']) == ['K', 'Q', 'V']
    assert payload['scalars'] == {'causal': False}
    assert len(payload['cases']) == 1 and len(payload['required_cases']) == 1
