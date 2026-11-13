"""The attention suite must declare exactly the cases it covers — including the large one."""
import pytest

from operator_agent.decide.bench_cases import BenchCaseError
from operator_agent.decide.correctness_profiles import ATTENTION_CASES, attention_suite
from operator_agent.decide.correctness_profiles import default_policy


def test_the_suite_has_the_eight_declared_cases():
    suite = attention_suite()
    assert [c.case_id for c in suite.cases] == [row[0] for row in ATTENTION_CASES]
    assert len(suite.cases) == 8


def test_every_case_carries_causal_by_value_and_real_shapes():
    suite = attention_suite()
    shapes = {c.inputs.tensors[0].shape for c in suite.cases}
    assert (1, 16, 1024, 128) in shapes and (1, 16, 4096, 128) in shapes
    assert (1, 16, 1000, 128) in shapes          # 非 2 的幂: 尾部 block
    assert all(set(c.inputs.scalars) == {'causal'} for c in suite.cases)
    assert all(type(c.inputs.scalars['causal']) is bool for c in suite.cases)
    assert any(c.inputs.scalars['causal'] is False for c in suite.cases)


def test_the_large_case_is_present_and_is_not_optional():
    """标定证明: 随机输入骗得过容差, 只有极值输入能抓到"没做 max 平移"。"""
    suite = attention_suite()
    patterns = {c.pattern for c in suite.cases}
    assert 'large' in patterns
    layouts = {c.layout for c in suite.cases}
    assert 'strided' in layouts


def test_attention_does_not_leak_into_the_axis_generated_profiles():
    # OPERATORS 那套是二维 axis 生成器, attention 不走它; 声明不支持的算子照旧拒绝。
    with pytest.raises(BenchCaseError):
        default_policy('attention', 'fp16')


def test_the_suite_is_deterministic_and_round_trips():
    first, second = attention_suite(), attention_suite()
    assert [c.content_dict() for c in first.cases] == [c.content_dict() for c in second.cases]


def test_every_declared_case_materializes_with_its_scalar_and_layout_intact():
    """声明了 8 条就得造得出 8 条。strided/zeros/large 曾因把 bool 当数组而抛
    AttributeError；这里逐条核标量值、张量形状/dtype，以及 pattern与 layout 真的落地。"""
    import numpy as np

    declared = {row[0]: row for row in ATTENTION_CASES}
    for case in attention_suite().cases:
        _, shape, causal, pattern, layout = declared[case.case_id]
        arrays = case.materialize()
        assert set(arrays) == {'Q', 'K', 'V', 'causal'}, case.case_id
        assert arrays['causal'] is causal, case.case_id
        assert type(arrays['causal']) is bool, case.case_id
        for name in ('Q', 'K', 'V'):
            tensor = arrays[name]
            assert tuple(tensor.shape) == shape, (case.case_id, name)
            assert tensor.dtype == np.float16, (case.case_id, name)
            if layout == 'strided':
                assert not tensor.flags.c_contiguous, (case.case_id, name)
                assert tensor.strides[-1] == tensor.itemsize * 2, (case.case_id, name)
            else:
                assert tensor.flags.c_contiguous, (case.case_id, name)
            if pattern == 'zeros':
                assert not tensor.any(), (case.case_id, name)
            elif pattern == 'large':
                assert (tensor == np.float16(100)).all(), (case.case_id, name)
            else:
                # fp16 上直接求 std 会溢出归约, 先升精度再判"不是退化输入"
                assert float(tensor.astype(np.float32).std()) > 0.1, (case.case_id, name)
        del arrays


def test_remateralizing_is_bit_identical_for_the_three_once_broken_cases():
    """verifier 的重复检查会再物化一次; 造不出或造得不一样, 确定性那一格就判不了。"""
    import numpy as np

    once_broken = {'s4096-causal-strided', 's1024-causal-zeros', 's1024-causal-large'}
    for case in attention_suite().cases:
        if case.case_id not in once_broken:
            continue
        first, second = case.materialize(), case.materialize()
        assert first['causal'] is second['causal'], case.case_id
        for name in ('Q', 'K', 'V'):
            np.testing.assert_array_equal(first[name], second[name])
        del first, second


def test_anticheat_blocks_the_attention_escape_hatches():
    from operator_agent.decide.anticheat import static_audit
    # 这里原来只测 `import torch.ops.aten as A` —— 那个写法在 torch 2.x 上直接
    # ModuleNotFoundError, 绿只证了"这条分支没人走"。换成**真能执行**的两种绕道。
    for code in ('import flash_attn\ndef kernel(Q,K,V):\n    return Q',
                 'from xformers.ops import memory_efficient_attention\ndef kernel(Q,K,V):\n    return Q',
                 'import torch\ndef kernel(Q,K,V,causal=False):\n'
                 '    return torch.ops.aten.scaled_dot_product_attention.default(Q,K,V,None,0.0,causal)',
                 'import torch\nfrom torch.ops import aten\ndef kernel(Q,K,V,causal=False):\n'
                 '    return aten.scaled_dot_product_attention.default(Q,K,V,None,0.0,causal)',
                 'import torch.nn.functional as F\ndef kernel(Q,K,V):\n    return F.scaled_dot_product_attention(Q,K,V)'):
        assert static_audit(code, 'triton').violations, code
    # 合法 triton 调用不误杀
    clean = 'import triton\nimport triton.language as tl\nimport torch\ndef kernel(Q,K,V):\n    return Q'
    assert static_audit(clean, 'triton').violations == []
