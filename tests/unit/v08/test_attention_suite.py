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


def test_anticheat_blocks_the_attention_escape_hatches():
    from operator_agent.decide.anticheat import static_audit
    for code in ('import flash_attn\ndef kernel(Q,K,V):\n    return Q',
                 'from xformers.ops import memory_efficient_attention\ndef kernel(Q,K,V):\n    return Q',
                 'import torch.ops.aten as A\ndef kernel(Q,K,V):\n    return Q',
                 'import torch.nn.functional as F\ndef kernel(Q,K,V):\n    return F.scaled_dot_product_attention(Q,K,V)'):
        assert static_audit(code, 'triton').violations, code
    # 合法 triton 调用不误杀
    clean = 'import triton\nimport triton.language as tl\nimport torch\ndef kernel(Q,K,V):\n    return Q'
    assert static_audit(clean, 'triton').violations == []
