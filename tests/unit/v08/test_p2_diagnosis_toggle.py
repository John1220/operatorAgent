"""P2 方案②诊断开关接线本地守卫（mock，无 GPU/无真模型）。

覆盖两臂固定次数的搜索调用、完整提示的差分、规则搜索回退记录与模板作用域。
"""
import hashlib

import pytest

from operator_agent.core.schemas import UnifiedSketch, TilingStrategy
from operator_agent.generate.generation_spec import GENERATION_SPEC
from operator_agent.generate.prompt_manager import PromptManager

from benchmarks.experiments.v08 import p2_optimize as po


METRICS = dict(
    gflops=4325.0, speedup_vs_baseline=1.46, bound_type='memory', avg_time_ms=0.14,
    memory_bandwidth_gbs=2883.0, hw_peak_gflops=148000.0, hw_peak_bandwidth_gbs=4000.0,
    gflops_utilization_pct=2.9, bandwidth_utilization_pct=72.1, arithmetic_intensity=1.75,
    ridge_point=37.0, roofline_bound='memory', name_bound='memory', hw_name='nvidia h20',
    hw_compute_capability='9.0', shared_mem_per_sm_kb=228)

REPLY = dict(analysis='a', suggestion='增大 BLOCK_N 提升占有率',
             new_tiling=dict(block_m=16, block_n=256, block_k=32),
             new_hints=['vectorize load'], optimization_stage='hardware_utilization')


class MockLLM:
    def __init__(self, raise_exc=None):
        self.calls = []
        self._raise = raise_exc

    def chat_json(self, messages, **kwargs):
        self.calls.append(messages[0]['content'])
        if self._raise:
            raise self._raise
        return dict(REPLY)


def _sketch():
    return UnifiedSketch(algorithm='layernorm', target_dsl='triton', version=1,
                         tiling=TilingStrategy(block_m=16, block_n=512, block_k=32))


def _generate(sketch, suggestion, idx):
    return f'# round {idx}\nimport triton\nimport triton.language as tl\n'


def _verify(code):
    return dict(METRICS)


def _run(arm, llm, n_rounds=2):
    return po.run_optimize_arm(seed_sketch=_sketch(), seed_metrics=METRICS,
                               min_speedup=1.0, arm=arm, n_rounds=n_rounds, llm=llm,
                               generate=_generate, verify=_verify)


@pytest.mark.parametrize('arm', ['on', 'off'])
def test_both_arms_really_call_the_optimize_model_every_round(arm):
    llm = MockLLM()
    result = _run(arm, llm, n_rounds=2)
    assert result['llm_optimize_calls'] == 2
    assert len(llm.calls) == 2
    assert result['fallbacks'] == []
    assert all(r['path'] == 'llm' for r in result['rounds'])
    assert [r['prompt_sha256'] for r in result['rounds']] == [
        hashlib.sha256(prompt.encode('utf-8')).hexdigest() for prompt in llm.calls
    ]


def test_the_only_prompt_difference_is_the_diagnosis_block():
    on_llm, off_llm = MockLLM(), MockLLM()
    _run('on', on_llm, n_rounds=1)
    _run('off', off_llm, n_rounds=1)
    on_prompt, off_prompt = on_llm.calls[0], off_llm.calls[0]
    prefix, start, rest = on_prompt.partition('\n\n## Roofline 诊断\n')
    diagnosis, end, suffix = rest.partition('\n\n## 优化历史 (已尝试的方案)\n')
    assert start and end
    assert off_prompt == prefix + end + suffix
    for marker in ('Roofline 判定', '瓶颈定位', '算数强度', 'Roofline拐点',
                   '硬件理论峰值', '148000', '72.1'):
        assert marker in diagnosis
    assert '起点算子已经达到当前性能目标' in off_prompt
    assert '当前算子性能未达标' not in off_prompt
    assert 'nvidia h20 (CC 9.0)' in off_prompt
    assert '优化历史' in off_prompt
    assert off_prompt.startswith(GENERATION_SPEC)
    assert 'SKILL SPECIFICATION (SEARCH role)' in off_prompt
    assert '{{' not in on_prompt and '{{' not in off_prompt


def test_speed_feedback_table_values_are_identical_across_arms():
    on_llm, off_llm = MockLLM(), MockLLM()
    _run('on', on_llm, n_rounds=1)
    _run('off', off_llm, n_rounds=1)
    # GFLOPS、带宽与加速比的实测值都在共享表中。
    for value in ('| GFLOPS | 4325.00', '2883.00 GB/s', '1.46x'):
        assert value in on_llm.calls[0]
        assert value in off_llm.calls[0]


def test_fallback_to_rule_search_is_recorded_not_silent():
    llm = MockLLM(raise_exc=RuntimeError('boom'))
    result = _run('on', llm, n_rounds=1)
    assert result['llm_optimize_calls'] == 0
    assert len(result['fallbacks']) == 1
    fb = result['fallbacks'][0]
    assert fb['path'] == 'rule_llm_fallback'
    assert fb['llm_error'] == 'boom'
    assert fb['prompt_sha256'] == hashlib.sha256(llm.calls[0].encode('utf-8')).hexdigest()


def test_p2_prompt_manager_only_swaps_search_optimize():
    default = PromptManager()
    on_pm = po.P2PromptManager(include_diagnosis=True)
    off_pm = po.P2PromptManager(include_diagnosis=False)
    assert on_pm.get_raw('search_optimize') != off_pm.get_raw('search_optimize')
    for name in ('coder_generate_triton', 'coder_repair', 'designer_sketch', 'fixer_repair'):
        assert on_pm.get_raw(name) == default.get_raw(name) == off_pm.get_raw(name)


def test_perf_data_carries_no_escalation_or_budget_gate_keys():
    pd = po.build_perf_data(METRICS, min_speedup=1.0, history=[])
    assert 'structural_escalation_denied' not in pd
    assert 'llm_budget_exhausted' not in pd
    assert pd['roofline_bound'] == 'memory'
    assert pd['target_speedup'] == 1.0
