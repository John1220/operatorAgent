"""The optimize diagnosis is deterministic, cited, and refuses to invent headroom."""
import copy

from operator_agent.generate.optimize_diagnoser import SCHEMA, diagnose_optimization


def perf(**overrides):
    base = dict(level='L3', speedup_vs_baseline=1.42, median_time_ms=1.34,
                gflops_utilization_pct=61.2, bandwidth_utilization_pct=48.0,
                launch_dominance=0.07, launch_floor_us=18.69,
                measurement_can_gate=True, measurement_can_disable_recipe=True,
                measurement_can_count_stagnation=True,
                improvement_band=0.001, binding_attainment_pct=61.2,
                reference_ceiling_reached=False,
                roofline=dict(bound_type='compute', compute_used='fp16_tensor_core',
                              reference_lower_bound=True, peak_trusted=True))
    base.update(overrides)
    return base


def envelope(performance, validity='valid', **levels):
    return dict(levels=dict({'L1': True, 'L2': True, 'L3': True}, **levels),
                measurement_validity=validity, performance=performance)


def test_compute_bound_reading_yields_cited_experiments_and_headroom():
    result = diagnose_optimization(envelope(perf()))
    assert result['schema'] == SCHEMA and result['status'] == 'diagnosed'
    assert result['bottleneck']['binding'] == 'compute_bound'
    assert result['bottleneck']['evidence'] == dict(launch_dominance=0.07, launch_floor_us=18.69,
                                                    bound_type='compute',
                                                    compute_used='fp16_tensor_core')
    assert [row['axis'] for row in result['experiments']] == [
        'precision_and_accumulation', 'tiling', 'pipelining']
    assert result['decision_band_pct'] == 0.1
    assert result['headroom']['remaining_pct'] == 38.8
    assert result['headroom']['basis'] == 'reference_lower_bound'
    assert result['stop'] is None


def test_memory_bound_reading_aims_at_bandwidth_not_flops():
    result = diagnose_optimization(envelope(perf(
        roofline=dict(bound_type='memory', memory_used='fp16', reference_lower_bound=True),
        gflops_utilization_pct=12.0, bandwidth_utilization_pct=73.5,
        binding_attainment_pct=73.5, improvement_band=0.004), validity='degraded'))
    assert result['bottleneck']['binding'] == 'memory_bound'
    assert result['bottleneck']['attainment_key'] == 'bandwidth_utilization_pct'
    assert [row['axis'] for row in result['experiments']][0] == 'coalescing'
    assert result['decision_band_pct'] == 0.4
    # degraded 仍可判定, 但不许拿来禁用配方或累计停滞 —— 由 basis 如实转述
    assert result['basis']['can_disable_recipe'] is True
    assert result['basis']['validity'] == 'degraded'


def test_launch_floor_share_becomes_its_own_axis():
    result = diagnose_optimization(envelope(perf(launch_dominance=0.14)))
    assert result['bottleneck']['binding'] == 'launch_floor'
    assert [row['axis'] for row in result['experiments']] == ['work_per_launch']


def test_uninformative_reading_supports_no_comparison_and_no_experiment():
    result = diagnose_optimization(envelope(perf(
        measurement_can_gate=False, measurement_can_disable_recipe=False,
        measurement_can_count_stagnation=False, launch_dominance=0.35,
        improvement_band=None), validity='uninformative'))
    assert result['status'] == 'hold_for_measurement'
    assert result['experiments'] == [] and result['headroom'] is None and result['stop'] is None
    assert result['decision_band_pct'] is None
    assert 'new frozen contract' in result['reason']


def test_reference_ceiling_stops_instead_of_suggesting_more_rounds():
    result = diagnose_optimization(envelope(perf(
        reference_ceiling_reached=True, gflops_utilization_pct=95.0,
        binding_attainment_pct=95.0)))
    assert result['status'] == 'stop_at_ceiling'
    assert result['stop']['reason'] == 'reference_ceiling_reached'
    assert result['stop']['evidence']['ceiling_pct'] == 90.0
    assert result['experiments'] == []


def test_correctness_and_missing_reading_have_their_own_states():
    failed = diagnose_optimization(envelope(perf(), L2=False))
    assert failed['status'] == 'correctness_first'
    missing = diagnose_optimization(dict(levels=dict(L1=True, L2=True), performance=None))
    assert missing['status'] == 'insufficient_evidence'


def test_headroom_is_not_claimed_when_the_denominator_is_unverified():
    result = diagnose_optimization(envelope(perf(
        roofline=dict(bound_type='compute', reference_lower_bound=False, peak_trusted=False))))
    assert result['headroom']['basis'] == 'unverified'
    assert result['headroom']['remaining_pct'] is None


def test_unknown_binding_is_unknown_not_a_guess():
    result = diagnose_optimization(envelope(perf(roofline=dict(bound_type='unknown'))))
    assert result['bottleneck']['binding'] == 'unknown'
    assert result['experiments'] == []
    assert result['status'] == 'diagnosed'


def test_diagnosis_is_deterministic_and_does_not_mutate_its_input():
    envelope_in = envelope(perf())
    before = copy.deepcopy(envelope_in)
    first, second = diagnose_optimization(envelope_in), diagnose_optimization(envelope_in)
    assert first == second
    assert envelope_in == before


def test_optimize_context_carries_the_diagnosis_and_other_phases_do_not():
    from operator_agent.app.hitl_tools import build_task
    from operator_agent.core.contract import freeze_from_task
    from operator_agent.core.run_contract import code_hash
    from operator_agent.generate.host_context import build_host_context

    code = 'import numpy as np\ndef kernel(X):\n    return X.copy()\n'
    task = build_task(dict(request='relu fp32', shapes=[[4, 8]], precision='fp32',
                           dsl='python', hardware='mock_gpu'))
    contract = freeze_from_task(task)
    metrics = dict(perf())
    metrics['roofline_analysis'] = metrics.pop('roofline')
    verification = dict(ok=True, source_hash=code_hash(code), contract_hash=contract.hash,
                        levels=dict(L1=True, L2=True, L3=True),
                        verdicts=[dict(level='L1', passed=True, metrics={}),
                                  dict(level='L2', passed=True, metrics=dict(max_error=0.0)),
                                  dict(level='L3', passed=True, metrics=metrics)],
                        validity='valid')
    optimize = build_host_context(task, contract, backend='fake', phase='optimize',
                                  source=code, verification=verification)
    assert optimize['optimization']['status'] == 'diagnosed'
    assert optimize['optimization']['bottleneck']['binding'] == 'compute_bound'
    repair = build_host_context(task, contract, backend='fake', phase='repair',
                                source=code, verification=verification)
    assert 'optimization' not in repair