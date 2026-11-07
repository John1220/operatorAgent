"""Deterministic optimization hypotheses from one bound verification reading.

The host owns the decision; this module only turns the projected performance facts
into a bounded, cited hypothesis: which axis the reading says is tight, what an
experiment on that axis would change, how large a change would even be visible, and
when to stop instead of burning budget. It never runs, probes, rewrites source or
calls a model, and it grants no correctness or performance credit.

Two rules carried over from ``decide.measurement`` rather than re-derived here:

* an ``uninformative`` reading supports **no** improvement comparison, so it yields
  no experiment list (a different declared shape is a new frozen contract, not an
  edit to this one);
* ``reference_ceiling_reached`` is fail-closed — ``False`` means "not proven at the
  reference ceiling", not "there is headroom". Headroom is only reported when the
  denominator is the reference lower bound, and then it is *under*-estimated.
"""
from __future__ import annotations

from ..decide.measurement import (LAUNCH_DOMINANCE_UNINFORMATIVE, LAUNCH_DOMINANCE_VALID,
                                  REFERENCE_CEILING_PCT)

SCHEMA = 'optimize-diagnosis-v1'
MAX_EXPERIMENTS = 3

POLICY = (
    'optimization is a caller-owned hypothesis derived only from the reading above; it '
    'grants no correctness or performance credit. Submit changed source to the '
    'verification tool and keep the frozen contract. Verify the change against '
    'decision_band_pct: anything smaller is indistinguishable from noise in this tier.')

#: 轴 → 具体实验。只写"改什么、为什么、该看哪个读数动", 不写代码。
_EXPERIMENTS = {
    'compute_bound': (
        dict(axis='precision_and_accumulation',
             change='feed the inner product from tensor-core dtypes and keep the accumulator in fp32',
             why='compute-bound: the tight axis is FLOP throughput, so dtype width buys the most',
             expected_signal='gflops_utilization_pct rises while max_error stays inside tolerance'),
        dict(axis='tiling',
             change='raise per-program work (BM/BN/BK) until SMEM or occupancy stops the growth',
             why='larger tiles raise arithmetic intensity and amortize each load',
             expected_signal='binding_attainment_pct rises; watch for register spills in the compile receipt'),
        dict(axis='pipelining',
             change='increase num_stages/num_warps to overlap loads with the MMA pipeline',
             why='compute-bound kernels still stall when loads are not overlapped',
             expected_signal='median_time_ms falls without a change in gflops'),
    ),
    'memory_bound': (
        dict(axis='coalescing',
             change='make the innermost per-thread access unit-stride; drop strided index arithmetic',
             why='memory-bound: the tight axis is bandwidth, and uncoalesced lanes waste it',
             expected_signal='bandwidth_utilization_pct rises; max_error unchanged'),
        dict(axis='vectorized_access',
             change='use 128-bit vector loads/stores with an aligned base offset',
             why='fewer, wider transactions per byte moved',
             expected_signal='effective bandwidth rises at the same working set'),
        dict(axis='dtype_width',
             change='narrow the loaded dtype where the frozen tolerance allows; accumulate in fp32',
             why='half the bytes moved is half the time on a bandwidth-bound kernel',
             expected_signal='median_time_ms falls; re-check matched_ratio, not just max_error'),
    ),
    'launch_floor': (
        dict(axis='work_per_launch',
             change='increase elements per program and remove tiny grids so dispatch cost is amortized',
             why='the launch floor is a measurable share of the median, so it caps any speedup',
             expected_signal='median_time_ms falls toward the floor, not below it'),
    ),
    'unknown': (),
}

#: 绑定类型 → 用哪个利用率当达成率 (取不到就缺席, 不兜 0)
_ATTAINMENT_KEY = {'compute_bound': 'gflops_utilization_pct',
                   'memory_bound': 'bandwidth_utilization_pct'}


def _result(status, **fields):
    out = dict(schema=SCHEMA, status=status, policy=POLICY)
    out.update(fields)
    return out


def _binding_of(perf):
    """(binding, evidence) —— 只读投影里的事实, 判不了就 unknown。"""
    roofline = perf.get('roofline') if isinstance(perf.get('roofline'), dict) else {}
    dominance = perf.get('launch_dominance')
    evidence = {}
    if type(dominance) in (int, float):
        evidence['launch_dominance'] = dominance
        if perf.get('launch_floor_us') is not None:
            evidence['launch_floor_us'] = perf['launch_floor_us']
    bound = roofline.get('bound_type')
    if type(bound) is str:
        evidence['bound_type'] = bound
    for key in ('compute_used', 'memory_used'):
        if type(roofline.get(key)) is str:
            evidence[key] = roofline[key]
    # `bound_type == 'unknown'` 是占位值 (缺分母), 不是判定 —— 照 unknown 处理。
    if type(dominance) in (int, float) and dominance > LAUNCH_DOMINANCE_VALID:
        return 'launch_floor', evidence
    if bound == 'compute':
        return 'compute_bound', evidence
    if bound == 'memory':
        return 'memory_bound', evidence
    return 'unknown', evidence


def diagnose_optimization(feedback):
    """Return one bounded diagnosis dict for the optimize phase.

    ``feedback`` is the context feedback envelope: ``levels`` plus the projected
    ``performance`` block. Nothing else is read, so the host can check every claim
    against the same facts it was given.
    """
    feedback = feedback if isinstance(feedback, dict) else {}
    levels = feedback.get('levels') if isinstance(feedback.get('levels'), dict) else {}
    perf = feedback.get('performance') if isinstance(feedback.get('performance'), dict) else None
    if levels.get('L2') is False:
        return _result('correctness_first',
                       reason='L2 did not pass: an optimization on an unverified kernel '
                              'cannot be attributed to the change. Fix correctness first.')
    if perf is None:
        return _result('insufficient_evidence',
                       reason='No L3 performance reading is bound to this source. Run a '
                              'verification first; an absent reading is not headroom.')

    validity = feedback.get('measurement_validity', perf.get('measurement_validity'))
    basis = dict(level=perf.get('level'),
                 validity=validity if validity in ('valid', 'degraded', 'uninformative') else 'unknown',
                 can_gate=perf.get('measurement_can_gate'),
                 can_disable_recipe=perf.get('measurement_can_disable_recipe'),
                 can_count_stagnation=perf.get('measurement_can_count_stagnation'))
    band = perf.get('improvement_band')
    band_pct = round(band * 100, 4) if type(band) in (int, float) else None
    binding, evidence = _binding_of(perf)

    if perf.get('measurement_can_gate') is False or basis['validity'] == 'uninformative':
        return _result('hold_for_measurement', basis=basis, bottleneck=dict(binding=binding,
                       evidence=evidence), decision_band_pct=band_pct, headroom=None,
                       stop=None, experiments=[],
                       reason=('This reading cannot support an improvement comparison'
                               + (': the launch floor dominates the median' if binding == 'launch_floor'
                                  else '') + '. Re-measuring the same source on a quieter window '
                               'is a measurement action; a different declared shape is a new frozen '
                               'contract, not an edit to this one.'))

    stop = None
    if perf.get('reference_ceiling_reached') is True:
        roofline = perf.get('roofline') if isinstance(perf.get('roofline'), dict) else {}
        stop = dict(reason='reference_ceiling_reached',
                    evidence=dict(binding_attainment_pct=perf.get('binding_attainment_pct'),
                                  ceiling_pct=REFERENCE_CEILING_PCT,
                                  reference_lower_bound=roofline.get('reference_lower_bound') is True),
                    note='further recipes would only burn GPU and model budget')

    attainment = perf.get('binding_attainment_pct')
    roofline = perf.get('roofline') if isinstance(perf.get('roofline'), dict) else {}
    headroom = None
    if type(attainment) in (int, float):
        if roofline.get('reference_lower_bound') is True:
            # 参考下界当下界用 ⇒ 利用率偏大 ⇒ 剩余空间**偏小**, 所以这是保守估计。
            headroom = dict(basis='reference_lower_bound',
                            remaining_pct=round(max(0.0, 100.0 - float(attainment)), 4),
                            note='remaining_pct is measured against a reference lower bound, '
                                 'so it under-states the real gap')
        else:
            headroom = dict(basis='unverified', remaining_pct=None,
                            note='no reference lower bound on this reading: do not read the '
                                 'attainment as distance to the hardware peak')

    experiments = [] if stop is not None else [dict(row) for row in _EXPERIMENTS[binding]][:MAX_EXPERIMENTS]
    out = _result('stop_at_ceiling' if stop is not None else 'diagnosed',
                  basis=basis, bottleneck=dict(binding=binding, evidence=evidence,
                                               attainment_key=_ATTAINMENT_KEY.get(binding),
                                               attainment_pct=attainment),
                  decision_band_pct=band_pct, headroom=headroom, stop=stop,
                  experiments=experiments)
    if band_pct is None and stop is None:
        out['reason'] = ('No decision band is attached to this reading: a delta cannot be '
                         'called an improvement, only an observation. Confirm the measurement '
                         'tier before comparing rounds.')
    return out