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
    'decision_band_pct: anything smaller is indistinguishable from noise in this tier. '
    'The experiment list is a MENU OF DIMENSIONS with expected signals, not instructions: '
    'which way to turn each knob depends on the source (accumulator size, head dim, '
    'occupancy), and a direction that looks obvious from the reading can be wrong.')

#: 轴 → 可动的维度。只写"能动哪一格、它换来什么、该看哪个读数动", **不写方向、不写代码**。
#:
#: 为什么去掉方向性祈使句(2026-11-08 真机): 原表在 compute_bound 下写的是
#: "raise per-program work until SMEM or occupancy stops the growth"。模型照做,
#: attention D=128 上 `BLOCK_M 64→128` 直接 +34.6%、`BLOCK_N 64→128` +10.8% ——
#: 累加器随 BLOCK_M 翻倍, 4 warps 撑不住。同一轮里唯一有效的是 num_stages 2→3(-2.1%)。
#: 方向属于"要读源码"的判断, 归宿主; 这张表只负责说清**有哪些维度、看什么信号**。
_EXPERIMENTS = {
    'compute_bound': (
        dict(axis='precision_and_accumulation',
             change='numeric width of the inner product vs. the accumulator (declared inputs stay frozen)',
             why='compute-bound readings are the ones where arithmetic width can matter at all',
             expected_signal='gflops_utilization_pct rises while max_error stays inside tolerance'),
        dict(axis='tiling',
             change='tile shape (BM / BN / BK) at fixed arithmetic — either direction is a candidate',
             why='tile shape trades arithmetic intensity against per-program registers, SMEM and occupancy',
             expected_signal='binding_attainment_pct up AND no new spills; buying one and losing the other is a loss'),
        dict(axis='pipelining',
             change='software pipeline depth (num_stages) and warp count around the current point',
             why='compute-bound kernels still stall when loads are not overlapped',
             expected_signal='median_time_ms falls without a change in gflops'),
    ),
    'memory_bound': (
        dict(axis='coalescing',
             change='access order and index arithmetic on the innermost lane (contiguous vs. strided reads)',
             why='memory-bound: the tight axis is bandwidth, and wasted lanes show up here',
             expected_signal='bandwidth_utilization_pct rises; max_error unchanged'),
        dict(axis='vectorized_access',
             change='transaction width of each load/store (vectorized vs. scalar) and base alignment',
             why='bytes per transaction is the lever a bandwidth-bound reading points at',
             expected_signal='effective bandwidth rises at the same working set'),
        dict(axis='dtype_width',
             change='width of the dtype actually loaded (the frozen contract decides what is allowed)',
             why='on a bandwidth-bound kernel, bytes moved is the first-order term',
             expected_signal='median_time_ms falls; re-check matched_ratio, not just max_error'),
    ),
    'launch_floor': (
        dict(axis='work_per_launch',
             change='elements per program (grid size against the dispatch floor)',
             why='the launch floor is a measurable share of the median, so it caps any speedup',
             expected_signal='median_time_ms falls toward the floor, not below it'),
    ),
    'unknown': (),
}

#: 这些 `compute_used` 说明核**已经**在张量核上跑 —— 再建议"换 dtype"就是废话。
_TENSOR_CORE_MARKERS = ('tensor_core', 'tensor-core', 'wgmma', 'mma')

#: 绑定类型 → 用哪个利用率当达成率 (取不到就缺席, 不兜 0)
_ATTAINMENT_KEY = {'compute_bound': 'gflops_utilization_pct',
                   'memory_bound': 'bandwidth_utilization_pct'}


def _eligible_experiments(binding, roofline):
    """去掉**证据已经满足**的轴: 建议一个已经做了的改动只会浪费一轮预算。"""
    used = str(roofline.get('compute_used') or '').lower()
    tensor_core = any(marker in used for marker in _TENSOR_CORE_MARKERS)
    rows = [dict(row) for row in _EXPERIMENTS[binding]]
    if tensor_core and binding == 'compute_bound':
        rows = [row for row in rows if row['axis'] != 'precision_and_accumulation']
    return rows[:MAX_EXPERIMENTS]


def _result(status, **fields):
    out = dict(schema=SCHEMA, status=status, policy=POLICY)
    out.update(fields)
    return out


def _binding_of(perf):
    """(binding, evidence) —— 绑定轴只看 roofline, 地板另算。

    地板**不是**与 compute/memory 并列的第三条轴: 它是「同一份代码再快也快不过地板」
    的串行限制 (2026-11-07 真机: relu 16384×6144 带宽利用率 81.2%、地板占 14.0%,
    把它判成 launch_floor 会把带宽轴整个吞掉)。所以绑定轴上并列的只有两条轴, 地板
    以 `launch_limit` 的形式单独报。
    """
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
    for key in ('compute_used', 'compute_path_evidence', 'memory_used'):
        if type(roofline.get(key)) is str:
            evidence[key] = roofline[key]
    # `bound_type == 'unknown'` 是占位值 (缺分母), 不是判定 —— 照 unknown 处理。
    if bound == 'compute':
        return 'compute_bound', evidence
    if bound == 'memory':
        return 'memory_bound', evidence
    if type(dominance) in (int, float) and dominance > LAUNCH_DOMINANCE_VALID:
        # 绑定轴判不出来, 但地板已经把上限压住了 —— 这是唯一能说的事。
        return 'launch_floor', evidence
    return 'unknown', evidence


def _launch_limit(perf):
    """地板给出的**串行上限**: 即使另一条轴做到完美, 也只剩 (1-地板占比) 可分。"""
    dominance = perf.get('launch_dominance')
    if type(dominance) not in (int, float) or not 0 <= dominance < 1:
        return None
    if dominance <= LAUNCH_DOMINANCE_VALID:
        return None
    return dict(dominance=round(float(dominance), 4),
                max_remaining_fraction=round(1.0 - float(dominance), 4),
                note='the launch floor is shared by every configuration of this kernel: '
                     'even at infinite bandwidth or FLOPs the median cannot fall below it')



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

    roofline = perf.get('roofline') if isinstance(perf.get('roofline'), dict) else {}
    # Older FP32 receipts inferred TF32 from the device. Keep their timing
    # usable while withholding the unsupported compute diagnosis and ceiling.
    if (roofline.get('compute_used') == 'unknown' or
            (roofline.get('compute_used') == 'tf32_tensor_core'
             and roofline.get('compute_path_evidence') != 'triton_ptx')):
        roofline = dict(roofline, compute_used='unknown', bound_type='unknown')
        perf = dict(perf, roofline=roofline, binding_attainment_pct=None,
                    reference_ceiling_reached=False)

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
                       evidence=evidence), decision_band_pct=band_pct,
                       launch_limit=_launch_limit(perf), headroom=None, stop=None, experiments=[],
                       reason=('This reading cannot support an improvement comparison'
                               + (': the launch floor dominates the median' if binding == 'launch_floor'
                                  else '') + '. Re-measuring the same source on a quieter window '
                               'is a measurement action; a different declared shape is a new frozen '
                               'contract, not an edit to this one.'))

    stop = None
    if perf.get('reference_ceiling_reached') is True:
        stop = dict(reason='reference_ceiling_reached',
                    evidence=dict(binding_attainment_pct=perf.get('binding_attainment_pct'),
                                  ceiling_pct=REFERENCE_CEILING_PCT,
                                  reference_lower_bound=roofline.get('reference_lower_bound') is True),
                    note='further recipes would only burn GPU and model budget')

    attainment = perf.get('binding_attainment_pct')
    headroom = None
    if type(attainment) in (int, float):
        if roofline.get('reference_lower_bound') is True:
            # 参考下界当下界用 ⇒ 利用率偏大 ⇒ 剩余空间**偏小**, 所以这是保守估计。
            headroom = dict(basis='reference_lower_bound',
                            remaining_pct=round(max(0.0, 100.0 - float(attainment)), 4),
                            note='remaining_pct is measured against a reference lower bound, '
                                 'so it under-states the real gap')
        elif roofline.get('peak_trusted') is True:
            # 分母是**核过口径的规格峰值** (不是实测峰值): 达成率可以读, 但它是规格推算,
            # 剩下的那点空间里还有时钟/温度/占用率。H20 fp16 4096³ 实测就到过 89.3%,
            # 这种读数下"再换配方"通常无收益, 所以余量必须报出来而不是说"不知道"。
            headroom = dict(basis='audited_peak',
                            remaining_pct=round(max(0.0, 100.0 - float(attainment)), 4),
                            note='remaining_pct is against a spec-audited peak, not a measured '
                                 'one; a small remainder usually means there is little to win')
        else:
            headroom = dict(basis='unverified', remaining_pct=None,
                            note='no reference lower bound on this reading: do not read the '
                                 'attainment as distance to the hardware peak')

    experiments = [] if stop is not None else _eligible_experiments(binding, roofline)
    out = _result('stop_at_ceiling' if stop is not None else 'diagnosed',
                  basis=basis, bottleneck=dict(binding=binding, evidence=evidence,
                                               attainment_key=_ATTAINMENT_KEY.get(binding),
                                               attainment_pct=attainment),
                  launch_limit=_launch_limit(perf),
                  decision_band_pct=band_pct, headroom=headroom, stop=stop,
                  experiments=experiments)
    if band_pct is None and stop is None:
        out['reason'] = ('No decision band is attached to this reading: a delta cannot be '
                         'called an improvement, only an observation. Confirm the measurement '
                         'tier before comparing rounds.')
    return out
