"""Comparable-window selection and delivered-source formal conclusions."""
import pytest

from operator_agent.execute import host_task_budget as hb


def rows(*medians, validity=None, persist=None):
    validity = validity or ['valid'] * len(medians)
    persist = persist or [True] * len(medians)
    base = 1795000000
    out = []
    for index, median in enumerate(medians, 1):
        out.append(dict(id=index, source_hash=f'{index:064x}', purpose='explore',
                        role='baseline' if index == 1 else 'candidate',
                        status='completed', full_l2_passed=True,
                        finished_after_deadline=False,
                        evidence=f'/tmp/verify-{index}/result.json',
                        performance=dict(median_time_ms=median,
                                         measurement_validity=validity[index-1],
                                         measurement_can_persist=persist[index-1],
                                         selected_gpu=3, gpu_uuid='GPU-test',
                                         sm_clock_mhz_before=1980.0,
                                         sm_clock_mhz_after=1980.0,
                                         started_at=base + index * 60,
                                         completed_at=base + index * 60 + 10,
                                         created_at=base + index * 60 + 10,
                                         measurement_intruder_suspected=False,
                                         benchmark_contaminated=False,
                                         timing_identity_hash='tid',
                                         timing_protocol='protocol',
                                         timing_channel='channel',
                                         torch_version='torch', triton_version='triton',
                                         cuda_version='cuda', l2_flush_bytes=1,
                                         pair_repeat=1, speedup_vs_baseline=1.0,
                                         l3_passed=True)))
    return out


def test_performance_projection_preserves_unknown_bound_axis():
    full = dict(created_at=1.0, verdicts=[dict(level='L3', passed=True, metrics=dict(
        median_time_ms=0.2, roofline_analysis=dict(bound_type='unknown')))])
    assert hb._performance_facts(full)['bound_type'] == 'unknown'


def test_p2_replay_keeps_the_seed():
    best = hb._selection(rows(0.268064, 0.390544, 0.357376), 0.337626)['best']
    assert best['source_hash'] == f'{1:064x}'


def test_d2_replay_promotes_only_strictly_beyond_the_band():
    result = hb._selection(rows(0.267632, 0.266176, 0.264128, 0.274304), 0.714285)
    assert result['best']['source_hash'] == f'{3:064x}'
    reasons = [row['reason'] for row in result['candidates']]
    assert reasons[1] == 'equal_or_inside_band'
    assert reasons[2] == 'strictly_beyond_band'
    assert reasons[3] == 'slower'


def test_invalid_or_unpersistable_l3_never_wins():
    result = hb._selection(rows(0.268, 0.20, 0.21,
                                validity=['valid', 'degraded', 'valid'],
                                persist=[True, False, False]), 0.1)
    assert result['best']['source_hash'] == f'{1:064x}'
    assert result['candidates'][1]['reason'] == 'l3_not_valid_or_not_persistable'
    assert result['candidates'][2]['reason'] == 'l3_not_valid_or_not_persistable'


def pair_rows(first, second):
    left = rows(first)[0]
    right = rows(second)[0]
    right.update(id=2, role='candidate', source_hash='f' * 64,
                 evidence='/tmp/verify-2/result.json')
    right['performance']['started_at'] = left['performance']['started_at'] + 60
    right['performance']['completed_at'] = left['performance']['completed_at'] + 60
    right['performance']['created_at'] = left['performance']['created_at'] + 60
    return left, right


def test_cross_gpu_candidate_without_a_new_baseline_is_not_compared():
    left, right = pair_rows(0.400, 0.300)
    right['performance'].update(selected_gpu=7, gpu_uuid='GPU-other')
    result = hb._selection([left, right], 0.1)
    assert result['best']['source_hash'] == left['source_hash']
    assert result['candidates'][1]['reason'] == 'not_comparable:no_window_baseline'


def test_clock_drift_and_span_are_not_comparable():
    left, right = pair_rows(0.400, 0.300)
    right['performance']['sm_clock_mhz_after'] = 2100.0
    result = hb._selection([left, right], 0.1)
    assert result['best']['source_hash'] == left['source_hash']
    assert 'clock_drift' in result['candidates'][1]['reason']
    left, right = pair_rows(0.400, 0.300)
    right['performance']['started_at'] += 3 * 3600
    right['performance']['completed_at'] += 3 * 3600
    result = hb._selection([left, right], 0.1)
    assert result['best']['source_hash'] == left['source_hash']
    assert 'span_exceeded' in result['candidates'][1]['reason']


def test_intruder_is_diagnostic_for_the_formal_gain():
    left, right = pair_rows(0.400, 0.300)
    for record in (left, right):
        record['performance']['measurement_intruder_suspected'] = True
    result = hb._selection([left, right], 0.1)
    assert result['best']['source_hash'] == right['source_hash']
    assert result['formal']['gain'] is True
    assert result['formal']['comparison']['intruder_suspected'] is True


def test_stale_preflight_promotes_observed_but_blocks_formal():
    left, right = pair_rows(0.400, 0.300)
    band = dict(pct=0.1, source='preflight', created_epoch=1795000000 - 3 * 3600)
    result = hb._selection([left, right], 0.1, band)
    assert result['best']['source_hash'] == right['source_hash']
    assert result['candidates'][1]['reason'] == 'strictly_beyond_band_formal_pending'
    assert result['formal']['gain'] is False
    assert 'preflight_stale' in result['formal']['reasons']


def preflight_band():
    return dict(pct=0.1, source='preflight', created_epoch=1794999900,
                gpu_uuid='GPU-test', gpu_index=3, channel='channel')


def test_delivery_formal_uses_the_submitted_final_source():
    left, right = pair_rows(0.400, 0.300)
    final = dict(right, purpose='final', role='candidate')
    result = hb._delivery_formal([left, final], final, preflight_band(), 1.0)
    assert result['gain'] is True and result['pending'] is False
    assert result['submitted_source'] == final['source_hash']
    assert result['delta_pct'] == pytest.approx(25.0)


def test_failed_final_cannot_inherit_the_explore_gain():
    left, right = pair_rows(0.400, 0.300)
    assert hb._selection([left, right], 0.1)['formal']['gain'] is True
    final = dict(right, purpose='final', role='candidate', status='failed')
    result = hb._delivery_formal([left, right, final], final, preflight_band(), 1.0)
    assert result['gain'] is False and result['pending'] is True
    assert result['reasons'] == ['final_failed']


def test_seed_final_is_no_gain_not_pending():
    left, _ = pair_rows(0.400, 0.300)
    final = dict(left, purpose='final', role='candidate', status='completed')
    result = hb._delivery_formal([left, final], final, preflight_band(), 1.0)
    assert result['gain'] is False and result['pending'] is False
    assert result['reasons'] == ['seed_delivered']


def test_timing_identity_change_is_not_comparable():
    left, right = pair_rows(0.400, 0.300)
    right['performance']['triton_version'] = 'different'
    result = hb._selection([left, right], 0.1)
    assert result['best']['source_hash'] == left['source_hash']
    assert 'timing_identity_mismatch' in result['candidates'][1]['reason']


def test_missing_clean_flags_block_only_the_formal_claim():
    left, right = pair_rows(0.400, 0.300)
    for row in (left, right):
        row['performance']['measurement_intruder_suspected'] = None
        row['performance']['benchmark_contaminated'] = None
    result = hb._selection([left, right], 0.1)
    assert result['best']['source_hash'] == right['source_hash']
    assert result['formal']['gain'] is False
    assert 'contamination_flag_missing' in result['formal']['reasons']
