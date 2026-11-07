"""One receipt: can this machine produce a decidable performance reading right now?

The eight-cell run answered that question by hand — run the same thing N times, take
``iqr/median``, compare against the tier table — and then spent 74 minutes and 195
model calls on a comparison that came back with zero valid readings. This module is
that procedure as code, with the thresholds **imported** rather than restated:
``DISPERSION_DEGRADED`` / ``DISPERSION_UNINFORMATIVE`` for the tier and ``NOISE_BAND``
for the band each tier supports.

What it measures is the machine around one reference kernel, repeated. It is not a
candidate's speed, not a correctness claim, and not a verdict on any task: it says
whether a paired comparison run *now* could distinguish an effect of a given size.
``predicted_tier`` is a prediction about the next measurement window, never a
substitute for one, and a shared host can change between windows.
"""
from __future__ import annotations

import datetime
import math

from .measurement import (DISPERSION_DEGRADED, DISPERSION_UNINFORMATIVE, NOISE_BAND,
                          Validity, _worse)

PROTOCOL = 'noise-preflight-v1'
MIN_REPS = 3
#: 判决带宽 = 3 × 同一份代码连跑多次的散布 —— 与 `measurement.NOISE_BAND` 的取法同一条
#: (见那里的注释)。这里只算, 不另立系数。
BAND_MULTIPLE = 3.0

BOUNDARY = (
    'This receipt describes the measurement window around one reference kernel. It is not '
    'a speedup for any candidate, not a correctness result, and not a verdict on any task. '
    'Rerun it in every new window: a shared host can move between runs, and a quiet window '
    'can be lost without any change to the code under test.')


def _nearest_rank(values, q):
    """最近秩分位 (与 `cuda_ruler_profile._quantile` 同一取法, 便于两份回执对上)。"""
    return values[max(0, math.ceil(q * len(values)) - 1)]


def summarize(costs):
    """Repeat-to-repeat statistics for one reference kernel.

    Refuses rather than repairs: fewer than `MIN_REPS` samples, or any sample that is
    non-finite / non-positive, is a defect signal about the *channel*, not a zero cost.
    """
    if not isinstance(costs, (list, tuple)):
        raise ValueError('costs must be a sequence of millisecond samples')
    xs = []
    for value in costs:
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'unusable timing sample {value!r}: a missing reading is not a zero cost')
        xs.append(float(value))
    if len(xs) < MIN_REPS:
        raise ValueError(f'at least {MIN_REPS} samples are required, got {len(xs)}')
    ordered = sorted(xs)
    median = _nearest_rank(ordered, 0.5) if len(ordered) % 2 else (
        (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2.0)
    q1, q3 = _nearest_rank(ordered, 0.25), _nearest_rank(ordered, 0.75)
    return dict(n=len(ordered), median_ms=round(median, 6),
                min_ms=round(ordered[0], 6), max_ms=round(ordered[-1], 6),
                iqr_over_median=round((q3 - q1) / median, 8),
                spread_over_median=round((ordered[-1] - ordered[0]) / median, 8),
                max_over_median=round(ordered[-1] / median, 8))


def empirical_band_pct(stats):
    """3 × 实测散布, 用百分点表示。"""
    return round(BAND_MULTIPLE * stats['iqr_over_median'] * 100.0, 6)


def predicted_tier(stats):
    """The tier the *next* window would most likely land in, from this window's spread.

    Only the two dispersion thresholds are used, exactly as ``measurement.assess`` uses
    them for its dispersion reason. The nominal band is **not** allowed to downgrade the
    tier: ``uninformative`` means "no comparison at all", while a wide measured band only
    means "small effects are unreadable" — that distinction is reported separately as
    ``band_consistent`` and carried into ``effective_band_pct``.
    """
    dispersion = stats['iqr_over_median']
    if dispersion > DISPERSION_UNINFORMATIVE:
        return Validity.UNINFORMATIVE.value
    return (Validity.DEGRADED if dispersion > DISPERSION_DEGRADED else Validity.VALID).value


def run_preflight(sample, *, reps=MIN_REPS, planned_effect_pct=None, identity=None,
                  channel='reference-kernel-repeatability', notes=()):
    """Run ``sample`` (a zero-arg callable returning median_ms) ``reps`` times.

    ``planned_effect_pct`` is the size of the effect the caller intends to ask about
    (for example the frozen ``min_speedup`` minus 1). With it, the receipt answers the
    question that actually matters: could this window resolve an effect that size?
    """
    if type(reps) is not int or reps < MIN_REPS:
        raise ValueError(f'reps must be an int >= {MIN_REPS}')
    if planned_effect_pct is not None and (type(planned_effect_pct) not in (int, float)
                                           or not math.isfinite(planned_effect_pct)
                                           or planned_effect_pct <= 0):
        raise ValueError('planned_effect_pct must be a positive percentage')
    costs, failure = [], None
    for index in range(reps):
        try:
            costs.append(sample())
        except Exception as error:              # 通道报错不是一次读数, 也不能被吞成 0
            failure = f'rep {index + 1}/{reps} raised {type(error).__name__}: {error}'
            break
    receipt = dict(schema=PROTOCOL,
                   created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                   channel=channel,
                   identity=dict(identity or {}),
                   reps_requested=reps, reps_completed=len(costs),
                   costs_ms=[round(float(value), 6) for value in costs],
                   notes=list(notes), boundary=BOUNDARY)
    if failure is not None:
        receipt.update(ok=False, action='refuse', reasons=[failure],
                       stats=None, empirical_band_pct=None, predicted_tier=None)
        return receipt
    stats = summarize(costs)
    tier = predicted_tier(stats)
    band_pct = empirical_band_pct(stats)
    nominal_pct = round(NOISE_BAND[Validity(tier)] * 100.0, 6)
    # 有效带宽取两者更宽的那个: 实测散布比标称表宽时, 标称表不再描述这台机器。
    effective_pct = max(band_pct, nominal_pct)
    receipt.update(ok=True, stats=stats, empirical_band_pct=band_pct,
                   predicted_tier=tier, nominal_band_pct=nominal_pct,
                   band_consistent=band_pct <= nominal_pct,
                   effective_band_pct=effective_pct,
                   planned_effect_pct=planned_effect_pct,
                   min_detectable_pct=effective_pct)
    reasons = []
    if tier == Validity.UNINFORMATIVE.value:
        reasons.append(f'repeat dispersion {stats["iqr_over_median"]:.4%} exceeds '
                       f'{DISPERSION_UNINFORMATIVE:.0%}: this window cannot resolve a '
                       f'performance difference, so a paired comparison would be noise')
    elif stats['iqr_over_median'] > DISPERSION_DEGRADED:
        reasons.append(f'repeat dispersion {stats["iqr_over_median"]:.4%} exceeds '
                       f'{DISPERSION_DEGRADED:.0%}: the tier still gates, but small deltas '
                       f'are unreadable')
    if not receipt['band_consistent']:
        reasons.append(f'measured band {band_pct:.4f}% is wider than the {tier} table band '
                       f'{nominal_pct:.4f}%: use the measured one for this window')
    if planned_effect_pct is not None and planned_effect_pct <= effective_pct:
        reasons.append(f'planned effect {planned_effect_pct:.4f}% does not clear the '
                       f'{effective_pct:.4f}% band: a pairing would not be able to show it')
    blocked = (tier == Validity.UNINFORMATIVE.value
               or (planned_effect_pct is not None and planned_effect_pct <= effective_pct))
    receipt.update(action='hold' if blocked else 'start', reasons=reasons)
    return receipt