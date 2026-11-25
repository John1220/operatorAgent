"""Durable verification reservations for an explicitly enabled host workflow.

The host owns model calls and conversation state. This ledger limits plugin
verification dispatches and elapsed task time; it does not measure GPU seconds,
provider usage or billing. Reservations are never refunded. POSIX file locking
arbitrates cooperating processes, not hostile mutation by the same user.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import uuid

from ..core.run_contract import atomic_json, digest
from ..core.run_layout import write_once_artifact
from .task_cancellation import valid_invocation
from .task_clock import monotonic
import time


POLICY = 'host-task-policy.json'
#: 旧任务的预算状态文件。**新任务不再写它**：预算状态住在任务账本里
#: （`budget_opened` / `budget_reserved` / `budget_settled` 携带整份快照）。
#: 这个常量只剩一个用处：认出「账本之前的老目录」，那条路只读。
STATE = 'host-task.json'
LOCK = '.host-task.lock'
SCHEMA = 'operator-host-task-budget-v1'
FINAL_SELECTION_SCHEMA = 'operator-final-selection-v1'
_HASH = re.compile(r'[0-9a-f]{64}\Z')
MODEL_LIMIT_FIELDS = ('max_model_calls', 'max_output_tokens_per_call',
                      'total_output_token_reservations')
MODEL_STATE = 'host-model-budget.json'


class HostTaskBudgetError(ValueError):
    def __init__(self, code, detail=None):
        self.code = code
        self.detail = detail or code
        super().__init__(self.detail)


@dataclass(frozen=True)
class HostTaskLimits:
    wall_clock_s: float = 600.0
    max_verifications: int = 6
    final_reserve: int = 1
    max_model_calls: int = 12
    max_output_tokens_per_call: int = 8192
    total_output_token_reservations: int = 98304

    def __post_init__(self):
        if (type(self.wall_clock_s) not in (int, float)
                or not math.isfinite(self.wall_clock_s) or self.wall_clock_s <= 0):
            raise ValueError('wall_clock_s must be finite and positive')
        if type(self.max_verifications) is not int or not 2 <= self.max_verifications <= 1000:
            raise ValueError('max_verifications must be an integer from 2 to 1000')
        if type(self.final_reserve) is not int or self.final_reserve != 1:
            raise ValueError('exactly one independent final verification must be reserved')
        for name, maximum in (('max_model_calls', 1000),
                              ('max_output_tokens_per_call', 40960),
                              ('total_output_token_reservations', 40960000)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(name+' must be a positive integer no greater than '+str(maximum))

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value=None):
        if value is None:
            return cls()
        if type(value) is not dict or set(value) - set(cls.__dataclass_fields__):
            raise ValueError('invalid host task budget fields')
        return cls(**value)


def _hash(value, field):
    if type(value) is not str or _HASH.fullmatch(value) is None:
        raise HostTaskBudgetError('host_task_binding', field+' must be a SHA256 digest')
    return value


def _owner(value, *, required=False):
    if value is None and not required:
        return None
    if type(value) is not str or not valid_invocation(value):
        raise HostTaskBudgetError('host_task_owner', 'owner invocation must be 32 lowercase hexadecimal characters')
    return value


def _timely_l2(record):
    return (record.get('full_l2_passed') is True and record['status']=='completed'
            and record.get('finished_after_deadline') is False
            and record.get('accepted_by_supervisor') is not False)


def _usable_measurement(record):
    """A record whose numbers may serve as a **denominator**.

    Same rule `_selection` already applies to its rows, kept in one place: full L2,
    finished before the deadline, not rejected by the supervisor, and an eligible
    L3 (`valid` + `can_persist`). A final confirmation is not an input.

    Why this exists (2026-11-20): `_delivery_formal` used to pick the *last* record
    of the seed's source with eligible performance. A seed **retest** the supervisor
    later revoked keeps its L3 numbers in the ledger (that is deliberate — the facts
    stay) but must not become the denominator: a revoked slow retest would turn a
    2.0% difference into a 21.6% "formal gain".
    """
    return (type(record) is dict and record.get('purpose') != 'final'
            and _timely_l2(record)
            and _eligible_performance(record.get('performance')))


SEED = 'seed.json'
BASELINE = 'baseline.json'
PREFLIGHT = 'preflight.json'
PREFLIGHT_PROTOCOL = 'noise-preflight-v1'
BAND_PREFLIGHT = 'preflight'
BAND_NOMINAL = 'nominal_table'
NOMINAL_BAND_PCT = 0.1
#: Paired-evidence preconditions from the v0.8.2 comparable-window rule.
CLOCK_TOLERANCE = 0.02
WINDOW_SPAN_S = 2 * 60 * 60
FORMAL_THRESHOLD_PCT = 2.5


def _optional_json(path):
    """Tolerant read for host-owned side receipts; never a journal file."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, UnicodeError):
        return None


def _finite_positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _performance_facts(full):
    """Project the L3 facts the host ledger may compare; None when L3 is absent.

    Only facts are carried. No new judgement is computed here: eligibility and the
    noise band are applied by ``_selection`` / ``_noise_band``.
    """
    if type(full) is not dict:
        return None
    verdicts = [row for row in full.get('verdicts') or []
                if type(row) is dict and row.get('level') == 'L3']
    if len(verdicts) != 1:
        return None
    metrics = verdicts[0].get('metrics')
    if type(metrics) is not dict:
        return None
    median = metrics.get('median_time_ms')
    if not _finite_positive(median):
        return None
    facts = dict(median_time_ms=float(median))
    for key in ('speedup_vs_baseline', 'dispersion_iqr_over_median',
                'sm_clock_mhz_before', 'sm_clock_mhz_after',
                'mem_clock_mhz_before', 'mem_clock_mhz_after',
                'bandwidth_utilization_pct'):
        value = metrics.get(key)
        if type(value) in (int, float) and math.isfinite(value):
            facts[key] = float(value)
    for key in ('baseline_type', 'measurement_validity'):
        value = metrics.get(key)
        if type(value) is str:
            facts[key] = value
    persist = metrics.get('measurement_can_persist')
    facts['measurement_can_persist'] = persist if type(persist) is bool else None
    passed = verdicts[0].get('passed')
    facts['l3_passed'] = passed if type(passed) is bool else None
    gpu = metrics.get('selected_gpu')
    facts['selected_gpu'] = gpu if type(gpu) is int else None
    windows = metrics.get('arm_windows')
    facts['window'] = (windows.get('candidate') if type(windows) is dict
                       and type(windows.get('candidate')) is str else None)
    for key in ('measurement_intruder_suspected', 'benchmark_contaminated',
                'gpu_shared_with_others'):
        value = metrics.get(key)
        facts[key] = value if type(value) is bool else None
    timing = metrics.get('execution_timing')
    started = timing.get('started_at') if type(timing) is dict else None
    completed = timing.get('completed_at') if type(timing) is dict else None
    facts['started_at'] = float(started) if type(started) in (int, float) else None
    facts['completed_at'] = float(completed) if type(completed) in (int, float) else None
    created = full.get('created_at')
    facts['created_at'] = float(created) if type(created) in (int, float) else None
    facts['hw_name'] = metrics.get('hw_name') if type(metrics.get('hw_name')) is str else None
    timing_receipt = metrics.get('timing_receipt')
    identity = (timing_receipt.get('calibration_identity')
                if type(timing_receipt) is dict else None)
    runtime = identity.get('runtime') if type(identity) is dict else None
    config = identity.get('config') if type(identity) is dict else None
    try:
        facts['gpu_uuid'] = (timing_receipt['calibration_identity']['runtime']['gpu_uuid']
                             if type(timing_receipt) is dict else None)
    except (KeyError, TypeError):
        facts['gpu_uuid'] = None
    if type(facts['gpu_uuid']) is not str:
        facts['gpu_uuid'] = None
    facts['timing_identity_hash'] = (
        timing_receipt.get('identity_hash') if type(timing_receipt) is dict
        and type(timing_receipt.get('identity_hash')) is str else None)
    facts['timing_protocol'] = (
        identity.get('protocol') if type(identity) is dict
        and type(identity.get('protocol')) is str else None)
    channel = None
    if type(identity) is dict and type(identity.get('channel')) is str:
        channel = identity['channel']
    elif type(timing_receipt) is dict and type(timing_receipt.get('channel')) is str:
        channel = timing_receipt['channel']
    facts['timing_channel'] = channel
    for key in ('torch', 'triton', 'cuda', 'driver'):
        value = runtime.get(key) if type(runtime) is dict else None
        facts[f'{key}_version'] = value if type(value) is str else None
    for key in ('l2_flush_bytes', 'pair_repeat'):
        value = config.get(key) if type(config) is dict else None
        facts[key] = int(value) if type(value) is int else None
    roofline = metrics.get('roofline_analysis')
    bound = roofline.get('bound_type') if type(roofline) is dict else None
    facts['bound_type'] = bound if type(bound) is str else None
    reasons = metrics.get('measurement_validity_reasons')
    facts['measurement_validity_reasons'] = (
        [row for row in reasons if type(row) is str][:8] if type(reasons) is list else [])
    return facts


def _eligible_performance(facts):
    return (type(facts) is dict and facts.get('measurement_validity') == 'valid'
            and facts.get('measurement_can_persist') is True
            and _finite_positive(facts.get('median_time_ms')))


def _epoch(value):
    if type(value) in (int, float):
        return float(value)
    if type(value) is str:
        try:
            return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
        except ValueError:
            return None
    return None


def _channel_family(value):
    if type(value) is not str or not value:
        return None
    lowered = value.lower()
    if 'cuda-event' in lowered or 'cuda_events' in lowered or 'cuda events' in lowered:
        return 'cuda-event'
    return value


def _noise_band(run):
    """The comparison band: a sealed preflight receipt, else the nominal table."""
    receipt = _optional_json(Path(run)/PREFLIGHT)
    if (type(receipt) is dict and receipt.get('schema') == PREFLIGHT_PROTOCOL
            and receipt.get('ok') is True and receipt.get('action') == 'start'
            and receipt.get('test_only') is not True):
        band = receipt.get('effective_band_pct')
        if _finite_positive(band):
            created = receipt.get('created_at')
            identity = receipt.get('identity') if type(receipt.get('identity')) is dict else {}
            device = identity.get('device') if type(identity.get('device')) is dict else None
            if device is None:
                device = receipt.get('device') if type(receipt.get('device')) is dict else None
            return dict(pct=float(band), source=BAND_PREFLIGHT,
                        empirical_pct=receipt.get('empirical_band_pct'),
                        nominal_pct=receipt.get('nominal_band_pct'),
                        planned_effect_pct=receipt.get('planned_effect_pct'),
                        channel=receipt.get('channel'), created_at=created,
                        created_epoch=_epoch(created), identity=identity,
                        device=device,
                        gpu_uuid=(device.get('uuid') if type(device) is dict
                                  and type(device.get('uuid')) is str else None),
                        gpu_index=(device.get('index') if type(device) is dict
                                   and type(device.get('index')) is int else None))
    return dict(pct=NOMINAL_BAND_PCT, source=BAND_NOMINAL, empirical_pct=None,
                nominal_pct=NOMINAL_BAND_PCT, planned_effect_pct=None, channel=None,
                created_at=None, created_epoch=None, identity={}, device=None,
                gpu_uuid=None, gpu_index=None)


def _verification_round(record):
    evidence = record.get('evidence')
    if type(evidence) is not str:
        return None
    name = Path(evidence).parent.name
    return int(name[7:]) if name.startswith('verify-') and name[7:].isdigit() else None


def _pair_evidence(left, right, band):
    """The comparable-window preconditions between two L3 records.

    ``reasons`` are hard conditions for comparing the measured effect (they keep
    the incumbent). ``formal_reasons`` are the registered conditions for calling
    that effect a **formal gain**; a missing flag/identity is not treated as clean.
    """
    lp = left.get('performance') or {}
    rp = right.get('performance') or {}
    reasons = []
    lg, rg = lp.get('selected_gpu'), rp.get('selected_gpu')
    if type(lg) is not int or type(rg) is not int:
        reasons.append('gpu_missing')
    elif lg != rg:
        reasons.append('gpu_index_mismatch')
    lu, ru = lp.get('gpu_uuid'), rp.get('gpu_uuid')
    if type(lu) is not str or type(ru) is not str:
        reasons.append('gpu_uuid_missing')
    elif lu != ru:
        reasons.append('gpu_uuid_mismatch')
    identity_keys = ('timing_identity_hash', 'timing_protocol', 'timing_channel',
                     'torch_version', 'triton_version', 'cuda_version',
                     'l2_flush_bytes', 'pair_repeat')
    if any(lp.get(key) is None or rp.get(key) is None for key in identity_keys):
        reasons.append('timing_identity_missing')
    elif any(lp.get(key) != rp.get(key) for key in identity_keys):
        reasons.append('timing_identity_mismatch')
    clocks = [lp.get('sm_clock_mhz_before'), lp.get('sm_clock_mhz_after'),
              rp.get('sm_clock_mhz_before'), rp.get('sm_clock_mhz_after')]
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
           for value in clocks):
        reasons.append('clock_missing')
    elif max(clocks) / min(clocks) - 1 > CLOCK_TOLERANCE:
        reasons.append('clock_drift')
    times = [lp.get('started_at'), lp.get('completed_at'),
             rp.get('started_at'), rp.get('completed_at')]
    times_known = all(type(value) in (int, float) for value in times)
    if not times_known:
        reasons.append('time_missing')
    elif max(times) - min(times) > WINDOW_SPAN_S:
        reasons.append('span_exceeded')
    formal_reasons = []
    if band.get('source') == BAND_PREFLIGHT:
        created = band.get('created_epoch')
        if type(created) not in (int, float):
            formal_reasons.append('preflight_time_missing')
        elif times_known:
            # The band must be released before the earliest covered window; a
            # preflight taken after the seed cannot back-date the seed. This
            # invalidates the formal claim, not the observed comparison.
            if created > min(times) + 1:
                formal_reasons.append('preflight_after_first_window')
            elif max(times) - created > WINDOW_SPAN_S:
                formal_reasons.append('preflight_stale')
    intruder_flags = [lp.get('measurement_intruder_suspected'),
                      rp.get('measurement_intruder_suspected')]
    contamination_flags = [lp.get('benchmark_contaminated'), rp.get('benchmark_contaminated')]
    # v0.8.2 decision 6 (2026-11-20): intruder_suspected is a long-tail
    # diagnostic only; benchmark_contaminated=true still blocks a formal gain.
    if any(flag is None for flag in contamination_flags):
        formal_reasons.append('contamination_flag_missing')
    if any(flag is True for flag in contamination_flags):
        formal_reasons.append('benchmark_contaminated')
    if band.get('source') == BAND_PREFLIGHT:
        pre_uuid = band.get('gpu_uuid')
        pre_index = band.get('gpu_index')
        if pre_uuid is None and pre_index is None:
            formal_reasons.append('preflight_device_missing')
        for perf in (lp, rp):
            if pre_uuid is not None and perf.get('gpu_uuid') != pre_uuid:
                formal_reasons.append('preflight_device_mismatch')
            if pre_index is not None and perf.get('selected_gpu') != pre_index:
                formal_reasons.append('preflight_device_mismatch')
        preflight_family = _channel_family(band.get('channel'))
        if preflight_family is None:
            formal_reasons.append('preflight_channel_missing')
        else:
            for perf in (lp, rp):
                if _channel_family(perf.get('timing_channel')) != preflight_family:
                    formal_reasons.append('preflight_channel_mismatch')
    formal_reasons = list(dict.fromkeys(formal_reasons))
    return dict(comparable=not reasons, reasons=reasons,
                formal_clean=not formal_reasons, formal_reasons=formal_reasons,
                intruder_suspected=bool(any(flag is True for flag in intruder_flags)),
                benchmark_contaminated=bool(any(flag is True for flag in contamination_flags)))


def _window_group_key(performance):
    """Identity of a comparable measurement window (card + timing stack)."""
    if type(performance) is not dict:
        return None
    identity = (performance.get('timing_identity_hash')
                if type(performance.get('timing_identity_hash')) is str else '')
    uuid = performance.get('gpu_uuid')
    if type(uuid) is str and uuid:
        return ('gpu', uuid, identity)
    index = performance.get('selected_gpu')
    if type(index) is int:
        return ('index', index, identity)
    return None


def _selection(records, band_pct, band=None):
    """Fastest eligible L3 median per comparable window group.

    A card/software change (or a new stable clock window) starts a new *group*:
    the same-source seed re-measurement becomes that group's baseline, and the
    group's candidates are compared to it. Old groups remain in ``candidates`` as
    history; the delivered/active best is the last group's best. This is what
    makes "re-measure seed and candidate together after a card change" work.
    """
    if type(band) is not dict:
        band = dict(pct=float(band_pct), source=BAND_NOMINAL, created_epoch=None)
    band = dict(band)
    band_pct = float(band.get('pct', band_pct))
    seed_source = next((row.get('source_hash') for row in records
                        if row.get('role') == 'baseline'), None)
    groups = {}
    rows, last_group = [], None
    for record in records:
        performance = record.get('performance')
        row = dict(id=record['id'], role=record.get('role', 'candidate'),
                   source_hash=record['source_hash'], status=record['status'],
                   purpose=record.get('purpose'), round=_verification_round(record),
                   median_time_ms=(performance or {}).get('median_time_ms'),
                   measurement_validity=(performance or {}).get('measurement_validity'),
                   evidence=record.get('evidence'), performance=performance,
                   eligible=False, promoted=False, reason=None, delta_vs_best_pct=None,
                   comparison=None, window_group=None)
        if record.get('purpose') == 'final':
            row['reason'] = 'final_confirmation_not_selection_input'
        elif record['status'] != 'completed':
            row['reason'] = 'not_completed'
        elif record.get('accepted_by_supervisor') is False:
            row['reason'] = 'rejected_by_supervisor'
        elif record.get('finished_after_deadline') is True:
            row['reason'] = 'finished_after_deadline'
        elif record.get('full_l2_passed') is not True:
            row['reason'] = 'full_l2_not_passed'
        elif not _eligible_performance(performance):
            row['reason'] = 'l3_not_valid_or_not_persistable'
        else:
            row['eligible'] = True
            key = _window_group_key(performance) or ('record', record['id'])
            row['window_group'] = list(key)
            group = groups.get(key)
            if group is None:
                # A new group needs its own baseline. The first eligible record
                # is the initial baseline; later groups must start with the same
                # seed source (a re-measurement), never with a bare candidate.
                if not groups or record.get('source_hash') == seed_source:
                    groups[key] = dict(baseline=record, baseline_row=row,
                                       best=record, best_row=row)
                    row.update(promoted=True, reason=(
                        'first_eligible' if len(rows) == 0 or not groups
                        else 'window_baseline'))
                    last_group = key
                else:
                    row['reason'] = 'not_comparable:no_window_baseline'
            else:
                if record.get('source_hash') == seed_source:
                    # Same-source re-measurement is a new baseline, not a new
                    # candidate. In the same comparable window keep a previously
                    # verified faster candidate; only a bare/blue baseline resets.
                    previous_best = group['best']
                    previous_baseline = group['baseline']
                    keep = False
                    pair = None
                    if previous_best.get('id') != previous_baseline.get('id'):
                        pair = _pair_evidence(record, previous_best, band)
                        previous_perf = previous_best.get('performance') or {}
                        if (pair['comparable'] and _finite_positive(
                                previous_perf.get('median_time_ms'))
                                and previous_perf['median_time_ms'] < performance['median_time_ms']):
                            keep = True
                    group['baseline'] = record
                    group['baseline_row'] = row
                    if keep:
                        row.update(reason='window_baseline_kept_candidate',
                                   comparison=pair)
                    else:
                        group['best'] = record
                        group['best_row'] = row
                        row.update(promoted=True, reason='window_baseline')
                    last_group = key
                else:
                    pair = _pair_evidence(group['best'], record, band)
                    row['comparison'] = pair
                    current = (group['best'].get('performance') or {})['median_time_ms']
                    delta = (current - performance['median_time_ms']) / current * 100.0
                    row['delta_vs_best_pct'] = delta
                    if not pair['comparable']:
                        row['reason'] = 'not_comparable:' + ','.join(pair['reasons'])
                    elif delta > band_pct:
                        group['best'] = record
                        group['best_row'] = row
                        row.update(promoted=True, reason=(
                            'strictly_beyond_band' if pair['formal_clean']
                            else 'strictly_beyond_band_formal_pending'))
                        last_group = key
                    else:
                        row['reason'] = 'equal_or_inside_band' if delta >= 0 else 'slower'
        rows.append(row)
    active = groups.get(last_group)
    best = active['best'] if active is not None else None
    formal = dict(gain=False, pending=True, delta_pct=None,
                  threshold_pct=FORMAL_THRESHOLD_PCT, band_pct=band_pct,
                  scope='explore_recommendation_not_delivery',
                  baseline_round=None, best_round=None, comparison=None, reasons=[])
    if best is not None:
        formal['best_round'] = _verification_round(best)
    if active is None:
        formal['reasons'] = ['no_eligible_baseline']
    else:
        baseline_row = active['baseline_row']
        formal['baseline_round'] = baseline_row['round']
        if best is not None and best['source_hash'] == baseline_row['source_hash']:
            formal.update(pending=False, reasons=['seed_retained'])
        elif best is not None:
            base_perf = baseline_row.get('performance') or {}
            best_perf = best.get('performance') or {}
            delta = ((base_perf['median_time_ms'] - best_perf['median_time_ms'])
                     / base_perf['median_time_ms'] * 100.0)
            pair = _pair_evidence(baseline_row, best, band)
            reasons = []
            if not pair['comparable']:
                reasons.append('not_comparable:' + ','.join(pair['reasons']))
            if not pair['formal_clean']:
                reasons.extend(pair['formal_reasons'] or ['intruder_or_contaminated'])
            if delta <= FORMAL_THRESHOLD_PCT:
                reasons.append('below_formal_threshold')
            formal.update(delta_pct=delta, comparison=pair, reasons=reasons,
                          gain=not reasons and delta > band_pct,
                          pending=bool(reasons))
    return dict(best=best, candidates=rows, formal=formal,
                active_baseline=(active['baseline'] if active is not None else None),
                active_group=(list(last_group) if last_group is not None else None))


def _delivery_formal(records, final_record, band, min_speedup=None):
    """Formal gain for the *submitted* source, confirmed by its own final L3.

    The explore recommendation is deliberately separate: a final that failed, was
    never run, or delivered the seed must not inherit the explore gain. A formal
    gain additionally needs a timely, accepted, full-L2 final whose **L3 verdict
    passed** and whose speedup clears the contract's min_speedup, plus a real
    in-window preflight (nominal is not enough). The denominator must itself be a
    usable measurement (`_usable_measurement`), never a revoked retest.
    """
    band_pct = float(band.get('pct', NOMINAL_BAND_PCT))
    result = dict(gain=False, pending=True, delta_pct=None,
                  threshold_pct=FORMAL_THRESHOLD_PCT, band_pct=band_pct,
                  scope='delivered_source_independent_final',
                  submitted_source=(final_record.get('source_hash')
                                    if type(final_record) is dict else None),
                  submitted_round=(_verification_round(final_record)
                                   if type(final_record) is dict else None),
                  baseline_round=None, comparison=None, reasons=[])
    if type(final_record) is not dict:
        result['reasons'] = ['final_not_reserved']
        return result
    if final_record.get('status') != 'completed':
        result['reasons'] = [f'final_{final_record.get("status") or "unknown"}']
        return result
    if final_record.get('finished_after_deadline') is True:
        result['reasons'] = ['final_after_deadline']
        return result
    if final_record.get('accepted_by_supervisor') is False:
        result['reasons'] = ['final_rejected']
        return result
    if final_record.get('full_l2_passed') is not True:
        result['reasons'] = ['final_l2_not_passed']
        return result
    if not _eligible_performance(final_record.get('performance')):
        result['reasons'] = ['final_l3_not_valid_or_not_persistable']
        return result
    seed_source = next((row.get('source_hash') for row in records
                        if row.get('role') == 'baseline'), None)
    final_key = _window_group_key(final_record.get('performance'))
    # 分母只取**可用测量**：被撤销/被拒绝/迟到的记录保留事实，但不参与比较。
    measured = [row for row in records if _usable_measurement(row)]
    same_source = [row for row in measured
                   if seed_source is not None and row.get('source_hash') == seed_source]
    same_group = [row for row in same_source
                  if _window_group_key(row.get('performance')) == final_key]
    group_measured = [row for row in measured
                      if _window_group_key(row.get('performance')) == final_key]
    role_baselines = [row for row in measured if row.get('role') == 'baseline']
    baseline = (same_group[-1] if same_group else
                role_baselines[-1] if role_baselines else
                same_source[-1] if same_source else
                group_measured[0] if group_measured else None)
    if baseline is None:
        result['reasons'] = ['no_eligible_baseline']
        return result
    result['baseline_round'] = _verification_round(baseline)
    if final_record.get('source_hash') == baseline.get('source_hash'):
        result.update(gain=False, pending=False,
                      reasons=['seed_delivered' if seed_source is not None else 'baseline_delivered'])
        return result
    final_perf = final_record.get('performance') or {}
    if min_speedup is None:
        result.update(gain=False, pending=True, reasons=['min_speedup_unknown'])
        return result
    speedup = final_perf.get('speedup_vs_baseline')
    if type(speedup) in (int, float) and speedup < min_speedup:
        result.update(gain=False, pending=False, reasons=['below_min_speedup'])
        return result
    # 数字够大（或压根没有数字）时，仍要**明确通过的独立 L3 verdict**：契约 L3 可能
    # 因正确性/反作弊/roofline 等原因判未通过，而时延读数依旧好看。2026-11-20 之前
    # 这里只在 speedup 缺失时看 `l3_passed`，于是「主用例 1.2× + verdict 未通过」
    # 会被报成正式增益。
    if final_perf.get('l3_passed') is not True:
        result.update(gain=False, pending=False, reasons=['final_l3_not_passed'])
        return result
    if band.get('source') != BAND_PREFLIGHT:
        result.update(gain=False, pending=True, reasons=['preflight_missing'])
        return result
    pair = _pair_evidence(baseline, final_record, band)
    baseline_perf = baseline.get('performance') or {}
    delta = ((baseline_perf['median_time_ms'] - final_perf['median_time_ms'])
             / baseline_perf['median_time_ms'] * 100.0)
    reasons = []
    if not pair['comparable']:
        reasons.append('not_comparable:' + ','.join(pair['reasons']))
    if not pair['formal_clean']:
        reasons.extend(pair['formal_reasons'] or ['intruder_or_contaminated'])
    if delta <= FORMAL_THRESHOLD_PCT:
        reasons.append('below_formal_threshold')
    result.update(delta_pct=delta, comparison=pair, reasons=reasons,
                  gain=not reasons and delta > band_pct, pending=bool(reasons))
    return result


def _run_path(value):
    path = Path(value).absolute()
    if path.is_symlink():
        raise HostTaskBudgetError('host_task_journal', 'ordinary run directory required')
    return path.resolve()


def _read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise HostTaskBudgetError('host_task_journal', 'missing or symlink journal file')
    try:
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError('duplicate key')
                result[key] = value
            return result
        def constant(_):
            raise ValueError('nonfinite JSON')
        value = json.loads(path.read_text(), object_pairs_hook=pairs,
                           parse_constant=constant)
        if type(value) is not dict:
            raise ValueError('object required')
        return value
    except (OSError, ValueError, UnicodeError) as exc:
        raise HostTaskBudgetError('host_task_journal', 'unreadable task journal') from exc


@contextmanager
def _locked(run):
    # Each operation opens its own description, so flock also serializes
    # independent ledger objects used by different threads in one process.
    try:
        import fcntl
    except ImportError as exc:
        raise HostTaskBudgetError('host_task_platform', 'host workflow requires POSIX file locking') from exc
    if run.is_symlink() or not run.is_dir():
        raise HostTaskBudgetError('host_task_journal', 'prepared ordinary run directory required')
    try:
        fd = os.open(run/LOCK, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    except OSError as exc:
        raise HostTaskBudgetError('host_task_journal', 'unable to open ordinary task lock') from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise HostTaskBudgetError('host_task_journal', 'ordinary lock file required')
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def host_task_enabled(run):
    """An incomplete journal is enabled too: it must fail closed, never reset.

    两种目录都算「有宿主任务」：账本里已经有预算事件的（新任务），以及只剩旧状态文件的
    老任务。后者**只读** —— 读不出来就报错，不重建额度。
    """
    run = Path(run)
    if any((run/name).exists() or (run/name).is_symlink() for name in (POLICY, STATE)):
        return True
    try:
        return _budget_state(run) is not None
    except HostTaskBudgetError:
        # 账本读不动也算「有宿主任务」：那条路会**拒绝**，不会静默重建额度。
        return True


def _budget_state(run):
    """预算状态**只从任务账本读**：最后一条带 `budget.host_task` 的事件就是现值。

    账本被换成软链 ⇒ 拒绝：那意味着「写下去的额度落在别处」，量出来的和记下来的
    就不是同一条流了。
    """
    from ..core import task_journal
    if (Path(run)/task_journal.JOURNAL).is_symlink():
        raise HostTaskBudgetError('host_task_journal', 'task journal must not be a symlink')
    try:
        state = task_journal.fold(task_journal.read_events(run))['budget'].get('host_task')
    except task_journal.JournalError as exc:
        raise HostTaskBudgetError('host_task_journal', f'账本读不动: {exc.code}') from exc
    return dict(state) if isinstance(state, dict) else None


def _write_state(run, state, kind):
    """把整份快照写进账本。**快照，不是增量**：与 `fold` 对 `budget` 的合并语义一致。

    调用方必须已经在 `_locked(run)` 里 —— 预算的读-改-写要整段串行，
    否则两个预留者会各自读到「还剩 2」，然后都占掉。
    """
    from ..core import task_journal
    body = {key: value for key, value in state.items() if key != 'state_hash'}
    task_journal.append(run, kind, budget={'host_task': body})


def _contract_cases(run, contract_hash):
    frozen = _read(run/'contract.json')
    if frozen.get('hash') != contract_hash:
        raise HostTaskBudgetError('host_task_binding', 'prepared contract differs from host task')
    suite = frozen.get('contract', {}).get('correctness_suite')
    cases = suite.get('cases', []) if type(suite) is dict else []
    ids = [case.get('case_id') for case in cases]
    if (any(type(name) is not str or not name for name in ids)
            or len(set(ids)) != len(ids)):
        raise HostTaskBudgetError('host_task_binding', 'invalid frozen correctness suite')
    return sorted(ids)


def initialize_host_task(run, contract_hash, limits=None):
    """Freeze once; a repeat with the same policy returns the original budget."""
    run = _run_path(run)
    _hash(contract_hash, 'contract_hash')
    if not isinstance(limits, HostTaskLimits):
        limits = HostTaskLimits.from_dict(limits)
    with _locked(run):
        cases = _contract_cases(run, contract_hash)
        if not host_task_enabled(run):
            policy = dict(schema=SCHEMA, contract_hash=contract_hash,
                          limits=limits.as_dict(), required_case_ids=cases)
            body = (json.dumps(policy, ensure_ascii=False, sort_keys=True,
                               separators=(',', ':'), allow_nan=False)+'\n').encode('utf-8')
            write_once_artifact(run/POLICY, body)
            _write_state(run, dict(schema=SCHEMA, policy_hash=digest(policy),
                task_id=uuid.uuid4().hex, started_epoch=time.time(),
                started_monotonic=monotonic(), max_elapsed_s=0.0, records=[]),
                'budget_opened')
        budget = HostTaskBudget(run, contract_hash)
        policy, state = budget._load()
        if policy['limits'] != limits.as_dict() or policy['required_case_ids'] != cases:
            raise HostTaskBudgetError('host_task_binding', 'frozen host task policy differs')
        return budget._snapshot(policy, state)


class HostTaskBudget:
    def __init__(self, run, contract_hash):
        self.run = _run_path(run)
        self.contract_hash = _hash(contract_hash, 'contract_hash')

    def _load(self):
        policy, state = _read(self.run/POLICY), _budget_state(self.run)
        if state is None:
            # 没有预算事件就是没有这张账。**不补**：额度是冻结出来的，不是推出来的。
            raise HostTaskBudgetError('host_task_journal',
                                      'missing budget events in the task journal')
        if (policy.get('schema') != SCHEMA or state.get('schema') != SCHEMA
                or policy.get('contract_hash') != self.contract_hash
                or state.get('policy_hash') != digest(policy)):
            raise HostTaskBudgetError('host_task_binding', 'host task journal binding differs')
        try:
            HostTaskLimits.from_dict(policy['limits'])
            model_fields = set(policy['limits']) & set(MODEL_LIMIT_FIELDS)
            if model_fields and model_fields != set(MODEL_LIMIT_FIELDS):
                raise ValueError('incomplete frozen model limits')
            records = state['records']
            if (type(records) is not list
                    or [r['id'] for r in records] != list(range(1, len(records)+1))
                    or any(type(r['id']) is not int
                           or r['purpose'] not in ('explore', 'final')
                           or r['status'] not in ('reserved', 'completed', 'failed', 'cancelled')
                           or (r.get('owner_invocation_id') is not None
                               and (type(r['owner_invocation_id']) is not str
                                    or not valid_invocation(r['owner_invocation_id'])))
                           or _HASH.fullmatch(r['source_hash']) is None for r in records)
                    or len(records) > policy['limits']['max_verifications']
                    or sum(r['purpose']=='final' for r in records) > 1
                    or sum(r['purpose']=='explore' for r in records) > policy['limits']['max_verifications']-1):
                raise ValueError('inconsistent records')
            for name in ('started_epoch','started_monotonic','max_elapsed_s'):
                if type(state[name]) not in (int, float) or not math.isfinite(state[name]) or state[name] < 0:
                    raise ValueError('invalid task clock')
            selection = state.get('final_selection')
            if selection is not None and (
                    type(selection) is not dict or selection.get('schema') != FINAL_SELECTION_SCHEMA
                    or type(selection.get('override')) is not bool
                    or _HASH.fullmatch(selection.get('actual_source') or '') is None):
                raise ValueError('invalid final selection')
            if policy['required_case_ids'] != _contract_cases(self.run, self.contract_hash):
                raise ValueError('frozen suite drift')
        except (KeyError, TypeError, ValueError) as exc:
            raise HostTaskBudgetError('host_task_journal', 'inconsistent host task journal') from exc
        return policy, state

    @staticmethod
    def _elapsed(state):
        current = monotonic()
        if current < state['started_monotonic']:
            raise HostTaskBudgetError('host_task_clock', 'monotonic epoch changed; resume refused')
        # Wall elapsed includes idle/downtime. Persisted elapsed prevents credit
        # being recreated by a cooperating caller after a backwards wall jump.
        return max(state['max_elapsed_s'], current-state['started_monotonic'],
                   time.time()-state['started_epoch'], 0.0)

    def _min_speedup(self):
        sealed = _optional_json(self.run/'contract.json')
        contract = sealed.get('contract') if type(sealed) is dict else None
        value = contract.get('min_speedup') if type(contract) is dict else None
        return (float(value) if type(value) in (int, float)
                and math.isfinite(value) else None)

    def _snapshot(self, policy, state):
        elapsed = self._elapsed(state)
        records = state['records']
        limits = policy['limits']
        correct = [r for r in records if _timely_l2(r)]
        last = correct[-1] if correct else None
        final = next((r for r in records if r['purpose']=='final'), None)
        band = _noise_band(self.run)
        selection = _selection(records, band['pct'], band)
        formal_delivery = _delivery_formal(records, final, band, self._min_speedup())
        best = selection['best']
        baseline = selection['active_baseline']
        seed_manifest = _optional_json(self.run/SEED)
        baseline_attempts = [r for r in records if r.get('role') == 'baseline']
        if baseline is None:
            baseline = next((r for r in baseline_attempts if _timely_l2(r)
                             and _eligible_performance(r.get('performance'))), None)
        seed = None
        if (type(seed_manifest) is dict and seed_manifest.get('schema') == 'operator-seed-v1'
                and type(seed_manifest.get('source_hash')) is str):
            seed_path = Path(seed_manifest.get('source') or (self.run/'seed.py'))
            if (seed_path.is_symlink() or not seed_path.is_file()
                    or hashlib.sha256(seed_path.read_bytes()).hexdigest()
                    != seed_manifest['source_hash']):
                raise HostTaskBudgetError('host_task_binding', 'sealed seed source changed')
            status = ('established' if baseline is not None else
                      'l2_failed' if any(r.get('status') == 'completed'
                                        and r.get('full_l2_passed') is not True
                                        for r in baseline_attempts) else 'pending')
            seed = dict(sealed=True, source=str(seed_path),
                        source_hash=seed_manifest['source_hash'],
                        contract_hash=seed_manifest.get('contract_hash'), status=status)
        seed_ready = seed is None or seed['status'] == 'established'
        pending = sum(r['status']=='reserved' for r in records)
        remaining_exploration = limits['max_verifications']-1-sum(r['purpose']=='explore' for r in records)
        expired = elapsed >= limits['wall_clock_s']
        if final is not None:
            if final['status']=='reserved':
                phase, stop_reason = 'finalizing', 'independent_final_in_flight'
            elif (final['status']=='completed' and final.get('full_l2_passed') is True
                  and final.get('finished_after_deadline') is False):
                phase, stop_reason = 'final_complete', 'independent_final_completed'
            else:
                phase, stop_reason = 'final_failed', (
                    'final_deadline' if final.get('finished_after_deadline') is True else
                    'final_'+final['status'] if final['status']!='completed' else 'final_l2_not_established')
        elif expired:
            phase, stop_reason = 'exhausted', 'host_task_deadline'
        elif pending:
            phase, stop_reason = 'verifying', None
        elif remaining_exploration <= 0:
            phase, stop_reason = ('awaiting_final', 'exploration_complete_final_reserved') if correct else (
                'exhausted', 'exploration_exhausted_without_correct_candidate')
        else:
            phase, stop_reason = 'awaiting_host', None
        development_closed = final is not None or expired or remaining_exploration <= 0
        return dict(schema=SCHEMA, task_id=state['task_id'], contract_hash=self.contract_hash,
            run_dir=str(self.run),
            phase=phase, stop_reason=stop_reason, development_closed=development_closed,
            terminal=phase in ('final_complete','final_failed','exhausted'),
            terminal_scope='task_budget_only_not_worker_or_provider_stop',
            can_reserve_exploration=not development_closed,
            can_reserve_final=final is None and not expired and bool(correct) and not pending and seed_ready,
            limits=limits, elapsed_s=elapsed, remaining_s=max(0.0, limits['wall_clock_s']-elapsed),
            deadline_epoch=state['started_epoch']+limits['wall_clock_s'],
            expired=expired, reserved=len(records),
            remaining_verifications=limits['max_verifications']-len(records),
            remaining_exploration=remaining_exploration,
            remaining_final=1-sum(r['purpose']=='final' for r in records),
            unfinished=pending, records=records,
            final_verification=({key:final.get(key) for key in ('id','source_hash','status','evidence',
                'full_l2_passed','finished_after_deadline')} if final is not None else None),
            last_correct_candidate=(dict(source_hash=last['source_hash'], evidence=last['evidence'],
                source=last.get('source'), report=last.get('report'), verification_id=last['id'],
                scope='latest_complete_l2_only_not_performance_best') if last else None),
            noise_band=band, selection=selection['formal'],
            formal_delivery=formal_delivery, seed=seed,
            baseline=(dict(source_hash=baseline['source_hash'],
                round=_verification_round(baseline), evidence=baseline.get('evidence'),
                source=baseline.get('source'), report=baseline.get('report'),
                performance=baseline.get('performance'), scope='active_window_baseline')
                if baseline is not None else None),
            best_candidate=(dict(source_hash=best['source_hash'],
                round=_verification_round(best), evidence=best.get('evidence'),
                source=best.get('source'), report=best.get('report'),
                performance=best.get('performance'), role=best.get('role', 'candidate'),
                scope='fastest_eligible_l3_strictly_beyond_band')
                if best is not None else None),
            final_selection=state.get('final_selection'),
            candidates=selection['candidates'],
            policy_hash=state['policy_hash'],
            model_budget=self._model_budget(policy, state),
            wall_clock_scope='task_elapsed_across_calls; worker supervisor enforces in-flight deadline',
            reservation_unit='verification_dispatches_no_refund_not_gpu_seconds')

    def snapshot(self, *, readonly=False):
        """现值。**不写**：预算状态只在真正的转移（开账/预占/结算）时落账。

        以前这里每次查询都把 `max_elapsed_s` 写一次文件。账本是只追加的，「查一下」
        不是一次转移；写进去就是把现场当历史。反悔不了的保证不靠它：
        `_elapsed` 取 `max(已落账的 max_elapsed_s, 单调钟走过的, 墙钟走过的)`，
        而落账的那一条发生在每一次真正的转移上。
        """
        with _locked(self.run):
            policy, state = self._load()
            return self._snapshot(policy, state)

    def _model_budget(self, policy, state):
        limits = policy['limits']
        base = dict(owner='host', calls=None, tokens=None, cost='unknown',
                    task_id=state['task_id'], scope='native_main_model_dispatches_after_successful_prepare',
                    preparation_request='excluded_before_successful_task_binding')
        if not all(name in limits for name in MODEL_LIMIT_FIELDS):
            return dict(base, enforcement='legacy_policy_without_model_limits',
                        receipt_verified=False, limits=None)
        model_limits = {name: limits[name] for name in MODEL_LIMIT_FIELDS}
        path = self.run/MODEL_STATE
        if not path.exists() and not path.is_symlink():
            return dict(base, enforcement='not_bound', receipt_verified=False,
                        limits=model_limits)
        try:
            receipt = _read(path)
            def integer(value, minimum=0):
                return type(value) is int and minimum <= value <= 9007199254740991
            def integer_json(value):
                if value is None or type(value) in (str, bool, int):
                    return type(value) is not int or -9007199254740991 <= value <= 9007199254740991
                if type(value) is list:
                    return all(integer_json(item) for item in value)
                return (type(value) is dict and all(type(key) is str and integer_json(item)
                        for key, item in value.items()))
            if (not integer_json(receipt)
                    or receipt.get('state_hash') != digest({k:v for k,v in receipt.items() if k!='state_hash'})
                    or receipt.get('schema') != 'operator-host-model-budget-v1'
                    or receipt.get('task_id') != state['task_id']
                    or receipt.get('contract_hash') != self.contract_hash
                    or receipt.get('policy_hash') != state['policy_hash']
                    or receipt.get('policy_bytes_sha256') != hashlib.sha256((self.run/POLICY).read_bytes()).hexdigest()
                    or receipt.get('limits') != model_limits
                    or receipt.get('scope') != base['scope']):
                raise ValueError('model receipt binding or digest differs')
            lease, records = receipt.get('lease'), receipt.get('records')
            if (type(lease) is not dict or not isinstance(lease.get('id'),str) or not lease['id']
                    or not isinstance(lease.get('session_id'),str) or not lease['session_id']
                    or not integer(lease.get('owner_pid'),1) or not integer(lease.get('generation'),1)
                    or not integer(lease.get('bound_at')) or type(records) is not list
                    or len(records) > limits['max_model_calls']):
                raise ValueError('invalid model lease or calls')
            usage_fields = ('inputTokens','outputTokens','totalTokens','cacheReadTokens','cacheWriteTokens')
            known_usage = []
            for index, record in enumerate(records,1):
                if (type(record) is not dict or type(record.get('id')) is not int or record['id']!=index
                        or record.get('status') not in ('reserved','answered','failed','cancelled','abandoned')
                        or any(type(record.get(key)) is not str or not record[key]
                               for key in ('request_id','session_id','lease_id','provider','model'))
                        or not integer(record.get('owner_pid'),1)
                        or not integer(record.get('reserved_at'))
                        or not integer(record.get('max_output_tokens_reserved'),1)
                        or record['max_output_tokens_reserved'] > limits['max_output_tokens_per_call']
                        or (record['status']=='reserved' and record.get('finished_at') is not None)
                        or (record['status']!='reserved' and not integer(record.get('finished_at')))):
                    raise ValueError('invalid model reservation')
                usage = record.get('usage')
                if usage is not None:
                    if (type(usage) is not dict or set(usage)-set(usage_fields)
                            or any(not integer(value) for value in usage.values())):
                        raise ValueError('invalid model usage counters')
                    if all(key in usage for key in ('inputTokens','outputTokens','totalTokens')):
                        base_tokens=usage['inputTokens']+usage['outputTokens']
                        cache_tokens=usage.get('cacheReadTokens',0)+usage.get('cacheWriteTokens',0)
                        if usage['totalTokens'] not in (base_tokens,base_tokens+cache_tokens):
                            raise ValueError('inconsistent model usage counters')
                    if (record['status']!='reserved' and integer(usage.get('inputTokens'))
                            and integer(usage.get('outputTokens')) and any(usage.values())):
                        known_usage.append(usage)
            reserved_output = sum(record['max_output_tokens_reserved'] for record in records)
            if reserved_output > limits['total_output_token_reservations']:
                raise ValueError('model output reservations exceed frozen policy')
            known_output = sum(usage['outputTokens'] for usage in known_usage)
            known_total = sum(usage.get('totalTokens',0) for usage in known_usage)
            all_known = len(known_usage)==len(records)
            return dict(base, enforcement='host_enforced_receipt', receipt_verified=True,
                verification_scope='Frozen binding, integer self-hash and arithmetic only; '
                                   'not proof of current hook installation, model attention or provider stop.',
                limits=model_limits, lease=lease,
                calls=dict(reserved=len(records),remaining=limits['max_model_calls']-len(records),
                    limit=limits['max_model_calls'],unfinished=sum(record['status']=='reserved' for record in records)),
                tokens=dict(output_reserved=reserved_output,
                    remaining_reservations=limits['total_output_token_reservations']-reserved_output,
                    reservation_limit=limits['total_output_token_reservations'],
                    known_api_output_tokens=known_output, api_output_tokens=known_output if all_known else None,
                    known_api_total_tokens=known_total,
                    api_total_tokens=known_total if all_known and all('totalTokens' in usage for usage in known_usage) else None),
                api_usage_known_calls=len(known_usage),api_usage_unknown_calls=len(records)-len(known_usage))
        except (HostTaskBudgetError, OSError, ValueError, TypeError, KeyError):
            return dict(base,enforcement='receipt_invalid',receipt_verified=False,limits=model_limits,
                        receipt_error='host_model_receipt_invalid')

    def reserve(self, source_hash, *, purpose='explore', evidence=None, owner_invocation_id=None,
                final_selection=None, lease_host=None, lease_gpu_uuid=None):
        _hash(source_hash, 'source_hash')
        owner_invocation_id = _owner(owner_invocation_id)
        if purpose not in ('explore','final'):
            raise HostTaskBudgetError('host_task_purpose', 'purpose must be explore or final')
        if final_selection is not None and (
                purpose != 'final' or type(final_selection) is not dict
                or final_selection.get('schema') != FINAL_SELECTION_SCHEMA
                or type(final_selection.get('override')) is not bool
                or final_selection.get('actual_source') != source_hash
                or _HASH.fullmatch(final_selection.get('actual_source') or '') is None):
            raise HostTaskBudgetError('host_final_selection', 'invalid final selection binding')
        if evidence is not None:
            evidence = self._evidence_path(evidence)
        with _locked(self.run):
            policy, state = self._load()
            elapsed = self._elapsed(state)
            snapshot = self._snapshot(policy, state)
            if snapshot['final_verification'] is not None:
                raise HostTaskBudgetError('host_task_finalized',
                    'independent final was reserved; no further verification dispatches are permitted')
            if snapshot['expired']:
                raise HostTaskBudgetError('host_task_deadline', 'host task wall clock exhausted')
            if snapshot['remaining_verifications'] <= 0 or (
                    purpose=='explore' and snapshot['remaining_exploration'] <= 0):
                raise HostTaskBudgetError('host_verification_budget', 'verification budget exhausted; final reserve is protected')
            if purpose=='final':
                if snapshot['unfinished']:
                    raise HostTaskBudgetError('host_final_in_flight',
                        'settle all in-flight exploration reservations before independent final')
                if snapshot['remaining_final'] <= 0:
                    raise HostTaskBudgetError('host_final_budget', 'independent final reservation already spent')
                if not any(r['source_hash']==source_hash and _timely_l2(r) for r in state['records']):
                    raise HostTaskBudgetError('host_final_source', 'final requires a prior complete L2 pass of this source')
            seed = _optional_json(self.run/SEED)
            seed_hash = seed.get('source_hash') if type(seed) is dict else None
            established = any(r.get('role') == 'baseline' and _timely_l2(r)
                              and _eligible_performance(r.get('performance'))
                              for r in state['records'])
            if seed_hash is not None and not established and (
                    purpose != 'explore' or source_hash != seed_hash):
                raise HostTaskBudgetError(
                    'host_seed_baseline_required',
                    'establish the sealed seed baseline (full L2 + eligible L3) before '
                    'dispatching other sources or the independent final')
            role = ('baseline' if purpose == 'explore' and seed_hash == source_hash
                    and not established else 'candidate')
            from .gpu_lease import GpuLeaseBusy, acquire as acquire_lease
            lease_owner = owner_invocation_id or 'verify-' + uuid.uuid4().hex
            try:
                lease = acquire_lease(self.run, owner=lease_owner, tool='verify',
                                      deadline=time.time() + float(snapshot['remaining_s']),
                                      host=lease_host, gpu_uuid=lease_gpu_uuid)
            except GpuLeaseBusy as error:
                raise HostTaskBudgetError(
                    'host_gpu_busy',
                    f'{error}; verification and NCU replay must not share the GPU') from error
            try:
                record = dict(id=len(state['records'])+1, source_hash=source_hash, purpose=purpose,
                              role=role, status='reserved', reserved_epoch=time.time(),
                              reserved_monotonic=monotonic(), evidence=evidence,
                              full_l2_passed=None, owner_invocation_id=owner_invocation_id,
                              gpu_lease_owner=lease_owner, gpu_lease_key=lease.get('machine_key'))
                state['records'].append(record)
                if final_selection is not None:
                    state['final_selection'] = dict(final_selection)
                state['max_elapsed_s'] = elapsed
                _write_state(self.run, state, 'budget_reserved')
                return dict(record, remaining_s=snapshot['remaining_s'])
            except BaseException:
                from .gpu_lease import release as release_lease
                release_lease(self.run, lease_owner)
                raise

    def _evidence_path(self, value):
        original = Path(value).absolute()
        path = original.resolve()
        if self.run not in path.parents or path.name!='result.json' or any(
                p.is_symlink() for p in (original, *original.parents)
                if self.run==p.resolve() or self.run in p.resolve().parents):
            raise HostTaskBudgetError('host_task_evidence', 'ordinary result.json inside the owned run required')
        return str(path)

    def _completed_result(self, record, result, evidence, required_case_ids, records):
        path = Path(self._evidence_path(evidence))
        if any(r.get('evidence')==str(path) and r['id']!=record['id'] for r in records):
            raise HostTaskBudgetError('host_task_evidence', 'each reservation needs new independent evidence')
        full, complete = _read(path), _read(path.parent/'complete.json')
        binding = full.get('binding', {})
        if (full.get('evidence_complete') is not True or complete.get('evidence_complete') is not True
                or binding != complete.get('binding')
                or binding.get('source_hash') != record['source_hash']
                or binding.get('contract_hash') != self.contract_hash
                or full.get('source_hash') != record['source_hash']
                or full.get('contract_hash') != self.contract_hash
                or type(full.get('created_at')) not in (int,float)
                or full['created_at'] < record['reserved_epoch']):
            raise HostTaskBudgetError('host_task_evidence', 'completed evidence binding or reservation time differs')
        members = complete.get('files_sha256')
        if type(members) is not dict or 'result.json' not in members:
            raise HostTaskBudgetError('host_task_evidence', 'complete evidence manifest required')
        for name, expected in members.items():
            relative = Path(name)
            member = path.parent/relative
            if relative.is_absolute() or '..' in relative.parts or member.is_symlink() or not member.is_file():
                raise HostTaskBudgetError('host_task_evidence', 'invalid evidence manifest member')
            if hashlib.sha256(member.read_bytes()).hexdigest()!=expected:
                raise HostTaskBudgetError('host_task_evidence', 'completed evidence member changed')
        if result is not None and (type(result) is not dict or any(
                result.get(k)!=full.get(k) for k in ('source_hash','contract_hash','levels'))):
            raise HostTaskBudgetError('host_task_evidence', 'returned result differs from persisted evidence')
        l2 = [v for v in full.get('verdicts', []) if v.get('level')=='L2']
        correctness = l2[0].get('metrics',{}).get('correctness',{}) if len(l2)==1 else {}
        cases = correctness.get('cases', [])
        full_l2 = (full.get('levels',{}).get('L2') is True and len(l2)==1
            and l2[0].get('passed') is True and correctness.get('verdict')=='passed'
            and type(cases) is list and bool(required_case_ids)
            and all(type(c) is dict and type(c.get('case_id')) is str for c in cases)
            and sorted(c['case_id'] for c in cases)==required_case_ids
            and all(c.get('verdict')=='passed' for c in cases)
            and correctness.get('cases_total',len(cases))==len(cases))
        return full, bool(full_l2), str(path)

    def finish(self, reservation_id, *, outcome='completed', result=None, evidence=None):
        if type(reservation_id) is not int or outcome not in ('completed','failed','cancelled'):
            raise HostTaskBudgetError('host_task_reservation', 'valid reservation and terminal outcome required')
        with _locked(self.run):
            policy, state = self._load()
            matches = [r for r in state['records'] if r['id']==reservation_id]
            if len(matches)!=1 or matches[0]['status']!='reserved':
                raise HostTaskBudgetError('host_task_reservation', 'reservation absent or already completed')
            record = matches[0]
            evidence = evidence or (result.get('evidence') if type(result) is dict else None) or record.get('evidence')
            if outcome=='completed':
                if evidence is None:
                    raise HostTaskBudgetError('host_task_evidence', 'completed verification needs persisted evidence')
                full, passed, evidence = self._completed_result(record,result,evidence,
                    policy['required_case_ids'],state['records'])
                record.update(full_l2_passed=passed, evidence=evidence,
                              source=full.get('source'), report=full.get('report'),
                              validity=full.get('validity'),
                              performance=_performance_facts(full))
            elif evidence is not None:
                record['evidence'] = self._evidence_path(evidence)
            state['max_elapsed_s'] = self._elapsed(state)
            record.update(status=outcome, finished_epoch=time.time(),
                finished_after_deadline=state['max_elapsed_s'] >= policy['limits']['wall_clock_s'])
            _write_state(self.run, state, 'budget_settled')
            from .gpu_lease import release as release_lease
            release_lease(self.run, record.get('gpu_lease_owner') or '')
            return self._snapshot(policy,state)

    def settle_owner(self, invocation_id, *, outcome='cancelled', reject_completed=False):
        """Settle only this owner's unfinished work after its supervisor reaped it.

        This internal API cannot independently prove worker death. It must be
        called by the owning supervisor after confirming worker_reaped. It does
        not complete evidence, refund reservations or establish remote shutdown.
        Repeated cleanup leaves settled records intact. With reject_completed,
        a response rejected by the supervisor also loses accepted-candidate
        credit while its verification facts and immutable artifacts remain.
        A cooperative signal handler may have recorded failed before the
        supervisor decides cancelled; that un-settled failure is normalized too.
        """
        invocation_id = _owner(invocation_id,required=True)
        if outcome not in ('cancelled','failed'):
            raise HostTaskBudgetError('host_task_reservation', 'owner settlement must be cancelled or failed')
        if type(reject_completed) is not bool:
            raise HostTaskBudgetError('host_task_reservation', 'reject_completed must be boolean')
        with _locked(self.run):
            policy,state = self._load()
            elapsed = self._elapsed(state)
            settled=[]
            for record in state['records']:
                cooperative_failure = (reject_completed and outcome=='cancelled'
                    and record['status']=='failed' and record.get('settled_by') is None)
                if (record.get('owner_invocation_id')==invocation_id
                        and (record['status']=='reserved'
                             or reject_completed and record['status']=='completed'
                             or cooperative_failure)):
                    if record['status'] in ('completed','failed'):
                        record['verification_status_before_settlement']=record['status']
                        record['verification_finished_epoch']=record.get('finished_epoch')
                        record['verification_finished_after_deadline']=record.get('finished_after_deadline')
                    record.update(status=outcome, finished_epoch=time.time(),
                        finished_after_deadline=elapsed >= policy['limits']['wall_clock_s'],
                        accepted_by_supervisor=False, settled_by='owning_supervisor',
                        settlement_basis='worker_reaped_response_rejected' if reject_completed else 'worker_reaped')
                    settled.append(record['id'])
            state['max_elapsed_s']=elapsed
            _write_state(self.run, state, 'budget_settled')
            from .gpu_lease import release as release_lease
            for record in state['records']:
                if record['id'] in settled:
                    release_lease(self.run, record.get('gpu_lease_owner') or '')
            return dict(self._snapshot(policy,state), settled_owner_invocation_id=invocation_id,
                        settled_reservation_ids=settled)
