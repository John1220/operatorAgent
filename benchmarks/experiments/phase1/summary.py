"""Read-only reconciliation of retained cells, ledgers, events and final verdicts."""
import copy
import hashlib
import math
from pathlib import Path
import re

from operator_agent.core.schemas import VerificationLevel, VerificationResult
from operator_agent.generate.repair_loop import _full_l2_passed
from operator_agent.execute.budget import BudgetPlan
from operator_agent.execute import s5_shared_budget as shared

from .events import collect_events
from . import plan as plans


def _read(path, problems):
    try:
        return plans.read(path)
    except (OSError, ValueError) as exc:
        problems.append(str(path)+': '+type(exc).__name__)
        return None


def _model_budget(cell, config, problems):
    result = dict(complete=False, reserved_calls=None, fixture_completed_calls=None,
                  provider_dispatches=None, api_usage=None, cost_usd=None, shared=None)
    if config is None:
        return result
    try:
        common = shared.snapshot(cell/'budget.sqlite', expected_task_id=config['shared_task_id'])
    except (shared.BudgetError, OSError) as exc:
        problems.append('shared model budget unavailable: '+str(exc))
        return result
    result.update(shared=common, reserved_calls=common['reserved_calls'])
    journal = cell/'work'/'model-journal'
    local = _read(journal/'summary.json', problems)
    calls = [_read(path, problems) for path in sorted(journal.glob('call-*.json'))]
    limits = _read(journal/'limits.json', problems)
    identity = _read(journal/'identity.json', problems)
    expected_calls = config['budget']['max_llm_calls']
    known = (isinstance(local, dict) and not local.get('aborted') and not common['aborted']
        and local.get('policy') == 'production_task' and local.get('attempts') == calls
        and local.get('identity') == identity and local.get('limits') == limits
        and isinstance(identity, dict) and identity.get('test_only') is True
        and identity.get('transport') == 'none' and identity.get('model') == 'fake-fixture-v1'
        and common['max_calls'] == expected_calls and isinstance(limits, dict)
        and limits.get('max_calls') == expected_calls
        and common['max_output_tokens'] == limits.get('max_output_tokens') == 40960
        and common['total_output_tokens'] == limits.get('total_output_token_reservations')
            == 40960*expected_calls
        and limits.get('campaign_seconds') == config['budget']['wall_clock_s']
        and local.get('reserved_calls') == len(calls) == common['reserved_calls']
        and [c.get('call') if isinstance(c, dict) else None for c in calls]
            == list(range(1, len(calls)+1)))
    if known:
        for local_call, common_call in zip(calls, common['calls']):
            if (any(local_call.get(k) != common_call.get(k) for k in
                    ('caller', 'request_sha256', 'output_tokens_reserved'))
                    or local_call.get('dispatched_monotonic') is not None
                    or common_call.get('dispatched_monotonic') is not None
                    or (local_call.get('completed_monotonic') is not None
                        and local_call.get('usage_source') != 'test_fixture')):
                known = False
                break
    result['complete'] = bool(known)
    if known:
        result.update(fixture_completed_calls=sum(c.get('completed_monotonic') is not None for c in calls),
                      provider_dispatches=0, cost_usd=0., cost_source='test_fixture')
    else:
        problems.append('local/shared model journal coverage is incomplete or inconsistent')
    # Undispatched fixture reservations are not provider completions or API usage.
    return result


def _oracle_budget(cell, plan, problems):
    value = _read(cell/'work'/'verification-budget.json', problems)
    complete = False
    if isinstance(value, dict) and isinstance(value.get('plan'), dict):
        spent = value.get('spent', {})
        caps = value.get('plan', {}).get('phase_caps', {})
        total = plan['budget']['oracle_total']
        complete = (value.get('plan', {}).get('oracle_total') == total
            and isinstance(spent, dict) and isinstance(caps, dict)
            and caps == BudgetPlan(oracle_total=total).phase_caps()
            and set(spent) == {'calibration', 'explore', 'refine', 'confirm'}
            and all(type(n) is int and n >= 0 for n in spent.values())
            and sum(spent.values()) == value.get('spent_all')
            and value['spent_all'] <= total and type(caps.get('confirm')) is int
            and spent['confirm'] <= caps['confirm']
            and sum(spent[k] for k in ('calibration', 'explore', 'refine')) <= total-caps['confirm']
            and value.get('reservation_policy') == 'pre_dispatch_non_refundable')
    if not complete:
        problems.append('verification reservations unavailable or inconsistent')
    return dict(complete=bool(complete), evidence=value)


def _events(cell, arm, record, problems):
    path = cell/'work'/'events.json'
    events = _read(path, problems) if path.is_file() else []
    if not isinstance(events, list):
        problems.append('event stream must be a list')
        events = [events]
    coverage = dict(conversation='missing', tools='missing', external_edits='missing')
    # Only a terminal normal worker receipt seals this adapter's tool capture.
    development = record.get('development')
    worker_path = cell/'work'/('baseline-result.json' if arm == 'A' else 'delegate-result.json')
    if (path.is_file() and worker_path.is_file() and record.get('worker_result') == str(worker_path)
            and isinstance(development, dict) and not development.get('stopped_reason')):
        coverage['tools'] = 'complete'
    checked = copy.deepcopy(events)
    for event in checked:
        if not isinstance(event, dict):
            continue
        ref = event.get('source')
        if not isinstance(ref, dict):
            continue
        try:
            raw_source = Path(ref['path'])
            source = raw_source.resolve()
            if cell not in source.parents or raw_source.is_symlink():
                raise ValueError('source is outside its own cell')
            if hashlib.sha256(source.read_bytes()).hexdigest() != ref.get('sha256'):
                raise ValueError('event source hash mismatch')
        except (KeyError, OSError, ValueError, TypeError) as exc:
            problems.append('unbound event source: '+str(exc))
            event['source'] = None
            coverage['tools'] = 'partial'
    result = collect_events(checked, coverage, arm=arm)
    result['raw_events'] = events
    return result


def _verdicts(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError('missing verification verdicts')
    if any(not isinstance(v, dict) or type(v.get('passed')) is not bool
           or not isinstance(v.get('metrics'), dict) for v in rows):
        raise ValueError('invalid verification verdict')
    return [VerificationResult(VerificationLevel(v['level']), v['passed'],
                               metrics=v['metrics']) for v in rows]


def _final(cell, record, task, plan, problems):
    result = dict(complete=False, full_l2_passed=None, levels=None, speedup=None,
                  validity=None, gpu_acceptance=False)
    outcome = record.get('final_evaluation')
    if not isinstance(outcome, dict) or outcome.get('stopped_reason') or outcome.get('returncode') != 0:
        return result
    value = _read(cell/'final'/'final-result.json', problems)
    if not isinstance(value, dict):
        return result
    pinned = cell/'selected.py'
    actual = hashlib.sha256(pinned.read_bytes()).hexdigest() if pinned.is_file() else None
    verified = value.get('result', {})
    if not isinstance(verified, dict):
        problems.append('final result must be an object')
        return result
    context = verified.get('execution_context', {})
    if (value.get('cell_id') != record['cell_id'] or value.get('plan_hash') != plan['hash']
            or value.get('source_hash') != actual or actual != record.get('selected_source_hash')
            or value.get('contract_hash') != task['contract_hash']
            or verified.get('contract_hash') != task['contract_hash']
            or type(value.get('verification_count')) is not int or value['verification_count'] != 1
            or context.get('test_only') is not True or context.get('gpu_acceptance') is not False):
        problems.append('independent final evaluation source/contract/context mismatch')
        return result
    verdicts = verified.get('verdicts', [])
    try:
        parsed = _verdicts(verdicts)
    except (ValueError, KeyError) as exc:
        problems.append('invalid final verdicts: '+str(exc))
        return result
    if (verified.get('levels') != {v.level.value: v.passed for v in parsed}
            or verified.get('passed') is not False):
        problems.append('test-only final levels/verdicts are inconsistent')
        return result
    l2 = next((v for v in verdicts if v.get('level') == 'L2'), None)
    l3 = next((v for v in verdicts if v.get('level') == 'L3'), {})
    metrics = l3.get('metrics') or {}
    if l3.get('passed') is True or metrics.get('measurement_can_gate') is True:
        problems.append('test-only final evaluation improperly claims GPU qualification')
        return result
    result.update(complete=True, full_l2_passed=_full_l2_passed(parsed),
        levels=verified.get('levels'), source_hash=actual, contract_hash=task['contract_hash'],
        speedup=metrics.get('speedup_vs_baseline'), validity=metrics.get('measurement_validity'),
        evidence=str(cell/'final'/'final-result.json'))
    return result


def _first_l2(cell, task, problems):
    observed = []
    for path in sorted((cell/'work').rglob('verification-event-*.json')):
        event = _read(path, problems)
        if not isinstance(event, dict):
            continue
        try:
            if (event.get('schema') != 'operator-verification-completion-v1'
                    or event.get('contract_hash') != task['contract_hash']
                    or not re.fullmatch('[0-9a-f]{64}', str(event.get('source_hash')))):
                raise ValueError('invalid level event identity')
            if event.get('level') != 'L2':
                continue
            stamp = event['timing']['completed_at']
            if type(stamp) not in (int, float) or not math.isfinite(stamp) or stamp < 0:
                raise ValueError('invalid level event timestamp')
            actual = _full_l2_passed(_verdicts(event['verdicts']))
            if event.get('full_l2_passed') is True and (event.get('passed') is not True or actual is not True):
                raise ValueError('full L2 summary contradicts retained verdicts')
            if actual is True and event.get('full_l2_passed') is True and event.get('passed') is True:
                observed.append(dict(completed_at=stamp, source_hash=event['source_hash'], evidence=str(path)))
        except (KeyError, ValueError, TypeError, OverflowError) as exc:
            problems.append('invalid L2 event '+str(path)+': '+str(exc))
    return min(observed, key=lambda r: r['completed_at']) if observed else None


def summarize(root):
    root = Path(root).resolve()
    plan = plans.load(root/'plan.json', check_implementation=False)
    execution = plans.read(root/'execution.json')
    if execution.get('plan_hash') != plan['hash']:
        raise ValueError('execution/plan identity mismatch')
    tasks = {task['task_id']: task for task in plan['tasks']}
    rows = []
    for slot in plan['schedule']:
        cell = root/'cells'/slot['cell_id']
        problems = []
        record = _read(cell/'record.json', problems)
        if not isinstance(record, dict):
            problems.append('cell record unavailable or malformed')
            record = dict(slot, phase='unknown')
        if any(record.get(key) != value for key, value in dict(slot, plan_hash=plan['hash']).items()):
            problems.append('cell record identity mismatch')
            record = dict(slot, phase='unknown', reason='record_identity_mismatch')
        config = _read(cell/'cell-config.json', problems) if (cell/'cell-config.json').is_file() else None
        if config is not None and not isinstance(config, dict):
            problems.append('cell configuration must be an object')
            config = None
        if config is not None and (config.get('plan_hash') != plan['hash']
                or config.get('cell_id') != slot['cell_id'] or config.get('budget') != plan['budget']
                or config.get('shared_task_id') != 'p1-'+plans.digest([plan['hash'], slot['cell_id']])[:24]
                or config.get('shared_ledger') != str(cell/'budget.sqlite')
                or config.get('run_dir') != str(cell/'work') or config.get('arm') != slot['arm']
                or config.get('declaration') != tasks[slot['task_id']]['declaration']
                or config.get('backend') != 'fake' or config.get('fake') != plan['fake']):
            problems.append('cell configuration mismatch')
            config = None
        model = _model_budget(cell, config, problems)
        oracle = _oracle_budget(cell, plan, problems)
        events = _events(cell, slot['arm'], record, problems)
        final = _final(cell, record, tasks[slot['task_id']], plan, problems)
        development = record.get('development') if isinstance(record.get('development'), dict) else {}
        final_outcome = record.get('final_evaluation') if isinstance(record.get('final_evaluation'), dict) else {}
        first_l2 = _first_l2(cell, tasks[slot['task_id']], problems)
        rows.append(dict(slot, phase=record.get('phase', 'unknown'), reason=record.get('reason'),
            model_budget=model, verification_budget=oracle, events=events,
            budget_auditable=bool(model['complete'] and oracle['complete'] and development and not problems),
            development_wall_s=development.get('elapsed_s'), final_wall_s=final_outcome.get('elapsed_s'),
            first_full_l2=first_l2,
            final_evaluation=final, problems=problems))
    return dict(schema='phase1-offline-summary-v1', plan_hash=plan['hash'],
        phase=execution.get('phase'), execution_context=plan['execution_context'],
        implementation_matches_current=plan['implementation'] == plans.implementation_identity(),
        planned_cells=8, retained_cells=len(rows), cells=rows,
        arms={arm: dict(planned=4, terminal=sum(r['phase'] in ('done', 'failed', 'cancelled')
            for r in rows if r['arm'] == arm), correct_candidates=sum(
            r['final_evaluation']['full_l2_passed'] is True for r in rows if r['arm'] == arm))
              for arm in ('A', 'B')},
        formal_comparison_credit=False, dsh_entry_verified=False, gpu_acceptance=False,
        workflow_benefit_established=False, human_minutes=None,
        scope='small offline fixtures; not the formal four-task GPU comparison')
