"""Offline A/C summaries for the plugin-only protocol.

All assigned attempts remain in the denominator. Repeated attempts are averaged
within each task before the descriptive task-cluster bootstrap. This module
checks record shape and identity, not the truth of the underlying evidence.
"""
from __future__ import annotations

import collections
import math
import random

from operator_agent.core.run_contract import digest

ARMS = ('A', 'C')
ORDINARY_NONCOMPLETION = frozenset({
    'call_budget_exhausted', 'output_budget_exhausted', 'per_call_output_exhausted',
    'wall_clock_exhausted', 'no_submission',
})
USAGE_FIELDS = ('inputTokens', 'cacheReadTokens', 'cacheWriteTokens', 'outputTokens', 'totalTokens')


def _grid(cases, repeats):
    if not isinstance(cases, list) or not cases or type(repeats) is not int or repeats < 1:
        raise ValueError('invalid task grid')
    ids = []
    for case in cases:
        name = case.get('case_id') if isinstance(case, dict) else None
        if not isinstance(name, str) or not name.strip() or name in ids:
            raise ValueError('invalid or duplicate task identity')
        ids.append(name)
    return ids


def schedule(cases, repeats=1):
    """Return every task/repeat/A-C slot once, alternating the first arm."""
    ids = _grid(cases, repeats)
    return [dict(case_id=name, repeat_id=r + 1, arm=arm)
            for r in range(repeats) for i, name in enumerate(ids)
            for arm in (ARMS if (i + r) % 2 == 0 else ARMS[::-1])]


def _cost(values, *, integer=False):
    for value in values:
        if value is None:
            continue
        if ((type(value) is not int if integer else type(value) not in (int, float))
                or (type(value) is float and not math.isfinite(value)) or value < 0):
            raise ValueError('invalid cost or usage')
    known = [value for value in values if value is not None]
    return dict(observed_sum=sum(known) if known else None, observed_rows=len(known),
                unknown_rows=len(values) - len(known), complete=len(known) == len(values))


def _mapping(row, field):
    value = row.get(field)
    if value is not None and not isinstance(value, dict):
        raise ValueError('invalid ' + field)
    return value or {}


def _success(row):
    if row.get('execution_state') == 'not_executed' or row.get('missing'):
        return None
    if row.get('outcome_class') == 'infrastructure_error':
        return None
    if row.get('error') and row.get('error') != 'TimeoutExpired':
        return None
    if row['status'] in ('passed', 'failed'):
        return int(row['status'] == 'passed')
    if row.get('outcome_class') in ORDINARY_NONCOMPLETION:
        return 0
    return None


def aggregate(plan, records, bootstrap_samples=10000, seed=20261029):
    """Summarize a fixed two-arm grid without promoting unknowns to failures."""
    if (not isinstance(plan, dict) or plan.get('arms') != list(ARMS)
            or not isinstance(plan.get('hash'), str) or not plan['hash']):
        raise ValueError('plugin-only plan requires A/C arms and a plan hash')
    if type(bootstrap_samples) is not int or not 100 <= bootstrap_samples <= 100000:
        raise ValueError('invalid bootstrap samples')
    if type(seed) is not int:
        raise ValueError('invalid bootstrap seed')
    repeats = plan.get('repeats', 1)
    ids = _grid(plan.get('cases'), repeats)
    slots = schedule(plan['cases'], repeats)
    if 'planned_runs' in plan and (type(plan['planned_runs']) is not int
                                   or plan['planned_runs'] != len(slots)):
        raise ValueError('planned denominator does not match task grid')
    expected = {(s['case_id'], s['repeat_id'], s['arm']) for s in slots}
    if not isinstance(records, list):
        raise ValueError('records must be a list')
    lookup = {}
    for row in records:
        if not isinstance(row, dict):
            raise ValueError('invalid record')
        repeat_id = row.get('repeat_id', 1)
        if (not isinstance(row.get('case_id'), str) or not isinstance(row.get('arm'), str)
                or type(repeat_id) is not int):
            raise ValueError('invalid record identity')
        key = row['case_id'], repeat_id, row['arm']
        if key not in expected or key in lookup:
            raise ValueError('duplicate or unplanned record')
        if row.get('plan_hash') != plan['hash']:
            raise ValueError('record plan drift')
        if row.get('status') not in ('passed', 'failed', 'unknown'):
            raise ValueError('invalid verdict')
        state = row.get('execution_state')
        if state not in ('finished', 'interrupted', 'unknown', 'not_executed'):
            raise ValueError('invalid execution state')
        if row['status'] in ('passed', 'failed') and state != 'finished':
            raise ValueError('unfinished row cannot have a definitive verdict')
        lookup[key] = row
    rows = [lookup.get((slot['case_id'], slot['repeat_id'], slot['arm']),
                dict(slot, plan_hash=plan['hash'], status='unknown',
                     execution_state='not_executed', outcome_class='not_executed', missing=True))
            for slot in slots]
    groups = {}
    for arm in ARMS:
        subset = [r for r in rows if r['arm'] == arm]
        counts = collections.Counter(r['status'] for r in subset)
        costs = dict(calls=_cost([_mapping(r, 'model_calls').get('reserved') for r in subset], integer=True),
                     wall_seconds=_cost([r.get('wall_seconds') for r in subset]))
        usage = {name: _cost([_mapping(r, 'dsh_usage').get(name) for r in subset], integer=True)
                 for name in USAGE_FIELDS}
        progress_fields = ('model_steps_observed', 'tool_calls', 'exploration_tool_calls',
                           'first_candidate_write_step', 'first_verification_step',
                           'development_verifications', 'development_feedback_observed')
        progress = [dict(case_id=r['case_id'], repeat_id=r.get('repeat_id', 1),
                         **{name: _mapping(r, 'workflow_progress').get(name)
                            for name in progress_fields}) for r in subset]
        groups[arm] = dict(planned=len(subset), passed=counts['passed'], failed=counts['failed'],
            unknown=counts['unknown'], not_executed=sum(r['execution_state'] == 'not_executed' for r in subset),
            success_rate=counts['passed'] / len(subset), costs=costs, dsh_usage=usage,
            mcp_used=sum(_mapping(r, 'development_audit').get('plugin_verification_observed') is True
                         for r in subset), workflow_progress=dict(
                basis='retained DSH model-step/tool events; model steps are not provider API calls',
                exploration_scope='obvious search/read/capability and shell inspection only',
                values=progress))
    completed = all(not r.get('missing') and r['execution_state'] != 'not_executed' for r in rows)
    by_key = {(r['case_id'], r.get('repeat_id', 1), r['arm']): r for r in rows}
    differences = []
    for name in ids:
        values = {arm: [_success(by_key[name, r, arm]) for r in range(1, repeats + 1)] for arm in ARMS}
        differences.append(None if any(value is None for items in values.values() for value in items)
                           else (sum(values['C']) - sum(values['A'])) / repeats)
    inference_allowed = completed and all(value is not None for value in differences)
    interval = None
    if inference_allowed:
        rng = random.Random(seed)
        samples = sorted(sum(rng.choice(differences) for _ in ids) / len(ids)
                         for _ in range(bootstrap_samples))
        interval = [samples[int(.025 * (bootstrap_samples - 1))],
                    samples[int(.975 * (bootstrap_samples - 1))]]
    comparison = dict(
        success_rate_difference=sum(differences) / len(ids) if inference_allowed else None,
        recorded_success_rate_difference=groups['C']['success_rate'] - groups['A']['success_rate'],
        paired_task_differences=differences, task_cluster_percentile_95=interval,
        inference_allowed=inference_allowed, bootstrap_samples=bootstrap_samples, bootstrap_seed=seed,
        note='descriptive task-cluster bootstrap; repeats averaged within task; unknown infrastructure suppresses inference',
        cost_per_attempt_difference={
            key: ((groups['C']['costs'][key]['observed_sum'] - groups['A']['costs'][key]['observed_sum'])
                  / (len(ids) * repeats))
            if groups['C']['costs'][key]['complete'] and groups['A']['costs'][key]['complete'] else None
            for key in ('calls', 'wall_seconds')})
    body = dict(schema='s5-plugin-summary-v1', plan_hash=plan['hash'], arms=list(ARMS),
        planned=len(rows), received=len(records), dispatched_runs=sum(r['execution_state'] != 'not_executed' for r in rows),
        counts={state: sum(r['status'] == state for r in rows) for state in ('passed', 'failed', 'unknown')},
        groups=groups, comparisons={'C-minus-A': comparison}, rows=rows, completed_denominator=completed,
        formal_s5_credit=False, primary_comparison='C-minus-A',
        billing='all assigned attempts retained; missing costs and DSH usage stay unknown; DSH usage is not a provider invoice',
        limitations=['fixed project tasks may have prior exposure; no training-unseen claim',
                     'small task-cluster bootstrap does not establish general operator-development gains'])
    return dict(body, hash=digest(body))
