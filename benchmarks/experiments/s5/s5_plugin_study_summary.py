"""Offline scoped-study tables; shape checks are not runtime/evidence attestation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.experiments.s5 import s5_minimal_tasks, s5_plugin_summary
from operator_agent.run_contract import digest

PLAN_SCHEMA = 's5-plugin-study-plan-v1'
SCHEMA = 's5-plugin-study-summary-v1'


def aggregate(plan, records, *, bootstrap_samples=10000, seed=20261029):
    """Keep all 24 assigned attempts, and restrict cost pairs to two successes."""
    tasks = s5_minimal_tasks.manifest()
    ids = [row['case_id'] for row in tasks['tasks']]
    if (not isinstance(plan, dict) or plan.get('schema') != PLAN_SCHEMA
            or plan.get('mode') != 'study' or plan.get('arms') != ['A', 'C']
            or plan.get('repeats') != 2 or plan.get('planned_runs') != 24
            or plan.get('manifest_hash') != tasks['hash']
            or [case.get('case_id') for case in plan.get('cases', [])] != ids
            or plan.get('schedule') != s5_plugin_summary.schedule(plan.get('cases'), 2)):
        raise ValueError('study summary requires the fixed prospective 24-slot grid')
    result = s5_plugin_summary.aggregate(plan, records,
        bootstrap_samples=bootstrap_samples, seed=seed)
    result.pop('hash')
    lookup = {(r['case_id'], r.get('repeat_id', 1), r['arm']): r for r in result['rows']}
    pairs = []
    for case_id in ids:
        for repeat_id in (1, 2):
            a, c = [lookup[case_id, repeat_id, arm] for arm in ('A', 'C')]
            success_a, success_c = map(s5_plugin_summary._success, (a, c))
            verdict = ('unknown' if None in (success_a, success_c) else
                       'win' if success_c > success_a else
                       'loss' if success_c < success_a else 'tie')
            both_passed = success_a == success_c == 1
            costs = {}
            for key in ('calls', 'wall_seconds'):
                values = [(r.get('model_calls') or {}).get('reserved') if key == 'calls'
                          else r.get(key) for r in (a, c)]
                costs[key] = (values[1] - values[0] if both_passed
                              and all(v is not None for v in values) else None)
            pairs.append(dict(case_id=case_id, repeat_id=repeat_id,
                a_success=success_a, c_success=success_c, outcome=verdict,
                both_passed=both_passed, success_pair_cost_difference=costs))
    metrics = []
    for row in result['rows']:
        development = row.get('development') or {}
        verification = row.get('verification') or {}
        metrics.append(dict(case_id=row['case_id'], repeat_id=row.get('repeat_id', 1),
            arm=row['arm'], outcome_class=row.get('outcome_class'),
            model_calls_completed=(row.get('model_calls') or {}).get('completed'),
            development_verifications=development.get('development_used'),
            final_verifications=development.get('final_used'),
            first_complete_correct=row.get('first_complete_correct'),
            final_source_sha256=row.get('candidate_sha256'),
            final_verification_source_sha256=verification.get('source_sha256'),
            plugin_verification_observed=(row.get('development_audit') or {}).get(
                'plugin_verification_observed'),
            final_submission_refused=row.get('final_submission_refused', False)))
    result.update(schema=SCHEMA, study_scope='outer development and supervised L2 verification only',
        execution_policy=plan.get('execution_policy', 'formal'),
        exploratory_limitations=plan.get('exploratory_limitations'),
        pair_outcomes={state: sum(pair['outcome'] == state for pair in pairs)
                       for state in ('win', 'tie', 'loss', 'unknown')},
        attempt_pairs=pairs, attempt_metrics=metrics,
        l3_performance=dict(status='not_evaluated', qualified_successes=None),
        provider_billing='unknown unless separate trustworthy provider records are available',
        formal_s5_credit=False,
        evidence_attestation='offline table consistency only; does not attest runtime or artifact truth')
    return dict(result, hash=digest(result))


def main(argv=None):
    """Recompute tables only; never load execution credentials or invoke a model/GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--execution', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    if digest({k: v for k, v in plan.items() if k != 'hash'}) != plan.get('hash'):
        raise ValueError('analysis plan hash mismatch')
    records = [json.loads(path.read_text()) for path in sorted((args.execution / 'records').glob('*.json'))]
    result = aggregate(plan, records)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write('\n')
    print(json.dumps(dict(hash=result['hash'], counts=result['counts'], planned=result['planned'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
