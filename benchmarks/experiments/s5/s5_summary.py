"""Offline S5 schedule and caller-reported result aggregation prototype.

This does not launch experiments or verify artifacts, isolation, origin or
provider usage. Its numbers are mock/caller-reported only and cannot establish
formal S5 results. Missing or inconsistent observations are never passes.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from benchmarks.experiments.s5.s5_task_manifest import manifest
from operator_agent.core.run_contract import digest

ARMS = ('A', 'B', 'C')
ROTATION = (('A', 'B', 'C'), ('B', 'C', 'A'), ('C', 'A', 'B'))
HASH = re.compile(r'[a-f0-9]{64}\Z')


def schedule():
    draft = manifest()
    rows = []
    for index, task in enumerate(draft['tasks']):
        for arm in ROTATION[index % len(ROTATION)]:
            rows.append(dict(sequence=len(rows) + 1, case_id=task['case_id'], arm=arm,
                             contract_hash=task['contract_hash']))
    body = dict(schema='s5-draft-schedule-v1', manifest_hash=draft['hash'],
                status='draft_not_sealed', rows=rows)
    return dict(body, hash=digest(body))


def _usage(record):
    """Return caller-reported provider totals only if all fields are explicit.

    This label is a caller assertion, not proof that provider usage was complete.
    Dollars, cache accounting and direct interception are deliberately absent.
    """
    usage = record.get('api_usage')
    if not isinstance(usage, dict) or usage.get('evidence') != 'provider_complete':
        return None
    prompt, completion = usage.get('prompt_tokens'), usage.get('completion_tokens')
    if any(type(n) is not int or n < 0 for n in (prompt, completion)):
        return None
    return dict(prompt_tokens=prompt, completion_tokens=completion)


def _observation(row, record):
    if record is None:
        return dict(**row, status='unexecuted', full_l2_passed=None,
                    api_usage=None, reason='no report supplied')
    if not isinstance(record, dict):
        raise ValueError('result record must be an object')
    state = record.get('execution_state')
    if state not in ('finished', 'interrupted', 'unknown'):
        raise ValueError('invalid execution state')
    reported = record.get('final_l2')
    result = None
    reason = 'execution not finished or final evidence absent'
    if state == 'finished' and isinstance(reported, dict):
        value = reported.get('full_l2_passed')
        source = reported.get('source_sha256')
        if (type(value) is bool and reported.get('contract_hash') == row['contract_hash']
                and isinstance(source, str) and HASH.fullmatch(source)):
            result = value
            reason = 'caller-reported L2 and identities; artifact not verified'
        else:
            reason = 'malformed or mismatched final evidence'
    return dict(**row, status=('reported_pass' if result is True else
                              'reported_fail' if result is False else 'unknown'),
                full_l2_passed=result, api_usage=_usage(record), reason=reason)


def summarize(records):
    """Aggregate caller-provided records against every row of the draft schedule.

    Duplicate or foreign task/arm identities are errors, never a reason to
    choose a convenient run. A pair is comparable only if both explicitly report
    finished and identity-bound boolean outcomes. This checks shape, not truth.
    """
    if not isinstance(records, list):
        raise ValueError('records must be a list')
    planned = schedule()
    identities = {(row['case_id'], row['arm']) for row in planned['rows']}
    provided = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError('result record must be an object')
        key = (record.get('case_id'), record.get('arm'))
        if any(not isinstance(value, str) for value in key):
            raise ValueError('invalid task or arm identity')
        if key not in identities:
            raise ValueError('unplanned task or arm')
        if key in provided:
            raise ValueError('duplicate task or arm')
        provided[key] = record
    rows = [_observation(row, provided.get((row['case_id'], row['arm'])))
            for row in planned['rows']]
    groups = {}
    for arm in ARMS:
        group = [r for r in rows if r['arm'] == arm]
        counts = {state: sum(r['status'] == state for r in group)
                  for state in ('unexecuted', 'unknown', 'reported_fail', 'reported_pass')}
        executed = [r for r in group if r['status'] != 'unexecuted']
        known = [r['api_usage'] for r in executed if r['api_usage'] is not None]
        groups[arm] = dict(planned=len(group), counts=counts,
                           usage=dict(reported_complete=len(known),
                                      unknown=len(executed)-len(known),
                                      prompt_tokens=(sum(v['prompt_tokens'] for v in known) if known else None),
                                      completion_tokens=(sum(v['completion_tokens'] for v in known) if known else None),
                                      total_is_partial=len(known) != len(executed)))
    lookup = {(r['case_id'], r['arm']): r for r in rows}
    pairs = {}
    for left, right in (('B', 'A'), ('C', 'B')):
        comparisons = []
        for task in manifest()['tasks']:
            name = task['case_id']
            a, b = lookup[name, left], lookup[name, right]
            if a['full_l2_passed'] is None or b['full_l2_passed'] is None:
                comparison = None
            else:
                comparison = int(a['full_l2_passed']) - int(b['full_l2_passed'])
            comparisons.append(dict(case_id=name, difference=comparison))
        known = [r['difference'] for r in comparisons if r['difference'] is not None]
        pairs[left+'-minus-'+right] = dict(planned=len(comparisons), comparable=len(known),
            unknown=len(comparisons)-len(known),
            mean_difference=(sum(known)/len(known) if known else None), tasks=comparisons)
    return dict(schema='s5-offline-summary-prototype-v1', schedule_hash=planned['hash'],
        manifest_hash=planned['manifest_hash'], mock_or_caller_reported_only=True,
        formal_s5_credit=False, artifact_identity_verified=False,
        scope='No artifacts, provenance, isolation or provider usage independently verified',
        groups=groups, pairs=pairs, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, help='JSON array of normalized caller reports; omitted means none')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.records.read_text(encoding='utf-8')) if args.records else []
    result = summarize(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(path=str(args.output), rows=len(result['rows']), formal_s5_credit=False)))


if __name__ == '__main__':
    main()
