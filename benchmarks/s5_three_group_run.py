"""Run one S5 task through arms A, B and C with a fake model and mock L2.

Every arm gets the same task text, the same scripted fake model behind the same
budget ledger, the same wall clock and the same supervisor-side verification
entry. Only the installed surface differs: A is bare DSH, B adds the core CLI to
``PATH``, C adds the core CLI plus the frozen plugin and MCP client.

This preflight proves the wiring, not quality, isolation, a real provider or a
GPU. Its records are marked as excluded from the formal denominator before the
first arm starts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

from benchmarks import s5_arm_env, s5_preflight_plan, s5_verify_service
from benchmarks.s5_fake_http import FakeHTTPServer
from benchmarks.s5_vertical_headless import NODE
from benchmarks.s5_vertical_validation import GOOD_FIXTURE
from operator_agent import s5_shared_budget as shared
from operator_agent.run_contract import digest

SCHEMA_RECORD = 's5-preflight-arm-record-v1'
_headless = s5_arm_env.start_headless
SCHEMA_AGGREGATE = 's5-three-group-preflight-aggregate-v1'
LIMITS = [
    'single-host macOS fixture; no Linux, container or namespace isolation',
    'scripted fake model; the provider is never contacted and usage stays unknown',
    'mock L2 only; no GPU, no performance measurement, no final-evaluation feedback',
    'arm labels describe installed configuration, not a verified process identity',
]


def _write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


def _write_text_once(path, text):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open('x', encoding='utf-8') as handle:
        handle.write(text)


def _sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _fixture_provider():
    """Scripted model: one streamed answer containing the trusted fixture."""
    calls = []

    def adapter(request):
        if request.get('stream') is not True:
            raise ValueError('fixture expects a streamed request')
        calls.append(_sha(json.dumps(request, sort_keys=True, separators=(',', ':'))))
        if len(calls) > 1:
            raise ValueError('one scripted request per arm')
        delta = {'id': 'fake-preflight-1', 'model': 'fake-model',
                 'choices': [dict(index=0, delta=dict(content=GOOD_FIXTURE),
                                  finish_reason=None)]}
        end = {'id': 'fake-preflight-1', 'model': 'fake-model',
               'choices': [dict(index=0, delta=dict(), finish_reason='stop')]}
        return [('data: ' + json.dumps(delta) + '\n\n').encode('utf-8'),
                ('data: ' + json.dumps(end) + '\n\n').encode('utf-8'),
                b'data: [DONE]\n\n']

    adapter.calls = calls
    return adapter


def _final_text(events):
    if not isinstance(events, list):
        return None
    texts = [item.get('text') for item in events
             if isinstance(item, dict) and item.get('type') == 'final']
    if len(texts) != 1 or not isinstance(texts[0], str):
        return None
    return texts[0]


def _budget_facts(snapshot):
    calls = snapshot.get('calls') if isinstance(snapshot, dict) else None
    if not isinstance(calls, list):
        return dict(reserved=None, completed=None, answered=None, aborted=snapshot.get('aborted'))
    return dict(reserved=snapshot.get('reserved_calls'), completed=snapshot.get('completed_calls'),
                answered=sum(1 for call in calls if call.get('outcome') == 'answered'),
                aborted=snapshot.get('aborted'),
                usage_present=sum(1 for call in calls if call.get('api_usage') is not None))


def _legacy_submission(plan, arm, candidate, arm_dir):
    verdict = s5_verify_service.submit(plan, arm=arm, candidate_text=candidate,
                                       run_dir=Path(arm_dir) / 'verification')
    fact = s5_verify_service.final_l2(plan, verdict)
    return dict(status=verdict['status'], reason=verdict.get('reason'),
                backend=verdict.get('backend'), verdict_sha256=fact['verdict_sha256'],
                final_l2=fact)


def run_arm(plan, arm, arm_dir, submit=None):
    """Execute one arm; never raises for an arm-level failure.

    ``submit`` lets the caller route the final answer through a different
    supervisor-side verification entry; the default keeps the offline mock judge.
    """
    arm_dir = Path(arm_dir)
    arm_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    task_id = plan['case_id'] + '-preflight-' + arm
    record = dict(schema=SCHEMA_RECORD, plan_hash=plan['hash'], case_id=plan['case_id'],
                  arm=arm, status='unknown', execution_state='unknown', final_l2=None,
                  api_usage=None, usage_evidence='missing', real_model_called=False,
                  gpu_used=False, isolation_verified=False, linux_host_verified=False,
                  mock_only=True, formal_s5_credit=False, limits=list(LIMITS),
                  error=None, reason=None)
    started = time.monotonic()
    try:
        ledger = arm_dir / 'budget.sqlite'
        budget = plan['budget']
        shared.create(ledger, task_id=task_id, max_calls=budget['max_model_calls'],
                      max_output_tokens=budget['max_output_tokens_per_call'],
                      total_output_tokens=budget['total_output_tokens'],
                      wall_clock_s=budget['wall_clock_s'])
        adapter = _fixture_provider()
        with FakeHTTPServer(ledger, task_id=task_id, model=plan['model']['id'],
                            participant='outer', adapter=adapter) as server:
            built = s5_arm_env.build(arm, base_url=server.base_url, arm_dir=arm_dir)
            Path(built['home']).mkdir(parents=True, exist_ok=True, mode=0o700)
            Path(built['work']).mkdir(parents=True, exist_ok=True, mode=0o700)
            _write_once(arm_dir / 'launch.json', built)
            probe = s5_arm_env.probe(built)
            _write_once(arm_dir / 'probe.json', probe)
            record['surface'] = built['surface']
            record['probe'] = dict(core_importable=probe['core_importable'],
                                   cli_on_path=probe['cli_on_path'],
                                   repo_canary_readable=probe['repo_canary_readable'],
                                   matches_expectation=probe['matches_expectation'])
            record['launch_sha256'] = built['patch_sha256']
            try:
                response = _headless(built, plan['task_text'],
                                     timeout=budget['wall_clock_s'])
            finally:
                # Record the shared ledger even when the arm itself failed, so a
                # partial run cannot disappear from the denominator.
                snapshot = shared.snapshot(ledger, expected_task_id=task_id)
                _write_once(arm_dir / 'budget.json', snapshot)
                record['budget'] = _budget_facts(snapshot)
        events = response['events']
        _write_once(arm_dir / 'headless-events.json', events)
        _write_once(arm_dir / 'headless-stderr.json',
                    dict(text=response['stderr'], sha256=_sha(response['stderr']),
                         activation_errors=response['stderr'].count('did not activate')))
        record['evidence'] = dict(
            events_sha256=digest(events),
            provider_requests=len(adapter.calls),
            provider_request_sha256=list(adapter.calls))
        if response['returncode'] != 0:
            record['execution_state'] = 'interrupted'
            record['reason'] = 'headless exited with %r' % response['returncode']
            return record
        candidate = _final_text(events)
        if candidate is None:
            record['execution_state'] = 'interrupted'
            record['reason'] = 'no single final answer in the DSH event stream'
            return record
        record['execution_state'] = 'finished'
        _write_text_once(arm_dir / 'candidate.txt', candidate)
        verdict = ((submit or _legacy_submission)(plan, arm, candidate, arm_dir)
                   if submit is None else submit(plan, arm, candidate, arm_dir))
        record['verification'] = dict(status=verdict['status'], reason=verdict.get('reason'),
                                      backend=verdict.get('backend'),
                                      verdict_sha256=verdict['verdict_sha256'],
                                      service_plan_sha256=verdict.get('service_plan_sha256'))
        record['final_l2'] = verdict['final_l2']
        record['status'] = verdict['status']
    except Exception as exc:  # noqa: BLE001 - an arm failure is data, not a crash
        record['error'] = type(exc).__name__
        record['reason'] = str(exc)[:200]
        if record['execution_state'] == 'unknown':
            record['execution_state'] = 'interrupted'
        record['status'] = 'unknown'
    record['wall_seconds'] = round(time.monotonic() - started, 3)
    return record


def _summary_row(record):
    return dict(case_id=record['case_id'], arm=record['arm'],
                execution_state=record['execution_state'],
                final_l2=record.get('final_l2'), api_usage=None,
                status=record['status'])


def aggregate(plan, records):
    """Pure aggregation: the same plan and records always yield the same hash."""
    if not isinstance(records, list) or len(records) != len(plan['arms']):
        raise ValueError('one record per arm is required')
    by_arm = {}
    for record in records:
        arm = record.get('arm')
        if arm not in plan['arms'] or arm in by_arm:
            raise ValueError('duplicate or unplanned arm record')
        if record.get('case_id') != plan['case_id']:
            raise ValueError('record task identity mismatch')
        by_arm[arm] = record
    rows = [_summary_row(by_arm[arm]) for arm in plan['arms']]
    counts = {state: sum(1 for row in rows if row['status'] == state)
              for state in ('passed', 'failed', 'unknown')}
    states = {state: sum(1 for row in rows if row['execution_state'] == state)
              for state in ('finished', 'interrupted', 'unknown')}
    body = dict(schema=SCHEMA_AGGREGATE, preflight=True, formal_s5_credit=False,
                counts_toward_denominator=False, plan_hash=plan['hash'],
                case_id=plan['case_id'], arms=list(plan['arms']),
                counts=counts, execution_states=states,
                model_calls=dict((arm, by_arm[arm].get('budget')) for arm in plan['arms']),
                provider_usage='unknown: scripted fake model reports no provider usage',
                rows=rows, record_hashes=dict((arm, digest(by_arm[arm])) for arm in plan['arms']),
                limits=list(LIMITS),
                wiring_complete=(states['finished'] == len(plan['arms'])
                                 and counts['unknown'] == 0))
    return dict(body, hash=digest(body))


def run(plan_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    plan = s5_preflight_plan.load(plan_path)
    started = dict(schema='s5-preflight-started-v1', plan_hash=plan['hash'],
                   arms=list(plan['arms']), case_id=plan['case_id'],
                   started_unix=time.time(), python=sys.executable,
                   platform=platform.platform(), node=NODE,
                   status='running')
    _write_once(output / 'started.json', started)
    records = []
    for arm in plan['arms']:
        record = run_arm(plan, arm, output / 'arms' / arm)
        _write_once(output / 'records' / ('arm-' + arm + '.json'), record)
        records.append(record)
    summary = aggregate(plan, records)
    _write_once(output / 'aggregate.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    summary = run(args.plan, args.output)
    print(json.dumps(dict(output=str(args.output), hash=summary['hash'],
                          counts=summary['counts'],
                          execution_states=summary['execution_states'],
                          wiring_complete=summary['wiring_complete']), sort_keys=True))
    return 0 if summary['wiring_complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())