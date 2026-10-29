"""Three-arm preflight whose final answers all go through the shared service.

The launch surfaces, fake model and budget ledger are unchanged. What changes is
the submission path: each arm's final answer is handed to
``benchmarks.s5_verify_remote`` — the supervisor-side service that prepares the
task itself, records the candidate before evaluating it, and only then runs the
production verifier (mock today, SSH later).

The preflight plan and the service plan must agree on the semantic contract, so a
service prepared for a different task cannot silently judge these arms.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from benchmarks import (s5_preflight_plan, s5_runtime, s5_task_manifest,
                        s5_three_group_run, s5_verify_remote)
from operator_agent.run_contract import digest

SCHEMA = 's5-three-group-service-run-v1'


def _write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


def submitter(service_plan, service_plan_sha256):
    """Adapt the service to the runner's submission hook."""

    def submit(preflight_plan, arm, candidate, arm_dir):
        result = s5_verify_remote.submit(service_plan, arm=arm, candidate_text=candidate,
                                         output_dir=Path(arm_dir) / 'verification')
        verdict_sha256 = digest({key: result.get(key) for key in
                                 ('status', 'source_sha256', 'contract_hash', 'reason')})
        return dict(status=result['status'], reason=result.get('reason'),
                    backend=result.get('backend'), verdict_sha256=verdict_sha256,
                    service_plan_sha256=service_plan_sha256,
                    final_l2=dict(full_l2_passed=result.get('full_l2_passed'),
                                  source_sha256=result.get('source_sha256'),
                                  contract_hash=result.get('contract_hash'),
                                  verdict_sha256=verdict_sha256))

    return submit


def run(preflight_plan_path, output, service_plan_path, service_plan_sha256):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    preflight = s5_preflight_plan.load(preflight_plan_path)
    service_plan = s5_verify_remote.load_plan(service_plan_path, service_plan_sha256)
    # The two plans bind through the task identity, not through one hash: the
    # draft-manifest contract (with its quality suite) and the production
    # verifier's contract are different evaluation regimes on the same task.
    manifest = s5_task_manifest.manifest()
    if preflight['manifest_hash'] != manifest['hash']:
        raise ValueError('draft manifest identity changed')
    row = next((item for item in manifest['tasks']
                if item['case_id'] == preflight['case_id']), None)
    if row is None:
        raise ValueError('preflight task is not in the draft manifest')
    declaration = service_plan['declaration']
    shape, precision = [list(row['shape'])], row['precision']
    if ([list(item) for item in declaration.get('shapes') or []] != shape
            or declaration.get('precision') != precision
            or row['kernel_type'] not in str(declaration.get('request', ''))):
        raise ValueError('verification service was prepared for a different task')
    _write_once(output / 'binding.json', dict(
        schema=SCHEMA + '-binding', preflight_plan_hash=preflight['hash'],
        service_plan_sha256=service_plan_sha256,
        preflight_contract_hash=preflight['contract_hash'],
        service_contract_hash=service_plan['contract_hash'],
        regime_note='draft-manifest quality suite is a different evaluation regime '
                    'from the production verifier contract bound here',
        declaration=declaration, backend=service_plan['backend'],
        mode=service_plan['mode'], task_case_id=preflight['case_id'],
        task_shape=shape, task_precision=precision, task_text=preflight['task_text'],
        started_unix=time.time(), runtime=s5_runtime.tree_digest(s5_runtime.deps_dir())
        if s5_runtime.available() else None))
    hook = submitter(service_plan, service_plan_sha256)
    records = []
    for arm in preflight['arms']:
        record = s5_three_group_run.run_arm(preflight, arm, output / 'arms' / arm, submit=hook)
        _write_once(output / 'records' / ('arm-' + arm + '.json'), record)
        records.append(record)
    summary = s5_three_group_run.aggregate(preflight, records)
    _write_once(output / 'aggregate.json', summary)
    facts = dict(schema=SCHEMA, preflight=True, formal_s5_credit=False,
                 counts_toward_denominator=False, service_plan_sha256=service_plan_sha256,
                 service_backend=service_plan['backend'], service_mode=service_plan['mode'],
                 contract_hash=preflight['contract_hash'], aggregate_hash=summary['hash'],
                 counts=summary['counts'], execution_states=summary['execution_states'],
                 limits=summary['limits'])
    _write_once(output / 'service.json', facts)
    return dict(summary, service=facts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight-plan', required=True, type=Path)
    parser.add_argument('--service-plan', required=True, type=Path)
    parser.add_argument('--service-plan-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    result = run(args.preflight_plan, args.output, args.service_plan, args.service_plan_sha256)
    print(json.dumps(dict(hash=result['hash'], counts=result['counts'],
                          wiring_complete=result['wiring_complete'],
                          service=result['service']['service_backend']), sort_keys=True))
    return 0 if result['wiring_complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())