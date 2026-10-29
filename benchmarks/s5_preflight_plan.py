"""Frozen plan for the S5 three-group preflight (one task, arms A/B/C).

The plan is written once, hashed, and refuses to be edited or reused. It is a
*preflight*: one task, one run per arm, fake model, mock verification. Its
records never count toward the formal S5 denominator, and the plan states that
before any arm runs so the result cannot be selected after the fact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from benchmarks import s5_arm_env, s5_runtime, s5_task_manifest
from operator_agent.run_contract import digest

SCHEMA = 's5-three-group-preflight-plan-v1'
CASE_ID = 'layernorm-tail'
TASK = ('Implement a Python function kernel(X) for layer normalization on the '
        'last axis with epsilon 1e-5. Return only the source code.')
ARMS = s5_arm_env.ARMS
MAX_PLAN_BYTES = 65536


def build(*, case_id=CASE_ID, task_text=TASK, model='fake-model',
          max_model_calls=2, max_output_tokens=1024, total_output_tokens=2048,
          wall_clock_s=300, verify_backend='mock', arms=ARMS, repeats=1):
    """Compose the preflight plan; the caller still has to persist it once."""
    if arms != ARMS:
        raise ValueError('preflight requires arms in the fixed order A,B,C')
    if repeats != 1:
        raise ValueError('preflight is one run per arm')
    if verify_backend not in ('mock', 'h20'):
        raise ValueError('unknown verification backend')
    for name, value in (('max_model_calls', max_model_calls),
                        ('max_output_tokens', max_output_tokens),
                        ('total_output_tokens', total_output_tokens),
                        ('wall_clock_s', wall_clock_s)):
        if type(value) is not int or value <= 0:
            raise ValueError('invalid budget ' + name)
    if not isinstance(task_text, str) or not task_text.strip() or len(task_text.encode()) > 4096:
        raise ValueError('invalid task text')
    manifest = s5_task_manifest.manifest()
    row = next((item for item in manifest['tasks'] if item['case_id'] == case_id), None)
    if row is None:
        raise ValueError('task is not in the draft manifest')
    runtime = s5_runtime_facts()
    body = dict(
        schema=SCHEMA, status='preflight_not_formal', preflight=True,
        formal_s5_credit=False, counts_toward_denominator=False,
        purpose='prove the three launch paths, shared verification submission and '
                'record pipeline for one task; not a quality or performance result',
        case_id=case_id, task_text=task_text, model=dict(id=model, provider='fixture',
                                                         real_provider=False),
        manifest_hash=manifest['hash'], contract_hash=row['contract_hash'],
        suite_hash=row['correctness_suite_hash'], arms=list(ARMS), repeats=repeats,
        budget=dict(max_model_calls=max_model_calls,
                    max_output_tokens_per_call=max_output_tokens,
                    total_output_tokens=total_output_tokens,
                    wall_clock_s=wall_clock_s),
        verify=dict(backend=verify_backend,
                    adapter=('s5_l2_evaluator.evaluate_mock_fixture' if verify_backend == 'mock'
                             else 'unwired_remote_adapter'),
                    final_feedback_allowed=False, mock_only=verify_backend == 'mock'),
        runtime=runtime,
        exclusion_note='this run is a preflight; its rows are excluded from any formal '
                       'denominator even if all three arms pass')
    return dict(body, hash=digest(body))


def s5_runtime_facts():
    deps = s5_runtime.deps_dir()
    facts = dict(deps_dir=str(deps), deps_present=s5_runtime.available(deps),
                 python=sys.executable)
    facts['tree'] = (s5_runtime.tree_digest(deps) if facts['deps_present'] else None)
    return facts


def write(plan, path):
    """Persist the plan exactly once and return the written path."""
    body = json.loads(json.dumps(plan, ensure_ascii=False, sort_keys=True, allow_nan=False))
    if body.get('schema') != SCHEMA or digest({k: v for k, v in body.items() if k != 'hash'}) != body.get('hash'):
        raise ValueError('plan hash mismatch')
    if body.get('formal_s5_credit') is not False or body.get('counts_toward_denominator') is not False:
        raise ValueError('preflight plan cannot claim formal credit')
    target = Path(path)
    with target.open('x', encoding='utf-8') as handle:
        json.dump(body, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')
    target.chmod(0o600)
    return target


def load(path):
    """Read and re-validate a frozen plan; any drift is an error, not a warning."""
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ValueError('plan missing')
    if target.stat().st_size > MAX_PLAN_BYTES:
        raise ValueError('plan too large')
    plan = json.loads(target.read_text(encoding='utf-8'))
    if not isinstance(plan, dict) or plan.get('schema') != SCHEMA:
        raise ValueError('not a preflight plan')
    stored = plan.pop('hash', None)
    if not isinstance(stored, str) or digest(plan) != stored:
        raise ValueError('plan hash mismatch')
    plan['hash'] = stored
    if (plan.get('status') != 'preflight_not_formal' or plan.get('formal_s5_credit') is not False
            or plan.get('counts_toward_denominator') is not False):
        raise ValueError('plan is not an excluded preflight')
    if plan.get('arms') != list(ARMS) or plan.get('repeats') != 1:
        raise ValueError('plan arm order or repeats changed')
    if plan.get('verify', {}).get('final_feedback_allowed') is not False:
        raise ValueError('plan allows verification feedback into the final answer')
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    plan = build()
    write(plan, args.output)
    print(json.dumps(dict(path=str(args.output), hash=plan['hash'],
                          case_id=plan['case_id'], arms=plan['arms']), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())