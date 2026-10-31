"""Create a reviewable, non-executing S5 task manifest draft.

The ten existing quality tasks retain their primary and holdout definitions.
New tasks are explicit candidates with unknown prior exposure, not a sealed
experiment or a claim of independent unseen data. No model or GPU is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from operator_agent.decide.bench_cases import BenchmarkCase, TensorSpec
from operator_agent.core.contract import freeze_from_task
from operator_agent.decide.correctness import CorrectnessCase, as_suite
from operator_agent.decide.correctness_profiles import with_default_correctness
from operator_agent.decide.quality_suite import suite_manifest
from operator_agent.core.run_contract import digest
from operator_agent.core.schemas import ConstraintSpec, Precision, SourceSpec, TargetSpec, TaskSchema

# These are candidate tasks, not successful preflight or holdout claims.
NEW_CASES = (
    ('matmul-1024-fp16', 'matmul', (1024, 1024, 1024), 'fp16', 101),
    ('matmul-tail-large-fp16', 'matmul', (1001, 1003, 997), 'fp16', 103),
    ('softmax-large-fp32', 'softmax', (4096, 1024), 'fp32', 107),
    ('layernorm-large-fp32', 'layernorm', (4096, 1024), 'fp32', 109),
)
HOLDOUT_SEEDS = (317, 619)


def _case(name, operator, shape, precision, seed):
    if operator == 'matmul':
        m, n, k = shape
        tensors = (TensorSpec('A', (m, k), precision), TensorSpec('B', (k, n), precision))
    else:
        tensors = (TensorSpec('X', shape, precision),)
    return BenchmarkCase(name=name, tensors=tensors, seed=seed)


def new_row(spec):
    name, operator, shape, precision, seed = spec
    primary = _case(name, operator, shape, precision, seed)
    task = TaskSchema(task_id=name, created_at=0, source=SourceSpec(kernel_type=operator),
        target=TargetSpec(hardware='nvidia_gpu', dsl='triton'),
        constraints=ConstraintSpec(precision=Precision(precision), cases=(primary,)))
    task = with_default_correctness(task)
    contract = freeze_from_task(task)
    checks = [CorrectnessCase(name+'-primary', primary, 'random', 'contiguous')]
    checks.extend(CorrectnessCase(name+'-holdout-'+str(n), _case(name, operator, shape, precision, n),
                                  'random', 'contiguous') for n in HOLDOUT_SEEDS)
    return dict(case_id=name, kernel_type=operator, shape=list(shape), precision=precision,
        pattern='random', layout='contiguous', seed=seed, holdout_seeds=list(HOLDOUT_SEEDS),
        contract_hash=contract.hash, correctness_profile=task.constraints.correctness_profile,
        correctness_suite_hash=as_suite(task.constraints.correctness_suite).digest(),
        evaluation_inputs=[c.content_dict() for c in checks], exposure='unknown',
        performance_feasibility='unknown')


def manifest():
    existing = suite_manifest()
    old = [dict(row, exposure='known_historical', performance_feasibility='unknown')
           for row in existing['cases']]
    rows = old + [new_row(row) for row in NEW_CASES]
    if len(rows) != 14 or len({row['case_id'] for row in rows}) != 14:
        raise ValueError('invalid task identity')
    body = dict(schema='s5-task-manifest-draft-v1', status='draft_not_sealed',
        scope='shared final evaluation, not an authorization to dispatch',
        quality_suite_hash=existing['hash'], holdouts_are_secret=False,
        production_entry_note='CLI declarations do not inject quality evaluation pattern/layout/seed',
        tasks=rows)
    return dict(body, hash=digest(body))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    value = manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps({'path': str(args.output), 'hash': value['hash'], 'tasks': len(value['tasks']),
                      'status': value['status']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
