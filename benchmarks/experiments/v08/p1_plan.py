"""P1 preregistration: performance PARITY of plugin-arm vs bare-arm delivered kernels.

What this round can and cannot claim, fixed before any L3 is re-run:

* The v0.8 A/C reproduction established a **delivery-count** gain on the weak host
  (27B/xhigh). It established **nothing about performance**: the independent final
  evaluation stopped at L2, so every delivered pair has `validity=None`.
* P1 re-runs an independent L3 on the pairs that delivered in **both** arms, under the
  **same frozen contract**, and asks only: is the plugin arm's delivered kernel **slower**
  than the bare arm's beyond the noise band? That is a *parity* question, not a gain question.
* The frozen contracts use **library baselines** (cuBLAS IEEE fp32 / SDPA fp16). The
  vs-library speedup is therefore expected < 1 and is recorded as **context**, never as a
  plugin performance claim. The "beat eager composition" positive claim from the task-family
  selection decision needs **new eager-baseline tasks** and is explicitly OUT of this round.
* Candidates are weak-host **one-shot deliveries**, not optimized kernels: this measures the
  performance of what each arm delivered, not the optimize loop (that is P2).

The judgment rule is pre-registered here as executable instrument (`judge_pair` /
`judge_series`); the readings themselves are taken at the GPU window, not invented here.
"""
from __future__ import annotations

import json
from pathlib import Path

from operator_agent.core.run_contract import digest

from . import r2_plan

SCHEMA = 'v08-workflow-performance-parity-v1'
PLAN_ID = 'operator-v08-workflow-p1-parity-v1'
PLAN_IDS = {PLAN_ID}
WITHDRAWN_PLAN_IDS: dict = {}

SUBJECT = ('performance parity of plugin-arm (C) vs bare-arm (A) delivered kernels on the '
           'v0.8 weak-host (Qwen3.8-27B, reasoning_effort=xhigh) reproduction tasks that '
           'delivered in BOTH arms, measured under their frozen contracts')

#: 两个任务沿用 R2 已标定的声明与窗口探针（同形状/同 dtype/同 mask）。声明逐字 reproduces
#: 交付时的契约哈希（见 PAIRS.contract_hash，validate 会重算核对）。
TASKS = [
    dict(task_id='matmul-4096-fp32',
         declaration=dict(request='fp32 4096x4096x4096 matmul, triton', shapes=[[4096, 4096, 4096]],
                          precision='fp32', dsl='triton', hardware='nvidia_gpu', min_speedup=1.0),
         baseline_context='cuBLAS IEEE fp32 (library); historical candidate ~0.37x, context only',
         mask=None),
    dict(task_id='attention-prefill-4096',
         declaration=dict(request='fp16 attention prefill 1x16x4096x128', shapes=[[1, 16, 4096, 128]],
                          precision='fp16', dsl='triton', hardware='nvidia_gpu', min_speedup=1.0),
         baseline_context='SDPA fp16 (library); historical candidate 0.8608x valid, context only',
         mask='nocausal'),
]

#: 只登记**两臂都交付**的配对（从 v6 两轮 cells 核出）。绑定用哈希，不用绝对路径：
#: 源码在仓外证据树，测量时按 (round, task, arm) 解析并重算哈希核对。
PAIRS = [
    dict(pair_id='p1-matmul-r1', task_id='matmul-4096-fp32', round='v6r1',
         contract_hash='f6062c865b786d1407fbc5608bbdbc8d0fcc33fd0e018c696c3ee613dd1f5d30',
         A_source_sha256='cadea73ad1d9b9fdb0bef2b0442c6621c3fdeebd3fa7cb8298abc0509eaef49a',
         C_source_sha256='6aea22b3a578ed56e4837aee2f4029b610207483d300c00931a67073bddf211d'),
    dict(pair_id='p1-matmul-r2', task_id='matmul-4096-fp32', round='v6r2',
         contract_hash='f6062c865b786d1407fbc5608bbdbc8d0fcc33fd0e018c696c3ee613dd1f5d30',
         A_source_sha256='f45a5b728d795f3912124e9e6066f5225109a1309dfe417e6cfb2fb6dcf0f951',
         C_source_sha256='b97209152a683c7c3937cb05ce774374a417b46f5f6fd15cb30b7917c2078cdc'),
    dict(pair_id='p1-attention-r2', task_id='attention-prefill-4096', round='v6r2',
         contract_hash='e9f75d60cf1c2b35372cf87e3c77d8b63e69514e426a558733363d0e2f96bfe5',
         A_source_sha256='8f84a010ebd57dd7a85f0ceb3426625c70759214bada0a8c238f7585867c40f2',
         C_source_sha256='abb98d4d8159584348513e823c877ea569ae9f0418ec64ec32d8d9dd2d9275ba'),
]

#: 探针沿用 R2 那两份（同通道/同命令）；历史带宽只作参考，P1 窗口必须**重跑**，不许继承读数。
PROBES = {task: json.loads(json.dumps(r2_plan.PROBES[task]))
          for task in ('matmul-4096-fp32', 'attention-prefill-4096')}

_GATE_POLICY = dict(r2_plan._GATE_POLICY)

METRIC = dict(
    primary=('paired difference in independent final L3 median time (C minus A), per pair, '
             'only when BOTH arms reach valid L3 under the same frozen contract and bound source hash'),
    parity_rule=('a decidable pair is C_worse iff (C_median - A_median) / A_median * 100 > window_band_pct; '
                 'the series parity claim holds iff at least one pair is decidable AND no decidable pair is C_worse'),
    minimum_detectable_difference='the per-task window band (noise floor); a smaller gap is not decidable',
    valid_requires=['both arms full L2 passed on the bound source hash',
                    'both arms measurement_validity == "valid"',
                    're-frozen contract hash equals the delivered pair contract_hash',
                    'independent final evaluation (purpose=final, stop_after=L3) per arm'],
    secondary=['speedup vs the contract library baseline as CONTEXT (expected < 1), never a plugin claim'],
    never_report=['a degraded/unknown reading as parity or gain',
                  'beating cuBLAS/SDPA as a plugin performance claim',
                  'a vs-eager-composition speedup (no eager-baseline task in this round)',
                  'an unpaired average as a comparison', 'developer-time',
                  'cost inferred from token counters'],
)

STOP_RULES = dict(
    per_pair=['preflight action != "start" -> hold, do not launch and spend no budget',
              're-frozen contract hash != pair contract_hash -> refuse the pair (wiring, not candidate)',
              'recomputed source hash != bound source_sha256 -> refuse the pair'],
    per_series=dict(no_valid_pairs_in_a_row=2,
                    on_stop='record the series as unknown; do not launch a third pair',
                    why='two consecutive pairs without a valid L3 is a systematic window or wiring problem'),
    budget='stop at the frozen budget; never raise it mid-series to reach a threshold',
)

FINAL_EVALUATION = dict(independent=True, same_source_hash=True, stop_after='L3',
                        reserve_per_pair=2,
                        note='one reserved independent L3 per arm per pair; no generation, no model calls')

BOUNDARIES = [
    'single device (H20), single model family (Qwen3.8-27B/xhigh); n=3 pairs over 2 tasks, no significance',
    'candidates are weak-host one-shot deliveries, NOT optimized: this is delivery performance, not the optimize loop (P2)',
    'contracts use library baselines (cuBLAS IEEE fp32 / SDPA fp16); parity is baseline-agnostic, vs-library absolute is context only',
    'the vs-eager-composition positive claim (task-family selection) is OUT of scope: it needs new eager-baseline tasks, a future round',
    'a pair that never reaches valid L3 in both arms is not decidable; it stays in the denominator and the series may be unknown',
    'no developer-time claim; no performance claim beyond parity on these two tasks',
]

BUDGET_FIELDS = ('pairs', 'max_model_calls', 'max_verifications', 'wall_clock_s', 'gpu_minutes')


def _task(task_id):
    return next((row for row in TASKS if row['task_id'] == task_id), None)


def design():
    """The freezable part: no budget, no outcome, no credit."""
    body = dict(schema=SCHEMA, plan_id=PLAN_ID, status='draft_awaiting_budget',
                subject=SUBJECT,
                tasks=json.loads(json.dumps(TASKS)), pairs=json.loads(json.dumps(PAIRS)),
                probes=json.loads(json.dumps(PROBES)), measurement_gate_policy=dict(_GATE_POLICY),
                metric=dict(METRIC), stop_rules=json.loads(json.dumps(STOP_RULES)),
                final_evaluation=dict(FINAL_EVALUATION), boundaries=list(BOUNDARIES),
                budget={name: None for name in BUDGET_FIELDS},
                formal_parity_claim=False, formal_gain_claim=False,
                formal_comparison_credit=False, outcome=None)
    body['design_hash'] = digest(body)
    return body


def _positive(value):
    return type(value) in (int, float) and value == value and value > 0


def _hex64(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate(plan):
    """Return a list of problems; empty means the plan may be frozen/executed."""
    problems = []
    if not isinstance(plan, dict) or plan.get('schema') != SCHEMA:
        return ['not a P1 performance-parity preregistration']
    if plan.get('plan_id') in WITHDRAWN_PLAN_IDS:
        problems.append(f'plan_id {plan["plan_id"]} was withdrawn: {WITHDRAWN_PLAN_IDS[plan["plan_id"]]}')
    elif plan.get('plan_id') not in PLAN_IDS:
        problems.append(f'plan_id must be one of {sorted(PLAN_IDS)}')
    body = {key: value for key, value in plan.items() if key != 'hash'}
    if plan.get('hash') is not None and digest(body) != plan.get('hash'):
        problems.append('plan hash does not match its content')
    for flag in ('formal_parity_claim', 'formal_gain_claim', 'formal_comparison_credit'):
        if plan.get(flag) is not False:
            problems.append(f'a preregistration cannot set {flag}')
    if plan.get('outcome') is not None:
        problems.append('a preregistration cannot carry an outcome')

    tasks = plan.get('tasks')
    if not isinstance(tasks, (list, tuple)) or not tasks:
        problems.append('at least one task must be registered')
        tasks = ()
    by_id = {row.get('task_id'): row for row in tasks if isinstance(row, dict)}

    # 声明必须能建任务，且重算契约哈希 == 配对登记的契约（否则测的不是交付那份契约）。
    from operator_agent.app.hitl_tools import build_task
    from operator_agent.core.contract import freeze_from_task
    contract_of = {}
    for task_id, row in by_id.items():
        declaration = row.get('declaration')
        try:
            contract_of[task_id] = freeze_from_task(build_task(json.loads(json.dumps(declaration)))).hash
        except Exception as error:
            problems.append(f'task {task_id!r} declaration does not build a task: {error}')

    pairs = plan.get('pairs')
    if not isinstance(pairs, (list, tuple)) or not pairs:
        problems.append('at least one delivered pair must be registered')
        pairs = ()
    seen_pairs = set()
    for pair in pairs:
        if not isinstance(pair, dict):
            problems.append('pair is not an object')
            continue
        pid = pair.get('pair_id')
        if pid in seen_pairs:
            problems.append(f'duplicate pair_id {pid!r}')
        seen_pairs.add(pid)
        task_id = pair.get('task_id')
        if task_id not in by_id:
            problems.append(f'pair {pid!r} references unregistered task {task_id!r}')
            continue
        for arm in ('A', 'C'):
            if not _hex64(pair.get(f'{arm}_source_sha256')):
                problems.append(f'pair {pid!r} arm {arm} needs a 64-hex source_sha256')
        if not _hex64(pair.get('contract_hash')):
            problems.append(f'pair {pid!r} needs a 64-hex contract_hash')
        elif task_id in contract_of and pair['contract_hash'] != contract_of[task_id]:
            problems.append(f'pair {pid!r} contract_hash does not match the re-frozen task contract '
                            '(the measurement would not be on the delivered contract)')

    # 探针必须量**登记的那个任务**（dtype/op/shape/mask），复用 R2 的同一套命令解析。
    probes = plan.get('probes') or {}
    for task_id, row in by_id.items():
        probe = probes.get(task_id)
        if not isinstance(probe, dict):
            problems.append(f'task {task_id!r} has no registered probe')
            continue
        command = probe.get('command')
        if not isinstance(command, str) or not command.strip():
            problems.append(f'task {task_id!r} probe must declare its command')
            continue
        declaration = row.get('declaration') or {}
        precision = declaration.get('precision')
        probed = r2_plan._command_dtype(command)
        if precision and probed != precision:
            problems.append(f'task {task_id!r} precision={precision} but probe probes {probed}')
        try:
            kernel_type = r2_plan._declared_kernel_type(declaration)
        except Exception as error:
            problems.append(f'task {task_id!r} declaration does not build a task: {error}')
            kernel_type = None
        probed_op = r2_plan._command_op(command)
        if kernel_type and probed_op != kernel_type:
            problems.append(f'task {task_id!r} is {kernel_type!r} but probe probes {probed_op!r}')
        probed_shape = r2_plan._command_shape(command)
        shapes = [[int(v) for v in shape] for shape in (declaration.get('shapes') or [])]
        if probed_shape is not None and shapes and probed_shape not in shapes:
            problems.append(f'task {task_id!r} declares {shapes} but probe probes {probed_shape}')
        if not _positive(probe.get('observed_band_pct')):
            problems.append(f'task {task_id!r} probe needs a positive observed_band_pct')
        mask = row.get('mask')
        if mask is not None:
            import re
            if bool(re.search(r'--causal\b', command)) != (mask == 'causal'):
                problems.append(f'task {task_id!r} mask={mask} but probe command disagrees about --causal')
            elif f'-{mask}-' not in (probe.get('channel') or ''):
                problems.append(f'task {task_id!r} probe channel does not name mask={mask}')

    series = (plan.get('stop_rules') or {}).get('per_series') or {}
    if series.get('no_valid_pairs_in_a_row') != 2:
        problems.append('the series stop rule must be "two consecutive pairs without a valid L3"')
    if (plan.get('final_evaluation') or {}).get('stop_after') != 'L3':
        problems.append('P1 final evaluation must stop_after L3 (L2-only leaves performance unknown)')

    budget = plan.get('budget') or {}

    def _budget_ok(name, value):
        if value is None or _positive(value):
            return True
        # P1 不生成候选：模型调用数允许为 0（其余字段仍须正）。
        return name == 'max_model_calls' and value == 0

    invalid = [name for name in BUDGET_FIELDS if not _budget_ok(name, budget.get(name))]
    if invalid:
        problems.append(f'invalid budget fields: {invalid}')
    unfilled = [name for name in BUDGET_FIELDS if budget.get(name) is None]
    if plan.get('status') == 'frozen':
        if unfilled:
            problems.append(f'a frozen plan needs every budget filled, missing: {unfilled}')
        if budget.get('max_model_calls') not in (0, None):
            problems.append('P1 re-measures existing deliveries: max_model_calls must be 0 (no generation)')
        if budget.get('pairs') != len(pairs):
            problems.append(f'budget.pairs ({budget.get("pairs")}) must equal the registered pair count ({len(pairs)})')
    return problems


def freeze(budget, output):
    """Write the plan once, with budgets filled. Refuses to overwrite or to guess."""
    plan = design()
    plan['budget'] = {name: (budget or {}).get(name) for name in BUDGET_FIELDS}
    plan['status'] = 'frozen'
    problems = validate(plan)
    if problems:
        raise ValueError('; '.join(problems))
    plan['hash'] = digest({key: value for key, value in plan.items() if key != 'hash'})
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(plan, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return plan


# ── 预登记判读量具（零机时可单测；真机读数在窗口取，不在此发明）─────────────────

def judge_pair(*, a_median_ms, c_median_ms, a_valid, c_valid, band_pct):
    """One pair: decidable only if both arms valid; parity judged against the noise band."""
    if not (a_valid and c_valid) or not _positive(a_median_ms) or not _positive(c_median_ms):
        return dict(decidable=False, verdict='not_decidable',
                    reason='both arms must reach a valid L3 with positive median times')
    difference_pct = (c_median_ms - a_median_ms) / a_median_ms * 100.0
    if difference_pct > band_pct:
        verdict = 'C_worse'
    elif difference_pct < -band_pct:
        verdict = 'C_faster'
    else:
        verdict = 'parity_within_band'
    return dict(decidable=True, verdict=verdict, difference_pct=round(difference_pct, 4),
                band_pct=band_pct)


def judge_series(pair_verdicts):
    """Series parity claim: >=1 decidable pair AND no decidable pair is C_worse. Scoped."""
    decidable = [v for v in pair_verdicts if v.get('decidable')]
    worse = [v for v in decidable if v.get('verdict') == 'C_worse']
    if not decidable:
        return dict(verdict='unknown', formal_parity_claim=False, decidable_pairs=0,
                    reason='no pair reached valid L3 in both arms; this is about the window, not the candidates')
    if worse:
        return dict(verdict='C_worse_observed', formal_parity_claim=False,
                    decidable_pairs=len(decidable), worse_pairs=len(worse),
                    reason='at least one decidable pair shows the plugin arm slower beyond the band')
    return dict(verdict='parity_holds', formal_parity_claim=True,
                decidable_pairs=len(decidable), worse_pairs=0, scope=SUBJECT,
                reason='no decidable pair shows the plugin arm slower beyond the band',
                boundaries='no gain claim; no vs-library claim; single device; single model family; n<=3 pairs')