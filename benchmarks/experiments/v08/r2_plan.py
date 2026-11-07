"""R2 preregistration: freeze the v0.7.2 vs v0.8 increment comparison before running it.

The point of writing this down first is that "what counts as a result" cannot be chosen
after the numbers arrive. Everything here is either a frozen declaration or a number that
was measured in a named window; nothing is a claim about an outcome.

Two budgets are deliberately left empty. `freeze()` refuses to write a plan while they are
`None` — a preregistration whose budget is decided later is not a preregistration.
"""
from __future__ import annotations

import json
from pathlib import Path

from operator_agent.core.run_contract import digest

SCHEMA = 'v08-increment-preregistration-v1'
PLAN_ID = 'operator-v08-increment-r2'

#: 窗口恶化多少倍时，预登记的效应仍要能判出来。带宽是 3× 实测散布，窗口一变吵它就变宽；
#: 留到 10 倍，是为了让"今天 0.2% 可判"不会因为明天 load 从 5 涨到 50 就作废。
WINDOW_DEGRADATION_MARGIN = 10.0

TASKS = [dict(
    task_id='matmul-4096',
    declaration=dict(request='fp16 4096x4096x4096 matmul, triton', shapes=[[4096, 4096, 4096]],
                     precision='fp16', dsl='triton', hardware='nvidia_gpu', min_speedup=1.0),
    why='the only shape measured in a decidable window: 0.203% band on 2026-11-07',
)]

MEASUREMENT_GATE = dict(
    command=('python -m operator_agent preflight --backend ssh --host <HOST> '
             '--shape 4096,4096,4096 --reps 5 --repeat 8 --planned-effect-pct 2.5'),
    channel='triton-matmul-4096x4096x4096-cuda-event',
    start_only_if='action == "start"',
    observed_band_pct=0.203,
    observed_window='2026-11-07 13:36-14:10 CST, h20, 192 cores, load 5.35, 8 GPUs at 0%',
    on_hold=('do not start the pair and do not count it as a failure: a held window '
             'consumes no model or verification budget'),
    rerun_every_window=True,
)

DETECTABILITY = dict(
    measured_band_pct=0.203,
    min_detectable_effect_pct=2.5,
    margin_x=round(2.5 / 0.203, 2),
    rule=('the registered effect must clear the window band by WINDOW_DEGRADATION_MARGIN; '
          'a pairing below that is not launched'),
)

METRIC = dict(
    primary='paired difference in final speedup, only for pairs whose final L3 is valid',
    valid_requires=['full L2 passed', 'measurement_validity == "valid"',
                    'independent final evaluation on the same source hash'],
    secondary=['dispatches used', 'wall clock to first full L2', 'verifications used',
               'agreement between the optimize diagnosis and the measured binding axis'],
    never_report=['a degraded reading as a gain', 'an unpaired average as a comparison',
                  'cost inferred from token counters'],
)

STOP_RULES = dict(
    per_pair=['preflight action != "start" -> hold, do not launch',
              'provider output-cap violation -> stop the pair'],
    per_series=dict(no_valid_pairs_in_a_row=2,
                    on_stop='record the series as unknown; do not launch a third pair',
                    why=('two degraded pairs in a row is a systematic window or wiring '
                         'problem, and a third pair only spends money')),
    budget='stop at the frozen budget; never raise it mid-series to reach a threshold',
)

FINAL_EVALUATION = dict(independent=True, same_source_hash=True, reserve_per_pair=1,
                        note='one reserved independent evaluation per pair, never spent on exploration')

BOUNDARIES = [
    'both arms run the same plugin; only the optimize phase differs, so this compares the increment, not the workflow',
    'sampling_comparability is not established (see operator_agent usage-reconcile): report paired observations, not absolute gains',
    'the A arm has no equivalent model-call hard gate, so no A/B workflow claim is made from this series',
    'a task that never reaches a valid final L3 leaves the series unknown; that is a result about the window, not about the candidate',
]

BUDGET_FIELDS = ('pairs', 'max_model_calls', 'max_output_tokens_per_call',
                 'total_output_token_reservations', 'max_verifications', 'wall_clock_s',
                 'gpu_minutes')


def design():
    """The freezable part: no budget, no outcome, no credit."""
    body = dict(schema=SCHEMA, plan_id=PLAN_ID, status='draft_awaiting_budget',
                tasks=[json.loads(json.dumps(row)) for row in TASKS],
                measurement_gate=dict(MEASUREMENT_GATE), detectability=dict(DETECTABILITY),
                metric=dict(METRIC), stop_rules=json.loads(json.dumps(STOP_RULES)),
                final_evaluation=dict(FINAL_EVALUATION), boundaries=list(BOUNDARIES),
                budget={name: None for name in BUDGET_FIELDS},
                window_degradation_margin=WINDOW_DEGRADATION_MARGIN,
                # 这三项是"不许写进去的东西": 收益主张、正式对照信用、已完成的结果。
                formal_gain_claim=False, formal_comparison_credit=False, outcome=None)
    body['design_hash'] = digest(body)
    return body


def _positive(value):
    return type(value) in (int, float) and value == value and value > 0


def validate(plan):
    """Return a list of problems; empty means the plan may be frozen/executed."""
    problems = []
    if not isinstance(plan, dict) or plan.get('schema') != SCHEMA:
        return ['not an R2 preregistration']
    if plan.get('plan_id') != PLAN_ID:
        problems.append(f'plan_id must be {PLAN_ID}')
    body = {key: value for key, value in plan.items() if key != 'hash'}
    if plan.get('hash') is not None and digest(body) != plan.get('hash'):
        problems.append('plan hash does not match its content')
    if plan.get('formal_gain_claim') is not False or plan.get('formal_comparison_credit') is not False:
        problems.append('a preregistration cannot claim a gain or comparison credit')
    if plan.get('outcome') is not None:
        problems.append('a preregistration cannot carry an outcome')
    gate, detect = plan.get('measurement_gate') or {}, plan.get('detectability') or {}
    band = gate.get('observed_band_pct')
    effect = detect.get('min_detectable_effect_pct')
    if not _positive(band) or not _positive(effect):
        problems.append('the measured band and the registered effect must both be positive')
    else:
        required = float(band) * float(plan.get('window_degradation_margin') or 0)
        if float(effect) < required:
            problems.append(f'registered effect {effect}% does not clear {required:.3f}% '
                            f'({plan.get("window_degradation_margin")}x the {band}% band)')
    series = (plan.get('stop_rules') or {}).get('per_series') or {}
    if series.get('no_valid_pairs_in_a_row') != 2:
        problems.append('the series stop rule must be "two consecutive pairs without a valid L3"')
    budget = plan.get('budget') or {}
    missing = [name for name in BUDGET_FIELDS
               if not (_positive(budget.get(name)) or budget.get(name) is None) ]
    if missing:
        problems.append(f'invalid budget fields: {missing}')
    unfilled = [name for name in BUDGET_FIELDS if budget.get(name) is None]
    if plan.get('status') == 'frozen' and unfilled:
        problems.append(f'a frozen plan needs every budget filled, missing: {unfilled}')
    if not plan.get('tasks'):
        problems.append('at least one task must be registered')
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