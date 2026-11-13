"""Turn a round of A/C cells into one delivery-count verdict, using the frozen plan's rules.

The difference that matters
---------------------------
`r2_compare.arm_summary` raises when an arm's final L2 did not pass, because in the
increment round both arms are expected to deliver and a failure means the input is broken.
Here a failure IS the measurement: the question is how many tasks each arm delivers inside
the same budget. So this module keeps three outcomes apart and never converts one into
another:

* ``delivered`` — an independent final evaluation passed full L2 on the submitted source;
* ``failed``    — the cell ran to a terminal state (completed, budget exhausted, deadline)
                  and did not deliver. This is a result about the arm;
* ``unknown``   — the record chain broke (crash, missing receipt, drifted source hash).
                  This is a result about the wiring, and it blocks a round-level claim.

Planned-but-unexecuted cells are counted separately and also block a round-level claim:
a pass rate whose denominator quietly shrank is the failure mode this round exists to avoid.
The source hash is re-computed from the file here; the receipt's claim about itself is not
trusted.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

from operator_agent.core.run_contract import digest

from . import ac_plan

SCHEMA = 'v08-workflow-ac-round-v1'

#: 驱动观测到的进程终态。分类放在驱动那一侧，比较器不猜"它为什么没交"。
TERMINAL_STATES = ('completed', 'budget_exhausted', 'deadline')
CELL_STATES = TERMINAL_STATES + ('crashed', 'not_executed')


class RoundInputError(ValueError):
    """输入本身坏了（漂移、重复格、不认识的任务/臂）——没有可判的东西，只能停止。"""


def _load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _file_hash(path):
    target = Path(path)
    if not target.is_file():
        raise RoundInputError(f'candidate source 不存在: {target}')
    return hashlib.sha256(target.read_bytes()).hexdigest()


def _l3(receipt):
    for verdict in receipt.get('verdicts') or ():
        if isinstance(verdict, dict) and verdict.get('level') == 'L3':
            return verdict
    return None


def cell_summary(cell, *, plan_tasks):
    """One cell's facts. A failed L2 is returned as ``delivered=False``, not raised."""
    if not isinstance(cell, dict):
        raise RoundInputError('cell 不是对象')
    task_id, arm = cell.get('task_id'), cell.get('arm')
    if task_id not in plan_tasks:
        raise RoundInputError(f'任务 {task_id!r} 不在这份冻结计划里')
    if arm not in ac_plan.ARMS:
        raise RoundInputError(f'臂 {arm!r} 不是 {list(ac_plan.ARMS)} 之一')
    state = cell.get('state')
    if state not in CELL_STATES:
        raise RoundInputError(f'{arm}/{task_id}: 未知的格状态 {state!r}，必须是 {list(CELL_STATES)}')

    summary = dict(task_id=task_id, arm=arm, state=state, delivered=False,
                   outcome='unknown', contract_hash=None, source_hash=None,
                   validity=None, speedup=None, median_time_ms=None,
                   wall_clock_to_first_full_L2_s=cell.get('wall_clock_to_first_full_L2_s'),
                   model_dispatches=cell.get('model_dispatches'),
                   verifications=cell.get('verifications'),
                   stop_reason=cell.get('stop_reason'), reasons=[])
    if state == 'not_executed':
        summary['outcome'] = 'not_executed'
        summary['reasons'].append('计划内但未派发：单列，不进已派发分母，并且让整轮记为不完整')
        return summary
    if state == 'crashed':
        summary['reasons'].append('记录链断了（进程崩溃）：这是接线的结果，不是候选的结果')
        return summary

    receipt = cell.get('final_receipt')
    if receipt is None:
        summary['outcome'] = 'failed'
        summary['reasons'].append(f'格走到终态 {state} 但没有独立终评回执：未交付')
        return summary
    if not isinstance(receipt, dict):
        raise RoundInputError(f'{arm}/{task_id}: 终评回执不是对象')
    if receipt.get('verification_purpose') != 'final':
        raise RoundInputError(
            f'{arm}/{task_id}: 要用独立终评的读数（verification_purpose=final），'
            f'收到 {receipt.get("verification_purpose")!r}')
    source = cell.get('source') or receipt.get('source')
    if not source:
        raise RoundInputError(f'{arm}/{task_id}: 回执没有绑定源码路径，无法重算哈希')
    actual = _file_hash(source)
    if receipt.get('source_hash') != actual:
        raise RoundInputError(
            f'{arm}/{task_id}: 回执绑定的源码哈希与实际文件不一致'
            f'（receipt={receipt.get("source_hash")}, file={actual}）—— 判的不是提交的那份代码')

    levels = receipt.get('levels') or {}
    l3 = _l3(receipt)
    metrics = (l3 or {}).get('metrics') or {}
    summary.update(contract_hash=receipt.get('contract_hash'), source_hash=actual,
                   validity=receipt.get('validity') or metrics.get('measurement_validity'),
                   speedup=metrics.get('speedup_vs_baseline'),
                   median_time_ms=metrics.get('median_time_ms'),
                   levels={key: bool(levels.get(key)) for key in ('L1', 'L2', 'L3')})
    if levels.get('L2') is True:
        summary['delivered'] = True
        summary['outcome'] = 'delivered'
    else:
        summary['outcome'] = 'failed'
        summary['reasons'].append('独立终评的完整 L2 没有通过：未交付（这是关于该臂的结果）')
    return summary


def _performance(tasks, cells, plan_tasks):
    """Secondary only: both delivered, both valid, same contract. Otherwise reasons, no number."""
    observations = []
    for task_id in plan_tasks:
        pair = {cell['arm']: cell for cell in cells if cell['task_id'] == task_id}
        left, right = pair.get('A'), pair.get('C')
        reasons = []
        for side, cell in (('A', left), ('C', right)):
            if cell is None:
                reasons.append(f'{side}: 这一格没有记录')
            elif not cell['delivered']:
                reasons.append(f'{side}: 未交付完整 L2，性能不进比较')
            elif cell['validity'] != 'valid':
                reasons.append(f'{side}: 终评 L3 有效性是 {cell["validity"]!r}，不是 valid')
            elif cell['speedup'] is None:
                reasons.append(f'{side}: 终评没有给出 speedup_vs_baseline')
        if left is not None and right is not None and left['contract_hash'] != right['contract_hash']:
            reasons.append('两臂契约哈希不同')
        comparable = not reasons
        observation = dict(task_id=task_id, comparable=comparable, reasons=reasons,
                           speedup_A=left['speedup'] if left else None,
                           speedup_C=right['speedup'] if right else None,
                           speedup_difference_C_minus_A=None,
                           relative_speedup_difference_pct=None)
        if comparable:
            a_speedup, c_speedup = left['speedup'], right['speedup']
            observation['speedup_difference_C_minus_A'] = c_speedup - a_speedup
            observation['relative_speedup_difference_pct'] = (
                (c_speedup / a_speedup - 1.0) * 100.0 if a_speedup else None)
        observations.append(observation)
    return observations


def compare_round(plan, cells, *, metering=None, now=None):
    """Build one round verdict from the frozen plan and every cell it registered."""
    problems = ac_plan.validate(plan)
    if problems:
        raise RoundInputError('plan 不可执行: ' + '; '.join(problems))
    if plan.get('status') != 'frozen':
        raise RoundInputError('只有冻结的计划才能产出整轮结论')
    plan_tasks = tuple(row['task_id'] for row in plan['tasks'])
    if not isinstance(cells, (list, tuple)):
        raise RoundInputError('cells 必须是列表')

    seen, summaries = set(), []
    for cell in cells:
        summary = cell_summary(cell, plan_tasks=plan_tasks)
        key = (summary['arm'], summary['task_id'])
        if key in seen:
            raise RoundInputError(f'重复的格 {key}：两段运行不能进同一份账')
        seen.add(key)
        summaries.append(summary)

    counts = {}
    for arm in ac_plan.ARMS:
        rows = [cell for cell in summaries if cell['arm'] == arm]
        counts[arm] = dict(dispatched=sum(1 for cell in rows if cell['state'] != 'not_executed'),
                           **{outcome: sum(1 for cell in rows if cell['outcome'] == outcome)
                              for outcome in ('delivered', 'failed', 'unknown', 'not_executed')})
    difference = counts['C']['delivered'] - counts['A']['delivered']
    incomplete = [f'{cell["arm"]}/{cell["task_id"]}' for cell in summaries
                  if cell['outcome'] in ('unknown', 'not_executed')]
    missing = [f'{arm}/{task_id}' for task_id in plan_tasks for arm in ac_plan.ARMS
               if (arm, task_id) not in seen]
    # 缺格与坏格一样让整轮不完整：分母不许悄悄缩小。
    incomplete.extend(missing)

    body = dict(
        schema=SCHEMA,
        created_at=(now or datetime.datetime.now(datetime.timezone.utc)).isoformat(timespec='seconds'),
        plan_id=plan['plan_id'], plan_hash=plan.get('hash'),
        plan_hash_verified=(plan.get('hash') is not None
                            and digest({k: v for k, v in plan.items() if k != 'hash'}) == plan['hash']),
        arms=list(ac_plan.ARMS), tasks=list(plan_tasks), cells=summaries,
        delivery=dict(
            per_arm=counts,
            denominator='dispatched cells per arm; not_executed and missing cells are listed '
                        'and make the round incomplete',
            paired_difference_C_minus_A=difference,
            minimum_detectable_unit=plan['metric']['minimum_detectable_unit'],
            clears_minimum_detectable_unit=bool(difference >= 1),
            round_complete=not incomplete,
            incomplete_cells=sorted(incomplete),
            interpretation=('a delivered-count difference is an observation about this round '
                            'and this budget, not a significance test')),
        performance=_performance(plan['tasks'], summaries, plan_tasks),
        sampling_comparability=(metering or {}).get('sampling_comparability', 'not_established'),
        formal_gain_claim=False, formal_comparison_credit=False)
    body['hash'] = digest(body)
    return body


def main(argv=None):
    parser = argparse.ArgumentParser(prog='ac_compare', description=__doc__.splitlines()[0])
    parser.add_argument('--plan', required=True, help='冻结的 A/C 计划 JSON')
    parser.add_argument('--cells', required=True, help='本格账 JSON（列表，含计划内未执行的格）')
    parser.add_argument('--metering', help='usage-reconcile 回执（可选）')
    parser.add_argument('--json-out', required=True)
    args = parser.parse_args(argv)
    verdict = compare_round(_load(args.plan), _load(args.cells),
                            metering=_load(args.metering) if args.metering else None)
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(verdict['delivery'], ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())