"""v7 融合 A/C 的整轮聚合：主指标是**达标交付**，不是仅 L2 交付。

规则（评审 2026-11-17 后定）：

* 单对：C 达标、A 不达标 ⇒ `+1`；A 达标、C 不达标 ⇒ `−1`；同结果 ⇒ `0`，
  且必须保留子型（both_qualified / both_correct_only / both_failed / mixed）。
* 整轮（2 对）：两对 `+1` ⇒ `reproduced`；两对 `0` ⇒ `delivery_parity`；
  一对 `+1`、一对 `0` ⇒ `not_established`；任一对 `−1` ⇒ `contrary`；
  有 incomplete/unknown ⇒ `unknown`。
* 仅正确交付、调用数、时间只进 `secondary`，不授正式信用。
"""
from __future__ import annotations

import hashlib
import json

from . import ac_plan

SCHEMA = 'v08-fused-ac-series-v1'
PAIR_SCHEMA = 'v08-fused-ac-pair-v1'
PAIRS = 2


def _digest(value):
    blob = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


def _unknown_reason(cell):
    state = cell.get('state')
    if state in ('not_executed', 'crashed'):
        return f'{cell.get("arm")}: state={state}'
    if cell.get('outcome') == 'unknown':
        return f'{cell.get("arm")}: outcome=unknown'
    receipt = cell.get('final_receipt')
    if not isinstance(receipt, dict):
        return f'{cell.get("arm")}: no final receipt'
    if receipt.get('verification_purpose') != 'final':
        return f'{cell.get("arm")}: receipt is not the independent final'
    return None


def _reading(cell, task_id):
    arm = cell.get('arm')
    if cell.get('task_id') != task_id:
        raise ValueError(f'{arm}: cell task {cell.get("task_id")!r} is not {task_id!r}')
    if arm not in ac_plan.ARMS:
        raise ValueError(f'unknown arm {arm!r}')
    unknown = _unknown_reason(cell)
    delivery_class = cell.get('delivery_class')
    if unknown is None and delivery_class not in ('qualified', 'correct_only', 'failed'):
        unknown = f'{arm}: missing delivery_class'
    if unknown is None:
        outcome = delivery_class            # qualified / correct_only / failed
    else:
        outcome = 'unknown'
    return dict(arm=arm, state=cell.get('state'), outcome=outcome,
                qualified=bool(cell.get('qualified')) and outcome == 'qualified',
                delivery_class=delivery_class,
                final_speedup=cell.get('final_speedup'),
                final_validity=cell.get('final_validity'),
                fusion_static=cell.get('fusion_static'),
                model_dispatches=cell.get('model_dispatches'),
                verifications=cell.get('verifications'),
                wall_clock_total_s=cell.get('wall_clock_total_s'),
                reason=unknown)


def pair_verdict(plan, cells):
    """One pair: C minus A on *qualified* delivery, with the 0-subtype kept."""
    problems = list(ac_plan.validate(plan))
    if problems:
        raise ValueError('plan 不可执行: ' + '; '.join(problems))
    if plan.get('plan_id') not in {ac_plan.PLAN_IDS[name] for name in ac_plan.FUSED_VERSIONS}:
        raise ValueError('pair_verdict 只接受融合计划（v7/v8）')
    if plan.get('status') != 'frozen':
        raise ValueError('只有冻结的计划才能产出配对判读')
    task_id = plan['tasks'][0]['task_id']
    if not isinstance(cells, (list, tuple)) or len(cells) != 2:
        raise ValueError('一对必须正好是 A/C 两格')
    readings = {cell.get('arm'): _reading(cell, task_id) for cell in cells}
    if set(readings) != set(ac_plan.ARMS):
        raise ValueError('一对必须是 A 和 C 各一格')
    a, c = readings['A'], readings['C']
    unknown = [row for row in (a, c) if row['outcome'] == 'unknown']
    complete = not unknown
    difference = (int(c['qualified']) - int(a['qualified'])) if complete else None
    classes = {a['delivery_class'], c['delivery_class']}
    if not complete:
        subtype = 'unknown'
    elif classes == {'qualified'}:
        subtype = 'both_qualified'
    elif classes == {'correct_only'}:
        subtype = 'both_correct_only'
    elif classes == {'failed'}:
        subtype = 'both_failed'
    else:
        subtype = 'mixed_' + '_'.join(
            f'{arm}_{row["delivery_class"]}' for arm, row in (('a', a), ('c', c)))
    body = dict(schema=PAIR_SCHEMA, plan_id=plan.get('plan_id'),
                plan_hash=plan.get('hash'), task_id=task_id,
                round_complete=complete, subtype=subtype,
                paired_difference_C_minus_A=difference,
                per_arm=readings,
                secondary=dict(
                    a_qualified=a['qualified'], c_qualified=c['qualified'],
                    a_correct_only=a['delivery_class'] == 'correct_only',
                    c_correct_only=c['delivery_class'] == 'correct_only',
                    a_model_dispatches=a['model_dispatches'],
                    c_model_dispatches=c['model_dispatches'],
                    a_wall_clock_total_s=a['wall_clock_total_s'],
                    c_wall_clock_total_s=c['wall_clock_total_s']),
                reason=('complete' if complete else '; '.join(row['reason'] for row in unknown)))
    body['hash'] = _digest(body)
    return body


def aggregate_fused(plan, pairs):
    """Two pairs -> round verdict (reproduced / delivery_parity / not_established / contrary / unknown)."""
    problems = list(ac_plan.validate(plan))
    if plan.get('plan_id') not in {ac_plan.PLAN_IDS[name] for name in ac_plan.FUSED_VERSIONS}:
        problems.append('不是融合计划（v7/v8）')
    if plan.get('status') != 'frozen':
        problems.append('只有冻结的计划才能产出整轮结论')
    if plan.get('pairs') != PAIRS:
        problems.append(f'计划登记的正式对数是 {plan.get("pairs")!r}，需要 {PAIRS}')
    if not isinstance(pairs, (list, tuple)) or len(pairs) != PAIRS:
        problems.append(f'必须正好 {PAIRS} 对，收到 {len(pairs or ())}')
    rows = []
    for index, pair in enumerate(pairs or (), start=1):
        if pair.get('plan_hash') != plan.get('hash'):
            problems.append(f'第 {index} 对的 plan_hash 与冻结计划不一致')
        rows.append(dict(pair=index, hash=pair.get('hash'),
                         round_complete=bool(pair.get('round_complete')),
                         paired_difference=pair.get('paired_difference_C_minus_A'),
                         subtype=pair.get('subtype')))
    if problems:
        verdict, reason = 'unknown', '; '.join(problems)
    elif not all(row['round_complete'] for row in rows):
        verdict, reason = 'unknown', '有配对不完整/unknown；按规则不追加轮次'
    else:
        diffs = [row['paired_difference'] for row in rows]
        if -1 in diffs:
            verdict, reason = 'contrary', '至少一对是 A 达标、C 不达标（−1）'
        elif diffs == [1, 1]:
            verdict, reason = 'reproduced', '两对均为 C 达标、A 不达标（+1）'
        elif diffs == [0, 0]:
            verdict, reason = 'delivery_parity', '两对达标数相同（0）；子型见 per_pair'
        elif set(diffs) <= {0, 1}:
            verdict, reason = 'not_established', '一对 +1、一对 0：未达每对同向 +1'
        else:
            verdict, reason = 'unknown', f'配对差无法判读: {diffs}'
    body = dict(schema=SCHEMA, plan_id=plan.get('plan_id'), plan_hash=plan.get('hash'),
                pairs_expected=PAIRS, pairs_observed=len(rows), per_pair=rows,
                verdict=verdict, reason=reason,
                formal_gain_claim=verdict == 'reproduced',
                formal_comparison_credit=verdict == 'reproduced',
                scope=('weak host 27B/xhigh, fused residual+LayerNorm 32768x6144, '
                       'GPU7, 2 pairs; qualified delivery only'),
                boundaries=('no performance claim beyond the qualified bar; '
                            'no developer-time claim; single device; single model'))
    body['hash'] = _digest(body)
    return body


def main(argv=None):
    import argparse
    from pathlib import Path
    from benchmarks.experiments.phase1.plan import write_once

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p_pair = sub.add_parser('pair')
    p_pair.add_argument('--plan', type=Path, required=True)
    p_pair.add_argument('--cells', type=Path, nargs=2, required=True)
    p_pair.add_argument('--output', type=Path, required=True)
    p_series = sub.add_parser('series')
    p_series.add_argument('--plan', type=Path, required=True)
    p_series.add_argument('--pairs', type=Path, nargs=2, required=True)
    p_series.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)

    def read(path):
        return json.loads(Path(path).read_text(encoding='utf-8'))

    if args.command == 'pair':
        result = pair_verdict(read(args.plan), [read(path) for path in args.cells])
        write_once(args.output, result)
        print(result['paired_difference_C_minus_A'], result['subtype'])
    else:
        result = aggregate_fused(read(args.plan), [read(path) for path in args.pairs])
        write_once(args.output, result)
        print(result['verdict'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
