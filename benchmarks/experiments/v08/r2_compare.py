"""Turn two arms' final receipts into one pair verdict, using the frozen plan's rules.

Why this is code and not a paragraph
------------------------------------
The runbook said so ("`pair.json` 的比较器还没写成代码；在此之前按第 3 节的规则人工判").
Manual judgement on a rule table is how a threshold quietly moves after the numbers
arrive. Everything here is decided from inputs that travel with the pair:

* the plan must be **frozen** and hash-consistent (``r2_plan.validate``);
* both arms must present an **independent final evaluation** — ``verification_purpose ==
  "final"``, L1+L2 passed, L3 present — and the evaluator's ``source_hash`` must equal a
  fresh hash of the submitted source file (the comparator re-hashes; it does not trust
  the receipt's claim about itself);
* a pair is **comparable** only when both arms' final L3 are ``valid`` *and* both were
  evaluated against the same contract hash;
* the primary number is the **paired difference** of final speedups. A ``degraded``
  reading is never a difference; an ``uninformative`` one makes the pair non-decidable;
* two consecutive non-decidable pairs end the series (the plan's own stop rule).

Broken input and "not comparable" are kept apart: a drifted source hash or a plan that
is not frozen raises, because there is nothing to judge; a valid-but-not-decidable pair
is a **result about the window**, recorded with reasons.
"""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

from operator_agent.core.run_contract import digest

from . import r2_plan

SCHEMA = 'v08-r2-native-pair-v1'


class PairInputError(ValueError):
    """输入本身坏了（漂移、半份回执、计划未冻结）——没有可判的东西，只能停止。"""


def _load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _file_hash(path):
    target = Path(path)
    if not target.is_file():
        raise PairInputError(f'candidate source 不存在: {target}')
    return hashlib.sha256(target.read_bytes()).hexdigest()


def _l3(receipt):
    for verdict in receipt.get('verdicts') or ():
        if isinstance(verdict, dict) and verdict.get('level') == 'L3':
            return verdict
    return None


def arm_summary(receipt, *, side, source=None):
    """One arm's final-evaluation facts, with the source hash re-computed here."""
    if not isinstance(receipt, dict):
        raise PairInputError(f'{side}: 回执不是对象')
    if receipt.get('verification_purpose') != 'final':
        raise PairInputError(
            f'{side}: 这一对要用**独立终评**的读数（verification_purpose=final），'
            f'收到 {receipt.get("verification_purpose")!r}')
    levels = receipt.get('levels') or {}
    if levels.get('L2') is not True:
        raise PairInputError(f'{side}: 终评的 L2 没有通过，不能进配对比较')
    l3 = _l3(receipt)
    if l3 is None:
        raise PairInputError(f'{side}: 回执里没有 L3 层')
    claimed = receipt.get('source_hash')
    actual = _file_hash(source or receipt.get('source'))
    if claimed != actual:
        raise PairInputError(
            f'{side}: 回执绑定的源码哈希与实际文件不一致（receipt={claimed}, file={actual}）'
            '—— 判的不是提交的那份代码，停止')
    metrics = l3.get('metrics') or {}
    return dict(contract_hash=receipt.get('contract_hash'),
                source_hash=actual,
                verification_purpose='final',
                levels={key: bool(levels.get(key)) for key in ('L1', 'L2', 'L3')},
                l3_passed=bool(l3.get('passed')),
                validity=receipt.get('validity') or metrics.get('measurement_validity'),
                speedup=metrics.get('speedup_vs_baseline'),
                median_time_ms=metrics.get('median_time_ms'),
                baseline_time_ms=metrics.get('baseline_time_ms'),
                measurement_can_gate=metrics.get('measurement_can_gate'),
                verification_dispatches=(receipt.get('host_task') or {}).get('used')
                if isinstance(receipt.get('host_task'), dict) else None)


def _gate_block(gate):
    if not isinstance(gate, dict):
        return None
    return dict(receipt=None, passed=gate.get('action') == 'start',
                action=gate.get('action'), channel=gate.get('channel'),
                effective_band_pct=gate.get('effective_band_pct'),
                identity=gate.get('identity'), reasons=list(gate.get('reasons') or ()))


def compare(plan, receipts, *, order=('B', 'C'), gate=None, metering=None, now=None):
    """Build one ``pair.json``. ``receipts`` maps arm name → final verify receipt."""
    problems = r2_plan.validate(plan)
    if problems:
        raise PairInputError('plan 不可执行: ' + '; '.join(problems))
    if plan.get('status') != 'frozen':
        raise PairInputError('只有冻结的计划才能产出配对结论')
    side_a, side_b = order
    if set(receipts) != {side_a, side_b}:
        raise PairInputError(f'需要恰好两臂 {list(order)}，收到 {sorted(receipts)}')
    arms = {side: arm_summary(receipts[side], side=side) for side in order}

    reasons = []
    if arms[side_a]['contract_hash'] != arms[side_b]['contract_hash']:
        reasons.append(f'两臂的契约哈希不同: {arms[side_a]["contract_hash"]} vs '
                       f'{arms[side_b]["contract_hash"]}')
    for side in order:
        arm = arms[side]
        if arm['validity'] != 'valid':
            reasons.append(f'{side}: 终评 L3 的有效性是 {arm["validity"]!r}，不是 "valid" '
                           '—— 不进入差值')
        if arm['speedup'] is None:
            reasons.append(f'{side}: 终评没有给出 speedup_vs_baseline')
    comparable = not reasons
    effect = (plan.get('detectability') or {}).get('min_detectable_effect_pct')
    comparison = dict(
        comparable=comparable, reasons=reasons,
        registered_effect_pct=effect,
        sampling_comparability=(metering or {}).get('sampling_comparability', 'not_established'),
        formal_gain_claim=False, formal_comparison_credit=False)
    if comparable:
        left, right = arms[side_a]['speedup'], arms[side_b]['speedup']
        difference = right - left
        relative = (right / left - 1.0) * 100.0 if left else None
        comparison[f'speedup_difference_{side_b}_minus_{side_a}'] = difference
        comparison['relative_speedup_difference_pct'] = relative
        comparison['clears_registered_positive_effect'] = bool(
            relative is not None and effect is not None and relative >= effect)
    else:
        # 键的形状与可比时一致（值为 None）：让读回执的人不必按"键在不在"分支。
        comparison[f'speedup_difference_{side_b}_minus_{side_a}'] = None
        comparison['relative_speedup_difference_pct'] = None
        comparison['clears_registered_positive_effect'] = False

    body = dict(
        schema=SCHEMA,
        created_at=(now or datetime.datetime.now(datetime.timezone.utc)).isoformat(timespec='seconds'),
        plan_id=plan['plan_id'], plan_hash=plan.get('hash'),
        plan_hash_verified=(plan.get('hash') is not None
                            and digest({k: v for k, v in plan.items() if k != 'hash'}) == plan['hash']),
        order=list(order), tasks=plan.get('tasks'), gate=_gate_block(gate),
        arms=arms, comparison=comparison)
    body['hash'] = digest(body)
    return body


def series_state(plan, prior_pairs, pair=None):
    """连续多少对不可判，以及是否该收工（计划自己的停止规则）。"""
    limit = ((plan.get('stop_rules') or {}).get('per_series') or {}).get('no_valid_pairs_in_a_row')
    pairs = [dict(item.get('comparison') or {}) for item in (prior_pairs or ())]
    if pair is not None:
        pairs.append(dict(pair.get('comparison') or {}))
    streak = 0
    for item in reversed(pairs):
        if item.get('comparable'):
            break
        streak += 1
    return dict(pairs_attempted=len(pairs),
                pairs_comparable=sum(1 for item in pairs if item.get('comparable')),
                consecutive_nondecidable=streak,
                stop_rule=limit, stop=(limit is not None and streak >= limit))


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(prog='r2_compare', description=__doc__.splitlines()[0])
    parser.add_argument('--plan', required=True, help='冻结的 R2 计划 JSON')
    parser.add_argument('--arm', action='append', default=[],
                        help='NAME=终评回执.json（可重复，恰好两个）')
    parser.add_argument('--order', default='B,C', help='两臂顺序（默认 B,C）')
    parser.add_argument('--gate', help='这一对开跑前的 preflight 回执（可选）')
    parser.add_argument('--metering', help='usage-reconcile 回执（可选）')
    parser.add_argument('--series', help='本系列已有的 pair.json 目录（可选）')
    parser.add_argument('--json-out', required=True)
    args = parser.parse_args(argv)

    receipts = {}
    for item in args.arm:
        if '=' not in item:
            raise SystemExit(f'--arm 需要 NAME=path，收到 {item!r}')
        name, path = item.split('=', 1)
        receipts[name] = _load(path)
    plan = _load(args.plan)
    pair = compare(plan, receipts, order=tuple(args.order.split(',')),
                   gate=_load(args.gate) if args.gate else None,
                   metering=_load(args.metering) if args.metering else None)
    prior = []
    if args.series and Path(args.series).is_directory():
        prior = [_load(path) for path in sorted(Path(args.series).glob('pair-*.json'))]
    pair['series'] = series_state(plan, prior, pair)
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(
        json.dumps(pair, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(pair['comparison'], ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())