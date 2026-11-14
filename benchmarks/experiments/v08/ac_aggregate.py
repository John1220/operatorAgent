"""复现轮聚合器：把 N 轮的单轮判读合并成一条**正式主张**（或 null）。

为什么单独一个模块：`ac_compare` 的语义是"**这一轮**的观测"，
它的 `formal_gain_claim` 硬编码 False 是**有意**的——单轮不是结论。
这里不碰它，只在它之上做合并，于是两种语义各归各位（用户 2026-11-10 选定 ③A）。

规则（用户 2026-11-10 选定 ①A）：**每一轮**的配对差都要 ≥ 最小可分辨单位；
任一轮掉下去就判 `null`，并且**不再追加轮次**。

进聚合的前置条件，缺一条就 `inconclusive`（不是 null，也不是 true）：

* 计划必须带 `reproduction` 块（复现计划），且 `validate` 通过；
* 轮数必须正好等于计划登记的 `rounds`（多跑的不许偷偷算进来）；
* 每轮 `round_complete=true`、`sampling_comparability=established`。
"""
from __future__ import annotations

import hashlib
import json

from . import ac_plan

SCHEMA = 'v08-workflow-ac-reproduction-v1'


def _digest(value):
    blob = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


def aggregate(plan, rounds):
    """`rounds` 是若干份 `ac_compare.compare_round` 的回执（按执行顺序）。"""
    problems = list(ac_plan.validate(plan))
    reproduction = plan.get('reproduction') or {}
    if not reproduction:
        problems.append('不是复现计划：缺少 reproduction 块')
    expected = reproduction.get('rounds')
    if expected is not None and len(rounds) != expected:
        problems.append(f'轮数 {len(rounds)} 与计划登记的 {expected} 不一致')

    rows = []
    for index, result in enumerate(rounds, start=1):
        delivery = result.get('delivery') or {}
        rows.append(dict(round=index,
                         plan_hash=result.get('plan_hash'),
                         round_hash=result.get('hash'),
                         paired_difference=delivery.get('paired_difference_C_minus_A'),
                         round_complete=bool(delivery.get('round_complete')),
                         sampling_comparability=result.get('sampling_comparability'),
                         per_arm=delivery.get('per_arm')))

    if problems:
        verdict, reason = 'inconclusive', '; '.join(problems)
    else:
        for row in rows:
            if not row['round_complete']:
                problems.append(f'第 {row["round"]} 轮不完整')
            if row['sampling_comparability'] != 'established':
                problems.append(f'第 {row["round"]} 轮采样身份未确立')
        unit = reproduction.get('minimum_detectable_unit', '+1 task')
        need = 1 if unit.startswith('+1') else None
        if need is None:
            problems.append(f'不支持的可分辨单位: {unit}')
        falling = [row['round'] for row in rows
                   if row['paired_difference'] is None or row['paired_difference'] < need]
        if problems:
            verdict, reason = 'inconclusive', '; '.join(problems)
        elif falling:
            verdict = 'null'
            reason = (f'第 {falling[0]} 轮配对差 < {unit}：按预登记规则判 null，'
                      f'不再追加轮次')
        else:
            verdict, reason = 'reproduced', (
                f'{len(rows)} 轮全部 ≥ {unit}；范围限定为 '
                f'{reproduction.get("subject")}')

    body = dict(schema=SCHEMA, plan_id=plan.get('plan_id'), plan_hash=plan.get('hash'),
                rule=reproduction.get('rule'),
                rounds_expected=expected, rounds_observed=len(rows),
                minimum_detectable_unit=reproduction.get('minimum_detectable_unit'),
                per_round=rows, verdict=verdict, reason=reason,
                formal_gain_claim=verdict == 'reproduced',
                formal_comparison_credit=verdict == 'reproduced',
                scope=reproduction.get('subject'),
                boundaries=('no performance claim; no developer-time claim; '
                            'single device; single model family'))
    body['hash'] = _digest(body)
    return body