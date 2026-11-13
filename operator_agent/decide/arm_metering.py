"""Reconcile per-arm model metering so "same budget, same sampling" can be judged.

The eight-cell run reported ``sampling_comparability=not_established`` and left cost and
human minutes unknown, because the two arms counted different things: B has a durable
pre-dispatch reservation ledger, A only has whatever the provider saw. This module reads
both sides, normalises them, and states **per check** whether they are comparable — it
does not estimate cost from tokens, does not treat a missing counter as zero, and does
not upgrade "the numbers happen to be close" into "the arms were metered the same way".

Two verdicts, deliberately separate:

* ``metering_comparable`` — do both arms count the same population of dispatches
  (every dispatch has validated usage counters, none left in flight)?
* ``sampling_comparability`` — is the sampling identity **fully recorded** on both sides
  and equal? A field that nobody recorded is ``unrecorded``, which blocks this verdict;
  it is never assumed equal.
"""
from __future__ import annotations

PROTOCOL = 'arm-metering-v1'

#: 采样身份必须逐项记录才能声称「同口径」。缺失不等于相同 —— 这是本期
#: `sampling_comparability=not_established` 的直接原因, 也是补齐它的清单。
REQUIRED_SAMPLING_FIELDS = ('provider', 'model', 'temperature', 'n',
                            'max_output_tokens', 'enable_thinking')

BOUNDARY = (
    'Both arms are described from their own logs. Coverage is declared, not proven: a '
    'provider-side log can miss a dispatch that bypassed it, and the host ledger only '
    'covers native main-agent dispatches after a successful preparation. Cost is unknown: '
    'token counters are not invoices, and no price is inferred here.')


class MeteringError(ValueError):
    def __init__(self, code, detail=None):
        self.code = code
        self.detail = detail or code
        super().__init__(self.detail)


_USAGE_KEYS = ('inputTokens', 'outputTokens', 'totalTokens', 'cacheReadTokens', 'cacheWriteTokens')


def _usage(value):
    """None 保持 None (未记录); 否则每个出现的计数器必须是非负整数。"""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - set(_USAGE_KEYS):
        raise MeteringError('metering_input', f'invalid usage counters {value!r}')
    for key, counter in value.items():
        if type(counter) is not int or counter < 0:
            raise MeteringError('metering_input', f'usage {key}={counter!r} is not a non-negative integer')
    if 'outputTokens' not in value:
        raise MeteringError('metering_input', 'usage without outputTokens cannot be reconciled')
    return value


def _count(value, field):
    if type(value) is not int or value < 0:
        raise MeteringError('metering_field', f'{field} must be a non-negative integer')
    return value


def _sum(rows, key):
    values = [row.get(key) for row in rows]
    if not values or any(type(value) is not int or value < 0 for value in values):
        return None
    return sum(values)


def _identity(rows, key):
    found = {row.get(key) for row in rows if row.get(key) is not None}
    return sorted(found) if found else []


def _uniform(rows, key):
    values = {row.get(key) for row in rows}
    if len(values) != 1:
        return None
    value = values.pop()
    return value


def _sampling(rows, fields):
    """Per-field value if uniform across dispatches, else ``None`` (mixed or absent)."""
    return {field: _uniform(rows, field) for field in fields}


def _finish(arm, source, metering, budget, rows, *, reserved_key=None,
            max_tokens_key=None):
    """Shared tail: counts, tokens, identity and the checks every arm must pass."""
    statuses = {}
    for row in rows:
        status = row.get('status')
        statuses[str(status)] = statuses.get(str(status), 0) + 1
    known = [row for row in rows if isinstance(row.get('usage'), dict)]
    unknown = len(rows) - len(known)
    in_flight = sum(1 for row in rows if row.get('status') == 'reserved')
    usage_rows = [row['usage'] for row in known]
    violations = []
    if reserved_key is not None:
        reserved = [row.get(reserved_key) for row in rows]
        if all(type(value) is int and value >= 0 for value in reserved):
            for row in known:
                cap, used = row.get(reserved_key), (row['usage'] or {}).get('outputTokens')
                if type(cap) is int and type(used) is int and used > cap:
                    violations.append(f'dispatch {row.get("id")}: outputTokens {used} > reserved {cap}')
    # 只要有一次用量未知, 这一臂的 token 合计就是**未知**, 不是「已知的那部分之和」:
    # 半截求和长得像总量, 正是这类对账最容易被引用错的地方。
    complete = unknown == 0 and bool(rows)
    observed_output = _sum(usage_rows, 'outputTokens') if complete else None
    reserved_total = (_sum(rows, reserved_key) if reserved_key is not None else None)
    return dict(
        arm=arm, source=source, metering=metering, budget=dict(budget),
        dispatches=len(rows), status_counts=statuses, in_flight=in_flight,
        usage_known=len(known), usage_unknown=unknown,
        input_tokens=_sum(usage_rows, 'inputTokens') if complete else None,
        output_tokens=observed_output,
        total_tokens=_sum(usage_rows, 'totalTokens') if complete else None,
        reserved_output_tokens=reserved_total,
        providers=_identity(rows, 'provider'), models=_identity(rows, 'model'),
        sampling=_sampling(rows, ('temperature', 'n', 'max_output_tokens',
                                  'enable_thinking', 'provider', 'model')),
        window=dict(started_at_ms=_min(rows, 'reserved_at') or _min(rows, 'started_at_ms'),
                    finished_at_ms=_max(rows, 'finished_at') or _max(rows, 'finished_at_ms')),
        violations=violations,
        cost='unknown')


def _min(rows, key):
    values = [row[key] for row in rows if type(row.get(key)) is int]
    return min(values) if values else None


def _max(rows, key):
    values = [row[key] for row in rows if type(row.get(key)) is int]
    return max(values) if values else None


def normalize_host_ledger(state, policy=None, *, arm='B'):
    """The plugin-side arm: its durable ledger (``host-model-budget.json``).

    ``reserved`` rows are in flight, not spent; an absent ``usage`` stays unknown.

    ``arm`` 是**这一份账本属于哪个臂**的标签。原来写死成 ``'B'``，因为 R2 的两臂是
    B（v0.7.2）/C（v0.8）；但 A/C 工作流轮的插件侧臂名是 ``C``，写死会让
    ``usage-reconcile`` 把 C 的预算记在 B 名下——两个臂名在回执里根本读不出来。
    默认仍是 ``'B'``：既有回执逐字不变（这条由测试与归档回执对拍）。
    """
    if not isinstance(arm, str) or not arm.strip():
        raise MeteringError('metering_input', f'invalid plugin arm label {arm!r}')
    if not isinstance(state, dict) or not isinstance(state.get('records'), list):
        raise MeteringError('metering_input', 'host ledger lacks a records list')
    records = [dict(row) for row in state['records']]
    for row in records:
        if type(row.get('id')) is not int or row.get('status') not in (
                'reserved', 'answered', 'failed', 'cancelled', 'abandoned'):
            raise MeteringError('metering_input', f'invalid ledger record {row.get("id")!r}')
        _usage(row.get('usage'))
    state_limits = state.get('limits') if isinstance(state.get('limits'), dict) else {}
    policy_limits = {}
    if isinstance(policy, dict):
        policy_limits = policy.get('limits') if isinstance(policy.get('limits'), dict) else {}
    limits = state_limits or policy_limits
    budget = dict(max_model_calls=limits.get('max_model_calls'),
                  max_output_tokens_per_call=limits.get('max_output_tokens_per_call'),
                  # 挂钟上限只在任务策略里 (ledger 只冻模型三项), 所以从这里取。
                  wall_clock_s=policy_limits.get('wall_clock_s'))
    normalized = _finish(arm.strip(), 'plugin host-model ledger', 'host_pre_dispatch_reservations',
                         budget, records, reserved_key='max_output_tokens_reserved')
    # 预留下限就是这个请求的输出上限 (插件把它夹在 policy 之内), 所以它可以当作
    # `max_output_tokens` 这一项**已记录** —— 但仍然是「预留」不是「实际用量」。
    normalized['sampling']['max_output_tokens'] = _uniform(records, 'max_output_tokens_reserved')
    return normalized


def normalize_arm_document(document):
    """A arm: a provider-proxy (or harness) log with one row per dispatch.

    Required per dispatch: ``status`` and ``usage`` (null when the provider gave none) —
    plus whatever sampling fields the producer could observe. Missing sampling fields are
    reported as unrecorded rather than assumed equal to the other arm.
    """
    if not isinstance(document, dict) or not isinstance(document.get('dispatches'), list):
        raise MeteringError('metering_input', 'arm log must be an object with a dispatches list')
    rows = [dict(row) for row in document['dispatches']]
    for row in rows:
        if row.get('status') not in ('reserved', 'answered', 'failed', 'cancelled', 'abandoned'):
            raise MeteringError('metering_input', f'invalid dispatch status {row.get("status")!r}')
        _usage(row.get('usage'))
    budget = document.get('budget') if isinstance(document.get('budget'), dict) else {}
    for key in ('max_calls', 'max_output_tokens_per_call'):
        if key in budget and budget[key] is not None:
            _count(budget[key], key)
    if budget.get('wall_clock_s') is not None:
        _count(budget['wall_clock_s'], 'wall_clock_s')
    return _finish(str(document.get('arm') or 'A'),
                   str(document.get('source') or 'declared arm log'),
                   str(document.get('metering') or 'declared'),
                   dict(max_model_calls=budget.get('max_calls'),
                        max_output_tokens_per_call=budget.get('max_output_tokens_per_call'),
                        wall_clock_s=budget.get('wall_clock_s')),
                   rows, reserved_key='max_output_tokens')


def _check(name, ok, detail):
    return dict(check=name, ok=bool(ok), detail=detail)


def compare(arm_a, arm_b):
    """Per-check comparison. ``established`` only when every required check passes."""
    checks = []
    for arm in (arm_a, arm_b):
        checks.append(_check(f'{arm["arm"]}.dispatches_counted', arm['dispatches'] > 0,
                             f'{arm["dispatches"]} dispatch(es) in this arm'))
        checks.append(_check(f'{arm["arm"]}.usage_complete', arm['usage_unknown'] == 0,
                             f'{arm["usage_known"]} known / {arm["usage_unknown"]} unknown usage counters'))
        checks.append(_check(f'{arm["arm"]}.nothing_in_flight', arm['in_flight'] == 0,
                             f'{arm["in_flight"]} reservation(s) still open'))
        checks.append(_check(f'{arm["arm"]}.no_cap_violation', not arm['violations'],
                             '; '.join(arm['violations']) or 'observed output within reserved cap'))
    for key, label in (('max_model_calls', 'model call cap'),
                       ('max_output_tokens_per_call', 'per-call output cap'),
                       ('wall_clock_s', 'wall clock cap')):
        left, right = arm_a['budget'].get(key), arm_b['budget'].get(key)
        checks.append(_check(f'budget.{key}', left is not None and left == right,
                             f'A={left!r} vs B={right!r} ({label})'))
    same_provider = arm_a['providers'] == arm_b['providers'] and len(arm_a['providers']) == 1
    same_model = arm_a['models'] == arm_b['models'] and len(arm_a['models']) == 1
    checks.append(_check('route.provider', same_provider,
                         f'A={arm_a["providers"]} vs B={arm_b["providers"]}'))
    checks.append(_check('route.model', same_model,
                         f'A={arm_a["models"]} vs B={arm_b["models"]}'))
    unrecorded, mismatched = [], []
    for field in REQUIRED_SAMPLING_FIELDS:
        left, right = arm_a['sampling'].get(field), arm_b['sampling'].get(field)
        if field in ('provider', 'model'):
            continue
        if left is None or right is None:
            unrecorded.append(f'{field}: A={left!r} B={right!r}')
        elif left != right:
            mismatched.append(f'{field}: A={left!r} B={right!r}')
    checks.append(_check('sampling.recorded', not unrecorded,
                         '; '.join(unrecorded) or 'every required field is recorded on both arms'))
    checks.append(_check('sampling.equal', not mismatched,
                         '; '.join(mismatched) or 'recorded fields agree across arms'))
    metering_comparable = all(row['ok'] for row in checks
                              if row['check'].endswith(('usage_complete', 'nothing_in_flight',
                                                        'no_cap_violation', 'dispatches_counted')))
    sampling_comparability = 'established' if all(row['ok'] for row in checks) else 'not_established'
    return dict(schema=PROTOCOL, arms={arm_a['arm']: arm_a, arm_b['arm']: arm_b},
                checks=checks, metering_comparable=metering_comparable,
                sampling_comparability=sampling_comparability,
                blockers=[row['check'] + ': ' + row['detail'] for row in checks if not row['ok']],
                unrecorded_sampling=unrecorded, mismatched_sampling=mismatched,
                cost='unknown', boundary=BOUNDARY)