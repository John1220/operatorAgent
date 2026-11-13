"""One ledger for every arm: a real budget gate, and the log `usage-reconcile` reads.

WHY this exists
---------------
R2's metering receipt names one blocker above all others: *the A arm has no equivalent
model-call hard gate*. B and C reserve before each dispatch through the plugin's durable
ledger; a plain agent has whatever its provider happened to log. "Same budget" was a
sentence in a plan, not a mechanism — so a pair could be reported as same-budget while
one side had actually spent twice the calls.

This module is that mechanism, and it is deliberately **not** an estimator:

* a dispatch is reserved *before* the call and settled *after* it (``reserved`` →
  ``answered`` / ``failed`` / ``cancelled`` / ``abandoned``), so a crash leaves a row
  that says "this was started", never a row that pretends it did not happen;
* the three caps (wall clock, model calls, reserved output tokens) are enforced, not
  reported: exceeding one raises ``ArmBudgetError`` instead of writing a reading;
* sampling fields are recorded **only when the caller knows them**. An unrecorded field
  stays unrecorded — the reconciler then says ``not_established`` rather than assuming
  equality (docs/testing/v08-arm-metering.md §2).

The document it writes is exactly the schema ``operator_agent usage-reconcile`` consumes
(``decide/arm_metering.normalize_arm_document``), so A and B/C are read by one reader.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

PROTOCOL = 'v08-arm-gate-v1'

#: 采样身份里"记不到就别记"的字段。记不到与相等是两件事：前者让对账回执
#: 报 `not_established`，后者才允许声称同口径。
SAMPLING_FIELDS = ('provider', 'model', 'temperature', 'n', 'max_output_tokens',
                   'enable_thinking')

STATUSES = ('reserved', 'answered', 'failed', 'cancelled', 'abandoned')


class ArmBudgetError(ValueError):
    def __init__(self, code, detail):
        self.code = code
        self.detail = detail
        super().__init__(detail)


def _count(value, field):
    if type(value) is not int or value < 0:
        raise ArmBudgetError('arm_gate_policy', f'{field} 必须是非负整数: {value!r}')
    return value


class ArmGate:
    """A-arm budget gate. One instance per arm per pair.

    用法（调用方在**发请求之前** reserve，回来之后 finish）::

        gate = ArmGate(path, budget=dict(max_calls=30, max_output_tokens_per_call=8192,
                                         total_output_tokens_reservations=245760,
                                         wall_clock_s=2400))
        rid = gate.reserve(provider='aliyun', model='...', max_output_tokens=8192,
                           sampling=dict(temperature=0.7, n=1, enable_thinking=True))
        ... call the model ...
        gate.finish(rid, usage=dict(inputTokens=..., outputTokens=..., totalTokens=...))
    """

    def __init__(self, path, *, budget, arm='A', source='declared arm log',
                 metering='arm_gate', clock=time.monotonic, wall=time.time):
        path = Path(path)
        if path.exists():
            # 同一份日志不许重开：两段互不相干的运行写进一个文件，对账会把
            # "两次预算"读成"一次超支"。新的一对就是新目录。
            raise ArmBudgetError('arm_gate_exists', f'arm log 已存在，不覆盖: {path}')
        self.path = path
        self.arm = str(arm)
        self.source = str(source)
        self.metering = str(metering)
        self._clock, self._wall = clock, wall
        self._started = self._clock()
        self._budget = dict(
            max_calls=_count(budget.get('max_calls'), 'max_calls')
            if budget.get('max_calls') is not None else None,
            max_output_tokens_per_call=_count(
                budget.get('max_output_tokens_per_call'), 'max_output_tokens_per_call')
            if budget.get('max_output_tokens_per_call') is not None else None,
            total_output_tokens_reservations=_count(
                budget.get('total_output_tokens_reservations'),
                'total_output_tokens_reservations')
            if budget.get('total_output_tokens_reservations') is not None else None,
            wall_clock_s=_count(budget.get('wall_clock_s'), 'wall_clock_s')
            if budget.get('wall_clock_s') is not None else None,
        )
        self._rows = []
        self._write()

    # ── 预算 ────────────────────────────────────────────────────────────────
    def elapsed_s(self):
        return round(self._clock() - self._started, 3)

    def remaining(self):
        reserved = sum(row.get('max_output_tokens') or 0 for row in self._rows)
        return dict(
            calls=(None if self._budget['max_calls'] is None
                   else self._budget['max_calls'] - len(self._rows)),
            output_tokens=(None if self._budget['total_output_tokens_reservations'] is None
                           else self._budget['total_output_tokens_reservations'] - reserved),
            wall_clock_s=(None if self._budget['wall_clock_s'] is None
                          else round(self._budget['wall_clock_s'] - self.elapsed_s(), 3)),
        )

    def _refuse_at_the_edge(self, max_output_tokens):
        left = self.remaining()
        if left['wall_clock_s'] is not None and left['wall_clock_s'] <= 0:
            raise ArmBudgetError('arm_deadline',
                                 f'墙钟预算用尽: {self.elapsed_s()}s / {self._budget["wall_clock_s"]}s')
        if left['calls'] is not None and left['calls'] <= 0:
            raise ArmBudgetError('arm_call_budget',
                                 f'模型调用预算用尽: 已派发 {len(self._rows)} / {self._budget["max_calls"]}')
        per_call = self._budget['max_output_tokens_per_call']
        if per_call is not None and max_output_tokens > per_call:
            raise ArmBudgetError('arm_output_cap',
                                 f'单次输出预留 {max_output_tokens} 超过上限 {per_call}')
        if left['output_tokens'] is not None and max_output_tokens > left['output_tokens']:
            raise ArmBudgetError('arm_token_budget',
                                 f'累计输出预留不足: 还剩 {left["output_tokens"]}, 本次要 {max_output_tokens}')

    # ── 派发 ────────────────────────────────────────────────────────────────
    def reserve(self, *, provider, model, max_output_tokens, sampling=None):
        """发请求**之前**记账。超预算直接拒绝——不写一条"本该被拒"的读数。"""
        max_output_tokens = _count(max_output_tokens, 'max_output_tokens')
        self._refuse_at_the_edge(max_output_tokens)
        row = dict(id=len(self._rows) + 1, status='reserved',
                   provider=str(provider), model=str(model),
                   max_output_tokens=max_output_tokens,
                   reserved_at=self._wall() * 1000.0, finished_at=None, usage=None)
        # 采样字段"能记就记"：None 的项不写键，回执里会以 unrecorded 点名。
        for field in ('temperature', 'n', 'enable_thinking'):
            value = (sampling or {}).get(field)
            if value is not None:
                row[field] = value
        self._rows.append(row)
        self._write()
        return row['id']

    def finish(self, reservation, *, usage=None, status='answered'):
        if status not in STATUSES or status == 'reserved':
            raise ArmBudgetError('arm_gate_status', f'非终态 status={status!r}')
        row = next((item for item in self._rows if item['id'] == reservation), None)
        if row is None:
            raise ArmBudgetError('arm_gate_unknown', f'没有 id={reservation} 的预留')
        if row['status'] != 'reserved':
            raise ArmBudgetError('arm_gate_settled', f'id={reservation} 已经结算过')
        if usage is not None:
            # 缺计数不是 0：`usage=None` 保持 None，回执里那一臂的 token 合计变 null。
            if not isinstance(usage, dict):
                raise ArmBudgetError('arm_gate_usage', f'usage 必须是对象或 None: {usage!r}')
            for key, value in usage.items():
                _count(value, key)
        row.update(status=status, usage=usage, finished_at=self._wall() * 1000.0)
        self._write()
        return dict(self.document())

    # ── 产物 ────────────────────────────────────────────────────────────────
    def document(self):
        """`usage-reconcile` 读的那份 arm log（一次派发一行，不聚合）。"""
        return dict(arm=self.arm, source=self.source, metering=self.metering,
                    schema=PROTOCOL, created_at=self._wall(),
                    elapsed_s=self.elapsed_s(),
                    budget=dict(max_calls=self._budget['max_calls'],
                                max_output_tokens_per_call=self._budget['max_output_tokens_per_call'],
                                total_output_tokens_reservations=self._budget[
                                    'total_output_tokens_reservations'],
                                wall_clock_s=self._budget['wall_clock_s']),
                    dispatches=[json.loads(json.dumps(row)) for row in self._rows])

    def _write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.document(), ensure_ascii=False, indent=2) + '\n',
                             encoding='utf-8')


def in_flight(document):
    """还有几条预留没结算——对账判 `nothing_in_flight` 用的就是这一格。"""
    return sum(1 for row in (document or {}).get('dispatches', [])
               if row.get('status') == 'reserved')

def wrap(dispatch, gate, *, provider, model, default_output_tokens=None):
    """把一次派发包成"先记账、再上游、再结算"——A 臂接进这条路径的唯一改动点。

    `dispatch` 是 `phase1/host.model_gateway` 里那个真正的上游调用。包在这里而不是
    在 DSH 里，是因为「这次请求到底发出去没有」只有一个地方知道。
    """
    def call(request):
        cap = (request or {}).get('max_tokens') or (request or {}).get('max_completion_tokens')
        cap = cap if type(cap) is int and cap >= 0 else (default_output_tokens or 0)
        sampling = (request or {}).get('sampling') if isinstance((request or {}).get('sampling'), dict) else None
        sampling = sampling or dict(temperature=(request or {}).get('temperature'),
                                    n=(request or {}).get('n'))
        reservation = gate.reserve(provider=provider, model=model, max_output_tokens=int(cap),
                                   sampling=sampling)
        try:
            result = dispatch(request)
        except BaseException:
            gate.finish(reservation, status='failed', usage=None)
            raise
        # 流式响应里的 usage 这里不解析：缺计数就保持 None（未知不等于 0）。
        gate.finish(reservation, status='answered', usage=None)
        return result
    return call


# ── 入口：把"第 N+1 次被拒、日志仍是 N 行"做成可复现的一条命令 ──────────────

def _summary(document):
    from operator_agent.decide.arm_metering import MeteringError, normalize_arm_document
    out = dict(arm=document['arm'], dispatches=len(document['dispatches']),
               budget=document['budget'], path=None)
    try:
        normal = normalize_arm_document(document)
        out.update(parsed=True, usage_known=normal['usage_known'],
                   usage_unknown=normal['usage_unknown'], in_flight=normal['in_flight'])
    except MeteringError as error:
        out.update(parsed=False, parse_error=f'{error.code}: {error.detail}')
    return out


def main(argv=None):
    """`python -m benchmarks.experiments.v08.arm_gate --budget '{"max_calls":30,…}' --probe 31`

    做的事只有一件：按预算连续 reserve N 次，报出第一次被拒的那条与最终日志行数。
    它是 P0-3 的验收动作，也是驱动里"预算真的拦住了"的最小复现——不碰网络、不调模型。
    """
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(prog='arm_gate', description=__doc__.splitlines()[0])
    parser.add_argument('--budget', required=True, help='JSON：max_calls / total_output_tokens_reservations / wall_clock_s …')
    parser.add_argument('--per-call', type=int, default=8192, help='每次预留的输出 token')
    parser.add_argument('--probe', type=int, default=31, help='连续 reserve 几次（默认 31）')
    parser.add_argument('--arm', default='A', help='臂名（默认 A）')
    parser.add_argument('--path', help='日志写到哪（默认临时目录）')
    parser.add_argument('--json-out', help='同时写一份 JSON 到文件')
    args = parser.parse_args(argv)

    budget = json.loads(args.budget)
    path = Path(args.path) if args.path else Path(tempfile.mkdtemp(prefix='arm-gate-')) / 'arm-A.json'
    gate = ArmGate(path, budget=budget, arm=args.arm)

    refusal = None
    completed = 0
    for index in range(args.probe):
        try:
            reservation = gate.reserve(provider='probe', model='probe',
                                       max_output_tokens=args.per_call)
        except ArmBudgetError as error:
            refusal = dict(code=error.code, detail=error.detail, at_attempt=index + 1)
            break
        gate.finish(reservation, usage=dict(inputTokens=0, outputTokens=0, totalTokens=0),
                    status='answered')
        completed += 1

    summary = _summary(gate.document())
    summary.update(attempts=args.probe, settled=completed, refusal=refusal,
                   accepted_as_designed=(refusal is not None and completed == args.probe - 1
                                         if refusal else completed == args.probe))
    text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(text + '\n', encoding='utf-8')
    print(text)
    return 0 if summary['parsed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
