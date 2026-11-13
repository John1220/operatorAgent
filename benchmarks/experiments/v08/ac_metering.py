"""把一格的 A 臂派发记录改写成 `usage-reconcile` 认的 arm log。

为什么需要这一步：`ac_live` 的 A 臂走的是 `model_gateway` + 共享账本，没有经过
`arm_gate.ArmGate`，所以盘上没有 `v08-arm-gate-v1` 文档。而 **每一行派发都是有据的**——
网关把每次请求原样落进了 `cell/outer-requests/request-NNNN.json`（含 model/temperature/
n/max_tokens/enable_thinking）。这个模块只做**转录**，不估、不补：

* 一行派发对应一个真实请求文件，`status='answered'` 只在响应确实完成时写（
  `transport-failures.jsonl` 里同 index 的 `outcome` 决定）；
* `usage` 一律 `None`——网关不记录 token 计数，"记不到"就是 `None`，
  对账回执会把那一臂的 token 合计报成 null，而不是假装它是 0；
* 采样字段只在请求体里真有的时候才写。
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit

PROTOCOL = 'v08-arm-gate-v1'


def _outcomes(cell):
    """每一行派发的终态：网关的 transport 记录说了算。"""
    path = Path(cell) / 'transport-failures.jsonl'
    rows = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            rows[int(row.get('index', -1))] = row.get('outcome')
    return rows


def arm_document(cell, *, arm='A', budget, provider=None, source='ac_live model_gateway transcript'):
    cell = Path(cell)
    requests = sorted((cell / 'outer-requests').glob('request-*.json'))
    if not requests:
        raise SystemExit(f'{arm}: {cell} 下没有 outer-requests 记录，无法转录')
    outcomes = _outcomes(cell)
    dispatches = []
    for index, path in enumerate(requests):
        body = json.loads(path.read_text())
        outcome = outcomes.get(index)
        row = dict(id=index + 1,
                   status='answered' if outcome == 'completed' else
                          'failed' if outcome == 'failed' else 'abandoned',
                   provider=str(provider or urlsplit(str(body.get('__base_url') or
                                                          'declared://unknown')).hostname or 'declared'),
                   model=str(body.get('model')), max_output_tokens=body.get('max_tokens'),
                   reserved_at=None, finished_at=None, usage=None)
        for field in ('temperature', 'n', 'enable_thinking'):
            if body.get(field) is not None:
                row[field] = body[field]
        dispatches.append(row)
    return dict(arm=arm, source=source, metering='ac_live_transcript', schema=PROTOCOL,
                created_at=None, elapsed_s=None, budget=dict(budget), dispatches=dispatches)


def write(cell, output, *, arm='A', budget, provider=None):
    document = arm_document(cell, arm=arm, budget=budget, provider=provider)
    Path(output).write_text(json.dumps(document, ensure_ascii=False, indent=2) + '\n',
                            encoding='utf-8')
    return document

#: 采样身份里"必须一致"的字段（与 `arm_gate.SAMPLING_FIELDS` 同一份语义）。
SAMPLING_FIELDS = ('model', 'temperature', 'n', 'max_tokens', 'enable_thinking',
                   'reasoning_effort')


def _identity(body):
    return tuple((field, body.get(field)) for field in SAMPLING_FIELDS)


def sampling_identity(cells, *, provider=None):
    """从两臂的**实际请求记录**核对采样身份是否一致。

    `usage-reconcile` 比的是"插件账本 vs 另一臂自报"；本轮的模型调用**两臂都走同一个
    supervisor 网关**，插件侧没有 `host-model-budget.json`，那条路不适用。但两臂的请求
    原文都落在 `cell/outer-requests/` 里，采样身份可以直接从中读出、逐字比对。
    """
    per_arm = {}
    for arm, cell in sorted(cells.items()):
        seen = {}
        for path in sorted((Path(cell) / 'outer-requests').glob('request-*.json')):
            body = json.loads(path.read_text())
            key = _identity(body)
            seen[key] = seen.get(key, 0) + 1
        per_arm[arm] = dict(dispatches=sum(seen.values()),
                            distinct=[dict(values) for values in seen],
                            counts=[seen[values] for values in seen])
        per_arm[arm]['provider'] = provider
    sets = {arm: {tuple(sorted(row.items())) for row in per_arm[arm]['distinct']}
            for arm in per_arm}
    arms = sorted(sets)
    equal = len(arms) == 2 and sets[arms[0]] == sets[arms[1]] and all(sets.values())
    reasons = []
    if not equal:
        reasons.append('两臂的采样身份不同' if len(arms) == 2 else '需要一个 A 和一个 C')
    for arm in arms:
        if len(sets[arm]) > 1:
            reasons.append(f'{arm} 臂内部采样身份不唯一（{len(sets[arm])} 种）')
    return dict(schema='v08-sampling-identity-v1',
                sampling_comparability='established' if equal and not reasons else 'not_established',
                fields=list(SAMPLING_FIELDS), per_arm=per_arm, reasons=reasons,
                source='每一条派发的请求原文（cell/outer-requests/request-NNNN.json）')
