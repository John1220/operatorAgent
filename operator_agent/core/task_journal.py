"""任务的**唯一权威状态**：一条只追加的事件流。

为什么要有它。此前一次 run 的状态散在多个文件里，各由不同模块更新：

    declaration.json / contract.json / run.json   ← prepare
    host-task.json / host-task-policy.json        ← host_task_budget
    host-model-budget.json                        ← JS 侧账本
    run-status.json                               ← tool_calls 或 autonomous

于是「终态变化时谁先写」这个问题，**每个写者按自己的理解回答**；`resume` 要同时看
几个文件才能拼出「现在到底是什么状态」；而父进程若在写之前崩掉，`run-status.json`
就永远停在 `running` —— 既不能说它还在跑，也不能说它跑完了。

现在的形状：**`journal.jsonl` 是唯一权威，其余全部是它的投影**。事件只追加、不改写，
每条带 `rev`（从 1 连续递增）。`status` / `best` / `resume` 都是 `fold(events)`。

三条纪律：

1. **只追加**。没有「更新某条事件」这种操作。改状态 = 追加一条新事件。
2. **`rev` 连续**。断号即视为损坏（不是「跳过」）—— 断号意味着有人绕过这个模块写过。
3. **每行一条、单次 `write`**。POSIX 下 `O_APPEND` 的小写入是原子的，所以 Python 侧与
   JS 侧可以各自追加而不会互相撕裂半行。跨进程的顺序由 `flock` 保证（见 `append`）。

投影**不缓存**：`fold` 每次都从事件流现算。事件数量是「一次任务的转移次数」，量级在
数十条，现算比维护缓存的一致性便宜得多。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

JOURNAL = 'journal.jsonl'

#: 事件 schema。形状一变就换号；旧流按旧规则解释或直接拒绝。
SCHEMA = 'operator-task-journal-v1'

#: 一次任务的生命周期。`result_unknown` 是与 `failed` **并列**的一格，不是它的子类：
#: 它说的是「执行可能已经发生，但结果没有到达账本」，与「执行失败了」是两件事。
PHASES = ('prepared', 'queued', 'running', 'completed', 'failed', 'cancelled')

#: 终止相位：到达之后不再接受新事件（除非是 `recovered` 一类的显式补偿）。
TERMINAL = frozenset({'completed', 'failed', 'cancelled'})

#: 结果传输三态。与相位正交 —— 相位说「任务走到哪了」，这一格说「结果回来了没有」。
RESULT_STATES = ('received', 'unknown', 'none')


class JournalError(ValueError):
    """事件流不可用。带稳定 code，便于调用方分支。"""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        self.detail = detail or code
        super().__init__(f"{code}: {self.detail}")


def path_for(run) -> Path:
    return Path(run) / JOURNAL


def _line(event: dict) -> bytes:
    return (json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False)
            + '\n').encode('utf-8')


def read_events(run) -> list:
    """读出全部事件。文件不存在 ⇒ 空流（不是错误：还没开始也是合法状态）。"""
    target = path_for(run)
    if not target.is_file():
        return []
    events = []
    with open(target, 'r', encoding='utf-8') as handle:
        for number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError as error:
                # 半行意味着有人没走 `append`，或者写到一半掉电。两种都不许猜。
                raise JournalError('journal_corrupt',
                                   f'第 {number} 行不是合法 JSON: {error}') from error
            if not isinstance(event, dict) or event.get('schema') != SCHEMA:
                raise JournalError('journal_corrupt',
                                   f'第 {number} 行 schema 不是 {SCHEMA!r}')
            events.append(event)
    return events


def last_rev(events: Iterable) -> int:
    return max((event.get('rev', 0) for event in events), default=0)


def append(run, kind: str, *, invocation: str = '', **fields) -> dict:
    """追加一条事件，返回写下去的那条（含分配到的 `rev`）。

    加锁的理由不是防止两个人同时追加（`O_APPEND` 已经保证不撕裂），而是保证
    `rev` **连续且不重复** —— 两者都读「当前最大值」再 +1，没有锁就会撞号。
    """
    if not isinstance(kind, str) or not kind.strip():
        raise JournalError('journal_bad_event', 'kind 必须是非空字符串')
    target = path_for(run)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_suffix('.lock')
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        import fcntl
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        events = read_events(run)
        # 用投影判断，而不是读某条事件的某个键：相位是「事件流走到哪」的**推论**，
        # 不是任何单条事件的字段。直接读键的话，这条门永远开着。
        current = fold(events)['phase'] if events else 'absent'
        if current in TERMINAL:
            raise JournalError(
                'journal_closed',
                f"任务已到达终止相位 {current!r}，不再接受新事件")
        rev = last_rev(events) + 1
        event = dict(schema=SCHEMA, rev=rev, kind=kind, at=time.time(),
                     invocation=invocation or '', **fields)
        handle = os.open(target, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
        try:
            os.write(handle, _line(event))
        finally:
            os.close(handle)
        return event
    finally:
        os.close(descriptor)


def fold(events: Iterable) -> dict:
    """事件流 → 现在是什么状态。**唯一**的投影函数。

    返回的每个键都是「问题 → 答案」，而不是某个文件的镜像：
    `phase` 走到哪一步、`result` 结果回来了没有、`budget` 记账、`last` 最后一条。
    """
    events = list(events)
    if not events:
        return dict(phase='absent', result='none', rev=0, invocation='',
                    budget={}, last=None, history=[])

    revs = [event.get('rev') for event in events]
    if revs != list(range(1, len(events) + 1)):
        # 断号不是「跳过」，是「有人绕过这个模块写过」—— 那种流的顺序不可信。
        raise JournalError('journal_gap', f'rev 不连续: {revs}')

    phase = 'prepared'
    result = 'none'
    budget: dict = {}
    invocation = ''
    for event in events:
        kind = event.get('kind', '')
        invocation = event.get('invocation') or invocation
        # 相位由 `kind` 显式推进；其余事件只更新各自的格子。
        if kind in PHASES:
            phase = kind
        elif kind.endswith('_requested'):
            phase = 'queued'
        elif kind.endswith('_started'):
            phase = 'running'
        if 'result_state' in event:
            result = event['result_state']
        for key in ('budget', 'reservations'):
            if key in event and isinstance(event[key], dict):
                budget.update(event[key])
        if kind == 'settled' and 'budget' in event:
            budget.update(event['budget'])
    return dict(phase=phase, result=result, rev=last_rev(events),
                invocation=invocation, budget=budget, last=events[-1],
                history=[event.get('kind') for event in events])


def needs_recovery(projection: dict) -> bool:
    """这个任务需要人（或恢复流程）看一眼吗。

    两个条件任一成立就要：结果未知（执行可能发生了），或者相位停在非终止处
    而事件流却不再增长（父进程死了）。
    """
    if projection.get('result') == 'unknown':
        return True
    return projection.get('phase') not in TERMINAL | {'absent'}