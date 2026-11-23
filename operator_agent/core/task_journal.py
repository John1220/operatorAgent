"""任务的**唯一权威状态**：一条只追加的事件流。

为什么要有它。此前一次 run 的状态散在多个文件里，各由不同模块更新：

    declaration.json / contract.json / run.json   ← prepare
    host-task.json / host-task-policy.json        ← host_task_budget
    host-model-budget.json                        ← JS 侧账本
    run-status.json                               ← tool_calls 或 autonomous

于是「终态变化时谁先写」这个问题，**每个写者按自己的理解回答**；`resume` 要同时看
几个文件才能拼出「现在到底是什么状态」；而父进程若在写之前崩掉，`run-status.json`
就永远停在 `running` —— 既不能说它还在跑，也不能说它跑完了。

现在的形状：**`journal.jsonl` 是唯一权威，其余全部是它的投影**。

## 两层，不是一层

一个 run 里会做**多次** verify（首轮 L2 失败 → 修复后通过 → 再跑一次独立 final）。
所以「任务」与「单次调用」必须是两层：

    task  一次 prepare 建立的整段工作        task_prepared → active → closed
    job   一次 verify / bench / profile      job_queued → started → finished

任务的终止只有一个出口 `task_closed`；单个 Job 的终态**只约束它自己**，不封整条流。
把这两层压成一根相位，会出现「第一个 Job 成功就锁死整条流」——这个错误实测过：
Job A 写 `completed` 之后，Job B 的 `job_queued` 被拒，`rev` 停在 4。

## 身份

`task_id` 由 prepare 建立，写在每条事件上；`invocation_id` 跟着**单次调用**走，写在
该 Job 的事件上。两者不是一回事：一个任务有多个 invocation。

## 三条纪律

1. **只追加**。改状态 = 追加一条新事件。没有「更新某条事件」这种操作。
2. **`rev` 连续**。断号即视为损坏（不是「跳过」）—— 断号意味着有人绕过这个模块写过。
3. **写入与回放共用同一套校验**。拼错的 `kind`、缺 `job_id`、越界的结果枚举，在
   `append` 与 `fold` 里都要被拒；拒绝时 `rev` 保持不变。

投影**不缓存**：`fold` 每次从事件流现算。事件是「一次任务的转移」，量级在数十条。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

JOURNAL = 'journal.jsonl'

#: 事件 schema。两层形状 + 事件词表都变了，所以换号。
#: 无法归属的旧流（v1 的 job 事件没有 `job_id`）**明确拒绝**，不做静默兼容。
SCHEMA = 'operator-task-journal-v2'

#: 任务相位。`closed` 是**唯一**的任务级终止。
TASK_PHASES = ('absent', 'prepared', 'active', 'closed')

#: 任务事件。
TASK_EVENTS = frozenset({'task_prepared', 'task_activated', 'task_closed'})

#: 单次调用的事件。`job_finished` 携带 `outcome` 与 `result`。
JOB_EVENTS = frozenset({'job_queued', 'job_started', 'job_finished'})

#: 预算写者接入时再用。这里先承认它们的合法性，免得届时又改一次词表。
BUDGET_EVENTS = frozenset({'budget_reserved', 'budget_settled'})

KINDS = TASK_EVENTS | JOB_EVENTS | BUDGET_EVENTS

#: 任务与 Job 共用的终态词表。
OUTCOMES = ('completed', 'failed', 'cancelled')

#: 结果回来了没有。与 `outcome` **正交**：
#: 「执行可能发生了、但结果没到账本」和「执行失败了」是两个事实。
RESULT_STATES = ('none', 'received', 'unknown')

#: `unknown` 的细分。diagnosis 用，不进相位。
RESULT_REASONS = ('receipt_absent', 'invocation_mismatch', 'invalid_timestamp')

#: 一次调用做的是什么。将来 profile 也走这套事件，只是这里不同。
OPERATIONS = ('verify', 'bench', 'profile')

#: 只有这些 Kind 需要 `job_id`。
NEEDS_JOB = JOB_EVENTS


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


def validate(event: dict) -> None:
    """一条事件是否合规。**append 与 fold 共用这一份**，所以写入侧挡不住的东西，
    回放侧也挡得住 —— 两侧规则不一致时，手写进文件的坏事件就成了唯一真相。"""
    if not isinstance(event, dict):
        raise JournalError('journal_bad_event', '事件必须是对象')
    kind = event.get('kind')
    if kind not in KINDS:
        raise JournalError(
            'journal_unknown_kind',
            f"kind={kind!r} 不在闭合词表里；已知的是 {sorted(KINDS)}")
    if kind in NEEDS_JOB:
        job_id = event.get('job_id')
        if not isinstance(job_id, str) or not job_id.strip():
            raise JournalError('journal_missing_job',
                               f"{kind} 必须带非空的 job_id（一次调用一个 Job）")
    if kind in JOB_EVENTS:
        operation = event.get('operation')
        if operation not in OPERATIONS:
            raise JournalError(
                'journal_unknown_operation',
                f"operation={operation!r} 不在闭合词表里；已知的是 {list(OPERATIONS)}")
    if kind == 'job_finished':
        if event.get('outcome') not in OUTCOMES:
            raise JournalError(
                'journal_unknown_outcome',
                f"outcome={event.get('outcome')!r} 必须是 {list(OUTCOMES)} 之一")
        result = event.get('result')
        if result not in RESULT_STATES:
            raise JournalError(
                'journal_unknown_result',
                f"result={result!r} 必须是 {list(RESULT_STATES)} 之一")
        if result == 'unknown' and event.get('result_reason') not in RESULT_REASONS:
            # 「未知」必须说清是怎么未知的，否则诊断入口就断了。
            raise JournalError(
                'journal_missing_result_reason',
                f"result=unknown 必须带 result_reason，选 {list(RESULT_REASONS)}")
    if kind == 'task_closed':
        if event.get('outcome') not in OUTCOMES:
            raise JournalError(
                'journal_unknown_outcome',
                f"task_closed 的 outcome={event.get('outcome')!r} 必须是 {list(OUTCOMES)} 之一")


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
            if not isinstance(event, dict):
                raise JournalError('journal_corrupt', f'第 {number} 行不是对象')
            if event.get('schema') != SCHEMA:
                raise JournalError(
                    'journal_schema_mismatch',
                    f"第 {number} 行 schema={event.get('schema')!r}，本层只认 {SCHEMA!r}；"
                    "旧流（没有 job_id 的 job 事件）无法归属，拒绝而不猜")
            validate(event)
            events.append(event)
    return events


def last_rev(events: Iterable) -> int:
    return max((event.get('rev', 0) for event in events), default=0)


def append(run, kind: str, *, task_id: str = '', job_id: str = '',
           invocation: str = '', **fields) -> dict:
    """追加一条事件，返回写下去的那条（含分配到的 `rev`）。

    加锁的理由不是防止两个人同时追加（`O_APPEND` 已经保证不撕裂半行），而是保证
    `rev` **连续且不重复** —— 两者都读「当前最大值」再 +1，没有锁就会撞号。

    拒绝时**不写任何东西**，`rev` 保持原值。
    """
    target = path_for(run)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_suffix('.lock')
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        import fcntl
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        events = read_events(run)
        current = fold(events)['task']['phase'] if events else 'absent'
        # **唯一的封账门**：只有任务被显式关闭才拒绝后续事件。
        # 单个 Job 的终态只约束它自己 —— 否则第一个 Job 成功就锁死整条流。
        if current == 'closed':
            raise JournalError(
                'journal_closed',
                "任务已显式关闭（task_closed），不再接受新事件；"
                "要开始新的工作请新建任务目录")
        if kind in JOB_EVENTS and current == 'prepared':
            raise JournalError(
                'journal_not_activated',
                "任务还没有激活（task_activated），Job 事件不能先到")

        event = dict(schema=SCHEMA, rev=last_rev(events) + 1, kind=kind,
                     at=time.time(), task_id=task_id or '', job_id=job_id or '',
                     invocation=invocation or '', **fields)
        validate(event)
        handle = os.open(target, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
        try:
            os.write(handle, _line(event))
        finally:
            os.close(handle)
        return event
    finally:
        os.close(descriptor)


def _job_row(event: dict) -> dict:
    kind = event['kind']
    if kind == 'job_queued':
        return dict(phase='queued')
    if kind == 'job_started':
        return dict(phase='running')
    return dict(phase=event['outcome'], outcome=event['outcome'],
                result=event['result'], result_reason=event.get('result_reason', ''))


def fold(events: Iterable) -> dict:
    """事件流 → 现在是什么状态。**唯一**的投影函数。

    返回的每个键都是「问题 → 答案」，而不是某个文件的镜像：

        task   {phase, task_id, outcome, reason}      任务走到哪、怎么结束的
        jobs   {job_id: {phase, outcome, result, ...}} 单次调用各自的账
        budget {…}                                     预算
        rev                                           最后一条的编号
    """
    events = list(events)
    if not events:
        return dict(task=dict(phase='absent', task_id='', outcome='', reason=''),
                    jobs={}, budget={}, rev=0, last=None, history=[])

    revs = [event.get('rev') for event in events]
    if revs != list(range(1, len(events) + 1)):
        # 断号不是「跳过」，是「有人绕过这个模块写过」—— 那种流的顺序不可信。
        raise JournalError('journal_gap', f'rev 不连续: {revs}')

    task = dict(phase='absent', task_id='', outcome='', reason='')
    jobs: dict = {}
    budget: dict = {}
    for event in events:
        validate(event)
        kind = event['kind']
        task['task_id'] = event.get('task_id') or task['task_id']
        if kind == 'task_prepared':
            task['phase'] = 'prepared'
        elif kind == 'task_activated':
            task['phase'] = 'active'
        elif kind == 'task_closed':
            task['phase'] = 'closed'
            task['outcome'] = event['outcome']
            task['reason'] = event.get('reason', '')
        elif kind in JOB_EVENTS:
            job_id = event['job_id']
            row = jobs.setdefault(job_id, dict(phase='queued', operation=event['operation'],
                                               invocation='', outcome='', result='none',
                                               result_reason=''))
            row.update(_job_row(event))
            row['operation'] = event['operation']
            row['invocation'] = event.get('invocation') or row['invocation']
            # 刻意**不**在这里把任务推进到 active：激活只有 `task_activated` 一条路。
            # 让 fold 也「顺手」推进的话，同一个状态就有两条到达路径 ——
            # 而那正是这次要消灭的那种东西。
        elif kind in BUDGET_EVENTS:
            if isinstance(event.get('budget'), dict):
                budget.update(event['budget'])
    return dict(task=task, jobs=jobs, budget=budget, rev=last_rev(events),
                last=events[-1], history=[event['kind'] for event in events])


def is_open(projection: dict) -> bool:
    """任务还没封账吗。**只回答任务相位**，不回答任何 Job 的进展。

    一个 Job 的结果仍为 `unknown` 并不让任务变「未结束」；反过来，任务 `closed`
    之后即使某个 Job 的回执还缺，这里也必须返回 False。

    自动恢复**不能**只靠这个函数：它还要与该 invocation 匹配的 worker / supervisor
    存活证据。见 `docs` 里「相位停在非终止处」那条。
    """
    return projection.get('task', {}).get('phase') in ('prepared', 'active')