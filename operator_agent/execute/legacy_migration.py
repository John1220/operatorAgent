"""把「账本之前」的任务目录迁进任务账本：**只搬已经发生的事实，不补没发生的**。

为什么需要它。`journal.jsonl` 成为唯一权威之后，仓库里不再有任何代码去读旧相位文件
—— 那些文件还在盘上（一次运行的原始记录），但读它的判据已经没了。于是「这个目录当年
跑成什么样」会从「读得慢」变成「读不到」。迁移把它们搬进同一条流，旧文件原地不动。

三条纪律：

1. **能归属才搬**。`declaration.json` + `contract.json` 是「这个目录被准备过」的凭据；
   缺一个就拒绝（`not_migratable`）—— 不从一个孤零零的相位文件反推任务身份。
2. **不覆盖已有的流**。目录里已经有 `journal.jsonl` 就拒绝（`already_migrated`），
   迁移不是「整理」，不能把两条历史缝在一起。
3. **旧文件里没有的东西不补**。`result` 这一格在旧相位里根本不存在；能确定的写
   `received`，不能确定的写 `unknown` 并说明理由（`receipt_absent` /
   `stopped_before_receipt`），绝不默认成「成功」或「失败」之外的解释。
"""
from __future__ import annotations

import json
from pathlib import Path
import time

from ..core import task_journal as journal

STATUS = 'run-status.json'
TERMINAL = {'done': 'completed', 'failed': 'failed', 'cancelled': 'cancelled'}
TOOLS = {'operator_verify': 'verify', 'operator_bench': 'bench',
         'operator_repair': 'repair'}
LIVE = {'queued', 'running', 'starting', 'executing'}


class MigrationRefused(ValueError):
    def __init__(self, code, detail):
        self.code, self.detail = code, detail
        super().__init__(f'{code}: {detail}')


def _read(path):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError, UnicodeError):
        return None
    return value if isinstance(value, dict) else None


def _job_events(status, invocation):
    """一次调用在旧相位文件里的样子 → Job 事件。`result` 只能按证据能支撑的程度写。"""
    tool = status.get('tool')
    operation = TOOLS.get(tool)
    if operation is None:
        raise MigrationRefused('unattributable_status',
                               f'{invocation}: 相位文件里没有可归属的 tool（{tool!r}）')
    phase = str(status.get('phase') or '')
    fields = dict(job_id=invocation, invocation=invocation, operation=operation, tool=tool)
    if phase in LIVE:
        # 活着的调用只搬「开始了」：它没结束，就不写终态。
        return [('job_queued', dict(fields, **{k: status[k] for k in
                                               ('pid', 'pid_identity', 'started_at') if k in status})),
                ('job_started', dict(fields))]
    if phase not in TERMINAL:
        raise MigrationRefused('unknown_phase', f'{invocation}: 不认识的相位 {phase!r}')
    outcome = TERMINAL[phase]
    stopped = str(status.get('stopped_reason') or '')
    if outcome == 'completed':
        result, reason = 'received', ''
    elif outcome == 'cancelled':
        has_receipt = (invocation and (Path(status.get('_dir', ''))/'result.json').is_file())
        result, reason = ('received', '') if has_receipt else ('unknown', 'stopped_before_receipt')
    elif stopped == 'worker_failed':
        result, reason = 'unknown', 'receipt_absent'
    else:
        result, reason = 'received', ''
    finished = dict(fields, outcome=outcome, result=result, stopped_reason=stopped)
    if reason:
        finished['result_reason'] = reason
    for key in ('cleanup', 'finished_at', 'worker_pid'):
        if key in status and status[key] is not None:
            finished[key] = status[key]
    return [('job_queued', dict(fields, **{k: status[k] for k in
                                           ('pid', 'pid_identity', 'started_at') if k in status})),
            ('job_started', dict(fields)),
            ('job_finished', finished)]


def _task_events(status, invocation):
    """整段运行（autonomous）的旧相位 → 任务事件。"""
    phase = str(status.get('phase') or '')
    activated = dict(invocation=invocation, pid=status.get('pid'),
                     pid_identity=status.get('pid_identity'), started_at=status.get('started_at'),
                     backend=status.get('backend'), max_rounds=status.get('max_rounds'),
                     task_budget=status.get('task_budget'), budget=status.get('budget'),
                     corpus=status.get('corpus'))
    activated = {key: value for key, value in activated.items() if value is not None}
    if phase in LIVE or phase == 'lost':
        # `lost` 在旧相位里是「进程没了、终态没写」。搬过来就只剩「激活过」这一条，
        # 读者用同一条判据（相位 active + owner 不可达）重新得出 lost —— 不写一条假的终态。
        return [('task_activated', activated)]
    if phase not in TERMINAL:
        raise MigrationRefused('unknown_phase', f'不认识的相位 {phase!r}')
    closed = dict(invocation=invocation, outcome=TERMINAL[phase],
                  reason=str(status.get('stopped_reason') or ''))
    for key in ('error', 'detail', 'cleanup', 'budget', 'finished_at', 'duration_s',
                'generation_evidence', 'coverage'):
        if key in status and status[key] is not None:
            closed[key] = status[key]
    return [('task_activated', activated), ('task_closed', closed)]


def migrate(run_dir, *, dry_run=False):
    """把 `run_dir` 的旧相位文件搬进账本。返回回执；拒绝时抛 `MigrationRefused`。"""
    run = Path(run_dir).resolve()
    if run.is_symlink() or not run.is_dir():
        raise MigrationRefused('not_migratable', '需要一个普通目录（不是软链）')
    if (run/journal.JOURNAL).exists():
        raise MigrationRefused('already_migrated', '这个目录已经有任务账本，迁移不改写历史')
    declaration, contract = _read(run/'declaration.json'), _read(run/'contract.json')
    if not declaration or not contract or not contract.get('hash'):
        # 没有契约就没有任务身份：从一个孤零零的相位文件反推「这是哪个任务」是猜。
        raise MigrationRefused(
            'not_migratable', '缺少 declaration.json / contract.json，无法把这个目录归属到任何任务')
    root_status = _read(run/STATUS)
    calls = sorted(p for p in (run/'tool-calls').glob('*') if p.is_dir())
    statuses = [(directory, _read(directory/STATUS)) for directory in calls]
    if root_status is None and not any(item is not None for _, item in statuses):
        raise MigrationRefused('not_migratable', '旧相位文件一个都没有：没有可搬的事实')

    recorded = _read(run/'run.json') or {}
    plan = [('task_prepared', dict(task_id=str(recorded.get('task_id') or '').strip(),
                                   contract_hash=contract['hash']))]
    closure = []
    if root_status is not None:
        invocation = str(root_status.get('invocation_id') or '')
        kind = 'tool_call' if root_status.get('kind') == 'tool_call' else 'autonomous'
        if kind == 'tool_call':
            raise MigrationRefused('unattributable_status',
                                   '运行根的相位文件写着 tool_call，但没有调用目录可归属')
        for event_kind, fields in _task_events(root_status, invocation):
            # 封账是**最后**一条：它之后不再接受任何事件，而那次运行里发生的调用
            # 都在它之前。顺序错了迁移就会被自己的流拒绝（这条拒绝是对的）。
            (closure if event_kind == 'task_closed' else plan).append((event_kind, fields))
    pending = [(directory, status) for directory, status in statuses if status is not None]
    if pending and root_status is None:
        # 有调用却没有运行根相位（host workflow 的目录就是这样）：任务确实被激活过。
        # 这一条不是补造 —— 一个 Job 跑起来了，任务就是 active。
        plan.append(('task_activated', dict(invocation=str(
            pending[0][1].get('invocation_id') or pending[0][0].name))))
    for directory, status in pending:
        invocation = str(status.get('invocation_id') or directory.name)
        if invocation != directory.name:
            raise MigrationRefused('unattributable_status',
                                   f'{directory.name}: 相位文件里的 invocation 与目录名不符')
        status['_dir'] = str(directory)
        plan.extend(_job_events(status, invocation))

    plan.extend(closure)
    if not dry_run:
        for kind, fields in plan:
            journal.append(run, kind, **fields)
    return dict(ok=True, run_dir=str(run), dry_run=bool(dry_run),
                events=[kind for kind, _ in plan],
                event_count=len(plan),
                calls_migrated=[directory.name for directory, status in statuses
                                if status is not None],
                task_migrated=root_status is not None,
                legacy_files_kept=[STATUS] if root_status is not None else [],
                scope='只搬旧相位文件里已经发生过的事实；没有账本之前的东西不补')