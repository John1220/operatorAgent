"""Host-owned generation: publish domain facts without making model requests."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import uuid
from contextlib import contextmanager

from ..core.contract import freeze_from_task
from ..core.run_contract import code_hash, digest
from ..core.run_layout import write_once_artifact


#: 默认工作流。**仍然是 legacy**：正式路径靠捆包 `cordis.patch.yml` 设 `host`。
#:
#: 为什么没翻（2026-11-23 复核）：翻默认值的代价远大于「四个环境变量」的估算。
#: 实测把默认改成 `host` 后，会挂的不止那些驱动 legacy 工具的用例 —— 还有一批
#: 走无界 `operator_verify` 的用例（host 模式要求先 `prepare` 绑定 host task），
#: 以及用 `setUpClass` 提前抓取工具面的类（函数级 fixture 到得太晚）。总影响面
#: ≈30 个用例 / 9 个文件，且其中几处的修法互不相同。
#:
#: 想翻的时候，正确做法是一次专门的迁移：给每个用例显式声明它测的是哪条路径
#: （`conftest.py` 已有 `legacy_workflow` / `host_workflow` 两个 fixture，
#: 部分文件也已经声明了），并且单独验证进程类用例（worker 子进程会继承环境变量）。
DEFAULT_WORKFLOW = 'legacy'

WORKFLOWS = ('host', 'legacy')


def mode():
    """当前的 workflow 名。非法值当场拒绝，不静默降级。"""
    value = os.environ.get('OPERATOR_AGENT_WORKFLOW', DEFAULT_WORKFLOW)
    if value not in WORKFLOWS:
        from .hitl_tools import ToolError
        raise ToolError('invalid_workflow',
                        f'OPERATOR_AGENT_WORKFLOW 只能是 {WORKFLOWS}，实际 {value!r}')
    return value


def enabled():
    return mode() == 'host'


@contextmanager
def workflow_mode(name):
    """在**限定作用域内**切到某个 workflow，退出时恢复。

    为什么不是直接改全局 env：pytest 是一个进程跑完的，`os.environ[...] = 'legacy'`
    会串到同一会话里后面所有用例上 —— 那正是本仓一直在防的「跨用例串味」。
    legacy 的 harness 只需要在**它自己那次分派期间**是 legacy。
    """
    if name not in WORKFLOWS:
        raise ValueError(f'workflow 只能是 {WORKFLOWS}，实际 {name!r}')
    previous = os.environ.get('OPERATOR_AGENT_WORKFLOW')
    os.environ['OPERATOR_AGENT_WORKFLOW'] = name
    try:
        yield name
    finally:
        if previous is None:
            os.environ.pop('OPERATOR_AGENT_WORKFLOW', None)
        else:
            os.environ['OPERATOR_AGENT_WORKFLOW'] = previous


@contextmanager
def prepare_lock(run):
    import fcntl
    run.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(run/'.host-prepare.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def prepared_task(run_dir):
    from .hitl_tools import ToolError, build_task
    run = Path(run_dir).resolve()
    try:
        declaration = json.loads((run/'declaration.json').read_text())
        recorded = json.loads((run/'contract.json').read_text())
    except (OSError, ValueError) as error:
        raise ToolError('run_not_prepared', 'Prepare a valid task directory first.') from error
    task = build_task(declaration)
    contract = freeze_from_task(task)
    if recorded.get('hash') != contract.hash or digest(recorded.get('contract')) != digest(contract.as_dict()):
        raise ToolError('contract_drift', 'The prepared task or frozen contract changed.')
    return run, task, contract


def _verification_materials(run, contract, backend, verification_round):
    from .hitl_tools import ToolError
    directory = run/f'verify-{verification_round}'
    try:
        if directory.is_symlink() or (directory/'complete.json').is_symlink():
            raise ValueError('verification directory or receipt is a symlink')
        complete = json.loads((directory/'complete.json').read_text())
        if (complete.get('schema') != 'operator-tool-evidence-complete-v1'
                or complete.get('evidence_complete') is not True):
            raise ValueError('complete verification receipt required')
        members = complete['files_sha256']
        checked = {}
        for name in ('result.json', 'candidate.py', 'report.md'):
            path = directory/name
            data = path.read_bytes()
            if path.is_symlink() or hashlib.sha256(data).hexdigest() != members[name]:
                raise ValueError('verification member drift')
            checked[name] = data
        verification = json.loads(checked['result.json'])
        source = checked['candidate.py'].decode('utf-8')
        if (verification.get('schema') != 'operator-tool-evidence-v1'
                or verification.get('evidence_complete') is not True
                or verification.get('backend') != backend
                or verification['contract_hash'] != contract.hash
                or verification['source_hash'] != code_hash(source)
                or complete['binding'] != verification['binding']):
            raise ValueError('verification binding drift')
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ToolError('verification_refused', 'Completed source-bound verification is required.') from error
    return source, verification, directory


def context(run_dir, *, phase='generate', backend='ssh', verification_round=None):
    from .hitl_tools import ToolError
    from ..generate.host_context import build_host_context
    from ..execute.host_task_budget import HostTaskBudget, host_task_enabled
    run, task, contract = prepared_task(run_dir)
    source = verification = None
    if phase in ('repair', 'optimize'):
        if type(verification_round) is not int or verification_round < 1:
            raise ToolError('verification_required', 'Select a completed verification_round.')
        source, verification, _ = _verification_materials(run, contract, backend, verification_round)
    elif verification_round is not None:
        raise ToolError('unexpected_verification', 'generate does not take verification_round.')
    try:
        materials = build_host_context(task, contract, backend=backend, phase=phase,
                                       source=source, verification=verification)
    except ValueError as error:
        raise ToolError('context_refused', str(error)) from error
    directory = run/'contexts'
    directory.mkdir(exist_ok=True)
    evidence = directory/(uuid.uuid4().hex+'.json')
    write_once_artifact(evidence, json.dumps(materials, ensure_ascii=False, sort_keys=True).encode())
    budget = HostTaskBudget(run, contract.hash).snapshot() if host_task_enabled(run) else None
    return dict(ok=True, workflow='host', run_dir=str(run), contract_hash=contract.hash,
                context=materials, evidence=str(evidence), host_task=budget,
                model_calls=0, context_owner='host')


def legacy_task_view(run_dir):
    """旧版本任务的**只读**恢复：把已经发生过的事实读出来，不重建任何东西。

    为什么不能重建：v0.9 的 `build_task` 要求显式 `operator` / `precision`（判据层
    不再从自然语言推关键词），而旧任务的 `declaration.json` 里没有这两个字段。用新
    规则去解释旧声明，得到的不是「这个任务」，而是「按今天的规则它应该是什么」——
    那会把一个已有冻结契约、已花预算、已交付源码的任务判成非法。

    所以这里只读，且只读三样东西：**它当初评的是什么（契约）**、**花了多少（两本账）**、
    **交付了哪份源码**。三者各自按自己的身份对齐，不能要求它们一致：

      * 契约按 `contract_hash` 对齐；
      * 两本预算账按**预算自己的** `task_id` 对齐 —— 它与 `run.json.task_id` 是两个
        不同的 ID（A4 实测：`56362c1f` vs `1ed2cb08e5c74b6cbe3ced1550e56b0f`）。
        要求三个 ID 全相等会把这份真实产物误判成损坏。
    """
    from ..core.contract import load_dict
    from ..core.run_contract import digest
    run = Path(run_dir).resolve()
    try:
        recorded_run = json.loads((run/'run.json').read_text(encoding='utf-8'))
        recorded_contract = json.loads((run/'contract.json').read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise ValueError(f'not a readable run directory: {error}') from error

    # 契约：用既有读取器重算，并与运行记录比对。重算不出来就说重算不出来。
    recomputed = None
    try:
        recomputed = load_dict(recorded_contract['contract']).hash
    except (KeyError, TypeError, ValueError):
        recomputed = None
    claimed = recorded_run.get('contract_hash')
    ledger = json.loads((run/'host-task.json').read_text(encoding='utf-8')) if (run/'host-task.json').is_file() else {}
    records = ledger.get('records') or []
    purposes = [row.get('purpose') for row in records]
    exploration = purposes.count('explore')
    final = purposes.count('final')
    model = (json.loads((run/'host-model-budget.json').read_text(encoding='utf-8'))
             if (run/'host-model-budget.json').is_file() else {})
    selection = (json.loads((run/'final-selection.json').read_text(encoding='utf-8'))
                 if (run/'final-selection.json').is_file() else {})
    return {
        'ok': True,
        'read_only': True,
        'run_dir': str(run),
        'why_read_only': ('这个任务是在要求显式声明的版本之前冻结的（声明里没有 operator / '
                          'precision）。本层不按新规则重新解释旧声明，只读出已发生的事实。'),
        'tool_version_recorded': recorded_run.get('tool_version'),
        'contract': {
            'recorded_hash': claimed,
            'recomputed_hash': recomputed,
            'matches_run_record': bool(claimed) and recomputed == claimed,
        },
        'verification_budget': {
            'ledger_task_id': ledger.get('task_id'),
            'exploration_completed': exploration,
            'final_completed': final,
            'records': len(records),
        },
        'model_ledger': {
            'calls': len(model.get('calls') or model.get('records') or []),
            'failed': len([row for row in (model.get('calls') or model.get('records') or [])
                           if isinstance(row, dict)
                           and str(row.get('status') or row.get('outcome') or '').lower()
                           in ('failed', 'error')]),
        },
        'delivery': {
            'source_hash': selection.get('actual_source'),
            'recommended_round': selection.get('recommended_round'),
            'override': selection.get('override'),
            'policy': selection.get('policy'),
        },
        'next': ('这是只读视图：可以查回「当初评了什么、花了多少、交付了哪份源码」。'
                 '要继续推进这个任务，请用产生它的那个版本的 operator-agent；'
                 '不要修改原 declaration.json —— 它是这份冻结契约的记录，改了就没有对照物了。'),
    }


def resume(run_dir, *, backend='ssh', verification_round=None):
    """Read the existing task without resetting policy, time or reservations.

    Conversation/user instructions remain host-owned. This tool only returns
    frozen domain facts and source-bound observations for the host to combine
    with its current conversation. It never reaps an unknown owner or opens a
    new task, and model-session attachment is left to the DSH adapter.
    """
    from .hitl_tools import ToolError
    from ..execute.host_task_budget import HostTaskBudget, host_task_enabled
    if Path(run_dir).is_symlink():
        raise ToolError('host_task_required', 'Resume an ordinary prepared host task directory.')
    try:
        run, task, contract = prepared_task(run_dir)
    except ToolError as error:
        # 旧版本的声明没有 `operator` / `precision`。用它本来就没有的字段去重建，
        # 得到的是「按今天的规则它应该是什么」，不是它。改走只读视图。
        if error.code in ('explicit_declaration_required', 'unknown_parameter'):
            return legacy_task_view(run_dir)
        raise
    if not host_task_enabled(run):
        raise ToolError('host_task_required', 'Resume requires the existing bounded host task.')
    budget = HostTaskBudget(run, contract.hash).snapshot(readonly=True)
    selected = None
    completed = []
    incomplete = []
    for directory in run.glob('verify-*'):
        if not directory.is_dir() or not directory.name[7:].isdigit():
            continue
        number = int(directory.name[7:])
        (completed if (directory/'complete.json').exists() else incomplete).append(number)
    completed.sort()
    if verification_round is not None:
        if type(verification_round) is not int or verification_round not in completed:
            raise ToolError('verification_required', 'Resume needs a completed verification_round.')
        selected = verification_round
    elif budget['can_reserve_final'] and not budget['can_reserve_exploration']:
        # When exploration is spent, return the accepted source eligible for
        # final rather than the last unsuccessful experiment.
        evidence = Path(budget['last_correct_candidate']['evidence'])
        if evidence.parent.parent == run and evidence.parent.name.startswith('verify-'):
            selected = int(evidence.parent.name[7:])
    if selected is None and completed:
        selected = completed[-1]

    source = verification = directory = None
    if selected is not None:
        source, verification, directory = _verification_materials(run, contract, backend, selected)
    candidate = (dict(source=str(directory/'candidate.py'), source_hash=code_hash(source),
        evidence=str(directory/'result.json'), report=str(directory/'report.md'), round=selected)
        if source is not None else None)
    accepted = [record for record in budget['records']
        if record['status']=='completed' and record.get('full_l2_passed') is True
        and record.get('finished_after_deadline') is False
        and record.get('accepted_by_supervisor') is not False]
    accepted_source = source is not None and any(record['source_hash']==code_hash(source) for record in accepted)
    model_calls = budget['model_budget'].get('calls')
    model_unfinished = model_calls.get('unfinished') if isinstance(model_calls,dict) else None
    if budget['terminal']:
        next_action = 'report_final' if budget['phase']=='final_complete' else 'report_stopped'
    elif budget['unfinished']:
        next_action = 'inspect_in_flight'
    elif model_unfinished:
        next_action = 'inspect_model_in_flight'
    elif accepted_source and budget['can_reserve_final']:
        next_action = 'final_verify'
    elif not budget['can_reserve_exploration']:
        next_action = 'select_accepted_candidate' if budget['can_reserve_final'] else 'report_stopped'
    else:
        next_action = ('verify_candidate' if verification is not None and verification.get('levels',{}).get('L2') is True
                       else 'repair' if verification is not None else 'generate')
    next_context = None
    if next_action in ('generate','repair'):
        next_context = dict(run_dir=str(run),backend=backend,
                            phase='repair' if next_action=='repair' else 'generate')
        if next_action=='repair':
            next_context['verification_round']=selected
    compact_budget = {key:value for key,value in budget.items() if key!='records'}
    compact_budget['records_evidence']=str(run/'host-task.json')
    unfinished = [dict(id=record['id'],owner_invocation_id=record.get('owner_invocation_id'),
                       source_hash=record['source_hash'],purpose=record['purpose'])
                  for record in budget['records'] if record['status']=='reserved']
    return dict(ok=True, workflow='host', resumed=True, run_dir=str(run),
        task_id=budget['task_id'], contract_hash=contract.hash, phase=budget['phase'],host_task=compact_budget,
        latest_completed_verification=completed[-1] if completed else None,
        selected_verification=selected, incomplete_verifications=sorted(incomplete),
        candidate=candidate, last_correct_candidate=budget['last_correct_candidate'],
        next_context=next_context, next_action=next_action, model_calls=0, context_owner='host',
        unfinished_model_calls=model_unfinished,
        unfinished_owners=unfinished[:16],unfinished_owners_truncated=len(unfinished)>16,
        resume_scope='Read existing task only; bind the DSH model session separately. '
                     'Preserve frozen contract, limits, deadline and all reservations.',
        user_instruction_owner='host_conversation_not_frozen_task_mutation')
