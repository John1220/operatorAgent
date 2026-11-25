"""Host-owned generation: publish domain facts without making model requests."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
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


def _profile_evidence(run, verification_round, source_hash):
    from .hitl_tools import ToolError
    from ..execute.profile_budget import ProfileBudget, ProfileBudgetError
    try:
        ledger = ProfileBudget(run).snapshot()
    except ProfileBudgetError as error:
        raise ToolError(error.code, error.detail) from error
    evidence = []
    for entry in ledger['entries']:
        if entry['status'] != 'completed' or not entry.get('receipt'):
            continue
        receipt_path = Path(entry['receipt'])
        if receipt_path.parent.parent != run or receipt_path.name != 'receipt.json':
            raise ToolError('profile_evidence_refused', 'profile receipt is outside the owned run')
        try:
            receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
            complete = json.loads((receipt_path.parent/'complete.json').read_text(encoding='utf-8'))
            if (receipt.get('schema') != 'operator-profile-v1'
                    or complete.get('schema') != 'operator-profile-complete-v1'
                    or complete.get('evidence_complete') is not True
                    or complete.get('binding', {}).get('profile_id') != receipt.get('profile_id')):
                raise ValueError('profile evidence envelope differs')
            for name, expected in complete['files_sha256'].items():
                path = receipt_path.parent/name
                if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise ValueError('profile evidence member drift')
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ToolError('profile_evidence_refused', 'completed profile evidence is unreadable') from error
        if (receipt.get('verification_round') != verification_round
                or receipt.get('source_hash') != source_hash):
            continue
        evidence.append(dict(
            profile_id=receipt['profile_id'], verification_round=verification_round,
            source_hash=source_hash, gpu=receipt.get('gpu'), totals=receipt.get('totals'),
            launches=receipt.get('launches'), range_ok=receipt.get('range_ok'),
            decoy_seen=receipt.get('decoy_seen'), reexport=receipt.get('reexport'),
            cache_conditions=receipt.get('cache_conditions'), ncu=receipt.get('ncu'),
            wall_s=receipt.get('wall_s'), evidence=str(receipt_path.parent/'complete.json')))
    return evidence


def _experiment_history(run):
    records = []
    for path in sorted((run/'contexts').glob('*.json')) if (run/'contexts').is_dir() else []:
        try:
            if path.is_symlink() or path.stat().st_size > 4 * 1024 * 1024:
                continue
            payload = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError, UnicodeError):
            continue
        experiment = (payload.get('optimization') or {}).get('experiment')
        if isinstance(experiment, dict) and experiment.get('schema') == 'operator-ncu-experiment-v1':
            records.append(experiment)
    return records[-32:]


def context(run_dir, *, phase='generate', backend='ssh', verification_round=None,
            hypothesis=None, expected_metric_change=None):
    from .hitl_tools import ToolError
    from ..generate.host_context import build_host_context
    from ..execute.host_task_budget import HostTaskBudget, host_task_enabled
    run, task, contract = prepared_task(run_dir)
    if phase != 'optimize' and (hypothesis is not None or expected_metric_change is not None):
        raise ToolError('unexpected_experiment', 'experiment hypotheses are only valid for optimize context')
    if phase == 'optimize':
        if (hypothesis is None) != (expected_metric_change is None):
            raise ToolError('experiment_incomplete', 'provide both hypothesis and expected_metric_change')
        for name, value in (('hypothesis', hypothesis), ('expected_metric_change', expected_metric_change)):
            if value is not None and (not isinstance(value, str) or not value.strip() or len(value) > 1000):
                raise ToolError('experiment_invalid', name+' must be non-empty text up to 1000 characters')
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
    if phase == 'optimize':
        profiles = _profile_evidence(run, verification_round, code_hash(source))
        record_id = uuid.uuid4().hex
        materials['optimization']['profile_evidence'] = profiles
        materials['optimization']['experiment_history'] = _experiment_history(run)
        materials['optimization']['experiment'] = dict(
            schema='operator-ncu-experiment-v1', id=record_id, created_at=time.time(),
            source_hash=code_hash(source), verification_round=verification_round,
            bottleneck_hypothesis=hypothesis, expected_metric_change=expected_metric_change,
            observed_result=profiles, status='observed' if profiles else 'planned',
            next_action='host_review_then_independent_l3')
        materials['optimization']['experiment_protocol'] = dict(
            required_fields=['source_hash', 'bottleneck_hypothesis', 'expected_metric_change',
                             'observed_result', 'next_action'],
            rule='Use the bound NCU facts as observations, not verdicts; state a falsifiable '
                 'next hypothesis and confirm it with independent L3.')
    else:
        record_id = uuid.uuid4().hex
    directory = run/'contexts'
    directory.mkdir(exist_ok=True)
    evidence = directory/(record_id+'.json')
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
    # 旧任务的账本在**它当年那个状态文件**里（账本之前没有 journal）。这是一次只读的
    # 历史回读，不是第二个权威：新任务的记录不在这里。
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


def _seed_attempt_is_infra_hold(record):
    """True when a seed attempt ended in an infra hold, not a numeric L2 failure.

    The remote reference-identity handshake can end with a transport outcome of
    ``unknown`` (cold environment, SSH hiccup). The host task still records
    ``full_l2_passed=False``; retrying the seed once within the same budget is
    safe, and must not be reported as ``seed_not_correct_for_contract``.
    """
    evidence = record.get('evidence')
    if not evidence:
        return False
    try:
        payload = json.loads(Path(evidence).read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError):
        return False
    if not isinstance(payload, dict):
        return False
    route = payload.get('route')
    if isinstance(route, dict) and route.get('route') == 'hold' and (
            route.get('kind') == 'infra' or route.get('counts_as_iteration') is False):
        return True
    for verdict in payload.get('verdicts') or ():
        metrics = verdict.get('metrics') if isinstance(verdict, dict) else None
        if not isinstance(metrics, dict):
            continue
        if (metrics.get('reference_identity_unknown') is True
                or metrics.get('transport_outcome') == 'outcome_unknown'):
            return True
    return False


def resume(run_dir, *, backend='ssh', verification_round=None):
    """Read the existing task without resetting policy, time or reservations.

    Conversation/user instructions remain host-owned. This tool only returns
    frozen domain facts and source-bound observations for the host to combine
    with its current conversation. It never reaps an unknown owner or opens a
    new task, and model-session attachment is left to the DSH adapter.
    """
    from .hitl_tools import ToolError
    from ..execute.host_task_budget import (HostTaskBudget, _verification_round,
                                            host_task_enabled)
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
    best = budget.get('best_candidate')
    final_record = next((record for record in budget['records']
                         if record.get('purpose') == 'final'
                         and record.get('status') == 'completed'), None)
    if verification_round is not None:
        if type(verification_round) is not int or verification_round not in completed:
            raise ToolError('verification_required', 'Resume needs a completed verification_round.')
        selected = verification_round
    elif final_record is not None and type(_verification_round(final_record)) is int:
        selected = _verification_round(final_record)
    elif best is not None and type(best.get('round')) is int:
        selected = best['round']
    elif budget['can_reserve_final'] and not budget['can_reserve_exploration']:
        recommended = budget.get('best_candidate') or budget['last_correct_candidate']
        evidence = Path(recommended['evidence'])
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
    seed = budget.get('seed')
    baseline = budget.get('baseline')
    baseline_attempts = [r for r in budget['records'] if r.get('role') == 'baseline']
    stop_reason = None
    next_context = None
    if budget['terminal']:
        next_action = 'report_final' if budget['phase']=='final_complete' else 'report_stopped'
        stop_reason = budget.get('stop_reason')
    elif budget['unfinished']:
        next_action = 'inspect_in_flight'
    elif model_unfinished:
        next_action = 'inspect_model_in_flight'
    elif seed is not None and baseline is None:
        last_attempt = baseline_attempts[-1] if baseline_attempts else None
        infra_holds = sum(
            1 for record in baseline_attempts
            if record.get('status') == 'completed'
            and record.get('full_l2_passed') is not True
            and _seed_attempt_is_infra_hold(record))
        if last_attempt is None:
            next_action = 'verify_seed'
        elif (last_attempt.get('status') == 'completed'
              and last_attempt.get('full_l2_passed') is not True
              and _seed_attempt_is_infra_hold(last_attempt)
              and infra_holds <= 1
              and budget.get('can_reserve_exploration')
              and budget.get('remaining_verifications', 0) > 0
              and not budget.get('expired')):
            next_action = 'verify_seed'
        elif last_attempt.get('status') == 'completed' and last_attempt.get('full_l2_passed') is not True:
            next_action = 'report_stopped'
            stop_reason = 'seed_not_correct_for_contract'
        elif last_attempt.get('status') == 'completed':
            if budget['can_reserve_exploration']:
                next_action = 'verify_seed'
            else:
                next_action = 'report_stopped'
                stop_reason = 'baseline_unavailable'
        else:
            next_action = 'verify_seed'
        if next_action == 'verify_seed':
            next_context = dict(run_dir=str(run), backend=backend, phase='verify_seed',
                                seed_source=(seed.get('source') or ''))
    elif seed is not None and baseline is not None and budget['can_reserve_exploration']:
        next_action = 'optimize'
    elif accepted_source and budget['can_reserve_final']:
        next_action = 'final_verify'
    elif not budget['can_reserve_exploration']:
        next_action = 'select_accepted_candidate' if budget['can_reserve_final'] else 'report_stopped'
        stop_reason = stop_reason or budget.get('stop_reason')
    else:
        next_action = ('verify_candidate' if verification is not None and verification.get('levels',{}).get('L2') is True
                       else 'repair' if verification is not None else 'generate')
    if next_context is None and next_action in ('generate','repair','optimize'):
        next_context = dict(run_dir=str(run),backend=backend,
                            phase=('optimize' if next_action=='optimize' else 'repair' if next_action=='repair' else 'generate'))
        if next_action in ('repair','optimize'):
            next_context['verification_round']=selected
    compact_budget = {key:value for key,value in budget.items() if key!='records'}
    compact_budget['records_evidence']=str(run/'journal.jsonl')
    unfinished = [dict(id=record['id'],owner_invocation_id=record.get('owner_invocation_id'),
                       source_hash=record['source_hash'],purpose=record['purpose'])
                  for record in budget['records'] if record['status']=='reserved']
    return dict(ok=True, workflow='host', resumed=True, run_dir=str(run),
        task_id=budget['task_id'], contract_hash=contract.hash, phase=budget['phase'],
        stop_reason=stop_reason or budget['stop_reason'], host_task=compact_budget,
        latest_completed_verification=completed[-1] if completed else None,
        selected_verification=selected, incomplete_verifications=sorted(incomplete),
        candidate=candidate, last_correct_candidate=budget['last_correct_candidate'],
        seed=seed, baseline=baseline, best_candidate=budget.get('best_candidate'),
        final_selection=budget.get('final_selection'),
        candidates=budget.get('candidates'), noise_band=budget.get('noise_band'),
        selection=budget.get('selection'), formal_delivery=budget.get('formal_delivery'),
        next_context=next_context, next_action=next_action, model_calls=0, context_owner='host',
        unfinished_model_calls=model_unfinished,
        unfinished_owners=unfinished[:16],unfinished_owners_truncated=len(unfinished)>16,
        resume_scope='Read existing task only; bind the DSH model session separately. '
                     'Preserve frozen contract, limits, deadline and all reservations.',
        user_instruction_owner='host_conversation_not_frozen_task_mutation')
