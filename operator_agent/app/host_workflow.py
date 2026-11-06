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


def enabled():
    mode = os.environ.get('OPERATOR_AGENT_WORKFLOW', 'legacy')
    if mode not in ('host', 'legacy'):
        from .hitl_tools import ToolError
        raise ToolError('invalid_workflow', 'OPERATOR_AGENT_WORKFLOW must be host or legacy.')
    return mode == 'host'


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
    run, task, contract = prepared_task(run_dir)
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
