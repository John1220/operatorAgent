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


def context(run_dir, *, phase='generate', backend='ssh', verification_round=None):
    from .hitl_tools import ToolError
    from ..generate.host_context import build_host_context
    from ..execute.host_task_budget import HostTaskBudget, host_task_enabled
    run, task, contract = prepared_task(run_dir)
    source = verification = None
    if phase in ('repair', 'optimize'):
        if type(verification_round) is not int or verification_round < 1:
            raise ToolError('verification_required', 'Select a completed verification_round.')
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
            for name in ('result.json', 'candidate.py'):
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
