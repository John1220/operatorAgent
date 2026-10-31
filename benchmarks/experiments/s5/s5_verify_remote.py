"""Supervisor-side S5 verification service over the production CLI backends.

All three comparison arms submit their final answer text here. The service owns
the task directory: it prepares the declaration once from the frozen plan, writes
the candidate itself, and only then runs the production ``operator_agent verify``
path (mock or SSH). The verdict is recorded after the run, never fed back into the
same arm, and anything the evidence cannot prove stays unknown.

This is not a hostile-code sandbox, a GPU lease, or a claim that remote cleanup is
complete. The SSH path is the production ``RemoteExecutor``; the mock path is the
trusted local fixture evaluator already used by the S5 prototypes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from benchmarks.experiments.s5 import s5_runtime
from operator_agent.app.hitl_tools import prepare
from operator_agent.core.run_contract import code_hash, digest

SCHEMA = 's5-remote-verification-v1'
PLAN_SCHEMA = 's5-remote-verification-plan-v1'
STOP_AFTER = ('L1', 'L2')
BASE_BACKENDS = {'mock': ('mock', None), 'h20': ('ssh', 'algorithm-h20')}
HASH = re.compile(r'[0-9a-f]{64}\Z')
HOST = re.compile(r'[A-Za-z0-9_.-]{1,64}\Z')
MAX_PAYLOAD_BYTES = 4 * 1024 * 1024


class VerificationRefused(RuntimeError):
    """Submission rejected before any verification ran."""


def _write_once(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


def _local_identity():
    """Hash the service code and the production CLI entry it dispatches."""
    root = Path(__file__).resolve().parents[3]
    files = [root / 'benchmarks' / 'experiments/s5/s5_verify_remote.py',
             root / 'operator_agent' / 'app/cli.py',
             root / 'operator_agent' / 'app/hitl_tools.py',
             root / 'operator_agent' / 'execute/remote_executor.py',
             root / 'remote_scripts' / 'run_kernel.py',
             root / 'remote_scripts' / 'runner_server.py',
             root / 'remote_scripts' / 'session_control.py']
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files if path.is_file()}


def freeze(output, *, declaration, backend='mock', host=None, timeout=300,
           gpu_index=None, gpu_uuid=None, gpu_name=None):
    """Prepare the task once and persist a non-overwritable service plan."""
    if not isinstance(declaration, dict):
        raise ValueError('declaration object required')
    if backend not in BASE_BACKENDS:
        raise ValueError('unknown verification backend')
    mode, default_host = BASE_BACKENDS[backend]
    host = host or default_host
    if mode == 'ssh' and (not isinstance(host, str) or not HOST.fullmatch(host)):
        raise ValueError('ssh backend needs a host alias')
    if type(timeout) is not int or not 5 <= timeout <= 1800:
        raise ValueError('invalid timeout')
    if gpu_index is not None and (type(gpu_index) is not int or not 0 <= gpu_index <= 63):
        raise ValueError('invalid gpu index')
    if gpu_index is not None and mode != 'ssh':
        raise ValueError('a GPU pin requires the ssh backend')
    if gpu_index is not None:
        # The production pinning path refuses a bare index: it compares the
        # index, UUID and name so a renumbered card cannot silently mis-pin.
        if (not isinstance(gpu_uuid, str) or not re.fullmatch(r'GPU-[0-9a-f-]{8,}', gpu_uuid)
                or not isinstance(gpu_name, str) or not gpu_name):
            raise ValueError('a GPU pin needs index, UUID and name')
    elif gpu_uuid is not None or gpu_name is not None:
        raise ValueError('GPU UUID or name without an index')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    prepared = prepare(declaration, root / 'task')
    plan = dict(schema=PLAN_SCHEMA, backend=backend, mode=mode, host=host,
                stop_after='L2', timeout=timeout, gpu_index=gpu_index,
                gpu_uuid=gpu_uuid, gpu_name=gpu_name,
                declaration=declaration,
                contract_hash=prepared['contract_hash'],
                case_id=declaration.get('request'), local_sha256=_local_identity())
    target = root / 'plan.json'
    _write_once(target, plan)
    return hashlib.sha256(target.read_bytes()).hexdigest()


def load_plan(plan_path, expected_sha256):
    path = Path(plan_path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('plan missing')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('plan hash mismatch')
    plan = json.loads(raw)
    if (not isinstance(plan, dict) or plan.get('schema') != PLAN_SCHEMA
            or plan.get('stop_after') != 'L2' or plan.get('backend') not in BASE_BACKENDS
            or plan.get('local_sha256') != _local_identity()
            or not isinstance(plan.get('contract_hash'), str)
            or not HASH.fullmatch(plan['contract_hash'])):
        raise ValueError('plan configuration or runtime identity mismatch')
    return plan


def _command(plan, task_dir, candidate):
    command = [sys.executable, '-m', 'operator_agent.app.cli', 'verify',
               '--run-dir', str(task_dir), '--code-file', str(candidate),
               '--stop-after', plan['stop_after'], '--backend', plan['mode'],
               '--timeout', str(plan['timeout'])]
    if plan['mode'] == 'ssh':
        command += ['--host', plan['host']]
    return command


def _hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _evidence(plan, task_dir, evidence_path):
    """Validate the production evidence file and its completion receipt."""
    problems = []
    evidence_file = Path(evidence_path)
    if evidence_file.is_symlink() or not evidence_file.is_file():
        return None, ['evidence file missing']
    rounds = sorted(path for path in task_dir.glob('verify-*') if path.is_dir())
    if len(rounds) != 1:
        return None, ['expected exactly one verification round']
    round_dir = rounds[0]
    evidence = json.loads(evidence_file.read_text(encoding='utf-8'))
    receipt_file = round_dir / 'complete.json'
    if not receipt_file.is_file():
        return None, ['completion receipt missing']
    receipt = json.loads(receipt_file.read_text(encoding='utf-8'))
    binding = evidence.get('binding') if isinstance(evidence.get('binding'), dict) else {}
    if evidence.get('schema') != 'operator-tool-evidence-v1' or evidence.get('evidence_complete') is not True:
        problems.append('evidence incomplete')
    if evidence.get('implementation_stable') is not True:
        problems.append('implementation not stable')
    if evidence.get('contract_hash') != plan['contract_hash'] or binding.get('contract_hash') != plan['contract_hash']:
        problems.append('contract identity mismatch')
    resolution = evidence.get('backend_resolution')
    if (not isinstance(resolution, dict) or resolution.get('degraded') is not False
            or resolution.get('requested') != plan['mode']):
        problems.append('backend degraded or misreported')
    files = receipt.get('files_sha256') if isinstance(receipt, dict) else None
    if (receipt.get('schema') != 'operator-tool-evidence-complete-v1'
            or receipt.get('evidence_complete') is not True or receipt.get('binding') != binding):
        problems.append('receipt identity mismatch')
    if not isinstance(files, dict) or not {'candidate.py', 'result.json', 'started.json'} <= set(files):
        problems.append('receipt file list incomplete')
    else:
        for name, checksum in files.items():
            rel = Path(name)
            target = round_dir / rel
            if (rel.is_absolute() or '..' in rel.parts or target.is_symlink()
                    or not target.is_file() or _hash_file(target) != checksum):
                problems.append('receipt member %s mismatch' % name)
                break
    l2 = next((item for item in evidence.get('verdicts') or []
               if isinstance(item, dict) and item.get('level') == 'L2'), None)
    metrics = (l2 or {}).get('metrics') if isinstance(l2, dict) else None
    if l2 is None and _l1_rejection_is_decisive(evidence):
        pass  # Syntax/compile rejection legitimately has no L2 metrics.
    elif not isinstance(metrics, dict):
        problems.append('L2 metrics missing')
    else:
        correctness = metrics.get('correctness')
        cases = correctness.get('cases') if isinstance(correctness, dict) else None
        if l2.get('passed') is True:
            # A pass needs the whole suite and the input identity behind it.
            if (not isinstance(correctness, dict) or correctness.get('verdict') != 'passed'
                    or not isinstance(cases, list) or not cases
                    or any(not isinstance(case, dict) or case.get('verdict') != 'passed'
                           for case in cases)):
                problems.append('complete correctness suite not proven')
            if metrics.get('num_input_sets') != 1:
                problems.append('input set count is not one')
            fidelity = metrics.get('input_fidelity_readings')
            if (not isinstance(fidelity, list) or not fidelity
                    or any(not isinstance(item, dict)
                           or item.get('input_fidelity_verified') is not True
                           for item in fidelity)):
                problems.append('input fidelity not proven')
        elif l2.get('passed') is False:
            # A rejection is decisive only when the verifier says the suite failed.
            if not isinstance(correctness, dict) or correctness.get('verdict') not in ('failed',):
                problems.append('rejection is not attributable to the correctness gate')
    if problems:
        return None, problems
    return dict(evidence_sha256=_hash_file(evidence_file),
                receipt_sha256=_hash_file(receipt_file),
                tool_version=binding.get('tool_version'),
                backend_resolution=resolution), []


def submit(plan, *, arm, candidate_text, output_dir):
    """Record the candidate first, then verify it once through the CLI."""
    if not isinstance(plan, dict) or plan.get('schema') != PLAN_SCHEMA:
        raise VerificationRefused('service plan required')
    if arm not in ('A', 'B', 'C'):
        raise VerificationRefused('unknown arm')
    if not isinstance(candidate_text, str) or not candidate_text.strip():
        raise VerificationRefused('empty candidate')
    if len(candidate_text.encode('utf-8')) > 65536:
        raise VerificationRefused('candidate too large')
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (output_dir / 'submission.json').exists() or (output_dir / 'result.json').exists():
        raise VerificationRefused('verification already recorded for this arm')
    task_dir = output_dir / 'task'
    if task_dir.exists():
        raise VerificationRefused('service task directory already exists')
    source_sha256 = code_hash(candidate_text)
    _write_once(output_dir / 'submission.json', dict(
        schema=SCHEMA + '-submission', arm=arm, candidate_sha256=source_sha256,
        candidate_bytes=len(candidate_text.encode('utf-8')),
        contract_hash=plan['contract_hash'], backend=plan['backend'],
        stop_after=plan['stop_after'], host=plan['host']))
    task_dir.mkdir(mode=0o700)
    prepared = prepare(plan['declaration'], task_dir)
    if prepared['contract_hash'] != plan['contract_hash']:
        return _finish(output_dir, dict(status='unknown', full_l2_passed=None,
                                        reason='prepared contract drifted from the plan',
                                        source_sha256=source_sha256, backend=plan['backend']), plan)
    candidate = task_dir / 'candidate.py'
    with candidate.open('x', encoding='utf-8') as handle:
        handle.write(candidate_text)
    started = time.monotonic()
    try:
        # The supervisor, not the tested agent, holds the SSH configuration; the
        # mock path keeps an isolated HOME so it cannot read the user's profile.
        home = (os.environ.get('HOME', str(output_dir)) if plan['mode'] == 'ssh'
                else str(output_dir))
        env = {'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': home,
               'PYTHONPATH': s5_runtime.python_path(), 'PYTHONNOUSERSITE': '1'}
        if plan.get('gpu_index') is not None:
            env['OPERATOR_AGENT_REQUIRED_GPU'] = str(plan['gpu_index'])
            env['OPERATOR_AGENT_REQUIRED_GPU_UUID'] = plan['gpu_uuid']
            env['OPERATOR_AGENT_REQUIRED_GPU_NAME'] = plan['gpu_name']
        proc = subprocess.run(_command(plan, task_dir, candidate), capture_output=True,
                              text=True, timeout=plan['timeout'], env=env,
                              cwd=str(Path(__file__).resolve().parents[3]))
        stdout, stderr = proc.stdout, proc.stderr
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        return _finish(output_dir, dict(status='unknown', full_l2_passed=None,
                                        reason='verification timed out; remote cleanup unknown',
                                        source_sha256=source_sha256, backend=plan['backend'],
                                        timed_out=True, wall_seconds=round(time.monotonic() - started, 3)), plan)
    wall = round(time.monotonic() - started, 3)
    if len(stdout.encode()) + len(stderr.encode()) > MAX_PAYLOAD_BYTES:
        return _finish(output_dir, dict(status='unknown', full_l2_passed=None,
                                        reason='verification output exceeded the service bound',
                                        source_sha256=source_sha256, backend=plan['backend'],
                                        wall_seconds=wall), plan)
    try:
        payload = json.loads(stdout)
    except ValueError:
        return _finish(output_dir, dict(status='unknown', full_l2_passed=None,
                                        reason='verification output was not JSON (exit %r)' % returncode,
                                        source_sha256=source_sha256, backend=plan['backend'],
                                        wall_seconds=wall), plan)
    evidence, problems = _evidence(plan, task_dir, payload.get('evidence', ''))
    levels = payload.get('levels') if isinstance(payload.get('levels'), dict) else {}
    identity_ok = (payload.get('contract_hash') == plan['contract_hash']
                   and payload.get('source_hash') == source_sha256
                   and payload.get('stop_after') == plan['stop_after'])
    resolution = payload.get('backend_resolution') if isinstance(payload.get('backend_resolution'), dict) else {}
    degraded = resolution.get('degraded') is not False
    l1, l2 = levels.get('L1'), levels.get('L2')
    if problems or not identity_ok or degraded or evidence is None:
        status, passed = 'unknown', None
        reason = '; '.join(problems) if problems else (
            'verification identity, backend or evidence incomplete')
    elif payload.get('ok') is True and l1 is True and l2 is True:
        status, passed, reason = 'passed', True, 'production CLI reported full L2 pass'
    elif l1 is True and l2 is False:
        status, passed, reason = 'failed', False, 'production CLI rejected the candidate at L2'
    elif _l1_rejection_is_decisive(payload):
        status, passed, reason = 'failed', False, 'production CLI rejected the candidate at L1'
    else:
        status, passed, reason = 'unknown', None, 'verification outcome not decisive'
    return _finish(output_dir, dict(
        status=status, full_l2_passed=passed, reason=reason, source_sha256=source_sha256,
        backend=plan['backend'], levels=dict(L1=l1, L2=l2),
        backend_resolution=resolution, evidence=evidence,
        payload_sha256=digest(payload), wall_seconds=wall), plan)


def _l1_rejection_is_decisive(payload):
    """A syntax pre-check or compile error is a candidate defect, not a transport gap.

    The verifier reports compile failures inside L1 metrics; an L1 failure with no
    compile status at all is its local syntax pre-check. Anything else (guard,
    resource, transport) stays unknown because it may not be the candidate's fault.
    """
    verdicts = payload.get('verdicts')
    if not isinstance(verdicts, list) or len(verdicts) != 1:
        return False
    only = verdicts[0]
    if not isinstance(only, dict) or only.get('level') != 'L1' or only.get('passed') is not False:
        return False
    metrics = only.get('metrics') if isinstance(only.get('metrics'), dict) else {}
    if not metrics:
        return True
    status = metrics.get('compile_status')
    if status is None and 'static_guards' not in metrics:
        return True
    return status in ('compile_error', 'compile_failed', 'failed', 'error', 'syntax_error')


def _finish(output_dir, result, plan=None):
    body = dict(schema=SCHEMA, contract_hash=(plan or {}).get('contract_hash'),
                service_plan_sha256=None, formal_s5_credit=False,
                mock_only=result.get('backend') == 'mock',
                remote_backend_executed=result.get('backend') == 'h20',
                isolation_verified=False, gpu_lease=False)
    body.update(result)
    _write_once(Path(output_dir) / 'result.json', body)
    return body


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--arm', required=True, choices=('A', 'B', 'C'))
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    plan = load_plan(args.plan, args.plan_sha256)
    result = submit(plan, arm=args.arm,
                    candidate_text=args.candidate.read_text(encoding='utf-8'),
                    output_dir=args.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
