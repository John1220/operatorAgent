"""One-shot production L2/L3 evaluation for the bounded S5 performance study.

freeze() is local only. Actual SSH/GPU work requires execute(..., exact_hash).
No model is invoked. Qualified timing is separate from passing a speed target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from benchmarks.experiments.s5 import s5_runtime
from benchmarks.experiments.s5 import s5_shared_gpu
from benchmarks.experiments.s5.s5_verify_remote import _l1_rejection_is_decisive
from operator_agent.contract import freeze_from_task
from operator_agent.correctness import as_suite
from operator_agent.hitl_tools import build_task, prepare
from operator_agent.run_contract import digest, protocol_identity

PLAN_SCHEMA = 's5-performance-evaluation-plan-v1'
RESULT_SCHEMA = 's5-performance-evaluation-v1'
ROOT = Path(__file__).resolve().parents[3]
HASH = re.compile(r'[0-9a-f]{64}\Z')
MAX_CAPTURE = 4 * 1024 * 1024


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


def implementation_identity():
    paths = [p for folder in ('operator_agent', 'remote_scripts')
             for p in (ROOT / folder).rglob('*')
             if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py', '.md')]
    paths += [Path(__file__), ROOT/'benchmarks/experiments/s5/s5_runtime.py',
              ROOT/'benchmarks/experiments/s5/s5_verify_remote.py', ROOT/'benchmarks/experiments/s5/s5_shared_gpu.py']
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}


def _pin(pin):
    if (not isinstance(pin, dict)
            or set(pin) != {'host', 'gpu_index', 'gpu_uuid', 'gpu_name'}
            or not isinstance(pin['host'], str)
            or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,252}', pin['host'])
            or type(pin['gpu_index']) is not int or not 0 <= pin['gpu_index'] <= 63
            or not isinstance(pin['gpu_uuid'], str)
            or not re.fullmatch(r'GPU-[0-9a-fA-F-]{8,}', pin['gpu_uuid'])
            or not isinstance(pin['gpu_name'], str) or not pin['gpu_name'].strip()
            or any(ord(c) < 32 for c in pin['gpu_name'])):
        raise ValueError('exact SSH/GPU pin required')


def freeze(output_dir, *, declaration, candidate_path, pin, resource_policy, timeout=600,
           stop_after='L3'):
    """Freeze one evaluation; no SSH, model or GPU work is performed."""
    _pin(pin)
    s5_shared_gpu.validate_policy(resource_policy)
    if type(timeout) is not int or not 5 <= timeout <= 1800:
        raise ValueError('timeout must be an integer in 5..1800')
    if stop_after not in ('L2', 'L3'):
        raise ValueError('only full L2 or L3 evaluations are supported')
    if not isinstance(declaration, dict):
        raise ValueError('declaration required')
    # JSON-normalize and reject nonfinite configuration before creating output.
    declaration = json.loads(json.dumps(declaration, allow_nan=False))
    candidate_path = Path(candidate_path)
    if candidate_path.is_symlink() or not candidate_path.is_file():
        raise ValueError('regular candidate file required')
    raw = candidate_path.read_bytes()
    if not raw.strip() or len(raw) > 65536:
        raise ValueError('candidate must contain 1..65536 bytes')
    raw.decode('utf-8')
    task = build_task(declaration)
    contract = freeze_from_task(task)
    suite = as_suite(task.constraints.correctness_suite)
    if suite is None or not suite.cases:
        raise ValueError('a declared full correctness suite is required')
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    (out/'candidate.py').write_bytes(raw)
    write(out/'declaration.json', declaration)
    plan = dict(schema=PLAN_SCHEMA, declaration=declaration, pin=dict(pin),
                timeout=timeout, stop_after=stop_after, persistent=True,
                candidate_sha256=sha(out/'candidate.py'),
                resource_policy=dict(resource_policy),
                contract_hash=contract.hash, suite_sha256=suite.digest(),
                suite_case_ids=[case.case_id for case in suite.cases],
                implementation=implementation_identity(), protocol=protocol_identity(),
                runtime_deps=(s5_runtime.tree_digest(s5_runtime.deps_dir())
                              if s5_runtime.available() else None),
                model_calls=0, max_verifications=1, formal_s5_credit=False,
                candidate_origin='outer',
                scope='same production L2/L3 evaluator; one reading, not independent confirmation')
    plan['hash'] = digest(plan)
    write(out/'plan.json', plan)
    return plan


def load_plan(path, expected_hash):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not isinstance(expected_hash, str) or not HASH.fullmatch(expected_hash):
        raise ValueError('plan and its exact content hash are required')
    p = read(path)
    if (not isinstance(p, dict) or p.get('schema') != PLAN_SCHEMA
            or digest({k: v for k, v in p.items() if k != 'hash'}) != expected_hash
            or p.get('hash') != expected_hash):
        raise ValueError('plan hash mismatch')
    _pin(p.get('pin'))
    s5_shared_gpu.validate_policy(p.get('resource_policy'))
    if (p.get('implementation') != implementation_identity()
            or p.get('protocol') != protocol_identity()):
        raise ValueError('frozen implementation changed')
    if (p.get('model_calls') != 0 or p.get('max_verifications') != 1
            or p.get('persistent') is not True or p.get('stop_after') not in ('L2', 'L3')
            or type(p.get('timeout')) is not int or not 5 <= p['timeout'] <= 1800):
        raise ValueError('evaluation limits changed')
    if sha(path.parent/'candidate.py') != p.get('candidate_sha256'):
        raise ValueError('candidate identity changed')
    if read(path.parent/'declaration.json') != p.get('declaration'):
        raise ValueError('declaration sidecar changed')
    task = build_task(p['declaration'])
    contract = freeze_from_task(task)
    suite = as_suite(task.constraints.correctness_suite)
    if (contract.hash != p['contract_hash'] or suite.digest() != p['suite_sha256']
            or [c.case_id for c in suite.cases] != p['suite_case_ids']):
        raise ValueError('contract or correctness suite changed')
    actual_deps = s5_runtime.tree_digest(s5_runtime.deps_dir()) if s5_runtime.available() else None
    if p.get('runtime_deps') != actual_deps:
        raise ValueError('frozen runtime dependencies changed')
    return p


def environment(plan):
    env = {k: os.environ[k] for k in ('HOME', 'SSH_AUTH_SOCK', 'TMPDIR', 'LANG')
           if k in os.environ}
    env.update(PATH='/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin',
               PYTHONPATH=s5_runtime.python_path(), PYTHONNOUSERSITE='1',
               OPERATOR_AGENT_EXISTING_REMOTE_ONLY='1',
               OPERATOR_AGENT_S4_GRACEFUL_CLEANUP='1',
               OPERATOR_AGENT_REQUIRED_GPU=str(plan['pin']['gpu_index']),
               OPERATOR_AGENT_REQUIRED_GPU_UUID=plan['pin']['gpu_uuid'],
               OPERATOR_AGENT_REQUIRED_GPU_NAME=plan['pin']['gpu_name'])
    return env


def _command(plan, out):
    return [sys.executable, '-m', 'operator_agent.cli', 'verify',
            '--run-dir', str(out/'task'), '--code-file', str(out/'candidate.py'),
            '--candidate-origin', plan['candidate_origin'], '--stop-after', plan['stop_after'],
            '--backend', 'ssh', '--host', plan['pin']['host'], '--persistent',
            '--timeout', str(plan['timeout'])]


def _run(command, out, env, timeout):
    """Allow CLI-owned cancellation first; never signal unrelated processes."""
    started = time.monotonic()
    stopped = None
    with (out/'cli.stdout').open('x') as stdout, (out/'cli.stderr').open('x') as stderr:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                start_new_session=True)
        try:
            proc.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            stopped = 'wall_clock_or_interrupt'
            try:
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=30)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                # The process group was created exclusively for this CLI call.
                try: os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError: pass
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try: os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                    proc.wait(timeout=5)
    return dict(returncode=proc.returncode, stopped_reason=stopped,
                wall_seconds=round(time.monotonic()-started, 3))


def _finite_positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def qualified_timing(m, arms, pin):
    """Accept idle known shared processes only when production gates pass."""
    if not isinstance(m, dict) or not isinstance(arms, dict):
        return False
    other_count = m.get('gpu_other_procs')
    if (m.get('measurement_validity') != 'valid'
            or m.get('measurement_can_gate') is not True
            or (m.get('eligibility') or {}).get('can_report') is not True
            or (m.get('eligibility') or {}).get('can_verdict') is not True
            or m.get('backend_real_device') is not True
            or m.get('correctness_verified') is not True
            or type(other_count) is not int or other_count < 0
            or m.get('gpu_shared_with_others') is not (other_count > 0)
            or m.get('benchmark_contaminated') is not False
            or m.get('measurement_scope') != 'same_remote_device'
            or m.get('measurement_window_scope') != 'same_window'
            or type(m.get('paired_samples')) is not int or m['paired_samples'] < 2
            or not all(_finite_positive(m.get(key)) for key in
                       ('candidate_median_time_ms', 'baseline_time_ms', 'speedup_vs_baseline'))):
        return False
    for key in ('candidate', 'baseline', 'after'):
        arm = arms.get(key) or {}
        if (arm.get('cuda_visible_devices') != str(pin['gpu_index'])
                or arm.get('count') != 1 or arm.get('name') != pin['gpu_name']):
            return False
    return True


def inspect_evidence(plan, task_dir):
    """Interpret full production evidence, never the CLI's compact metrics."""
    task_dir = Path(task_dir)
    rd = task_dir/'verify-1'
    result = dict(schema=RESULT_SCHEMA, status='unknown', full_l2_passed=None,
                  qualified=False, performance_target_passed=None,
                  candidate_median_time_ms=None, baseline_time_ms=None,
                  speedup_vs_baseline=None, measurement_validity=None,
                  measurement_can_gate=None, gpu_other_procs=None,
                  gpu_shared_with_others=None, reason=None, model_calls=0,
                  formal_s5_credit=False)
    problems = []
    try:
        if any(p.is_symlink() for p in (rd, rd/'result.json', rd/'complete.json', rd/'candidate.py')):
            raise ValueError('symlink evidence refused')
        e, receipt = read(rd/'result.json'), read(rd/'complete.json')
        binding = e.get('binding') or {}
        if (e.get('schema') != 'operator-tool-evidence-v1' or e.get('evidence_complete') is not True
                or e.get('implementation_stable') is not True
                or e.get('implementation') != plan['protocol']):
            problems.append('production evidence or implementation identity incomplete')
        if (binding.get('source_hash') != plan['candidate_sha256']
                or e.get('source_hash') != plan['candidate_sha256']
                or sha(rd/'candidate.py') != plan['candidate_sha256']
                or binding.get('contract_hash') != plan['contract_hash']
                or e.get('contract_hash') != plan['contract_hash']
                or binding.get('stop_after') != plan['stop_after']):
            problems.append('source, contract or level identity mismatch')
        resolution = e.get('backend_resolution') or {}
        if resolution.get('requested') != 'ssh' or resolution.get('degraded') is not False:
            problems.append('backend degraded or not SSH')
        files = receipt.get('files_sha256')
        if (receipt.get('schema') != 'operator-tool-evidence-complete-v1'
                or receipt.get('evidence_complete') is not True
                or receipt.get('binding') != binding or not isinstance(files, dict)
                or not {'candidate.py', 'result.json', 'started.json'} <= set(files)):
            problems.append('completion receipt incomplete')
        else:
            for name, checksum in files.items():
                rel = Path(name)
                path = rd/rel
                if (rel.is_absolute() or '..' in rel.parts or path.is_symlink()
                        or not path.is_file() or sha(path) != checksum):
                    problems.append('receipt member mismatch: '+name)
                    break
        if problems:
            result['reason'] = '; '.join(problems)
            return result
        result.update(evidence_sha256=sha(rd/'result.json'), receipt_sha256=sha(rd/'complete.json'),
                      source_sha256=plan['candidate_sha256'], contract_hash=plan['contract_hash'])
        verdicts = e.get('verdicts') or []
        l1 = next((v for v in verdicts if v.get('level') == 'L1'), {})
        l2 = next((v for v in verdicts if v.get('level') == 'L2'), {})
        if _l1_rejection_is_decisive(e):
            result.update(status='failed', full_l2_passed=False, reason='candidate compile/syntax rejection')
            return result
        correct = (l2.get('metrics') or {}).get('correctness') or {}
        if l1.get('passed') is True and l2.get('passed') is False and correct.get('verdict') == 'failed':
            result.update(status='failed', full_l2_passed=False, reason='candidate numerical rejection')
            return result
        cases = correct.get('cases') or []
        fidelity = (l2.get('metrics') or {}).get('input_fidelity_readings') or []
        if (l1.get('passed') is not True or l2.get('passed') is not True
                or correct.get('verdict') != 'passed'
                or correct.get('suite_hash') != plan['suite_sha256']
                or len(cases) != len(plan['suite_case_ids'])
                or {c.get('case_id') for c in cases} != set(plan['suite_case_ids'])
                or any(c.get('verdict') != 'passed'
                       or (c.get('metrics') or {}).get('input_fidelity_verified') is not True for c in cases)
                or not fidelity or any(f.get('input_fidelity_verified') is not True for f in fidelity)):
            result['reason'] = 'full correctness suite or input fidelity not proven'
            return result
        result['full_l2_passed'] = True
        if plan['stop_after'] == 'L2':
            result.update(status='passed', reason='full L2 passed; no performance measured')
            return result
        l3 = next((v for v in verdicts if v.get('level') == 'L3'), {})
        m = l3.get('metrics') or {}
        validity, gate = m.get('measurement_validity'), m.get('measurement_can_gate')
        result.update(measurement_validity=validity, measurement_can_gate=gate,
                      performance_target_passed=l3.get('passed'),
                      baseline_type=m.get('baseline_type'), paired_samples=m.get('paired_samples'),
                      gpu_other_procs=m.get('gpu_other_procs'),
                      gpu_shared_with_others=m.get('gpu_shared_with_others'))
        candidate = m.get('candidate_median_time_ms', m.get('median_time_ms'))
        baseline, speedup = m.get('baseline_time_ms'), m.get('speedup_vs_baseline')
        arms = m.get('arm_identity') or {}
        qualified_metrics = dict(m, candidate_median_time_ms=candidate)
        if not qualified_timing(qualified_metrics, arms, plan['pin']):
            problems.append('valid qualified same-window real-device timing not proven')
        if problems:
            result['reason'] = '; '.join(problems)
            return result
        result.update(status='measured', qualified=True, candidate_median_time_ms=candidate,
                      baseline_time_ms=baseline, speedup_vs_baseline=speedup,
                      reason='full L2 plus valid qualified L3; one independent reading')
        return result
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        result['reason'] = 'evidence unavailable or malformed: '+type(exc).__name__
        return result


def execute(plan_path, expected_hash):
    """Dispatch exactly once. Missing/invalid evidence or cleanup stays unknown."""
    plan_path = Path(plan_path).resolve()
    plan = load_plan(plan_path, expected_hash)
    out = plan_path.parent
    write(out/'execution-started.json', dict(plan_hash=expected_hash, started_at=time.time(),
                                           model_calls=0, max_verifications=1, no_resume=True))
    summary = dict(schema=RESULT_SCHEMA, status='unknown', qualified=False,
                   full_l2_passed=None, reason=None, model_calls=0, formal_s5_credit=False)
    started = time.monotonic()
    try:
        prepared = prepare(plan['declaration'], out/'task')
        if prepared['contract_hash'] != plan['contract_hash']:
            raise ValueError('prepared contract changed')
        before = s5_shared_gpu.snapshot(plan['pin'], plan['resource_policy'], environment(plan))
        write(out/'resources-before.json', before)
        write(out/'dispatch.json', dict(command=_command(plan, out), candidate_sha256=plan['candidate_sha256'],
                                        plan_hash=expected_hash, model_calls=0))
        supervision = _run(_command(plan, out), out, environment(plan), plan['timeout'])
        write(out/'supervision.json', supervision)
        after = s5_shared_gpu.snapshot(plan['pin'], plan['resource_policy'], environment(plan))
        write(out/'resources-after.json', after)
        before_pids = [item['pid'] for item in before['compute_apps']]
        after_pids = [item['pid'] for item in after['compute_apps']]
        if before_pids != after_pids:
            raise ValueError('resident GPU process set changed during evaluation')
        summary = inspect_evidence(plan, out/'task')
        summary.update(resource_mode=plan['resource_policy']['mode'],
                       other_compute_processes=len(before['compute_apps']),
                       gpu_util_before_pct=before['util_pct'], gpu_util_after_pct=after['util_pct'])
        if supervision.get('stopped_reason') or supervision.get('returncode') != 0:
            summary.update(status='unknown', qualified=False,
                           reason='CLI stopped or failed; remote cleanup may be unknown')
        if implementation_identity() != plan['implementation'] or protocol_identity() != plan['protocol']:
            summary.update(status='unknown', qualified=False, reason='implementation changed during verification')
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        summary.update(status='unknown', qualified=False,
                       reason='evaluation unavailable: '+type(exc).__name__)
    finally:
        summary.update(plan_hash=expected_hash, candidate_sha256=plan['candidate_sha256'],
                       contract_hash=plan['contract_hash'], wall_seconds=round(time.monotonic()-started, 3))
        write(out/'result.json', summary)
    return summary


def evaluate(declaration, candidate_path, output_dir, pin, resource_policy, timeout=600, stop_after='L3', *,
             execute=False, plan_hash=None):
    """Default is freeze-only. Pass execute=True and the returned plan hash to run."""
    if not execute:
        return freeze(output_dir, declaration=declaration, candidate_path=candidate_path,
                      pin=pin, resource_policy=resource_policy,
                      timeout=timeout, stop_after=stop_after)
    plan_path = Path(output_dir)/'plan.json'
    plan = load_plan(plan_path, plan_hash)
    if (plan['declaration'] != declaration or plan['pin'] != pin
            or plan['resource_policy'] != resource_policy or plan['timeout'] != timeout
            or plan['stop_after'] != stop_after or sha(candidate_path) != plan['candidate_sha256']):
        raise ValueError('requested evaluation differs from frozen plan')
    return globals()['execute'](plan_path, plan_hash)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    f = sub.add_parser('freeze')
    f.add_argument('--output', required=True, type=Path)
    f.add_argument('--declaration', required=True, type=Path)
    f.add_argument('--candidate', required=True, type=Path)
    f.add_argument('--pin', required=True, type=Path)
    f.add_argument('--resource-policy', required=True, type=Path)
    f.add_argument('--timeout', type=int, default=600)
    f.add_argument('--stop-after', choices=['L2', 'L3'], default='L3')
    e = sub.add_parser('execute')
    e.add_argument('--plan', required=True, type=Path)
    e.add_argument('--plan-hash', required=True)
    args = parser.parse_args(argv)
    if args.action == 'freeze':
        result = freeze(args.output, declaration=read(args.declaration), candidate_path=args.candidate,
                        pin=read(args.pin), resource_policy=read(args.resource_policy),
                        timeout=args.timeout, stop_after=args.stop_after)
    else:
        result = execute(args.plan, args.plan_hash)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if args.action == 'freeze' or result['status'] in ('measured', 'passed', 'failed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
