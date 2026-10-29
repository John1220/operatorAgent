"""Controlled S4 cancellation via the public CLI; no model calls.

This is a one-shot utility, not a general experiment runner. A timeout or missing
owned cleanup receipt is UNKNOWN and must stop the campaign.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
GPU_UUID = 'GPU-52ca6155-d8bc-a416-847a-6f0c71db1552'
FIXTURE = ROOT/'tests/fixtures/layernorm_missing_column_stride.txt'
EXPECTED_CONTRACT = '96cdb38deafce9282485b5beaa555a738e573a1c0892292554b9d7f70685c7a7'


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def cli(env, *args, timeout=20):
    return subprocess.run([sys.executable, '-m', 'operator_agent', *args], cwd=ROOT,
        env=env, text=True, capture_output=True, timeout=timeout)


def main(destination):
    dest = Path(destination).resolve()
    dest.mkdir(parents=True, exist_ok=False)
    run = dest/'run'
    env = os.environ.copy()
    env.update(OPERATOR_AGENT_REQUIRED_GPU='7', OPERATOR_AGENT_REQUIRED_GPU_UUID=GPU_UUID,
        OPERATOR_AGENT_REQUIRED_GPU_NAME='NVIDIA H20', OPERATOR_AGENT_EXISTING_REMOTE_ONLY='1',
        OPERATOR_AGENT_S4_GRACEFUL_CLEANUP='1')
    prepared = cli(env, 'prepare', '--request', 'layernorm fp32 triton', '--shapes', '7,33',
        '--hardware', 'nvidia_gpu', '--dsl', 'triton', '--precision', 'fp32',
        '--min-speedup', '1.0', '--run-dir', str(run))
    (dest/'prepare.stdout.json').write_text(prepared.stdout)
    data = read(dest/'prepare.stdout.json')
    if prepared.returncode or not isinstance(data, dict) or data.get('contract_hash') != EXPECTED_CONTRACT:
        raise RuntimeError('prepare failed or contract drifted; no remote operation started')
    command = [sys.executable, '-m', 'operator_agent', 'verify', '--run-dir', str(run),
        '--code-file', str(FIXTURE), '--stop-after', 'L2', '--backend', 'ssh',
        '--host', 'algorithm-h20', '--persistent', '--timeout', '90']
    started = time.monotonic()
    with (dest/'verify.stdout.json').open('w') as stdout, (dest/'verify.stderr.log').open('w') as stderr:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                start_new_session=True)
        invocation = None
        readiness = False
        while proc.poll() is None and time.monotonic()-started < 90:
            root = run/'tool-calls'
            dirs = list(root.iterdir()) if root.is_dir() else []
            if len(dirs) == 1:
                directory = dirs[0]
                launch = read(directory/'launch.json') or {}
                candidate = launch.get('invocation_id')
                if isinstance(candidate, str) and len(candidate) == 32:
                    invocation = candidate
                    for event_path in directory.glob('owned-session-ready-*.json'):
                        event = read(event_path) or {}
                        if event.get('schema') == 'operator-owned-session-ready-v1' and event.get('invocation_id') == invocation:
                            readiness = True
                            break
            elif len(dirs) > 1:
                break
            if readiness:
                break
            time.sleep(.05)
        cancel = None
        if invocation and proc.poll() is None:
            try:
                cancel = cli(env, 'cancel', '--run-dir', str(run), '--invocation-id', invocation)
                (dest/'cancel.stdout.json').write_text(cancel.stdout)
            except subprocess.TimeoutExpired:
                if proc.poll() is None:
                    proc.send_signal(signal.SIGTERM)
        elif proc.poll() is None:
            # Signal the public CLI owner; its handler requests cancellation.
            proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            # Never guess that killing the local CLI stops its remote worker.
            (dest/'result.json').write_text(json.dumps(dict(state='unknown', reason='CLI did not exit',
                invocation_id=invocation, ready=readiness), indent=2)+'\n')
            return 1
    status = read(run/'tool-calls'/str(invocation)/'run-status.json') if invocation else None
    cleanup = (status or {}).get('cleanup') or {}
    ok = bool(readiness and cancel is not None and cancel.returncode == 0 and
              (status or {}).get('phase') == 'cancelled' and
              cleanup.get('remote_execution') == 'quiescent')
    result = dict(state='passed' if ok else 'unknown', invocation_id=invocation,
        ready=readiness, cli_exit=proc.returncode, status=status,
        elapsed_seconds=round(time.monotonic()-started, 3),
        scope='Only this owned persistent session; provider state not established.')
    (dest/'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    return 0 if ok else 1


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: s4_cancel_smoke.py NEW_DESTINATION')
    raise SystemExit(main(sys.argv[1]))
