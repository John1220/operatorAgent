"""Read-only checks for a frozen S5 shared GPU with known resident processes.

This is an observation before dispatch, not a GPU reservation. Existing PIDs may
exit; a new target-device PID, missing identity or unreadable resource data
refuses dispatch. The historical ruler's exclusive-device policy is unchanged.
"""
from __future__ import annotations

import csv
import io
import os
import re
import subprocess
import time


POLICY_KEYS = frozenset({'mode', 'min_free_mib', 'max_util_pct', 'allowed_compute_pids'})
PIN_KEYS = frozenset({'host', 'gpu_index', 'gpu_uuid', 'gpu_name'})
UUID = re.compile(r'GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z')
HOST = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,252}\Z')
UINT = re.compile(r'[0-9]+\Z')
MAX_QUERY_BYTES = 1024 * 1024


def validate_policy(policy):
    """Reject ambiguous or weakened versions of the declared shared policy."""
    if not isinstance(policy, dict) or set(policy) != POLICY_KEYS:
        raise ValueError('invalid shared GPU policy fields')
    pids = policy['allowed_compute_pids']
    if (policy['mode'] != 'shared-known-processes'
            or type(policy['min_free_mib']) is not int or policy['min_free_mib'] != 8192
            or type(policy['max_util_pct']) is not int or policy['max_util_pct'] != 5
            or not isinstance(pids, list)
            or any(type(pid) is not int or pid <= 0 for pid in pids)
            or pids != sorted(set(pids))):
        raise ValueError('invalid shared GPU policy values')
    return policy


def _validate_pin(pin):
    if not isinstance(pin, dict) or set(pin) != PIN_KEYS:
        raise ValueError('invalid shared GPU pin fields')
    if (not isinstance(pin['host'], str) or not HOST.fullmatch(pin['host'])
            or type(pin['gpu_index']) is not int or not 0 <= pin['gpu_index'] <= 63
            or not isinstance(pin['gpu_uuid'], str) or not UUID.fullmatch(pin['gpu_uuid'])
            or not isinstance(pin['gpu_name'], str) or not pin['gpu_name'].strip()
            or len(pin['gpu_name']) > 128 or any(ord(c) < 32 for c in pin['gpu_name'])):
        raise ValueError('invalid shared GPU pin values')


def _uint(value, field):
    if not UINT.fullmatch(value):
        raise ValueError('malformed GPU ' + field)
    return int(value)


def snapshot(pin, policy, env=None):
    """Check exact device identity, capacity, utilization and resident PID set."""
    validate_policy(policy)
    _validate_pin(pin)
    environment = os.environ.copy() if env is None else env

    def query(fields, kind):
        command = 'nvidia-smi --query-' + kind + '=' + fields + ' --format=csv,noheader,nounits'
        proc = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                               pin['host'], command], env=environment,
                              capture_output=True, text=True, check=True, timeout=15)
        if not isinstance(proc.stdout, str) or len(proc.stdout.encode('utf-8')) > MAX_QUERY_BYTES:
            raise ValueError('GPU query output exceeds bound or is missing')
        try:
            rows = list(csv.reader(io.StringIO(proc.stdout), skipinitialspace=True, strict=True))
        except csv.Error:
            raise ValueError('malformed GPU CSV query') from None
        return [[item.strip() for item in row] for row in rows]

    gpu_rows = query('index,uuid,name,memory.free,utilization.gpu', 'gpu')
    app_rows = query('gpu_uuid,pid,process_name,used_gpu_memory', 'compute-apps')
    gpus = []
    for row in gpu_rows:
        if len(row) != 5 or not UUID.fullmatch(row[1]) or not row[2]:
            raise ValueError('GPU identity query incomplete')
        index = _uint(row[0], 'index')
        free = _uint(row[3], 'free memory')
        util = _uint(row[4], 'utilization')
        if not 0 <= index <= 63 or not 0 <= util <= 100:
            raise ValueError('GPU identity query outside valid bounds')
        gpus.append(dict(gpu_index=index, uuid=row[1], name=row[2], free_mib=free, util_pct=util))
    found = [gpu for gpu in gpus if gpu['uuid'] == pin['gpu_uuid']]
    if len(found) != 1:
        raise ValueError('GPU identity missing or ambiguous')
    gpu = found[0]
    if (gpu['gpu_index'] != pin['gpu_index'] or gpu['name'] != pin['gpu_name']
            or gpu['free_mib'] < policy['min_free_mib']
            or gpu['util_pct'] > policy['max_util_pct']):
        raise ValueError('GPU identity/capacity/utilization does not satisfy frozen shared plan')

    apps = []
    for row in app_rows:
        if (len(row) != 4 or not UUID.fullmatch(row[0]) or not row[2]
                or any(ord(c) < 32 for c in row[2])):
            raise ValueError('compute process query incomplete')
        pid = _uint(row[1], 'process PID')
        memory = _uint(row[3], 'process memory')
        if pid <= 0:
            raise ValueError('compute process PID must be positive')
        if row[0] == pin['gpu_uuid']:
            apps.append(dict(pid=pid, process_name=row[2], used_gpu_memory_mib=memory))
    observed_pids = [app['pid'] for app in apps]
    if len(observed_pids) != len(set(observed_pids)):
        raise ValueError('compute process query has duplicate target PIDs')
    if not set(observed_pids).issubset(policy['allowed_compute_pids']):
        raise ValueError('GPU has an unapproved compute process; no task dispatched')
    return dict(observed_at=time.time(), **gpu, resource_mode=policy['mode'],
                other_compute_processes=len(apps), compute_apps=sorted(apps, key=lambda r: r['pid']),
                exclusive_lease=False)
