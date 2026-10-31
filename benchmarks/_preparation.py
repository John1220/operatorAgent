"""Bounded preparation receipts. No raw commands, output or exception text.

These same-UID receipts detect drift, not hostile forgery. A started scratch
creation with a lost reply is UNKNOWN, never proof that no scratch exists.
"""
from __future__ import annotations

import copy
from pathlib import Path
import subprocess
import threading
import time

from operator_agent.execute.llm_budget import strict_object
from operator_agent.core.run_contract import atomic_json, json_value

STAGES = ('runtime_probe', 'gpu_query', 'gpu_select', 'scratch_create',
          'source_upload', 'source_readback', 'runner_readback')
POLICY = dict(schema='preparation-receipts-v1', stages=list(STAGES),
    max_records=15, max_record_bytes=4096, max_gpu_rows=8, max_gpu_reply_bytes=4096,
    max_utilization=5, min_free_mb=1024, raw_output=False, exception_text=False,
    return_code='only when exposed by the transport exception; otherwise unknown')


def gpu_snapshot(raw):
    """Keep numeric GPU state, not remote-provided names or arbitrary text."""
    if not isinstance(raw, str) or len(raw.encode('utf-8')) > POLICY['max_gpu_reply_bytes']:
        raise ValueError('invalid GPU reply')
    lines = raw.splitlines()
    if not 1 <= len(lines) <= POLICY['max_gpu_rows']:
        raise ValueError('invalid GPU row count')
    rows = []
    for line in lines:
        parts = line.split(',')
        if len(parts) != 4:
            raise ValueError('invalid GPU row')
        index, name, free, util = (p.strip() for p in parts)
        index, free, util = int(index), int(free), int(util)
        if (not 0 <= index < 8 or not 0 <= free <= 2**31-1 or not 0 <= util <= 100
                or not name or len(name) > 128 or any(r['index'] == index for r in rows)):
            raise ValueError('invalid GPU values')
        rows.append(dict(index=index, is_h20='H20' in name, free_mb=free, utilization=util))
    return rows


def failure(stage, exc):
    """Fixed vocabulary; even exception class names and messages are untrusted."""
    rc = exc.returncode if isinstance(exc, subprocess.CalledProcessError) else getattr(exc, 'ssh_rc', None)
    rc = rc if type(rc) is int and 0 <= rc <= 255 else None
    if isinstance(exc, KeyboardInterrupt): reason = 'interrupted'
    elif isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)): reason = 'timeout'
    elif rc is not None: reason = 'ssh_command_failed'
    elif isinstance(exc, (ValueError, KeyError, TypeError)): reason = 'invalid_reply_or_guard'
    elif stage == 'gpu_select' and isinstance(exc, RuntimeError): reason = 'gpu_ineligible'
    else: reason = 'stage_failed'
    return dict(reason=reason, ssh_returncode=rc)


class PreparationJournal:
    def __init__(self, directory, requested_gpu):
        if type(requested_gpu) is not int or not 0 <= requested_gpu < 8:
            raise ValueError('explicit requested GPU required')
        self.requested_gpu = requested_gpu
        self.directory = Path(directory)
        if any(p.is_symlink() for p in (self.directory, *self.directory.parents)):
            raise ValueError('symlink preparation directory')
        self.directory.mkdir(mode=0o700)  # Fresh, no reopening a prior attempt.
        self.records = {}; self.events = []; self.poisoned = False
        self._lock = threading.Lock()
        self._write('policy.json', dict(POLICY, requested_gpu=requested_gpu))

    def _write(self, name, value):
        from json import dumps
        value = json_value(value)
        if len(dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode()) > POLICY['max_record_bytes']:
            raise ValueError('preparation receipt exceeds byte cap')
        p = self.directory/name
        if p.exists() or p.is_symlink() or len(self.records) >= POLICY['max_records']:
            raise ValueError('preparation receipt cannot be replaced or expanded')
        atomic_json(p, value, readonly=True)
        if p.stat().st_size > POLICY['max_record_bytes']:
            raise ValueError('serialized preparation receipt exceeds byte cap')
        self.records[name] = value

    def step(self, stage, operation, counts, details=lambda value: {}):
        with self._lock:
            i = len(self.events)
            if self.poisoned or i >= len(STAGES) or stage != STAGES[i]:
                raise ValueError('reordered/retried preparation refused')
            # Even failure to publish the start receipt cannot buy a retry.
            self.poisoned = True
        def bounded_counts():
            v = dict(counts())
            if set(v) != {'rpc', 'command'} or any(type(n) is not int or not 0 <= n <= 512 for n in v.values()):
                raise ValueError('invalid preparation transport counts')
            return v
        stem = f'{i+1:02d}-{stage}'
        started = dict(stage=stage, state='started', counts=bounded_counts())
        self._write(stem+'-start.json', started)  # Durable BEFORE possible side effects.
        self.events.append(started)
        start = time.monotonic()
        try:
            result = operation()
            ended = dict(stage=stage, state='passed', details=details(result))
        except (Exception, KeyboardInterrupt) as exc:
            ended = dict(stage=stage, state='failed', **failure(stage, exc))
            ended.update(counts=bounded_counts(), elapsed_s=round(time.monotonic()-start, 3))
            self._write(stem+'-finish.json', ended)
            self.events[-1] = ended
            raise
        ended.update(counts=bounded_counts(), elapsed_s=round(time.monotonic()-start, 3))
        self._write(stem+'-finish.json', ended)
        self.events[-1] = ended; self.poisoned = False
        return result

    def snapshot(self):
        """Read back all receipts; missing/torn/extra/symlink evidence fails closed."""
        if self.directory.is_symlink() or any(p.is_symlink() for p in self.directory.parents):
            raise ValueError('symlink preparation journal')
        if {p.name for p in self.directory.iterdir()} != set(self.records):
            raise ValueError('missing/extra preparation receipt')
        for name, expected in self.records.items():
            p = self.directory/name
            if p.is_symlink() or not p.is_file() or p.stat().st_size > POLICY['max_record_bytes']:
                raise ValueError('unsafe preparation receipt')
            if strict_object(p.read_text()) != expected:
                raise ValueError('preparation receipt drift')
        return dict(schema=POLICY['schema'], read_only=True, evidence_consistent=True,
            requested_gpu=self.requested_gpu, events=copy.deepcopy(self.events),
            completed=len(self.events) == len(STAGES) and not self.poisoned,
            failed_stage=next((e['stage'] for e in self.events if e['state'] != 'passed'), None),
            scratch_creation_attempted=any(e['stage'] == 'scratch_create' for e in self.events))
