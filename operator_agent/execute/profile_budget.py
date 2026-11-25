"""Independent accounting for on-demand NCU profiles.

Profiles are not verification dispatches: they never touch the host task's
reservation budget or its independent final slot. This ledger is separate, is
not reset by resume, counts failures/timeouts, and caps two profiles per task.
"""
from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path
import time

from .gpu_lease import GpuLeaseBusy, acquire as acquire_lease, release as release_lease
from .task_cancellation import valid_invocation


PROFILE_FILE = 'profiles.json'
PROFILE_LOCK = '.profile.lock'
SCHEMA = 'operator-profile-ledger-v1'
MAX_PROFILES = 2
MAX_WALL_S = 300.0


class ProfileBudgetError(ValueError):
    def __init__(self, code, detail=None):
        self.code = code
        self.detail = detail or code
        super().__init__(self.detail)


@contextmanager
def _locked(run):
    descriptor = os.open(run/PROFILE_LOCK, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


class ProfileBudget:
    def __init__(self, run):
        self.run = Path(run)
        self.path = self.run/PROFILE_FILE

    def _load(self):
        if not self.path.exists():
            return dict(schema=SCHEMA, entries=[])
        if self.path.is_symlink():
            raise ProfileBudgetError('profile_ledger', 'profile ledger must be an ordinary file')
        try:
            value = json.loads(self.path.read_text(encoding='utf-8'))
        except (OSError, ValueError, UnicodeError) as error:
            raise ProfileBudgetError('profile_ledger', 'unreadable profile ledger') from error
        if (type(value) is not dict or value.get('schema') != SCHEMA
                or type(value.get('entries')) is not list):
            raise ProfileBudgetError('profile_ledger', 'invalid profile ledger')
        for entry in value['entries']:
            if (type(entry) is not dict or type(entry.get('id')) is not int
                    or entry.get('status') not in ('reserved', 'completed', 'failed', 'cancelled')):
                raise ProfileBudgetError('profile_ledger', 'invalid profile entry')
        return value

    def _write(self, value):
        temporary = self.path.with_suffix(self.path.suffix + '.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
                             encoding='utf-8')
        os.replace(temporary, self.path)

    def snapshot(self):
        with _locked(self.run):
            value = self._load()
            entries = value['entries']
            return dict(schema=SCHEMA, limit=MAX_PROFILES, used=len(entries),
                        remaining=max(0, MAX_PROFILES - len(entries)),
                        in_flight=sum(entry['status'] == 'reserved' for entry in entries),
                        max_wall_s=MAX_WALL_S, entries=entries)

    def reserve(self, *, timeout_s, owner_invocation_id=None, lease_host=None,
                lease_gpu_uuid=None):
        if owner_invocation_id is not None and not valid_invocation(owner_invocation_id):
            raise ProfileBudgetError('profile_reservation', 'invalid owner invocation')
        timeout_s = min(MAX_WALL_S, max(1.0, float(timeout_s)))
        lease_owner = owner_invocation_id or 'profile-' + os.urandom(16).hex()
        try:
            lease = acquire_lease(self.run, owner=lease_owner, tool='profile',
                                  deadline=time.time() + timeout_s,
                                  host=lease_host, gpu_uuid=lease_gpu_uuid)
        except GpuLeaseBusy as error:
            raise ProfileBudgetError(
                'profile_gpu_busy',
                f'{error}; NCU replay and the independent L3 must not share the card') from error
        try:
            with _locked(self.run):
                value = self._load()
                entries = value['entries']
                if any(entry['status'] == 'reserved' for entry in entries):
                    raise ProfileBudgetError('profile_in_flight',
                                             'another profile is still in flight')
                if len(entries) >= MAX_PROFILES:
                    raise ProfileBudgetError('profile_budget',
                                             f'profile budget exhausted ({MAX_PROFILES} per task)')
                entry = dict(id=len(entries) + 1, status='reserved', timeout_s=timeout_s,
                             started_epoch=time.time(), receipt=None, error=None, wall_s=None,
                             owner_invocation_id=owner_invocation_id, lease_owner=lease_owner,
                             lease_key=lease.get('machine_key'), cleanup=None)
                entries.append(entry)
                self._write(value)
                return dict(entry)
        except BaseException:
            release_lease(self.run, lease_owner)
            raise

    def settle_owner(self, owner_invocation_id, *, outcome='cancelled', error=None,
                     cleanup=None, cleanup_s=None, cleanup_grace_s=None):
        """Settle only this owner's reserved profile after its supervisor reaped it.

        The consumed slot is retained: settlement frees ``in_flight`` but never
        refunds or removes the entry. Callers must have confirmed worker death.
        """
        if not valid_invocation(owner_invocation_id):
            raise ProfileBudgetError('profile_reservation', 'invalid owner invocation')
        if outcome not in ('cancelled', 'failed'):
            raise ProfileBudgetError('profile_reservation', 'owner settlement must be cancelled or failed')
        with _locked(self.run):
            value = self._load()
            active = [entry for entry in value['entries']
                      if entry.get('owner_invocation_id') == owner_invocation_id
                      and entry['status'] == 'reserved']
            if not active:
                return dict(settled=0, entries=[])
            now = time.time()
            for entry in active:
                entry.update(status=outcome, finished_epoch=now,
                             wall_s=max(0.0, now - entry['started_epoch']),
                             error=error, cleanup=cleanup, cleanup_s=cleanup_s,
                             cleanup_grace_s=cleanup_grace_s)
            self._write(value)
            for entry in active:
                release_lease(self.run, entry.get('lease_owner')
                              or entry.get('owner_invocation_id') or '')
            return dict(settled=len(active), entries=[dict(entry) for entry in active])

    def finish(self, profile_id, *, status='completed', receipt=None, wall_s=None,
               error=None, cleanup=None, cleanup_s=None, cleanup_grace_s=None):
        if status not in ('completed', 'failed', 'cancelled'):
            raise ProfileBudgetError('profile_reservation', 'invalid profile outcome')
        with _locked(self.run):
            value = self._load()
            matches = [entry for entry in value['entries'] if entry['id'] == profile_id]
            if len(matches) != 1 or matches[0]['status'] != 'reserved':
                raise ProfileBudgetError('profile_reservation',
                                         'profile reservation absent or already finished')
            entry = matches[0]
            entry.update(status=status, finished_epoch=time.time(),
                         receipt=receipt, wall_s=wall_s, error=error,
                         cleanup=cleanup, cleanup_s=cleanup_s,
                         cleanup_grace_s=cleanup_grace_s)
            self._write(value)
            release_lease(self.run, entry.get('lease_owner')
                          or entry.get('owner_invocation_id') or '')
            return dict(entry)
