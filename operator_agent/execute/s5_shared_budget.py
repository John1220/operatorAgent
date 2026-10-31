"""Local cross-process S5 model-call reservations; no dispatch or network I/O.

A single SQLite database supplies the same pre-dispatch reservation pool to
cooperating outer and inner processes. This is a primitive, not a claim that all
host/provider calls have been intercepted or that same-UID code is isolated.
The audit directory must be on one local filesystem. SQLite transactions + FULL
synchronous durability serialize writers. Requests are represented by SHA256,
never prompts or credentials. A reservation can be consumed even if its owner
dies before dispatch; it is never refunded.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
import time
import uuid

from ..execute.task_clock import monotonic
from ..core.run_layout import create_private_mutable_file_once

SCHEMA = 's5-shared-call-budget-v1'
LABEL = re.compile(r'[A-Za-z0-9_.:-]{1,64}\Z')
HASH = re.compile(r'[0-9a-f]{64}\Z')
CALL_ID = re.compile(r'[0-9a-f]{32}\Z')
PARTICIPANTS = ('outer', 'inner')
OUTCOMES = ('answered', 'failed', 'cancelled', 'unknown')
ABORT_REASONS = ('audit_failure', 'budget_controller_failure', 'participant_crash', 'manual_stop')
MAX_INTEGER = 2**63 - 1


class BudgetError(RuntimeError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def _pos_int(value, name):
    if type(value) is not int or not 0 < value <= MAX_INTEGER:
        raise BudgetError('invalid_' + name)
    return value


def _label(value, name):
    if type(value) is not str or not LABEL.fullmatch(value):
        raise BudgetError('invalid_' + name)
    return value


def _now():
    return monotonic()


def _connect(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise BudgetError('ledger_missing')
    try:
        conn = sqlite3.connect(str(path), timeout=15, isolation_level=None)
        conn.execute('PRAGMA busy_timeout=15000')
        conn.execute('PRAGMA foreign_keys=ON')
        conn.execute('PRAGMA synchronous=FULL')
        return conn
    except sqlite3.Error:
        raise BudgetError('ledger_unavailable') from None


def _meta(conn):
    try:
        rows = conn.execute('SELECT key,value FROM settings').fetchall()
        if len(rows) != 1 or rows[0][0] != 'config':
            raise ValueError('settings')
        data = json.loads(rows[0][1])
        expected = {'schema','task_id','max_calls','max_output_tokens',
                    'total_output_tokens','start_monotonic','start_wall','deadline_monotonic'}
        if type(data) is not dict or set(data) != expected or data['schema'] != SCHEMA:
            raise ValueError('config')
        _label(data['task_id'], 'task_id')
        for key in ('max_calls','max_output_tokens','total_output_tokens'):
            _pos_int(data[key], key)
        a,b,w=data['start_monotonic'],data['deadline_monotonic'],data['start_wall']
        if (type(a) not in (int,float) or type(b) not in (int,float) or
                type(w) not in (int,float) or not math.isfinite(w) or
                not math.isfinite(a) or not math.isfinite(b) or not 60 < a < b or w<=0):
            raise ValueError('time')
        now=_now()
        # On a reboot, elapsed monotonic and elapsed wall time diverge by at
        # least the original uptime. A clock jump/sleep also fails closed rather
        # than granting a longer budget. This is not a cryptographic boot ID.
        if now < a or abs((time.time()-w)-(now-a)) > 5:
            raise BudgetError('clock_identity_changed')
        return data
    except BudgetError:
        raise
    except (ValueError, TypeError, KeyError, sqlite3.Error):
        raise BudgetError('ledger_corrupt') from None


def create(path, *, task_id, max_calls, max_output_tokens, total_output_tokens, wall_clock_s):
    path=Path(path)
    _label(task_id,'task_id')
    for n,v in (('max_calls',max_calls),('max_output_tokens',max_output_tokens),
                ('total_output_tokens',total_output_tokens)):
        _pos_int(v,n)
    if (type(wall_clock_s) not in (int,float) or not math.isfinite(wall_clock_s)
            or wall_clock_s<=0 or wall_clock_s>86400):
        raise BudgetError('invalid_wall_clock_s')
    started=_now(); wall=time.time(); deadline=started+wall_clock_s
    if started <= 60: raise BudgetError('clock_identity_unavailable')
    if not math.isfinite(deadline): raise BudgetError('invalid_wall_clock_s')
    config=dict(schema=SCHEMA,task_id=task_id,max_calls=max_calls,
                max_output_tokens=max_output_tokens,total_output_tokens=total_output_tokens,
                start_monotonic=started,start_wall=wall,deadline_monotonic=deadline)
    try:
        create_private_mutable_file_once(str(path))
    except FileExistsError:
        raise BudgetError('ledger_exists') from None
    except OSError:
        raise BudgetError('ledger_create_failed') from None
    try:
        conn=_connect(path)
        try:
            conn.execute('PRAGMA journal_mode=DELETE')
            conn.execute('BEGIN IMMEDIATE')
            conn.execute('CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            conn.execute('CREATE TABLE reservations (sequence INTEGER PRIMARY KEY, call_id TEXT NOT NULL UNIQUE, participant TEXT NOT NULL, caller TEXT NOT NULL, output_tokens INTEGER NOT NULL, request_sha256 TEXT NOT NULL, reserved_monotonic REAL NOT NULL)')
            conn.execute('CREATE TABLE events (sequence INTEGER PRIMARY KEY, call_id TEXT NOT NULL REFERENCES reservations(call_id), kind TEXT NOT NULL, monotonic REAL NOT NULL, outcome TEXT, usage_json TEXT, usage_evidence TEXT, UNIQUE(call_id,kind))')
            conn.execute('CREATE TABLE aborts (id INTEGER PRIMARY KEY CHECK(id=1), reason TEXT NOT NULL, monotonic REAL NOT NULL)')
            conn.execute('INSERT INTO settings(key,value) VALUES (?,?)',('config',json.dumps(config,sort_keys=True,allow_nan=False)))
            for table in ('settings','reservations','events','aborts'):
                for op in ('UPDATE','DELETE'):
                    conn.execute(f"CREATE TRIGGER {table}_no_{op.lower()} BEFORE {op} ON {table} BEGIN SELECT RAISE(ABORT,'append only'); END")
            conn.execute('COMMIT')
        finally: conn.close()
        # Persist creation of the database name itself where the platform permits.
        parent=os.open(str(path.parent), os.O_RDONLY)
        try: os.fsync(parent)
        finally: os.close(parent)
    except (sqlite3.Error,OSError):
        # Keep the exclusive, possibly incomplete path; never offer a fresh
        # allowance under the same identity after an ambiguous failure.
        raise BudgetError('ledger_create_failed') from None
    return dict(ok=True,schema=SCHEMA,task_id=task_id)


def _transaction(path, action, expected_task_id=None):
    if expected_task_id is not None: _label(expected_task_id, 'task_id')
    conn=_connect(path)
    try:
        conn.execute('BEGIN IMMEDIATE')
        meta=_meta(conn)
        if expected_task_id is not None and meta['task_id'] != expected_task_id:
            raise BudgetError('task_identity_mismatch')
        out=action(conn,meta)
        conn.execute('COMMIT')
        return out
    except BudgetError:
        try: conn.execute('ROLLBACK')
        except sqlite3.Error: pass
        raise
    except (sqlite3.Error,ValueError,TypeError,OverflowError):
        try: conn.execute('ROLLBACK')
        except sqlite3.Error: pass
        raise BudgetError('ledger_failure') from None
    finally: conn.close()


def reserve(path, *, participant, caller, max_output_tokens, request_sha256, expected_task_id=None):
    if participant not in PARTICIPANTS: raise BudgetError('invalid_participant')
    _label(caller,'caller'); _pos_int(max_output_tokens,'max_output_tokens')
    if type(request_sha256) is not str or not HASH.fullmatch(request_sha256):
        raise BudgetError('invalid_request_sha256')
    def action(conn,meta):
        now=_now()
        if conn.execute('SELECT 1 FROM aborts WHERE id=1').fetchone(): raise BudgetError('campaign_aborted')
        if now >= meta['deadline_monotonic']: raise BudgetError('wall_clock_exhausted')
        if max_output_tokens>meta['max_output_tokens']: raise BudgetError('per_call_output_exhausted')
        count,total=conn.execute('SELECT COUNT(*), COALESCE(SUM(output_tokens),0) FROM reservations').fetchone()
        if count>=meta['max_calls']: raise BudgetError('call_budget_exhausted')
        if total+max_output_tokens>meta['total_output_tokens']: raise BudgetError('output_budget_exhausted')
        call_id=uuid.uuid4().hex
        cur=conn.execute('INSERT INTO reservations(call_id,participant,caller,output_tokens,request_sha256,reserved_monotonic) VALUES (?,?,?,?,?,?)',
                         (call_id,participant,caller,max_output_tokens,request_sha256,now))
        return dict(ok=True,call_id=call_id,sequence=cur.lastrowid)
    return _transaction(path,action,expected_task_id)


def _call_id(value):
    if type(value) is not str or not CALL_ID.fullmatch(value): raise BudgetError('invalid_call_id')


def dispatch(path, *, call_id, expected_task_id=None):
    _call_id(call_id)
    def action(conn,meta):
        now=_now()
        if conn.execute('SELECT 1 FROM aborts WHERE id=1').fetchone(): raise BudgetError('campaign_aborted')
        if now >= meta['deadline_monotonic']: raise BudgetError('wall_clock_exhausted')
        if not conn.execute('SELECT 1 FROM reservations WHERE call_id=?',(call_id,)).fetchone(): raise BudgetError('reservation_missing')
        if conn.execute('SELECT 1 FROM events WHERE call_id=?',(call_id,)).fetchone(): raise BudgetError('dispatch_already_recorded')
        conn.execute('INSERT INTO events(call_id,kind,monotonic) VALUES (?,?,?)',(call_id,'dispatch',now))
        return dict(ok=True,call_id=call_id)
    return _transaction(path,action,expected_task_id)


def abort(path, *, reason, expected_task_id=None):
    """Irreversibly stop future dispatches while allowing known calls to finish.

    If durable recording itself fails, the controlling supervisor must stop all
    participants; a failed abort is never evidence the campaign is stopped.
    """
    if reason not in ABORT_REASONS: raise BudgetError('invalid_abort_reason')
    def action(conn,meta):
        old=conn.execute('SELECT reason,monotonic FROM aborts WHERE id=1').fetchone()
        if old is not None:
            return dict(ok=True,aborted=True,reason=old[0],already_aborted=True)
        conn.execute('INSERT INTO aborts(id,reason,monotonic) VALUES (1,?,?)',(reason,_now()))
        return dict(ok=True,aborted=True,reason=reason,already_aborted=False)
    return _transaction(path,action,expected_task_id)


def _usage(value,evidence):
    if evidence not in ('provider','unverified','missing'): raise BudgetError('invalid_usage_evidence')
    if evidence != 'provider':
        # Non-provider data can be a synthesized zero. Discard it rather than
        # granting it audit significance, but retain its provenance category.
        return None
    keys={'prompt_tokens','completion_tokens','total_tokens','cache_read_tokens','cache_write_tokens'}
    if type(value) is not dict or not {'prompt_tokens','completion_tokens'} <= set(value) or set(value)-keys:
        raise BudgetError('invalid_provider_usage')
    for n,v in value.items():
        if type(v) is not int or v<0 or v>MAX_INTEGER: raise BudgetError('invalid_provider_usage')
    if 'total_tokens' in value and value['total_tokens'] != value['prompt_tokens']+value['completion_tokens']:
        raise BudgetError('invalid_provider_usage')
    if any(value.get(n,0)>value['prompt_tokens'] for n in ('cache_read_tokens','cache_write_tokens')):
        raise BudgetError('invalid_provider_usage')
    if value.get('cache_read_tokens',0)+value.get('cache_write_tokens',0)>value['prompt_tokens']:
        raise BudgetError('invalid_provider_usage')
    return value


def finish(path, *, call_id, outcome, usage=None, usage_evidence='missing', expected_task_id=None):
    _call_id(call_id)
    if outcome not in OUTCOMES: raise BudgetError('invalid_outcome')
    counters=_usage(usage,usage_evidence)
    def action(conn,meta):
        reservation=conn.execute('SELECT output_tokens FROM reservations WHERE call_id=?',(call_id,)).fetchone()
        if not reservation: raise BudgetError('reservation_missing')
        if counters is not None and counters['completion_tokens'] > reservation[0]: raise BudgetError('invalid_provider_usage')
        if not conn.execute('SELECT 1 FROM events WHERE call_id=? AND kind=?',(call_id,'dispatch')).fetchone(): raise BudgetError('dispatch_not_recorded')
        if conn.execute('SELECT 1 FROM events WHERE call_id=? AND kind=?',(call_id,'finish')).fetchone(): raise BudgetError('finish_already_recorded')
        conn.execute('INSERT INTO events(call_id,kind,monotonic,outcome,usage_json,usage_evidence) VALUES (?,?,?,?,?,?)',
                     (call_id,'finish',_now(),outcome,json.dumps(counters,sort_keys=True) if counters is not None else None,usage_evidence))
        return dict(ok=True,call_id=call_id)
    return _transaction(path,action,expected_task_id)


def snapshot(path, *, expected_task_id=None):
    def action(conn,meta):
        reservations=conn.execute('SELECT call_id,participant,caller,output_tokens,request_sha256,reserved_monotonic FROM reservations ORDER BY sequence').fetchall()
        events=conn.execute('SELECT call_id,kind,monotonic,outcome,usage_json,usage_evidence FROM events ORDER BY sequence').fetchall()
        stopped=conn.execute('SELECT reason,monotonic FROM aborts WHERE id=1').fetchone()
        calls=[]
        for cid,participant,caller,out,req,stamp in reservations:
            facts=[e for e in events if e[0]==cid]
            d=next((e for e in facts if e[1]=='dispatch'),None)
            f=next((e for e in facts if e[1]=='finish'),None)
            calls.append(dict(call_id=cid,participant=participant,caller=caller,output_tokens_reserved=out,
                request_sha256=req,reserved_monotonic=stamp,dispatched_monotonic=d[2] if d else None,
                completed_monotonic=f[2] if f else None,outcome=f[3] if f else None,
                api_usage=json.loads(f[4]) if f and f[4] else None,usage_evidence=f[5] if f else 'not_observed'))
        return dict(ok=True,schema=SCHEMA,task_id=meta['task_id'],max_calls=meta['max_calls'],
            max_output_tokens=meta['max_output_tokens'],total_output_tokens=meta['total_output_tokens'],
            reserved_calls=len(calls),remaining_calls=max(0,meta['max_calls']-len(calls)),
            reserved_output_tokens=sum(x['output_tokens_reserved'] for x in calls),
            completed_calls=sum(x['outcome'] is not None for x in calls),
            usage_unknown_calls=sum(x['api_usage'] is None for x in calls),calls=calls,
            aborted=bool(stopped),abort_reason=stopped[0] if stopped else None,
            aborted_monotonic=stopped[1] if stopped else None,
            provider_billing_cap=False,network_interception_proven=False)
    return _transaction(path,action,expected_task_id)


def _parser():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='action',required=True)
    c=sub.add_parser('create'); c.add_argument('--ledger',required=True); c.add_argument('--task-id',required=True)
    c.add_argument('--max-calls',required=True,type=int); c.add_argument('--max-output-tokens',required=True,type=int)
    c.add_argument('--total-output-tokens',required=True,type=int); c.add_argument('--wall-clock-s',required=True,type=float)
    c=sub.add_parser('reserve'); c.add_argument('--ledger',required=True); c.add_argument('--participant',required=True)
    c.add_argument('--caller',required=True); c.add_argument('--max-output-tokens',required=True,type=int); c.add_argument('--request-sha256',required=True)
    c=sub.add_parser('dispatch'); c.add_argument('--ledger',required=True); c.add_argument('--call-id',required=True)
    c=sub.add_parser('finish'); c.add_argument('--ledger',required=True); c.add_argument('--call-id',required=True)
    c.add_argument('--outcome',required=True); c.add_argument('--usage-json'); c.add_argument('--usage-evidence',default='missing')
    c=sub.add_parser('abort'); c.add_argument('--ledger',required=True); c.add_argument('--reason',required=True)
    c=sub.add_parser('snapshot'); c.add_argument('--ledger',required=True)
    for name, child in sub.choices.items():
        if name != 'create': child.add_argument('--expected-task-id')
    return p


def main(argv=None):
    args=vars(_parser().parse_args(argv)); action=args.pop('action'); path=args.pop('ledger')
    try:
        if action=='finish':
            raw=args.pop('usage_json')
            if raw is not None:
                try: args['usage']=json.loads(raw)
                except (ValueError,TypeError): raise BudgetError('invalid_usage_json') from None
        result={'create':create,'reserve':reserve,'dispatch':dispatch,'finish':finish,'abort':abort,'snapshot':snapshot}[action](path,**args)
    except BudgetError as exc:
        print(json.dumps(dict(ok=False,error=exc.reason))); return 2
    print(json.dumps(result,sort_keys=True,allow_nan=False)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
