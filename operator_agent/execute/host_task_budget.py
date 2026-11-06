"""Durable verification reservations for an explicitly enabled host workflow.

The host owns model calls and conversation state. This ledger limits plugin
verification dispatches and elapsed task time; it does not measure GPU seconds,
provider usage or billing. Reservations are never refunded. POSIX file locking
arbitrates cooperating processes, not hostile mutation by the same user.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import uuid

from ..core.run_contract import atomic_json, digest
from ..core.run_layout import write_once_artifact
from .task_cancellation import valid_invocation
from .task_clock import monotonic
import time


POLICY = 'host-task-policy.json'
STATE = 'host-task.json'
LOCK = '.host-task.lock'
SCHEMA = 'operator-host-task-budget-v1'
_HASH = re.compile(r'[0-9a-f]{64}\Z')
MODEL_LIMIT_FIELDS = ('max_model_calls', 'max_output_tokens_per_call',
                      'total_output_token_reservations')
MODEL_STATE = 'host-model-budget.json'


class HostTaskBudgetError(ValueError):
    def __init__(self, code, detail=None):
        self.code = code
        self.detail = detail or code
        super().__init__(self.detail)


@dataclass(frozen=True)
class HostTaskLimits:
    wall_clock_s: float = 600.0
    max_verifications: int = 6
    final_reserve: int = 1
    max_model_calls: int = 12
    max_output_tokens_per_call: int = 8192
    total_output_token_reservations: int = 98304

    def __post_init__(self):
        if (type(self.wall_clock_s) not in (int, float)
                or not math.isfinite(self.wall_clock_s) or self.wall_clock_s <= 0):
            raise ValueError('wall_clock_s must be finite and positive')
        if type(self.max_verifications) is not int or not 2 <= self.max_verifications <= 1000:
            raise ValueError('max_verifications must be an integer from 2 to 1000')
        if type(self.final_reserve) is not int or self.final_reserve != 1:
            raise ValueError('exactly one independent final verification must be reserved')
        for name, maximum in (('max_model_calls', 1000),
                              ('max_output_tokens_per_call', 40960),
                              ('total_output_token_reservations', 40960000)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(name+' must be a positive integer no greater than '+str(maximum))

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value=None):
        if value is None:
            return cls()
        if type(value) is not dict or set(value) - set(cls.__dataclass_fields__):
            raise ValueError('invalid host task budget fields')
        return cls(**value)


def _hash(value, field):
    if type(value) is not str or _HASH.fullmatch(value) is None:
        raise HostTaskBudgetError('host_task_binding', field+' must be a SHA256 digest')
    return value


def _owner(value, *, required=False):
    if value is None and not required:
        return None
    if type(value) is not str or not valid_invocation(value):
        raise HostTaskBudgetError('host_task_owner', 'owner invocation must be 32 lowercase hexadecimal characters')
    return value


def _timely_l2(record):
    return (record.get('full_l2_passed') is True and record['status']=='completed'
            and record.get('finished_after_deadline') is False
            and record.get('accepted_by_supervisor') is not False)


def _run_path(value):
    path = Path(value).absolute()
    if path.is_symlink():
        raise HostTaskBudgetError('host_task_journal', 'ordinary run directory required')
    return path.resolve()


def _read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise HostTaskBudgetError('host_task_journal', 'missing or symlink journal file')
    try:
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError('duplicate key')
                result[key] = value
            return result
        def constant(_):
            raise ValueError('nonfinite JSON')
        value = json.loads(path.read_text(), object_pairs_hook=pairs,
                           parse_constant=constant)
        if type(value) is not dict:
            raise ValueError('object required')
        return value
    except (OSError, ValueError, UnicodeError) as exc:
        raise HostTaskBudgetError('host_task_journal', 'unreadable task journal') from exc


@contextmanager
def _locked(run):
    # Each operation opens its own description, so flock also serializes
    # independent ledger objects used by different threads in one process.
    try:
        import fcntl
    except ImportError as exc:
        raise HostTaskBudgetError('host_task_platform', 'host workflow requires POSIX file locking') from exc
    if run.is_symlink() or not run.is_dir():
        raise HostTaskBudgetError('host_task_journal', 'prepared ordinary run directory required')
    try:
        fd = os.open(run/LOCK, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    except OSError as exc:
        raise HostTaskBudgetError('host_task_journal', 'unable to open ordinary task lock') from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise HostTaskBudgetError('host_task_journal', 'ordinary lock file required')
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def host_task_enabled(run):
    """An incomplete journal is enabled too: it must fail closed, never reset."""
    run = Path(run)
    return any((run/name).exists() or (run/name).is_symlink() for name in (POLICY, STATE))


def _write_state(run, state):
    state['state_hash'] = digest({k:v for k,v in state.items() if k!='state_hash'})
    atomic_json(run/STATE,state)


def _contract_cases(run, contract_hash):
    frozen = _read(run/'contract.json')
    if frozen.get('hash') != contract_hash:
        raise HostTaskBudgetError('host_task_binding', 'prepared contract differs from host task')
    suite = frozen.get('contract', {}).get('correctness_suite')
    cases = suite.get('cases', []) if type(suite) is dict else []
    ids = [case.get('case_id') for case in cases]
    if (any(type(name) is not str or not name for name in ids)
            or len(set(ids)) != len(ids)):
        raise HostTaskBudgetError('host_task_binding', 'invalid frozen correctness suite')
    return sorted(ids)


def initialize_host_task(run, contract_hash, limits=None):
    """Freeze once; a repeat with the same policy returns the original budget."""
    run = _run_path(run)
    _hash(contract_hash, 'contract_hash')
    if not isinstance(limits, HostTaskLimits):
        limits = HostTaskLimits.from_dict(limits)
    with _locked(run):
        cases = _contract_cases(run, contract_hash)
        if not host_task_enabled(run):
            policy = dict(schema=SCHEMA, contract_hash=contract_hash,
                          limits=limits.as_dict(), required_case_ids=cases)
            body = (json.dumps(policy, ensure_ascii=False, sort_keys=True,
                               separators=(',', ':'), allow_nan=False)+'\n').encode('utf-8')
            write_once_artifact(run/POLICY, body)
            _write_state(run, dict(schema=SCHEMA, policy_hash=digest(policy),
                task_id=uuid.uuid4().hex, started_epoch=time.time(),
                started_monotonic=monotonic(), max_elapsed_s=0.0, records=[]))
        budget = HostTaskBudget(run, contract_hash)
        policy, state = budget._load()
        if policy['limits'] != limits.as_dict() or policy['required_case_ids'] != cases:
            raise HostTaskBudgetError('host_task_binding', 'frozen host task policy differs')
        return budget._snapshot(policy, state)


class HostTaskBudget:
    def __init__(self, run, contract_hash):
        self.run = _run_path(run)
        self.contract_hash = _hash(contract_hash, 'contract_hash')

    def _load(self):
        policy, state = _read(self.run/POLICY), _read(self.run/STATE)
        if (policy.get('schema') != SCHEMA or state.get('schema') != SCHEMA
                or policy.get('contract_hash') != self.contract_hash
                or state.get('policy_hash') != digest(policy)
                or state.get('state_hash') != digest({k:v for k,v in state.items() if k!='state_hash'})):
            raise HostTaskBudgetError('host_task_binding', 'host task journal binding differs')
        try:
            HostTaskLimits.from_dict(policy['limits'])
            model_fields = set(policy['limits']) & set(MODEL_LIMIT_FIELDS)
            if model_fields and model_fields != set(MODEL_LIMIT_FIELDS):
                raise ValueError('incomplete frozen model limits')
            records = state['records']
            if (type(records) is not list
                    or [r['id'] for r in records] != list(range(1, len(records)+1))
                    or any(type(r['id']) is not int
                           or r['purpose'] not in ('explore', 'final')
                           or r['status'] not in ('reserved', 'completed', 'failed', 'cancelled')
                           or (r.get('owner_invocation_id') is not None
                               and (type(r['owner_invocation_id']) is not str
                                    or not valid_invocation(r['owner_invocation_id'])))
                           or _HASH.fullmatch(r['source_hash']) is None for r in records)
                    or len(records) > policy['limits']['max_verifications']
                    or sum(r['purpose']=='final' for r in records) > 1
                    or sum(r['purpose']=='explore' for r in records) > policy['limits']['max_verifications']-1):
                raise ValueError('inconsistent records')
            for name in ('started_epoch','started_monotonic','max_elapsed_s'):
                if type(state[name]) not in (int, float) or not math.isfinite(state[name]) or state[name] < 0:
                    raise ValueError('invalid task clock')
            if policy['required_case_ids'] != _contract_cases(self.run, self.contract_hash):
                raise ValueError('frozen suite drift')
        except (KeyError, TypeError, ValueError) as exc:
            raise HostTaskBudgetError('host_task_journal', 'inconsistent host task journal') from exc
        return policy, state

    @staticmethod
    def _elapsed(state):
        current = monotonic()
        if current < state['started_monotonic']:
            raise HostTaskBudgetError('host_task_clock', 'monotonic epoch changed; resume refused')
        # Wall elapsed includes idle/downtime. Persisted elapsed prevents credit
        # being recreated by a cooperating caller after a backwards wall jump.
        return max(state['max_elapsed_s'], current-state['started_monotonic'],
                   time.time()-state['started_epoch'], 0.0)

    def _snapshot(self, policy, state):
        elapsed = self._elapsed(state)
        records = state['records']
        limits = policy['limits']
        correct = [r for r in records if _timely_l2(r)]
        last = correct[-1] if correct else None
        final = next((r for r in records if r['purpose']=='final'), None)
        pending = sum(r['status']=='reserved' for r in records)
        remaining_exploration = limits['max_verifications']-1-sum(r['purpose']=='explore' for r in records)
        expired = elapsed >= limits['wall_clock_s']
        if final is not None:
            if final['status']=='reserved':
                phase, stop_reason = 'finalizing', 'independent_final_in_flight'
            elif (final['status']=='completed' and final.get('full_l2_passed') is True
                  and final.get('finished_after_deadline') is False):
                phase, stop_reason = 'final_complete', 'independent_final_completed'
            else:
                phase, stop_reason = 'final_failed', (
                    'final_deadline' if final.get('finished_after_deadline') is True else
                    'final_'+final['status'] if final['status']!='completed' else 'final_l2_not_established')
        elif expired:
            phase, stop_reason = 'exhausted', 'host_task_deadline'
        elif pending:
            phase, stop_reason = 'verifying', None
        elif remaining_exploration <= 0:
            phase, stop_reason = ('awaiting_final', 'exploration_complete_final_reserved') if correct else (
                'exhausted', 'exploration_exhausted_without_correct_candidate')
        else:
            phase, stop_reason = 'awaiting_host', None
        development_closed = final is not None or expired or remaining_exploration <= 0
        return dict(schema=SCHEMA, task_id=state['task_id'], contract_hash=self.contract_hash,
            run_dir=str(self.run),
            phase=phase, stop_reason=stop_reason, development_closed=development_closed,
            terminal=phase in ('final_complete','final_failed','exhausted'),
            terminal_scope='task_budget_only_not_worker_or_provider_stop',
            can_reserve_exploration=not development_closed,
            can_reserve_final=final is None and not expired and bool(correct) and not pending,
            limits=limits, elapsed_s=elapsed, remaining_s=max(0.0, limits['wall_clock_s']-elapsed),
            deadline_epoch=state['started_epoch']+limits['wall_clock_s'],
            expired=expired, reserved=len(records),
            remaining_verifications=limits['max_verifications']-len(records),
            remaining_exploration=remaining_exploration,
            remaining_final=1-sum(r['purpose']=='final' for r in records),
            unfinished=pending, records=records,
            final_verification=({key:final.get(key) for key in ('id','source_hash','status','evidence',
                'full_l2_passed','finished_after_deadline')} if final is not None else None),
            last_correct_candidate=(dict(source_hash=last['source_hash'], evidence=last['evidence'],
                source=last.get('source'), report=last.get('report'), verification_id=last['id'],
                scope='latest_complete_l2_only_not_performance_best') if last else None),
            policy_hash=state['policy_hash'],
            model_budget=self._model_budget(policy, state),
            wall_clock_scope='task_elapsed_across_calls; worker supervisor enforces in-flight deadline',
            reservation_unit='verification_dispatches_no_refund_not_gpu_seconds')

    def snapshot(self, *, readonly=False):
        with _locked(self.run):
            policy, state = self._load()
            if not readonly:
                state['max_elapsed_s'] = self._elapsed(state)
                _write_state(self.run,state)
            return self._snapshot(policy, state)

    def _model_budget(self, policy, state):
        limits = policy['limits']
        base = dict(owner='host', calls=None, tokens=None, cost='unknown',
                    task_id=state['task_id'], scope='native_main_model_dispatches_after_successful_prepare',
                    preparation_request='excluded_before_successful_task_binding')
        if not all(name in limits for name in MODEL_LIMIT_FIELDS):
            return dict(base, enforcement='legacy_policy_without_model_limits',
                        receipt_verified=False, limits=None)
        model_limits = {name: limits[name] for name in MODEL_LIMIT_FIELDS}
        path = self.run/MODEL_STATE
        if not path.exists() and not path.is_symlink():
            return dict(base, enforcement='not_bound', receipt_verified=False,
                        limits=model_limits)
        try:
            receipt = _read(path)
            def integer(value, minimum=0):
                return type(value) is int and minimum <= value <= 9007199254740991
            def integer_json(value):
                if value is None or type(value) in (str, bool, int):
                    return type(value) is not int or -9007199254740991 <= value <= 9007199254740991
                if type(value) is list:
                    return all(integer_json(item) for item in value)
                return (type(value) is dict and all(type(key) is str and integer_json(item)
                        for key, item in value.items()))
            if (not integer_json(receipt)
                    or receipt.get('state_hash') != digest({k:v for k,v in receipt.items() if k!='state_hash'})
                    or receipt.get('schema') != 'operator-host-model-budget-v1'
                    or receipt.get('task_id') != state['task_id']
                    or receipt.get('contract_hash') != self.contract_hash
                    or receipt.get('policy_hash') != state['policy_hash']
                    or receipt.get('policy_bytes_sha256') != hashlib.sha256((self.run/POLICY).read_bytes()).hexdigest()
                    or receipt.get('limits') != model_limits
                    or receipt.get('scope') != base['scope']):
                raise ValueError('model receipt binding or digest differs')
            lease, records = receipt.get('lease'), receipt.get('records')
            if (type(lease) is not dict or not isinstance(lease.get('id'),str) or not lease['id']
                    or not isinstance(lease.get('session_id'),str) or not lease['session_id']
                    or not integer(lease.get('owner_pid'),1) or not integer(lease.get('generation'),1)
                    or not integer(lease.get('bound_at')) or type(records) is not list
                    or len(records) > limits['max_model_calls']):
                raise ValueError('invalid model lease or calls')
            usage_fields = ('inputTokens','outputTokens','totalTokens','cacheReadTokens','cacheWriteTokens')
            known_usage = []
            for index, record in enumerate(records,1):
                if (type(record) is not dict or type(record.get('id')) is not int or record['id']!=index
                        or record.get('status') not in ('reserved','answered','failed','cancelled','abandoned')
                        or any(type(record.get(key)) is not str or not record[key]
                               for key in ('request_id','session_id','lease_id','provider','model'))
                        or not integer(record.get('owner_pid'),1)
                        or not integer(record.get('reserved_at'))
                        or not integer(record.get('max_output_tokens_reserved'),1)
                        or record['max_output_tokens_reserved'] > limits['max_output_tokens_per_call']
                        or (record['status']=='reserved' and record.get('finished_at') is not None)
                        or (record['status']!='reserved' and not integer(record.get('finished_at')))):
                    raise ValueError('invalid model reservation')
                usage = record.get('usage')
                if usage is not None:
                    if (type(usage) is not dict or set(usage)-set(usage_fields)
                            or any(not integer(value) for value in usage.values())):
                        raise ValueError('invalid model usage counters')
                    if all(key in usage for key in ('inputTokens','outputTokens','totalTokens')):
                        base_tokens=usage['inputTokens']+usage['outputTokens']
                        cache_tokens=usage.get('cacheReadTokens',0)+usage.get('cacheWriteTokens',0)
                        if usage['totalTokens'] not in (base_tokens,base_tokens+cache_tokens):
                            raise ValueError('inconsistent model usage counters')
                    if (record['status']!='reserved' and integer(usage.get('inputTokens'))
                            and integer(usage.get('outputTokens')) and any(usage.values())):
                        known_usage.append(usage)
            reserved_output = sum(record['max_output_tokens_reserved'] for record in records)
            if reserved_output > limits['total_output_token_reservations']:
                raise ValueError('model output reservations exceed frozen policy')
            known_output = sum(usage['outputTokens'] for usage in known_usage)
            known_total = sum(usage.get('totalTokens',0) for usage in known_usage)
            all_known = len(known_usage)==len(records)
            return dict(base, enforcement='host_enforced_receipt', receipt_verified=True,
                verification_scope='Frozen binding, integer self-hash and arithmetic only; '
                                   'not proof of current hook installation, model attention or provider stop.',
                limits=model_limits, lease=lease,
                calls=dict(reserved=len(records),remaining=limits['max_model_calls']-len(records),
                    limit=limits['max_model_calls'],unfinished=sum(record['status']=='reserved' for record in records)),
                tokens=dict(output_reserved=reserved_output,
                    remaining_reservations=limits['total_output_token_reservations']-reserved_output,
                    reservation_limit=limits['total_output_token_reservations'],
                    known_api_output_tokens=known_output, api_output_tokens=known_output if all_known else None,
                    known_api_total_tokens=known_total,
                    api_total_tokens=known_total if all_known and all('totalTokens' in usage for usage in known_usage) else None),
                api_usage_known_calls=len(known_usage),api_usage_unknown_calls=len(records)-len(known_usage))
        except (HostTaskBudgetError, OSError, ValueError, TypeError, KeyError):
            return dict(base,enforcement='receipt_invalid',receipt_verified=False,limits=model_limits,
                        receipt_error='host_model_receipt_invalid')

    def reserve(self, source_hash, *, purpose='explore', evidence=None, owner_invocation_id=None):
        _hash(source_hash, 'source_hash')
        owner_invocation_id = _owner(owner_invocation_id)
        if purpose not in ('explore','final'):
            raise HostTaskBudgetError('host_task_purpose', 'purpose must be explore or final')
        if evidence is not None:
            evidence = self._evidence_path(evidence)
        with _locked(self.run):
            policy, state = self._load()
            elapsed = self._elapsed(state)
            snapshot = self._snapshot(policy, state)
            if snapshot['final_verification'] is not None:
                raise HostTaskBudgetError('host_task_finalized',
                    'independent final was reserved; no further verification dispatches are permitted')
            if snapshot['expired']:
                raise HostTaskBudgetError('host_task_deadline', 'host task wall clock exhausted')
            if snapshot['remaining_verifications'] <= 0 or (
                    purpose=='explore' and snapshot['remaining_exploration'] <= 0):
                raise HostTaskBudgetError('host_verification_budget', 'verification budget exhausted; final reserve is protected')
            if purpose=='final':
                if snapshot['unfinished']:
                    raise HostTaskBudgetError('host_final_in_flight',
                        'settle all in-flight exploration reservations before independent final')
                if snapshot['remaining_final'] <= 0:
                    raise HostTaskBudgetError('host_final_budget', 'independent final reservation already spent')
                if not any(r['source_hash']==source_hash and _timely_l2(r) for r in state['records']):
                    raise HostTaskBudgetError('host_final_source', 'final requires a prior complete L2 pass of this source')
            record = dict(id=len(state['records'])+1, source_hash=source_hash, purpose=purpose,
                          status='reserved', reserved_epoch=time.time(),
                          reserved_monotonic=monotonic(), evidence=evidence,
                          full_l2_passed=None, owner_invocation_id=owner_invocation_id)
            state['records'].append(record)
            state['max_elapsed_s'] = elapsed
            _write_state(self.run,state)
            return dict(record, remaining_s=snapshot['remaining_s'])

    def _evidence_path(self, value):
        original = Path(value).absolute()
        path = original.resolve()
        if self.run not in path.parents or path.name!='result.json' or any(
                p.is_symlink() for p in (original, *original.parents)
                if self.run==p.resolve() or self.run in p.resolve().parents):
            raise HostTaskBudgetError('host_task_evidence', 'ordinary result.json inside the owned run required')
        return str(path)

    def _completed_result(self, record, result, evidence, required_case_ids, records):
        path = Path(self._evidence_path(evidence))
        if any(r.get('evidence')==str(path) and r['id']!=record['id'] for r in records):
            raise HostTaskBudgetError('host_task_evidence', 'each reservation needs new independent evidence')
        full, complete = _read(path), _read(path.parent/'complete.json')
        binding = full.get('binding', {})
        if (full.get('evidence_complete') is not True or complete.get('evidence_complete') is not True
                or binding != complete.get('binding')
                or binding.get('source_hash') != record['source_hash']
                or binding.get('contract_hash') != self.contract_hash
                or full.get('source_hash') != record['source_hash']
                or full.get('contract_hash') != self.contract_hash
                or type(full.get('created_at')) not in (int,float)
                or full['created_at'] < record['reserved_epoch']):
            raise HostTaskBudgetError('host_task_evidence', 'completed evidence binding or reservation time differs')
        members = complete.get('files_sha256')
        if type(members) is not dict or 'result.json' not in members:
            raise HostTaskBudgetError('host_task_evidence', 'complete evidence manifest required')
        for name, expected in members.items():
            relative = Path(name)
            member = path.parent/relative
            if relative.is_absolute() or '..' in relative.parts or member.is_symlink() or not member.is_file():
                raise HostTaskBudgetError('host_task_evidence', 'invalid evidence manifest member')
            if hashlib.sha256(member.read_bytes()).hexdigest()!=expected:
                raise HostTaskBudgetError('host_task_evidence', 'completed evidence member changed')
        if result is not None and (type(result) is not dict or any(
                result.get(k)!=full.get(k) for k in ('source_hash','contract_hash','levels'))):
            raise HostTaskBudgetError('host_task_evidence', 'returned result differs from persisted evidence')
        l2 = [v for v in full.get('verdicts', []) if v.get('level')=='L2']
        correctness = l2[0].get('metrics',{}).get('correctness',{}) if len(l2)==1 else {}
        cases = correctness.get('cases', [])
        full_l2 = (full.get('levels',{}).get('L2') is True and len(l2)==1
            and l2[0].get('passed') is True and correctness.get('verdict')=='passed'
            and type(cases) is list and bool(required_case_ids)
            and all(type(c) is dict and type(c.get('case_id')) is str for c in cases)
            and sorted(c['case_id'] for c in cases)==required_case_ids
            and all(c.get('verdict')=='passed' for c in cases)
            and correctness.get('cases_total',len(cases))==len(cases))
        return full, bool(full_l2), str(path)

    def finish(self, reservation_id, *, outcome='completed', result=None, evidence=None):
        if type(reservation_id) is not int or outcome not in ('completed','failed','cancelled'):
            raise HostTaskBudgetError('host_task_reservation', 'valid reservation and terminal outcome required')
        with _locked(self.run):
            policy, state = self._load()
            matches = [r for r in state['records'] if r['id']==reservation_id]
            if len(matches)!=1 or matches[0]['status']!='reserved':
                raise HostTaskBudgetError('host_task_reservation', 'reservation absent or already completed')
            record = matches[0]
            evidence = evidence or (result.get('evidence') if type(result) is dict else None) or record.get('evidence')
            if outcome=='completed':
                if evidence is None:
                    raise HostTaskBudgetError('host_task_evidence', 'completed verification needs persisted evidence')
                full, passed, evidence = self._completed_result(record,result,evidence,
                    policy['required_case_ids'],state['records'])
                record.update(full_l2_passed=passed, evidence=evidence,
                              source=full.get('source'), report=full.get('report'),
                              validity=full.get('validity'))
            elif evidence is not None:
                record['evidence'] = self._evidence_path(evidence)
            state['max_elapsed_s'] = self._elapsed(state)
            record.update(status=outcome, finished_epoch=time.time(),
                finished_after_deadline=state['max_elapsed_s'] >= policy['limits']['wall_clock_s'])
            _write_state(self.run,state)
            return self._snapshot(policy,state)

    def settle_owner(self, invocation_id, *, outcome='cancelled', reject_completed=False):
        """Settle only this owner's unfinished work after its supervisor reaped it.

        This internal API cannot independently prove worker death. It must be
        called by the owning supervisor after confirming worker_reaped. It does
        not complete evidence, refund reservations or establish remote shutdown.
        Repeated cleanup leaves settled records intact. With reject_completed,
        a response rejected by the supervisor also loses accepted-candidate
        credit while its verification facts and immutable artifacts remain.
        A cooperative signal handler may have recorded failed before the
        supervisor decides cancelled; that un-settled failure is normalized too.
        """
        invocation_id = _owner(invocation_id,required=True)
        if outcome not in ('cancelled','failed'):
            raise HostTaskBudgetError('host_task_reservation', 'owner settlement must be cancelled or failed')
        if type(reject_completed) is not bool:
            raise HostTaskBudgetError('host_task_reservation', 'reject_completed must be boolean')
        with _locked(self.run):
            policy,state = self._load()
            elapsed = self._elapsed(state)
            settled=[]
            for record in state['records']:
                cooperative_failure = (reject_completed and outcome=='cancelled'
                    and record['status']=='failed' and record.get('settled_by') is None)
                if (record.get('owner_invocation_id')==invocation_id
                        and (record['status']=='reserved'
                             or reject_completed and record['status']=='completed'
                             or cooperative_failure)):
                    if record['status'] in ('completed','failed'):
                        record['verification_status_before_settlement']=record['status']
                        record['verification_finished_epoch']=record.get('finished_epoch')
                        record['verification_finished_after_deadline']=record.get('finished_after_deadline')
                    record.update(status=outcome, finished_epoch=time.time(),
                        finished_after_deadline=elapsed >= policy['limits']['wall_clock_s'],
                        accepted_by_supervisor=False, settled_by='owning_supervisor',
                        settlement_basis='worker_reaped_response_rejected' if reject_completed else 'worker_reaped')
                    settled.append(record['id'])
            state['max_elapsed_s']=elapsed
            _write_state(self.run,state)
            return dict(self._snapshot(policy,state), settled_owner_invocation_id=invocation_id,
                        settled_reservation_ids=settled)
