"""Pre-dispatch LLM campaign limits, independent of the usage/cost estimate.

Reservations never refund. A fresh journal is required; old journals are audit
data, not resumable credit. These limits bound calls/bytes/output reservations,
NOT exact input tokens, provider billing or server-side cancellation.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
from ..execute.task_clock import monotonic
import tempfile

from ..core.run_contract import atomic_json, json_value


class LLMCallRefused(RuntimeError):
    """No implicit fallback/retry is permitted after this failure."""


class ProductionBudgetExhausted(LLMCallRefused):
    """A production task limit, with a stable controller-facing stop reason."""

    def __init__(self, reason, message=None):
        self.reason = reason
        super().__init__(message or reason)


def strict_object(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result

    def constant(_):
        raise ValueError('nonfinite JSON')

    result = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(result, dict):
        raise ValueError('JSON object required')
    return result


def validated_usage(raw):
    """Do not coerce bool/fractional/negative counters into plausible costs."""
    usage = {} if raw is None else raw
    if type(usage) is not dict:
        raise ValueError('invalid usage')
    for name in ('prompt_tokens', 'input_tokens', 'completion_tokens', 'output_tokens',
                 'total_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens'):
        if name in usage and (type(usage[name]) is not int or usage[name] < 0):
            raise ValueError('invalid usage counter')
    for name in ('prompt_tokens_details', 'input_tokens_details'):
        details = usage.get(name)
        details = {} if details is None else details
        if type(details) is not dict:
            raise ValueError('invalid usage details')
        cached = details.get('cached_tokens', 0)
        if type(cached) is not int or cached < 0:
            raise ValueError('invalid cached usage')
    complete = (any(n in usage for n in ('prompt_tokens', 'input_tokens'))
                and any(n in usage for n in ('completion_tokens', 'output_tokens')))
    return usage, 'api_usage' if complete else 'estimated'


_USAGE_COUNTERS = ('prompt_tokens', 'input_tokens', 'completion_tokens', 'output_tokens',
                   'total_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens')


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='microseconds').replace('+00:00', 'Z')


def _api_usage_evidence(usage, source):
    """Only validated, complete API counters cross the production audit boundary.

    Provider extensions can contain arbitrary text. Never persist the raw usage
    object, and never turn CostTracker's character estimates into API evidence.
    """
    if usage is None:
        return None, source or 'not_observed'
    try:
        checked, inferred = validated_usage(usage)
    except (ValueError, TypeError):
        return None, 'invalid_api_usage'
    if inferred != 'api_usage' or source not in (None, 'api_usage'):
        return None, source or inferred
    counters = {name: checked[name] for name in _USAGE_COUNTERS if name in checked}
    for holder in ('prompt_tokens_details', 'input_tokens_details'):
        details = checked.get(holder)
        if type(details) is dict and 'cached_tokens' in details:
            counters[holder] = {'cached_tokens': details['cached_tokens']}
    return counters, 'api_usage'


@dataclass(frozen=True)
class LLMCallLimits:
    max_calls: int
    max_output_tokens: int
    total_output_token_reservations: int
    max_request_bytes: int = 65536
    max_response_bytes: int = 131072
    call_seconds: float = 60.
    campaign_seconds: float = 1800.
    allowed_callers: tuple = ('SingleCandidateCoder',)

    def __post_init__(self):
        for name in ('max_calls', 'max_output_tokens', 'total_output_token_reservations',
                     'max_request_bytes', 'max_response_bytes'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(name + ' must be a positive integer')
        for name in ('call_seconds', 'campaign_seconds'):
            value = getattr(self, name)
            if (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
                raise ValueError(name + ' must be finite and positive')
        if (type(self.allowed_callers) is not tuple or not self.allowed_callers
                or any(type(x) is not str or not x for x in self.allowed_callers)
                or len(set(self.allowed_callers)) != len(self.allowed_callers)):
            raise ValueError('allowed_callers must be a nonempty unique tuple')


class LLMCallBudget:
    """One client/campaign lifetime, thread-safe and durable before any HTTP.

    The controller and CostTracker cannot reset this ledger. On any dispatched
    failure (including invalid output), poison the campaign so an agent's outer
    retry loop cannot silently spend another request. Journal I/O failure also
    poisons it. Same-UID hostile rewrites are outside this integrity boundary.
    """

    @property
    def strict_response(self):
        return True

    @property
    def poison_dispatched_failure(self):
        return True

    def __init__(self, limits: LLMCallLimits, journal_dir):
        if not isinstance(limits, LLMCallLimits):
            raise TypeError('LLMCallLimits required')
        self._limits = limits
        self.root = Path(journal_dir)
        self.root.mkdir(parents=True, exist_ok=False, mode=0o700)
        atomic_json(self.root / 'limits.json', asdict(limits), readonly=True)
        self._deadline = self._now() + limits.campaign_seconds
        self._lock = threading.RLock()
        self._identity = None
        self._records = []
        self._aborted = None

    @property
    def limits(self):
        return self._limits

    def bind(self, identity):
        with self._lock:
            if self._identity is not None:
                raise LLMCallRefused('budget already bound; create a fresh campaign')
            self._identity = copy.deepcopy(identity)
            try:
                path = self.root / 'identity.json'
                if path.exists() or path.is_symlink():
                    raise ValueError('existing identity evidence')
                atomic_json(self.root / 'identity.json', identity, readonly=True)
            except Exception:
                self._aborted = 'journal_failure'
                raise LLMCallRefused('identity journal failed; no dispatch') from None

    def _check_journal(self):
        try:
            expected = [('limits.json', json_value(asdict(self.limits))),
                        ('identity.json', self._identity)]
            expected += [('call-%03d.json' % r['call'], r) for r in self._records]
            for name, value in expected:
                path = self.root / name
                if path.is_symlink() or strict_object(path.read_text()) != value:
                    raise ValueError('journal drift')
        except Exception:
            self._aborted = 'journal_drift'
            raise LLMCallRefused('frozen budget journal drift; no dispatch') from None

    def reserve(self, identity, body_text, caller, timeout):
        with self._lock:
            if self._aborted:
                raise LLMCallRefused('campaign stopped: ' + self._aborted)
            if self._identity is None or identity != self._identity:
                raise LLMCallRefused('frozen model configuration drift')
            self._check_journal()
            if caller not in self.limits.allowed_callers:
                raise LLMCallRefused('undeclared model caller')
            if len(self._records) >= self.limits.max_calls:
                self._exhausted('llm_call_budget', 'call budget exhausted before dispatch')
            body = strict_object(body_text)
            output = body.get('max_tokens')
            if (type(output) is not int or not 0 < output <= self.limits.max_output_tokens):
                raise LLMCallRefused('per-call output reservation exceeds limit')
            if sum(r['output_tokens_reserved'] for r in self._records) + output > self.limits.total_output_token_reservations:
                self._exhausted('llm_output_budget', 'total output reservations exhausted before dispatch')
            size = len(body_text.encode('utf-8'))
            if size > self.limits.max_request_bytes:
                raise LLMCallRefused('request byte limit before dispatch')
            if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
                raise LLMCallRefused('invalid model timeout')
            seconds = self._timeout_seconds(timeout)
            record = dict(call=len(self._records) + 1, caller=caller, status='reserved',
                          request=self._request_fields(body), request_bytes=size,
                          request_sha256=hashlib.sha256(body_text.encode()).hexdigest(),
                          output_tokens_reserved=output, timeout_seconds=seconds)
            record.update(self._reservation_fields())
            self._records.append(record)
            path = self.root / ('call-%03d.json' % record['call'])
            try:
                # mkstemp owns a 0600 inode; link publishes it exclusively,
                # without chmod/opening a write window on existing evidence.
                # Existing files AND symlinks make link fail before dispatch.
                fd, tmp = tempfile.mkstemp(prefix=path.name + '.', dir=str(self.root))
                try:
                    with os.fdopen(fd, 'w', encoding='utf-8') as f:
                        json.dump(record, f, ensure_ascii=False, allow_nan=False)
                        f.flush()
                        os.fsync(f.fileno())
                    os.link(tmp, path)
                finally:
                    os.unlink(tmp)
                fd = os.open(str(self.root), os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            except Exception:
                self._aborted = 'journal_failure'
                raise LLMCallRefused('reservation journal failed; no dispatch') from None
            return record['call'], seconds

    def _exhausted(self, reason, message, *, before_dispatch=True):
        raise LLMCallRefused(message)

    def _now(self):
        return time.monotonic()

    def _timeout_seconds(self, timeout):
        seconds = min(timeout, self.limits.call_seconds)
        if self._now() + seconds > self._deadline:
            raise LLMCallRefused('insufficient campaign time before dispatch')
        return seconds

    def _response_fields(self, response):
        return dict(response_text=response,
                    response_sha256=hashlib.sha256(response.encode()).hexdigest(),
                    response_bytes=len(response.encode()))

    def _request_fields(self, body):
        return body

    def _reservation_fields(self):
        # Strict single-candidate journals retain their original exact format.
        return {}

    def _completion_fields(self, usage, usage_source):
        return {}

    def finish(self, call, *, response=None, outcome='answered', wall_seconds=0., usage_source=None,
               usage=None):
        with self._lock:
            record = self._records[call - 1]
            if record['status'] != 'reserved':
                raise LLMCallRefused('attempt already finalized')
            self._check_journal()
            record.update(status=outcome, wall_seconds=wall_seconds, usage_source=usage_source)
            record.update(self._completion_fields(usage, usage_source))
            if response is not None:
                record.update(self._response_fields(response))
            if outcome != 'answered' and self.poison_dispatched_failure:
                self._aborted = outcome
            try:
                atomic_json(self.root / ('call-%03d.json' % call), record, readonly=True)
            except Exception:
                self._aborted = 'journal_failure'
                raise LLMCallRefused('result journal failed; campaign stopped') from None

    def abort(self, reason):
        with self._lock:
            self._aborted = reason
            try:
                atomic_json(self.root / 'abort.json', {'reason': reason}, readonly=True)
            except Exception:
                self._aborted = 'journal_failure'
                raise LLMCallRefused('abort journal failed; campaign stopped') from None

    def summary(self):
        with self._lock:
            return copy.deepcopy(dict(schema='llm-call-budget-v1', scope='campaign_lifetime',
                limits=asdict(self.limits), identity=self._identity, aborted=self._aborted,
                attempts=self._records, reserved_calls=len(self._records),
                output_tokens_reserved=sum(r['output_tokens_reserved'] for r in self._records),
                exact_input_token_cap=False, provider_billing_cap=False,
                retries=False, fallback=False, redirects=False))


class ProductionLLMCallBudget(LLMCallBudget):
    """One production task, including every explicitly requested retry.

    Ordinary request/output failures consume their reservations but do not
    revoke the remaining task allowance. Configuration/journal failures still
    fail closed. There are no transport retries, redirects or hidden fallback.
    Responses are hashed instead of retained as a second source-writing path.
    """

    @property
    def strict_response(self):
        return False

    @property
    def poison_dispatched_failure(self):
        return False

    def __init__(self, limits, journal_dir, *, deadline=None):
        super().__init__(limits, journal_dir)
        if deadline is not None:
            if type(deadline) not in (int, float) or not math.isfinite(deadline):
                raise ValueError('finite monotonic deadline required')
            self._deadline = min(self._deadline, deadline)
        self._thread_calls = {}
        self._refusal_reason = None

    def _now(self):
        return monotonic()

    def _reservation_fields(self):
        return dict(reserved_at_utc=_utc_now(), reserved_monotonic=self._now(),
            dispatched_at_utc=None, dispatched_monotonic=None,
            completed_at_utc=None, completed_monotonic=None,
            api_usage=None, usage_source='not_observed')

    def _completion_fields(self, usage, usage_source):
        counters, source = _api_usage_evidence(usage, usage_source)
        return dict(completed_at_utc=_utc_now(), completed_monotonic=self._now(),
                    api_usage=counters, usage_source=source)

    def mark_dispatched(self, call):
        """Durably record a transport handoff attempt, not provider acceptance.

        A reservation can remain undispatched after cancellation or journal
        failure. Publishing this boundary separately prevents counting either as
        an observed API response or an exact provider bill.
        """
        with self._lock:
            self._check_journal()
            record = self._records[call - 1]
            if record['status'] != 'reserved' or record['dispatched_monotonic'] is not None:
                raise LLMCallRefused('attempt already dispatched or finalized')
            dispatched = dict(record, dispatched_at_utc=_utc_now(),
                              dispatched_monotonic=self._now())
            try:
                atomic_json(self.root / ('call-%03d.json' % call), dispatched, readonly=True)
            except Exception:
                self._aborted = 'journal_failure'
                self._persist_summary()
                raise LLMCallRefused('result journal failed; no dispatch') from None
            self._records[call - 1] = dispatched
            self._persist_summary()

    def _persist_summary(self):
        """Publish the in-memory index before dispatch and after each outcome.

        If the index cannot be refreshed, remove any stale readable snapshot
        rather than let a supervisor mistake the old counters for a full ledger.
        """
        try:
            atomic_json(self.root / 'summary.json', self.summary(), readonly=True)
        except Exception:
            self._aborted = 'journal_failure'
            try:
                atomic_json(self.root / 'summary.json', self.summary(), readonly=True)
            except Exception:
                try:
                    (self.root / 'summary.json').unlink(missing_ok=True)
                except OSError:
                    pass
            try:
                atomic_json(self.root / 'abort.json', dict(reason=self._aborted), readonly=True)
            except Exception:
                pass
            raise LLMCallRefused('summary journal failed; campaign stopped') from None

    def bind(self, identity):
        with self._lock:
            try:
                super().bind(identity)
            except Exception:
                self._persist_summary()
                raise
            self._persist_summary()

    def _check_journal(self):
        try:
            super()._check_journal()
        except LLMCallRefused:
            self._persist_summary()
            raise

    def remaining_seconds(self):
        return max(0., self._deadline - self._now())

    def _exhausted(self, reason, message, *, before_dispatch=True):
        with self._lock:
            self._refusal_reason = reason
            try:
                atomic_json(self.root / 'refusal.json', dict(reason=reason,
                    before_dispatch=before_dispatch, reserved_calls=len(self._records),
                    remaining_seconds=self.remaining_seconds()), readonly=True)
            except Exception:
                self._aborted = 'journal_failure'
                self._persist_summary()
                raise LLMCallRefused('refusal journal failed; campaign stopped') from None
            self._persist_summary()
        raise ProductionBudgetExhausted(reason, message) from None

    def _timeout_seconds(self, timeout):
        remaining = self.remaining_seconds()
        if remaining <= 0:
            self._exhausted('task_wall_clock', 'task deadline exhausted before dispatch')
        return min(timeout, self.limits.call_seconds, remaining)

    def reserve(self, identity, body_text, caller, timeout):
        with self._lock:
            try:
                call, seconds = super().reserve(identity, body_text, caller, timeout)
            except Exception:
                self._persist_summary()
                raise
            self._thread_calls[threading.get_ident()] = call
            self._persist_summary()
            remaining = self.remaining_seconds()
            if remaining <= 0:
                self.finish(call, outcome='task_wall_clock')
                self._exhausted('task_wall_clock', 'task deadline exhausted before dispatch')
            seconds = min(seconds, remaining)
            return call, seconds

    def finish(self, call, *, response=None, outcome='answered', wall_seconds=0., usage_source=None,
               usage=None):
        with self._lock:
            try:
                super().finish(call, response=response, outcome=outcome,
                               wall_seconds=wall_seconds, usage_source=usage_source, usage=usage)
            except Exception:
                self._persist_summary()
                raise
            self._persist_summary()

    def _request_fields(self, body):
        return copy.deepcopy({name: body[name] for name in
            ('model', 'temperature', 'max_tokens', 'stream', 'chat_template_kwargs', 'enable_thinking')
            if name in body})

    def _response_fields(self, response):
        return dict(response_sha256=hashlib.sha256(response.encode()).hexdigest(),
                    response_bytes=len(response.encode()))

    def abort(self, reason):
        if reason != 'invalid_json':
            with self._lock:
                try:
                    super().abort(reason)
                except Exception:
                    self._persist_summary()
                    raise
                self._persist_summary()
            return
        # chat_json validates the content after the HTTP envelope has already
        # been finalized. Attribute a parse failure to this thread's call,
        # without poisoning another agent's remaining production allowance.
        with self._lock:
            self._check_journal()
            call = self._thread_calls.get(threading.get_ident())
            if call is None:
                raise LLMCallRefused('no production call for invalid output')
            record = self._records[call - 1]
            record.update(status='invalid_json')
            try:
                atomic_json(self.root / ('call-%03d.json' % call), record, readonly=True)
            except Exception:
                self._aborted = 'journal_failure'
                self._persist_summary()
                raise LLMCallRefused('result journal failed; campaign stopped') from None
            self._persist_summary()

    def summary(self):
        summary = super().summary()
        records = summary['attempts']
        known = [record['api_usage'] for record in records if record.get('api_usage') is not None]
        known_totals, field_unknown_calls = {}, {}
        for field in _USAGE_COUNTERS:
            values = [usage[field] for usage in known if field in usage]
            known_totals[field] = sum(values) if values else None
            field_unknown_calls[field] = len(records) - len(values)
        dispatched = sum(record.get('dispatched_monotonic') is not None for record in records)
        completed = sum(record.get('completed_monotonic') is not None for record in records)
        summary.update(policy='production_task', explicit_retries=True,
            failures_refunded=False, response_retention='hash_only',
            request_retention='metadata_and_hash_only', stop_reason=self._refusal_reason,
            remaining_calls=max(0, self.limits.max_calls - summary['reserved_calls']),
            remaining_seconds=self.remaining_seconds(),
            dispatched_calls=dispatched, completed_calls=completed,
            unfinished_calls=len(records) - completed,
            undispatched_reservations=len(records) - dispatched,
            dispatch_scope='transport_handoff_attempts_not_provider_acceptance',
            api_usage=dict(source='validated_api_counters_only', known_calls=len(known),
                unknown_calls=len(records) - len(known), known_totals=known_totals,
                field_unknown_calls=field_unknown_calls,
                totals={field: total if field_unknown_calls[field] == 0 else None
                        for field, total in known_totals.items()}))
        return summary
