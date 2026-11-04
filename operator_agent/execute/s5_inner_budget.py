"""Experimental S5 bridge: a production inner ledger and a shared call pool.

Not enabled by the normal CLI. Both ledgers reserve before network handoff;
partial failures consume any successful reservation and never refund it. This
only covers callers that actually use this budget, not arbitrary model traffic.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path

from ..execute.llm_budget import (ProductionLLMCallBudget, ProductionBudgetExhausted,
                                 LLMCallRefused, strict_object)
from ..execute import s5_shared_budget as shared


_SHARED_LIMIT_REASONS = {
    'call_budget_exhausted': 'llm_call_budget',
    'wall_clock_exhausted': 'task_wall_clock',
    'per_call_output_exhausted': 'llm_output_budget',
    'output_budget_exhausted': 'llm_output_budget',
}


def _provider_counters(usage, source):
    if source != 'api_usage' or type(usage) is not dict:
        return None
    prompt = usage.get('prompt_tokens')
    completion = usage.get('completion_tokens')
    if type(prompt) is not int or prompt < 0 or type(completion) is not int or completion < 0:
        return None
    out = dict(prompt_tokens=prompt, completion_tokens=completion)
    if 'total_tokens' in usage:
        if type(usage['total_tokens']) is not int or usage['total_tokens'] != prompt+completion:
            return None
        out['total_tokens'] = usage['total_tokens']
    details = usage.get('prompt_tokens_details', usage.get('input_tokens_details'))
    if isinstance(details, dict) and 'cached_tokens' in details:
        cache = details['cached_tokens']
        if type(cache) is not int or not 0 <= cache <= prompt:
            return None
        out['cache_read_tokens'] = cache
    return out


class SharedProductionBudget(ProductionLLMCallBudget):
    """One task's inner production journal plus S5 shared outer/inner pool."""

    def __init__(self, limits, journal_dir, *, shared_path, task_id, deadline=None):
        path = Path(shared_path)
        try:
            if path.is_symlink():
                raise OSError('symlink')
            stat = path.stat()
            info = shared.snapshot(path, expected_task_id=task_id)
            after = path.stat()
            if path.is_symlink() or (stat.st_dev, stat.st_ino) != (after.st_dev, after.st_ino):
                raise OSError('changed ledger')
        except (shared.BudgetError, OSError):
            raise LLMCallRefused('shared ledger unavailable before initialization') from None
        if info.get('task_id') != task_id:
            raise LLMCallRefused('shared ledger task identity mismatch')
        self.shared_path = path
        self.shared_task_id = task_id
        self._shared_file_identity = (stat.st_dev, stat.st_ino)
        self._shared_calls = {}
        self._shared_dispatched = set()
        try:
            super().__init__(limits, journal_dir, deadline=deadline)
        except Exception:
            stopped = False
            try:
                self._check_shared_identity()
                shared.abort(self.shared_path, reason='audit_failure', expected_task_id=self.shared_task_id)
                stopped = True
            except Exception:
                pass
            raise LLMCallRefused('local budget initialization failed; ' +
                ('campaign stop recorded' if stopped else 'global stop unproven')) from None

    def _check_shared_identity(self):
        try:
            st = self.shared_path.stat()
            if self.shared_path.is_symlink() or (st.st_dev, st.st_ino) != self._shared_file_identity:
                raise ValueError('changed ledger file')
        except (OSError, ValueError):
            raise LLMCallRefused('shared ledger identity unavailable or changed') from None

    def _poison(self, phase):
        """Stop this client and attempt a durable stop for all participants.

        A missing/replaced database cannot safely receive an abort. The caller
        must notify its supervisor, which must stop owned workers independently.
        """
        self._aborted = 'shared_' + phase
        stopped = False
        try:
            self._check_shared_identity()
            shared.abort(self.shared_path, reason='audit_failure', expected_task_id=self.shared_task_id)
            stopped = True
        except Exception:
            pass
        try:
            self._persist_summary()
        except Exception:
            pass
        raise LLMCallRefused('shared audit failure during ' + phase +
            ('; campaign stop recorded' if stopped else '; global stop unproven')) from None

    def bind(self, identity):
        try:
            return super().bind(identity)
        except Exception:
            if self._aborted:
                self._poison('local_binding')
            raise

    def reserve(self, identity, body_text, caller, timeout):
        with self._lock:
            if self._aborted:
                raise LLMCallRefused('campaign stopped: ' + self._aborted)
            try:
                self._check_shared_identity()
            except LLMCallRefused:
                self._poison('identity')
            try:
                body = strict_object(body_text)
                token_cap = body.get('max_tokens')
                req_hash = hashlib.sha256(body_text.encode('utf-8')).hexdigest()
                item = shared.reserve(self.shared_path, participant='inner', caller=caller,
                                      max_output_tokens=token_cap, request_sha256=req_hash,
                                      expected_task_id=self.shared_task_id)
            except shared.BudgetError as exc:
                reason = _SHARED_LIMIT_REASONS.get(exc.reason)
                if reason is None:
                    self._poison('reservation')
                try:
                    self._exhausted(reason, 'shared budget refused: ' + exc.reason)
                except ProductionBudgetExhausted:
                    raise
                except Exception:
                    # A normal limit still needs a durable refusal receipt.
                    # Failure to publish it stops all participants as before.
                    self._poison('reservation_refusal')
            except Exception:
                self._poison('reservation')
            # If the local reservation fails, the shared reservation remains
            # spent and undispatched; granting it back would violate the cap.
            try:
                local_call, seconds = super().reserve(identity, body_text, caller, timeout)
            except Exception as exc:
                if self._aborted or not isinstance(exc, LLMCallRefused):
                    self._poison('local_reservation')
                raise
            self._shared_calls[local_call] = item['call_id']
            return local_call, seconds

    def mark_dispatched(self, call):
        with self._lock:
            if self._aborted:
                raise LLMCallRefused('campaign stopped: ' + self._aborted)
            try:
                self._check_shared_identity()
            except LLMCallRefused:
                self._poison('identity')
            cid = self._shared_calls.get(call)
            if cid is None:
                self._poison('reservation_identity')
            try:
                shared.dispatch(self.shared_path, call_id=cid, expected_task_id=self.shared_task_id)
            except Exception:
                self._poison('dispatch')
            self._shared_dispatched.add(call)
            try:
                # The shared event marks intent before transport handoff; it
                # cannot prove that the HTTP request reached a provider.
                super().mark_dispatched(call)
            except Exception:
                self._poison('local_dispatch')

    def finish(self, call, *, response=None, outcome='answered', wall_seconds=0., usage_source=None, usage=None):
        with self._lock:
            cid = self._shared_calls.get(call)
            failed = False
            if call in self._shared_dispatched:
                try:
                    self._check_shared_identity()
                    counters = _provider_counters(usage, usage_source)
                    normalized_outcome = outcome if outcome in shared.OUTCOMES else 'unknown'
                    shared.finish(self.shared_path, call_id=cid, outcome=normalized_outcome,
                                  usage=counters, usage_evidence='provider' if counters is not None else 'missing',
                                  expected_task_id=self.shared_task_id)
                except Exception:
                    failed = True
            try:
                super().finish(call, response=response, outcome=outcome,
                               wall_seconds=wall_seconds, usage_source=usage_source, usage=usage)
            except Exception:
                self._poison('local_completion')
            if failed:
                self._poison('completion')


def production_budget(limits, journal_dir, *, deadline=None):
    """Select S5 only with a complete explicit process environment binding."""
    path = os.environ.get('OPERATOR_AGENT_S5_SHARED_LEDGER')
    task_id = os.environ.get('OPERATOR_AGENT_S5_TASK_ID')
    if path is None and task_id is None:
        return ProductionLLMCallBudget(limits, journal_dir, deadline=deadline)
    if not path or not task_id:
        raise LLMCallRefused('incomplete S5 shared ledger binding')
    return SharedProductionBudget(limits, journal_dir, shared_path=path,
                                  task_id=task_id, deadline=deadline)
