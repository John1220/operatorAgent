"""In-process, offline-only S5 budget-gateway prototype.

This module performs no network I/O and accepts only a *trusted offline test
fixture* as its adapter. It is not an HTTP server, a sandbox, a production
gateway, or evidence that any real model calls are intercepted. A caller that
supplies an arbitrary Python callable could itself perform I/O; it is outside
the scope of this prototype. Do not load user-generated or provider code here.

The caller's ``participant`` label is self-reported. The shared ledger records
the request hash and a pre-handoff intent, not prompts, response text, verified
provider usage, actual network dispatch or a bill. The actual isolation and
supervisor must be designed and validated separately.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import threading

from operator_agent.execute import s5_shared_budget as shared


class GatewayRefused(RuntimeError):
    """The fake request was refused or cannot be audited safely."""


class GatewayFakeFailure(RuntimeError):
    """A trusted fake adapter or its response failed."""


_EXPECTED_EXHAUSTION = frozenset({
    'campaign_aborted', 'wall_clock_exhausted', 'call_budget_exhausted',
    'output_budget_exhausted', 'per_call_output_exhausted',
})
_FIELDS = frozenset({'model', 'messages', 'max_tokens', 'stream', 'temperature',
                     'tools', 'tool_choice', 'n'})


class FakeGateway:
    """Use one SQLite pool for trusted fake outer and inner fixture requests.

    ``adapter`` receives a detached request object and returns an iterable of
    bytes, either a single JSON body chunk or SSE chunks. Payload contents are
    passed through without judging their truth or interpreting provider usage.
    Fake adapters are in-process trusted fixtures: neither their execution time
    nor their side effects can be constrained by this class.

    Reservations are taken on first iteration and never refunded. An iterator
    must be exhausted or explicitly closed to finalize it; an abandoned or
    crashed iterator remains unknown in the ledger. A process supervisor is
    still necessary, particularly if durable abort cannot be recorded.
    """

    def __init__(self, path, *, task_id, model, adapter, max_request_bytes=1048576,
                 max_response_bytes=1048576, max_chunks=1024):
        if (not isinstance(model, str) or not model or len(model) > 128
                or not callable(adapter)):
            raise ValueError('a fixed model and trusted fake adapter are required')
        for value in (max_request_bytes, max_response_bytes, max_chunks):
            if type(value) is not int or not 1 <= value <= 1048576:
                raise ValueError('invalid gateway limit')
        self.path = Path(path)
        self.task_id = task_id
        self.model = model
        self.adapter = adapter
        self.max_request_bytes = max_request_bytes
        self.max_response_bytes = max_response_bytes
        self.max_chunks = max_chunks
        self._failed = threading.Event()
        try:
            before = self._identity()
            shared.snapshot(self.path, expected_task_id=self.task_id)
            if self._identity() != before:
                raise GatewayRefused('ledger identity changed')
        except Exception:
            raise GatewayRefused('ledger identity unavailable; global stop unproven') from None
        self._ledger_identity = before

    def _identity(self):
        if self.path.is_symlink():
            raise GatewayRefused('ledger identity unavailable')
        stat = self.path.stat()
        if not self.path.is_file():
            raise GatewayRefused('ledger identity unavailable')
        return stat.st_dev, stat.st_ino

    def _check_identity(self):
        if self._identity() != self._ledger_identity:
            raise GatewayRefused('ledger identity changed')

    def _poison(self):
        self._failed.set()
        try:
            self._check_identity()
            shared.abort(self.path, reason='audit_failure', expected_task_id=self.task_id)
        except Exception:
            raise GatewayRefused('audit failed; global stop unproven') from None
        raise GatewayRefused('audit failed; campaign stop recorded') from None

    def _call(self, operation, **kwargs):
        try:
            self._check_identity()
            result = operation(self.path, expected_task_id=self.task_id, **kwargs)
            self._check_identity()
            return result
        except shared.BudgetError as exc:
            if exc.reason in _EXPECTED_EXHAUSTION:
                raise GatewayRefused('shared budget refused: ' + exc.reason) from None
            self._poison()
        except Exception:
            self._poison()

    def _request(self, request, participant):
        if participant not in ('outer', 'inner') or type(request) is not dict:
            raise GatewayRefused('invalid fake request')
        if (set(request) - _FIELDS or request.get('model') != self.model
                or not isinstance(request.get('messages'), list)
                or not request['messages'] or request.get('n', 1) != 1
                or type(request.get('n', 1)) is not int
                or type(request.get('max_tokens')) is not int
                or not 0 < request['max_tokens'] <= 40960
                or type(request.get('stream', False)) is not bool):
            raise GatewayRefused('invalid fake request')
        temp = request.get('temperature')
        if temp is not None and (type(temp) not in (int, float) or not math.isfinite(temp)):
            raise GatewayRefused('invalid fake request')
        try:
            body = json.dumps(request, ensure_ascii=False, allow_nan=False,
                              separators=(',', ':'), sort_keys=True).encode('utf-8')
            if len(body) > self.max_request_bytes:
                raise ValueError('too large')
            detached = json.loads(body)
        except (TypeError, ValueError, OverflowError, RecursionError):
            raise GatewayRefused('invalid fake request') from None
        return detached, hashlib.sha256(body).hexdigest()

    def stream(self, request, *, participant):
        """Return a lazy byte iterator; exhaust or close it to record outcome."""
        detached, request_hash = self._request(request, participant)

        def consume():
            if self._failed.is_set():
                raise GatewayRefused('gateway stopped')
            reservation = self._call(shared.reserve, participant=participant,
                caller='fake_gateway', max_output_tokens=detached['max_tokens'],
                request_sha256=request_hash)
            if self._failed.is_set():
                raise GatewayRefused('gateway stopped')
            cid = reservation['call_id']
            # This is a pre-handoff intent, never evidence of a network request.
            self._call(shared.dispatch, call_id=cid)
            outcome = 'cancelled'
            chunks = None
            try:
                if self._failed.is_set():
                    raise GatewayRefused('gateway stopped')
                chunks = iter(self.adapter(detached))
                size = count = 0
                for chunk in chunks:
                    if self._failed.is_set():
                        raise GatewayRefused('gateway stopped')
                    count += 1
                    if type(chunk) is not bytes or count > self.max_chunks:
                        raise GatewayFakeFailure('invalid fake response')
                    size += len(chunk)
                    if size > self.max_response_bytes:
                        raise GatewayFakeFailure('fake response too large')
                    yield chunk
                outcome = 'answered'
            except GeneratorExit:
                raise
            except GatewayRefused:
                outcome = 'unknown'
                raise
            except Exception:
                outcome = 'failed'
                raise GatewayFakeFailure('fake adapter or response failed') from None
            finally:
                close_failed = False
                close = getattr(chunks, 'close', None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        close_failed = True
                        outcome = 'unknown'
                # Fake usage is never provider-reported usage, including zero.
                self._call(shared.finish, call_id=cid, outcome=outcome, usage_evidence='missing')
                if close_failed:
                    raise GatewayFakeFailure('fake adapter cleanup failed') from None

        return consume()
