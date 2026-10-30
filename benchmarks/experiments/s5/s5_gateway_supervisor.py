"""Isolated upstream worker for the experimental S5 model gateway.

The worker receives its fixed endpoint, credential and request over stdin, not
argv or files, and returns length-prefixed response chunks over stdout. The
parent applies a monotonic wall deadline and kills its own worker process group
on timeout/cancel. A reaped direct child does not prove escaped descendants or
the remote provider stopped; cleanup and provider state remain unknown.

This is not a sandbox against a hostile same-UID process and is not a provider
billing cap. It has no CLI mode that starts a real request; the only entrypoint
is the parent-facing callable configured by a trusted supervisor.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import selectors
import signal
import struct
import subprocess
import sys
import threading
import time

from benchmarks.experiments.s5.s5_gateway_transport import (UpstreamTransport, UpstreamError,
                                           MAX_WALL_SECONDS, failure_detail)

_MAX_FRAME = 65536
_FRAME = struct.Struct('!cI')
_ROOT = str(Path(__file__).resolve().parents[3])


def _frame(kind, data=b''):
    if kind not in (b'D',b'F',b'E') or len(data)>_MAX_FRAME:
        raise ValueError('invalid worker frame')
    sys.stdout.buffer.write(_FRAME.pack(kind,len(data))+data)
    sys.stdout.buffer.flush()


def _worker():
    # stdin is private to the owned process; never log the body or credential.
    try:
        raw=sys.stdin.buffer.read(10485761)
        if len(raw)>10485760: return 2
        obj=json.loads(raw)
        if type(obj) is not dict or set(obj)!={'endpoint','api_key','request','timeout_seconds',
                                              'max_request_bytes','max_response_bytes'}:
            return 2
        upstream=UpstreamTransport(obj['endpoint'],obj['api_key'],
            timeout_seconds=obj['timeout_seconds'],max_request_bytes=obj['max_request_bytes'],
            max_response_bytes=obj['max_response_bytes'])
        exchange=upstream.stream(obj['request'])
        try:
            for chunk in exchange:
                # Transport chunks are at most 64 KiB; split defensively.
                for i in range(0,len(chunk),_MAX_FRAME): _frame(b'D',chunk[i:i+_MAX_FRAME])
            _frame(b'F',b'answered' if exchange.outcome=='answered' else b'unknown')
        finally:
            exchange.close()
        return 0
    except BaseException as exc:
        detail=(exc.diagnostic if isinstance(exc,UpstreamError) else
                dict(stage='worker_input',code='unexpected_error',http_status=None))
        try: _frame(b'E',json.dumps(failure_detail(detail),sort_keys=True).encode('utf-8'))
        except BaseException: pass
        return 1


class SupervisedExchange:
    """Single-use iterable; cancellation can be requested from another thread."""
    def __init__(self, owner, request, call_sequence):
        self._owner=owner
        self._request=request
        self._lock=threading.Lock()
        self._stop=threading.Event()
        self._done=threading.Event()
        self._process=None
        self._iterator=self._run()
        self.outcome='not_started'
        # No claim about processes that escape a process group or provider work.
        self.cleanup='not_started'
        self.provider_usage=None
        self.call_sequence=call_sequence
        self.failure=None
        self._deadline_expired=threading.Event()
        self._cancel_requested=threading.Event()

    def __iter__(self): return self
    def __next__(self): return next(self._iterator)

    def _terminate(self):
        self._stop.set()
        with self._lock:
            proc=self._process
            if proc is None:
                return
            if proc.poll() is None:
                try: os.killpg(proc.pid,signal.SIGKILL)
                except OSError:
                    # Fall back to the known direct child if the process-group
                    # operation is unavailable; descendants remain unknown.
                    try: proc.kill()
                    except (ProcessLookupError,PermissionError,OSError): pass
            try: proc.wait(timeout=1)
            except (subprocess.TimeoutExpired,OSError):
                self.cleanup='unknown'
                return
            self.cleanup='direct_child_reaped_descendants_unknown'

    def close(self):
        # Do not call generator.close() from another thread while it is running.
        # Terminating the worker causes the running read to return.
        if self.outcome in ('not_started','in_flight'):
            self.outcome='cancelled'
            self._cancel_requested.set()
            self.failure=dict(stage='worker_cancelled',code='cancelled',http_status=None)
        self._terminate()
        try:
            self._iterator.close()
        except ValueError:
            # Another thread is executing __next__; the killed worker will
            # unblock it, and its finally block will close the pipes.
            pass
        if self.failure is not None:
            self._owner._remember_failure(self)

    def _watch(self, deadline):
        remaining=max(0.,deadline-time.monotonic())
        if not self._done.wait(remaining):
            self._deadline_expired.set()
            self._terminate()

    def _run(self):
        self.outcome='in_flight'
        deadline=time.monotonic()+self._owner.wall_seconds
        watchdog=None
        phase='request_encoding'
        try:
            # Body and credential exist in memory only, not in command args or
            # the environment. env intentionally contains no inherited keys.
            obj=dict(endpoint=self._owner._endpoint,api_key=self._owner._api_key,
                     request=self._request,timeout_seconds=self._owner.wall_seconds,
                     max_request_bytes=self._owner.max_request_bytes,
                     max_response_bytes=self._owner.max_response_bytes)
            encoded=json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',', ':'),
                               allow_nan=False).encode('utf-8')
            if len(encoded)>10485760:
                raise UpstreamError('worker request exceeds bound',stage=phase,code='request_limit')
            phase='worker_start'
            with self._lock:
                if self._stop.is_set(): raise UpstreamError('worker cancelled')
                proc=subprocess.Popen([sys.executable,'-m','benchmarks.experiments.s5.s5_gateway_supervisor','--worker'],
                    cwd=_ROOT,env={'PATH':os.environ.get('PATH','/usr/bin:/bin'),'PYTHONNOUSERSITE':'1'},
                    stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
                    start_new_session=True,bufsize=0)
                self._process=proc
                self.cleanup='unknown'
            watchdog=threading.Thread(target=self._watch,args=(deadline,),daemon=True)
            watchdog.start()
            phase='worker_input'
            try:
                remaining=memoryview(encoded)
                while remaining:
                    sent=proc.stdin.write(remaining)
                    if not sent: raise OSError('worker input closed')
                    remaining=remaining[sent:]
                proc.stdin.close()
            except (BrokenPipeError,OSError):
                raise UpstreamError('worker request unavailable',stage=phase,code='network_error') from None
            buffer=bytearray(); terminal=False
            phase='worker_framing'
            with selectors.DefaultSelector() as sel:
                sel.register(proc.stdout,selectors.EVENT_READ)
                while not terminal:
                    remaining=deadline-time.monotonic()
                    if remaining<=0 or self._stop.is_set():
                        raise UpstreamError('worker deadline or cancellation',stage='worker_deadline',code='timeout')
                    ready=sel.select(remaining)
                    if not ready: raise UpstreamError('worker deadline',stage='worker_deadline',code='timeout')
                    chunk=os.read(proc.stdout.fileno(),_MAX_FRAME+_FRAME.size)
                    if not chunk:
                        raise UpstreamError('worker ended without terminal record',stage=phase,code='worker_eof')
                    buffer.extend(chunk)
                    while len(buffer)>=_FRAME.size:
                        kind,length=_FRAME.unpack(buffer[:_FRAME.size])
                        if kind not in (b'D',b'F',b'E') or length>_MAX_FRAME:
                            raise UpstreamError('worker framing invalid',stage=phase,code='invalid_frame')
                        if len(buffer)<_FRAME.size+length: break
                        payload=bytes(buffer[_FRAME.size:_FRAME.size+length])
                        del buffer[:_FRAME.size+length]
                        if kind==b'D':
                            if not payload: raise UpstreamError('worker framing invalid',stage=phase,code='invalid_frame')
                            yield payload
                        elif kind==b'F':
                            if payload != b'answered' or buffer:
                                raise UpstreamError('worker completion unavailable',stage='worker_completion',code='incomplete')
                            terminal=True
                            break
                        else:
                            try:
                                detail=json.loads(payload)
                            except (UnicodeDecodeError,ValueError):
                                detail=None
                            clean=failure_detail(detail)
                            if clean['stage']=='unknown':
                                clean=dict(stage='worker_framing',code='invalid_frame',http_status=None)
                            raise UpstreamError('upstream outcome unknown',**clean)
            phase='worker_completion'
            remaining=max(0.,deadline-time.monotonic())
            try: rc=proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                raise UpstreamError('worker deadline',stage='worker_deadline',code='timeout') from None
            if rc!=0: raise UpstreamError('worker completion unavailable',stage=phase,code='nonzero_exit')
            self.cleanup='direct_child_reaped_descendants_unknown'
            self.outcome='answered'
        except GeneratorExit:
            self.outcome='cancelled'
            self.failure=dict(stage='worker_cancelled',code='cancelled',http_status=None)
            raise
        except BaseException as exc:
            if self.outcome!='cancelled': self.outcome='unknown'
            if self._cancel_requested.is_set():
                self.failure=dict(stage='worker_cancelled',code='cancelled',http_status=None)
            elif self._deadline_expired.is_set() or isinstance(exc,subprocess.TimeoutExpired):
                self.failure=dict(stage='worker_deadline',code='timeout',http_status=None)
            elif isinstance(exc,UpstreamError):
                self.failure=failure_detail(exc.diagnostic)
                if self.failure['stage']=='unknown':
                    self.failure=dict(stage=phase,code='unexpected_error',http_status=None)
            else:
                self.failure=failure_detail(dict(stage=phase,code='unexpected_error',http_status=None))
            raise UpstreamError('supervised upstream unavailable; outcome unknown',**self.failure) from None
        finally:
            self._done.set()
            self._terminate()
            if watchdog is not None and watchdog is not threading.current_thread():
                watchdog.join(timeout=1)
            proc=self._process
            if proc is not None:
                for pipe in (proc.stdin,proc.stdout):
                    try: pipe.close()
                    except (OSError,ValueError): pass
            if self.failure is not None:
                self._owner._remember_failure(self)


class SupervisedTransport:
    """Factory for one supervised exchange per call, suitable as policy adapter."""
    def __init__(self, endpoint, api_key, *, wall_seconds=30,
                 max_request_bytes=1048576,max_response_bytes=1048576):
        if (type(wall_seconds) not in (int,float) or not math.isfinite(wall_seconds)
                or not 0<wall_seconds<=MAX_WALL_SECONDS):
            raise ValueError('invalid wall time')
        # Reuse transport's endpoint/credential/limit checks without dispatch.
        UpstreamTransport(endpoint,api_key,timeout_seconds=wall_seconds,
                          max_request_bytes=max_request_bytes,max_response_bytes=max_response_bytes)
        self._endpoint=endpoint
        self._api_key=api_key
        self.wall_seconds=float(wall_seconds)
        self.max_request_bytes=max_request_bytes
        self.max_response_bytes=max_response_bytes
        self._audit_lock=threading.Lock()
        self._call_sequence=0
        self._failures={}
        self._failures_truncated=False

    def __call__(self, request):
        with self._audit_lock:
            self._call_sequence+=1
            sequence=self._call_sequence
        return SupervisedExchange(self,request,sequence)

    def _remember_failure(self, exchange):
        # Only controlled labels survive; never retain the exchange/request.
        detail=failure_detail(exchange.failure)
        cleanup=(exchange.cleanup if exchange.cleanup in ('not_started','unknown',
                 'direct_child_reaped_descendants_unknown') else 'unknown')
        outcome=exchange.outcome if exchange.outcome in ('unknown','cancelled') else 'unknown'
        with self._audit_lock:
            if len(self._failures)>=1024 and exchange.call_sequence not in self._failures:
                self._failures_truncated=True
                return
            self._failures[exchange.call_sequence]=dict(call_sequence=exchange.call_sequence,
                **detail,cleanup=cleanup,outcome=outcome)

    def failure_snapshot(self):
        """Supervisor-side audit only; never delivered in model-facing errors."""
        with self._audit_lock:
            return dict(schema='s5-upstream-failures-v1',calls_created=self._call_sequence,
                failures=[dict(self._failures[n])for n in sorted(self._failures)],
                truncated=self._failures_truncated,
                provider_billing_known=False)


if __name__=='__main__':
    if sys.argv[1:] != ['--worker']:
        raise SystemExit(2)
    raise SystemExit(_worker())
