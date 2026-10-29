"""Bounded supervisor-owned development feedback and independent final evaluation.

Loopback and tokens prevent accidental cross-run submissions, not same-UID access
or model/GPU budget bypass. Only the supervisor's Python API can submit a final.
Each service is one task/arm, serialized, exclusive on-disk, with no refunds.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import secrets
import threading
import time

from benchmarks import s5_verify_remote
from operator_agent.run_contract import code_hash

MAX_BODY = 131072
MAX_SOURCE = 65536


def encoded(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def write_once(path, obj):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(obj, f, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def unique(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError('duplicate field')
        obj[key] = value
    return obj


class _HTTP(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def handle_error(self, *args):
        pass


class _Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.0'

    def log_message(self, *args):
        pass

    def send(self, status, obj):
        raw = encoded(obj)
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        self.connection.settimeout(5)
        owner = self.server.owner
        auth = self.headers.get_all('Authorization', [])
        if len(auth) != 1 or not hmac.compare_digest(auth[0], 'Bearer ' + owner.token):
            return self.send(403, dict(ok=False, error='unauthorized'))
        lengths = self.headers.get_all('Content-Length', [])
        if (self.path != '/development/verify' or self.headers.get('Transfer-Encoding') is not None
                or len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit()
                or not 0 < int(lengths[0]) <= MAX_BODY
                or self.headers.get_content_type() != 'application/json'):
            return self.send(400, dict(ok=False, error='invalid_request'))
        try:
            raw = self.rfile.read(int(lengths[0]))
            obj = json.loads(raw, object_pairs_hook=unique,
                             parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
            if len(raw) != int(lengths[0]) or type(obj) is not dict or set(obj) != {'code', 'contract_hash'}:
                raise ValueError('unexpected fields')
            result = owner.develop(obj['code'], obj['contract_hash'])
        except (ValueError, TypeError, KeyError):
            return self.send(400, dict(ok=False, error='invalid_request'))
        self.send(200, result)


class DevelopmentService:
    def __init__(self, plan_path, plan_sha256, output, *, arm, max_development=2,
                 wall_seconds=2400):
        if arm not in ('A', 'B', 'C') or type(max_development) is not int or not 1 <= max_development <= 12:
            raise ValueError('invalid arm or verification budget')
        if type(wall_seconds) is not int or not 1 <= wall_seconds <= 14400:
            raise ValueError('invalid deadline')
        self.plan_path = Path(plan_path)
        self.plan_sha256 = plan_sha256
        self.plan = s5_verify_remote.load_plan(self.plan_path, plan_sha256)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False, mode=0o700)
        self.arm = arm
        self.max_development = max_development
        self.deadline = time.monotonic() + wall_seconds
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.Lock()
        self.records = []
        self.development_count = 0
        self.final_count = 0
        self.closed = False
        write_once(self.output / 'policy.json', dict(schema='s5-development-policy-v1',
            arm=arm, service_plan_sha256=plan_sha256, contract_hash=self.plan['contract_hash'],
            max_development=max_development, final_reserve=1, wall_seconds=wall_seconds,
            final_feedback_to_agent=False, isolation_verified=False))
        self.server = None
        self.thread = None

    @property
    def url(self):
        if self.server is None:
            raise RuntimeError('service not started')
        return 'http://127.0.0.1:%d/development/verify' % self.server.server_address[1]

    def snapshot(self):
        with self.lock:
            return dict(schema='s5-development-snapshot-v1', arm=self.arm,
                        contract_hash=self.plan['contract_hash'], max_development=self.max_development,
                        development_used=self.development_count, final_used=self.final_count,
                        records=list(self.records), formal_s5_credit=False)

    def develop(self, code, contract_hash):
        return self._submit('development', code, contract_hash)

    def final(self, code):
        """Separate one-shot evaluation. Never returned to an agent tool."""
        return self._submit('final', code, self.plan['contract_hash'])

    def _submit(self, phase, code, contract_hash):
        if (not isinstance(code, str) or not code.strip() or len(code.encode()) > MAX_SOURCE
                or contract_hash != self.plan['contract_hash']):
            raise ValueError('candidate or contract invalid')
        with self.lock:
            if self.closed or self.final_count:
                return dict(ok=False, status='unknown', error='verification_closed')
            if phase == 'development' and (self.development_count >= self.max_development
                                          or time.monotonic() >= self.deadline):
                return dict(ok=False, status='unknown', error='development_budget_exhausted',
                            development_used=self.development_count,
                            max_development=self.max_development)
            sequence = len(self.records) + 1
            attempt = self.output / ('%03d-%s' % (sequence, phase))
            attempt.mkdir(mode=0o700)
            source = code_hash(code)
            start = time.monotonic()
            request = dict(sequence=sequence, phase=phase, arm=self.arm,
                           source_sha256=source, contract_hash=contract_hash,
                           started_monotonic=start, previous_sequence=sequence - 1 or None,
                           service_plan_sha256=self.plan_sha256)
            # Reservation and source precede execution; a failure never refunds it.
            write_once(attempt / 'request.json', request)
            (attempt / 'candidate.py').write_text(code, encoding='utf-8')
            if phase == 'development':
                self.development_count += 1
            else:
                self.final_count += 1
            try:
                plan = s5_verify_remote.load_plan(self.plan_path, self.plan_sha256)
                if plan['backend'] == 'mock':
                    from benchmarks.s5_vertical_validation import GOOD_FIXTURE, BAD_FIXTURE
                    if code not in (GOOD_FIXTURE, BAD_FIXTURE):
                        raise ValueError('mock mode only executes the two reviewed exact fixtures')
                verdict = s5_verify_remote.submit(plan, arm=self.arm, candidate_text=code,
                                                  output_dir=attempt / 'evaluation')
                if verdict.get('source_sha256') != source or verdict.get('contract_hash') != contract_hash:
                    raise ValueError('verification identity mismatch')
                # A bounded, identical feedback projection for every arm. Full
                # verifier data stays supervisor-side and is hash-bound below.
                messages = []
                for f in sorted((attempt / 'evaluation' / 'task').glob('verify-*/result.json')):
                    for v in json.loads(f.read_text()).get('verdicts', []):
                        messages.append(dict(level=v.get('level'), passed=v.get('passed'),
                            message=str(v.get('message') or '')[:8000],
                            error_log=str(v.get('error_log') or '')[:4000]))
                result = dict(ok=True, status=verdict['status'],
                              passed=verdict.get('full_l2_passed'),
                              full_l2_passed=verdict.get('full_l2_passed'),
                              reason=verdict.get('reason'), levels=verdict.get('levels'),
                              feedback=messages, backend=plan['mode'])
            except Exception as exc:
                result = dict(ok=True, status='unknown', passed=None,
                              reason='verification unavailable: ' + type(exc).__name__,
                              feedback=[], backend=self.plan['mode'])
            result.update(schema='s5-development-feedback-v1', sequence=sequence, phase=phase,
                          source_sha256=source, contract_hash=contract_hash,
                          development_used=self.development_count,
                          max_development=self.max_development, formal_s5_credit=False)
            payload = encoded(result)
            record = dict(request, completed_monotonic=time.monotonic(),
                          response_sha256=hashlib.sha256(payload).hexdigest(),
                          status=result['status'], result=result)
            write_once(attempt / 'response.json', result)
            write_once(attempt / 'record.json', record)
            self.records.append(record)
            return result

    def __enter__(self):
        self.server = _HTTP(('127.0.0.1', 0), _Handler)
        self.server.owner = self
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=6)
        with self.lock:
            self.closed = True
