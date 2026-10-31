"""Loopback-only mock validation fixture for the S5 tool-calling experiment.

This is not a production GPU service or a security boundary. It can execute
only the exact fixtures whitelisted by VerticalValidator; the counter and
evidence live in this process, outside the temporary DSH workspace.
"""
from __future__ import annotations

import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import subprocess
import sys
import threading

from benchmarks.experiments.s5 import s5_runtime
from benchmarks.experiments.s5.s5_vertical_validation import VerticalValidator, GOOD_FIXTURE, BAD_FIXTURE
from operator_agent.execute.task_clock import monotonic

MAX_BODY = 131072


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def _write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def handle_error(self, request, client_address):
        pass


class _Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.0'

    def log_message(self, *args):
        pass

    def _send(self, status, body):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.connection.settimeout(5)
        lengths = self.headers.get_all('Content-Length', [])
        if (self.path != '/validate' or self.headers.get('Transfer-Encoding') is not None
                or len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit()
                or not 0 < int(lengths[0]) <= MAX_BODY
                or self.headers.get_content_type() != 'application/json'):
            return self._send(400, b'{"error":"invalid request"}')
        try:
            raw = self.rfile.read(int(lengths[0]))
            def unique(pairs):
                out = {}
                for key, value in pairs:
                    if key in out: raise ValueError('duplicate key')
                    out[key] = value
                return out
            obj = json.loads(raw, object_pairs_hook=unique,
                             parse_constant=lambda _: (_ for _ in ()).throw(ValueError('constant')))
            if len(raw) != int(lengths[0]) or type(obj) is not dict or set(obj) != {'source'}:
                raise ValueError('bad request')
        except (ValueError, UnicodeDecodeError, TypeError):
            return self._send(400, b'{"error":"invalid request"}')
        try:
            with self.server.lock:
                number = self.server.requests + 1
                self.server.requests = number
                start = monotonic()
                _write(self.server.output/f'request-{number:03d}.json',
                       dict(sequence=number, start_monotonic=start,
                            request_sha256=hashlib.sha256(raw).hexdigest()))
                if number > 1:
                    result = dict(status='budget_exhausted', candidate_sha256=None,
                                  verification_attempts=1, max_verification_attempts=1)
                elif obj['source'] not in (GOOD_FIXTURE, BAD_FIXTURE):
                    # Unknown source never enters a worker that can execute it.
                    result = self.server.validator.validate(obj['source'], arm='A')
                else:
                    # A single-threaded worker lets the existing execution guard
                    # use its normal fork path instead of launching a fresh Python
                    # interpreter for every one of the many L2 suite cases.
                    env = {'PATH':os.environ.get('PATH','/usr/bin:/bin'),
                           'PYTHONPATH':s5_runtime.python_path(),
                           'PYTHONNOUSERSITE':'1'}
                    worker = subprocess.run([sys.executable,'-m','benchmarks.experiments.s5.s5_vertical_tool_worker'],
                        input=json.dumps(obj), text=True, capture_output=True, timeout=60,
                        env=env, cwd=str(Path(__file__).resolve().parents[3]))
                    if worker.returncode != 0 or len(worker.stdout.encode()) > 2*1024*1024:
                        raise ValueError('mock worker unavailable')
                    result = json.loads(worker.stdout)
                end = monotonic()
                payload = _json_bytes(dict(status=result['status'],
                    candidate_sha256=result['candidate_sha256'],
                    verification_attempts=result['verification_attempts'],
                    mock_only=True, formal_s5_credit=False))
                record = dict(sequence=number, start_monotonic=start, end_monotonic=end,
                              request_sha256=hashlib.sha256(raw).hexdigest(),
                              response_sha256=hashlib.sha256(payload).hexdigest(), result=result)
                _write(self.server.output/f'result-{number:03d}.json', record)
                self.server.records.append(record)
            self._send(200, payload)
        except Exception:
            self._send(500, b'{"error":"verification unavailable"}')


class MockValidationService:
    def __init__(self, output):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
        self._server = _Server(('127.0.0.1', 0), _Handler)
        self._server.output = output
        self._server.validator = VerticalValidator(max_attempts=1)
        self._server.requests = 0
        self._server.records = []
        self._server.lock = threading.Lock()
        self._thread = None

    @property
    def url(self):
        return 'http://127.0.0.1:%d' % self._server.server_address[1]

    @property
    def records(self):
        with self._server.lock:
            return list(self._server.records)

    def __enter__(self):
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=6)
