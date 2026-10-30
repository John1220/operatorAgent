"""Supervisor-owned access to the real MCP server, outside the participant jail.

The HTTP capability is per run and only forwards the frozen development surface.
It is not a hostile same-UID/network boundary. Core schemas and instructions are
returned unchanged by the real server; final evaluation is never an RPC tool.
"""
from __future__ import annotations

import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import threading
import time

from benchmarks import s5_runtime
from benchmarks.s5_development_service import encoded, unique, write_once

ROOT = Path(__file__).resolve().parents[1]
MAX_REQUEST = 262144
MAX_REPLY = 2 * 1024 * 1024
TOOLS = frozenset(('operator_prepare', 'operator_verify', 'operator_capabilities'))

#: Refusals must stay fixable: a bare "refused" made a real model repeat the same
#: wrong call until its budget ran out (2026-10-30 Matmul C, two `run_dir=/work`
#: submissions).  Every reason therefore carries a hint that names the argument
#: and the values the frozen surface accepts.  Hints never echo supervisor paths
#: or the frozen task directory: the participant only learns its workspace-relative
#: location, which it already owns.
STATIC_HINTS = {
    'invalid RPC envelope': 'send one JSON-RPC 2.0 object with only jsonrpc/id/method/params',
    'invalid RPC id': 'use a string or integer id of at most 128 characters',
    'invalid RPC params': 'send params as a JSON object',
    'notification id refused': 'send notifications without an id',
    'request id required': 'include an id on every request',
    'method refused': 'only initialize, ping, tools/list and tools/call are available',
    'tool refused': 'available tools: operator_prepare, operator_verify, operator_capabilities',
    'capabilities arguments refused': 'operator_capabilities takes no arguments',
    'prepared run required': 'run_dir is required (the task directory operator_prepare returned)',
    'frozen declaration required': 'operator_prepare only accepts the frozen declaration from the task material',
    'single code source required': 'pass exactly one of code or code_file',
    'candidate invalid': 'code must be non-empty source text of at most 65536 bytes',
    'candidate path required': 'code_file must be a non-empty path inside the workspace',
    'regular candidate file required': 'code_file must be a regular, unlinked file inside the workspace',
    'candidate outside workspace': 'code_file must resolve inside the workspace',
    'candidate link refused': 'code_file must not sit under a symlink',
    'candidate exceeds bound': 'candidate source must be at most 65536 bytes',
}


class Refused(ValueError):
    """Every violated argument rule at once, so one reply is enough to fix them."""

    def __init__(self, problems):
        super().__init__('; '.join(reason for reason, _ in problems))
        self.problems = tuple(problems)


def refusal_text(problems):
    return 'Frozen development surface refused this request: ' + '; '.join(
        '%s (%s)' % (reason, hint) for reason, hint in problems)


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
            return self.send(403, dict(error='unauthorized'))
        lengths = self.headers.get_all('Content-Length', [])
        if (self.path != '/mcp' or self.headers.get('Transfer-Encoding') is not None
                or self.headers.get_content_type() != 'application/json'
                or len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit()
                or not 0 < int(lengths[0]) <= MAX_REQUEST):
            return self.send(400, dict(error='invalid_request'))
        try:
            raw = self.rfile.read(int(lengths[0]))
            if len(raw) != int(lengths[0]):
                raise ValueError('short body')
            request = json.loads(raw, object_pairs_hook=unique,
                                 parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
            reply = owner.request(request)
        except (ValueError, TypeError, KeyError):
            return self.send(400, dict(error='invalid_request'))
        self.send(200, reply if reply is not None else dict(notification=True))


def _key(value):
    return type(value).__name__, json.dumps(value, sort_keys=True, allow_nan=False)


def _error(identity, code, message):
    return dict(jsonrpc='2.0', id=identity, error=dict(code=code, message=message))


class MCPRelay:
    def __init__(self, development, task_dir, declaration, output, *, timeout=700):
        if type(timeout) is not int or not 1 <= timeout <= 1810:
            raise ValueError('invalid MCP timeout')
        self.development = dict(development)
        if set(self.development) != {'url', 'token', 'contract', 'backend', 'timeout'}:
            raise ValueError('invalid development configuration')
        self.task_dir = Path(task_dir).resolve()
        self.workspace = self.task_dir.parent
        self.declaration = dict(declaration)
        if not self.task_dir.is_dir() or not (self.task_dir/'contract.json').is_file():
            raise ValueError('prepared task required')
        if json.loads((self.task_dir/'contract.json').read_text())['hash'] != development['contract']:
            raise ValueError('prepared task identity mismatch')
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False, mode=0o700)
        self.timeout = timeout
        self.token = secrets.token_urlsafe(32)
        self.server = self.thread = self.proc = self.reader = self.stderr = None
        self.pending = {}
        self.lock = threading.Lock()
        self.write_lock = threading.Lock()
        self.records = []
        self.closed = False
        self.cleanup = 'not_started'
        self.protocol_failed = False
        self._buffer = bytearray()

    @property
    def url(self):
        if self.server is None:
            raise RuntimeError('MCP relay not started')
        return 'http://127.0.0.1:%d/mcp' % self.server.server_address[1]

    def client_config(self):
        return dict(url=self.url, token=self.token, timeout=self.timeout)

    def snapshot(self):
        with self.lock:
            return dict(schema='s5-plugin-mcp-relay-v1', records=list(self.records),
                        cleanup=self.cleanup, closed=self.closed,
                        process_returncode=self.proc.poll() if self.proc is not None else None,
                        protocol_failed=self.protocol_failed, real_core_server=True,
                        provider_credentials_passed=False)

    def _candidate_file(self, name):
        if not isinstance(name, str) or not name:
            raise ValueError('candidate path required')
        original = Path(name)
        original = original if original.is_absolute() else self.workspace/original
        if original.is_symlink() or not original.is_file() or original.stat().st_nlink != 1:
            raise ValueError('regular candidate file required')
        resolved = original.resolve()
        if not resolved.is_relative_to(self.workspace):
            raise ValueError('candidate outside workspace')
        # No ancestor symlinks, including links to another file within work.
        cur = original.absolute()
        while cur != self.workspace and cur != cur.parent:
            if cur.is_symlink():
                raise ValueError('candidate link refused')
            cur = cur.parent
        with resolved.open('rb') as f:
            raw = f.read(65537)
        if len(raw) > 65536:
            raise ValueError('candidate exceeds bound')
        return raw.decode('utf-8')

    def _task_relative(self):
        try:
            return self.task_dir.relative_to(self.workspace).as_posix()
        except ValueError:
            return 'the task directory'

    def _hint(self, reason, extra=''):
        """Actionable text for one refusal reason; never a supervisor path."""
        if reason == 'run path refused':
            return 'run_dir must resolve to the prepared task directory; pass %r' % self._task_relative()
        if reason == 'verification option refused':
            return 'remove unsupported argument(s)' + (': ' + extra if extra else '')
        if reason == 'verification identity refused':
            return ("operator_verify accepts stop_after='L2', backend=%r and "
                    '1 <= timeout <= %s' % (self.development['backend'], self.development['timeout']))
        return STATIC_HINTS.get(reason, 'resend a request that matches the tool schema')

    def _refuse(self, reason, extra=''):
        return (reason, self._hint(reason, extra))

    def _validate(self, message):
        if (type(message) is not dict or message.get('jsonrpc') != '2.0'
                or set(message) - {'jsonrpc', 'id', 'method', 'params'}):
            raise ValueError('invalid RPC envelope')
        identity = message.get('id')
        if identity is not None and (type(identity) not in (str, int) or len(str(identity)) > 128):
            raise ValueError('invalid RPC id')
        method = message.get('method')
        params = message.get('params', {})
        if type(params) is not dict:
            raise ValueError('invalid RPC params')
        if method in ('notifications/initialized', 'notifications/cancelled'):
            if identity is not None:
                raise ValueError('notification id refused')
            return message
        if identity is None:
            raise ValueError('request id required')
        if method in ('initialize', 'ping', 'tools/list'):
            return message
        if method != 'tools/call' or set(params) - {'name', 'arguments', '_meta'}:
            raise ValueError('method refused')
        name = params.get('name')
        args = params.get('arguments', {})
        if name not in TOOLS or type(args) is not dict:
            raise ValueError('tool refused')
        if name == 'operator_capabilities':
            if args:
                raise ValueError('capabilities arguments refused')
            return message
        problems = []
        run_dir = args.get('run_dir')
        if not isinstance(run_dir, str):
            problems.append(self._refuse('prepared run required'))
        else:
            run_path = Path(run_dir)
            run_path = run_path if run_path.is_absolute() else self.workspace/run_path
            if run_path.is_symlink() or run_path.resolve() != self.task_dir:
                problems.append(self._refuse('run path refused'))
        if name == 'operator_prepare':
            if {k:v for k,v in args.items() if k != 'run_dir'} != self.declaration:
                problems.append(self._refuse('frozen declaration required'))
            if problems:
                raise Refused(problems)
        else:
            allowed = {'run_dir', 'code', 'code_file', 'candidate_origin', 'stop_after', 'backend', 'timeout'}
            unsupported = set(args) - allowed
            if unsupported:
                problems.append(self._refuse('verification option refused', ', '.join(sorted(unsupported))))
            if (args.get('stop_after', 'L2') != 'L2'
                    or args.get('backend', self.development['backend']) != self.development['backend']
                    or ('timeout' in args and (type(args['timeout']) is not int
                        or not 1 <= args['timeout'] <= int(self.development['timeout'])))):
                problems.append(self._refuse('verification identity refused'))
            if ('code' in args) == ('code_file' in args):
                problems.append(self._refuse('single code source required'))
            elif 'code_file' in args:
                try:
                    args = dict(args)
                    args['code'] = self._candidate_file(args.pop('code_file'))
                except ValueError as exc:
                    problems.append(self._refuse(str(exc)))
            if problems:
                raise Refused(problems)
            if not isinstance(args['code'], str) or not args['code'].strip() or len(args['code'].encode()) > 65536:
                raise Refused([self._refuse('candidate invalid')])
            args.setdefault('stop_after', 'L2')
            args.setdefault('backend', self.development['backend'])
        args = dict(args, run_dir=str(self.task_dir))
        return dict(message, params=dict(params, arguments=args))

    def _read(self):
        try:
            while True:
                line = self.proc.stdout.readline(MAX_REPLY+1)
                if not line:
                    break
                if len(line) > MAX_REPLY or not line.endswith(b'\n'):
                    raise ValueError('MCP reply bound')
                reply = json.loads(line)
                if type(reply) is not dict or reply.get('jsonrpc') != '2.0' or 'id' not in reply:
                    raise ValueError('invalid MCP reply')
                key = _key(reply['id'])
                with self.lock:
                    waiting = self.pending.get(key)
                    if waiting is None:
                        raise ValueError('unexpected MCP reply')
                    waiting['reply'] = reply
                    waiting['event'].set()
        except (ValueError, OSError):
            self.protocol_failed = True
        finally:
            with self.lock:
                for waiting in self.pending.values():
                    waiting['event'].set()

    def request(self, message):
        identity = message.get('id') if isinstance(message, dict) else None
        try:
            outgoing = self._validate(message)
        except (ValueError, TypeError, OSError) as exc:
            problems = getattr(exc, 'problems', None) or (
                self._refuse(str(exc) or type(exc).__name__),)
            try:
                digest = hashlib.sha256(encoded(message)).hexdigest()
            except (TypeError, ValueError):
                digest = None
            with self.lock:
                self.records.append(dict(sequence=len(self.records)+1, outcome='refused',
                    request_sha256=digest, refusal=[reason for reason, _ in problems]))
            return _error(identity, -32602, refusal_text(problems))
        method = outgoing['method']
        tool = outgoing.get('params', {}).get('name') if method == 'tools/call' else None
        record = dict(method=method, tool=tool, request_sha256=hashlib.sha256(encoded(message)).hexdigest(),
                      started_monotonic=time.monotonic(), outcome='unknown')
        waiting = dict(event=threading.Event(), reply=None)
        key = _key(identity) if identity is not None else None
        with self.lock:
            record['sequence'] = len(self.records)+1
            self.records.append(record)
            if self.closed or self.protocol_failed or self.proc is None or self.proc.poll() is not None:
                return _error(identity, -32603, 'MCP backend unavailable')
            if key in self.pending:
                record['outcome'] = 'refused'
                return _error(identity, -32600, 'Request ID already active')
            if key is not None:
                self.pending[key] = waiting
        try:
            raw = encoded(outgoing)+b'\n'
            if len(raw) > MAX_REQUEST:
                raise ValueError('RPC bound')
            with self.write_lock:
                self.proc.stdin.write(raw)
                self.proc.stdin.flush()
            if key is None:
                record['outcome'] = 'notification_forwarded'
                return None
            if not waiting['event'].wait(self.timeout) or waiting['reply'] is None:
                raise ValueError('RPC response unavailable')
            reply = waiting['reply']
            record['outcome'] = 'answered'
            record['response_sha256'] = hashlib.sha256(encoded(reply)).hexdigest()
            if 'error' in reply:
                record['rpc_error_code'] = reply['error'].get('code')
            return reply
        except (OSError, ValueError):
            return _error(identity, -32603, 'MCP backend unavailable')
        finally:
            record['completed_monotonic'] = time.monotonic()
            if key is not None:
                with self.lock:
                    self.pending.pop(key, None)

    def __enter__(self):
        config = self.output/'development-client.json'
        write_once(config, {k.upper(): str(v) for k,v in self.development.items()})
        config.chmod(0o600)
        home = self.output/'home'; home.mkdir(mode=0o700)
        env = dict(PATH=os.environ.get('PATH', '/usr/bin:/bin'), HOME=str(home),
                   PYTHONPATH=s5_runtime.python_path(), PYTHONNOUSERSITE='1',
                   PYTHONDONTWRITEBYTECODE='1', OPERATOR_AGENT_S5_DEV_CONFIG=str(config))
        self.stderr = (self.output/'mcp-stderr.log').open('xb')
        try:
            self.proc = subprocess.Popen([sys.executable, '-m', 'operator_agent.mcp_server'],
                cwd=str(ROOT), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.stderr, start_new_session=True, bufsize=0)
            self.cleanup = 'running'
            self.reader = threading.Thread(target=self._read, daemon=True); self.reader.start()
            self.server = _HTTP(('127.0.0.1', 0), _Handler); self.server.owner = self
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True); self.thread.start()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        with self.lock:
            self.closed = True
        if self.server is not None:
            self.server.shutdown(); self.server.server_close()
        if self.thread is not None:
            self.thread.join(timeout=6)
        if self.proc is not None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                try: os.killpg(self.proc.pid, signal.SIGTERM)
                except ProcessLookupError: pass
                try: self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try: os.killpg(self.proc.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                    self.proc.wait(timeout=3)
            self.cleanup = 'direct_child_reaped_descendants_unknown'
            if self.reader is not None: self.reader.join(timeout=3)
            if self.proc.stdout is not None: self.proc.stdout.close()
        if self.stderr is not None: self.stderr.close()
        (self.output/'development-client.json').unlink(missing_ok=True)
        target = self.output/'relay-snapshot.json'
        if not target.exists(): write_once(target, self.snapshot())
