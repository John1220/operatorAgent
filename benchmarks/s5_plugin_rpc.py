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
        run_dir = args.get('run_dir')
        if not isinstance(run_dir, str):
            raise ValueError('prepared run required')
        run_path = Path(run_dir)
        run_path = run_path if run_path.is_absolute() else self.workspace/run_path
        if run_path.is_symlink() or run_path.resolve() != self.task_dir:
            raise ValueError('run path refused')
        if name == 'operator_prepare':
            if {k:v for k,v in args.items() if k != 'run_dir'} != self.declaration:
                raise ValueError('frozen declaration required')
        else:
            allowed = {'run_dir', 'code', 'code_file', 'candidate_origin', 'stop_after', 'backend', 'timeout'}
            if set(args) - allowed:
                raise ValueError('verification option refused')
            if (args.get('stop_after', 'L2') != 'L2'
                    or args.get('backend', self.development['backend']) != self.development['backend']
                    or ('timeout' in args and (type(args['timeout']) is not int
                        or not 1 <= args['timeout'] <= int(self.development['timeout'])))):
                raise ValueError('verification identity refused')
            if ('code' in args) == ('code_file' in args):
                raise ValueError('single code source required')
            args = dict(args)
            if 'code_file' in args:
                args['code'] = self._candidate_file(args.pop('code_file'))
            if not isinstance(args['code'], str) or not args['code'].strip() or len(args['code'].encode()) > 65536:
                raise ValueError('candidate invalid')
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
        except (ValueError, TypeError, OSError):
            with self.lock:
                self.records.append(dict(sequence=len(self.records)+1, outcome='refused',
                    request_sha256=hashlib.sha256(encoded(message)).hexdigest()))
            return _error(identity, -32602, 'Frozen development surface refused this request')
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
