"""Offline loopback HTTP harness around the in-process S5 fake gateway.

This is a trusted test fixture, not a network proxy, sandbox, production server,
or proof that real DSH traffic is intercepted. It has no upstream transport;
the supplied trusted fake adapter can itself do arbitrary I/O. It binds only an
ephemeral IPv4 loopback port. Provider credentials and payloads are not logged.
The ledger hashes the normalized request (the two output-limit spellings map
to ``max_tokens``), not the exact HTTP bytes. Participant is fixed by the test
server, not a verified process identity.
"""
from __future__ import annotations

import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

from benchmarks.experiments.s5.s5_fake_gateway import FakeGateway, GatewayFakeFailure, GatewayRefused

_COMPAT = frozenset({'chat_template_kwargs', 'enable_thinking', 'stream_options', 'store'})


class _CompatGateway(FakeGateway):
    def _request(self, request, participant):
        if type(request) is not dict:
            raise GatewayRefused('invalid fake request')
        normalized = dict(request)
        if 'max_completion_tokens' in normalized:
            if 'max_tokens' in normalized:
                raise GatewayRefused('conflicting output limits')
            normalized['max_tokens'] = normalized.pop('max_completion_tokens')
        compat = {k: normalized.pop(k) for k in _COMPAT if k in normalized}
        if 'chat_template_kwargs' in compat:
            opt = compat['chat_template_kwargs']
            if (type(opt) is not dict or set(opt) != {'enable_thinking'}
                    or type(opt['enable_thinking']) is not bool):
                raise GatewayRefused('invalid thinking options')
        if 'enable_thinking' in compat and type(compat['enable_thinking']) is not bool:
            raise GatewayRefused('invalid thinking options')
        if 'enable_thinking' in compat and 'chat_template_kwargs' in compat:
            raise GatewayRefused('conflicting thinking options')
        if 'stream_options' in compat:
            opt = compat['stream_options']
            if (type(opt) is not dict or set(opt) != {'include_usage'}
                    or type(opt['include_usage']) is not bool or normalized.get('stream') is not True):
                raise GatewayRefused('invalid stream options')
        if 'store' in compat and compat['store'] is not False:
            raise GatewayRefused('invalid store option')
        detached, _ = super()._request(normalized, participant)
        detached.update(compat)
        try:
            body = json.dumps(detached, ensure_ascii=False, allow_nan=False,
                              separators=(',', ':'), sort_keys=True).encode('utf-8')
        except (TypeError, ValueError, OverflowError, RecursionError):
            raise GatewayRefused('invalid fake request') from None
        if len(body) > self.max_request_bytes:
            raise GatewayRefused('fake request too large')
        return json.loads(body), hashlib.sha256(body).hexdigest()


def _strict_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError('duplicate key')
        out[key] = value
    return out


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def handle_error(self, request, client_address):
        # Never log request bytes, headers, or arbitrary fixture exceptions.
        pass


class _Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.0'

    def log_message(self, *args):
        pass

    def _error(self, status):
        body = b'{"error":"fake gateway refused"}'
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.connection.settimeout(5)
        if self.path != '/v1/chat/completions':
            return self._error(404)
        lengths = self.headers.get_all('Content-Length', [])
        if (self.headers.get('Transfer-Encoding') is not None or len(lengths) != 1
                or not lengths[0].isascii() or not lengths[0].isdigit()):
            return self._error(400)
        length = int(lengths[0])
        if length < 1 or length > self.server.gateway.max_request_bytes:
            return self._error(413)
        if self.headers.get_content_type() != 'application/json':
            return self._error(415)
        try:
            body = self.rfile.read(length)
            if len(body) != length:
                return self._error(400)
            request = json.loads(body, object_pairs_hook=_strict_object,
                                 parse_constant=lambda _: (_ for _ in ()).throw(ValueError('constant')))
            iterator = self.server.gateway.stream(request, participant=self.server.participant)
            # Initiate reservation before returning HTTP success. Empty streams
            # may be legitimate fake responses, but receive no inferred usage.
            first = next(iterator, None)
        except (UnicodeDecodeError, ValueError, TypeError):
            return self._error(400)
        except GatewayRefused:
            return self._error(429)
        except GatewayFakeFailure:
            return self._error(502)
        try:
            streaming = request.get('stream', False) is True
            if not streaming:
                pieces = ([] if first is None else [first])
                try:
                    pieces.extend(iterator)
                except (GatewayFakeFailure, GatewayRefused):
                    return self._error(502)
                payload = b''.join(pieces)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            else:
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Connection', 'close')
                self.end_headers()
                if first is not None:
                    self.wfile.write(first)
                    self.wfile.flush()
                for chunk in iterator:
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, TimeoutError, GatewayFakeFailure, GatewayRefused):
            # Once headers are sent, terminate rather than claim a successful
            # stream. A closed iterator records cancellation/unknown.
            self.close_connection = True
        finally:
            try:
                iterator.close()
            except (GatewayFakeFailure, GatewayRefused):
                self.close_connection = True


class FakeHTTPServer:
    """Context-managed localhost-only test server with a fixed participant label."""
    def __init__(self, path, *, task_id, model, participant, adapter, **limits):
        if participant not in ('outer', 'inner'):
            raise ValueError('invalid participant')
        gateway = _CompatGateway(path, task_id=task_id, model=model, adapter=adapter, **limits)
        self._server = _Server(('127.0.0.1', 0), _Handler)
        self._server.gateway = gateway
        self._server.participant = participant
        self._thread = None

    @property
    def base_url(self):
        return 'http://127.0.0.1:%d/v1' % self._server.server_address[1]

    def __enter__(self):
        if self._thread is not None:
            raise RuntimeError('server already started')
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=6)
