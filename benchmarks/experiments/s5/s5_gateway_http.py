"""Loopback listener for the experimental S5 budgeted model gateway.

The caller supplies a fixed upstream transport and a shared budget policy.
This server is not authentication, a process supervisor or network isolation;
it cannot prevent same-host clients from bypassing it. No caller-supplied URL,
authorization header, cookies or other headers are sent upstream.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

from benchmarks.experiments.s5.s5_gateway_policy import GatewayPolicy, PolicyRefused, PolicyUpstreamFailure


def _unique(pairs):
    result={}
    for k,v in pairs:
        if k in result: raise ValueError('duplicate key')
        result[k]=v
    return result


class _Server(ThreadingHTTPServer):
    daemon_threads=True
    allow_reuse_address=False

    def handle_error(self, request, client_address):
        # Never log request bytes, authorization headers or upstream exceptions.
        pass


class _Handler(BaseHTTPRequestHandler):
    protocol_version='HTTP/1.0'

    def log_message(self, *args):
        pass

    def _error(self, status):
        body=b'{"error":"gateway request unavailable"}'
        self.send_response(status)
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.connection.settimeout(5)
        lengths=self.headers.get_all('Content-Length',[])
        if (self.path!=self.server.expected_path or self.headers.get('Transfer-Encoding') is not None
                or len(lengths)!=1 or not lengths[0].isascii() or not lengths[0].isdigit()
                or not 0<int(lengths[0])<=self.server.gateway.max_request_bytes
                or self.headers.get_content_type()!='application/json'):
            return self._error(400)
        try:
            body=self.rfile.read(int(lengths[0]))
            if len(body)!=int(lengths[0]): raise ValueError('short body')
            request=json.loads(body,object_pairs_hook=_unique,
                parse_constant=lambda _: (_ for _ in ()).throw(ValueError('invalid number')))
            iterator=self.server.gateway.stream(request,participant=self.server.participant)
            first=next(iterator,None)  # Budget and upstream open precede success headers.
        except (UnicodeDecodeError,ValueError,TypeError,TimeoutError):
            return self._error(400)
        except PolicyRefused:
            return self._error(429)
        except PolicyUpstreamFailure:
            return self._error(502)
        try:
            if request.get('stream',False) is True:
                self.send_response(200)
                self.send_header('Content-Type','text/event-stream')
                self.send_header('Connection','close')
                self.end_headers()
                if first is not None:
                    self.wfile.write(first); self.wfile.flush()
                for chunk in iterator:
                    self.wfile.write(chunk); self.wfile.flush()
            else:
                pieces=[] if first is None else [first]
                try: pieces.extend(iterator)
                except (PolicyUpstreamFailure,PolicyRefused): return self._error(502)
                payload=b''.join(pieces)
                self.send_response(200)
                self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        except (BrokenPipeError,ConnectionResetError,TimeoutError,PolicyUpstreamFailure,PolicyRefused):
            # Once headers are emitted, disconnect rather than synthesize a
            # successful end marker. Closing the iterator cancels its adapter.
            self.close_connection=True
        finally:
            try: iterator.close()
            except (PolicyUpstreamFailure,PolicyRefused): self.close_connection=True


class GatewayServer:
    """Experimental local listener; credentials remain in the supplied transport."""
    def __init__(self, ledger, *, task_id, model, participant, adapter,
                 protocol='chat_completions', max_calls_limit=2, reasoning_effort=None,
                 **limits):
        if participant not in ('outer','inner'):
            raise ValueError('invalid participant')
        if protocol not in ('chat_completions','responses'):
            raise ValueError('unknown wire protocol')
        self._server=_Server(('127.0.0.1',0),_Handler)
        self._server.expected_path=('/v1/responses' if protocol=='responses'
                                    else '/v1/chat/completions')
        try:
            self._server.gateway=GatewayPolicy(ledger,task_id=task_id,model=model,adapter=adapter,
                                               protocol=protocol,max_calls_limit=max_calls_limit,
                                               reasoning_effort=reasoning_effort,**limits)
        except Exception:
            self._server.server_close()
            raise
        self._server.participant=participant
        self._thread=None

    @property
    def base_url(self):
        return 'http://127.0.0.1:%d/v1' % self._server.server_address[1]

    def __enter__(self):
        if self._thread is not None: raise RuntimeError('server already started')
        self._thread=threading.Thread(target=self._server.serve_forever,daemon=True)
        self._thread.start()
        return self

    def __exit__(self,*exc):
        # Stops accepting connections, not proof that active upstream calls or
        # same-UID workers have stopped. A real supervisor remains necessary.
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=6)
