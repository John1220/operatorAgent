"""Loopback fake upstream speaking the OpenAI Responses wire protocol.

This is a test fixture, not a provider emulator or a network boundary. It
records only request hashes and bounded metadata, and it can deliberately emit
truncated or failed streams so the gateway's unknown handling is testable.
"""
from __future__ import annotations

import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading


def _event(name, obj):
    return ('event: ' + name + '\ndata: ' + json.dumps(obj) + '\n\n').encode('utf-8')


def transcript(text='OK', *, terminal='completed', include_usage=True):
    """Build the Responses event sequence a real client consumes."""
    message = {'id': 'msg_fixture', 'type': 'message', 'role': 'assistant',
               'status': terminal, 'content': [{'type': 'output_text', 'text': text,
                                                'annotations': []}]}
    response = {'id': 'resp_fixture', 'object': 'response', 'status': terminal,
                'output': [message]}
    if include_usage:
        response['usage'] = {'input_tokens': 7, 'output_tokens': 1, 'total_tokens': 8}
    events = [
        _event('response.created', {'type': 'response.created',
                                    'response': {'id': 'resp_fixture', 'status': 'in_progress'}}),
        _event('response.output_item.added', {'type': 'response.output_item.added',
                                              'output_index': 0,
                                              'item': dict(message, status='in_progress',
                                                           content=[])}),
        _event('response.content_part.added', {'type': 'response.content_part.added',
                                               'item_id': 'msg_fixture', 'output_index': 0,
                                               'content_index': 0,
                                               'part': {'type': 'output_text', 'text': '',
                                                        'annotations': []}}),
        _event('response.output_text.delta', {'type': 'response.output_text.delta',
                                              'item_id': 'msg_fixture', 'output_index': 0,
                                              'content_index': 0, 'delta': text}),
        _event('response.output_text.done', {'type': 'response.output_text.done',
                                             'item_id': 'msg_fixture', 'output_index': 0,
                                             'content_index': 0, 'text': text}),
        _event('response.content_part.done', {'type': 'response.content_part.done',
                                              'item_id': 'msg_fixture', 'output_index': 0,
                                              'content_index': 0,
                                              'part': {'type': 'output_text', 'text': text,
                                                       'annotations': []}}),
        _event('response.output_item.done', {'type': 'response.output_item.done',
                                             'output_index': 0, 'item': message}),
    ]
    if terminal == 'completed':
        events.append(_event('response.completed',
                             {'type': 'response.completed', 'response': response}))
    elif terminal == 'failed':
        events.append(_event('response.failed', {'type': 'response.failed',
                                                 'response': dict(response, status='failed')}))
    return events


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def handle_error(self, request, client_address):
        pass


class _Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.0'

    def log_message(self, *args):
        pass

    def do_POST(self):
        server = self.server
        lengths = self.headers.get_all('Content-Length', [])
        if (self.path != '/v1/responses' or len(lengths) != 1
                or not lengths[0].isdigit() or self.headers.get('Transfer-Encoding')):
            self.send_response(400)
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        body = self.rfile.read(int(lengths[0]))
        try:
            fields = json.loads(body)
        except ValueError:
            fields = {}
        server.requests.append(dict(
            request_sha256=hashlib.sha256(body).hexdigest(), bytes=len(body),
            authorization=self.headers.get('Authorization'),
            model=fields.get('model'), max_output_tokens=fields.get('max_output_tokens'),
            reasoning=fields.get('reasoning'),
            stream=fields.get('stream'), store=fields.get('store'),
            tool_count=len(fields.get('tools') or [])))
        events = transcript(server.text, terminal=server.terminal,
                            include_usage=server.include_usage)
        if server.truncate_after is not None:
            events = events[:server.truncate_after]
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Connection', 'close')
        self.end_headers()
        for event in events:
            self.wfile.write(event)
            self.wfile.flush()


class FakeResponsesUpstream:
    """Context-managed loopback Responses fixture with a fixed reply."""

    def __init__(self, *, text='OK', terminal='completed', include_usage=True,
                 truncate_after=None):
        self._server = _Server(('127.0.0.1', 0), _Handler)
        self._server.text = text
        self._server.terminal = terminal
        self._server.include_usage = include_usage
        self._server.truncate_after = truncate_after
        self._server.requests = []
        self._thread = None

    @property
    def url(self):
        return 'http://127.0.0.1:%d/v1/responses' % self._server.server_address[1]

    @property
    def requests(self):
        return self._server.requests

    def __enter__(self):
        if self._thread is not None:
            raise RuntimeError('upstream already started')
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=6)
