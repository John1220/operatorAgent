"""Loopback fake upstream checks; no real model, provider, or GPU."""
import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time

import pytest
from benchmarks.experiments.s5.s5_gateway_transport import UpstreamError, UpstreamTransport


FINISH=b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
JSON_OK=b'{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}'


@contextlib.contextmanager
def fake_server(handler):
    class Server(ThreadingHTTPServer):
        daemon_threads=True
        def handle_error(self,*_): pass
    server=Server(('127.0.0.1',0),handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        yield 'http://127.0.0.1:%d/v1/chat/completions' % server.server_address[1]
    finally:
        server.shutdown();server.server_close();thread.join(3)


def handler_for(status=200, body=JSON_OK, content_type='application/json', headers=None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_): pass
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(status);self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(body)))
            for name,value in (headers or {}).items(): self.send_header(name,value)
            self.end_headers()
            try: self.wfile.write(body)
            except OSError: pass
    return Handler


def test_responses_stream_may_end_without_a_trailing_blank_line():
    from benchmarks.experiments.s5.s5_gateway_transport import _valid_response
    stream=(b'event: response.created\n'
            b'data: {"type":"response.created","response":{"status":"in_progress"}}\n\n'
            b'event: response.completed\n'
            b'data: {"type":"response.completed","response":{"status":"completed"}}')
    assert _valid_response(stream,True,'responses') is True
    truncated=stream.split(b'event: response.completed')[0]
    assert _valid_response(truncated,True,'responses') is False


def test_json_and_bounded_request_are_sent_without_changing_fields():
    received=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_): pass
        def do_POST(self):
            body=self.rfile.read(int(self.headers['Content-Length']))
            received.append((self.headers.get('Authorization'),json.loads(body)))
            self.send_response(200);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(JSON_OK)));self.end_headers();self.wfile.write(JSON_OK)
    request={'model':'fixture','stream':False,'max_completion_tokens':12,
             'chat_template_kwargs':{'enable_thinking':False},'tools':[{'type':'function'}]}
    with fake_server(Handler) as url:
        exchange=UpstreamTransport(url,'test-secret').stream(request)
        assert b''.join(exchange)==JSON_OK
    assert received==[('Bearer test-secret',request)]
    assert exchange.outcome=='answered' and exchange.provider_usage is None
    assert exchange.usage_evidence=='not_collected'


def test_stream_first_chunk_arrives_before_server_finishes():
    sent=threading.Event();release=threading.Event()
    first=b'data: {"choices":[{"delta":{"content":"a"},"finish_reason":null}]}\n\n'
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_): pass
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200);self.send_header('Content-Type','text/event-stream')
            self.end_headers();self.wfile.write(first);self.wfile.flush();sent.set()
            release.wait(5)
            try: self.wfile.write(FINISH);self.wfile.flush()
            except OSError: pass
    with fake_server(Handler) as url:
        exchange=UpstreamTransport(url,'test',timeout_seconds=8).stream({'stream':True})
        try:
            initial=next(exchange)
            assert sent.wait(1) and not release.is_set() and initial==first
            release.set(); assert b''.join(exchange)==FINISH
            assert exchange.outcome=='answered'
        finally:
            release.set();exchange.close()


@pytest.mark.parametrize('separator',[b'\r\n',b'\r'])
def test_sse_preserves_line_endings_including_final_cr(separator):
    body=(b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":null}]}' +
          separator*2 + b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}' +
          separator*2 + b'data: [DONE]' + separator*2)
    with fake_server(handler_for(body=body,content_type='text/event-stream')) as url:
        exchange=UpstreamTransport(url,'test').stream({'stream':True})
        assert b''.join(exchange)==body and exchange.outcome=='answered'


@pytest.mark.parametrize('body',[
    b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
    b'data: {"choices":[{"delta":{},"finish_reason":null}]}\n\ndata: [DONE]\n\n',
    b'data: not-json\n\ndata: [DONE]\n\n',
    b'data: {"choices":[{"finish_reason":"stop"}]}\n\ndata:  [DONE]\n\n',
    b'data: {"choices":[{"finish_reason":"stop"}]}\n\ndata: {"error":"bad"}\n\ndata: [DONE]\n\n',
])
def test_incomplete_or_malformed_sse_is_unknown(body):
    with fake_server(handler_for(body=body,content_type='text/event-stream')) as url:
        exchange=UpstreamTransport(url,'test').stream({'stream':True})
        with pytest.raises(UpstreamError,match='unavailable or incomplete'):
            list(exchange)
        assert exchange.outcome=='unknown' and exchange.provider_usage is None


@pytest.mark.parametrize('body',[
    b'data: {"choices":[{"delta":{},"finish_reason":null}]}\n\ndata: [DONE]\n\n',
    b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
    b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\ndata: {"error":"bad"}\n\n',
])
def test_invalid_stream_never_delivers_completion_markers(body):
    with fake_server(handler_for(body=body,content_type='text/event-stream')) as url:
        exchange=UpstreamTransport(url,'test').stream({'stream':True})
        delivered=[]
        with pytest.raises(UpstreamError):
            while True: delivered.append(next(exchange))
        wire=b''.join(delivered)
        assert b'[DONE]' not in wire
        assert b'"finish_reason":"stop"' not in wire
        assert exchange.outcome=='unknown'


def test_completion_is_withheld_until_valid_eof():
    initial=b'data: {"choices":[{"delta":{"content":"a"},"finish_reason":null}]}\n\n'
    ending=b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
    sent=threading.Event();release=threading.Event();delivered=threading.Event();chunks=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_): pass
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
            try:
                self.wfile.write(initial+ending);self.wfile.flush();sent.set();release.wait(5)
            except OSError: pass
    with fake_server(Handler) as url:
        exchange=UpstreamTransport(url,'test',timeout_seconds=8).stream({'stream':True})
        try:
            assert next(exchange)==initial and sent.wait(1)
            def consume():
                chunks.extend(exchange);delivered.set()
            thread=threading.Thread(target=consume,daemon=True);thread.start()
            assert not delivered.wait(.2) and not chunks
            release.set();thread.join(3)
            assert delivered.is_set() and b''.join(chunks)==ending
            assert exchange.outcome=='answered'
        finally:
            release.set()


@pytest.mark.parametrize('body',[b'{}',b'{"choices":[{"message":{},"finish_reason":null}]}',b'invalid',
                                  b'{"error":"bad","choices":[{"message":{},"finish_reason":"stop"}]}'])
def test_incomplete_json_is_unknown(body):
    with fake_server(handler_for(body=body)) as url:
        exchange=UpstreamTransport(url,'test').stream({'stream':False})
        with pytest.raises(UpstreamError): list(exchange)
        assert exchange.outcome=='unknown'


def test_redirect_not_followed_and_error_never_leaks_secret_or_endpoint():
    with fake_server(handler_for(status=302,headers={'Location':'http://127.0.0.1:9/secret'})) as url:
        exchange=UpstreamTransport(url,'a-private-credential').stream({'stream':False})
        with pytest.raises(UpstreamError) as error: list(exchange)
        assert 'a-private-credential' not in str(error.value) and url not in str(error.value)
        assert exchange.outcome=='unknown'


def test_deadline_is_total_across_multiple_slow_chunks():
    first=b'data: {"choices":[{"delta":{},"finish_reason":null}]}\n\n'
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_): pass
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
            try:
                for _ in range(4):
                    self.wfile.write(first);self.wfile.flush();time.sleep(.12)
                self.wfile.write(FINISH);self.wfile.flush()
            except OSError: pass
    with fake_server(Handler) as url:
        exchange=UpstreamTransport(url,'test',timeout_seconds=.25).stream({'stream':True})
        with pytest.raises(UpstreamError): list(exchange)
        assert exchange.outcome=='unknown'


def test_cancel_after_first_chunk_marks_cancelled():
    release=threading.Event();first=b'data: {"choices":[]}\n\n'
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_): pass
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
            try:
                self.wfile.write(first);self.wfile.flush();release.wait(2);self.wfile.write(FINISH)
            except OSError: pass
    with fake_server(Handler) as url:
        exchange=UpstreamTransport(url,'test').stream({'stream':True})
        try: assert next(exchange)==first
        finally: exchange.close();release.set()
        assert exchange.outcome=='cancelled'


def test_size_and_endpoint_checks():
    for endpoint in ('http://example.com/v1/chat/completions',
                     'https://user:secret@example.com/v1/chat/completions',
                     'https://example.com/v1/chat/completions?key=secret',
                     'https://example.com/v1/other'):
        with pytest.raises(ValueError): UpstreamTransport(endpoint,'test')
    with fake_server(handler_for(body=JSON_OK)) as url:
        transport=UpstreamTransport(url,'test',max_request_bytes=20,max_response_bytes=10)
        with pytest.raises(ValueError): transport.stream({'data':'x'*100})
        exchange=transport.stream({'stream':False})
        with pytest.raises(UpstreamError): list(exchange)
        assert exchange.outcome=='unknown'


def test_compatible_mode_path_is_preserved_and_ambiguous_paths_rejected():
    seen=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_): pass
        def do_POST(self):
            seen.append(self.path); self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(JSON_OK)));self.end_headers();self.wfile.write(JSON_OK)
    with fake_server(Handler) as url:
        prefix=url.removesuffix('/v1/chat/completions')
        endpoint=prefix+'/compatible-mode/v1/chat/completions'
        exchange=UpstreamTransport(endpoint,'test').stream({'stream':False})
        assert b''.join(exchange)==JSON_OK and exchange.outcome=='answered'
        for path in ('/compatible-mode/../v1/chat/completions',
                     '/compatible-mode%2fv1/chat/completions',
                     '/compatible-mode/v1/chat/completions?key=x'):
            with pytest.raises(ValueError): UpstreamTransport(prefix+path,'test')
    assert seen==['/compatible-mode/v1/chat/completions']
