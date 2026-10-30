"""Offline loopback checks for supervised upstream process lifecycle."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

import pytest
from benchmarks.s5_gateway_supervisor import SupervisedTransport
from benchmarks.s5_gateway_transport import UpstreamError

SECRET='fixture-worker-secret'
FIRST=b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n\n'
LAST=b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
REQUEST={'model':'fake','messages':[{'role':'user','content':'test'}],'max_tokens':8,'stream':True}


class Server:
    def __init__(self,mode):
        self.entered=threading.Event();self.release=threading.Event();self.auth=[]
        self.mode=mode
        owner=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                owner.auth.append(self.headers.get('Authorization'))
                owner.entered.set()
                if owner.mode=='headers':
                    owner.release.wait(4)
                    return
                if owner.mode=='drip':
                    try:
                        for byte in b'HTTP/1.0 200 OK\r\nContent-Type: text/event-stream\r\n\r\n':
                            if owner.release.is_set(): break
                            self.wfile.write(bytes([byte]));self.wfile.flush()
                            time.sleep(.1)
                    except (BrokenPipeError,ConnectionResetError): pass
                    return
                if owner.mode=='http_error':
                    self.send_response(503)
                    self.send_header('Content-Type','application/json')
                    self.end_headers()
                    self.wfile.write(SECRET.encode())
                    return
                if owner.mode in ('invalid_event','missing_terminal','wrong_type'):
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json' if owner.mode=='wrong_type' else 'text/event-stream')
                    self.end_headers()
                    self.wfile.write(('data: '+SECRET+'\n\n').encode() if owner.mode=='invalid_event' else FIRST)
                    return
                try:
                    self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
                    self.wfile.write(FIRST);self.wfile.flush()
                    owner.release.wait(4)
                    self.wfile.write(LAST);self.wfile.flush()
                except (BrokenPipeError,ConnectionResetError): pass
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.server.daemon_threads=True
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
    @property
    def endpoint(self): return 'http://127.0.0.1:%d/v1/chat/completions' % self.server.server_address[1]
    def __enter__(self): self.thread.start();return self
    def __exit__(self,*exc):
        self.release.set();self.server.shutdown();self.server.server_close();self.thread.join(timeout=3)


def test_stream_delivers_before_upstream_completion_and_reaps_worker():
    with Server('stream') as upstream:
        transport=SupervisedTransport(upstream.endpoint,SECRET,wall_seconds=5)
        exchange=transport(REQUEST)
        started=time.monotonic();first=next(exchange)
        assert FIRST in first and time.monotonic()-started<3
        assert upstream.auth==['Bearer '+SECRET]
        upstream.release.set();rest=b''.join(exchange)
        assert b'[DONE]' in rest and exchange.outcome=='answered'
        assert exchange.cleanup=='direct_child_reaped_descendants_unknown'
        assert SECRET not in repr(exchange._process.args)


def test_worker_deadline_stops_stalled_headers_and_leaves_outcome_unknown():
    with Server('headers') as upstream:
        exchange=SupervisedTransport(upstream.endpoint,SECRET,wall_seconds=0.8)(REQUEST)
        started=time.monotonic()
        with pytest.raises(UpstreamError) as caught: list(exchange)
        assert time.monotonic()-started<2.5
        assert exchange.outcome=='unknown' and exchange._process.poll() is not None
        assert exchange.cleanup=='direct_child_reaped_descendants_unknown'
        assert SECRET not in str(caught.value) and upstream.endpoint not in str(caught.value)


def test_worker_deadline_stops_trickled_headers_despite_socket_activity():
    with Server('drip') as upstream:
        exchange=SupervisedTransport(upstream.endpoint,SECRET,wall_seconds=0.8)(REQUEST)
        started=time.monotonic()
        with pytest.raises(UpstreamError): list(exchange)
        assert time.monotonic()-started<2.5
        assert exchange.outcome=='unknown' and exchange._process.poll() is not None


def test_watchdog_stops_worker_even_if_consumer_pauses_after_first_chunk():
    with Server('stream') as upstream:
        exchange=SupervisedTransport(upstream.endpoint,SECRET,wall_seconds=0.8)(REQUEST)
        assert FIRST in next(exchange)
        time.sleep(1.0)
        assert exchange._process.poll() is not None
        with pytest.raises(UpstreamError): next(exchange)
        assert exchange.outcome=='unknown'


def test_cross_thread_cancellation_unblocks_waiting_reader():
    with Server('headers') as upstream:
        exchange=SupervisedTransport(upstream.endpoint,SECRET,wall_seconds=5)(REQUEST)
        errors=[]
        def read():
            try: next(exchange)
            except BaseException as exc: errors.append(exc)
        thread=threading.Thread(target=read);thread.start()
        assert upstream.entered.wait(3)
        exchange.close();thread.join(timeout=2)
        assert not thread.is_alive() and errors and isinstance(errors[0],UpstreamError)
        assert exchange.outcome=='cancelled' and exchange._process.poll() is not None
        assert exchange.cleanup=='direct_child_reaped_descendants_unknown'


def test_abandoned_stream_close_stops_worker_before_provider_end():
    with Server('stream') as upstream:
        exchange=SupervisedTransport(upstream.endpoint,SECRET,wall_seconds=5)(REQUEST)
        assert FIRST in next(exchange)
        exchange.close()
        assert exchange.outcome=='cancelled' and exchange._process.poll() is not None
        assert not upstream.release.is_set()


def test_only_fixed_safe_destination_and_bounded_wall_time():
    for endpoint in ('http://example.com/v1/chat/completions',
                     'https://user:pass@example.com/v1/chat/completions',
                     'https://example.com/v1/chat/completions?secret=x'):
        with pytest.raises(ValueError): SupervisedTransport(endpoint,SECRET)
    with pytest.raises(ValueError): SupervisedTransport('http://127.0.0.1:1/v1/chat/completions',SECRET,wall_seconds=float('nan'))


@pytest.mark.parametrize('mode,stage,code,status',[
    ('http_error','response_headers','http_status',503),
    ('wrong_type','response_headers','content_type',None),
    ('invalid_event','response_validation','invalid_event',None),
    ('missing_terminal','response_validation','incomplete',None),
])
def test_failure_audit_preserves_only_verified_stage_and_http_status(mode,stage,code,status):
    with Server(mode) as upstream:
        transport=SupervisedTransport(upstream.endpoint,SECRET,wall_seconds=5)
        exchange=transport(REQUEST)
        with pytest.raises(UpstreamError):list(exchange)
        report=transport.failure_snapshot()
    assert report['calls_created']==1 and not report['truncated']
    assert report['failures']==[dict(call_sequence=1,stage=stage,code=code,http_status=status,
        cleanup='direct_child_reaped_descendants_unknown',outcome='unknown')]
    wire=json.dumps(report)
    assert SECRET not in wire and upstream.endpoint not in wire and 'test' not in wire
    assert set(report['failures'][0])=={'call_sequence','stage','code','http_status','cleanup','outcome'}


def test_failure_sequence_includes_prior_success_and_has_no_automatic_retry():
    with Server('stream') as upstream:
        upstream.release.set()
        transport=SupervisedTransport(upstream.endpoint,SECRET,wall_seconds=5)
        assert b'[DONE]' in b''.join(transport(REQUEST))
        assert transport.failure_snapshot()['failures']==[]
        upstream.mode='http_error'
        with pytest.raises(UpstreamError):list(transport(REQUEST))
        assert len(upstream.auth)==2
        assert transport.failure_snapshot()['failures'][0]['call_sequence']==2


def test_failure_snapshot_is_a_copy_and_deadline_cleanup_is_recorded():
    with Server('headers') as upstream:
        transport=SupervisedTransport(upstream.endpoint,SECRET,wall_seconds=.8)
        with pytest.raises(UpstreamError):list(transport(REQUEST))
        report=transport.failure_snapshot()
        row=report['failures'][0]
        # The worker's socket timeout and the parent's watchdog may race; both
        # are verified timeout stages, never an inferred provider HTTP error.
        assert row['stage'] in ('worker_deadline','response_headers') and row['code']=='timeout'
        assert row['http_status'] is None and row['cleanup']=='direct_child_reaped_descendants_unknown'
        report['failures'][0]['code']='altered'
        assert transport.failure_snapshot()['failures'][0]['code']=='timeout'


@pytest.mark.parametrize('value',[
    {'stage':SECRET,'code':'unexpected_error','http_status':None},
    {'stage':'response_headers','code':SECRET,'http_status':None},
    {'stage':'response_headers','code':'http_status','http_status':SECRET},
    {'stage':'response_headers','code':'http_status','http_status':True},
    {'stage':'response_headers','code':'http_status','http_status':503,'message':SECRET},
])
def test_untrusted_worker_diagnostic_cannot_add_raw_fields_or_labels(value):
    from benchmarks.s5_gateway_transport import failure_detail
    report=failure_detail(value)
    assert report==dict(stage='unknown',code='unexpected_error',http_status=None)
    assert SECRET not in json.dumps(report)


def test_cancelled_exchange_keeps_cleanup_in_failure_audit():
    with Server('stream') as upstream:
        transport=SupervisedTransport(upstream.endpoint,SECRET,wall_seconds=5)
        exchange=transport(REQUEST)
        assert FIRST in next(exchange)
        exchange.close()
        row=transport.failure_snapshot()['failures'][0]
        assert row==dict(call_sequence=1,stage='worker_cancelled',code='cancelled',http_status=None,
                        cleanup='direct_child_reaped_descendants_unknown',outcome='cancelled')
