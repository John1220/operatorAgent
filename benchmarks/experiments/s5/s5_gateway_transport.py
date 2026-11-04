"""Bounded, direct upstream transport for the experimental S5 gateway.

This module is an experimental building block, not an isolation boundary or a
complete cost-control system. It has no fallback, proxy, redirect following,
or automatic retry. It only accepts HTTPS endpoints, except IPv4 loopback HTTP
for offline tests. The credential remains in memory, never in an error message.

The caller must exhaust or close each exchange and must not present an EOF as
a completed answer unless ``outcome == 'answered'``. An HTTP 200 alone is not
enough: streamed completions require a well-formed SSE terminal [DONE] and a
finish reason; JSON requires a complete choices/message/finish_reason envelope.
Incomplete or malformed responses remain unknown. Provider usage is not parsed
here and remains unknown until independently observed from the raw response.
The monotonic deadline is checked around socket operations and on each read;
DNS, TLS, slow headers and a blocked write are not independently preemptible.
Cross-thread close is not a reliable interruption of an active iterator. A
separate process supervisor is required for a hard wall clock and cancellation.
"""
from __future__ import annotations

import http.client
import json
import os
import math
import socket
import ssl
import time
from urllib.parse import urlsplit

MAX_RESPONSE_BYTES = 64 * 1024 * 1024
MAX_WALL_SECONDS = 1800

# Only these literal labels cross the worker boundary or reach an audit file.
# Never derive labels from exception text, provider bodies, URLs or headers.
FAILURE_STAGES = frozenset({'unknown', 'request_encoding', 'request_send',
    'response_headers', 'response_body', 'response_validation', 'worker_start',
    'worker_input', 'worker_framing', 'worker_completion', 'worker_deadline',
    'worker_cancelled'})
FAILURE_CODES = frozenset({'unexpected_error', 'request_limit', 'request_invalid',
    'timeout', 'network_error', 'http_status', 'content_type', 'content_length',
    'response_limit', 'truncated', 'invalid_event', 'incomplete', 'worker_eof',
    'invalid_frame', 'nonzero_exit', 'cancelled', 'invalid_input'})


def failure_detail(value):
    """Copy only a closed diagnostic schema, including untrusted worker input."""
    if (type(value) is not dict or set(value) != {'stage', 'code', 'http_status'}
            or type(value.get('stage')) is not str or value['stage'] not in FAILURE_STAGES
            or type(value.get('code')) is not str or value['code'] not in FAILURE_CODES
            or (value['http_status'] is not None and
                (type(value['http_status']) is not int or not 100 <= value['http_status'] <= 599))
            or (value['http_status'] is not None and value['code'] != 'http_status')):
        return dict(stage='unknown', code='unexpected_error', http_status=None)
    return {key:value[key] for key in ('stage', 'code', 'http_status')}


class UpstreamError(RuntimeError):
    """Bounded upstream request could not be established or completed."""
    def __init__(self, message, *, stage='unknown', code='unexpected_error', http_status=None):
        super().__init__(message)
        self.diagnostic = failure_detail(dict(stage=stage, code=code, http_status=http_status))


def _positive_int(value, name, maximum):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError('invalid ' + name)
    return value


def _endpoint(value):
    if not isinstance(value, str) or len(value) > 2048:
        raise ValueError('invalid upstream endpoint')
    try:
        u=urlsplit(value)
        port=u.port
    except ValueError:
        raise ValueError('invalid upstream endpoint') from None
    allowed_paths=('/v1/chat/completions', '/compatible-mode/v1/chat/completions',
                   '/v1/responses', '/compatible-mode/v1/responses')
    if (u.scheme not in ('https','http') or not u.hostname or u.username or u.password
            or u.path not in allowed_paths or u.query or u.fragment
            or (u.scheme == 'http' and u.hostname != '127.0.0.1')):
        raise ValueError('invalid upstream endpoint')
    return u.scheme,u.hostname,port or (443 if u.scheme == 'https' else 80),u.path


def _choice_finished(value, require_message):
    if (not isinstance(value,dict) or 'error' in value
            or not isinstance(value.get('choices'),list)):
        return False
    for choice in value['choices']:
        if (isinstance(choice,dict) and isinstance(choice.get('finish_reason'),str)
                and choice['finish_reason'] and
                (not require_message or isinstance(choice.get('message'),dict))):
            return True
    return False


def _sse_boundary(data):
    """Find an empty SSE line, preserving LF, CRLF and CR separators."""
    start=i=0
    while i < len(data):
        char=data[i]
        if char not in (10,13):
            i+=1
            continue
        if char==13 and i+1==len(data):
            return None  # The next chunk may complete a CRLF separator.
        end=i+1
        if char==13 and data[i+1]==10: end+=1
        if i==start: return end
        start=end
        i=end
    return None


def _sse_event(raw):
    """Return (is_data, is_done, is_finish) for one complete event."""
    text=raw.decode('utf-8').replace('\r\n','\n').replace('\r','\n')
    lines=[]
    for line in text.split('\n'):
        if line.startswith('event:') and line[6:].strip()=='error':
            raise ValueError('error event')
        if line=='data':
            lines.append('')
        elif line.startswith('data:'):
            value=line[5:]
            lines.append(value[1:] if value.startswith(' ') else value)
    if not lines: return False,False,False
    data='\n'.join(lines)
    if data=='[DONE]': return True,True,False
    obj=json.loads(data,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('constant')))
    if not isinstance(obj,dict) or 'error' in obj: raise ValueError('invalid event')
    choices=obj.get('choices',[])
    if not isinstance(choices,list): raise ValueError('invalid choices')
    for choice in choices:
        if not isinstance(choice,dict): raise ValueError('invalid choice')
        reason=choice.get('finish_reason')
        if reason is not None and (not isinstance(reason,str) or not reason):
            raise ValueError('invalid finish reason')
    return True,False,_choice_finished(obj,False)


def _responses_sse_event(raw):
    """Return (is_data, is_terminal, event_type) for one Responses SSE event."""
    text=raw.decode('utf-8').replace('\r\n','\n').replace('\r','\n')
    lines=[]
    for line in text.split('\n'):
        if line.startswith('event:') and line[6:].strip()=='error':
            raise ValueError('error event')
        if line=='data':
            lines.append('')
        elif line.startswith('data:'):
            value=line[5:]
            lines.append(value[1:] if value.startswith(' ') else value)
    if not lines: return False,False,None
    obj=json.loads('\n'.join(lines),parse_constant=lambda _: (_ for _ in ()).throw(ValueError('constant')))
    if not isinstance(obj,dict) or 'error' in obj: raise ValueError('invalid event')
    kind=obj.get('type')
    if not isinstance(kind,str) or not kind: raise ValueError('invalid event type')
    if kind=='error': raise ValueError('error event')
    terminal=kind in ('response.completed','response.failed','response.incomplete')
    if terminal and kind=='response.completed':
        response=obj.get('response')
        if not isinstance(response,dict) or response.get('status')!='completed':
            raise ValueError('incomplete terminal event')
    return True,terminal,kind


def _valid_responses(data, streaming):
    try:
        text=data.decode('utf-8')
        if not streaming:
            obj=json.loads(text,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('constant')))
            return (isinstance(obj,dict) and obj.get('status')=='completed'
                    and isinstance(obj.get('output'),list))
        normalized=text.replace('\r\n','\n').replace('\r','\n')
        # A provider may end the stream with a final event that has no trailing
        # blank line; that event is still complete when its JSON parses.
        events=[]
        for part in normalized.split('\n\n'):
            if not part.strip(): continue
            is_data,terminal,kind=_responses_sse_event((part+'\n\n').encode('utf-8'))
            if is_data: events.append((terminal,kind))
        if not events: return False
        if any(terminal for terminal,_ in events[:-1]): return False
        terminal,kind=events[-1]
        return terminal and kind=='response.completed'
    except (UnicodeDecodeError,ValueError,TypeError):
        return False


def _valid_response(data, streaming, protocol='chat_completions'):
    if protocol=='responses':
        return _valid_responses(data,streaming)
    try:
        text=data.decode('utf-8')
        if not streaming:
            return _choice_finished(json.loads(text,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('constant'))), True)
        normalized=text.replace('\r\n','\n').replace('\r','\n')
        if not normalized.endswith('\n\n'):
            return False
        events=[]
        for part in normalized.split('\n\n'):
            if not part:
                continue
            is_data,is_done,is_finish=_sse_event((part+'\n\n').encode('utf-8'))
            if is_data: events.append((is_done,is_finish))
        if not events or not events[-1][0] or any(event[0] for event in events[:-1]):
            return False
        return any(event[1] for event in events[:-1])
    except (UnicodeDecodeError,ValueError,TypeError):
        return False


class UpstreamExchange:
    """A single-use iterable; close early on cancellation.

    status is not a provider receipt or billing statement. A yielded chunk may
    reach the consumer before a later framing/timeout error is discovered.
    """
    def __init__(self, transport, body, streaming, protocol='chat_completions'):
        self._transport=transport
        self._body=body
        self._streaming=streaming
        self._protocol=protocol
        self._iterator=self._run()
        self._closed=False
        self.outcome='not_started'
        self.provider_usage=None
        self.usage_evidence='not_collected'
        self.failure=None

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator)

    def close(self):
        if not self._closed:
            self._iterator.close()
            self._closed=True
            if self.outcome in ('not_started','in_flight'):
                self.outcome='cancelled'

    def _run(self):
        transport=self._transport
        started=time.monotonic()
        deadline=started+transport.timeout_seconds
        conn=None
        response=None
        self.outcome='in_flight'
        phase='request_send'
        try:
            conn_type=http.client.HTTPSConnection if transport.scheme=='https' else http.client.HTTPConnection
            kwargs={'timeout':transport.timeout_seconds}
            if transport.scheme=='https':
                kwargs['context']=ssl.create_default_context()
            conn=conn_type(transport.host,transport.port,**kwargs)
            # http.client does not consult proxy environment, redirect or retry.
            conn.request('POST',transport.path,body=self._body,headers={
                'Content-Type':'application/json',
                'Accept':'text/event-stream' if self._streaming else 'application/json',
                'Authorization':'Bearer '+transport._api_key,
                'Connection':'close',
            })
            if time.monotonic() >= deadline:
                raise UpstreamError('upstream deadline exceeded',stage=phase,code='timeout')
            if conn.sock is not None:
                conn.sock.settimeout(max(.001,deadline-time.monotonic()))
            phase='response_headers'
            response=conn.getresponse()
            # HTTP/1.0 or Connection: close can detach conn.sock while the
            # response's buffered reader still owns the socket.
            response_socket = conn.sock or getattr(getattr(getattr(response,'fp',None),'raw',None),'_sock',None)
            if response.status != 200:
                raise UpstreamError('upstream rejected request',stage=phase,
                                    code='http_status',http_status=response.status)
            expected='text/event-stream' if self._streaming else 'application/json'
            actual=response.getheader('Content-Type','').split(';',1)[0].strip().lower()
            if actual != expected:
                raise UpstreamError('upstream content type invalid',stage=phase,code='content_type')
            content_length=response.getheader('Content-Length')
            if content_length is not None:
                if not content_length.isascii() or not content_length.isdigit():
                    raise UpstreamError('upstream length invalid',stage=phase,code='content_length')
                if int(content_length)>transport.max_response_bytes:
                    raise UpstreamError('upstream response exceeds bound',stage=phase,code='response_limit')
            received=bytearray()
            pending=bytearray()
            withheld=bytearray()
            holding=False
            phase='response_body'
            while True:
                # read1 closes its buffered socket once Content-Length is
                # consumed. Do not touch that detached socket on the next
                # iteration; validation below still checks length and EOF.
                if response.isclosed():
                    break
                remaining=deadline-time.monotonic()
                if remaining <= 0: raise UpstreamError('upstream deadline exceeded',stage=phase,code='timeout')
                if response_socket is not None: response_socket.settimeout(max(.001,remaining))
                chunk=response.read1(min(65536,transport.max_response_bytes+1-len(received)))
                if not chunk:
                    break
                received.extend(chunk)
                if len(received)>transport.max_response_bytes:
                    raise UpstreamError('upstream response exceeds bound',stage=phase,code='response_limit')
                if not self._streaming:
                    yield chunk
                    continue
                pending.extend(chunk)
                while True:
                    boundary=_sse_boundary(pending)
                    if boundary is None: break
                    event=bytes(pending[:boundary]); del pending[:boundary]
                    try:
                        if self._protocol=='responses':
                            _,terminal,_=_responses_sse_event(event)
                        else:
                            _,done,finished=_sse_event(event)
                            terminal=done or finished
                    except (ValueError,UnicodeDecodeError,TypeError):
                        raise UpstreamError('upstream event invalid',stage='response_validation',code='invalid_event') from None
                    # Never forward finish_reason, [DONE] or response.completed
                    # until the complete upstream stream and EOF have both been
                    # validated. A consumer may otherwise treat a later
                    # truncated stream as complete even without an end marker.
                    if terminal: holding=True
                    if holding: withheld.extend(event)
                    else: yield event
            if content_length is not None and len(received)!=int(content_length):
                raise UpstreamError('upstream response truncated',stage=phase,code='truncated')
            phase='response_validation'
            if self._streaming and pending:
                # A provider may end the stream without the trailing blank line
                # that normally closes the last event. Try that tail as one
                # complete event before declaring the stream truncated.
                try:
                    if self._protocol=='responses':
                        is_data,terminal,_kind=_responses_sse_event(bytes(pending))
                    else:
                        is_data,done,finished=_sse_event(bytes(pending))
                        terminal=done or finished
                except (ValueError,UnicodeDecodeError,TypeError):
                    raise UpstreamError('upstream event truncated',stage=phase,code='truncated') from None
                if not is_data:
                    raise UpstreamError('upstream event truncated',stage=phase,code='truncated')
                if terminal: holding=True
                if holding: withheld.extend(pending)
                else: yield bytes(pending)
                pending.clear()
            if not _valid_response(bytes(received),self._streaming,self._protocol):
                raise UpstreamError('upstream response incomplete',stage=phase,code='incomplete')
            if withheld:
                yield bytes(withheld)
            self.outcome='answered'
        except GeneratorExit:
            self.outcome='cancelled'
            self.failure=dict(stage='worker_cancelled',code='cancelled',http_status=None)
            raise
        except Exception as exc:
            self.outcome='unknown'
            # Underlying exceptions may contain provider URLs, credentials,
            # headers or payloads. Do not chain or echo them.
            code=('timeout' if isinstance(exc,(TimeoutError,socket.timeout)) else
                  'network_error' if isinstance(exc,(OSError,http.client.HTTPException)) else 'unexpected_error')
            self.failure=(failure_detail(exc.diagnostic) if isinstance(exc,UpstreamError) else
                          failure_detail(dict(stage=phase,code=code,http_status=None)))
            raise UpstreamError('upstream request unavailable or incomplete',**self.failure) from None
        finally:
            if response is not None:
                response.close()
            if conn is not None:
                conn.close()
            self._closed=True


class UpstreamTransport:
    def __init__(self, endpoint, api_key, *, timeout_seconds=30,
                 max_request_bytes=1048576, max_response_bytes=1048576):
        self.scheme,self.host,self.port,self.path=_endpoint(endpoint)
        if (not isinstance(api_key,str) or not api_key or len(api_key)>4096
                or '\n' in api_key or '\r' in api_key):
            raise ValueError('invalid upstream credential')
        self.protocol='responses' if self.path.endswith('/responses') else 'chat_completions'
        if (type(timeout_seconds) not in (int,float) or not math.isfinite(timeout_seconds)
                or not 0 < timeout_seconds <= MAX_WALL_SECONDS):
            raise ValueError('invalid upstream timeout')
        self.timeout_seconds=float(timeout_seconds)
        self.max_request_bytes=_positive_int(max_request_bytes,'request bound',10485760)
        self.max_response_bytes=_positive_int(max_response_bytes,'response bound',MAX_RESPONSE_BYTES)
        self._api_key=api_key

    def stream(self, request):
        if type(request) is not dict or type(request.get('stream',False)) is not bool:
            raise ValueError('invalid upstream request')
        try:
            body=json.dumps(request,ensure_ascii=False,allow_nan=False,sort_keys=True,
                            separators=(',', ':')).encode('utf-8')
        except (TypeError,ValueError,OverflowError,RecursionError,UnicodeError):
            raise ValueError('invalid upstream request') from None
        if len(body)>self.max_request_bytes:
            raise ValueError('upstream request exceeds bound')
        return UpstreamExchange(self,body,request.get('stream',False),self.protocol)
