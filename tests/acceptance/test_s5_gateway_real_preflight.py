"""One-call DSH preflight tests, with a loopback fake upstream only."""
import json
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from benchmarks.experiments.s5 import s5_gateway_real_preflight as pre


class FakeUpstream:
    def __init__(self,with_usage=True):
        calls=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                calls.append((body,self.headers.get('Authorization')))
                chunks=[{'id':'fake-1','model':'fake-model','choices':[{'index':0,
                          'delta':{'content':'OK'},'finish_reason':None}]},
                        {'id':'fake-1','choices':[{'index':0,'delta':{},'finish_reason':'stop'}]}]
                if with_usage: chunks.append({'choices':[],'usage':{'prompt_tokens':4,'completion_tokens':1,'total_tokens':5}})
                data=''.join('data: '+json.dumps(chunk)+'\n\n' for chunk in chunks)+'data: [DONE]\n\n'
                raw=data.encode();self.send_response(200);self.send_header('Content-Type','text/event-stream')
                self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler);self.calls=calls
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
    @property
    def endpoint(self): return 'http://127.0.0.1:%d/v1/chat/completions' % self.server.server_address[1]
    def __enter__(self): self.thread.start();return self
    def __exit__(self,*exc): self.server.shutdown();self.server.server_close();self.thread.join()


@pytest.mark.skipif(not shutil.which('node') or not (pre.DSH_ROOT/'dsh-llm-pi-ai/lib/index.js').is_file(),
                    reason='installed DSH unavailable')
@pytest.mark.parametrize('with_usage',[True,False])
def test_exactly_one_dsh_call_and_no_secret_on_disk(tmp_path,with_usage):
    secret='local-placeholder-secret'
    with FakeUpstream(with_usage) as upstream:
        out=tmp_path/'trial';sha=pre.freeze(out,upstream.endpoint,'fake-model')
        result=pre.execute(out,sha,secret)
        assert result['status']=='passed', result
        assert result['model_reservations']==1 and result['gpu_used'] is False
        assert result['response_bytes']==2 and len(result['response_sha256'])==64
        assert result['direct_worker_cleanup']=='direct_child_reaped_descendants_unknown'
        assert result['api']=='openai-completions'
        assert result['provider_usage']==({'prompt_tokens':4,'completion_tokens':1,'total_tokens':5} if with_usage else None)
        assert result['usage_source']==('raw_upstream' if with_usage else 'unknown')
        assert len(upstream.calls)==1 and upstream.calls[0][1]=='Bearer '+secret
        assert upstream.calls[0][0]['max_tokens']==256 and not upstream.calls[0][0].get('tools')
        with pytest.raises(FileExistsError): pre.execute(out,sha,secret)
    for path in out.rglob('*'):
        if path.is_file(): assert secret.encode() not in path.read_bytes()


def test_bad_or_changed_plan_refused_before_execution(tmp_path):
    out=tmp_path/'trial';sha=pre.freeze(out,'http://127.0.0.1:1/v1/chat/completions','fake-model')
    with pytest.raises(ValueError,match='hash mismatch'): pre.execute(out,'0'*64,'x')
    assert not (out/'execution-started.json').exists()
    plan=json.loads((out/'plan.json').read_text());plan['max_calls']=2
    (out/'plan.json').write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='hash mismatch'): pre.execute(out,sha,'x')


@pytest.mark.parametrize('url',['http://example.com/v1/chat/completions',
    'https://user:password@example.com/v1/chat/completions',
    'https://example.com/v1/chat/completions?token=secret',
    'https://example.com/untrusted/v1/chat/completions'])
def test_unsafe_endpoint_refused(tmp_path,url):
    with pytest.raises(ValueError): pre.freeze(tmp_path/'out',url,'fake-model')


def test_ambiguous_or_invalid_raw_usage_remains_unknown():
    assert pre._usage_sse(b'data: {"usage":{"prompt_tokens":0}}\n\ndata: [DONE]\n\n') is None
    assert pre._usage_sse(b'data: {"usage":{"prompt_tokens":true,"completion_tokens":1}}\n\n') is None
    assert pre._usage_sse(b'data: {"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":0}}\n\n') is None
    assert pre._usage_sse(b'data: {"usage":{"prompt_tokens":1,"completion_tokens":257}}\n\n') is None


@pytest.mark.parametrize('changes',[
    {'messages':[{'role':'system','content':'private'},{'role':'user','content':pre.PROMPT}]},
    {'messages':[{'role':'user','content':'different prompt'}]},
    {'tools':[{'type':'function'}]},
    {'max_tokens':257},
])
def test_capture_refuses_unapproved_context_before_upstream(changes):
    calls=[]
    capture=pre._Capture(lambda request: calls.append(request))
    request=dict(stream=True,max_tokens=256,messages=[{'role':'user','content':pre.PROMPT}])
    request.update(changes)
    with pytest.raises(ValueError): capture(request)
    assert not calls and capture.exchange is None


def test_parent_timeout_closes_owned_exchange_and_reports_cleanup(tmp_path, monkeypatch):
    closed=[]
    class Exchange:
        outcome='in_flight';cleanup='unknown'
        def close(self):
            closed.append(True);self.cleanup='direct_child_reaped_descendants_unknown'
    class Capture:
        def __init__(self,transport): self.exchange=Exchange();self.usage=None
        def __call__(self,request): raise AssertionError('unexpected model call')
    monkeypatch.setattr(pre,'_Capture',Capture)
    monkeypatch.setattr(pre.subprocess,'run',lambda *a,**kw:(_ for _ in ()).throw(subprocess.TimeoutExpired('node',1)))
    out=tmp_path/'run';sha=pre.freeze(out,'http://127.0.0.1:1/v1/chat/completions','fake-model')
    result=pre.execute(out,sha,'placeholder')
    assert result['status']=='unknown' and result['error']=='TimeoutExpired'
    assert result['direct_worker_cleanup']=='direct_child_reaped_descendants_unknown' and closed==[True]
