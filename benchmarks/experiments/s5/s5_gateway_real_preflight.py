"""One explicitly frozen DSH→gateway request; never invokes GPU or tools.

This experimental helper is not an isolation boundary or billing cap. Actual
HTTPS dispatch is initiated only by an authorized caller of execute(), after
freeze() and exact plan-hash confirmation. Credentials are passed as a Python
argument to the in-memory gateway supervisor; they never enter DSH, argv,
environment, the plan or result files. Tests use loopback fake providers only.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from benchmarks.experiments.s5.s5_gateway_http import GatewayServer
from benchmarks.experiments.s5.s5_gateway_supervisor import SupervisedTransport
from benchmarks.experiments.s5.s5_gateway_transport import _endpoint as _transport_endpoint
from benchmarks.experiments.s5.s5_vertical_headless import DSH_ROOT
from operator_agent import s5_shared_budget as shared

SCHEMA='s5-single-model-preflight-v1'
PROMPT='Reply with exactly OK.'
MODEL_PATTERN=re.compile(r'[A-Za-z0-9_.:/-]{1,128}\Z')


def _json(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()


def _write(path, value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())


def _identities(node):
    """Bind the direct client and gateway sources; not a full dependency lock."""
    root=Path(__file__).resolve().parent
    files={'node':Path(node),'dsh_cli':DSH_ROOT/'dsh/lib/bin.js',
           'dsh_llm':DSH_ROOT/'dsh-llm/lib/index.js',
           'dsh_pi_ai':DSH_ROOT/'dsh-llm-pi-ai/lib/index.js',
           'preflight':Path(__file__).resolve(),
           **{name:root.parents[2]/'operator_agent'/(name+'.py') for name in
              ('s5_shared_budget','run_layout','task_clock')},
           **{name:root/(name+'.py') for name in
              ('s5_gateway_http','s5_gateway_policy','s5_gateway_supervisor','s5_gateway_transport')}}
    return {name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}


def _endpoint(value):
    _transport_endpoint(value)
    return value


def freeze(output, endpoint, model, *, wall_seconds=30):
    """Create one non-overwritable plan and return its on-disk SHA256."""
    endpoint=_endpoint(endpoint)
    if not isinstance(model,str) or not MODEL_PATTERN.fullmatch(model): raise ValueError('invalid model')
    if type(wall_seconds) not in (int,float) or not math.isfinite(wall_seconds) or not 5<=wall_seconds<=60:
        raise ValueError('invalid wall_seconds')
    node=shutil.which('node'); cli=DSH_ROOT/'dsh/lib/bin.js'
    if not node or not cli.is_file() or not (DSH_ROOT/'dsh-llm-pi-ai/lib/index.js').is_file():
        raise RuntimeError('installed DSH unavailable')
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False,mode=0o700)
    plan=dict(schema=SCHEMA, endpoint=endpoint,model=model,api='openai-completions',
              prompt=PROMPT,max_calls=1,max_output_tokens=256,max_request_bytes=65536,
              max_response_bytes=1048576,wall_seconds=float(wall_seconds),node=node,cli=str(cli),
              source_sha256=_identities(node),client_sha256=hashlib.sha256(_DSH_CLIENT.encode()).hexdigest(),
              scope='one DSH LlmRuntime/pi-ai request via gateway, no agent or tools, no GPU')
    _write(root/'plan.json',plan)
    return hashlib.sha256((root/'plan.json').read_bytes()).hexdigest()


def _usage_sse(body):
    """Use only explicit valid raw upstream usage, never normalized DSH zeros."""
    try:
        text=body.decode('utf-8').replace('\r\n','\n').replace('\r','\n')
        values=[]
        for part in text.split('\n\n'):
            data=[]
            for line in part.split('\n'):
                if line.startswith('data:'):
                    value=line[5:]; data.append(value[1:] if value.startswith(' ') else value)
            if not data or '\n'.join(data)=='[DONE]': continue
            obj=json.loads('\n'.join(data))
            if isinstance(obj,dict) and 'usage' in obj:
                if obj['usage'] is None: continue
                values.append(obj['usage'])
        if len(values)!=1 or not isinstance(values[0],dict): return None
        u=values[0];prompt=u.get('prompt_tokens');completion=u.get('completion_tokens')
        if any(type(x) is not int or not 0<=x<=2**63-1 for x in (prompt,completion)):
            return None
        if completion > 256:
            return None
        result=dict(prompt_tokens=prompt,completion_tokens=completion)
        if 'total_tokens' in u:
            total=u['total_tokens']
            if type(total) is not int or total != prompt+completion: return None
            result['total_tokens']=total
        return result
    except (ValueError,TypeError,UnicodeDecodeError): return None


class _Capture:
    """Captures bounded raw bytes in memory to inspect explicitly supplied usage."""
    def __init__(self,transport):
        self.transport=transport;self.exchange=None;self.usage=None
    def __call__(self, request):
        if (request.get('stream') is not True or request.get('max_tokens')!=256
                or request.get('tools') not in (None,[]) or request.get('tool_choice') is not None
                or request.get('messages')!=[{'role':'user','content':PROMPT}]):
            raise ValueError('preflight requires one bounded tool-free request')
        exchange=self.transport(request);self.exchange=exchange
        def consume():
            data=bytearray()
            try:
                for chunk in exchange:
                    data.extend(chunk)
                    if len(data)>1048576: raise ValueError('response limit')
                    yield chunk
                if exchange.outcome=='answered': self.usage=_usage_sse(bytes(data))
            finally: exchange.close()
        return consume()


def execute(output, expected_plan_sha256, api_key):
    """Execute exactly once, writing no credential, raw prompts or responses."""
    root=Path(output).resolve();plan_path=root/'plan.json'
    if plan_path.is_symlink() or not plan_path.is_file(): raise ValueError('plan missing')
    raw=plan_path.read_bytes()
    if not isinstance(expected_plan_sha256,str) or hashlib.sha256(raw).hexdigest()!=expected_plan_sha256:
        raise ValueError('plan hash mismatch')
    plan=json.loads(raw)
    if (type(plan) is not dict or plan.get('schema')!=SCHEMA or plan.get('api')!='openai-completions'
        or plan.get('prompt')!=PROMPT or plan.get('max_calls')!=1 or plan.get('max_output_tokens')!=256
        or plan.get('max_request_bytes')!=65536 or plan.get('max_response_bytes')!=1048576
        or not isinstance(plan.get('model'),str) or not MODEL_PATTERN.fullmatch(plan['model'])
        or type(plan.get('wall_seconds')) not in (float,int) or not 5<=plan['wall_seconds']<=60
        or plan.get('cli')!=str(DSH_ROOT/'dsh/lib/bin.js') or plan.get('node')!=shutil.which('node')):
        raise ValueError('plan configuration mismatch')
    _endpoint(plan.get('endpoint'))
    if (plan.get('source_sha256')!=_identities(plan['node']) or
            plan.get('client_sha256')!=hashlib.sha256(_DSH_CLIENT.encode()).hexdigest()):
        raise ValueError('runtime identity drift')
    if not isinstance(api_key,str) or not api_key or '\n' in api_key or '\r' in api_key:
        raise ValueError('credential unavailable')
    # No rerun under an existing output identity, including failures/timeouts.
    _write(root/'execution-started.json',dict(plan_sha256=expected_plan_sha256,no_resume=True))
    start=time.monotonic();ledger=root/'budget.sqlite'
    result=dict(schema='s5-single-model-preflight-result-v1',status='unknown',error=None,
                real_provider_call='unknown',provider_usage=None,usage_source='unknown',
                gpu_used=False,formal_s5_credit=False,isolation_verified=False,
                api=plan['api'],model=plan['model'],plan_sha256=expected_plan_sha256,
                direct_worker_cleanup='unknown',remote_provider_stopped='unknown')
    capture=None
    try:
        shared.create(ledger,task_id='one-call-preflight',max_calls=1,max_output_tokens=256,
                      total_output_tokens=256,wall_clock_s=plan['wall_seconds']+30)
        transport=SupervisedTransport(plan['endpoint'],api_key,wall_seconds=plan['wall_seconds'],
                                     max_request_bytes=65536,max_response_bytes=1048576)
        capture=_Capture(transport)
        home=root/'home';home.mkdir(mode=0o700)
        work=root/'work';work.mkdir(mode=0o700)
        script=root/'client.mjs'
        script.write_text(_DSH_CLIENT,encoding='utf-8')
        with GatewayServer(ledger,task_id='one-call-preflight',model=plan['model'],
                           participant='outer',adapter=capture,max_request_bytes=65536) as gateway:
            env={'PATH':'/usr/local/bin:/usr/bin:/bin','HOME':str(home),'DSH_HOME':str(home/'dsh'),
                 'S5_FAKE_API_KEY':'placeholder','NO_PROXY':'127.0.0.1','no_proxy':'127.0.0.1'}
            proc=subprocess.run([plan['node'],str(script),gateway.base_url,plan['model'],str(DSH_ROOT)],
                cwd=work,env=env,stdin=subprocess.DEVNULL,capture_output=True,
                text=True,timeout=plan['wall_seconds']+20)
        if len(proc.stdout.encode())>1048576 or len(proc.stderr.encode())>1048576:
            raise ValueError('headless output too large')
        event=json.loads(proc.stdout)
        snapshot=shared.snapshot(ledger,expected_task_id='one-call-preflight')
        _write(root/'budget.json',snapshot)
        result['model_reservations']=snapshot['reserved_calls']
        if capture.exchange is not None:
            result['direct_worker_cleanup']=capture.exchange.cleanup
            result['real_provider_call']='handoff_attempted'
        if (proc.returncode!=0 or snapshot['reserved_calls']!=1 or snapshot['completed_calls']!=1
            or capture.exchange is None or capture.exchange.outcome!='answered'
            or event.get('reason')!='stop' or event.get('tool_calls')!=0
            or not isinstance(event.get('text'),str) or not event['text'].strip()):
            raise ValueError('preflight evidence incomplete')
        result['provider_usage']=capture.usage
        result['usage_source']='raw_upstream' if capture.usage is not None else 'unknown'
        response_bytes=event['text'].encode('utf-8')
        result['response_sha256']=hashlib.sha256(response_bytes).hexdigest()
        result['response_bytes']=len(response_bytes)
        result['status']='passed'
    except Exception as exc:
        result['error']=type(exc).__name__
        if capture is not None and capture.exchange is not None:
            result['direct_worker_cleanup']=capture.exchange.cleanup
            result['real_provider_call']='handoff_attempted'
        try:
            if ledger.exists() and not (root/'budget.json').exists():
                _write(root/'budget.json',shared.snapshot(ledger,expected_task_id='one-call-preflight'))
        except Exception:
            pass
    finally:
        # The listener uses daemon handlers. Explicitly stop the owned upstream
        # exchange after success, timeout or failure; a reaped direct worker is
        # not proof that the remote service or escaped descendants have stopped.
        if capture is not None and capture.exchange is not None:
            try:
                capture.exchange.close()
                result['direct_worker_cleanup']=capture.exchange.cleanup
            except Exception:
                result['direct_worker_cleanup']='unknown'
                result['status']='unknown'
        if (result['status']=='passed' and
                result['direct_worker_cleanup']!='direct_child_reaped_descendants_unknown'):
            result['status']='unknown'
            result['error']='worker_cleanup_unproven'
    result['wall_seconds']=round(time.monotonic()-start,3)
    _write(root/'result.json',result)
    return result


_DSH_CLIENT = r'''import {pathToFileURL} from 'node:url';
import path from 'node:path';
const [base,model,root]=process.argv.slice(2);
const {Context}=await import(pathToFileURL(path.join(root,'cordis/lib/index.js')));
const {LlmRuntime}=await import(pathToFileURL(path.join(root,'dsh-llm/lib/index.js')));
const piAi=await import(pathToFileURL(path.join(root,'dsh-llm-pi-ai/lib/index.js')));
const ctx=new Context();ctx.plugin(LlmRuntime);
ctx.plugin(piAi,{providers:{fixture:{api:'openai-completions',baseURL:base,
  apiKeyEnv:'S5_FAKE_API_KEY',models:[{id:model,contextWindow:16384,maxTokens:256}],
  compat:{maxTokensField:'max_tokens',supportsDeveloperRole:false}}}});
await new Promise(resolve=>setTimeout(resolve,100));
const chunks=await Array.fromAsync(ctx.llm.stream({provider:'fixture',model,maxTokens:256,
 messages:[{role:'user',content:[{type:'text',text:'Reply with exactly OK.'}]}]}));
const text=chunks.filter(c=>c.type==='text-delta').map(c=>c.text).join('');
const reason=chunks.find(c=>c.type==='finish')?.reason?.kind;
const tool_calls=chunks.filter(c=>c.type==='tool-call').length;
process.stdout.write(JSON.stringify({text,reason,tool_calls})+'\n');
process.exit(0);
'''
