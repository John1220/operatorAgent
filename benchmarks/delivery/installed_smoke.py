"""Validate installed wheel entry points from a fresh workspace, using fake only.

This controller uses only the standard library. The supplied Python must import
operator_agent from site-packages; no source tree is added to its import path.
Export is a delivery helper copying source/report with hashes, not a new product
CLI command. No provider, SSH or GPU is used.
"""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,sort_keys=True,indent=2)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition,message):
    if not condition:raise ValueError(message)


def environment(root,python):
    env={key:os.environ[key] for key in ('PATH','LANG','LC_ALL','SYSTEMROOT') if key in os.environ}
    env.update(HOME=str(root/'home'),TMPDIR=str(root/'tmp'),PYTHONNOUSERSITE='1',
        PYTHONDONTWRITEBYTECODE='1',OPERATOR_AGENT_OUTPUT_DIR=str(root/'output'),
        OPERATOR_AGENT_LOG_DIR=str(root/'log'),OPERATOR_AGENT_KERNEL_DB=str(root/'kernel_db.jsonl'),
        OPERATOR_AGENT_FAILURE_DB=str(root/'failure_db.jsonl'))
    env['PATH']=str(Path(python).parent)+os.pathsep+env.get('PATH','')
    for name in ('home','tmp'): (root/name).mkdir()
    return env


def invoke(python,argv,root,env):
    value=subprocess.run([str(python),'-I',*argv],cwd=root,env=env,
        capture_output=True,text=True,timeout=30)
    require(value.returncode==0,'installed command failed: '+value.stderr[-2000:])
    return json.loads(value.stdout)


class MCP:
    def __init__(self,python,root,env):
        self.root=root;self.counter=0;self.records=[];self.queue=queue.Queue()
        self.stderr=(root/'mcp-stderr.log').open('w',encoding='utf-8')
        self.process=subprocess.Popen([str(python),'-I','-m','operator_agent.app.mcp_server'],
            cwd=root,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,
            text=True,bufsize=1,start_new_session=True)
        def reader():
            for line in self.process.stdout:self.queue.put(line)
            self.queue.put(None)
        self.thread=threading.Thread(target=reader,daemon=True);self.thread.start()

    def rpc(self,method,params=None):
        self.counter+=1
        request=dict(jsonrpc='2.0',id=self.counter,method=method)
        if params is not None:request['params']=params
        self.process.stdin.write(json.dumps(request)+'\n');self.process.stdin.flush()
        deadline=time.monotonic()+45
        while time.monotonic()<deadline:
            try:line=self.queue.get(timeout=max(.01,deadline-time.monotonic()))
            except queue.Empty:raise TimeoutError('MCP request did not return') from None
            require(line is not None,'MCP exited before returning its response')
            response=json.loads(line)
            if 'id' not in response:continue
            require(response['id']==self.counter,'unexpected MCP response identity')
            self.records.append(dict(request=request,response=response))
            require('error' not in response,'MCP protocol refusal: '+repr(response))
            return response['result']
        raise TimeoutError('MCP timeout')

    def tool(self,name,arguments):
        result=self.rpc('tools/call',dict(name=name,arguments=arguments))
        body=json.loads(result['content'][0]['text'])
        require(not result.get('isError'),'MCP tool refused: '+repr(body))
        return body

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=3)
        self.thread.join(timeout=1);self.stderr.close()
        dump(self.root/'mcp-transcript.json',self.records)


def wait(mcp,run,predicate,timeout=35):
    end=time.monotonic()+timeout
    last=None
    while time.monotonic()<end:
        last=mcp.tool('operator_status',dict(run_dir=str(run)))
        if predicate(last):return last
        time.sleep(.05)
    raise TimeoutError('autonomous state did not settle: '+repr(last))


def test_only(value):
    context=value.get('execution_context',{})
    require(context.get('test_only') is True and context.get('gpu_acceptance') is False,
            'fake result missing permanent test-only boundary')


def _terminal(value):
    if isinstance(value,str):
        try:return _terminal(json.loads(value))
        except ValueError:return False
    if isinstance(value,dict):
        if 'phase' in value:return value['phase'] in ('done','failed','cancelled')
        return _terminal(value.get('content',value.get('text')))
    return isinstance(value,list) and any(_terminal(v) for v in value)


def dsh_headless(python,plugin,root,env):
    """Drive actual installed DSH+packed plugin with a local outer fixture."""
    runtime=Path('/Applications/DSH Desktop.app/Contents/Resources/app/node_modules/@deepseek-ai')
    entry=runtime/'dsh/lib/bin.js';client=runtime/'dsh-mcp-client/lib/index.js';node=shutil.which('node')
    require(bool(node) and entry.is_file() and client.is_file(),'installed DSH/Node is unavailable')
    work=root/'dsh';work.mkdir();run_dir=work/'task';requests=[];errors=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*a):pass
        def do_POST(self):
            try:
                require(self.path=='/v1/chat/completions','unexpected fixture endpoint')
                request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(request)
                require(request.get('model')=='installed-delivery-fixture' and request.get('stream') is True,
                        'DSH fixture model/protocol changed')
                names={v['function']['name'] for v in request.get('tools',[])}
                require({'mcp__operator__operator_run','mcp__operator__operator_status'}<=names,
                        'installed DSH did not expose MCP generation/status tools')
                tool_messages=[m for m in request['messages'] if m.get('role')=='tool']
                index=len(requests)
                if index>1 and tool_messages and _terminal(tool_messages[-1].get('content')):
                    delta=dict(content='Observed the installed operator task terminal status.');finish='stop'
                else:
                    require(index<=60,'DSH fixture did not observe terminal status')
                    if index>1:time.sleep(.25)
                    args=dict(run_dir=str(run_dir))
                    name='mcp__operator__operator_status'
                    if index==1:
                        name='mcp__operator__operator_run'
                        args.update(request='matmul fp32',shapes=[[8,8,8]],dsl='python',precision='fp32',
                            backend='fake',fake=dict(llm='fake'),max_rounds=1,wall_clock_s=35,max_llm_calls=8)
                    delta=dict(tool_calls=[dict(index=0,id='installed-%d'%index,type='function',
                        function=dict(name=name,arguments=json.dumps(args)))])
                    finish='tool_calls'
                chunks=[dict(id='installed-%d'%index,model=request['model'],
                    choices=[dict(index=0,delta=delta,finish_reason=None)]),
                    dict(id='installed-%d'%index,model=request['model'],
                    choices=[dict(index=0,delta={},finish_reason=finish)])]
                payload=(''.join('data: '+json.dumps(c)+'\n\n' for c in chunks)+'data: [DONE]\n\n').encode()
                self.send_response(200);self.send_header('Content-Type','text/event-stream')
                self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
            except Exception as exc:
                errors.append(str(exc));self.send_response(500);self.end_headers()
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.daemon_threads=True
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    disabled=('session-title-llm','llm-retry','otel','session-telemetry-otel','tool-working-directory',
        'tool-bash','tool-pwsh','tool-jobs','tool-fs','tool-fs-search','tool-skill','tool-subagent-control',
        'tool-subagent-list-agents','tool-subagent','tool-subagent-fork','tool-workflow','tool-todo',
        'tool-goal','web','web-search-deepseek','web-deepseek-api','web-fetch-http','tool-web')
    patch=[dict(id=name,disabled=True) for name in disabled]
    patch.extend([dict(id='agent-default-model',config=dict(provider='fixture',model='installed-delivery-fixture')),
        dict(id='llm-pi-ai',config=dict(providers=dict(fixture=dict(api='openai-completions',
            baseURL='http://127.0.0.1:%d/v1'%server.server_port,apiKeyEnv='DELIVERY_FIXTURE_KEY',
            models=[dict(id='installed-delivery-fixture',contextWindow=65536,maxTokens=4096)],
            compat=dict(maxTokensField='max_tokens'))))),
        dict(insert=[dict(id='mcp-operator',name=str(client),config=dict(serverName='operator',
            transport='stdio',command=str(python),args=['-I','-m','operator_agent.app.mcp_server'],
            toolCallTimeoutMs=60000,failOnStartupError=True)),
            dict(id='operator-tools',name=str(plugin/'index.mjs'),config=dict(command=str(python),
                args=['-I','-m','operator_agent']))])])
    patch_file=work/'patch.json';dump(patch_file,patch)
    dsh_env=dict(env,DSH_HOME=str(work/'profiles'),DELIVERY_FIXTURE_KEY='local-synthetic-fixture',
        NO_PROXY='127.0.0.1',no_proxy='127.0.0.1')
    try:
        completed=subprocess.run([node,'--import','data:text/javascript,console.debug%20%3D%20console.error%3B',
            str(entry),'--profile','headless','--patch',str(patch_file),'--json',
            'Use the operator MCP tools to develop the declared fixture, then poll the task until terminal.'],
            cwd=work,env=dsh_env,capture_output=True,text=True,timeout=75)
        (work/'stdout.jsonl').write_text(completed.stdout)
        (work/'stderr.log').write_text(completed.stderr)
        require(completed.returncode==0,'installed DSH headless failed; see '+str(work/'stderr.log'))
        require(not errors and len(requests)>=3,'DSH fixture failed: '+repr(errors))
        require(_terminal([m['content'] for m in requests[-1]['messages'] if m.get('role')=='tool']),
                'actual DSH did not return tool terminal state to the model')
        status=invoke(python,['-m','operator_agent','status','--run-dir',str(run_dir)],root,env)
        require(status['phase']=='done' and status['summary']['levels']==dict(L1=True,L2=True,L3=False),
                'installed DSH candidate did not pass full L2')
        test_only(status)
        selected=Path(status['summary']['path'])
        require((selected/'code.py').is_file() and (selected/'report.md').is_file(),
                'installed DSH did not retain source/report')
        result=dict(passed=True,test_only=True,gpu_acceptance=False,outer_transport='local_fixture',
            inner_model='scripted_fake',outer_calls=len(requests),status=status,
            dsh_entry=str(entry),dsh_entry_sha256=digest(entry),mcp_client_sha256=digest(client),
            packed_plugin_sha256=digest(plugin/'index.mjs'))
        dump(work/'result.json',result)
        return result
    finally:
        server.shutdown();server.server_close();thread.join(timeout=1)
        dump(work/'model-requests.json',requests)
        # If the outer host exits early, cancel only this fixture's owned task.
        if (run_dir/'launch.json').is_file():
            try:
                state=invoke(python,['-m','operator_agent','status','--run-dir',str(run_dir)],root,env)
                if state.get('phase') in ('running','cancelling'):
                    invocation=json.loads((run_dir/'launch.json').read_text())['invocation_id']
                    invoke(python,['-m','operator_agent','cancel','--run-dir',str(run_dir),
                        '--invocation-id',invocation],root,env)
                    deadline=time.monotonic()+10
                    while time.monotonic()<deadline:
                        state=invoke(python,['-m','operator_agent','status','--run-dir',str(run_dir)],root,env)
                        if state.get('phase') not in ('running','cancelling'):break
                        time.sleep(.05)
            except (OSError,ValueError,KeyError,subprocess.TimeoutExpired):pass


def run(python,plugin,output,*,dsh=False):
    # Resolve the parent directory but retain the venv executable symlink itself.
    python=Path(python).absolute();plugin=Path(plugin).resolve();root=Path(output).resolve()
    require(not root.exists(),'use a fresh output directory')
    require((plugin/'package.json').is_file() and (plugin/'index.mjs').is_file(),
            'plugin must be an extracted package directory')
    root.mkdir(parents=True,mode=0o700)
    env=environment(root,python)
    provenance=invoke(python,['-c',
        'import importlib.metadata,json,pathlib,sys,operator_agent;'
        'print(json.dumps(dict(python=sys.version,executable=sys.executable,'
        'module=str(pathlib.Path(operator_agent.__file__).resolve()),'
        'path=sys.path,package_version=importlib.metadata.version("operator-agent"))))'],root,env)
    require('site-packages' in Path(provenance['module']).parts,'wheel was not imported from site-packages')
    require(provenance['python'].startswith('3.12.'),'delivery default must be Python 3.12')
    dump(root/'provenance.json',provenance)
    cli=invoke(python,['-m','operator_agent','capabilities'],root,env)
    dump(root/'cli-capabilities.json',cli)
    mcp=MCP(python,root,env)
    owned=[]
    receipt=dict(schema='v07-installed-fake-smoke-v1',test_only=True,gpu_acceptance=False,
        paid_model_calls=0,ssh_executions=0,package_version=provenance['package_version'],
        plugin_version=json.loads((plugin/'package.json').read_text())['version'],passed=False)
    try:
        mcp.rpc('initialize',dict(protocolVersion='2025-11-25',capabilities={},
            clientInfo=dict(name='v07-installed-delivery',version='1')))
        tools=mcp.rpc('tools/list')['tools']
        required={'operator_capabilities','operator_prepare','operator_verify','operator_run',
                  'operator_status','operator_cancel'}
        require(required <= {t['name'] for t in tools},'installed MCP tools incomplete')
        cap=mcp.tool('operator_capabilities',{})
        dump(root/'mcp-capabilities.json',cap)
        declaration=dict(request='relu fp32',shapes=[[4,8]],dsl='python',precision='fp32')
        verify_dir=root/'verification'
        prepared=mcp.tool('operator_prepare',dict(declaration,run_dir=str(verify_dir)))
        code='import numpy as np\ndef kernel(X):\n    return np.maximum(X,0)\n'
        verified=mcp.tool('operator_verify',dict(run_dir=str(verify_dir),code=code,backend='fake'))
        test_only(verified)
        require(verified['levels']==dict(L1=True,L2=True,L3=False),'installed full verifier did not pass L2')
        run_dir=root/'repair-run'
        launched=mcp.tool('operator_run',dict(declaration,run_dir=str(run_dir),backend='fake',
            fake=dict(scenario='correctness-repair'),max_rounds=2,wall_clock_s=30,max_llm_calls=8))
        owned.append((run_dir,launched['invocation_id']))
        test_only(launched)
        status=wait(mcp,run_dir,lambda s:s.get('phase') in ('done','failed','lost','cancelled'))
        require(status['phase']=='done','installed generation/repair failed: '+repr(status))
        test_only(status)
        require(status['summary']['levels']==dict(L1=True,L2=True,L3=False),'repaired candidate incorrect')
        events=[json.loads(line) for line in (run_dir/'fake-events.jsonl').read_text().splitlines()]
        require(len({e['source_hash'] for e in events if e['operation']=='compile'})>=2,
            'repair did not produce and verify a new candidate')
        journals=[json.loads(p.read_text()) for p in (run_dir/'model-journal').glob('call-*.json')]
        require(any('Fixer' in str(c.get('caller')) for c in journals),'repair model call was not retained')
        require(all(c.get('dispatched_monotonic') is None for c in journals),
            'fixture unexpectedly dispatched a provider request')
        selected=Path(status['summary']['path'])
        require(root in selected.resolve().parents,'saved result escaped isolated workspace')
        export=root/'export';export.mkdir()
        entries=[]
        for name in ('code.py','report.md'):
            source=selected/name
            require(source.is_file() and not source.is_symlink(),'missing selected artifact '+name)
            shutil.copyfile(source,export/name)
            entries.append(dict(file=name,sha256=digest(export/name),source=str(source),source_sha256=digest(source)))
        require('FAKE / test_only' in (export/'report.md').read_text(),'report lacks fake boundary')
        independent=root/'independent'
        independent_prepared=mcp.tool('operator_prepare',dict(declaration,run_dir=str(independent)))
        final=mcp.tool('operator_verify',dict(run_dir=str(independent),backend='fake',
            code=(export/'code.py').read_text()))
        test_only(final)
        require(final['levels']==dict(L1=True,L2=True,L3=False),'independent installed final failed')
        require(final.get('source_hash')==digest(export/'code.py'),'independent final source binding changed')
        require(prepared['contract_hash']==independent_prepared['contract_hash']==final['contract_hash'],
            'independent contract binding changed')
        dump(export/'manifest.json',dict(schema='v07-delivery-copy-export-v1',test_only=True,
            product_export_command=False,files=entries,contract_hash=final['contract_hash'],
            independent_evidence=final['evidence']))
        blocked=root/'cancel-run'
        launched=mcp.tool('operator_run',dict(declaration,run_dir=str(blocked),backend='fake',
            fake=dict(scenario='timeout'),wall_clock_s=30,max_llm_calls=8))
        owned.append((blocked,launched['invocation_id']))
        def started(_):
            path=blocked/'fake-events.jsonl'
            return path.is_file() and any(json.loads(l)['operation']=='blocked' for l in path.read_text().splitlines())
        wait(mcp,blocked,started)
        cancelled=mcp.tool('operator_cancel',dict(run_dir=str(blocked),invocation_id=launched['invocation_id']))
        require(cancelled.get('accepted') is True,'cancel request refused')
        terminal=wait(mcp,blocked,lambda s:s.get('phase') in ('done','failed','lost','cancelled'))
        test_only(terminal)
        require(terminal['phase']=='cancelled' and terminal.get('cleanup',{}).get('worker_reaped') is True,
            'installed cancellation did not reap its worker')
        require(not (root/'kernel_db.jsonl').exists(),'control-plane production corpus unexpectedly written')
        require(not (run_dir/'kernel_db.jsonl').exists(),'fake positive candidate entered production corpus')
        receipt.update(tools=sorted(required),repair_status=status,cancel_status=terminal,
            independent_final=final,export=str(export/'manifest.json'),
            plugin_files={str(p.relative_to(plugin)):digest(p) for p in plugin.rglob('*') if p.is_file()})
        if dsh:receipt['dsh']=dsh_headless(python,plugin,root,env)
        receipt['passed']=True
    finally:
        for run_dir,invocation in owned:
            try:
                state=mcp.tool('operator_status',dict(run_dir=str(run_dir),invocation_id=invocation))
                if state.get('phase') in ('running','cancelling'):
                    mcp.tool('operator_cancel',dict(run_dir=str(run_dir),invocation_id=invocation))
                    wait(mcp,run_dir,lambda s:s.get('phase') not in ('running','cancelling'),timeout=10)
            except Exception:pass
        mcp.close()
        dump(root/'smoke-result.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python',required=True)
    parser.add_argument('--plugin',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--dsh',action='store_true',help='also drive installed DSH with local SSE fixture')
    args=parser.parse_args()
    result=run(args.python,args.plugin,args.output,dsh=args.dsh)
    print(json.dumps(dict(passed=result['passed'],evidence=str(Path(args.output).resolve()/'smoke-result.json'))))


if __name__=='__main__':main()
