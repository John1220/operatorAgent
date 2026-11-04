"""Per-cell MCP surface: real production tools with supervisor-pinned inputs."""
import copy
import json
from pathlib import Path
import secrets
import threading
import time
from contextlib import nullcontext

from benchmarks.experiments.s5.s5_plugin_rpc import _HTTP, _Handler, _error
from operator_agent.app import mcp_server, hitl_tools as ht
from operator_agent.execute.task_clock import monotonic
from .connection import options
from .plan import write_once

TOOLS=('operator_capabilities','operator_run','operator_status','operator_cancel')


class Relay:
    def __init__(self,config):
        self.config=config
        self.token=secrets.token_urlsafe(32)
        self.timeout=max(1,min(1810,int(config['budget']['wall_clock_s'])+5))
        self.lock=threading.Lock()
        self.records=[]
        self.invocation=None
        self.run_requested=False
        self.server=None

    def __enter__(self):
        self.server=_HTTP(('127.0.0.1',0),_Handler)
        self.server.owner=self
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        return self

    def client_config(self):
        return dict(url='http://127.0.0.1:%d/mcp'%self.server.server_address[1],
                    token=self.token,timeout=self.timeout)

    def _dispatch(self,request):
        if not isinstance(request,dict) or set(request)-{'jsonrpc','id','method','params'} or request.get('jsonrpc')!='2.0':
            raise ValueError('invalid JSON-RPC envelope')
        method=request.get('method')
        if not isinstance(request.get('params',{}),dict):
            raise ValueError('RPC params must be an object')
        if method=='notifications/initialized' and 'id' not in request:
            return mcp_server.handle_message(request)
        identity=request.get('id')
        if type(identity) not in (str,int) or len(str(identity))>128:
            raise ValueError('request id required')
        if method in ('initialize','ping'):
            return mcp_server.handle_message(request)
        if method=='tools/list':
            reply=mcp_server.handle_message(request)
            tools=[]
            for raw in reply['result']['tools']:
                if raw['name'] in TOOLS:
                    tool=copy.deepcopy(raw)
                    if tool['name']!='operator_capabilities':
                        tool['inputSchema']=dict(type='object',properties={},additionalProperties=False)
                        tool['description']+=' Frozen integration cell: takes no arguments; task, backend, budget and owned invocation are supplied by the supervisor.'
                    tools.append(tool)
            reply['result']['tools']=tools
            return reply
        params=request.get('params',{})
        if (method!='tools/call' or not isinstance(params,dict) or
                set(params)-{'name','arguments'} or params.get('name') not in TOOLS or params.get('arguments',{})!={}):
            raise ValueError('only listed tools with empty arguments are permitted')
        name=params['name']
        args={}
        if name=='operator_run':
            if self.run_requested:raise ValueError('one operator_run per cell; poll operator_status')
            self.run_requested=True
            remaining=self.config['deadline_monotonic']-monotonic()
            if remaining<=0:raise ValueError('cell deadline reached')
            args=dict(self.config['declaration'],run_dir=self.config['run_dir'],
                backend=self.config['backend'],max_rounds=self.config['max_rounds'],
                **options(self.config),**dict(self.config['budget'],wall_clock_s=remaining))
            if self.config['backend']=='fake':args['fake']=dict(llm='live')
        elif name in ('operator_status','operator_cancel'):
            if not self.invocation:raise ValueError('operator_run has not returned an owned invocation')
            args=dict(run_dir=self.config['run_dir'],invocation_id=self.invocation)
        pinned=dict(request,params=dict(name=name,arguments=args))
        from .sampling import autonomous_launcher
        with autonomous_launcher(self.config) if name=='operator_run' else nullcontext():
            reply=mcp_server.handle_message(pinned)
        if name=='operator_run' and 'result' in reply:
            result=reply['result']
            try:
                body=json.loads(result['content'][0]['text'])
                self.invocation=body.get('invocation_id')
            except (KeyError,IndexError,ValueError,TypeError):pass
        return reply

    def request(self,request):
        with self.lock:
            try:
                reply=self._dispatch(request)
            except (ValueError,TypeError,KeyError) as exc:
                reply=_error(request.get('id') if isinstance(request,dict) else None,-32602,str(exc))
            row=dict(index=len(self.records),timestamp=time.time(),request=request,reply=reply)
            self.records.append(row)
            write_once(Path(self.config['run_dir']).parent/'mcp'/('rpc-%04d.json'%row['index']),row)
            return reply

    def __exit__(self,*exc):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=2)
        if self.invocation:
            status=ht.autonomous_status(self.config['run_dir'],self.invocation)
            if status.get('phase') in ('running','cancelling'):
                ht.cancel_task(self.config['run_dir'],self.invocation)
                # Parent process-group supervision is the final cleanup owner.
