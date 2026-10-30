"""Experimental relay to the unchanged direct-mode MCP implementation.

Only the outer DSH process sees the HTTP proxy. Product code and SSH credentials
stay in the supervisor-owned MCP process. This is a cooperative capability, not
hostile-code or network isolation.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

from benchmarks.experiments.s5 import s5_plugin_rpc as rpc
from benchmarks.experiments.s5 import s5_runtime, s5_shared_gpu

ALLOWED = frozenset(('operator_capabilities', 'operator_prepare', 'operator_verify', 'operator_bench'))


class PerformanceRelay(rpc.MCPRelay):
    def __init__(self, task_dir, declaration, output, *, pin, resource_policy,
                 timeout=660, max_candidates=3, max_verify_calls=3, max_bench_calls=3,
                 max_tool_gpu_calls=6, allow_prepare=True):
        contract = json.loads((Path(task_dir)/'contract.json').read_text())['hash']
        super().__init__(dict(url='http://127.0.0.1:1/development/verify', token='unused',
                             contract=contract, backend='ssh', timeout=timeout),
                         task_dir, declaration, output, timeout=timeout)
        self.pin = dict(pin)
        self.resource_policy = dict(resource_policy)
        s5_shared_gpu.validate_policy(self.resource_policy)
        self.max_candidates = max_candidates
        self.max_verify_calls = max_verify_calls
        self.max_bench_calls = max_bench_calls
        self.max_tool_gpu_calls = max_tool_gpu_calls
        self.allowed_tools = set(ALLOWED)
        if not allow_prepare:
            self.allowed_tools.discard('operator_prepare')
        self.candidates = {}
        self.reservation_lock = threading.Lock()
        self.gpu_observations = []
        self.resource_error = None
        self.verify_calls = 0
        self.bench_calls = 0
        self.prepare_calls = 0

    def _gpu_environment(self):
        env = {key: os.environ[key] for key in ('HOME', 'SSH_AUTH_SOCK', 'TMPDIR', 'LANG')
               if key in os.environ}
        env.update(PATH='/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin',
                   PYTHONPATH=s5_runtime.python_path(), PYTHONNOUSERSITE='1',
                   OPERATOR_AGENT_REQUIRED_GPU=str(self.pin['gpu_index']),
                   OPERATOR_AGENT_REQUIRED_GPU_UUID=self.pin['gpu_uuid'],
                   OPERATOR_AGENT_REQUIRED_GPU_NAME=self.pin['gpu_name'])
        return env

    def _validate(self, message):
        # Retain the existing envelope/initialization/capabilities validation.
        if message.get('method') != 'tools/call' or message.get('params',{}).get('name') == 'operator_capabilities':
            return super()._validate(message)
        if (message.get('jsonrpc') != '2.0' or message.get('id') is None
                or set(message) - {'jsonrpc','id','method','params'}):
            raise ValueError('invalid RPC envelope')
        params=message.get('params',{});name=params.get('name');args=dict(params.get('arguments',{}))
        if name not in self.allowed_tools:
            raise ValueError('inner generation/repair is outside this performance experiment')
        path=Path(args.get('run_dir',''))
        if not path.is_absolute():path=self.workspace/path
        if path.resolve()!=self.task_dir or path.is_symlink():raise ValueError('fixed task required')
        if name=='operator_prepare':
            if {k:v for k,v in args.items() if k!='run_dir'}!=self.declaration:raise ValueError('fixed declaration required')
            if self.prepare_calls >= 1:raise ValueError('prepare call limit')
            self.prepare_calls += 1
            return dict(message,params=dict(params,arguments=dict(args,run_dir=str(self.task_dir))))
        allowed={'run_dir','backend','host','timeout','persistent','candidate_origin','stop_after','code','code_file'} if name=='operator_verify' else {'run_dir','backend','host','timeout','persistent','candidates','max_candidates','deadline_s'}
        if set(args)-allowed:raise ValueError('connection/measurement override refused')
        if args.get('backend','ssh')!='ssh' or args.get('host',self.pin['host'])!=self.pin['host']:
            raise ValueError('fixed backend required')
        if 'timeout' in args and (type(args['timeout']) is not int or not 1<=args['timeout']<=600):raise ValueError('timeout refused')
        args.update(run_dir=str(self.task_dir),backend='ssh',host=self.pin['host'])
        args.setdefault('timeout',600)
        sources=[]
        if name=='operator_verify':
            if args.get('stop_after','L2') != 'L2':raise ValueError('verification is limited to L2')
            if self.verify_calls >= self.max_verify_calls:raise ValueError('verification call limit')
            if ('code'in args)==('code_file'in args):raise ValueError('single code source required')
            if 'code_file'in args:args['code']=self._candidate_file(args.pop('code_file'))
            sources=[args['code']]
        else:
            if self.bench_calls >= self.max_bench_calls:raise ValueError('benchmark call limit')
            candidates=args.get('candidates')
            if not isinstance(candidates,list) or not 1<=len(candidates)<=self.max_candidates:raise ValueError('candidate limit')
            normalized=[]
            for item in candidates:
                if not isinstance(item,dict) or set(item)-{'name','code','code_file'} or ('code'in item)==('code_file'in item):raise ValueError('candidate object invalid')
                code=self._candidate_file(item['code_file']) if 'code_file'in item else item['code']
                sources.append(code);normalized.append(dict(name=item.get('name','candidate'),code=code))
            args['candidates']=normalized;args['max_candidates']=self.max_candidates
            args.setdefault('deadline_s',600.)
            if type(args['deadline_s']) not in (int,float) or not 0<args['deadline_s']<=600:raise ValueError('bench deadline refused')
        for code in sources:
            if not isinstance(code,str) or not code.strip() or len(code.encode())>65536:raise ValueError('candidate invalid')
            if 'triton.autotune' in code or '@autotune' in code:raise ValueError('each submission must contain one explicit configuration')
        # Distinct source reservations never refund; repeated measurement is visible
        # in records but does not introduce a fourth candidate.
        with self.reservation_lock:
            if self.verify_calls + self.bench_calls >= self.max_tool_gpu_calls:
                raise ValueError('GPU tool call limit')
            pending={hashlib.sha256(code.encode()).hexdigest():code for code in sources}
            if len(set(self.candidates)|set(pending))>self.max_candidates:raise ValueError('three candidate limit')
            for checksum,code in pending.items():
                if checksum not in self.candidates:
                    target=self.output/('candidate-'+checksum+'.py')
                    target.write_text(code);self.candidates[checksum]=str(target)
            if name == 'operator_verify':
                self.verify_calls += 1
            else:
                self.bench_calls += 1
        return dict(message,params=dict(params,arguments=args))

    def request(self,message):
        name = (message.get('params') or {}).get('name') if isinstance(message, dict) else None
        is_gpu_call = message.get('method') == 'tools/call' and name in ('operator_verify', 'operator_bench')
        observation = None
        if is_gpu_call:
            if self.resource_error:
                return rpc._error(message.get('id'), -32001, 'Frozen GPU resource policy failed')
            try:
                observation = s5_shared_gpu.snapshot(self.pin, self.resource_policy,
                                                     self._gpu_environment())
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                self.resource_error = type(exc).__name__
                return rpc._error(message.get('id'), -32001, 'Frozen GPU resource policy failed')
        reply=super().request(message)
        if is_gpu_call:
            try:
                after = s5_shared_gpu.snapshot(self.pin, self.resource_policy,
                                               self._gpu_environment())
                before_pids = [row['pid'] for row in observation['compute_apps']]
                after_pids = [row['pid'] for row in after['compute_apps']]
                if before_pids != after_pids:
                    raise ValueError('resident GPU process set changed')
                self.gpu_observations.append(dict(tool=name, before=observation, after=after,
                                                  process_set_stable=True))
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                self.resource_error = type(exc).__name__
                self.gpu_observations.append(dict(tool=name, before=observation,
                                                  after=None, process_set_stable=False,
                                                  error=type(exc).__name__))
                return rpc._error(message.get('id'), -32001, 'Frozen GPU resource policy failed')
        if message.get('method')=='tools/list' and reply and isinstance(reply.get('result'),dict):
            result=dict(reply['result']);result['tools']=[t for t in result.get('tools',[]) if t.get('name')in self.allowed_tools]
            reply=dict(reply,result=result)
        return reply

    def snapshot(self):
        value = super().snapshot()
        value.update(schema='s5-performance-mcp-relay-v1',
                     candidate_hashes=sorted(self.candidates),
                     max_candidates=self.max_candidates,
                     verify_calls=self.verify_calls, max_verify_calls=self.max_verify_calls,
                     bench_calls=self.bench_calls, max_bench_calls=self.max_bench_calls,
                     max_tool_gpu_calls=self.max_tool_gpu_calls,
                     prepare_calls=self.prepare_calls,
                     resource_error=self.resource_error,
                     gpu_observations=list(self.gpu_observations))
        return value

    def __enter__(self):
        env={k:os.environ[k] for k in ('PATH','HOME','SSH_AUTH_SOCK') if k in os.environ}
        env.update(PYTHONPATH=s5_runtime.python_path(),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
                   OPERATOR_AGENT_REQUIRED_GPU=str(self.pin['gpu_index']),
                   OPERATOR_AGENT_REQUIRED_GPU_UUID=self.pin['gpu_uuid'],
                   OPERATOR_AGENT_REQUIRED_GPU_NAME=self.pin['gpu_name'],
                   OPERATOR_AGENT_EXISTING_REMOTE_ONLY='1',OPERATOR_AGENT_S4_GRACEFUL_CLEANUP='1',
                   OPERATOR_AGENT_KERNEL_DB='off')
        self.stderr=(self.output/'mcp-stderr.log').open('xb')
        try:
            self.proc=subprocess.Popen([sys.executable,'-m','operator_agent.mcp_server'],cwd=str(rpc.ROOT),env=env,
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,start_new_session=True,bufsize=0)
            self.cleanup='running';self.reader=threading.Thread(target=self._read,daemon=True);self.reader.start()
            self.server=rpc._HTTP(('127.0.0.1',0),rpc._Handler);self.server.owner=self
            self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
            return self
        except BaseException:
            self.__exit__(None,None,None);raise
