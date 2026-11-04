"""Installed DSH + source plugin adapter; outer and inner share one ledger."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from contextlib import contextmanager

from benchmarks.experiments.s5 import s5_arm_env, s5_plugin_env
from benchmarks.experiments.s5.s5_gateway_http import GatewayServer
from benchmarks.experiments.s5.s5_gateway_transport import UpstreamTransport
from operator_agent.app import hitl_tools as ht
from operator_agent.execute.task_clock import monotonic
from .connection import credential
from .delegate import source_ref
from .plan import read, write_once
from .relay import Relay
from .sampling import DEFAULT_OUTPUT_CAP, output_cap, request_timeout as upstream_deadline_s


def runtime_identity():
    paths=[s5_arm_env.DSH_ROOT/'dsh/lib/bin.js',s5_arm_env.MCP_CLIENT,
           s5_arm_env.DSH_ROOT/'dsh/package.json',s5_arm_env.DSH_ROOT/'dsh-mcp-client/package.json']
    for package in ('dsh','dsh-mcp-client'):
        paths.extend(p for p in (s5_arm_env.DSH_ROOT/package/'lib').rglob('*') if p.is_file())
    if s5_arm_env.NODE:paths.append(Path(s5_arm_env.NODE))
    value={str(p):hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None for p in paths}
    if not s5_arm_env.NODE:value['missing_node']=None
    return value


def preflight():
    from benchmarks.experiments.s5 import s5_seatbelt
    if (sys.platform!='darwin' or not Path(s5_seatbelt.SANDBOX).is_file()
            or not s5_arm_env.NODE or any(v is None for v in runtime_identity().values())):
        raise ValueError('installed DSH/node and macOS Seatbelt required; no core fallback')


def entry_observed(records):
    """Require actual discovery, one accepted run and an Agent terminal poll."""
    handshake=discovery=launched=polled=False
    for row in records:
        request=row.get('request') or {};reply=row.get('reply') or {}
        if not isinstance(request,dict) or not isinstance(reply,dict):continue
        result=reply.get('result')
        if request.get('method')=='initialize' and isinstance(result,dict):handshake=True
        if request.get('method')=='tools/list' and isinstance(result,dict):
            discovery=any(t.get('name')=='operator_run' for t in result.get('tools',[]) if isinstance(t,dict))
        params=request.get('params') or {}
        if not isinstance(params,dict) or not isinstance(result,dict) or result.get('isError'):continue
        try:body=json.loads(result['content'][0]['text'])
        except (KeyError,IndexError,ValueError,TypeError):continue
        if not isinstance(body,dict):continue
        if params.get('name')=='operator_run' and body.get('ok') is True and body.get('invocation_id'):launched=True
        if params.get('name')=='operator_status' and body.get('phase') in ('done','failed','cancelled'):polled=True
    return bool(handshake and discovery and launched and polled)


def dsh_model_options(config):
    """Both DSH arms advertise the same limit before the gateway sees a request."""
    options = dict(max_tokens=output_cap(config), context_window=65536)
    if 'sampling' in config:
        options['stream_idle_timeout_ms'] = max(1, int(upstream_deadline_s(config) * 1000))
    return options


def stream_limits(config):
    """(max_response_bytes, max_chunks)：按输出上限给流设界。

    SSE 每个片段都带一遍封套，所以一个 32K 输出的响应在**远没到 token 上限**时就会
    撞到网关的 1 MiB / 4096 块老界 —— 表现是 `UpstreamError(stage=response_body,
    code=response_limit)`，到 DSH 那边只剩 `Stream ended without finish_reason`，
    看起来像网络或模型故障。2026-11-04 实测：同一请求重复 6 次，**5 次在
    1,048,xxx 字节 / 约 3734 块处被掐**，而直连上游同一条 prompt 能完整跑完。

    老界保留在 4K 输出档（那是它原本的用途），大输出档放到 32 MiB / 65536 块。
    """
    large = output_cap(config) > DEFAULT_OUTPUT_CAP
    return ((32 if large else 1) * 1024 * 1024, 65536 if large else 4096)


@contextmanager
def model_gateway(config,cell):
    model=config['model']
    max_response_bytes, max_chunks = stream_limits(config)
    transport=UpstreamTransport(model['base_url']+'/chat/completions',credential(model),
        timeout_seconds=upstream_deadline_s(config),max_response_bytes=max_response_bytes)
    counter=[0]
    def dispatch(request):
        index=counter[0]
        write_once(cell/'outer-requests'/('request-%04d.json'%index),request)
        counter[0]+=1
        return _observed_stream(transport,request,cell,index)
    with GatewayServer(config['shared_ledger'],task_id=config['shared_task_id'],model=model['id'],
            participant='outer',adapter=dispatch,max_calls_limit=config['budget']['max_llm_calls'],
            max_response_bytes=max_response_bytes,max_chunks=max_chunks) as gateway:
        from .sampling import attach_gateway
        attach_gateway(gateway,config,config['run_dir'])
        yield gateway


def _observed_stream(transport,request,cell,index):
    """把上游那次流包一层观测：只记事实（耗时、块数、字节、失败码），不记内容。

    为什么值得单独一层：2026-11-04 A 臂反复死在
    `dsh: TRANSPORT: Stream ended without finish_reason`，而网关会把上游异常统一
    改写成 `PolicyUpstreamFailure`（不带上游诊断）—— 于是「是不是超时、是不是网络
    断了、断在第几块」这三点全都查不到，只能靠重跑猜。这里把它们落成一行 JSON，
    挂在格目录里，失败格和成功格都写（成功那行是失败行的对照）。
    """
    started=time.monotonic()
    chunks=0
    size=0
    first_after=None
    saw_finish=False
    try:
        for chunk in transport.stream(request):
            if first_after is None: first_after=time.monotonic()-started
            chunks+=1
            if isinstance(chunk,(bytes,bytearray)):
                size+=len(chunk)
                if not saw_finish and b'finish_reason' in chunk: saw_finish=True
            yield chunk
    except Exception as exc:                      # noqa: BLE001 —— 只观测，不改变传播
        _append_stream_record(cell,dict(index=index,outcome='failed',
            elapsed_s=round(time.monotonic()-started,3),first_chunk_s=None if first_after is None else round(first_after,3),
            chunks=chunks,bytes=size,failure=getattr(exc,'diagnostic',None),
            exception=type(exc).__name__))
        raise
    else:
        _append_stream_record(cell,dict(index=index,outcome='completed',
            elapsed_s=round(time.monotonic()-started,3),first_chunk_s=None if first_after is None else round(first_after,3),
            chunks=chunks,bytes=size,failure=None,exception=None,finish_seen=saw_finish))


def _append_stream_record(cell,row):
    """追加一行流观测。写不进不挡测量（观测面不是入场券）。"""
    try:
        path=Path(cell)/'transport-failures.jsonl'
        line=json.dumps(dict(schema='phase1-transport-stream-v1',at=time.time(),**row),
                        ensure_ascii=False,sort_keys=True)+'\n'
        fd=os.open(str(path),os.O_CREAT|os.O_WRONLY|os.O_APPEND,0o600)
        try: os.write(fd,line.encode())
        finally: os.close(fd)
    except OSError:
        pass


def run(config):
    cell=Path(config['run_dir']).parent
    model=config['model']
    prepared=ht.prepare(config['declaration'],config['run_dir'])
    result=dict(cell_id=config['cell_id'],plan_hash=config['plan_hash'],arm='B',phase='failed',
        reason='host_incomplete',selected_source=None,source_hash=None,levels={},
        contract_hash=prepared['contract_hash'],execution_context=config['execution_context'],
        dsh_entry_verified=False)
    runtime=s5_plugin_env.stage_runtime(cell/'plugin-runtime')
    write_once(cell/'host-runtime.json',dict(installed=runtime_identity(),plugin=runtime,
        installed_plugin_acceptance=False))
    relay=Relay(config)
    try:
        with relay, model_gateway(config,cell) as gateway:
            built=s5_plugin_env.build('C',runtime=runtime,relay=relay,base_url=gateway.base_url,
                arm_dir=cell/'dsh',model=model['id'],
                plugin_tool_timeout_ms=relay.timeout*1000, **dsh_model_options(config))
            # The child gets only the gateway placeholder key and per-cell relay capability.
            write_once(cell/'host-surface.json',dict(surface=built['surface'],
                patch_sha256=built['patch_sha256'],env_keys=sorted(built['env']),
                provider_credential_in_child=False,source_plugin=True))
            prompt=('Develop the frozen operator using mcp__operator__operator_run with empty arguments. '
                'The supervisor has pinned all task and execution arguments. Call operator_run exactly once, '
                'then poll mcp__operator__operator_status with empty arguments until a terminal phase. '
                'Do not generate source through shell or edit files. Report the actual terminal outcome. '
                'Frozen declaration: '+json.dumps(config['declaration'],ensure_ascii=False))
            response=s5_arm_env.start_headless(built,prompt,
                timeout=max(.01,config['deadline_monotonic']-monotonic()),capture_dir=cell/'host-output')
            write_once(cell/'host-response.json',response)
            status=ht.autonomous_status(config['run_dir'],relay.invocation) if relay.invocation else {}
            write_once(cell/'host-terminal-status.json',status)
            summary=read(Path(config['run_dir'])/'conductor.json') if (Path(config['run_dir'])/'conductor.json').is_file() else {}
            source=Path(summary['path'])/'code.py' if summary.get('path') else None
            completed=status.get('phase') in ('done','failed','cancelled')
            result.update(phase=status.get('phase','failed') if completed else 'failed',
                reason=status.get('stopped_reason') or ('host_returned' if completed else 'host_left_task_incomplete'),
                selected_source=str(source) if source and source.is_file() and completed else None,
                source_hash=source_ref(source)['sha256'] if source and source.is_file() and completed else None,
                levels=summary.get('levels',{}),dsh_entry_verified=bool(
                    response['returncode']==0 and entry_observed(relay.records) and relay.invocation and completed),
                entry_evidence=[source_ref(cell/'mcp'/('rpc-%04d.json'%row['index'])) for row in relay.records],
                host_response=source_ref(cell/'host-response.json'))
    except Exception as exc:
        result.update(phase='failed',reason=type(exc).__name__)
        raise
    finally:
        events=[]
        for row in relay.records:
            params=row['request'].get('params',{}) if isinstance(row['request'],dict) else {}
            name=params.get('name') if isinstance(params,dict) else None
            if not name:continue
            ref=source_ref(cell/'mcp'/('rpc-%04d.json'%row['index']))
            events.append(dict(id='rpc-%d'%row['index'],timestamp=row['timestamp'],actor='agent',
                stage='delegate',kind='tool_call',action='candidate_verify' if name=='operator_run' else name,
                tool=name,source=ref))
        write_once(Path(config['run_dir'])/'events.json',events)
        write_once(Path(config['run_dir'])/'connected-result.json',result)
    return result
