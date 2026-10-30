"""Actual DSH A/C with fake replies, real MCP and mock L2 outside the jail.

No real provider, credential or GPU is used. This proves the tested file paths,
public-client/MCP feedback and common quota, not complete network/IPC isolation.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile

from benchmarks import s5_arm_env, s5_plugin_env as surface, s5_seatbelt, s5_verify_remote
from benchmarks.s5_development_evidence import _feedback
from benchmarks.s5_development_service import DevelopmentService, encoded
from benchmarks.s5_gateway_http import GatewayServer
from benchmarks.s5_plugin_rpc import MCPRelay
from benchmarks.s5_plugin_flow import progress as workflow_progress
from benchmarks.s5_vertical_tool_provider import _sse
from benchmarks.s5_vertical_validation import GOOD_FIXTURE, BAD_FIXTURE
from operator_agent import s5_shared_budget as shared
from operator_agent.hitl_tools import prepare
from operator_agent.run_contract import digest

DECLARATION=dict(request='layernorm fp32',shapes=[[7,33]],hardware='mock_gpu',
                 dsl='python',precision='fp32')
FORBIDDEN_TEXT='PRIVATE_PLUGIN_PROBE_CANARY_MUST_NOT_ESCAPE'
CALLS=10
SCHEMA='s5-plugin-only-readiness-v1'


def contains_message_text(value, expected):
    """Inspect message text without JSON escaping quotes in instructions."""
    if isinstance(value, str):return expected in value
    if isinstance(value, list):return any(contains_message_text(item, expected) for item in value)
    if isinstance(value, dict):return any(contains_message_text(item, expected) for item in value.values())
    return False


def write(path,value):
    with Path(path).open('x',encoding='utf-8')as f:
        json.dump(value,f,indent=2,sort_keys=True,ensure_ascii=False);f.write('\n')


def matrix_expected(arm):
    return dict(probe='s5-plugin-read-matrix',other=False,credentials=False,ledger=False,
        source_repository=False,core=False,plugin=arm=='C',public_test=True,
        core_importable=False,core_module_callable=False,cli_on_path=False,
        child_read=False,hardlink_read=False)


def find_matrix(text):
    for i,c in enumerate(text):
        if c!='{':continue
        try:obj,_=json.JSONDecoder().raw_decode(text[i:])
        except ValueError:continue
        if isinstance(obj,dict)and obj.get('probe')=='s5-plugin-read-matrix':return obj
    raise ValueError('read matrix missing')


def audit(events,snapshot,*,arm):
    """Bind real MCP/public-Python replies to retained service projections."""
    calls={};matches={};problems=[]
    rows={r['sequence']:r for r in snapshot['records']if r['phase']=='development'}
    for index,event in enumerate(events):
        if event.get('type')=='tool_call':
            identity=event.get('callId')
            if identity in calls:problems.append('duplicate tool call id')
            calls[identity]=(index,event)
        if event.get('type')!='tool_result':continue
        body=_feedback(event.get('result'))
        if body is None:continue
        seq=body.get('sequence');row=rows.get(seq);call=calls.get(event.get('callId'))
        if row is None or call is None or seq in matches or event.get('status')!='completed':
            problems.append('unbound or duplicate development feedback');continue
        if any(body.get(k)!=v for k,v in row['result'].items()):
            problems.append('service feedback differs from DSH tool result');continue
        tool=call[1].get('tool')
        command=(call[1].get('input')or{}).get('command','')
        if tool=='mcp__operator__operator_verify'and arm=='C':ingress='operator_mcp'
        elif tool=='bash'and 'check.py'in command:ingress='public_python'
        else:problems.append('unexpected development tool');continue
        matches[seq]=dict(sequence=seq,tool_call_id=event.get('callId'),ingress=ingress,
            source_sha256=row['source_sha256'],service_response_sha256=row['response_sha256'],
            tool_result_sha256=hashlib.sha256(event['result'].encode()).hexdigest(),
            call_event_index=call[0],result_event_index=index)
    missing=sorted(set(rows)-set(matches))
    if missing:problems.append('development response missing from DSH transcript')
    ordered=[matches[k]for k in sorted(matches)]
    if any(a['result_event_index']>=b['call_event_index']for a,b in zip(ordered,ordered[1:])):
        problems.append('feedback/revision order is not serial')
    return dict(schema='s5-plugin-development-transcript-audit-v1',complete=not problems,
        problems=problems,receipts=ordered,missing_sequences=missing,
        plugin_verification_observed=any(r['ingress']=='operator_mcp'for r in ordered),
        feedback_causality_proven=False,isolation_verified=False)


class BoundaryProvider:
    def __init__(self,arm,built,private,service,mcp_schema=None,mcp_instructions=None):
        self.arm,self.built,self.private,self.service=arm,built,private,service
        self.requests=[];self.mcp_schema=mcp_schema;self.mcp_instructions=mcp_instructions
        self.schema_discovered=False;self.instructions_discovered=False

    def command(self):
        root=Path(self.built['isolation']['runtime']['root']);work=Path(self.built['work'])
        paths=dict(other=self.private/'other-arm/candidate.py',
            credentials=self.private/'host-home/credentials',ledger=self.private/'supervisor-ledger',
            source_repository=surface.ROOT/'benchmarks/s5_formal_run.py',
            core=surface.ROOT/'operator_agent/__init__.py',plugin=root/'plugin/index.mjs',
            public_test=work/'check.py')
        source=('import json,pathlib,subprocess,sys,shutil,os\n'
            'result={"probe":"s5-plugin-read-matrix"}\n'
            'for label,path in '+repr({k:str(v)for k,v in paths.items()})+'.items():\n'
            ' try:pathlib.Path(path).read_bytes();result[label]=True\n'
            ' except PermissionError:result[label]=False\n'
            'try:import operator_agent;result["core_importable"]=True\n'
            'except (ImportError,PermissionError):result["core_importable"]=False\n'
            'result["cli_on_path"]=shutil.which("operator-agent") is not None\n'
            'r=subprocess.run([sys.executable,"-m","operator_agent","--help"],capture_output=True)\n'
            'result["core_module_callable"]=r.returncode==0\n'
            'r=subprocess.run(["/bin/cat",'+repr(str(paths['other']))+'],capture_output=True)\n'
            'result["child_read"]=r.returncode==0\n'
            'try:\n'
            ' os.link('+repr(str(paths['other']))+','+repr(str(work/'hardlink'))+')\n'
            ' pathlib.Path('+repr(str(work/'hardlink'))+').read_bytes();result["hardlink_read"]=True\n'
            'except (PermissionError,FileNotFoundError):result["hardlink_read"]=False\n'
            'print(json.dumps(result))\n')
        return shlex.quote(sys.executable)+' -c '+shlex.quote(source)

    def public_call(self):
        return 'bash',dict(description='Use common Python verification client',
            command=shlex.quote(sys.executable)+' check.py --code-file candidate.py')

    def verify_call(self,source):
        if self.arm=='A':return self.public_call()
        return 'mcp__operator__operator_verify',dict(run_dir=str(Path(self.built['work'])/'task'),
            code=source,stop_after='L2',backend='mock')

    def __call__(self,request):
        number=len(self.requests)+1
        record=dict(sequence=number,request_sha256=digest(request));self.requests.append(record)
        if number==1:
            tools={t.get('function',{}).get('name'):t.get('function',{})for t in request.get('tools',[])}
            self.schema_discovered='mcp__operator__operator_verify'in tools
            self.instructions_discovered=bool(self.mcp_instructions and
                contains_message_text(request.get('messages'), self.mcp_instructions))
            if self.schema_discovered!=(self.arm=='C'):raise ValueError('unexpected MCP tool surface')
            if self.arm=='C':
                actual=tools['mcp__operator__operator_verify'].get('parameters')
                if actual!=self.mcp_schema:raise ValueError('MCP schema changed at DSH discovery')
                if not self.instructions_discovered:raise ValueError('real MCP instructions missing')
        if number>1:
            latest=[m for m in request.get('messages',[])if m.get('role')=='tool'][-1]
            text=latest.get('content','');record['previous_tool_result']=text
            if latest.get('tool_call_id')!='plugin_boundary_'+str(number-1):raise ValueError('tool ID drift')
            if FORBIDDEN_TEXT in text:raise ValueError('private canary escaped')
            if number==2 and 'OWN_CANARY'not in text:raise ValueError('own read failed')
            if number in(3,4)and not any(t in text.lower()for t in('error','denied','permitted','eacces')):
                raise ValueError('denied read not observed')
            if number==5:
                matrix=find_matrix(text)
                if matrix!=matrix_expected(self.arm):raise ValueError('read matrix mismatch')
                record['matrix']=matrix
            if number in(7,9):
                actual=self.service.snapshot()['records'][0 if number==7 else 1]['result']
                feedback=_feedback(text)
                if not isinstance(feedback,dict)or any(feedback.get(k)!=v for k,v in actual.items()):
                    raise ValueError('service feedback changed')
                if feedback['status']!=('failed'if number==7 else'passed'):raise ValueError('unexpected mock verdict')
                record['service_feedback_sha256']=hashlib.sha256(encoded(actual)).hexdigest()
            if number==10:
                if 'development_budget_exhausted'not in text:raise ValueError('common quota bypassed')
                if self.service.snapshot()['development_used']!=2:raise ValueError('extra development reserved')
                record['extra_public_submission_refused']=True
        work=Path(self.built['work'])
        if number==1:tool,args='read',dict(file_path=str(work/'own.txt'))
        elif number==2:tool,args='read',dict(file_path=str(self.private/'other-arm/candidate.py'))
        elif number==3:tool,args='read',dict(file_path=str(work/'alias'))
        elif number==4:tool,args='bash',dict(description='Probe actual inherited black-box boundary',command=self.command())
        elif number in(5,7):tool,args='write',dict(file_path=str(work/'candidate.py'),content=BAD_FIXTURE if number==5 else GOOD_FIXTURE)
        elif number in(6,8):tool,args=self.verify_call(BAD_FIXTURE if number==6 else GOOD_FIXTURE)
        elif number==9:tool,args=self.public_call()
        elif number==10:return _sse(dict(content=GOOD_FIXTURE),'stop')
        else:raise ValueError('unexpected fake model request')
        call=dict(index=0,id='plugin_boundary_'+str(number),type='function',
                  function=dict(name=tool,arguments=json.dumps(args)))
        return _sse(dict(tool_calls=[call]),'tool_calls')


def implementation():
    paths=('benchmarks/s5_plugin_probe.py','benchmarks/s5_plugin_env.py',
        'benchmarks/s5_arm_env.py','benchmarks/s5_seatbelt.py','benchmarks/s5_plugin_rpc.py',
        'benchmarks/s5_plugin_proxy.py','benchmarks/s5_development_service.py',
        'benchmarks/s5_verify_remote.py','operator_agent/s5_verify_client.py','operator_agent/mcp_server.py')
    return {p:s5_seatbelt.checksum(surface.ROOT/p)for p in paths}


def run(output):
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False,mode=0o700)
    runtime=surface.stage_runtime(root/'runtime')
    private=root/'supervisor-private';private.mkdir(mode=0o700)
    for name in('other-arm/candidate.py','host-home/credentials','supervisor-ledger'):
        p=private/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(FORBIDDEN_TEXT)
    # Positive control: the forbidden targets really exist and links work on
    # this filesystem before testing denial inside the participant jail.
    control=private/'control-hardlink';os.link(private/'other-arm/candidate.py',control)
    assert control.read_text()==FORBIDDEN_TEXT;control.unlink()
    for p in(surface.ROOT/'operator_agent/__init__.py',surface.ROOT/'benchmarks/s5_formal_run.py'):
        assert p.is_file()
    service_sha=s5_verify_remote.freeze(root/'service-plan',declaration=DECLARATION,backend='mock')
    participants=Path(tempfile.mkdtemp(prefix='operator-s5-plugin-')).resolve()
    write(root/'participant-location.json',dict(path=str(participants),temporary=True))
    records=[]
    for arm in('A','C'):
        out=root/'arms'/arm;out.mkdir(parents=True,mode=0o700)
        with ExitStack()as stack:
            svc=stack.enter_context(DevelopmentService(root/'service-plan/plan.json',service_sha,
                out/'verification',arm=arm,wall_seconds=240))
            work=participants/arm/'work';work.mkdir(parents=True,mode=0o700)
            prepare(DECLARATION,work/'task')
            development=dict(url=svc.url,token=svc.token,contract=svc.plan['contract_hash'],backend='mock',timeout=70)
            relay=None;schema=instructions=None
            if arm=='C':
                relay=stack.enter_context(MCPRelay(development,work/'task',DECLARATION,out/'mcp-relay',timeout=100))
                init=relay.request(dict(jsonrpc='2.0',id='supervisor-init',method='initialize',params=dict(
                    protocolVersion='2025-11-25',capabilities={},clientInfo=dict(name='s5-probe',version='1'))))
                listing=relay.request(dict(jsonrpc='2.0',id='supervisor-tools',method='tools/list',params={}))
                write(out/'real-mcp-initialize.json',init);write(out/'real-mcp-tools.json',listing)
                instructions=init['result']['instructions']
                schema=next(t['inputSchema']for t in listing['result']['tools']if t['name']=='operator_verify')
            ledger=out/'budget.sqlite'
            shared.create(ledger,task_id='plugin-probe-'+arm,max_calls=CALLS,max_output_tokens=2048,
                          total_output_tokens=CALLS*2048,wall_clock_s=240)
            with GatewayServer(ledger,task_id='plugin-probe-'+arm,model='fake-model',participant='outer',
                adapter=lambda request:provider(request),max_calls_limit=CALLS)as gateway:
                built=surface.build(arm,runtime=runtime,relay=relay,base_url=gateway.base_url,
                    arm_dir=participants/arm,max_tokens=2048,development=development)
                (work/'own.txt').write_text('OWN_CANARY');(work/'alias').symlink_to(private/'other-arm/candidate.py')
                provider=BoundaryProvider(arm,built,private,svc,schema,instructions)
                response=s5_arm_env.start_headless(built,'Execute only the scripted local black-box fixture.',
                    timeout=180,capture_dir=out/'headless-output')
            write(out/'events.json',response['events']);write(out/'stderr.json',dict(text=response['stderr']))
            write(out/'provider-records.json',provider.requests)
            snapshot=svc.snapshot();write(out/'verification.json',snapshot)
            proof=audit(response['events'],snapshot,arm=arm);write(out/'transcript-audit.json',proof)
            progress=workflow_progress(response['events'],snapshot)
            write(out/'workflow-progress.json',progress)
            budget=shared.snapshot(ledger);write(out/'budget.json',budget)
            write(out/'launch.json',built)
            archive=s5_seatbelt.archive_work(built,out/'participant-work');write(out/'workspace-archive.json',archive)
            from benchmarks.s5_formal_run import _final_text
            refused=any(r.get('extra_public_submission_refused')for r in provider.requests)
            passed=(response['returncode']==0 and len(provider.requests)==CALLS and proof['complete']
                and snapshot['development_used']==2 and snapshot['final_used']==0 and archive['complete']
                and _final_text(response['events'])==GOOD_FIXTURE and refused
                and(arm!='C'or(proof['plugin_verification_observed']and provider.schema_discovered
                               and provider.instructions_discovered)))
            row=dict(arm=arm,status='passed'if passed else'unknown',calls=budget['reserved_calls'],
                matrix=next((r['matrix']for r in provider.requests if'matrix'in r),None),
                transcript_complete=proof['complete'],plugin_verification_observed=proof['plugin_verification_observed'],
                schema_discovered=provider.schema_discovered,instructions_discovered=provider.instructions_discovered,
                extra_public_submission_refused=refused,development_used=snapshot['development_used'],
                final_used=snapshot['final_used'],workflow_progress=progress)
            write(out/'result.json',row);records.append(row)
        if not passed:break
    surface.verify_runtime(runtime)
    summary=dict(schema=SCHEMA,records=records,passed=len(records)==2 and all(r['status']=='passed'for r in records),
        runtime=runtime,runtime_files_hash=digest(runtime['files']),isolation_implementation=implementation(),
        real_model_calls=0,gpu_executions=0,network_isolation_verified=False,formal_s5_credit=False)
    summary['hash']=digest(summary);write(root/'summary.json',summary)
    return summary


def load_readiness(path):
    path=Path(path)
    if path.is_symlink()or not path.is_file():raise ValueError('plugin-only readiness report missing')
    report=json.loads(path.read_text())
    if(report.get('schema')!=SCHEMA or report.get('passed')is not True
        or report.get('real_model_calls')!=0 or report.get('gpu_executions')!=0
        or digest({k:v for k,v in report.items()if k!='hash'})!=report.get('hash')
        or report.get('isolation_implementation')!=implementation()
        or report.get('runtime_files_hash')!=digest(report['runtime']['files'])
        or[ r.get('arm')for r in report.get('records',[])]!=['A','C']):
        raise ValueError('plugin-only readiness report incomplete or drifted')
    for row in report['records']:
        c=row['arm']=='C'
        if(row.get('status')!='passed' or row.get('calls')!=CALLS or row.get('matrix')!=matrix_expected(row['arm'])
            or row.get('transcript_complete')is not True or row.get('development_used')!=2
            or row.get('final_used')!=0 or row.get('extra_public_submission_refused')is not True
            or row.get('plugin_verification_observed')is not c or row.get('schema_discovered')is not c
            or row.get('instructions_discovered')is not c):
            raise ValueError('plugin-only readiness evidence incomplete')
    surface.verify_runtime(report['runtime'])
    return dict(report_sha256=s5_seatbelt.checksum(path),report_path=str(path.resolve()),
        runtime=report['runtime'],runtime_files_hash=report['runtime_files_hash'],
        isolation_implementation=report['isolation_implementation'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    report=run(parser.parse_args().output);print(json.dumps(report,indent=2));return 0 if report['passed']else 1


if __name__=='__main__':raise SystemExit(main())
