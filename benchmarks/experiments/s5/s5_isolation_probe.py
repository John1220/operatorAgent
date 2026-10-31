"""No-model/no-GPU acceptance: actual DSH read/bash/CLI/MCP in outer Seatbelt.

Only scripted local model replies and two reviewed mock candidates are used.
Private canaries, ledger and evaluator stay outside the participant boundary.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys
import tempfile

from benchmarks.experiments.s5 import s5_arm_env, s5_seatbelt, s5_verify_remote
from benchmarks.experiments.s5.s5_development_service import DevelopmentService
from benchmarks.experiments.s5.s5_development_evidence import audit, _feedback
from benchmarks.experiments.s5.s5_gateway_http import GatewayServer
from benchmarks.experiments.s5.s5_vertical_tool_provider import _sse
from benchmarks.experiments.s5.s5_vertical_validation import GOOD_FIXTURE, BAD_FIXTURE
from operator_agent.execute import s5_shared_budget as shared
from operator_agent.app.hitl_tools import prepare
from operator_agent.core.run_contract import digest

DECLARATION = dict(request='layernorm fp32', shapes=[[7,33]], hardware='mock_gpu',
                   dsl='python', precision='fp32')
FORBIDDEN_TEXT = 'PRIVATE_CANARY_MUST_NOT_REACH_DSH'


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=False);f.write('\n')


def find_json(text):
    for i, char in enumerate(text):
        if char != '{': continue
        try: obj, _ = json.JSONDecoder().raw_decode(text[i:])
        except ValueError: continue
        if isinstance(obj,dict) and obj.get('probe') == 's5-read-matrix':return obj
    raise ValueError('read matrix output missing')


class BoundaryProvider:
    def __init__(self, arm, built, private, service):
        self.arm,self.built,self.private,self.service=arm,built,private,service
        self.requests=[]

    def command(self):
        root=Path(self.built['isolation']['runtime']['root'])
        paths=dict(other=self.private/'other-arm/candidate.py',
            credentials=self.private/'host-home/credentials',
            ledger=self.private/'supervisor-ledger',
            source_repository=s5_seatbelt.ROOT/'operator_agent/app/hitl_tools.py',
            core=root/'core/operator_agent/__init__.py',plugin=root/'plugin/index.mjs')
        source=('import json,pathlib\n'
                'result={"probe":"s5-read-matrix"}\n'
                'for label,path in '+repr({k:str(v)for k,v in paths.items()})+'.items():\n'
                ' try:\n'
                '  pathlib.Path(path).read_bytes();result[label]=True\n'
                ' except PermissionError:result[label]=False\n'
                'print(json.dumps(result))\n')
        return shlex.quote(sys.executable)+' -c '+shlex.quote(source)

    def verify_call(self, source):
        task=Path(self.built['work'])/'task'
        if self.arm=='C':
            return 'mcp__operator__operator_verify',dict(run_dir=str(task),code=source,stop_after='L2',backend='mock')
        command=('s5-verify --code-file candidate.py' if self.arm=='A' else
            'operator-agent verify --run-dir '+shlex.quote(str(task))+' --code-file candidate.py --stop-after L2 --backend mock')
        return 'bash',dict(description='Verify reviewed mock candidate',
            command='printf %s '+shlex.quote(source)+' > candidate.py && '+command)

    def __call__(self, request):
        number=len(self.requests)+1
        record=dict(sequence=number,request_sha256=digest(request))
        self.requests.append(record)
        if number>1:
            replies=[m for m in request.get('messages',[]) if m.get('role')=='tool']
            latest=replies[-1]
            text=latest.get('content','')
            record['previous_tool_result']=text
            if latest.get('tool_call_id')!='boundary_'+str(number-1):raise ValueError('tool ID drift')
            if FORBIDDEN_TEXT in text:raise ValueError('private canary escaped')
            if number==2 and 'OWN_CANARY' not in text:raise ValueError('own read failed')
            if number in (3,4) and not any(t in text.lower()for t in ('error','denied','permitted','eacces')):
                raise ValueError('denied read not observed')
            if number==5:
                matrix=find_json(text)
                expected=dict(probe='s5-read-matrix',other=False,credentials=False,ledger=False,
                              source_repository=False,core=self.arm!='A',plugin=self.arm=='C')
                if matrix!=expected:raise ValueError('read matrix mismatch')
                record['matrix']=matrix
            if number in (6,7):
                feedback=_feedback(text)
                actual=self.service.snapshot()['records'][number-6]['result']
                if not isinstance(feedback,dict) or any(feedback.get(k)!=v for k,v in actual.items()):
                    raise ValueError('service feedback changed')
                if feedback['status']!=('failed'if number==6 else'passed'):raise ValueError('unexpected mock verdict')
                record['service_feedback_sha256']=hashlib.sha256(json.dumps(actual,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        work=Path(self.built['work'])
        if number==1:tool,args='read',dict(file_path=str(work/'own.txt'))
        elif number==2:tool,args='read',dict(file_path=str(self.private/'other-arm/candidate.py'))
        elif number==3:tool,args='read',dict(file_path=str(work/'alias'))
        elif number==4:tool,args='bash',dict(description='Probe inherited file read restrictions',command=self.command())
        elif number in (5,6):tool,args=self.verify_call(BAD_FIXTURE if number==5 else GOOD_FIXTURE)
        elif number==7:return _sse({'content':GOOD_FIXTURE},'stop')
        else:raise ValueError('unexpected extra fake model call')
        call=dict(index=0,id='boundary_'+str(number),type='function',
                  function=dict(name=tool,arguments=json.dumps(args)))
        return _sse({'tool_calls':[call]},'tool_calls')


def run(output):
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False,mode=0o700)
    runtime=s5_seatbelt.stage_runtime(root/'runtime')
    private=root/'supervisor-private';private.mkdir(mode=0o700)
    for name in ('other-arm/candidate.py','host-home/credentials','supervisor-ledger'):
        path=private/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(FORBIDDEN_TEXT)
    service_sha=s5_verify_remote.freeze(root/'service-plan',declaration=DECLARATION,backend='mock')
    participants=Path(tempfile.mkdtemp(prefix='operator-s5-seatbelt-')).resolve()
    write(root/'participant-location.json',dict(path=str(participants),temporary=True))
    records=[]
    for arm in 'ABC':
        out=root/'arms'/arm;out.mkdir(parents=True,mode=0o700)
        with DevelopmentService(root/'service-plan/plan.json',service_sha,out/'verification',arm=arm,wall_seconds=180)as svc:
            ledger=out/'budget.sqlite'
            shared.create(ledger,task_id='isolation-'+arm,max_calls=7,max_output_tokens=2048,
                          total_output_tokens=14336,wall_clock_s=180)
            # Trusted callback, no real upstream transport exists in this driver.
            with GatewayServer(ledger,task_id='isolation-'+arm,model='fake-model',participant='outer',
                adapter=lambda request:provider(request),max_calls_limit=7)as gateway:
                built=s5_arm_env.build(arm,base_url=gateway.base_url,arm_dir=participants/arm,
                    tools=True,max_tokens=2048,development=dict(url=svc.url,token=svc.token,
                        contract=svc.plan['contract_hash'],backend='mock',timeout=70),isolation_runtime=runtime)
                work=Path(built['work']);(work/'own.txt').write_text('OWN_CANARY')
                (work/'alias').symlink_to(private/'other-arm/candidate.py')
                if arm!='A':prepare(DECLARATION,work/'task')
                provider=BoundaryProvider(arm,built,private,svc)
                response=s5_arm_env.start_headless(built,'Run only the scripted local isolation fixture.',timeout=120)
            write(out/'events.json',response['events']);write(out/'stderr.json',dict(text=response['stderr']))
            write(out/'provider-records.json',provider.requests)
            snapshot=svc.snapshot();write(out/'verification.json',snapshot)
            proof=audit(response['events'],snapshot,arm=arm);write(out/'transcript-audit.json',proof)
            budget=shared.snapshot(ledger);write(out/'budget.json',budget)
            write(out/'launch.json',built)
            write(out/'workspace-archive.json',s5_seatbelt.archive_work(built,out/'participant-work'))
            passed=(response['returncode']==0 and len(provider.requests)==7 and proof['complete']
                    and snapshot['development_used']==2 and snapshot['final_used']==0
                    and (arm!='C' or proof['plugin_verification_observed']))
            row=dict(arm=arm,status='passed'if passed else'unknown',calls=budget['reserved_calls'],
                     matrix=next((r['matrix']for r in provider.requests if'matrix'in r),None),
                     transcript_complete=proof['complete'],plugin_verification_observed=proof['plugin_verification_observed'])
            write(out/'result.json',row);records.append(row)
        if not passed:break
    s5_seatbelt.verify_runtime(runtime)
    summary=dict(schema='s5-read-isolation-probe-v1',records=records,
        passed=len(records)==3 and all(r['status']=='passed'for r in records),
        runtime=runtime, runtime_files_hash=digest(runtime['files']),
        isolation_implementation=implementation(),
        real_model_calls=0,gpu_executions=0,network_isolation_verified=False,formal_s5_credit=False)
    summary['hash']=digest(summary);write(root/'summary.json',summary)
    return summary


def implementation():
    return {name:s5_seatbelt.checksum(Path(__file__).resolve().parent/name)
            for name in ('s5_seatbelt.py','s5_arm_env.py','s5_isolation_probe.py')}


def load_readiness(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file():raise ValueError('read isolation report missing')
    report=json.loads(path.read_text())
    if (report.get('schema')!='s5-read-isolation-probe-v1' or report.get('passed')is not True
        or report.get('real_model_calls')!=0 or report.get('gpu_executions')!=0
        or digest({k:v for k,v in report.items()if k!='hash'})!=report.get('hash')
        or report.get('isolation_implementation')!=implementation()
        or report.get('runtime_files_hash')!=digest(report['runtime']['files'])
        or [r.get('arm')for r in report['records']]!=list('ABC')
        or any(r.get('status')!='passed' for r in report['records'])):
        raise ValueError('read isolation report incomplete or drifted')
    for row in report['records']:
        arm=row['arm']
        expected=dict(probe='s5-read-matrix',other=False,credentials=False,ledger=False,
                      source_repository=False,core=arm!='A',plugin=arm=='C')
        if (row.get('matrix')!=expected or row.get('calls')!=7
                or row.get('transcript_complete')is not True
                or (arm=='C' and row.get('plugin_verification_observed')is not True)):
            raise ValueError('read isolation evidence incomplete')
    s5_seatbelt.verify_runtime(report['runtime'])
    s5_seatbelt.verify_source_version(report['runtime'])
    return dict(report_sha256=s5_seatbelt.checksum(path),report_path=str(path.resolve()),
                runtime=report['runtime'],runtime_files_hash=report['runtime_files_hash'],
                isolation_implementation=report['isolation_implementation'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    result=run(p.parse_args().output);print(json.dumps(result,indent=2));return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
