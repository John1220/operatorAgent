"""Freeze or run two bounded production-CLI H20 ruler checks, no LLM calls.

`freeze` is local-only. `execute` requires the exact frozen plan SHA256 and
performs real GPU work; each invocation is one-shot and is not a performance
comparison experiment. No remote service installation or broad process kill.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from operator_agent.contract import freeze_from_task
from operator_agent.correctness import as_suite
from operator_agent.hitl_tools import build_task
from operator_agent.run_contract import digest, protocol_identity

UUID = 'GPU-0f61d0d6-767e-44a6-0cd0-a43892bf17e2'
SOURCE = ROOT/'output/plugin_completion/20261027-s4/gpu6-8004/tasks/matmul-aligned/candidate/code.py'
SOURCE_SHA = '1cc3de9144390a883b4e64e8df1dc54a7ed4e708fef231fd66511dd3329b89d2'
EXPECTED = {2048:'92edfa5c8a389f012c87ed5745567c597812ed7a7b11046ba14894cf7e4ad086',
            4096:'7770b8be83fd04c7f614763c770401f63d97118508e7134bdcd86898d08c38ae'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def declaration(n):
    return dict(request='matmul fp16 triton', shapes=[[n,n,n]], hardware='nvidia_gpu',
                dsl='triton', precision='fp16', min_speedup=1.0, required_caps=['real_device'])


def remote_files():
    # These include session_control.py and gpu_claim.py, which are not all in
    # protocol_identity(). Bind the complete deployment set before any SSH.
    from operator_agent.remote_executor import SSHExecutor
    ex=SSHExecutor(host='algorithm-h20',work_dir='~/operator_agent',
                   remote_runner_path='~/operator_agent/run_kernel.py',persistent=True)
    files=ex._remote_files()
    result={str(remote):sha(local) for local,remote in files}
    if len(result)!=15: raise ValueError('remote file set changed')
    return result


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')


def freeze(out):
    out = Path(out).resolve()
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError('S4 selected candidate missing or hash changed')
    rows=[]
    for n in (2048, 4096):
        spec=declaration(n); task=build_task(spec); contract=freeze_from_task(task)
        if contract.hash != EXPECTED[n]:
            raise ValueError('local task contract drift')
        rows.append(dict(name='matmul-'+str(n), declaration=spec, contract_sha256=contract.hash,
                         suite_sha256=as_suite(task.constraints.correctness_suite).digest()))
    out.mkdir(parents=True, exist_ok=False)
    (out/'candidate.py').write_bytes(SOURCE.read_bytes())
    plan=dict(schema='s5-ruler-preflight-v1', purpose='measurement qualification only; not S5 comparison',
              host='algorithm-h20', gpu_index=6, gpu_uuid=UUID, gpu_name='NVIDIA H20',
              min_free_mib=4096, max_util_pct=5, candidate_sha256=SOURCE_SHA,
              candidate_origin='inner_coder', protocol_sha256=digest(protocol_identity()),
              remote_files=remote_files(),
              script_sha256=sha(__file__), tasks=rows, model_calls=0, max_verifications=2,
              task_timeout_s=600, cleanup_grace_s=30, ssh_rpc_timeout_s=300,
              gates='existing L1, full L2, L3 and existing measurement validity; no threshold change',
              scope='single readings only; no independent performance confirmation or speedup claim')
    write(out/'plan.json', plan)
    print(json.dumps({'plan':str(out/'plan.json'), 'sha256':sha(out/'plan.json'),
                      'tasks':rows, 'model_calls':0}, ensure_ascii=False, indent=2))


def read(path):
    try: return json.loads(Path(path).read_text())
    except (OSError, ValueError): return None


def env_for(plan):
    env=os.environ.copy()
    for key in list(env):
        if key.startswith('LLM_') or key in ('HY_API_KEY','REMOTE_WORK_DIR','REMOTE_RUNNER_PATH',
                                              'REMOTE_SSH_PORT','REMOTE_SSH_USER','REMOTE_SSH_KEY'):
            env.pop(key,None)
    env.update(OPERATOR_AGENT_REQUIRED_GPU=str(plan['gpu_index']), OPERATOR_AGENT_REQUIRED_GPU_UUID=plan['gpu_uuid'],
               OPERATOR_AGENT_REQUIRED_GPU_NAME=plan['gpu_name'], OPERATOR_AGENT_EXISTING_REMOTE_ONLY='1',
               OPERATOR_AGENT_S4_GRACEFUL_CLEANUP='1')
    return env


def device_snapshot(plan, env):
    def query(command):
        p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',plan['host'],command],
                         env=env,capture_output=True,text=True,check=True,timeout=15)
        return list(csv.reader(io.StringIO(p.stdout),skipinitialspace=True))
    gpus=query('nvidia-smi --query-gpu=index,uuid,name,memory.free,utilization.gpu --format=csv,noheader,nounits')
    apps=query('nvidia-smi --query-compute-apps=gpu_uuid,used_gpu_memory --format=csv,noheader,nounits')
    if any(len(x)!=2 for x in apps): raise ValueError('compute process query incomplete')
    found=[x for x in gpus if len(x)==5 and x[1].strip()==plan['gpu_uuid']]
    if len(found)!=1: raise ValueError('GPU identity missing or ambiguous')
    row=found[0]
    if (int(row[0])!=plan['gpu_index'] or row[2].strip()!=plan['gpu_name'] or
            int(row[3])<plan['min_free_mib'] or int(row[4])>plan['max_util_pct'] or int(row[4])<0):
        raise ValueError('GPU identity/capacity/utilization does not satisfy frozen plan')
    if any(x[0].strip()==plan['gpu_uuid'] for x in apps):
        raise ValueError('GPU has another compute process; no task dispatched')
    return dict(observed_at=time.time(), gpu_index=int(row[0]), uuid=row[1].strip(),
                name=row[2].strip(), free_mib=int(row[3]), util_pct=int(row[4]),
                other_compute_processes=0, exclusive_lease=False)


def validate(out, plan_sha):
    plan_file=out/'plan.json'
    if sha(plan_file)!=plan_sha: raise ValueError('plan sha mismatch')
    p=read(plan_file)
    if not isinstance(p,dict) or p.get('schema')!='s5-ruler-preflight-v1': raise ValueError('invalid plan')
    if sha(out/'candidate.py')!=SOURCE_SHA or p.get('candidate_sha256')!=SOURCE_SHA:
        raise ValueError('candidate drift')
    if p.get('script_sha256')!=sha(__file__) or p.get('protocol_sha256')!=digest(protocol_identity()):
        raise ValueError('frozen implementation drift')
    if p.get('remote_files')!=remote_files(): raise ValueError('frozen remote file identity drift')
    if (p.get('host'),p.get('gpu_index'),p.get('gpu_uuid'),p.get('gpu_name')) != ('algorithm-h20',6,UUID,'NVIDIA H20'):
        raise ValueError('GPU/host identity drift')
    if (p.get('min_free_mib'),p.get('max_util_pct'),p.get('model_calls'),p.get('max_verifications'),
        p.get('task_timeout_s'),p.get('cleanup_grace_s'),p.get('ssh_rpc_timeout_s')) != (4096,5,0,2,600,30,300):
        raise ValueError('execution budget drift')
    rows=p.get('tasks')
    if not isinstance(rows,list) or len(rows)!=2: raise ValueError('task count drift')
    for row,n in zip(rows,(2048,4096)):
        if row.get('name')!='matmul-'+str(n) or row.get('declaration')!=declaration(n): raise ValueError('task drift')
        task=build_task(row['declaration']); contract=freeze_from_task(task)
        if contract.hash!=EXPECTED[n] or row.get('contract_sha256')!=contract.hash or row.get('suite_sha256')!=as_suite(task.constraints.correctness_suite).digest():
            raise ValueError('contract/suite drift')
    return p


def execute(out, plan_sha):
    # SSHExecutor reads process environment. Restore all affected names even
    # when validation, the one-shot marker or remote setup fails. This helper
    # must run without concurrent threads that also mutate these environment
    # variables; subprocess environments remain independently explicit.
    prefixes = ('REMOTE_', 'OPERATOR_AGENT_')
    saved = {key: value for key, value in os.environ.items() if key.startswith(prefixes)}
    try:
        return _execute_inner(out, plan_sha)
    finally:
        for key in list(os.environ):
            if key.startswith(prefixes):
                os.environ.pop(key, None)
        os.environ.update(saved)


def _execute_inner(out, plan_sha):
    out=Path(out).resolve(); plan=validate(out,plan_sha); env=env_for(plan)
    # This process only uses the controlled environment for the local SSH executor.
    for key in list(os.environ):
        if key.startswith('REMOTE_'): os.environ.pop(key,None)
    for key,value in env.items():
        if key.startswith('OPERATOR_AGENT_'): os.environ[key]=value
    from operator_agent.remote_executor import SSHExecutor
    start=out/'execution-started.json'
    write(start,dict(plan_sha256=plan_sha, started_at=time.time(), no_resume=True))
    summaries=[]
    for row in plan['tasks']:
        validate(out,plan_sha)
        ex=SSHExecutor(host=plan['host'],work_dir='~/operator_agent',
                       remote_runner_path='~/operator_agent/run_kernel.py',persistent=True,
                       timeout=plan['ssh_rpc_timeout_s'])
        if not ex.deploy(): raise RuntimeError('existing remote environment unavailable')
        snapshot=device_snapshot(plan,env)
        result_dir=out/row['name']; result_dir.mkdir(exist_ok=False)
        write(result_dir/'preflight.json',snapshot)
        spec=result_dir/'declaration.json'; write(spec,row['declaration'])
        run=result_dir/'run'
        prep=subprocess.run([sys.executable,'-m','operator_agent','prepare','--declaration',str(spec),
                             '--run-dir',str(run)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=60)
        (result_dir/'prepare.stdout.json').write_text(prep.stdout)
        (result_dir/'prepare.stderr.log').write_text(prep.stderr)
        if prep.returncode!=0 or (read(result_dir/'prepare.stdout.json') or {}).get('contract_hash')!=row['contract_sha256']:
            raise RuntimeError('CLI prepare failed or contract drift; no verify dispatched')
        cmd=[sys.executable,'-m','operator_agent','verify','--run-dir',str(run),'--code-file',str(out/'candidate.py'),
             '--candidate-origin',plan['candidate_origin'],'--stop-after','L3','--backend','ssh','--host',plan['host'],
             '--persistent','--timeout',str(plan['ssh_rpc_timeout_s'])]
        write(result_dir/'verify-dispatch.json',dict(started_at=time.time(),task=row['name'],candidate_sha256=SOURCE_SHA))
        sig=[False]
        def request_stop(*_): sig[0]=True
        old={s:signal.signal(s,request_stop) for s in (signal.SIGTERM,signal.SIGINT)}
        try:
            with (result_dir/'verify.stdout.json').open('w') as stdout,(result_dir/'verify.stderr.log').open('w') as stderr:
                proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
                deadline=time.monotonic()+plan['task_timeout_s']; reason=None
                while proc.poll() is None and time.monotonic()<deadline and not sig[0]: time.sleep(.1)
                if proc.poll() is None:
                    reason='interrupted' if sig[0] else 'wall_clock'
                    try: proc.send_signal(signal.SIGTERM)
                    except ProcessLookupError: pass
                    try: proc.wait(timeout=plan['cleanup_grace_s'])
                    except subprocess.TimeoutExpired: pass
        finally:
            for s,h in old.items(): signal.signal(s,h)
        calls=list((run/'tool-calls').iterdir()) if (run/'tool-calls').is_dir() else []
        status=read(calls[0]/'run-status.json') if len(calls)==1 else None
        cleanup=((status or {}).get('cleanup') or {}).get('remote_execution')
        payload=read(result_dir/'verify.stdout.json')
        item=dict(task=row['name'], cli_exit=proc.poll(), stopped_reason=reason,
                  remote_execution=cleanup, status=status, payload=payload)
        write(result_dir/'supervision.json',item)
        if proc.poll() is None or cleanup not in ('quiescent','not_started'):
            raise RuntimeError('remote cleanup unproven; stop campaign')
        if reason or proc.returncode!=0 or not isinstance(payload,dict):
            raise RuntimeError('verification incomplete; stop campaign')
        l3=next((x for x in payload.get('verdicts',[]) if x.get('level')=='L3'),{})
        metric=l3.get('metrics') or {}
        summaries.append(dict(task=row['name'],levels=payload.get('levels'),passed=payload.get('passed'),
                              validity=metric.get('measurement_validity'),
                              measurement_can_gate=metric.get('measurement_can_gate'),
                              remote_execution=cleanup,results='raw evidence in task folder'))
    write(out/'summary.json',dict(schema='s5-ruler-preflight-result-v1', plan_sha256=plan_sha,
        scope=plan['scope'],tasks=summaries,model_calls=0))
    print(json.dumps(summaries,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['freeze','execute'])
    p.add_argument('--output',required=True); p.add_argument('--plan-sha256')
    args=p.parse_args()
    if args.mode=='freeze': freeze(args.output)
    else:
        if not args.plan_sha256: p.error('execute requires --plan-sha256')
        execute(args.output,args.plan_sha256)


if __name__=='__main__': main()
