"""Nine excluded real workflow runs; stop subsequent dispatch on wiring unknowns.

No candidate reruns, no services stopped, no formal comparison credit. Requires
an explicit plan hash. GPU snapshots are observations, never exclusive leases.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import time

from benchmarks.experiments.s5 import s5_formal_run as workflow
from benchmarks.experiments.s5.s5_ruler_preflight import device_snapshot
from operator_agent.core.run_contract import digest
from benchmarks.experiments.s5.s5_outcomes import fatal, classify

TASKS = ('matmul-tail-fp16', 'layernorm-tail', 'softmax-extreme')
RESOURCE_POLICY = dict(min_free_mib=4096, max_util_pct=5, other_compute_processes=0)


def driver_identity():
    return {name: hashlib.sha256((Path(__file__).resolve().parent / name).read_bytes()).hexdigest()
            for name in ('s5_workflow_preflight.py', 's5_ruler_preflight.py','s5_outcomes.py',
                         's5_shared_gpu.py')}


def freeze(output, isolation_report=None, *, study=False, gpu=None, shared_compute_pids=None):
    from benchmarks.experiments.s5.s5_isolation_probe import load_readiness
    if isolation_report is None:
        raise ValueError('a passed no-model read isolation report is required')
    readiness=load_readiness(isolation_report)
    resource_policy=dict(RESOURCE_POLICY)
    if shared_compute_pids is not None:
        if gpu is None:
            raise ValueError('shared GPU requires an explicit device pin')
        from benchmarks.experiments.s5.s5_shared_gpu import validate_policy
        resource_policy=dict(mode='shared-known-processes',min_free_mib=8192,max_util_pct=5,
                             allowed_compute_pids=shared_compute_pids)
        validate_policy(resource_policy)
    root = Path(output)
    if study:
        from benchmarks.experiments.s5.s5_minimal_tasks import manifest
        plan=workflow.freeze(root,task_manifest=manifest(),repeats=2,read_isolation=readiness,
                             gpu=gpu,resource_policy=resource_policy)
    else:
        plan = workflow.freeze(root, case_ids=TASKS, preflight=True,read_isolation=readiness,
                               gpu=gpu,resource_policy=resource_policy)
    policy = dict(schema='s5-workflow-preflight-policy-v1', plan_hash=plan['hash'],
        driver=driver_identity(), tasks=[c['case_id']for c in plan['cases']], resources=resource_policy,
        phase='minimal_comparison' if study else 'preflight',
        planned_runs=plan['planned_runs'], max_model_reservations=plan['planned_runs'] * plan['budget']['max_calls'],
        max_output_reservations=plan['planned_runs'] * plan['budget']['total_output_tokens'],
        max_verifications=plan['planned_runs']*3, stop_on_wiring_unknown=True,
        ordinary_noncompletion_continues=True, plugin_unused_rows_retained=True,
        counts_toward_denominator=False, formal_s5_credit=False)
    policy['hash'] = digest(policy)
    workflow._write_once(root / 'execution-policy.json', policy)
    return plan, policy


def resources(plan):
    gpu = plan['verification']['gpu']
    policy=plan.get('resource_policy')
    pin=dict(host=plan['verification']['host'],gpu_index=gpu['index'],
             gpu_uuid=gpu['uuid'],gpu_name=gpu['name'])
    if isinstance(policy,dict) and policy.get('mode')=='shared-known-processes':
        from benchmarks.experiments.s5.s5_shared_gpu import snapshot
        return snapshot(pin,policy,os.environ.copy())
    if policy!=RESOURCE_POLICY:
        raise ValueError('unknown frozen resource policy')
    return device_snapshot(dict(pin,**policy),os.environ.copy())


def credential():
    import yaml
    path = Path.home() / '.dsh/.credentials.yaml'
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or stat.S_IMODE(st.st_mode) != 0o600:
        raise ValueError('protected credential file unavailable')
    obj = yaml.safe_load(path.read_text())
    key = obj.get('refs', {}).get('ALIYUN_API_KEY') if isinstance(obj, dict) else None
    if not isinstance(key, str) or not key or '\n' in key or '\r' in key:
        raise ValueError('configured provider credential unavailable')
    return key


def execute(output, expected_plan_hash, expected_policy_hash):
    root = Path(output).resolve()
    plan = workflow.load_plan(root / 'plan.json')
    policy = workflow._read(root / 'execution-policy.json')
    from benchmarks.experiments.s5.s5_isolation_probe import load_readiness
    isolation=plan.get('read_isolation')
    if not isolation or load_readiness(isolation['report_path'])!=isolation:
        raise ValueError('passed read isolation report required before any real dispatch')
    if (plan['hash'] != expected_plan_hash or policy['hash'] != expected_policy_hash
            or digest({k: v for k, v in policy.items() if k != 'hash'}) != policy['hash']
            or policy['plan_hash'] != plan['hash'] or policy['driver'] != driver_identity()
            or policy['tasks'] != [c['case_id']for c in plan['cases']]
            or policy.get('planned_runs') != plan['planned_runs']
            or policy.get('resources') != plan.get('resource_policy')
            or plan.get('counts_toward_denominator') is not False
            or plan['implementation'] != workflow.runtime_identity()):
        raise ValueError('preflight identity drift')
    # Check before the execution marker and before reading the credential.
    initial = resources(plan)
    workflow._write_once(root / 'execution-started.json', dict(plan_hash=plan['hash'],
        policy_hash=policy['hash'], started_at=time.time(), initial_resources=initial))
    api_key = credential()
    stopped = None
    records = []
    cases={c['case_id']:c for c in plan['cases']}
    schedule=plan.get('schedule') or [dict(case_id=c['case_id'],repeat_id=1,arm=a)for c in plan['cases']for a in c['order']]
    for slot in schedule:
        case=cases[slot['case_id']];arm=slot['arm'];repeat_id=slot['repeat_id']
        label=case['case_id']+('-repeat-%02d'%repeat_id if plan.get('repeats',1)>1 else '')+'-'+arm
        if stopped is None:
            try:
                observation = resources(plan)
                workflow._write_once(root / 'resource-checks' / (label+'.json'), observation)
            except (ValueError, OSError, subprocess.SubprocessError) as exc:
                stopped = 'resource precheck refused: ' + type(exc).__name__
        if stopped is None:
            try:
                record = (workflow.run_one(plan, case, arm, root, api_key,repeat_id=repeat_id)
                          if plan.get('repeats',1)>1 else workflow.run_one(plan, case, arm, root, api_key))
            except Exception as exc:
                record=dict(case_id=case['case_id'],arm=arm,plan_hash=plan['hash'],
                    status='unknown',execution_state='interrupted',error=type(exc).__name__,
                    reason='run launch or evidence collection refused',model_calls=None,
                    wall_seconds=None,repo_path_mentions=0,formal_s5_credit=False)
            record['outcome_class']=classify(record)
            if fatal(record):
                stopped = 'wiring unknown after ' + case['case_id'] + '/' + arm
        else:
            record = dict(case_id=case['case_id'], arm=arm, plan_hash=plan['hash'],
                status='unknown', execution_state='not_executed', reason=stopped,
                model_calls=None, wall_seconds=None, repo_path_mentions=0,
                formal_s5_credit=False)
        record['repeat_id']=repeat_id
        workflow._write_once(root / 'records' / (label+'.json'), record)
        records.append(record)
        print(json.dumps(dict(case_id=case['case_id'], arm=arm, status=record['status'],
            state=record['execution_state'], calls=(record.get('model_calls') or {}).get('reserved'),
            development=(record.get('development') or {}).get('development_used'),
            wall_seconds=record['wall_seconds']), sort_keys=True), flush=True)
    if plan.get('repeats',1)>1:
        from benchmarks.experiments.s5.s5_minimal_summary import aggregate
        summary=aggregate(plan,records)
    else:summary = workflow.aggregate(plan, records)
    summary['preflight'] = plan.get('preflight',False)
    summary['counts_toward_denominator'] = False
    summary['formal_s5_credit'] = False
    summary['stopped_reason'] = stopped
    summary['dispatched_runs'] = sum(r['execution_state'] != 'not_executed' for r in records)
    summary['execution_gate_passed'] = stopped is None
    summary['workflow_gate_passed'] = (stopped is None and all(
        r.get('development_audit',{}).get('complete') is True for r in records))
    summary['tool_usage']={arm:sum((r.get('development') or {}).get('development_used',0)>0
                                 for r in records if r['arm']==arm)for arm in 'ABC'}
    summary['mcp_used_runs']=sum(r.get('development_audit',{}).get('plugin_verification_observed')is True
                                for r in records)
    summary.pop('hash', None)
    summary['hash'] = digest(summary)
    workflow._write_once(root / 'aggregate.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--freeze', action='store_true')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--plan-hash')
    p.add_argument('--policy-hash')
    p.add_argument('--isolation-report',type=Path)
    p.add_argument('--study',action='store_true',help='freeze prospective six-task/two-repeat study; not execution authorization')
    p.add_argument('--gpu-index',type=int)
    p.add_argument('--gpu-uuid')
    p.add_argument('--shared-pids',type=int,nargs='+')
    a = p.parse_args()
    if a.freeze and not a.execute:
        if (a.gpu_index is None)!=(a.gpu_uuid is None):
            p.error('GPU index and UUID must be supplied together')
        gpu=(dict(index=a.gpu_index,uuid=a.gpu_uuid,name='NVIDIA H20')
             if a.gpu_index is not None else None)
        plan, policy = freeze(a.output,a.isolation_report,study=a.study,gpu=gpu,
                              shared_compute_pids=a.shared_pids)
        print(json.dumps(dict(plan_hash=plan['hash'], policy_hash=policy['hash'],
                              max_calls=policy['max_model_reservations'],
                              max_output_tokens=policy['max_output_reservations']), sort_keys=True))
    elif a.execute and not a.freeze and a.plan_hash and a.policy_hash:
        summary = execute(a.output, a.plan_hash, a.policy_hash)
        print(json.dumps(dict(summary_hash=summary['hash'], counts=summary['counts'],
                              gate=summary['workflow_gate_passed']), sort_keys=True))
    else:
        p.error('choose --freeze, or --execute with both frozen hashes')


if __name__ == '__main__':
    main()
