"""Real-client baseline and production delegated-worker adapters."""
import json
from pathlib import Path
import re
import sys
import time
import uuid

from operator_agent.app import autonomous, hitl_tools as ht
from operator_agent.execute.budget import BudgetLedger, BudgetPlan
from operator_agent.execute.llm_budget import ProductionBudgetExhausted
from operator_agent.execute.task_budget import TaskBudgetConfig, snapshot
from operator_agent.execute.task_clock import monotonic

from .baseline import _persist_verification
from .connection import client, environment, options
from .delegate import source_ref
from .plan import read, write_once


def _code(reply):
    blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', reply, flags=re.S)
    return (blocks[-1] if blocks else reply).strip()+'\n'


def baseline(config):
    work = Path(config['run_dir'])
    work.mkdir(parents=True,exist_ok=True)
    write_once(work/'connected-start.json',dict(started_at=time.time(),cell_id=config['cell_id']))
    prepared = ht.prepare(config['declaration'],work)
    policy = TaskBudgetConfig.from_dict(config['budget'])
    ledger = BudgetLedger(BudgetPlan(oracle_total=policy.oracle_total, wall_clock_s=policy.wall_clock_s))
    _persist_verification(work,ledger,policy.oracle_total)
    result = dict(cell_id=config['cell_id'],plan_hash=config['plan_hash'],arm='A',phase='done',
        levels={},reason='rounds_exhausted',selected_source=None,source_hash=None,
        contract_hash=prepared['contract_hash'],execution_context=config['execution_context'])
    events=[]
    llm=client(config,work)
    messages=[dict(role='system',content=(
        'Return only source defining kernel with the declared operands. '
        'Use CPU NumPy for dsl=python; use Triton for dsl=triton. '
        'Support the complete input layouts and builtin mathematical semantics. '
        'Revise source yourself using verification feedback; do not delegate generation.')),
        dict(role='user',content=json.dumps(config['declaration'],sort_keys=True))]
    try:
        for index in range(1,config['max_rounds']+1):
            phase='explore' if index==1 else 'refine'
            if monotonic() >= config['deadline_monotonic']:
                result['reason']='budget_wall_clock'; break
            if not ledger.oracle_may(phase,1):
                ledger.oracle_refuse(phase)
                result['reason']='verification_budget_exhausted'; break
            code=_code(llm.chat(messages,caller_tag='Coder' if index==1 else 'Fixer',
                               temperature=config['model']['temperature']))
            path=work/('candidate-%d.py'%index)
            with path.open('x',encoding='utf-8') as f:f.write(code)
            ref=source_ref(path)
            result.update(selected_source=str(path),source_hash=ref['sha256'])
            ledger.oracle_spend(phase,1)
            _persist_verification(work,ledger,policy.oracle_total)
            events.append(dict(id='verify-%d-call'%index,timestamp=time.time(),stage='check',
                actor='agent',kind='tool_call',tool='baseline.verify',action='candidate_verify',source=ref))
            autonomous._atomic_json(work/'events.json',events)
            verdict=ht.verify(work,code,backend=config['backend'],candidate_origin='outer',**options(config))
            events.append(dict(id='verify-%d-result'%index,timestamp=time.time(),stage='check',
                actor='agent',kind='tool_result',tool='baseline.verify',action='candidate_verify',
                ok=verdict['passed'],source=source_ref(verdict['evidence'])))
            result['levels']=verdict['levels']
            if verdict['passed'] or (config['backend']=='fake' and verdict['levels'].get('L2') is True):
                result['reason']='candidate_verified'; break
            events.append(dict(id='retry-%d'%index,timestamp=time.time(),stage='modify',actor='agent',
                kind='retry',source=source_ref(verdict['evidence']),reason='verification_failed'))
            messages.extend([dict(role='assistant',content=code),dict(role='user',content=json.dumps(
                {k:verdict[k] for k in ('levels','verdicts')},ensure_ascii=False))])
    except ProductionBudgetExhausted as exc:
        result['reason']=dict(llm_call_budget='budget_llm_calls',task_wall_clock='budget_wall_clock',
                              llm_output_budget='budget_llm_output').get(exc.reason,exc.reason)
    except Exception as exc:
        result.update(phase='failed',reason=type(exc).__name__)
        raise
    finally:
        _persist_verification(work,ledger,policy.oracle_total)
        autonomous._atomic_json(work/'events.json',events)
        result['budget']=snapshot(work,policy,config['started_monotonic'])
        write_once(work/'connected-result.json',result)
    return result


def delegated(config):
    work=Path(config['run_dir']); work.mkdir(parents=True,exist_ok=True)
    prepared=ht.prepare(config['declaration'],work)
    cfg=dict(backend=config['backend'],max_rounds=config['max_rounds'],
        task_budget=config['budget'],invocation_id=uuid.uuid4().hex,
        **options(config))
    if config['backend']=='fake':cfg['fake']=dict(scenario='success',profile='small',llm='live')
    if 'sampling' in config:
        from .sampling import request_timeout
        cfg['llm_timeout']=request_timeout(config)
    write_once(work/'autonomous.json',cfg)
    autonomous._worker_main(work,config['deadline_monotonic'],config['started_monotonic'],cfg['invocation_id'])
    raw=read(work/'worker-result.json')
    summary=read(work/'conductor.json') if (work/'conductor.json').is_file() else {}
    source=Path(summary['path'])/'code.py' if summary.get('path') else None
    result=dict(cell_id=config['cell_id'],plan_hash=config['plan_hash'],arm='B',phase=raw['phase'],
        reason=raw.get('stopped_reason') or raw.get('error') or 'worker_returned',
        selected_source=str(source) if source and source.is_file() else None,
        source_hash=source_ref(source)['sha256'] if source and source.is_file() else None,
        contract_hash=prepared['contract_hash'],levels=summary.get('levels',{}),
        execution_context=config['execution_context'])
    write_once(work/'connected-result.json',result)
    events=[]
    for index,path in enumerate(sorted(work.rglob('verification-event-*.json'))):
        value=read(path)
        events.append(dict(id='level-%d'%index,timestamp=value['timing']['completed_at'],
            stage=value['level'],actor='agent',kind='tool_result',action='candidate_verify',
            tool='operator_run',ok=value['passed'],source=source_ref(path)))
    write_once(work/'events.json',events)
    return result


def run(config):
    if config.get('schema')!='phase1-connected-cell-v1':raise ValueError('invalid connected cell')
    from .sampling import bounded
    with environment(config), bounded(config):
        if config['arm']=='A':
            if config.get('entry_a')=='dsh_native':
                from .native import run as native_run
                return native_run(config)
            return baseline(config)
        if config['entry_b']=='core':return delegated(config)
        from .host import run as host_run
        return host_run(config)


if __name__=='__main__':
    print(json.dumps(run(read(sys.argv[1])),ensure_ascii=False))
