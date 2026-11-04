"""Read-only reconciliation of connected entry evidence; no formal study credit."""
import copy
import hashlib
import json
import math
from pathlib import Path

from operator_agent.execute import s5_shared_budget as shared
from operator_agent.execute.s5_inner_budget import _provider_counters
from operator_agent.generate.repair_loop import _full_l2_passed
from . import connected_plan as plans, plan as files
from .connected import configuration
from .events import collect_events
from .summary import _read, _oracle_budget, _verdicts
from .sampling import FIELDS, matches


def _budget(cell,cfg,problems):
    result=dict(complete=False,reserved_calls=None,provider_dispatches=None,api_usage=None,cost_usd=None)
    if cfg is None:return result
    try:
        common=shared.snapshot(cell/'budget.sqlite',expected_task_id=cfg['shared_task_id'])
    except (ValueError,OSError,shared.BudgetError):
        problems.append('shared model ledger unavailable');return result
    calls=common['calls']
    n=cfg['budget']['max_llm_calls']
    known=(not common['aborted'] and common['max_calls']==n and common['max_output_tokens']==40960
        and common['total_output_tokens']==n*40960 and common['reserved_calls']==len(calls)<=n
        and common['reserved_output_tokens']==sum(c['output_tokens_reserved'] for c in calls)
        and all(c['participant'] in ('outer','inner') and type(c['output_tokens_reserved']) is int
                and 0<c['output_tokens_reserved']<=40960 for c in calls))
    inner=[c for c in calls if c['participant']=='inner']
    journal=cell/'work/model-journal'
    local_calls=[_read(p,problems) for p in sorted(journal.glob('call-*.json'))]
    sampling=[]
    if inner or journal.exists():
        local=_read(journal/'summary.json',problems)
        identity=_read(journal/'identity.json',problems)
        limits=_read(journal/'limits.json',problems)
        known &= (isinstance(local,dict) and not local.get('aborted') and local.get('policy')=='production_task'
            and local.get('attempts')==local_calls and local.get('reserved_calls')==len(inner)==len(local_calls)
            and local.get('identity')==identity and local.get('limits')==limits
            and isinstance(identity,dict) and identity.get('model')==cfg['model']['id']
            and identity.get('endpoint_sha256')==cfg['model']['endpoint_sha256']
            and identity.get('credential_present') is True and isinstance(limits,dict)
            and limits.get('max_calls')==n and limits.get('max_output_tokens')==40960
            and limits.get('total_output_token_reservations')==n*40960)
        for i,(local_call,shared_call) in enumerate(zip(local_calls,inner),1):
            if not isinstance(local_call,dict):known=False;continue
            local_usage=local_call.get('api_usage')
            counters=_provider_counters(local_usage,local_call.get('usage_source'))
            known &= (local_call.get('call')==i and all(local_call.get(k)==shared_call.get(k)
                for k in ('caller','request_sha256','output_tokens_reserved'))
                and (local_usage is None or counters is not None)
                and counters==shared_call.get('api_usage')
                and (local_call.get('dispatched_monotonic') is None)==(shared_call.get('dispatched_monotonic') is None)
                and (local_call.get('completed_monotonic') is None)==(shared_call.get('completed_monotonic') is None)
                and local_call.get('request',{}).get('model')==cfg['model']['id'])
            sampling.append(dict(participant='inner',caller=local_call.get('caller'),
                request=local_call.get('request'),request_sha256=local_call.get('request_sha256')))
    outer=[c for c in calls if c['participant']=='outer']
    retained=[_read(p,problems) for p in sorted((cell/'outer-requests').glob('request-*.json'))]
    # Reservations refused before dispatch legitimately have no upstream body.
    dispatched=[c for c in outer if c['dispatched_monotonic'] is not None]
    known &= len(dispatched)==len(retained) and ((cfg['arm']=='B' and cfg['entry_b']=='dsh')
        or (cfg['arm']=='A' and cfg.get('entry_a')=='dsh_native') or not outer)
    for request,call in zip(retained,dispatched):
        if not isinstance(request,dict):known=False;continue
        encoded=json.dumps(request,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
        known &= (request.get('model')==cfg['model']['id'] and hashlib.sha256(encoded).hexdigest()==call['request_sha256'])
        sampling.append(dict(participant='outer',request={k:request[k] for k in
            FIELDS if k in request},
            request_sha256=call['request_sha256']))
    usages=[]
    for call in calls:
        usage=call.get('api_usage')
        if (call.get('usage_evidence')=='provider' and isinstance(usage,dict)
                and all(type(usage.get(k)) is int and usage[k]>=0 for k in ('prompt_tokens','completion_tokens'))):
            usages.append(usage)
    totals={k:sum(u[k] for u in usages) if usages else None for k in ('prompt_tokens','completion_tokens')}
    result.update(complete=bool(known),reserved_calls=len(calls),shared=common,
        provider_dispatches=sum(c['dispatched_monotonic'] is not None for c in calls),
        dispatch_scope='transport_handoffs_not_provider_acceptance',sampling=sampling,
        api_usage=dict(source='validated_api_counters_only',known_calls=len(usages),unknown_calls=len(calls)-len(usages),
            known_totals=totals,totals=totals if len(usages)==len(calls) else None))
    if 'sampling' in cfg:
        intents=[]
        wire_policy=cfg['sampling']
        try:
            for path in sorted((cell/'work').glob('sampling-*.jsonl')):
                intents.extend(json.loads(line) for line in path.read_text().splitlines())
            result['sampling_aligned']=(bool(known and calls)
                and all(matches(body,wire_policy) for body in retained)
                and all(call['output_tokens_reserved']==wire_policy['max_tokens'] and any(
                    isinstance(row,dict) and row.get('request_sha256')==call['request_sha256']
                    and row.get('participant')==call['participant']
                    and matches(row.get('effective'),wire_policy) for row in intents) for call in calls))
        except (OSError,ValueError,TypeError):result['sampling_aligned']=False
        result['sampling_scope']='prepared wire requests; provider honoring is not inferred'
        if not result['sampling_aligned']:problems.append('sampling policy lacks complete request bindings')
    if not known:problems.append('connected local/shared model evidence is incomplete or inconsistent')
    return result


def _final(cell,record,task,plan,development_config,problems):
    result=dict(complete=False,full_l2_passed=None,levels=None,gpu_acceptance=False,performance_evidence=False)
    outcome=record.get('final_evaluation')
    if not isinstance(outcome,dict) or outcome.get('returncode')!=0 or outcome.get('stopped_reason'):return result
    value=_read(cell/'final/final-result.json',problems)
    cfg=_read(cell/'final-config.json',problems)
    if not isinstance(value,dict) or not isinstance(cfg,dict):return result
    source=cell/'selected.py'
    actual=hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() and not source.is_symlink() else None
    verified=value.get('result',{})
    expected=dict(development_config or {},run_dir=str(cell/'final'),source=str(source),
        source_hash=actual,contract_hash=task['contract_hash'])
    finished=value.get('finished_monotonic')
    if (not actual or value.get('schema')!='phase1-connected-final-v1' or value.get('source_hash')!=actual
            or value.get('source_hash')!=outcome.get('source_hash') or value.get('cell_id')!=record['cell_id']
            or value.get('plan_hash')!=plan['hash'] or value.get('contract_hash')!=task['contract_hash']
            or cfg!=expected
            or value.get('execution_context')!=plan['execution_context']
            or type(value.get('verification_count')) is not int or value['verification_count']!=1
            or type(finished) not in (int,float) or not math.isfinite(finished)
            or not outcome['started_monotonic']<=finished<=outcome['deadline_monotonic']
            or not isinstance(verified,dict) or verified.get('contract_hash')!=task['contract_hash']):
        problems.append('final source/contract/context/deadline binding mismatch');return result
    try:parsed=_verdicts(verified.get('verdicts'))
    except (ValueError,KeyError,TypeError):
        problems.append('invalid independent final verdicts');return result
    levels={v.level.value:v.passed for v in parsed}
    if verified.get('levels')!=levels or verified.get('passed')!=all(v.passed for v in parsed):
        problems.append('final verdicts contradict level summary');return result
    if plan['backend']=='fake' and (levels.get('L3') is True or
            verified.get('execution_context',{}).get('test_only') is not True):
        problems.append('fake final improperly claims GPU qualification');return result
    result.update(complete=True,full_l2_passed=_full_l2_passed(parsed),levels=levels,
        source_hash=actual,contract_hash=task['contract_hash'],evidence=str(cell/'final/final-result.json'))
    return result


def _events(cell,cfg,record,problems):
    raw=_read(cell/'work/events.json',problems) if (cell/'work/events.json').is_file() else []
    if not isinstance(raw,list):problems.append('invalid event stream');raw=[]
    checked=copy.deepcopy(raw)
    coverage=dict(conversation='missing',tools='partial',external_edits='missing')
    for event in checked:
        try:
            ref=event['source'];source=Path(ref['path'])
            if source.is_symlink() or cell not in source.resolve().parents or hashlib.sha256(source.read_bytes()).hexdigest()!=ref['sha256']:
                raise ValueError('event source mismatch')
        except (KeyError,ValueError,OSError,TypeError):
            problems.append('unbound connected event source')
            if isinstance(event,dict):event['source']=None
    if cfg and cfg['arm']=='A' and cfg.get('entry_a')!='dsh_native' and record.get('worker_result') and not problems:coverage['tools']='complete'
    # DSH transcript is retained, but external edits and human messages are not
    # exhaustively observed. Never promote zero visible events to zero humans.
    result=collect_events(checked,coverage,arm=cfg['arm'] if cfg else record['arm'])
    result['raw_events']=raw
    return result


def _entry(cell,cfg,record,plan,task,problems):
    if not cfg or cfg['arm']!='B' or cfg['entry_b']!='dsh':return False
    path=cell/'work/connected-result.json'
    value=_read(path,problems)
    try:
        if (not isinstance(value,dict) or record.get('worker_result')!=str(path)
                or record.get('worker_result_sha256')!=hashlib.sha256(path.read_bytes()).hexdigest()
                or value.get('cell_id')!=cfg['cell_id'] or value.get('plan_hash')!=plan['hash']
                or value.get('contract_hash')!=task['contract_hash']
                or value.get('execution_context')!=plan['execution_context']
                or value.get('dsh_entry_verified') is not True):return False
        def bound(ref):
            source=Path(ref['path'])
            if source.is_symlink() or cell not in source.resolve().parents or hashlib.sha256(source.read_bytes()).hexdigest()!=ref['sha256']:
                raise ValueError('host entry source binding mismatch')
            return files.read(source)
        records=[bound(ref) for ref in value['entry_evidence']]
        response=bound(value['host_response'])
        from .host import entry_observed
        return bool(response['returncode']==0 and entry_observed(records))
    except (KeyError,OSError,ValueError,TypeError):
        problems.append('host entry evidence incomplete or changed');return False


def summarize(root):
    root=Path(root).resolve()
    plan=plans.load(root/'plan.json',check_implementation=False)
    execution=files.read(root/'execution.json')
    if execution.get('plan_hash')!=plan['hash']:raise ValueError('execution plan mismatch')
    tasks={r['task_id']:r for r in plan['tasks']}
    rows=[]
    for slot in plan['schedule']:
        cell=root/'cells'/slot['cell_id'];problems=[]
        record=_read(cell/'record.json',problems) or dict(**slot,phase='unknown')
        if any(record.get(k)!=v for k,v in slot.items()):raise ValueError('connected record slot mismatch')
        cfg=_read(cell/'cell-config.json',problems) if (cell/'cell-config.json').is_file() else None
        if cfg:
            expected=configuration(plan,tasks[slot['task_id']],slot,cell,cfg.get('started_monotonic',0))
            if cfg!=expected:raise ValueError('frozen connected cell configuration mismatch')
        model=_budget(cell,cfg,problems)
        oracle=_oracle_budget(cell,plan,problems) if cfg else dict(complete=False,evidence=None)
        final=_final(cell,record,tasks[slot['task_id']],plan,cfg,problems)
        events=_events(cell,cfg,record,problems)
        entry=_entry(cell,cfg,record,plan,tasks[slot['task_id']],problems)
        rows.append(dict(**slot,phase=record['phase'],reason=record.get('reason'),model_budget=model,
            verification_budget=oracle,final_evaluation=final,events=events,
            budget_auditable=bool(model['complete'] and oracle['complete']),dsh_entry_verified=entry,problems=problems))
    return dict(schema='phase1-connected-summary-v1',plan_hash=plan['hash'],phase=execution['phase'],
        execution_context=plan['execution_context'],cells=rows,formal_comparison_credit=False,
        gpu_acceptance=False,performance_evidence=False,installed_plugin_acceptance=False,
        dsh_entry_verified=bool(plan['entry_b']=='dsh' and all(r['dsh_entry_verified'] for r in rows if r['arm']=='B')),
        sampling_comparability=plan['sampling_comparability'],human_minutes=None,
        wire_sampling=dict(configured='sampling' in plan,
            aligned_for_all_started=bool(rows) and all(r['model_budget'].get('sampling_aligned') is True for r in rows),
            provider_honoring='not_inferred'),
        support=plan['support'])
