"""Offline paired-task summaries; repeats are not independent task samples."""
import collections
import math
import random
from operator_agent.core.run_contract import digest


def aggregate(plan, records, bootstrap_samples=10000, seed=20261029):
    if type(bootstrap_samples)is not int or not 100<=bootstrap_samples<=100000:
        raise ValueError('invalid bootstrap samples')
    ids=[c['case_id']for c in plan['cases']];repeats=plan.get('repeats',1)
    if not ids or len(set(ids))!=len(ids)or type(repeats)is not int or repeats<1:
        raise ValueError('invalid task grid')
    expected={(case,r,arm)for case in ids for r in range(1,repeats+1)for arm in 'ABC'}
    lookup={}
    for row in records:
        key=(row.get('case_id'),row.get('repeat_id',1),row.get('arm'))
        if key not in expected or key in lookup:raise ValueError('duplicate or unplanned record')
        if row.get('plan_hash')!=plan['hash']:raise ValueError('record plan drift')
        if row.get('status')not in ('passed','failed','unknown'):raise ValueError('invalid verdict')
        if row['status']=='passed' and row.get('execution_state')!='finished':
            raise ValueError('unfinished row cannot be passed')
        lookup[key]=row
    rows=[]
    for key in sorted(expected):
        rows.append(lookup.get(key,dict(case_id=key[0],repeat_id=key[1],arm=key[2],
            status='unknown',execution_state='not_executed',missing=True)))
    groups={}
    for arm in 'ABC':
        subset=[r for r in rows if r['arm']==arm];counts=collections.Counter(r['status']for r in subset)
        costs={}
        for field in ('calls','wall_seconds'):
            values=[(r.get('model_calls')or{}).get('reserved')if field=='calls'else r.get(field)for r in subset]
            if any(v is not None and (type(v)not in (int,float)or not math.isfinite(v)or v<0)for v in values):
                raise ValueError('invalid cost')
            costs[field]=dict(observed_sum=sum(v for v in values if v is not None),
                unknown_rows=sum(v is None for v in values),complete=all(v is not None for v in values))
        groups[arm]=dict(planned=len(subset),passed=counts['passed'],failed=counts['failed'],unknown=counts['unknown'],
            not_executed=sum(r.get('execution_state')=='not_executed'for r in subset),
            success_rate=counts['passed']/len(subset),costs=costs,
            mcp_used=sum(r.get('development_audit',{}).get('plugin_verification_observed')is True for r in subset))
        groups[arm]['dsh_usage']={key:dict(
            observed_sum=sum((r.get('dsh_usage')or{}).get(key,0)for r in subset),
            unknown_rows=sum(key not in (r.get('dsh_usage')or{})for r in subset))
            for key in ('inputTokens','cacheReadTokens','outputTokens')}
    complete=all(not r.get('missing') and r.get('execution_state')!='not_executed'for r in rows)
    definitive=all(r['status']in ('passed','failed') or r.get('outcome_class')in (
        'call_budget_exhausted','output_budget_exhausted','per_call_output_exhausted',
        'wall_clock_exhausted','no_submission') for r in rows)
    comparisons={};rng=random.Random(seed)
    for a,b in [('C','B'),('B','A'),('C','A')]:
        differences=[]
        for case in ids:
            rates=[]
            for arm in (a,b):
                rates.append(sum(lookup.get((case,r,arm),{}).get('status')=='passed'
                                 for r in range(1,repeats+1))/repeats)
            differences.append(rates[0]-rates[1])
        samples=sorted(sum(rng.choice(differences)for _ in ids)/len(ids)for _ in range(bootstrap_samples))
        interval=[samples[int(.025*(bootstrap_samples-1))],samples[int(.975*(bootstrap_samples-1))]]
        comparisons[a+'-minus-'+b]=dict(success_rate_difference=sum(differences)/len(ids),
            paired_task_differences=differences,task_cluster_percentile_95=interval,
            inference_allowed=complete and definitive,
            note='descriptive small-task bootstrap; no guaranteed power; unknowns are not successes')
        comparisons[a+'-minus-'+b]['cost_per_attempt_difference']={
            key:(groups[a]['costs'][key]['observed_sum']-groups[b]['costs'][key]['observed_sum'])/(len(ids)*repeats)
            if groups[a]['costs'][key]['complete'] and groups[b]['costs'][key]['complete'] else None
            for key in ('calls','wall_seconds')}
    body=dict(schema='s5-minimal-summary-v1',plan_hash=plan['hash'],planned=len(rows),
        received=len(records),groups=groups,comparisons=comparisons,rows=rows,
        completed_denominator=complete,formal_s5_credit=False,primary_comparison='C-minus-B',
        billing='costs cover all assigned attempts; absent usage is unknown, not zero; no invoice claim',
        limitations=['few prospective project tasks; training exposure unknown',
            'task-cluster bootstrap does not establish general operator-development gains'])
    return dict(body,hash=digest(body))
