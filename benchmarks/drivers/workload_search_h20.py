"""Fixed two-candidate H20 search wiring audit; local-plan-only by default.

NOT an LLM search or deployment speedup claim. Frozen min_speedup=10 forces
bounded search of num_warps=4/8 LayerNorm variants, each measured ONCE on two
fixed large cases. No retries, shape/weight/window selection or gate relaxation.
Owned scratch only, no shared runtime edits, installs or shared process kills.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmarks.drivers import workload_decision_h20 as wd
from operator_agent import Conductor, KernelDatabase
from operator_agent.core.artifact_store import ArtifactStore
from operator_agent.execute.budget import BudgetPlan
from operator_agent.core.contract import case_key, freeze_from_task
from operator_agent.core.kernel_db import FailureDatabase
from operator_agent.core.schemas import UnifiedSketch, TilingStrategy, VerificationLevel as Level
from operator_agent.generate.search_objective import reading, validate_snapshot, results_dict
from operator_agent.core.run_contract import atomic_json, audit_value, code_hash, digest, json_value, protocol_identity, load_checkpoint
from operator_agent.decide.reference_contract import registered_reference, validate_receipt


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--host',default='algorithm-h20')
    p.add_argument('--gpu',type=int,choices=range(8))
    p.add_argument('--output',default=str(ROOT/'output'/'workload_search_h20'))
    p.add_argument('--cuda-ruler-profile')
    p.set_defaults(cuda_calibration=None)
    return p


def candidates():
    base=wd.trial_code('unfused-fp32')
    marker='enable_fp_fusion=False'
    assert base.count(marker)==1
    return [dict(name='warps-'+str(w),source=base.replace(marker,marker+',num_warps='+str(w))) for w in (4,8)]


def declaration(args):
    t=wd.task('workload-search-h20')
    t.constraints.min_speedup=10.0  # Frozen search-forcing threshold, never relaxed.
    profile,calibration=wd.ruler(args)
    cs=candidates()
    sources=wd.evaluator_sources()
    sources['benchmarks/drivers/workload_search_h20.py']=code_hash(Path(__file__).read_text())
    body=dict(schema='workload-search-h20-v1',contract=json_value(freeze_from_task(t).as_dict()),
        candidates=[dict(c,hash=code_hash(c['source'])) for c in cs],
        supplied_cases=['X=8192x15360','X=8192x12288'],
        measurement_protocol=protocol_identity(),evaluator_sources=sources,
        budget_accounting=dict(unit='candidate_verification_dispatch_per_input_group',
            reservation_policy='pre_dispatch_non_refundable',
            calibration_policy='observe_only_may_exceed_total',
            fixed_dispatch_sequence_unchanged=True),
        controls=dict(host=args.host,gpu=args.gpu,max_rounds=3,oracle_total=7,heartbeat_interval=1,
            wall_clock_s=1800,max_rpcs=wd.MAX_RPCS,max_commands=wd.MAX_COMMANDS,
            rpc_timeout=wd.RPC_TIMEOUT,cuda_ruler_profile=profile,cuda_calibration=calibration,
            # Two dispatches × two groups + two heartbeat batches = six;
            # one confirm allowance remains protected. No extra candidate/RPC.
            budget=dict(oracle_total=7,wall_clock_s=1800,llm_tokens=128)),
        input_storage_limit=wd.MAX_STORAGE_BYTES,
        retry_windows=False,adaptive=False,performance_scope='fixed-candidate search wiring only')
    return t,json_value(dict(body,hash=digest(body)))


def check_plan(path,plan,t):
    if json.loads(Path(path).read_text())!=plan or plan['hash']!=digest({k:v for k,v in plan.items() if k!='hash'}):
        raise ValueError('frozen search plan drift')
    sources=wd.evaluator_sources()
    sources['benchmarks/drivers/workload_search_h20.py']=code_hash(Path(__file__).read_text())
    if sources!=plan['evaluator_sources'] or protocol_identity()!=plan['measurement_protocol']:
        raise ValueError('frozen search evaluator drift')
    if json_value(freeze_from_task(t).as_dict())!=plan['contract']:
        raise ValueError('frozen search contract drift')


def check_candidate(plan,rounds,code):
    # Enforce fixed order BEFORE Verifier can compile/dispatch, not just in the
    # retrospective audit. A Fixer retry/substitution must not buy a new window.
    if len(rounds)>=2 or code!=plan['candidates'][len(rounds)]['source']:
        raise ValueError('undeclared/retried search candidate refused before verification')


def audit_durable(artifact, rounds, plan, *, task, input_sets, controls):
    """Read-only final-product audit. No metadata backfill, save, resume or RPC.

    Check the actual files independently against raw candidate observations and
    a fully sealed checkpoint. Loading an artifact is NOT live verification.
    Missing/torn/mismatched durable evidence fails this audit, never triggers a
    measurement retry or patches the original product to make it look accepted.
    """
    errors=[];proof=dict(completed=False,errors=errors,read_only=True,
                        gpu_measurements=0,resumed=False)
    try:
        from operator_agent.core.contract import load_dict
        from operator_agent.core.report import build_report
        directory=Path(artifact.saved_dir)
        contract=load_dict(plan['contract'])
        if contract.hash!=freeze_from_task(task).hash or protocol_identity()!=plan['measurement_protocol']:
            raise ValueError('durable audit frozen contract/protocol drift')
        required=('result.json','sketch.json','verification.json','benchmark.json',
                  'execution_snapshot.json','contract.json','run_manifest.json','report.md')
        code_file=ArtifactStore._find_code_file(directory)
        if code_file is None:raise ValueError('durable selected source missing')
        paths=[directory/name for name in required]+[code_file]
        if directory.is_symlink() or any(p.is_symlink() or not p.is_file() for p in paths):
            raise ValueError('durable evidence missing/symlink')
        before={p.name:code_hash(p.read_text()) for p in paths}
        loaded=ArtifactStore(str(directory)).load(str(directory))
        raw=json.loads((directory/'result.json').read_text())
        snap,binding=load_checkpoint(directory/'execution_snapshot.json',task,input_sets,controls)
        scores=[]
        for code,results in rounds:
            l3=next(r for r in results if r.level is Level.L3_PERFORMANCE)
            point=reading(l3.metrics,contract)
            if not point['can_select'] or point['score'] is None:
                raise ValueError('durable winner lacks qualified raw workload evidence')
            scores.append(point['score'])
            sidecar_errors=wd.binding_errors(SimpleNamespace(saved_dir=str(directory),code=code),
                l3.metrics,contract,dict(plan,candidate=dict(hash=code_hash(code))))
            if sidecar_errors:raise ValueError('; '.join(sidecar_errors))
        if len(scores)!=2:raise ValueError('durable audit requires exactly two frozen observations')
        winner=max(range(len(scores)),key=lambda i:scores[i])
        source,results=rounds[winner]
        expected_selection=dict(basis='objective',selected_round=winner+1,
            fallback_to_best=winner!=len(rounds)-1,evidence_scope='current_process_selection_only')
        search=raw.get('search_objective');selection=raw.get('candidate_selection')
        if (search is None or search!=snap['search_objective']
                or selection!=expected_selection
                or search!=json_value(audit_value(artifact.metadata['search_objective']))
                or selection!=artifact.metadata['candidate_selection']
                or loaded.metadata.get('search_objective')!=search
                or loaded.metadata.get('candidate_selection')!=selection):
            raise ValueError('durable search identity/selection missing or changed')
        validate_snapshot(search,contract,snap['round_done'])
        best=search['best']
        if (best['round']!=winner+1 or best['score']!=scores[winner]
                or loaded.code!=source or artifact.code!=source or best['code']!=source
                or loaded.sketch.to_dict()!=best['sketch']
                or loaded.sketch.to_dict()!=artifact.sketch.to_dict()
                or raw.get('iteration_history')!=snap['iteration_history']
                or loaded.iteration_history!=json_value(audit_value(artifact.iteration_history))):
            raise ValueError('durable winner source/sketch/score/history drift')
        # Only known derived publication/case annotations may differ. All raw
        # timings, receipts, windows and bindings must belong to the selected
        # candidate -- swapping the last probe's results is NOT a roundtrip.
        derived={'claim_level','objective','implementation_decision',
                 'case_key','case_source','case_declared','case_match'}
        def core(rows):
            return json_value(audit_value([dict(level=r.level.value,passed=r.passed,message=r.message,
                metrics={k:v for k,v in r.metrics.items() if k not in derived}) for r in rows]))
        if core(loaded.verification_results)!=core(results):
            raise ValueError('durable selected verification not its own raw evidence')
        selected_l3=next(r for r in loaded.verification_results if r.level is Level.L3_PERFORMANCE)
        if (loaded.benchmark_data!=json_value(audit_value(artifact.benchmark_data))
                or reading(loaded.benchmark_data,contract)['score']!=scores[winner]
                or core([SimpleNamespace(level=Level.L3_PERFORMANCE,passed=selected_l3.passed,
                    message=selected_l3.message,metrics=loaded.benchmark_data)])!=core([selected_l3])):
            raise ValueError('durable selected benchmark drift')
        if (snap['current_code']!=rounds[-1][0] or snap['round_done']!=len(rounds)
                or snap['best_gflops']!=0. or loaded.all_passed or raw.get('all_passed') is not False
                or loaded.metadata.get('stop')!=artifact.metadata['stop']
                or loaded.metadata['stop']['reason']!='budget_exhausted'
                or (loaded.metadata.get('corpus_admission') or {}).get('kept') is not False
                or loaded.benchmark_data.get('implementation_decision',{}).get('decision')!='inconclusive'
                or any((directory/name).exists() for name in ('confirmation.json','admission.json'))):
            raise ValueError('durable last probe/stop/non-admission drift')
        report=(directory/'report.md').read_text()
        regenerated=build_report(artifact=loaded,task=task,generated_at='audit-only')
        start='### 声明 workload 的搜索选优';end='## 2. 证据链 (逐轮)'
        section=regenerated.split(start,1)[1].split(end,1)[0]
        chain=regenerated.split(end,1)[1].split('## 3.',1)[0]
        if start+section not in report or end+chain not in report:
            raise ValueError('durable report disagrees with loaded selection/history')
        after={p.name:code_hash(p.read_text()) for p in paths}
        if after!=before:raise ValueError('durable audit mutated or observed changing evidence')
        proof.update(completed=True,selected_round=winner+1,candidate_hash=code_hash(source),
            score=scores[winner],checkpoint_verified=True,
            protocol_hash=digest(binding['manifest']['measurement_protocol']),file_hashes=before)
    except (OSError,ValueError,KeyError,TypeError,AttributeError,IndexError,StopIteration) as exc:
        errors.append('durable search audit: '+type(exc).__name__+': '+str(exc))
    return proof


def audit(artifact, rounds, trace, plan, generated, corpus):
    """Independently reconstruct the ranking/reward from retained Oracle rows."""
    errors=[]
    from operator_agent.core.contract import load_dict
    contract=load_dict(plan['contract'])
    supplied=plan['supplied_cases'];expected_codes=[c['source'] for c in plan['candidates']]
    if [code for code,_ in rounds]!=expected_codes or generated!=expected_codes:
        errors.append('did not evaluate exactly the two frozen candidates in order')
    metrics=[];scores=[]
    for i,(code,results) in enumerate(rounds):
        levels={r.level.value:r.passed for r in results}
        l3=next((r for r in results if r.level is Level.L3_PERFORMANCE),None)
        if levels.get('L1') is not True or levels.get('L2') is not True or not l3:
            errors.append('candidate missing complete real L1/L2/L3: '+str(i));continue
        m=l3.metrics;metrics.append(m)
        errors+=wd.assess(SimpleNamespace(verification_results=results), 'passed',
            wd.as_suite(contract.correctness_suite), [r for r in trace if r['request'].get('code')==code], False)
        errors+=wd.metrics_errors(m,contract,supplied)
        errors+=wd.trace_metrics_errors([m],[r for r in trace if r['request'].get('code')==code],
                                       supplied,code,plan['controls']['gpu'])
        point=reading(m,contract);scores.append(point['score'])
        if not point['can_select'] or point['score'] is None or l3.passed:
            errors.append('candidate not a valid below-target full-workload search observation: '+str(i))
        for result in results:
            if result.metrics.get('run_binding',{}).get('candidate_hash')!=code_hash(code):
                errors.append('candidate result binding drift')
    errors+=wd.receipt_errors(registered_reference(),trace)+wd.transfer_errors(trace)
    errors+=wd.ruler_errors(metrics,trace,plan['controls']['cuda_calibration'],plan['controls']['cuda_ruler_profile'],plan['controls']['gpu'])
    windows=[m['cases'][k]['arm_windows']['candidate'] for m in metrics for k in supplied]
    windows += [r['response']['paired']['arm_windows']['candidate'] for r in trace if r['request']['action']=='cuda_calibration_probe']
    if len(windows)!=len(set(windows)):errors.append('reused candidate/case/heartbeat windows')
    if len(scores)==2 and all(s is not None for s in scores):
        winner=max(range(2),key=lambda i:scores[i]) # Stable tie: first.
        selected=artifact.metadata['candidate_selection']['selected_round']
        if selected!=winner+1 or artifact.code!=expected_codes[winner]:errors.append('did not return objective winner')
        history=artifact.iteration_history
        p1,p2=[r['search_objective'] for r in history]
        if p1['best_before'] is not None or p1['reward'] is not None:
            errors.append('first sample was not a fresh search baseline')
        if (p2['best_before']!=scores[0] or p2['reward']!=(scores[1]-scores[0])/scores[0]
                or p2['score']!=scores[1]):errors.append('reward compared with wrong objective/incumbent')
        arms=artifact.metadata['bandit']['arms']
        if artifact.metadata['bandit']['total_scored']!=1 or sum(a['pulls'] for a in arms.values())!=2:
            errors.append('bandit skipped or duplicated a frozen candidate observation')
    snap=json.loads((Path(artifact.saved_dir)/'execution_snapshot.json').read_text())
    validate_snapshot(snap['search_objective'],contract,snap['round_done'])
    if (artifact.all_passed or artifact.remeasure or artifact.metadata['stop']['reason']!='budget_exhausted'
            or artifact.metadata['budget']['spent']['confirm']!=0
            or (corpus.exists() and corpus.read_bytes())):
        errors.append('search exit not bounded non-admitting budget stop')
    if artifact.benchmark_data.get('implementation_decision',{}).get('decision')!='inconclusive':
        errors.append('below-target search winner granted deployment claim')
    return dict(errors=errors,objective_scores=scores,case_and_heartbeat_windows=windows,
                selected_round=artifact.metadata['candidate_selection']['selected_round'],
                stop=artifact.metadata['stop'],budget=artifact.metadata['budget'],
                decision=artifact.benchmark_data.get('implementation_decision'),bandit=artifact.metadata['bandit'])


def run(args):
    if args.execute and (args.gpu is None or not args.cuda_ruler_profile):
        raise ValueError('--execute requires authorized --gpu AND --cuda-ruler-profile')
    t,plan=declaration(args)
    budget=BudgetPlan(**plan['controls']['budget'])  # Validate before any SSH/preparation.
    parent=Path(args.output).resolve();parent.mkdir(parents=True,exist_ok=True)
    root=Path(tempfile.mkdtemp(prefix='workload-search-',dir=parent))
    atomic_json(root/'plan.json',plan,readonly=True)
    (root/'driver.py').write_text(Path(__file__).read_text())
    print('PLAN '+str(root/'plan.json')+' HASH '+plan['hash'],flush=True)
    summary=dict(completed=False,mode='execute' if args.execute else 'plan-only',run_root=str(root),
                 plan_hash=plan['hash'],errors=[],cleanup=None)
    if not args.execute:
        atomic_json(root/'summary.json',summary);return 0
    backend=None;rounds=[];generated=[];art=None
    log_keys=('OPERATOR_AGENT_LOG_DIR','OPERATOR_AGENT_LOG_KEEP')
    previous={k:os.environ.get(k) for k in log_keys}
    os.environ.update(OPERATOR_AGENT_LOG_DIR=str(root/'logs'),OPERATOR_AGENT_LOG_KEEP='10000')
    try:
        check_plan(root/'plan.json',plan,t)
        backend=wd.BoundedProfileSSH(host=args.host,timeout=wd.RPC_TIMEOUT)
        environment=backend.prepare(args.gpu);wd.validate_scratch(backend.scratch)
        if 'H20' not in environment.get('selected',{}).get('name',''):raise ValueError('not H20')
        reference=backend.reference_identity(registered_reference().content_dict())
        validate_receipt(registered_reference(),reference.get('reference_receipt'))
        atomic_json(root/'environment.json',dict(environment=environment,reference=reference))
        xs=[wd.primary(s) for s in wd.LARGE]
        if [case_key(x) for x in xs]!=plan['supplied_cases']:raise ValueError('actual inputs differ from frozen plan')
        corpus=root/'corpus'/'kernel_db.jsonl'
        c=Conductor(backend=backend,kernel_db=KernelDatabase(str(corpus),persist_retrievals=False),
            failure_db=FailureDatabase(str(root/'failures'/'failure_db.jsonl')),use_coder_pipeline=False,
            max_optimize_rounds=3,heartbeat_interval=1,budget=budget,
            cuda_ruler_profile_path=args.cuda_ruler_profile,
            cuda_calibration_identity_hash=plan['controls']['cuda_calibration']['identity_hash'])
        c.store.output_dir=root/'artifacts'
        c.designer.run=lambda *a,**k: UnifiedSketch(algorithm='layernorm',target_dsl='triton')
        def generate(*a,**k):
            check_plan(root/'plan.json',plan,t)
            if len(generated)>=2:raise ValueError('attempted third candidate')
            code=plan['candidates'][len(generated)]['source'];generated.append(code);return code
        c.coder=SimpleNamespace(run=generate)
        c.search=SimpleNamespace(run=lambda sketch,perf: (
            UnifiedSketch(algorithm='layernorm',target_dsl='triton',version=sketch.version+1,
                          tiling=TilingStrategy(128,64,32)),'fixed next candidate, not LLM search'))
        verify=c.verifier.verify
        def capture(*a,**k):
            check_plan(root/'plan.json',plan,t)
            check_candidate(plan,rounds,k['kernel_code'])
            results=verify(*a,**k);rounds.append((k['kernel_code'],results));return results
        c.verifier.verify=capture
        art=c.execute(t,xs,target_gflops=0.)
        check_plan(root/'plan.json',plan,t)
        proof=audit(art,rounds,backend.trace,plan,generated,corpus)
        durable=audit_durable(art,rounds,plan,task=t,input_sets=xs,controls=c._run_controls)
        summary.update(proof=proof,durable=durable,saved_dir=art.saved_dir)
        summary['errors']+=proof['errors']+durable['errors']
    except (Exception,KeyboardInterrupt) as exc:
        summary['errors'].append(type(exc).__name__+': '+str(exc))
    finally:
        if backend:
            try:summary['cleanup']=backend.cleanup()
            except (Exception,KeyboardInterrupt) as exc:summary['cleanup']=dict(quiescent=False,removed=False,error=str(exc))
            summary['errors']+=wd.cleanup_errors(summary['cleanup'],backend)
            summary['forbidden_actions']=backend.forbidden_actions
            atomic_json(root/'evidence.json',audit_value(dict(rounds=[dict(code=code,results=results_dict(rs)) for code,rs in rounds],
                trace=wd.compact_trace(backend.trace),work_counts=getattr(backend,'work_counts',{}),generated=generated)))
        for key,old in previous.items():
            if old is None:os.environ.pop(key,None)
            else:os.environ[key]=old
    summary['completed']=bool(not summary['errors'] and summary.get('durable',{}).get('completed')
        and summary['cleanup'] and summary['cleanup']['quiescent'] and summary['cleanup']['removed'] and len(rounds)==2)
    atomic_json(root/'summary.json',audit_value(summary));print(wd.dumps(summary),flush=True)
    return 0 if summary['completed'] else 1


if __name__=='__main__':raise SystemExit(run(parser().parse_args()))
