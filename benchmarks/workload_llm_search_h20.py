"""Two real, constrained LLM launch proposals; plan-only by default.

NOT free-form code generation: the model proposes num_warps and a rationale;
trusted LayerNorm source is materialized without executing model-authored code.
No model/GPU retries, fallback, repair, window picking or corpus admission.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks import workload_search_h20 as fixed
from benchmarks import workload_decision_h20 as wd
from benchmarks._bounded_proposals import BoundedModel, LIMITS, model_config, model_identity
from operator_agent import Conductor, KernelDatabase
from operator_agent.budget import BudgetPlan
from operator_agent.contract import case_key, freeze_from_task
from operator_agent.kernel_db import FailureDatabase
from operator_agent.reference_contract import registered_reference, validate_receipt
from operator_agent.run_contract import atomic_json, audit_value, code_hash, digest, json_value, protocol_identity
from operator_agent.schemas import UnifiedSketch, TilingStrategy, VerificationLevel as Level
from operator_agent.search_objective import reading, results_dict

WARPS = (4, 8, 16)
PROMPT = ('You propose a bounded NVIDIA H20 Triton LayerNorm launch configuration. '
    'Return ONLY a JSON object with exactly num_warps (integer) and rationale '
    '(nonempty string, at most 1000 characters). Choose one unused allowed_num_warps. '
    'Reference, precision semantics, shapes, weights, safety gates, and threshold '
    'are immutable. The 10x threshold forces search; this is not deployment acceptance. '
    'No source code, shell commands, new dependencies, precision changes or retries. '
    'Only qualified full-workload evidence may inform the second proposal; '
    'primary-case GFLOPS is not the ranking objective.')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--host', default='algorithm-h20')
    p.add_argument('--gpu', type=int, choices=range(8))
    p.add_argument('--output', default=str(ROOT/'output'/'workload_llm_search_h20'))
    p.add_argument('--cuda-ruler-profile')
    p.set_defaults(cuda_calibration=None)
    return p


def source(warps):
    if type(warps) is not int or warps not in WARPS:
        raise ValueError('undeclared launch parameter')
    template = wd.trial_code('unfused-fp32')
    marker = 'enable_fp_fusion=False'
    if template.count(marker) != 1:
        raise ValueError('trusted template marker drift')
    return template.replace(marker, marker + ',num_warps=' + str(warps))


def sources():
    files = wd.evaluator_sources()
    for name in ('workload_search_h20.py', 'workload_llm_search_h20.py', '_bounded_proposals.py'):
        files['benchmarks/'+name] = code_hash((ROOT/'benchmarks'/name).read_text())
    return files


def declaration(args, identity=None):
    task = wd.task('workload-llm-search-h20')
    task.constraints.min_speedup = 10.
    profile, calibration = wd.ruler(args)
    body = dict(schema='workload-llm-launch-search-v1', contract=json_value(freeze_from_task(task).as_dict()),
        supplied_cases=['X=8192x15360','X=8192x12288'], measurement_protocol=protocol_identity(),
        evaluator_sources=sources(), model=identity, model_limits=dict(LIMITS),
        proposal_policy=dict(allowed_num_warps=list(WARPS), max_candidates=2, max_repairs=0,
            source_author='trusted_template', template_hash=code_hash(wd.trial_code('unfused-fp32')),
            prompt=PROMPT, prompt_hash=code_hash(PROMPT), duplicate_candidates=False),
        controls=dict(host=args.host,gpu=args.gpu,max_rounds=3,heartbeat_interval=1,
            wall_clock_s=1800,max_rpcs=wd.MAX_RPCS,max_commands=wd.MAX_COMMANDS,
            rpc_timeout=wd.RPC_TIMEOUT,cuda_ruler_profile=profile,cuda_calibration=calibration,
            budget=dict(oracle_total=5,wall_clock_s=1800,llm_tokens=40000)),
        input_storage_limit=wd.MAX_STORAGE_BYTES, retry_windows=False,
        performance_scope='real constrained LLM proposals, not free-form generation or deployment acceptance')
    return task, json_value(dict(body, hash=digest(body)))


def check_plan(path, plan, task):
    if json.loads(Path(path).read_text()) != plan or plan['hash'] != digest({k:v for k,v in plan.items() if k!='hash'}):
        raise ValueError('frozen LLM search plan drift')
    if sources() != plan['evaluator_sources'] or protocol_identity() != plan['measurement_protocol']:
        raise ValueError('frozen LLM evaluator drift')
    if json_value(freeze_from_task(task).as_dict()) != plan['contract']:
        raise ValueError('frozen LLM contract drift')


def parse_proposal(proposal, used):
    if (not isinstance(proposal, dict) or set(proposal) != {'num_warps','rationale'}
            or type(proposal['num_warps']) is not int or proposal['num_warps'] not in WARPS
            or proposal['num_warps'] in used or not isinstance(proposal['rationale'], str)
            or not proposal['rationale'].strip() or len(proposal['rationale']) > 1000):
        raise ValueError('invalid/duplicate/unsafe model proposal; no retry')
    return proposal['num_warps']


def feedback(rounds, contract):
    """Derive second prompt from its OWN complete raw workload, never GFLOPS."""
    if len(rounds) != 1:
        raise ValueError('second proposal requires exactly one fresh observation')
    code, results = rounds[0]
    levels = {r.level.value:r.passed for r in results}
    l3 = next((r for r in results if r.level is Level.L3_PERFORMANCE), None)
    if levels.get('L1') is not True or levels.get('L2') is not True or l3 is None:
        raise ValueError('no correct first candidate for model feedback')
    for r in results:
        binding=r.metrics.get('run_binding',{})
        if binding.get('candidate_hash') != code_hash(code) or binding.get('contract_hash') != contract.hash:
            raise ValueError('model feedback candidate binding drift')
    point = reading(l3.metrics, contract)
    if not point['can_select'] or point['score'] is None:
        raise ValueError('unqualified workload cannot drive launch tuning')
    return dict(candidate_hash=code_hash(code), objective=point['score'],
        identity=point['identity'], cases={k:dict(speedup=v['speedup_vs_baseline'],
            validity=v['measurement_validity'], candidate_window=v['arm_windows']['candidate'])
            for k,v in l3.metrics['cases'].items()})


class ProposalSession:
    def __init__(self, model, root, plan, task, rounds):
        self.model=model; self.root=Path(root); self.plan=plan; self.task=task; self.rounds=rounds
        self.proposals=[]; self.generated=[]; self.dispatched=[]; self.repair_refusals=0
        self.terminal_search_skips=0

    def propose(self, stage, previous=None):
        check_plan(self.root/'plan.json', self.plan, self.task)
        if len(self.proposals)>=2 or len(self.proposals)!=len(self.rounds):
            raise ValueError('candidate proposal bound/order before model dispatch')
        data=dict(contract=self.plan['contract'], plan_hash=self.plan['hash'],
            allowed_num_warps=[w for w in WARPS if w not in [p['num_warps'] for p in self.proposals]],
            used_proposals=self.proposals, feedback=previous,
            template=self.plan['proposal_policy']['template_hash'])
        messages=[dict(role='system', content=PROMPT),dict(role='user', content=json.dumps(data,ensure_ascii=False))]
        proposal=self.model.ask(messages,stage)
        warps=parse_proposal(proposal,[p['num_warps'] for p in self.proposals])
        item=dict(proposal,call=len(self.model.records),source=source(warps),source_hash=code_hash(source(warps)),
            feedback=previous,feedback_hash=digest(previous))
        self.proposals.append(item)
        atomic_json(self.root/('proposal-%02d.json'%len(self.proposals)),item,readonly=True)
        return UnifiedSketch(algorithm='layernorm',target_dsl='triton',version=len(self.proposals),
            tiling=TilingStrategy(64 if warps==4 else 128 if warps==8 else 256,64,32),
            optimization_hints=['real LLM num_warps='+str(warps)+'; '+proposal['rationale']])

    def design(self,*args,**kwargs):
        return self.propose('InitialProposal')

    def search(self,sketch,perf):
        check_plan(self.root/'plan.json',self.plan,self.task)
        # Conductor asks Search at the END of a measured round, before the
        # next round's Oracle-budget gate. After proposal #2 there is no new
        # proposal/candidate: return the same sketch only to reach that gate.
        # generate()/before_verify() still refuse any third/repeated source.
        if len(self.proposals)==2 and len(self.rounds)==2:
            self.terminal_search_skips+=1
            return sketch, 'proposal cap reached; no model call or new candidate'
        from operator_agent.contract import load_dict
        previous=feedback(self.rounds,load_dict(self.plan['contract']))
        return self.propose('SearchProposal',previous), 'real bounded LLM launch proposal'

    def generate(self,*args,**kwargs):
        check_plan(self.root/'plan.json',self.plan,self.task)
        if len(self.generated)>=2 or len(self.generated)+1!=len(self.proposals):
            raise ValueError('unproposed/repeated candidate generation')
        code=self.proposals[len(self.generated)]['source']
        self.generated.append(code)
        return code

    def before_verify(self,code):
        check_plan(self.root/'plan.json',self.plan,self.task)
        i=len(self.dispatched)
        if i>=2 or i>=len(self.generated) or code!=self.generated[i] or code_hash(code)!=self.proposals[i]['source_hash']:
            raise ValueError('unproposed/repeated/repaired candidate before verification')
        # Count dispatch even if Verifier raises: a failure cannot buy a retry.
        self.dispatched.append(code_hash(code))

    def refuse_repair(self,*args,**kwargs):
        self.repair_refusals+=1
        raise ValueError('frozen repair budget is zero')


def audit_model(session, root):
    errors=[]
    try:
        paths=[Path(root)/(kind+'-%02d.json'%i) for i in (1,2) for kind in ('model-call','proposal')]
        if any(p.is_symlink() or not p.is_file() for p in paths):
            raise ValueError('missing/symlink model evidence')
        before={p.name:code_hash(p.read_text()) for p in paths}
        if len(session.model.records)!=2 or len(session.proposals)!=2 or len(session.generated)!=2 or len(session.rounds)!=2:
            raise ValueError('requires two real, distinct proposals and fresh observations')
        used=[]
        for i,(record,proposal) in enumerate(zip(session.model.records,session.proposals)):
            retained=json.loads((Path(root)/('model-call-%02d.json'%(i+1))).read_text())
            declared=json.loads((Path(root)/('proposal-%02d.json'%(i+1))).read_text())
            if retained!=record or declared!=proposal or record['status']!='answered':
                raise ValueError('retained model/proposal evidence drift')
            warps=parse_proposal(record['proposal'],used); used.append(warps)
            previous=None if i==0 else feedback(session.rounds[:1],freeze_from_task(session.task))
            body=record['request']; query=json.loads(body['messages'][1]['content'])
            response=json.loads(record['response_text'])
            content=response['choices'][0]
            if (proposal['call']!=i+1 or proposal['source']!=source(warps)
                    or session.rounds[i][0]!=proposal['source'] or proposal['source_hash']!=code_hash(proposal['source'])
                    or proposal['feedback']!=previous or proposal['feedback_hash']!=digest(previous)
                    or query['feedback']!=previous or query['contract']!=session.plan['contract']
                    or query['plan_hash']!=session.plan['hash'] or body['messages'][0]['content']!=PROMPT
                    or record['request_hash']!=digest(body) or record['identity']!=session.plan['model']
                    or query['allowed_num_warps']!=[w for w in WARPS if w not in used[:-1]]
                    or query['used_proposals']!=session.proposals[:i]
                    or body['max_tokens']!=LIMITS['max_output_tokens'] or body['temperature']!=LIMITS['temperature']
                    or body['stream'] is not False or body['chat_template_kwargs']!={'enable_thinking':False}
                    or record['output_tokens_reserved']!=LIMITS['max_output_tokens']
                    or json.loads(content['message']['content'])!=record['proposal']
                    or content.get('finish_reason')!='stop' or response.get('model')!=record['reported_model']
                    or response.get('usage')!=record['usage']
                    or record['stage']!=('InitialProposal' if i==0 else 'SearchProposal')
                    or record['response_hash']!=code_hash(record['response_text'])
                    or record['reported_model']!=session.plan['model']['model']):
                raise ValueError('real request/response/feedback/source lineage mismatch')
        if session.dispatched!=[p['source_hash'] for p in session.proposals] or session.repair_refusals:
            raise ValueError('unexpected candidate dispatch/repair')
        if before!={p.name:code_hash(p.read_text()) for p in paths}:
            raise ValueError('model audit observed changing evidence')
    except (OSError,ValueError,KeyError,TypeError,IndexError) as exc:
        errors.append('model audit: '+str(exc))
    return dict(errors=errors,completed=not errors,attempts=len(session.model.records),
        candidates=len(session.proposals),repair_refusals=session.repair_refusals,
        terminal_search_skips=session.terminal_search_skips,
        file_hashes=before if not errors else {},read_only=True,
        output_token_reservations=sum(r['output_tokens_reserved'] for r in session.model.records),
        usage=session.model.cost_tracker.summary(),cost_scope='repository estimate, not provider bill')


def audit_input(plan, proposals):
    # This is not a rehashed execution plan. Never retain the original `hash`
    # under a modified body and risk presenting generated sources as predeclared.
    value={k:copy.deepcopy(v) for k,v in plan.items() if k!='hash'}
    value.update(schema='workload-llm-search-audit-input-v1',frozen_plan_hash=plan['hash'],
        candidates=[dict(source=p['source'],hash=p['source_hash']) for p in proposals])
    return value


def run(args):
    if args.execute and (args.gpu is None or not args.cuda_ruler_profile):
        raise ValueError('--execute requires authorized --gpu AND --cuda-ruler-profile')
    config=model_config(ROOT/'config.yaml') if args.execute else None
    task,plan=declaration(args,model_identity(config) if config else None)
    budget=BudgetPlan(**plan['controls']['budget'])
    parent=Path(args.output).resolve();parent.mkdir(parents=True,exist_ok=True)
    root=Path(tempfile.mkdtemp(prefix='workload-llm-search-',dir=parent))
    atomic_json(root/'plan.json',plan,readonly=True)
    # Exact source snapshots supplement hashes; do not auto-commit user changes.
    for name in plan['evaluator_sources']:
        p=root/'frozen-sources'/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes((ROOT/name).read_bytes());p.chmod(0o444)
    print('PLAN '+str(root/'plan.json')+' HASH '+plan['hash'],flush=True)
    summary=dict(completed=False,mode='execute' if args.execute else 'plan-only',run_root=str(root),
        plan_hash=plan['hash'],errors=[],cleanup=None,scope=plan['performance_scope'])
    if not args.execute:
        atomic_json(root/'summary.json',summary);return 0
    backend=None;session=None;rounds=[];art=None
    previous={k:os.environ.get(k) for k in ('OPERATOR_AGENT_LOG_DIR','OPERATOR_AGENT_LOG_KEEP')}
    os.environ.update(OPERATOR_AGENT_LOG_DIR=str(root/'logs'),OPERATOR_AGENT_LOG_KEEP='10000')
    try:
        check_plan(root/'plan.json',plan,task)
        model=BoundedModel(config,root,plan['model'])
        session=ProposalSession(model,root,plan,task,rounds)
        backend=wd.BoundedProfileSSH(host=args.host,timeout=wd.RPC_TIMEOUT)
        environment=backend.prepare(args.gpu);wd.validate_scratch(backend.scratch)
        if 'H20' not in environment.get('selected',{}).get('name',''):raise ValueError('not H20')
        reference=backend.reference_identity(registered_reference().content_dict())
        validate_receipt(registered_reference(),reference.get('reference_receipt'))
        atomic_json(root/'environment.json',dict(environment=environment,reference=reference))
        xs=[wd.primary(s) for s in wd.LARGE]
        if [case_key(x) for x in xs]!=plan['supplied_cases']:raise ValueError('actual input drift')
        corpus=root/'corpus'/'kernel_db.jsonl'
        c=Conductor(backend=backend,llm=model,kernel_db=KernelDatabase(str(corpus),persist_retrievals=False),
            failure_db=FailureDatabase(str(root/'failures'/'failure_db.jsonl')),use_coder_pipeline=False,
            max_optimize_rounds=3,heartbeat_interval=1,budget=budget,
            cuda_ruler_profile_path=args.cuda_ruler_profile,
            cuda_calibration_identity_hash=plan['controls']['cuda_calibration']['identity_hash'])
        c.store.output_dir=root/'artifacts'
        c.designer=SimpleNamespace(run=session.design)
        c.coder=SimpleNamespace(run=session.generate)
        c.search=SimpleNamespace(run=session.search)
        c.fixer=SimpleNamespace(run_with_action=session.refuse_repair)
        verify=c.verifier.verify
        def capture(*a,**k):
            session.before_verify(k['kernel_code'])
            results=verify(*a,**k);rounds.append((k['kernel_code'],results));return results
        c.verifier.verify=capture
        art=c.execute(task,xs,target_gflops=0.)
        check_plan(root/'plan.json',plan,task)
        model_proof=audit_model(session,root)
        summary.update(model=model_proof,saved_dir=art.saved_dir)
        summary['errors']+=model_proof['errors']
        if (art.metadata.get('llm_cost',{}).get('total_calls')!=len(model.records)
                or json.loads((Path(art.saved_dir)/'result.json').read_text()).get('llm_cost')!=json_value(art.metadata.get('llm_cost'))):
            summary['errors'].append('durable model usage missing or changed')
        if model_proof['completed']:
            # A declared AUDIT INPUT, not a replacement frozen plan. Candidate
            # sources are proved from retained real responses above. Reuse the
            # unchanged raw-workload and durable-product auditors for 2 VALID
            # observations; partial/failed/degraded campaigns remain not accepted.
            projection=audit_input(plan,session.proposals)
            atomic_json(root/'audit-input.json',projection,readonly=True)
            proof=fixed.audit(art,rounds,backend.trace,projection,session.generated,corpus)
            durable=fixed.audit_durable(art,rounds,plan,task=task,input_sets=xs,controls=c._run_controls)
            summary.update(proof=proof,durable=durable)
            summary['errors']+=proof['errors']+durable['errors']
    except (Exception,KeyboardInterrupt) as exc:
        summary['errors'].append(type(exc).__name__+': '+str(exc))
    finally:
        if backend:
            try:summary['cleanup']=backend.cleanup()
            except (Exception,KeyboardInterrupt):summary['cleanup']=dict(quiescent=False,removed=False,error='owned cleanup failed')
            summary['errors']+=wd.cleanup_errors(summary['cleanup'],backend)
            summary['forbidden_actions']=backend.forbidden_actions
            atomic_json(root/'evidence.json',audit_value(dict(rounds=[dict(code=code,results=results_dict(rs)) for code,rs in rounds],
                trace=wd.compact_trace(backend.trace),work_counts=getattr(backend,'work_counts',{}),
                proposals=session.proposals if session else [],dispatched=session.dispatched if session else [])))
        if session and 'model' not in summary:
            summary['model']=audit_model(session,root)
        for key,old in previous.items():
            if old is None:os.environ.pop(key,None)
            else:os.environ[key]=old
    summary['completed']=bool(not summary['errors'] and summary.get('model',{}).get('completed')
        and summary.get('durable',{}).get('completed') and summary['cleanup']
        and summary['cleanup']['quiescent'] and summary['cleanup']['removed'])
    atomic_json(root/'summary.json',audit_value(summary));print(wd.dumps(summary),flush=True)
    return 0 if summary['completed'] else 1


if __name__=='__main__':raise SystemExit(run(parser().parse_args()))
