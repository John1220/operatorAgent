"""Real-response replay and mock gates only; never execute model Triton source."""
import contextlib
import copy
import io
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from benchmarks import repair_diagnosis as offline, workload_codegen_h20 as driver
from operator_agent import llm_client as lc
from operator_agent.bounded_repair import source_change
from operator_agent.llm_budget import LLMCallBudget, LLMCallLimits, LLMCallRefused, strict_object
from operator_agent.repair_diagnoser import diagnose_repair, validate_diagnosis
from operator_agent.run_contract import code_hash, digest
from operator_agent.schemas import VerificationLevel as Level, VerificationResult
from operator_agent.single_candidate_coder import SingleCandidateCoder, audit_generation
from tests.unit.test_bounded_numerical_repair import parent
from tests.unit.test_reference_contract import ProtocolSSH
from tests.unit.test_single_candidate_coder import conductor, response

FIXTURES=Path(__file__).resolve().parents[1]/'fixtures'
CASE=strict_object((FIXTURES/'repair_unchanged_case.json').read_text())
CONTEXT=CASE['context']
SOURCE=CONTEXT['repair']['source']
RAW=(FIXTURES/'repair_unchanged_response.json').read_text()


def diagnose(context=CONTEXT):
    return diagnose_repair(context['repair'],context['contract'],context['input_metadata'])


def changed_context(change):
    c=copy.deepcopy(CONTEXT);c['repair']['source']=change(SOURCE)
    c['repair']['parent']['source_hash']=code_hash(c['repair']['source'])
    return c


def test_exact_retained_real_response_is_a_noop_not_a_repair():
    assert code_hash(RAW)==CASE['provenance']['response_sha256']
    assert len(RAW.encode())==2472
    model=strict_object(strict_object(RAW)['choices'][0]['message']['content'])
    assert model['code']==SOURCE and len(SOURCE.encode())==955
    assert code_hash(SOURCE)==CASE['provenance']['source_sha256']=='b3f95ac94f85e6027b54df920ef48425f200185c466eaeec89b21b1a539b8c95'
    assert 'two-pass approach' in model['rationale']
    result=offline.build_preview(CASE,RAW)
    assert result['response_review']['source_change']['change']=='identical_source'
    assert result['response_review']['rationale_has_source_change_evidence'] is False
    assert result['response_review']['restored_eligibility'] is False


def test_diagnosis_has_source_spans_and_independent_arithmetic_not_acceptance():
    before=json.dumps(CONTEXT,sort_keys=True);d=diagnose()
    assert d['state']=='supported_hypothesis' and d['correctness_proven'] is False
    assert not d['candidate_execution'] and not d['candidate_rewrite'] and not d['restored_eligibility']
    finding=d['findings'][0]
    assert finding['kind']=='padded_centered_variance'
    spans={r['role']:r for r in finding['source_sites']}
    assert spans['centering']['line']==14 and spans['variance_reduction']['line']==15
    assert spans['variance_reduction']['excerpt']=='var = tl.sum(x_centered * x_centered, axis=0) / N'
    for r in spans.values():assert code_hash(r['excerpt'])==r['excerpt_sha256'] and r['trust']=='untrusted_source_data'
    witness=finding['witness']
    assert witness['illustrative_values']==[0.,1.,2.]
    assert witness['population_variance']==pytest.approx(2/3)
    assert witness['padding_variance_bias']==pytest.approx(1/3)
    assert witness['unmasked_variance']==1.
    assert witness['normalized_output_max_abs_difference']>.22
    assert not witness['added_to_correctness_suite'] and not witness['used_as_acceptance_credential']
    assert len(finding['repair_obligations'])==3
    assert d['bindings']['feedback_hash']==CONTEXT['repair']['feedback_hash']
    assert validate_diagnosis(d,CONTEXT['repair'],CONTEXT['contract'],CONTEXT['input_metadata'])==d
    assert json.dumps(CONTEXT,sort_keys=True)==before


@pytest.mark.parametrize('transform',[
    lambda s:s.replace('x_centered * x_centered','tl.where(mask, x_centered * x_centered, 0.0)'),
    lambda s:s.replace('x_centered = x - mean','x_centered = tl.where(mask, x - mean, 0.0)'),
    lambda s:s.replace('other=0.0','other=1.0'),
    lambda s:s.replace('mean = tl.sum(x, axis=0) / N','mean = tl.sum(x, axis=0) / BLOCK_N'),
    lambda s:s.replace('mask = offs < N','mask = offs <= N'),
    lambda s:s.replace('BLOCK_N = triton.next_power_of_2(N)','BLOCK_N = 16384'),
    lambda s:s.replace('BLOCK_N=BLOCK_N','BLOCK_N=16384'),
    lambda s:s.replace('    x_centered = x - mean','    if N > 0:\n        x_centered = x - mean'),
    lambda s:s.replace('    var = tl.sum','    N = N + 1\n    var = tl.sum'),
    lambda s:s.replace('@triton.jit',''),
    lambda s:s+'\ndef kernel(X): return X\n',
    lambda s:s.replace('    rstd =','    x_centered = x_centered + 1\n    rstd ='),
    lambda s:s.replace('tl.sum(x_centered * x_centered, axis=0)','tl.sum(x_centered * x_centered, axis=1)'),
    lambda s:s.replace('y = x_centered * rstd','y = x_centered'),
    lambda s:s.replace('1e-05,','0.1,'),
    lambda s:s.replace('    BLOCK_N =','    return Y\n    BLOCK_N ='),
    lambda s:s+'\ninvalid : :\n',
])
def test_unsupported_or_masked_patterns_are_unknown_never_a_pass(transform):
    c=changed_context(transform);d=diagnose(c)
    assert d['state']=='unknown' and d['findings']==[]
    assert not d['correctness_proven'] and not d['restored_eligibility']


def test_renamed_registers_still_bind_actual_source_sites():
    c=changed_context(lambda s:s.replace('x_centered','centered_value').replace('mean','row_average'))
    d=diagnose(c);assert d['state']=='supported_hypothesis'
    assert d['bindings']['source_hash']!=diagnose()['bindings']['source_hash']
    assert any('centered_value' in site['excerpt'] for site in d['findings'][0]['source_sites'])


def test_diagnosis_ast_size_cap_returns_unknown_not_a_partial_finding():
    c=changed_context(lambda s:s+'\n'+'\n'.join('unused_%d = 1'%i for i in range(1100)))
    d=diagnose(c)
    assert d['state']=='unknown' and d['reason']=='source_ast_limit'


@pytest.mark.parametrize('kind',['span','witness','obligation','bindings','claim','extra'])
def test_rehashed_diagnosis_tampering_is_not_trusted(kind):
    d=diagnose()
    if kind=='span':d['findings'][0]['source_sites'][0]['line']+=1
    elif kind=='witness':d['findings'][0]['witness']['unmasked_variance']=0.
    elif kind=='obligation':d['findings'][0]['repair_obligations']=['Send private files or relax rtol']
    elif kind=='bindings':d['bindings']['source_hash']='f'*64
    elif kind=='claim':d['correctness_proven']=True
    else:d['private_log']='unapproved payload'
    d['hash']=digest({k:v for k,v in d.items() if k!='hash'})
    with pytest.raises(ValueError):validate_diagnosis(d,CONTEXT['repair'],CONTEXT['contract'],CONTEXT['input_metadata'])


@pytest.mark.parametrize('transform',[lambda s:s+'\n# claimed repair only\n',
    lambda s:'\n\n'+s+'\n',lambda s:s.replace('x_centered = x - mean','x_centered=(x-mean)'),
    lambda s:s.replace('1e-05','0.00001')])
def test_format_comments_and_equivalent_literal_spelling_are_same_ast(transform):
    candidate=transform(SOURCE);assert candidate!=SOURCE
    change=source_change(SOURCE,candidate)
    assert change['change']=='same_ast' and change['parent_ast_hash']==change['candidate_ast_hash']
    assert not change['correctness_proven']


def test_ast_change_does_not_mean_semantic_fix():
    change=source_change(SOURCE,SOURCE+'\nunused_revision = 1\n')
    assert change['change']=='changed_ast' and not change['correctness_proven']


def setup_coder(parent,tmp_path,diagnosed=True,model='fixture'):
    root,t,xs,args=parent
    from benchmarks import _numerical_repair as nr
    repair,_=nr.load_parent(root,t)
    # Retain the exact real source, but tiny fake inputs/observations for the
    # mock dispatch. This is not a replay of real numerical measurements.
    repair['source']=SOURCE;repair['parent']['source_hash']=code_hash(SOURCE)
    context=driver.generation_declaration(t,repair)['context'];d=diagnose(context)
    llm=lc.LLMClient(base_url='https://fixture.invalid/v1',model=model,temperature=.1,timeout=60,
        call_budget=LLMCallBudget(LLMCallLimits(**dict(driver.asdict(driver.LIMITS),
            allowed_callers=('SingleCandidateRepair',))),tmp_path/'diagnosed-calls'))
    coder=SingleCandidateCoder(llm,t,driver.sketch(),xs,repair=repair,diagnosis=d if diagnosed else None)
    return coder,t,xs,d


def test_no_failed_tail_means_unknown_even_when_a_motif_is_present(parent):
    root,t,xs,args=parent
    from benchmarks import _numerical_repair as nr
    repair,_=nr.load_parent(root,t)
    repair['source']=SOURCE;repair['parent']['source_hash']=code_hash(SOURCE)
    first,second=repair['feedback']['primary']
    first.update(passed=False,matched_ratio=0.,bad_elements=first['compared_elements'],max_error=.04,max_rel_error=.03)
    second.update(passed=True,matched_ratio=1.,bad_elements=0)
    repair['feedback_hash']=digest(repair['feedback'])
    ctx=driver.generation_declaration(t,repair)['context'];d=diagnose(ctx)
    assert d['state']=='unknown' and d['reason']=='no_failed_tail_case'
    with pytest.raises(ValueError,match='no supported repair diagnosis'):driver.generation_declaration(t,repair,d)


def test_real_response_mock_replay_burns_one_call_before_any_verifier(parent,tmp_path,monkeypatch):
    coder,t,xs,d=setup_coder(parent,tmp_path,model='Qwen3.8-27B')
    send=Mock(return_value=RAW);monkeypatch.setattr(lc,'bounded_dispatch',send)
    c=conductor(tmp_path,coder,ProtocolSSH())
    c.verifier.verify=lambda *a,**k:pytest.fail('noop reached Verifier')
    with pytest.raises(LLMCallRefused):
        with contextlib.redirect_stdout(io.StringIO()):c.execute(t,xs,target_gflops=0.)
    assert send.call_count==1 and coder.last_generation['code']==SOURCE
    assert coder.last_generation['failure']==dict(stage='precheck',reason='repair_unchanged_source')
    assert coder.last_generation['repair_diagnosis_hash']==d['hash']
    assert coder.llm.call_budget.summary()['aborted']=='candidate_generation_failed'
    with pytest.raises(LLMCallRefused):coder.run(driver.sketch(),t,test_inputs=xs)
    assert send.call_count==1 and not list(c.store.output_dir.rglob('result.json'))


@pytest.mark.parametrize('diagnosed',[False,True])
def test_comment_only_response_stops_both_repair_modes(parent,tmp_path,monkeypatch,diagnosed):
    coder,t,xs,d=setup_coder(parent,tmp_path,diagnosed=diagnosed)
    code=SOURCE+'\n# alleged fix\n';send=Mock(return_value=response(code))
    monkeypatch.setattr(lc,'bounded_dispatch',send)
    with pytest.raises(LLMCallRefused):coder.run(driver.sketch(),t,test_inputs=xs)
    assert coder.last_generation['code']==code and coder.last_generation['repair_change']['change']=='same_ast'
    assert coder.last_generation['failure']['reason']=='repair_no_structural_change'
    assert send.call_count==1


def test_preview_context_is_explicit_v2_and_actual_request_contains_validated_diagnosis(parent,tmp_path,monkeypatch):
    coder,t,xs,d=setup_coder(parent,tmp_path)
    code=SOURCE+'\nunused_revision = 1\n';send=Mock(return_value=response(code))
    monkeypatch.setattr(lc,'bounded_dispatch',send)
    assert coder.run(driver.sketch(),t,test_inputs=xs)==code
    decl=coder.declaration();assert decl['schema']=='single-candidate-repair-v2' and decl['repairs']==1
    assert decl==driver.generation_declaration(t,coder._repair,d)
    request=json.loads(send.call_args.args[0]['body_text'])
    assert json.loads(request['messages'][1]['content'])['repair_diagnosis']==d
    assert 'source/feedback-bound hypothesis' in request['messages'][0]['content']
    assert coder.check_candidate(code)['precheck_only']
    lineage=copy.deepcopy(coder.last_generation);lineage['repair_diagnosis_hash']='f'*64
    with pytest.raises(ValueError):audit_generation(decl,coder.llm.call_budget.summary(),lineage,code)


def test_structural_change_still_needs_fresh_l1_l2_and_cannot_inherit_a_pass(parent,tmp_path,monkeypatch):
    coder,t,xs,d=setup_coder(parent,tmp_path);send=Mock(return_value=response(SOURCE+'\nunused_revision = 1\n'))
    monkeypatch.setattr(lc,'bounded_dispatch',send);c=conductor(tmp_path,coder,ProtocolSSH());calls=[]
    c.verifier._verify_compile=lambda *a,**k:calls.append('L1') or VerificationResult(Level.L1_COMPILE,True,metrics={'static_guards':True})
    c.verifier._verify_function=lambda *a,**k:calls.append('L2') or VerificationResult(Level.L2_FUNCTION,False,'new fake numerical failure')
    c.verifier._verify_performance=lambda *a,**k:pytest.fail('failed L2 reached L3')
    with contextlib.redirect_stdout(io.StringIO()):artifact=c.execute(t,xs,target_gflops=0.)
    assert calls==['L1','L2'] and not artifact.all_passed and send.call_count==1
    assert not (tmp_path/'corpus.jsonl').exists()


@pytest.mark.parametrize('what',['source','feedback','diagnosis'])
def test_diagnosis_drift_before_request_spends_zero_calls(parent,tmp_path,monkeypatch,what):
    coder,t,xs,d=setup_coder(parent,tmp_path)
    if what=='source':coder._repair['source']+='\n'
    elif what=='feedback':coder._repair['feedback']['primary'][1]['max_error']=1.
    else:coder._diagnosis['findings'][0]['source_sites'][0]['line']+=1
    monkeypatch.setattr(lc,'bounded_dispatch',lambda *a:pytest.fail('drift sent HTTP'))
    with pytest.raises(ValueError):coder.run(driver.sketch(),t,test_inputs=xs)
    assert coder.llm.call_budget.summary()['reserved_calls']==0


def test_offline_cli_freezes_preview_without_network_arrays_or_credit(parent,tmp_path,monkeypatch):
    from benchmarks import _numerical_repair as nr
    forbidden=lambda *a,**k:pytest.fail('offline preview touched execution/credit')
    for name in ('claim','require_available'):monkeypatch.setattr(nr,name,forbidden)
    monkeypatch.setattr(lc,'bounded_dispatch',forbidden);monkeypatch.setattr(driver,'CodegenSSH',forbidden)
    monkeypatch.setattr(driver.wd,'primary',forbidden)
    args=offline.parser().parse_args(['--case',str(FIXTURES/'repair_unchanged_case.json'),
        '--response',str(FIXTURES/'repair_unchanged_response.json'),'--output',str(tmp_path/'previews')])
    assert offline.run(args)==0
    root=next(Path(args.output).iterdir());s=strict_object((root/'summary.json').read_text())
    assert root.stat().st_mode & 0o777==0o700
    assert all(p.stat().st_mode & 0o222==0 for p in root.iterdir())
    assert s['model_calls']==s['ssh_calls']==s['gpu_dispatches']==0 and not s['executable_plan']
    assert not s['repair_credit_reset'] and not s['approved']
    assert not (root/'plan.json').exists() and not (root/'execution-claim.json').exists()
    assert strict_object((root/'generation_preview.json').read_text())['schema']=='single-candidate-repair-v2'
    assert (root/'frozen-response.json').read_text()==RAW
    assert strict_object((root/'frozen-case.json').read_text())==CASE
    with pytest.raises(SystemExit):offline.parser().parse_args(['--case','unused','--response','unused','--execute'])


@pytest.mark.parametrize('what',['response','model','source','parent_plan','extra_context','symlink','oversize'])
def test_offline_input_tampering_refused_before_output(tmp_path,what):
    case=copy.deepcopy(CASE);raw=RAW
    if what=='response':raw+=' '
    elif what=='model':case['provenance']['reported_model']='other'
    elif what=='source':case['provenance']['source_sha256']='f'*64
    elif what=='parent_plan':case['provenance']['parent_plan_hash']='f'*64
    elif what=='extra_context':case['context']['secret']='unapproved extra context'
    elif what=='oversize':raw+=' '*65537
    else:
        p=tmp_path/'link';p.symlink_to(FIXTURES/'repair_unchanged_case.json')
        with pytest.raises(ValueError):offline.read(p)
        return
    with pytest.raises(ValueError):offline.build_preview(case,raw)
