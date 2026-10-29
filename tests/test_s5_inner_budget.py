"""Experimental shared pool must constrain the real bounded inner client."""
import json
import pytest
from operator_agent import s5_shared_budget as shared
from operator_agent.llm_budget import LLMCallLimits, LLMCallRefused
from operator_agent.llm_client import LLMClient
from operator_agent.s5_inner_budget import SharedProductionBudget, production_budget, _provider_counters
from operator_agent.llm_budget import ProductionLLMCallBudget


def create(tmp_path, max_calls=2):
    path=tmp_path/'shared.sqlite'
    shared.create(path,task_id='task-01',max_calls=max_calls,max_output_tokens=100,
                  total_output_tokens=100*max_calls,wall_clock_s=120)
    limits=LLMCallLimits(max_calls=5,max_output_tokens=100,total_output_token_reservations=500,
                         allowed_callers=('Coder',),call_seconds=20,campaign_seconds=120)
    return path,SharedProductionBudget(limits,tmp_path/'inner',shared_path=path,task_id='task-01')


def test_outer_and_actual_inner_client_share_dispatch_limit(tmp_path, monkeypatch):
    path,budget=create(tmp_path,max_calls=2)
    outer=shared.reserve(path,participant='outer',caller='agent',max_output_tokens=10,request_sha256='a'*64)
    shared.dispatch(path,call_id=outer['call_id'])
    shared.finish(path,call_id=outer['call_id'],outcome='answered',usage_evidence='unverified')
    seen=[]
    def send(req,seconds):
        seen.append(True)
        return json.dumps({'model':'test','choices':[{'message':{'content':'ok'},'finish_reason':'stop'}],
                           'usage':{'prompt_tokens':11,'completion_tokens':2,'total_tokens':13}})
    monkeypatch.setattr('operator_agent.llm_client.bounded_dispatch',send)
    client=LLMClient(base_url='https://example.test/v1',model='test',api_key='test',max_tokens=20,
                     call_budget=budget,timeout=10)
    assert client.chat([{'role':'user','content':'x'}],caller_tag='Coder')=='ok'
    with pytest.raises(LLMCallRefused): client.chat([{'role':'user','content':'x'}],caller_tag='Coder')
    s=shared.snapshot(path)
    assert len(seen)==1 and s['reserved_calls']==2 and s['remaining_calls']==0
    assert s['calls'][0]['api_usage'] is None and s['calls'][1]['api_usage']['total_tokens']==13


def test_failed_local_reservation_does_not_refund_shared_pool(tmp_path):
    path,budget=create(tmp_path)
    identity={'model':'test'}; budget.bind(identity)
    with pytest.raises(LLMCallRefused):
        budget.reserve(identity,json.dumps({'max_tokens':10}),caller='not-allowed',timeout=5)
    s=shared.snapshot(path)
    assert s['reserved_calls']==1 and s['calls'][0]['dispatched_monotonic'] is None


def test_replaced_or_wrong_ledger_fails_closed(tmp_path):
    path,budget=create(tmp_path)
    identity={'model':'test'}; budget.bind(identity)
    path.rename(tmp_path/'old.sqlite')
    shared.create(path,task_id='task-01',max_calls=2,max_output_tokens=100,total_output_tokens=200,wall_clock_s=120)
    with pytest.raises(LLMCallRefused,match='identity'):
        budget.reserve(identity,json.dumps({'max_tokens':10}),caller='Coder',timeout=5)
    assert shared.snapshot(path)['reserved_calls']==0


def test_missing_usage_remains_unknown(tmp_path, monkeypatch):
    path,budget=create(tmp_path)
    monkeypatch.setattr('operator_agent.llm_client.bounded_dispatch',lambda *_:json.dumps(
        {'model':'test','choices':[{'message':{'content':'ok'},'finish_reason':'stop'}]}))
    client=LLMClient(base_url='https://example.test/v1',model='test',api_key='test',max_tokens=20,
                     call_budget=budget,timeout=10)
    assert client.chat([{'role':'user','content':'x'}],caller_tag='Coder')=='ok'
    assert shared.snapshot(path)['calls'][0]['api_usage'] is None


def test_ambiguous_provider_counter_names_remain_unknown():
    assert _provider_counters({'input_tokens':10,'output_tokens':2}, 'api_usage') is None
    assert _provider_counters({'prompt_tokens':10,'completion_tokens':2}, 'api_usage') == {
        'prompt_tokens':10,'completion_tokens':2}


def test_dispatch_audit_failure_stops_all_and_cannot_be_caught_to_retry(tmp_path, monkeypatch):
    path,budget=create(tmp_path)
    identity={'model':'test'}; budget.bind(identity)
    call,_=budget.reserve(identity,json.dumps({'max_tokens':10}),'Coder',5)
    def fail(*args,**kwargs): raise shared.BudgetError('ledger_failure')
    monkeypatch.setattr(shared,'dispatch',fail)
    with pytest.raises(LLMCallRefused,match='campaign stop recorded'):
        budget.mark_dispatched(call)
    assert shared.snapshot(path)['aborted']
    assert shared.snapshot(path)['calls'][0]['dispatched_monotonic'] is None
    with pytest.raises(LLMCallRefused,match='campaign stopped'):
        budget.reserve(identity,json.dumps({'max_tokens':10}),'Coder',5)
    with pytest.raises(shared.BudgetError,match='campaign_aborted'):
        shared.reserve(path,participant='outer',caller='loop',max_output_tokens=10,request_sha256='a'*64)


def test_local_dispatch_failure_records_intent_and_stops_before_handoff(tmp_path, monkeypatch):
    path,budget=create(tmp_path)
    identity={'model':'test'}; budget.bind(identity)
    call,_=budget.reserve(identity,json.dumps({'max_tokens':10}),'Coder',5)
    def fail(*args,**kwargs): raise LLMCallRefused('local journal failed')
    monkeypatch.setattr(ProductionLLMCallBudget,'mark_dispatched',fail)
    with pytest.raises(LLMCallRefused,match='local_dispatch'):
        budget.mark_dispatched(call)
    snapshot=shared.snapshot(path)
    assert snapshot['aborted'] and snapshot['calls'][0]['dispatched_monotonic'] is not None
    assert budget.summary()['attempts'][0]['dispatched_monotonic'] is None
    assert snapshot['network_interception_proven'] is False


def test_completion_audit_failure_stops_other_participants(tmp_path, monkeypatch):
    path,budget=create(tmp_path)
    identity={'model':'test'}; budget.bind(identity)
    call,_=budget.reserve(identity,json.dumps({'max_tokens':10}),'Coder',5)
    budget.mark_dispatched(call)
    def fail(*args,**kwargs): raise shared.BudgetError('ledger_failure')
    monkeypatch.setattr(shared,'finish',fail)
    with pytest.raises(LLMCallRefused,match='completion'):
        budget.finish(call,outcome='answered')
    assert shared.snapshot(path)['aborted']
    assert shared.snapshot(path)['calls'][0]['outcome'] is None
    assert budget.summary()['attempts'][0]['status']=='answered'


def test_unexpected_dispatch_exception_poisoned_globally(tmp_path, monkeypatch):
    path,budget=create(tmp_path)
    identity={'model':'test'}; budget.bind(identity)
    call,_=budget.reserve(identity,json.dumps({'max_tokens':10}),'Coder',5)
    def fail(*args,**kwargs): raise OSError('injected controller error')
    monkeypatch.setattr(shared,'dispatch',fail)
    with pytest.raises(LLMCallRefused,match='campaign stop recorded'): budget.mark_dispatched(call)
    assert shared.snapshot(path)['aborted']


def test_local_binding_failure_stops_other_participants(tmp_path, monkeypatch):
    path,budget=create(tmp_path)
    def fail(*args,**kwargs): raise OSError('injected journal error')
    monkeypatch.setattr('operator_agent.llm_budget.atomic_json',fail)
    with pytest.raises(LLMCallRefused,match='local_binding'): budget.bind({'model':'test'})
    assert shared.snapshot(path)['aborted']


def test_local_initialization_failure_stops_other_participants(tmp_path):
    path,budget=create(tmp_path)
    with pytest.raises(LLMCallRefused,match='initialization failed; campaign stop recorded'):
        SharedProductionBudget(budget.limits,tmp_path/'inner',shared_path=path,task_id='task-01')
    assert shared.snapshot(path)['aborted']


def test_identity_checked_inside_transaction_not_only_before_call(tmp_path, monkeypatch):
    path,budget=create(tmp_path)
    identity={'model':'test'}; budget.bind(identity)
    original=shared.reserve
    def replace_then_reserve(path,**kwargs):
        path.rename(tmp_path/'previous.sqlite')
        shared.create(path,task_id='another-task',max_calls=2,max_output_tokens=100,
                      total_output_tokens=200,wall_clock_s=120)
        return original(path,**kwargs)
    monkeypatch.setattr(shared,'reserve',replace_then_reserve)
    with pytest.raises(LLMCallRefused,match='global stop unproven'):
        budget.reserve(identity,json.dumps({'max_tokens':10}),'Coder',5)
    other=shared.snapshot(path)
    assert other['task_id']=='another-task' and not other['aborted'] and other['reserved_calls']==0


def test_replacement_during_initialization_is_rejected_without_aborting_replacement(tmp_path, monkeypatch):
    path,budget=create(tmp_path)
    original=shared.snapshot
    def swap_after_snapshot(path,**kwargs):
        info=original(path,**kwargs)
        path.rename(tmp_path/'original.sqlite')
        shared.create(path,task_id='another-task',max_calls=2,max_output_tokens=100,
                      total_output_tokens=200,wall_clock_s=120)
        return info
    monkeypatch.setattr(shared,'snapshot',swap_after_snapshot)
    with pytest.raises(LLMCallRefused,match='before initialization'):
        SharedProductionBudget(budget.limits,tmp_path/'inner2',shared_path=path,task_id='task-01')
    other=original(path)
    assert other['task_id']=='another-task' and not other['aborted']


def test_production_factory_requires_complete_explicit_binding(tmp_path, monkeypatch):
    path,budget=create(tmp_path)
    limits=budget.limits
    monkeypatch.delenv('OPERATOR_AGENT_S5_SHARED_LEDGER',raising=False)
    monkeypatch.delenv('OPERATOR_AGENT_S5_TASK_ID',raising=False)
    normal=production_budget(limits,tmp_path/'normal')
    assert type(normal) is ProductionLLMCallBudget
    monkeypatch.setenv('OPERATOR_AGENT_S5_SHARED_LEDGER',str(path))
    with pytest.raises(LLMCallRefused,match='incomplete'):
        production_budget(limits,tmp_path/'partial')
    monkeypatch.setenv('OPERATOR_AGENT_S5_TASK_ID','task-01')
    bound=production_budget(limits,tmp_path/'bound')
    assert isinstance(bound,SharedProductionBudget)


def test_repair_entry_routes_inner_calls_to_shared_ledger(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from operator_agent import hitl_tools as ht, llm_client, fixer, repair_loop, tool_calls
    from operator_agent.schemas import VerificationLevel, VerificationResult
    ledger_path=tmp_path/'shared.sqlite'
    shared.create(ledger_path,task_id='task-01',max_calls=1,max_output_tokens=40960,
                  total_output_tokens=40960,wall_clock_s=120)
    monkeypatch.setenv('OPERATOR_AGENT_S5_SHARED_LEDGER',str(ledger_path))
    monkeypatch.setenv('OPERATOR_AGENT_S5_TASK_ID','task-01')
    monkeypatch.setattr(ht,'_llm_problem',lambda:None)
    monkeypatch.setattr(ht,'_make_backend',lambda *a,**kw:(object(),{}))
    monkeypatch.setattr(ht,'Verifier',lambda backend:object())
    monkeypatch.setattr(tool_calls,'_REPAIR_CONTEXT',None)
    class OfflineClient:
        def __init__(self,*,call_budget,**kwargs):
            self.call_budget=call_budget; self.identity={'model':'offline'}; call_budget.bind(self.identity)
        def call(self):
            n,_=self.call_budget.reserve(self.identity,json.dumps({'max_tokens':10}),'Fixer',5)
            self.call_budget.mark_dispatched(n)
            self.call_budget.finish(n,outcome='answered',usage_source='api_usage',
                                    usage={'prompt_tokens':10,'completion_tokens':2,'total_tokens':12})
    monkeypatch.setattr(llm_client,'LLMClient',OfflineClient)
    monkeypatch.setattr(fixer,'FixerAgent',lambda llm:SimpleNamespace(llm=llm))
    def loop(code,task,inputs,verifier,fixer,*,before_verification,**kwargs):
        assert before_verification(1,code)
        fixer.llm.call()
        return dict(code=code,passed=False,rounds=[],results=[VerificationResult(
            VerificationLevel.L1_COMPILE,False,'offline')],log='',initial_full_l2_passed=None,
            final_full_l2_passed=None,candidate_records=[],selected_source_hash=ht.code_hash(code))
    monkeypatch.setattr(repair_loop,'repair',loop)
    run=tmp_path/'run'; ht.prepare({'request':'matmul fp32','shapes':[[8,8,8]],'precision':'fp32'},run)
    ht.repair(run,'def kernel(A, B):\n    return A @ B\n',max_llm_calls=3,oracle_total=5)
    data=shared.snapshot(ledger_path)
    assert data['reserved_calls']==1 and data['calls'][0]['participant']=='inner'
    assert data['calls'][0]['api_usage']['total_tokens']==12
