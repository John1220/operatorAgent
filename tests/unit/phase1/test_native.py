"""Native baseline verification owns one frozen task and its actual budgets."""
import hashlib
import http.client
import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from benchmarks.experiments.phase1 import native
from benchmarks.experiments.phase1.plan import read
from operator_agent.app import hitl_tools as ht
from operator_agent.execute.task_clock import monotonic

GOOD='def kernel(A, B):\n    return A @ B\n'
BAD='def kernel(A, B):\n    return (A @ B)*0\n'
DECLARATION=dict(request='matmul fp32',dsl='python',precision='fp32',shapes=[[4,4,4]])


@pytest.fixture
def service(tmp_path):
    work=tmp_path/'work'
    prepared=ht.prepare(DECLARATION,work)
    config=dict(arm='A',entry_a='dsh_native',run_dir=str(work),declaration=DECLARATION,
        backend='fake',connection={},deadline_monotonic=monotonic()+30,
        budget=dict(wall_clock_s=30,max_llm_calls=8,oracle_total=20),max_rounds=2,
        execution_context=dict(mode='fake',test_only=True,gpu_acceptance=False,
                               performance_evidence=False))
    return native.VerificationService(config,prepared['contract_hash'])


def test_ordinary_complete_verification_and_last_submission_not_best(service):
    good=service.develop(GOOD,service.contract_hash)
    assert good['status']=='passed' and good['full_l2_passed'] is True
    assert good['levels']==dict(L1=True,L2=True,L3=False)
    assert good['backend']=='fake' and good['execution_context']['test_only'] is True
    assert good['development_used']==1 and good['max_development']==18
    assert good['confirm_reserved']==2
    assert good['remaining']==dict(non_confirm=17,confirm=2)
    assert good['oracle_unit']=='candidate_verification_dispatch_per_input_group'
    assert read(good['verification_evidence']['path'])['contract_hash']==service.contract_hash
    bad=service.develop(BAD,service.contract_hash)
    assert bad['status']=='failed' and bad['full_l2_passed'] is False
    assert bad['development_used']==2 and bad['remaining']==dict(non_confirm=16,confirm=2)
    state=service.snapshot()
    assert state['verification_count']==2 and state['last_submission_verified'] is True
    assert Path(state['selected_source']).read_text()==BAD
    assert state['source_hash']==hashlib.sha256(BAD.encode()).hexdigest()
    ledger=read(service.work/'verification-budget.json')
    assert ledger['spent_all']==2 and ledger['spent']['confirm']==0
    assert ledger['remaining']['confirm']>0
    for event in read(service.work/'events.json'):
        ref=event['source']
        assert hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest()==ref['sha256']


@pytest.mark.parametrize('code,contract',[('',None),(' ',None),(None,None),('a'*65537,None),(GOOD,'x'*64)])
def test_invalid_source_or_contract_cannot_reserve_or_replace_candidate(service,code,contract):
    with pytest.raises(ValueError):service.develop(code,contract or service.contract_hash)
    assert service.used==0 and service.snapshot()['selected_source'] is None
    assert read(service.work/'verification-budget.json')['spent_all']==0


def test_common_oracle_cap_does_not_reset_and_final_uses_last_submitted_source(service):
    # One development dispatch and one protected confirm slot: the next
    # submission must not borrow the reserved final confirmation allowance.
    config=dict(service.config,budget=dict(wall_clock_s=30,max_llm_calls=8,oracle_total=2))
    service=native.VerificationService(config,service.contract_hash)
    assert service.develop(GOOD,service.contract_hash)['status']=='passed'
    refused=service.develop(BAD,service.contract_hash)
    assert refused['ok'] is False and refused['error']=='development_budget_exhausted'
    assert refused['development_used']==refused['max_development']==1
    assert refused['confirm_reserved']==1 and refused['remaining']==dict(non_confirm=0,confirm=1)
    assert service.used==1
    assert Path(service.snapshot()['selected_source']).read_text()==BAD
    assert service.snapshot()['last_submission_verified'] is False
    assert service.snapshot()['levels']=={}
    assert read(service.work/'verification-budget.json')['spent_all']==1
    assert service.records[-1]['admitted'] is False


def test_verifier_failure_consumes_reservation_and_records_unknown(service,monkeypatch):
    def broken(*args,**kwargs):raise RuntimeError('cannot verify')
    monkeypatch.setattr(service,'_verify',broken)
    result=service.develop(GOOD,service.contract_hash)
    assert result['status']=='unknown' and result['passed'] is None
    assert result['error']=='RuntimeError'
    assert result['development_used']==1 and result['remaining']==dict(non_confirm=17,confirm=2)
    assert result['full_l2_passed'] is None
    assert service.used==1 and service.snapshot()['last_submission_verified'] is False
    assert read(service.work/'verification-budget.json')['spent_all']==1


def test_deadline_refusal_still_retains_the_submitted_source(service):
    service.config['deadline_monotonic']=monotonic()-1
    result=service.develop(GOOD,service.contract_hash)
    assert result['ok'] is False and result['error']=='verification_closed'
    assert result['development_used']==0 and result['remaining']==dict(non_confirm=18,confirm=2)
    assert service.used==0 and service.snapshot()['source_hash']==hashlib.sha256(GOOD.encode()).hexdigest()
    assert service.snapshot()['last_submission_verified'] is False


def post(service,body,token=None):
    conn=http.client.HTTPConnection('127.0.0.1',service.server.server_address[1],timeout=5)
    try:
        conn.request('POST','/development/verify',body=json.dumps(body),headers={
            'Authorization':'Bearer '+(service.token if token is None else token),
            'Content-Type':'application/json'})
        response=conn.getresponse()
        return response.status,json.loads(response.read())
    finally:conn.close()


def test_real_public_http_client_protocol_and_closed_service(service,monkeypatch):
    from operator_agent.execute import s5_verify_client
    with service:
        cfg=service.client_config()
        assert set(cfg)=={'url','token','contract','backend','timeout'}
        # The compatibility label does not change the executed backend.
        assert cfg['backend']=='mock' and service.config['backend']=='fake'
        from urllib.parse import urlsplit
        monkeypatch.setattr(s5_verify_client,'configuration',lambda:(
            {key.upper():str(value) for key,value in cfg.items()},urlsplit(cfg['url']),cfg['timeout']))
        response=s5_verify_client.submit(GOOD,contract_hash=service.contract_hash)
        assert response['status']=='passed' and response['levels']['L2'] is True
    assert service.closed is True and not service.thread.is_alive()


@pytest.mark.parametrize('change',[{'backend':'ssh'},{'max_rounds':99},{'contract_hash':'wrong'},{}])
def test_http_rejects_auth_or_extra_execution_options_without_spend(service,change):
    with service:
        body=dict(code=GOOD,contract_hash=service.contract_hash,**{k:v for k,v in change.items() if k!='contract_hash'})
        if 'contract_hash' in change:body['contract_hash']=change['contract_hash']
        status,response=post(service,body,token='wrong' if not change else None)
        assert status in (400,403)
        assert response['ok'] is False
    assert service.used==0 and not service.records


def test_native_run_requires_explicit_matching_arm_before_any_environment_access():
    with pytest.raises(ValueError):native.run(dict(arm='B',entry_a='dsh_native'))
    with pytest.raises(ValueError):native.run(dict(arm='A',entry_a='llm_loop'))


@pytest.mark.parametrize('tamper',['source','contract'])
def test_trusted_verifier_rechecks_frozen_source_and_contract_before_backend_use(service,monkeypatch,tamper):
    source=service.work/'candidate.py';source.write_text(GOOD)
    request=dict(run_dir=str(service.work),source=str(source),source_hash=hashlib.sha256(GOOD.encode()).hexdigest(),
        contract_hash=service.contract_hash,backend='fake',connection={},result=str(service.work/'worker-result.json'))
    if tamper=='source':source.write_text(BAD)
    else:request['contract_hash']='a'*64
    def forbidden(*args,**kwargs):raise AssertionError('mismatched input reached the backend')
    monkeypatch.setattr(ht,'verify',forbidden)
    with pytest.raises(ValueError,match='binding mismatch'):native.verify_worker(request)
    assert not Path(request['result']).exists()


def test_native_host_keeps_ordinary_surface_and_pins_the_last_public_submission(tmp_path,monkeypatch):
    from benchmarks.experiments.phase1 import host
    cfg=dict(arm='A',entry_a='dsh_native',cell_id='matmul-A',plan_hash='a'*64,
        run_dir=str(tmp_path/'cell/work'),declaration=DECLARATION,backend='fake',connection={},
        deadline_monotonic=monotonic()+30,budget=dict(wall_clock_s=30,max_llm_calls=8,oracle_total=20),
        max_rounds=2,model=dict(id='fixture'),execution_context=dict(mode='fake',test_only=True),
        sampling=dict(max_tokens=4096))
    captured={}
    @contextmanager
    def gateway(config,cell):
        assert config==cfg and cell==tmp_path/'cell'
        yield type('Gateway',(),dict(base_url='http://127.0.0.1:1/v1'))()
    monkeypatch.setattr(host,'model_gateway',gateway,raising=False)
    monkeypatch.setattr(host,'runtime_identity',lambda:{'fixture':'a'*64})
    def build(arm,**kwargs):
        captured.update(kwargs)
        assert arm=='A' and 'relay' not in kwargs
        assert kwargs['development']['backend']=='mock'
        return dict(surface=dict(core_cli=False,core_source=False,plugin=False,mcp=False),
            env=dict(PATH='/bin'),patch_sha256='b'*64)
    monkeypatch.setattr(native.s5_plugin_env,'build',build)
    def headless(built,prompt,**kwargs):
        assert 'ordinary shell, read and edit' in prompt and 'check.py --code-file' in prompt
        assert 'Frozen common budget: '+json.dumps(cfg['budget'],sort_keys=True) in prompt
        assert 'Effective wire sampling policy: '+json.dumps(cfg['sampling'],sort_keys=True) in prompt
        assert 'not a GPU launch, GPU seconds' in prompt and 'confirm allowance is protected' in prompt
        endpoint=captured['development']['url']
        from urllib.parse import urlsplit
        for code in (GOOD,BAD):
            url=urlsplit(endpoint);connection=http.client.HTTPConnection(url.hostname,url.port,timeout=10)
            connection.request('POST',url.path,json.dumps(dict(code=code,contract_hash=captured['development']['contract'])),
                headers={'Content-Type':'application/json','Authorization':'Bearer '+captured['development']['token']})
            response=connection.getresponse()
            assert response.status==200;response.read();connection.close()
        return dict(returncode=0,events=[],stderr='')
    monkeypatch.setattr(native.s5_arm_env,'start_headless',headless)
    result=native.run(cfg)
    assert result['phase']=='done' and result['levels']['L2'] is False
    assert Path(result['selected_source']).read_text()==BAD
    assert result['source_hash']==hashlib.sha256(BAD.encode()).hexdigest()
    assert result['dsh_entry_verified'] is False
    assert result['last_submission_verified'] is True
    work=Path(cfg['run_dir'])
    assert read(work/'native-service.json')['verification_count']==2
    assert read(work/'connected-result.json')==result
    assert read(tmp_path/'cell/host-surface.json')['ordinary_shell_edit'] is True
    assert read(work/'verification-budget.json')['spent']['confirm']==0


def test_last_passing_submission_keeps_a_pass_after_the_host_dies():
    """宿主退出不该抹掉「验证服务已经判过」的通过。

    2026-11-04 真机：一格连着 verify-5/7/8 全过，随后模型调用额度用满、网关回 429、
    DSH 非零退出 ⇒ 整格记成 native_host_failed、levels 为空。候选的判定来自可信
    验证服务，与宿主死活无关；选择规则本来就是「最后一次提交」，这里只是把它落到
    「最后一次**通过**的提交」，不挑最快的那个。
    """
    from benchmarks.experiments.phase1.native import last_passing_submission
    rows = [
        {"index": 1, "response": {"passed": False, "levels": {"L1": False}}, "source_hash": "a"},
        {"index": 2, "response": {"levels": {"L1": True, "L2": True, "L3": False}}, "source_hash": "b"},
        {"index": 3, "response": {"levels": {"L1": True, "L2": True, "L3": True}}, "source_hash": "c"},
        {"index": 4, "response": {"error": "verification_closed"}, "source_hash": "d"},
    ]
    best = last_passing_submission(rows)
    assert best["index"] == 3 and best["source_hash"] == "c"


def test_last_passing_submission_is_none_without_any_pass():
    from benchmarks.experiments.phase1.native import last_passing_submission
    rows = [{"index": 1, "response": {"levels": {"L1": False}}},
            {"index": 2, "response": {"levels": {"L1": True, "L2": False}}}]
    assert last_passing_submission(rows) is None


def test_consecutive_no_gain_closes_the_native_verifier(tmp_path):
    """A 臂也要有「连续 2 次跑对但没更快 → 停」这条规则。

    2026-11-04 实测：一格提交 10 次，其中 6 次是 L1✓L2✓L3✗，一路烧到模型调用用满
    （773 s / 30 次调用）。A 臂不经过 Conductor，所以它此前**只有预算**这一个限制。
    """
    from operator_agent.app.conductor import MAX_STAGNANT_ROUNDS
    work = tmp_path / 'work'
    prepared = ht.prepare(DECLARATION, work)
    config = dict(arm='A', entry_a='dsh_native', run_dir=str(work), declaration=DECLARATION,
                  backend='fake', connection={}, deadline_monotonic=monotonic() + 30,
                  budget=dict(wall_clock_s=30, max_llm_calls=8, oracle_total=20), max_rounds=2,
                  execution_context=dict(mode='fake', test_only=True, gpu_acceptance=False,
                                         performance_evidence=False))
    service = native.VerificationService(config, prepared['contract_hash'])
    # fake 后端的 L3 读数是 uninformative（不能裁决策），按设计**不计数** —— 所以这里
    # 直接喂一条「L3 有效但没过」的反馈，验的是计数与关闭本身。
    slow = {'verdicts': [{'level': 'L1', 'passed': True},
                         {'level': 'L2', 'passed': True},
                         {'level': 'L3', 'passed': False,
                          'metrics': {'measurement_validity': 'valid'}}]}
    for _ in range(MAX_STAGNANT_ROUNDS):
        service._note_no_gain(slow)
    assert service.no_gain_streak >= MAX_STAGNANT_ROUNDS
    assert service.closed is True
    assert service.stop_reason == 'no_gain_%d_rounds' % MAX_STAGNANT_ROUNDS
    after = service.develop(GOOD, service.contract_hash)
    assert after['ok'] is False and after['error'] == 'verification_closed'
    assert after['stop_reason'] == service.stop_reason


def test_no_gain_only_counts_valid_l3_and_ignores_compile_failures():
    """只有「L3 有效读数但没过」才计数；编译失败与无效测量都不许算停滞。"""
    from benchmarks.experiments.phase1.native import l3_no_gain
    compile_fail = {'verdicts': [{'level': 'L1', 'passed': False}]}
    valid_slow = {'verdicts': [{'level': 'L3', 'passed': False,
                                'metrics': {'measurement_validity': 'valid'}}]}
    degraded = {'verdicts': [{'level': 'L3', 'passed': False,
                              'metrics': {'measurement_validity': 'degraded'}}]}
    faster = {'verdicts': [{'level': 'L3', 'passed': True,
                            'metrics': {'measurement_validity': 'valid'}}]}
    assert l3_no_gain(compile_fail) is None
    assert l3_no_gain(valid_slow) is True
    assert l3_no_gain(degraded) is None
    assert l3_no_gain(faster) is False
