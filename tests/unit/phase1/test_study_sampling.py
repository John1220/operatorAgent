"""Study wire controls precede the real local/shared budget and request hash."""
import copy
import unittest
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from benchmarks.experiments.phase1 import sampling
from benchmarks.experiments.phase1.plan import write_once
from benchmarks.experiments.s5.s5_gateway_policy import GatewayPolicy
from operator_agent.execute import llm_client, s5_shared_budget as shared
from operator_agent.execute.llm_client import LLMClient
from operator_agent.execute.s5_inner_budget import SharedProductionBudget
from operator_agent.execute.task_budget import TaskBudgetConfig


def cell(tmp_path):
    cfg=dict(schema='phase1-connected-cell-v1',run_dir=str(tmp_path/'work'),sampling=sampling.policy())
    write_once(tmp_path/'cell-config.json',cfg)
    return cfg


def ledger(tmp_path):
    path=tmp_path/'shared.sqlite'
    shared.create(path,task_id='study-test',max_calls=4,max_output_tokens=40960,
        total_output_tokens=4*40960,wall_clock_s=30)
    budget=SharedProductionBudget(TaskBudgetConfig(wall_clock_s=30,max_llm_calls=4).llm_limits(10),
        tmp_path/'work/model-journal',shared_path=path,task_id='study-test')
    return path,budget


def body():
    return dict(model='fixture',messages=[dict(role='user',content='local only')],
        temperature=.9,max_completion_tokens=2048,stream=True,
        chat_template_kwargs=dict(enable_thinking=True))


def test_dsh_model_options_preserve_legacy_integration_and_bound_study_idle_time():
    from benchmarks.experiments.phase1.host import dsh_model_options
    legacy = {'budget': {'wall_clock_s': 2400.0}}
    assert dsh_model_options(legacy) == dict(max_tokens=4096, context_window=65536)
    study = dict(sampling=sampling.policy(), budget={'wall_clock_s': 180.0})
    assert dsh_model_options(study) == dict(max_tokens=32768, context_window=65536,
                                           stream_idle_timeout_ms=180000)


def test_policy_is_copied_and_normalization_preserves_payload():
    value=sampling.policy();value['temperature']=1.7
    assert sampling.policy()['temperature']==.1
    raw=body();unchanged=copy.deepcopy(raw)
    actual=sampling.normalized(raw,sampling.policy())
    assert raw==unchanged
    assert actual['messages']==raw['messages'] and actual['stream'] is True
    assert sampling.matches(actual,sampling.policy())
    assert 'chat_template_kwargs' not in actual and 'max_completion_tokens' not in actual


@pytest.mark.parametrize('field,value',[('temperature',.3),('max_tokens',8192),
    ('enable_thinking',False),('schema','other'),('n',2),('n',True),('enable_thinking',0),
    ('n',1.),('max_tokens',32768.),('max_tokens',32000),('reasoning_effort','high'),
    ('reasoning_effort','low'),('reasoning_effort',None)])
def test_policy_cannot_be_changed_by_a_role(field,value):
    changed=sampling.policy();changed[field]=value
    with pytest.raises(ValueError,match='fixed wire'):
        sampling.normalized(body(),changed)


def test_inner_real_ledger_reserves_the_effective_body_before_transport(tmp_path,monkeypatch):
    cfg=cell(tmp_path);path,budget=ledger(tmp_path)
    client=LLMClient(base_url='http://127.0.0.1:1/v1',api_key='synthetic',model='fixture',
        temperature=.8,max_tokens=2048,enable_thinking=True,call_budget=budget,timeout=5)
    sent=[]
    def dispatch(request,seconds):
        raw=request['body_text'];actual=json.loads(raw)
        assert sampling.matches(actual,sampling.policy())
        common=shared.snapshot(path,expected_task_id='study-test')
        assert common['reserved_calls']==1
        assert common['calls'][0]['output_tokens_reserved']==32768
        assert common['calls'][0]['request_sha256']==hashlib.sha256(raw.encode()).hexdigest()
        assert common['calls'][0]['dispatched_monotonic'] is not None
        sent.append(actual)
        return json.dumps(dict(choices=[dict(message=dict(content='ok'),finish_reason='stop')],
            usage=dict(prompt_tokens=2,completion_tokens=1,total_tokens=3)))
    monkeypatch.setattr(llm_client,'bounded_dispatch',dispatch)
    original=LLMClient._call_bounded_model
    with sampling.bounded(cfg):
        assert client.chat([dict(role='user',content='hello')],temperature=.7,max_tokens=1500,caller_tag='Coder')=='ok'
    assert LLMClient._call_bounded_model is original
    assert len(sent)==1
    local=json.loads((tmp_path/'work/model-journal/call-001.json').read_text())
    assert local['request']['temperature']==.1 and local['request']['max_tokens']==32768
    assert local['request']['enable_thinking'] is True
    # `reasoning_effort` 由策略层注入到**发出去的 body**，不是客户端自己那四格；
    # 所以它出现在 sent/intent 里，而 model-journal 的 request 记的是客户端意图。
    assert sent[0]['reasoning_effort']=='medium'
    intent=json.loads((tmp_path/'work/sampling-inner.jsonl').read_text())
    assert intent['requested']['temperature']==.7 and intent['requested']['max_tokens']==1500
    assert intent['requested']['chat_template_kwargs']['enable_thinking'] is True
    assert intent['effective']['reasoning_effort']=='medium'
    assert intent['request_sha256']==local['request_sha256']==shared.snapshot(path)['calls'][0]['request_sha256']


def test_outer_policy_hash_and_reservation_are_built_after_normalization(tmp_path):
    path,_=ledger(tmp_path);sent=[]
    def adapter(request):
        state=shared.snapshot(path)
        encoded=json.dumps(request,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
        assert state['calls'][0]['request_sha256']==hashlib.sha256(encoded).hexdigest()
        assert state['calls'][0]['output_tokens_reserved']==32768
        assert sampling.matches(request,sampling.policy())
        sent.append(request)
        yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
    policy=GatewayPolicy(path,task_id='study-test',model='fixture',adapter=adapter,max_calls_limit=4)
    gateway=SimpleNamespace(_server=SimpleNamespace(gateway=policy))
    sampling.attach_gateway(gateway,dict(sampling=sampling.policy()),tmp_path/'work')
    raw=body();assert b'[DONE]' in b''.join(policy.stream(raw,participant='outer'))
    assert sent and raw['temperature']==.9
    intent=json.loads((tmp_path/'work/sampling-outer.jsonl').read_text())
    assert intent['requested']['max_completion_tokens']==2048
    assert intent['request_sha256']==shared.snapshot(path)['calls'][0]['request_sha256']


def test_no_study_binding_does_not_patch_an_ordinary_client(tmp_path,monkeypatch):
    monkeypatch.delenv('OPERATOR_AGENT_PHASE1_CELL_CONFIG',raising=False)
    original=LLMClient._call_bounded_model
    with sampling.bounded({}):assert LLMClient._call_bounded_model is original
    with sampling.bounded():assert LLMClient._call_bounded_model is original
    target=SimpleNamespace(_request=lambda *a:'original')
    gateway=SimpleNamespace(_server=SimpleNamespace(gateway=target))
    before=target._request
    sampling.attach_gateway(gateway,{},tmp_path)
    assert target._request is before


def test_context_restores_method_and_environment_on_exception(tmp_path,monkeypatch):
    cfg=cell(tmp_path)
    monkeypatch.setenv('OPERATOR_AGENT_PHASE1_CELL_CONFIG','previous-path')
    monkeypatch.setenv('OPERATOR_AGENT_PHASE1_CELL_SHA256','previous-digest')
    original=LLMClient._call_bounded_model
    with pytest.raises(RuntimeError,match='intentional'):
        with sampling.bounded(cfg):raise RuntimeError('intentional')
    assert LLMClient._call_bounded_model is original
    assert os.environ['OPERATOR_AGENT_PHASE1_CELL_CONFIG']=='previous-path'
    assert os.environ['OPERATOR_AGENT_PHASE1_CELL_SHA256']=='previous-digest'


def test_mutated_cell_refuses_before_budget_reservation(tmp_path,monkeypatch):
    cfg=cell(tmp_path);path,budget=ledger(tmp_path)
    client=LLMClient(base_url='http://127.0.0.1:1/v1',api_key='synthetic',model='fixture',call_budget=budget)
    with sampling.bounded(cfg):
        (tmp_path/'cell-config.json').write_text('{}')
        with pytest.raises(ValueError,match='changed before dispatch'):
            client.chat([dict(role='user',content='hello')],caller_tag='Coder')
    assert shared.snapshot(path)['reserved_calls']==0
    assert not (tmp_path/'work/sampling-inner.jsonl').exists()


@pytest.mark.parametrize('symlink',[False,True])
def test_inherited_worker_refuses_unbound_or_symlinked_config(tmp_path,monkeypatch,symlink):
    cfg=cell(tmp_path);source=tmp_path/'cell-config.json'
    if symlink:
        alias=tmp_path/'alias.json';alias.symlink_to(source);source=alias
    monkeypatch.setenv('OPERATOR_AGENT_PHASE1_CELL_CONFIG',str(source))
    monkeypatch.setenv('OPERATOR_AGENT_PHASE1_CELL_SHA256','0'*64)
    with pytest.raises(ValueError,match='configuration changed'):
        with sampling.bounded():pass


def test_fresh_interpreter_inherits_policy_without_changing_global_client(tmp_path):
    cfg=cell(tmp_path);path=tmp_path/'cell-config.json'
    script='''import json
from benchmarks.experiments.phase1.sampling import bounded
from operator_agent.execute.llm_client import LLMClient
def original(self,url,key,model,body,raw,caller):return body
LLMClient._call_bounded_model=original
with bounded():
 c=object.__new__(LLMClient)
 body=c._call_bounded_model('url','synthetic','fixture',dict(model='fixture',temperature=.8,max_tokens=77,stream=False),False,'Coder')
 assert LLMClient._call_bounded_model is not original
assert LLMClient._call_bounded_model is original
print(json.dumps(body))
'''
    env=dict(os.environ,OPERATOR_AGENT_PHASE1_CELL_CONFIG=str(path),
        OPERATOR_AGENT_PHASE1_CELL_SHA256=hashlib.sha256(path.read_bytes()).hexdigest())
    result=subprocess.run([sys.executable,'-c',script],cwd=Path(__file__).resolve().parents[3],
        env=env,capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
    assert sampling.matches(json.loads(result.stdout),sampling.policy())


def test_autonomous_launcher_rewrites_only_the_known_production_entry(tmp_path,monkeypatch):
    from operator_agent.execute import exec_guard
    calls=[]
    def launch(argv,*args,**kwargs):calls.append((argv,args,kwargs));return 'owned-process'
    monkeypatch.setattr(exec_guard,'start_background_worker',launch)
    run_dir = tmp_path / 'run'
    worker_config = run_dir / 'autonomous.json'
    write_once(worker_config, {'invocation_id': 'owned'})
    cfg = dict(sampling=sampling.policy(), run_dir=str(run_dir), budget={'wall_clock_s': 2400.0})
    with sampling.autonomous_launcher(cfg):
        assert exec_guard.start_background_worker([sys.executable,'-m','operator_agent.app.autonomous',str(run_dir)],cwd='cwd')=='owned-process'
        with pytest.raises(ValueError,match='unexpected'):
            exec_guard.start_background_worker([sys.executable,'-m','other',str(run_dir)])
        with pytest.raises(ValueError,match='unexpected'):
            exec_guard.start_background_worker([sys.executable,'-m','operator_agent.app.autonomous',str(tmp_path/'other')])
    assert exec_guard.start_background_worker is launch
    assert calls==[([sys.executable,'-m','benchmarks.experiments.phase1.study_autonomous',str(run_dir)],(),dict(cwd='cwd'))]
    assert json.loads(worker_config.read_text()) == {'invocation_id':'owned', 'llm_timeout':524.288}


def test_upstream_deadline_scales_with_the_output_cap():
    """单次上游请求的墙钟上界必须随输出上限走。

    2026-11-04 真机：固定 120 s 的 deadline 配 32768/medium，A 臂一次写整份核就超过
    2 分钟，网关切流 ⇒ 子进程报 `Stream ended without finish_reason`，两格被判成
    native_host_failed。失败理由长得像模型故障，其实是常数太小。
    """
    from benchmarks.experiments.phase1.host import upstream_deadline_s
    def cfg(max_tokens, wall=2400.0):
        return dict(sampling={'max_tokens': max_tokens}, budget={'wall_clock_s': wall})
    assert upstream_deadline_s(cfg(4096)) == 120.0, '4096 时应保持原来的 120 s'
    assert upstream_deadline_s(cfg(32768)) > 300.0, '32K 不能仍是两分钟级'
    assert upstream_deadline_s(cfg(32768, wall=300.0)) == 300.0, '不许超过该格墙钟预算'
    assert upstream_deadline_s(dict(sampling={}, budget={'wall_clock_s': 2400.0})) == 120.0, \
        '缺 sampling 时按 4096 处理，不抛也不放大'


def test_study_gateway_preserves_finish_after_many_small_stream_events(tmp_path,monkeypatch):
    from urllib.request import Request, urlopen
    from benchmarks.experiments.phase1.host import model_gateway
    from tests.acceptance.test_s5_gateway_transport import FINISH, fake_server, handler_for

    # A small text fragment can carry a much larger SSE envelope. Exercise
    # both old wire limits while staying below the model's token allowance.
    fragment = ('data: ' + json.dumps(dict(id='fixture-' + 'x'*170,
        choices=[dict(delta={'reasoning_content':'a'},finish_reason=None)])) + '\n\n').encode()
    payload = fragment * 5000 + FINISH
    assert len(payload) > 1024 * 1024
    path, _ = ledger(tmp_path)
    monkeypatch.setenv('STUDY_FIXTURE_KEY','loopback-only')
    with fake_server(handler_for(body=payload,content_type='text/event-stream')) as endpoint:
        config = dict(model=dict(base_url=endpoint.removesuffix('/chat/completions'),
                                 id='fixture',api_key_env='STUDY_FIXTURE_KEY'),
                      budget=dict(max_llm_calls=4,wall_clock_s=30),sampling=sampling.policy(),
                      run_dir=str(tmp_path/'work'),shared_ledger=str(path),shared_task_id='study-test')
        with model_gateway(config,tmp_path) as gateway:
            request = Request(gateway.base_url+'/chat/completions',data=json.dumps(body()).encode(),
                              headers={'Content-Type':'application/json'})
            with urlopen(request,timeout=10) as response:
                assert response.read() == payload
    receipt = json.loads((tmp_path/'transport-failures.jsonl').read_text())
    assert receipt['outcome']=='completed' and receipt['finish_seen'] is True
    calls = shared.snapshot(path)['calls']
    assert len(calls)==1 and calls[0]['output_tokens_reserved']==32768


class TestUpstreamStreamObservation(unittest.TestCase):
    """断流必须留下可查的事实，而不是只留一句 `Stream ended without finish_reason`。

    2026-11-04 A 臂反复断在 ~60 s，网关把上游异常改写成 `PolicyUpstreamFailure`
    （不带诊断），于是「超时还是网络断、断在第几块」全查不到。
    """

    def _run(self, stream_factory):
        import tempfile
        from benchmarks.experiments.phase1 import host
        cell = Path(tempfile.mkdtemp())
        transport = SimpleNamespace(stream=stream_factory)
        chunks, error = [], None
        try:
            chunks = list(host._observed_stream(transport, {"model": "x"}, cell, 3))
        except Exception as exc:                  # noqa: BLE001
            error = exc
        rows = [json.loads(line) for line in
                (cell / 'transport-failures.jsonl').read_text().splitlines()]
        return chunks, rows, error

    def test_failed_stream_records_diagnostic_and_still_raises(self):
        from benchmarks.experiments.s5.s5_gateway_transport import UpstreamError

        def boom(request):
            yield b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
            raise UpstreamError('upstream request unavailable or incomplete',
                                stage='response_body', code='timeout')

        _, rows, error = self._run(boom)
        self.assertIsInstance(error, UpstreamError, '观测层不许把异常吞掉')
        self.assertEqual(rows[0]['outcome'], 'failed')
        self.assertEqual(rows[0]['failure']['code'], 'timeout')
        self.assertEqual(rows[0]['chunks'], 1, '断之前收到的块数要能看见')
        self.assertIsNotNone(rows[0]['first_chunk_s'])

    def test_completed_stream_records_control_row(self):
        def ok(request):
            yield b'data: {"choices":[{"delta":{"content":"a"},"finish_reason":"stop"}]}\n\n'
            yield b'data: [DONE]\n\n'

        chunks, rows, error = self._run(ok)
        self.assertIsNone(error)
        self.assertEqual(len(chunks), 2, '观测层不许吞块')
        self.assertEqual(rows[0]['outcome'], 'completed')
        self.assertTrue(rows[0]['finish_seen'])
        self.assertIsNone(rows[0]['failure'])


class TestStreamLimitScalesWithOutputCap(unittest.TestCase):
    """网关的流上限必须跟着输出上限走，否则长响应会被自己的界掐断。

    2026-11-04 实测：`UpstreamTransport` 的老界是 1 MiB / 4096 块，而一个 32K 输出
    的 SSE 响应约 1.05 MB / 3734 块 —— 同一请求重复 6 次，5 次在 1,048,xxx 字节处
    被掐，DSH 只看到 `Stream ended without finish_reason`，看着像网络故障。
    """

    def test_legacy_limits_kept_for_the_default_cap(self):
        from benchmarks.experiments.phase1.host import stream_limits
        self.assertEqual(stream_limits({'sampling': {'max_tokens': 4096}}), (1048576, 4096))

    def test_large_cap_gets_room_for_a_32k_stream(self):
        from benchmarks.experiments.phase1.host import stream_limits
        size, chunks = stream_limits({'sampling': {'max_tokens': 32768}})
        self.assertGreaterEqual(size, 32 * 1024 * 1024, '32K 输出仍被 1 MiB 档卡住')
        self.assertGreaterEqual(chunks, 65536, '块数上限会把长流提前掐断')
