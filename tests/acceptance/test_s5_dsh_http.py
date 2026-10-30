"""Installed DSH pi-ai adapter and bounded Python client against loopback fake.

No real provider, model, GPU, user DSH profile or external network is used.
This exercises an installed runtime, not the full DSH agent or a sandbox.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from benchmarks.experiments.s5.s5_fake_http import FakeHTTPServer
from operator_agent import s5_shared_budget as shared
from operator_agent.llm_budget import LLMCallLimits, ProductionLLMCallBudget
from operator_agent.llm_client import LLMClient

DSH_ROOT=Path('/Applications/DSH Desktop.app/Contents/Resources/app/node_modules/@deepseek-ai')
FIXTURE=Path(__file__).resolve().parents[1]/'fixtures'/'s5_dsh_pi_ai_client.mjs'
NODE=shutil.which('node')


@pytest.mark.skipif(not NODE or any(not (DSH_ROOT/p/'lib/index.js').exists()
    for p in ('cordis','dsh-llm','dsh-llm-pi-ai')), reason='installed DSH pi-ai runtime unavailable')
def test_real_python_and_installed_dsh_pi_ai_share_fake_gateway_cap(tmp_path, monkeypatch):
    path=tmp_path/'shared.sqlite'
    shared.create(path,task_id='fixture',max_calls=2,max_output_tokens=40,
                  total_output_tokens=80,wall_clock_s=120)
    calls=[]
    def adapter(request):
        calls.append(request)
        if request.get('stream'):
            return [b'data: {"id":"fake-1","model":"fake-model","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}\n\n',
                    b'data: {"id":"fake-1","model":"fake-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
                    b'data: [DONE]\n\n']
        return [json.dumps({'model':'fake-model','choices':[{'message':{'content':'ok'},
                 'finish_reason':'stop'}]}).encode()]
    limits=LLMCallLimits(max_calls=1,max_output_tokens=40,total_output_token_reservations=40,
                         allowed_callers=('Coder',),call_seconds=10,campaign_seconds=60)
    local=ProductionLLMCallBudget(limits,tmp_path/'local')
    monkeypatch.setenv('LLM_FALLBACK_MODEL','')
    with FakeHTTPServer(path,task_id='fixture',model='fake-model',participant='inner',adapter=adapter) as inner, \
         FakeHTTPServer(path,task_id='fixture',model='fake-model',participant='outer',adapter=adapter) as outer:
        client=LLMClient(base_url=inner.base_url,api_key='placeholder',model='fake-model',
                         max_tokens=20,timeout=10,call_budget=local)
        assert client.chat([{'role':'user','content':'fixture'}],caller_tag='Coder')=='ok'
        env={'PATH':os.environ.get('PATH','/usr/bin:/bin'),'HOME':str(tmp_path),
             'DSH_HOME':str(tmp_path/'dsh'),'S5_FAKE_API_KEY':'placeholder',
             'NO_PROXY':'127.0.0.1','no_proxy':'127.0.0.1'}
        result=subprocess.run([NODE,str(FIXTURE),outer.base_url,str(DSH_ROOT)],env=env,
                              capture_output=True,text=True,timeout=30,check=True)
        data=json.loads(result.stdout.strip().splitlines()[-1])
    assert data['first']=={'text':'ok','reason':'stop'}
    assert data['second']['reason']=='error'
    assert len(calls)==2 and [row['stream'] for row in calls]==[False,True]
    snapshot=shared.snapshot(path)
    assert snapshot['reserved_calls']==snapshot['completed_calls']==2
    assert [row['participant'] for row in snapshot['calls']]==['inner','outer']
    assert all(row['api_usage'] is None for row in snapshot['calls'])
