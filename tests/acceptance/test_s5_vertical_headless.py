"""Actual installed headless CLI with a local fake model; no real provider/GPU."""
import json
import shutil

import pytest
from benchmarks.experiments.s5.s5_fake_http import FakeHTTPServer
from benchmarks.experiments.s5.s5_vertical_headless import DSH_ROOT, run_headless
from operator_agent import s5_shared_budget as shared


@pytest.mark.skipif(not shutil.which('node') or not (DSH_ROOT/'dsh/lib/bin.js').is_file(),
                    reason='installed DSH Desktop unavailable')
def test_headless_one_task_with_loopback_fixture(tmp_path):
    ledger=tmp_path/'budget.sqlite'
    shared.create(ledger,task_id='headless-fixture',max_calls=1,max_output_tokens=4096,
                  total_output_tokens=4096,wall_clock_s=120)
    calls=[]
    content='def kernel(X):\n    return X\n'
    def adapter(request):
        calls.append(request)
        assert request.get('stream') is True
        chunk={'id':'fake-1','model':'fake-model','choices':[{'index':0,
               'delta':{'content':content},'finish_reason':None}]}
        end={'id':'fake-1','model':'fake-model','choices':[{'index':0,
             'delta':{},'finish_reason':'stop'}]}
        return [('data: '+json.dumps(chunk)+'\n\n').encode(),
                ('data: '+json.dumps(end)+'\n\n').encode(),b'data: [DONE]\n\n']
    with FakeHTTPServer(ledger,task_id='headless-fixture',model='fake-model',
                        participant='outer',adapter=adapter) as server:
        result=run_headless(server.base_url,tmp_path/'private','Return the requested Python code only.')
    assert result['returncode']==0, result['stderr']+'\n'+json.dumps(result['events'])
    assert len(calls)==1
    assert content in json.dumps(result['events'],ensure_ascii=False).replace('\\n','\n')
    info=shared.snapshot(ledger)
    assert info['reserved_calls']==1 and info['calls'][0]['api_usage'] is None
