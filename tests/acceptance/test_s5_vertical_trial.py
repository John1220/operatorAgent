"""The single local vertical trial is an offline fixture, not a GPU test."""
import json
import shutil
import urllib.request

import pytest
from benchmarks import s5_vertical_trial as trial
from benchmarks.s5_vertical_headless import DSH_ROOT


@pytest.mark.skipif(not shutil.which('node') or not (DSH_ROOT/'dsh/lib/bin.js').is_file(),
                    reason='installed DSH Desktop unavailable')
def test_actual_headless_fixture_reaches_budgeted_mock_l2(tmp_path):
    out=tmp_path/'run'
    result=trial.run(out)
    assert result['status']=='passed' and result['fake_requests']==1
    assert result['mock_verifications']==1 and result['formal_s5_credit'] is False
    assert result['real_model_called'] is False and result['gpu_used'] is False
    assert result['linux_host_verified'] is False and result['isolation_verified'] is False
    details=json.loads((out/'validation.json').read_text())
    assert len(details['mock_evidence']['inputs'])==3
    assert all(row['full_l2_passed'] is True for row in details['mock_evidence']['inputs'])
    assert json.loads((out/'budget.json').read_text())['calls'][0]['api_usage'] is None


def test_unknown_candidate_is_never_executed(tmp_path, monkeypatch):
    def fake_headless(base_url, workspace, task):
        body=json.dumps({'model':'fake-model','messages':[{'role':'user','content':'x'}],
                         'stream':True,'max_tokens':1024}).encode()
        req=urllib.request.Request(base_url+'/chat/completions',data=body,
                                   headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=5) as response:
            response.read()
        return {'returncode':0,'stderr':'','events':[
            {'type':'status','phase':'turn_end','reason':{'kind':'completed'}},
            {'type':'final','text':'not a trusted fixture'}]}
    monkeypatch.setattr(trial,'run_headless',fake_headless)
    monkeypatch.setattr(trial,'VerticalValidator',lambda **kw:pytest.fail('must not execute unknown code'))
    out=tmp_path/'run'; result=trial.run(out)
    assert result['status']=='unknown' and result['error']=='ValueError'
    assert not (out/'candidate.py').exists() and not (out/'validation.json').exists()
    assert json.loads((out/'budget.json').read_text())['reserved_calls']==1
