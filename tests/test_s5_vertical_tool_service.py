"""Offline validation service: only trusted fixtures run and budget is spent."""
import json
import urllib.request

from benchmarks.s5_vertical_tool_service import MockValidationService
from benchmarks.s5_vertical_validation import GOOD_FIXTURE


def post(url, source):
    req=urllib.request.Request(url+'/validate',data=json.dumps({'source':source}).encode(),
                               headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=10) as response:
        return json.loads(response.read())


def test_service_records_one_complete_mock_result_and_refuses_second(tmp_path):
    out=tmp_path/'evidence'
    with MockValidationService(out) as service:
        result=post(service.url,GOOD_FIXTURE)
        assert result['status']=='passed' and result['verification_attempts']==1
        assert post(service.url,GOOD_FIXTURE)['status']=='budget_exhausted'
        assert len(service.records)==2
    assert (out/'request-001.json').is_file() and (out/'result-001.json').is_file()


def test_unknown_source_spends_budget_but_is_never_executed(tmp_path, monkeypatch):
    from benchmarks import s5_l2_evaluator
    monkeypatch.setattr(s5_l2_evaluator,'evaluate_mock_fixture',
                        lambda *a,**k: (_ for _ in ()).throw(AssertionError('must not execute')))
    with MockValidationService(tmp_path/'evidence') as service:
        result=post(service.url,'not an approved fixture')
        assert result['status']=='unknown' and result['verification_attempts']==1
        assert post(service.url,GOOD_FIXTURE)['status']=='budget_exhausted'
