"""Offline validation service: only trusted fixtures run and budget is spent."""
import json
import urllib.request

from benchmarks.experiments.s5.s5_vertical_tool_service import MockValidationService
from benchmarks.experiments.s5.s5_vertical_validation import GOOD_FIXTURE

#: 这一组驱动的是 s5 的验证管线，它对**归档依赖树**有硬依赖：目录在 ≠ 当前解释器能用
#: （归档那份 numpy 是 cpython-39，换 3.12 跑会走到「不要从源码目录导入」）。
#: 所以按**探针**跳过，而不是让 20 条用例各自以「输出不是 JSON」的形式红。
import pytest as _pytest  # noqa: E402
from benchmarks.experiments.s5 import s5_runtime as _s5_runtime  # noqa: E402
if not _s5_runtime.available():
    pytestmark = _pytest.mark.skipif(
        True, reason='s5 runtime deps 不可用: ' + '; '.join(_s5_runtime.deps_problems())
        + '。冻结的那棵树是按 Python 3.9 建的，用当前解释器重建一次即可：'
        + 'python -m benchmarks.experiments.s5.s5_rebuild_deps')


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
    from benchmarks.experiments.s5 import s5_l2_evaluator
    monkeypatch.setattr(s5_l2_evaluator,'evaluate_mock_fixture',
                        lambda *a,**k: (_ for _ in ()).throw(AssertionError('must not execute')))
    with MockValidationService(tmp_path/'evidence') as service:
        result=post(service.url,'not an approved fixture')
        assert result['status']=='unknown' and result['verification_attempts']==1
        assert post(service.url,GOOD_FIXTURE)['status']=='budget_exhausted'
