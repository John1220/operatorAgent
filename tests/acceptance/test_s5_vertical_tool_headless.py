"""Actual DSH headless tool invocation with trusted, local fixtures only."""
import json
import hashlib
import shutil
import pytest
from benchmarks.experiments.s5.s5_fake_http import FakeHTTPServer
from benchmarks.experiments.s5.s5_vertical_headless import DSH_ROOT
from benchmarks.experiments.s5.s5_vertical_tool_headless import run_tool_headless
from benchmarks.experiments.s5.s5_vertical_tool_service import MockValidationService
from benchmarks.experiments.s5.s5_vertical_validation import GOOD_FIXTURE
from operator_agent.execute import s5_shared_budget as shared

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


def chunk(delta, reason=None):
    event={'id':'fake-1','model':'fake-model','choices':[{'index':0,'delta':delta,'finish_reason':reason}]}
    return ('data: '+json.dumps(event)+'\n\n').encode()


@pytest.mark.skipif(not shutil.which('node') or not (DSH_ROOT/'dsh/lib/bin.js').is_file(),
                    reason='installed DSH Desktop unavailable')
def test_installed_headless_calls_validation_and_continues(tmp_path):
    ledger=tmp_path/'ledger.sqlite'
    shared.create(ledger,task_id='fixture',max_calls=2,max_output_tokens=1024,
                  total_output_tokens=2048,wall_clock_s=120)
    requests=[]
    def adapter(req):
        requests.append(req)
        if len(requests)==1:
            delta={'tool_calls':[{'index':0,'id':'call_fixture_1','type':'function',
                      'function':{'name':'validate_kernel','arguments':json.dumps({'source':GOOD_FIXTURE})}}]}
            return [chunk(delta),chunk({},'tool_calls'),b'data: [DONE]\n\n']
        assert len(requests)==2
        tools=[m for m in req['messages'] if m.get('role')=='tool']
        assert len(tools)==1 and tools[0].get('tool_call_id')=='call_fixture_1'
        content=tools[0]['content']
        assert json.loads(content)['status']=='passed'
        return [chunk({'content':'validation passed'}),chunk({},'stop'),b'data: [DONE]\n\n']
    with MockValidationService(tmp_path/'validation') as validator, FakeHTTPServer(
            ledger,task_id='fixture',model='fake-model',participant='outer',adapter=adapter) as provider:
        result=run_tool_headless(provider.base_url,validator.url+'/validate',tmp_path/'workspace',
                                 'Validate this Python fixture and report the result.')
        records=validator.records
    assert result['returncode']==0, result['stderr']+'\n'+json.dumps(result['events'])
    assert len(requests)==2 and len(records)==1
    assert records[0]['result']['status']=='passed'
    tool_events=[x for x in result['events'] if x.get('type')=='tool_result']
    assert len(tool_events)==1 and tool_events[0]['callId']=='call_fixture_1'
    assert tool_events[0]['status']=='completed'
    text=tool_events[0]['result']
    assert isinstance(text,str) and hashlib.sha256(text.encode()).hexdigest()==records[0]['response_sha256']
    tool_message=next(x for x in requests[1]['messages'] if x.get('role')=='tool')
    assert tool_message['content']==text and tool_message['tool_call_id']=='call_fixture_1'
    assert any(x.get('type')=='status' and x.get('phase')=='step_start' and x.get('step')==2
               for x in result['events'])
    assert any(x.get('type')=='final' and x.get('text')=='validation passed' for x in result['events'])
    assert shared.snapshot(ledger)['reserved_calls']==2
