"""Installed DSH headless and repository MCP, with local fake model only."""
import hashlib
import json
import shutil

import pytest
from benchmarks.experiments.s5.s5_fake_http import FakeHTTPServer
from benchmarks.experiments.s5.s5_vertical_headless import DSH_ROOT
from benchmarks.experiments.s5.s5_vertical_mcp_headless import run_mcp_headless
from benchmarks.experiments.s5.s5_vertical_mcp_provider import ScriptedMCPProvider, CALL_ID, TOOL_NAME
from benchmarks.experiments.s5.s5_vertical_validation import GOOD_FIXTURE
from operator_agent.execute import s5_shared_budget as shared
from operator_agent.app.hitl_tools import prepare
from operator_agent.core.run_contract import code_hash

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


@pytest.mark.skipif(not shutil.which('node') or not (DSH_ROOT/'dsh/lib/bin.js').is_file()
                    or not (DSH_ROOT/'dsh-mcp-client/lib/index.js').is_file(),
                    reason='installed DSH Desktop or MCP client unavailable')
def test_actual_mcp_verify_feedback_through_headless(tmp_path):
    run=(tmp_path/'run').resolve()
    prepare({"operator": 'layernorm', 'request':'layernorm fp32','shapes':[[7,33]],'precision':'fp32'},run)
    path=tmp_path/'ledger.sqlite'
    shared.create(path,task_id='fixture',max_calls=2,max_output_tokens=1024,
                  total_output_tokens=2048,wall_clock_s=120)
    provider=ScriptedMCPProvider(run)
    with FakeHTTPServer(path,task_id='fixture',model='fake-model',participant='outer',
                        adapter=provider) as server:
        result=run_mcp_headless(server.base_url,tmp_path/'workspace',
                                'Verify the candidate and report the result.')
    assert result['returncode']==0, result['stderr']
    events=result['events']
    calls=[x for x in events if x.get('type')=='tool_call']
    replies=[x for x in events if x.get('type')=='tool_result']
    assert len(calls)==len(replies)==1
    assert calls[0]['callId']==replies[0]['callId']==CALL_ID
    assert calls[0]['tool']==TOOL_NAME and calls[0]['input']['code']==GOOD_FIXTURE
    assert calls[0]['input']['backend']=='mock' and calls[0]['input']['stop_after']=='L2'
    assert replies[0]['status']=='completed'
    data=json.loads(replies[0]['result'])
    assert data['ok'] and data['passed'] and data['levels']=={'L1':True,'L2':True}
    assert data['source_hash']==code_hash(GOOD_FIXTURE)
    assert data['backend_resolution']['resolved']=='MockGPU'
    assert len(provider.requests)==2 and provider.requests[1]['tool_call_id']==CALL_ID
    assert hashlib.sha256(replies[0]['result'].encode()).hexdigest()==provider.requests[1]['tool_reply_sha256']
    budget=shared.snapshot(path)
    assert budget['reserved_calls']==budget['completed_calls']==2
    assert all(c['api_usage'] is None for c in budget['calls'])
    assert (run/'verify-1'/'complete.json').is_file()
