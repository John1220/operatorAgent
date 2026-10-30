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
from operator_agent import s5_shared_budget as shared
from operator_agent.hitl_tools import prepare
from operator_agent.run_contract import code_hash


@pytest.mark.skipif(not shutil.which('node') or not (DSH_ROOT/'dsh/lib/bin.js').is_file()
                    or not (DSH_ROOT/'dsh-mcp-client/lib/index.js').is_file(),
                    reason='installed DSH Desktop or MCP client unavailable')
def test_actual_mcp_verify_feedback_through_headless(tmp_path):
    run=(tmp_path/'run').resolve()
    prepare({'request':'layernorm fp32','shapes':[[7,33]],'precision':'fp32'},run)
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
