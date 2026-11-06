"""The host gets materials and verifies its own source, with no inner model."""
import io
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from operator_agent.app import hitl_tools as ht, mcp_server as mcp
from operator_agent.app.host_workflow import context
from operator_agent.core.run_contract import code_hash


DECL = dict(request='relu fp32', shapes=[[4, 8]], precision='fp32', dsl='python')
WRONG = 'import numpy as np\ndef kernel(X):\n    return X.copy()\n'
CORRECT = 'import numpy as np\ndef kernel(X):\n    return np.maximum(X, 0)\n'


def tool(name, **args):
    result = mcp._call_tool(name, args)
    return json.loads(result['content'][0]['text'])


def test_host_generates_repairs_and_independently_verifies_without_inner_llm(tmp_path, monkeypatch):
    from operator_agent.execute import llm_client
    monkeypatch.setenv('OPERATOR_AGENT_WORKFLOW', 'host')
    monkeypatch.setattr(llm_client, 'LLMClient', lambda **kw: pytest.fail('host flow called an inner model'))
    run = str(tmp_path/'task')
    prepared = tool('operator_prepare', **DECL, run_dir=run,
                    host_budget=dict(max_verifications=3, wall_clock_s=30))
    materials = tool('operator_context', run_dir=run, backend='fake')
    assert materials['context_owner'] == 'host' and materials['model_calls'] == 0
    assert materials['context']['authoritative']['task']['declared_shapes'] == [[4, 8]]
    assert 'Import only numpy/math.' in json.dumps(materials)
    failed = tool('operator_verify', run_dir=run, code=WRONG, backend='fake', stop_after='L2', candidate_origin='outer')
    assert failed['levels']['L2'] is False
    stopped = tool('operator_status', run_dir=run)
    assert stopped['last_correct_candidate'] is None
    assert stopped['latest_verification']['source_hash'] == code_hash(WRONG)
    assert Path(stopped['latest_verification']['report']).is_file()
    repair = tool('operator_context', run_dir=run, backend='fake', phase='repair', verification_round=failed['round'])
    assert repair['context']['candidate']['source'] == WRONG
    assert repair['context']['candidate']['source_hash'] == code_hash(WRONG)
    correct = tool('operator_verify', run_dir=run, code=CORRECT, backend='fake', stop_after='L2', candidate_origin='outer')
    assert correct['levels']['L2'] is True
    assert correct['host_task']['remaining_exploration'] == 0
    with pytest.raises(ht.ToolError, match='host_verification_budget'):
        tool('operator_verify', run_dir=run, code=CORRECT, backend='fake')
    final = tool('operator_verify', run_dir=run, code=CORRECT, backend='fake', stop_after='L2',
                 candidate_origin='outer', verification_purpose='final')
    assert final['contract_hash'] == prepared['contract_hash']
    assert final['levels']['L2'] is True and final['evidence'] != correct['evidence']
    assert final['host_task']['remaining_final'] == 0 and final['host_task']['reserved'] == 3
    recovered = tool('operator_status', run_dir=run)
    assert recovered['last_correct_candidate']['source_hash'] == code_hash(CORRECT)
    assert recovered['host_task']['model_budget']['enforcement'] == 'not_bound'
    assert not (Path(run)/'model-journal').exists()


def test_host_surface_and_stale_direct_calls_refuse_inner_tools_before_worker(tmp_path, monkeypatch):
    from operator_agent.app import tool_calls
    monkeypatch.setenv('OPERATOR_AGENT_WORKFLOW', 'host')
    assert {t['name'] for t in mcp._tool_definitions()} == mcp.HOST_TOOLS
    instructions = mcp.handle_message(dict(jsonrpc='2.0', id=1, method='initialize'))['result']['instructions']
    assert instructions == mcp.HOST_INSTRUCTIONS and 'operator_context' in instructions
    monkeypatch.setattr(tool_calls, '_start_call_worker', lambda *a: pytest.fail('legacy worker spawned'))
    requests = [dict(jsonrpc='2.0', id=i, method='tools/call',
        params=dict(name=name, arguments=dict(run_dir=str(tmp_path/'not-prepared'))))
        for i, name in enumerate(('operator_run', 'operator_repair', 'operator_bench'), start=1)]
    output = io.StringIO()
    mcp.serve(io.StringIO(''.join(json.dumps(r)+'\n' for r in requests)), output)
    results = [json.loads(line)['result'] for line in output.getvalue().splitlines()]
    assert len(results) == 3
    assert all(r['isError'] and json.loads(r['content'][0]['text'])['error'] == 'host_workflow_only' for r in results)


def test_host_rejects_null_budget_and_unmanaged_legacy_directories(tmp_path, monkeypatch):
    legacy = tmp_path/'legacy'
    ht.prepare(DECL, legacy)
    monkeypatch.setenv('OPERATOR_AGENT_WORKFLOW', 'host')
    with pytest.raises(ht.ToolError, match='invalid_host_budget'):
        tool('operator_prepare', **DECL, run_dir=str(tmp_path/'null-budget'), host_budget=None)
    assert not (tmp_path/'null-budget').exists()
    for name, arguments in [('operator_verify', dict(code=CORRECT, backend='fake')),
                            ('operator_context', dict(backend='fake'))]:
        with pytest.raises(ht.ToolError, match='host_task_required'):
            tool(name, run_dir=str(legacy), **arguments)


def test_host_prepare_cannot_reset_or_rewrite_task_and_legacy_cannot_bypass(tmp_path):
    run = tmp_path/'task'
    first = ht.prepare(DECL, run, host_budget=dict(max_verifications=2))
    ht.verify(run, CORRECT, backend='fake', stop_after='L2')
    repeated = ht.prepare(DECL, run)
    assert repeated['host_task']['task_id'] == first['host_task']['task_id']
    assert repeated['host_task']['reserved'] == 1
    original = (run/'contract.json').read_bytes()
    with pytest.raises(ht.ToolError, match='host_task_binding'):
        ht.prepare(dict(DECL, shapes=[[8, 8]]), run, host_budget={})
    assert (run/'contract.json').read_bytes() == original
    for call in (lambda: ht.repair(run, WRONG), lambda: ht.bench(run, []),
                 lambda: ht.run_autonomous(DECL, run)):
        with pytest.raises(ht.ToolError, match='host_workflow_only'):
            call()


def test_repair_context_refuses_mutated_or_incomplete_evidence(tmp_path):
    ht.prepare(DECL, tmp_path)
    result = ht.verify(tmp_path, WRONG, backend='fake', stop_after='L2')
    directory = Path(result['evidence']).parent
    (directory/'candidate.py').chmod(0o600)
    (directory/'candidate.py').write_text(CORRECT)
    with pytest.raises(ht.ToolError, match='verification_refused'):
        context(tmp_path, backend='fake', phase='repair', verification_round=result['round'])


def test_host_tool_supervisor_enforces_task_deadline_and_reaps_worker(tmp_path, monkeypatch):
    from operator_agent.app import tool_calls
    from operator_agent.execute import exec_guard
    ht.prepare(DECL, tmp_path, host_budget=dict(wall_clock_s=1, max_verifications=2))
    monkeypatch.setattr(tool_calls, '_start_call_worker', lambda *a: subprocess.Popen(
        [sys.executable, '-c', 'import time;time.sleep(60)'], start_new_session=True))
    call = tool_calls.ToolCall('operator_verify', dict(run_dir=str(tmp_path), code=CORRECT, backend='fake'))
    try:
        end = time.monotonic()+8
        reply = None
        while reply is None and time.monotonic() < end:
            reply = call.poll()
            time.sleep(.02)
        assert reply is not None
        value = json.loads(reply['content'][0]['text'])
        assert value['error'] == 'task_budget_exhausted'
        assert value['host_task']['remaining_s'] == 0
        assert call.status['cleanup']['worker_reaped'] is True
        assert call.process.poll() is not None
    finally:
        call.close()


def test_cancelled_worker_settles_only_its_private_reservation_and_allows_final(tmp_path, monkeypatch):
    from operator_agent.app import tool_calls
    ht.prepare(DECL, tmp_path, host_budget=dict(wall_clock_s=30, max_verifications=3))
    script = '''import json,sys,time
from pathlib import Path
from operator_agent.app import hitl_tools as ht,tool_calls
directory=Path(sys.argv[1]);request=json.loads((directory/'request.json').read_text())
tool_calls._HOST_CONTEXT=request['host_budget_context']
def blocked(*args,**kwargs):
 (directory/'ready').touch();time.sleep(60)
ht._verify_candidate=blocked
ht.verify(**request['arguments'])
'''
    monkeypatch.setattr(tool_calls, '_start_call_worker', lambda directory: subprocess.Popen(
        [sys.executable, '-c', script, str(directory)], start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    call = tool_calls.ToolCall('operator_verify', dict(run_dir=str(tmp_path), code=CORRECT, backend='fake'))
    try:
        end = time.monotonic()+10
        while not (call.directory/'ready').exists() and time.monotonic() < end:
            assert call.process.poll() is None
            time.sleep(.02)
        assert (call.directory/'ready').exists()
        assert call.host_budget.snapshot()['unfinished'] == 1
        call.cancel()
        reply = None
        while reply is None and time.monotonic() < end:
            reply = call.poll()
            time.sleep(.02)
        assert reply is not None and call.status['cleanup']['worker_reaped'] is True
        snapshot = call.host_budget.snapshot()
        assert snapshot['reserved'] == 1 and snapshot['unfinished'] == 0
        assert snapshot['records'][0]['status'] == 'cancelled'
        assert snapshot['records'][0]['owner_invocation_id'] == call.invocation_id
        ht.verify(tmp_path, CORRECT, backend='fake', stop_after='L2')
        final = ht.verify(tmp_path, CORRECT, backend='fake', stop_after='L2', verification_purpose='final')
        assert final['host_task']['phase'] == 'final_complete'
    finally:
        call.close()
