"""Offline DSH → repository MCP operator_verify → mock L2 → DSH feedback.

Only a scripted fake model and a fixed trusted Python fixture are allowed.
This macOS fixture is not Linux isolation, a real model or GPU test, or a
formal S5 comparison. The supervisor prepares the run before the DSH session.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from benchmarks.s5_fake_http import FakeHTTPServer
from benchmarks.s5_vertical_mcp_headless import run_mcp_headless
from benchmarks.s5_vertical_mcp_provider import ScriptedMCPProvider, TOOL_NAME
from operator_agent import s5_shared_budget as shared
from operator_agent.hitl_tools import prepare

TASK_ID='vertical-mcp-fixture-01'
DECLARATION=dict(request='layernorm fp32',shapes=[[7,33]],hardware='mock_gpu',
                 dsl='python',precision='fp32')
TASK=('Use the operator MCP tool to verify the supplied Python kernel against '
      'the prepared mock task. Use the tool result in the final reply.')


def _write(path,value):
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
        f.write('\n')


def run(output):
    from benchmarks.s5_vertical_mcp_evidence import evaluate
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False,mode=0o700)
    start=time.monotonic()
    result=dict(schema='s5-vertical-mcp-trial-v1',status='unknown',error=None,
                scope='macOS DSH headless, repository plugin and MCP, mock backend',
                real_model_called=False,gpu_used=False,formal_s5_credit=False,
                linux_host_verified=False,isolation_verified=False,provider_usage=None)
    try:
        run_dir=output/'operator-run'
        prepared=prepare(DECLARATION,run_dir)
        _write(output/'prepare.json',prepared)
        ledger=output/'budget.sqlite'
        shared.create(ledger,task_id=TASK_ID,max_calls=2,max_output_tokens=1024,
                      total_output_tokens=2048,wall_clock_s=180)
        provider=ScriptedMCPProvider(run_dir)
        with FakeHTTPServer(ledger,task_id=TASK_ID,model='fake-model',
                            participant='outer',adapter=provider) as server:
            response=run_mcp_headless(server.base_url,output/'headless',TASK,timeout=120)
        _write(output/'headless-events.json',response['events'])
        (output/'headless-stderr.txt').write_text(response['stderr'],encoding='utf-8')
        facts=shared.snapshot(ledger,expected_task_id=TASK_ID)
        _write(output/'budget.json',facts)
        observations=[]
        for item in provider.requests:
            row=dict(request_sha256=item['request_sha256'],
                     observed_monotonic=item['observed_monotonic'],tool_feedback=[])
            if 'tool_call_id' in item:
                row['tool_feedback']=[dict(tool_call_id=item['tool_call_id'],
                                           content_sha256=item['tool_reply_sha256'])]
            observations.append(row)
        observations=dict(requests=observations)
        _write(output/'provider-observations.json',observations)
        if response['returncode']!=0:
            raise ValueError('headless failed')
        checked=evaluate(response['events'],facts,observations,task_id=TASK_ID,
                         run_dir=run_dir,expected_tool_name=TOOL_NAME,
                         expected_contract_hash=prepared['contract_hash'])
        _write(output/'evidence-check.json',checked)
        if checked.get('evidence_status')!='complete':
            result['reason']=checked.get('reason')
            raise ValueError('MCP evidence incomplete')
        result.update(status='passed',fake_requests=2,mock_verifications=1,
            contract_hash=prepared['contract_hash'],candidate_sha256=checked['candidate_sha256'],
            tool_call_id=checked['tool_call_id'],tool_reply_sha256=checked['tool_reply_sha256'])
    except Exception as exc:
        result['error']=type(exc).__name__
    result['wall_seconds']=round(time.monotonic()-start,3)
    _write(output/'result.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    result=run(p.parse_args().output)
    print(json.dumps(result,ensure_ascii=False,sort_keys=True))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())
