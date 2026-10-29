"""One DSH tool-feedback cycle using only local fake model and mock L2.

Experimental standalone fixture, not the operator plugin, Linux, GPU, a real
model gateway, or a security sandbox. The fake model sends a fixed, trusted
candidate; the validation service refuses to execute all other source text.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from benchmarks.s5_fake_http import FakeHTTPServer
from benchmarks.s5_vertical_tool_evidence import evaluate
from benchmarks.s5_vertical_tool_headless import run_tool_headless
from benchmarks.s5_vertical_tool_provider import ScriptedToolProvider, TOOL_NAME
from benchmarks.s5_vertical_tool_service import MockValidationService
from benchmarks.s5_vertical_validation import GOOD_FIXTURE
from operator_agent import s5_shared_budget as shared
from operator_agent.run_contract import code_hash

TASK_ID='vertical-tool-fixture-01'
TASK=('Validate the supplied Python kernel using validate_kernel. '
      'Use the tool feedback to complete the task.')


def _write(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
        f.write('\n')


def run(output):
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False,mode=0o700)
    started=time.monotonic()
    result=dict(schema='s5-vertical-tool-trial-v1',status='unknown',error=None,
        scope='macOS DSH headless; local scripted fake provider and mock tool',
        real_model_called=False,gpu_used=False,formal_s5_credit=False,
        linux_host_verified=False,isolation_verified=False,operator_plugin_tested=False,
        provider_usage=None)
    ledger=output/'budget.sqlite'
    provider=None; service=None; response=None
    try:
        shared.create(ledger,task_id=TASK_ID,max_calls=2,max_output_tokens=1024,
                      total_output_tokens=2048,wall_clock_s=120)
        with MockValidationService(output/'mock-validation') as service:
            provider=ScriptedToolProvider(service)
            with FakeHTTPServer(ledger,task_id=TASK_ID,model='fake-model',
                                participant='outer',adapter=provider) as server:
                response=run_tool_headless(server.base_url,service.url+'/validate',
                                           output/'headless',TASK)
        _write(output/'headless-events.json',response['events'])
        (output/'headless-stderr.txt').write_text(response['stderr'],encoding='utf-8')
        snapshot=shared.snapshot(ledger,expected_task_id=TASK_ID)
        _write(output/'budget.json',snapshot)
        records=service.records
        observations=[]
        for item in provider.requests:
            row=dict(request_sha256=item['request_sha256'],
                     observed_monotonic=item['observed_monotonic'],tool_feedback=[])
            if 'tool_call_id' in item:
                row['tool_feedback']=[dict(tool_call_id=item['tool_call_id'],
                                           content_sha256=item['tool_reply_sha256'])]
            observations.append(row)
        observed=dict(requests=observations,validation_requests=len(records))
        _write(output/'provider-observations.json',observed)
        if response['returncode'] != 0 or len(records) != 1:
            raise ValueError('headless or single verification incomplete')
        checked=evaluate(response['events'],snapshot,records[0],observed,
                         task_id=TASK_ID,expected_candidate_sha256=code_hash(GOOD_FIXTURE),
                         expected_tool_name=TOOL_NAME)
        _write(output/'evidence-check.json',checked)
        if checked['evidence_status'] != 'complete':
            result['reason']=checked.get('reason')
            raise ValueError('tool evidence incomplete')
        result.update(status='passed',fake_requests=checked['fake_calls'],
            mock_verifications=checked['mock_verifications'],tool_call_id=checked['tool_call_id'],
            candidate_sha256=checked['candidate_sha256'],tool_reply_sha256=checked['tool_reply_sha256'])
    except Exception as exc:
        result['error']=type(exc).__name__
    result['wall_seconds']=round(time.monotonic()-started,3)
    _write(output/'result.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    result=run(p.parse_args().output)
    print(json.dumps(result,ensure_ascii=False,sort_keys=True))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())
