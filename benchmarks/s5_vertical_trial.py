"""One offline DSH headless → fake model → trusted mock L2 vertical trial.

This is a macOS/local fixture, not a sandbox, GPU execution or formal S5 run.
It never forwards requests to a real provider; the only model answer is an
in-process fixed fixture. The returned text must exactly match that fixture
before the trusted mock validator will execute it.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from benchmarks.s5_fake_http import FakeHTTPServer
from benchmarks.s5_vertical_headless import run_headless
from benchmarks.s5_vertical_validation import GOOD_FIXTURE, VerticalValidator
from operator_agent import s5_shared_budget as shared
from operator_agent.run_contract import code_hash

TASK = ('Implement a Python function kernel(X) for layer normalization on the '
        'last axis with epsilon 1e-5. Return only the source code.')
TASK_ID = 'vertical-fixture-01'


def _write(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def run(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    started = time.monotonic()
    result = dict(schema='s5-vertical-trial-v1', status='unknown',
        scope='macOS installed DSH headless, fixed fake model, trusted mock L2',
        formal_s5_credit=False, real_model_called=False, gpu_used=False,
        linux_host_verified=False, isolation_verified=False, provider_usage=None,
        expected_candidate_sha256=code_hash(GOOD_FIXTURE), error=None)
    ledger = output/'budget.sqlite'
    try:
        shared.create(ledger, task_id=TASK_ID, max_calls=1, max_output_tokens=1024,
                      total_output_tokens=1024, wall_clock_s=120)
        def adapter(request):
            if request.get('stream') is not True:
                raise ValueError('fixture expects streamed request')
            delta = {'id':'fake-vertical-1','model':'fake-model','choices':[{'index':0,
                     'delta':{'content':GOOD_FIXTURE},'finish_reason':None}]}
            end = {'id':'fake-vertical-1','model':'fake-model','choices':[{'index':0,
                   'delta':{},'finish_reason':'stop'}]}
            return [('data: '+json.dumps(delta)+'\n\n').encode(),
                    ('data: '+json.dumps(end)+'\n\n').encode(), b'data: [DONE]\n\n']
        with FakeHTTPServer(ledger, task_id=TASK_ID, model='fake-model',
                            participant='outer', adapter=adapter) as server:
            response = run_headless(server.base_url, output/'headless', TASK)
        _write(output/'headless-events.json', response['events'])
        (output/'headless-stderr.txt').write_text(response['stderr'], encoding='utf-8')
        facts = shared.snapshot(ledger, expected_task_id=TASK_ID)
        _write(output/'budget.json', facts)
        finals = [x for x in response['events'] if isinstance(x, dict) and x.get('type') == 'final']
        completed = any(isinstance(x, dict) and x.get('type') == 'status' and x.get('phase') == 'turn_end' and
                        isinstance(x.get('reason'), dict) and x['reason'].get('kind') == 'completed'
                        for x in response['events'])
        if (response['returncode'] != 0 or len(finals) != 1 or not completed or
                facts['reserved_calls'] != 1 or facts['completed_calls'] != 1 or
                facts['calls'][0]['outcome'] != 'answered'):
            raise ValueError('headless result or budget evidence incomplete')
        candidate = finals[0].get('text')
        # The trusted mock evaluator is not a sandbox. Never execute a model
        # response unless it is exactly the known fixture in this script.
        if candidate != GOOD_FIXTURE:
            raise ValueError('returned source differs from the trusted fixture')
        (output/'candidate.py').write_text(candidate, encoding='utf-8')
        verification = VerticalValidator(max_attempts=1).validate(candidate, arm='A')
        _write(output/'validation.json', verification)
        result.update(status=verification['status'],
                      candidate_sha256=verification['candidate_sha256'],
                      manifest_hash=verification['draft_manifest_hash'],
                      contract_hash=verification['contract_hash'],
                      suite_hash=verification['suite_hash'],
                      fake_requests=facts['reserved_calls'],
                      mock_verifications=verification['verification_attempts'])
    except Exception as exc:
        # Error class only: no arbitrary provider data or credentials in summary.
        result['error'] = type(exc).__name__
    result['wall_seconds'] = round(time.monotonic()-started, 3)
    _write(output/'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    result = run(parser.parse_args().output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
