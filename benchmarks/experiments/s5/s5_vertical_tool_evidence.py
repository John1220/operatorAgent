"""Cross-check a local fake-provider/tool-feedback vertical trial.

The observations are supplied by the test supervisor, not an independent
security boundary. This checker cannot prove process isolation, HTTP delivery,
provider billing or provenance. It requires mutually consistent local records
and never judges a model's final answer as evidence of verification.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

from benchmarks.experiments.s5.s5_vertical_validation import GOOD_FIXTURE
from operator_agent.core.run_contract import code_hash

HASH = re.compile(r'[a-f0-9]{64}\Z')


def _hash(value):
    return isinstance(value, str) and HASH.fullmatch(value) is not None


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _canonical_response(result):
    value = dict(status=result['status'], candidate_sha256=result['candidate_sha256'],
        verification_attempts=result['verification_attempts'], mock_only=True,
        formal_s5_credit=False)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False)


def evaluate(events, ledger, validation_record, observations, *, task_id,
             expected_candidate_sha256, expected_tool_name):
    """Check local records; any missing/inconsistent evidence remains unknown.

    ``validation_record`` is the trusted mock service's sole persisted record.
    ``observations`` is {'validation_requests': 1, 'requests': [
    {'request_sha256', 'observed_monotonic', 'tool_feedback':
    [{'tool_call_id', 'content_sha256'}]}, ...]}. Request
    identities are computed from the exact canonical requests supplied to the
    fake adapter; second request feedback hashes the raw UTF-8 tool text.
    """
    result = dict(schema='s5-vertical-tool-evidence-v1', evidence_status='unknown',
        reason=None, formal_s5_credit=False, real_model_called=False, gpu_used=False,
        isolation_verified=False, provider_usage=None,
        scope='supervisor-provided offline fixture records; not independent provenance')
    def unknown(reason):
        result['reason'] = reason
        return result
    try:
        if (not isinstance(events, list) or not isinstance(ledger, dict)
                or not isinstance(validation_record, dict) or not isinstance(observations, dict)
                or not isinstance(task_id, str) or not isinstance(expected_tool_name, str)
                or not _hash(expected_candidate_sha256)):
            return unknown('invalid evidence')
        if expected_candidate_sha256 != code_hash(GOOD_FIXTURE):
            return unknown('fixture identity mismatch')
        if observations.get('validation_requests') != 1 or type(observations.get('validation_requests')) is not int:
            return unknown('validation request count incomplete')
        if (ledger.get('task_id') != task_id or ledger.get('schema') != 's5-shared-call-budget-v1'
                or ledger.get('aborted') is not False or ledger.get('max_calls') != 2
                or ledger.get('reserved_calls') != 2 or ledger.get('completed_calls') != 2
                or ledger.get('network_interception_proven') is not False):
            return unknown('budget evidence incomplete')
        calls = ledger.get('calls')
        requests = observations.get('requests')
        if not isinstance(calls, list) or len(calls) != 2 or not isinstance(requests, list) or len(requests) != 2:
            return unknown('two calls not established')
        for call, request in zip(calls, requests):
            if (not isinstance(call, dict) or not isinstance(request, dict)
                    or call.get('participant') != 'outer' or call.get('outcome') != 'answered'
                    or call.get('api_usage') is not None or not _hash(call.get('request_sha256'))
                    or request.get('request_sha256') != call['request_sha256']
                    or not all(_number(call.get(k)) for k in ('reserved_monotonic', 'dispatched_monotonic', 'completed_monotonic'))
                    or not _number(request.get('observed_monotonic'))
                    or not (call['reserved_monotonic'] <= call['dispatched_monotonic'] <=
                            request['observed_monotonic'] <= call['completed_monotonic'])):
                return unknown('model request identity or ordering incomplete')
        if calls[0].get('call_id') == calls[1].get('call_id'):
            return unknown('duplicate model call identity')
        v = validation_record.get('result')
        request_bytes = json.dumps({'source': GOOD_FIXTURE}, ensure_ascii=False,
            separators=(',', ':'), allow_nan=False).encode('utf-8')
        expected_request_hash = hashlib.sha256(request_bytes).hexdigest()
        if (validation_record.get('sequence') != 1 or not isinstance(v, dict)
                or validation_record.get('request_sha256') != expected_request_hash
                or not _number(validation_record.get('start_monotonic'))
                or not _number(validation_record.get('end_monotonic'))
                or not (calls[0]['completed_monotonic'] <= validation_record['start_monotonic'] <=
                        validation_record['end_monotonic'] <= calls[1]['reserved_monotonic'])
                or v.get('schema') != 's5-vertical-mock-validation-v1'
                or v.get('status') != 'passed' or v.get('candidate_sha256') != expected_candidate_sha256
                or v.get('verification_attempts') != 1 or v.get('max_verification_attempts') != 1
                or type(v.get('verification_attempts')) is not int
                or type(v.get('max_verification_attempts')) is not int
                or v.get('mock_only') is not True or v.get('formal_s5_credit') is not False
                or v.get('gpu_used') is not False or v.get('untrusted_source_executed') is not False):
            return unknown('mock validation evidence incomplete')
        detail = v.get('mock_evidence')
        if (not isinstance(detail, dict) or detail.get('full_l2_passed') is not True
                or detail.get('source_hash') != expected_candidate_sha256
                or detail.get('draft_contract_hash') != v.get('contract_hash')
                or detail.get('suite_hash') != v.get('suite_hash')
                or not all(_hash(v.get(k)) for k in ('contract_hash','suite_hash','draft_manifest_hash'))):
            return unknown('mock validation identity mismatch')
        inputs = detail.get('inputs')
        expected_inputs = ('layernorm-tail-primary','layernorm-tail-holdout-317','layernorm-tail-holdout-619')
        if (not isinstance(inputs, list) or len(inputs) != 3 or
                tuple(x.get('case_id') for x in inputs if isinstance(x, dict)) != expected_inputs or
                any(x.get('full_l2_passed') is not True or x.get('source_hash') != expected_candidate_sha256 for x in inputs)):
            return unknown('mock input evidence incomplete')
        tool_calls = [(i,x) for i,x in enumerate(events) if isinstance(x, dict) and x.get('type') == 'tool_call']
        tool_results = [(i,x) for i,x in enumerate(events) if isinstance(x, dict) and x.get('type') == 'tool_result']
        finals = [(i,x) for i,x in enumerate(events) if isinstance(x, dict) and x.get('type') == 'final']
        if len(tool_calls) != 1 or len(tool_results) != 1 or len(finals) != 1:
            return unknown('agent tool event count incomplete')
        call_index, tool_call = tool_calls[0]
        result_index, tool_result = tool_results[0]
        call_id = tool_call.get('callId')
        if (not isinstance(call_id, str) or not call_id or tool_call.get('tool') != expected_tool_name
                or tool_call.get('input') != {'source': GOOD_FIXTURE}
                or tool_result.get('callId') != call_id or tool_result.get('status') != 'completed'
                or not call_index < result_index < finals[0][0]):
            return unknown('agent tool call or result identity mismatch')
        if not any(isinstance(x, dict) and x.get('type') == 'status' and x.get('phase') == 'step_start'
                   for x in events[result_index+1:finals[0][0]]):
            return unknown('agent next step not observed')
        if not any(isinstance(x, dict) and x.get('type') == 'status' and x.get('phase') == 'turn_end'
                   and isinstance(x.get('reason'), dict) and x['reason'].get('kind') == 'completed'
                   for x in events[result_index+1:finals[0][0]]):
            return unknown('agent turn not completed')
        response = _canonical_response(v)
        response_hash = hashlib.sha256(response.encode('utf-8')).hexdigest()
        if (validation_record.get('response_sha256') != response_hash
                or tool_result.get('result') != response):
            return unknown('validation feedback identity mismatch')
        if requests[0].get('tool_feedback') != []:
            return unknown('unexpected initial tool feedback')
        if requests[1].get('tool_feedback') != [dict(tool_call_id=call_id, content_sha256=response_hash)]:
            return unknown('next request did not contain matching feedback')
        result.update(evidence_status='complete', reason='local cross-checks complete',
                      tool_call_id=call_id, candidate_sha256=expected_candidate_sha256,
                      tool_reply_sha256=response_hash, fake_calls=2, mock_verifications=1)
        return result
    except (TypeError, ValueError, KeyError, UnicodeError, OverflowError):
        return unknown('malformed evidence')
