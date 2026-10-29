"""Scripted fake model for the local two-turn DSH tool experiment.

The second reply is produced only when the actual model request contains the
matching tool result returned by our mock service. No model inference or real
provider is involved; the fixed candidate is a trusted Python fixture.
"""
from __future__ import annotations

import hashlib
import json
import threading

from benchmarks.s5_vertical_validation import GOOD_FIXTURE
from operator_agent.task_clock import monotonic

CALL_ID = 'call_s5_mock_01'
TOOL_NAME = 'validate_kernel'
FINAL_TEXT = 'Mock validation feedback received.'


def _hash(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                         allow_nan=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _sse(delta, reason):
    chunk = {'id':'fake-tool-1','model':'fake-model','choices':[{'index':0,
             'delta':delta,'finish_reason':None}]}
    end = {'id':'fake-tool-1','model':'fake-model','choices':[{'index':0,
           'delta':{},'finish_reason':reason}]}
    return [('data: '+json.dumps(chunk)+'\n\n').encode(),
            ('data: '+json.dumps(end)+'\n\n').encode(), b'data: [DONE]\n\n']


class ScriptedToolProvider:
    def __init__(self, service):
        self.service = service
        self.requests = []
        self._lock = threading.Lock()

    def __call__(self, request):
        with self._lock:
            number = len(self.requests) + 1
            record = dict(sequence=number, request_sha256=_hash(request),
                          observed_monotonic=monotonic())
            self.requests.append(record)
            if request.get('stream') is not True:
                raise ValueError('stream required')
            if number == 1:
                calls = [{'index':0,'id':CALL_ID,'type':'function',
                          'function':{'name':TOOL_NAME,
                                      'arguments':json.dumps({'source':GOOD_FIXTURE})}}]
                return _sse({'tool_calls':calls}, 'tool_calls')
            if number != 2:
                raise ValueError('unexpected extra fake model request')
            records = self.service.records
            if len(records) != 1:
                raise ValueError('one validation record required before second request')
            messages = request.get('messages')
            if not isinstance(messages, list):
                raise ValueError('missing messages')
            replies = [m for m in messages if isinstance(m, dict) and m.get('role') == 'tool']
            if len(replies) != 1 or replies[0].get('tool_call_id') != CALL_ID:
                raise ValueError('tool response identity mismatch')
            text = replies[0].get('content')
            if not isinstance(text, str):
                raise ValueError('tool response text unavailable')
            text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
            if text_hash != records[0]['response_sha256']:
                raise ValueError('tool feedback changed before model request')
            record.update(tool_call_id=CALL_ID, tool_reply_sha256=text_hash)
            return _sse({'content':FINAL_TEXT}, 'stop')
