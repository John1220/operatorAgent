"""Scripted fake model for one real operator MCP verification.

Only a fixed Python fixture, a pre-prepared run directory, mock backend and L2
are ever sent to the real tool. No actual model or GPU is used.
"""
from __future__ import annotations
import hashlib
import json
import threading
from pathlib import Path

from benchmarks.experiments.s5.s5_vertical_tool_provider import _hash, _sse
from benchmarks.experiments.s5.s5_vertical_validation import GOOD_FIXTURE
from operator_agent.execute.task_clock import monotonic

CALL_ID='call_operator_verify_01'
TOOL_NAME='mcp__operator__operator_verify'
FINAL_TEXT='Mock operator verification feedback received.'


class ScriptedMCPProvider:
    def __init__(self, run_dir):
        self.run_dir=Path(run_dir).resolve()
        self.requests=[]
        self._lock=threading.Lock()

    def __call__(self, request):
        with self._lock:
            number=len(self.requests)+1
            record=dict(sequence=number,request_sha256=_hash(request),observed_monotonic=monotonic())
            self.requests.append(record)
            if request.get('stream') is not True:
                raise ValueError('stream required')
            if number == 1:
                args=dict(run_dir=str(self.run_dir),code=GOOD_FIXTURE,backend='mock',stop_after='L2')
                call=[dict(index=0,id=CALL_ID,type='function',
                           function=dict(name=TOOL_NAME,arguments=json.dumps(args)))]
                return _sse({'tool_calls':call},'tool_calls')
            if number != 2:
                raise ValueError('unexpected extra request')
            messages=request.get('messages')
            if not isinstance(messages,list): raise ValueError('missing messages')
            replies=[m for m in messages if isinstance(m,dict) and m.get('role')=='tool']
            if len(replies)!=1 or replies[0].get('tool_call_id')!=CALL_ID:
                raise ValueError('tool reply identity mismatch')
            text=replies[0].get('content')
            if not isinstance(text,str): raise ValueError('tool reply text unavailable')
            # The real tool must have persisted a round before the final fake
            # answer. A later evidence check validates its actual contents.
            if not (self.run_dir/'verify-1'/'complete.json').is_file():
                raise ValueError('operator round not complete')
            record.update(tool_call_id=CALL_ID,tool_reply_sha256=hashlib.sha256(text.encode()).hexdigest())
            return _sse({'content':FINAL_TEXT},'stop')
