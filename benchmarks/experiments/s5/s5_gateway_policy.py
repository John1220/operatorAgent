"""Experimental shared budget policy for an S5 supervised model gateway.

This component does no network I/O and is not an isolation boundary. An adapter
provided by a trusted supervisor performs the actual transport. The policy is
the *only* shared reservation authority for its requests; enabling another S5
shared interceptor for the same request would double-charge the budget.

The ledger stores a SHA256 of a canonical request, not prompt or credentials.
Every handoff is intent only. A fully consumed byte iterator is still outcome
unknown because the adapter contract does not establish provider completion.
The adapter must itself enforce network destination, timeouts and byte limits.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import threading

from operator_agent.execute import s5_shared_budget as shared
from benchmarks.experiments.s5.s5_gateway_transport import MAX_RESPONSE_BYTES

_ALLOWED = frozenset({'model','messages','max_tokens','max_completion_tokens',
    'stream','stream_options','temperature','tools','tool_choice','n',
    'chat_template_kwargs','enable_thinking','reasoning_effort','store'})
_RESPONSES_ALLOWED = frozenset({'model','input','max_output_tokens','stream','store',
    'tools','tool_choice','temperature','instructions','prompt_cache_key','reasoning','include'})
PROTOCOLS = ('chat_completions','responses')
_EXPECTED = frozenset({'campaign_aborted','wall_clock_exhausted',
    'call_budget_exhausted','output_budget_exhausted','per_call_output_exhausted'})


class PolicyRefused(RuntimeError):
    """Request refused, budget exhausted, or audit failed."""


class PolicyUpstreamFailure(RuntimeError):
    """Trusted transport failed after reservation; provider outcome unknown."""


class GatewayPolicy:
    def __init__(self, ledger_path, *, task_id, model, adapter, protocol='chat_completions',
                 max_request_bytes=1048576, max_response_bytes=1048576, max_chunks=4096,
                 max_calls_limit=2, reasoning_effort=None):
        if (not isinstance(model,str) or not model or len(model)>128 or not callable(adapter)
                or not isinstance(task_id,str) or not task_id):
            raise ValueError('fixed task, model and trusted transport required')
        if protocol not in PROTOCOLS:
            raise ValueError('unknown wire protocol')
        if reasoning_effort is not None:
            if protocol != 'responses' or reasoning_effort not in ('none','low','medium','high'):
                raise ValueError('invalid reasoning effort')
        self.protocol=protocol
        self.last_budget_refusal=None
        self.reasoning_effort=reasoning_effort
        for value, maximum in ((max_request_bytes, 1048576),
                               (max_response_bytes, MAX_RESPONSE_BYTES),
                               (max_chunks, 1048576)):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError('invalid gateway limit')
        self.path=Path(ledger_path); self.task_id=task_id; self.model=model; self.adapter=adapter
        self.max_request_bytes=max_request_bytes; self.max_response_bytes=max_response_bytes
        self.max_chunks=max_chunks; self._failed=threading.Event()
        try:
            before=self._identity()
            snapshot=shared.snapshot(self.path,expected_task_id=self.task_id)
            if self._identity()!=before: raise ValueError('identity changed')
            if (type(max_calls_limit) is not int or not 1 <= max_calls_limit <= 64):
                raise ValueError('invalid call limit')
            if (snapshot['max_calls']>max_calls_limit or snapshot['max_output_tokens']>40960
                    or snapshot['aborted']):
                raise ValueError('unsafe or stopped budget')
        except Exception:
            raise PolicyRefused('ledger unavailable or unsuitable; global stop unproven') from None
        self._pinned=before

    def _identity(self):
        if self.path.is_symlink(): raise ValueError('ledger symlink')
        st=self.path.stat()
        if not self.path.is_file(): raise ValueError('ledger is not a file')
        return (st.st_dev,st.st_ino)

    def _check_identity(self):
        if self._identity()!=self._pinned: raise ValueError('ledger changed')

    def _poison(self):
        self._failed.set()
        try:
            self._check_identity()
            shared.abort(self.path,reason='audit_failure',expected_task_id=self.task_id)
        except Exception:
            raise PolicyRefused('audit failed; global stop unproven') from None
        raise PolicyRefused('audit failed; campaign stop recorded') from None

    def _call(self, operation, **kwargs):
        try:
            self._check_identity()
            result=operation(self.path,expected_task_id=self.task_id,**kwargs)
            self._check_identity()
            return result
        except shared.BudgetError as exc:
            if exc.reason in _EXPECTED:
                self.last_budget_refusal=exc.reason
                raise PolicyRefused('shared budget refused: '+exc.reason) from None
            self._poison()
        except Exception:
            self._poison()

    def _request(self, request, participant):
        if self.protocol=='responses':
            return self._request_responses(request,participant)
        if (participant not in ('outer','inner') or type(request) is not dict
                or set(request)-_ALLOWED or request.get('model') != self.model
                or type(request.get('messages')) is not list or not request['messages']
                or type(request.get('stream',False)) is not bool
                or type(request.get('n',1)) is not int or request.get('n',1) != 1):
            raise PolicyRefused('invalid request')
        caps=[request[key] for key in ('max_tokens','max_completion_tokens') if key in request]
        if len(caps)!=1 or type(caps[0]) is not int or not 0<caps[0]<=40960:
            raise PolicyRefused('explicit bounded output required')
        temp=request.get('temperature')
        if temp is not None and (type(temp) not in (int,float) or not math.isfinite(temp)):
            raise PolicyRefused('invalid request')
        if 'tools' in request and not isinstance(request['tools'],list):
            raise PolicyRefused('invalid request')
        if 'store' in request and request['store'] is not False:
            raise PolicyRefused('invalid request')
        if 'stream_options' in request:
            options=request['stream_options']
            if (not request.get('stream') or type(options) is not dict
                    or set(options)!={'include_usage'} or type(options['include_usage']) is not bool):
                raise PolicyRefused('invalid request')
        if 'enable_thinking' in request and type(request['enable_thinking']) is not bool:
            raise PolicyRefused('invalid request')
        if 'chat_template_kwargs' in request:
            options=request['chat_template_kwargs']
            if (type(options) is not dict or set(options)!={'enable_thinking'}
                    or type(options['enable_thinking']) is not bool or 'enable_thinking' in request):
                raise PolicyRefused('invalid request')
        if any(type(row) is not dict or row.get('role') not in ('system','user','assistant','tool')
               for row in request['messages']):
            raise PolicyRefused('invalid request')
        try:
            raw=json.dumps(request,ensure_ascii=False,sort_keys=True,separators=(',', ':'),
                           allow_nan=False).encode('utf-8')
            if len(raw)>self.max_request_bytes: raise ValueError('request too large')
            detached=json.loads(raw)
        except (TypeError,ValueError,OverflowError,RecursionError,UnicodeError):
            raise PolicyRefused('invalid request') from None
        return detached,caps[0],hashlib.sha256(raw).hexdigest()

    def _request_responses(self, request, participant):
        """Validate an OpenAI Responses request; output cap is still explicit."""
        if (participant not in ('outer','inner') or type(request) is not dict
                or set(request)-_RESPONSES_ALLOWED or request.get('model') != self.model
                or type(request.get('stream',False)) is not bool):
            raise PolicyRefused('invalid request')
        cap=request.get('max_output_tokens')
        if type(cap) is not int or not 0<cap<=40960:
            raise PolicyRefused('explicit bounded output required')
        if 'store' in request and request['store'] is not False:
            raise PolicyRefused('invalid request')
        prompt=request.get('input')
        if isinstance(prompt,str):
            if not prompt.strip(): raise PolicyRefused('invalid request')
        elif isinstance(prompt,list):
            # A follow-up turn replays prior output items (function_call,
            # function_call_output, reasoning) next to role-tagged messages.
            roles=('user','assistant','system','developer','tool')
            items=('message','function_call','function_call_output','reasoning',
                   'item_reference')
            if not prompt or any(type(row) is not dict
                                 or (row.get('role') not in roles and row.get('type') not in items)
                                 for row in prompt):
                raise PolicyRefused('invalid request')
        else:
            raise PolicyRefused('invalid request')
        if 'tools' in request and not isinstance(request['tools'],list):
            raise PolicyRefused('invalid request')
        if 'instructions' in request and not isinstance(request['instructions'],str):
            raise PolicyRefused('invalid request')
        if 'reasoning' in request:
            reasoning=request['reasoning']
            if (type(reasoning) is not dict or set(reasoning)-{'effort','summary'}
                    or reasoning.get('effort') not in ('none','low','medium','high')
                    or ('summary' in reasoning and reasoning['summary'] not in ('auto','concise','detailed'))):
                raise PolicyRefused('invalid request')
        if 'include' in request and request['include'] not in ([], ['reasoning.encrypted_content']):
            raise PolicyRefused('invalid request')
        if 'prompt_cache_key' in request:
            key=request['prompt_cache_key']
            if not isinstance(key,str) or not 0<len(key)<=256:
                raise PolicyRefused('invalid request')
        if 'tools' in request and any(not isinstance(tool,dict) for tool in request['tools']):
            raise PolicyRefused('invalid request')
        temp=request.get('temperature')
        if temp is not None and (type(temp) not in (int,float) or not math.isfinite(temp)):
            raise PolicyRefused('invalid request')
        # The harness may cap reasoning effort for every arm; the ledger hashes
        # the request that is actually dispatched.
        if self.reasoning_effort is not None:
            request=dict(request, reasoning=dict(request.get('reasoning') or {},
                                                 effort=self.reasoning_effort))
        try:
            raw=json.dumps(request,ensure_ascii=False,sort_keys=True,separators=(',',':'),
                           allow_nan=False).encode('utf-8')
            if len(raw)>self.max_request_bytes: raise ValueError('request too large')
            detached=json.loads(raw)
        except (TypeError,ValueError,OverflowError,RecursionError,UnicodeError):
            raise PolicyRefused('invalid request') from None
        return detached,cap,hashlib.sha256(raw).hexdigest()

    def stream(self, request, *, participant):
        detached,cap,request_hash=self._request(request,participant)
        def consume():
            if self._failed.is_set(): raise PolicyRefused('gateway stopped')
            reservation=self._call(shared.reserve,participant=participant,caller='s5.gateway',
                max_output_tokens=cap,request_sha256=request_hash)
            if self._failed.is_set(): raise PolicyRefused('gateway stopped')
            cid=reservation['call_id']
            self._call(shared.dispatch,call_id=cid)
            source=None
            outcome='unknown'
            try:
                if self._failed.is_set(): raise PolicyRefused('gateway stopped')
                source=iter(self.adapter(detached))
                size=count=0
                for chunk in source:
                    if self._failed.is_set(): raise PolicyRefused('gateway stopped')
                    count+=1
                    if type(chunk) is not bytes or count>self.max_chunks:
                        raise ValueError('invalid upstream chunk')
                    size+=len(chunk)
                    if size>self.max_response_bytes: raise ValueError('upstream body exceeds limit')
                    yield chunk
            except GeneratorExit:
                outcome='cancelled'
                raise
            except PolicyRefused:
                raise
            except Exception:
                raise PolicyUpstreamFailure('upstream unavailable; provider outcome unknown') from None
            finally:
                close=getattr(source,'close',None)
                close_failed=False
                if callable(close):
                    try: close()
                    except Exception:
                        close_failed=True
                        outcome='unknown'
                self._call(shared.finish,call_id=cid,outcome=outcome,usage_evidence='missing')
                if close_failed:
                    raise PolicyUpstreamFailure('upstream cleanup unavailable; outcome unknown') from None
        return consume()
