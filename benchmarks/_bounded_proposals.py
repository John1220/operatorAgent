"""Single-endpoint, no-retry model transport for a frozen acceptance campaign.

Not the production client's fallback policy. Calls (including failures) are
reserved durably BEFORE dispatch. A disposable child has a parent-enforced
deadline; a socket timeout alone would not bound a slowly streaming response.
Credentials travel only over private stdin, never argv or evidence files.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operator_agent.execute.llm_client import CostTracker, parse_usage, _estimate_tokens_from_text
from operator_agent.core.run_contract import atomic_json, code_hash, digest

LIMITS = dict(max_calls=2, max_prompt_bytes=16384, max_response_bytes=65536,
              max_output_tokens=512, total_output_token_reservations=1024,
              call_seconds=60, campaign_seconds=1800, temperature=0.3, enable_thinking=False,
              retries=0, fallback=False, redirects=False)


def model_config(path):
    """Read the existing local config without copying endpoint/key to evidence."""
    import yaml
    cfg = (yaml.safe_load(Path(path).read_text()) or {}).get('llm') or {}
    url = (cfg.get('base_url') or os.environ.get('LLM_BASE_URL', '')).rstrip('/')
    model = cfg.get('model') or os.environ.get('LLM_MODEL', '')
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme not in ('http', 'https') or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or not isinstance(model, str) or not model):
        raise ValueError('missing/unsafe configured model endpoint or identity')
    return dict(url=url, model=model, key=os.environ.get('LLM_API_KEY', ''))


def model_identity(config):
    return dict(endpoint_hash=code_hash(config['url']), model=config['model'],
                credential_present=bool(config['key']), transport='configured-single-endpoint',
                limits=dict(LIMITS))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def worker(payload):
    """Only child entry point may use network. No proxy/redirect/fallback."""
    headers = {'Content-Type': 'application/json'}
    if payload['key']:
        headers['Authorization'] = 'Bearer ' + payload['key']
    request = urllib.request.Request(payload['url'] + '/chat/completions',
        data=json.dumps(payload['body'], ensure_ascii=False).encode(), headers=headers, method='POST')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(request, timeout=LIMITS['call_seconds']) as response:
            raw = response.read(LIMITS['max_response_bytes'] + 1)
        if len(raw) > LIMITS['max_response_bytes']:
            return {'error': 'response_size_limit'}
        return {'response_text': raw.decode('utf-8')}
    except Exception as exc:
        # API error bodies/URLs can contain credentials; never propagate them.
        return {'error': type(exc).__name__, 'http_status': getattr(exc, 'code', None)}


def dispatch(payload):
    try:
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'],
            input=json.dumps(payload), capture_output=True, text=True,
            timeout=LIMITS['call_seconds'], check=False)
    except subprocess.TimeoutExpired:
        raise ValueError('model deadline exceeded; child killed, no retry') from None
    if process.returncode:
        raise ValueError('model worker failed; no retry')
    try:
        result = json.loads(process.stdout)
        if 'error' in result:
            raise ValueError('model transport refused/failed: ' + str(result['error']))
        return result['response_text']
    except (KeyError, json.JSONDecodeError):
        raise ValueError('invalid model worker response') from None


class BoundedModel:
    def __init__(self, config, root, identity):
        if model_identity(config) != identity:
            raise ValueError('model identity drift')
        self._config = dict(config)
        self.identity = identity
        self.root = Path(root)
        self.records = []
        self.cost_tracker = CostTracker()
        self.deadline = time.monotonic() + LIMITS['campaign_seconds']

    def __repr__(self):
        return 'BoundedModel(model=' + repr(self.identity['model']) + ', endpoint=redacted)'

    def ask(self, messages, stage):
        if model_identity(self._config) != self.identity:
            raise ValueError('model configuration drift')
        if len(self.records) >= LIMITS['max_calls']:
            raise ValueError('model call budget exhausted before dispatch')
        if time.monotonic() + LIMITS['call_seconds'] > self.deadline:
            raise ValueError('insufficient campaign time before model dispatch')
        if stage not in ('InitialProposal', 'SearchProposal'):
            raise ValueError('undeclared model caller')
        if len(json.dumps(messages, ensure_ascii=False).encode()) > LIMITS['max_prompt_bytes']:
            raise ValueError('model prompt byte limit before dispatch')
        body = dict(model=self.identity['model'], messages=messages,
            temperature=LIMITS['temperature'], max_tokens=LIMITS['max_output_tokens'],
            stream=False, chat_template_kwargs={'enable_thinking': False})
        if len(json.dumps(body, ensure_ascii=False).encode()) > LIMITS['max_prompt_bytes']:
            raise ValueError('model request byte limit before dispatch')
        record = dict(call=len(self.records)+1, stage=stage, request=body,
            request_hash=digest(body), identity=self.identity, status='reserved',
            output_tokens_reserved=LIMITS['max_output_tokens'])
        path = self.root / ('model-call-%02d.json' % record['call'])
        if path.exists() or path.is_symlink():
            raise ValueError('refusing to overwrite an existing model attempt')
        self.records.append(record)
        atomic_json(path, record)  # Must succeed BEFORE spending a call.
        start = time.monotonic()
        try:
            raw = dispatch(dict(self._config, body=body))
            record.update(response_text=raw, response_hash=code_hash(raw))
            result = json.loads(raw)
            choice = result['choices'][0]
            content = choice['message']['content']
            record.update(finish_reason=choice.get('finish_reason'),
                          reported_model=result.get('model'), usage=result.get('usage'))
            norm = parse_usage(result.get('usage'))
            if min(norm['prompt_tokens'], norm['completion_tokens']) < 0:
                raise ValueError('negative model usage')
            usage_source = 'api_usage' if result.get('usage') else 'estimated'
            self.cost_tracker.record(model=self.identity['model'],
                prompt_tokens=norm['prompt_tokens'] if usage_source == 'api_usage' else _estimate_tokens_from_text(json.dumps(messages)),
                completion_tokens=norm['completion_tokens'] if usage_source == 'api_usage' else _estimate_tokens_from_text(content or ''),
                usage_source=usage_source, caller_tag=stage,
                cached_prompt_tokens=norm['cached_prompt_tokens'], cache_semantics=norm['cache_semantics'],
                reported_model=result.get('model'), temperature=body['temperature'],
                max_tokens=body['max_tokens'], enable_thinking=False)
            if (result.get('model') != self.identity['model']
                    or choice.get('finish_reason') != 'stop' or not isinstance(content, str)
                    or not content.strip() or norm['completion_tokens'] > body['max_tokens']):
                raise ValueError('empty/truncated/over-budget model output')
            proposal = json.loads(content)  # Strict JSON; no repair or retry.
            if not isinstance(proposal, dict):
                raise ValueError('model proposal must be an object')
            record.update(status='answered', proposal=proposal)
            return proposal
        except Exception:
            # Do not log exception text (could quote URL, key or HTTP response).
            record['status'] = 'failed'
            raise ValueError('model call failed; original evidence retained, no retry') from None
        finally:
            record['wall_seconds'] = time.monotonic() - start
            atomic_json(path, record, readonly=True)


if __name__ == '__main__':
    print(json.dumps(worker(json.loads(sys.stdin.read()))))
