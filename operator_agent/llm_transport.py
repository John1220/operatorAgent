"""Disposable HTTP worker for bounded production LLM requests.

No application imports, redirect, proxy, retry, or fallback. The parent kills
and reaps this child at its wall deadline, including a slowly dripping socket.
Credentials travel over stdin only; transport errors never include API bodies.
"""
import json
from pathlib import Path
import subprocess
import sys
import urllib.request


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def worker(payload):
    headers = {'Content-Type': 'application/json'}
    if payload['key']:
        headers['Authorization'] = 'Bearer ' + payload['key']
    req = urllib.request.Request(payload['url'], data=payload['body_text'].encode('utf-8'),
                                 headers=headers, method='POST')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(req, timeout=payload['timeout']) as response:
            raw = response.read(payload['max_response_bytes'] + 1)
        if len(raw) > payload['max_response_bytes']:
            return {'error': 'response_size_limit'}
        return {'response_text': raw.decode('utf-8')}
    except Exception:
        return {'error': 'transport_failure'}


def dispatch(payload, timeout):
    try:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'],
            input=json.dumps(payload, ensure_ascii=False), text=True, capture_output=True,
            timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        raise RuntimeError('model deadline exceeded; child killed and reaped') from None
    if result.returncode:
        raise RuntimeError('model worker failed')
    try:
        reply = json.loads(result.stdout)
        if 'error' in reply:
            raise RuntimeError('bounded model transport failed')
        return reply['response_text']
    except (KeyError, TypeError, json.JSONDecodeError):
        raise RuntimeError('invalid model worker response') from None


if __name__ == '__main__':
    print(json.dumps(worker(json.loads(sys.stdin.read())), ensure_ascii=False))
