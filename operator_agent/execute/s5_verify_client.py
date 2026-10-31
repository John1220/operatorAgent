"""Opt-in S5 verification transport, also usable as a standalone public client.

No operator imports: the public A-arm copy exposes only source submission. This
is a cooperative loopback transport, not an isolation or provenance boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

PREFIX = 'OPERATOR_AGENT_S5_DEV_'
KEYS = ('URL', 'TOKEN', 'CONTRACT', 'BACKEND', 'TIMEOUT')


def enabled():
    return any(PREFIX + key in os.environ for key in (*KEYS, 'CONFIG'))


def configuration():
    config_path = os.environ.get(PREFIX + 'CONFIG')
    if config_path:
        path = Path(config_path)
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise ValueError('development configuration file unavailable')
        with path.open('rb') as f:
            raw = f.read(8193)
        if len(raw) > 8192:
            raise ValueError('development configuration too large')
        conf = json.loads(raw)
        if not isinstance(conf, dict) or set(conf) != set(KEYS):
            raise ValueError('invalid development configuration file')
    else:
        conf = {key: os.environ.get(PREFIX + key) for key in KEYS}
    if any(not conf[key] for key in KEYS):
        raise ValueError('incomplete S5 development configuration')
    u = urlsplit(conf['URL'])
    if (u.scheme != 'http' or u.hostname != '127.0.0.1' or not u.port
            or u.path != '/development/verify' or u.username or u.password or u.query or u.fragment):
        raise ValueError('fixed loopback development endpoint required')
    if conf['BACKEND'] not in ('mock', 'ssh') or len(conf['CONTRACT']) != 64:
        raise ValueError('invalid development identity')
    timeout = int(conf['TIMEOUT'])
    if not 1 <= timeout <= 1810:
        raise ValueError('invalid development timeout')
    return conf, u, timeout


def submit(code, *, contract_hash=None):
    conf, u, timeout = configuration()
    if not isinstance(code, str) or not code.strip() or len(code.encode('utf-8')) > 65536:
        raise ValueError('invalid candidate')
    if contract_hash is not None and contract_hash != conf['CONTRACT']:
        raise ValueError('prepared contract differs from supervised task')
    body = json.dumps(dict(code=code, contract_hash=conf['CONTRACT']), ensure_ascii=False).encode()
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    try:
        conn.request('POST', u.path, body=body, headers={
            'Authorization': 'Bearer ' + conf['TOKEN'], 'Content-Type': 'application/json'})
        response = conn.getresponse()
        raw = response.read(131073)
        if response.status != 200 or len(raw) > 131072:
            raise ValueError('development service refused or exceeded response limit')
        result = json.loads(raw)
    finally:
        conn.close()
    if not isinstance(result, dict):
        raise ValueError('invalid development response')
    if result.get('ok') is True:
        if (result.get('schema') != 's5-development-feedback-v1'
                or result.get('phase') != 'development'
                or result.get('source_sha256') != hashlib.sha256(code.encode()).hexdigest()
                or result.get('contract_hash') != conf['CONTRACT']
                or result.get('status') not in ('passed', 'failed', 'unknown')
                or result.get('passed') is not {'passed': True, 'failed': False, 'unknown': None}[result['status']]):
            raise ValueError('development response binding invalid')
    elif result.get('error') not in ('development_budget_exhausted', 'verification_closed'):
        raise ValueError('unexpected development service refusal')
    return result


def delegated_verify(run_dir, code, *, recorded_hash, stop_after, backend, host=None,
                     port=None, user=None, key_path=None, work_dir=None,
                     persistent=False, target_gflops=0.0):
    conf, _, _ = configuration()
    level = getattr(stop_after, 'value', stop_after)
    if (level != 'L2' or backend != conf['BACKEND'] or persistent or target_gflops != 0
            or any(x is not None for x in (host, port, user, key_path, work_dir))):
        raise ValueError('S5 permits only supervised L2; remote connection options are supervisor-owned')
    return submit(code, contract_hash=recorded_hash)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Submit one candidate to supervised development L2.')
    parser.add_argument('--code-file', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        # Bound allocation before reading the full file.
        with args.code_file.open('rb') as f:
            code = f.read(65537).decode('utf-8')
        result = submit(code)
    except (OSError, ValueError, http.client.HTTPException):
        result = dict(ok=False, status='unknown', error='development_transport_failed')
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get('ok') else 2


if __name__ == '__main__':
    raise SystemExit(main())
