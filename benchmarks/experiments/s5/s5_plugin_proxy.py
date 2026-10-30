"""Standalone stdlib-only bridge to one supervisor-owned real MCP process.

This file is copied to the participant's public runtime; it imports no core
package and offers no arbitrary CLI execution or final evaluation endpoint.
"""
from __future__ import annotations
import argparse
import http.client
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit

MAX_REQUEST = 262144
MAX_REPLY = 2 * 1024 * 1024


def configuration(path):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError('proxy configuration unavailable')
    with path.open('rb') as f:
        raw = f.read(8193)
    if len(raw) > 8192:
        raise ValueError('proxy configuration bound')
    conf = json.loads(raw)
    if (type(conf) is not dict or set(conf) != {'url', 'token', 'timeout'}
            or type(conf['timeout']) is not int or not 1 <= conf['timeout'] <= 1810
            or not isinstance(conf['token'], str) or not 16 <= len(conf['token']) <= 256
            or any(ord(c) < 33 or ord(c) > 126 for c in conf['token'])):
        raise ValueError('proxy configuration invalid')
    url = urlsplit(conf['url'])
    if (url.scheme != 'http' or url.hostname != '127.0.0.1' or not url.port
            or url.path != '/mcp' or url.username or url.password or url.query or url.fragment):
        raise ValueError('fixed loopback MCP endpoint required')
    return conf, url


def forward(config, request):
    conf, url = configuration(config)
    raw = json.dumps(request, ensure_ascii=False, allow_nan=False).encode()
    if len(raw) > MAX_REQUEST:
        raise ValueError('request exceeds bound')
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=conf['timeout'])
    try:
        conn.request('POST', url.path, body=raw, headers={
            'Authorization': 'Bearer '+conf['token'], 'Content-Type': 'application/json'})
        response = conn.getresponse()
        body = response.read(MAX_REPLY+1)
        if response.status != 200 or len(body) > MAX_REPLY:
            raise ValueError('MCP relay unavailable')
        result = json.loads(body)
    finally:
        conn.close()
    if request.get('id') is None:
        if result != {'notification': True}:
            raise ValueError('invalid notification response')
        return None
    if (type(result) is not dict or result.get('jsonrpc') != '2.0'
            or type(result.get('id')) is not type(request['id']) or result.get('id') != request['id']):
        raise ValueError('MCP response identity mismatch')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--stdio', action='store_true')
    parser.add_argument('--capabilities', action='store_true')
    parser.add_argument('command', nargs='?', choices=['capabilities'])
    args = parser.parse_args(argv)
    if args.stdio and not args.capabilities and args.command is None:
        while True:
            raw = sys.stdin.buffer.readline(MAX_REQUEST+1)
            if not raw:
                break
            if len(raw) > MAX_REQUEST:
                return 2
            identity = None
            try:
                message = json.loads(raw)
                if type(message) is not dict:
                    raise ValueError('RPC object required')
                identity = message.get('id')
                reply = forward(args.config, message)
            except (OSError, ValueError, http.client.HTTPException):
                reply = (dict(jsonrpc='2.0', id=identity, error=dict(code=-32603,
                         message='MCP relay unavailable')) if identity is not None else None)
            if reply is not None:
                print(json.dumps(reply, ensure_ascii=False, allow_nan=False), flush=True)
        return 0
    if not args.stdio and (args.capabilities or args.command == 'capabilities'):
        try:
            reply = forward(args.config, dict(jsonrpc='2.0', id='capabilities',method='tools/call',
                params=dict(name='operator_capabilities', arguments={})))
            result = reply.get('result', {})
            content = result.get('content', [])
            if 'error' in reply or result.get('isError') or not content:
                raise ValueError('capabilities unavailable')
            # Match the real CLI payload consumed by the unchanged /op plugin.
            print('\n'.join(block['text'] for block in content if block.get('type') == 'text'))
            return 0
        except (OSError, ValueError, http.client.HTTPException):
            print(json.dumps(dict(ok=False,error='mcp_unavailable')))
            return 2
    parser.error('choose --stdio or capabilities')


if __name__ == '__main__': raise SystemExit(main())
