"""Reproduce the current A/C loopback egress gap, without model/GPU/SSH calls.

Two synthetic HTTP listeners bind 127.0.0.1 only. The actual participant
file-policy builder/wrapper confines Python and shell-child probes. Success at
the second, undeclared listener is evidence of a gap, never formal readiness.
This does not start a DSH model loop or probe any real provider/host proxy.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import threading
from urllib.parse import urlsplit

from benchmarks import s5_plugin_env
from operator_agent.run_contract import digest

SCHEMA = 's5-loopback-network-gap-probe-v1'


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write('\n')


class Listener:
    def __init__(self, label):
        self.label = label
        self.requests = []
        self.lock = threading.Lock()
        self.server = self.thread = None
        self.cleaned = False

    def __enter__(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                path = urlsplit(self.path).path
                with owner.lock:
                    owner.requests.append(dict(path=path, client_host=self.client_address[0]))
                raw = json.dumps(dict(listener=owner.label, path=path)).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    @property
    def port(self):
        return self.server.server_address[1]

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.cleaned = not self.thread.is_alive()


class ConfigurationOnlyRelay:
    """Provide C's normal on-disk shape; never start a core/MCP process."""
    def __init__(self, port):
        self.port = port

    def client_config(self):
        return dict(url=f'http://127.0.0.1:{self.port}/mcp',
                    token='public-loopback-probe-placeholder', timeout=5)


def child_source(gateway_port, sentinel_port, label):
    # No URL/proxy environment or DNS path: literal 127.0.0.1 and two owned ports.
    return (
        'import http.client,json\n'
        'result={}\n'
        f'for name,port in {[("gateway", gateway_port), ("sentinel", sentinel_port)]!r}:\n'
        ' conn=http.client.HTTPConnection("127.0.0.1",port,timeout=2)\n'
        ' try:\n'
        f'  conn.request("GET", "/{label}/"+name)\n'
        '  response=conn.getresponse();body=response.read(4096)\n'
        '  parsed=json.loads(body);result[name]={'
        '"reachable":response.status==200 and parsed.get("listener")==name,"reply":parsed}\n'
        ' except Exception as exc:result[name]={"reachable":False,"error":type(exc).__name__}\n'
        ' finally:conn.close()\n'
        'print(json.dumps(result,sort_keys=True))\n'
    )


def execute(command, *, env=None, cwd=None):
    try:
        p = subprocess.run(command, env=env, cwd=cwd, capture_output=True,
                           text=True, timeout=15, stdin=subprocess.DEVNULL)
        try:
            data = json.loads(p.stdout)
        except ValueError:
            data = None
        return dict(returncode=p.returncode, stdout=p.stdout[:8192],
                    stderr=p.stderr[:4096], data=data)
    except (OSError, subprocess.SubprocessError) as exc:
        return dict(returncode=None, error=type(exc).__name__, data=None)


def run(output):
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    runtime = s5_plugin_env.stage_runtime(out / 'runtime')
    rows = []
    gateway = Listener('gateway')
    sentinel = Listener('sentinel')
    error = None
    controls = []
    try:
        with ExitStack() as stack:
            stack.enter_context(gateway)
            stack.enter_context(sentinel)
            for route in ('python', 'shell-child'):
                src = child_source(gateway.port, sentinel.port, f'control-{route}')
                command = [sys.executable, '-c', src]
                if route == 'shell-child':
                    command = ['/bin/sh', '-c', ' '.join(shlex.quote(v) for v in command)]
                controls.append(dict(route=route, **execute(command)))
            for arm in ('A', 'C'):
                temporary = stack.enter_context(tempfile.TemporaryDirectory(prefix='s5-network-gap-'))
                built = s5_plugin_env.build(arm, runtime=runtime,
                    relay=ConfigurationOnlyRelay(gateway.port) if arm == 'C' else None,
                    base_url=f'http://127.0.0.1:{gateway.port}/v1', arm_dir=temporary,
                    api='responses', model='local-network-probe-no-model-call',
                    max_tokens=1024, context_window=16384, tools=True,
                    development=dict(url=f'http://127.0.0.1:{gateway.port}/development/verify',
                        token='public-loopback-probe-placeholder', contract='a' * 64,
                        backend='mock', timeout=5))
                policy = built['isolation']
                original = Path(policy['profile']).read_bytes()
                (out / f'{arm}-actual-profile.sb').write_bytes(original)
                write(out / f'{arm}-policy.json', dict(profile_sha256=policy['profile_sha256'],
                    read_roots=policy['read_roots'], read_files=policy['read_files'],
                    write_roots=policy['write_roots'], declared_gateway_port=gateway.port,
                    undeclared_sentinel_port=sentinel.port,
                    same_wrapper_as_dsh=True, dsh_model_loop_started=False))
                for route in ('python', 'shell-child'):
                    label = f'{arm}-{route}'
                    src = child_source(gateway.port, sentinel.port, label)
                    command = [sys.executable, '-c', src]
                    if route == 'shell-child':
                        command = ['/bin/sh', '-c', ' '.join(shlex.quote(v) for v in command)]
                    wrapped = s5_plugin_env.wrap(built, command)
                    result = execute(wrapped, env=built['env'], cwd=built['work'])
                    rows.append(dict(arm=arm, route=route, label=label,
                        profile_sha256=hashlib.sha256(original).hexdigest(), **result))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        error = type(exc).__name__
    def reached(record):
        data = record.get('data') or {}
        return record.get('returncode') == 0 and all(
            (data.get(name) or {}).get('reachable') is True for name in ('gateway', 'sentinel'))
    positive = len(controls) == 2 and all(reached(r) for r in controls)
    server_paths = {name: [r['path'] for r in endpoint.requests]
                    for name, endpoint in (('gateway', gateway), ('sentinel', sentinel))}
    receipts = len(rows) == 4 and all(
        f"/{r['label']}/{name}" in server_paths[name] for r in rows for name in server_paths)
    reproduced = positive and receipts and all(reached(r) for r in rows)
    body = dict(schema=SCHEMA, status='bypass_reproduced' if reproduced else 'not_demonstrated',
        current_policy_gap_reproduced=reproduced, formal_ready=False,
        network_isolation_verified=False, positive_control=positive,
        server_receipts_match=receipts, controls=controls, records=rows,
        listener_requests=dict(gateway=gateway.requests, sentinel=sentinel.requests),
        cleanup=dict(gateway_listener_stopped=gateway.cleaned, sentinel_listener_stopped=sentinel.cleaned),
        error=error, real_model_calls=0, gpu_executions=0, ssh_calls=0,
        external_network_probed=False, dsh_model_loop_started=False,
        scope='actual A/C build+wrap policy applied to Python and shell-child; two owned loopback listeners only',
        runtime=runtime)
    body['hash'] = digest(body)
    write(out / 'summary.json', body)
    return body


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.output)
    print(json.dumps({k: result[k] for k in ('status', 'hash', 'error', 'cleanup')}, sort_keys=True))
    return 0 if result['current_policy_gap_reproduced'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
