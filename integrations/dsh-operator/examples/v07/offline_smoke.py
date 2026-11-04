"""Exercise installed CLI/MCP with a scripted model and CPU FakeBackend.

Run with the Python environment that has operator-agent and NumPy installed.
The output directory must be new. This never selects SSH or a live model.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


class Client:
    def __init__(self, entry, output):
        self.entry, self.output, self.count = entry, output, 0
        self.proc = None
        # The example chooses the installed package, independently of a checkout.
        self.env = dict(os.environ)
        self.env.pop('PYTHONPATH', None)
        if any(k.startswith('OPERATOR_AGENT_S5_DEV_') for k in self.env):
            raise ValueError('run outside an inherited S5 development-service environment')
        if entry == 'mcp':
            self.stderr = (output/'mcp-stderr.log').open('w', encoding='utf-8')
            self.proc = subprocess.Popen([sys.executable, '-I', '-m', 'operator_agent.app.mcp_server'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
                text=True, bufsize=1, cwd=output, env=self.env)
            self.inbox = queue.Queue()
            def read():
                for line in self.proc.stdout:
                    self.inbox.put(line)
                self.inbox.put(None)
            self.reader = threading.Thread(target=read, daemon=True)
            self.reader.start()
            try:
                self.rpc('initialize', dict(protocolVersion='2024-11-05', capabilities={},
                    clientInfo=dict(name='v07-offline-example', version='1')))
                tools = self.rpc('tools/list', {})
                required = {'operator_run', 'operator_status', 'operator_cancel'}
                if not required <= {t['name'] for t in tools['tools']}:
                    raise RuntimeError('installed MCP server lacks required tools')
            except Exception:
                self.close()
                raise

    def rpc(self, method, params):
        self.count += 1
        request = dict(jsonrpc='2.0', id=self.count, method=method, params=params)
        self.proc.stdin.write(json.dumps(request) + '\n')
        self.proc.stdin.flush()
        end = time.monotonic() + 15
        while True:
            raw = self.inbox.get(timeout=max(.01, end-time.monotonic()))
            if raw is None:
                raise RuntimeError('MCP server exited before returning a response')
            reply = json.loads(raw)
            if reply.get('id') == self.count:
                break
        save(self.output/('rpc-%03d.json' % self.count), dict(request=request, reply=reply))
        if 'error' in reply:
            raise RuntimeError('MCP error: ' + json.dumps(reply['error']))
        return reply['result']

    def call(self, action, args):
        if self.entry == 'mcp':
            result = self.rpc('tools/call', dict(name='operator_'+action, arguments=args))
            value = json.loads(result['content'][0]['text'])
            if result.get('isError'):
                raise RuntimeError('MCP tool failed: ' + json.dumps(value))
            return value
        command = [sys.executable, '-I', '-m', 'operator_agent', action]
        for key, value in args.items():
            if key == 'fake':
                command.extend(['--fake-scenario', value['scenario'], '--fake-profile', value['profile'],
                                '--llm-mode', value['llm']])
            elif key == 'shapes':
                for shape in value:
                    command.extend(['--shapes', ','.join(map(str, shape))])
            else:
                command.extend(['--'+key.replace('_', '-'), str(value)])
        response = subprocess.run(command, cwd=self.output, env=self.env,
            capture_output=True, text=True, timeout=15)
        self.count += 1
        save(self.output/('cli-%03d.json' % self.count), dict(action=action, arguments=args,
            returncode=response.returncode, stdout=response.stdout, stderr=response.stderr))
        value = json.loads(response.stdout)
        # The production CLI uses 130 for an observed cancelled terminal state.
        if response.returncode not in (0, 130):
            raise RuntimeError('CLI failed: ' + json.dumps(value))
        return value

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            self.stderr.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry', choices=('cli', 'mcp'), default='cli')
    parser.add_argument('--scenario', choices=('success', 'correctness-repair', 'cancel'), default='success')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    run = output/'run'
    client = Client(args.entry, output)
    started = None
    try:
        started = client.call('run', dict(request='relu', shapes=[[4, 8]], dsl='python',
            run_dir=str(run), backend='fake', fake=dict(scenario='timeout' if args.scenario=='cancel'
                else args.scenario, profile='small', llm='fake'), max_rounds=2,
            wall_clock_s=30, max_llm_calls=8, oracle_total=20))
        invocation = started['invocation_id']
        status_args = dict(run_dir=str(run), invocation_id=invocation)
        end = time.monotonic()+40
        cancelled = False
        while time.monotonic() < end:
            status = client.call('status', status_args)
            if status.get('phase') in ('done', 'failed', 'cancelled', 'lost'):
                break
            if args.scenario == 'cancel' and not cancelled:
                events = run/'fake-events.jsonl'
                if events.exists() and any(json.loads(line).get('operation')=='blocked'
                                          for line in events.read_text().splitlines()):
                    client.call('cancel', status_args)
                    cancelled = True
            time.sleep(.1)
        else:
            raise RuntimeError('example deadline expired')
        save(output/'status.json', status)
        if status.get('execution_context', {}).get('test_only') is not True:
            raise RuntimeError('missing test_only provenance')
        expected = 'cancelled' if args.scenario == 'cancel' else 'done'
        if status.get('phase') != expected:
            raise RuntimeError('unexpected terminal phase: '+str(status.get('phase')))
        exported = output/'export'
        exported.mkdir()
        shutil.copyfile(output/'status.json', exported/'status.json')
        if args.scenario != 'cancel':
            summary = status['summary']
            if summary['levels'] != dict(L1=True, L2=True, L3=False) or summary['all_passed'] is not False:
                raise RuntimeError('unexpected fake correctness/performance result')
            source = Path(summary['path']).resolve()
            if run.resolve() not in source.parents:
                raise RuntimeError('result is outside its run directory')
            for name in ('code.py', 'report.md'):
                shutil.copyfile(source/name, exported/name)
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(exported.iterdir())}
        save(exported/'manifest.json', dict(test_only=True, gpu_acceptance=False,
            entry=args.entry, scenario=args.scenario, invocation_id=invocation, files=hashes))
        print(json.dumps(dict(ok=True, phase=status['phase'], test_only=True,
            gpu_acceptance=False, export=str(exported)), ensure_ascii=False))
    except Exception:
        if started:
            try:
                client.call('cancel', dict(run_dir=str(run), invocation_id=started['invocation_id']))
            except Exception:
                pass
        raise
    finally:
        client.close()


if __name__ == '__main__':
    main()
