"""Run the installed DSH headless CLI against a *local fake* endpoint only.

Test harness, not a sandbox or Linux deployment. The caller must supply a
trusted loopback fake provider; the fake provider controls every reply. No real
credential, provider, GPU or user DSH profile is used. DSH runs a real agent
turn, but this harness does not execute or judge the returned text as code.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from urllib.parse import urlsplit

DSH_ROOT = Path('/Applications/DSH Desktop.app/Contents/Resources/app/node_modules/@deepseek-ai')
NODE = shutil.which('node')
DISABLED = ('session-title-llm', 'llm-retry', 'otel', 'session-telemetry-otel',
    'tool-working-directory', 'tool-bash', 'tool-pwsh', 'tool-jobs', 'tool-fs',
    'tool-fs-search', 'tool-skill', 'tool-subagent-control', 'tool-subagent-list-agents',
    'tool-subagent', 'tool-subagent-fork', 'tool-workflow', 'tool-todo', 'tool-goal',
    'web', 'web-search-deepseek', 'web-deepseek-api', 'web-fetch-http', 'tool-web')


def run_headless(base_url, workspace, task, *, timeout=45):
    url = urlsplit(base_url)
    if (url.scheme != 'http' or url.hostname != '127.0.0.1' or not url.port
            or url.path != '/v1' or url.username or url.password or url.query or url.fragment):
        raise ValueError('an explicit IPv4 loopback fake /v1 endpoint is required')
    if not NODE or not (DSH_ROOT/'dsh/lib/bin.js').is_file():
        raise RuntimeError('installed DSH Desktop runtime unavailable')
    if not isinstance(task, str) or not task.strip() or len(task.encode()) > 4096:
        raise ValueError('invalid task')
    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    home = root/'home'; home.mkdir(mode=0o700)
    work = root/'work'; work.mkdir(mode=0o700)
    patch = [dict(id=name, disabled=True) for name in DISABLED]
    patch += [dict(id='agent-default-model',config=dict(provider='fixture',model='fake-model')),
        dict(id='llm-pi-ai',config=dict(providers=dict(fixture=dict(
            api='openai-completions',baseURL=base_url,apiKeyEnv='S5_FAKE_API_KEY',
            models=[dict(id='fake-model',contextWindow=16384,maxTokens=1024)],
            compat=dict(maxTokensField='max_tokens')))))]
    patch_path = root/'patch.json'
    patch_path.write_text(json.dumps(patch),encoding='utf-8')
    env={'PATH':'/usr/local/bin:/usr/bin:/bin','HOME':str(home),'DSH_HOME':str(home/'dsh'),
         'S5_FAKE_API_KEY':'placeholder','NO_PROXY':'127.0.0.1','no_proxy':'127.0.0.1'}
    command=[NODE,str(DSH_ROOT/'dsh/lib/bin.js'),'--profile','headless',
             '--patch',str(patch_path),'--json',task]
    proc=subprocess.run(command,cwd=work,env=env,stdin=subprocess.DEVNULL,
                        capture_output=True,text=True,timeout=timeout)
    if len(proc.stdout.encode())>2*1024*1024 or len(proc.stderr.encode())>2*1024*1024:
        raise RuntimeError('headless output exceeded fixture bound')
    events=[]
    for line in proc.stdout.splitlines():
        try: events.append(json.loads(line))
        except ValueError: raise RuntimeError('headless output is not JSONL') from None
    return dict(returncode=proc.returncode,events=events,stderr=proc.stderr,
                scope='installed macOS DSH headless with trusted loopback fixture; not isolation')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',required=True)
    parser.add_argument('--workspace',required=True)
    parser.add_argument('--task',required=True)
    args=parser.parse_args()
    result=run_headless(args.base_url,args.workspace,args.task)
    print(json.dumps(result,ensure_ascii=False))
    return 0 if result['returncode']==0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
