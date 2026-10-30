"""Actual DSH headless plus repository MCP/plugin, only local fake model.

The experimental caller controls every model reply and provides a pre-prepared
mock task. This is not a sandbox, a Linux run, or a real-provider/GPU test.
"""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit

from benchmarks.experiments.s5 import s5_runtime
from benchmarks.experiments.s5.s5_vertical_headless import DSH_ROOT, NODE, DISABLED

ROOT = Path(__file__).resolve().parents[3]
PLUGIN = ROOT/'integrations'/'dsh-operator'/'index.mjs'
MCP_CLIENT = DSH_ROOT/'dsh-mcp-client'/'lib'/'index.js'


def run_mcp_headless(base_url, workspace, task, *, timeout=90):
    u=urlsplit(base_url)
    if (u.scheme!='http' or u.hostname!='127.0.0.1' or not u.port or u.path!='/v1'
            or u.username or u.password or u.query or u.fragment):
        raise ValueError('explicit local fixture model endpoint required')
    if not NODE or not (DSH_ROOT/'dsh/lib/bin.js').is_file() or not MCP_CLIENT.is_file() or not PLUGIN.is_file():
        raise RuntimeError('installed DSH or repository plugin unavailable')
    if not isinstance(task,str) or not task.strip() or len(task.encode())>4096:
        raise ValueError('invalid task')
    root=Path(workspace)
    root.mkdir(parents=True,exist_ok=False,mode=0o700)
    home=root/'home'; home.mkdir(mode=0o700)
    work=root/'work'; work.mkdir(mode=0o700)
    patch=[dict(id=name,disabled=True) for name in DISABLED]
    patch += [dict(id='agent-default-model',config=dict(provider='fixture',model='fake-model')),
        dict(id='llm-pi-ai',config=dict(providers=dict(fixture=dict(
            api='openai-completions',baseURL=base_url,apiKeyEnv='S5_FAKE_API_KEY',
            models=[dict(id='fake-model',contextWindow=16384,maxTokens=1024)],
            compat=dict(maxTokensField='max_tokens'))))),
        dict(insert=[
            dict(id='mcp-operator',name=str(MCP_CLIENT),config=dict(
                serverName='operator',transport='stdio',command=sys.executable,
                args=['-m','operator_agent.mcp_server'],toolCallTimeoutMs=60000,
                failOnStartupError=True,env=dict(PYTHONPATH=s5_runtime.python_path(ROOT),PYTHONNOUSERSITE='1'))),
            dict(id='operator-tools',name=str(PLUGIN),config=dict(
                command=sys.executable,args=['-m','operator_agent'],
                env=dict(PYTHONPATH=s5_runtime.python_path(ROOT),PYTHONNOUSERSITE='1'))),
        ])]
    patch_path=root/'patch.json'; patch_path.write_text(json.dumps(patch),encoding='utf-8')
    env={'PATH':'/usr/local/bin:/usr/bin:/bin','HOME':str(home),'DSH_HOME':str(home/'dsh'),
        'S5_FAKE_API_KEY':'placeholder','NO_PROXY':'127.0.0.1','no_proxy':'127.0.0.1'}
    command=[NODE,str(DSH_ROOT/'dsh/lib/bin.js'),'--profile','headless',
             '--patch',str(patch_path),'--json',task]
    proc=subprocess.run(command,cwd=work,env=env,stdin=subprocess.DEVNULL,
                        capture_output=True,text=True,timeout=timeout)
    if len(proc.stdout.encode())>2*1024*1024 or len(proc.stderr.encode())>2*1024*1024:
        raise RuntimeError('headless output exceeded fixture bound')
    try: events=[json.loads(line) for line in proc.stdout.splitlines()]
    except ValueError: raise RuntimeError('headless output is not JSONL') from None
    return dict(returncode=proc.returncode,events=events,stderr=proc.stderr,
                scope='installed macOS DSH headless, repository MCP/plugin, no isolation')
