"""Run the actual installed DSH headless with a loopback mock validation tool.

Only local fixture HTTP URLs and placeholder credentials are accepted. Tool
restrictions are configuration, not an OS sandbox. Never run untrusted models.
"""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
from urllib.parse import urlsplit

from benchmarks.experiments.s5.s5_vertical_headless import DSH_ROOT, NODE, DISABLED

PLUGIN = Path(__file__).with_name('s5_vertical_tool_plugin.mjs').resolve()


def _local_url(value, path):
    u=urlsplit(value)
    if (u.scheme!='http' or u.hostname!='127.0.0.1' or not u.port or u.path!=path
            or u.username or u.password or u.query or u.fragment):
        raise ValueError('explicit local fixture endpoint required')
    return value


def run_tool_headless(base_url, validation_url, workspace, task, *, timeout=45):
    _local_url(base_url,'/v1'); _local_url(validation_url,'/validate')
    if not NODE or not (DSH_ROOT/'dsh/lib/bin.js').is_file():
        raise RuntimeError('installed DSH Desktop runtime unavailable')
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
        dict(insert=[dict(id='s5-fixture-validation',name=str(PLUGIN),
                          inject=['tools'],config=dict(validationUrl=validation_url))])]
    patch_path=root/'patch.json'; patch_path.write_text(json.dumps(patch),encoding='utf-8')
    env={'PATH':'/usr/local/bin:/usr/bin:/bin','HOME':str(home),'DSH_HOME':str(home/'dsh'),
        'S5_FAKE_API_KEY':'placeholder','NO_PROXY':'127.0.0.1','no_proxy':'127.0.0.1',
        'S5_DSH_TOOLS_MODULE':str(DSH_ROOT/'dsh-tools/lib/index.js')}
    command=[NODE,str(DSH_ROOT/'dsh/lib/bin.js'),'--profile','headless',
             '--patch',str(patch_path),'--json',task]
    proc=subprocess.run(command,cwd=work,env=env,stdin=subprocess.DEVNULL,
                        capture_output=True,text=True,timeout=timeout)
    if len(proc.stdout.encode())>2*1024*1024 or len(proc.stderr.encode())>2*1024*1024:
        raise RuntimeError('headless output exceeded fixture bound')
    try: events=[json.loads(line) for line in proc.stdout.splitlines()]
    except ValueError: raise RuntimeError('headless output is not JSONL') from None
    return dict(returncode=proc.returncode,events=events,stderr=proc.stderr,
                scope='installed macOS DSH headless; local fixture tool; no isolation')
