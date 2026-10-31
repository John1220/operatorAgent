"""Per-arm launch surfaces for the S5 three-group preflight.

Arm A is bare DSH, B adds the core CLI on ``PATH``, C adds the same core CLI plus
the frozen operator plugin and MCP client. This module only *installs* those
surfaces and records what the arm's runtime environment can reach; it is not an
operating-system isolation boundary. On a shared macOS host a same-UID process
can still read the repository, so the probe result is stored as a fact instead of
being promoted into an isolation claim.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from benchmarks.experiments.s5 import s5_runtime
from benchmarks.experiments.s5.s5_vertical_headless import DSH_ROOT, DISABLED, NODE

ROOT = Path(__file__).resolve().parents[3]
ARMS = ('A', 'B', 'C')
SURFACE = {
    'A': dict(label='bare DSH', core_cli=False, plugin=False, mcp=False),
    'B': dict(label='DSH + core CLI', core_cli=True, plugin=False, mcp=False),
    'C': dict(label='DSH + core CLI + plugin', core_cli=True, plugin=True, mcp=True),
}
MCP_CLIENT = DSH_ROOT / 'dsh-mcp-client' / 'lib' / 'index.js'
PLUGIN = ROOT / 'integrations' / 'dsh-operator' / 'index.mjs'
CANARY = ROOT / 'benchmarks' / 'experiments/s5/s5_task_manifest.py'
BASE_PATH = '/usr/local/bin:/usr/bin:/bin'
# The installed MCP SDK logs missing optional resources with console.debug,
# whose Node default is stdout. Keep diagnostics off the strict JSONL stream.
# Apply the same preload to every arm; leave process.stdout.write untouched.
DEBUG_PRELOAD = 'data:text/javascript,console.debug%20%3D%20console.error%3B'


def surface(arm):
    if arm not in SURFACE:
        raise ValueError('unknown arm')
    return dict(SURFACE[arm], arm=arm)


def _write_shim(path, argv):
    body = '#!/bin/sh\nexec ' + ' '.join("'" + str(a) + "'" for a in argv) + ' "$@"\n'
    path.write_text(body, encoding='utf-8')
    path.chmod(0o755)


API_NAMES = {'chat_completions': 'openai-completions', 'responses': 'openai-responses'}


KEEP_DISABLED = frozenset({'session-title-llm', 'llm-retry', 'otel', 'session-telemetry-otel',
                           'tool-pwsh', 'tool-jobs', 'tool-skill', 'tool-subagent-control',
                           'tool-subagent-list-agents', 'tool-subagent', 'tool-subagent-fork',
                           'tool-workflow', 'tool-todo', 'tool-goal', 'web', 'web-search-deepseek',
                           'web-deepseek-api', 'web-fetch-http', 'tool-web'})


def build(arm, *, base_url, arm_dir, root=ROOT, deps=None, api='chat_completions',
          max_tokens=1024, model='fake-model', tools=False, development=None,
          context_window=16384, reasoning_effort=None, isolation_runtime=None,
          stream_idle_timeout_ms=None):
    """Return the launch configuration for one arm without starting anything."""
    facts = surface(arm)
    plugin = PLUGIN
    if isolation_runtime is not None:
        runtime_root = Path(isolation_runtime['root'])
        root, deps = runtime_root / 'core', runtime_root / 'deps'
        plugin = runtime_root / 'plugin/index.mjs'
    if api not in API_NAMES:
        raise ValueError('unknown model wire protocol')
    arm_dir = Path(arm_dir).resolve()
    home = arm_dir / 'home'
    work = arm_dir / 'work'
    env = {'PATH': BASE_PATH, 'HOME': str(home), 'DSH_HOME': str(home / 'dsh'),
           'S5_FAKE_API_KEY': 'placeholder', 'NO_PROXY': '127.0.0.1',
           'no_proxy': '127.0.0.1', 'PYTHONNOUSERSITE': '1'}
    disabled = (KEEP_DISABLED if tools else frozenset(DISABLED))
    patch = [dict(id=name, disabled=True) for name in sorted(disabled)]
    if not isinstance(model, str) or not model or len(model) > 128:
        raise ValueError('invalid model id')
    if (type(context_window) is not int or type(max_tokens) is not int
            or not 0 < max_tokens < context_window):
        raise ValueError('output tokens must fit inside the model context window')
    model_config = dict(provider='fixture', model=model)
    if reasoning_effort is not None:
        if api != 'responses' or reasoning_effort not in ('none', 'low', 'medium', 'high'):
            raise ValueError('unsupported reasoning configuration')
        model_config['reasoningEffort'] = 'off' if reasoning_effort == 'none' else reasoning_effort
    patch.append(dict(id='agent-default-model', config=model_config))
    provider = dict(api=API_NAMES[api], baseURL=base_url, apiKeyEnv='S5_FAKE_API_KEY',
                    models=[dict(id=model, contextWindow=context_window, maxTokens=max_tokens)])
    if stream_idle_timeout_ms is not None:
        if (type(stream_idle_timeout_ms) is not int
                or not 1 <= stream_idle_timeout_ms <= 2147483647):
            raise ValueError('invalid adapter stream idle timeout')
        provider['streamIdleTimeoutMs'] = stream_idle_timeout_ms
    if reasoning_effort is not None:
        provider['models'][0]['reasoningEfforts'] = dict(
            off=None, low='low', medium='medium', high='high')
    if api == 'chat_completions':
        provider['compat'] = dict(maxTokensField='max_tokens')
    patch.append(dict(id='llm-pi-ai', config=dict(providers=dict(fixture=provider))))
    shim = None
    development_env = {}
    if development is not None:
        if not tools:
            raise ValueError('development verification requires tools')
        from operator_agent.execute.s5_verify_client import PREFIX
        config_file = arm_dir / 'development-client.json'
        config_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with config_file.open('x', encoding='utf-8') as f:
            json.dump({key: str(development[key.lower()])
                       for key in ('URL', 'TOKEN', 'CONTRACT', 'BACKEND', 'TIMEOUT')}, f)
        config_file.chmod(0o600)
        # DSH deliberately scrubs TOKEN-like env names from bash children.
        # Pass a per-run private file reference, never provider credentials.
        development_env = {PREFIX + 'CONFIG': str(config_file)}
        env.update(development_env)
        patch.append(dict(id='bash-sandbox', config=dict(
            timeoutMs=(int(development['timeout']) + 30) * 1000,
            maxTimeoutMs=(int(development['timeout']) + 30) * 1000)))
    if facts['core_cli']:
        deps = s5_runtime.deps_dir() if deps is None else Path(deps)
        env['PYTHONPATH'] = s5_runtime.python_path(root, deps)
        bindir = arm_dir / 'bin'
        bindir.mkdir(parents=True, exist_ok=True, mode=0o700)
        shim = bindir / 'operator-agent'
        _write_shim(shim, (sys.executable, '-m', 'operator_agent.app.cli'))
        env['PATH'] = str(bindir) + os.pathsep + env['PATH']
    if facts['plugin']:
        patch.append(dict(insert=[
            dict(id='mcp-operator', name=str(MCP_CLIENT), config=dict(
                serverName='operator', transport='stdio', command=sys.executable,
                args=['-m', 'operator_agent.app.mcp_server'],
                toolCallTimeoutMs=((int(development['timeout']) + 30) * 1000
                                   if development is not None else 60000),
                failOnStartupError=True,
                env=dict(PYTHONPATH=env['PYTHONPATH'], PYTHONNOUSERSITE='1', **development_env))),
            dict(id='operator-tools', name=str(plugin), config=dict(
                command=sys.executable, args=['-m', 'operator_agent'],
                env=dict(PYTHONPATH=env['PYTHONPATH'], PYTHONNOUSERSITE='1', **development_env)))]))
    if development is not None and arm == 'A':
        # Standalone stdlib-only public client: no core import/PYTHONPATH added.
        bindir = arm_dir / 'bin'
        bindir.mkdir(parents=True, exist_ok=True, mode=0o700)
        client = bindir / 's5_public_verify.py'
        shutil.copyfile(root / 'operator_agent' / 'execute/s5_verify_client.py', client)
        _write_shim(bindir / 's5-verify', (sys.executable, client))
        env['PATH'] = str(bindir) + os.pathsep + env['PATH']
    body = dict(schema='s5-arm-launch-v1', arm=arm, surface=facts, model_api=api,
                stream_idle_timeout_ms=stream_idle_timeout_ms,
                diagnostic_console='console.debug to stderr via Node preload',
                tools_enabled=bool(tools),
                home=str(home), work=str(work), cli_shim=(str(shim) if shim else None),
                python=sys.executable, env=env, patch=patch,
                development_enabled=development is not None,
                dsh_entry=str(DSH_ROOT / 'dsh' / 'lib' / 'bin.js'))
    body['patch_sha256'] = hashlib.sha256(json.dumps(
        patch, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
    if isolation_runtime is not None:
        from benchmarks.experiments.s5.s5_seatbelt import configure
        configure(body, isolation_runtime)
    return body


OUTPUT_BOUND = 2 * 1024 * 1024


def headless_failure_diagnostics(response):
    """Identify a known host timeout without guessing the upstream root cause."""
    import re
    if response.get('returncode') == 0:
        return None
    match = re.fullmatch(r'dsh: TIMEOUT: pi-ai stream idle timeout after ([0-9]{1,10})ms',
                         (response.get('stderr') or '').strip())
    if match:
        return dict(kind='adapter_stream_idle_timeout', idle_timeout_ms=int(match.group(1)),
                    upstream_root_cause='unknown', budget_refund=False)
    return None


def _capture_output(directory, stdout, stderr, *, returncode=None, timed_out=False):
    """Keep bounded original bytes before decoding, including failed launches."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    metadata = dict(schema='s5-headless-output-v1', returncode=returncode,
                    timed_out=timed_out, byte_limit_per_stream=OUTPUT_BOUND)
    for name, data in (('stdout', stdout), ('stderr', stderr)):
        data = data or b''
        if isinstance(data, str):
            data = data.encode('utf-8')
        with (directory / (name + '.bin')).open('xb') as f:
            f.write(data[:OUTPUT_BOUND])
        metadata[name] = dict(bytes=len(data), saved_bytes=min(len(data), OUTPUT_BOUND),
            truncated=len(data) > OUTPUT_BOUND, sha256=hashlib.sha256(data).hexdigest())
    with (directory / 'metadata.json').open('x', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write('\n')
    return metadata


def _parse_events(stdout):
    # JSONL is LF-delimited. str.splitlines() also splits legal JSON string
    # characters such as U+0085 and U+2028, corrupting otherwise valid events.
    lines = stdout.split('\n')
    if lines and lines[-1] == '':
        lines.pop()
    events = []
    for number, line in enumerate(lines, 1):
        try:
            event = json.loads(line)
        except ValueError:
            raise RuntimeError('headless output is not JSONL at line %d' % number) from None
        if not isinstance(event, dict):
            raise RuntimeError('headless event is not an object at line %d' % number)
        events.append(event)
    return events


def start_headless(built, task, *, timeout=120, capture_dir=None):
    """Start the arm's DSH entry with its own patch, env and working directory."""
    if not NODE or not Path(built['dsh_entry']).is_file():
        raise RuntimeError('node or the DSH headless entry is unavailable')
    patch_path = Path(built['home']).parent / 'patch.json'
    patch_path.write_text(json.dumps(built['patch']), encoding='utf-8')
    command = [NODE, '--import', DEBUG_PRELOAD, built['dsh_entry'], '--profile', 'headless',
               '--patch', str(patch_path), '--json', task]
    if (built.get('isolation') or built.get('outer_isolation_required')
            or built['env'].get('OPERATOR_AGENT_S5_OUTER_REQUIRED') == '1'):
        if built.get('protocol_mode')=='plugin-only-ac-v1':
            from benchmarks.experiments.s5.s5_plugin_env import wrap
        else:
            from benchmarks.experiments.s5.s5_seatbelt import wrap
        command = wrap(built, command)
    capture_dir = (Path(capture_dir) if capture_dir is not None else
                   Path(built['home']).parent / 'headless-output')
    try:
        try:
            proc = subprocess.run(command, cwd=built['work'], env=built['env'],
                                  stdin=subprocess.DEVNULL, capture_output=True,
                                  timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            _capture_output(capture_dir, exc.stdout, exc.stderr, timed_out=True)
            raise
        captured = _capture_output(capture_dir, proc.stdout, proc.stderr,
                                   returncode=proc.returncode)
    finally:
        cleanup_development(built)
    if any(captured[name]['truncated'] for name in ('stdout', 'stderr')):
        raise RuntimeError('headless output exceeded the harness bound')
    try:
        stdout = proc.stdout.decode('utf-8')
        stderr = proc.stderr.decode('utf-8')
    except UnicodeDecodeError:
        raise RuntimeError('headless output is not UTF-8') from None
    return dict(returncode=proc.returncode, events=_parse_events(stdout), stderr=stderr,
                output_capture=captured)


def cleanup_development(built):
    if built.get('development_enabled'):
        (Path(built['home']).parent / 'patch.json').unlink(missing_ok=True)
        Path(built['env']['OPERATOR_AGENT_S5_DEV_CONFIG']).unlink(missing_ok=True)


def _run(argv, env, cwd, timeout=60):
    try:
        proc = subprocess.run(argv, env=env, cwd=str(cwd), stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=timeout)
        return dict(returncode=proc.returncode, stdout=proc.stdout[:4096],
                    stderr=proc.stderr[:4096])
    except (OSError, subprocess.SubprocessError) as exc:
        return dict(returncode=None, stdout='', stderr=type(exc).__name__)


def probe(built):
    """Record what the arm environment actually reaches. Never an isolation proof."""
    env = dict(built['env'])
    work = Path(built['work'])
    work.mkdir(parents=True, exist_ok=True, mode=0o700)
    facts = dict(schema='s5-arm-probe-v1', arm=built['arm'],
                 isolation_verified=False, linux_host_verified=False,
                 scope='same-host environment probe; not a file/network boundary')
    imp = _run([sys.executable, '-c', 'import operator_agent'], env, work)
    facts['core_importable'] = imp['returncode'] == 0
    facts['core_import_error'] = None if imp['returncode'] == 0 else imp['stderr'].strip().splitlines()[-1:] or ['failed']
    facts['cli_on_path'] = shutil.which('operator-agent', path=env.get('PATH', '')) is not None
    facts['python_path_entries'] = len(env.get('PYTHONPATH', '').split(os.pathsep)) if env.get('PYTHONPATH') else 0
    if facts['cli_on_path']:
        help_run = _run(['operator-agent', '--help'], env, work)
        facts['cli_help_returncode'] = help_run['returncode']
        facts['cli_help_sha256'] = (hashlib.sha256(help_run['stdout'].encode()).hexdigest()
                                    if help_run['returncode'] == 0 else None)
    else:
        facts['cli_help_returncode'] = None
        facts['cli_help_sha256'] = None
    canary = _run([sys.executable, '-c',
                   'import sys;print(len(open(sys.argv[1],"rb").read(1)))', str(CANARY)],
                  env, work)
    facts['repo_canary_readable'] = canary['returncode'] == 0
    facts['plugin_entry_configured'] = bool(built['surface']['plugin'])
    facts['expectation'] = dict(core_importable=built['surface']['core_cli'],
                                cli_on_path=built['surface']['core_cli'])
    facts['matches_expectation'] = (facts['core_importable'] == built['surface']['core_cli']
                                    and facts['cli_on_path'] == built['surface']['core_cli'])
    return facts
