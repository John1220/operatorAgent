"""Validate a local plugin archive with a private DSH profile, without booting an agent.

Only DSH's plugin manager and boot-free configuration dump execute. Python and
Node standard libraries drive the lifecycle; the installed DSH YAML parser reads
its own config dump. No personal profile, model credential or GPU is used.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
REPO = (_SOURCE_ROOT if (_SOURCE_ROOT/'pyproject.toml').is_file()
        and (_SOURCE_ROOT/'operator_agent').is_dir() else None)
DEFAULT_DSH = Path('/Applications/DSH Desktop.app/Contents/Resources/app/node_modules/@deepseek-ai/dsh/lib/bin.js')
PROFILE = 'v07-delivery'
NAME = 'dsh-operator-agent'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)+'\n')


def archive_identity(path):
    with tarfile.open(path, 'r:gz') as archive:
        files = {}
        for member in archive.getmembers():
            parts = Path(member.name).parts
            if not parts or parts[0] != 'package' or '..' in parts or member.issym() or member.islnk():
                raise ValueError('plugin archive must contain regular package-relative files')
            if member.isfile():
                files[str(Path(*parts[1:]))] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
        member = archive.extractfile('package/package.json')
        package = json.load(member)
    if package.get('name') != NAME:
        raise ValueError('unexpected plugin package name')
    for required in ('index.mjs', 'cordis.patch.yml', 'LICENSE', 'NOTICE'):
        if required not in files:
            raise ValueError('plugin archive missing '+required)
    return dict(path=str(path), sha256=sha(path), name=NAME, version=package['version'], files=files)


def invoke(argv, label, root, env, commands, *, timeout=180):
    started = time.time()
    process = subprocess.Popen(list(map(str, argv)), cwd=root, env=env,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True)
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate(timeout=3)
    (root/(label+'.stdout')).write_bytes(stdout)
    (root/(label+'.stderr')).write_bytes(stderr)
    receipt = dict(argv=list(map(str, argv)), started_at=started, elapsed_s=time.time()-started,
        returncode=process.returncode, timed_out=timed_out,
        stdout=dict(path=label+'.stdout', sha256=sha(root/(label+'.stdout'))),
        stderr=dict(path=label+'.stderr', sha256=sha(root/(label+'.stderr'))))
    commands.append(receipt)
    save(root/(label+'.json'), receipt)
    if process.returncode != 0 or timed_out:
        raise RuntimeError(label+' failed; see retained stdout/stderr')
    return stdout.decode('utf-8')


def state(profile, artifact):
    manifest = profile/'package.json'
    value = json.loads(manifest.read_text()) if manifest.is_file() else {}
    installed = profile/'node_modules'/NAME
    present = os.path.lexists(installed)
    package = json.loads((installed/'package.json').read_text()) if (installed/'package.json').is_file() else {}
    return dict(dependency=value.get('dependencies', {}).get(NAME),
        bundle_registered=NAME in value.get('dsh', {}).get('profile', {}).get('bundles', []),
        package_installed=present, installed_version=package.get('version'),
        installed_files_match=bool(present and all((installed/name).is_file() and
            sha(installed/name)==digest for name, digest in artifact['files'].items())))


def run(plugin, python, output, *, dsh_bin=DEFAULT_DSH):
    plugin = Path(plugin).resolve()
    # Preserve the venv executable symlink: resolving it selects the base Python.
    python = Path(os.path.abspath(python))
    root = Path(output).resolve()
    dsh_bin = Path(dsh_bin).resolve()
    node = shutil.which('node')
    if not node or not dsh_bin.is_file():
        raise ValueError('installed Node and DSH CLI are required')
    console = {name:python.parent/name for name in ('operator-agent', 'operator-agent-mcp')}
    if not python.is_file() or not all(path.is_file() for path in console.values()):
        raise ValueError('clean installed Python environment must expose both console scripts')
    if REPO is not None and (root == REPO or REPO in root.parents):
        raise ValueError('profile evidence must be outside the source repository')
    artifact = archive_identity(plugin)
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    home, dsh_home = root/'home', root/'dsh-home'
    home.mkdir(mode=0o700);dsh_home.mkdir(mode=0o700)
    env = dict(PATH=str(python.parent)+os.pathsep+os.environ.get('PATH', '/usr/bin:/bin'),
        HOME=str(home), DSH_HOME=str(dsh_home), NO_COLOR='1',
        DSH_TELEMETRY_DISABLED='1', npm_config_cache=str(root/'npm-cache'),
        XDG_CACHE_HOME=str(root/'cache'), TMPDIR=str(root/'tmp'))
    (root/'tmp').mkdir(mode=0o700)
    profile = dsh_home/'profiles'/PROFILE
    commands = []
    manifest = dict(schema='v07-dsh-profile-lifecycle-v1', accepted=False,
        plugin=artifact, python=str(python), node=node, dsh_bin=str(dsh_bin),
        dsh_bin_sha256=sha(dsh_bin), profile=PROFILE, home=str(home), dsh_home=str(dsh_home),
        personal_profile_accessed=False, model_calls=0, gpu_executions=0,
        gpu_acceptance=False, formal_comparison_credit=False, commands=commands)
    saved_patch = None
    def call(args, label, **kwargs):
        return invoke([node, dsh_bin, *args], label, root, env, commands, **kwargs)
    try:
        identity = json.loads(invoke([python, '-I', '-c',
            'import importlib.metadata,importlib.util,json,sys;print(json.dumps(dict('
            'version=list(sys.version_info[:3]),prefix=sys.prefix,base_prefix=sys.base_prefix,'
            'package_version=importlib.metadata.version("operator-agent"),'
            'package_origin=importlib.util.find_spec("operator_agent").origin)))'],
            'python-identity', root, env, commands))
        manifest['python_identity'] = identity
        if identity['version'][:2] != [3,12] or identity['prefix'] == identity['base_prefix']:
            raise ValueError('acceptance requires an isolated Python 3.12 venv')
        origin = Path(identity['package_origin']).resolve()
        prefix = Path(identity['prefix']).resolve()
        if (prefix not in origin.parents or 'site-packages' not in origin.parts
                or REPO is not None and (origin == REPO or REPO in origin.parents)):
            raise ValueError('Python must resolve operator_agent from the specified venv site-packages')
        manifest['dsh_version'] = call(['--version'], 'dsh-version').strip()
        manifest['node_version'] = invoke([node, '--version'], 'node-version', root, env, commands).strip()
        manifest['before'] = state(profile, artifact)
        save(root/'profile-before.json', manifest['before'])
        call(['plugin', '--profile', PROFILE, 'add', str(plugin)], 'profile-add')
        manifest['added'] = state(profile, artifact)
        save(root/'profile-added.json', manifest['added'])
        if not all(manifest['added'][key] for key in ('dependency','bundle_registered','package_installed','installed_files_match')):
            raise ValueError('installed archive/bundle identity check failed')
        if manifest['added']['installed_version'] != artifact['version']:
            raise ValueError('installed plugin version differs from archive')
        patch = profile/'cordis.patch.yml'
        saved_patch = patch.read_bytes()
        (root/'profile-patch-before.yml').write_bytes(saved_patch)
        # DSH replaces the whole config in an override, including required MCP fields.
        mcp_config = dict(serverName='operator', transport='stdio',
                          command=str(console['operator-agent-mcp']),
                          toolCallTimeoutMs=3600000, failOnStartupError=True,
                          env=dict(OPERATOR_AGENT_WORKFLOW='host'))
        slash_config = dict(command=str(console['operator-agent']), timeoutMs=60000, hostModelBudget={})
        example = profile/'node_modules'/NAME/'examples/v07/profile-absolute-paths.patch.yml'
        override = example.read_text(encoding='utf-8')
        for name, path in console.items():
            placeholder = '/absolute/path/to/v07-venv/bin/'+name
            if not any(line.strip() == 'command: '+placeholder for line in override.splitlines()):
                raise ValueError('shipped profile example lacks the command placeholder for '+name)
            override = override.replace('command: '+placeholder+'\n',
                                        'command: '+json.dumps(str(path))+'\n')
        patch.write_text(override, encoding='utf-8')
        (root/'profile-patch-installed.yml').write_bytes(patch.read_bytes())
        call(['--profile', PROFILE, '--dump-config'], 'profile-composed')
        # Parse the real DSH composition with its installed YAML dependency.
        parser = ('import {createRequire} from "node:module";import{readFileSync}from"node:fs";'
            'const require=createRequire(process.argv[1]);const yaml=require("js-yaml");'
            'const schema=yaml.DEFAULT_SCHEMA.extend([new yaml.Type("tag:yaml.org,2002:js",'
            '{kind:"scalar",construct:value=>value})]);'
            'const entries=yaml.load(readFileSync(process.argv[2],"utf8"),{schema});'
            'console.log(JSON.stringify(entries.filter(x=>["mcp-operator","operator-tools"].includes(x.id))))')
        app_boot = dsh_bin.parents[2]/'dsh-app-boot/package.json'
        entries = json.loads(invoke([node, '--input-type=module', '-e', parser, app_boot,
            root/'profile-composed.stdout'], 'profile-inspected', root, env, commands))
        by_id = {entry['id']:entry for entry in entries}
        if (by_id.get('mcp-operator', {}).get('config') != mcp_config
                or by_id.get('operator-tools', {}).get('config') != slash_config):
            raise ValueError('DSH composition did not preserve the installed entry configuration')
        manifest['composed_entries'] = entries
        manifest['installation_verified'] = True
    except Exception as exc:
        manifest['error'] = dict(type=type(exc).__name__, detail=str(exc))
    finally:
        try:
            if saved_patch is not None:
                (profile/'cordis.patch.yml').write_bytes(saved_patch)
            if state(profile,artifact)['dependency']:
                call(['plugin', '--profile', PROFILE, 'remove', NAME], 'profile-remove')
            removed = state(profile,artifact)
            manifest['removed'] = removed
            save(root/'profile-removed.json', removed)
            if removed['dependency'] or removed['bundle_registered'] or removed['package_installed']:
                raise ValueError('profile removal left an installed dependency/bundle')
            if (profile/'package.json').is_file():
                composed = call(['--profile', PROFILE, '--dump-config'], 'profile-after-remove')
                if '\n- id: mcp-operator\n' in composed or '\n- id: operator-tools\n' in composed:
                    raise ValueError('removed plugin still appears in composed profile')
            manifest['removal_verified'] = True
        except Exception as exc:
            manifest['cleanup_error'] = dict(type=type(exc).__name__, detail=str(exc))
        manifest['accepted'] = bool(manifest.get('installation_verified') and
            manifest.get('removal_verified') and not manifest.get('error') and not manifest.get('cleanup_error'))
        manifest['finished_at'] = time.time()
        save(root/'manifest.json',manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plugin', required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dsh-bin', default=str(DEFAULT_DSH))
    args = parser.parse_args(argv)
    result = run(args.plugin,args.python,args.output,dsh_bin=args.dsh_bin)
    print(json.dumps(dict(accepted=result['accepted'],manifest=str(Path(args.output).resolve()/'manifest.json'))))
    return 0 if result['accepted'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
