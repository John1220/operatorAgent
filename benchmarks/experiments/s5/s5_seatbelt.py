"""Experimental macOS file-read boundary around the entire DSH process tree.

No model/GPU work. Public code is staged separately from supervisor evidence.
This is a file-access experiment, not a complete hostile-process/network jail.
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
from benchmarks.experiments.s5.s5_vertical_headless import DSH_ROOT, NODE

SANDBOX = '/usr/bin/sandbox-exec'
ROOT = Path(__file__).resolve().parents[3]
PLUGIN_FILES = ('index.mjs', 'model-relay.mjs', 'host-model-budget.mjs',
                'host-model-ledger.mjs', 'host-session-lifecycle.mjs', 'package.json')


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage_runtime(output):
    """Copy public runtime inputs once; never expose the whole repository."""
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '.DS_Store')
    for name in ('operator_agent', 'remote_scripts'):
        shutil.copytree(ROOT / name, out / 'core' / name, ignore=ignore)
    shutil.copytree(s5_runtime.deps_dir(), out / 'deps', ignore=ignore)
    (out / 'plugin').mkdir()
    for name in PLUGIN_FILES:
        shutil.copyfile(ROOT / 'integrations/dsh-operator' / name, out / 'plugin' / name)
    files = {}
    for f in sorted(out.rglob('*')):
        if f.is_symlink():
            raise ValueError('runtime symlinks are not permitted')
        if f.is_file():
            files[str(f.relative_to(out))] = checksum(f)
    manifest = dict(schema='s5-isolation-runtime-v1', root=str(out), files=files)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    return manifest


def verify_runtime(runtime):
    root = Path(runtime['root'])
    if root.is_symlink() or not root.is_dir():
        raise ValueError('runtime missing')
    actual = {str(f.relative_to(root)): checksum(f) for f in sorted(root.rglob('*'))
              if f.is_file() and f != root / 'manifest.json'}
    if any(f.is_symlink() for f in root.rglob('*')) or actual != runtime['files']:
        raise ValueError('public runtime drift')


def verify_source_version(runtime):
    from operator_agent.core.run_contract import protocol_identity
    for name, expected in protocol_identity().items():
        if runtime['files'].get('core/'+name)!=expected:
            raise ValueError('staged core no longer matches the supervisor source version')
    for name in PLUGIN_FILES:
        if runtime['files'].get('plugin/'+name)!=checksum(ROOT/'integrations/dsh-operator'/name):
            raise ValueError('staged plugin no longer matches the source version')


def _quote(path):
    return json.dumps(str(Path(path).resolve()), ensure_ascii=False)


def python_files():
    """Freeze only Python's launcher/link chain and framework executables."""
    files = set()
    path = Path(sys.executable).absolute()
    for _ in range(16):
        files.add(path)
        files.update(parent for parent in path.parents if parent.is_symlink())
        physical = path.parent.resolve() / path.name
        files.add(physical)
        if not physical.is_symlink():
            break
        target = Path(os.readlink(physical))
        path = Path(os.path.abspath(target if target.is_absolute() else physical.parent / target))
    else:
        raise ValueError('Python executable symlink chain exceeds bound')
    base = Path(sys.base_prefix).resolve()
    files.update(p for p in (base / 'Python', base / 'Resources/Python.app/Contents/MacOS/Python')
                 if p.is_file())
    return sorted(files)


def profile(read_roots, read_files, write_roots, *, python_prefix):
    """Reads default-denied; runtime libraries exclude ambient site-packages."""
    forms = ['(version 1)', '(allow default)', '(deny file-read*)', '(deny file-write*)',
             '(deny appleevent-send)', '(allow file-read* (literal "/"))']
    # macOS exposes /System/Volumes/Data as another spelling of user data.
    forms += ['(allow file-read* (require-all (subpath "/System") '
              '(require-not (subpath "/System/Volumes/Data"))))']
    for p in ('/usr', '/bin', '/sbin', '/dev', '/Library/Apple', '/private/var/db/dyld'):
        forms.append('(allow file-read* (subpath '+_quote(p)+'))')
    py_lib = Path(python_prefix) / 'lib'
    forms.append('(allow file-read* (require-all (subpath '+_quote(py_lib)+') '
                 '(require-not (regex #"/site-packages(/|$)"))))')
    for p in read_roots:
        forms.append('(allow file-read* (subpath '+_quote(p)+'))')
    for p in read_files:
        for spelling in sorted({str(Path(p).absolute()), str(Path(p).resolve())}):
            forms.append('(allow file-read* (literal '+json.dumps(spelling)+'))')
    for p in write_roots:
        forms.append('(allow file-read* file-write* (subpath '+_quote(p)+'))')
    forms.append('(allow file-write* (literal "/dev/null"))')
    # Ancestor traversal/stat is necessary; directory enumeration stays denied.
    ancestors = set()
    for p in [*read_roots, *read_files, *write_roots, py_lib]:
        ancestors.update(Path(p).resolve().parents)
        ancestors.update(Path(p).absolute().parents)
    for p in sorted(ancestors):
        forms.append('(allow file-read-metadata (literal '+_quote(p)+'))')
    return '\n'.join(forms)+'\n'


def configure(built, runtime):
    if sys.platform != 'darwin' or not Path(SANDBOX).is_file():
        raise RuntimeError('macOS Seatbelt required; unconfined fallback forbidden')
    verify_runtime(runtime)
    arm_root = Path(built['home']).parent
    if arm_root.resolve().is_relative_to(ROOT):
        raise ValueError('isolated participant must live outside the source repository')
    tmp = arm_root / 'tmp'
    for p in (Path(built['home']), Path(built['work']), tmp):
        p.mkdir(parents=True, exist_ok=True, mode=0o700)
    built['env'].update(TMPDIR=str(tmp), PYTHONDONTWRITEBYTECODE='1')
    runtime_root = Path(runtime['root'])
    reads = [Path(DSH_ROOT).parents[1], arm_root / 'bin', runtime_root / 'deps']
    if built['arm'] in ('B', 'C'):
        reads.append(runtime_root / 'core')
    if built['arm'] == 'C':
        reads.append(runtime_root / 'plugin')
    files = [*python_files(), Path(NODE), arm_root / 'patch.json']
    files += [Path(p) for p in ('/private/etc/hosts', '/private/etc/resolv.conf',
                               '/private/etc/localtime', '/private/etc/services',
                               '/private/var/select/sh')]
    if built.get('development_enabled'):
        files.append(Path(built['env']['OPERATOR_AGENT_S5_DEV_CONFIG']))
    writes = [Path(built['home']), Path(built['work']), tmp]
    text = profile(reads, files, writes, python_prefix=sys.base_prefix)
    target = arm_root / 'isolation.sb'
    with target.open('x') as f:
        f.write(text)
    built['isolation'] = dict(schema='s5-seatbelt-launch-v1', kind='macos-file-read-whitelist',
        profile=str(target), profile_sha256=checksum(target), runtime=runtime,
        read_roots=list(map(str, reads)), read_files=list(map(str, files)),
        write_roots=list(map(str, writes)), file_probe_passed=False,
        network_isolation_verified=False)
    # macOS rejects sandbox_init inside an already sandboxed process. Delegate
    # DSH's inner write-only fence to this mandatory outer read/write policy.
    # Never retry outside the outer policy on runner failure.
    built['env']['DSH_PERMISSION_MODE'] = 'danger-full-access'
    built['env']['OPERATOR_AGENT_S5_OUTER_REQUIRED'] = '1'
    built['outer_isolation_required'] = True
    built['isolation']['inner_policy_delegated_to_outer'] = True
    return built


def wrap(built, command):
    policy = built.get('isolation')
    if not policy:
        if (built.get('outer_isolation_required')
                or built.get('env', {}).get('OPERATOR_AGENT_S5_OUTER_REQUIRED') == '1'):
            raise ValueError('required outer isolation policy missing')
        return command
    if sys.platform != 'darwin' or not Path(SANDBOX).is_file():
        raise RuntimeError('required sandbox unavailable')
    verify_runtime(policy['runtime'])
    path = Path(policy['profile'])
    expected = profile(policy['read_roots'], policy['read_files'], policy['write_roots'],
                       python_prefix=sys.base_prefix)
    if (path.is_symlink() or not path.is_file() or checksum(path) != policy['profile_sha256']
            or path.read_text() != expected):
        raise ValueError('isolation policy drift; refusing unsandboxed launch')
    return [SANDBOX, '-f', str(path), *command]


def run(built, command, timeout=15):
    return subprocess.run(wrap(built, command), env=built['env'], cwd=built['work'],
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)


def archive_work(built, dest, max_bytes=10*1024*1024):
    """Copy participant-owned files without following links into private data."""
    source=Path(built['work']);dest=Path(dest)
    dest.mkdir(parents=True,exist_ok=False,mode=0o700)
    links=[];omitted=[];total=0
    for directory, dirs, files in os.walk(source,followlinks=False):
        folder=Path(directory)
        for name in list(dirs):
            item=folder/name
            if item.is_symlink():links.append(str(item.relative_to(source)));dirs.remove(name)
        for name in files:
            item=folder/name;rel=item.relative_to(source)
            if item.is_symlink():links.append(str(rel));continue
            if not item.is_file() or total+item.stat().st_size>max_bytes:
                omitted.append(str(rel));continue
            target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(item,target);total+=target.stat().st_size
    return dict(bytes=total,skipped_symlinks=links,omitted=omitted,complete=not omitted)
