"""Frozen local Python dependencies for the S5 arm entry points.

The arms run with ``PYTHONNOUSERSITE=1`` so a comparison cannot silently depend
on whatever happens to be installed in the invoking user's site-packages. The
few third-party imports the core CLI and MCP server need (numpy) therefore live
in an explicit, hashed directory that the harness adds to ``PYTHONPATH``.

This is a local dependency-freeze helper, not a sandbox, a lockfile for a Linux
host, or evidence that the dependency set is complete for a real GPU run.
"""
from __future__ import annotations

import hashlib
import json
import functools
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DEPS = ROOT / 'output' / 'plugin_completion' / '20261027-s5' / 'runtime-deps' / 'site'
MANIFEST = 'runtime-deps-manifest.json'
HASH = re.compile(r'[0-9a-f]{64}\Z')
OWNED = ('numpy', 'numpy.libs')


def deps_dir():
    """Return the frozen dependency directory, honouring an explicit override."""
    override = os.environ.get('S5_RUNTIME_DEPS')
    return Path(override).resolve() if override else DEFAULT_DEPS


def deps_problems(path=None):
    """归档依赖树**在当前解释器下**能不能用？返回问题清单，空 list 表示能用。

    为什么要有这个：`DEFAULT_DEPS` 里的 numpy 扩展是 `cpython-39-darwin.so`
    （冻结在 Python 3.9 上）。用 3.12 跑时它会走到 numpy 的「不要从源码目录导入」
    分支，于是**每一个** s5 验收用例都以 `verification output was not JSON (exit 1)`
    的样子失败 —— 看起来像判定层坏了，实际上是依赖树与解释器版本对不上。
    这个检查把那件事变成一个**点名的**原因。
    """
    target = Path(path or deps_dir())
    if not target.is_dir():
        return [f'runtime deps 目录不存在: {target}']
    tag = f'cpython-{sys.version_info.major}{sys.version_info.minor}'
    extensions = [item for item in target.rglob('*.so') if item.is_file()]
    if not extensions:
        return [f'runtime deps 里没有任何 .so 扩展（纯源码树？）: {target}']
    matching = [item for item in extensions if tag in item.name]
    if not matching:
        found = sorted({re.search(r'cpython-\d+', item.name).group(0)
                        for item in extensions if re.search(r'cpython-\d+', item.name)})
        return [f'runtime deps 的扩展是 {found}，当前解释器要 {tag} —— '
                f'这棵树在 {sys.version_info.major}.{sys.version_info.minor} 下不可用: {target}']
    return []


def tree_digest(path):
    """Hash a directory tree by relative name and file contents (no symlinks)."""
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError('dependency tree unavailable')
    entries = []
    for item in sorted(path.rglob('*')):
        rel = item.relative_to(path).as_posix()
        if item.is_symlink():
            raise ValueError('dependency tree contains a symlink')
        if item.is_dir():
            continue
        if not item.is_file():
            raise ValueError('dependency tree contains a non-file')
        entries.append((rel, hashlib.sha256(item.read_bytes()).hexdigest()))
    body = json.dumps(entries, separators=(',', ':')).encode('utf-8')
    return dict(files=len(entries), sha256=hashlib.sha256(body).hexdigest())


def python_path(root=ROOT, deps=None):
    """PYTHONPATH value shared by the B/C arm entry points."""
    deps = deps_dir() if deps is None else Path(deps)
    parts = [str(root)]
    if deps.is_dir() and not deps.is_symlink():
        parts.append(str(deps))
    return os.pathsep.join(parts)


def available(deps=None):
    """依赖树可用吗 —— **包含解释器兼容性**，不只是「目录在不在」。

    只查存在性是不够的：`DEFAULT_DEPS` 里的 numpy 扩展冻结在 Python 3.9 上，
    目录明明在、`is_dir()` 明明为真，但一导入就炸。用这种半吊子判据当 skip 守卫，
    结果就是守卫放行、用例全红，而红的理由看起来像判定层坏了。
    """
    deps = deps_dir() if deps is None else Path(deps)
    return deps.is_dir() and not deps.is_symlink() and not deps_problems(deps)


@functools.lru_cache(maxsize=None)
def sandbox_probe():
    """`sandbox-exec` 现在**真的能应用策略**吗？→ (bool, 原因)。

    文件存在只是一半。宿主自己跑在沙箱里时，嵌套的 `sandbox_apply` 会被拒绝
    （`Operation not permitted`），于是需要 Seatbelt 的那批用例会以各种不相干的
    理由失败。探一次比让它们各自失败强。
    """
    executable = Path('/usr/bin/sandbox-exec')
    if not executable.is_file():
        return False, 'sandbox-exec 不存在（非 macOS？）'
    try:
        probe = subprocess.run([str(executable), '-p', '(version 1)(allow default)',
                                '/usr/bin/true'],
                               capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as error:
        return False, f'sandbox-exec 起不来: {error}'
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout or '').strip().splitlines()
        return False, ('sandbox-exec 无法应用策略（嵌套沙箱？）: '
                       + (detail[-1][:160] if detail else f'exit {probe.returncode}'))
    return True, ''


def materialize(from_site, dest=None):
    """Copy a reviewed numpy installation into ``dest`` and hash the copy.

    The source is the invoking user's site-packages; nothing is downloaded and
    the copy is recorded so a later run can detect that the set changed. Only
    the numpy package, its shared-library directory and dist-info are copied.
    """
    source = Path(from_site).resolve()
    dest = deps_dir() if dest is None else Path(dest).resolve()
    if dest.exists() or dest.is_symlink():
        raise FileExistsError('runtime dependency directory already exists')
    names = [name for name in OWNED if (source / name).exists()]
    if not (source / 'numpy').is_dir():
        raise ValueError('numpy installation not found in the supplied site-packages')
    metadata = sorted(p.name for p in source.glob('numpy-*.dist-info') if p.is_dir())
    if len(metadata) != 1:
        raise ValueError('exactly one numpy distribution metadata directory is required')
    names.extend(metadata)
    dest.mkdir(parents=True, mode=0o700)
    for name in names:
        src, target = source / name, dest / name
        if src.is_dir():
            shutil.copytree(src, target)
        else:
            shutil.copy2(src, target)
    digest = tree_digest(dest)
    facts = dict(schema='s5-runtime-deps-v1', source=str(source), tree=digest,
                 python=sys.version.split()[0], copied=names)
    (dest / MANIFEST).write_text(
        json.dumps(facts, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    return facts


def verify_import(deps=None, timeout=60):
    """Import numpy in an isolated interpreter that only sees the frozen deps."""
    deps = deps_dir() if deps is None else Path(deps)
    if not available(deps):
        return dict(available=False, numpy_version=None, reason='dependency directory missing')
    env = {'PATH': '/usr/local/bin:/usr/bin:/bin', 'PYTHONPATH': str(deps),
           'PYTHONNOUSERSITE': '1', 'HOME': os.environ.get('HOME', '/tmp')}
    try:
        proc = subprocess.run([sys.executable, '-c',
                               'import numpy;print(numpy.__version__)'],
                              env=env, stdin=subprocess.DEVNULL, capture_output=True,
                              text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return dict(available=True, numpy_version=None, reason=type(exc).__name__)
    version = proc.stdout.strip()
    ok = proc.returncode == 0 and bool(version)
    return dict(available=True, numpy_version=version if ok else None,
                reason=None if ok else 'isolated numpy import failed')


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', metavar='SITE_PACKAGES',
                        help='copy numpy from this site-packages directory')
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args(argv)
    if args.prepare:
        print(json.dumps(materialize(args.prepare), sort_keys=True))
    if args.status or not args.prepare:
        deps = deps_dir()
        print(json.dumps(dict(path=str(deps), exists=available(deps),
                              tree=(tree_digest(deps) if available(deps) else None),
                              import_check=verify_import(deps)), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
