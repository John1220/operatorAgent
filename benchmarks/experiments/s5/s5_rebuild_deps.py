"""把归档依赖树按**当前解释器**重建一遍。

为什么需要这个命令：`s5_runtime.DEFAULT_DEPS` 里冻结的 numpy 是按**建树时的
Python 版本**编译的（归档那份是 `cpython-39-darwin.so`）。换一个解释器跑时，
numpy 一导入就走到「不要从源码目录导入」的分支，于是整组 s5 验收会以
`verification output was not JSON (exit 1)` 这种看起来像判定层坏了的理由失败。

`s5_runtime.deps_problems()` 会先一步把这件事点名，验收组据此 skip。修的办法就是
用当前解释器重新冻一棵树：

    python -m benchmarks.experiments.s5.s5_rebuild_deps
    python -m benchmarks.experiments.s5.s5_rebuild_deps --from /path/to/site-packages

默认从**当前运行的解释器**的 site-packages 取 numpy（`site.getsitepackages()[0]`）。
不下载任何东西：只把 numpy、它的共享库目录与 dist-info 复制过去并记账。

已有的树会先被移到 `<deps>.old-<时间戳>` 而不是删除 —— 冻结的树是证据，换掉它应该是
可回退的。
"""
import argparse
import site
import sys
import time
from pathlib import Path

from benchmarks.experiments.s5 import s5_runtime


def _default_site() -> str:
    packages = site.getsitepackages()
    return packages[0]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--from', dest='source', default=None,
                        help='site-packages 目录（默认取当前解释器的第一个）')
    parser.add_argument('--dest', default=None, help='目标 deps 目录（默认 DEFAULT_DEPS）')
    parser.add_argument('--keep-old', action='store_true', default=True,
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    source = Path(args.source or _default_site())
    dest = Path(args.dest) if args.dest else s5_runtime.deps_dir()

    print(f'解释器 : {sys.version.split()[0]} ({sys.executable})')
    print(f'来源   : {source}')
    print(f'目标   : {dest}')

    if not (source / 'numpy').is_dir():
        print(f'✗ {source} 里没有 numpy —— 换一个 site-packages', file=sys.stderr)
        return 2

    staged = None
    if dest.exists() or dest.is_symlink():
        staged = dest.with_name(f'{dest.name}.old-{time.strftime("%Y%m%d-%H%M%S")}')
        dest.rename(staged)
        print(f'旧的依赖树已移到 : {staged}')

    try:
        receipt = s5_runtime.materialize(source, dest)
    except Exception:
        # 重建失败要把旧树放回去：把唯一可用的那棵弄丢，比一开始就不重建更糟。
        if staged is not None and not dest.exists():
            staged.rename(dest)
            print(f'重建失败，已恢复旧树 : {dest}', file=sys.stderr)
        raise

    problems = s5_runtime.deps_problems(dest)
    if problems:
        print('✗ 重建后仍不可用：' + '; '.join(problems), file=sys.stderr)
        return 1
    print(f'✓ 依赖树已按当前解释器重建')
    print(f'  {receipt}')
    if staged is not None:
        print(f'  确认无误后可删掉旧的: {staged}')
    return 0


if __name__ == '__main__':
    sys.exit(main())