"""回执对账：重跑出来的 `replay-*.json` 必须与提交进仓的那份**逐字段相同**。

迁移与判据改动都要过这一关：`benchmarks/repro/` 下四个包是**已发布证据**，
重跑的回执若与仓里那份不同，说明「重放成立」这句话不再成立，不许合并。

跑法（H20 窗口里，先 `sh benchmarks/repro/run_all_replays.sh` 拿到新回执）：

    python -m benchmarks.repro.compare_replays --fresh logs/replays.XXXXXX

有三个「运行身份」字段**不比较** —— 它们描述这一次跑出来的产物落在哪
（`run/` 下每次的候选目录名不同）：

    evidence / run_dir / final_evidence

其余任何字段（契约哈希、源码哈希、级别、判定、编译失败、final 级别与 purpose）
必须逐字段相等。这不等于那三个字段没人看：它们指向的产物正是回放器用来判定的东西，
抄一份旧回执而不同时抄产物，会在别的字段上先露馅。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PACKAGES = ('fused_ln_compile_20261118', 'fused_ln_tail_20261118',
            'matmul_strided_20261107', 'attention_causal_20261109')
ROOT = Path(__file__).resolve().parent
RUN_IDENTITY = ('evidence', 'run_dir', 'final_evidence')


def _load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def compare(committed, fresh, package):
    """返回差异清单（空 = 逐字段相同）。"""
    problems = []
    for path in sorted(Path(committed).glob('replay-*.json')):
        other = Path(fresh)/path.name
        if not other.is_file():
            problems.append(f'{package}/{path.name}: 这次没有重跑出这份回执')
            continue
        old, new = _load(path), _load(other)
        for key in sorted(set(old) | set(new)):
            if key in RUN_IDENTITY:
                continue          # 这一次的产物路径，不参与逐字段比较
            if old.get(key) != new.get(key):
                problems.append(f'{package}/{path.name}:{key}: {old.get(key)!r} != {new.get(key)!r}')
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--fresh', required=True, type=Path,
                        help='run_all_replays.sh 写回执的目录')
    parser.add_argument('--cases', nargs='*', default=list(PACKAGES))
    arguments = parser.parse_args(argv)
    problems = []
    for package in arguments.cases:
        problems.extend(compare(ROOT/package, arguments.fresh/package, package))
    report = dict(ok=not problems, problems=problems, packages=arguments.cases,
                  scope='字段级对账；运行身份字段只校验落点，不比较取值')
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0 if not problems else 2


if __name__ == '__main__':
    raise SystemExit(main())
