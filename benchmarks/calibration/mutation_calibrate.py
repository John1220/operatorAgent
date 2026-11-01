"""变异标定 harness：**每行变异要带 kill witness，分母只数有行为证人的行**

外部一手 (arXiv:2609.22220v1, 行 246-251, 逐字):
    "Scoring suites against unkillable rows deflates every protocol equally and informs
     about none. We therefore admit a mutant into the denominator only with a kill witness —
     a concrete, validity-gated input on which it verifiably fails — and quarantine the rest."

我方那张 21 行的表以前只报一个数 (「21/21 拦住、0 逃逸」)。这句话混了两件事:
  · **行为证人**行 —— 存在一个具体输入让变异真的改变可观察结果 (M1 恒返回编译取向:
    把一条 L2 失败送进 `run()` 就能看到差异);
  · **只有结构证人**的行 —— 今天在行为层与原文不可区分, 拦住它靠的是 AST/装配守卫
    (M16「取向自己判文案而不走分类器」就是这一类: 今天两条通道给出同一个桶名)。
分母把它们混在一起, 那个 21 就既不能说明覆盖、也不能说明缺什么 (I15/I11 那一族的又一形态:
报表里并排的数各取自不同的东西)。所以这里把两栏**分开计数**, 且两栏都必须被拦住。

三件让它不空转的事 (都在 `tests/unit/test_mutation_calibration.py` 里有对照):
  1. 跑不起来不等于拦住 —— pytest 没有汇总行就直接拒绝判定 (doc 49 §八 第 25 条 / I15);
  2. 锚点不唯一 ⇒ 这行根本没跑, 记成 `ANCHOR` 而不是 `KILLED`;
  3. 每条变异跑完立刻复原并断言逐字节相等, 否则红名单会串味道 (doc 55 §六 / I11)。

用法:
    python3 -B -m benchmarks.calibration.mutation_calibrate --spec benchmarks/mut_specs/fixer_stage_branch.py
    python3 -B -m benchmarks.calibration.mutation_calibrate --spec … --check-only   # 只核锚点与证人, 不改文件
"""
from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from benchmarks.experiments.misc import provenance
from typing import Dict, List, Sequence, Tuple

_PYtest_OK = (0, 1)  # 1 = 有断言失败, 这是预期的读数; 其它码 = 收集/语法错


@dataclass(frozen=True)
class Row:
    """一行变异。`witness` 不是注释, 它是这行能不能进分母的凭据。

    behavioral=True —— 存在一个**具体输入或条件**让这次改动能被观察到 (外部那句 kill witness
      的形状: "a concrete, validity-gated input on which it verifiably fails"), witness 写清
      那个输入是什么。
    behavioral=False —— 今天在行为层与原文不可区分, 拦住它的只可能是 AST / 装配这类**结构守卫**;
      这种行进结构分母, witness 写「为什么今天行为不可区分」, 也就是它的隔离理由。

    两栏都必须被拦住。分开计数不是为了放过结构行, 是为了不让「21/21」听起来像「21 条行为覆盖」。
    """
    name: str
    file_key: str
    old: str
    new: str
    witness: str
    behavioral: bool = True


@dataclass
class Result:
    name: str
    verdict: str          # KILLED / ESCAPED / ANCHOR / ERROR / STRUCTURAL
    fails: Tuple[str, ...]
    witness: str
    behavioral: bool


def load_spec(spec_path: str) -> Tuple[Dict[str, str], List[str], List[Row]]:
    """按路径执行 spec 模块并取 FILES/TARGETS/ROWS。

    ⚠️ 这里**不用 `isinstance(r, Row)`**: 用 `python -m benchmarks.calibration.mutation_calibrate` 跑时本文件
    的名字是 `__main__`, 而 spec 里 `from benchmarks.calibration.mutation_calibrate import Row` 会把它**再导
    一份** ⇒ 两个 Row 类对象, isinstance 对完全合法的规格报「不是 Row」。改成按字段结构核,
    与类身份无关。
    """
    loader = importlib.machinery.SourceFileLoader("mut_spec", spec_path)
    module = importlib.util.module_from_spec(
        importlib.util.spec_from_loader(loader.name, loader))
    loader.exec_module(module)
    rows = list(module.ROWS)
    fields = ("name", "file_key", "old", "new", "witness", "behavioral")
    for r in rows:
        missing = [f for f in fields if not hasattr(r, f)]
        if missing:
            raise TypeError(f"{getattr(r, 'name', r)!r} 缺字段 {missing} —— 每行必须带 kill witness")
        if not isinstance(r.behavioral, bool):
            raise TypeError(f"{r.name}: behavioral 必须是 bool (False = 只有结构证人)")
        if not r.witness.strip():
            raise TypeError(f"{r.name}: witness 不能为空 —— 没有证人的行不该存在于表里")
    return dict(module.FILES), list(module.TARGETS), rows


def check_shape(root: str, files: Dict[str, str], rows: Sequence[Row]) -> List[str]:
    """不动盘: 每行的锚点在它要改的那个文件里必须**恰好出现一次**, 且证人不是空话"""
    problems: List[str] = []
    for r in rows:
        if r.file_key not in files:
            problems.append(f"{r.name}: file_key {r.file_key!r} 不在 FILES 里")
            continue
        if not r.witness.strip():
            problems.append(f"{r.name}: 没有 kill witness —— 这行不该进任何分母")
        try:
            text = open(f"{root}/{files[r.file_key]}", encoding="utf-8").read()
        except OSError as exc:
            problems.append(f"{r.name}: 读不到文件 ({exc})")
            continue
        hits = text.count(r.old)
        if hits != 1:
            problems.append(f"{r.name}: 锚点在 {files[r.file_key]} 里出现 {hits} 次 (要恰好 1)")
    return problems


def run_suite(root: str, targets: Sequence[str]) -> Tuple[List[str], str]:
    # 继承环境, 只加禁字节码那一项: 本仓被 `sys.pycache_prefix` 的 pyc 骗过一次
    # (还原后仍执行被变异的字节码), 而清仓库 `__pycache__` 对它毫无作用。
    # 不自造 PATH/HOME —— 套件里有读环境的用例, 换掉环境就是在测另一套代码。
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([sys.executable, "-B", "-m", "pytest", *targets, "-q",
                           "-p", "no:cacheprovider"],
                          cwd=root, capture_output=True, text=True, env=env)
    if not re.search(r"\d+ (?:passed|failed)", proc.stdout):
        # I15: 没有汇总行不是「拦住」, 是压根没跑 (第一版就是这样报出过「8 条全逃逸」)
        raise RuntimeError("pytest 没跑出汇总行 ⇒ 拒绝判定:\n"
                           + proc.stdout[-2000:] + "\n" + proc.stderr[-2000:])
    fails = sorted({m.group(1) for m in
                    re.finditer(r"^FAILED \S+::(\S+)", proc.stdout, re.M)})
    if proc.returncode not in _PYtest_OK:
        raise RuntimeError(f"pytest 退出码 {proc.returncode} (收集/语法错?)")
    return fails, proc.stdout


def calibrate(root: str, files: Dict[str, str], targets: Sequence[str],
              rows: Sequence[Row], only: Sequence[str] = ()) -> List[Result]:
    pristine = {k: open(f"{root}/{v}", encoding="utf-8").read() for k, v in files.items()}
    selected = [r for r in rows if not only or r.name in only]
    results: List[Result] = []
    prev_fails: List[str] = []
    try:
        base_fails, _ = run_suite(root, targets)
        if base_fails:
            raise RuntimeError(f"基线不绿: {base_fails} ⇒ 先修基线再来标定")
        for r in selected:
            path = f"{root}/{files[r.file_key]}"
            if pristine[r.file_key].count(r.old) != 1:
                results.append(Result(r.name, "ANCHOR", (), r.witness, r.behavioral))
                continue
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(pristine[r.file_key].replace(r.old, r.new, 1))
            try:
                fails, _ = run_suite(root, targets)
            finally:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(pristine[r.file_key])
                assert open(path, encoding="utf-8").read() == pristine[r.file_key], path
            verdict = "KILLED" if fails else "ESCAPED"
            if fails and prev_fails and fails == prev_fails:
                # I11: 红名单与上一行逐字相同通常是归属串了味道, 不是两条真等价
                print(f"  ⚠️ {r.name}: 红名单与上一行完全相同 ({len(fails)} 条) —— 复核归属")
            prev_fails = fails
            results.append(Result(r.name, verdict, tuple(fails), r.witness, r.behavioral))
    finally:
        for k, text in pristine.items():
            with open(f"{root}/{files[k]}", "w", encoding="utf-8") as fh:
                fh.write(text)
            assert open(f"{root}/{files[k]}", encoding="utf-8").read() == text, k
    return results


def summarize(results: Sequence[Result]) -> str:
    beh = [r for r in results if r.behavioral]
    stru = [r for r in results if not r.behavioral]
    lines = [f"行为分母 (带 kill witness): {sum(1 for r in beh if r.verdict == 'KILLED')}"
             f"/{len(beh)} 拦住"]
    lines.append(f"结构分母 (今天行为不可区分, 只有结构守卫杀得掉): "
                 f"{sum(1 for r in stru if r.verdict == 'KILLED')}/{len(stru)} 拦住")
    other = [r for r in results if r.verdict not in ("KILLED",)]
    for r in other:
        lines.append(f"  {r.verdict}: {r.name}"
                     + (f"  ← 逃逸 ⇒ {r.witness[:70]}" if r.verdict == "ESCAPED" else ""))
    return "\n".join(lines)


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--check-only", action="store_true",
                    help="只核锚点唯一性与证人字段, 不改任何文件")
    args = ap.parse_args(argv)
    root = os.path.abspath(args.root)
    files, targets, rows = load_spec(args.spec)

    print(provenance.stamp(root))
    problems = check_shape(root, files, rows)
    if problems:
        print("❌ 规格本身有问题:")
        for p in problems:
            print("  - " + p)
        return 2
    print(f"锚点与证人核对: {len(rows)} 行全部唯一且有 witness")
    if args.check_only:
        return 0

    try:
        results = calibrate(root, files, targets, rows, only=args.only)
    except RuntimeError as exc:
        print(f"❌ {exc}")
        return 2
    for r in results:
        print(f"- {r.name}: {r.verdict} ({len(r.fails)} 条红)  witness={r.witness[:60]}")
    print("\n" + summarize(results))
    escaped = [r for r in results if r.verdict in ("ESCAPED", "ANCHOR", "ERROR")]
    return 1 if escaped else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
