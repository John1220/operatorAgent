"""「这份读数属于哪一份代码」—— 跑动窗口内的代码指纹, 由**另一个进程**算

为什么需要它 (doc 50 §7.5 / doc 56 用 Hearsay 那条读我方自己):
2026-09-29 那次五臂真机端到端跑 (12 分钟、约 10 万次 token) 跑完才发现, 跑动窗口内有并发会话
正在改同一条路径上的 5 个文件 ⇒ 五臂不是同一份代码, 而**一次真机跑没法「重跑一遍就好」**。
当时的报表里没有任何一行说明这件事。

为什么哈希要由子进程算, 而不是驱动自己算 (doc 56 逐字引的外部结论:
"The remedy is another writer, not richer logging"):
驱动说「我跑的是这份代码」是自述 (introspective) —— 它算的哈希只能证明它读到了什么, 不能证明
跑动期间盘上没变。子进程的价值有两层: ① 它是**另一个写者**, 与被测的 Conductor 不共享任何状态;
② 它失败时返回 `None` 而不是兜成空 dict —— 那正是 I21/I24 那一族 (兜底值冒充读数): 「取不到」与
「取到了但为空」在报表里必须长得不一样。

依赖清单是**算出来的**不是手抄的 (closure): 手抄的名单从抄下那一刻就开始漂, 而漂的方向恰好是
「漏掉新加的那个依赖」—— 那正是这份名单要抓的事。

⚠️ 名字撞车: 同目录的 `verify_provenance_gate.py` 管的是「出处门」(这条**配方/读数声称来自哪台
设备**, doc 49 §7.9)，本模块管的是「这份**读数**属于哪一份代码」。两件事都叫 provenance 是巧合,
读代码时别把它们当成同一个门。
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

# 只跟着这些顶层包走 (外部库的字节不属于本仓, 变了也不该由本仓的报表解释)
LOCAL_PACKAGES = ("operator_agent", "remote_scripts", "benchmarks")

_ABSENT = "ABSENT"          # 声明了但不存在 —— 必须与「存在且内容已知」区分开


def stamp(root: str = ".") -> str:
    """**读数条件与读数同印**: 哪套解释器 / 装不装 torch / 哪个 HEAD / 工作树干不干净

    2026-09-30 实撞: 同一份 HEAD 的干净 worktree, 用 `~/opt/miniconda3/bin/python3` (3.9.12,
    torch 2.8.0) 读是 986 passed / 1 skipped, 用 `/usr/bin/python3` (3.9.6, 无 torch) 读是
    981 passed / 1 failed / 5 skipped —— 两边总数都是 987, 差的 5 条全是 CHECKLOG I5 那一族
    (用例判定随本机有无 torch 翻转), 零行代码不同。`python3` 这个名字解析到哪一套取决于当时的
    `PATH`, 所以**同一台机器上也有两套读数条件** (CHECKLOG I24)。
    """
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                             capture_output=True, text=True).stdout.strip() or "?"
        dirty = subprocess.run(["git", "status", "--porcelain", "-uall"], cwd=root,
                               capture_output=True, text=True).stdout.strip()
    except OSError:
        rev, dirty = "?", "?"
    return (f"读数条件: python={sys.version.split()[0]} @ {sys.executable} "
            f"torch={'有' if _has_torch() else '无'} "
            f"HEAD={rev} 未提交改动={'有' if dirty else '无'}")


def _has_torch() -> bool:
    import importlib.util
    return importlib.util.find_spec("torch") is not None


def _module_to_rel(mod: str) -> Optional[str]:
    """`operator_agent.conductor` -> `operator_agent/conductor.py`; 包 -> `<pkg>/__init__.py`"""
    parts = mod.split(".")
    if parts[0] not in LOCAL_PACKAGES:
        return None
    return "/".join(parts) + ".py"


def _climb(package: str, level: int) -> str:
    """`from .` / `from ..` 的层数: level=1 是当前包, level=2 是上一层"""
    parts = package.split(".") if package else []
    drop = level - 1
    if drop:
        parts = parts[:-drop] if drop < len(parts) else []
    return ".".join(parts)


def _is_package(mod: str, root: str) -> bool:
    """`operator_agent` 是包 (有目录), `operator_agent.verifier` 是模块 —— 决定 names 要不要展开

    第一版无条件展开, 于是 `from .verifier import Verifier` 里的**类名** `Verifier` 被当成子模块,
    闭包多出一个 `operator_agent/Verifier.py` —— 在大小写不敏感的 macOS 盘上 `exists()` 竟然为真,
    于是同一个文件被登记了两份 (29 个「文件」里有一个是幻影)。名单的用途是「谁变了这份读数就
    不能归给 HEAD」, 多算一条不会让结论变安全, 只会让「29 个文件」这个数字不可信。
    """
    return (Path(root).resolve() / mod.replace(".", os.sep)).is_dir()


def local_imports(path: Path, root: str = ".") -> List[str]:
    """这个文件 import 了哪些本仓模块 —— 含**相对 import** (`from .verifier import …`)

    ⚠️ 第一版只认绝对名, 于是 `operator_agent/conductor.py` 里那一串 `from .xxx import` 一条也没
    认出来, 闭包停在 8 个文件上、verifier/measurement 全都不在名单里 —— 而这份名单的用途恰恰是
    「谁变了这份读数就不该归给 HEAD」。少一层等于这条守卫在最重要的那条边上从没跑过。
    只看可执行语句: 注释与 docstring 里提到的模块名不算 (doc 49 §八 那条「被 docstring 喂出假阳性」)。
    """
    rel = os.path.relpath(str(path), str(Path(root).resolve()))
    package = os.path.dirname(rel).replace(os.sep, ".")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    mods: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                prefix = _climb(package, node.level)
                if node.module:
                    mods.append(f"{prefix}.{node.module}" if prefix else node.module)
                elif _is_package(prefix, root):
                    mods.extend(f"{prefix}.{a.name}" for a in node.names)
            elif node.module:
                mods.append(node.module)
                # `from operator_agent import conductor` —— 子模块名在 names 里, 不在 module 里
                if (_is_package(node.module, root)
                        and node.module.split(".")[0] in LOCAL_PACKAGES):
                    mods.extend(f"{node.module}.{a.name}" for a in node.names)
        elif isinstance(node, ast.Import):
            mods.extend(a.name for a in node.names)
    return mods


def closure(entry: str, root: str = ".", extra_entries: Sequence[str] = ()) -> List[str]:
    """从入口出发, 沿「本仓模块」的 import 边做 BFS, 返回所有会被装载的仓内文件 (相对路径)

    找不到的模块直接跳过 (那是外部依赖或可选 import); 但**入口文件自己**必须存在, 否则抛 ——
    入口拼错时静默返回空清单就等于这条守卫从没跑过 (doc 49 §八 那条「守卫的空转」)。
    """
    rp = Path(root).resolve()
    queue: List[str] = []
    for e in [entry, *extra_entries]:
        p = Path(root) / e if not os.path.isabs(e) else Path(e)
        if not p.exists():
            raise FileNotFoundError(f"closure 的入口不存在: {e}")
        queue.append(str(p.relative_to(rp)) if p.is_relative_to(rp) else str(p))
    # 按 **realpath** 去重: 大小写不敏感的盘上 `Verifier.py` 与 `verifier.py` 是同一个文件,
    # 各登记一次就会让「多少个文件」这个数不可信 (名单的用途就是让那个数可信)
    found: Dict[str, str] = {}
    visited = set()
    while queue:
        rel = queue.pop(0)
        if rel in visited:
            continue
        visited.add(rel)
        path = rp / rel
        if not path.exists():
            continue            # 声明过但不在盘上: 不登记, 也不报错 (可选依赖)
        real = os.path.realpath(str(path))
        key = os.path.relpath(real, str(rp)) if real.startswith(str(rp) + os.sep) else rel
        found.setdefault(key, rel)
        # 包的 `__init__.py` 也是「跑过的代码」: `import operator_agent.conductor` 会先执行
        # `operator_agent/__init__.py`, 而那个文件自己又 import 了一批模块
        parts = rel.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            queue.append("/".join(parts[:i]) + "/__init__.py")
        for mod in local_imports(path, root):
            r = _module_to_rel(mod)
            if r and r not in visited:
                queue.append(r)
    return sorted(found)


def snapshot(paths: Iterable, root: str = ".") -> Dict[str, str]:
    """每个文件的内容哈希; 不在盘上的登记成 `_ABSENT` 而不是跳过

    ⚠️ `root` 不存在 ⇒ 抛, 不返回一张全 ABSENT 的表。这条是写测试时撞出来的: 第一版拿一个不存在的
    根目录去跑, 子进程「成功」地算出两份**全 ABSENT** 的清单, 于是 `diff` 报「什么都没变」——
    门在一份证明不了任何事的名单上绿着, 正是 doc 49 §八 那条「守卫的空转比守卫的缺失更安全-looking」。
    """
    rp = Path(root).resolve()
    if not rp.is_dir():
        raise FileNotFoundError(f"指纹根目录不存在: {root}")
    out: Dict[str, str] = {}
    for rel in sorted(set(paths)):
        p = rp / rel
        if not p.exists():
            out[rel] = _ABSENT
            continue
        out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return out


def snapshot_via_subprocess(paths: Iterable, root: str = ".") -> Optional[Dict[str, str]]:
    """让**另一个进程**去读盘 (见模块 docstring)。失败返回 None, 不兜成空 dict。"""
    payload = json.dumps({"paths": sorted(set(paths)), "root": str(Path(root).resolve())})
    code = (
        "import json,sys;sys.path.insert(0,%r);"
        "from benchmarks.experiments.misc.provenance import snapshot;"
        "d=json.loads(sys.stdin.read());"
        "print(json.dumps(snapshot(d['paths'], d['root'])))" % os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    )
    try:
        proc = subprocess.run([sys.executable, "-B", "-c", code], input=payload,
                              capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"  ⚠️ 指纹子进程没跑起来: {type(exc).__name__}: {exc}")
        return None
    if proc.returncode != 0:
        print(f"  ⚠️ 指纹子进程退出码 {proc.returncode}: {proc.stderr.strip()[:200]}")
        return None
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        print(f"  ⚠️ 指纹子进程的读数不是 JSON: {proc.stdout[:120]!r}")
        return None


def diff(before: Optional[Dict[str, str]], after: Optional[Dict[str, str]]) -> List[str]:
    """两次快照之间变了什么。`before/after` 为 None ⇒ 当成「取到了但为空」—— 不许被读成「没变」

    两份**全 ABSENT** 的清单也算「证明不了事」: 名单或根目录错了的时候, 逐元素比反而完全相同,
    门会绿 (上面那条注释里撞到的就是这个)。
    """
    if before is None or after is None:
        return ["指纹没取到 (取不到 ≠ 没变化)"]
    empty = [m for m in (before, after) if m and all(v == _ABSENT for v in m.values())]
    if empty:
        return [f"指纹清单整张都是 ABSENT ({len(empty)} 份) ⇒ 名单或根目录不对, 这道门在空转"]
    changes: List[str] = []
    for rel in sorted(set(before) | set(after)):
        b, a = before.get(rel), after.get(rel)
        if b is None:
            changes.append(f"新增 {rel}")
        elif a is None:
            changes.append(f"这次没登记 {rel}")
        elif b != a:
            if b == _ABSENT:
                changes.append(f"跑动期间出现 {rel}")
            elif a == _ABSENT:
                changes.append(f"跑动期间被删 {rel}")
            else:
                changes.append(f"被改 {rel}")
    return changes


def report(before: Optional[Dict[str, str]], after: Optional[Dict[str, str]],
           entry: str, root: str = ".") -> bool:
    """打印归属结论; True = 这份读数可以归给当前 HEAD"""
    changes = diff(before, after)
    print(f"  代码指纹: {len(before or {})} 个仓内文件 ({entry} 的 import 闭包), "
          f"另一个进程各读一次盘")
    if not changes:
        print("  ✅ 跑动窗口内这些文件的哈希未变 ⇒ 读数可归给上面那个 HEAD")
        return True
    print(f"  ❌ 跑动窗口内 {len(changes)} 处不同 ⇒ **这份读数不能归给任何 commit**:")
    for c in changes[:12]:
        print(f"     - {c}")
    if len(changes) > 12:
        print(f"     - …另有 {len(changes) - 12} 条")
    return False
