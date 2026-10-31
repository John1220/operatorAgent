"""§22.1 目录权限本地零依赖演示 —— 亲眼看一下「越权写在写的那一刻失败」

不需要 GPU、不需要 LLM、不联网，全部投影落在一个临时目录里:

    python3 benchmarks/demos/demo_run_layout.py

四个场景回答的是**同一句话**: 设计 §22.1 那句「违反时写入必须**失败**，而不是
『事后被审计发现』」在代码里到底是什么样子。

  1. 七格布局与权限位 —— 五个受保护格 `0500`、`candidate`/`build` `0700`，
     以及已有的那种「权限位不对就报错，不静默修好」
  2. **越权写**：候选往 `measurements/` / `report/` / `contract/` 写 ⇒ `CrossRoleWrite`，
     而且失败**不留痕**（连 `.tmp` 都没有）
  3. **裸写**（不过唯一入口）：新建条目、覆盖已有 artifact —— 都抛 `PermissionError`；
     同时经 `write_for` 的正当写入是好的（否则上面那些「拦住了」没有意义）
  4. **同源**：§21.5 的判据拿**真路径**仍然逐格命中，而且判据与布局读**同一个对象**

可执行文档: `tests/unit/test_run_layout.py`(32 条); 变异表 8 行:

    python3 -B -m benchmarks.mutation_calibrate --spec benchmarks/mut_specs/run_layout_doors.py

⚠️ 边界 (与 CHECKLOG I82 / 状态文档同一句话): 这只证明 **harness 侧**那一半 ——
候选与 harness 同一个 uid，属主可以自己 `chmod` 回去；远端 runner 侧的布局一个字都没说；
§22.2 的超时/watchdog/quota/网络关闭/进程树清理**一条都没落**。所以这不是沙箱。
"""
from __future__ import annotations

import os
import stat
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from operator_agent.generate import agent_runtime as ar  # noqa: E402
from operator_agent.core import run_layout as rl  # noqa: E402


def _line(title: str) -> None:
    print("\n" + "=" * 68 + f"\n{title}\n" + "=" * 68)


def scene_1_layout(root: str) -> rl.RunLayout:
    _line("1. 七格布局与权限位（形状本身就是数据）")
    layout = rl.run_layout(root)
    for name in rl.DIR_NAMES:
        row = layout.describe()["dirs"][name]
        flag = "受保护" if row["protected"] else "可写  "
        print(f"  {name:<13} {flag}  写者={row['writer']:<13} mode={row['mode']}")
    print(f"  否定声明: 挡住同 uid 候选={layout.describe()['blocks_same_uid_candidate']}"
          f" / 覆盖远端={layout.describe()['covers_remote_runner']}")
    print("  ⚠️ 这两个 False 是给读者看的**否定**声明，不是「还没填」")
    os.chmod(os.path.join(root, "report"), 0o700)
    try:
        rl.run_layout(root)
        print("  !! 权限位被改过却通过了 —— 守卫坏了")
    except rl.LayoutError as exc:
        print(f"  权限位被改 ⇒ 报错而不是顺手改回来: {str(exc)[:64]}…")
    os.chmod(os.path.join(root, "report"), rl.PROTECTED_DIR_MODE)
    return layout


def scene_2_cross_role(layout: rl.RunLayout) -> None:
    _line("2. 越权写：候选往受保护格写 ⇒ 在**碰文件系统之前**失败")
    for name in rl.PROTECTED_DIR_NAMES:
        try:
            layout.write_for(rl.ROLE_CANDIDATE, f"{name}/cheat.py", b"x")
            print(f"  !! candidate 写进了 {name}/")
        except rl.CrossRoleWrite as exc:
            leftovers = os.listdir(os.path.join(layout.root, name))
            print(f"  candidate → {name}/ 拒 ({str(exc).split('（')[0].strip()}); "
                  f"格内残留={leftovers}")
    try:
        layout.write_for(rl.ROLE_HARNESS, "measurements/l3.json", b"x")
        print("  !! harness 也能写 measurements/ —— 那就不是一格一个写者")
    except rl.CrossRoleWrite:
        print("  harness → measurements/ 也拒（一格一个写者，不是「harness 什么都能写」）")


def scene_3_raw_writes(layout: rl.RunLayout) -> None:
    _line("3. 裸写：不过唯一入口，在同一 uid 下也失败")
    for name in rl.PROTECTED_DIR_NAMES:
        try:
            with open(os.path.join(layout.root, name, "new.txt"), "w") as fh:
                fh.write("x")
            print(f"  !! {name}/ 能裸写")
        except PermissionError:
            print(f"  open({name}/new.txt, 'w') → PermissionError（目录写位那层）")
    path = layout.write_for(rl.ROLE_ORACLE, "measurements/l3.json", b"original")
    print(f"  经入口写出的 artifact mode={oct(stat.S_IMODE(os.stat(path).st_mode))}")
    try:
        with open(path, "w") as fh:
            fh.write("tampered")
        print("  !! 覆盖成功")
    except PermissionError:
        print("  覆盖已有 artifact → PermissionError（文件写位那层）")
    scratch = os.path.join(layout.root, "candidate", "scratch.txt")
    with open(scratch, "w") as fh:
        fh.write("ok")
    print(f"  candidate/ 里的**新**文件仍然可写（可写格是刻意可写的）: "
          f"{os.path.basename(scratch)}")
    print(f"  写完之后受保护格的权限位复原: measurements = {oct(layout.mode_of('measurements'))}")


def scene_4_one_source(layout: rl.RunLayout) -> None:
    _line("4. 同源：§21.5 的判据拿**真路径**仍然命中，而且读的是同一个对象")
    print(f"  agent_runtime.PROTECTED_DIR_NAMES is run_layout.PROTECTED_DIR_NAMES = "
          f"{ar.PROTECTED_DIR_NAMES is rl.PROTECTED_DIR_NAMES}")
    print(f"  candidate_path(root) == layout.path('candidate/code.py') = "
          f"{ar.candidate_path(layout.root) == layout.path('candidate/code.py')}")
    print(f"  角色路径那一档（无 run_root, 今天生产用的）: {ar.candidate_path()!r}")
    ctx = ar.ActionContext(spend_committed=True,
                           authorized_roots=(os.path.join(layout.root, "candidate"),))
    for name in rl.PROTECTED_DIR_NAMES:
        out = ar._check_protected_paths(
            ar.Patch(path=os.path.join(layout.root, name, "x.py"), new_source="x = 1"), ctx)
        print(f"  判据 protected_paths @ 真路径 {name}/x.py → {out.state}")
    inside = ar._check_write_scope(
        ar.Patch(path=os.path.join(layout.root, "candidate", "code.py"), new_source="x = 1"),
        ctx)
    print(f"  判据 write_scope @ candidate/code.py → {inside.state}")


def main() -> None:
    root = os.path.join(tempfile.mkdtemp(prefix="demo-run-layout-"), "run")
    print(f"投影目录: {root}")
    layout = scene_1_layout(root)
    scene_2_cross_role(layout)
    scene_3_raw_writes(layout)
    scene_4_one_source(layout)
    _line("边界（不许读成已经做到的事）")
    print("  · 候选与 harness **同一个 uid** ⇒ 属主可自己 chmod 回去（要 §22.2）")
    print("  · 远端 runner 侧的布局与权限位：一个字都没说")
    print("  · 不是沙箱：§22.2 的超时/watchdog/quota/网络关闭/进程树清理一条都没落")
    print("  · `run_root` 今天没有生产调用点 ⇒ 形态是「声明 + 可运行守卫」")


if __name__ == "__main__":
    main()