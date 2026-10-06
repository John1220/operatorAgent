"""§22.1 的运行布局与目录权限位：**越权写入必须在写的那一刻失败**

设计原文（`docs/design/autonomous-operator-agent-design.md:1536-1550`）给了七个目录格
和一句判据：「候选**不能**写入 `contract/`、`harness/`、`baseline/`、`measurements/`、
`report/`。这条不是『约定』，而是权限位：违反时写入必须**失败**，而不是『事后被审计
发现』。」

这一组用例钉的就是「失败」这两个字，分四层：
  ① **形状**：七格、权限位、写者声明，都是数据（`describe()`），不靠读文档；
  ② **越权写**：角色矩阵逐个组合 —— 该失败的在**碰文件系统之前**就失败（不留临时文件、
     不改权限位）；
  ③ **裸写**：不经过唯一入口的写入在同一 uid 下也失败（目录写位 + 文件写位两层）；
  ④ **同源**：§21.5 的判据与 §22.1 的目录名读**同一个元组**，路径由同一个构造点给出。

⚠️ 本文件**不**证明、也不许被读成：
  · 「候选被挡住了」—— 候选与 harness 同一个 uid，属主可以自己 `chmod` 回去（§22.2）；
  · 「远端也被管住了」—— 这里一个字都没说远端 runner 侧的布局；
  · 「有沙箱」—— §22.2 的编译/运行超时、watchdog、quota、网络关闭、进程树清理、
    artifact 大小上限**都还没落地**（`run_layout.describe()` 里那两个恒 False 的键
    就是给读者看的否定声明）。
"""

import ast
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operator_agent import agent_runtime as ar
from operator_agent import run_layout as rl
from operator_agent.run_layout import (
    ARTIFACT_MODE, DIR_NAMES, PROTECTED_DIR_MODE, PROTECTED_DIR_NAMES, ROLES,
    ROLE_CANDIDATE, ROLE_ORACLE, ROLE_REPORT_WRITER, ROLE_BUILD, ROLE_HARNESS,
    ROLE_WRITERS, WRITABLE_DIR_MODE, WRITABLE_DIR_NAMES,
    CrossRoleWrite, LayoutError, RunLayout, layout_path, run_layout,
)


def _tmp_root() -> str:
    return os.path.join(tempfile.mkdtemp(prefix="runlayout-test-"), "run")


class TestLayoutShape(unittest.TestCase):
    """七格、权限位、写者：形状本身是数据"""

    def setUp(self):
        self.layout = run_layout(_tmp_root())

    def test_the_seven_dirs_exist_with_the_declared_modes(self):
        for name in DIR_NAMES:
            p = os.path.join(self.layout.root, name)
            self.assertTrue(os.path.isdir(p), f"{name}/ 没建出来")
            want = (PROTECTED_DIR_MODE if name in PROTECTED_DIR_NAMES
                    else WRITABLE_DIR_MODE)
            self.assertEqual(want, stat.S_IMODE(os.lstat(p).st_mode),
                             f"{name}/ 的权限位不是 {oct(want)}")

    def test_the_five_protected_dirs_are_exactly_the_design_ones(self):
        self.assertEqual(("contract", "harness", "baseline", "measurements", "report"),
                         PROTECTED_DIR_NAMES)
        self.assertEqual(("candidate", "build"), WRITABLE_DIR_NAMES)
        self.assertEqual(set(DIR_NAMES),
                         set(PROTECTED_DIR_NAMES) | set(WRITABLE_DIR_NAMES))

    def test_every_dir_has_exactly_one_declared_writer(self):
        """闭合词汇: 每格一个写者, 值域不跑出 `ROLES`, 七格全覆盖"""
        self.assertEqual(set(DIR_NAMES), set(ROLE_WRITERS))
        self.assertTrue(set(ROLE_WRITERS.values()) <= ROLES,
                        f"写者里有未声明的角色: {set(ROLE_WRITERS.values()) - ROLES}")
        for d in DIR_NAMES:
            self.assertEqual(1, len(rl.WRITABLE_BY[d]), f"{d} 有不止一个写者")

    def test_describe_says_what_is_not_covered(self):
        """否定声明必须是数据: 「挡住同 uid 候选」「覆盖远端」两个键恒 False"""
        d = self.layout.describe()
        self.assertEqual(self.layout.root, d["root"])
        self.assertEqual(7, len(d["dirs"]))
        for key in ("blocks_same_uid_candidate", "covers_remote_runner"):
            self.assertIn(key, d)
            self.assertIs(False, d[key],
                          f"{key} 变成了 {d[key]!r} —— 要么真的做到了 (那要证据), "
                          "要么这个键被写坏了")

    def test_an_empty_run_root_is_refused(self):
        """空串不是「默认布局」。建一个相对 CWD 的布局等于把语义偷偷换掉"""
        for bad in ("", "   ", None, 0):
            with self.assertRaises(LayoutError):
                run_layout(bad)

    def test_rerunning_verifies_instead_of_repairing(self):
        """权限位不对 ⇒ 报错, 不静默改回来 (那会把「上一个写窗没复原」藏起来)"""
        os.chmod(os.path.join(self.layout.root, "measurements"), 0o700)
        with self.assertRaises(LayoutError) as ctx:
            run_layout(self.layout.root)
        self.assertIn("权限位", str(ctx.exception))
        # 报错之后**没有**被动过: 还是 0700, 而不是被顺手修成 0500
        self.assertEqual(0o700, stat.S_IMODE(
            os.lstat(os.path.join(self.layout.root, "measurements")).st_mode))

    def test_a_symlinked_dir_is_not_a_layout_slot(self):
        root = _tmp_root()
        os.makedirs(root)
        elsewhere = tempfile.mkdtemp(prefix="runlayout-elsewhere-")
        os.mkdir(os.path.join(root, "contract"))
        os.rmdir(os.path.join(root, "contract"))
        os.symlink(elsewhere, os.path.join(root, "contract"))
        with self.assertRaises(LayoutError) as ctx:
            run_layout(root)
        self.assertIn("符号链接", str(ctx.exception))

    def test_a_same_root_can_be_reopened(self):
        """幂等: 已经建好的布局再核一次是允许的 (resume 要用)"""
        again = run_layout(self.layout.root)
        self.assertEqual(self.layout.root, again.root)
        self.assertEqual(self.layout.describe()["dirs"], again.describe()["dirs"])


class TestCrossRoleWritesFail(unittest.TestCase):
    """② 越权写: 角色矩阵逐个组合, 失败点在**碰文件系统之前**"""

    def setUp(self):
        self.layout = run_layout(_tmp_root())

    def test_the_whole_role_by_dir_matrix(self):
        """5 角色 × 7 格 = 35 种组合: 只有 7 种合法, 其余 28 种必须抛 CrossRoleWrite"""
        legal, refused = 0, 0
        for dir_name in DIR_NAMES:
            for role in sorted(ROLES):
                rel = f"{dir_name}/probe.txt"
                if ROLE_WRITERS[dir_name] == role:
                    path = self.layout.write_for(role, rel, b"ok")
                    self.assertTrue(os.path.exists(path))
                    legal += 1
                else:
                    with self.assertRaises(CrossRoleWrite, msg=f"{role} 写进了 {dir_name}"):
                        self.layout.write_for(role, rel, b"should-not-land")
                    refused += 1
        self.assertEqual((7, 28), (legal, refused))

    def test_the_candidate_cannot_write_any_protected_dir(self):
        """§22.1 那一句, 逐格验一遍 (设计点名了这五个)"""
        for name in PROTECTED_DIR_NAMES:
            with self.assertRaises(CrossRoleWrite):
                self.layout.write_for(ROLE_CANDIDATE, f"{name}/x.py", b"cheat")

    def test_a_refused_write_leaves_no_trace(self):
        """失败要**在碰文件系统之前**: 不留临时文件、不改权限位、不建目录

        这条不是洁癖: `measurements/` 正是要被保护的那一格, 一次失败的越权写如果在
        里面留下 `.tmp` 残骸, 「失败」本身就污染了证据。
        """
        before_mode = self.layout.mode_of("measurements")
        with self.assertRaises(CrossRoleWrite):
            self.layout.write_for(ROLE_CANDIDATE, "measurements/l3.json", b"x")
        self.assertEqual(before_mode, self.layout.mode_of("measurements"))
        self.assertEqual([], os.listdir(os.path.join(self.layout.root, "measurements")))

    def test_ensure_dir_goes_through_the_same_door(self):
        """嵌套 artifact 路径要能被表达, 但不能绕开角色判据"""
        sub = self.layout.ensure_dir(ROLE_ORACLE, "measurements/gemm/h20")
        self.assertTrue(os.path.isdir(sub))
        self.assertEqual(PROTECTED_DIR_MODE, stat.S_IMODE(os.lstat(sub).st_mode),
                         "受保护格里的子目录也必须是只读的 (否则它是绕过写位的那扇门)")
        with self.assertRaises(CrossRoleWrite):
            self.layout.ensure_dir(ROLE_CANDIDATE, "measurements/gemm/h100")
        self.assertFalse(os.path.exists(
            os.path.join(self.layout.root, "measurements", "gemm", "h100")))

    def test_ensure_dir_keeps_a_writable_slot_writable(self):
        """回归: 在**可写格**里建子目录不许把它收成只读

        这不是洁癖: 候选/构建进程的 scratch 就在这一格里, 收成 0500 等于在一格「可写」
        的空间里造出一格不可写的空间 —— 与本节自己的声明自相矛盾。
        """
        sub = self.layout.ensure_dir(ROLE_CANDIDATE, "candidate/scratch/tmp")
        self.assertEqual(WRITABLE_DIR_MODE, stat.S_IMODE(os.lstat(sub).st_mode))
        probe = os.path.join(sub, "blob.bin")
        with open(probe, "wb") as fh:
            fh.write(b"scratch")
        self.assertTrue(os.path.exists(probe))
        # 受保护格同一操作仍然收 0500 (两层行为都要在)
        deep = self.layout.ensure_dir(ROLE_ORACLE, "measurements/a/b/c")
        self.assertEqual(PROTECTED_DIR_MODE, stat.S_IMODE(os.lstat(deep).st_mode))
        self.assertEqual(PROTECTED_DIR_MODE, self.layout.mode_of("measurements"))

    def test_an_undeclared_role_is_refused(self):
        for bad in ("", "oracle ", "Oracle", "root", None):
            with self.assertRaises((LayoutError, CrossRoleWrite)):
                self.layout.write_for(bad, "measurements/x.json", b"x")


class TestRawWritesFailAtTheOsLevel(unittest.TestCase):
    """③ 裸写: 不走唯一入口, 在同一 uid 下也失败 (这一层不靠任何调用方自觉)"""

    def setUp(self):
        self.layout = run_layout(_tmp_root())

    def test_creating_a_new_entry_in_a_protected_dir_fails(self):
        for name in PROTECTED_DIR_NAMES:
            p = os.path.join(self.layout.root, name, "new.txt")
            with self.assertRaises(PermissionError, msg=f"{name}/ 居然能裸写新文件"):
                with open(p, "w") as fh:
                    fh.write("x")

    def test_overwriting_an_existing_artifact_fails(self):
        """目录写位管「新建条目」, 文件写位管「覆盖内容」—— 两层都要有"""
        path = self.layout.write_for(ROLE_ORACLE, "measurements/l3.json", b"original")
        self.assertEqual(ARTIFACT_MODE, stat.S_IMODE(os.stat(path).st_mode))
        with self.assertRaises(PermissionError):
            with open(path, "w") as fh:
                fh.write("tampered")
        with open(path) as fh:
            self.assertEqual("original", fh.read())

    def test_the_writable_dirs_still_allow_new_files(self):
        """可写格是刻意可写的: 候选/构建进程有自己的 scratch (比如编译缓存)"""
        p = os.path.join(self.layout.root, "candidate", "scratch.txt")
        with open(p, "w") as fh:
            fh.write("scratch")
        self.assertTrue(os.path.exists(p))

    def test_even_the_writable_dir_refuses_to_rewrite_the_artifact(self):
        path = self.layout.write_for(ROLE_CANDIDATE, "candidate/code.py", b"a = 1")
        with self.assertRaises(PermissionError):
            with open(path, "w") as fh:
                fh.write("a = 2  # 绕过唯一入口")


class TestTheSanctionedPathWorks(unittest.TestCase):
    """唯一入口自己得是好的 —— 否则上面那些「拦住了」没有意义"""

    def setUp(self):
        self.layout = run_layout(_tmp_root())

    def test_write_read_roundtrip_and_modes(self):
        self.layout.ensure_dir(ROLE_ORACLE, "measurements/gemm/h20")
        path = self.layout.write_for(ROLE_ORACLE, "measurements/gemm/h20/l3.json",
                                     '{"speedup": 1.4}')
        with open(path) as fh:
            self.assertEqual({"speedup": 1.4}, __import__("json").load(fh))
        self.assertEqual(ARTIFACT_MODE, stat.S_IMODE(os.stat(path).st_mode))
        self.assertEqual(PROTECTED_DIR_MODE, self.layout.mode_of("measurements"),
                         "写完之后受保护格的权限位必须复原")

    def test_rewriting_goes_through_the_door_and_replaces(self):
        p1 = self.layout.write_for(ROLE_ORACLE, "measurements/l3.json", b"v1")
        p2 = self.layout.write_for(ROLE_ORACLE, "measurements/l3.json", b"v2")
        self.assertEqual(p1, p2)
        with open(p2, "rb") as fh:
            self.assertEqual(b"v2", fh.read())

    def test_the_write_window_is_restored_even_when_the_write_fails(self):
        """`finally` 复原那一句是**不变量**: 留一个 0700 的 measurements/ 比写失败更糟"""
        original = os.replace
        os.replace = lambda *a, **k: (_ for _ in ()).throw(OSError("模拟写失败"))
        try:
            with self.assertRaises(OSError):
                self.layout.write_for(ROLE_ORACLE, "measurements/l3.json", b"v3")
        finally:
            os.replace = original
        self.assertEqual(PROTECTED_DIR_MODE, self.layout.mode_of("measurements"))
        self.assertEqual([], os.listdir(os.path.join(self.layout.root, "measurements")),
                         "失败的写留下了残骸")

    def test_paths_cannot_escape_the_layout(self):
        for rel in ("../x.py", "/etc/passwd", "measurements/../../x",
                    "candidate/../../x", "notaslot/x.py", "", "   "):
            with self.assertRaises(LayoutError, msg=f"{rel!r} 没被拦住"):
                self.layout.path(rel)
        self.assertTrue(self.layout.path("candidate/code.py").startswith(
            self.layout.root + os.sep))

    def test_str_and_bytes_both_work(self):
        a = self.layout.write_for(ROLE_ORACLE, "measurements/a.json", "文本")
        b = self.layout.write_for(ROLE_ORACLE, "measurements/b.json", b"\x00\x01")
        self.assertEqual("文本".encode("utf-8"), open(a, "rb").read())
        self.assertEqual(b"\x00\x01", open(b, "rb").read())


class TestTheJudgementAndTheLayoutReadOneSource(unittest.TestCase):
    """④ 同源: §21.5 的判据与 §22.1 的目录名/路径不许各存一份"""

    def test_the_protected_names_are_the_same_object(self):
        self.assertIs(ar.PROTECTED_DIR_NAMES, rl.PROTECTED_DIR_NAMES,
                      "两份名单了 —— 判据与文件系统迟早给出不同答案")
        self.assertIs(ar.CANDIDATE_DIR, rl.CANDIDATE_DIR)

    def test_the_candidate_path_comes_from_the_layout_module(self):
        root = _tmp_root()
        layout = run_layout(root)
        self.assertEqual(layout.path("candidate/code.py"), ar.candidate_path(root))
        self.assertEqual(layout_path(root, "candidate/code.py"), ar.candidate_path(root))
        self.assertEqual(os.path.join(root, "candidate", "code.py"), ar.candidate_path(root))

    def test_the_role_path_form_survives(self):
        """没有 run_root 时仍是角色路径 (今天生产用的就是这一档), 而且**无副作用**"""
        self.assertEqual("candidate/code.py", ar.candidate_path())
        self.assertEqual("candidate/code.trl", ar.candidate_path(ext=".trl"))
        probe = _tmp_root()
        self.assertEqual(os.path.join(probe, "candidate", "code.py"),
                         ar.candidate_path(probe))
        self.assertFalse(os.path.exists(probe), "candidate_path 顺手建了目录 (它只该算字符串)")

    def test_the_absolute_path_keeps_its_root(self):
        """回归: 以前那份实现 `strip("/")` 会把绝对路径的根吃掉"""
        self.assertTrue(ar.candidate_path("/a/b").startswith("/a/b/"))

    def test_the_judge_still_fires_on_a_real_path(self):
        """接了真路径之后, §21.5 的判据仍然逐格命中 (判据一个字没改)"""
        root = run_layout(_tmp_root()).root
        ctx = ar.ActionContext(
            spend_committed=True,
            authorized_roots=(os.path.join(root, "candidate"),))
        for name in PROTECTED_DIR_NAMES:
            real = os.path.join(root, name, "x.py")
            out = ar._check_protected_paths(ar.Patch(path=real, new_source="x = 1"), ctx)
            self.assertEqual(ar.STATE_VIOLATION, out.state, f"{name} 在真路径下没被拦住")
        inside = ar._check_write_scope(
            ar.Patch(path=os.path.join(root, "candidate", "code.py"), new_source="x = 1"),
            ctx)
        self.assertEqual(ar.STATE_OK, inside.state)


class TestOnlyOneDoorInTheHarness(unittest.TestCase):
    """结构守卫: 打开写窗的能力（chmod / O_CREAT）在 `operator_agent/` 里只有一处"""

    def test_only_run_layout_opens_a_write_window(self):
        pkg = Path(__file__).resolve().parents[1] / "operator_agent"
        offenders = []
        for path in sorted(pkg.glob("*.py")):
            if path.name == "run_layout.py":
                continue  # 它**就是**那扇门（本用例要钉的是「只有它一扇」）
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fn = node.func
                name = fn.attr if isinstance(fn, ast.Attribute) else (
                    fn.id if isinstance(fn, ast.Name) else "")
                if name == "chmod":
                    offenders.append(f"{path.name}:{node.lineno} os.chmod")
                if name == "open":
                    src = ast.dump(node)
                    if "O_CREAT" in src:
                        offenders.append(f"{path.name}:{node.lineno} os.open(O_CREAT)")
        self.assertEqual([], offenders,
                         "这些地方自己开了写窗 —— §22.1 只有一个入口, 否则「越权写会失败」"
                         f"就不是不变量: {offenders}")

    def test_the_module_is_wired_into_the_conductor(self):
        """接线也是不变量的一部分: 模块写得再好, 没人调就是死词汇"""
        src = (Path(__file__).resolve().parents[1]
               / "operator_agent" / "conductor.py").read_text(encoding="utf-8")
        self.assertIn("from .run_layout import RunLayout, run_layout", src)
        self.assertIn("self.layout = run_layout(self.run_root) if self.run_root.strip() else None",
                      src, "布局的创建点变了 —— 复核「一次 execute 一次」与「空根不建」")
        self.assertIn("self.layout: Optional[RunLayout] = None", src)


class _Sentinel(Exception):
    pass


class TestTheLayoutIsBuiltOncePerExecute(unittest.TestCase):
    """`execute` 建布局的**时机**与**次数**: 写任何 artifact 之前, 且一次运行一次"""

    def _conductor(self, run_root):
        from operator_agent import Conductor
        from operator_agent.kernel_db import KernelDatabase
        return Conductor(run_root=run_root, kernel_db=KernelDatabase(jsonl_path=None))

    def test_no_run_root_means_no_layout(self):
        c = self._conductor("")
        self.assertIsNone(c.layout)  # 构造期不建
        self.assertEqual("candidate/code.py", ar.candidate_path(c.run_root))

    def test_execute_builds_it_before_the_inner_run(self):
        from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec,
                                            TargetSpec, TaskSchema, TaskType)
        root = _tmp_root()
        c = self._conductor(root)
        seen = {}

        def _inner(task, *a, **kw):
            seen["layout_at_inner"] = c.layout
            raise _Sentinel()

        c._execute_inner = _inner
        task = TaskSchema(task_type=TaskType.OPERATOR_GENERATION,
                          source=SourceSpec(kernel_type="matmul"),
                          target=TargetSpec(hardware="h20", dsl="triton"),
                          constraints=ConstraintSpec(precision=Precision.FP32,
                                                     min_speedup=1.0, max_error=1e-3))
        with self.assertRaises(_Sentinel):
            c.execute(task, {"A": 1}, None)
        self.assertIsInstance(seen["layout_at_inner"], RunLayout,
                              "布局是在内层跑起来之后才建的 —— 那时 artifact 已经在写了")
        for name in DIR_NAMES:
            self.assertTrue(os.path.isdir(os.path.join(root, name)))

    def test_the_patch_path_becomes_a_real_path_when_a_root_is_given(self):
        """§21 ↔ §22.1 的接缝: 同一个判据, 输入端从「声明」换成「真路径」"""
        root = _tmp_root()
        c = self._conductor(root)
        c.layout = run_layout(root)
        real = ar.candidate_path(c.run_root)
        self.assertTrue(real.startswith(root + os.sep), real)
        self.assertEqual(c.layout.path("candidate/code.py"), real)


if __name__ == "__main__":
    unittest.main()