"""代码指纹量具自己的标定 + 「驱动必须问过它」的结构守卫 (doc 50 §7.5 / #62)

外部依据 (doc 56, Hearsay `2609.32495v1`): 自述 (introspective) 与证据 (evidentiary) 的区别是
「一个不在场的读者能不能不采信系统的话就核一遍, **包括核『某件事没发生』**」, 而它的结论句是
"The remedy is another writer, not richer logging." —— 所以哈希由**子进程**算, 且失败返回 `None`
而不是兜成空 dict。

本文件干三件事, 顺序即重要性:
  A. **绝对标定**: 已知内容的文件就该量到已知哈希 (能口算出理论值的那种);
  B. **失效形态**: 被改 / 被删 / 从来没存在 / 子进程没跑起来 —— 四种都必须被 `diff` 说出来,
     尤其「取不到」不许被读成「没变」;
  C. **闭包不是手抄的**: 用一个小假包证明相对 import 真的被跟着走 (第一版没认相对 import,
     闭包停在 8 个文件上、`verifier.py` 压根不在名单里 —— 而这份名单的用途就是抓它);
     最后拿真驱动核: 上一轮那次不可归属的事故里被改的 5 个文件, 今天必须全在名单上。

运行: python3 -m pytest tests/unit/test_run_provenance.py -q
"""
import ast
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from benchmarks import provenance  # noqa: E402
from benchmarks.provenance import (  # noqa: E402
    closure, diff, local_imports, snapshot, snapshot_via_subprocess, stamp,
)

DRIVER = "benchmarks/verify_resume_equivalence.py"


class TestFingerprintsAreAbsolute(unittest.TestCase):
    """A 组: 已知内容 ⇒ 已知哈希, 而且四种失效都得被说出来"""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="prov-")
        self.addCleanup(__import__("shutil").rmtree, self.dir)
        with open(os.path.join(self.dir, "a.py"), "wb") as fh:
            fh.write(b"x = 1\n")
        with open(os.path.join(self.dir, "b.py"), "wb") as fh:
            fh.write(b"y = 2\n")

    def _write(self, name, data):
        with open(os.path.join(self.dir, name), "wb") as fh:
            fh.write(data)

    def test_a_known_file_hashes_to_the_known_digest(self):
        """绝对锚点: 不是「它算出了个十六进制」, 而是它算出的**就是**那 16 位"""
        snap = snapshot(["a.py", "b.py"], self.dir)
        self.assertEqual(2, len(snap))
        self.assertEqual(hashlib.sha256(b"x = 1\n").hexdigest()[:16], snap["a.py"])
        self.assertEqual(hashlib.sha256(b"y = 2\n").hexdigest()[:16], snap["b.py"])

    def test_changing_one_file_names_exactly_that_file(self):
        before = snapshot(["a.py", "b.py"], self.dir)
        self._write("a.py", b"x = 999\n")
        after = snapshot(["a.py", "b.py"], self.dir)
        self.assertEqual(["被改 a.py"], diff(before, after))

    def test_deleting_a_listed_file_is_not_read_as_unchanged(self):
        before = snapshot(["a.py", "b.py"], self.dir)
        os.remove(os.path.join(self.dir, "a.py"))
        after = snapshot(["a.py", "b.py"], self.dir)
        self.assertEqual(provenance._ABSENT, after["a.py"], "不在盘上必须显式登记")
        self.assertEqual(["跑动期间被删 a.py"], diff(before, after))

    def test_a_file_that_never_existed_is_visible_not_silently_absent(self):
        """漏登记的依赖自己会红: 名单里写了它、盘上没有 ⇒ 快照里带 ABSENT, 而不是少一行"""
        snap = snapshot(["a.py", "ghost.py"], self.dir)
        self.assertEqual(provenance._ABSENT, snap["ghost.py"])
        self.assertEqual(2, len(snap))

    def test_no_snapshot_is_never_read_as_no_change(self):
        """`None` (子进程没跑起来) 与 `{}` (什么都没变) 必须是两个不同的结论"""
        self.assertTrue(diff(None, None))
        self.assertEqual([], diff({"a.py": "zz"}, {"a.py": "zz"}))

    def test_the_subprocess_writer_agrees_with_the_in_process_one(self):
        """另一个写者不是「更贵的同一个写者」: 两边读同一盘必须给出同一份指纹"""
        via_child = snapshot_via_subprocess(["a.py", "b.py"], self.dir)
        self.assertIsNotNone(via_child)
        self.assertEqual(snapshot(["a.py", "b.py"], self.dir), via_child)

    def test_the_subprocess_returns_none_rather_than_an_empty_dict_when_it_cannot_run(self):
        bad = snapshot_via_subprocess(["a.py"], os.path.join(self.dir, "no-such-root"))
        self.assertIsNone(bad)

    def test_a_missing_root_raises_instead_of_reporting_everything_absent(self):
        """根目录本身不存在时不许「成功地」算出一张全 ABSENT 的表 —— 那是空转的起点"""
        with self.assertRaises(FileNotFoundError):
            snapshot(["a.py"], os.path.join(self.dir, "no-such-root"))

    def test_an_all_absent_manifest_does_not_open_the_gate(self):
        """两份全 ABSENT 的清单逐元素相等, 但**证明不了任何事** ⇒ 门必须关

        这条是写上一轮那条测试时撞出来的真洞: 传错 root 时子进程照样退 0, 于是驱动拿到
        「什么都没变」的结论, 而实际上它从没看过盘。
        """
        wrong = tempfile.mkdtemp(prefix="prov-wrong-")
        self.addCleanup(__import__("shutil").rmtree, wrong)
        b = snapshot(["a.py", "b.py"], wrong)
        a = snapshot(["a.py", "b.py"], wrong)
        self.assertEqual(provenance._ABSENT, b["a.py"])
        self.assertEqual(b, a, "夹具本身要真的是「逐元素相等」")
        self.assertTrue(diff(b, a), "全 ABSENT 必须让门失败, 而不是报「没变」")

    def test_stamp_names_the_interpreter_and_the_commit(self):
        line = stamp(_ROOT)
        for key in ("python=", "torch=", "HEAD=", "未提交改动="):
            self.assertIn(key, line, line)
        self.assertIn(sys.executable, line)
        self.assertNotIn("HEAD=?", line, "取不到 commit 不许兜成 '?' 印出去 (I21/I24)")


class TestClosureIsNotHandCopied(unittest.TestCase):
    """C 组: 名单是算出来的 —— 手抄的名单从抄下那刻就开始漂, 而且总是漂在新加的那个依赖上"""

    MINI = {
        "entry.py": "import pkg.a\n",
        "pkg/__init__.py": "",
        "pkg/a.py": "from .b import B\n\n\nclass A:\n    pass\n",
        "pkg/b.py": "class B:\n    pass\n",
        "pkg/unused.py": "class Unused:\n    pass\n",
    }

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="prov-pkg-")
        self.addCleanup(__import__("shutil").rmtree, self.dir)
        os.makedirs(os.path.join(self.dir, "pkg"))
        for name, text in self.MINI.items():
            with open(os.path.join(self.dir, name), "w", encoding="utf-8") as fh:
                fh.write(text)

    def _closure(self):
        # 假包要过 LOCAL_PACKAGES 这道门, 否则解析出的名字会被丢掉
        real = provenance.LOCAL_PACKAGES
        provenance.LOCAL_PACKAGES = tuple(list(real) + ["pkg"])
        self.addCleanup(setattr, provenance, "LOCAL_PACKAGES", real)
        return closure("entry.py", self.dir)

    def test_it_follows_relative_imports_one_hop(self):
        """反空转: 这条是整组的存在理由 —— 第一版认不出 `from .b import B`, 结果全绿是假的"""
        found = self._closure()
        self.assertIn("pkg/b.py", found, found)
        self.assertIn("pkg/__init__.py", found, "包的 __init__ 也是跑过的代码")
        self.assertGreaterEqual(len(found), 3, found)
        self.assertNotIn("pkg/unused.py", found)

    def test_a_class_name_does_not_become_a_phantom_file(self):
        """`from .b import B` 里的 `B` 是属性不是模块。

        第一版无条件展开 names, 真仓库上就多出 `operator_agent/Verifier.py` —— 而 macOS 的盘
        大小写不敏感, `exists()` 为真, 于是**同一个文件被登记两次**, 「多少个文件」这个数不可信。
        """
        found = self._closure()
        self.assertNotIn("pkg/B.py", found, found)
        self.assertNotIn("pkg/a.B.py", found, found)

    def test_docstring_text_is_not_treated_as_an_import(self):
        with open(os.path.join(self.dir, "pkg", "a.py"), "w", encoding="utf-8") as fh:
            fh.write('"""这里写着 from .unused import Unused 只是为了说明"""\n'
                     'class A:\n    pass\n')
        found = self._closure()
        self.assertNotIn("pkg/unused.py", found, found)


class TestRealDriverIsCovered(unittest.TestCase):
    """C 组 (真仓库那半): 上一轮不可归属的事故, 今天必须被这份名单抓住"""

    # doc 50 §7.5: 那次跑动窗口内被并发会话改掉的就是这五个文件
    INCIDENT = ["operator_agent/schemas.py", "operator_agent/remote_executor.py",
                "operator_agent/verifier.py", "operator_agent/conductor.py",
                "remote_scripts/run_kernel.py"]

    def test_the_five_files_that_broke_the_last_run_are_all_listed(self):
        found = closure(DRIVER, _ROOT,
                        extra_entries=["remote_scripts/run_kernel.py"])
        missing = [f for f in self.INCIDENT if f not in found]
        self.assertEqual([], missing, f"闭包漏了: {missing}")

    def test_closure_entries_all_exist(self):
        found = closure(DRIVER, _ROOT)
        ghost = [f for f in found if not os.path.exists(os.path.join(_ROOT, f))]
        self.assertEqual([], ghost, "名单里不许有盘上不存在的条目 (计数要可信)")
        self.assertGreaterEqual(len(found), 20, found)

    def test_local_imports_on_a_real_module_finds_relative_names(self):
        mods = local_imports(Path(_ROOT) / "operator_agent" / "conductor.py", _ROOT)
        self.assertTrue(any(m.endswith("verifier") for m in mods), mods[:12])

    def test_driver_asks_twice_and_gates_its_conclusion_on_the_answer(self):
        """#62 的自检不是加一行 print: 取两次快照, 且结论行**在门后面**

        机器可读的形状: `report(...)` 的结果存进一个名字; 有一个 `if <那个名字>` 里面带着
        `return`; 而打印「汇总」的那个 print 的 `lineno` 必须**大于**那道门的 `lineno`。
        只数「main 里存在某个 `if`」是不够的 —— 门写在汇总后面就什么都没挡住, 而随便一个 `if na:`
        也能让那种断言绿; 所以这条判据要的是**顺序**关系, 不是存在关系。
        """
        src = Path(_ROOT, DRIVER).read_text(encoding="utf-8")
        self.assertGreaterEqual(src.count("snapshot_via_subprocess("), 2,
                                src.count("snapshot_via_subprocess("))
        tree = ast.parse(src)
        main = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "main")

        gated_names = set()
        for node in main.body:                      # 只认赋值给名字的 report(...)
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                f = node.value.func
                if getattr(f, "id", None) == "report" or getattr(f, "attr", None) == "report":
                    gated_names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        self.assertTrue(gated_names, "main() 里没有把 report() 的结果存进名字 ⇒ 没法门")

        gates = []
        for node in ast.walk(main):
            if isinstance(node, ast.If) and any(
                    isinstance(t, ast.Name) and t.id in gated_names
                    for t in ast.walk(node.test)):
                if any(isinstance(r, ast.Return) for r in ast.walk(node)):
                    gates.append(node.lineno)
        self.assertTrue(gates, "没有一道 `if` 是按归属结果提前 return 的")

        summary_lines = []
        for node in ast.walk(main):
            if (isinstance(node, ast.Call)
                    and (getattr(node.func, "id", "") == "print"
                         or getattr(node.func, "attr", "") == "print")
                    and node.args and isinstance(node.args[0], ast.JoinedStr)):
                text = "".join(v.value for v in node.args[0].values if isinstance(v, ast.Constant))
                if "汇总" in text:
                    summary_lines.append(node.lineno)
        self.assertTrue(summary_lines, "main() 里找不到那行汇总 print")
        self.assertTrue(all(min(gates) < ln for ln in summary_lines),
                        f"门在 {min(gates)} 行, 而汇总 print 在 {summary_lines} —— 汇总必须全在门后面")


if __name__ == "__main__":
    unittest.main(verbosity=2)
