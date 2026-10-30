"""变异标定 harness 自己的标定 —— 一台量具在报「N/N 拦住」之前, 要先证明它能看见逃逸

外部一手 (doc 57 §二 第 1 条, arXiv:2609.22220v1 行 246-251):
    "Scoring suites against unkillable rows deflates every protocol equally and informs
     about none. We therefore admit a mutant into the denominator only with a kill witness."
这条用到我方那张 21 行表上, 分出来的正是两栏: 18 行有行为证人, 3 行 (M11/M16/M21) 今天在
行为层与原文不可区分, 拦住它们的是 AST / 契约那类结构守卫。混在一个「21/21」里, 那句话就把
两种不同的覆盖说成了一种。

本文件干两件事:
  A. **拿假仓库标定 harness** (tmpdir 里一份 mini 源码 + mini 测试): 种一条真逃逸、种一个
     锚点不唯一的行、把 pytest 引到没有用例的目标上 —— 三种失效形态都必须被它自己说出来,
     而不是被读成「拦住」。这是 I15 (「变异脚本一个用例都没跑, 报表却印成 8 条全逃逸」) 与
     I11 (归属串味道) 的正面防御。
  B. **核真规格不腐烂**: 21 行的锚点今天在各自文件里必须仍然恰好出现一次, 目标测试文件必须
     还在, 结构行必须写明隔离理由 —— 否则「可复现」这句话又是空口 (本会话刚为它推翻过
     doc 41 的「13/13」)。

运行: python3 -m pytest tests/unit/test_mutation_calibration.py -q
"""
import os
import re
import shutil
import sys
import tempfile
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from benchmarks.mutation_calibrate import (  # noqa: E402
    Row, calibrate, check_shape, load_spec, main, summarize,
)

SPEC = os.path.join(_ROOT, "benchmarks", "mut_specs", "fixer_stage_branch.py")

# ── A 组用的假仓库 ──────────────────────────────────────────────────────────────
_PROD = """
def pick(x):
    return "a" if x else "b"
"""
_TEST_COVERED = """
import unittest
from prod import pick


class T(unittest.TestCase):
    def test_pick_both_sides(self):
        self.assertEqual("a", pick(True))
        self.assertEqual("b", pick(False))
"""
# 这份测试**只测常量**, 所以「把 pick 改成恒返 a」是真逃逸 —— 用来证明 harness 会报 ESCAPED
_TEST_BLIND = """
import unittest


class T(unittest.TestCase):
    def test_constant(self):
        self.assertEqual(1, 1)
"""


def _mini_repo(test_src: str, prod_src: str = _PROD) -> str:
    root = tempfile.mkdtemp(prefix="mutcal-")
    with open(os.path.join(root, "prod.py"), "w", encoding="utf-8") as fh:
        fh.write(prod_src)
    with open(os.path.join(root, "test_mini.py"), "w", encoding="utf-8") as fh:
        fh.write(test_src)
    return root


class TestHarnessSeesItsOwnFailures(unittest.TestCase):
    """每种「报表会说谎」的形态都要被 harness 自己拦住, 而不是被我读成成功"""

    def test_a_planted_escape_is_reported_as_escape(self):
        """对照: 用例照跑、全绿, 而变异真的没被任何用例看见 ⇒ 必须是 ESCAPED

        这条是整个 harness 的存在理由。若它反过来报 KILLED, 那「N/N 拦住」这个数字就没读过。
        """
        root = _mini_repo(_TEST_BLIND)
        try:
            rows = [Row("E1 pick 恒返 a", "prod", '    return "a" if x else "b"',
                        '    return "a"', "x=False 的输入应返回 b")]
            res = calibrate(root, {"prod": "prod.py"}, ["test_mini.py"], rows)
            self.assertEqual([("ESCAPED", True)],
                             [(r.verdict, r.behavioral) for r in res])
            self.assertIn("0/1 拦住", summarize(res))
        finally:
            shutil.rmtree(root)

    def test_the_same_mutation_is_killed_when_a_test_actually_covers_it(self):
        """反空转: 同一行变异在**有覆盖**的那份测试上必须变红 —— 否则上面那条 ESCAPED
        可能只是「测试压根没跑」的另一种表现

        ⚠️ 这条对照第一次是**红的**，原因在夹具而不是产品：`_TEST_COVERED` 原先只调
        `pick(True)`，于是「恒返 a」这个变异在它面前行为完全一致 ⇒ 「有覆盖的对照组」自己
        只覆盖了条件的一侧 —— 正是本仓记过的那条「只在阈值一侧取样，删掉整个条件也看不见」，
        这次落在我写的那份对照上。现在两份都测。
        """
        root = _mini_repo(_TEST_COVERED)
        try:
            rows = [Row("E1 pick 恒返 a", "prod", '    return "a" if x else "b"',
                        '    return "a"', "x=False 的输入应返回 b")]
            res = calibrate(root, {"prod": "prod.py"}, ["test_mini.py"], rows)
            self.assertEqual(["KILLED"], [r.verdict for r in res])
            self.assertEqual(1, len(res[0].fails))
        finally:
            shutil.rmtree(root)

    def test_no_summary_line_is_refused_not_read_as_evidence(self):
        """I15: 目标里没有用例时 pytest 不报 passed/failed, 那种「绿」必须直接拒绝判定"""
        root = _mini_repo(_TEST_COVERED)
        try:
            with self.assertRaises(RuntimeError) as ctx:
                calibrate(root, {"prod": "prod.py"}, ["test_nomhere.py"],
                          [Row("X", "prod", "x", "y", "w")])
            self.assertIn("汇总行", str(ctx.exception))
        finally:
            shutil.rmtree(root)

    def test_anchor_not_unique_is_anchor_not_kill(self):
        """锚点出现 0 或 2 次 ⇒ 这行**根本没跑**, 不许被算进分母的任何一栏"""
        dup = _PROD + '\ndef pick2(x):\n    return "a" if x else "b"\n'
        root = _mini_repo(_TEST_COVERED, dup)
        try:
            files = {"prod": "prod.py"}
            rows = [Row("A1 锚点重复", "prod", '    return "a" if x else "b"',
                        '    return "a"', "w")]
            problems = check_shape(root, files, rows)
            self.assertEqual(1, len(problems), problems)
            self.assertIn("出现 2 次", problems[0])
            res = calibrate(root, files, ["test_mini.py"], rows)
            self.assertEqual(["ANCHOR"], [r.verdict for r in res])
            self.assertIn("A1 锚点重复", summarize(res))
        finally:
            shutil.rmtree(root)

    def test_files_are_restored_byte_identically_even_after_a_red_row(self):
        """I11: 逐轮复原。跑完红名单归属才只对应那一行, 盘上也不留变异"""
        root = _mini_repo(_TEST_COVERED)
        before = {f: open(os.path.join(root, f), encoding="utf-8").read()
                  for f in ("prod.py", "test_mini.py")}
        try:
            rows = [Row("K1", "prod", '    return "a" if x else "b"', '    return "a"', "w"),
                    Row("K2", "prod", '    return "a" if x else "b"', '    return "b"', "w")]
            calibrate(root, {"prod": "prod.py"}, ["test_mini.py"], rows)
            after = {f: open(os.path.join(root, f), encoding="utf-8").read()
                     for f in before}
            self.assertEqual(before, after)
        finally:
            shutil.rmtree(root)

    def test_the_two_denominators_are_always_both_printed(self):
        """0 条结构行也要显式印出来 —— 否则有人删掉整类结构行时, 报表看起来更像「全守住」"""
        self.assertIn("行为分母", summarize([]))
        self.assertIn("结构分母", summarize([]))

    def test_main_exits_nonzero_on_an_escape(self):
        """退出码是判据的**出口**: 有逃逸却退 0, 前面印的 ❌ 会被 CI 读成通过"""
        root = _mini_repo(_TEST_BLIND)
        spec = os.path.join(root, "spec.py")
        with open(spec, "w", encoding="utf-8") as fh:
            fh.write("FILES = {'prod': 'prod.py'}\nTARGETS = ['test_mini.py']\n"
                     "import sys; sys.path.insert(0, %r)\n"
                     "from benchmarks.mutation_calibrate import Row\n"
                     "ROWS = [Row('E1', 'prod', %r, %r, 'x=False 时应返 b')]\n"
                     % (_ROOT, '    return "a" if x else "b"', '    return "a"'))
        try:
            code = main(["--spec", spec, "--root", root])
            self.assertEqual(1, code, "逃逸必须让退出码非 0")
        finally:
            shutil.rmtree(root)


class TestRealSpecDoesNotRot(unittest.TestCase):
    """B 组: 真那张 21 行的表 —— 「可复现」要有机械守卫, 不能靠下次有人重新数一遍"""

    @classmethod
    def setUpClass(cls):
        cls.files, cls.targets, cls.rows = load_spec(SPEC)

    def test_every_anchor_is_still_unique_in_the_current_tree(self):
        problems = check_shape(_ROOT, self.files, self.rows)
        self.assertEqual([], problems,
                         "锚点漂了 ⇒ 本文件那两句计数要重跑重数, 不许沿用旧数字")

    def test_target_files_exist(self):
        missing = [t for t in self.targets if not os.path.exists(os.path.join(_ROOT, t))]
        self.assertEqual([], missing, "规格里指着已经不存在的测试文件")

    def test_both_witness_classes_are_present(self):
        """两栏都得有人: 只有行为行 ⇒ 「分开计数」这句话本身是空的"""
        beh = [r for r in self.rows if r.behavioral]
        stru = [r for r in self.rows if not r.behavioral]
        self.assertGreaterEqual(len(beh), 10)
        self.assertGreaterEqual(len(stru), 1)

    def test_a_structural_row_must_state_why_it_is_quarantined(self):
        """结构行的 witness 不是标签而是**理由** —— 没有理由的隔离就是偷偷把它请出分母"""
        for r in self.rows:
            if not r.behavioral:
                self.assertTrue(r.witness.startswith("隔离理由"),
                                f"{r.name}: 结构行没写隔离理由")

    def test_row_names_match_the_table_in_the_test_docstring(self):
        """表在两个地方各一份: 规格 (可执行) 与测试文件 docstring (人读的) —— 差集要为空

        只核一侧时另一侧改名只有专门这条契约守卫能看见 (本仓记过: 装配契约两侧都要查)。
        """
        doc = open(os.path.join(_ROOT, "tests", "unit", "test_fixer_stage_branch.py"),
                   encoding="utf-8").read()
        spec_ids = {r.name.split()[0] for r in self.rows}
        in_doc = set(re.findall(r"\bM\d+[a-z]?\b", doc))
        self.assertTrue(spec_ids, "规格里一行都没有")
        self.assertEqual(set(), spec_ids - in_doc,
                         "规格里有、而 docstring 那张表里点不到的行 (改名或漏记)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
