"""`benchmarks/experiments/misc/arxiv_html_to_lines.py` 的读者 —— 调研文档里的行号必须能重建

为什么要有这个文件（不是「再补一组用例」）：抽取器自带 `--selftest`，但**全套测试里没有谁调用它**，
所以那条 selftest 就是本仓一直反对的形状 ——「算了没人读」。doc 58 每一条「行 N」都声称由这个脚本
产生；脚本坏掉时那篇文章不会变红，只有下一个想复核行号的人发现自己对不上。

判据按本仓那条老规矩成对写（CHECKLOG I19：只在条件成立的那一侧取样，测的是代码在不在，
不是条件对不对）：
  · NBSP 保留 **且** ASCII 空白折叠 —— 只测前一条的话，「什么都不压」也能绿；
  · 字面 `&lt;p&gt;` 存活 **且** 真 `<p>` 不出现 —— 只测后一条的话，「不删任何标签」也能绿；
  · `<script>` 里的词消失 **且** 同一个词写在正文里必须存活 —— 否则「把整段都扔了」也能绿。
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.experiments.misc.arxiv_html_to_lines import selftest, to_lines, write_numbered  # noqa: E402


class TestExtractorIsNotANoOp(unittest.TestCase):
    def test_the_bundled_selftest_has_a_reader_in_the_suite(self) -> None:
        # 这条就是「selftest 不再没人读」本身：返回 0 表示它内部那 6 项都过了
        self.assertEqual(selftest(), 0)

    def test_nb_sp_survives_and_ascii_whitespace_collapses(self) -> None:
        lines = to_lines("<p>2.1.\u00a0Method</p><p>a\t  b   c</p>")
        self.assertEqual(["2.1.\u00a0Method", "a b c"], lines)
        self.assertNotIn("\t", lines[1])

    def test_literal_markup_in_prose_survives_while_real_tags_do_not(self) -> None:
        # 两侧都钉死成整行的绝对值: 「把 &lt;p&gt; 连同真标签一起删掉」和「什么都不删」
        # 各自只能活过一半
        lines = to_lines("<p>quoted: &lt;p&gt;x&lt;/p&gt;</p><p>plain</p>")
        self.assertEqual(["quoted: <p>x</p>", "plain"], lines)


class TestProvenanceOfLineNumbers(unittest.TestCase):
    def test_script_body_is_gone_but_the_same_word_in_prose_survives(self) -> None:
        # 成对的那一侧：只断言 "hidden" 消失的话, 「把整个输入扔了」也绿
        gone = to_lines("<p>shown</p><script>var s='hidden';</script>")
        kept = to_lines("<p>shown and hidden together</p>")
        self.assertEqual(["shown"], gone)
        self.assertEqual(["shown and hidden together"], kept)

    def test_written_line_numbers_are_one_based_and_contiguous(self) -> None:
        lines = to_lines("<p>alpha</p><p>beta</p><p>gamma</p>")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "out.txt"
            write_numbered(lines, str(path))
            rows = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(["1\talpha", "2\tbeta", "3\tgamma"], rows)
        self.assertEqual(len(lines), len(rows))

    def test_a_newline_in_the_source_is_also_a_line_break(self) -> None:
        # 行号由**两种**东西决定：块级标签 *和* 源文件里本来就有的换行。
        # 所以「3,470 行」只在同一批字节上可重建 —— 这就是为什么报表把字节数和行数一起印
        # （2026-09-30: 这一句写完的第二天真被重跑了, KernelOPT 那两个数对不上, 见 doc 58 §1.1）
        self.assertEqual(["one", "two"], to_lines("<p>one\ntwo</p>"))


if __name__ == "__main__":
    unittest.main()
