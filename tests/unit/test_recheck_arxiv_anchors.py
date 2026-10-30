"""`benchmarks/recheck_arxiv_anchors.py` 的读者 —— 「出处可重建」这句话自己得有人重跑

为什么要有这个文件：上一片把抽取器进仓库，理由是「行号不在仓里就不可重建」；可**那张锚点表
自己也没被重建过**。今天照表重跑，四篇里 `2609.30059v1` 的字节数与行数两个数都是错的
（表的 1,058,127 / 3,344 vs 实际 1,087,672 / 3,470），而引文里我自己凭印象拼的一个 needle
也红了。这两件事都是**跑一次**才发现的 —— 所以这个脚本要走网络，而它自带 `--selftest`，
两者都需要读者（CHECKLOG I28）。

这套用例不打网络（真跑那条是 `python3 -B benchmarks/recheck_arxiv_anchors.py`），
它钉的是这台机器上就能判的三件事：
  · **解析不许空转** —— 解析出空表时后面每一条相等断言都会永真地绿 (I25 那一族)，所以既测
    「合成表解析对」也测「真实 doc 58 解析出 4 行、且散文里提到的 id 不算表行」；
  · **行号区间必须真的在约束** —— 成立那一侧（片段确实在区间内）与缺席那一侧（片段在区间外
    /行号越界）各一条。只测前者的话「把整篇拼起来找」也能绿，而那样等于什么都没核；
  · **取不到字节不许算通过** —— 与 #53/#64 同族（缺证据当否定证据的反面：当肯定证据）。

第四件是 2026-09-30 读 doc 60 时撞出来的，它把上面第三条拆成了两格：
  · **带正文的 404 是响应，没正文才是取不到**；**`--doc` 只载入一份表时，全局引文清单要按表筛**。
    前一条的意义不只是「能跑」：Vestrum `2609.33822v1` 那两个主机都给 404，而 `export.arxiv.org`
    （脚本走的那条）只有 126 B / 1 行 `404`，`arxiv.org` 才给 7,901 B / 31 行那句
    `No HTML for '…'` ⇒ 表与 needle 都必须按**脚本能重取的那一份**记（doc 60 §七）。
    后一条的实跑形态是 **32/65** —— 31 条别的文档的引文被当成「锚点对不上」，而模块 docstring
    的用法行本身就在推荐 `--doc 只核一份`。⇒ 新增 5 例（23 → 28）与另一张表
    `benchmarks/mut_specs/anchor_fetch_is_a_response.py`（6 行，**6 杀 / 0 逃逸**）。
    其中 R6 第一次是**假绿**：我断言 `In("doc.md")`，而变异印出的是 `['/tmp/…/doc.md']` ——
    列 repr 里也含那个子串 ⇒ 逃逸；改成同时断言「没有 `None`、没有 `['`」才杀得掉（I32 那一族）。

变异标定 (每条只改一处 `benchmarks/recheck_arxiv_anchors.py`，跑完立刻逐字节复原并断言 sha256
一致。2026-09-30 实跑，基线 22 passed，**6 杀 / 1 逃逸**):
    M1 区间不约束（整篇拼起来找）              → 3 条红  ← 这条判据的全部价值就在这
    M2 拆掉 `start < 1` 那道前置守卫           → 1 条红
    M3 同一 id 改成首行赢（`setdefault`）      → 1 条红（selftest 里的重复 id 那条）
    M4 千分位不剥                              → 4 条红
    M5 没取到字节算通过                        → 2 条红
    M6 只核字节、不核行数                      → 1 条红（`test_wrong_line_count_alone_is_flagged`）
    M7 命中判据退回真值判断 (`got is not None` → `got`)  → **全绿，逃逸**
M7 的逃逸是**结构性的、不是用例写松了**：`citation_in_range` 只在命中时返回 `start`，而
`start == 0` 恰好是 M2 那道守卫挡掉的输入 ⇒ 单独改 M7 在任何输入下都不可观察，只有
「守卫被拆 **且** 用真值判断」两件事同时发生才会出错。这两处是**一组**，登记在这里而不是
给 M7 编一个证人（I23 那条「两种覆盖分开计数」的又一例：一条变异杀不掉时，先问它是没被
测到，还是它本来就不可观察 —— 后者的正确动作是把它写进注释，不是把用例改成假的绿）。

⚠️ **上面那张表是 2026-09-30 在「只有一份表」的世界里标的**。同日读 Φ-Bench 开出 doc 59 之后，
`main()` 改成按 `DEFAULT_DOCS` 并读多份（doc 59 §一 记了那次假红：指到 doc 59 跑报 13/32，
19 条 ❌ 全是 doc 58 的引文），这套里随之加了两条守卫：孤儿判据改用 `main()` 真读的那组文件
（`test_every_cited_id_appears_in_a_table_that_main_actually_reads`），以及 `DEFAULT_DOCS` 里
每个模式必须仍匹配到文件（`test_every_default_doc_pattern_still_matches_a_file`）。
**这两条各自的变异没有重跑** —— 不是它们不可杀，是这次没标；M1~M7 那七条也不覆盖新的多文档
路径（它们改的是单份表内的判据）。下一个人若要动 `DEFAULT_DOCS` 或 `main()` 的读表逻辑，
先补这两条的变异标定再动。

⇒ **这句欠账同日已清**（读 doc 60 时我就是那个「下一个人」）：`main()` 的读表/取数侧改了一处
（带正文的 404 进比对）与两处新增（引文按表筛、全量跑的孤儿引文拉退出码 2），对应的标定在
`benchmarks/mut_specs/anchor_fetch_is_a_response.py`（6 行 6 杀，其中 R3/R4/R5 三条打的正是
「多文档路径」，R5 是 R3/R4 那句「筛掉了 N 条」的读者 —— 沉默的筛选与没有筛选不可区分）。
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.recheck_arxiv_anchors import (CITATIONS, DEFAULT_DOCS, audit,  # noqa: E402
                                              citation_in_range, closing_advice,
                                              main, parse_anchor_table, to_int)

REPO = Path(__file__).resolve().parents[2]
DOC58 = REPO / "docs/research/58-缓存语义与墙钟上界等新吸收.md"

BODY = b"<!DOCTYPE html><body><p>one</p><p>two</p><p>three</p></body>"


def _read_default_tables() -> dict:
    """把 `DEFAULT_DOCS` 指到的那些表并起来解析 —— 与 `main()` 读同一组文件"""
    texts = []
    for pattern in DEFAULT_DOCS:
        for path in sorted(REPO.glob(pattern)):
            texts.append(path.read_text(encoding="utf-8"))
    return parse_anchor_table("\n".join(texts))


def _table_row(arxiv_id: str, body: bytes = BODY) -> str:
    """按真实值造一行表 —— 「ok」那一侧要有样本，不能只测不匹配"""
    from benchmarks.arxiv_html_to_lines import to_lines
    n = len(to_lines(body.decode()))
    return f"| x | `{arxiv_id}` | {len(body):,} | {n} |"


class TestParseIsNotEmpty(unittest.TestCase):
    """解析空转 = 全绿。这一族本仓撞过不止一次"""

    def test_the_real_doc_58_table_parses_four_papers(self):
        table = parse_anchor_table(DOC58.read_text(encoding="utf-8"))
        self.assertEqual({"2609.11987v1", "2609.30059v1", "2609.33074v1", "2609.35140v1"},
                         set(table),
                         "表形状变了 ⇒ 那条重跑检查已经在空转（一条都没解析出来时它永真）")
        self.assertTrue(all(b > 0 and l > 0 for b, l in table.values()),
                        f"字节数/行数必须是正整数: {table}")

    def test_prose_mentioning_an_id_is_not_a_table_row(self):
        # 缺席的那一侧: README 那种「一句话里提到 `2609.30059v1`」不能被当表行读进来
        md = "| a | `1111.2222v1` | 10 | 2 |\n| 散文里提到 `5555.6666v1` 不是表行 |\n"
        self.assertEqual({"1111.2222v1"}, set(parse_anchor_table(md)))

    def test_thousands_separators_are_stripped_both_columns(self):
        table = parse_anchor_table("| a | `1111.2222v1` | 1,087,672 | 3,470 |")
        self.assertEqual((1087672, 3470), table["1111.2222v1"])
        self.assertEqual(9999, to_int(" 9,999 "))


class TestRangeIsTheWholePoint(unittest.TestCase):
    """区间判据: 它的价值全部在「区间外的不许命中」"""

    LINES = ["alpha", "beta gamma", "delta"]

    def test_needle_inside_the_declared_range_hits(self):
        self.assertEqual(2, citation_in_range(self.LINES, 2, 2, "beta"))
        self.assertEqual(2, citation_in_range(self.LINES, 2, 3, "delta"),
                         "跨行的引文要按区间拼起来找 (行 N-M 里的 M 常常是被块级换行劈开的)")

    def test_needle_outside_the_declared_range_does_not_hit(self):
        self.assertIsNone(citation_in_range(self.LINES, 2, 2, "alpha"),
                          "片段在行 1 而声明的是行 2 却判成命中 ⇒ 区间没在约束任何东西")
        self.assertIsNone(citation_in_range(self.LINES, 1, 2, "delta"),
                          "区间被写坏了 (end < 实际行) 也不许把窗外的东西算进来")

    def test_out_of_bounds_line_numbers_do_not_hit(self):
        self.assertIsNone(citation_in_range(self.LINES, 9, 9, "alpha"))
        self.assertIsNone(citation_in_range(self.LINES, 3, 2, "alpha"))

    def test_line_zero_is_not_a_line_even_if_the_text_matches(self):
        """`start=0` 会让切片变成 `lines[-1:…]` —— 倒数第一行, 一个「存在」的行号

        没有那个前置守卫时这里会**命中** (返回 0)。而 0 在真值判断下等于「没命中」,
        所以两种错会互相掩盖 —— 上面那条守卫与 `audit` 里的 `is not None` 是配对的。
        """
        self.assertIsNone(citation_in_range(self.LINES, 0, 3, "delta"))

    def test_empty_extraction_cannot_produce_a_citation(self):
        # 「抽取空转」与「引文不存在」在这里必须同形: 都判不命中, 而不是永真
        self.assertIsNone(citation_in_range([], 1, 1, "anything"))


class TestAuditVerdicts(unittest.TestCase):
    """逐条判定的形状: ok / anchor_mismatch / no_bytes / cited / citation_absent"""

    def test_matching_anchors_are_ok_and_citations_are_cited(self):
        md = _table_row("1111.2222v1")
        rows = audit(md, {"1111.2222v1": BODY}, [("1111.2222v1", 2, 2, "two")])
        self.assertEqual("ok", rows[0]["status"])
        self.assertEqual("cited", rows[1]["status"])

    def test_wrong_numbers_in_the_table_are_flagged(self):
        # 这一条就是今天真跑出来的那个形状 (表里的字节数与实际不符)
        rows = audit("| x | `1111.2222v1` | 99 | 99 |", {"1111.2222v1": BODY}, [])
        self.assertEqual("anchor_mismatch", rows[0]["status"])
        self.assertEqual(len(BODY), rows[0]["got_bytes"])

    def test_wrong_line_count_alone_is_flagged(self):
        """两侧分开取样的理由: 只核字节时上面那条也红, 而这一条会绿

        `and` 的两半必须各自有证人 —— 今天错的那两个数恰好是**同时**错的, 只看那一侧的
        判据下次单独错一个数时就放过去了 (I19/I27 那一族)。
        """
        right_bytes = len(BODY)
        rows = audit(f"| x | `1111.2222v1` | {right_bytes} | 4242 |",
                     {"1111.2222v1": BODY}, [])
        self.assertEqual("anchor_mismatch", rows[0]["status"],
                         f"字节对、行数不对却判成 ok: {rows[0]}")

    def test_missing_bytes_are_no_bytes_not_ok(self):
        rows = audit("| x | `3333.4444v1` | 9 | 9 |", {}, [])
        self.assertEqual("no_bytes", rows[0]["status"])

    def test_citation_of_an_unfetched_paper_is_absent_not_ok(self):
        rows = audit("| x | `1111.2222v1` | 1 | 1 |", {},
                     [("9999.0000v1", 1, 1, "anything")])
        self.assertEqual("citation_absent", rows[-1]["status"],
                         "取不到正文时引文不许算命中 —— 那是拿缺证据当肯定证据")


class TestCitationRegistryIsUsable(unittest.TestCase):
    """CITATIONS 自己不许是一串装饰"""

    def test_every_cited_id_appears_in_a_table_that_main_actually_reads(self):
        """CITATIONS 里每个 id 都要落在**这条检查真会读到的那些表**里

        原先它只读 doc 58 那一份。`main()` 现在按 `DEFAULT_DOCS` 并读几份 (doc 59 起每轮调研
        各开一份文档)，两边口径一旦分叉就有两种失效：新文档的引文被误判成孤儿（今天真的红了），
        或者真正的孤儿因为藏在某份被读到的表里而蒙过去。所以这里读的是 `DEFAULT_DOCS`，
        与 `main()` 同一组文件 —— 判据与被测对象共用一个真源，而不是各写一遍 glob。
        """
        table = _read_default_tables()
        orphans = {c[0] for c in CITATIONS} - set(table)
        self.assertEqual(set(), orphans,
                         f"引文引用了表里没有的论文, 它永远不会被取到: {sorted(orphans)}")

    def test_every_default_doc_pattern_still_matches_a_file(self):
        """`DEFAULT_DOCS` 里的模式不许悄悄指到空集

        文档改名而这里没跟着改时，`main()` 会退回「只解析到剩下那几份」—— 少一份表就是少一批
        从来没被重跑过的锚点，而它表现的形态是**一条绿**（I28：可重建这件事没被重建过就只是修辞）。
        """
        for pattern in DEFAULT_DOCS:
            hits = sorted(REPO.glob(pattern))
            self.assertTrue(hits, f"DEFAULT_DOCS 的模式没匹配到任何文件: {pattern}")
            self.assertTrue(len(hits) <= 3,
                            f"模式 {pattern} 匹配到 {len(hits)} 份, 宽到会把别的东西算进表里")

    def test_ranges_are_well_formed_and_non_empty(self):
        self.assertTrue(CITATIONS)
        for arxiv_id, start, end, needle in CITATIONS:
            self.assertGreaterEqual(end, start, f"{arxiv_id}:{start}-{end} 区间是反的")
            self.assertGreaterEqual(start, 1, f"{arxiv_id} 行号从 1 起")
            self.assertTrue(needle.strip(), f"{arxiv_id}:{start} 的 needle 是空的")

    def test_the_bundled_selftest_has_a_reader_in_the_suite(self):
        self.assertEqual(0, main(["--selftest"]))


class TestAdviceIsTriaged(unittest.TestCase):
    """收尾那句要分诊：「取数失败」与「文档里的表错了」的处置**相反**

    2026-09-30 实跑撞出来的：五个 id 里有一个瞬时取回失败，那次读数是 22/36，而原句无条件
    写着「要改的是文档里那张表」。照做就把一次网络抖动「核」成了一份错的表 —— 比不核更糟，
    因为留下的改动看着像有依据。
    """

    def test_all_green_says_nothing(self):
        """反向对照：全对时不许冒出一句「要改表」"""
        rows = [{"arxiv_id": "1", "status": "ok"}, {"arxiv_id": "1", "status": "cited",
                                                  "needle": "x"}]
        self.assertEqual("", closing_advice(rows, []))

    def test_a_fetch_failure_is_not_a_table_error(self):
        rows = [{"arxiv_id": "9.9v1", "status": "no_bytes", "want_bytes": 1,
                 "needle": None},
                {"arxiv_id": "9.9v1", "status": "citation_absent", "needle": "x",
                 "start": 1, "end": 1}]
        advice = closing_advice(rows, ["9.9v1"])
        self.assertIn("没取回字节", advice)
        self.assertIn("不是表错", advice)
        self.assertNotIn("要改的是**文档里那张表**", advice)

    def test_a_real_mismatch_still_points_at_the_table(self):
        """另一端也要有：字节都取回了、锚点却对不上 ⇒ 那才是要改表"""
        rows = [{"arxiv_id": "1", "status": "anchor_mismatch", "want_bytes": 1,
                 "got_bytes": 2, "got_lines": 3, "want_lines": 4, "needle": None},
                {"arxiv_id": "1", "status": "citation_absent", "needle": "x",
                 "start": 1, "end": 1}]
        advice = closing_advice(rows, [])
        self.assertIn("要改的是**文档里那张表**", advice)
        self.assertNotIn("没取回字节", advice)

    def test_no_bytes_row_alone_triggers_the_fetch_branch(self):
        """只靠行状态也要走取数那一支 —— `fetch_failed` 可能因为表里没这行而空着"""
        rows = [{"arxiv_id": "1", "status": "no_bytes", "want_bytes": 5, "needle": None}]
        self.assertIn("没取回字节", closing_advice(rows, []))

    def test_the_fetch_failure_reaches_the_advice_through_main(self):
        """接线也要有证人：`except` 那支若不记 id，只剩「0 个 id 没取回字节」这种空话

        打的是整条路 —— 假 urlopen 全部抛错，`main` 必须**既**报非零退出码，**又**在收尾
        那句里带上具体的 id（而不是把取数失败说成表错）。
        """
        import io
        import contextlib
        from unittest import mock
        from benchmarks import recheck_arxiv_anchors as rc

        def _boom(*a, **kw):
            raise OSError("模拟：端点不可达")

        buf = io.StringIO()
        with mock.patch.object(rc.urllib.request, "urlopen", _boom):
            with contextlib.redirect_stdout(buf):
                code = main(["--doc", "docs/research/58-*.md"])
        out = buf.getvalue()
        self.assertEqual(1, code, "全部取数失败却退出码 0 ⇒ 这条检查在 CI 等于没跑")
        self.assertIn("没取回字节", out)
        # ⚠️ 断言要打在**那句建议里的位置**上，不是「输出里有没有这个 id」：每个失败 id 都会
        # 单独印一行 `❌ 取 <id> 失败`，所以只查 id 的话，R5（except 里不记 id）也能全绿 ——
        # 与 H3 同一条 I32，本会话第二次撞在同一个机制上。
        self.assertIn("没取回字节**（2609.", out,
                      "收尾那句没把具体 id 跟在短语后面 ⇒ fetch_failed 那条接线断了")
        self.assertNotIn("0 个 id **没取回字节**", out)
        self.assertNotIn("要改的是**文档里那张表**", out)

    def test_main_actually_asks_for_the_advice(self):
        """函数写了没人调 = 本仓反复反对的那一种 (变异表里同款一行)"""
        import ast
        src = (REPO / "benchmarks/recheck_arxiv_anchors.py").read_text(encoding="utf-8")
        mfn = next(n for n in ast.walk(ast.parse(src))
                   if isinstance(n, ast.FunctionDef) and n.name == "main")
        self.assertIn("closing_advice", ast.dump(mfn),
                      "main() 不叫 closing_advice ⇒ 那句误诊还留在输出里")


class TestResponseVsFetchFailure(unittest.TestCase):
    """`main()` 的取数侧：一条**带正文的 404** 是响应，不是「取不到」

    2026-09-30 读 doc 60 时撞出来的，两处都在 `main()` 里，而且都不是「表错了」：

    ① Vestrum `2609.33822v1` 没有 HTML —— `export.arxiv.org` 给 **HTTP 404 + 126 B 正文**，
       而 `arxiv.org` 给 404 + 7,901 B / 31 行的成样式页（第 8 行逐字 `No HTML for '…'`）。
       旧代码任何 `HTTPError` 都记成取数失败 ⇒ **那一篇永远核不了**，而它正文恰恰是要引的东西。
       把「服务器明确告诉你没有」与「网络没答话」压成同一格，就是被引那篇 (Limbo) 批评的
       「把未知当失败」长在核它的工具上 (doc 60 §七)。
    ② `--doc` 只载入一份表，`CITATIONS` 却是全局扁平清单 ⇒ 别的文档的引文全被判成
       「锚点对不上」。第一次实测是 **32/65**，而模块 docstring 的用法行就在推荐这条 (`# 只核一份`)。
       只补「默认并读多份」不够 —— 那修的是默认路径，没修 `--doc` 本身。
    """

    @staticmethod
    def _raises(body: bytes):
        """一个真的 `urlopen`: 抛 HTTPError 并把正文挂在 fp 上

        第一版这里写成 `side_effect=lambda *a: self._http_error(body)` —— 那**不会抛**,
        Mock 把返回的异常实例当普通返回值, 而 `HTTPError` 自己就支持 `with` 与 `.read()`,
        于是代码走的是 200 那条成功分支, 测出来的绿与我打算测的那件事毫无关系 (I32 那一族:
        判据的输入是空气)。
        """
        def _call(*a, **k):
            raise urllib.error.HTTPError("https://x/y", 404, "Not Found", {}, io.BytesIO(body))
        return _call

    def _run_main(self, table_md: str, side_effect, citations, only_one_doc=True):
        """把 `main()` 的取数换成注入的假响应；表住在临时文件里"""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "doc.md")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(table_md)
            argv = ["--doc", path] if only_one_doc else []
            docs = (path,) if not only_one_doc else DEFAULT_DOCS
            with mock.patch.object(urllib.request, "urlopen", side_effect=side_effect), \
                 mock.patch("benchmarks.recheck_arxiv_anchors.CITATIONS", citations), \
                 mock.patch("benchmarks.recheck_arxiv_anchors.DEFAULT_DOCS", docs), \
                 contextlib.redirect_stdout(io.StringIO()) as buf:
                code = main(argv)
            return code, buf.getvalue()

    def test_a_404_with_a_body_is_compared_against_the_table(self):
        """成立那一侧：404 的正文照常进字节/行数比对，表对得上就绿、对不上就红

        只测「不崩」是不够的 —— 判据是它**真的进了比对**：表里数字错时同一条路径必须红，
        否则「404 也收」会被写成「404 一律放过」（那是把缺证据读成通过的另一半）。
        """
        body = b"<html><body><p>404</p></body>"
        row = _table_row("2609.33822v1", body)
        cites = [("2609.33822v1", 1, 1, "404")]
        code, out = self._run_main(row + "\n", self._raises(body), cites)
        self.assertEqual(0, code, out)
        self.assertIn("带正文的响应", out, "那句要印出来, 否则下一个人读不出这是 404 不是 200")

        code2, out2 = self._run_main("| x | `2609.33822v1` | 9,999 | 99 |\n",
                                     self._raises(body), cites)
        self.assertEqual(1, code2, out2)
        self.assertIn("文档里那张表", out2, "取到了正文却对不上 ⇒ 建议要指表, 不能指网络")

    def test_a_404_with_no_body_is_still_a_fetch_failure(self):
        """反侧：404 且**没有正文**时不许当成响应 —— 那是真取不到"""
        row = _table_row("1111.2222v1", BODY)
        code, out = self._run_main(row + "\n", self._raises(b""), [])
        self.assertEqual(1, code, out)
        self.assertIn("没有正文", out)
        self.assertNotIn("带正文的响应", out)

    def test_single_doc_run_does_not_consent_to_other_docs_citations(self):
        """`--doc` 载入一份表时, 别的引文要**筛掉并说明**, 不许判成红

        旧行为实测是 32/65：31 条别的文档的引文全被判成「锚点对不上」。
        """
        row = _table_row("1111.2222v1", BODY)
        others = [c for c in CITATIONS if c[0] != "1111.2222v1"]
        self.assertTrue(others, "构造前提: 全局清单里得有别的文档的引文")
        fake = mock.MagicMock()
        fake.__enter__.return_value.read.return_value = BODY
        code, out = self._run_main(row + "\n", lambda *a, **k: fake, others)
        self.assertEqual(0, code, out)
        self.assertIn("本轮未核", out)
        self.assertNotIn("❌ 行 ", out, "别的文档的引文不许变成红点")

    def test_full_run_with_an_orphan_citation_is_exit_2_not_green(self):
        """全量跑里有引文的 id 不在任何表 ⇒ **退出码 2**：那批锚点从来没被重跑过 (I28)

        这一条不 patch `DEFAULT_DOCS` 就测不到 —— 真实表覆盖了所有在册 id，所以把 `CITATIONS`
        换成一条不存在的 id，让它必然落进「筛掉了、但不是因为只读一份」那一格。
        """
        nb, nl = _table_counts(BODY)
        row = f"| x | `1111.2222v1` | {nb} | {nl} |\n"
        fake = mock.MagicMock()
        fake.__enter__.return_value.read.return_value = BODY
        code, out = self._run_main(row, lambda *a, **k: fake,
                                   [("9999.9999v9", 1, 1, "no-such-needle")],
                                   only_one_doc=False)
        self.assertEqual(2, code, out)
        self.assertIn("从来没被重跑过", out)


    def test_the_empty_table_message_names_the_files_it_read(self):
        """「一条锚点都没解析出来」那句要说**读了哪些文件**，不许说 `None`

        原句是 `f"❌ 在 {args.doc} 里…"` —— 全量跑时 `args.doc` 就是 `None`，于是消息把原因指向
        一个根本没被读过的东西。一条指错原因的错误消息比没有消息更贵（处方式错误那条纪律）。
        """
        code, out = self._run_main("| 散文里提到 `1111.2222v1` 但不是四列的表行 |\n",
                                   lambda *a, **k: mock.MagicMock(), [])
        self.assertEqual(2, code, out)
        self.assertIn("doc.md", out, "那句要列出真读过的文件名")
        # ↓ 两条**反列 repr**: 变异 R6 把消息换成 `{args.doc}` 时, 印出来是 `['/tmp/…/doc.md']`,
        #   它**也含** `doc.md` ⇒ 只断言 `In("doc.md")` 杀不掉那条 (第一次标定就逃逸在此)。
        #   真正要钉的是「消息里说的是文件名, 不是某个参数的 repr / None」。
        self.assertNotIn("None", out, out)
        self.assertNotIn("['", out, "列 repr 混进消息 ⇒ 它说的是「参数」而不是「读了哪些文件」")


def _table_counts(body: bytes) -> tuple:
    from benchmarks.arxiv_html_to_lines import to_lines
    return len(body), len(to_lines(body.decode()))


if __name__ == "__main__":
    unittest.main()
