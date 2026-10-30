"""把 doc 58 §一 那张「出处锚点表」从**一句声明**变成**一条能跑的检查**

为什么需要它 (2026-09-30 实跑撞出来的): 上一片把 `arxiv_html_to_lines.py` 进仓库, 理由是
「调研文档里的行号是同一种数, 抽取代码不在仓里就不可重建」(doc 55 §六 / I23 的续账)。可那条
**可重建**这件事本身从来没被重建过一次 —— 今天照表重跑四篇 (重新 `curl` 同一批 URL、跑同一把
尺), 四篇里**三篇的字节数与行数逐字对上**, 第四篇 (`2609.30059v1` KernelOPT) 表里印的
1,058,127 B / 3,344 行 两个数都对不上: 真实是 1,087,672 B / 3,470 行。而那个 3,344 是从一份
**不是这把尺产出的**文件上抄来的 (`/tmp` 里那份有 4,208 行、连行号前缀都没有)。
⇒ 与 CHECKLOG「出处门自己也是一个算了没人读」同族: 一个声称可复现的锚点, 没被复现过就只是
一句修辞。这一片把这句话变成一条退出码。

姿势:
  · **表是唯一真源**。脚本从 markdown 里读那张表 (哪个 id、多少字节、多少行), 不另存一份副本 ——
    副本就是第二个真源, 会漂。
  · **引用要落在它声明的那个行号区间里**。`CITATIONS` 是本文档真正据以下结论的那几句 (每条带
    一个行号区间与一段只能在**该区间内**找到的字面片段)。跨行的引文必须按区间找 (行 499-501
    那句被块级换行劈成两半), 而「按区间找」要有牙齿, 判据就得成对: 片段落在区间外时**不许找到**。
  · 取不到 / 抽不出来 ⇒ 如实报错退出, 不许「拿不到就算通过」(那是把缺证据读成通过)。

用法:
    python3 -B benchmarks/experiments/misc/recheck_arxiv_anchors.py            # 走网络, 真重跑
    python3 -B benchmarks/experiments/misc/recheck_arxiv_anchors.py --selftest # 不打网络, 只核这台机器
    python3 -B benchmarks/experiments/misc/recheck_arxiv_anchors.py --doc docs/research/59-*.md   # 只核一份

⚠️ 锚点表**可以住在多份文档里** (每轮调研各开一份), 而 `CITATIONS` 是一份扁平清单。第一次只读
一份文档的表时, 报的是 **13/32** —— 19 条 ❌ 全是另一份里的引文, 它们没被解析到表里, 于是被判成
「锚点对不上」。修法有两层, 第二层是 2026-09-30 补的: ① 默认并读 `DEFAULT_DOCS` 那几份表;
② `--doc` 只载入一份表时, **引文要按表筛** —— 只改① 的话 `--doc 只核这一份` 那条用法本身就是
一条必然假红 (实跑 32/65, 31 条别的文档的引文被判红)。筛掉的条数会印出来: 全量跑里出现「有引文
不在任何表里」是**退出码 2**, 因为那意味着那批锚点从来没被重跑过 (I28 那一族)。
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import urllib.request
from urllib.error import HTTPError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arxiv_html_to_lines import to_lines  # noqa: E402  ← 同一把尺, 不复制实现

URL = "https://export.arxiv.org/html/{arxiv_id}"

#: 带锚点表的调研文档 (glob, 相对仓库根)。新开的调研单文档要往这里加一行 ——
#: 漏掉的表现不是报错, 是那份表里的锚点从来没被重跑过 (I28 那一族)。
DEFAULT_DOCS = ("docs/research/58-*.md", "docs/research/59-*.md", "docs/research/60-*.md")

# 那张表的形状: | 论文 | `arXiv id` | 字节 | 行数 | 读到哪 |
# 只认「第 2 格是个 arXiv id、第 3/4 格是带千分位的数」的行 —— README 里那种把 id 写进长句子的
# 行不能被当成表行 (它没有那四列的形状)。
_ROW = re.compile(r"^\|[^|]*\|\s*`?(?P<id>\d{4}\.\d{4,5}(v\d+)?)`?\s*"
                  r"\|\s*(?P<bytes>[\d,]+)\s*\|\s*(?P<lines>[\d,]+)\s*\|")


def to_int(s: str) -> int:
    return int(s.replace(",", "").strip())


def parse_anchor_table(md: str) -> dict:
    """从 markdown 里读出 {arxiv_id: (html_bytes, line_count)}

    同一个 id 出现多次时**后写的赢** (表被改过而旧行没删, 那要看得见而不是被第一次出现定死)。
    """
    out: dict = {}
    for line in md.splitlines():
        m = _ROW.match(line)
        if m:
            out[m.group("id")] = (to_int(m.group("bytes")), to_int(m.group("lines")))
    return out


def check_bytes_and_lines(rec: tuple, body: bytes) -> dict:
    """核一篇: 表里印的字节数/行数 vs 这批字节经同一把尺实际产出的字节数/行数

    `body` 收**字节**而不是字符串: 「表里那个字节数」是 HTTP 响应体的长度, 解码后再 encode
    回去只在「原文是合法 UTF-8」时才相等 —— 而那条前提不该被这条判据依赖。
    """
    want_bytes, want_lines = rec
    lines = to_lines(body.decode("utf-8", "replace"))
    return {
        "want_bytes": want_bytes, "got_bytes": len(body),
        "want_lines": want_lines, "got_lines": len(lines),
        "lines": lines,
    }


def citation_in_range(lines: list[str], start: int, end: int, needle: str):
    """引文核对: 只在声明的那个行号区间内找 (`行 N-M` 的 M 可能是被块级换行劈开的)

    返回命中的 1-based 行号, 找不到返回 None。**区间是判据的全部价值所在** —— 把整篇拼起来找
    任何片段都会命中, 那等于什么都没核 (I25 那一族: 判据的输入是空气时断言永真)。
    """
    if not lines or start < 1 or end < start:
        return None
    window = " ".join(lines[start - 1:min(end, len(lines))])
    if needle not in window:
        return None
    return start


def audit(doc_md: str, fetched: dict, citations: list) -> list[dict]:
    """把表 + 引文一起过一遍, 返回逐条判定 (不打网络, 便于测试注入假 fetch)

    `fetched` = {arxiv_id: 响应体字节}。表里有、`fetched` 里没有的 id 记成 `no_bytes` 而不是通过。
    """
    table = parse_anchor_table(doc_md)
    rows = []
    for arxiv_id, rec in sorted(table.items()):
        body = fetched.get(arxiv_id)
        if body is None:
            rows.append({"arxiv_id": arxiv_id, "status": "no_bytes", **{"want_bytes": rec[0],
                                                                        "want_lines": rec[1]}})
            continue
        r = check_bytes_and_lines(rec, body)
        r["arxiv_id"] = arxiv_id
        r["status"] = ("ok" if r["want_bytes"] == r["got_bytes"]
                       and r["want_lines"] == r["got_lines"] else "anchor_mismatch")
        rows.append(r)
    by_id = {r["arxiv_id"]: r for r in rows}
    for arxiv_id, start, end, needle in citations:
        row = by_id.get(arxiv_id)
        got = citation_in_range(row["lines"] if row and "lines" in row else [],
                                start, end, needle)
        # `is not None` 而不是真值: 命中的**行号**可能是 0 (行号从 1 起 ⇒ 0 是非法输入),
        # 用真值判断会把「非法输入恰好命中」读成没命中 —— 那是两种不同的错被折成一个。
        yield_citation = {"arxiv_id": arxiv_id, "start": start, "end": end,
                          "needle": needle,
                          "status": "cited" if got is not None else "citation_absent"}
        rows.append(yield_citation)
    return rows


# 本文档真正据以下结论的那几句。**逐字取自 `to_lines` 的输出** —— 一条「逐字引用」如果不在
# 这把尺产出的文本里, 那它引用的是我改写过的另一份 (doc 55 §一 那条 NBSP 教训的原样复现)。
CITATIONS = [
    ("2609.35140v1", 150, 150, "immutable evaluation specification"),
    ("2609.35140v1", 209, 209, "fully measured"),
    ("2609.35140v1", 214, 216, "revisions must start from a retained verified parent"),
    ("2609.11987v1", 78, 78, "a normalizer that assumes one convention doubles the other"),
    ("2609.11987v1", 349, 351, "treated the wall-clock ceiling as a graded failure"),
    # 跨行的那两句: 声明的区间必须覆盖到断行之后的那一半, 否则逐字引文会「查无此文」。
    ("2609.33074v1", 499, 501, "compared with 17.6 and 29 with CUDA"),
    # ⚠️ 这一条第一版写的是 needle="cache-inclusive" —— **那是我自己拼的词**, 原文写的是
    # "(inclusive)"。跑真机重核时它红了, 而红的是**我的引文**不是那篇论文。⇒ needle 必须逐字
    # 取自这把尺产出的文本, 不能凭印象拼 (与 doc 55 §一 那条 NBSP 教训同一族: 逐字引用的
    # 反义词不是「错」, 是「引用了一份我改写过的文本」)。
    ("2609.11987v1", 1108, 1111, "exclusive of cache reads and writes"),
    # ↓ 2026-09-30 读的 KernelOPT §4.4 与附录 B (doc 58 §四/§4.1)。这四条是那片代码与
    #   那篇文档共同依赖的锚点 —— 引它们而不登记, 就还是「出处门没人读」。
    ("2609.30059v1", 1344, 1347, "a float64 reference"),
    ("2609.30059v1", 1351, 1351, "correct TF32 kernels yield"),
    ("2609.30059v1", 363, 364, "The four gates are applied sequentially"),
    ("2609.30059v1", 418, 421, "marginal improvements"),
    # ↓ 2026-09-30 补读 KernelZero 方法节 §2 (行 115-198, doc 58 §四)。这六条是那一段的
    #   全部引用点 —— 上一轮那句「方法节未读」就是没有这些锚点时的样子 (I28: 声明「可重建」
    #   的东西必须真重跑一次, 而引文本身也是被引的东西)。
    ("2609.33074v1", 144, 144, "matches the sampled API set"),
    ("2609.33074v1", 145, 145, "following the format of KernelBench"),
    ("2609.33074v1", 147, 147, "NaN or Inf is filtered out"),
    ("2609.33074v1", 158, 158, "approximately half of the sampled kernels are correct"),
    ("2609.33074v1", 158, 158, "neither trivial nor impossible"),
    ("2609.33074v1", 117, 117, "explicitly defined interfaces"),
    ("2609.33074v1", 181, 181, "performance optimization is activated only when correctness becomes sufficiently reliable"),
    ("2609.33074v1", 191, 191, "their normalized speedup scores are set to zero"),
    # ↓ 2026-09-30 读的 Φ-Bench (doc 59)。这十二句是那一篇里被我方**用来下结论**的全部引用,
    #   每条都先在 `arxiv_html_to_lines.py` 产出的文本里 `grep -F -c` 核过命中数为 1 (I28:
    #   needle 不许凭印象拼)。行 78 与 103 是**同一意思的两个措辞** —— 分别登记而不是复用一条,
    #   因为 `citation_in_range` 按声明区间找, 措辞相同的两句落在不同行时必须各自可重建。
    ("2609.10226v1", 172, 172, "the reference solution must achieve a stable performance of at least 1.15"),
    ("2609.10226v1", 172, 172, "the starter solution must fail at least one test"),
    ("2609.10226v1", 173, 173, "revised or excluded rather than assigned a score"),
    ("2609.10226v1", 177, 177, "buildability, functional correctness, edit constraints, and anti-cheating checks are evaluated before any reward"),
    ("2609.10226v1", 181, 181, "AB-BA paired measurement protocol with at least five measurement pairs"),
    ("2609.10226v1", 183, 183, "A candidate receives a reward of zero if the reference solution does not yield a valid performance"),
    ("2609.10226v1", 193, 193, "recovering upstream code from published Python packages"),
    ("2609.10226v1", 193, 193, "hard-coding expected outputs, skipping or bypassing correctness checks"),
    ("2609.10226v1", 306, 306, "a single NVIDIA H20 GPU to every task, while disabling all other GPU types"),
    ("2609.10226v1", 319, 319, "performance does not increase consistently with reasoning budget"),
    ("2609.10226v1", 103, 103, "their evaluation settings are typically restricted to individual functions or small collections of isolated components"),
    ("2609.10226v1", 78, 78, "existing benchmarks mainly focus on isolated kernels, predefined operators, or pre-specified optimization targets"),
    # ↓ 2026-09-30 读的 Limbo `2609.29095v1` 与 HarnessLens `2608.27311v1` (doc 60)。
    #   Limbo 这十九句是本篇 §三~§五 的全部引用点。**needle 一律只取散文半句**: 这篇的公式在
    #   抽取文本里是 Unicode+LaTeX **双语并列** (`s∈{absent,…}` 紧跟 `s\in\{\textsf{absent}…\}`),
    #   取带数学的那半 = 把噪声抄成锚点, 下一次抽取实现改动时它会莫名变红 (doc 60 §一)。
    ("2609.29095v1", 114, 114, "the action may or may not have taken effect"),
    ("2609.29095v1", 118, 118, "idempotency keys, conditional writes and queryable operation"),
    ("2609.29095v1", 164, 164, "Transparent client-side retries cut exactly-once success from 72% to 50%"),
    ("2609.29095v1", 222, 222, "hidden fact the outcome state"),
    ("2609.29095v1", 224, 225, "reflects only effects committed before"),
    ("2609.29095v1", 230, 230, "A recovery policy therefore cannot condition on the outcome state"),
    ("2609.29095v1", 238, 239, "a refunded double charge was still charged"),
    ("2609.29095v1", 244, 245, "the late original becomes a no-op in every world"),
    ("2609.29095v1", 476, 477, "an annotation telling the"),
    ("2609.29095v1", 531, 531, "duplicate rate is 0.5%"),
    # ⚠️ 这一条第一版写的是 needle="under a misleading HTTP 500 it is 8%" —— 红。原因是原文那个
    # "HTTP" 与 "500" 之间是 **U+00A0 不换行空格** (`HTTP\xa0500`)，而抽取器按设计**不折叠 NBSP**
    # (doc 55 §一 那条教训的本体)。⇒ needle 从 NBSP 之后起, 用一个普通空格永远撞不中。
    ("2609.29095v1", 532, 532, "500 it is 8%"),
    ("2609.29095v1", 536, 536, "56% of late-commit and 74% of"),
    ("2609.29095v1", 538, 539, "69% produced a duplicate"),
    ("2609.29095v1", 541, 541, "duplicated (100%)"),
    ("2609.29095v1", 544, 545, "Redelivery never produces an error at all"),
    ("2609.29095v1", 1356, 1357, "regenerating it on a retry"),
    ("2609.29095v1", 1360, 1361, "abandon escalation in favour of a verification that cannot see"),
    ("2609.29095v1", 1405, 1405, "exactly once on a key-less write"),
    ("2609.29095v1", 1408, 1408, "Agents rarely report the duplicates they cause"),
    # HarnessLens: 门的那三条 (伪码 / 附录声明 / 「preservation 不够」) 是 #103 直接引的形状,
    # 行 1050 那句「Nothing is re-executed」是本篇推荐的**零机时做法**, 所以它也必须有锚点。
    ("2608.27311v1", 99, 99, "allowing aggregate scores to obscure specific regressions"),
    ("2608.27311v1", 187, 187, "A deterministic controller"),
    ("2608.27311v1", 199, 199, "are not incorporated into"),
    ("2608.27311v1", 200, 200, "executes no tasks and has no access to trajectories"),
    ("2608.27311v1", 258, 258, "is inaccessible throughout harness evolution"),
    ("2608.27311v1", 715, 715, "if attributable positive evidence and no attributable regression then"),
    ("2608.27311v1", 878, 878, "A verification stage advances only if it contains attributable positive evidence"),
    ("2608.27311v1", 879, 879, "preservation alone is insufficient"),
    ("2608.27311v1", 1050, 1050, "both rules are applied to the same stored evidence"),
    ("2608.27311v1", 1051, 1051, "The two rules differ on 7 of the 19 paired iterations"),
    ("2608.27311v1", 1053, 1053, "found no task recovery attributable to a modified component"),
    # ↓ 这一条引的是**「没有全文」这件事本身**。Vestrum 的 /html 两个主机都给 404, 但正文不同:
    #   `arxiv.org` 给 7,901 B / 31 行的成样式页 (第 8 行逐字 "No HTML for '…'"), 而**本脚本走的是
    #   `export.arxiv.org`**, 那里只有 126 B / 1 行 `404`。表与 needle 都按脚本能重取的那一份记 ——
    #   抄那句 `No HTML for` 会得到一条**永远红**的锚点 (doc 60 §七: 可核的那份 ≠ 好看的那份)。
    ("2609.33822v1", 1, 1, "404"),
]


def closing_advice(rows: list, fetch_failed=None) -> str:
    """红名单**在说什么** —— 「取数失败」与「文档里的表错了」处置相反，不许共用一句建议

    2026-09-30 撞出来的：五个 id 里有一个瞬时取回失败，那次读数是 **22/36**，而收尾那句无条件
    写着「要改的是**文档里那张表**」。照做就把一次网络抖动「核」成了一份错的表 —— 比不核更糟，
    因为它留下的改动看着像有依据。⇒ 先分诊再给建议。
    """
    failed = list(fetch_failed or [])
    bad = [r for r in rows if r.get("status") not in ("ok", "cited")]
    if not bad:
        return ""
    n_no_bytes = sum(1 for r in bad if r.get("status") == "no_bytes")
    if failed or n_no_bytes:
        ids = ", ".join(failed) if failed else "见上面那行 ❌"
        return (f"⇒ 本轮 {len(failed)} 个 id **没取回字节**（{ids}），"
                f"{n_no_bytes} 行锚点因此无从比对 —— 那是**取数失败**，不是表错。"
                f"这句建议不能拿去改文档里的表：先重跑，仍失败就把那些条如实记成「未核」")
    return "⇒ 上面这些数要改的是**文档里那张表**, 不是这里 (表是唯一真源)"


def selftest() -> int:
    """这台机器上不需要网络的那部分: 解析、区间判据的两侧、缺字节的诚实处理"""
    md = ("| 论文 | arXiv id | HTML 字节 | 抽出行数 | 读到哪 |\n"
          "|---|---|---|---|---|\n"
          "| A | `1111.2222v1` | 12 | 3 |\n"
          "| B | `3333.4444v1` | 9,999 | 42 |\n"
          "| 一句散文里提到 `5555.6666v1` 不是表行 |\n")
    fails = []

    table = parse_anchor_table(md)
    if set(table) != {"1111.2222v1", "3333.4444v1"}:
        fails.append(f"表解析到的 id 不对: {sorted(table)}")
    if table.get("3333.4444v1") != (9999, 42):
        fails.append(f"千分位没被拆开: {table.get('3333.4444v1')}")
    # 反空转: 一条都没解析出来时上面两条都会红, 但这条把「为什么」说清
    if not table:
        fails.append("解析出空表 ⇒ 后面所有相等断言都会永真地绿")

    fake = {"1111.2222v1": b"<!DOCTYPE html><body><p>one</p><p>two</p><p>three</p></body>"}
    n_bytes = len(fake["1111.2222v1"])
    n_lines = len(to_lines(fake["1111.2222v1"].decode()))
    # 表里 A 那一行按**真实值**写 (它今天写的是 12/3, 那是**我自己估的** —— 见下面 kinds 的断言),
    # 于是「ok」这一侧也有样本; B 那一行故意不给字节, 用来核「没取到 ≠ 通过」。
    md_ok = (f"| A | `1111.2222v1` | {n_bytes} | {n_lines} |\n"
             "| B | `3333.4444v1` | 9,999 | 42 |\n"
             "| 一句散文里提到 `5555.6666v1` 不是表行 |\n")
    rows = audit(md_ok, fake, [("1111.2222v1", 2, 2, "two"),      # 成立那一侧
                               ("1111.2222v1", 2, 2, "one"),      # 片段在**别的行** ⇒ 不许命中
                               ("1111.2222v1", 9, 9, "three")])   # 行号越界 ⇒ 不许命中
    # ⚠️ 三条的 needle 必须互不相同: 下面那个字典按 (id, needle) 索引, 两条同名时后写的会
    # 覆盖前一条 —— 第一版就是「two」既当命中样本又当越界样本, 于是命中样本被覆盖成
    # 「citation_absent」, selftest 自己报了一条红。判据的输入互相覆盖时, 读数就不是那条判据的。
    by = {(r.get("arxiv_id"), r.get("needle")): r.get("status") for r in rows}
    if by.get(("1111.2222v1", "two")) != "cited":
        fails.append("判据在**成立**那一侧就不工作: 行 2 上的 'two' 没被判成 cited")
    if by.get(("1111.2222v1", "one")) != "citation_absent":
        fails.append("'one' 在行 1 而声明的是行 2, 竟然也算命中 ⇒ 区间没在约束任何东西")
    # 行号越界 (声明行 9 而全文只有 3 行) 不许变成命中
    if any(r.get("start") == 9 and r.get("status") == "cited" for r in rows):
        fails.append("行号越界却判成命中")
    # 表里有、字节没取到 ⇒ 必须是 no_bytes, 不许算通过
    kinds = {r["arxiv_id"]: r["status"] for r in rows if "needle" not in r}
    if kinds.get("3333.4444v1") != "no_bytes":
        fails.append(f"没取到字节却判成 {kinds.get('3333.4444v1')} —— 缺证据被读成通过")
    if kinds.get("1111.2222v1") != "ok":
        fails.append(f"表与实际相符时却没判成 ok: {kinds.get('1111.2222v1')}")
    # 锚点对不上必须被点名 (这一条就是今天真跑出来的那个形状: 表里的字节数/行数与实际不符)
    rows_bad = audit("| x | `1111.2222v1` | 99 | 99 |\n", fake, [])
    if not any(r["status"] == "anchor_mismatch" for r in rows_bad):
        fails.append("表里印的字节/行数与实际不符, 却没被判出来")
    # 同一个 id 在表里出现两次 (旧行没删) ⇒ 后写的赢, 这条是 parse 的 docstring 亲口说的
    if parse_anchor_table("| a | `1111.2222v1` | 1 | 1 |\n"
                          "| b | `1111.2222v1` | 7,000 | 8 |\n").get("1111.2222v1") != (7000, 8):
        fails.append("docstring 承诺「后写的赢」, 实际不是")

    print("selftest:", "FAIL " + "; ".join(fails) if fails
          else "ok（解析/区间判据两侧/缺字节/锚点不符/重复 id 都取过样）")
    return 1 if fails else 0


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--doc", action="append", default=None,
                    help="锚点表所在的文档, 可重复; 不给就用 DEFAULT_DOCS 那两份 glob")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()

    # 表可以住在**多份**文档里 (doc 58 之后每轮调研各开一份), 而 `CITATIONS` 是一份扁平清单。
    # 只读一份的后果不是报错, 是**假红**: 另一份里那些句子会被判成「锚点对不上」(第一轮跑
    # doc 59 时 13/32, 而 19 条 ❌ 全是 doc 58 的引文 —— 与「表改了这里没跟着改」是同一种失效)。
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    patterns = args.doc or [os.path.join(root, p) for p in DEFAULT_DOCS]
    texts = []
    for pattern in patterns:
        hits = sorted(glob.glob(pattern))
        if not hits:
            print(f"❌ 找不到锚点表所在的文档: {pattern}"
                  " —— 表搬了家而这里的模式没跟着改, 这条检查已经空转")
            return 2
        for path in hits:
            texts.append(open(path, encoding="utf-8").read())
    doc_md = "\n".join(texts)
    table = parse_anchor_table(doc_md)
    if not table:
        # 那句「在 None 里」是错的: 全量跑时 `args.doc` 就是 None, 而表其实是从 `DEFAULT_DOCS`
        # 拼出来的一组路径里读的 —— 一条指错原因的错误消息, 比没有消息更贵 (处方式错误那条纪律)。
        print(f"❌ 在这些文件里一条锚点都没解析出来: {', '.join(patterns)} —— "
              "表的形状变了, 这条检查已经空转")
        return 2
    fetched: dict = {}
    fetch_failed: list = []
    for arxiv_id in sorted(table):
        url = URL.format(arxiv_id=arxiv_id)
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                body = resp.read()
            fetched[arxiv_id] = body
            print(f"  取回 {arxiv_id}: {len(body):,} B")
        except HTTPError as exc:
            # **带正文的错误响应是响应**。`/html/<id>` 对「没有 HTML」的那篇返回 **404 + 一页正文**,
            # 而那页正文正是 doc 60 要引的东西 (第 8 行逐字印着 `No HTML for '<id>'`)。把它记成
            # 「取数失败」就等于把「服务器明确告诉你没有」与「网络没答话」压成同一格 ——
            # 与被引那篇 (Limbo) 批评的「把未知当失败」是同一个形状, 长在检查它的工具上。
            err_body = b""
            try:
                err_body = exc.read()
            except Exception:  # noqa: BLE001
                pass
            if err_body:
                fetched[arxiv_id] = err_body
                print(f"  取回 {arxiv_id}: {len(err_body):,} B (HTTP {exc.code} —— "
                      "那是**带正文的响应**, 不是取数失败)")
            else:
                fetch_failed.append(arxiv_id)
                print(f"  ❌ 取 {arxiv_id} 失败: HTTP {exc.code} 且**没有正文**")
        except Exception as exc:  # noqa: BLE001
            fetch_failed.append(arxiv_id)
            print(f"  ❌ 取 {arxiv_id} 失败: {type(exc).__name__}: {exc}")

    # 引文清单是**全局扁平**的, 表却可以只载入一份 (`--doc`)。不按表筛引文的后果是一条**必然假红**:
    # 2026-09-30 拿 `--doc docs/research/60-*.md` 跑, 报的是 32/65 —— 31 条 doc 58/59 的引文被判成
    # 「锚点对不上」, 而它们只是没被读进来。那句 `--doc # 只核这一份` 的用法本身在推荐这个坑。
    cits = [c for c in CITATIONS if c[0] in table]
    skipped = len(CITATIONS) - len(cits)
    if skipped:
        if not args.doc:
            print(f"❌ 全量跑却有 {skipped} 条引文的 id 不在任何一份表里 —— "
                  "那些锚点从来没被重跑过 (I28 那一族), 不能算通过")
            return 2
        print(f"  本次只载入 {len(table)} 篇的表 ⇒ {skipped} 条引文**本轮未核**"
              " (只核一份时的正常形状, 不代表那几篇有问题)")
    rows = audit(doc_md, fetched, cits)
    bad = 0
    for r in rows:
        if "needle" in r:
            flag = "✓" if r["status"] == "cited" else "❌"
            print(f"  {flag} 行 {r['start']}-{r['end']} {r['arxiv_id']}: {r['needle'][:58]!r}")
        else:
            ok = r["status"] == "ok"
            flag = "✓" if ok else "❌"
            got_b = f"{r['got_bytes']:,}" if "got_bytes" in r else "没取到"
            got_l = f"{r['got_lines']:,}" if "got_lines" in r else "没取到"
            print(f"  {flag} {r['arxiv_id']}: 字节 {r['want_bytes']:,} vs {got_b}"
                  f" · 行数 {r['want_lines']:,} vs {got_l}")
        bad += 0 if r.get("status") in ("ok", "cited") else 1
    print(f"\n锚点重跑: {len(rows) - bad}/{len(rows)} 对上")
    if bad:
        advice = closing_advice(rows, fetch_failed)
        if advice:
            print(advice)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
