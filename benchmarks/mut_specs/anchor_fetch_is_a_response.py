"""`tests/test_recheck_arxiv_anchors.py::TestResponseVsFetchFailure` 那张变异表的可执行形态

这一片修的是**检查器自己的两格**（doc 60 §七）：

① 带正文的 404 被记成「取数失败」⇒ Vestrum `2609.33822v1` 那条「无全文」的锚点永远核不了。
   把「服务器明确告诉你没有」与「网络没答话」压成同一格，正是本篇引的 Limbo 批评的
   「把未知当失败」，而它长在用来核 Limbo 的工具上。
② `--doc` 只载入一份表，`CITATIONS` 却是全局扁平清单 ⇒ 别的文档的引文全被判红
   （实跑 32/65）。只补「默认并读多份」修的是默认路径，没修 `--doc` 本身。

跑: python3 -B -m benchmarks.mutation_calibrate \
      --spec benchmarks/mut_specs/anchor_fetch_is_a_response.py
"""
import sys
from os.path import abspath, dirname

_ROOT = dirname(dirname(dirname(abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmarks.mutation_calibrate import Row  # noqa: E402

FILES = {"rc": "benchmarks/recheck_arxiv_anchors.py"}

TARGETS = ["tests/test_recheck_arxiv_anchors.py"]

ROWS = [
    Row("R1 带正文的 404 被记成取数失败 (整条 remedy 不走)", "rc",
        "            if err_body:\n                fetched[arxiv_id] = err_body",
        "            if False:\n                fetched[arxiv_id] = err_body",
        "⇒ `test_a_404_with_a_body_is_compared_against_the_table` 红: 表对得上却没绿, "
        "因为那条锚点根本没进比对"),

    Row("R2 空正文也算响应 (缺证据读成通过的另一半)", "rc",
        "            if err_body:",
        "            if True:",
        "⇒ `test_a_404_with_no_body_is_still_a_fetch_failure` 红: 404 且 0 字节被当成取到了, "
        "于是那条只能以「表错」的形态出现, 收尾建议会指错地方"),

    Row("R3 引文不按表筛 (全局清单原样交给 audit)", "rc",
        "    cits = [c for c in CITATIONS if c[0] in table]",
        "    cits = list(CITATIONS)",
        "⇒ `test_single_doc_run_does_not_consent_to_other_docs_citations` 红: 就是本轮实测的 "
        "32/65 那个形状 —— 别的文档的引文变成一片假 ❌"),

    Row("R4 全量跑的孤儿引文不再拉红 (缺的那批锚点没人知道)", "rc",
        "        if not args.doc:",
        "        if False:",
        "⇒ `test_full_run_with_an_orphan_citation_is_exit_2_not_green` 红: 表少一份时它表现成"
        "一条绿, 而那批锚点从来没被重跑过 (I28)"),

    Row("R5 「筛掉了 N 条」那句不印 (筛是静默的)", "rc",
        "    if skipped:\n        if not args.doc:",
        "    if False:\n        if not args.doc:",
        "⇒ 两条红 (单文档那条要看「本轮未核」印出来了, 全量那条要看退出码 2)。"
        "沉默的筛选与没有筛选不可区分 —— 这一行是 R3/R4 的读者"),

    Row("R6 空表那句消息把原因指向 `args.doc` (全量跑时它是 None)", "rc",
        "一条锚点都没解析出来: {', '.join(patterns)}",
        "一条锚点都没解析出来: {args.doc}",
        "⇒ `test_the_empty_table_message_names_the_files_it_read` 红: 消息说「在 None 里」, "
        "而表其实是从 `DEFAULT_DOCS` 那组文件读的 —— 指错原因的错误消息比没有消息更贵"),
]
