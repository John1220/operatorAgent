"""arXiv HTML → 带行号的纯文本 —— 让文档里每条「行 N-M」引用都能被重建

为什么进仓库（而不是又写成一个 /tmp 脚本）：本仓刚为同一件事发过两个提交 —— 那张变异表只在
文档里数过、锚点没落到能重建脚本的地方，于是「13/13」不可复现（doc 55 §六 / CHECKLOG I23）。
调研文档里的**行号**是同一种数：它由一段抽取代码产生，代码不在仓里，行号就不可重建。

两条自己的规矩（都是这仓撞过的坑）：
  · **不许折叠 NBSP**。doc 55 §一 记过一次：上一版抽取把 `&nbsp;` 变成普通空格，于是
    「逐字引用」其实是在引用一份我改写过的文本。这里只压 ASCII 空白（空格/制表/换行），
    U+00A0 与其它非断行字符原样保留。
  · **抽出来多少段要印出来**，而且要能被反查。抽取器空转的形态是「两边都返回空集 ⇒ 任何
    相等断言永真地绿」（doc 49 §八 那条），所以 `--selftest` 拿一段合成 HTML 验它真认得段落。

用法：
    python3 -B benchmarks/experiments/misc/arxiv_html_to_lines.py in.html out.txt
    python3 -B benchmarks/experiments/misc/arxiv_html_to_lines.py --selftest
"""
from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

# 只压 ASCII 空白：NBSP(U+00A0)、窄空格(U+202F)、零宽(U+200B) 等一律原样留着
_ASCII_WS = re.compile(r"[ \t\r\n\f\v]+")
_TAG = re.compile(r"<[^>]+>")
_BLOCK = re.compile(r"</(p|div|h[1-6]|li|tr|figcaption|section)>", re.I)
_STRIP = re.compile(r"<(script|style)\b.*?</\1>", re.I | re.S)
_DROP_HEAD = re.compile(r"^<!DOCTYPE.*?<body[^>]*>", re.I | re.S)


def to_lines(raw: str) -> list[str]:
    """HTML 字符串 → 行列表（一行一个块级元素，块内空白压成单个空格）"""
    body = _DROP_HEAD.sub("", raw, count=1)
    body = _STRIP.sub(" ", body)
    # 块级闭合标签换成换行, 其余标签删掉 —— **先删标签、后还原实体**: 反过来做的话,
    # 正文里被引用的 `&lt;p&gt;` 会先变成真标签再被删掉, 那半句话就从引文里消失了
    # (selftest 里那条 `kept` 断言盯的就是这件事)
    body = _BLOCK.sub("\n", body)
    text = html.unescape(_TAG.sub("", body))
    lines: list[str] = []
    for chunk in text.split("\n"):
        chunk = _ASCII_WS.sub(" ", chunk).strip()
        if chunk:
            lines.append(chunk)
    return lines


def write_numbered(lines: list[str], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as fh:
        for i, line in enumerate(lines, 1):
            fh.write(f"{i}\t{line}\n")


SELFTEST_HTML = (
    "<!DOCTYPE html><html><head><script>var x='<p>stub</p>';</script></head><body>"
    "<h2>2.1.\u00a0Method</h2>"
    "<p>We report <i>pass&#8209;at&#8209;one</i> on 53 tasks; the harness stays fixed.</p>"
    "<p>Second&nbsp;paragraph keeps a NBSP: 2.5e&#8722;6.</p>"
    "<p>Literal markup stays: &lt;p&gt;kept&lt;/p&gt;</p>"
    "<div><p>nested block</p></div>"
    "</body></html>"
)


def selftest() -> int:
    """反空转: 抽取器必须真认出 4 个块, 且必须**没**把 NBSP 变成空格"""
    lines = to_lines(SELFTEST_HTML)
    print(f"合成 HTML 抽出 {len(lines)} 行:")
    for i, l in enumerate(lines, 1):
        print(f"  {i}\t{l!r}")
    fails = []
    if len(lines) != 5:
        fails.append(f"应该认出 5 个块, 实得 {len(lines)}")
    if "\u00a0" not in lines[0]:
        fails.append("标题里的 U+00A0 被折叠了 —— 那就不再是「逐字」引用")
    if "\u00a0" not in lines[2]:
        fails.append("段落里的 &nbsp; 被压成普通空格")
    if "stub" in " ".join(lines):
        fails.append("<script> 里的内容没被剥掉")
    if "2.5e\u2212" not in lines[2]:
        fails.append("实体 &#8722; 未还原 ⇒ 先删标签后还原实体会丢字符")
    if "<p>kept</p>" not in lines[3]:
        fails.append("正文里字面写的 &lt;p&gt; 被当成真标签删掉了 (顺序错)")
    for f in fails:
        print("❌ " + f)
    print("✅ selftest 通过" if not fails else "⇒ 上面这些是抽取器自己的缺陷")
    return 1 if fails else 0


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", nargs="?")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.src or not args.dst:
        print("用法: arxiv_html_to_lines.py in.html out.txt   （或 --selftest）")
        return 2
    raw = Path(args.src).read_text(encoding="utf-8", errors="replace")
    lines = to_lines(raw)
    write_numbered(lines, args.dst)
    # 字节数与行数一起报: 它是那份「全文我确实取到了」的唯一凭据 (doc 55 §1.1)
    print(f"{args.src}: {len(raw.encode('utf-8')):,} B -> {args.dst}: {len(lines)} 行")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
