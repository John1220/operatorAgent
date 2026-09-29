"""CHECKLOG 台账的结构守卫 (形制借自 LeetCUDA book/CHECKLOG.md)

守的不是内容正确性 (那要人读证据), 是**立条纪律**:
四件套 (原话/实证/出处/后果) + 类别 + 状态。缺一件的条目等于没挂号 ——
「据某某说」这种证据栏是我们自己在 doc 43/45/47 连栽三次的那个坑。
"""
import re
import unittest
from pathlib import Path

CHECKLOG = Path(__file__).resolve().parents[1] / "docs" / "research" / "CHECKLOG.md"

CATEGORIES = {"C", "I", "M", "P", "S", "H"}
STATUSES = {"✅", "⚠️", "🚫"}


def _entries(text: str) -> list:
    """按 `### ID ·` 切条目, 返回 (id, body)"""
    parts = re.split(r"\n### ", text)
    out = []
    for p in parts[1:]:
        head, _, body = p.partition("\n")
        m = re.match(r"([CIMPSH][0-9]+)\s*·", head)
        if not m:
            continue
        out.append((m.group(1).replace(" ", ""), head, body))
    return out


class TestChecklogStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = CHECKLOG.read_text(encoding="utf-8")
        cls.entries = _entries(cls.text)

    def test_file_exists_and_has_entries(self):
        self.assertTrue(CHECKLOG.exists())
        self.assertGreaterEqual(len(self.entries), 10, "台账条目不该只有零星几条")

    def test_every_entry_has_the_four_required_fields(self):
        for eid, head, body in self.entries:
            joined = head + body
            for field in ("原话", "实证", "出处", "后果"):
                self.assertIn(field, joined, f"{eid} 缺「{field}」—— 四件套不齐")

    def test_every_entry_has_category_and_status(self):
        for eid, head, body in self.entries:
            joined = head + body
            self.assertRegex(joined, r"类别", f"{eid} 没写类别")
            self.assertTrue(any(s in joined for s in STATUSES),
                            f"{eid} 状态不是 ✅/⚠️/🚫 之一")

    def test_evidence_is_not_a_hearsay_only(self):
        """证据栏不许只有「据某某说」—— 必须带文件:行 / URL / commit / 实测数字"""
        for eid, head, body in self.entries:
            m = re.search(r"\*\*实证\*\*(.+?)(?:\n\*\*|\Z)", head + body, re.S)
            self.assertIsNotNone(m, f"{eid} 找不到实证段")
            ev = m.group(1)
            self.assertTrue(
                re.search(r"\.py|:\d|arXiv|http|doc \d|commit|[0-9]+\.[0-9]|%\b|×",
                          ev),
                f"{eid} 实证没有可核对的锚点: {ev[:80]!r}")

    def test_index_lists_every_id_exactly_once(self):
        body = self.text.split("## 索引")[-1]
        idx_ids = re.findall(r"^\|\s*([CIMP SH]\d+)\s*\|", body, re.M)
        entry_ids = [e[0] for e in self.entries]
        self.assertEqual(sorted(idx_ids), sorted(set(entry_ids)),
                         f"索引与正文对不上: idx={idx_ids} body={entry_ids}")
        self.assertEqual(len(idx_ids), len(set(idx_ids)), "索引里有重复行")

    def test_statuses_are_consistent_between_index_and_body(self):
        body = self.text.split("## 索引")[-1]
        for eid, head, b in self.entries:
            row = next((ln for ln in body.splitlines()
                        if re.match(rf"^\|\s*{eid}\s*\|", ln)), None)
            self.assertIsNotNone(row, f"索引里没有 {eid}")
            # 状态是多码点 emoji (⚠️ = U+26A0 U+FE0F), 按字符类匹配会漏
            idx_hits = [s for s in STATUSES if s in row]
            self.assertEqual(1, len(idx_hits), f"{eid} 索引行状态不明确: {row}")
            self.assertIn(idx_hits[0], head + b,
                          f"{eid} 索引状态 {idx_hits[0]} 与正文不一致")

    def test_frozen_sources_are_not_told_to_be_rewritten(self):
        """使用约定必须写明「不回写冻结源」—— 否则后人会去改上游文档"""
        self.assertIn("不回写", self.text)
        self.assertIn("冻结源", self.text)


if __name__ == "__main__":
    unittest.main()
