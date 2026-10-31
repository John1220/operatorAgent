# -*- coding: utf-8 -*-
"""`operator_agent/core/human_ledger.py` 的三条不变量，正反例都要有活的用例。

这个量具的全部风险都是同一个形状：**把「没问」显示成「问过」**。
所以每一组都有配对 —— 一条断言默认是 unknown，另一条断言显式写 `no` 才变成 `no`；
否则「全表 no」与「全表没填」在报表上长成一个样，那就是本仓反复吃过的那条
「全绿可能只证这条分支没人走」。
"""
import json
import os
import tempfile
import unittest

from operator_agent.core.human_ledger import (
    MINUTE_BANDS, STAGES, UNKNOWN, LedgerError, as_report_rows, counts, new_ledger,
    read_ledger, record, write_ledger)


class TestUnknownIsNotNo(unittest.TestCase):

    def test_an_empty_ledger_is_all_unknown_over_the_full_denominator(self):
        ledger = new_ledger("run-x")
        got = counts(ledger)
        self.assertEqual(got["total"], len(STAGES))
        self.assertEqual(got["answered"], 0, "新表一条都没问过，不能算已答")
        self.assertEqual(got["unknown"], len(STAGES))
        self.assertEqual(got["no"], 0, "没问 ≠ 没人做：no 必须由人显式写")
        for entry in ledger["entries"].values():
            self.assertEqual(entry["involved"], UNKNOWN)

    def test_recording_without_involved_keeps_the_cell_unknown(self):
        ledger = record(new_ledger("run-x"), "analyze", action="读了 L3 回执")
        entry = ledger["entries"]["analyze"]
        self.assertEqual(entry["action"], "读了 L3 回执")
        self.assertEqual(entry["involved"], UNKNOWN,
                         "只填动作不填「人是否介入」，仍是没问过")

    def test_explicit_no_is_counted_as_no(self):
        ledger = record(new_ledger("run-x"), "compare", involved="no")
        self.assertEqual(ledger["entries"]["compare"]["involved"], "no")
        got = counts(ledger)
        self.assertEqual((got["answered"], got["no"], got["unknown"]), (1, 1, 5))


class TestRefusalsTellHowToFix(unittest.TestCase):
    """拒绝消息要能自我纠正：说出合法取值，别让调用方猜。"""

    def _assert_refused_with_values(self, call, *needles):
        with self.assertRaises(LedgerError) as caught:
            call()
        message = str(caught.exception)
        for needle in needles:
            self.assertIn(needle, message)

    def test_unknown_stage_names_the_six(self):
        ledger = new_ledger("run-x")
        self._assert_refused_with_values(
            lambda: record(ledger, "integrate"), "integrate", "prepare", "accept")

    def test_unknown_involved_names_the_three(self):
        ledger = new_ledger("run-x")
        self._assert_refused_with_values(
            lambda: record(ledger, "prepare", involved="maybe"), "maybe", "yes", "no", UNKNOWN)

    def test_minutes_band_is_a_band_not_a_number(self):
        ledger = new_ledger("run-x")
        self._assert_refused_with_values(
            lambda: record(ledger, "prepare", involved="yes", minutes_band=25),
            "25", MINUTE_BANDS[1])

    def test_minutes_band_requires_yes(self):
        """否则「没问过」会被伪装成「问过且很快」。"""
        ledger = new_ledger("run-x")
        self._assert_refused_with_values(
            lambda: record(ledger, "analyze", minutes_band="5-15"),
            "requires involved='yes'", UNKNOWN)

    def test_write_refuses_a_foreign_schema(self):
        with tempfile.TemporaryDirectory() as root:
            path = root + "/human-ledger.json"
            with self.assertRaises(LedgerError) as caught:
                write_ledger(path, {"schema": "other", "entries": {}})
            self.assertIn("other", str(caught.exception))
            self.assertFalse(os.path.exists(path), "拒绝之后不该留下一半文件")

    def test_read_refuses_a_foreign_schema(self):
        with tempfile.TemporaryDirectory() as root:
            path = root + "/human-ledger.json"
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"schema": "other", "entries": {}}))
            with self.assertRaises(LedgerError) as caught:
                read_ledger(path)
            self.assertIn("other", str(caught.exception))

    def test_read_refuses_a_table_that_dropped_a_stage(self):
        """少一格就是有人在偷偷删 unknown，而不是「数据简洁」。"""
        with tempfile.TemporaryDirectory() as root:
            path = root + "/human-ledger.json"
            ledger = new_ledger("run-x")
            del ledger["entries"]["compare"]
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(ledger, handle)
            with self.assertRaises(LedgerError) as caught:
                read_ledger(path)
            self.assertIn("compare", str(caught.exception))


class TestReportKeepsUnitsApart(unittest.TestCase):

    def test_rows_have_two_separate_columns_and_no_total(self):
        ledger = record(new_ledger("run-x"), "check_run", involved="yes", minutes_band="15-60")
        rows = as_report_rows(ledger, {"check_run": 241.2, "accept": 96.05})
        self.assertEqual(len(rows), len(STAGES))
        for row in rows:
            self.assertNotIn("total", row)
            self.assertNotIn("sum", row)
            band, wall = row["human_minutes_band"], row["machine_wall_s"]
            self.assertTrue(band is None or isinstance(band, str), "量级档必须是字符串")
            self.assertTrue(wall is None or isinstance(wall, (int, float)), "秒必须是数值")
        by_stage = {row["stage"]: row for row in rows}
        self.assertEqual(by_stage["check_run"]["human_minutes_band"], "15-60")
        self.assertEqual(by_stage["check_run"]["machine_wall_s"], 241.2)
        self.assertIsNone(by_stage["prepare"]["human_minutes_band"],
                          "未答的环节两栏都得空着，不能填 0")
        self.assertIsNone(by_stage["prepare"]["machine_wall_s"])
        self.assertEqual(by_stage["accept"]["involved"], UNKNOWN,
                         "机器有秒数不等于人介入过")

    def test_no_machine_numbers_means_no_invented_zero(self):
        rows = as_report_rows(new_ledger("run-x"))
        self.assertTrue(all(row["machine_wall_s"] is None for row in rows))

    def test_boolean_machine_wall_is_refused(self):
        """bool 是 int 的子类 —— 不挡住就会把 True 当成 1 秒写进耗时列。"""
        with self.assertRaises(LedgerError):
            as_report_rows(new_ledger("run-x"), {"prepare": True})

    def test_non_numeric_machine_wall_is_refused(self):
        with self.assertRaises(LedgerError):
            as_report_rows(new_ledger("run-x"), {"prepare": "241.2"})


class TestRoundTrip(unittest.TestCase):

    def test_write_then_read_preserves_unknown_cells(self):
        with tempfile.TemporaryDirectory() as root:
            path = root + "/human-ledger.json"
            ledger = record(new_ledger("run-x"), "modify", involved="yes",
                            action="手写 3 个 tile 变体", reason="模板空间缺 stages=4 这一轴",
                            minutes_band="15-60")
            write_ledger(path, ledger)
            again = read_ledger(path)
            self.assertEqual(again, ledger)
            self.assertEqual(again["entries"]["modify"]["reason"], "模板空间缺 stages=4 这一轴")
            self.assertEqual(counts(again)["answered"], 1)
            self.assertEqual(counts(again)["unknown"], len(STAGES) - 1)


if __name__ == "__main__":
    unittest.main()
