# -*- coding: utf-8 -*-
"""`operator-agent ledger-new|fill|report` 的用例。

CLI 是这张表唯一的填写入口（量具刻意没接进 MCP 工具回包），所以它必须守住表本身的
三条不变量：没问过 ≠ 没人做、分母恒为六个环节、人自报档与机器秒不合算。
"""
import io
import json
import contextlib
import tempfile
import unittest
from pathlib import Path

from operator_agent.app import cli
from operator_agent.core import human_ledger


def run(argv):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = cli.main(argv)
    return code, buffer.getvalue()


class TestLedgerNew(unittest.TestCase):

    def test_new_writes_six_unknown_cells(self):
        with tempfile.TemporaryDirectory() as root:
            code, out = run(["ledger-new", "--run-dir", root])
            self.assertEqual(code, 0)
            ledger = human_ledger.read_ledger(Path(root) / human_ledger.LEDGER_FILENAME)
            self.assertEqual(ledger["run_id"], Path(root).name)
            self.assertEqual(sorted(ledger["entries"]), sorted(human_ledger.STAGES))
            for entry in ledger["entries"].values():
                self.assertEqual(entry["involved"], human_ledger.UNKNOWN)

    def test_new_refuses_to_clobber_a_filled_table(self):
        with tempfile.TemporaryDirectory() as root:
            run(["ledger-new", "--run-dir", root])
            run(["ledger-fill", "--run-dir", root, "--stage", "analyze",
                 "--involved", "yes", "--reason", "门槛在 ±3% 刀刃上"])
            code, out = run(["ledger-new", "--run-dir", root])
            self.assertEqual(code, 2, "覆盖已答的表必须是拒绝，不是静默重来")
            again = human_ledger.read_ledger(Path(root) / human_ledger.LEDGER_FILENAME)
            self.assertEqual(again["entries"]["analyze"]["involved"], "yes",
                             "被拒之后已答的格子必须原样还在")


class TestLedgerFill(unittest.TestCase):

    def test_fill_records_the_tuple_and_counts(self):
        with tempfile.TemporaryDirectory() as root:
            run(["ledger-new", "--run-dir", root])
            code, _ = run(["ledger-fill", "--run-dir", root, "--stage", "modify",
                           "--involved", "yes", "--action-taken", "手写 3 个 tile 变体",
                           "--reason", "模板空间缺 stages=4 这一轴",
                           "--minutes-band", "15-60"])
            self.assertEqual(code, 0)
            entry = human_ledger.read_ledger(
                Path(root) / human_ledger.LEDGER_FILENAME)["entries"]["modify"]
            self.assertEqual(entry["action"], "手写 3 个 tile 变体")
            self.assertEqual(entry["minutes_band"], "15-60")

    def test_unknown_stage_is_refused_with_the_accepted_list(self):
        with tempfile.TemporaryDirectory() as root:
            run(["ledger-new", "--run-dir", root])
            code, out = run(["ledger-fill", "--run-dir", root, "--stage", "integrate",
                             "--involved", "yes"])
            self.assertEqual(code, 2)
            payload = json.loads(out)
            self.assertEqual(payload["error"], "ledger_refused")
            for stage in human_ledger.STAGES:
                self.assertIn(stage, payload["detail"], "拒绝消息要能自我纠正")

    def test_band_without_involved_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            run(["ledger-new", "--run-dir", root])
            code, out = run(["ledger-fill", "--run-dir", root, "--stage", "compare",
                             "--minutes-band", "5-15"])
            self.assertEqual(code, 2)
            self.assertIn("requires involved='yes'", json.loads(out)["detail"])


class TestLedgerReport(unittest.TestCase):

    def test_two_columns_and_no_folded_total(self):
        with tempfile.TemporaryDirectory() as root:
            run(["ledger-new", "--run-dir", root])
            run(["ledger-fill", "--run-dir", root, "--stage", "check_run",
                 "--involved", "yes", "--minutes-band", "15-60"])
            wall = Path(root) / "wall.json"
            wall.write_text(json.dumps({"check_run": 241.2}), encoding="utf-8")
            code, out = run(["ledger-report", "--run-dir", root,
                             "--machine-wall-json", str(wall)])
            self.assertEqual(code, 0)
            payload = json.loads(out)
            self.assertEqual(len(payload["rows"]), len(human_ledger.STAGES))
            self.assertEqual(payload["counts"]["total"], len(human_ledger.STAGES))
            self.assertEqual(payload["counts"]["unknown"], len(human_ledger.STAGES) - 1)
            for key in ("total_seconds", "total_minutes", "sum"):
                self.assertNotIn(key, payload, "两栏不许被折成一个合计")
            for row in payload["rows"]:
                self.assertIsInstance(row["human_minutes_band"], (str, type(None)))
                self.assertIsInstance(row["machine_wall_s"], (int, float, type(None)))
            row = [r for r in payload["rows"] if r["stage"] == "check_run"][0]
            self.assertEqual(row["human_minutes_band"], "15-60")
            self.assertEqual(row["machine_wall_s"], 241.2)

    def test_machine_wall_truth_does_not_imply_human_presence(self):
        with tempfile.TemporaryDirectory() as root:
            run(["ledger-new", "--run-dir", root])
            wall = Path(root) / "wall.json"
            wall.write_text(json.dumps({"accept": 96.05}), encoding="utf-8")
            _, out = run(["ledger-report", "--run-dir", root,
                          "--machine-wall-json", str(wall)])
            row = [r for r in json.loads(out)["rows"] if r["stage"] == "accept"][0]
            self.assertEqual(row["involved"], human_ledger.UNKNOWN,
                             "机器有秒数不等于人介入过")

    def test_non_numeric_machine_wall_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            run(["ledger-new", "--run-dir", root])
            wall = Path(root) / "wall.json"
            wall.write_text(json.dumps({"accept": "96 seconds"}), encoding="utf-8")
            code, out = run(["ledger-report", "--run-dir", root,
                             "--machine-wall-json", str(wall)])
            self.assertEqual(code, 2)
            self.assertIn("ledger_refused", json.loads(out)["error"])


if __name__ == "__main__":
    unittest.main()
