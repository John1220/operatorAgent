/**
 * 插件侧的账本入口：**只提交，不分配、不落盘**。
 *
 * 守卫三件事：
 * 1. 提交形状正确（`journal-append --run-dir`，事件走 stdin）；
 * 2. 账本拒绝是**回执**，不是异常 —— 模型预算门不该因为一条事件写不进去就崩；
 * 3. CLI 不在也只是回执（`journal_unavailable`），调用方自己决定要不要报。
 */
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { appendJournal, configureJournalRunner } from "../journal.mjs";

/** 一个假 CLI：把收到的参数与 stdin 记下来，再按给定回包作答。 */
function stub(t, body, { exitCode = 0 } = {}) {
  const dir = mkdtempSync(join(tmpdir(), "journal-stub-"));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const log = join(dir, "calls.jsonl");
  const script = join(dir, "cli.mjs");
  writeFileSync(script, `
import { appendFileSync } from "node:fs";
const chunks = [];
for await (const chunk of process.stdin) chunks.push(chunk);
appendFileSync(${JSON.stringify(log)}, JSON.stringify({
  args: process.argv.slice(2), input: Buffer.concat(chunks).toString("utf8") }) + "\\n");
process.stdout.write(${JSON.stringify(JSON.stringify(body))});
process.exit(${exitCode});
`);
  configureJournalRunner({ command: process.execPath, args: [script] });
  t.after(() => configureJournalRunner({ command: "operator-agent", args: [] }));
  return () => readFileSync(log, "utf8").trim().split("\n").filter(Boolean).map(JSON.parse);
}

test("a submitted event goes through the CLI and returns the written one", (t) => {
  const calls = stub(t, { ok: true, event: { rev: 7, kind: "task_closed" } });
  const receipt = appendJournal("/tmp/run", { kind: "task_closed", outcome: "completed" });
  assert.equal(receipt.ok, true);
  assert.equal(receipt.event.rev, 7);
  const sent = calls();
  assert.equal(sent.length, 1);
  assert.deepEqual(sent[0].args, ["journal-append", "--run-dir", "/tmp/run"]);
  assert.equal(JSON.parse(sent[0].input).kind, "task_closed");
});

test("a refusal is a receipt (exit 2 still carries the structured answer)", (t) => {
  stub(t, { ok: false, error: "journal_closed", detail: "任务已显式关闭" }, { exitCode: 2 });
  const receipt = appendJournal("/tmp/run", { kind: "task_closed", outcome: "completed" });
  assert.equal(receipt.ok, false);
  assert.equal(receipt.error, "journal_closed");
});

test("a missing CLI is reported, not thrown", (t) => {
  configureJournalRunner({ command: "/nonexistent/operator-agent", args: [] });
  t.after(() => configureJournalRunner({ command: "operator-agent", args: [] }));
  const receipt = appendJournal("/tmp/run", { kind: "task_closed", outcome: "completed" });
  assert.equal(receipt.ok, false);
  assert.equal(receipt.error, "journal_unavailable");
});

test("no run directory means no submission at all", (t) => {
  const calls = stub(t, { ok: true, event: { rev: 1 } });
  assert.equal(appendJournal("", { kind: "task_closed" }).error, "journal_run_dir");
  assert.throws(() => calls(), /ENOENT/);      // 一次都没调过 CLI
});