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

import { appendJournal, configureJournalRunner, drainJournal, readJournal } from "../journal.mjs";

/** 一个假 CLI：把收到的参数、stdin 与环境探针记下来，再按给定回包作答。 */
function stub(t, body, { exitCode = 0, env } = {}) {
  const dir = mkdtempSync(join(tmpdir(), "journal-stub-"));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const log = join(dir, "calls.jsonl");
  const script = join(dir, "cli.mjs");
  writeFileSync(script, `
import { appendFileSync } from "node:fs";
const chunks = [];
for await (const chunk of process.stdin) chunks.push(chunk);
appendFileSync(${JSON.stringify(log)}, JSON.stringify({
  args: process.argv.slice(2), input: Buffer.concat(chunks).toString("utf8"),
  canary: process.env.OPERATOR_JOURNAL_CANARY ?? null,
  secret: process.env.SECRET_TOKEN ?? null }) + "\\n");
process.stdout.write(${JSON.stringify(JSON.stringify(body))});
process.exit(${exitCode});
`);
  configureJournalRunner({ command: process.execPath, args: [script],
    env: env ?? { PATH: process.env.PATH || "" } });
  t.after(() => configureJournalRunner({ command: "operator-agent", args: [],
    env: { PATH: process.env.PATH || "" } }));
  return () => readFileSync(log, "utf8").trim().split("\n").filter(Boolean).map(JSON.parse);
}

test("a submitted event goes through the CLI and returns the written one", async (t) => {
  const calls = stub(t, { ok: true, event: { rev: 7, kind: "task_closed" } });
  const receipt = await appendJournal("/tmp/run", { kind: "task_closed", outcome: "completed" });
  assert.equal(receipt.ok, true);
  assert.equal(receipt.event.rev, 7);
  const sent = calls();
  assert.equal(sent.length, 1);
  assert.deepEqual(sent[0].args, ["journal-append", "--run-dir", "/tmp/run"]);
  assert.equal(JSON.parse(sent[0].input).kind, "task_closed");
});

test("a refusal is a receipt (exit 2 still carries the structured answer)", async (t) => {
  stub(t, { ok: false, error: "journal_closed", detail: "任务已显式关闭" }, { exitCode: 2 });
  const receipt = await appendJournal("/tmp/run", { kind: "task_closed", outcome: "completed" });
  assert.equal(receipt.ok, false);
  assert.equal(receipt.error, "journal_closed");
});

test("a missing CLI is reported, not thrown", async (t) => {
  configureJournalRunner({ command: "/nonexistent/operator-agent", args: [] });
  t.after(() => configureJournalRunner({ command: "operator-agent", args: [] }));
  const receipt = await appendJournal("/tmp/run", { kind: "task_closed", outcome: "completed" });
  assert.equal(receipt.ok, false);
  assert.equal(receipt.error, "journal_unavailable");
});

test("no run directory means no submission at all", async (t) => {
  const calls = stub(t, { ok: true, event: { rev: 1 } });
  assert.equal((await appendJournal("", { kind: "task_closed" })).error, "journal_run_dir");
  assert.throws(() => calls(), /ENOENT/);      // 一次都没调过 CLI
});

test("same-task submissions stay ordered and drain waits for them", async t => {
  const calls = stub(t, { ok: true, event: { rev: 1 } });
  await Promise.all([
    appendJournal("/tmp/run", { kind: "budget_reserved" }),
    appendJournal("/tmp/run", { kind: "budget_settled" }),
  ]);
  await drainJournal();
  assert.deepEqual(calls().map(call => JSON.parse(call.input).kind),
    ["budget_reserved", "budget_settled"]);
});

test("journal subprocess receives only the configured environment", async t => {
  const calls = stub(t, { ok: true, event: { rev: 1 } }, {
    env: { PATH: "", OPERATOR_JOURNAL_CANARY: "yes" },
  });
  await appendJournal("/tmp/run", { kind: "budget_reserved" });
  assert.equal(calls()[0].canary, "yes");
  assert.equal(calls()[0].secret, null);
});

function journal(t, rows) {
  const dir = mkdtempSync(join(tmpdir(), "journal-read-"));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  writeFileSync(join(dir, "journal.jsonl"), rows.map(row => JSON.stringify({
    schema: "operator-task-journal-v2", task_id: "a".repeat(32), ...row })).join("\n") + "\n");
  return dir;
}

test("readJournal accepts a valid task and job sequence", t => {
  const dir = journal(t, [
    { rev: 1, kind: "task_prepared" },
    { rev: 2, kind: "budget_opened", budget: { host_task: { task_id: "a".repeat(32) } } },
    { rev: 3, kind: "task_activated" },
    { rev: 4, kind: "job_queued", job_id: "job", operation: "verify" },
    { rev: 5, kind: "job_started", job_id: "job", operation: "verify" },
    { rev: 6, kind: "job_finished", job_id: "job", operation: "verify",
      outcome: "completed", result: "received" },
  ]);
  assert.equal(readJournal(dir).length, 6);
});

test("readJournal keeps the journal task id distinct from the stable budget id", t => {
  const dir = journal(t, [
    { rev: 1, kind: "task_prepared" },
    { rev: 2, kind: "budget_opened", budget: { host_task: { task_id: "b".repeat(32) } } },
    { rev: 3, kind: "budget_reserved", budget: { host_task: { task_id: "b".repeat(32) } } },
  ]);
  assert.equal(readJournal(dir).length, 3);
});

test("readJournal rejects budget snapshots from different ledgers", t => {
  const dir = journal(t, [
    { rev: 1, kind: "task_prepared" },
    { rev: 2, kind: "budget_opened", budget: { host_task: { task_id: "b".repeat(32) } } },
    { rev: 3, kind: "budget_reserved", budget: { host_task: { task_id: "c".repeat(32) } } },
  ]);
  assert.throws(() => readJournal(dir), /journal_budget_mismatch/);
});

test("readJournal rejects a rev gap", t => {
  const dir = journal(t, [
    { rev: 1, kind: "task_prepared" },
    { rev: 3, kind: "budget_opened", budget: { host_task: { task_id: "a".repeat(32) } } },
  ]);
  assert.throws(() => readJournal(dir), /journal_gap/);
});

test("readJournal rejects mixed task identities", t => {
  const dir = journal(t, [
    { rev: 1, kind: "task_prepared" },
    { rev: 2, kind: "budget_opened", task_id: "b".repeat(32),
      budget: { host_task: { task_id: "b".repeat(32) } } },
  ]);
  assert.throws(() => readJournal(dir), /journal_task_mismatch|journal_budget_mismatch/);
});

test("readJournal rejects a job transition that bypasses queued", t => {
  const dir = journal(t, [
    { rev: 1, kind: "task_prepared" },
    { rev: 2, kind: "task_activated" },
    { rev: 3, kind: "job_started", job_id: "job", operation: "verify" },
  ]);
  assert.throws(() => readJournal(dir), /journal_job_unknown/);
});

test("readJournal rejects events after task_closed", t => {
  const dir = journal(t, [
    { rev: 1, kind: "task_prepared" },
    { rev: 2, kind: "task_closed", outcome: "completed" },
    { rev: 3, kind: "budget_reserved", budget: { model: {} } },
  ]);
  assert.throws(() => readJournal(dir), /journal_closed/);
});