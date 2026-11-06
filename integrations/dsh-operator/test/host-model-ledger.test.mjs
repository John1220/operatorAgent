import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { spawn, spawnSync } from "node:child_process";
import { once } from "node:events";
import {
  chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync,
  symlinkSync, unlinkSync, writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import {
  HostModelBudgetError, HostModelLedger, MODEL_BUDGET_FILE, MODEL_BUDGET_SCHEMA,
} from "../host-model-ledger.mjs";

const TASK = "a".repeat(32);
const CONTRACT = "b".repeat(64);
const MODULE = new URL("../host-model-ledger.mjs", import.meta.url).href;
function canonical(value) {
  if (Array.isArray(value)) return "[" + value.map(canonical).join(",") + "]";
  if (value && typeof value === "object") return "{" + Object.keys(value).sort()
    .map(key => JSON.stringify(key) + ":" + canonical(value[key])).join(",") + "}";
  return JSON.stringify(value);
}
const hash = text => createHash("sha256").update(text).digest("hex");
const code = expected => error => error instanceof HostModelBudgetError && error.code === expected;

function fixture(t, overrides = {}) {
  const dir = mkdtempSync(join(tmpdir(), "operator-host-model-"));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const policy = { schema: "operator-host-task-budget-v1", contract_hash: CONTRACT,
    limits: { wall_clock_s: 600, max_verifications: 6, final_reserve: 1,
      max_model_calls: 3, max_output_tokens_per_call: 64, total_output_token_reservations: 192,
      ...overrides }, required_case_ids: ["fixture-case"] };
  const policyText = canonical(policy);
  const started = Date.now() / 1000;
  writeFileSync(join(dir, "host-task-policy.json"), policyText + "\n", { mode: 0o400 });
  writeFileSync(join(dir, "host-task.json"), JSON.stringify({ task_id: TASK,
    policy_hash: hash(policyText), records: [], started_epoch: started,
    started_monotonic: 0, max_elapsed_s: 0 }) + "\n", { mode: 0o600 });
  const rawLedger = () => new HostModelLedger(dir, { taskId: TASK, contractHash: CONTRACT });
  const ledger = () => {
    const value = rawLedger();
    const bind = value.bind.bind(value);
    value.bind = (...args) => {
      bind(...args);
      value.clock({ remainingSeconds: Math.max(0, policy.limits.wall_clock_s - (Date.now()/1000 - started)),
        deadlineEpoch: started + policy.limits.wall_clock_s });
      return value.summary();
    };
    return value;
  };
  return { dir, policy, ledger, rawLedger, file: join(dir, MODEL_BUDGET_FILE), started };
}

function reserve(ledger, lease, requestId, maxTokens = 64) {
  return ledger.reserve(lease.session_id, lease.id,
    { requestId, maxTokens, provider: "fixture", model: "fixture-model" });
}

function clockSetup(dir) {
  return `
    const originalBind=ledger.bind.bind(ledger);
    ledger.bind=(...args)=>{
      originalBind(...args);
      const host=JSON.parse(readFileSync(${JSON.stringify(join(dir, "host-task.json"))},'utf8'));
      const policy=JSON.parse(readFileSync(${JSON.stringify(join(dir, "host-task-policy.json"))},'utf8'));
      ledger.clock({remainingSeconds:Math.max(0,host.started_epoch+policy.limits.wall_clock_s-Date.now()/1000),
        deadlineEpoch:host.started_epoch+policy.limits.wall_clock_s});
      return ledger.summary();
    };
  `;
}

function child(dir, body) {
  return spawnSync(process.execPath, ["--input-type=module", "-e", `
    import { HostModelLedger } from ${JSON.stringify(MODULE)};
    import { readFileSync } from "node:fs";
    const ledger=new HostModelLedger(${JSON.stringify(dir)}, {taskId:${JSON.stringify(TASK)},contractHash:${JSON.stringify(CONTRACT)}});
    ${clockSetup(dir)}
    ${body}
  `], { encoding: "utf8", timeout: 10_000 });
}

test("binds prepared identity and records pre-dispatch credit, never the preparation call", t => {
  const { dir, ledger } = fixture(t);
  const budget = ledger();
  const bound = budget.bind("session-1");
  assert.equal(bound.schema, MODEL_BUDGET_SCHEMA);
  assert.equal(bound.task_id, TASK);
  assert.equal(bound.preparation_dispatch, "excluded_before_successful_prepare");
  assert.equal(bound.calls_reserved, 0);
  assert.equal(bound.remaining_calls, 3);
  assert.equal(bound.policy_bytes_sha256, hash(readFileSync(join(dir, "host-task-policy.json"))));
  const call = reserve(budget, bound.lease, "request-1");
  assert.equal(call.status, "reserved");
  assert.equal(budget.summary().unfinished, 1);
  const ended = budget.finish("session-1", bound.lease.id, call.id, {
    usage: { inputTokens: 10, outputTokens: 5, totalTokens: 23, cacheReadTokens: 8 },
  });
  assert.equal(ended.calls_reserved, 1);
  assert.equal(ended.output_tokens_reserved, 64);
  assert.equal(ended.remaining_output_token_reservations, 128);
  assert.equal(ended.api_usage_known_calls, 1);
  assert.equal(ended.provider_execution_stop, "unknown");
});

test("failure, cancellation and smaller actual output never refund reservations", t => {
  const { ledger } = fixture(t);
  const budget = ledger();
  const { lease } = budget.bind("session-1");
  for (const [index, status] of ["failed", "cancelled", "answered"].entries()) {
    const call = reserve(budget, lease, "request-" + index);
    budget.finish(lease.session_id, lease.id, call.id, { status,
      usage: status === "answered" ? { inputTokens: 1, outputTokens: 1, totalTokens: 2 } : null });
  }
  const summary = budget.summary();
  assert.equal(summary.calls_reserved, 3);
  assert.equal(summary.output_tokens_reserved, 192);
  assert.equal(summary.api_usage_known_calls, 1);
  assert.equal(summary.api_usage_unknown_calls, 2);
  assert.throws(() => reserve(budget, lease, "request-over"), code("host_model_call_budget"));
});

test("per-call and total output limits are checked before any new credit is spent", t => {
  const { ledger } = fixture(t, { total_output_token_reservations: 96 });
  const budget = ledger();
  const { lease } = budget.bind("s");
  assert.throws(() => reserve(budget, lease, "too-large", 65), code("host_model_output_limit"));
  assert.equal(budget.summary().calls_reserved, 0);
  const first = reserve(budget, lease, "first");
  budget.finish("s", lease.id, first.id, { status: "failed" });
  assert.throws(() => reserve(budget, lease, "over-total"), code("host_model_token_budget"));
  reserve(budget, lease, "remaining", 32);
  assert.equal(budget.summary().output_tokens_reserved, 96);
  assert.throws(() => reserve(budget, lease, "one-more", 1), code("host_model_token_budget"));
  assert.equal(budget.summary().calls_reserved, 2);
});

test("reopening and repeated binding cannot recreate credit", t => {
  const { ledger } = fixture(t);
  const first = ledger();
  const { lease } = first.bind("s");
  const call = reserve(first, lease, "one");
  first.finish("s", lease.id, call.id, { status: "failed" });
  const again = ledger().bind("s");
  assert.equal(again.lease.id, lease.id);
  assert.equal(again.calls_reserved, 1);
  assert.equal(again.remaining_calls, 2);
  assert.equal(again.output_tokens_reserved, 64);
});

test("session handoff requires explicit reattachment and invalidates the old lease", t => {
  const { ledger } = fixture(t);
  const budget = ledger();
  const { lease } = budget.bind("first");
  const one = reserve(budget, lease, "one");
  budget.finish("first", lease.id, one.id);
  assert.throws(() => budget.bind("second"), code("host_model_reattach_required"));
  const resumed = budget.bind("second", { reattach: true });
  assert.equal(resumed.calls_reserved, 1);
  assert.equal(resumed.lease.generation, 2);
  assert.notEqual(resumed.lease.id, lease.id);
  assert.throws(() => reserve(budget, lease, "old"), code("host_model_lease"));
  const two = reserve(budget, resumed.lease, "two");
  assert.throws(() => budget.finish("first", lease.id, two.id), code("host_model_reservation"));
  assert.throws(() => budget.bind("third", { reattach: true }), code("host_model_in_flight"));
});

test("duplicate dispatch ids and duplicate completion cannot spend or rewrite another record", t => {
  const { ledger } = fixture(t);
  const budget = ledger();
  const { lease } = budget.bind("s");
  const call = reserve(budget, lease, "one");
  assert.throws(() => reserve(budget, lease, "one"), code("host_model_duplicate"));
  assert.equal(budget.summary().calls_reserved, 1);
  budget.finish("s", lease.id, call.id);
  assert.throws(() => budget.finish("s", lease.id, call.id, { status: "failed" }), code("host_model_reservation"));
  assert.equal(budget.summary().records[0].status, "answered");
});

test("a different process cannot silently use the persisted old dispatch lease", t => {
  const { dir, ledger } = fixture(t);
  const { lease } = ledger().bind("s");
  const result = child(dir, `
    try { ledger.reserve("s",${JSON.stringify(lease.id)}, {requestId:"foreign",maxTokens:64,provider:"fixture",model:"fixture-model"}); }
    catch(error) { console.log(error.code); }
  `);
  assert.equal(result.status, 0, result.stderr);
  assert.equal(result.stdout.trim(), "host_model_lease");
  assert.equal(ledger().summary().calls_reserved, 0);
});

test("completed calls survive process restart and session reattachment", t => {
  const { dir, ledger } = fixture(t, { max_model_calls: 2 });
  const result = child(dir, `
    const {lease}=ledger.bind("before-restart");
    const call=ledger.reserve(lease.session_id,lease.id,{requestId:"one",maxTokens:64,provider:"fixture",model:"fixture-model"});
    ledger.finish(lease.session_id,lease.id,call.id,{status:"failed"});
  `);
  assert.equal(result.status, 0, result.stderr);
  const after = ledger();
  assert.throws(() => after.bind("after-restart"), code("host_model_reattach_required"));
  const bound = after.bind("after-restart", { reattach: true });
  assert.equal(bound.calls_reserved, 1);
  reserve(after, bound.lease, "two");
  assert.throws(() => reserve(after, bound.lease, "three"), code("host_model_call_budget"));
});

test("a dead owner can be explicitly abandoned without refunds or invented usage", t => {
  const { dir, ledger } = fixture(t);
  const result = child(dir, `
    const {lease}=ledger.bind("dead-session");
    ledger.reserve(lease.session_id,lease.id,{requestId:"unfinished",maxTokens:64,provider:"fixture",model:"fixture-model"});
    console.log(JSON.stringify(lease));
  `);
  assert.equal(result.status, 0, result.stderr);
  const lease = JSON.parse(result.stdout);
  const after = ledger();
  assert.throws(() => after.bind("new-session", { reattach: true }), code("host_model_in_flight"));
  assert.throws(() => after.abandon(lease.session_id, lease.id, { checked_dead_pid: lease.owner_pid + 1 }), code("host_model_owner_unknown"));
  const abandoned = after.abandon(lease.session_id, lease.id, { checked_dead_pid: lease.owner_pid });
  assert.equal(abandoned.records[0].status, "abandoned");
  assert.equal(abandoned.records[0].usage, null);
  assert.equal(abandoned.calls_reserved, 1);
  assert.equal(abandoned.output_tokens_reserved, 64);
  assert.equal(abandoned.api_usage_unknown_calls, 1);
  assert.equal(abandoned.abandonment_events[0].checked_dead_pid, lease.owner_pid);
  assert.throws(() => after.finish(lease.session_id, lease.id, 1), code("host_model_reservation"));
  const rebound = after.bind("new-session", { reattach: true });
  assert.equal(rebound.remaining_calls, 2);
});

test("a live owner, including one in another process, cannot be declared abandoned", async t => {
  const { dir, ledger } = fixture(t);
  const script = `import {HostModelLedger} from ${JSON.stringify(MODULE)};
    import {readFileSync} from "node:fs";
    const ledger=new HostModelLedger(${JSON.stringify(dir)},{taskId:${JSON.stringify(TASK)},contractHash:${JSON.stringify(CONTRACT)}});
    ${clockSetup(dir)}
    const {lease}=ledger.bind("alive-session");
    ledger.reserve(lease.session_id,lease.id,{requestId:"live",maxTokens:64,provider:"fixture",model:"fixture-model"});
    console.log(JSON.stringify(lease));setInterval(()=>{},1000);`;
  const process = spawn(globalThis.process.execPath, ["--input-type=module", "-e", script], { stdio: ["ignore", "pipe", "pipe"] });
  t.after(() => process.kill("SIGTERM"));
  const [line] = await once(process.stdout, "data");
  const lease = JSON.parse(String(line));
  assert.throws(() => ledger().abandon(lease.session_id, lease.id,
    { checked_dead_pid: lease.owner_pid }), code("host_model_owner_unknown"));
  assert.equal(ledger().summary().unfinished, 1);
  process.kill("SIGTERM");
  await once(process, "exit");
});

test("stale or concurrent locks fail closed and are never automatically cleared", t => {
  const { dir, ledger } = fixture(t);
  const lock = join(dir, ".host-model-budget.lock");
  mkdirSync(lock, { mode: 0o700 });
  writeFileSync(join(lock, "owner.json"), canonical({ pid: 99999999, nonce: "stale-fixture" }) + "\n", { mode: 0o600 });
  assert.throws(() => ledger().bind("s"), code("host_model_locked"));
  assert.ok(existsSync(lock));
  assert.ok(!existsSync(join(dir, MODEL_BUDGET_FILE)));
});

test("old policies without model limits are unsupported rather than granted defaults", t => {
  const { dir, policy, ledger } = fixture(t);
  delete policy.limits.max_model_calls;
  const text = canonical(policy);
  chmodSync(join(dir, "host-task-policy.json"), 0o600);
  writeFileSync(join(dir, "host-task-policy.json"), text + "\n");
  writeFileSync(join(dir, "host-task.json"), JSON.stringify({ task_id: TASK, policy_hash: hash(text) }));
  assert.throws(() => ledger().bind("s"), code("host_model_policy_missing"));
  assert.ok(!existsSync(join(dir, MODEL_BUDGET_FILE)));
});

test("Python canonical policy float spellings verify without changing the opaque policy hash", t => {
  const { dir, policy, ledger } = fixture(t);
  const output = spawnSync("python3", ["-c", `import hashlib,json,sys
value=json.loads(sys.argv[1]);value['limits']['wall_clock_s']=600.0
value['required_case_ids']=['中文材料']
text=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))
print(json.dumps({'text':text,'hash':hashlib.sha256(text.encode()).hexdigest()}))`, JSON.stringify(policy)], { encoding: "utf8" });
  assert.equal(output.status, 0, output.stderr);
  const written = JSON.parse(output.stdout);
  chmodSync(join(dir, "host-task-policy.json"), 0o600);
  writeFileSync(join(dir, "host-task-policy.json"), written.text + "\n");
  writeFileSync(join(dir, "host-task.json"), JSON.stringify({
    ...JSON.parse(readFileSync(join(dir, "host-task.json"))), task_id: TASK, policy_hash: written.hash }));
  assert.equal(ledger().bind("s").policy_hash, written.hash);
});

test("policy or task identity drift never silently starts a new ledger", t => {
  const { dir, ledger, policy } = fixture(t);
  const budget = ledger();
  const { lease } = budget.bind("s");
  reserve(budget, lease, "one");
  assert.throws(() => new HostModelLedger(dir, { taskId: "f".repeat(32), contractHash: CONTRACT }).summary(), code("host_model_binding"));
  const text = canonical({ ...policy, limits: { ...policy.limits, max_model_calls: 100 } });
  chmodSync(join(dir, "host-task-policy.json"), 0o600);
  writeFileSync(join(dir, "host-task-policy.json"), text + "\n");
  assert.throws(() => budget.summary(), code("host_model_binding"));
  const raw = JSON.parse(readFileSync(join(dir, MODEL_BUDGET_FILE)));
  assert.equal(raw.records.length, 1);
});

test("ledger checksum and canonical bytes reject resets and duplicate JSON fields", t => {
  const { ledger, file } = fixture(t);
  const budget = ledger();
  const { lease } = budget.bind("s");
  reserve(budget, lease, "one");
  const saved = readFileSync(file, "utf8");
  const changed = JSON.parse(saved);
  changed.records = [];
  writeFileSync(file, canonical(changed) + "\n");
  assert.throws(() => budget.summary(), code("host_model_binding"));
  writeFileSync(file, saved.replace('"schema":', '"schema":"ignored-duplicate","schema":'));
  assert.throws(() => budget.summary(), code("host_model_journal"));
});

test("invalid existing journals are never overwritten or reset", t => {
  const { ledger, file } = fixture(t);
  writeFileSync(file, "{incomplete", { mode: 0o600 });
  assert.throws(() => ledger().bind("s"), code("host_model_journal"));
  assert.equal(readFileSync(file, "utf8"), "{incomplete");
});

test("symlink policies, task directories and nonprivate ledgers are rejected", t => {
  const { dir, ledger, file } = fixture(t);
  const budget = ledger();
  budget.bind("s");
  chmodSync(file, 0o644);
  assert.throws(() => budget.summary(), code("host_model_journal"));
  chmodSync(file, 0o600);
  const alias = dir + "-alias";
  symlinkSync(dir, alias);
  t.after(() => unlinkSync(alias));
  assert.throws(() => new HostModelLedger(alias, { taskId: TASK, contractHash: CONTRACT }), code("host_model_journal"));
  const policyPath = join(dir, "host-task-policy.json");
  const policyText = readFileSync(policyPath);
  unlinkSync(policyPath);
  writeFileSync(join(dir, "actual-policy.json"), policyText);
  symlinkSync(join(dir, "actual-policy.json"), policyPath);
  assert.throws(() => budget.summary(), code("host_model_journal"));
});

test("unobserved, zero-default or malformed provider counters remain unknown", t => {
  const { ledger } = fixture(t);
  const budget = ledger();
  const { lease } = budget.bind("s");
  for (const [index, usage] of [null, { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
    { inputTokens: 1, outputTokens: -1, totalTokens: 0 }].entries()) {
    const call = reserve(budget, lease, "call-" + index);
    budget.finish("s", lease.id, call.id, { usage });
  }
  const summary = budget.summary();
  assert.equal(summary.api_usage_known_calls, 0);
  assert.equal(summary.api_usage_unknown_calls, 3);
  assert.equal(summary.output_tokens_reserved, 192);
});

test("observed provider output cap violations stop further dispatch without claiming provider shutdown", t => {
  const { ledger } = fixture(t);
  const budget = ledger();
  const { lease } = budget.bind("s");
  const call = reserve(budget, lease, "one");
  const summary = budget.finish("s", lease.id, call.id, {
    usage: { inputTokens: 1, outputTokens: 65, totalTokens: 66 },
  });
  assert.equal(summary.provider_output_cap_violation, true);
  assert.equal(summary.provider_execution_stop, "unknown");
  assert.throws(() => reserve(budget, lease, "two"), code("host_model_provider_cap"));
  assert.equal(budget.summary().calls_reserved, 1);
});

test("missing persisted clock cannot be restored or silently recalibrated from a transcript", t => {
  const { rawLedger, started } = fixture(t);
  const budget = rawLedger();
  const bound = budget.bind("s");
  assert.throws(() => budget.clock({ remainingSeconds: 600, deadlineEpoch: started + 600,
    restoring: true }), code("host_model_clock_missing"));
  assert.throws(() => reserve(budget, bound.lease, "before-clock"), code("host_model_clock"));
  assert.equal(budget.summary().calls_reserved, 0);
  budget.clock({ remainingSeconds: 600, deadlineEpoch: started + 600 });
  reserve(budget, bound.lease, "after-clock");
  assert.equal(budget.summary().calls_reserved, 1);
});

test("wall clock rollback cannot extend a task past its persistent host monotonic deadline", t => {
  const { ledger } = fixture(t);
  const budget = ledger();
  const { lease } = budget.bind("s");
  const now = Date.now();
  const later = process.hrtime.bigint() + 601000000000n;
  t.mock.method(Date, "now", () => now - 60000);
  t.mock.method(process.hrtime, "bigint", () => later);
  assert.equal(budget.clock().remaining_ms, 0);
  assert.throws(() => reserve(budget, lease, "expired"), code("host_task_deadline"));
  assert.equal(budget.summary().calls_reserved, 0);
});

test("an observed wall-clock expiry cannot be revived by a later clock correction", t => {
  const { ledger } = fixture(t);
  const budget = ledger();
  const { lease } = budget.bind("s");
  const now = Date.now();
  const wall = t.mock.method(Date, "now", () => now + 601000);
  assert.equal(budget.clock().remaining_ms, 0);
  wall.mock.restore();
  assert.equal(budget.clock().remaining_ms, 0);
  assert.throws(() => reserve(budget, lease, "revived"), code("host_task_deadline"));
});

test("session reattachment retains the old monotonic anchor and consumed time", t => {
  const { rawLedger, started } = fixture(t);
  const budget = rawLedger();
  budget.bind("old");
  budget.clock({ remainingSeconds: 30, deadlineEpoch: started + 600 });
  const anchor = budget.summary().clock.anchor_hrtime_ns;
  const newer = process.hrtime.bigint() + 31000000000n;
  t.mock.method(process.hrtime, "bigint", () => newer);
  assert.equal(budget.clock().remaining_ms, 0);
  budget.bind("new", { reattach: true });
  assert.equal(budget.clock({ remainingSeconds: 600, deadlineEpoch: started + 600,
    restoring: true }).remaining_ms, 0);
  assert.equal(budget.summary().clock.anchor_hrtime_ns, anchor);
});

test("kernel boot identity or monotonic regression prevents a time-epoch guess", t => {
  const { ledger, file } = fixture(t);
  const budget = ledger();
  budget.bind("s");
  const saved = readFileSync(file, "utf8");
  const body = JSON.parse(saved);
  body.clock.boot_id = body.clock.boot_id === "0".repeat(36) ? "1".repeat(36) : "0".repeat(36);
  delete body.state_hash;
  writeFileSync(file, canonical({ ...body, state_hash: hash(canonical(body)) }) + "\n");
  assert.throws(() => budget.clock(), code("host_model_clock"));
  writeFileSync(file, saved);
  const anchor = BigInt(JSON.parse(saved).clock.anchor_hrtime_ns);
  t.mock.method(process.hrtime, "bigint", () => anchor - 1n);
  assert.throws(() => budget.clock(), code("host_model_clock"));
});

test("live Python elapsed high-water mark may shorten time but never restore it", t => {
  const { dir, ledger } = fixture(t);
  const budget = ledger();
  budget.bind("s");
  const path = join(dir, "host-task.json");
  const state = JSON.parse(readFileSync(path, "utf8"));
  writeFileSync(path, JSON.stringify({ ...state, max_elapsed_s: 601 }));
  assert.equal(budget.clock().remaining_ms, 0);
  writeFileSync(path, JSON.stringify(state));
  assert.equal(budget.clock().remaining_ms, 0);
  writeFileSync(path, JSON.stringify({ ...state, started_epoch: state.started_epoch + 1 }));
  assert.throws(() => budget.clock(), code("host_model_clock"));
});

test("a fresh proof cannot replace the frozen deadline or grant more than the task wall limit", t => {
  const { rawLedger, started } = fixture(t);
  const budget = rawLedger();
  budget.bind("s");
  assert.throws(() => budget.clock({ remainingSeconds: 600, deadlineEpoch: started + 1200 }),
    code("host_model_clock"));
  assert.throws(() => budget.clock({ remainingSeconds: 1200, deadlineEpoch: started + 600 }),
    code("host_model_clock"));
  assert.equal(budget.summary().clock, null);
});
