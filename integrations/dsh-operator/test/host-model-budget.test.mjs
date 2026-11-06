import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { installHostModelBudget, recoverHostBinding } from "../host-model-budget.mjs";
import { HostModelLedger, MODEL_BUDGET_FILE } from "../host-model-ledger.mjs";

const PREFIX = "mcp__operator__";
const TASK = "a".repeat(32);
const CONTRACT = "b".repeat(64);
const canonical = value => Array.isArray(value) ? "[" + value.map(canonical).join(",") + "]"
  : value && typeof value === "object" ? "{" + Object.keys(value).sort()
    .map(key => JSON.stringify(key) + ":" + canonical(value[key])).join(",") + "}"
    : JSON.stringify(value);
const hash = value => createHash("sha256").update(value).digest("hex");

function fixture(t, overrides = {}, taskId = TASK, directory) {
  const dir = directory ?? mkdtempSync(join(tmpdir(), "operator-host-hook-"));
  if (directory) mkdirSync(dir, { mode: 0o700 });
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const policy = { schema: "operator-host-task-budget-v1", contract_hash: CONTRACT,
    limits: { wall_clock_s: 60, max_verifications: 6, final_reserve: 1,
      max_model_calls: 3, max_output_tokens_per_call: 64,
      total_output_token_reservations: 192, ...overrides }, required_case_ids: ["case"] };
  const bytes = canonical(policy);
  writeFileSync(join(dir, "host-task-policy.json"), bytes + "\n", { mode: 0o400 });
  const started = Date.now() / 1000;
  const hostState = {
    schema: policy.schema, task_id: taskId, contract_hash: CONTRACT,
    policy_hash: hash(bytes), started_epoch: started, started_monotonic: 0,
    max_elapsed_s: 0, records: [],
  };
  writeFileSync(join(dir, "host-task.json"), JSON.stringify({
    ...hostState, state_hash: hash(canonical(hostState)),
  }) + "\n", { mode: 0o600 });
  const body = { ok: true, workflow: "host", run_dir: dir, contract_hash: CONTRACT,
    host_task: { task_id: taskId, contract_hash: CONTRACT, run_dir: dir,
      remaining_s: policy.limits.wall_clock_s,
      deadline_epoch: started + policy.limits.wall_clock_s } };
  return { dir, body,
    ledger: () => new HostModelLedger(dir, { taskId, contractHash: CONTRACT }),
    state: () => JSON.parse(readFileSync(join(dir, MODEL_BUDGET_FILE), "utf8")) };
}

function fakeHost({ events = [], inheritedEventCount = 0, sessionId = "session-1", header = {} } = {}) {
  const handlers = new Map();
  const guards = [];
  let currentController;
  const agent = { session: { id: sessionId }, status: "idle", cancel: () => currentController?.abort() };
  const errors = [];
  let observed = 0, disposed = 0;
  const ctx = {
    on(name, callback) {
      const list = handlers.get(name) ?? [];
      list.push(callback); handlers.set(name, list);
      return () => list.splice(list.indexOf(callback), 1);
    },
    tools: { guard(callback) { guards.push(callback); return () => guards.splice(guards.indexOf(callback), 1); } },
    logger: { error: value => errors.push(value) },
    sessionQuery: { async observeSession(id) {
      assert.equal(id, sessionId); observed += 1;
      return { events, inheritedEventCount, header, [Symbol.dispose]() { disposed += 1; } };
    } },
  };
  const dispose = installHostModelBudget(ctx, {});
  const emit = (name, ...args) => (handlers.get(name) ?? []).map(handler => handler(...args));
  let seq = Math.max(0, ...events.map(event => event.seq ?? 0));
  const ready = Promise.all(emit("agent/created", { agent, signal: new AbortController().signal }));
  // Restore errors remain observable through request(), without an unhandled promise.
  ready.catch(() => {});
  const native = event => {
    const value = { ...event, seq: ++seq };
    events.push(value);
    emit("session/event", agent.session, value);
    return value;
  };
  const emitTool = async (body, name = "operator_prepare", { isError = false } = {}) => {
    await ready;
    if (!events.some(event => event.type === "turn/start")) native({ type: "turn/start", data: { turn: 1 } });
    const callId = "call-" + events.length;
    const content = [{ type: "text", text: JSON.stringify(body) }];
    native({ type: "tool/call", data: { callId, name: PREFIX + name, arguments: JSON.stringify({ run_dir: body.run_dir }) } });
    emit("tools/result", { name: PREFIX + name, callId, agent, arguments: { run_dir: body.run_dir } },
      { isError, value: { content }, content });
    native({ type: "tool/result", data: { message: { role: "tool", toolCallId: callId,
      source: { kind: "tool", callId }, isError, content } } });
  };
  const request = async (base = {}, controller = new AbortController()) => {
    await ready;
    currentController = controller;
    let next = async () => ({ provider: "fixture", model: "model", maxTokens: 1000, ...base });
    for (const handler of [...(handlers.get("agent/request") ?? [])].reverse()) {
      const rest = next;
      next = () => handler({ agent, signal: controller.signal, turn: 1, step: 1 }, rest);
    }
    return { ...await next(), sessionId, signal: controller.signal };
  };
  const stream = (options, provider) => {
    let next = provider;
    for (const handler of [...(handlers.get("llm/stream") ?? [])].reverse()) {
      const rest = next;
      next = () => handler(options, rest);
    }
    return next();
  };
  const dispatch = async (provider, base, controller) => {
    const options = await request(base, controller);
    for await (const _chunk of stream(options, provider)) { /* exercise iterator to completion */ }
    return options;
  };
  const newTask = async (runDir, { source = "user", nativeCommand = true } = {}) => {
    await ready;
    const commandId = "command-" + (events.length + 1);
    if (nativeCommand) native({ type: "command/run", data: { commandId, name: "op",
      args: "new" + (runDir ? " " + runDir : ""), source: { kind: source } } });
    try {
      const result = await dispose.newTask({ agent, commandId, runDir, signal: new AbortController().signal });
      native({ type: "command/done", data: { commandId, kind: "success", text: result } });
      return result;
    } catch (error) {
      if (nativeCommand) native({ type: "command/done", data: { commandId, kind: "error", text: error.message } });
      throw error;
    }
  };
  return { agent, ctx, events, errors, emit, emitTool, request, stream, dispatch, dispose, ready, native, newTask,
    guard: (name, arguments_ = {}, callId) => guards.map(guard => guard({ name, arguments: arguments_, agent, callId })).find(Boolean),
    observations: () => ({ observed, disposed }) };
}

function provider(usage = { inputTokens: 5, outputTokens: 2, totalTokens: 7 }) {
  let dispatched = 0;
  const next = async function* () {
    dispatched += 1;
    if (usage) yield { type: "usage", usage };
    yield { type: "finish", reason: { kind: "stop" } };
  };
  return { next, count: () => dispatched };
}

test("successful prepare binds real durable credit, clamps native requests and settles API usage", async t => {
  const task = fixture(t);
  const host = fakeHost(); t.after(host.dispose);
  const upstream = provider();
  await host.dispatch(upstream.next); // preparation dispatch is not part of this task
  await host.emitTool(task.body);
  assert.equal(task.ledger().summary().calls_reserved, 0);
  const options = await host.dispatch(upstream.next);
  assert.equal(options.maxTokens, 64);
  assert.equal(upstream.count(), 2);
  const state = task.ledger().summary();
  assert.equal(state.calls_reserved, 1);
  assert.equal(state.output_tokens_reserved, 64);
  assert.equal(state.records[0].status, "answered");
  assert.equal(state.api_usage_known_calls, 1);
  assert.deepEqual(host.observations(), { observed: 1, disposed: 1 });
});

test("exhausted call count stops before invoking provider next", async t => {
  const task = fixture(t, { max_model_calls: 1 });
  const host = fakeHost(); t.after(host.dispose); await host.emitTool(task.body);
  const upstream = provider();
  await host.dispatch(upstream.next);
  await assert.rejects(() => host.dispatch(upstream.next), error => error.code === "host_model_call_budget");
  assert.equal(upstream.count(), 1);
  assert.equal(task.state().records.length, 1);
});

test("total output reservations shrink the next request then stop without refunds", async t => {
  const task = fixture(t, { total_output_token_reservations: 96 });
  const host = fakeHost(); t.after(host.dispose); await host.emitTool(task.body);
  const upstream = provider({ inputTokens: 1, outputTokens: 1, totalTokens: 2 });
  assert.equal((await host.dispatch(upstream.next)).maxTokens, 64);
  assert.equal((await host.dispatch(upstream.next)).maxTokens, 32);
  await assert.rejects(() => host.dispatch(upstream.next), error => error.code === "host_model_token_budget");
  assert.equal(upstream.count(), 2);
  assert.equal(task.ledger().summary().output_tokens_reserved, 96);
});

test("a later middleware cannot raise the actual output cap past frozen policy", async t => {
  const task = fixture(t);
  const host = fakeHost(); t.after(host.dispose); await host.emitTool(task.body);
  const options = await host.request();
  const upstream = provider();
  await assert.rejects(async () => {
    for await (const _ of host.stream({ ...options, maxTokens: 65 }, upstream.next)) { /* rejected */ }
  }, error => error.code === "host_model_output_limit");
  assert.equal(upstream.count(), 0);
  assert.equal(task.ledger().summary().calls_reserved, 0);
});

test("provider failure and client cancellation each consume their own full reservation", async t => {
  const task = fixture(t, { max_model_calls: 2, total_output_token_reservations: 128 });
  const host = fakeHost(); t.after(host.dispose); await host.emitTool(task.body);
  await assert.rejects(() => host.dispatch(async function* () { throw new Error("provider unavailable"); }), /provider unavailable/);
  const controller = new AbortController();
  await host.dispatch(async function* () {
    controller.abort();
    yield { type: "finish", reason: { kind: "aborted" } };
  }, undefined, controller);
  const summary = task.ledger().summary();
  assert.deepEqual(summary.records.map(record => record.status), ["failed", "cancelled"]);
  assert.equal(summary.output_tokens_reserved, 128);
  assert.equal(summary.api_usage_unknown_calls, 2);
});

test("native tool-call completion is answered while output truncation remains failed and charged", async t => {
  const task = fixture(t);
  const host = fakeHost(); t.after(host.dispose); await host.emitTool(task.body);
  for (const kind of ["tool-calls", "max-tokens"]) {
    await host.dispatch(async function* () {
      yield { type: "usage", usage: { inputTokens: 8, outputTokens: 3, totalTokens: 11 } };
      yield { type: "finish", reason: { kind } };
    });
  }
  const summary = task.ledger().summary();
  assert.deepEqual(summary.records.map(record => record.status), ["answered", "failed"]);
  assert.equal(summary.calls_reserved, 2);
  assert.equal(summary.output_tokens_reserved, 128);
  assert.equal(summary.api_usage_known_calls, 2);
});

test("bound tasks reject new directories and alternate model tools, keeping operator status available", async t => {
  const task = fixture(t);
  const host = fakeHost(); t.after(host.dispose); await host.emitTool(task.body);
  assert.match(host.guard(PREFIX + "operator_prepare", { run_dir: join(task.dir, "refill") }), /当前任务目录/);
  for (const name of ["subagent", "renamed-subagent", "run_code", "shell", "workflow"]) {
    assert.match(host.guard(name), /only its operator tools/);
  }
  assert.equal(host.guard(PREFIX + "operator_status", { run_dir: task.dir }), undefined);
  assert.equal(host.guard(PREFIX + "operator_cancel", { run_dir: task.dir, invocation_id: "a".repeat(32) }), undefined);
});

test("unbound context and verify cannot use another task without first binding its model credit", async t => {
  const task = fixture(t);
  const host = fakeHost(); t.after(host.dispose); await host.ready;
  for (const name of ["operator_context", "operator_verify"]) {
    assert.equal(typeof host.guard(PREFIX + name, { run_dir: task.dir }), "string");
  }
  assert.equal(host.guard(PREFIX + "operator_prepare", { run_dir: task.dir }), undefined);
  assert.equal(host.guard(PREFIX + "operator_resume", { run_dir: task.dir }), undefined);
});

test("a fresh plugin restores successful durable binding without resetting counters", async t => {
  const task = fixture(t, { max_model_calls: 1 });
  const first = fakeHost(); await first.emitTool(task.body);
  await first.dispatch(provider().next);
  first.dispose();
  const restored = fakeHost({ events: structuredClone(first.events) }); t.after(restored.dispose);
  await Promise.all(restored.emit("agent/created", { agent: restored.agent, signal: new AbortController().signal }));
  const upstream = provider();
  await assert.rejects(() => restored.dispatch(upstream.next), error => error.code === "host_model_call_budget");
  assert.equal(upstream.count(), 0);
  assert.equal(task.ledger().summary().calls_reserved, 1);
});

test("a backward wall-clock change cannot renew the main model deadline", async t => {
  const task = fixture(t);
  const host = fakeHost(); t.after(host.dispose); await host.emitTool(task.body);
  const wall = Date.now();
  const monotonic = process.hrtime.bigint();
  t.mock.method(Date, "now", () => wall - 60_000);
  t.mock.method(process.hrtime, "bigint", () => monotonic + 61_000_000_000n);
  const upstream = provider();
  await assert.rejects(() => host.dispatch(upstream.next), error => error.code === "host_task_deadline");
  assert.equal(upstream.count(), 0);
  assert.equal(task.ledger().summary().calls_reserved, 0);
  assert.equal(host.guard(PREFIX + "operator_status", { run_dir: task.dir }), undefined);
  assert.equal(host.guard(PREFIX + "operator_cancel", { run_dir: task.dir, invocation_id: "a".repeat(32) }), undefined);
});

test("a backward wall-clock change cannot postpone cancellation of an in-flight main model", async t => {
  const task = fixture(t);
  const host = fakeHost(); t.after(host.dispose); await host.emitTool(task.body);
  const wall = Date.now();
  t.mock.method(Date, "now", () => wall - 60_000);
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const controller = new AbortController();
  const options = await host.request({}, controller);
  const stream = host.stream(options, async function* () {
    await new Promise(resolve => controller.signal.addEventListener("abort", resolve, { once: true }));
    yield { type: "finish", reason: { kind: "aborted" } };
  })[Symbol.asyncIterator]();
  const first = stream.next();
  t.mock.timers.tick(60_001);
  assert.equal(controller.signal.aborted, true, "deadline must retain its original elapsed-time limit");
  await first;
  await stream.next();
  const summary = task.ledger().summary();
  assert.equal(summary.records[0].status, "cancelled");
  assert.equal(summary.output_tokens_reserved, 64);
});

test("durable restoration cannot restart the deadline from an old prepare transcript", async t => {
  const task = fixture(t);
  const first = fakeHost(); await first.emitTool(task.body); first.dispose();
  const wall = Date.now();
  const monotonic = process.hrtime.bigint();
  t.mock.method(Date, "now", () => wall - 60_000);
  t.mock.method(process.hrtime, "bigint", () => monotonic + 61_000_000_000n);
  const restored = fakeHost({ events: structuredClone(first.events) }); t.after(restored.dispose);
  const upstream = provider();
  await assert.rejects(() => restored.dispatch(upstream.next), error => error.code === "host_task_deadline");
  assert.equal(upstream.count(), 0);
  assert.equal(task.ledger().summary().calls_reserved, 0);
});

test("an observed exhausted wall clock cannot gain time when the clock returns or the session restarts", async t => {
  const task = fixture(t);
  const first = fakeHost(); t.after(first.dispose); await first.emitTool(task.body);
  const initial = Date.now();
  let wall = initial + 61_000;
  t.mock.method(Date, "now", () => wall);
  const upstream = provider();
  await assert.rejects(() => first.dispatch(upstream.next), error => error.code === "host_task_deadline");
  wall = initial;
  await assert.rejects(() => first.dispatch(upstream.next), error => error.code === "host_task_deadline");
  const restored = fakeHost({ events: structuredClone(first.events) }); t.after(restored.dispose);
  await assert.rejects(() => restored.dispatch(upstream.next), error => error.code === "host_task_deadline");
  assert.equal(upstream.count(), 0);
});

test("fork-inherited prepare events do not become the child's own task binding", async t => {
  const task = fixture(t);
  const parent = fakeHost(); await parent.emitTool(task.body); parent.dispose();
  const child = fakeHost({ events: structuredClone(parent.events), inheritedEventCount: parent.events.length,
    sessionId: "fork-child" }); t.after(child.dispose);
  await Promise.all(child.emit("agent/created", { agent: child.agent, signal: new AbortController().signal }));
  assert.equal(recoverHostBinding(child.events, child.events.length), undefined);
  assert.equal(typeof child.guard(PREFIX + "operator_verify", { run_dir: task.dir }), "string");
  assert.equal(task.ledger().summary().lease.session_id, "session-1");
});

test("delegated and forked sessions cannot create or resume a task under a fresh uncounted lease", async t => {
  const task = fixture(t);
  for (const [index, header] of [{ origin: "subagent" }, { origin: "fork", parentSession: "parent" }].entries()) {
    const child = fakeHost({ sessionId: "child-" + index, header }); t.after(child.dispose);
    await Promise.all(child.emit("agent/created", { agent: child.agent, signal: new AbortController().signal }));
    for (const name of ["operator_prepare", "operator_resume", "operator_context", "operator_verify"]) {
      assert.match(child.guard(PREFIX + name, { run_dir: task.dir }), /仅支持主会话/);
    }
    assert.equal(child.guard(PREFIX + "operator_status", { run_dir: task.dir }), undefined);
  }
});

test("binding failure blocks later native dispatch instead of observer errors being ignored", async t => {
  const task = fixture(t);
  const host = fakeHost(); t.after(host.dispose);
  await host.emitTool({ ...task.body, host_task: { ...task.body.host_task, task_id: "c".repeat(32) } });
  const upstream = provider();
  await assert.rejects(() => host.dispatch(upstream.next), /host_model_binding/);
  assert.equal(upstream.count(), 0);
  assert.ok(host.errors.length > 0);
});

test("unreadable successful host preparation fails closed before another model dispatch", async t => {
  const host = fakeHost(); t.after(host.dispose); await host.ready;
  host.emit("tools/result", { name: PREFIX + "operator_prepare", agent: host.agent },
    { isError: false, value: { content: [{ type: "text", text: "not the host task result" }] } });
  const upstream = provider();
  await assert.rejects(() => host.dispatch(upstream.next), /host_model/);
  assert.equal(upstream.count(), 0);
});

test("unreadable successful binding remains blocked after durable session restoration", async t => {
  for (const name of ["operator_prepare", "operator_resume"]) {
    const events = [
      { type: "tool/call", data: { callId: "broken-bind", name: PREFIX + name } },
      { type: "tool/result", data: { message: { toolCallId: "broken-bind", isError: false,
        content: [{ type: "text", text: "not the host task result" }] } } },
    ];
    const restored = fakeHost({ events }); t.after(restored.dispose);
    const upstream = provider();
    await assert.rejects(() => restored.dispatch(upstream.next), /host_model/);
    assert.equal(upstream.count(), 0);
    assert.deepEqual(restored.observations(), { observed: 1, disposed: 1 });
  }
});

test("legacy relay auxiliary requests remain outside the native main-task counter", async t => {
  const task = fixture(t);
  const host = fakeHost(); t.after(host.dispose); await host.emitTool(task.body);
  const upstream = provider();
  for await (const _ of host.stream({ sessionId: host.agent.session.id,
    purpose: "operator-delegation", signal: new AbortController().signal,
    provider: "fixture", model: "model", maxTokens: 64 }, upstream.next)) { /* legacy adapter */ }
  assert.equal(upstream.count(), 1);
  assert.equal(task.ledger().summary().calls_reserved, 0);
});

test("final summary remains charged until turn/end; discussion then leaves the old journal byte-identical", async t => {
  const task = fixture(t);
  const host = fakeHost(); t.after(host.dispose); await host.emitTool(task.body);
  await host.dispatch(provider().next);
  await host.emitTool({ ...task.body, host_task: { ...task.body.host_task, terminal: true, phase: "final_complete" } }, "operator_verify");
  await host.dispatch(provider().next); // original turn's final summary
  assert.equal(task.state().records.length, 2);
  host.native({ type: "turn/end", data: { turn: 1, reason: { kind: "completed" } } });
  const original = readFileSync(join(task.dir, MODEL_BUDGET_FILE), "utf8");
  host.native({ type: "turn/start", data: { turn: 2 } });
  await host.dispatch(provider().next);
  assert.equal(readFileSync(join(task.dir, MODEL_BUDGET_FILE), "utf8"), original);
  assert.match(host.guard(PREFIX + "operator_verify", { run_dir: task.dir }), /讨论模式/);
  assert.match(host.guard(PREFIX + "operator_prepare", { run_dir: task.dir + "-other" }), /op new/);
  await host.emitTool(task.body, "operator_resume");
  await host.dispatch(provider().next);
  assert.equal(readFileSync(join(task.dir, MODEL_BUDGET_FILE), "utf8"), original, "read-only resume cannot reactivate execution");
});

test("a budget stop ends execution scope; the next turn can discuss but not verify", async t => {
  const task = fixture(t, { max_model_calls: 1 });
  const host = fakeHost(); t.after(host.dispose); await host.emitTool(task.body);
  await host.dispatch(provider().next);
  await assert.rejects(() => host.dispatch(provider().next), /本任务已停止.*|Task model calls exhausted/s);
  host.native({ type: "turn/end", data: { turn: 1, reason: { kind: "error" } } });
  const original = readFileSync(join(task.dir, MODEL_BUDGET_FILE), "utf8");
  host.native({ type: "turn/start", data: { turn: 2 } });
  await host.dispatch(provider().next);
  assert.equal(readFileSync(join(task.dir, MODEL_BUDGET_FILE), "utf8"), original);
  assert.match(host.guard(PREFIX + "operator_verify", { run_dir: task.dir }), /讨论模式/);
});

test("only a native user command permits a fresh task, preserving the previous task", async t => {
  const a = fixture(t), fresh = join(a.dir, "fresh-task");
  const host = fakeHost(); t.after(host.dispose); await host.emitTool(a.body);
  await host.dispatch(provider().next);
  host.native({ type: "turn/end", data: { turn: 1, reason: { kind: "completed" } } });
  const original = readFileSync(join(a.dir, MODEL_BUDGET_FILE), "utf8");
  await assert.rejects(() => host.newTask(fresh, { source: "tool" }), /原生用户/);
  await assert.rejects(() => host.newTask(fresh, { nativeCommand: false }), /原生用户/);
  await host.newTask(fresh);
  assert.match(host.guard(PREFIX + "operator_prepare", { run_dir: a.dir }), /目录|旧任务/);
  assert.equal(host.guard(PREFIX + "operator_prepare", { run_dir: fresh }), undefined);
  host.native({ type: "turn/start", data: { turn: 2 } });
  const b = fixture(t, {}, "b".repeat(32), fresh);
  await host.emitTool(b.body);
  await host.dispatch(provider().next);
  assert.equal(b.state().records.length, 1);
  assert.equal(readFileSync(join(a.dir, MODEL_BUDGET_FILE), "utf8"), original);
});

test("new task refuses a running turn, unfinished model record, and unknown verification owner", async t => {
  const task = fixture(t);
  const host = fakeHost(); t.after(host.dispose); await host.emitTool(task.body);
  host.agent.status = "running";
  await assert.rejects(() => host.newTask(), /仍在运行/);
  host.agent.status = "idle";
  const ledger = task.ledger(), state = ledger.summary();
  const reservation = ledger.reserve(host.agent.session.id, state.lease.id,
    { requestId: "unfinished", maxTokens: 32, provider: "fixture", model: "model" });
  await assert.rejects(() => host.newTask(), /未结算模型|未清理/);
  ledger.finish(host.agent.session.id, state.lease.id, reservation.id, { status: "cancelled" });
  const path = join(task.dir, "host-task.json"), taskState = JSON.parse(readFileSync(path, "utf8"));
  taskState.records.push({ status: "reserved", owner_invocation_id: "f".repeat(32) });
  writeFileSync(path, JSON.stringify(taskState));
  await assert.rejects(() => host.newTask(), /未清理/);
  assert.equal(task.state().records.length, 1);
  assert.equal(task.state().records[0].status, "cancelled");
});

test("new task cannot abandon an unconfirmed prepare across process restart", async t => {
  const events = [
    { type: "turn/start", data: { turn: 1 } },
    { type: "tool/call", data: { callId: "interrupted-prepare", name: PREFIX + "operator_prepare",
      arguments: JSON.stringify({ run_dir: "/tmp/interrupted-operator-task" }) } },
  ];
  const restored = fakeHost({ events }); t.after(restored.dispose);
  await assert.rejects(() => restored.newTask("/tmp/different-operator-task"), /结果尚未确认/);
  assert.match(restored.guard(PREFIX + "operator_prepare", { run_dir: "/tmp/different-operator-task" }), /绑定目标目录/);
});

test("initial and manual entries reject a second unresolved prepare but admit their current call and failed retry", async t => {
  const root = fixture(t);
  for (const manual of [false, true]) {
    const host = fakeHost(); t.after(host.dispose); await host.ready;
    const run = join(root.dir, manual ? "manual-entry" : "initial-entry");
    if (manual) await host.newTask(run);
    host.native({ type: "turn/start", data: { turn: 1 } });
    const call = callId => ({ type: "tool/call", data: { callId, name: PREFIX + "operator_prepare", arguments: { run_dir: run } } });
    const failed = callId => ({ type: "tool/result", data: { message: { role: "tool", toolCallId: callId, isError: true, content: [] } } });
    host.native(call("first"));
    assert.equal(host.guard(PREFIX + "operator_prepare", { run_dir: run }, "first"), undefined,
      "native tool/call may be recorded before its own guard runs");
    assert.match(host.guard(PREFIX + "operator_resume", { run_dir: run }, "first"), /尚未确认/);
    assert.match(host.guard(PREFIX + "operator_prepare", { run_dir: run }, "parallel"), /尚未确认/);
    host.native(call("parallel"));
    assert.match(host.guard(PREFIX + "operator_prepare", { run_dir: run }, "parallel"), /尚未确认/);
    host.native(failed("parallel"));
    assert.match(host.guard(PREFIX + "operator_prepare", { run_dir: run }, "another"), /尚未确认/);
    host.native(failed("first"));
    host.native(call("retry"));
    assert.equal(host.guard(PREFIX + "operator_prepare", { run_dir: run }, "retry"), undefined);
    assert.match(host.guard(PREFIX + "operator_prepare", { run_dir: run + "-other" }, "retry"), /绑定目标目录/);
  }
});

test("a pending automatic prepare cannot make a read-only resume bind model credit", async t => {
  const a = fixture(t), host = fakeHost(); t.after(host.dispose);
  await host.emitTool(a.body);
  await host.emitTool({ ...a.body, host_task: { ...a.body.host_task, terminal: true } }, "operator_verify");
  host.native({ type: "turn/end", data: { turn: 1 } });
  host.native({ type: "turn/start", data: { turn: 2 } });
  host.native({ type: "user/message", data: { role: "user", source: { kind: "user" } } });
  const run = join(a.dir, "next-task"), callId = "automatic-prepare";
  host.native({ type: "tool/call", data: { callId, name: PREFIX + "operator_prepare", arguments: { run_dir: run } } });
  assert.equal(host.guard(PREFIX + "operator_prepare", { run_dir: run }, callId), undefined);
  const original = readFileSync(join(a.dir, MODEL_BUDGET_FILE), "utf8");
  await host.emitTool(a.body, "operator_resume");
  const result = body => ({ isError: false, value: { content: [{ type: "text", text: JSON.stringify(body) }] } });
  host.emit("tools/result", { name: PREFIX + "operator_resume", callId, agent: host.agent }, result(a.body));
  assert.equal(readFileSync(join(a.dir, MODEL_BUDGET_FILE), "utf8"), original,
    "read-only resume remains read-only while a different binding is pending");
  const b = fixture(t, {}, "b".repeat(32), run);
  host.emit("tools/result", { name: PREFIX + "operator_prepare", callId, agent: host.agent }, result(b.body));
  host.native({ type: "tool/result", data: { message: { role: "tool", toolCallId: callId,
    isError: false, content: result(b.body).value.content } } });
  await host.dispatch(provider().next);
  assert.equal(b.state().records.length, 1, "the matching result still binds synchronously before dispatch");
  assert.equal(readFileSync(join(a.dir, MODEL_BUDGET_FILE), "utf8"), original);
});

test("an unrelated successful result cannot bind an initial or manual pending entry before native replay", async t => {
  for (const { manual, pendingName, resultCallId } of [
    { manual: false, pendingName: "operator_prepare", resultCallId: "unrelated" },
    { manual: false, pendingName: "operator_resume", resultCallId: "pending" },
    { manual: true, pendingName: "operator_prepare", resultCallId: "unrelated" },
  ]) {
    const task = fixture(t), host = fakeHost(); t.after(host.dispose); await host.ready;
    if (manual) await host.newTask();
    host.native({ type: "tool/call", data: { callId: "pending", name: PREFIX + pendingName, arguments: { run_dir: task.dir } } });
    host.emit("tools/result", { name: PREFIX + "operator_prepare",
      callId: resultCallId, agent: host.agent },
    { isError: false, value: { content: [{ type: "text", text: JSON.stringify(task.body) }] } });
    assert.equal(existsSync(join(task.dir, MODEL_BUDGET_FILE)), false);
    await assert.rejects(() => host.dispatch(provider().next), /matching authorized pending call/);
  }
});

test("new user turn can automatically prepare a fresh task but cannot refill old credit", async t => {
  const a = fixture(t), host = fakeHost(); t.after(host.dispose);
  await host.emitTool(a.body);
  await host.dispatch(provider().next);
  await host.emitTool({ ...a.body, host_task: { ...a.body.host_task, terminal: true } }, "operator_verify");
  host.native({ type: "turn/end", data: { turn: 1 } });
  const old = readFileSync(join(a.dir, MODEL_BUDGET_FILE), "utf8");
  const fresh = join(a.dir, "automatic-task");
  assert.match(host.guard(PREFIX + "operator_prepare", { run_dir: fresh }), /讨论模式/);
  host.native({ type: "turn/start", data: { turn: 2 } });
  host.native({ type: "user/message", data: { role: "user", source: { kind: "user" } } });
  assert.equal(host.guard(PREFIX + "operator_prepare", { run_dir: fresh }), undefined);
  assert.match(host.guard(PREFIX + "operator_prepare", { run_dir: a.dir }), /全新/);
  const b = fixture(t, {}, "b".repeat(32), fresh);
  await host.emitTool(b.body);
  await host.dispatch(provider().next);
  assert.equal(b.state().records.length, 1);
  assert.equal(readFileSync(join(a.dir, MODEL_BUDGET_FILE), "utf8"), old);
});
