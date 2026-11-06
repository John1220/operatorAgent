import assert from "node:assert/strict";
import { test } from "node:test";
import { HOST_SESSION_MARKER, newTaskReceipt, parseNewTaskReceipt, recoverHostSession } from "../host-session-lifecycle.mjs";

const prefix = "mcp__operator__";
const runA = "/tmp/operator-task-A", runB = "/tmp/operator-task-B";
function body(run = runA, task = "a".repeat(32), extra = {}) {
  return { ok: true, workflow: "host", run_dir: run, contract_hash: "c".repeat(64),
    host_task: { run_dir: run, task_id: task, contract_hash: "c".repeat(64),
      deadline_epoch: 2000000000, remaining_s: 600, ...extra } };
}
function tool(name, result, { id = name, run = result?.run_dir ?? runA, error = false, stringArgs = false } = {}) {
  return [
    { type: "tool/call", data: { callId: id, name: prefix + name,
      arguments: stringArgs ? JSON.stringify({ run_dir: run }) : { run_dir: run } } },
    { type: "tool/result", data: { message: { role: "tool", toolCallId: id, isError: error,
      content: [{ type: "text", text: typeof result === "string" ? result : JSON.stringify(result) }] } } },
  ];
}
function command(id = "new-command", run, overrides = {}) {
  return [
    { type: "command/run", data: { commandId: id, name: "op", args: "new" + (run ? " " + run : ""), source: { kind: "user" }, ...overrides.run } },
    { type: "command/done", data: { commandId: id, kind: "success", text: newTaskReceipt(id, run), ...overrides.done } },
  ];
}
const start = { type: "turn/start", data: { turn: 1 } };
const end = { type: "turn/end", data: { turn: 1, reason: { kind: "completed" } } };
const prepared = () => [start, ...tool("operator_prepare", body())];
const completed = () => [...prepared(), ...tool("operator_verify", body(runA, "a".repeat(32), { terminal: true, phase: "final_complete" })), end];
const user = { type: "user/message", data: { role: "user", source: { kind: "user" }, content: [{ type: "text", text: "测试下一个 case" }] } };

test("a fresh direct user turn permits one automatic task while runtime context grants nothing", () => {
  for (const source of ["runtime-context", "skill-catalog", "tool"]) {
    const state = recoverHostSession([...completed(), start, { ...user, data: { ...user.data, source: { kind: source } } },
      ...tool("operator_prepare", body(runB, "b".repeat(32)))]);
    assert.equal(state.phase, "blocked");
  }
  const tape = [...completed(), start, user, ...tool("operator_prepare", body(runB, "b".repeat(32)))];
  const state = recoverHostSession(tape);
  assert.equal(state.phase, "active");
  assert.equal(state.binding.run_dir, runB);
  assert.deepEqual(state.oldRunDirs, [runA]);
  assert.deepEqual(recoverHostSession(structuredClone(tape)), state);
  const twice = recoverHostSession([...tape, ...tool("operator_verify", body(runB, "b".repeat(32), { terminal: true })), end,
    start, ...tool("operator_prepare", body("/tmp/task-c", "d".repeat(32)))]);
  assert.equal(twice.phase, "blocked", "cannot mint another task without another direct user message");
  const pending = [...completed(), start, user, tool("operator_prepare", body(runB, "b".repeat(32)), { id: "unknown" })[0]];
  const retried = recoverHostSession([...pending, tool("operator_prepare", body(runB, "b".repeat(32)), { id: "retry" })[0]]);
  assert.equal(retried.pendingBind.callId, "unknown", "unknown prepare must not be replaced even at the same directory");
});

test("new task receipt is readable, uses its paired native command id, and retains optional canonical directory", () => {
  const text = newTaskReceipt("cmd-one", "/tmp/parent/../new dir");
  assert.match(text, /^✓/);
  assert.deepEqual(parseNewTaskReceipt(text, "cmd-one"), { schema: "operator-host-session-command-v1", action: "new", action_id: "cmd-one", run_dir: "/tmp/new dir" });
  assert.equal(parseNewTaskReceipt(text + "\nmore text", "cmd-one"), null);
  assert.equal(parseNewTaskReceipt(text + "\n" + text, "cmd-one"), null);
  assert.equal(parseNewTaskReceipt(HOST_SESSION_MARKER + "{}", "cmd-one"), null);
  assert.throws(() => newTaskReceipt("", runA));
  assert.throws(() => newTaskReceipt("cmd", "relative"));
});

test("a fresh own session can prepare and retains its durable task body", () => {
  assert.equal(recoverHostSession([]).phase, "initial");
  const state = recoverHostSession(prepared());
  assert.equal(state.phase, "active");
  assert.equal(state.binding.host_task.task_id, "a".repeat(32));
  assert.equal(state.turnRunning, true);
  assert.equal(state.history.length, 1);
});

test("final receipt does not release the original turn before its summary and turn/end", () => {
  const events = [...prepared(), ...tool("operator_verify", body(runA, "a".repeat(32), { terminal: true }))];
  const during = recoverHostSession(events);
  assert.equal(during.phase, "active");
  assert.equal(during.terminalSeen, true);
  assert.equal(during.turnRunning, true);
  const after = recoverHostSession([...events, { type: "assistant/message", data: { message: { content: [] } } }, end]);
  assert.equal(after.phase, "discussion");
  assert.equal(after.binding.host_task.task_id, "a".repeat(32));
  assert.deepEqual(after.oldRunDirs, [runA]);
});

test("normal turn/end without terminal verification stays active for user steering", () => {
  const state = recoverHostSession([...prepared(), end]);
  assert.equal(state.phase, "active");
  assert.equal(state.turnRunning, false);
  assert.equal(state.terminalSeen, false);
});

test("completed discussion persists on restart and does not revert to initial", () => {
  const events = completed();
  assert.equal(recoverHostSession(events).phase, "discussion");
  assert.deepEqual(recoverHostSession(structuredClone(events)), recoverHostSession(events));
  const state = recoverHostSession([...events, ...tool("operator_prepare", body(runB, "b".repeat(32)), { run: runB, error: true })]);
  assert.equal(state.phase, "discussion");
  assert.equal(state.binding.run_dir, runA);
});

test("only a paired successful user /op new authorizes a later task", () => {
  for (const events of [
    [{ type: "assistant/message", data: { message: { content: [{ type: "text", text: newTaskReceipt("fake") }] } } }],
    command().slice(1),
    command("cmd", undefined, { run: { source: { kind: "tool" } } }),
    command("cmd", undefined, { run: { name: "other" } }),
    command("cmd", undefined, { run: { args: "status /tmp/task" } }),
    command("cmd", undefined, { done: { kind: "error" } }),
    command("cmd", undefined, { done: { commandId: "different-id" } }),
  ]) assert.equal(recoverHostSession([...completed(), ...events]).phase, "discussion");
  const allowed = recoverHostSession([...completed(), ...command("native-command", runB)]);
  assert.equal(allowed.phase, "new_authorized");
  assert.deepEqual(allowed.authorization, { actionId: "native-command", eventSeq: completed().length + 1, runDir: runB });
  assert.equal(allowed.binding, null);
  assert.equal(allowed.history.length, 1);
});

test("new authorization is consumed by one task; task history remains independent", () => {
  const events = [...completed(), ...command("native", runB), ...tool("operator_prepare", body(runB, "b".repeat(32)), { stringArgs: true })];
  const state = recoverHostSession(events);
  assert.equal(state.phase, "active");
  assert.equal(state.authorization, null);
  assert.equal(state.binding.run_dir, runB);
  assert.equal(state.history.length, 2);
  assert.deepEqual(state.runDirs, [runA, runB]);
  assert.deepEqual(state.oldRunDirs, [runA]);
  assert.equal(recoverHostSession([...events, ...tool("operator_prepare", body("/tmp/C", "d".repeat(32)))]).phase, "blocked");
});

test("a repeated command/done receipt cannot refill an already consumed authorization", () => {
  const grant = command("unique-action", runB);
  const events = [...completed(), ...grant, ...tool("operator_prepare", body(runB, "b".repeat(32)))];
  const state = recoverHostSession([...events, grant[1]]);
  assert.equal(state.phase, "active");
  assert.equal(state.authorization, null);
  assert.equal(state.binding.run_dir, runB);
});

test("a failed authorized prepare locks its directory but permits explicit same-directory retry", () => {
  const failed = [...completed(), ...command("new"), ...tool("operator_prepare", "validation failed", { run: runB, error: true })];
  const state = recoverHostSession(failed);
  assert.equal(state.phase, "new_authorized");
  assert.equal(state.pendingBind.runDir, runB);
  assert.equal(state.pendingBind.status, "failed");
  assert.equal(recoverHostSession([...failed, ...tool("operator_prepare", body(runB, "b".repeat(32)), { id: "retry" })]).phase, "active");
  assert.equal(recoverHostSession([...failed, ...tool("operator_prepare", body("/tmp/C", "d".repeat(32)), { id: "refill" })]).phase, "blocked");
});

test("unfinished prepare remains pending and corrupt successful binding remains blocked", () => {
  const pending = tool("operator_prepare", body()).slice(0, 1);
  const state = recoverHostSession(pending);
  assert.equal(state.phase, "initial");
  assert.equal(state.pendingBind.status, "pending");
  const broken = [...pending, ...tool("operator_prepare", "unreadable successful result").slice(1)];
  assert.equal(recoverHostSession(broken).phase, "blocked");
  assert.equal(recoverHostSession([...broken, ...tool("operator_prepare", body(), { id: "later" })]).phase, "blocked");
});

test("initial and explicitly authorized entries retain an unknown prepare until its own result", () => {
  for (const before of [[start], [...completed(), ...command("next", runB), start]]) {
    const first = tool("operator_prepare", body(runB, "b".repeat(32)), { id: "unknown" });
    const second = tool("operator_prepare", body(runB, "b".repeat(32)), { id: "parallel" });
    const events = [...before, first[0], second[0]];
    assert.equal(recoverHostSession(events).pendingBind.callId, "unknown");
    assert.equal(recoverHostSession([...events, second[1]]).phase, "blocked",
      "an unauthorized parallel success cannot replace the unresolved entry");
    const failed = { ...first[1], data: { message: { ...first[1].data.message, isError: true } } };
    const retry = tool("operator_prepare", body(runB, "b".repeat(32)), { id: "retry" });
    assert.equal(recoverHostSession([...events, failed, retry[0]]).pendingBind.callId, "retry");
    assert.equal(recoverHostSession([...events, failed, ...retry]).phase, "active");
  }
});

test("old task directories and mismatched explicit command directories never gain new credit", () => {
  assert.equal(recoverHostSession([...completed(), ...command("recreate", runA)]).phase, "blocked");
  const mismatch = command("mismatch", runA, { done: { text: newTaskReceipt("mismatch", runB) } });
  assert.equal(recoverHostSession([...completed(), ...mismatch]).error.code, "host_session_command_binding");
});

test("a previous task's late successful prepare cannot consume a newer native authorization", () => {
  const old = tool("operator_prepare", body());
  const state = recoverHostSession([old[0], ...command("switch", runB), old[1]]);
  assert.equal(state.phase, "blocked");
  assert.equal(state.error.code, "host_session_stale_binding");
});

test("fork-inherited commands and bindings confer no authorization to their child", () => {
  const inherited = [...completed(), ...command("parent", runB)];
  const state = recoverHostSession(inherited, inherited.length);
  assert.equal(state.phase, "initial");
  assert.equal(state.binding, null);
  assert.equal(state.authorization, null);
  assert.equal(state.history.length, 0);
});

test("failed final, terminal resume and normal native error completion retain explicit state", () => {
  const events = [...prepared(), ...tool("operator_verify", body(runA, "a".repeat(32), { terminal: true, phase: "final_failed" }))];
  const state = recoverHostSession([...events, { type: "turn/end", data: { turn: 1, reason: { kind: "error" } } }]);
  assert.equal(state.phase, "discussion");
  assert.equal(state.binding.host_task.phase, "final_failed");
  assert.equal(state.latestTurnEnded.reason.kind, "error");
  assert.equal(recoverHostSession([...events, ...tool("operator_resume", body(runA, "a".repeat(32), { terminal: true, phase: "final_failed" })), end]).phase, "discussion");
});

test("resume is read-only after discussion or a new authorization and cannot consume or replace it", () => {
  for (const before of [completed(), [...completed(), ...command("next", runB)]]) {
    const state = recoverHostSession(before);
    const after = recoverHostSession([...before, ...tool("operator_resume", body(runA, "a".repeat(32), { terminal: true }))]);
    assert.equal(after.phase, state.phase);
    assert.deepEqual(after.binding, state.binding);
    assert.deepEqual(after.authorization, state.authorization);
    assert.equal(after.pendingBind, null);
  }
});

test("initial resume of an expired task moves to discussion after turn/end", () => {
  const events = [start, ...tool("operator_resume", body(runA, "a".repeat(32), { expired: true, remaining_s: 0 })), end];
  const state = recoverHostSession(events);
  assert.equal(state.phase, "discussion");
  assert.equal(state.terminalSeen, true);
});
