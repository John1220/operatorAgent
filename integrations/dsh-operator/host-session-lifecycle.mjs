/** Pure replay of DSH-owned task lifecycle events. Never creates model credit. */
import { isAbsolute, resolve } from "node:path";

export const HOST_SESSION_MARKER = "OPERATOR_TASK_SESSION_V1:";
export const HOST_SESSION_RECEIPT_SCHEMA = "operator-host-session-command-v1";
// Intent is decided by the DSH agent. This gate only proves a fresh, direct
// user turn exists; one successful task consumes that opportunity.
export function canAutoStartTask(state) {
  return !!state?.binding && state.turnRunning && state.latestTurnEnded
    && state.lastUserEventSeq > state.latestTurnEnded.seq
    && state.latestTurnEnded.seq > state.lastBindingEventSeq;
}
const PREFIX = "mcp__operator__";
const BINDS = new Set([PREFIX + "operator_prepare", PREFIX + "operator_resume"]);
const HASH = /^[0-9a-f]{64}$/;
const TASK = /^[0-9a-f]{32}$/;
const identity = value => typeof value === "string" && value.length > 0 && value.length <= 256
  && !/[\u0000-\u001f]/u.test(value);
const contentText = content => Array.isArray(content)
  ? content.filter(block => block?.type === "text").map(block => block.text).join("\n") : "";

function directory(value) {
  return typeof value === "string" && isAbsolute(value) ? resolve(value) : null;
}

function parseArguments(value) {
  try { return typeof value === "string" ? JSON.parse(value) : value; }
  catch { return null; }
}

function hostBody(message) {
  try {
    const body = JSON.parse(contentText(message?.content));
    const budget = body?.host_task;
    const run = directory(budget?.run_dir ?? body?.run_dir);
    if (body?.ok !== true || body.workflow !== "host" || !run || !budget
        || !TASK.test(budget.task_id ?? "") || !HASH.test(budget.contract_hash ?? "")
        || body.contract_hash !== budget.contract_hash
        || !Number.isFinite(budget.deadline_epoch)
        || !Number.isFinite(budget.remaining_s) || budget.remaining_s < 0) return null;
    return body;
  } catch { return null; }
}

export function newTaskReceipt(actionId, runDir) {
  if (!identity(actionId)) throw new TypeError("A native commandId is required.");
  if (runDir !== undefined && !directory(runDir)) throw new TypeError("A new task directory must be absolute.");
  return ["✓ 已开启新任务入口", "下一次 prepare 将使用独立任务；旧任务与已用预算保留。",
    ...(runDir === undefined ? [] : [`新任务目录：${directory(runDir)}`])].join("\n");
}

/** Native command ids provide causality; user-facing text contains no opaque ids. */
export function parseNewTaskReceipt(text, commandId) {
  if (typeof text !== "string" || !identity(commandId)) return null;
  const lines = text.trimEnd().split(/\r?\n/);
  if (lines.length < 2 || lines.length > 3 || lines[0] !== "✓ 已开启新任务入口"
      || lines[1] !== "下一次 prepare 将使用独立任务；旧任务与已用预算保留。") return null;
  const run = lines.length === 3 && lines[2].startsWith("新任务目录：")
    ? directory(lines[2].slice("新任务目录：".length)) : null;
  if (lines.length === 3 && !run) return null;
  return { schema: HOST_SESSION_RECEIPT_SCHEMA, action: "new", action_id: commandId,
    ...(run ? { run_dir: run } : {}) };
}

/** Replay only this session's own native events; inherited fork events grant nothing. */
export function recoverHostSession(events, inheritedCount = 0) {
  if (!Array.isArray(events) || !Number.isInteger(inheritedCount) || inheritedCount < 0) {
    throw new TypeError("Native session events and a nonnegative inherited count are required.");
  }
  const state = { phase: "initial", binding: null, history: [], runDirs: [], oldRunDirs: [],
    authorization: null, pendingBind: null, terminalSeen: false, turnRunning: false,
    latestTurnEnded: null, lastBindingEventSeq: null, lastUserEventSeq: -1, generation: 0, error: null };
  const calls = new Map(), commands = new Map(), seenActions = new Set();
  const rememberOld = body => {
    const run = directory(body?.host_task?.run_dir ?? body?.run_dir);
    if (run && !state.oldRunDirs.includes(run)) state.oldRunDirs.push(run);
  };
  const remember = body => {
    const run = directory(body.host_task.run_dir ?? body.run_dir);
    if (!state.runDirs.includes(run)) state.runDirs.push(run);
    const index = state.history.findIndex(item => item.host_task.task_id === body.host_task.task_id);
    if (index < 0) state.history.push(body); else state.history[index] = body;
  };
  const block = (code, detail) => {
    state.phase = "blocked"; state.authorization = null;
    state.error = { code, detail };
  };

  for (const [offset, event] of events.slice(inheritedCount).entries()) {
    if (!event || typeof event !== "object") continue;
    const data = event.data ?? {};
    const seq = event.seq ?? inheritedCount + offset;
    if (event.type === "user/message" && data.role === "user" && data.source?.kind === "user") {
      state.lastUserEventSeq = seq;
    } else if (event.type === "turn/start") {
      state.turnRunning = true;
    } else if (event.type === "turn/end") {
      state.turnRunning = false;
      state.latestTurnEnded = { seq, ...data };
      if (state.phase === "active" && state.terminalSeen) {
        state.phase = "discussion";
        rememberOld(state.binding);
      }
    } else if (event.type === "command/run" && identity(data.commandId)) {
      // A duplicate pairing id cannot be used to replace an earlier command.
      if (!commands.has(data.commandId)) commands.set(data.commandId, { ...data, seq });
    } else if (event.type === "command/done") {
      const command = commands.get(data.commandId);
      commands.delete(data.commandId);
      if (!command || command.name !== "op" || command.source?.kind !== "user"
          || typeof command.args !== "string" || !/^new(?:\s+[\s\S]+)?$/.test(command.args.trim())
          || data.kind !== "success") continue;
      const receipt = parseNewTaskReceipt(data.text, command.commandId);
      if (!receipt || receipt.action_id !== command.commandId || seenActions.has(receipt.action_id)) continue;
      seenActions.add(receipt.action_id);
      const rawRun = command.args.trim().slice(3).trim().replace(/^(["'])([\s\S]*)\1$/, "$2");
      if (isAbsolute(rawRun) && directory(rawRun) !== receipt.run_dir) {
        block("host_session_command_binding", "The new-task receipt does not match its native command directory.");
        continue;
      }
      rememberOld(state.binding);
      if (receipt.run_dir && state.oldRunDirs.includes(receipt.run_dir)) {
        block("host_session_old_task", "A new task cannot recreate credit in an earlier task directory.");
        continue;
      }
      // The native handler admits this action only after checking live owners.
      // Its successful receipt can also settle an interrupted previous process's turn.
      state.phase = "new_authorized"; state.binding = null; state.pendingBind = null;
      state.generation += 1;
      state.terminalSeen = false; state.turnRunning = false; state.error = null;
      state.authorization = { actionId: receipt.action_id, eventSeq: seq, runDir: receipt.run_dir ?? null };
    } else if (event.type === "tool/call" && identity(data.callId)) {
      const args = parseArguments(data.arguments);
      const run = directory(args?.run_dir);
      const call = { ...data, runDir: run, eventSeq: seq, generation: state.generation };
      calls.set(data.callId, call);
      if (!BINDS.has(data.name)) continue;
      call.autoNew = data.name === PREFIX + "operator_prepare" && canAutoStartTask(state)
        && run && !state.runDirs.includes(run)
        && (!state.pendingBind || (state.pendingBind.status === "failed" && state.pendingBind.runDir === run));
      call.readOnlyResume = data.name === PREFIX + "operator_resume"
        && ["discussion", "new_authorized", "blocked"].includes(state.phase);
      if (call.readOnlyResume) continue;
      const locked = state.pendingBind?.runDir ?? state.authorization?.runDir;
      const allowed = (["initial", "new_authorized"].includes(state.phase) || call.autoNew)
        && run && !state.oldRunDirs.includes(run) && (!locked || locked === run)
        && (!state.pendingBind || (state.pendingBind.status === "failed" && state.pendingBind.runDir === run));
      call.bindingAllowed = allowed || (state.phase === "active" && state.binding
        && run === directory(state.binding.host_task.run_dir ?? state.binding.run_dir));
      if (allowed) state.pendingBind = { callId: data.callId, name: data.name,
        runDir: run, eventSeq: seq, status: "pending", autoNew: !!call.autoNew };
    } else if (event.type === "tool/result") {
      const message = data.message;
      const callId = message?.toolCallId ?? message?.source?.callId;
      const call = calls.get(callId);
      if (!call) continue;
      calls.delete(callId);
      if (BINDS.has(call.name)) {
        if (call.readOnlyResume) continue;
        if (message.isError) {
          if (state.pendingBind?.callId === callId) state.pendingBind.status = "failed";
          continue;
        }
        if (state.phase === "blocked") continue;
        if (call.generation !== state.generation) {
          block("host_session_stale_binding", "A previous task's late binding cannot consume a new task authorization.");
          continue;
        }
        const body = hostBody(message);
        if (!call.bindingAllowed || !body || call.runDir !== directory(body.host_task.run_dir ?? body.run_dir)) {
          block("host_session_binding", "A successful prepare/resume has no authorized, matching task binding.");
          continue;
        }
        if (call.autoNew) {
          rememberOld(state.binding);
          state.binding = null; state.generation += 1;
        }
        const prior = state.binding;
        if (prior && (prior.host_task.task_id !== body.host_task.task_id
            || prior.contract_hash !== body.contract_hash)) {
          block("host_session_binding", "The active task identity changed.");
          continue;
        }
        state.binding = body; remember(body); state.phase = "active";
        state.authorization = null; state.pendingBind = null; state.error = null;
        state.lastBindingEventSeq = seq;
        state.terminalSeen = body.host_task.terminal === true || body.host_task.expired === true
          || body.host_task.remaining_s === 0;
      } else if (call.name === PREFIX + "operator_verify" && !message.isError && state.phase === "active") {
        const body = hostBody(message);
        if (body && body.host_task.task_id === state.binding.host_task.task_id
            && body.contract_hash === state.binding.contract_hash
            && directory(body.host_task.run_dir ?? body.run_dir) === directory(state.binding.host_task.run_dir ?? state.binding.run_dir)) {
          state.binding = body; remember(body);
          state.terminalSeen = body.host_task.terminal === true;
        }
      }
    }
  }
  return structuredClone(state);
}
