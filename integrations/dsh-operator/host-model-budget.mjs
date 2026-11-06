/** Native DSH task binding and pre-dispatch gates; never an agent loop. */
import { existsSync, readFileSync, readdirSync, lstatSync } from "node:fs";
import { isAbsolute, resolve } from "node:path";
import { randomUUID } from "node:crypto";
import { performance } from "node:perf_hooks";
import { HostModelLedger, HostModelBudgetError } from "./host-model-ledger.mjs";
import { recoverHostSession, newTaskReceipt, canAutoStartTask } from "./host-session-lifecycle.mjs";

const PREFIX = "mcp__operator__";
const TASK_TOOLS = new Set(["operator_prepare", "operator_resume", "operator_context",
  "operator_verify", "operator_status", "operator_cancel", "operator_capabilities"].map(name => PREFIX + name));
const BIND_TOOLS = new Set([PREFIX + "operator_prepare", PREFIX + "operator_resume"]);
const USAGE_FIELDS = ["inputTokens", "outputTokens", "totalTokens", "cacheReadTokens", "cacheWriteTokens"];
const text = content => Array.isArray(content)
  ? content.filter(block => block?.type === "text").map(block => block.text).join("\n") : "";

export function decodeHostResult(value) {
  try {
    const body = JSON.parse(text(value?.content));
    return body?.ok === true && body.workflow === "host" ? body : null;
  } catch { return null; }
}

/** Recover only this session's own successful durable binding, never a fork's. */
export function recoverHostBinding(events, inheritedEventCount = 0) {
  const calls = new Map();
  let binding;
  for (const event of events.slice(inheritedEventCount)) {
    const data = event.data;
    if (event.type === "tool/call" && data?.callId) calls.set(data.callId, data);
    if (event.type !== "tool/result") continue;
    const message = data?.message;
    const call = calls.get(message?.toolCallId ?? message?.source?.callId);
    if (!call || !BIND_TOOLS.has(call.name) || message.isError) continue;
    const body = decodeHostResult(message);
    if (body) binding = body;
    else throw new HostModelBudgetError("host_model_restore_binding", "A successful durable prepare/resume has no verifiable task binding; refusing an unbound restart.");
  }
  return binding;
}

function boundFacts(body) {
  const budget = body?.host_task;
  const runDir = budget?.run_dir ?? body?.run_dir;
  if (!budget?.task_id || !budget?.contract_hash || budget.contract_hash !== body.contract_hash
      || typeof runDir !== "string" || !isAbsolute(runDir)
      || !Number.isFinite(budget.deadline_epoch)
      || !Number.isFinite(budget.remaining_s) || budget.remaining_s < 0) {
    throw new HostModelBudgetError("host_model_binding", "A canonical prepared host task and deadline are required.");
  }
  return { runDir: resolve(runDir), taskId: budget.task_id, contractHash: budget.contract_hash,
    deadline: budget.deadline_epoch * 1000, remainingSeconds: budget.remaining_s };
}

function nativeUsage(value) {
  if (!value || typeof value !== "object") return null;
  return Object.fromEntries(USAGE_FIELDS.filter(key => value[key] !== undefined).map(key => [key, value[key]]));
}

export function installHostModelBudget(ctx, config = {}) {
  if (!config || typeof config !== "object" || Array.isArray(config) || Object.keys(config).length) {
    throw new Error("hostModelBudget accepts an empty object; task limits are frozen by operator_prepare.");
  }
  const sessions = new Map(), blocked = new Map(), restores = new Map();
  const childSessions = new Set(), nativeSignals = new WeakMap(), disposers = [];
  const READ_ONLY = new Set(["operator_capabilities", "operator_status", "operator_resume", "operator_context"].map(n => PREFIX + n));
  const read = path => {
    if (lstatSync(path).isSymbolicLink()) throw new Error("Task evidence must not be a symbolic link.");
    return JSON.parse(readFileSync(path, "utf8"));
  };
  const fail = (id, error) => {
    const err = new Error(`[${error.code ?? "host_model_binding_failed"}] ${error.message}`);
    blocked.set(id, err); ctx.logger?.error?.(`operator-tools: ${err.message}`); return err;
  };
  function taskState(binding) {
    const model = binding.ledger.summary(), task = read(resolve(binding.runDir, "host-task.json"));
    if (task.task_id !== binding.taskId || task.policy_hash !== model.policy_hash || !Array.isArray(task.records)) {
      throw new Error("The task identity or verification journal cannot be established.");
    }
    const unfinished = task.records.some(r => r.status === "reserved") || model.unfinished > 0;
    let cleanupPending = unfinished;
    const calls = resolve(binding.runDir, "tool-calls");
    if (existsSync(calls)) for (const name of readdirSync(calls)) {
      const path = resolve(calls, name, "run-status.json");
      if (!existsSync(path)) { cleanupPending = true; continue; }
      const status = read(path);
      if (!["done", "failed", "cancelled"].includes(status.phase)
          || status.cleanup?.worker_reaped !== true) cleanupPending = true;
    }
    const final = task.records.find(r => r.purpose === "final");
    return { model, task, cleanupPending, terminal: !!final && final.status !== "reserved" };
  }
  function attach(id, body, restoring, execute) {
    const facts = boundFacts(body), old = sessions.get(id)?.binding;
    const ledger = new HostModelLedger(facts.runDir, facts);
    let summary;
    if (execute) {
      const exists = existsSync(resolve(facts.runDir, "host-model-budget.json"));
      if (exists && restoring) {
        const prior = ledger.summary();
        if (prior.unfinished) ledger.abandon(prior.lease.session_id, prior.lease.id, { checked_dead_pid: prior.lease.owner_pid });
      }
      summary = ledger.bind(id, { reattach: exists });
      const clock = ledger.clock({ remainingSeconds: facts.remainingSeconds, deadlineEpoch: facts.deadline / 1000, restoring });
      return { ...facts, ledger, lease: summary.lease, monotonicDeadline: Math.min(
        old?.taskId === facts.taskId ? old.monotonicDeadline ?? Infinity : Infinity, performance.now() + clock.remaining_ms) };
    }
    return { ...facts, ledger }; // discussion never takes a dispatch lease or recreates time
  }
  function replay(id, data, restoring = false) {
    const previous = data.state;
    const next = recoverHostSession(data.events, data.inherited);
    if (next.error) throw fail(id, new HostModelBudgetError(next.error.code, next.error.detail));
    // Budget stops have native turn/end records, but no new plugin event. The
    // frozen ledger determines the stopped scope after restart, never its prose.
    if (data.closed && next.phase === "active" && next.binding?.host_task.task_id === data.closed) next.phase = "discussion";
    data.state = next;
    if (next.binding) {
      const changed = data.binding?.taskId !== next.binding.host_task.task_id;
      if (changed || (next.phase === "active" && !data.binding?.lease)) {
        let execute = next.phase === "active";
        const readonly = attach(id, next.binding, restoring, false);
        if (execute && !next.turnRunning) {
          const facts = taskState(readonly);
          const spent = facts.model.remaining_calls <= 0 || facts.model.remaining_output_token_reservations <= 0
            || facts.model.provider_output_cap_violation;
          if (!facts.cleanupPending && (facts.terminal || spent || readonly.ledger.clock({ readonly: true }).expired)) {
            next.phase = "discussion"; data.closed = readonly.taskId; execute = false;
          }
        }
        data.binding = execute ? attach(id, next.binding, restoring, true) : readonly;
      }
      if (next.phase === "discussion") data.closed = next.binding.host_task.task_id;
    } else if (next.phase === "new_authorized") {
      data.previousBinding = data.binding ?? data.previousBinding;
      data.binding = null; data.closed = null;
    }
    if (next.phase === "active" && !next.turnRunning && data.binding) {
      const facts = taskState(data.binding);
      let expired = false;
      if (!facts.terminal) expired = data.binding.ledger.clock({ readonly: true }).remaining_ms <= 0;
      if (!facts.cleanupPending && (facts.terminal || expired || facts.model.remaining_calls <= 0
          || facts.model.remaining_output_token_reservations <= 0 || facts.model.provider_output_cap_violation)) {
        next.phase = "discussion"; data.closed = data.binding.taskId;
      }
    }
    return previous;
  }
  async function restore(agent, signal) {
    const id = agent.session.id;
    if (!restores.has(id)) restores.set(id, (async () => {
      const observation = await ctx.sessionQuery.observeSession(id, { signal, projectionMode: "none" });
      try {
        if (observation.header?.origin === "subagent" || observation.header?.parentSession !== undefined) childSessions.add(id);
        const data = { events: [...observation.events], inherited: observation.inheritedEventCount ?? 0,
          state: null, binding: null, previousBinding: null, closed: null };
        sessions.set(id, data); replay(id, data, true);
      } finally { observation[Symbol.dispose](); }
    })().catch(error => { throw fail(id, error); }));
    await restores.get(id);
    return sessions.get(id);
  }
  function remainingTime(binding) {
    const persisted = binding.ledger.clock().remaining_ms, mono = performance.now();
    const remaining = Math.min(persisted, binding.deadline - Date.now(), binding.monotonicDeadline - mono);
    binding.monotonicDeadline = Math.min(binding.monotonicDeadline, mono + remaining);
    if (remaining <= 0) throw new HostModelBudgetError("host_task_deadline", "Task wall clock exhausted before model dispatch.");
    return remaining;
  }
  function stopMessage(data, error) {
    const b = data.binding;
    const reason = ({ host_task_deadline: "原任务截止时间已到，不会自动续时。",
      host_model_call_budget: "原任务的模型调用额度已用完。",
      host_model_token_budget: "原任务的输出 token 预留额度已用完。",
      host_model_provider_cap: "服务商输出超过预留上限，已停止后续任务派发。" })[error.code] ?? "任务执行条件不再满足。";
    return new HostModelBudgetError(error.code ?? "host_task_stopped", [
      "本任务已停止后续模型派发；会话仍可用于讨论。", reason,
      `任务目录：${b?.runDir ?? "未知"}`,
      "本轮停止后可继续提问；下一轮直接说明要测试的新任务即可，DSH 会使用独立目录和预算。",
      `查看已有结果：/op status ${b?.runDir ?? "<run_dir>"}`,
      "原任务的截止和消耗保留，普通讨论按 DSH 正常计费。",
    ].join("\n"));
  }
  const listener = (session, event) => {
    const id = session.id, data = sessions.get(id);
    if (!data) return;
    if (event.seq !== undefined && data.events.some(e => e.seq === event.seq)) return;
    data.events.push(event);
    try { replay(id, data); } catch (error) { fail(id, error); }
  };
  disposers.push(ctx.on("session/event", listener));
  disposers.push(ctx.on("agent/created", async ({ agent, signal }) => restore(agent, signal)));
  disposers.push(ctx.on("agent/request", async ({ agent, signal }, next) => {
    if (!agent?.session?.id) return next();
    const id = agent.session.id, data = await restore(agent, signal);
    if (blocked.has(id)) throw blocked.get(id);
    if (signal) nativeSignals.set(signal, { sessionId: id, agent });
    const options = await next();
    if (data.state.phase !== "active" || !data.binding) return options;
    const b = data.binding, state = b.ledger.summary();
    try {
      remainingTime(b);
      if (state.remaining_calls <= 0) throw new HostModelBudgetError("host_model_call_budget", "Task model calls exhausted before dispatch.");
      if (state.remaining_output_token_reservations <= 0) throw new HostModelBudgetError("host_model_token_budget", "Task output reservations exhausted before dispatch.");
      if (state.provider_output_cap_violation) throw new HostModelBudgetError("host_model_provider_cap", "Provider output exceeded the reserved cap.");
    } catch (error) { data.stopped = true; throw stopMessage(data, error); }
    return { ...options, maxTokens: Math.min(options.maxTokens ?? state.limits.max_output_tokens_per_call,
      state.limits.max_output_tokens_per_call, state.remaining_output_token_reservations) };
  }));
  disposers.push(ctx.tools.guard(exec => {
    const id = exec.agent?.session.id, data = sessions.get(id);
    if (blocked.has(id)) return blocked.get(id).message;
    const developing = BIND_TOOLS.has(exec.name) || exec.name === PREFIX + "operator_context" || exec.name === PREFIX + "operator_verify";
    if (childSessions.has(id) && developing) return "算子任务仅支持主会话，不能通过子代理或分支开启执行。";
    const state = data?.state;
    if (!state) return;
    const run = typeof exec.arguments?.run_dir === "string" && isAbsolute(exec.arguments.run_dir) ? resolve(exec.arguments.run_dir) : null;
    if (state.phase === "discussion") {
      if (exec.name === PREFIX + "operator_prepare" && canAutoStartTask(state)) {
        if (!run || state.runDirs.includes(run) || existsSync(resolve(run, "host-task-policy.json"))) return "下一项任务须使用全新绝对目录，旧任务不能重开。";
        if (state.pendingBind && ((state.pendingBind.status === "pending" && state.pendingBind.callId !== exec.callId) || state.pendingBind.runDir !== run)) return "上次 prepare 尚未确认或已锁定目录，不能换目录补额度。";
        try { if (taskState(data.binding).cleanupPending) return "旧任务尚有未结算调用或未回收 worker，不能开启下一项任务。"; }
        catch { return "旧任务清理状态未知，不能开启下一项任务。"; }
        return;
      }
      if (!READ_ONLY.has(exec.name)) return "当前为结果讨论模式；下一轮说明新任务即可自动切换，也可手动 /op new。旧任务不会获得新额度。";
      return;
    }
    if (state.phase === "active") {
      if (!TASK_TOOLS.has(exec.name)) return "A bound operator task uses only its operator tools; child agents, shell and alternate workflows are outside this task budget.";
      if (exec.name !== PREFIX + "operator_capabilities" && run !== data.binding?.runDir) return "请保留当前任务目录；开始独立的新任务需用户直接输入 /op new。";
      return;
    }
    if (exec.name === PREFIX + "operator_context" || exec.name === PREFIX + "operator_verify") return "Prepare or explicitly resume the original host task before using its development tools.";
    if (BIND_TOOLS.has(exec.name)) {
      if (!run) return "任务目录必须是绝对路径。";
      const lock = state.pendingBind?.runDir ?? state.authorization?.runDir;
      if (lock && lock !== run) return "本次任务入口已绑定目标目录，不能改目录补额度。";
      if (state.pendingBind?.status === "pending" && (state.pendingBind.callId !== exec.callId || state.pendingBind.name !== exec.name)) return "上次 prepare/resume 的结果尚未确认，不能并行重试任务入口。";
      if (state.oldRunDirs.includes(run)) return "旧任务目录不能重新 prepare 取得额度；请只读查看其结果。";
      if (state.phase === "new_authorized" && (exec.name !== PREFIX + "operator_prepare" || existsSync(resolve(run, "host-task-policy.json")))) return "新任务入口仅允许 prepare 一个全新目录；旧任务不会重开。";
    }
  }));
  // Bind synchronously before the next request. Native tool/result then supplies
  // the durable replay record, including the phase and final completion hint.
  disposers.push(ctx.on("tools/result", (exec, result) => {
    if (!BIND_TOOLS.has(exec.name) || result.isError) return;
    const id = exec.agent?.session.id, data = sessions.get(id);
    if (!data) return;
    const pending = data.state.pendingBind;
    const pendingResult = pending?.status === "pending" && pending.callId === exec.callId && pending.name === exec.name;
    if (["discussion", "new_authorized", "blocked"].includes(data.state.phase) && exec.name === PREFIX + "operator_resume") return;
    if (data.state.phase === "blocked") return;
    try {
      if (data.state.phase !== "active" && (!pendingResult || (data.state.phase === "discussion" && !pending.autoNew))) {
        throw new HostModelBudgetError("host_model_result", "A successful task binding has no matching authorized pending call.");
      }
      const body = decodeHostResult(result.value ?? result);
      if (!body) throw new HostModelBudgetError("host_model_result", "Successful prepare/resume returned no verifiable host task binding.");
      if (pendingResult && boundFacts(body).runDir !== pending.runDir) throw new HostModelBudgetError("host_model_result", "The task binding result does not match its pending call directory.");
      if (data.binding && data.binding.taskId !== body.host_task.task_id && !(pendingResult && pending.autoNew)) throw new Error("The active task identity changed.");
      data.binding = attach(id, body, false, true);
      // Native replay still owns authorization consumption and durable phase.
      data.state.phase = "active";
    } catch (error) { fail(id, error); }
  }));
  disposers.push(ctx.on("llm/stream", (options, next) => {
    const id = options.sessionId, data = sessions.get(id);
    if (!data?.binding || data.state.phase !== "active" || options.purpose !== undefined) return next();
    const b = data.binding;
    return (async function* () {
      if (blocked.has(id)) throw blocked.get(id);
      const native = nativeSignals.get(options.signal);
      if (native?.sessionId !== id) throw fail(id, new HostModelBudgetError("host_model_native_request", "Cannot identify the bound main-agent request."));
      let record, status = "failed", usage = null, timer;
      try {
        remainingTime(b);
        record = b.ledger.reserve(id, b.lease.id, { requestId: randomUUID(), maxTokens: options.maxTokens, provider: options.provider, model: options.model });
        timer = setTimeout(() => native.agent.cancel({ kind: "user" }), Math.max(1, remainingTime(b))); timer.unref?.();
        for await (const chunk of next()) {
          if (chunk.type === "usage") usage = nativeUsage(chunk.usage);
          if (chunk.type === "finish" && ["stop", "tool-calls"].includes(chunk.reason?.kind)) status = "answered";
          yield chunk;
        }
      } finally {
        clearTimeout(timer);
        if (options.signal?.aborted) status = "cancelled";
        if (record) try { b.ledger.finish(id, b.lease.id, record.id, { status, usage }); }
        catch (error) { throw fail(id, error); }
      }
    })();
  }));
  const newTask = async ({ agent, commandId, runDir, signal }) => {
    if (!agent?.session?.id || !commandId) throw new Error("新任务入口需要 DSH 的原生用户命令。");
    const id = agent.session.id, data = await restore(agent, signal);
    if (blocked.has(id)) throw blocked.get(id);
    if (childSessions.has(id)) throw new Error("请在主会话开启新任务。");
    if (agent.status === "running") throw new Error("当前轮仍在运行，请先停止并等执行结算后再开启新任务。");
    if (data.state.pendingBind?.status === "pending") throw new Error("上次 prepare/resume 的结果尚未确认，请先核对原任务；不能切换目录开启新任务。");
    const command = data.events.findLast(e => e.type === "command/run" && e.data?.commandId === commandId);
    if (command?.data?.name !== "op" || command.data.source?.kind !== "user" || !/^new(?:\s+[\s\S]+)?$/.test(command.data.args?.trim() ?? "")) throw new Error("缺少可核对的原生用户新任务命令。");
    const prior = data.binding ?? data.previousBinding;
    if (prior && taskState(prior).cleanupPending) throw new Error("旧任务还有未结算模型或未清理验证实例；先用 /op status 查询并取消，不能并行新开执行。");
    if (runDir && (!isAbsolute(runDir) || data.state.runDirs.includes(resolve(runDir)) || existsSync(resolve(runDir, "host-task-policy.json")))) throw new Error("新任务请指定全新绝对目录，不能复用旧任务。");
    return newTaskReceipt(commandId, runDir);
  };
  const sessionStatus = async ({ agent, signal }) => {
    const data = await restore(agent, signal), b = data.binding ?? data.previousBinding;
    const label = ({ initial: "等待首项任务", active: "当前任务执行范围内", discussion: "结果讨论",
      new_authorized: "新任务入口已开启，等待 prepare", blocked: "任务绑定异常" })[data.state.phase] ?? "未知";
    const lines = [`✓ 会话任务状态：${label}`, "新任务：直接说明下一项任务，DSH 自动准备独立目录；/op new 为可选手动入口。普通讨论按 DSH 正常计费。"];
    if (b) lines.push(`上次任务：${b.runDir}`, `结果与预算：/op status ${b.runDir}`);
    return lines.join("\n");
  };
  const dispose = () => { for (const fn of disposers.reverse()) if (typeof fn === "function") fn(); };
  dispose.newTask = newTask; dispose.sessionStatus = sessionStatus;
  return dispose;
}
