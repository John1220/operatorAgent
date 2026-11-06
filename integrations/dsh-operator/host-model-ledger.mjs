/** Durable reservations for native host model requests after task preparation. */
import { createHash, randomUUID } from "node:crypto";
import { execFileSync } from "node:child_process";
import {
  closeSync, constants, fsyncSync, linkSync, lstatSync, mkdirSync, openSync,
  readFileSync, realpathSync, renameSync, rmdirSync, unlinkSync, writeFileSync,
} from "node:fs";
import { dirname, isAbsolute, join, resolve } from "node:path";

export const MODEL_BUDGET_SCHEMA = "operator-host-model-budget-v1";
export const MODEL_BUDGET_FILE = "host-model-budget.json";
export const MODEL_BUDGET_SCOPE = "native_main_model_dispatches_after_successful_prepare";
const LOCK = ".host-model-budget.lock";
const HASH = /^[0-9a-f]{64}$/;
const TASK = /^[0-9a-f]{32}$/;
const INSTANCE = randomUUID();
const STATUSES = new Set(["reserved", "answered", "failed", "cancelled", "abandoned"]);
const LIMITS = ["max_model_calls", "max_output_tokens_per_call", "total_output_token_reservations"];
const USAGE = ["inputTokens", "outputTokens", "totalTokens", "cacheReadTokens", "cacheWriteTokens"];
const CLOCK_SCHEMA = "operator-host-model-clock-v1";
let processBootId;

export class HostModelBudgetError extends Error {
  constructor(code, detail = code) {
    super(detail);
    this.name = "HostModelBudgetError";
    this.code = code;
    this.detail = detail;
  }
}

function refuse(code, detail) { throw new HostModelBudgetError(code, detail); }
function object(value) { return value !== null && typeof value === "object" && !Array.isArray(value); }
function integer(value, minimum = 0) { return Number.isSafeInteger(value) && value >= minimum; }
function string(value, limit = 256) {
  return typeof value === "string" && value.length > 0 && value.length <= limit
    && !/[\u0000-\u001f\ud800-\udfff]/u.test(value);
}

function canonical(value) {
  if (Array.isArray(value)) return "[" + value.map(canonical).join(",") + "]";
  if (object(value)) return "{" + Object.keys(value).sort()
    .map(key => JSON.stringify(key) + ":" + canonical(value[key])).join(",") + "}";
  return JSON.stringify(value);
}
function digest(value) { return createHash("sha256").update(canonical(value)).digest("hex"); }
function bytesHash(value) { return createHash("sha256").update(value).digest("hex"); }

function bootIdentity() {
  if (processBootId) return processBootId;
  let value;
  try {
    if (process.platform === "linux") value = readFileSync("/proc/sys/kernel/random/boot_id", "utf8").trim();
    else if (process.platform === "darwin") value = execFileSync("/usr/sbin/sysctl",
      ["-n", "kern.bootsessionuuid"], { encoding: "utf8", timeout: 2000,
        stdio: ["ignore", "pipe", "pipe"] }).trim();
  } catch { refuse("host_model_clock", "Kernel boot identity cannot be established; task time remains unknown."); }
  if (typeof value !== "string" || !/^[0-9a-f-]{36}$/i.test(value)) {
    refuse("host_model_clock", "This platform has no supported kernel boot identity.");
  }
  processBootId = value.toLowerCase();
  return processBootId;
}

function uid() {
  if (typeof process.getuid !== "function" || constants.O_NOFOLLOW === undefined) {
    refuse("host_model_platform", "Host task model reservations require POSIX file ownership.");
  }
  return process.getuid();
}

function ordinary(path, { directory = false, mutable = false } = {}) {
  const stat = lstatSync(path);
  if (stat.isSymbolicLink() || stat.uid !== uid()
      || (directory ? !stat.isDirectory() : !stat.isFile())
      || (mutable && (stat.mode & 0o077) !== 0)) {
    refuse("host_model_journal", "An owned ordinary task directory/private ledger file is required.");
  }
  return stat;
}

function readObject(path, { mutable = false } = {}) {
  ordinary(path, { mutable });
  const fd = openSync(path, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK);
  let bytes;
  try {
    const stat = ordinary(path, { mutable });
    if (stat.size > 4 * 1024 * 1024) refuse("host_model_journal", "Task journal exceeds its bound.");
    bytes = readFileSync(fd);
  } finally { closeSync(fd); }
  let value;
  try {
    value = JSON.parse(bytes.toString("utf8"));
    if (mutable && bytes.toString("utf8") !== canonical(value) + "\n") {
      refuse("host_model_journal", "Private model journal must retain canonical bytes.");
    }
  }
  catch { refuse("host_model_journal", "Unreadable task journal."); }
  if (!object(value)) refuse("host_model_journal", "Task journal must be an object.");
  return { value, bytes };
}

function publish(path, value, { initial = false } = {}) {
  const temporary = path + ".pending-" + randomUUID();
  const fd = openSync(temporary, constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL
    | constants.O_NOFOLLOW, 0o600);
  try { writeFileSync(fd, canonical(value) + "\n"); fsyncSync(fd); }
  finally { closeSync(fd); }
  try {
    if (initial) linkSync(temporary, path);
    else { ordinary(path, { mutable: true }); renameSync(temporary, path); }
    const parent = openSync(dirname(path), constants.O_RDONLY | constants.O_DIRECTORY | constants.O_NOFOLLOW);
    try { fsyncSync(parent); } finally { closeSync(parent); }
  } finally {
    try { unlinkSync(temporary); } catch (error) { if (error.code !== "ENOENT") throw error; }
  }
}

function normalizeUsage(usage) {
  if (!object(usage) || Object.keys(usage).some(key => !USAGE.includes(key))
      || Object.values(usage).some(value => !integer(value))
      || !Object.values(usage).some(value => value > 0)) return null;
  if (usage.inputTokens !== undefined && usage.outputTokens !== undefined
      && usage.totalTokens !== undefined) {
    const base = usage.inputTokens + usage.outputTokens;
    const cache = (usage.cacheReadTokens ?? 0) + (usage.cacheWriteTokens ?? 0);
    if (!integer(base + cache) || ![base, base + cache].includes(usage.totalTokens)) return null;
  }
  return Object.fromEntries(USAGE.filter(key => usage[key] !== undefined).map(key => [key, usage[key]]));
}

/** All operations are synchronous so a tool-result observer can bind before return. */
export class HostModelLedger {
  constructor(runDir, { taskId, contractHash } = {}) {
    if (typeof runDir !== "string" || !isAbsolute(runDir) || !TASK.test(taskId ?? "")
        || !HASH.test(contractHash ?? "")) {
      refuse("host_model_binding", "Absolute run_dir, task_id and contract_hash are required.");
    }
    ordinary(resolve(runDir), { directory: true });
    this.runDir = realpathSync(runDir);
    this.taskId = taskId;
    this.contractHash = contractHash;
  }

  _locked(action) {
    ordinary(this.runDir, { directory: true });
    const lock = join(this.runDir, LOCK);
    try { mkdirSync(lock, { mode: 0o700 }); }
    catch (error) {
      if (error.code === "EEXIST") refuse("host_model_locked",
        "Model journal is locked; stale ownership is unknown and is not automatically cleared.");
      throw error;
    }
    const nonce = randomUUID();
    const owner = join(lock, "owner.json");
    try {
      const fd = openSync(owner, constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL
        | constants.O_NOFOLLOW, 0o600);
      try { writeFileSync(fd, canonical({ nonce, pid: process.pid }) + "\n"); fsyncSync(fd); }
      finally { closeSync(fd); }
      return action();
    } finally {
      // Never remove a replaced lock or a lock whose owner cannot be established.
      const saved = readObject(owner, { mutable: true }).value;
      if (saved.nonce !== nonce) refuse("host_model_locked", "Model journal lock ownership changed.");
      unlinkSync(owner);
      rmdirSync(lock);
    }
  }

  _policy() {
    const policy = readObject(join(this.runDir, "host-task-policy.json"));
    const host = readObject(join(this.runDir, "host-task.json")).value;
    if (policy.value.schema !== "operator-host-task-budget-v1"
        || policy.value.contract_hash !== this.contractHash || host.task_id !== this.taskId
        || !HASH.test(host.policy_hash ?? "")) {
      refuse("host_model_binding", "Prepared task identity differs from the model binding.");
    }
    const limits = Object.fromEntries(LIMITS.map(key => [key, policy.value.limits?.[key]]));
    if (!integer(limits.max_model_calls, 1) || limits.max_model_calls > 1000
        || !integer(limits.max_output_tokens_per_call, 1) || limits.max_output_tokens_per_call > 40960
        || !integer(limits.total_output_token_reservations, 1)
        || limits.total_output_token_reservations > 40960000) {
      refuse("host_model_policy_missing", "Prepared task has no supported frozen model budget; do not invent credit.");
    }
    // New Python policies are published as canonical UTF-8 JSON plus one LF.
    // Hash raw canonical bytes to retain Python float spelling (e.g. 600.0).
    const canonicalPolicy = policy.bytes.at(-1) === 10 ? policy.bytes.subarray(0, -1) : policy.bytes;
    if (bytesHash(canonicalPolicy) !== host.policy_hash) {
      refuse("host_model_binding", "Frozen policy bytes differ from the Python policy digest.");
    }
    return { policy_hash: host.policy_hash, policy_bytes_sha256: bytesHash(policy.bytes), limits };
  }

  _hostClock() {
    const policy = readObject(join(this.runDir, "host-task-policy.json")).value;
    const state = readObject(join(this.runDir, "host-task.json")).value;
    if (!Number.isFinite(policy.limits?.wall_clock_s) || policy.limits.wall_clock_s <= 0
        || !Number.isFinite(state.started_epoch) || state.started_epoch <= 0
        || !Number.isFinite(state.started_monotonic) || state.started_monotonic < 0
        || !Number.isFinite(state.max_elapsed_s) || state.max_elapsed_s < 0) {
      refuse("host_model_clock", "Live prepared task clock metadata is required.");
    }
    const limitMs = Math.floor(policy.limits.wall_clock_s * 1000);
    const deadlineMs = Math.floor((state.started_epoch + policy.limits.wall_clock_s) * 1000);
    const hostElapsedMs = Math.ceil(state.max_elapsed_s * 1000);
    if (![limitMs, deadlineMs, hostElapsedMs].every(value => integer(value))) {
      refuse("host_model_clock", "Prepared task clock exceeds supported integer precision.");
    }
    return { limitMs, deadlineMs, hostElapsedMs,
      startedEpoch: String(state.started_epoch), startedMonotonic: String(state.started_monotonic) };
  }

  _clock(state, { remainingSeconds, deadlineEpoch, restoring = false } = {}) {
    if (typeof restoring !== "boolean") refuse("host_model_clock", "Clock restore mode must be explicit.");
    const live = this._hostClock();
    const bootId = bootIdentity();
    const now = process.hrtime.bigint();
    if (state.clock === null && !restoring && state.records.length === 0 && state.lease.generation === 1) {
      if (!Number.isFinite(remainingSeconds) || remainingSeconds < 0
          || !Number.isFinite(deadlineEpoch) || deadlineEpoch <= 0
          || Math.abs(deadlineEpoch * 1000 - live.deadlineMs) > 1
          || remainingSeconds * 1000 > live.limitMs + 1) {
        refuse("host_model_clock", "Initial clock needs a live prepared task remaining-time receipt.");
      }
      const remaining = Math.max(0, Math.min(Math.floor(remainingSeconds * 1000),
        live.limitMs - live.hostElapsedMs, live.deadlineMs - Date.now()));
      state.clock = { schema: CLOCK_SCHEMA, boot_id: bootId, anchor_hrtime_ns: String(now),
        last_hrtime_ns: String(now), remaining_ms_at_anchor: remaining, minimum_remaining_ms: remaining,
        max_elapsed_ms: 0, wall_limit_ms: live.limitMs, deadline_epoch_ms: live.deadlineMs,
        host_started_epoch: live.startedEpoch, host_started_monotonic: live.startedMonotonic };
    }
    const clock = state.clock;
    if (!object(clock) || clock.schema !== CLOCK_SCHEMA || typeof clock.boot_id !== "string"
        || !/^(?:0|[1-9][0-9]*)$/.test(clock.anchor_hrtime_ns ?? "")
        || !/^(?:0|[1-9][0-9]*)$/.test(clock.last_hrtime_ns ?? "")
        || ![clock.remaining_ms_at_anchor, clock.minimum_remaining_ms, clock.max_elapsed_ms,
          clock.wall_limit_ms, clock.deadline_epoch_ms].every(value => integer(value))) {
      refuse("host_model_clock_missing", "Persisted task clock is missing or invalid; an old transcript cannot recreate time.");
    }
    if (clock.boot_id !== bootId) refuse("host_model_clock", "Kernel boot identity changed; task time cannot be restored.");
    if (clock.host_started_epoch !== live.startedEpoch || clock.host_started_monotonic !== live.startedMonotonic
        || clock.wall_limit_ms !== live.limitMs || clock.deadline_epoch_ms !== live.deadlineMs) {
      refuse("host_model_clock", "Prepared task clock identity changed.");
    }
    const anchor = BigInt(clock.anchor_hrtime_ns), last = BigInt(clock.last_hrtime_ns);
    if (now < anchor || now < last) refuse("host_model_clock", "Host monotonic time moved backwards.");
    const elapsed = Number((now - anchor + 999999n) / 1000000n);
    if (!integer(elapsed)) refuse("host_model_clock", "Host elapsed time exceeds supported precision.");
    clock.max_elapsed_ms = Math.max(clock.max_elapsed_ms, elapsed);
    clock.last_hrtime_ns = String(now);
    clock.minimum_remaining_ms = Math.max(0, Math.min(clock.minimum_remaining_ms,
      clock.remaining_ms_at_anchor - clock.max_elapsed_ms, live.limitMs - live.hostElapsedMs,
      live.deadlineMs - Date.now()));
    return { schema: CLOCK_SCHEMA, task_id: this.taskId, contract_hash: this.contractHash,
      boot_id: bootId, remaining_ms: clock.minimum_remaining_ms,
      elapsed_ms: live.limitMs - clock.minimum_remaining_ms, expired: clock.minimum_remaining_ms === 0,
      deadline_epoch_ms: clock.deadline_epoch_ms,
      scope: "live_prepare_calibration_then_persistent_same_boot_host_monotonic_no_epoch_guess" };
  }

  _load() {
    const policy = this._policy();
    const state = readObject(join(this.runDir, MODEL_BUDGET_FILE), { mutable: true }).value;
    const { state_hash: checksum, ...body } = state;
    if (!HASH.test(checksum ?? "") || digest(body) !== checksum
        || state.schema !== MODEL_BUDGET_SCHEMA || state.task_id !== this.taskId
        || state.contract_hash !== this.contractHash || state.scope !== MODEL_BUDGET_SCOPE
        || state.policy_hash !== policy.policy_hash || state.policy_bytes_sha256 !== policy.policy_bytes_sha256
        || canonical(state.limits) !== canonical(policy.limits)) {
      refuse("host_model_binding", "Model journal checksum or frozen task policy changed.");
    }
    const lease = state.lease;
    if (!object(lease) || !string(lease.session_id) || !string(lease.id)
        || !integer(lease.owner_pid, 1) || !integer(lease.generation, 1)
        || !string(lease.owner_instance_id) || !integer(lease.bound_at)
        || !Array.isArray(state.lease_history) || !Array.isArray(state.abandonment_events)
        || !Array.isArray(state.records) || state.records.length > policy.limits.max_model_calls) {
      refuse("host_model_journal", "Invalid session lease or model records.");
    }
    const requests = new Set();
    let total = 0;
    for (const [index, record] of state.records.entries()) {
      if (!object(record) || record.id !== index + 1 || !STATUSES.has(record.status)
          || !string(record.session_id) || !string(record.lease_id) || !integer(record.owner_pid, 1)
          || !string(record.request_id) || requests.has(record.request_id)
          || !string(record.provider, 512) || !string(record.model, 512)
          || !integer(record.max_output_tokens_reserved, 1)
          || record.max_output_tokens_reserved > policy.limits.max_output_tokens_per_call
          || !integer(record.reserved_at)
          || (record.status === "reserved" && record.finished_at !== null)
          || (record.status !== "reserved" && !integer(record.finished_at))
          || (record.usage !== null && canonical(normalizeUsage(record.usage)) !== canonical(record.usage))) {
        refuse("host_model_journal", "Invalid model reservation or API usage counters.");
      }
      requests.add(record.request_id);
      total += record.max_output_tokens_reserved;
    }
    if (!integer(total) || total > policy.limits.total_output_token_reservations) {
      refuse("host_model_journal", "Model reservations exceed frozen credit.");
    }
    return state;
  }

  _write(state, initial = false) {
    const { state_hash: ignored, ...body } = state;
    publish(join(this.runDir, MODEL_BUDGET_FILE), { ...body, state_hash: digest(body) }, { initial });
  }

  _summary(state) {
    const reserved = state.records.reduce((sum, record) => sum + record.max_output_tokens_reserved, 0);
    const unfinished = state.records.filter(record => record.status === "reserved").length;
    const known = state.records.filter(record => record.usage !== null).length;
    const capViolation = state.records.some(record => record.usage?.outputTokens > record.max_output_tokens_reserved);
    return structuredClone({ ...state, calls_reserved: state.records.length,
      output_tokens_reserved: reserved, remaining_calls: state.limits.max_model_calls - state.records.length,
      remaining_output_token_reservations: state.limits.total_output_token_reservations - reserved,
      unfinished, api_usage_known_calls: known, api_usage_unknown_calls: state.records.length - known,
      provider_output_cap_violation: capViolation,
      scope: MODEL_BUDGET_SCOPE, preparation_dispatch: "excluded_before_successful_prepare",
      enforcement: "host_pre_dispatch_reservations", usage_scope: "native_validated_counters_no_cost_inference",
      cost: "unknown", provider_execution_stop: "unknown" });
  }

  _lease(state, sessionId, leaseId, requireCurrentProcess = false) {
    if (state.lease.session_id !== sessionId || state.lease.id !== leaseId
        || (requireCurrentProcess && (state.lease.owner_pid !== process.pid
          || state.lease.owner_instance_id !== INSTANCE))) {
      refuse("host_model_lease", "This session or process no longer owns the model dispatch lease.");
    }
  }

  bind(sessionId, { reattach = false } = {}) {
    if (!string(sessionId) || typeof reattach !== "boolean") refuse("host_model_session", "A valid session identity is required.");
    return this._locked(() => {
      const policy = this._policy();
      let state;
      try { state = this._load(); }
      catch (error) {
        if (error.code !== "ENOENT") throw error;
        state = { schema: MODEL_BUDGET_SCHEMA, task_id: this.taskId, contract_hash: this.contractHash,
          ...policy, scope: MODEL_BUDGET_SCOPE, records: [], lease_history: [], abandonment_events: [], clock: null };
      }
      if (state.lease?.session_id === sessionId && state.lease.owner_pid === process.pid
          && state.lease.owner_instance_id === INSTANCE) return this._summary(state);
      if (state.lease) {
        if (!reattach) refuse("host_model_reattach_required", "Explicit reattachment is required; credit is preserved.");
        if (state.records.some(record => record.status === "reserved")) {
          refuse("host_model_in_flight", "Unfinished model reservations must retain unknown ownership until resolved.");
        }
        state.lease_history.push(state.lease);
      }
      const initial = !state.lease;
      state.lease = { id: randomUUID(), session_id: sessionId, owner_pid: process.pid,
        owner_instance_id: INSTANCE, generation: (state.lease?.generation ?? 0) + 1, bound_at: Date.now() };
      this._write(state, initial);
      return this._summary(this._load());
    });
  }

  summary() { return this._locked(() => this._summary(this._load())); }

  clock(options = {}) {
    const { readonly = false, ...calibration } = options;
    if (typeof readonly !== "boolean") refuse("host_model_clock", "Clock read mode must be explicit.");
    return this._locked(() => {
      const state = this._load();
      const value = this._clock(state, calibration);
      if (!readonly) this._write(state);
      return value;
    });
  }

  reserve(sessionId, leaseId, { requestId, maxTokens, provider, model } = {}) {
    if (!string(requestId) || !string(provider, 512) || !string(model, 512) || !integer(maxTokens, 1)) {
      refuse("host_model_request", "Bound request identity, route and positive maxTokens are required.");
    }
    return this._locked(() => {
      const state = this._load();
      this._lease(state, sessionId, leaseId, true);
      const clock = this._clock(state);
      if (clock.expired) {
        this._write(state);
        refuse("host_task_deadline", "Task wall clock exhausted before model reservation.");
      }
      const summary = this._summary(state);
      if (state.records.some(record => record.request_id === requestId)) refuse("host_model_duplicate", "Request already has a reservation.");
      if (summary.provider_output_cap_violation) refuse("host_model_provider_cap", "A provider exceeded the reserved output cap; further dispatch is stopped.");
      if (maxTokens > state.limits.max_output_tokens_per_call) refuse("host_model_output_limit", "Requested output exceeds the per-call limit.");
      if (summary.remaining_calls <= 0) refuse("host_model_call_budget", "Task model call budget is exhausted.");
      if (maxTokens > summary.remaining_output_token_reservations) refuse("host_model_token_budget", "Task output reservations are exhausted.");
      const record = { id: state.records.length + 1, request_id: requestId, session_id: sessionId,
        lease_id: leaseId, owner_pid: state.lease.owner_pid, provider, model,
        max_output_tokens_reserved: maxTokens, status: "reserved", reserved_at: Date.now(),
        finished_at: null, usage: null };
      state.records.push(record);
      this._write(state);
      return structuredClone(record);
    });
  }

  finish(sessionId, leaseId, reservationId, { status = "answered", usage = null } = {}) {
    if (!integer(reservationId, 1) || !["answered", "failed", "cancelled"].includes(status)) {
      refuse("host_model_reservation", "A model reservation and terminal status are required.");
    }
    return this._locked(() => {
      const state = this._load();
      const record = state.records[reservationId - 1];
      if (!record || record.session_id !== sessionId || record.lease_id !== leaseId || record.status !== "reserved") {
        refuse("host_model_reservation", "Reservation is absent, closed or owned by another lease.");
      }
      record.status = status;
      record.finished_at = Date.now();
      record.usage = normalizeUsage(usage);
      this._write(state);
      return this._summary(this._load());
    });
  }

  abandon(sessionId, leaseId, { checked_dead_pid: checkedPid } = {}) {
    return this._locked(() => {
      const state = this._load();
      this._lease(state, sessionId, leaseId);
      if (!integer(checkedPid, 1) || checkedPid !== state.lease.owner_pid || checkedPid === process.pid) {
        refuse("host_model_owner_unknown", "The old model owner's dead PID must be explicitly checked.");
      }
      try { process.kill(checkedPid, 0); }
      catch (error) {
        if (error.code !== "ESRCH") refuse("host_model_owner_unknown", "Owner liveness is unknown; reservations remain untouched.");
        const abandoned = [];
        for (const record of state.records) {
          if (record.session_id === sessionId && record.lease_id === leaseId && record.status === "reserved") {
            record.status = "abandoned";
            record.finished_at = Date.now();
            record.usage = null;
            abandoned.push(record.id);
          }
        }
        state.abandonment_events.push({ checked_dead_pid: checkedPid, checked_at: Date.now(),
          lease_id: leaseId, reservation_ids: abandoned, provider_execution_stop: "unknown" });
        this._write(state);
        return this._summary(this._load());
      }
      refuse("host_model_owner_unknown", "Owner PID is alive or reused; no abandonment is inferred.");
    });
  }
}
