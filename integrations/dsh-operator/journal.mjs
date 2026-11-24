/**
 * 跨进程追加一条任务账本事件：**写入只有一处**。
 *
 * `rev` 是「读当前最大值 + 1」，落盘还要 `flock`；这两件事都在 Python 的
 * `task_journal.append` 里。JS 侧自己写会有两个问题：两侧各读各的会撞号；而
 * `proper-lockfile` 用 `mkdir` 取目录锁，与 `flock` 不是同一个互斥协议，同名的
 * `.lock` 提供不了共同互斥。所以这里只**提交**，不分配、不落盘。
 *
 * 失败**不抛**：CLI 不在、账本拒绝，都不该让模型预算门崩掉。调用方按回执决定要不要
 * 报出来 —— 静默吞掉和崩溃都不可接受。
 */
import { spawn } from "node:child_process";
import { lstatSync, readFileSync } from "node:fs";
import { join } from "node:path";

export const JOURNAL_FILE = "journal.jsonl";
export const JOURNAL_SCHEMA = "operator-task-journal-v2";
const BOUND_BYTES = 4 * 1024 * 1024;
const TASK_EVENTS = new Set(["task_prepared", "task_activated", "task_closed"]);
const JOB_EVENTS = new Set(["job_queued", "job_started", "job_finished"]);
const BUDGET_EVENTS = new Set(["budget_opened", "budget_reserved", "budget_settled"]);
const KINDS = new Set([...TASK_EVENTS, ...JOB_EVENTS, ...BUDGET_EVENTS]);
const OUTCOMES = new Set(["completed", "failed", "cancelled"]);
const RESULTS = new Set(["none", "received", "unknown"]);
const RESULT_REASONS = new Set(["receipt_absent", "invocation_mismatch", "invalid_timestamp", "stopped_before_receipt"]);
const OPERATIONS = new Set(["verify", "bench", "profile", "repair"]);
const TERMINAL = new Set(["completed", "failed", "cancelled"]);

function validateSequence(events) {
  let phase = "absent", taskId = "", budgetTaskId = "";
  const jobs = new Map();
  for (let index = 0; index < events.length; index += 1) {
    const event = events[index], line = index + 1, kind = event.kind;
    if (event.rev !== line) throw new Error(`journal_gap: 第 ${line} 行 rev=${event.rev}`);
    if (typeof event.task_id !== "string" || !event.task_id) {
      throw new Error(`journal_task_missing: 第 ${line} 行没有 task_id`);
    }
    if (!taskId) taskId = event.task_id;
    if (event.task_id !== taskId) throw new Error(`journal_task_mismatch: 第 ${line} 行属于另一个任务`);
    if (!KINDS.has(kind)) throw new Error(`journal_unknown_kind: 第 ${line} 行 kind=${kind}`);
    const snapshotTaskId = event.budget?.host_task?.task_id;
    if (snapshotTaskId !== undefined) {
      if (typeof snapshotTaskId !== "string" || !snapshotTaskId) {
        throw new Error(`journal_budget_mismatch: 第 ${line} 行预算快照缺 task_id`);
      }
      if (!budgetTaskId) budgetTaskId = snapshotTaskId;
      if (snapshotTaskId !== budgetTaskId) {
        throw new Error(`journal_budget_mismatch: 第 ${line} 行预算快照属于另一个账本`);
      }
    }
    if (phase === "closed") throw new Error(`journal_closed: 第 ${line} 行出现在封账之后`);
    if (kind === "task_prepared") {
      if (phase !== "absent") throw new Error(`journal_already_prepared: 第 ${line} 行重复 prepare`);
      phase = "prepared";
      continue;
    }
    if (kind === "task_activated") {
      if (phase !== "prepared" && phase !== "active") {
        throw new Error(`journal_not_prepared: 第 ${line} 行 activate 早于 prepare`);
      }
      phase = "active";
      continue;
    }
    if (kind === "task_closed") {
      if (!OUTCOMES.has(event.outcome)) throw new Error(`journal_unknown_outcome: 第 ${line} 行`);
      phase = "closed";
      continue;
    }
    if (JOB_EVENTS.has(kind)) {
      if (phase !== "active") throw new Error(`journal_not_activated: 第 ${line} 行 Job 早于 activate`);
      if (typeof event.job_id !== "string" || !event.job_id || !OPERATIONS.has(event.operation)) {
        throw new Error(`journal_bad_job: 第 ${line} 行缺少 job_id 或 operation`);
      }
      const job = jobs.get(event.job_id);
      if (job && TERMINAL.has(job)) throw new Error(`journal_job_closed: 第 ${line} 行 Job 已终态`);
      if (kind === "job_queued") {
        if (job) throw new Error(`journal_job_exists: 第 ${line} 行重复排队`);
        jobs.set(event.job_id, "queued");
      } else if (!job) {
        throw new Error(`journal_job_unknown: 第 ${line} 行 Job 从未排队`);
      } else if (kind === "job_started") {
        if (job !== "queued") throw new Error(`journal_bad_transition: 第 ${line} 行 start 不接 queued`);
        jobs.set(event.job_id, "running");
      } else {
        if (job !== "running") throw new Error(`journal_bad_transition: 第 ${line} 行 finish 不接 running`);
        if (!OUTCOMES.has(event.outcome) || !RESULTS.has(event.result)) {
          throw new Error(`journal_bad_result: 第 ${line} 行终态不完整`);
        }
        if (event.result === "unknown" && !RESULT_REASONS.has(event.result_reason)) {
          throw new Error(`journal_missing_result_reason: 第 ${line} 行未知结果缺原因`);
        }
        jobs.set(event.job_id, event.outcome);
      }
    }
  }
}

/** 读整条流。没有账本 ⇒ 空流；坏账本或非法序列 ⇒ 抛。 */
export function readJournal(runDir) {
  const path = join(runDir, JOURNAL_FILE);
  let stat;
  try {
    stat = lstatSync(path);
  } catch (error) {
    if (error.code === "ENOENT") return [];
    throw new Error(`journal_unusable: ${error.message}`);
  }
  if (stat.isSymbolicLink() || !stat.isFile()) {
    throw new Error("journal_unusable: 账本必须是普通文件，不是软链或目录");
  }
  if (typeof process.getuid === "function" && stat.uid !== process.getuid()) {
    throw new Error("journal_unusable: 账本属主不是本进程用户");
  }
  if (stat.size > BOUND_BYTES) throw new Error("journal_unusable: 账本超过大小上限");
  const text = readFileSync(path, "utf8");
  const events = text.split("\n").filter(line => line.trim()).map((line, index) => {
    let event;
    try {
      event = JSON.parse(line);
    } catch {
      throw new Error(`journal_corrupt: 第 ${index + 1} 行不是合法 JSON`);
    }
    if (event?.schema !== JOURNAL_SCHEMA) {
      throw new Error(`journal_schema_mismatch: 第 ${index + 1} 行 schema=${event?.schema}`);
    }
    return event;
  });
  validateSequence(events);
  return events;
}

/**
 * 预算状态：**只从任务账本读**（最后一条带 `budget.host_task` 的事件）。
 *
 * 宿主侧要的是 `task_id` / `policy_hash` / `started_*` / `max_elapsed_s` 这几个数，
 * 它们和 Python 的 `HostTaskBudget` 读的是同一条事件 —— 不存在第二份状态文件。
 */
export function readHostTask(runDir) {
  const state = readJournal(runDir).filter(event => event.budget?.host_task).pop()?.budget.host_task;
  if (!state || typeof state !== "object") {
    throw new Error("host_task_missing: 账本里没有预算状态（budget_opened 之后才有）");
  }
  return state;
}

const baseEnv = Object.fromEntries(["HOME", "PATH", "LANG", "SYSTEMROOT"]
  .filter(key => process.env[key] !== undefined).map(key => [key, process.env[key]]));
let runner = {
  command: process.env.OPERATOR_AGENT_CLI || "operator-agent",
  args: [],
  timeoutMs: 20000,
  env: baseEnv,
};
const queues = new Map(), pending = new Map();
const MAX_PENDING = 64;

/** 由插件入口注入与 `/op` 相同的 command/args/timeout 和已清洗 env。 */
export function configureJournalRunner(next = {}) {
  runner = { ...runner, ...next };
}

function submit(runDir, event, config) {
  return new Promise(resolve => {
    let stdout = "", stderr = "", settled = false, timer;
    const finish = receipt => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve(receipt);
    };
    let child;
    try {
      child = spawn(config.command, [...config.args, "journal-append", "--run-dir", runDir],
        { env: config.env, stdio: ["pipe", "pipe", "pipe"] });
    } catch (error) {
      finish({ ok: false, error: "journal_unavailable", detail: String(error?.message ?? error) });
      return;
    }
    timer = setTimeout(() => child.kill(), config.timeoutMs);
    child.stdout.on("data", chunk => {
      stdout += chunk;
      if (stdout.length > 1024 * 1024) child.kill();
    });
    child.stderr.on("data", chunk => { stderr += chunk; });
    child.stdin.on("error", error => finish({
      ok: false, error: "journal_unavailable", detail: String(error?.message ?? error) }));
    child.on("error", error => finish({
      ok: false, error: "journal_unavailable", detail: String(error?.message ?? error) }));
    child.on("close", () => {
      const out = stdout.trim();
      if (!out) {
        finish({ ok: false, error: "journal_unavailable",
          detail: stderr.trim() || `journal CLI exited without a receipt` });
        return;
      }
      try {
        const body = JSON.parse(out);
        if (body?.ok) finish({ ok: true, event: body.event });
        else finish({ ok: false, error: body?.error || "journal_refused", detail: body?.detail });
      } catch {
        finish({ ok: false, error: "journal_unreadable", detail: out.slice(0, 400) });
      }
    });
    child.stdin.end(JSON.stringify(event));
  });
}

/** 按任务串行提交；同一任务的预留、结算、封账不会乱序。 */
export function appendJournal(runDir, event) {
  if (typeof runDir !== "string" || !runDir) {
    return Promise.resolve({ ok: false, error: "journal_run_dir" });
  }
  if ((pending.get(runDir) ?? 0) >= MAX_PENDING) {
    return Promise.resolve({ ok: false, error: "journal_queue_full" });
  }
  const config = { ...runner, args: [...runner.args], env: { ...runner.env } };
  const previous = queues.get(runDir) ?? Promise.resolve();
  const current = previous.catch(() => {}).then(() => submit(runDir, event, config));
  queues.set(runDir, current);
  pending.set(runDir, (pending.get(runDir) ?? 0) + 1);
  current.finally(() => {
    const left = (pending.get(runDir) ?? 1) - 1;
    if (left) pending.set(runDir, left);
    else pending.delete(runDir);
    if (queues.get(runDir) === current) queues.delete(runDir);
  }).catch(() => {});
  return current;
}

/** 卸载或封账前等待已入队提交全部落账。 */
export async function drainJournal() {
  while (queues.size) await Promise.all([...queues.values()].map(queue => queue.catch(() => {})));
}