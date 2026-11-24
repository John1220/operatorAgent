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
import { execFileSync } from "node:child_process";
import { lstatSync, readFileSync } from "node:fs";
import { join } from "node:path";

export const JOURNAL_FILE = "journal.jsonl";
export const JOURNAL_SCHEMA = "operator-task-journal-v2";
const BOUND_BYTES = 4 * 1024 * 1024;

/** 读整条流。没有账本 ⇒ 空流（那是状态，不是错误）；坏账本 ⇒ 抛。 */
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
  return text.split("\n").filter(line => line.trim()).map((line, index) => {
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

let runner = {
  command: process.env.OPERATOR_AGENT_CLI || "operator-agent",
  args: [],
  timeoutMs: 20000,
};

/** 由插件入口注入它自己的 CLI 配置（与 `/op` 命令用的是同一套 command/args/timeout）。 */
export function configureJournalRunner(next = {}) {
  runner = { ...runner, ...next };
}

/** 提交一条事件。返回 `{ok:true,event}` 或 `{ok:false,error,detail}`。 */
export function appendJournal(runDir, event) {
  if (typeof runDir !== "string" || !runDir) return { ok: false, error: "journal_run_dir" };
  const { command, args, timeoutMs } = runner;
  let out;
  try {
    out = execFileSync(command, [...args, "journal-append", "--run-dir", runDir],
      { input: JSON.stringify(event), encoding: "utf8", timeout: timeoutMs });
  } catch (error) {
    // 账本拒绝时 CLI 退出码是 2，stdout 里仍有结构化回包 —— 那才是要读的东西。
    out = typeof error?.stdout === "string" ? error.stdout.trim() : "";
    if (!out) {
      return { ok: false, error: "journal_unavailable", detail: String(error?.message ?? error) };
    }
  }
  try {
    const body = JSON.parse(out);
    if (body?.ok) return { ok: true, event: body.event };
    return { ok: false, error: body?.error || "journal_refused", detail: body?.detail };
  } catch {
    return { ok: false, error: "journal_unreadable", detail: String(out).slice(0, 400) };
  }
}