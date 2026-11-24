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