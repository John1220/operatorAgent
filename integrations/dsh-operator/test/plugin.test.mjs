// 这个插件的守卫：**注册面**（命令名/提示段）+ **命令行为**（成功/失败/未安装）
// + **格式化契约**（拿真 CLI 的输出做 fixture，钉住静态壳读的字段）。
// 用替身 ctx 与一个假 CLI，不依赖 DSH、不跑 Python 判定。
import { test } from "node:test";
import assert from "node:assert/strict";
import { chmodSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { apply, formatCapabilities, formatStatus, runCli } from "../index.mjs";

/** 真 CLI 的输出：reshape 判定层 payload 时，这两个 fixture 先红。 */
const fixture = (name) =>
  JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), "utf8"));

function fakeCtx() {
  const commandDefs = [];
  const sections = [];
  const logs = [];
  const record = (level) => (message) => logs.push([level, String(message)]);
  return {
    commandDefs,
    sections,
    logs,
    commands: { register: (def) => commandDefs.push(def) },
    systemPrompt: { section: (s) => sections.push(s) },
    logger: { info: record("info"), warn: record("warn"), error: record("error") },
  };
}

function fakeCli(body) {
  const dir = mkdtempSync(join(tmpdir(), "fake-operator-"));
  const file = join(dir, "operator-agent");
  writeFileSync(file, `#!/bin/sh\n${body}\n`, "utf8");
  chmodSync(file, 0o755);
  return file;
}

test("注册了 /op 命令与 system prompt 段", () => {
  const ctx = fakeCtx();
  apply(ctx, { command: "operator-agent" });
  assert.equal(ctx.commandDefs.length, 1);
  assert.equal(ctx.commandDefs[0].name, "op");
  // 稳定身份：UI 侧靠它跨重启配对同一条命令。
  assert.equal(ctx.commandDefs[0].definitionId, "dsh-operator-agent");
  assert.equal(typeof ctx.commandDefs[0].handler, "function");
  assert.equal(ctx.sections.length, 1);
  assert.equal(ctx.sections[0].name, "tool:operator");
  assert.ok(ctx.sections[0].text.includes("mcp__operator__"));
  assert.ok(ctx.sections[0].text.includes("eligibility"), "提示段必须要求 route 与资格一起读");
});

test("apply 不返回任何东西：生命周期归 cordis，别自己攒 disposer", () => {
  const ctx = fakeCtx();
  assert.equal(apply(ctx, { command: "operator-agent" }), undefined);
});

test("/op help 返回用法且不调用 CLI", async () => {
  const ctx = fakeCtx();
  apply(ctx, { command: "/nonexistent/never-run" });
  const out = await ctx.commandDefs[0].handler({ rawInput: "help" });
  assert.equal(out.kind, "success");
  assert.ok(out.text.includes("/op status"));
});

test("/op capabilities 把 JSON 折成人读的几行", async () => {
  const cli = fakeCli(`cat <<'JSON'
{"ok":true,"version":"0.6.0","tool_version":1,
 "backends":{"mock":{"wired":true},"local":{"wired":true},"ssh":{"wired":true}},
 "runtime":{"python":"3.9.12","torch_installed":true,"triton_installed":true},
 "kernel_types_with_input_layout":["matmul","softmax"]}
JSON`);
  const ctx = fakeCtx();
  apply(ctx, { command: cli });
  const out = await ctx.commandDefs[0].handler({ rawInput: "capabilities" });
  assert.equal(out.kind, "success");
  assert.ok(out.text.includes("mock / local / ssh"));
  assert.ok(out.text.includes("triton 有"));
  assert.ok(out.text.includes("matmul, softmax"));
});

test("/op status 读相位与摘要；lost 也要能显示", async () => {
  const cli = fakeCli(`cat <<'JSON'
{"ok":true,"phase":"lost","process_alive":false,"duration_s":12.5,
 "summary":{"all_passed":false,"levels":{"L1":true,"L2":false},"speedup":null},
 "log_tail":["最后一轮：L2 失败"]}
JSON`);
  const ctx = fakeCtx();
  apply(ctx, { command: cli });
  const out = await ctx.commandDefs[0].handler({ rawInput: "status /tmp/run-1" });
  assert.equal(out.kind, "success");
  assert.ok(out.text.includes("lost"));
  assert.ok(out.text.includes("L1✓"));
  assert.ok(out.text.includes("L2✗"));
  assert.ok(out.text.includes("最后一轮：L2 失败"));
});

test("/op status 缺参数时报错", async () => {
  const ctx = fakeCtx();
  apply(ctx, { command: "operator-agent" });
  const out = await ctx.commandDefs[0].handler({ rawInput: "status" });
  assert.equal(out.kind, "error");
  assert.ok(out.text.includes("run_dir"));
});

test("命令不存在时给可执行的提示，而不是抛异常", async () => {
  const ctx = fakeCtx();
  apply(ctx, { command: "/definitely/not/installed" });
  const out = await ctx.commandDefs[0].handler({ rawInput: "capabilities" });
  assert.equal(out.kind, "error");
  assert.ok(out.text.includes("pip install operator-agent"));
});

test("未知子命令回 error 并带上用法", async () => {
  const ctx = fakeCtx();
  apply(ctx, { command: "operator-agent" });
  const out = await ctx.commandDefs[0].handler({ rawInput: "nope" });
  assert.equal(out.kind, "error");
  assert.ok(out.text.includes("用法"));
});

test("runCli 把非零退出折成结构化结果（不抛）", async () => {
  const cli = fakeCli("echo 'boom' >&2; exit 2");
  const out = await runCli(cli, []);
  assert.equal(out.ok, true);
  assert.equal(out.code, 2);
  assert.equal(out.stderr, "boom");
});

test("失败在宿主日志里留痕，且同类只记一次（不刷屏）", async () => {
  const ctx = fakeCtx();
  apply(ctx, { command: "/nonexistent/never-run" });
  const handler = ctx.commandDefs[0].handler;

  await handler({ rawInput: "capabilities" });
  await handler({ rawInput: "capabilities" });      // 第二次同类失败
  await handler({ rawInput: "status /tmp/nope" });  // 也是 not_installed
  assert.equal(ctx.logs.length, 1, `同类失败记了多次：${JSON.stringify(ctx.logs)}`);
  assert.equal(ctx.logs[0][0], "error", "命令不存在 = 插件根本没接线，按 error 记");
  assert.ok(ctx.logs[0][1].includes("找不到命令"), ctx.logs[0][1]);
});

test("超时记 warn，取消不记（取消是人的正常操作）", async () => {
  const ctx = fakeCtx();
  apply(ctx, { command: fakeCli("exec sleep 5"), timeoutMs: 300 });
  const out = await ctx.commandDefs[0].handler({ rawInput: "capabilities" });
  assert.equal(out.kind, "error");
  assert.equal(ctx.logs.length, 1);
  assert.equal(ctx.logs[0][0], "warn");
  assert.ok(ctx.logs[0][1].includes("超时"), ctx.logs[0][1]);

  const cancelled = fakeCtx();
  apply(cancelled, { command: fakeCli("exec sleep 30") });
  const controller = new AbortController();
  setTimeout(() => controller.abort(), 50);
  const res = await cancelled.commandDefs[0].handler(
    { rawInput: "capabilities", signal: controller.signal });
  assert.equal(res.kind, "error");
  assert.deepEqual(cancelled.logs, [], "取消不该进日志");
});

test("CLI 回非 JSON 时记 bad_output（否则用户只看到命令报错，日志里没线索）", async () => {
  const ctx = fakeCtx();
  apply(ctx, { command: fakeCli("echo '这不是 JSON'") });
  const out = await ctx.commandDefs[0].handler({ rawInput: "capabilities" });
  assert.equal(out.kind, "error");
  assert.equal(ctx.logs.length, 1);
  assert.equal(ctx.logs[0][0], "warn");
  assert.ok(ctx.logs[0][1].includes("不是 JSON"), ctx.logs[0][1]);
});

test("runCli 把超时折成 reason=timeout，而不是抛异常", async () => {
  // `exec sleep` 让被杀的进程就是 sleep 本体：否则它作为孙进程持有 stdout，
  // 回调要等它自然结束才回来，这条测试会变成 5 秒。
  const cli = fakeCli("exec sleep 5");
  const out = await runCli(cli, [], 300);
  assert.equal(out.ok, false);
  assert.equal(out.reason, "timeout");
  assert.ok(out.detail.includes("300"));
});

test("runCli 把取消折成 reason=cancelled，并且真的把子进程收掉", async () => {
  const cli = fakeCli("exec sleep 30");
  const controller = new AbortController();
  setTimeout(() => controller.abort(), 50);
  const started = Date.now();
  const out = await runCli(cli, [], 60_000, undefined, controller.signal);
  const elapsed = Date.now() - started;
  assert.equal(out.ok, false);
  // 取消排在超时前面：abort 也会让 err.killed 为 true，别把它读成超时。
  assert.equal(out.reason, "cancelled");
  assert.ok(elapsed < 5_000, `取消没有真的杀掉子进程：等了 ${elapsed} ms`);
});

test("已经 abort 的 signal 也不抛", async () => {
  const cli = fakeCli("exec sleep 30");
  const out = await runCli(cli, [], 60_000, undefined, AbortSignal.abort());
  assert.equal(out.ok, false);
  assert.equal(out.reason, "cancelled");
});

test("config.env 合并进子进程（没 pip install、从源码树跑的那条路）", async () => {
  const cli = fakeCli(`if [ "$DSH_OPERATOR_TEST_FLAG" = "on" ]; then
  cat <<'JSON'
{"ok":true,"version":"ENV-ON","backends":{},"runtime":{},"kernel_types_with_input_layout":[]}
JSON
else
  cat <<'JSON'
{"ok":true,"version":"ENV-MISSING","backends":{},"runtime":{},"kernel_types_with_input_layout":[]}
JSON
fi`);
  const ctx = fakeCtx();
  apply(ctx, { command: cli, env: { DSH_OPERATOR_TEST_FLAG: "on" } });
  const out = await ctx.commandDefs[0].handler({ rawInput: "capabilities" });
  assert.equal(out.kind, "success");
  assert.ok(out.text.includes("ENV-ON"), out.text);
});

test("formatCapabilities 吃真 CLI 的 capabilities fixture（字段名不许漂）", () => {
  const text = formatCapabilities(fixture("capabilities.json"));
  assert.ok(text.includes("算子判定层"), text);
  // 真 CLI 的 JSON 是 sort_keys=True，所以这里是 local / mock / ssh（不是声明顺序）。
  assert.ok(text.includes("local / mock / ssh"), text);
  assert.ok(text.includes("python"), text);
  assert.ok(text.includes("matmul"), text);
  // 字段名漂了就会退化成 "?" —— 钉住这一格。
  assert.ok(!text.includes("?"), `有字段没读到：\n${text}`);
});

test("formatStatus 吃真 CLI 的 done fixture：终态不许说「在跑」", () => {
  const payload = fixture("status-done.json");
  assert.equal(payload.phase, "done");
  // Python 侧终态给的 process_alive 是 null —— 壳必须按相位判，不按死活判。
  assert.equal(payload.process_alive, null);
  const text = formatStatus(payload);
  assert.ok(text.includes("相位：done（进程已结束）"), text);
  assert.ok(!text.includes("在跑"), text);
  assert.ok(text.includes("2025-10-09T08:53:20.000Z"), text);
  assert.ok(text.includes("全部通过"), text);
  assert.ok(text.includes("1.83"), text);
});

test("formatStatus 四态各有各的进程文案，未知相位不猜", () => {
  const base = { ok: true, levels: {} };
  assert.ok(formatStatus({ ...base, phase: "running", process_alive: true }).includes("进程在跑"));
  assert.ok(formatStatus({ ...base, phase: "running", process_alive: false }).includes("进程已退出"));
  assert.ok(formatStatus({ ...base, phase: "lost", process_alive: false }).includes("进程已退出"));
  assert.ok(formatStatus({ ...base, phase: "failed", process_alive: null }).includes("进程已结束"));
  assert.ok(formatStatus({ ...base, phase: "weird", process_alive: null }).includes("进程状态未知"));
});

test("/op status 对 ok:false 的回包回 error，不伪装成成功", async () => {
  const cli = fakeCli(`cat <<'JSON'
{"ok":false,"error":"run_not_started","detail":"run-status.json 不存在（还没调用 operator_run）"}
JSON`);
  const ctx = fakeCtx();
  apply(ctx, { command: cli });
  const out = await ctx.commandDefs[0].handler({ rawInput: "status /tmp/never" });
  assert.equal(out.kind, "error");
  assert.ok(out.text.includes("run-status.json 不存在"), out.text);
});

test("/op capabilities 对 ok:false 的回包回 error（不打一排「?」）", async () => {
  const cli = fakeCli(`cat <<'JSON'
{"ok":false,"error":"tool_error","detail":"这台机器上没有 numpy"}
JSON`);
  const ctx = fakeCtx();
  apply(ctx, { command: cli });
  const out = await ctx.commandDefs[0].handler({ rawInput: "capabilities" });
  assert.equal(out.kind, "error");
  assert.ok(out.text.includes("没有 numpy"), out.text);
  assert.ok(!out.text.includes("?"), out.text);
});
