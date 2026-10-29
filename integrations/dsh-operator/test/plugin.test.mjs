// 这个插件的守卫：**注册面**（命令名/提示段）+ **命令行为**（成功/失败/未安装）
// + **格式化契约**（拿真 CLI 的输出做 fixture，钉住静态壳读的字段）。
// 用替身 ctx 与一个假 CLI，不依赖 DSH、不跑 Python 判定。
import { test } from "node:test";
import assert from "node:assert/strict";
import { chmodSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { apply, formatCapabilities, formatStatus, normalizeConfig, runCli, scrubEnv }
  from "../index.mjs";

/** 真 CLI 的输出：reshape 判定层 payload 时，这两个 fixture 先红。 */
const fixture = (name) =>
  JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), "utf8"));

function fakeCtx({ cwd = undefined } = {}) {
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
    // 宿主的 `ctx.get(服务名)` 对未提供的服务返回 undefined（不抛），替身照这个语义来。
    get: (name) => (name === "workingDirectory" && cwd !== undefined
      ? { get: () => cwd } : undefined),
  };
}

/** 命令的 invocation 里带着 agent（宿主：{commandId, agent, rawInput, attachments, signal}）。 */
const fakeAgent = { session: { id: "s" } };

function fakeCli(body) {
  const dir = mkdtempSync(join(tmpdir(), "fake-operator-"));
  const file = join(dir, "operator-agent");
  writeFileSync(file, `#!/bin/sh\n${body}\n`, "utf8");
  chmodSync(file, 0o755);
  return file;
}

test("注册了 /op 命令；提示段不归这里管（由 server 的 instructions 提供）", () => {
  const ctx = fakeCtx();
  apply(ctx, { command: "operator-agent" });
  assert.equal(ctx.commandDefs.length, 1);
  assert.equal(ctx.commandDefs[0].name, "op");
  // 稳定身份：UI 侧靠它跨重启配对同一条命令。
  assert.equal(ctx.commandDefs[0].definitionId, "dsh-operator-agent");
  assert.equal(typeof ctx.commandDefs[0].handler, "function");
  // 「什么时候该用」那段是 MCP server 的 initialize.instructions（宿主渲染成 mcp:operator），
  // 插件再注册一遍就变成两份会漂的副本 —— 这里钉住"别偷偷加回来"。
  assert.deepEqual(ctx.sections, [], "插件不该再注册 system prompt 段");
});

test("没有 systemPrompt 服务的 ctx 也能装（inject 只剩 commands）", () => {
  const ctx = fakeCtx();
  ctx.systemPrompt = undefined;
  assert.equal(apply(ctx, { command: "operator-agent" }), undefined);
  assert.equal(ctx.commandDefs.length, 1);
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


test("scrubEnv 用宿主那套规则洗：*KEY*/*PASSWORD*/*SECRET*/*TOKEN* 与全部 DSH_*", () => {
  const cleaned = scrubEnv({
    PATH: "/usr/bin",
    LLM_API_KEY_FILE: "/home/u/llm.key",
    LLM_API_KEY: "sk-secret",
    GITHUB_TOKEN: "t",
    DB_PASSWORD: "p",
    MY_SECRET: "s",
    DSH_HOME: "/home/synthetic/.dsh",
    DSH_OPERATOR_ANYTHING: "x",
    OPERATOR_AGENT_OUTPUT_DIR: "/tmp/out",
    PYTHONPATH: "/repo",
  });
  assert.deepEqual(Object.keys(cleaned).sort(),
                   ["OPERATOR_AGENT_OUTPUT_DIR", "PATH", "PYTHONPATH"]);
});

test("normalizeConfig：拼错的键会说，写错的类型会退回默认值", () => {
  const warned = [];
  const good = normalizeConfig(
    { command: "python3", args: ["-m", "operator_agent"], timeoutMs: 1000, env: { A: "1" } },
    (m) => warned.push(m));
  assert.deepEqual(good, { command: "python3", args: ["-m", "operator_agent"],
                           timeoutMs: 1000, env: { A: "1" } });
  assert.deepEqual(warned, [], "正常配置不该报警");

  // 两个真实的静默陷阱：键名拼错（会被忽略）、毫秒写成带单位（会退回默认）。
  const w = [];
  const bad = normalizeConfig({ comand: "python3", timeoutMs: "30s" }, (m) => w.push(m));
  assert.equal(bad.command, "operator-agent", "拼错的 comand 不该被当成 command");
  assert.equal(bad.timeoutMs, 60_000);
  assert.equal(w.length, 2, JSON.stringify(w));
  assert.ok(w[0].includes("comand"), w[0]);
  assert.ok(w[1].includes("timeoutMs"), w[1]);

  const w2 = [];
  assert.equal(normalizeConfig({ timeoutMS: 1 }, (m) => w2.push(m)).timeoutMs, 60_000);
  assert.ok(w2[0].includes("正确写法是 timeoutMs"), w2[0]);

  const w3 = [];
  const bad2 = normalizeConfig({ command: "", args: "x", env: { A: 1 } }, (m) => w3.push(m));
  assert.equal(bad2.command, "operator-agent");
  assert.deepEqual(bad2.args, []);
  assert.equal(bad2.env, undefined);
  assert.equal(w3.length, 3, JSON.stringify(w3));
});

test("配置错一次只在宿主日志里说一次（apply 加载时不刷屏）", () => {
  const ctx = fakeCtx();
  apply(ctx, { comand: "x", timeoutMs: "30s" });
  assert.equal(ctx.logs.length, 2, JSON.stringify(ctx.logs));
  assert.ok(ctx.logs.every(([level]) => level === "warn"));
  assert.ok(ctx.logs[0][1].startsWith("operator-tools: 配置项 comand"), ctx.logs[0][1]);
});

test("/op status：相对路径按**会话工作区**解析（宿主 cwd 不是人的工作区）", async () => {
  // 假 CLI 把收到的 --run-dir 原样回进 payload：直接证明**到 CLI 手里的那个路径**是什么。
  const cli = fakeCli(`printf '{"ok":true,"phase":"%s"}' "$3"`);
  const ctx = fakeCtx({ cwd: "/workspace/operatorAgent" });
  apply(ctx, { command: cli });
  const out = await ctx.commandDefs[0].handler({
    rawInput: "status runs/2026-10", agent: fakeAgent, signal: undefined });

  assert.equal(out.kind, "success");
  assert.ok(out.text.includes("目录：/workspace/operatorAgent/runs/2026-10"), out.text);
  // 这里的假 CLI 把路径塞进了 phase 字段，所以首行也会带上它——两处都认，但
  // 「实际读了哪个目录」的权威落点是 `目录：` 行。
  assert.ok(out.text.includes("/workspace/operatorAgent/runs/2026-10"), out.text);
});

test("/op status：含空格的路径整段保留（按空白切会碎）", async () => {
  const cli = fakeCli(`printf '{"ok":true,"phase":"%s"}' "$3"`);
  const ctx = fakeCtx({ cwd: "/w" });
  apply(ctx, { command: cli });

  const plain = await ctx.commandDefs[0].handler({
    rawInput: "status /workspace/my runs/x", agent: fakeAgent });
  assert.ok(plain.text.includes("目录：/workspace/my runs/x"), plain.text);

  const quoted = await ctx.commandDefs[0].handler({
    rawInput: 'status "my runs/x"', agent: fakeAgent });
  assert.ok(quoted.text.includes("目录：/w/my runs/x"), quoted.text);
});

test("/op status：绝对路径与 ~ 都不动会话工作区那套", async () => {
  const cli = fakeCli(`printf '{"ok":true,"phase":"%s"}' "$3"`);
  const ctx = fakeCtx({ cwd: "/w" });
  apply(ctx, { command: cli });

  const abs = await ctx.commandDefs[0].handler({ rawInput: "status /tmp/r", agent: fakeAgent });
  assert.ok(abs.text.includes("目录：/tmp/r"), abs.text);

  const home = await ctx.commandDefs[0].handler({ rawInput: "status ~/r", agent: fakeAgent });
  assert.ok(home.text.includes(`目录：${homedir()}/r`), home.text);
});

test("拿不到 workingDirectory 时原样传下去，不把命令变成不可用", async () => {
  const cli = fakeCli(`printf '{"ok":true,"phase":"%s"}' "$3"`);
  const ctx = fakeCtx();                       // 没有 workingDirectory
  apply(ctx, { command: cli });
  const out = await ctx.commandDefs[0].handler({ rawInput: "status rel/dir", agent: fakeAgent });
  assert.ok(out.text.includes("目录：rel/dir"), out.text);
});

test("显式 config.env 在清洗**之后**合并：要传仍传得进去", async () => {
  process.env.DSH_OPERATOR_TEST_SCRUB = "dsh-value";       // DSH_ 前缀 → 必须洗掉
  process.env.LLM_API_KEY_FROM_HOST = "sk-from-host";      // 含 KEY → 必须洗掉
  try {
    // 三个探针按顺序回进 version：DSH_* / 宿主里的敏感变量 / 显式写进 config 的那个。
    const cli = fakeCli(
      "printf '{\"ok\":true,\"version\":\"%s|%s|%s\",\"backends\":{},\"runtime\":{}," +
      "\"kernel_types_with_input_layout\":[]}' " +
      '"\${DSH_OPERATOR_TEST_SCRUB:-unset}" "\${LLM_API_KEY_FROM_HOST:-unset}" ' +
      '"\${LLM_API_KEY_FILE:-unset}"');
    const ctx = fakeCtx();
    apply(ctx, { command: cli, env: { LLM_API_KEY_FILE: "/tmp/inside.key" } });
    const out = await ctx.commandDefs[0].handler({ rawInput: "capabilities" });
    assert.equal(out.kind, "success", out.text);
    assert.ok(out.text.includes("unset|unset|/tmp/inside.key"),
              `前两个该被洗掉、第三个该来自 config.env：${out.text}`);
  } finally {
    delete process.env.DSH_OPERATOR_TEST_SCRUB;
    delete process.env.LLM_API_KEY_FROM_HOST;
  }
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

// 宿主的 GenericCommandCard 折叠时把**全文**塞进单行摘要、换行被压平（README §5.5）。
// 所以首行是折叠态下用户唯一能看见的东西：它必须自己就能读完，而不是一个「算子判定层 0.6.0」的孤零零开头。
test("formatCapabilities 首行自带摘要：折叠态不看明细也知道后端与覆盖数", () => {
  const payload = fixture("capabilities.json");
  const [first] = formatCapabilities(payload).split("\n");
  assert.ok(first.includes("算子判定层 0.6.0"), first);
  assert.ok(first.includes("local/mock/ssh"), `首行缺后端：${first}`);
  // 布局数是「10」，不是把十个名字全铺在首行（铺了就会溢出折行）。
  assert.ok(first.includes("10 个算子有输入布局"), `首行缺覆盖数：${first}`);
  assert.ok(!first.includes("matmul"), `首行不该铺算子名：${first}`);
  // 首行要短到折叠态一行放得下（原实现 25 字符切开，这里放宽到 60 仍留余量）。
  assert.ok(first.length <= 60, `首行过长（${first.length}）：${first}`);
});

test("formatCapabilities 明细仍在后面，展开能看全（首行只是摘要，不是砍功能）", () => {
  const payload = fixture("capabilities.json");
  const lines = formatCapabilities(payload).split("\n");
  assert.ok(lines.length >= 3, `明细被吃了：\n${lines.join("\n")}`);
  const body = lines.slice(1).join("\n");
  assert.ok(body.includes("已接线后端：local / mock / ssh"), body);
  assert.ok(body.includes("python 3.9.12"), body);
  // 十个算子名一个不少地留在明细里。
  for (const name of payload.kernel_types_with_input_layout) {
    assert.ok(body.includes(name), `明细缺 ${name}：\n${body}`);
  }
});

test("formatCapabilities 后端为空/缺字段时首行不崩，也不编造", () => {
  const first = formatCapabilities({ version: "9.9.9", tool_version: 2, backends: {},
    kernel_types_with_input_layout: [] }).split("\n")[0];
  assert.ok(first.includes("9.9.9"), first);
  assert.ok(first.includes("后端 无"), `空后端要明说：${first}`);
  assert.ok(first.includes("0 个算子有输入布局"), `空布局要明说：${first}`);
});

// ── 回执语义 ────────────────────────────────────────────────────────────────
// 宿主折叠态只显示首行（nowrap + ellipsis）。所以「跑没跑成」必须由首行自己说，
// 不能指望用户点开才发现。成功 ✓ / 失败 ✗ / 取消明说「已取消」而不挂 ✗。

test("回执：成功的两条命令首行都以 ✓ 开头，且说的是完成而不是数据", () => {
  const caps = formatCapabilities(fixture("capabilities.json")).split("\n")[0];
  assert.ok(caps.startsWith("✓ 已执行"), `capabilities 首行缺回执：${caps}`);
  const status = formatStatus(fixture("status-done.json")).split("\n")[0];
  assert.ok(status.startsWith("✓ 查询完成"), `status 首行缺回执：${status}`);
});

test("回执：首行装得下（摘要 nowrap 截断，超了就只看见省略号）", () => {
  // 81 字符是 comfortably 放得下的上限；再长就会在常见窗口宽度下被截。
  const caps = formatCapabilities(fixture("capabilities.json")).split("\n")[0];
  assert.ok(caps.length <= 60, `capabilities 首行过长（${caps.length}）：${caps}`);
  const status = formatStatus(fixture("status-done.json"), "/some/really/long/run/dir").split("\n")[0];
  assert.ok(status.length <= 40, `status 首行过长（${status.length}）：${status}`);
  // 首行不该把路径塞进来——路径在明细的 `目录：` 行，长了会把相位挤掉。
  assert.ok(!status.includes("/some/really/long"), `路径不该进首行：${status}`);
});

test("回执：tool 号降到明细，不占首行的预算", () => {
  const text = formatCapabilities(fixture("capabilities.json"));
  assert.ok(!text.split("\n")[0].includes("tool"), `tool 号仍占首行：${text.split("\n")[0]}`);
  // 但它没丢，只是挪到明细（回包契约版本和判定层版本不是一回事，不能合并）。
  assert.ok(text.includes("工具版本：tool 1"), text);
});

test("回执：判定层拒绝走 ✗，一眼能和成功分开", () => {
  const cli = fakeCli(`printf '{"ok":false,"error":"BAD_REQUEST","detail":"形状不合法"}'`);
  const ctx = fakeCtx();
  apply(ctx, { command: cli });
  return ctx.commandDefs[0].handler({ rawInput: "capabilities" }).then((out) => {
    assert.equal(out.kind, "error");
    assert.ok(out.text.startsWith("✗"), `拒绝没有 ✗ 回执：${out.text}`);
    assert.ok(out.text.includes("形状不合法"), out.text);
  });
});

test("回执：CLI 没装走 ✗ 未执行，reason 进首行", async () => {
  const ctx = fakeCtx();
  apply(ctx, { command: "/nonexistent/operator-agent-xyz" });
  const out = await ctx.commandDefs[0].handler({ rawInput: "capabilities" });
  assert.equal(out.kind, "error");
  assert.ok(out.text.startsWith("✗ 未执行"), `没装时缺回执：${out.text}`);
  assert.ok(out.text.split("\n")[0].includes("not_installed"), out.text);
});

test("回执：非 JSON 走 ✗ 读数不可用（不是「成功但空的」）", async () => {
  const cli = fakeCli(`printf 'garbage'`);
  const ctx = fakeCtx();
  apply(ctx, { command: cli });
  const out = await ctx.commandDefs[0].handler({ rawInput: "capabilities" });
  assert.equal(out.kind, "error");
  assert.ok(out.text.startsWith("✗ 读数不可用"), out.text);
});

test("回执：缺参数与未知子命令也都以 ✗ 开头（不许静默）", async () => {
  const ctx = fakeCtx();
  apply(ctx, { command: fakeCli(`printf '{"ok":true}'`) });
  const noArg = await ctx.commandDefs[0].handler({ rawInput: "status" });
  assert.ok(noArg.text.startsWith("✗ 缺参数"), noArg.text);
  const unknown = await ctx.commandDefs[0].handler({ rawInput: "wat" });
  assert.ok(unknown.text.startsWith("✗ 未知子命令"), unknown.text);
});

test("回执：/op help 也带 ✓，且说明里写清折叠态只显示首行", async () => {
  const ctx = fakeCtx();
  apply(ctx, { command: fakeCli(`printf '{"ok":true}'`) });
  const out = await ctx.commandDefs[0].handler({ rawInput: "help" });
  assert.equal(out.kind, "success");
  assert.ok(out.text.startsWith("✓ 已执行"), out.text);
  assert.ok(out.text.includes("折叠时只显示首行"), out.text);
});

test("formatStatus 吃真 CLI 的 done fixture：终态不许说「在跑」", () => {
  const payload = fixture("status-done.json");
  assert.equal(payload.phase, "done");
  // Python 侧终态给的 process_alive 是 null —— 壳必须按相位判，不按死活判。
  assert.equal(payload.process_alive, null);
  const text = formatStatus(payload);
  // 相位在首行（折叠态唯一可见的那行），且只出现一次——展开态不重复。
  assert.ok(text.split("\n")[0].includes("相位 done（进程已结束）"), text);
  assert.equal(text.split("相位").length - 1, 1, `相位重复了：\n${text}`);
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

test("formatStatus 展示任务预算、确认预留与明确停止原因", () => {
  const text = formatStatus({ ok: true, phase: "done", process_alive: null,
    stopped_reason: "budget_llm_calls",
    task_budget: { wall_clock_s: 1800, max_llm_calls: 24, oracle_total: 20 },
    budget: {
      wall_clock: { limit_s: 1800, remaining_s: 123.5 },
      model_calls: { limit: 24, reserved: 24, remaining: 0 },
      verification: { plan: { oracle_total: 20, phase_caps: { confirm: 2 } },
        spent_all: 18, spent: { confirm: 1 }, remaining: { total: 2, confirm: 1 } },
    }, summary: { all_passed: false, levels: { L1: true, L2: false } } });
  const first = text.split("\n")[0];
  assert.ok(first.startsWith("✓ 查询完成") && first.includes("预算停止"), first);
  assert.ok(first.length <= 40 && !first.includes("全部通过"), first);
  assert.ok(text.includes("停止原因：budget_llm_calls"), text);
  assert.ok(text.includes("任务墙钟：剩余 123.5 s / 上限 1800 s"), text);
  assert.ok(text.includes("模型调用：预留 24 / 上限 24 · 剩余 0"), text);
  assert.ok(text.includes("验证输入组：预留 18 / 上限 20 · 剩余 2"), text);
  assert.ok(text.includes("最终确认：预留 2 · 已用 1 · 剩余 1"), text);
});

test("formatStatus 从 summary 取预算，缺失用量显示未知", () => {
  const text = formatStatus({ ok: true, phase: "done", process_alive: null,
    summary: { stopped_reason: "wall_clock_exhausted",
      task_budget: { wall_clock_s: 12.5, max_llm_calls: 7, oracle_total: 11 },
      budget: { wall_clock: { remaining_s: 0 }, model_calls: { reserved: null, remaining: null } },
    } });
  assert.ok(text.split("\n")[0].includes("预算停止"), text);
  assert.ok(text.includes("停止原因：wall_clock_exhausted"), text);
  assert.ok(text.includes("任务墙钟：剩余 0 s / 上限 12.5 s"), text);
  assert.ok(text.includes("模型调用：预留 未知 / 上限 7 · 剩余 未知"), text);
  assert.ok(text.includes("验证输入组：预留 未知 / 上限 11 · 剩余 未知"), text);
  assert.ok(text.includes("最终确认：预留 未知 · 已用 未知 · 剩余 未知"), text);
});

test("formatStatus 不把无效计数或仍在运行的任务猜成预算停止", () => {
  const text = formatStatus({ ok: true, phase: "running", process_alive: true,
    stopped_reason: "budget_exhausted", task_budget: { max_llm_calls: 24 },
    budget: { wall_clock: { remaining_s: NaN },
      model_calls: { reserved: false, remaining: -1 },
      verification: { spent_all: 1.5 } } });
  assert.ok(!text.split("\n")[0].includes("预算停止"), text);
  assert.ok(text.includes("任务墙钟：剩余 未知 s"), text);
  assert.ok(text.includes("模型调用：预留 未知 / 上限 24 · 剩余 未知"), text);
  assert.ok(text.includes("验证输入组：预留 未知"), text);
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

test("formatStatus 展示取消终态、实例和清理未知，不推断远端停止", () => {
  const text = formatStatus({ ok: true, phase: "cancelled", invocation_id: "a".repeat(32),
    stopped_reason: "user_cancelled", cancellation: {}, cleanup: {
      status: "cleanup_unknown", remote_execution: "unknown", provider_call: "unknown" } });
  assert.ok(text.split("\n")[0].includes("cancelled（进程已结束）"), text);
  assert.ok(text.includes("任务实例：" + "a".repeat(32)), text);
  assert.ok(text.includes("本机清理：cleanup_unknown"), text);
  assert.ok(text.includes("远端执行：unknown · 服务商调用：unknown"), text);
});

test("/op cancel 保留含空格目录，按实例请求并提示查终态", async () => {
  const cli = fakeCli(`test "$1" = cancel && test "$2" = --run-dir && test "$3" = '/tmp/run space' && test "$4" = --invocation-id && test "$5" = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa || exit 3
printf '%s' '{"ok":true,"accepted":true,"invocation_id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}'`);
  const ctx = fakeCtx();
  apply(ctx, { command: cli });
  const out = await ctx.commandDefs[0].handler({ rawInput: 'cancel "/tmp/run space" ' + "a".repeat(32) });
  assert.equal(out.kind, "success");
  assert.ok(out.text.includes("取消已请求"), out.text);
  assert.ok(out.text.includes("确认终态"), out.text);
});

test("/op cancel 缺身份拒绝；已结束回包不说刚刚取消", async () => {
  const cli = fakeCli(`printf '%s' '{"ok":true,"accepted":false,"phase":"done","invocation_id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}'`);
  const ctx = fakeCtx();
  apply(ctx, { command: cli });
  const missing = await ctx.commandDefs[0].handler({ rawInput: "cancel /tmp/run" });
  assert.equal(missing.kind, "error");
  const done = await ctx.commandDefs[0].handler({ rawInput: "cancel /tmp/run " + "a".repeat(32) });
  assert.ok(done.text.includes("任务已结束 · done"), done.text);
  assert.ok(!done.text.includes("取消已请求"), done.text);
});

test("formatStatus 可列出待取消工具实例", () => {
  const text = formatStatus({ ok: true, phase: "tool_calls", calls: [
    { invocation_id: "a".repeat(32), phase: "running" },
    { invocation_id: "b".repeat(32), phase: "cancelled" }] });
  assert.ok(text.includes("2 次工具调用"), text);
  assert.ok(text.includes("a".repeat(32) + " · running"), text);
});

test("/op cancel 不把 lost 的监督者失联说成任务已结束", async () => {
  const cli = fakeCli(`printf '%s' '{"ok":true,"accepted":false,"phase":"lost","reason":"owner_unavailable","invocation_id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}'`);
  const ctx = fakeCtx();
  apply(ctx, { command: cli });
  const out = await ctx.commandDefs[0].handler({ rawInput: "cancel /tmp/run " + "a".repeat(32) });
  assert.ok(out.text.includes("监督者不可用"), out.text);
  assert.ok(out.text.includes("本机清理结果未知"), out.text);
  assert.ok(!out.text.includes("任务已结束"), out.text);
});
