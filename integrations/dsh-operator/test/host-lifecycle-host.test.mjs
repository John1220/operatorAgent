// Native DSH + local HTTP fixture. Scripted decisions test plumbing only;
// no account/provider credential, GPU or personal profile participates.
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { chmodSync, existsSync, mkdtempSync, readFileSync, readdirSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";

const repo = fileURLToPath(new URL("../../../", import.meta.url));
const plugin = process.env.OPERATOR_AGENT_TEST_DSH_MODEL_RELAY_PLUGIN ?? fileURLToPath(new URL("../index.mjs", import.meta.url));
const app = process.env.DSH_APP_ROOT ?? "/Applications/DSH Desktop.app/Contents/Resources/app";
const entry = join(app, "node_modules/@deepseek-ai/dsh/lib/bin.js");
const model = "host-lifecycle-local-fixture";
const key = "local-lifecycle-fixture-key";
const correct = "import numpy as np\ndef kernel(X):\n    return np.maximum(X, 0)\n";
const disabled = ["session-title-llm", "llm-retry", "otel", "session-telemetry-otel", "tool-working-directory",
  "tool-bash", "tool-pwsh", "tool-jobs", "tool-fs", "tool-fs-search", "tool-skill", "tool-subagent-control",
  "tool-subagent-list-agents", "tool-subagent", "tool-subagent-fork", "tool-workflow", "tool-todo", "tool-goal",
  "web", "web-search-deepseek", "web-deepseek-api", "web-fetch-http", "tool-web"];
const skip = process.env.OPERATOR_AGENT_TEST_DSH_HOST_BUDGET !== "1"
  ? "set OPERATOR_AGENT_TEST_DSH_HOST_BUDGET=1 for real DSH/local-provider integration"
  : !existsSync(entry) && "set DSH_APP_ROOT to an installed DSH";

const load = path => JSON.parse(readFileSync(path, "utf8"));
const records = path => readFileSync(path, "utf8").trim().split("\n").filter(Boolean).map(JSON.parse);
function decoded(value) {
  if (typeof value === "string") { try { return decoded(JSON.parse(value)); } catch { return null; } }
  if (Array.isArray(value)) return value.map(decoded).find(Boolean) ?? null;
  if (value && typeof value === "object") return "ok" in value ? value : decoded(value.text ?? value.content);
  return null;
}
const toolReplyText = body => body.messages.filter(message => message.role === "tool").at(-1)?.content ?? "";
function reply(response, action, ordinal) {
  const base = { id: `lifecycle-${ordinal}`, model, created: 1, object: "chat.completion.chunk" };
  const delta = action.tool ? { role: "assistant", tool_calls: [{ index: 0, id: `lifecycle-tool-${ordinal}`,
    type: "function", function: { name: action.tool, arguments: JSON.stringify(action.args) } }] }
    : { role: "assistant", content: action.text ?? "Local lifecycle fixture completed." };
  const data = [
    { ...base, choices: [{ index: 0, delta, finish_reason: null }] },
    { ...base, choices: [{ index: 0, delta: {}, finish_reason: action.tool ? "tool_calls" : "stop" }] },
    { ...base, choices: [], usage: { prompt_tokens: 20, completion_tokens: 12, total_tokens: 32 } },
  ];
  response.writeHead(200, { "Content-Type": "text/event-stream" });
  response.end(data.map(x => "data: " + JSON.stringify(x) + "\n\n").join("") + "data: [DONE]\n\n");
}
async function run(argv, env, work, label) {
  const child = spawn(process.execPath, ["--import", "data:text/javascript,console.debug%20%3D%20console.error%3B", entry, ...argv],
    { cwd: work, env, stdio: ["ignore", "pipe", "pipe"] });
  let stdout = "", stderr = "", timedOut = false;
  child.stdout.setEncoding("utf8").on("data", x => { stdout += x; });
  child.stderr.setEncoding("utf8").on("data", x => { stderr += x; });
  const timer = setTimeout(() => { timedOut = true; child.kill("SIGTERM"); }, 30_000);
  const hard = setTimeout(() => child.kill("SIGKILL"), 35_000);
  try {
    const code = await new Promise((resolve, reject) => { child.once("error", reject); child.once("close", resolve); });
    writeFileSync(join(work, `${label}.stdout.jsonl`), stdout);
    writeFileSync(join(work, `${label}.stderr.log`), stderr);
    assert.equal(timedOut, false, `native DSH timeout; receipts: ${work}`);
    return { code, stdout, stderr };
  } finally { clearTimeout(timer); clearTimeout(hard); }
}
function writable(dir) {
  chmodSync(dir, 0o700);
  for (const item of readdirSync(dir, { withFileTypes: true })) if (item.isDirectory()) writable(join(dir, item.name));
}
function prepare(runDir, maxCalls = 8, tokens = 1024, total = 8192) {
  return { tool: "mcp__operator__operator_prepare", args: { request: "relu fp32", shapes: [[4, 8]], precision: "fp32",
    dsl: "python", hardware: "mock_gpu", run_dir: runDir,
    host_budget: { wall_clock_s: 120, max_verifications: 3, max_model_calls: maxCalls,
      max_output_tokens_per_call: tokens, total_output_token_reservations: total } } };
}
const context = runDir => ({ tool: "mcp__operator__operator_context", args: { run_dir: runDir, backend: "fake", phase: "generate" } });
const verify = (runDir, purpose) => ({ tool: "mcp__operator__operator_verify", args: {
  run_dir: runDir, backend: "fake", stop_after: "L2", candidate_origin: "outer", code: correct, verification_purpose: purpose,
} });

async function fixture(t, choose) {
  const work = realpathSync(mkdtempSync(join(tmpdir(), "operator-dsh-host-lifecycle-")));
  const installed = process.env.OPERATOR_AGENT_TEST_DSH_MODEL_RELAY_INSTALLED_PYTHON;
  const python = installed ?? process.env.OPERATOR_AGENT_PYTHON ??
    (existsSync(join(repo, ".venv/bin/python")) ? join(repo, ".venv/bin/python") : "python3");
  const observer = join(work, "observer.mjs"), observerLog = join(work, "observer.jsonl");
  writeFileSync(observer, `import {appendFileSync} from 'node:fs';
export const inject=['commands'];
export function apply(ctx, config) {
  const log = row => appendFileSync(config.log,JSON.stringify({...row,phase:config.phase})+'\\n');
  ctx.on('session/event',(session,event)=>{
    if(['command/run','command/done','turn/end'].includes(event.type))log({sessionId:session.id,type:event.type,data:event.data});
  });
  ctx.on('agent/created',async ({agent})=>{
    log({type:'agent/created',sessionId:agent.session.id});
    if(config.authorizeNew){
      const receipt=await ctx.commands.execute(agent,'/op new',[],new AbortController().signal);
      log({type:'native-command-result',sessionId:agent.session.id,receipt});
      if(receipt?.result?.kind!=='success')throw new Error('native /op new did not succeed');
    }
  });
}
`);
  let phase = "", local = 0, ordinal = 0;
  const faults = [], requests = [];
  const server = createServer(async (request, response) => {
    try {
      const chunks = []; for await (const chunk of request) chunks.push(chunk);
      const body = JSON.parse(Buffer.concat(chunks).toString("utf8"));
      assert.equal(request.headers.authorization, "Bearer " + key);
      assert.equal(body.model, model); assert.equal(body.stream, true);
      local += 1; ordinal += 1;
      requests.push({ phase, index: local });
      const action = choose({ phase, index: local, ordinal, body, work });
      reply(response, action, ordinal);
    } catch (error) { faults.push(error); response.writeHead(500).end("local lifecycle assertion failed"); }
  });
  const port = await new Promise((resolve, reject) => {
    server.once("error", reject); server.listen(0, "127.0.0.1", () => resolve(server.address().port));
  });
  const patch = join(work, "patch.json");
  const env = { PATH: process.env.PATH, HOME: work, DSH_HOME: join(work, "dsh-home"),
    DSH_TELEMETRY_DISABLED: "1", HOST_LIFECYCLE_FIXTURE_KEY: key, NO_PROXY: "127.0.0.1", no_proxy: "127.0.0.1" };
  let sessionId, passed = false;
  async function launch(name, prompt, authorizeNew = false) {
    phase = name; local = 0;
    writeFileSync(patch, JSON.stringify([
      ...disabled.map(id => ({ id, disabled: true })),
      { id: "agent-default-model", config: { provider: "fixture", model } },
      { id: "llm-pi-ai", config: { providers: { fixture: { api: "openai-completions", baseURL: `http://127.0.0.1:${port}/v1`,
        apiKeyEnv: "HOST_LIFECYCLE_FIXTURE_KEY", compat: { maxTokensField: "max_tokens" },
        models: [{ id: model, contextWindow: 65536, maxTokens: 4096 }] } } } },
      { insert: [
        { id: "mcp-operator", name: join(app, "node_modules/@deepseek-ai/dsh-mcp-client/lib/index.js"), config: {
          serverName: "operator", transport: "stdio", command: python,
          args: [...(installed ? ["-I"] : []), "-m", "operator_agent.app.mcp_server"], cwd: installed ? work : repo,
          failOnStartupError: true, toolCallTimeoutMs: 30_000,
          env: { OPERATOR_AGENT_WORKFLOW: "host", ...(installed ? {} : { PYTHONPATH: repo }),
            OPERATOR_AGENT_OUTPUT_DIR: join(work, "output"), OPERATOR_AGENT_LOG_DIR: join(work, "log"),
            OPERATOR_AGENT_KERNEL_DB: join(work, "kernel_db.jsonl"), OPERATOR_AGENT_FAILURE_DB: join(work, "failure_db.jsonl") } } },
        { id: "operator-tools", name: plugin, config: { command: python,
          args: [...(installed ? ["-I"] : []), "-m", "operator_agent"],
          ...(installed ? {} : { env: { PYTHONPATH: repo } }), hostModelBudget: {} } },
        { id: "lifecycle-observer", name: observer, config: { log: observerLog, phase: name, authorizeNew } },
      ] },
    ]));
    const result = await run(["--profile", "headless", "--patch", patch, "--json",
      ...(sessionId ? ["--session-id", sessionId] : []), prompt], env, work, name);
    assert.deepEqual(faults, [], `local fixture failure; receipts: ${work}`);
    const own = records(observerLog).filter(row => row.type === "agent/created");
    sessionId ??= own[0].sessionId;
    assert.ok(own.every(row => row.sessionId === sessionId), "every process must use the original session");
    return result;
  }
  return { work, requests, faults, launch, observerLog,
    pass() { passed = true; },
    async close() {
      server.closeAllConnections(); await new Promise(resolve => server.close(resolve));
      if (passed) { writable(work); rmSync(work, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 }); }
      else t.diagnostic(`Preserved native lifecycle receipts: ${work}`);
    } };
}

test("real DSH ends task scope, discusses without old credit, and authorizes a second task across restarts", {
  skip, timeout: 150_000,
}, async t => {
  const f = await fixture(t, ({ phase, index, body, work }) => {
    const a = join(work, "task-a"), b = join(work, "task-b");
    if (phase === "task-a" || phase === "task-b") {
      const dir = phase === "task-a" ? a : b;
      if (index === 1) return prepare(dir);
      if (index === 2) return context(dir);
      if (index === 3) return verify(dir, "explore");
      if (index === 4) return verify(dir, "final");
      if (index === 5) {
        const final = decoded(toolReplyText(body));
        assert.equal(final?.levels?.L2, true); assert.equal(final?.verification_purpose, "final");
        const journal = load(join(dir, "host-model-budget.json"));
        assert.equal(journal.records.length, 4, "final summary is reserved before provider dispatch");
        assert.equal(journal.records.at(-1).status, "reserved");
        return { text: "Independent final passed. This summary remains part of the original task budget." };
      }
    } else if (phase === "unauthorized-prepare") {
      if (index === 1) return prepare(a);
      if (index === 2) {
        assert.match(JSON.stringify(toolReplyText(body)), /\/op new|新任务|新建任务|全新|旧任务/);
        return { text: "A new task requires the user's native command." };
      }
    } else if (["discussion-a", "authorize-only", "discussion-b"].includes(phase) && index === 1) {
      return { text: "Normal conversation remains available without reactivating previous task execution." };
    }
    throw new Error(`unexpected local request ${phase}/${index}`);
  });
  try {
    const a = join(f.work, "task-a"), b = join(f.work, "task-b");
    await f.launch("task-a", "Run the local ReLU task and independent final.");
    const beforeA = load(join(a, "host-model-budget.json"));
    assert.equal(beforeA.records.length, 4); assert.ok(beforeA.records.every(row => row.status === "answered"));
    const oldA = readFileSync(join(a, "host-model-budget.json"), "utf8");
    await f.launch("discussion-a", "Explain the result; do not start another task.");
    assert.equal(readFileSync(join(a, "host-model-budget.json"), "utf8"), oldA, "discussion must leave old model ledger untouched");
    await f.launch("unauthorized-prepare", "Start another task without a native user command.");
    assert.equal(existsSync(join(b, "host-task-policy.json")), false, "model intent must not mint task authorization");
    assert.equal(readFileSync(join(a, "host-model-budget.json"), "utf8"), oldA);
    await f.launch("authorize-only", "The user has authorized a new task; wait for its declaration.", true);
    const commands = records(f.observerLog);
    const runs = commands.filter(row => row.type === "command/run" && row.data.name === "op" && row.data.args.trim() === "new");
    assert.equal(runs.length, 1); assert.equal(runs[0].data.source.kind, "user");
    assert.ok(commands.some(row => row.type === "command/done" && row.data.commandId === runs[0].data.commandId && row.data.kind === "success"));
    await f.launch("task-b", "Start the newly authorized independent ReLU task and final.");
    const beforeB = load(join(b, "host-model-budget.json"));
    assert.equal(beforeB.records.length, 4); assert.notEqual(beforeB.task_id, beforeA.task_id);
    assert.equal(beforeB.lease.session_id, beforeA.lease.session_id);
    assert.equal(readFileSync(join(a, "host-model-budget.json"), "utf8"), oldA, "second task cannot refill or rewrite task A");
    const oldB = readFileSync(join(b, "host-model-budget.json"), "utf8");
    await f.launch("discussion-b", "Discuss the final task B result.");
    assert.equal(readFileSync(join(b, "host-model-budget.json"), "utf8"), oldB);
    assert.equal(f.requests.length, 15);
    f.pass();
  } finally { await f.close(); }
});

test("real DSH automatically starts the next task from a new user turn without a slash command", {
  skip, timeout: 100_000,
}, async t => {
  const f = await fixture(t, ({ phase, index, work }) => {
    const dir = join(work, phase);
    if (index === 1) return prepare(dir);
    if (index === 2) return context(dir);
    if (index === 3) return verify(dir, "explore");
    if (index === 4) return verify(dir, "final");
    return { text: "Independent final passed." };
  });
  try {
    await f.launch("auto-a", "Run a ReLU case and independent final.");
    const pathA = join(f.work, "auto-a", "host-model-budget.json");
    const old = readFileSync(pathA, "utf8");
    await f.launch("auto-b", "Test another independent ReLU case in a fresh directory.");
    const a = JSON.parse(old), b = load(join(f.work, "auto-b", "host-model-budget.json"));
    assert.notEqual(a.task_id, b.task_id);
    assert.equal(a.lease.session_id, b.lease.session_id);
    assert.equal(b.records.length, 4);
    assert.equal(readFileSync(pathA, "utf8"), old);
    assert.equal(records(f.observerLog).some(row => row.type === "command/run"), false);
    f.pass();
  } finally { await f.close(); }
});

test("real DSH stops the exhausted turn then permits discussion while keeping old execution closed", {
  skip, timeout: 100_000,
}, async t => {
  const f = await fixture(t, ({ phase, index, body, work }) => {
    const dir = join(work, "task-exhausted");
    if (phase === "exhaust") {
      if (index === 1) return prepare(dir, 1, 128, 128);
      if (index === 2) return context(dir);
    } else if (phase === "discuss" && index === 1) return { text: "The previous task is stopped; ordinary conversation is available." };
    else if (phase === "forbidden-old-verify") {
      if (index === 1) return verify(dir, "explore");
      if (index === 2) {
        assert.match(JSON.stringify(toolReplyText(body)), /\/op new|停止|结束|执行|任务|closed|discussion/i);
        return { text: "The stopped task cannot resume execution without changing its original budget." };
      }
    }
    throw new Error(`unexpected exhausted-task request ${phase}/${index}`);
  });
  try {
    const dir = join(f.work, "task-exhausted");
    const first = await f.launch("exhaust", "Exercise the single-call local task budget.");
    assert.equal(f.requests.length, 2, "hard stop must not dispatch an extra provider request");
    assert.match(first.stdout + first.stderr, /预算|额度|exhausted|停止/);
    const path = join(dir, "host-model-budget.json"), old = load(path);
    assert.equal(old.records.length, 1); assert.equal(old.records[0].status, "answered");
    const bytes = readFileSync(path, "utf8");
    await f.launch("discuss", "Why did that task stop? Discuss only.");
    assert.equal(readFileSync(path, "utf8"), bytes);
    await f.launch("forbidden-old-verify", "Try to verify the stopped old task.");
    assert.equal(readFileSync(path, "utf8"), bytes);
    assert.equal(load(join(dir, "host-task.json")).records.length, 0, "old execution may not resume in discussion mode");
    assert.equal(existsSync(join(dir, "verify-1/result.json")), false);
    f.pass();
  } finally { await f.close(); }
});
