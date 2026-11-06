// Real DSH and MCP plumbing with a local provider only. No account or GPU is used.
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { chmodSync, existsSync, mkdtempSync, readFileSync, readdirSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { isAbsolute, join } from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";

const repo = fileURLToPath(new URL("../../../", import.meta.url));
const plugin = process.env.OPERATOR_AGENT_TEST_DSH_MODEL_RELAY_PLUGIN ??
  fileURLToPath(new URL("../index.mjs", import.meta.url));
const app = process.env.DSH_APP_ROOT ?? "/Applications/DSH Desktop.app/Contents/Resources/app";
const entry = join(app, "node_modules/@deepseek-ai/dsh/lib/bin.js");
const model = "host-budget-local-fixture";
const providerKey = "local-budget-fixture-key";
const disabled = ["session-title-llm", "llm-retry", "otel", "session-telemetry-otel",
  "tool-working-directory", "tool-bash", "tool-pwsh", "tool-jobs", "tool-fs", "tool-fs-search",
  "tool-skill", "tool-subagent-control", "tool-subagent-list-agents", "tool-subagent",
  "tool-subagent-fork", "tool-workflow", "tool-todo", "tool-goal", "web", "web-search-deepseek",
  "web-deepseek-api", "web-fetch-http", "tool-web"];

function listen(server) {
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      server.removeListener("error", reject);
      resolve(server.address().port);
    });
  });
}

function respondTool(response, name, args, index) {
  const chunk = (choices, extra = {}) => ({
    id: `budget-${index}`, model, created: 1, object: "chat.completion.chunk", choices, ...extra,
  });
  const tool = { index: 0, id: `budget-tool-${index}`, type: "function",
    function: { name, arguments: JSON.stringify(args) } };
  const events = [
    chunk([{ index: 0, delta: { role: "assistant", tool_calls: [tool] }, finish_reason: null }]),
    chunk([{ index: 0, delta: {}, finish_reason: "tool_calls" }]),
    chunk([], { usage: { prompt_tokens: 20, completion_tokens: 12, total_tokens: 32 } }),
  ];
  response.writeHead(200, { "Content-Type": "text/event-stream" });
  response.end(events.map(event => "data: " + JSON.stringify(event) + "\n\n").join("") + "data: [DONE]\n\n");
}

function decode(value) {
  if (typeof value === "string") {
    try { return decode(JSON.parse(value)); } catch { return null; }
  }
  if (Array.isArray(value)) {
    for (const item of value) { const found = decode(item); if (found) return found; }
  }
  if (value && typeof value === "object") {
    if ("ok" in value) return value;
    return decode(value.text ?? value.content);
  }
  return null;
}

async function runDsh(argv, env, cwd, label) {
  const child = spawn(process.execPath, ["--import", "data:text/javascript,console.debug%20%3D%20console.error%3B",
    entry, ...argv], { cwd, env, stdio: ["ignore", "pipe", "pipe"] });
  let stdout = "", stderr = "";
  child.stdout.setEncoding("utf8").on("data", value => { stdout += value; });
  child.stderr.setEncoding("utf8").on("data", value => { stderr += value; });
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; child.kill("SIGTERM"); }, 30_000);
  const hardStop = setTimeout(() => child.kill("SIGKILL"), 35_000);
  try {
    const code = await new Promise((resolve, reject) => {
      child.once("error", reject);
      child.once("close", resolve);
    });
    writeFileSync(join(cwd, `${label}.stdout.jsonl`), stdout);
    writeFileSync(join(cwd, `${label}.stderr.log`), stderr);
    assert.equal(timedOut, false, `DSH did not settle its bounded task; receipts: ${cwd}`);
    return { code, stdout, stderr };
  } finally { clearTimeout(timer); clearTimeout(hardStop); }
}

function makeWritable(dir) {
  chmodSync(dir, 0o700);
  for (const item of readdirSync(dir, { withFileTypes: true })) {
    if (item.isDirectory()) makeWritable(join(dir, item.name));
  }
}

for (const scenario of [
  { name: "call budget", maxModelCalls: 1, remainingCalls: 0,
    refusal: /模型调用额度已用完/ },
  { name: "cumulative token reservations", maxModelCalls: 3, remainingCalls: 2,
    refusal: /输出 token 预留额度已用完/ },
  { name: "in-flight main model cancellation", maxModelCalls: 3, remainingCalls: 2,
    cancelAfterText: true, refusal: /输出 token 预留额度已用完/ },
]) test(`real DSH ${scenario.cancelAfterText ? "cancels an in-flight main model request" : `rejects exhausted ${scenario.name} before provider`} and resumes discussion without changing credit`, {
  skip: process.env.OPERATOR_AGENT_TEST_DSH_HOST_BUDGET !== "1"
    ? "set OPERATOR_AGENT_TEST_DSH_HOST_BUDGET=1 for real DSH/local-provider integration"
    : !existsSync(entry) && "set DSH_APP_ROOT to an installed DSH",
  timeout: 80_000,
}, async (t) => {
  assert.ok(isAbsolute(plugin) && existsSync(plugin), "set an absolute plugin index.mjs");
  const installed = process.env.OPERATOR_AGENT_TEST_DSH_MODEL_RELAY_INSTALLED_PYTHON;
  const python = installed ?? process.env.OPERATOR_AGENT_PYTHON ??
    (existsSync(join(repo, ".venv/bin/python")) ? join(repo, ".venv/bin/python") : "python3");
  if (installed) assert.ok(isAbsolute(python), "installed Python must be absolute");
  const work = realpathSync(mkdtempSync(join(tmpdir(), "operator-dsh-host-budget-")));
  const runDir = join(work, "task");
  const sessionLog = join(work, "sessions.jsonl");
  const observer = join(work, "session-observer.mjs");
  writeFileSync(observer, `import {appendFileSync} from 'node:fs';
export function apply(ctx, config) {
  const agents = new Map();
  ctx.on('agent/created', ({agent}) => {
    agents.set(agent.session.id, agent);
    appendFileSync(config.path, JSON.stringify({sessionId:agent.session.id})+'\\n');
  });
  if (config.cancelAfterText) ctx.on('llm/stream', (options, next) => {
    if (options.purpose !== undefined) return next();
    return (async function* () {
      for await (const chunk of next()) {
        if (chunk.type === 'text-delta') {
          appendFileSync(config.path, JSON.stringify({cancelledAfterText:true,sessionId:options.sessionId})+'\\n');
          queueMicrotask(() => agents.get(options.sessionId).cancel({kind:'user'}));
        }
        yield chunk;
      }
    })();
  });
}
`);
  const requests = [], faults = [];
  let boundProviderClosed = false;
  const provider = createServer(async (request, response) => {
    try {
      const chunks = [];
      for await (const chunk of request) chunks.push(chunk);
      const body = JSON.parse(Buffer.concat(chunks).toString("utf8"));
      assert.equal(request.url, "/v1/chat/completions");
      assert.equal(request.headers.authorization, "Bearer " + providerKey);
      assert.equal(body.model, model);
      assert.equal(body.stream, true);
      requests.push(body);
      if (requests.length === 1) {
        const prepare = body.tools.find(tool => tool.function.name === "mcp__operator__operator_prepare");
        assert.ok(prepare?.function.parameters.properties.host_budget.properties.max_model_calls,
          "MCP prepare must expose the frozen model policy");
        respondTool(response, "mcp__operator__operator_prepare", {
          request: "relu fp32", shapes: [[4, 8]], hardware: "mock_gpu", dsl: "python", precision: "fp32",
          run_dir: runDir, host_budget: { wall_clock_s: 120, max_verifications: 3,
            max_model_calls: scenario.maxModelCalls, max_output_tokens_per_call: 128,
            total_output_token_reservations: 128 },
        }, 1);
      } else if (requests.length === 2) {
        assert.equal(body.max_tokens, 128, "native request must use the frozen per-call cap");
        const last = body.messages.filter(message => message.role === "tool").at(-1);
        const prepared = decode(last.content);
        assert.equal(prepared?.ok, true);
        assert.equal(prepared?.workflow, "host");
        const ledger = JSON.parse(readFileSync(join(runDir, "host-model-budget.json"), "utf8"));
        assert.equal(ledger.records.length, 1, "reservation must precede provider handoff");
        assert.equal(ledger.records[0].status, "reserved");
        assert.equal(ledger.records[0].max_output_tokens_reserved, 128);
        if (scenario.cancelAfterText) {
          response.writeHead(200, { "Content-Type": "text/event-stream" });
          response.write("data: " + JSON.stringify({ id: "budget-2", model, created: 1,
            object: "chat.completion.chunk", choices: [{ index: 0,
              delta: { role: "assistant", content: "local partial answer" }, finish_reason: null }] }) + "\n\n");
          const stop = setTimeout(() => response.destroy(new Error("local cancellation fixture timed out")), 10_000);
          stop.unref();
          response.on("close", () => { boundProviderClosed = true; clearTimeout(stop); });
        } else {
          respondTool(response, "mcp__operator__operator_context", { run_dir: runDir, backend: "fake" }, 2);
        }
      } else {
        assert.equal(requests.length, 3, "only one ordinary discussion after the stopped turn");
        response.writeHead(200, { "Content-Type": "text/event-stream" });
        response.end('data: '+JSON.stringify({id:'discussion',model,choices:[{index:0,delta:{role:'assistant',content:'Discussing existing results only.'},finish_reason:null}]})+'\n\n'+
          'data: '+JSON.stringify({id:'discussion',model,choices:[{index:0,delta:{},finish_reason:'stop'}]})+'\n\ndata: [DONE]\n\n');
      }
    } catch (error) {
      faults.push(error);
      response.writeHead(500).end("local fixture assertion failed");
    }
  });
  let passed = false;
  try {
    const port = await listen(provider);
    const patch = join(work, "patch.json");
    writeFileSync(patch, JSON.stringify([
      ...disabled.map(id => ({ id, disabled: true })),
      { id: "agent-default-model", config: { provider: "fixture", model } },
      { id: "llm-pi-ai", config: { providers: { fixture: { api: "openai-completions",
        baseURL: `http://127.0.0.1:${port}/v1`, apiKeyEnv: "HOST_BUDGET_FIXTURE_KEY",
        compat: { maxTokensField: "max_tokens" }, models: [{ id: model, contextWindow: 65536, maxTokens: 4096 }] } } } },
      { insert: [
        { id: "mcp-operator", name: join(app, "node_modules/@deepseek-ai/dsh-mcp-client/lib/index.js"),
          config: { serverName: "operator", transport: "stdio", command: python,
            args: [...(installed ? ["-I"] : []), "-m", "operator_agent.app.mcp_server"],
            cwd: installed ? work : repo, failOnStartupError: true, toolCallTimeoutMs: 30_000,
            env: { OPERATOR_AGENT_WORKFLOW: "host", ...(installed ? {} : { PYTHONPATH: repo }),
              OPERATOR_AGENT_OUTPUT_DIR: join(work, "output"), OPERATOR_AGENT_LOG_DIR: join(work, "log"),
              OPERATOR_AGENT_KERNEL_DB: join(work, "kernel_db.jsonl"), OPERATOR_AGENT_FAILURE_DB: join(work, "failure_db.jsonl") } } },
        { id: "operator-tools", name: plugin, config: { hostModelBudget: {} } },
        { id: "budget-session-observer", name: observer,
          config: { path: sessionLog, cancelAfterText: scenario.cancelAfterText ?? false } },
      ] },
    ]));
    const env = { PATH: process.env.PATH, HOME: work, DSH_HOME: join(work, "dsh-home"),
      DSH_TELEMETRY_DISABLED: "1", HOST_BUDGET_FIXTURE_KEY: providerKey,
      NO_PROXY: "127.0.0.1", no_proxy: "127.0.0.1" };
    const base = ["--profile", "headless", "--patch", patch, "--json"];
    const first = await runDsh([...base, "Use the operator host workflow for the local bounded fixture."], env, work, "initial");
    assert.deepEqual(faults, [], `fixture failure; receipts: ${work}`);
    assert.equal(requests.length, 2, `initial prepare plus one bound request only; receipts: ${work}`);
    const events = first.stdout.trim().split("\n").map(line => JSON.parse(line));
    if (scenario.cancelAfterText) {
      assert.ok(boundProviderClosed, "main-agent cancellation must abort the held provider connection");
      assert.match(readFileSync(sessionLog, "utf8"), /"cancelledAfterText":true/);
    } else {
      assert.match(first.stdout + first.stderr, scenario.refusal);
      assert.equal(events.find(event => event.type === "tool_result" && event.callId === "budget-tool-2")?.status,
        "completed", "the one admitted dispatch must complete its context tool call");
    }
    const ledgerFile = join(runDir, "host-model-budget.json");
    const before = JSON.parse(readFileSync(ledgerFile, "utf8"));
    assert.equal(before.records.length, 1);
    assert.equal(before.records[0].status, scenario.cancelAfterText ? "cancelled" : "answered");
    assert.equal(before.records[0].max_output_tokens_reserved, 128);
    assert.deepEqual(before.records[0].usage, scenario.cancelAfterText ? null
      : { inputTokens: 20, outputTokens: 12, totalTokens: 32 });
    assert.equal(before.limits.max_model_calls - before.records.length, scenario.remainingCalls);
    assert.equal(before.limits.total_output_token_reservations - before.records.reduce(
      (total, record) => total + record.max_output_tokens_reserved, 0), 0,
    "output reservations are exhausted even when model calls remain");
    const sessionId = JSON.parse(readFileSync(sessionLog, "utf8").trim().split("\n")[0]).sessionId;
    assert.ok(sessionId);
    const resumedPatch = JSON.parse(readFileSync(patch, "utf8"));
    const observerEntry = resumedPatch.flatMap(item => item.insert ?? []).find(item => item.id === "budget-session-observer");
    observerEntry.config.cancelAfterText = false; // cancel only the original in-flight task
    writeFileSync(patch, JSON.stringify(resumedPatch));
    const resumed = await runDsh([...base, "--session-id", sessionId,
      "Discuss the existing result without executing any task or changing its policy."], env, work, "resumed");
    assert.deepEqual(faults, [], `fixture failure after resume; receipts: ${work}`);
    assert.equal(requests.length, 3, "ordinary discussion is allowed outside the stopped task scope");
    assert.match(resumed.stdout, /Discussing existing results only/);
    const after = JSON.parse(readFileSync(ledgerFile, "utf8"));
    assert.equal(after.task_id, before.task_id);
    assert.deepEqual(after.records, before.records, "model credit and observed usage cannot be reset on resume");
    assert.deepEqual(after, before, "discussion must not rewrite the stopped task journal");
    passed = true;
  } finally {
    provider.closeAllConnections();
    await new Promise(resolve => provider.close(resolve));
    if (passed) { makeWritable(work); rmSync(work, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 }); }
    else t.diagnostic(`Preserved local host-budget receipts: ${work}`);
  }
});
