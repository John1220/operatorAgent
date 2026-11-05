// Opt in because this test needs a real DSH installation and local HTTP listeners.
// The provider is a loopback fixture. No GPU or provider account is used.
import assert from "node:assert/strict";
import { execFile, spawn } from "node:child_process";
import { chmodSync, existsSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { createConnection, createServer as createTcpServer } from "node:net";
import { tmpdir } from "node:os";
import { isAbsolute, join } from "node:path";
import { randomBytes } from "node:crypto";
import { fileURLToPath, pathToFileURL } from "node:url";
import { promisify } from "node:util";
import { test } from "node:test";

const execFileAsync = promisify(execFile);
const repo = fileURLToPath(new URL("../../../", import.meta.url));
const plugin = process.env.OPERATOR_AGENT_TEST_DSH_MODEL_RELAY_PLUGIN ??
  fileURLToPath(new URL("../index.mjs", import.meta.url));
const app = process.env.DSH_APP_ROOT ?? "/Applications/DSH Desktop.app/Contents/Resources/app";
const entry = join(app, "node_modules/@deepseek-ai/dsh/lib/bin.js");
const outerModel = "relay-test-outer";
const innerModel = "relay-test-inner";
const providerKey = "fixture-provider-key";
const responseText = "bounded host answer";
const sketch = JSON.stringify({
  algorithm: "matmul", tiling: { block_m: 16, block_n: 16, block_k: 16 },
  memory_plan: {}, pipeline: [], optimization_hints: [],
});
const source = "import numpy as np\n\ndef kernel(A, B):\n    return np.matmul(A, B)\n";
const messages = [
  { role: "system", content: "Keep the frozen kernel contract." },
  { role: "user", content: "Write an FP16 Softmax kernel for shape 4096×4096." },
];

const pythonScript = `
import json
import sys
from operator_agent.execute.llm_budget import LLMCallLimits, ProductionBudgetExhausted, ProductionLLMCallBudget
from operator_agent.execute.llm_client import LLMClient

budget = ProductionLLMCallBudget(
    LLMCallLimits(max_calls=1, max_output_tokens=64, total_output_token_reservations=64,
                  call_seconds=10, allowed_callers=("Coder",)),
    sys.argv[1],
)
client = LLMClient(timeout=10, call_budget=budget)
messages = json.loads(sys.argv[2])
answer = client.chat(messages, temperature=0.1, max_tokens=64, caller_tag="Coder")
try:
    client.chat(messages, temperature=0.1, max_tokens=64, caller_tag="Coder")
except ProductionBudgetExhausted as exc:
    assert exc.reason == "llm_call_budget"
else:
    raise AssertionError("task dispatched beyond the one-call budget")
print(json.dumps({"answer": answer, "budget": budget.summary()}))
`;

function listen(server) {
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      server.removeListener("error", reject);
      resolve(server.address().port);
    });
  });
}

function close(server) {
  return new Promise((resolve) => server.close(resolve));
}

function makeWritable(dir) {
  // Operator evidence directories are intentionally sealed to mode 0500.
  chmodSync(dir, 0o700);
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (entry.isDirectory()) makeWritable(join(dir, entry.name));
  }
}

function finish(response, model, content, usage) {
  const chunk = (choices, extra = {}) => ({
    id: "relay-fixture", model, created: 1, object: "chat.completion.chunk", choices, ...extra,
  });
  const events = [
    chunk([{ index: 0, delta: { role: "assistant", content }, finish_reason: null }]),
    chunk([{ index: 0, delta: {}, finish_reason: "stop" }]),
  ];
  if (usage) events.push(chunk([], { usage }));
  response.writeHead(200, { "Content-Type": "text/event-stream" });
  response.end(events.map((event) => "data: " + JSON.stringify(event) + "\n\n").join("") +
    "data: [DONE]\n\n");
}

function withTimeout(promise, ms, message) {
  let timer;
  const timeout = new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(message)), ms); });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

async function waitForRelay(port) {
  const deadline = Date.now() + 5_000;
  while (Date.now() < deadline) {
    const ready = await new Promise((resolve) => {
      const socket = createConnection({ host: "127.0.0.1", port });
      socket.once("connect", () => { socket.destroy(); resolve(true); });
      socket.once("error", () => resolve(false));
      socket.setTimeout(200, () => { socket.destroy(); resolve(false); });
    });
    if (ready) return;
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  throw new Error("DSH model relay did not listen");
}

async function verifyWorker(python, pythonArgs, cwd, env, work) {
  const fromApp = (name) => import(pathToFileURL(join(app, "node_modules", name)).href);
  const [{ Client }, { StdioClientTransport }] = await Promise.all([
    fromApp("@modelcontextprotocol/client/dist/index.mjs"),
    fromApp("@modelcontextprotocol/client/dist/stdio.mjs"),
  ]);
  const client = new Client({ name: "relay-host-test", version: "1" }, { capabilities: {} });
  await client.connect(new StdioClientTransport({
    command: python, args: [...pythonArgs, "-m", "operator_agent.app.mcp_server"], cwd, env,
  }));
  const tool = async (name, args) => {
    const reply = await client.callTool({ name, arguments: args });
    const body = JSON.parse(reply.content[0].text);
    assert.ok(!reply.isError && body.ok, JSON.stringify(body));
    return body;
  };
  const runDir = join(work, "operator-run");
  let invocation;
  let status;
  try {
    const launched = await tool("operator_run", {
      run_dir: runDir, request: "matmul fp32", shapes: [[8, 8, 8]], dsl: "python", precision: "fp32",
      backend: "fake", fake: { llm: "live" }, max_rounds: 1, wall_clock_s: 20, max_llm_calls: 2,
    });
    invocation = { run_dir: runDir, invocation_id: launched.invocation_id };
    const deadline = Date.now() + 16_000;
    do {
      status = await tool("operator_status", invocation);
      if (status.phase !== "running") break;
      await new Promise((resolve) => setTimeout(resolve, 100));
    } while (Date.now() < deadline);
    assert.equal(status.phase, "done", JSON.stringify(status));
    assert.equal(status.execution_context.model_mode, "live");
    assert.equal(status.execution_context.test_only, true);
    assert.deepEqual(status.summary.levels, { L1: true, L2: true, L3: false });
    const journal = JSON.parse(readFileSync(join(runDir, "model-journal/summary.json"), "utf8"));
    assert.deepEqual(journal.attempts.map(({ caller }) => caller), ["Designer", "Coder"]);
    assert.equal(journal.dispatched_calls, 2);
    assert.equal(journal.api_usage.known_calls, 2);
    assert.equal(journal.api_usage.totals.total_tokens, 36);
  } finally {
    try {
      if (invocation && (!status || status.phase === "running")) {
        await tool("operator_cancel", invocation);
      }
    } finally {
      await client.close();
    }
  }
}

test("bounded Python model call uses the real DSH host route and returns provider usage", {
  skip: process.env.OPERATOR_AGENT_TEST_DSH_MODEL_RELAY !== "1"
    ? "set OPERATOR_AGENT_TEST_DSH_MODEL_RELAY=1 to run the local DSH integration"
    : !existsSync(entry) && "set DSH_APP_ROOT to a DSH installation",
  timeout: 45_000,
}, async () => {
  assert.ok(isAbsolute(plugin) && existsSync(plugin), "set an absolute path to the plugin index.mjs");
  const work = mkdtempSync(join(tmpdir(), "operator-dsh-relay-"));
  const tokenFile = join(work, "model-relay.token");
  writeFileSync(tokenFile, randomBytes(32).toString("hex") + "\n", { mode: 0o600 });
  const portProbe = createTcpServer();
  const relayPort = await listen(portProbe);
  await close(portProbe);
  const installedPython = process.env.OPERATOR_AGENT_TEST_DSH_MODEL_RELAY_INSTALLED_PYTHON;
  const python = installedPython ?? process.env.OPERATOR_AGENT_PYTHON ??
    (existsSync(join(repo, ".venv/bin/python")) ? join(repo, ".venv/bin/python") : "python3");
  if (installedPython) assert.ok(isAbsolute(python), "set an absolute path to the installed Python");
  const pythonArgs = installedPython ? ["-I"] : [];
  const pythonCwd = installedPython ? work : repo;

  const requests = [];
  const faults = [];
  let releaseOuter;
  const outerRequest = new Promise((resolve) => { releaseOuter = resolve; });
  const provider = createServer(async (request, response) => {
    try {
      const chunks = [];
      for await (const chunk of request) chunks.push(chunk);
      const body = JSON.parse(Buffer.concat(chunks).toString("utf8"));
      assert.equal(request.url, "/v1/chat/completions");
      assert.equal(request.headers.authorization, "Bearer " + providerKey);
      assert.equal(body.stream, true);
      requests.push(body);
      if (body.model === outerModel) {
        releaseOuter(response);
        return;
      }
      assert.equal(body.model, innerModel);
      const output = new Map([[64, responseText], [2048, sketch], [40960, source]]).get(body.max_tokens);
      assert.notEqual(output, undefined, `unexpected worker request cap ${body.max_tokens}`);
      finish(response, innerModel, output, { prompt_tokens: 13, completion_tokens: 5, total_tokens: 18 });
    } catch (error) {
      faults.push(error);
      response.writeHead(400).end();
    }
  });
  let child;
  let exited;
  let stderr = "";
  try {
    const providerPort = await listen(provider);
    const patch = join(work, "profile.patch.json");
    writeFileSync(patch, JSON.stringify([
      ...["session-title-llm", "llm-retry", "otel", "session-telemetry-otel"]
        .map((id) => ({ id, disabled: true })),
      { id: "agent-default-model", config: { provider: "fixture", model: outerModel } },
      { id: "llm-pi-ai", config: { providers: { fixture: {
        api: "openai-completions", baseURL: `http://127.0.0.1:${providerPort}/v1`,
        apiKeyEnv: "MODEL_RELAY_TEST_PROVIDER_KEY", compat: { maxTokensField: "max_tokens" },
        models: [outerModel, innerModel].map((id) => ({ id, contextWindow: 65536, maxTokens: 40960 })),
      } } } },
      { insert: [{ id: "operator-tools", name: plugin, config: {
        modelRelay: { provider: "fixture", model: innerModel, port: relayPort, tokenFile },
      } }] },
    ]));
    child = spawn(process.execPath, [
      "--import", "data:text/javascript,console.debug%20%3D%20console.error%3B",
      entry, "--profile", "headless", "--patch", patch, "--json", "Wait for the local test.",
    ], {
      cwd: work,
      env: { PATH: process.env.PATH, HOME: work, DSH_HOME: join(work, "dsh"),
        MODEL_RELAY_TEST_PROVIDER_KEY: providerKey, NO_PROXY: "127.0.0.1", no_proxy: "127.0.0.1" },
      stdio: ["ignore", "ignore", "pipe"],
    });
    child.stderr.setEncoding("utf8").on("data", (data) => { stderr += data; });
    exited = new Promise((resolve, reject) => {
      child.once("error", reject);
      child.once("close", resolve);
    });
    const outer = await withTimeout(Promise.race([
      outerRequest,
      exited.then((code) => { throw new Error(`DSH exited before provider request: ${code}\n${stderr}`); }),
    ]), 18_000, "DSH did not reach the loopback fixture");
    await waitForRelay(relayPort);

    const env = { PATH: process.env.PATH, HOME: work, LLM_BASE_URL: `http://127.0.0.1:${relayPort}/v1`,
      LLM_MODEL: innerModel, LLM_API_KEY_FILE: tokenFile };
    const { stdout } = await execFileAsync(python, [...pythonArgs, "-c", pythonScript, join(work, "journal"),
      JSON.stringify(messages)], { cwd: pythonCwd, timeout: 15_000, env });
    const result = JSON.parse(stdout);
    assert.equal(result.answer, responseText);
    assert.equal(result.budget.dispatched_calls, 1);
    assert.equal(result.budget.stop_reason, "llm_call_budget");
    assert.equal(result.budget.api_usage.known_calls, 1);
    assert.equal(result.budget.api_usage.totals.total_tokens, 18);
    const inner = requests.filter((request) => request.model === innerModel);
    assert.equal(inner.length, 1);
    assert.deepEqual(inner[0].messages.map(({ role, content }) => ({ role, content })), messages);
    assert.equal(inner[0].temperature, 0.1);
    assert.equal(inner[0].max_tokens, 64);
    await verifyWorker(python, pythonArgs, pythonCwd, env, work);
    assert.deepEqual(requests.filter(({ model }) => model === innerModel)
      .map(({ max_tokens }) => max_tokens), [64, 2048, 40960]);
    assert.deepEqual(faults, []);
    finish(outer, outerModel, "Finished the local test.");
    assert.equal(await withTimeout(exited, 5_000, "DSH did not stop"), 0, stderr);
  } finally {
    if (child?.exitCode === null) {
      child.kill("SIGKILL");
      await exited;
    }
    provider.closeAllConnections();
    await close(provider);
    makeWritable(work);
    rmSync(work, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 });
  }
});
