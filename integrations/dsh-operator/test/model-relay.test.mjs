import { test } from "node:test";
import assert from "node:assert/strict";
import { chmod, mkdtemp, rm, symlink, writeFile } from "node:fs/promises";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { startModelRelay } from "../model-relay.mjs";

const TOKEN = "relay_fixture_token_for_local_tests_only";
const MODEL = "fixture-model";
const BODY = {
  model: MODEL,
  temperature: 0.3,
  max_tokens: 128,
  stream: false,
  chat_template_kwargs: { enable_thinking: false },
  messages: [{ role: "system", content: "Return JSON" }, { role: "user", content: "写一个算子" }],
};

async function freePort() {
  const server = createServer();
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const { port } = server.address();
  await new Promise((resolve) => server.close(resolve));
  return port;
}

function host(stream, modelInfo = {}) {
  const calls = [];
  const ctx = { llm: { async prepareCall({ provider, model, ...controls }, signal, configure) {
    const config = { provider, model, ...configure(controls, modelInfo) };
    calls.push(config);
    return { config, stream };
  } } };
  return { ctx, calls };
}

async function fixture(t, ctx, options = {}) {
  const home = await mkdtemp(join(tmpdir(), "operator-relay-test-"));
  let close;
  t.after(async () => {
    await close?.();
    await rm(home, { recursive: true, force: true });
  });
  const tokenFile = join(home, "token");
  await writeFile(tokenFile, TOKEN + "\n", { mode: 0o600 });
  const port = await freePort();
  close = await startModelRelay(ctx, { port, provider: "fixture", model: MODEL, tokenFile, ...options });
  function post(body = BODY, { token = TOKEN, signal, path = "/v1/chat/completions" } = {}) {
    return fetch("http://127.0.0.1:" + port + path, {
      method: "POST", signal, body: JSON.stringify(body),
      headers: { "Content-Type": "application/json", Authorization: "Bearer " + token },
    });
  }
  return { post, close };
}

function* completion(usage) {
  yield { type: "block-start", blockType: "text", index: 0 };
  yield { type: "text-delta", text: "answer", index: 0 };
  yield { type: "block-end", block: { type: "text", text: "answer" }, index: 0 };
  if (usage !== undefined) yield { type: "usage", usage };
  yield { type: "finish", reason: { kind: "stop" } };
}

test("routes exactly one text completion with pinned sampling and normalized usage", async (t) => {
  const { ctx, calls } = host(function* ({ messages, purpose, signal }) {
    assert.deepEqual(messages.map(({ role, content }) => ({ role, content })),
      BODY.messages.map(({ role, content }) => ({ role, content: [{ type: "text", text: content }] })));
    assert.equal(purpose, "operator-delegation");
    assert.equal(signal.aborted, false);
    yield* completion({ inputTokens: 10, outputTokens: 2, cacheReadTokens: 4, cacheWriteTokens: 1, totalTokens: 17 });
  }, { reasoning: { efforts: [{ id: "off" }, { id: "high" }], defaultEffort: "high" } });
  const { post } = await fixture(t, ctx);
  const response = await post();
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), {
    choices: [{ message: { role: "assistant", content: "answer" }, finish_reason: "stop" }],
    usage: { input_tokens: 10, output_tokens: 2, cache_read_input_tokens: 4,
      cache_creation_input_tokens: 1, total_tokens: 17 },
  });
  assert.deepEqual(calls, [{ provider: "fixture", model: MODEL, temperature: 0.3,
    maxTokens: 128, reasoningEffort: "off" }]);
});

test("refuses unauthenticated, oversized and unsupported requests before model preparation", async (t) => {
  const { ctx, calls } = host(() => completion());
  const { post } = await fixture(t, ctx);
  assert.equal((await post(BODY, { token: TOKEN + "x" })).status, 401);
  assert.equal((await post(BODY, { path: "/other" })).status, 404);
  assert.equal((await post({ ...BODY, messages: [{ role: "user", content: "x".repeat(1 << 20) }] })).status, 413);
  for (const invalid of [
    { model: "other" }, { max_tokens: 40_961 }, { stream: true }, { tools: [] },
    { enable_thinking: true }, { chat_template_kwargs: { enable_thinking: true } },
    { messages: [{ role: "tool", content: "result" }] },
    { messages: [{ role: "user", content: [{ type: "image", data: "x" }] }] },
  ]) {
    assert.equal((await post({ ...BODY, ...invalid })).status, 400);
  }
  assert.deepEqual(calls, []);
});

test("omits adapter-default zero usage and refuses incomplete output", async (t) => {
  let finish = true;
  let usage = { inputTokens: 0, outputTokens: 0, totalTokens: 0 };
  const { ctx } = host(function* () {
    yield { type: "text-delta", text: "answer", index: 0 };
    yield { type: "usage", usage };
    if (finish) yield { type: "finish", reason: { kind: "stop" } };
  });
  const { post } = await fixture(t, ctx);
  const response = await post();
  assert.equal(response.status, 200);
  assert.equal("usage" in await response.json(), false);
  usage = { inputTokens: 10, outputTokens: 2, cacheReadTokens: 4, cacheWriteTokens: 1, totalTokens: 12 };
  const inclusive = await post();
  assert.equal(inclusive.status, 200);
  assert.deepEqual((await inclusive.json()).usage, { input_tokens: 5, output_tokens: 2,
    cache_read_input_tokens: 4, cache_creation_input_tokens: 1, total_tokens: 12 });
  usage = { inputTokens: 10, outputTokens: 2, cacheReadTokens: 4 };
  assert.equal("usage" in await (await post()).json(), false);
  finish = false;
  const truncated = await post();
  assert.equal(truncated.status, 502);
  assert.deepEqual(await truncated.json(), { error: { code: "model_call_failed" } });
});

test("aborts the provider stream on client disconnect and relay disposal", { timeout: 5000 }, async (t) => {
  const entered = [];
  const stopped = [];
  const { ctx } = host(async function* ({ signal }) {
    entered.shift()();
    await new Promise((resolve) => signal.addEventListener("abort", resolve, { once: true }));
    stopped.shift()(signal.aborted);
  });
  const { post, close } = await fixture(t, ctx);
  for (const dispose of [false, true]) {
    const controller = new AbortController();
    const ready = new Promise((resolve) => entered.push(resolve));
    const aborted = new Promise((resolve) => stopped.push(resolve));
    const request = post(BODY, { signal: controller.signal });
    const rejected = assert.rejects(request);
    await ready;
    if (dispose) await close();
    else controller.abort();
    assert.equal(await aborted, true);
    await rejected;
  }
  await close();
});

test("aborts and returns a deadline failure when the provider stalls", { timeout: 5000 }, async (t) => {
  let aborted = false;
  const { ctx } = host(async function* ({ signal }) {
    await new Promise((resolve) => signal.addEventListener("abort", resolve, { once: true }));
    aborted = signal.aborted;
  });
  const { post } = await fixture(t, ctx, { timeoutMs: 500 });
  const response = await post();
  assert.equal(response.status, 504);
  assert.deepEqual(await response.json(), { error: { code: "model_deadline" } });
  assert.equal(aborted, true);
});

test("limits simultaneous model work before preparing another call", { timeout: 5000 }, async (t) => {
  let started = 0;
  let signalReady;
  let signalRelease;
  const ready = new Promise((resolve) => { signalReady = resolve; });
  const release = new Promise((resolve) => { signalRelease = resolve; });
  const { ctx, calls } = host(async function* ({ signal }) {
    if (++started === 4) signalReady();
    await Promise.race([release, new Promise((resolve) => signal.addEventListener("abort", resolve, { once: true }))]);
    yield* completion();
  });
  const { post } = await fixture(t, ctx);
  const results = Promise.all(Array.from({ length: 4 }, () => post()));
  try {
    await ready;
    assert.equal((await post()).status, 429);
    assert.equal(calls.length, 4);
  } finally {
    signalRelease();
    assert.deepEqual((await results).map((response) => response.status), [200, 200, 200, 200]);
  }
});

test("rejects a public or symlinked bearer secret file before listening", async (t) => {
  const home = await mkdtemp(join(tmpdir(), "operator-relay-test-"));
  t.after(() => rm(home, { recursive: true, force: true }));
  const tokenFile = join(home, "token");
  const link = join(home, "link");
  await writeFile(tokenFile, TOKEN, { mode: 0o644 });
  await chmod(tokenFile, 0o644);
  await symlink(tokenFile, link);
  const port = await freePort();
  const { ctx } = host(() => completion());
  const config = { port, provider: "fixture", model: MODEL, tokenFile };
  await assert.rejects(startModelRelay(ctx, config), /mode 0600/);
  await assert.rejects(startModelRelay(ctx, { ...config, tokenFile: link }));
});
