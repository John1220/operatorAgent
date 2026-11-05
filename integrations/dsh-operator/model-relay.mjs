/** Bounded local bridge from the Python operator worker to the DSH model service. */
import { randomUUID, timingSafeEqual } from "node:crypto";
import { constants } from "node:fs";
import { open } from "node:fs/promises";
import { createServer } from "node:http";
import { isAbsolute } from "node:path";

const BODY_LIMIT = 1 << 20;
const MAX_TOKENS = 40_960;
const MAX_STREAM_CHUNKS = 65_536;
const MAX_ACTIVE = 4;
const DEFAULT_TIMEOUT_MS = 120_000;
const COMPLETIONS = "/v1/chat/completions";
const REQUEST_KEYS = new Set([
  "model", "messages", "temperature", "max_tokens", "stream",
  "chat_template_kwargs", "enable_thinking",
]);

function object(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function counter(value) {
  return Number.isSafeInteger(value) && value >= 0;
}

async function readToken(path) {
  if (typeof path !== "string" || !isAbsolute(path) || constants.O_NOFOLLOW === undefined) {
    throw new Error("modelRelay tokenFile must be an absolute, private regular file");
  }
  const file = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK);
  try {
    const stat = await file.stat();
    if (!stat.isFile() || (stat.mode & 0o777) !== 0o600 || stat.uid !== process.getuid?.()
        || stat.size > 130) {
      throw new Error("modelRelay tokenFile must be owned by the host user and mode 0600");
    }
    const buffer = Buffer.alloc(131);
    const { bytesRead } = await file.read(buffer);
    if (bytesRead !== stat.size) throw new Error("modelRelay tokenFile changed while being read");
    const token = buffer.subarray(0, bytesRead).toString("utf8").replace(/\r?\n$/, "");
    if (!/^[a-zA-Z0-9_-]{32,128}$/.test(token)) {
      throw new Error("modelRelay token must contain 32 to 128 URL-safe characters");
    }
    return Buffer.from("Bearer " + token);
  } finally {
    await file.close();
  }
}

function authorized(header, expected) {
  if (typeof header !== "string") return false;
  const actual = Buffer.from(header);
  return actual.length === expected.length && timingSafeEqual(actual, expected);
}

function invalidRequest() {
  return new Error("unsupported_model_request");
}

function requestOptions(body, provider, model) {
  if (!object(body) || Object.keys(body).some((key) => !REQUEST_KEYS.has(key))
      || body.model !== model || (body.stream !== undefined && body.stream !== false)
      || typeof body.temperature !== "number" || !Number.isFinite(body.temperature)
      || body.temperature < 0 || body.temperature > 2
      || !Number.isSafeInteger(body.max_tokens) || body.max_tokens < 1 || body.max_tokens > MAX_TOKENS
      || (body.enable_thinking !== undefined && body.enable_thinking !== false)
      || !Array.isArray(body.messages) || body.messages.length < 1 || body.messages.length > 128) {
    throw invalidRequest();
  }
  const nested = body.chat_template_kwargs;
  if (nested !== undefined && (!object(nested) || nested.enable_thinking !== false
      || Object.keys(nested).length !== 1)) {
    throw invalidRequest();
  }
  const messages = body.messages.map((item) => {
    if (!object(item) || Object.keys(item).length !== 2 || typeof item.content !== "string") {
      throw invalidRequest();
    }
    const { role } = item;
    let source;
    switch (role) {
      case "user": source = { kind: "operator-model-relay" }; break;
      case "system": source = { kind: "system-prompt" }; break;
      case "assistant": source = { kind: "model", provider, model }; break;
      default: throw invalidRequest();
    }
    return { id: randomUUID(), role, source, content: [{ type: "text", text: item.content }] };
  });
  return { controls: { provider, model, temperature: body.temperature, maxTokens: body.max_tokens }, messages };
}

function disableReasoning(controls, modelInfo) {
  if (modelInfo.reasoning === undefined) return controls;
  if (!modelInfo.reasoning.efforts.some(({ id }) => id === "off")) {
    throw new Error("configured model does not support disabling reasoning");
  }
  return { ...controls, reasoningEffort: "off" };
}

function usageReceipt(usage) {
  if (!object(usage) || !counter(usage.inputTokens) || !counter(usage.outputTokens)) return;
  const read = usage.cacheReadTokens ?? 0;
  const write = usage.cacheWriteTokens ?? 0;
  if (!counter(read) || !counter(write)) return;
  const cache = read + write;
  const base = usage.inputTokens + usage.outputTokens;
  const total = usage.totalTokens;
  // Some adapters initialize omitted provider counters to zero.
  if (base + cache === 0 || !counter(base + cache)) return;
  let uncached = usage.inputTokens;
  if (total === undefined) {
    if (cache !== 0) return;
  } else if (total !== base + cache) {
    if (total !== base || cache > uncached) return;
    uncached -= cache;
  }
  return { input_tokens: uncached, output_tokens: usage.outputTokens,
    ...(total === undefined ? {} : { total_tokens: total }),
    ...(read ? { cache_read_input_tokens: read } : {}),
    ...(write ? { cache_creation_input_tokens: write } : {}) };
}

async function complete(llm, { controls, messages }, signal) {
  // Preparation pins the adapter generation through dispatch and enforces the
  // requested sampling controls even when host settings reload in the meantime.
  const prepared = await llm.prepareCall(controls, signal, disableReasoning);
  signal.throwIfAborted();
  const blocks = new Map();
  let usage;
  let finished = false;
  let bytes = 0;
  let count = 0;
  for await (const chunk of prepared.stream({
    ...prepared.config, messages, signal, purpose: "operator-delegation",
  })) {
    signal.throwIfAborted();
    if (finished || ++count > MAX_STREAM_CHUNKS) throw new Error("invalid provider stream");
    if (chunk.type === "finish") {
      if (chunk.reason?.kind !== "stop") throw new Error("incomplete model output");
      finished = true;
    } else if (chunk.type === "usage") {
      usage = usageReceipt(chunk.usage);
    } else if (chunk.type === "text-delta" || chunk.type === "block-start" || chunk.type === "block-end") {
      const type = chunk.type === "text-delta" ? "text"
        : chunk.type === "block-start" ? chunk.blockType : chunk.block?.type;
      if (type === "reasoning") continue;
      if (type !== "text" || !counter(chunk.index)) throw new Error("unsupported provider block");
      const block = blocks.get(chunk.index) ?? { text: "", closed: false };
      if (block.closed) throw new Error("duplicate provider block");
      const text = chunk.type === "text-delta" ? chunk.text
        : chunk.type === "block-end" ? chunk.block.text : "";
      if (typeof text !== "string") throw new Error("invalid provider text");
      if (chunk.type === "block-end") {
        bytes -= Buffer.byteLength(block.text);
        block.text = text;
        block.closed = true;
      } else {
        block.text += text;
      }
      bytes += Buffer.byteLength(text);
      if (bytes > BODY_LIMIT) throw new Error("provider response exceeds local limit");
      blocks.set(chunk.index, block);
    } else if (chunk.type !== "reasoning-delta") {
      throw new Error("unsupported provider stream event");
    }
  }
  signal.throwIfAborted();
  const content = [...blocks.values()].map((block) => block.text).join("");
  if (!finished || !content.trim()
      || (usage?.completion_tokens ?? usage?.output_tokens ?? 0) > controls.maxTokens) {
    throw new Error("invalid model completion");
  }
  const response = JSON.stringify({
    choices: [{ message: { role: "assistant", content }, finish_reason: "stop" }],
    ...(usage === undefined ? {} : { usage }) });
  if (Buffer.byteLength(response) > BODY_LIMIT) throw new Error("provider response exceeds local limit");
  return response;
}

function send(response, status, body) {
  if (response.writableEnded || response.destroyed) return;
  response.writeHead(status, { "Content-Type": "application/json", "Connection": "close" });
  response.end(body);
}

function failure(response, status, code) {
  send(response, status, JSON.stringify({ error: { code } }));
}

/** Start a private IPv4-loopback relay. Return an idempotent async disposer. */
export async function startModelRelay(ctx, { port, provider, model, tokenFile, timeoutMs = DEFAULT_TIMEOUT_MS }) {
  if (!Number.isSafeInteger(port) || port < 1 || port > 65_535
      || typeof provider !== "string" || !provider.trim() || provider.trim() !== provider
      || typeof model !== "string" || !model.trim() || model.trim() !== model
      || !Number.isSafeInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > DEFAULT_TIMEOUT_MS
      || typeof ctx.llm?.prepareCall !== "function") {
    throw new Error("modelRelay requires a valid port, provider, model, timeout and DSH model service");
  }
  const credential = await readToken(tokenFile);
  const active = new Set();
  let closing;
  const server = createServer({ requestTimeout: timeoutMs, headersTimeout: timeoutMs }, async (request, response) => {
    if (request.method !== "POST" || request.url !== COMPLETIONS) {
      return failure(response, 404, "unknown_endpoint");
    }
    if (!authorized(request.headers.authorization, credential)) {
      return failure(response, 401, "unauthorized");
    }
    const length = Number(request.headers["content-length"]);
    if (length > BODY_LIMIT) return failure(response, 413, "request_too_large");
    if (!Number.isSafeInteger(length) || length < 1 || request.headers["transfer-encoding"] !== undefined
        || !/^application\/json(?:\s*;|$)/i.test(request.headers["content-type"] ?? "")) {
      return failure(response, 400, "unsupported_model_request");
    }
    if (closing || active.size >= MAX_ACTIVE) return failure(response, 429, "relay_busy");

    const controller = new AbortController();
    active.add(controller);
    response.once("close", () => {
      if (!response.writableFinished) controller.abort();
    });
    const timeout = setTimeout(() => {
      controller.abort();
      failure(response, 504, "model_deadline");
      if (!request.complete) request.destroy();
    }, timeoutMs);
    let parsed = false;
    try {
      const chunks = [];
      let received = 0;
      for await (const chunk of request) {
        received += chunk.length;
        if (received > length) throw invalidRequest();
        chunks.push(chunk);
      }
      controller.signal.throwIfAborted();
      const body = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(Buffer.concat(chunks)));
      const options = requestOptions(body, provider, model);
      parsed = true;
      send(response, 200, await complete(ctx.llm, options, controller.signal));
    } catch {
      if (!controller.signal.aborted) {
        failure(response, parsed ? 502 : 400, parsed ? "model_call_failed" : "unsupported_model_request");
      }
    } finally {
      clearTimeout(timeout);
      active.delete(controller);
    }
  });
  server.maxConnections = MAX_ACTIVE * 4;
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, "127.0.0.1", () => {
      server.removeListener("error", reject);
      resolve();
    });
  });
  return () => closing ??= new Promise((resolve) => {
    for (const controller of active) controller.abort();
    server.close(resolve);
    server.closeAllConnections();
  });
}
