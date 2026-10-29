/** Offline test-only DSH LLM runtime + installed pi-ai adapter, loopback only. */
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import path from 'node:path';

const base = new URL(process.argv[2]);
assert.equal(base.protocol, 'http:');
assert.equal(base.hostname, '127.0.0.1');
assert.match(base.port, /^\d+$/);
assert.equal(base.pathname, '/v1');
const root = process.argv[3];
const {Context} = await import(pathToFileURL(path.join(root, 'cordis/lib/index.js')));
const {LlmRuntime} = await import(pathToFileURL(path.join(root, 'dsh-llm/lib/index.js')));
const piAi = await import(pathToFileURL(path.join(root, 'dsh-llm-pi-ai/lib/index.js')));
const context = new Context();
context.plugin(LlmRuntime);
context.plugin(piAi, {providers:{fixture:{api:'openai-completions',baseURL:base.href,
  apiKeyEnv:'S5_FAKE_API_KEY', models:[{id:'fake-model',contextWindow:4096,maxTokens:20}],
  compat:{maxTokensField:'max_tokens'}}}});
await new Promise(resolve => setTimeout(resolve, 100));
const options = {provider:'fixture',model:'fake-model',maxTokens:20,
  messages:[{role:'user',content:[{type:'text',text:'fixture message'}]}]};
async function run() {
  const chunks=await Array.fromAsync(context.llm.stream(options));
  return {text:chunks.filter(c=>c.type==='text-delta').map(c=>c.text).join(''),
          reason:chunks.find(c=>c.type==='finish')?.reason?.kind};
}
const first=await run();
const second=await run();
process.stdout.write(JSON.stringify({first,second})+'\n');
process.exit(0);
