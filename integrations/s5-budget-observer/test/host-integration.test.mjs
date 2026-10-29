/** Offline integration: installed DSH waterfall, fake provider, real local ledger. */
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {existsSync,mkdtempSync,rmSync,renameSync} from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {pathToFileURL} from 'node:url';
import {createCliLedger} from '../cli-ledger.mjs';
import {installS5Interceptor} from '../interceptor.mjs';

const ROOT=path.resolve(import.meta.dirname,'../../..');
// 需显式提供：S5_DSH_ROOT=<...>/@deepseek-ai、S5_PYTHON=<abs python>；缺了就 skip，
// 不把某台机器的家目录写进仓库。
const APP=process.env.S5_DSH_ROOT || '';
const PYTHON=process.env.S5_PYTHON || '';
const available=existsSync(path.join(APP,'cordis/lib/index.js')) && existsSync(path.join(APP,'dsh-llm/lib/index.js')) && existsSync(PYTHON);
function command(action,args){
 const p=spawnSync(PYTHON,['-m','operator_agent.s5_shared_budget',action,...args],{cwd:ROOT,encoding:'utf8',env:{PATH:process.env.PATH,PYTHONNOUSERSITE:'1'}});
 if(p.status!==0) throw Error(`ledger ${action} failed: ${p.stdout}`);
 return JSON.parse(p.stdout);
}
function makeLedger(max=2){
 const dir=mkdtempSync(path.join(os.tmpdir(),'s5-host-integration-')), file=path.join(dir,'budget.sqlite');
 command('create',['--ledger',file,'--task-id','test-task','--max-calls',String(max),'--max-output-tokens','40960','--total-output-tokens',String(max*40960),'--wall-clock-s','60']);
 return {dir,file,bridge:createCliLedger({python:PYTHON,cwd:ROOT,ledgerPath:file,taskId:'test-task'})};
}

test('installed DSH waterfall shares budget with Python and never trusts normalized zeros',{skip:!available},async()=>{
 const {Context}=await import(pathToFileURL(path.join(APP,'cordis/lib/index.js')));
 const {LlmRuntime,LlmAdapter}=await import(pathToFileURL(path.join(APP,'dsh-llm/lib/index.js')));
 const context=new Context();context.plugin(LlmRuntime);await new Promise(r=>setTimeout(r,80));
 const {dir,file,bridge}=makeLedger();let calls=0;
 try{
  command('reserve',['--ledger',file,'--participant','inner','--caller','inner.test','--max-output-tokens','512','--request-sha256','a'.repeat(64)]);
  class Fake extends LlmAdapter {async *stream(){calls++;yield {type:'usage',usage:{inputTokens:0,outputTokens:0,totalTokens:0}};yield {type:'finish',reason:{kind:'stop'}};}}
  context.llm.registerAdapter(['fixture'],new Fake());installS5Interceptor(context,bridge);
  const req={provider:'fixture',model:'test',messages:[],maxTokens:1024,purpose:'session-title'};
  const first=context.llm.stream(req); const chunks=await Array.fromAsync(first);assert.equal(chunks.length,2);assert.equal(calls,1);
  assert.deepEqual(await Array.fromAsync(first),[]);assert.equal(calls,1);
  await assert.rejects(Array.fromAsync(context.llm.stream(req)),/reserve unavailable or rejected/);assert.equal(calls,1);
  const snapshot=await bridge.snapshot();assert.equal(snapshot.reserved_calls,2);
  const outer=snapshot.calls.find(row=>row.participant==='outer');assert.equal(outer.caller,'dsh.outer.session-title');
  assert.equal(outer.outcome,'answered');assert.equal(outer.usage_evidence,'unverified');assert.equal(outer.api_usage,null);
 }finally{rmSync(dir,{recursive:true,force:true});}
});

test('installed DSH maps fake adapter exception to failure and charges it',{skip:!available},async()=>{
 const {Context}=await import(pathToFileURL(path.join(APP,'cordis/lib/index.js')));
 const {LlmRuntime,LlmAdapter}=await import(pathToFileURL(path.join(APP,'dsh-llm/lib/index.js')));
 const context=new Context();context.plugin(LlmRuntime);await new Promise(r=>setTimeout(r,80));
 const {dir,bridge}=makeLedger(1);
 try {
  class Broken extends LlmAdapter {async *stream(){throw Error('provider fault');}}
  context.llm.registerAdapter(['fixture'],new Broken());installS5Interceptor(context,bridge);
  const chunks=await Array.fromAsync(context.llm.stream({provider:'fixture',model:'test',messages:[],maxTokens:64}));
  assert.equal(chunks.at(-1).reason.kind,'error');
  const result=await bridge.snapshot();assert.equal(result.calls[0].outcome,'failed');assert.equal(result.calls[0].usage_evidence,'missing');
 }finally{rmSync(dir,{recursive:true,force:true});}
});

test('mismatched task identity refuses before reservation or adapter',{skip:!available},async()=>{
 const {Context}=await import(pathToFileURL(path.join(APP,'cordis/lib/index.js')));
 const {LlmRuntime,LlmAdapter}=await import(pathToFileURL(path.join(APP,'dsh-llm/lib/index.js')));
 const context=new Context();context.plugin(LlmRuntime);await new Promise(r=>setTimeout(r,80));
 const {dir,file,bridge}=makeLedger(1);let calls=0;
 try {
  const other=createCliLedger({python:PYTHON,cwd:ROOT,ledgerPath:file,taskId:'other-task'});
  class Fake extends LlmAdapter {async *stream(){calls++;yield {type:'finish',reason:{kind:'stop'}};}}
  context.llm.registerAdapter(['fixture'],new Fake());installS5Interceptor(context,other);
  await assert.rejects(Array.fromAsync(context.llm.stream({provider:'fixture',model:'test',messages:[],maxTokens:64})),/shared campaign stop unproven/);
  const snapshot=await bridge.snapshot();
  assert.equal(calls,0);assert.equal(snapshot.reserved_calls,0);assert.equal(snapshot.aborted,false);
 }finally{rmSync(dir,{recursive:true,force:true});}
});

test('replaced ledger is never dispatched to or aborted as the original',async()=>{
 const {dir,file,bridge}=makeLedger(1);
 try {
  await bridge.snapshot();
  renameSync(file,file+'.old');
  command('create',['--ledger',file,'--task-id','test-task','--max-calls','1','--max-output-tokens','40960','--total-output-tokens','40960','--wall-clock-s','60']);
  let next=0;
  const hook=(await import('../interceptor.mjs')).createS5Interceptor(bridge);
  await assert.rejects(Array.fromAsync(hook({provider:'fixture',model:'test',messages:[],maxTokens:64},()=>{next++;return (async function*(){})();})),/shared campaign stop unproven/);
  assert.equal(next,0);
  const replacement=command('snapshot',['--ledger',file]);
  const original=command('snapshot',['--ledger',file+'.old']);
  assert.equal(replacement.reserved_calls,0);assert.equal(replacement.aborted,false);
  assert.equal(original.reserved_calls,0);assert.equal(original.aborted,false);
 } finally {rmSync(dir,{recursive:true,force:true});}
});

test('real ledger preserves abort after reserve without falsely reporting a dispatch',{skip:!available},async()=>{
 const {Context}=await import(pathToFileURL(path.join(APP,'cordis/lib/index.js')));
 const {LlmRuntime,LlmAdapter}=await import(pathToFileURL(path.join(APP,'dsh-llm/lib/index.js')));
 const context=new Context();context.plugin(LlmRuntime);await new Promise(r=>setTimeout(r,80));
 const {dir,bridge}=makeLedger(1), ac=new AbortController();let calls=0;
 try {
  class Fake extends LlmAdapter {async *stream(){calls++;yield {type:'finish',reason:{kind:'stop'}};}}
  context.llm.registerAdapter(['fixture'],new Fake());
  installS5Interceptor(context,{...bridge,reserve:async options=>{const value=await bridge.reserve(options);ac.abort();return value;}});
  await assert.rejects(Array.fromAsync(context.llm.stream({provider:'fixture',model:'test',messages:[],maxTokens:64,signal:ac.signal})));
  const snapshot=await bridge.snapshot();assert.equal(calls,0);assert.equal(snapshot.reserved_calls,1);
  assert.equal(snapshot.calls[0].dispatched_monotonic,null);assert.equal(snapshot.calls[0].completed_monotonic,null);
 }finally{rmSync(dir,{recursive:true,force:true});}
});
