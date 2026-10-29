import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createS5Interceptor, installS5Interceptor } from '../interceptor.mjs';

function ledger(cap=100) {
  let count=0;
  const calls=[];
  return { calls, aborted:false, async abort(){this.aborted=true;}, async reserve(req){ if(count>=cap) throw new Error('budget exhausted'); count++; const r={...req, call_id:String(count).padStart(32,'0'), dispatched:false,finish:null}; calls.push(r); return {call_id:r.call_id,sequence:count}; },
    async dispatch({call_id}){calls.find(r=>r.call_id===call_id).dispatched=true;},
    async finish({call_id,...result}){calls.find(r=>r.call_id===call_id).finish=result;}};
}
const request=(extra={})=>({provider:'route',model:'model-id',maxTokens:1024,messages:[{role:'user',content:'secret request body'}],...extra});
async function consume(stream){ const out=[]; for await(const c of stream) out.push(c); return out; }
const chunks=(...items)=>async function*(){for(const item of items)yield item;};

test('reserves only on consumption, records before next and preserves stream',async()=>{
 const l=ledger(), hook=createS5Interceptor(l); let n=0;
 const stream=hook(request(),()=>{n++;assert.equal(l.calls[0].dispatched,true);return chunks({type:'text-delta',text:'example'},{type:'usage',usage:{inputTokens:0,outputTokens:0}},{type:'finish',reason:{kind:'stop'}})();});
 assert.equal(l.calls.length,0);const output=await consume(stream); assert.equal(n,1);assert.equal(output.length,3);
 assert.equal(l.calls[0].finish.outcome,'answered');assert.equal(l.calls[0].finish.usage_evidence,'unverified');
 assert.match(l.calls[0].request_sha256,/^[0-9a-f]{64}$/);assert.ok(!JSON.stringify(l.calls).includes('secret request body'));
});

test('missing usage remains missing and does not depend on session id',async()=>{
 const l=ledger(),hook=createS5Interceptor(l);
 await consume(hook(request({purpose:'session-title'}),()=>chunks({type:'finish',reason:{kind:'stop'}})()));
 assert.equal(l.calls[0].caller,'dsh.outer.session-title');assert.equal(l.calls[0].finish.usage_evidence,'missing');
});

test('failed reservation or dispatch prevents calling provider',async()=>{
 for(const method of ['reserve','dispatch']){
  const l=ledger();l[method]=async()=>{throw new Error('failed')}; const hook=createS5Interceptor(l);let next=0;
  await assert.rejects(consume(hook(request(),()=>{next++;return chunks()();})));assert.equal(next,0);
 }
});

test('retry is a separate reservation and capacity is not refunded',async()=>{
 const l=ledger(1), hook=createS5Interceptor(l);let next=0;
 const bad=hook(request(),()=>{next++;throw new Error('provider failed');});await assert.rejects(consume(bad));
 assert.equal(l.calls[0].finish.outcome,'failed');
 await assert.rejects(consume(hook(request(),()=>{next++;return chunks()();})),/budget exhausted/);assert.equal(next,1);
});

test('consumer cancellation closes underlying stream and retains charge',async()=>{
 const l=ledger(),hook=createS5Interceptor(l);let closed=false;
 async function* source(){try{yield {type:'text-delta',text:'partial'};yield {type:'finish',reason:{kind:'stop'}};}finally{closed=true;}}
 const it=hook(request(),()=>source());await it.next();await it.return();
 assert.equal(closed,true);assert.equal(l.calls[0].finish.outcome,'cancelled');assert.equal(l.calls[0].finish.usage_evidence,'missing');
});

test('signal cancellation before dispatch and invalid configuration fail closed',async()=>{
 const l=ledger(), hook=createS5Interceptor(l);let next=0;const ac=new AbortController();ac.abort();
 await assert.rejects(consume(hook(request({signal:ac.signal}),()=>{next++;return chunks()();})));assert.equal(next,0);assert.equal(l.calls.length,0);
 for(const maxTokens of [undefined,0,40961,1.2,true]){
  await assert.rejects(consume(createS5Interceptor(ledger())(request({maxTokens}),()=>{next++;return chunks()();})));assert.equal(next,0);
 }
});

test('an unaccountable request poisons subsequent campaign dispatches',async()=>{
 const l=ledger(),hook=createS5Interceptor(l);let next=0;
 await assert.rejects(consume(hook(request({maxTokens:undefined}),()=>{next++;return chunks()();})));
 await assert.rejects(consume(hook(request(),()=>{next++;return chunks()();})),/campaign must stop/);
 assert.equal(next,0);assert.equal(l.calls.length,0);
});

test('concurrent audit failure blocks another pending reservation from dispatch',async()=>{
 const l=ledger(),original=l.reserve;let unblock;const gate=new Promise(r=>{unblock=r;});let i=0,n=0;
 l.reserve=async req=>{i++;if(i===1){const receipt=await original(req);await gate;return receipt;}throw Error('ledger failed');};
 const hook=createS5Interceptor(l),next=()=>{n++;return chunks()();};
 const first=consume(hook(request(),next));await Promise.resolve();
 await assert.rejects(consume(hook(request(),next)),/ledger failed/);unblock();
 await assert.rejects(first,/campaign must stop/);assert.equal(n,0);assert.equal(l.calls[0].dispatched,false);
});

test('abort while reservation is pending retains reservation and never dispatches',async()=>{
 const l=ledger(), ac=new AbortController(), reserve=l.reserve;
 l.reserve=async req=>{const result=await reserve(req);ac.abort();return result;};
 const hook=createS5Interceptor(l);let n=0;
 await assert.rejects(consume(hook(request({signal:ac.signal}),()=>{n++;return chunks()();})));
 assert.equal(n,0);assert.equal(l.calls.length,1);assert.equal(l.calls[0].dispatched,false);
 assert.equal(l.calls[0].finish,null);
});

test('audit finish failure propagates and stops downstream success',async()=>{
 const l=ledger(); l.finish=async()=>{throw Error('journal unavailable')};const hook=createS5Interceptor(l);
 await assert.rejects(consume(hook(request(),()=>chunks({type:'finish',reason:{kind:'stop'}})())),/journal unavailable/);
 let next=0; l.finish=async()=>{};
 await assert.rejects(consume(hook(request(),()=>{next++;return chunks()();})),/campaign must stop/);
 assert.equal(next,0);
});

test('missing terminal chunk is unknown; aborted terminal is cancelled',async()=>{
 const l=ledger(),hook=createS5Interceptor(l);
 await consume(hook(request(),()=>chunks({type:'text-delta',text:'partial'})()));
 await consume(hook(request(),()=>chunks({type:'finish',reason:{kind:'aborted'}})()));
 assert.equal(l.calls[0].finish.outcome,'unknown');assert.equal(l.calls[1].finish.outcome,'cancelled');
});

test('global hook observes auxiliary and unscoped calls rather than filtering session id',()=>{
 const seen=[];const ctx={on:(...args)=>seen.push(args)};installS5Interceptor(ctx,ledger());
 assert.equal(seen.length,1);assert.equal(seen[0][0],'llm/stream');assert.deepEqual(seen[0][2],{global:true,prepend:true});
});
