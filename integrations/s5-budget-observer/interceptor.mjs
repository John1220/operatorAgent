/** S5 experimental DSH outer-call observer. This is not installed by default.
 *
 * A trusted, isolated experiment must install it globally before any call and
 * supply a durable ledger. `next()` is never invoked until reserve and dispatch
 * have both committed. Normalized DSH usage is NOT provider usage evidence:
 * the pi-ai adapter may synthesize zeros when the response omitted usage.
 */
import { createHash } from 'node:crypto';

const FINISH_OUTCOMES = Object.freeze({stop:'answered','tool-calls':'answered',
  'max-tokens':'answered', aborted:'cancelled', error:'failed'});
const PURPOSES = new Set(['compaction', 'session-title']);

function requestIdentity(options) {
  if (!options || typeof options !== 'object' || typeof options.provider !== 'string' ||
      !options.provider || typeof options.model !== 'string' || !options.model ||
      !Array.isArray(options.messages)) throw new TypeError('invalid model request identity');
  if (!Number.isSafeInteger(options.maxTokens) || options.maxTokens < 1 || options.maxTokens > 40960)
    throw new RangeError('request maxTokens must be explicitly bounded within 1..40960');
  const purpose=options.purpose;
  if (purpose !== undefined && !PURPOSES.has(purpose))
    throw new TypeError('unknown auxiliary request purpose');
  // Explicit selection excludes AbortSignal and credentials. The resulting
  // message only exists in memory; the ledger receives its SHA256, not content.
  let serialized;
  try { serialized=JSON.stringify({provider:options.provider,model:options.model,
    messages:options.messages,system:options.system,tools:options.tools,
    temperature:options.temperature,maxTokens:options.maxTokens,stop:options.stop,
    reasoningEffort:options.reasoningEffort,purpose,sessionId:options.sessionId}); }
  catch { throw new TypeError('request identity is not serializable'); }
  if (Buffer.byteLength(serialized,'utf8') > 1048576)
    throw new RangeError('request identity exceeds 1 MiB');
  return {participant:'outer',caller:'dsh.outer.'+(purpose ?? 'conversation'),
    max_output_tokens:options.maxTokens,
    request_sha256:createHash('sha256').update(serialized,'utf8').digest('hex')};
}

function validateLedger(ledger) {
  if (!ledger || ['reserve','dispatch','finish','abort'].some(k=>typeof ledger[k]!=='function'))
    throw new TypeError('durable ledger reserve/dispatch/finish/abort required');
}

export function createS5Interceptor(ledger) {
  validateLedger(ledger);
  let auditFault=false;
  async function poison(error) {
    auditFault=true;
    try { await ledger.abort({reason:'audit_failure'}); }
    catch { throw new Error('audit unavailable; shared campaign stop unproven'); }
    throw error;
  }
  return function intercept(options, next) {
    // The DSH waterfall returns an iterable synchronously; all asynchronous
    // budget work happens on its first consumption, before the adapter runs.
    return (async function* () {
      if (auditFault) throw new Error('audit unavailable; campaign must stop');
      if (options?.signal?.aborted) throw new Error('request cancelled before reservation');
      let identity;
      try { identity=requestIdentity(options); }
      catch (error) { await poison(error); }
      let reservation;
      try { reservation=await ledger.reserve(identity); }
      catch (error) { await poison(error); }
      const id=reservation?.call_id;
      if (typeof id!=='string' || !/^[0-9a-f]{32}$/.test(id)) {
        await poison(new Error('ledger reservation identity unavailable'));
      }
      let outcome='cancelled';
      let usageEvidence='missing';
      let terminal=false;
      let dispatched=false;
      try {
        if (auditFault) throw new Error('audit unavailable; campaign must stop');
        if (options?.signal?.aborted) throw new Error('request cancelled before dispatch');
        try { await ledger.dispatch({call_id:id}); }
        catch (error) { await poison(error); }
        dispatched=true;
        if (auditFault) throw new Error('audit unavailable; campaign must stop');
        if (options?.signal?.aborted) throw new Error('request cancelled during dispatch');
        // No async provider operation may occur before the durable dispatch.
        const stream=next();
        if (!stream || typeof stream[Symbol.asyncIterator] !== 'function')
          throw new TypeError('adapter stream is not async iterable');
        for await (const chunk of stream) {
          if (chunk?.type==='usage') usageEvidence='unverified';
          if (chunk?.type==='finish') {
            terminal=true;
            const kind=typeof chunk.reason==='object' ? chunk.reason?.kind : undefined;
            outcome=FINISH_OUTCOMES[kind] ?? 'unknown';
          }
          yield chunk;
        }
        if (!terminal) outcome='unknown';
      } catch (error) {
        outcome=options?.signal?.aborted ? 'cancelled' : 'failed';
        throw error;
      } finally {
        // This call may fail after the consumer received chunks; propagating
        // failure prevents presenting an unaudited stream as complete.
        // The shared ledger only accepts completion after a proven dispatch.
        // Pre-dispatch cancellation leaves an unrefunded, unfinished reservation;
        // a failed dispatch response is ambiguous and poisons this interceptor.
        if (dispatched) {
          try { await ledger.finish({call_id:id,outcome,usage_evidence:usageEvidence}); }
          catch (error) { await poison(error); }
        }
      }
    })();
  };
}

export function installS5Interceptor(ctx, ledger) {
  if (!ctx || typeof ctx.on!=='function') throw new TypeError('DSH context required');
  const hook=createS5Interceptor(ledger);
  return ctx.on('llm/stream',hook,{global:true,prepend:true});
}
