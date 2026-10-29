/** Bridge the experimental JS waterfall to one local shared Python ledger.
 *
 * subprocess arguments carry only bounded metadata, request SHA256 and IDs;
 * prompts, API keys and model responses are never passed on argv. This is
 * trusted same-host coordination, not isolation against hostile same-UID code.
 */
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import path from 'node:path';
import { lstatSync } from 'node:fs';

const execFileAsync=promisify(execFile);
const ID=/^[0-9a-f]{32}$/;

export function createCliLedger({python, cwd, ledgerPath, taskId, timeoutMs=20000}) {
  if (![python,cwd,ledgerPath].every(x=>typeof x==='string' && path.isAbsolute(x)))
    throw new TypeError('absolute python, cwd and ledgerPath required');
  if (typeof taskId!=='string' || !/^[a-zA-Z0-9_.:-]{1,64}$/.test(taskId))
    throw new TypeError('invalid frozen task id');
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs<100 || timeoutMs>60000)
    throw new TypeError('invalid ledger timeout');
  // Deliberately do not inherit any API/provider credentials or Python path.
  const env={PATH:process.env.PATH || '/usr/bin:/bin',PYTHONNOUSERSITE:'1',PYTHONUNBUFFERED:'1'};
  let pinnedIdentity;
  function fileIdentity() {
    let st;
    try { st=lstatSync(ledgerPath,{bigint:true}); }
    catch { throw new Error('ledger identity unavailable'); }
    if (!st.isFile() || st.isSymbolicLink()) throw new Error('ledger identity unavailable');
    return `${st.dev}:${st.ino}`;
  }
  async function call(operation, args=[]) {
    const argv=['-m','operator_agent.s5_shared_budget',operation,'--ledger',ledgerPath,
      '--expected-task-id',taskId,...args];
    let stdout;
    try { ({stdout}=await execFileAsync(python,argv,{cwd,env,timeout:timeoutMs,maxBuffer:1048576})); }
    catch { throw new Error('ledger '+operation+' unavailable or rejected'); }
    let result;
    try { result=JSON.parse(stdout); }
    catch { throw new Error('ledger '+operation+' returned invalid JSON'); }
    if (!result || result.ok!==true) throw new Error('ledger '+operation+' rejected');
    return result;
  }
  async function checkedSnapshot() {
      const before=fileIdentity();
      if (pinnedIdentity!==undefined && pinnedIdentity!==before)
        throw new Error('ledger file identity changed');
      const result=await call('snapshot');
      if (fileIdentity()!==before) throw new Error('ledger file identity changed');
      if (result.task_id!==taskId) throw new Error('ledger task identity mismatch');
      // Never pin, modify or abort a ledger belonging to another task.
      if (pinnedIdentity!==undefined && pinnedIdentity!==before)
        throw new Error('ledger file identity changed');
      pinnedIdentity=before;
      return result;
  }
  async function checkedCall(operation,args=[]) {
    await checkedSnapshot();
    const result=await call(operation,args);
    if (fileIdentity()!==pinnedIdentity) throw new Error('ledger file identity changed');
    return result;
  }
  return {
    async snapshot() { return checkedSnapshot(); },
    async reserve({participant,caller,max_output_tokens,request_sha256}) {
      if (participant!=='outer' || typeof caller!=='string' || !Number.isSafeInteger(max_output_tokens) ||
          max_output_tokens<1 || max_output_tokens>40960 || !/^[0-9a-f]{64}$/.test(request_sha256))
        throw new TypeError('invalid outer reservation');
      const result=await checkedCall('reserve',['--participant','outer','--caller',caller,
        '--max-output-tokens',String(max_output_tokens),'--request-sha256',request_sha256]);
      if (!ID.test(result.call_id) || !Number.isSafeInteger(result.sequence) || result.sequence<1)
        throw new Error('ledger returned invalid reservation');
      return result;
    },
    async dispatch({call_id}) {
      if (typeof call_id!=='string' || !ID.test(call_id)) throw new TypeError('invalid reservation identity');
      return checkedCall('dispatch',['--call-id',call_id]);
    },
    async finish({call_id,outcome,usage_evidence}) {
      if (typeof call_id!=='string' || !ID.test(call_id) || !['answered','failed','cancelled','unknown'].includes(outcome) ||
          !['unverified','missing'].includes(usage_evidence)) throw new TypeError('invalid completion');
      // No provider payload is exposed by DSH: normalized usage is never
      // promoted to provider evidence, nor persisted as factual token counts.
      return checkedCall('finish',['--call-id',call_id,'--outcome',outcome,'--usage-evidence',usage_evidence]);
    },
    async abort({reason}) {
      if (!['audit_failure','budget_controller_failure','participant_crash','manual_stop'].includes(reason))
        throw new TypeError('invalid shared stop reason');
      return checkedCall('abort',['--reason',reason]);
    },
  };
}
