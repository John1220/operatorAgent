/** One offline-only headless fixture tool. This is not an isolation boundary. */
import { pathToFileURL } from 'node:url';

export const name = 's5-fixture-validation';
export const inject = ['tools'];

export async function apply(ctx, config) {
  const url = new URL(config.validationUrl);
  if (url.protocol !== 'http:' || url.hostname !== '127.0.0.1' || !url.port ||
      url.pathname !== '/validate' || url.username || url.password || url.search || url.hash)
    throw new Error('local fixture validator required');
  const modulePath = process.env.S5_DSH_TOOLS_MODULE;
  if (!modulePath || !modulePath.startsWith('/')) throw new Error('tools module unavailable');
  const { defineTool } = await import(pathToFileURL(modulePath));
  ctx.tools.register(defineTool({
    name:'validate_kernel',
    description:'Validate the proposed Python kernel on the fixed mock task.',
    parameters:{source:{type:'string',required:true,description:'Exact Python candidate source.'}},
    output:{schema:{type:'string'},render:(_args,value)=>[{type:'text',text:value}]},
    timeoutMs:15000,
    isConcurrencySafe:()=>false,
    async execute(args, exec) {
      if (Buffer.byteLength(args.source,'utf8') > 65536) throw new Error('source too large');
      const signal = exec?.signal ? AbortSignal.any([exec.signal,AbortSignal.timeout(12000)]) : AbortSignal.timeout(12000);
      const response = await fetch(url, {method:'POST',redirect:'error',signal,
        headers:{'Content-Type':'application/json'},body:JSON.stringify({source:args.source})});
      if (!response.ok) throw new Error('fixture validator refused');
      const text = await response.text();
      if (Buffer.byteLength(text,'utf8') > 65536) throw new Error('fixture validator response too large');
      JSON.parse(text);
      return text;
    }
  }));
}
