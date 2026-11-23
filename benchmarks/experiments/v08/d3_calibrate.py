"""方向3 真机标定：先证明缓存与异步这两件量具在真 SSH 上读得对、跑得通，再谈信读数。

两件量具（v0.8.1 `9fb3ce7`）各自的标定面：

A. ``identity_cache`` —— 判定缓存的键里有一格是**环境身份**。在真 SSH 后端上：
   1. 环境身份必须回读到真实 H20 **型号级**字段（name/device/cc/torch/triton），不是空字典；
      驱动/CUDA/UUID 在 SSH 上不产出——型号级身份是已接受的边界（2026-11-15），记为观测项不作门；
   2. 同源码同契约重复 verify（explore, L1）⇒ 第二次 ``cache_hit=True`` 且 ``cached_from`` 指回第一轮；
   3. 换一份源码（哈希不同）⇒ 不命中；
   4. 预算按**派发**扣，命中也扣（登记单位不是 GPU 秒）。
   用 L1（只编译，秒级）做命中/未命中机制标定，环境身份与级别无关。

B. ``async_chain`` —— 异步验证全链路。spawn 真实 ``operator-agent-mcp`` 入口，走 JSON-RPC：
   ``operator_verify(wait=false)`` 必须**立即**返回 invocation_id（不阻塞分钟级 L2），
   ``operator_status`` 轮询到终态判定，``operator_cancel`` 能取消在飞验证。
   用分钟级 attention L2 才证明得了「不阻塞」——秒级核会在首次轮询前就结束。

候选复用 P1 已绑定哈希的 attention 核（``abb98d4d…``），不新生成、不调模型。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ATTENTION_DECL = dict(operator='attention', request='fp16 attention prefill 1x16x4096x128',
                      shapes=[[1, 16, 4096, 128]], hardware='nvidia_gpu',
                      dsl='triton', precision='fp16', min_speedup=1.0)
ATTENTION_SOURCE_HASH = 'abb98d4d8159584348513e823c877ea569ae9f0418ec64ec32d8d9dd2d9275ba'


def _gpu_env(gpu, gpu_uuid, gpu_name):
    env = dict(os.environ)
    env['OPERATOR_AGENT_REQUIRED_GPU'] = str(gpu)
    if gpu_uuid:
        env['OPERATOR_AGENT_REQUIRED_GPU_UUID'] = gpu_uuid
    if gpu_name:
        env['OPERATOR_AGENT_REQUIRED_GPU_NAME'] = gpu_name
    return env


def _attention_code(ac_root):
    rel = 'output/v08-p1-20261114/run3/p1-attention-r2/C/verify-2/candidate.py'
    path = Path(ac_root) / rel if ac_root else Path(rel)
    code = path.read_text(encoding='utf-8')
    import hashlib
    digest = hashlib.sha256(code.encode('utf-8')).hexdigest()
    if digest != ATTENTION_SOURCE_HASH:
        raise SystemExit(f'attention 源码哈希不符: {digest} != {ATTENTION_SOURCE_HASH}')
    return code


def _budget():
    return dict(wall_clock_s=1800, max_verifications=8, max_model_calls=1,
                max_output_tokens_per_call=8192, total_output_token_reservations=8192)


def identity_cache(*, host, work_dir, root, ac_root, gpu_env):
    from operator_agent.app import hitl_tools as ht

    os.environ.update({k: v for k, v in gpu_env.items()
                       if k.startswith('OPERATOR_AGENT_REQUIRED_GPU')})
    code = _attention_code(ac_root)
    run = Path(root) / 'identity-cache'
    run.mkdir(parents=True, exist_ok=True)
    ht.prepare(ATTENTION_DECL, run, host_budget=_budget())
    opts = dict(backend='ssh', host=host, work_dir=work_dir, timeout=600, stop_after='L1')

    t0 = time.time()
    r1 = ht.verify(run, code, **opts)
    t1 = time.time()
    r2 = ht.verify(run, code, **opts)                      # 同源码 ⇒ 应命中
    t2 = time.time()
    r3 = ht.verify(run, code + '\n# d3 calibration: distinct source hash\n', **opts)  # 换哈希 ⇒ 不命中
    t3 = time.time()

    env = r1.get('environment') or {}
    checks = {
        'environment_nonempty': bool(env),
        'environment_has_device': bool(env.get('device') or env.get('name')),
        'environment_has_torch': bool(env.get('torch_version')),
        'environment_model_level_complete': all(
            bool(env.get(k)) for k in ('name', 'device', 'compute_capability',
                                       'torch_version', 'triton_version')),
        'first_not_cached': not r1.get('cache_hit'),
        'second_cache_hit': r2.get('cache_hit') is True,
        'cached_from_round1': (r2.get('cached_from') or {}).get('round') == r1.get('round'),
        'third_not_cached': not r3.get('cache_hit'),
        'budget_charged_thrice': (r1.get('host_task', {}).get('reserved') == 1
                                  and r2.get('host_task', {}).get('reserved') == 2
                                  and r3.get('host_task', {}).get('reserved') == 3),
    }
    receipt = dict(part='identity_cache', checks=checks,
                   all_pass=all(checks.values()),
                   environment=env,
                   identity_scope='gpu_model_level_accepted',
                   gpu_uuid_bound=bool(env.get('gpu_uuid')),
                   accepted_limitation='SSH 不产出 gpu_uuid/driver/cuda；型号级身份为已接受边界，非缺陷',
                   rounds=[dict(round=r.get('round'), cache_hit=r.get('cache_hit'),
                                cached_from=r.get('cached_from'),
                                levels=r.get('levels'),
                                reserved=(r.get('host_task') or {}).get('reserved'))
                           for r in (r1, r2, r3)],
                   seconds=dict(first=round(t1 - t0, 2), second_cached=round(t2 - t1, 2),
                                third=round(t3 - t2, 2)),
                   evidence=[r.get('evidence') for r in (r1, r2, r3)])
    _write(root, 'identity-cache.json', receipt)
    return receipt


class _Mcp:
    """最小 JSON-RPC 行客户端：顺序请求/应答。stdout 只走协议行（库 print 已改道 stderr）。"""

    def __init__(self, entry, env, log_path):
        self._id = 0
        self._log = open(log_path, 'w')
        self.proc = subprocess.Popen(entry, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self._log, text=True, env=env, bufsize=1)

    def _send(self, obj):
        self.proc.stdin.write(json.dumps(obj, ensure_ascii=False) + '\n')
        self.proc.stdin.flush()

    def call(self, name, arguments, timeout=650):
        self._id += 1
        mid = self._id
        self._send(dict(jsonrpc='2.0', id=mid, method='tools/call',
                        params=dict(name=name, arguments=arguments)))
        return self._read(mid, timeout)

    def notify(self, method, params=None):
        self._send(dict(jsonrpc='2.0', method=method, params=params or {}))

    def initialize(self):
        self._id += 1
        mid = self._id
        self._send(dict(jsonrpc='2.0', id=mid, method='initialize',
                        params=dict(protocolVersion='2025-11-25', capabilities={},
                                    clientInfo=dict(name='d3-calibrate', version='0'))))
        reply = self._read(mid, 60)
        self.notify('notifications/initialized')
        return reply

    def _read(self, mid, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                raise SystemExit('MCP stdout 关闭（进程退出？见 stderr 日志）')
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            if msg.get('id') == mid:
                if 'error' in msg:
                    raise SystemExit(f'MCP error: {msg["error"]}')
                return _unwrap(msg)
        raise SystemExit(f'MCP 应答超时 id={mid}')

    def close(self):
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
        self._log.close()


def _unwrap(msg):
    content = (msg.get('result') or {}).get('content') or []
    if content and content[0].get('type') == 'text':
        return json.loads(content[0]['text'])
    return msg.get('result')


def async_chain(*, host, work_dir, root, ac_root, gpu_env, entry):
    code = _attention_code(ac_root)
    run = Path(root) / 'async-chain'
    run.mkdir(parents=True, exist_ok=True)
    client = _Mcp(entry, gpu_env, str(Path(root) / 'async-mcp.stderr.log'))
    facts = dict(part='async_chain')
    try:
        client.initialize()
        prep = client.call('operator_prepare', dict(
            operator=ATTENTION_DECL['operator'],
            request=ATTENTION_DECL['request'], shapes=ATTENTION_DECL['shapes'],
            run_dir=str(run), hardware='nvidia_gpu', dsl='triton', precision='fp16'))
        facts['contract_hash'] = prep.get('contract_hash')

        t0 = time.time()
        submit = client.call('operator_verify', dict(
            run_dir=str(run), code=code, backend='ssh', host=host, work_dir=work_dir,
            stop_after='L2', timeout=600, wait=False))
        submit_latency = time.time() - t0
        inv = submit.get('invocation_id')
        facts['submit'] = dict(latency_s=round(submit_latency, 3), phase=submit.get('phase'),
                               submitted=submit.get('submitted'), invocation_id=inv)

        terminal = None
        polls = 0
        t_poll = time.time()
        while time.time() - t_poll < 650:
            polls += 1
            status = client.call('operator_status', dict(run_dir=str(run), invocation_id=inv))
            phase = status.get('phase')
            if phase in ('done', 'failed', 'cancelled', 'lost'):
                terminal = status
                break
            time.sleep(5)
        facts['poll'] = dict(count=polls, elapsed_s=round(time.time() - t_poll, 1),
                             terminal_state=(terminal or {}).get('phase'))
        result = (terminal or {}).get('result') or {}
        facts['terminal_levels'] = result.get('levels')
        facts['terminal_ok'] = result.get('ok')

        # 取消链路：再提交一个在飞验证，立刻取消，确认终态。
        submit2 = client.call('operator_verify', dict(
            run_dir=str(run), code=code, backend='ssh', host=host, work_dir=work_dir,
            stop_after='L2', timeout=600, wait=False))
        inv2 = submit2.get('invocation_id')
        cancel = client.call('operator_cancel', dict(run_dir=str(run), invocation_id=inv2))
        cancel_state = None
        for _ in range(60):
            st = client.call('operator_status', dict(run_dir=str(run), invocation_id=inv2))
            if st.get('phase') in ('done', 'failed', 'cancelled', 'lost'):
                cancel_state = st.get('phase')
                break
            time.sleep(2)
        facts['cancel'] = dict(accepted=cancel.get('accepted') or cancel.get('ok'),
                               invocation_id=inv2, terminal_state=cancel_state)
    finally:
        client.close()

    checks = {
        'submit_immediate': facts['submit']['latency_s'] < 5.0,
        'submit_returned_invocation': bool(facts['submit']['invocation_id']),
        'submit_phase_running': facts['submit']['phase'] == 'running',
        'reached_terminal': facts['poll']['terminal_state'] in ('done', 'failed'),
        'terminal_l2_present': bool((facts.get('terminal_levels') or {}).get('L2') is not None),
        'cancel_reached_terminal': facts['cancel']['terminal_state'] in ('cancelled', 'done', 'failed'),
    }
    facts['checks'] = checks
    facts['all_pass'] = all(checks.values())
    _write(root, 'async-chain.json', facts)
    return facts


def _write(root, name, payload):
    out = Path(root) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'[d3] {name}: all_pass={payload.get("all_pass")}', flush=True)
    for key, value in (payload.get('checks') or {}).items():
        print(f'    {"PASS" if value else "FAIL"}  {key}', flush=True)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(prog='d3_calibrate', description=__doc__.splitlines()[0])
    parser.add_argument('--part', choices=['identity', 'async', 'all'], default='all')
    parser.add_argument('--host', default='h20')
    parser.add_argument('--work-dir', default='/root/operator-d3-20261114')
    parser.add_argument('--root', default='output/v08-d3-20261114')
    parser.add_argument('--ac-root', default=None)
    parser.add_argument('--gpu', default='3')
    parser.add_argument('--gpu-uuid', default='GPU-634bbc76-9ab6-220a-48a8-27247d7acbe2')
    parser.add_argument('--gpu-name', default='NVIDIA H20')
    parser.add_argument('--mcp-entry', default=None)
    args = parser.parse_args(argv)

    gpu_env = _gpu_env(args.gpu, args.gpu_uuid, args.gpu_name)
    Path(args.root).mkdir(parents=True, exist_ok=True)
    entry = args.mcp_entry or [sys.executable, '-m', 'operator_agent.app.mcp_server']
    if isinstance(entry, str):
        entry = [entry]

    if args.part in ('identity', 'all'):
        identity_cache(host=args.host, work_dir=args.work_dir, root=args.root,
                       ac_root=args.ac_root, gpu_env=gpu_env)
    if args.part in ('async', 'all'):
        async_chain(host=args.host, work_dir=args.work_dir, root=args.root,
                    ac_root=args.ac_root, gpu_env=gpu_env, entry=entry)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())