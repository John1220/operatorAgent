"""P2 step2：真机重测冻结核的 L3 测量有效性（先测起点核，再花模型预算）。

seed7 在 16384×6144 拿到 L3 1.46× 但 measurement_validity=degraded（launch 地板 12.7%>10%
+ 当时 1 个 harness 共卡进程）。本探针在空闲卡重做同形状 L3，看 validity 是否成立；
若仍 degraded（launch 地板是核时间 intrinsic，小形状核快→地板占比高），按用户定**优先试
更大输入形状**并为新形状重新验收起点核（核源码哈希不变 cb851b67，契约哈希随形状变）。

复用 ac_driver.final_evaluation（prepare→完整 L2→独立终评），stop_after='L3' 取 L3 metrics。
不花模型预算（纯验证）。GPU env 钉住测量卡 UUID；只读探测他人进程，不 kill。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from operator_agent.core.run_contract import digest

SEED_JSON = Path('output/v08-p2-seed7-medium-20261116/seed.json')
GPU3_UUID = 'GPU-634bbc76-9ab6-220a-48a8-27247d7acbe2'


def _declaration(rows, cols):
    return dict(operator='layernorm',
                request=f'fp16 {rows}x{cols} layernorm', shapes=[[rows, cols]],
                precision='fp16', dsl='triton', hardware='nvidia_gpu', min_speedup=1.0)


def _l3_metrics(receipt):
    for verdict in (receipt or {}).get('verdicts') or []:
        if isinstance(verdict, dict) and verdict.get('level') == 'L3':
            return verdict.get('metrics') or {}
    return {}


def _set_gpu_env(gpu, gpu_uuid, gpu_name):
    os.environ['OPERATOR_AGENT_REQUIRED_GPU'] = str(gpu)
    os.environ['OPERATOR_AGENT_REQUIRED_GPU_UUID'] = gpu_uuid
    os.environ['OPERATOR_AGENT_REQUIRED_GPU_NAME'] = gpu_name


def _establish_reference(declaration, host, work_dir, gpu, gpu_uuid, gpu_name):
    """L3 前确立该契约的基线参考身份（新形状=新契约，work_dir 无现成参考回执）。

    复用 p2_seed._provision_remote 同款：不带 EXISTING_REMOTE_ONLY 跑 reference_preflight
    （start_run=True），触发参考标定。返回非 None ⇒ 参考身份未确立，L3 会 uninformative。
    """
    from operator_agent.app import hitl_tools as ht
    from operator_agent.decide.verifier import Verifier
    saved = os.environ.pop('OPERATOR_AGENT_EXISTING_REMOTE_ONLY', None)
    _set_gpu_env(gpu, gpu_uuid, gpu_name)
    try:
        backend = ht._backend('ssh', host=host, work_dir=work_dir, persistent=True, timeout=600)
        task = ht.build_task(declaration)
        preflight = Verifier(backend).reference_preflight(task, start_run=True)
        if preflight is not None:
            raise SystemExit(f'[p2-probe] 参考身份未确立: {preflight.message}')
        print('[p2-probe] reference identity established for contract', flush=True)
    finally:
        if saved is not None:
            os.environ['OPERATOR_AGENT_EXISTING_REMOTE_ONLY'] = saved


def probe_seed_l3(*, code, declaration, root, host, work_dir, gpu, gpu_uuid, gpu_name,
                  budget_wall=600.0, timeout=600):
    from .ac_driver import final_evaluation
    from operator_agent.app.hitl_tools import build_task
    from operator_agent.core.contract import freeze_from_task

    _set_gpu_env(gpu, gpu_uuid, gpu_name)
    _establish_reference(declaration, host, work_dir, gpu, gpu_uuid, gpu_name)
    expected_contract = freeze_from_task(build_task(declaration)).hash
    run = Path(root)
    run.mkdir(parents=True, exist_ok=True)
    host_budget = dict(wall_clock_s=budget_wall, max_verifications=2, max_model_calls=1,
                       max_output_tokens_per_call=8192, total_output_token_reservations=8192)
    result = final_evaluation(run, code, declaration=declaration, host_budget=host_budget,
                              backend='ssh', stop_after='L3', host=host, work_dir=work_dir,
                              timeout=timeout, persistent=True)
    contract = json.loads((run / 'contract.json').read_text()).get('hash')
    if contract != expected_contract:
        raise ValueError(f'重冻结契约漂移: {contract} != {expected_contract}')
    explore_m = _l3_metrics(result.get('explore'))
    final_m = _l3_metrics(result.get('final'))
    summary = dict(
        contract_hash=contract, delivered=result.get('delivered'),
        explore_l2=bool((result.get('explore', {}).get('levels') or {}).get('L2')),
        explore_l3=bool((result.get('explore', {}).get('levels') or {}).get('L3')),
        final_l3=bool((result.get('final', {}).get('levels') or {}).get('L3')),
        explore_validity=explore_m.get('measurement_validity'),
        final_validity=final_m.get('measurement_validity'),
        explore_speedup=explore_m.get('speedup_vs_baseline'),
        final_speedup=final_m.get('speedup_vs_baseline'),
        explore_launch_dominance=explore_m.get('launch_dominance'),
        final_launch_dominance=final_m.get('launch_dominance'),
        explore_validity_reasons=explore_m.get('measurement_validity_reasons'),
        final_validity_reasons=final_m.get('measurement_validity_reasons'),
        baseline_type=final_m.get('baseline_type'),
        reason=result.get('reason'))
    valid = summary['final_validity'] == 'valid' and summary['final_l3']
    summary['valid_window'] = valid
    (run / 'probe-l3.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
                                       encoding='utf-8')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(prog='p2_seed_probe', description=__doc__.splitlines()[0])
    parser.add_argument('--rows', type=int, default=16384)
    parser.add_argument('--cols', type=int, default=6144)
    parser.add_argument('--root', default='output/v08-p2-probe-l3-20261116')
    parser.add_argument('--host', default='h20')
    parser.add_argument('--work-dir', default='/root/operator-p2-seed-20261116')
    parser.add_argument('--gpu', default='3')
    parser.add_argument('--gpu-uuid', default=GPU3_UUID)
    parser.add_argument('--gpu-name', default='NVIDIA H20')
    parser.add_argument('--seed-json', default=str(SEED_JSON))
    args = parser.parse_args(argv)

    seed = json.loads(Path(args.seed_json).read_text())
    code = Path(seed['source']).read_text(encoding='utf-8')
    actual = hashlib.sha256(code.encode('utf-8')).hexdigest()
    if actual != seed['source_hash']:
        raise SystemExit(f'起点核哈希漂移: {actual} != {seed["source_hash"]}')
    declaration = _declaration(args.rows, args.cols)
    print(f'[p2-probe] seed hash={actual[:16]} shape={args.rows}x{args.cols} gpu={args.gpu}', flush=True)
    summary = probe_seed_l3(code=code, declaration=declaration, root=args.root,
                            host=args.host, work_dir=args.work_dir, gpu=args.gpu,
                            gpu_uuid=args.gpu_uuid, gpu_name=args.gpu_name)
    print(f'[p2-probe] explore: L3={summary["explore_l3"]} validity={summary["explore_validity"]} '
          f'speedup={summary["explore_speedup"]} launch_dom={summary["explore_launch_dominance"]}', flush=True)
    print(f'[p2-probe] final:   L3={summary["final_l3"]} validity={summary["final_validity"]} '
          f'speedup={summary["final_speedup"]} launch_dom={summary["final_launch_dominance"]}', flush=True)
    print(f'[p2-probe] valid_window={summary["valid_window"]} contract={summary["contract_hash"][:16]}', flush=True)
    if summary['final_validity_reasons']:
        for r in summary['final_validity_reasons']:
            print(f'    reason: {r}', flush=True)
    return 0 if summary['valid_window'] else 1


if __name__ == '__main__':
    raise SystemExit(main())