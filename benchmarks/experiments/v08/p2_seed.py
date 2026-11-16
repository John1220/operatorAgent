"""P2 起点核（B 方案）：用内层 27B/xhigh 一次生成 layernorm fp16 核，过完整 L2 后冻住单一哈希。

P2 测的是 optimize 诊断闭环的因果增量，两臂共用同一份正确起点核。起点核来源已定 B
（2026-11-15 用户定）：一份由 27B 一次生成、过完整 L2 后冻住的核，不每对重新生成。

**内层采样身份**：草案 §13 已定「27B/xhigh（与 v6 一致）」。v6 的 xhigh 是外层 DSH 宿主经
phase1 ``wire_sampling`` 注入的 ``reasoning_effort``；内层 Conductor 自己的 ``LLMClient``
默认是思考关/4096/无 reasoning_effort（autonomous.py 的 ``LLMClient(max_tokens=4096, ...)``）。
要让内层真做到 xhigh，本驱动**复用 phase1 已标定的 study 机制**，不改核心默认值：
``connected_worker.run`` 在 ``environment(config) + bounded(config)`` 上下文里进程内跑
``autonomous._worker_main``，``bounded`` 把冻结 cell config 的 ``sampling`` 段
（reasoning_effort=xhigh / enable_thinking=True / max_tokens）normalize 进内层 bounded
client 的请求体，并落 ``sampling-inner.jsonl`` 供审计。这样起点核与后续 optimize 闭环
用的是同一份可审计的冻结采样身份。

凭据：本机 vLLM 27B 无鉴权（不带 key 也 200，已实证）。client 强制要非空 key，本驱动用
**非密占位** ``LLM_API_KEY=EMPTY``（用户 2026-11-16 定），不落盘任何真实密钥。

量具纪律：先 ``--probe-sampling`` 做一次便宜正控（一次小调用，确认 xhigh 真发出去），
再跑整核生成；生成后独立完整 L2 复核，过了才 write-once 冻住。草案阶段不写正式轮读数。

**2026-11-16 截断成分探查 + 用户定案**：xhigh 真实 Coder（11 配方）三次撞顶 40960 未冻住。
成分探查（``p2_truncation_probe.py``）发现代码本身极小（~0.8–1.5k token），**思考占 91–95% 是撞顶主因**：
xhigh 思考随配方复杂度涨——简化 prompt 31295 能 fit、真实 11 配方 Coder ≥40960 截断；medium 思考只有
xhigh ~1/4（7963 vs 29743，简化 prompt 8750 总 token、完整核、finish=stop）。故「xhigh 必然 >40960」
不准确，是配方复杂度×xhigh 共同推高思考。**用户定：起点核生成降 medium**（保 40960 上限、零核心改动，
``--reasoning-effort medium``）；不提内层硬上限（避免改核心预算）。optimize 闭环采样身份另定
（Search 诊断调用产小 JSON 非整核，可仍 xhigh）。v6 提示做法（源码只进最终输出）保留。
回执见 ``output/v08-p2-seed{2,3,4,5}-20261116/``（xhigh 失败）与 ``output/v08-p2-trunc-{xhigh,medium}-20261116/``（成分）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from operator_agent.core.run_contract import digest
from operator_agent.execute.task_clock import monotonic

#: 本地纯函数标定过的 layernorm v2 声明（build_task 产出 kernel_type=layernorm、
#: correctness_profile=builtin-layernorm-fp16-cuda-v2、shape 16384×6144）。
DECL = dict(request='fp16 16384x6144 layernorm', shapes=[[16384, 6144]],
            precision='fp16', dsl='triton', hardware='nvidia_gpu', min_speedup=1.0)
CONTRACT_HASH = '642b52781c88fb890fb7f96110fda1fa42b71574438a3c7f6c8175721499cee7'

GPU3_UUID = 'GPU-634bbc76-9ab6-220a-48a8-27247d7acbe2'


def _write_once(path, value):
    from benchmarks.experiments.phase1.plan import write_once
    write_once(path, value)


def _task_budget(budget):
    """TaskBudgetConfig 只认这三格；共享账本/终评预留用的是另外几格，分开投影。"""
    return {k: budget[k] for k in ('wall_clock_s', 'max_llm_calls', 'oracle_total')}


def _wire_policy(reasoning_effort, max_tokens):
    """把 phase1 线上采样策略设成 P2 冻结身份，再取副本进 cell config。

    ``bounded`` 会 ``validate(sampling == POLICY)``，所以必须先改 POLICY 再取 policy()
    （与 ac_live._apply_wire_policy 同款）。max_tokens 不得超过账本每次预留。
    """
    from benchmarks.experiments.phase1 import sampling as wire_sampling
    wire_sampling.POLICY['reasoning_effort'] = reasoning_effort
    wire_sampling.POLICY['max_tokens'] = int(max_tokens)
    return dict(wire_sampling.policy())


def _cell_config(*, root, host, work_dir, gpu, gpu_uuid, gpu_name, base_url, model_id,
                 budget, max_rounds, sampling):
    root = Path(root).resolve()
    run_dir = root / 'run'
    ledger = (root / 'budget.sqlite').resolve()
    task_id = 'p2-seed-' + CONTRACT_HASH[:16]
    started = monotonic()
    config = dict(
        schema='phase1-connected-cell-v1',
        cell_id='p2-seed-layernorm',
        plan_hash=digest(dict(decl=DECL, contract_hash=CONTRACT_HASH,
                              sampling=sampling, gpu=dict(index=gpu, uuid=gpu_uuid, name=gpu_name))),
        arm='B', entry_b='core',
        declaration=DECL,
        model=dict(base_url=base_url, id=model_id, api_key_env='LLM_API_KEY', temperature=0.1),
        gpu=dict(index=int(gpu), uuid=gpu_uuid, name=gpu_name),
        connection=dict(host=host, work_dir=work_dir),
        backend='ssh',
        budget=_task_budget(budget),
        shared_ledger=str(ledger),
        shared_task_id=task_id,
        sampling=dict(sampling),
        run_dir=str(run_dir),
        max_rounds=int(max_rounds),
        started_monotonic=started,
        deadline_monotonic=started + float(budget['wall_clock_s']),
        execution_context={},
    )
    return config, ledger, task_id


def _create_ledger(ledger, task_id, budget):
    from operator_agent.execute import s5_shared_budget as shared
    if Path(ledger).exists():
        raise SystemExit(f'共享账本已存在，拒绝复用: {ledger}（用新的 --root）')
    shared.create(ledger, task_id=task_id,
                  max_calls=int(budget['max_model_calls']),
                  max_output_tokens=int(budget['max_output_tokens_per_call']),
                  total_output_tokens=int(budget['total_output_token_reservations']),
                  wall_clock_s=float(budget['wall_clock_s']))


def _widen_call_timeout(per_1k, wall_clock_s):
    """放宽内层每调用超时。phase1 ``request_timeout`` 用 16s/1k，隐含要求 ≥62.5 tok/s；
    本机 vLLM 实测 ~53 tok/s，xhigh Coder 把整核写进思考、一路写到 32k 上限需 ~618s，
    16s/1k×32.768k=524s 必被掐（2026-11-16 实测 call-003 wall=524.3s 超时失败）。
    ``delegated`` 在调用时 ``from .sampling import request_timeout`` 取当前函数，所以
    patch ``sampling.request_timeout`` 即可把内层 llm_timeout 放宽（与 ac_live 放宽外层
    流式截止同理）。仍受墙钟预算约束。
    """
    from benchmarks.experiments.phase1 import sampling

    def _timeout(config):
        cap = sampling.output_cap(config)
        return min(max(120.0, per_1k * cap / 1000.0), float(wall_clock_s))

    sampling.request_timeout = _timeout


def _provision_remote(host, work_dir, gpu, gpu_uuid, gpu_name):
    """先把远端 work_dir 备好并标定参考身份，再进 delegated。

    ``connected_worker`` 的 ``environment`` 设 ``OPERATOR_AGENT_EXISTING_REMOTE_ONLY=1``：
    deploy 只**检查**现有环境（``.venv/bin/python`` + 脚本哈希），不 mkdir/安装/部署。
    首跑 work_dir 全新时那条检查会 ``HardwareProbeFailed`` ⇒ 参考身份 unknown ⇒ Conductor
    在 Designer 前早退（rounds=0）。所以这里**不带** EXISTING_REMOTE_ONLY 跑一次
    ``reference_preflight``：它触发完整 provisioning（uv 环境 + 脚本部署）并校验冻结参考
    身份。返回 unknown 就带原因停，不烧生成。
    """
    from operator_agent.app import hitl_tools as ht
    from operator_agent.decide.verifier import Verifier
    saved = os.environ.pop('OPERATOR_AGENT_EXISTING_REMOTE_ONLY', None)
    os.environ['OPERATOR_AGENT_REQUIRED_GPU'] = str(gpu)
    os.environ['OPERATOR_AGENT_REQUIRED_GPU_UUID'] = gpu_uuid
    os.environ['OPERATOR_AGENT_REQUIRED_GPU_NAME'] = gpu_name
    try:
        backend = ht._backend('ssh', host=host, work_dir=work_dir, persistent=True, timeout=600)
        task = ht.build_task(DECL)
        preflight = Verifier(backend).reference_preflight(task, start_run=True)
        if preflight is not None:
            raise SystemExit(f'[p2-seed] 远端参考身份未确立，不生成: {preflight.message}')
        print('[p2-seed] remote work_dir provisioned + reference identity matched', flush=True)
    finally:
        if saved is not None:
            os.environ['OPERATOR_AGENT_EXISTING_REMOTE_ONLY'] = saved


def probe_sampling(*, root, host, work_dir, gpu, gpu_uuid, gpu_name, base_url, model_id,
                   budget, sampling):
    """便宜正控：在 bounded 上下文里发一次小调用，确认 xhigh 真进了请求体。

    不跑 Conductor、不烧整核生成。读 sampling-inner.jsonl 的 effective 格核对
    reasoning_effort / enable_thinking / max_tokens 是否等于冻结身份。
    """
    from benchmarks.experiments.phase1.connection import environment
    from benchmarks.experiments.phase1.sampling import bounded
    from operator_agent.execute.llm_client import LLMClient
    from operator_agent.execute.s5_inner_budget import SharedProductionBudget
    from operator_agent.execute.task_budget import TaskBudgetConfig

    config, ledger, task_id = _cell_config(
        root=root, host=host, work_dir=work_dir, gpu=gpu, gpu_uuid=gpu_uuid,
        gpu_name=gpu_name, base_url=base_url, model_id=model_id, budget=budget,
        max_rounds=1, sampling=sampling)
    _create_ledger(ledger, task_id, budget)
    _write_once(Path(config['run_dir']).parent / 'cell-config.json', config)
    policy = TaskBudgetConfig.from_dict(_task_budget(budget))
    from benchmarks.experiments.phase1.sampling import request_timeout
    timeout = request_timeout(config)
    journal = Path(config['run_dir']) / 'model-journal'
    with environment(config), bounded(config):
        budget_obj = SharedProductionBudget(policy.llm_limits(timeout), journal,
                                            shared_path=ledger, task_id=task_id,
                                            deadline=config['deadline_monotonic'])
        llm = LLMClient(base_url=base_url, model=model_id, api_key=os.environ['LLM_API_KEY'],
                        max_tokens=4096, enable_thinking=False, timeout=timeout,
                        call_budget=budget_obj)
        reply = llm.chat([dict(role='user', content='Reply with the single word: ok')],
                         caller_tag='Coder')
    inner = Path(config['run_dir']) / 'sampling-inner.jsonl'
    rows = [json.loads(line) for line in inner.read_text().splitlines()] if inner.is_file() else []
    effective = (rows[-1].get('effective') if rows else {}) or {}
    checks = {
        'journal_written': bool(rows),
        'reasoning_effort_matches': effective.get('reasoning_effort') == sampling['reasoning_effort'],
        'enable_thinking_true': effective.get('enable_thinking') is True,
        'max_tokens_matches': effective.get('max_tokens') == sampling['max_tokens'],
        'reply_nonempty': bool((reply or '').strip()),
    }
    receipt = dict(part='probe_sampling', checks=checks, all_pass=all(checks.values()),
                   effective=effective, requested=(rows[-1].get('requested') if rows else {}),
                   reply_preview=(reply or '')[:80], sampling=sampling)
    _write_once(Path(root) / 'probe-sampling.json', receipt)
    print(f'[p2-seed] probe-sampling: all_pass={receipt["all_pass"]}', flush=True)
    for key, value in checks.items():
        print(f'    {"PASS" if value else "FAIL"}  {key}', flush=True)
    return receipt


def _l2_correctness(receipt):
    for verdict in (receipt or {}).get('verdicts') or []:
        if isinstance(verdict, dict) and verdict.get('level') == 'L2':
            return ((verdict.get('metrics') or {}).get('correctness') or {}).get('verdict')
    return None


def accept_seed(code, *, declaration, root, budget, backend='ssh', host=None, work_dir=None,
                timeout=600, persistent=True):
    """验收顺序复用 ``ac_driver.final_evaluation``：同一份代码**先过完整 L2**（explore），
    再发起**独立终评**（``purpose=final``），终评通过才算 delivered。不信任 Conductor 自报 levels。

    顺序不是风格：``host_final_source`` 拒绝没有前置完整 L2 通过的源码，所以「直接 final」
    这条路在机制上不存在（与 ac_driver.final_evaluation 同一条纪律）。契约由 declaration 现算
    （``freeze_from_task``），重冻结漂移就拒。``backend`` 可换 mock 做本地成功/失败两路测试。
    """
    from .ac_driver import final_evaluation
    from operator_agent.app.hitl_tools import build_task
    from operator_agent.core.contract import freeze_from_task

    expected_hash = freeze_from_task(build_task(declaration)).hash
    verify_dir = Path(root) / 'independent-l2'
    verify_dir.mkdir(parents=True, exist_ok=True)
    host_budget = dict(wall_clock_s=budget['wall_clock_s'], max_verifications=2,
                       max_model_calls=1, max_output_tokens_per_call=8192,
                       total_output_token_reservations=8192)
    result = final_evaluation(verify_dir, code, declaration=declaration,
                              host_budget=host_budget, backend=backend, stop_after='L2',
                              host=host, work_dir=work_dir, timeout=timeout, persistent=persistent)
    contract = json.loads((verify_dir / 'contract.json').read_text()).get('hash')
    if contract != expected_hash:
        raise ValueError(f'重冻结契约漂移: {contract} != {expected_hash}')
    explore = result.get('explore') or {}
    final = result.get('final') or {}
    explore_l2 = bool((explore.get('levels') or {}).get('L2'))
    l2_passed = bool((final.get('levels') or {}).get('L2'))
    correctness = _l2_correctness(final)
    delivered = bool(result.get('delivered'))
    frozen = delivered and explore_l2 and l2_passed and correctness == 'passed'
    return dict(frozen=frozen, delivered=delivered, explore_l2=explore_l2,
                l2_passed=l2_passed, correctness=correctness, contract_hash=contract,
                final_evidence=final.get('evidence'),
                measurement_validity=final.get('validity'), reason=result.get('reason'))


def generate_seed(*, root, host, work_dir, gpu, gpu_uuid, gpu_name, base_url, model_id,
                  budget, max_rounds, sampling, call_s_per_1k=30.0):
    """进程内跑内层 Conductor（真 xhigh），取交付核，独立完整 L2 复核，过了才冻住。"""
    from benchmarks.experiments.phase1 import connected_worker
    from operator_agent.app import hitl_tools as ht

    _widen_call_timeout(call_s_per_1k, budget['wall_clock_s'])
    config, ledger, task_id = _cell_config(
        root=root, host=host, work_dir=work_dir, gpu=gpu, gpu_uuid=gpu_uuid,
        gpu_name=gpu_name, base_url=base_url, model_id=model_id, budget=budget,
        max_rounds=max_rounds, sampling=sampling)
    _create_ledger(ledger, task_id, budget)
    _write_once(Path(config['run_dir']).parent / 'cell-config.json', config)

    _provision_remote(host, work_dir, gpu, gpu_uuid, gpu_name)

    print(f'[p2-seed] generating layernorm seed via inner Conductor '
          f'(reasoning_effort={sampling["reasoning_effort"]}, '
          f'max_tokens={sampling["max_tokens"]}, max_rounds={max_rounds})', flush=True)
    t0 = time.time()
    result = connected_worker.run(config)
    elapsed = round(time.time() - t0, 1)
    print(f'[p2-seed] conductor returned in {elapsed}s: phase={result.get("phase")} '
          f'reason={result.get("reason")} levels={result.get("levels")}', flush=True)

    source_path = result.get('selected_source')
    source_hash = result.get('source_hash')
    if not source_path or not Path(source_path).is_file():
        _write_once(Path(root) / 'seed-result.json',
                    dict(part='generate_seed', frozen=False, elapsed_s=elapsed,
                         result=result, reason='no delivered source'))
        raise SystemExit(f'[p2-seed] 没有交付核（reason={result.get("reason")}）；不冻结')

    code = Path(source_path).read_text(encoding='utf-8')
    actual = hashlib.sha256(code.encode('utf-8')).hexdigest()
    if actual != source_hash:
        raise SystemExit(f'[p2-seed] 交付核哈希漂移: {actual} != {source_hash}')

    # 验收顺序：同源码先过完整 L2（explore）→ 独立终评（purpose=final）→ 过了才冻。
    # 复用 ac_driver.final_evaluation，不信任 Conductor 自报 levels；独立 run 目录避免污染生成证据。
    acceptance = accept_seed(code, declaration=DECL, root=root, budget=budget,
                             backend='ssh', host=host, work_dir=work_dir)
    print(f'[p2-seed] independent acceptance: explore_l2={acceptance["explore_l2"]} '
          f'final_l2={acceptance["l2_passed"]} correctness={acceptance["correctness"]} '
          f'delivered={acceptance["delivered"]}', flush=True)

    frozen = acceptance['frozen']
    seed = dict(
        part='generate_seed', frozen=frozen, elapsed_s=elapsed,
        source_hash=actual, contract_hash=acceptance['contract_hash'],
        source=str(source_path),
        sampling_identity=dict(sampling),
        gpu_identity=dict(index=int(gpu), uuid=gpu_uuid, name=gpu_name),
        model_identity=dict(id=model_id, base_url_sha256=digest(base_url)),
        acceptance=acceptance,
        conductor_result=dict(phase=result.get('phase'), reason=result.get('reason'),
                              levels=result.get('levels')),
        candidate_origin='inner_coder',
        note='B 方案起点核：27B 生成（采样身份见 sampling_identity，含插件 bounded 修复循环）、'
             '先过完整 L2 再独立终评通过后冻住单一哈希；两臂共用',
    )
    _write_once(Path(root) / 'seed-result.json', seed)
    if frozen:
        _write_once(Path(root) / 'seed.json',
                    dict(source_hash=actual, contract_hash=acceptance['contract_hash'],
                         sampling_identity=dict(sampling),
                         gpu_identity=dict(index=int(gpu), uuid=gpu_uuid, name=gpu_name),
                         source=str(source_path)))
        print(f'[p2-seed] FROZEN seed source_hash={actual}', flush=True)
    else:
        print(f'[p2-seed] NOT frozen: acceptance failed (delivered={acceptance["delivered"]}, '
              f'explore_l2={acceptance["explore_l2"]}, final_l2={acceptance["l2_passed"]}, '
              f'correctness={acceptance["correctness"]})', flush=True)
    return seed


def _budget(args):
    per_call = int(args.max_tokens)
    calls = int(args.max_llm_calls)
    return dict(wall_clock_s=float(args.wall_clock_s), max_llm_calls=calls,
                oracle_total=int(args.oracle_total), max_model_calls=calls,
                max_output_tokens_per_call=per_call,
                total_output_token_reservations=per_call * calls,
                max_verifications=int(args.oracle_total))


def main(argv=None):
    parser = argparse.ArgumentParser(prog='p2_seed', description=__doc__.splitlines()[0])
    parser.add_argument('--mode', choices=['probe', 'generate'], default='probe',
                        help='probe=便宜采样正控；generate=整核生成+独立L2+冻结')
    parser.add_argument('--host', default='h20')
    parser.add_argument('--work-dir', default='/root/operator-p2-seed-20261116')
    parser.add_argument('--root', default='output/v08-p2-seed-20261116')
    parser.add_argument('--gpu', default='3')
    parser.add_argument('--gpu-uuid', default=GPU3_UUID)
    parser.add_argument('--gpu-name', default='NVIDIA H20')
    parser.add_argument('--base-url', default='http://127.0.0.1:18003/v1')
    parser.add_argument('--model-id', default='Qwen3.8-27B')
    parser.add_argument('--reasoning-effort', default='medium',
                        help='起点核生成档位；用户 11-16 定 medium（成分探查：思考占 91–95%% 是撞顶主因，'
                             'medium 思考只有 xhigh ~1/4，full prompt 也轻松 fit 40960）。optimize 闭环身份另定')
    parser.add_argument('--max-tokens', type=int, default=40960,
                        help='内层冻结输出上限；40960 是 llm_limits 硬上限、也是 Coder 正常请求值。'
                             'medium 起点核生成远低于此（成分探查简化 prompt 仅 8750 总 token）')
    parser.add_argument('--max-llm-calls', type=int, default=16)
    parser.add_argument('--oracle-total', type=int, default=10)
    parser.add_argument('--wall-clock-s', type=float, default=2400.0)
    parser.add_argument('--max-rounds', type=int, default=1)
    parser.add_argument('--call-s-per-1k', type=float, default=30.0,
                        help='内层每调用超时按每 1k 输出 token 多少秒（本机 ~53 tok/s，默认 30 留余量）')
    args = parser.parse_args(argv)

    # 非密占位 key（本机 vLLM 无鉴权）；不覆盖用户已设的真实值。
    os.environ.setdefault('LLM_API_KEY', 'EMPTY')
    os.environ.setdefault('LLM_THINKING_PARAMETER', 'enable_thinking')
    budget = _budget(args)
    sampling = _wire_policy(args.reasoning_effort, args.max_tokens)
    if sampling['max_tokens'] > budget['max_output_tokens_per_call']:
        raise SystemExit('线上 max_tokens 不能超过账本每次预留')

    common = dict(root=args.root, host=args.host, work_dir=args.work_dir, gpu=args.gpu,
                  gpu_uuid=args.gpu_uuid, gpu_name=args.gpu_name, base_url=args.base_url,
                  model_id=args.model_id, budget=budget, sampling=sampling)
    Path(args.root).mkdir(parents=True, exist_ok=True)
    if args.mode == 'probe':
        receipt = probe_sampling(**common)
        return 0 if receipt['all_pass'] else 1
    seed = generate_seed(max_rounds=args.max_rounds, call_s_per_1k=args.call_s_per_1k, **common)
    return 0 if seed['frozen'] else 1


if __name__ == '__main__':
    raise SystemExit(main())