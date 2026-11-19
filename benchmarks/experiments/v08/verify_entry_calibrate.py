"""对比 A 与 C 两条**验证入口**：同一份候选、同一契约、零模型调用。

为什么要这一件：v9 那对里两臂各验证 4 次，但 C 的第二次验证耗时 298.7 s、A 的只有
102.3 s —— 差的 196 s 几乎就是整对墙钟差（369.8 vs 541.9）。如果这个差能复现，
它属于**验证链路**，不属于模型，也不该记在某一边头上。

两条入口：

* **A**：`VerificationService.develop()` —— 与 `check.py` 走完全相同的一条路
  （本地 supervisor → 可信 worker 进程 → `hitl_tools.verify`）；
* **C**：直接 `hitl_tools.verify` —— 与 `operator_verify` 工具同一条路。

每次派发都数远端 RPC（按 action 分组）并计时，这样差值能落到具体环节，
而不是只说"C 慢一点"。

用法::

    python -m benchmarks.experiments.v08.verify_entry_calibrate \
        --code benchmarks/repro/matmul_strided_20261107/historical_fix.py \
        --declaration-json benchmarks/repro/matmul_strided_20261107/case.json \
        --root logs/verify-entry-calib-20261119 --rounds 2
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

from operator_agent.app import hitl_tools
from operator_agent.execute import remote_executor as rx
from operator_agent.execute.backend_ownership import owned_worker_scope
from operator_agent.execute.task_clock import monotonic

SCHEMA = 'v08-verify-entry-calibration-v1'


class RpcCounter:
    """数远端 RPC（按 action），并累计耗时。"""

    def __init__(self):
        self.calls = []
        self._original = None

    def __enter__(self):
        self._original = rx.SSHExecutor._ssh_exec_json
        original = self._original

        def counted(inner, payload, *a, **k):
            started = time.time()
            try:
                return original(inner, payload, *a, **k)
            finally:
                self.calls.append(dict(action=payload.get('action'),
                                       seconds=round(time.time() - started, 2)))
        rx.SSHExecutor._ssh_exec_json = counted
        return self

    def __exit__(self, *exc):
        rx.SSHExecutor._ssh_exec_json = self._original

    def summary(self):
        groups = {}
        for row in self.calls:
            count, total = groups.get(row['action'], (0, 0.0))
            groups[row['action']] = (count + 1, total + row['seconds'])
        return {action: dict(calls=count, seconds=round(total, 1))
                for action, (count, total) in sorted(groups.items())}


def _declaration(case_json):
    case = json.loads(Path(case_json).read_text(encoding='utf-8'))
    return dict(case['declaration'])


def run_a_entry(*, root, declaration, code, host, work_dir, timeout, label):
    """A 入口：supervisor 的 VerificationService（与 check.py 同一条路）。"""
    from benchmarks.experiments.phase1.native import VerificationService

    run = root / label
    run.mkdir(parents=True, exist_ok=False)
    budget = dict(wall_clock_s=1800, max_model_calls=4, max_output_tokens_per_call=8192,
                  total_output_token_reservations=32768, max_verifications=4,
                  final_reserve=1)
    prepared = hitl_tools.prepare(declaration, run, host_budget=dict(budget))
    service = VerificationService(dict(
        run_dir=str(run), backend='ssh', connection=dict(host=host, work_dir=work_dir,
                                                        persistent=True),
        declaration=declaration, execution_context=dict(entry='A', label=label),
        stop_after='L2',
        budget=dict(wall_clock_s=1800, max_llm_calls=4, oracle_total=4),
        deadline_monotonic=monotonic() + 1800), prepared['contract_hash'])
    with RpcCounter() as rpc:
        started = time.time()
        with service:
            reply = service.develop(code, prepared['contract_hash'])
        elapsed = round(time.time() - started, 1)
    return dict(entry='A', label=label, seconds=elapsed,
                levels=reply.get('levels'), status=reply.get('status'),
                rpc=rpc.summary(), evidence=reply.get('verification_evidence'))


def run_c_entry(*, root, declaration, code, host, work_dir, timeout, label):
    """C 入口：`hitl_tools.verify`（与 operator_verify 同一条路）。"""
    run = root / label
    run.mkdir(parents=True, exist_ok=False)
    hitl_tools.prepare(declaration, run, host_budget=dict(
        wall_clock_s=1800, max_verifications=4, max_model_calls=1,
        max_output_tokens_per_call=8192, total_output_token_reservations=8192))
    worker = run / 'owned-worker' / uuid.uuid4().hex
    worker.mkdir(parents=True, exist_ok=True)
    with RpcCounter() as rpc:
        started = time.time()
        with owned_worker_scope(worker, uuid.uuid4().hex):
            reply = hitl_tools.verify(run, code, backend='ssh', host=host,
                                      work_dir=work_dir, timeout=timeout,
                                      persistent=True, stop_after='L2',
                                      candidate_origin='outer')
        elapsed = round(time.time() - started, 1)
    return dict(entry='C', label=label, seconds=elapsed, levels=reply.get('levels'),
                status='passed' if (reply.get('levels') or {}).get('L2') else 'failed',
                rpc=rpc.summary(), evidence=reply.get('evidence'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--declaration-json', type=Path, required=True,
                        help='case.json（取其 declaration 字段）')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--host', default='h20')
    parser.add_argument('--work-dir', default='/root/operator-p2-seed-20261116')
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--rounds', type=int, default=2)
    args = parser.parse_args(argv)

    declaration = _declaration(args.declaration_json)
    code = args.code.read_text(encoding='utf-8')
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    records = []
    for index in range(1, args.rounds + 1):
        for runner in (run_a_entry, run_c_entry):
            label = f'round{index}-{runner.__name__.split("_")[1]}'
            record = runner(root=root, declaration=declaration, code=code,
                            host=args.host, work_dir=args.work_dir,
                            timeout=args.timeout, label=label)
            records.append(record)
            print(json.dumps({k: record[k] for k in
                              ('entry', 'label', 'seconds', 'levels', 'status')},
                             ensure_ascii=False), flush=True)
    summary = dict(schema=SCHEMA, code_sha=__import__('hashlib').sha256(
        code.encode()).hexdigest(), rounds=args.rounds, records=records)
    (root / 'calibration.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())