"""No-model MCP discovery probe for the performance-study tool surface."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.s5_perf_mcp import PerformanceRelay, ALLOWED


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


def run(*, task_dir, declaration_path, pin_path, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    relay = PerformanceRelay(task_dir, read(declaration_path), output/'relay',
                             pin=read(pin_path),
                             resource_policy=dict(mode='shared-known-processes',
                                 min_free_mib=8192, max_util_pct=5,
                                 allowed_compute_pids=[]),
                             timeout=660, max_candidates=3)
    result = dict(schema='s5-performance-mcp-probe-v1', status='unknown',
                  model_calls=0, candidate_submissions=0, gpu_verifications=0,
                  gpu_benchmarks=0, formal_s5_credit=False)
    try:
        relay.__enter__()
        initialize = relay.request(dict(jsonrpc='2.0', id='probe-initialize',
            method='initialize', params=dict(protocolVersion='2025-11-25', capabilities={},
            clientInfo=dict(name='s5-performance-probe', version='1'))))
        listing = relay.request(dict(jsonrpc='2.0', id='probe-tools',
                                     method='tools/list', params={}))
        if not isinstance(initialize, dict) or 'result' not in initialize:
            raise ValueError('MCP initialization missing')
        if not isinstance(listing, dict) or 'result' not in listing:
            raise ValueError('MCP tool listing missing')
        tools = listing['result'].get('tools')
        if not isinstance(tools, list):
            raise ValueError('MCP tools missing')
        by_name = {tool.get('name'): tool for tool in tools if isinstance(tool, dict)}
        visible = sorted(by_name)
        bench = by_name.get('operator_bench')
        verify = by_name.get('operator_verify')
        if not isinstance(bench, dict) or not isinstance(verify, dict):
            raise ValueError('required performance tools missing')
        props = (bench.get('inputSchema') or {}).get('properties') or {}
        if not {'run_dir', 'candidates'} <= set(props):
            raise ValueError('operator_bench schema drift')
        if set(visible) != set(ALLOWED):
            raise ValueError('MCP tool allowlist mismatch')
        result.update(status='passed', visible_tools=visible,
            allowed_tools=sorted(ALLOWED), bench_schema_sha256=hashlib.sha256(
                json.dumps(bench.get('inputSchema'), sort_keys=True,
                           separators=(',', ':')).encode()).hexdigest(),
            verify_schema_sha256=hashlib.sha256(
                json.dumps(verify.get('inputSchema'), sort_keys=True,
                           separators=(',', ':')).encode()).hexdigest(),
            relay=relay.snapshot())
    finally:
        relay.__exit__(None, None, None)
        result['relay_final'] = relay.snapshot()
        write(output/'result.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-dir', required=True, type=Path)
    parser.add_argument('--declaration', required=True, type=Path)
    parser.add_argument('--pin', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    result = run(task_dir=args.task_dir, declaration_path=args.declaration,
                 pin_path=args.pin, output=args.output)
    print(json.dumps(result, sort_keys=True))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
