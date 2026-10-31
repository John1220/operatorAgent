"""One DSH agent turn through the budgeted gateway on the Responses protocol.

The gateway is the only caller of the supplied upstream transport, so the arm
never holds the provider credential. This module is backend-agnostic: a loopback
Responses fixture for offline tests, or a real HTTPS endpoint supplied by an
authorized caller. It does not claim isolation, GPU execution or provider
completion beyond what the ledger and the raw upstream framing support.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from benchmarks.experiments.s5 import s5_arm_env
from benchmarks.experiments.s5.s5_gateway_http import GatewayServer
from benchmarks.experiments.s5.s5_gateway_transport import UpstreamTransport
from operator_agent.execute import s5_shared_budget as shared

SCHEMA = 's5-responses-turn-v1'
PLAN_SCHEMA = 's5-responses-plan-v1'
TASK = 'Reply with exactly OK.'
TASK_ID = 'responses-turn-1'


def _write_once(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


def _final_text(events):
    texts = [item.get('text') for item in events
             if isinstance(item, dict) and item.get('type') == 'final']
    return texts[0] if len(texts) == 1 and isinstance(texts[0], str) else None


def _identities():
    """Hash the code that would dispatch the request; not a dependency lock."""
    root = Path(__file__).resolve().parent
    files = [root / name for name in ('s5_gateway_http.py', 's5_gateway_policy.py',
                                      's5_gateway_transport.py', 's5_gateway_supervisor.py',
                                      's5_arm_env.py', 's5_responses_turn.py')]
    files += [root.parents[2] / 'operator_agent' / name for name in
              ('execute/s5_shared_budget.py', 'core/run_layout.py', 'execute/task_clock.py')]
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def freeze(output, *, endpoint, model='fake-model', transport='direct',
           max_output_tokens=256, timeout=60):
    """Write one non-overwritable plan and return its on-disk SHA256."""
    if transport not in ('direct', 'supervised'):
        raise ValueError('unknown transport kind')
    if type(max_output_tokens) is not int or not 0 < max_output_tokens <= 40960:
        raise ValueError('invalid output bound')
    if type(timeout) is not int or not 5 <= timeout <= 120:
        raise ValueError('invalid wall clock')
    UpstreamTransport(endpoint, 'placeholder', timeout_seconds=min(60, timeout))
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    plan = dict(schema=PLAN_SCHEMA, endpoint=endpoint, model=model, transport=transport,
                prompt=TASK, max_calls=1, max_output_tokens=max_output_tokens,
                timeout=timeout, source_sha256=_identities())
    target = root / 'plan.json'
    _write_once(target, plan)
    return hashlib.sha256(target.read_bytes()).hexdigest()


def run_planned(output, plan_path, expected_sha256, *, api_key, upstream_kind='real'):
    """Refuse to run unless the frozen plan is unchanged; then execute once."""
    plan_path = Path(plan_path)
    if plan_path.is_symlink() or not plan_path.is_file():
        raise ValueError('plan missing')
    raw = plan_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('plan hash mismatch')
    plan = json.loads(raw)
    if (not isinstance(plan, dict) or plan.get('schema') != PLAN_SCHEMA
            or plan.get('prompt') != TASK or plan.get('max_calls') != 1
            or plan.get('source_sha256') != _identities()
            or plan.get('transport') not in ('direct', 'supervised')):
        raise ValueError('plan configuration mismatch')
    UpstreamTransport(plan['endpoint'], 'placeholder', timeout_seconds=min(60, plan['timeout']))
    return run(Path(output), upstream_url=plan['endpoint'], api_key=api_key,
               model=plan['model'], transport=plan['transport'],
               timeout=plan['timeout'], upstream_kind=upstream_kind,
               max_output_tokens=plan['max_output_tokens'])


def run(output, *, upstream_url, api_key='placeholder', model='fake-model',
        transport='direct', timeout=60, upstream_kind='fixture', max_output_tokens=256):
    """Run exactly one agent turn; never raises for a turn-level failure."""
    if transport not in ('direct', 'supervised'):
        raise ValueError('unknown transport kind')
    if upstream_kind not in ('fixture', 'real'):
        raise ValueError('unknown upstream kind')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    started = time.monotonic()
    result = dict(schema=SCHEMA, status='unknown', error=None, model=model,
                  gateway_protocol='responses', transport=transport,
                  upstream_kind=upstream_kind, real_provider_call=False,
                  provider_usage=None, usage_evidence='missing',
                  isolation_verified=False, gpu_used=False, formal_s5_credit=False,
                  linux_host_verified=False)
    ledger = output / 'budget.sqlite'
    try:
        shared.create(ledger, task_id=TASK_ID, max_calls=1,
                      max_output_tokens=max_output_tokens,
                      total_output_tokens=max_output_tokens, wall_clock_s=timeout + 60)
        if transport == 'supervised':
            from benchmarks.experiments.s5.s5_gateway_supervisor import SupervisedTransport
            transport_object = SupervisedTransport(upstream_url, api_key,
                                                   wall_seconds=min(60, timeout),
                                                   max_request_bytes=65536)
            call_upstream = transport_object
        else:
            transport_object = UpstreamTransport(upstream_url, api_key,
                                                 timeout_seconds=min(60, timeout),
                                                 max_request_bytes=65536)
            call_upstream = transport_object.stream
        holder = {}

        def adapter(request):
            exchange = call_upstream(request)
            holder['exchange'] = exchange
            return exchange
        home = output / 'arm' / 'home'
        home.mkdir(parents=True, mode=0o700)
        (output / 'arm' / 'work').mkdir(mode=0o700)
        with GatewayServer(ledger, task_id=TASK_ID, model=model, participant='outer',
                           adapter=adapter, protocol='responses',
                           max_request_bytes=65536) as gateway:
            built = s5_arm_env.build('A', base_url=gateway.base_url,
                                     arm_dir=output / 'arm', api='responses',
                                     max_tokens=max_output_tokens, model=model)
            _write_once(output / 'launch.json', built)
            response = s5_arm_env.start_headless(built, TASK, timeout=timeout)
        _write_once(output / 'headless-events.json', response['events'])
        _write_once(output / 'headless-stderr.json',
                    dict(text=response['stderr'],
                         sha256=hashlib.sha256(response['stderr'].encode()).hexdigest()))
        snapshot = shared.snapshot(ledger, expected_task_id=TASK_ID)
        _write_once(output / 'budget.json', snapshot)
        result['model_calls'] = dict(reserved=snapshot['reserved_calls'],
                                     completed=snapshot['completed_calls'],
                                     aborted=snapshot['aborted'],
                                     outcomes=[call.get('outcome') for call in snapshot['calls']])
        exchange = holder.get('exchange')
        result['upstream_framing'] = getattr(exchange, 'outcome', 'not_started')
        result['upstream_cleanup'] = getattr(exchange, 'cleanup', 'unknown')
        result['usage_evidence'] = ('provider_complete' if any(
            call.get('api_usage') is not None for call in snapshot['calls']) else 'missing')
        text = _final_text(response['events'])
        if response['returncode'] != 0:
            result['error'] = 'headless_exit_%r' % response['returncode']
            return result
        if text is None:
            result['error'] = 'final_text_missing'
            return result
        result['final_text_sha256'] = hashlib.sha256(text.encode()).hexdigest()
        result['final_text_bytes'] = len(text.encode())
        if (snapshot['reserved_calls'] != 1 or snapshot['completed_calls'] != 1
                or snapshot['aborted'] is not False):
            result['error'] = 'model_budget_evidence_incomplete'
            return result
        if result['upstream_framing'] != 'answered':
            result['error'] = 'upstream_framing_' + str(result['upstream_framing'])
            return result
        if not text.strip():
            result['error'] = 'empty_final_text'
            return result
        result['status'] = 'passed'
    except Exception as exc:  # noqa: BLE001 - a failed turn is a recorded fact
        result['error'] = type(exc).__name__
        try:
            if ledger.is_file() and not (output / 'budget.json').exists():
                _write_once(output / 'budget.json',
                            shared.snapshot(ledger, expected_task_id=TASK_ID))
        except Exception:
            pass
    finally:
        result['wall_seconds'] = round(time.monotonic() - started, 3)
        _write_once(output / 'result.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--upstream', required=True, help='Responses endpoint URL')
    parser.add_argument('--model', default='fake-model')
    parser.add_argument('--transport', choices=('direct', 'supervised'), default='direct')
    parser.add_argument('--api-key-file', type=Path,
                        help='read the upstream credential from this file')
    args = parser.parse_args(argv)
    key = (args.api_key_file.read_text(encoding='utf-8').strip()
           if args.api_key_file else 'placeholder')
    result = run(args.output, upstream_url=args.upstream, api_key=key,
                 model=args.model, transport=args.transport,
                 upstream_kind='real' if args.api_key_file else 'fixture')
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())