"""Freeze, execute, cancel or replay a Phase 1 offline/connected integration."""
import argparse
import json
from pathlib import Path

from operator_agent.execute.fake_backend.config import PROFILES, SCENARIOS

from . import plan, runner, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    freeze = commands.add_parser('freeze')
    freeze.add_argument('--output', required=True)
    freeze.add_argument('--scenario', choices=SCENARIOS, default='success')
    freeze.add_argument('--profile', choices=PROFILES, default='small')
    freeze.add_argument('--wall-clock-s', type=float, default=30.)
    freeze.add_argument('--max-llm-calls', type=int, default=8)
    freeze.add_argument('--oracle-total', type=int, default=20)
    freeze.add_argument('--max-rounds', type=int, default=2)
    execute = commands.add_parser('run')
    execute.add_argument('--plan', required=True)
    execute.add_argument('--output', required=True)
    connected_freeze = commands.add_parser('freeze-connected')
    connected_freeze.add_argument('--config', required=True)
    connected_freeze.add_argument('--output', required=True)
    connected_run = commands.add_parser('run-connected')
    connected_run.add_argument('--plan', required=True)
    connected_run.add_argument('--output', required=True)
    prereg = commands.add_parser('preregister')
    prereg.add_argument('--output',required=True)
    prereg.add_argument('--model-config',help='JSON model metadata only, never a literal API key')
    inspect_study=commands.add_parser('check-study')
    inspect_study.add_argument('--preregistration',required=True)
    for action in ('collect-instance','bind-study'):
        binding=commands.add_parser(action)
        binding.add_argument('--preregistration',required=True)
        binding.add_argument('--instance-config',required=True)
        binding.add_argument('--output',required=True)
    for name in ('summary', 'cancel'):
        commands.add_parser(name).add_argument('--execution', required=True)
    args = parser.parse_args(argv)
    try:
        if args.action in ('preregister','check-study','collect-instance','bind-study'):
            from . import study
            if args.action=='preregister':
                frozen=study.prepare(args.output,plan.read(args.model_config) if args.model_config else None)
                value=dict(preregistration_hash=frozen['hash'],status=frozen['status'],
                    model_configured=frozen['bindings']['model_configured'],
                    planned_cells=len(frozen['schedule']),formal_execution_ready=False)
            elif args.action=='check-study':
                frozen=study.load(args.preregistration)
                value=dict(ok=True,preregistration_hash=frozen['hash'],formal_execution_ready=False)
            elif args.action=='collect-instance':
                value=study.collect(args.preregistration,plan.read(args.instance_config),args.output)
            else:
                frozen=study.bind(args.preregistration,plan.read(args.instance_config),args.output)
                value=dict(plan_hash=frozen['hash'],plan_path=str(Path(args.output).resolve()/'plan.json'),
                    status='bound_requires_per_cell_instance_checks',formal_comparison_credit=False)
        elif args.action == 'freeze':
            value = plan.freeze(args.output, fake=dict(scenario=args.scenario, profile=args.profile),
                budget=dict(wall_clock_s=args.wall_clock_s, max_llm_calls=args.max_llm_calls,
                            oracle_total=args.oracle_total), max_rounds=args.max_rounds)
            value = dict(plan_hash=value['hash'], planned_cells=len(value['schedule']),
                         plan_path=str(Path(args.output).resolve()/'plan.json'))
        elif args.action == 'freeze-connected':
            from .connected_plan import freeze as freeze_connected
            frozen = freeze_connected(args.output, plan.read(args.config))
            value = dict(plan_hash=frozen['hash'], planned_cells=len(frozen['schedule']),
                         plan_path=str(Path(args.output).resolve()/'plan.json'))
        elif args.action == 'run-connected':
            from .connected import run as run_connected
            value = run_connected(args.plan,args.output)
        elif args.action == 'run':
            value = runner.run(args.plan, args.output)
        elif args.action == 'summary':
            if plan.read(Path(args.execution)/'plan.json')['schema']=='phase1-connected-plan-v1':
                from .connected_summary import summarize
                value=summarize(args.execution)
            else:
                value = summary.summarize(args.execution)
        else:
            value = runner.cancel(args.execution)
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps(dict(ok=False, error=type(exc).__name__, detail=str(exc)), ensure_ascii=False))
        return 2
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
