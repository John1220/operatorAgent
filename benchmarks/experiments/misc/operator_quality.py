"""Plan, replay, execute or compare fixed operator quality experiments."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from operator_agent.quality_evaluation import make_plan, run_plan, run_preflight
from operator_agent.quality_pairing import (PAIRED_PLAN_SCHEMAS, make_paired_plan,
    make_seeded_paired_plan, run_paired_plan, validate_paired_plan)
from operator_agent.quality_portfolio import (make_portfolio_plan, run_portfolio_plan,
    validate_portfolio_plan)
from operator_agent.quality_suite import compare_reports

PORTFOLIO_PLAN_SCHEMA = 'operator-portfolio-quality-plan-v1'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--execute', metavar='PLAN')
    actions.add_argument('--preflight', metavar='PLAN', help='Check an existing SSH runtime; no model calls')
    actions.add_argument('--replay', metavar='RESPONSES')
    actions.add_argument('--compare', nargs=2, metavar=('LEFT', 'RIGHT'))
    parser.add_argument('--model', default=os.environ.get('LLM_MODEL', 'offline-replay'))
    parser.add_argument('--backend', choices=('mock', 'ssh'), default='mock')
    parser.add_argument('--host', default='')
    parser.add_argument('--strategy', choices=('baseline', 'diagnosis'))
    parser.add_argument('--paired', action='store_true', help='Compare both repair strategies from one initial source')
    parser.add_argument('--portfolio', action='store_true',
                        help='Generate three strategy candidates and select the first full L2 pass')
    parser.add_argument('--initial-source', action='append', metavar='CASE=FILE',
                        help='Freeze a supplied source per paired case; no shared generation')
    parser.add_argument('--case', action='append', dest='cases')
    parser.add_argument('--output', help='Plan file, or a NEW directory for execution/replay')
    args = parser.parse_args(argv)
    if args.portfolio and (args.paired or args.strategy is not None or
                           args.initial_source or args.compare):
        parser.error('--portfolio fixes three generation strategies; do not combine with '
                     '--paired/--strategy/--initial-source/--compare')
    if args.paired and (args.strategy is not None or args.compare):
        parser.error('--paired already fixes both strategies; do not combine with --strategy/--compare')
    if args.initial_source and (not args.paired or args.execute or args.preflight or args.compare):
        parser.error('--initial-source requires --paired planning/replay; execution uses frozen sources')
    initial_sources = None
    if args.initial_source:
        initial_sources = {}
        for declaration in args.initial_source:
            case, separator, path = declaration.partition('=')
            if not separator or not case or not path or case in initial_sources:
                parser.error('--initial-source needs unique CASE=FILE entries')
            try:
                with Path(path).open('rb') as stream:
                    data = stream.read(32769)
                if len(data) > 32768:
                    raise ValueError('source byte limit')
                initial_sources[case] = data.decode('utf-8')
            except (OSError, ValueError):
                parser.error('initial source must be an existing UTF-8 file of at most 32768 bytes')
        if args.cases is None:
            args.cases = list(initial_sources)
    def paired_plan(**options):
        if initial_sources is None:
            return make_paired_plan(**options)
        try:
            return make_seeded_paired_plan(initial_sources=initial_sources, **options)
        except ValueError:
            parser.error('initial sources must exactly cover valid selected cases and contain nonempty source')
    read = lambda path: json.loads(Path(path).read_text())
    if args.compare:
        result = compare_reports(*(read(path) for path in args.compare))
    elif args.execute:
        if not args.output:
            parser.error('--execute requires --output NEW_DIRECTORY')
        plan = read(args.execute)
        paired = plan.get('schema') in PAIRED_PLAN_SCHEMAS
        portfolio = plan.get('schema') == PORTFOLIO_PLAN_SCHEMA
        if args.paired and not paired:
            parser.error('--paired requires a paired frozen plan')
        if args.portfolio and not portfolio:
            parser.error('--portfolio requires a portfolio frozen plan')
        runner = run_portfolio_plan if portfolio else run_paired_plan if paired else run_plan
        result = runner(plan, args.output, host=args.host)
    elif args.preflight:
        if not args.output:
            parser.error('--preflight requires --output NEW_DIRECTORY')
        plan = read(args.preflight)
        portfolio = plan.get('schema') == PORTFOLIO_PLAN_SCHEMA
        if args.portfolio and not portfolio:
            parser.error('--portfolio requires a portfolio frozen plan')
        if portfolio:
            validate_portfolio_plan(plan)
            plan = plan['base_plan']
        elif plan.get('schema') in PAIRED_PLAN_SCHEMAS:
            validate_paired_plan(plan)
            plan = plan['base_plan']
        result = run_preflight(plan, args.output, host=args.host)
    elif args.replay:
        if not args.output:
            parser.error('--replay requires --output NEW_DIRECTORY')
        if args.portfolio:
            plan = make_portfolio_plan(model='offline-replay', case_ids=args.cases)
            result = run_portfolio_plan(plan, args.output, responses=read(args.replay))
        elif args.paired:
            plan = paired_plan(model='offline-replay', case_ids=args.cases)
            result = run_paired_plan(plan, args.output, responses=read(args.replay))
        else:
            plan = make_plan(model='offline-replay', strategy=args.strategy or 'diagnosis', case_ids=args.cases)
            result = run_plan(plan, args.output, responses=read(args.replay))
    else:
        options = dict(model=args.model, endpoint=os.environ.get('LLM_BASE_URL', ''),
                       backend=args.backend, host=args.host, case_ids=args.cases)
        result = (make_portfolio_plan(**options) if args.portfolio else
                  paired_plan(**options) if args.paired else
                  make_plan(strategy=args.strategy or 'diagnosis', **options))
        if args.output:
            with Path(args.output).open('x') as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
