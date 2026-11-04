"""Freeze small offline fixtures separately from the real four-task study."""
import hashlib
import json
from pathlib import Path

from operator_agent.app.hitl_tools import build_task
from operator_agent.core.contract import freeze_from_task
from operator_agent.core.run_contract import digest, protocol_identity
from operator_agent.execute.fake_backend import FakeConfig, execution_context
from operator_agent.execute.task_budget import TaskBudgetConfig

SCHEMA = 'phase1-offline-plan-v1'
REPO = Path(__file__).resolve().parents[3]
FIXTURES = (('matmul-small', 'matmul fp32', [8, 8, 8]),
            ('relu-wide', 'relu fp32', [8, 16]),
            ('matmul-wide', 'matmul fp32', [16, 16, 16]),
            ('relu-small', 'relu fp32', [4, 8]))
ENTRIES = dict(A='benchmarks.experiments.phase1.baseline',
               B='benchmarks.experiments.phase1.delegate')
MODEL = dict(id='fake-fixture-v1', transport='none', provider_dispatch=False)
COVERAGE = dict(conversation='missing', tools='complete', external_edits='missing')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def fresh_directory(path):
    path = Path(path).resolve()
    if path == REPO or REPO in path.parents:
        raise ValueError('store execution evidence outside the source tree')
    path.mkdir(parents=True, exist_ok=False, mode=0o700)
    return path


def implementation_identity():
    files = sorted(Path(__file__).parent.glob('*.py'))
    return dict(protocol=protocol_identity(), harness={p.name: hashlib.sha256(
        p.read_bytes()).hexdigest() for p in files})


def schedule(tasks):
    return [dict(cell_id=task['task_id']+'-'+arm, task_id=task['task_id'], arm=arm)
            for index, task in enumerate(tasks)
            for arm in (('A', 'B') if index % 2 == 0 else ('B', 'A'))]


def freeze(output, *, fake=None, budget=None, max_rounds=2, final_wall_clock_s=30.):
    config = FakeConfig.from_dict(fake)
    if config.llm != 'fake':
        raise ValueError('offline rehearsal requires fake model; live adapters are not enabled')
    policy = TaskBudgetConfig.from_dict(budget or dict(
        wall_clock_s=30., max_llm_calls=8, oracle_total=20))
    if policy.max_llm_calls > 30 or policy.wall_clock_s > 2400 or policy.oracle_total > 20:
        raise ValueError('rehearsal may reduce, but cannot enlarge, the planned cell budget')
    if type(max_rounds) is not int or not 1 <= max_rounds <= 8:
        raise ValueError('max_rounds must be between 1 and 8')
    TaskBudgetConfig(wall_clock_s=final_wall_clock_s)
    tasks = []
    for task_id, request, shape in FIXTURES:
        declaration = dict(request=request, shapes=[shape], dsl='python',
                           precision='fp32', min_speedup=1.)
        contract = freeze_from_task(build_task(declaration))
        tasks.append(dict(task_id=task_id, declaration=declaration, contract_hash=contract.hash))
    plan = dict(schema=SCHEMA, plan_id='phase1-offline-rehearsal-v1',
        target_plan_id='phase1-agent-comparison-v2', backend='fake',
        execution_context=execution_context(config), fake=config.as_dict(),
        model=dict(MODEL),
        budget=policy.as_dict(), max_rounds=max_rounds,
        final_evaluation=dict(max_verifications=1, wall_clock_s=final_wall_clock_s),
        entries=dict(ENTRIES),
        visibility=dict(isolation_verified=False, shared_source_tree=True,
                        note='separate work directories, no hostile-code read isolation'),
        coverage=dict(COVERAGE),
        implementation=implementation_identity(), tasks=tasks, schedule=schedule(tasks),
        formal_comparison_credit=False, dsh_entry_verified=False)
    plan['hash'] = digest(plan)
    root = fresh_directory(output)
    write_once(root/'plan.json', plan)
    return plan


def validate(plan, *, check_implementation=True):
    if not isinstance(plan, dict) or plan.get('schema') != SCHEMA:
        raise ValueError('not a phase1 offline rehearsal plan')
    body = {k: v for k, v in plan.items() if k != 'hash'}
    if digest(body) != plan.get('hash'):
        raise ValueError('plan hash mismatch')
    if (plan.get('plan_id') != 'phase1-offline-rehearsal-v1'
            or plan.get('target_plan_id') != 'phase1-agent-comparison-v2'
            or plan.get('entries') != ENTRIES or plan.get('model') != MODEL
            or plan.get('coverage') != COVERAGE
            or plan.get('visibility') != dict(isolation_verified=False, shared_source_tree=True,
                note='separate work directories, no hostile-code read isolation')):
        raise ValueError('offline plan identity/entry/model/coverage mismatch')
    if plan.get('backend') != 'fake' or FakeConfig.from_dict(plan.get('fake')).llm != 'fake':
        raise ValueError('offline plan cannot dispatch live model/GPU work')
    if plan.get('execution_context') != execution_context(plan['fake']):
        raise ValueError('fake execution context mismatch')
    policy = TaskBudgetConfig.from_dict(plan['budget'])
    if policy.max_llm_calls > 30 or policy.wall_clock_s > 2400 or policy.oracle_total > 20:
        raise ValueError('unplanned budget expansion')
    if (type(plan['max_rounds']) is not int or not 1 <= plan['max_rounds'] <= 8
            or type(plan['final_evaluation']['max_verifications']) is not int
            or plan['final_evaluation']['max_verifications'] != 1):
        raise ValueError('invalid round/final evaluation policy')
    TaskBudgetConfig(wall_clock_s=plan['final_evaluation']['wall_clock_s'])
    tasks = plan.get('tasks')
    if not isinstance(tasks, list) or len(tasks) != 4:
        raise ValueError('four fixture tasks are required')
    for row, (task_id, request, shape) in zip(tasks, FIXTURES):
        expected = dict(request=request, shapes=[shape], dsl='python', precision='fp32', min_speedup=1.)
        if row.get('task_id') != task_id or row.get('declaration') != expected:
            raise ValueError('fixture task mismatch')
        if freeze_from_task(build_task(expected)).hash != row.get('contract_hash'):
            raise ValueError('task contract drift')
    if plan.get('schedule') != schedule(tasks):
        raise ValueError('schedule must retain all eight interleaved slots')
    if plan.get('formal_comparison_credit') is not False or plan.get('dsh_entry_verified') is not False:
        raise ValueError('offline rehearsal cannot claim formal/DSH credit')
    if check_implementation and plan.get('implementation') != implementation_identity():
        raise ValueError('implementation drift: freeze a new rehearsal')
    return plan


def load(path, *, check_implementation=True):
    return validate(read(path), check_implementation=check_implementation)
