"""R2 preregistration: freeze the v0.7.2 vs v0.8 increment comparison before running it.

The point of writing this down first is that "what counts as a result" cannot be chosen
after the numbers arrive. Everything here is either a frozen declaration or a number that
was measured in a named window; nothing is a claim about an outcome.

Two budgets are deliberately left empty. `freeze()` refuses to write a plan while they are
`None` — a preregistration whose budget is decided later is not a preregistration.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from operator_agent.core.run_contract import digest

SCHEMA = 'v08-increment-preregistration-v1'
PLAN_ID = 'operator-v08-increment-r2'

#: 窗口恶化多少倍时，预登记的效应仍要能判出来。带宽是 3× 实测散布，窗口一变吵它就变宽；
#: 留到 10 倍，是为了让"今天 0.2% 可判"不会因为明天 load 从 5 涨到 50 就作废。
WINDOW_DEGRADATION_MARGIN = 10.0

TASKS = [dict(
    task_id='matmul-4096',
    declaration=dict(operator='matmul', request='fp16 4096x4096x4096 matmul, triton', shapes=[[4096, 4096, 4096]],
                     precision='fp16', dsl='triton', hardware='nvidia_gpu', min_speedup=1.0),
    why='the only shape measured in a decidable window: 0.203% band on 2026-11-07',
    measured_headroom_pct=7.7,
    measured_baseline_gap='naive triton reaches 92.3% of peak and 0.99x of the torch baseline',
), dict(
    task_id='matmul-4096-fp32',
    declaration=dict(operator='matmul', request='fp32 4096x4096x4096 matmul, triton', shapes=[[4096, 4096, 4096]],
                     precision='fp32', dsl='triton', hardware='nvidia_gpu', min_speedup=1.0),
    # pair 1 证明 fp16 4096³ 已贴着峰值 (余量 7.7%), 门槛 2.5% 在噪声边缘;
    # fp32 版本同窗口实测: 探针带宽 0.10%, 朴素实现只到 33.8% 峰值 (余量 66.2%),
    # 且相对基线只有 0.35x —— 余量与基线差距都足够大。
    why='measured 66.2% headroom and a 0.35x baseline gap on 2026-11-07; gate band 0.10%',
    measured_headroom_pct=66.2,
    measured_baseline_gap='naive triton fp32 runs at 0.3545x of the torch baseline',
), dict(
    task_id='attention-prefill-1024',
    declaration=dict(operator='attention', request='fp16 attention prefill 1x16x1024x128',
                     shapes=[[1, 16, 1024, 128]],
                     precision='fp16', dsl='triton', hardware='nvidia_gpu', min_speedup=1.0),
    # 前两个任务都贴着峰值（fp16 余量 7.7%）或只差一个配置（fp32 余量 66.2% 但基线 0.35x）。
    # attention 是唯一同时"余量大 + 基线差距大 + 绑定轴不显然"的热点：C5 实测达成率 60.2%
    # （粗估模型），同一个正确的 triton 核只有 SDPA 基线的 0.88x。
    why='measured 39.8% headroom and a 0.88x baseline gap on 2026-11-08; gate band 0.1%',
    measured_headroom_pct=39.8,
    measured_baseline_gap=('one correct triton candidate ran at 0.8826x of the same-card '
                           'SDPA baseline (degraded L3, so this is a task-selection fact, '
                           'not a performance claim)'),
    # 门控必须量**这个**任务的 mask 形态：causal 会把 FLOPs 砍半，两种形态的窗口读数
    # 不能互相顶替（与 fp32 计划当年继承 fp16 探针是同一条纪律）。
    mask='nocausal',
    notes=[
        'primary case is causal=False: prepare() has no way to ask for causal=True today, '
        'so the causal mask is exercised by the 8-case correctness suite only',
        'achievement (60.2%) comes from the coarse roofline model attention_coarse_v1, '
        'not from a measured peak; reference_lower_bound stays false',
        'the 8-case suite is a correctness scope, not 8 performance cases: L3 measures '
        'the primary case alone and no case_weights/max_regression are declared',
        'SUPERSEDED: S=1024 cannot produce a valid L3 (launch floor 17.3% > 10%), so a '
        'pair on this task would be non-decidable by construction',
    ],
), dict(
    task_id='attention-prefill-4096',
    declaration=dict(operator='attention', request='fp16 attention prefill 1x16x4096x128',
                     shapes=[[1, 16, 4096, 128]],
                     precision='fp16', dsl='triton', hardware='nvidia_gpu', min_speedup=1.0),
    # S=1024 的核只有 96 µs，而 launch 地板是 20 µs：地板占 17.3% > 10%，测量档位
    # 结构性停在 degraded —— 那样的配对一律不可判。S=4096 把核拉长到 1.21 ms，
    # 地板占比降到 1.6%，2026-11-08 实测拿到 **valid**。
    why=('measured 23.4% headroom and a 0.86x baseline gap, with a VALID L3 '
         '(launch dominance 1.6%, dispersion 0.07%, 47 runs) on 2026-11-08; gate band 0.1%'),
    measured_headroom_pct=23.4,
    measured_baseline_gap=('one correct triton candidate ran at 0.8608x of the same-card '
                           'SDPA baseline in a VALID reading (2026-11-08)'),
    mask='nocausal',
    notes=[
        'valid L3 comes from S=4096 only: at S=1024 the same kernel is structurally degraded',
        'achievement (76.6%) comes from the coarse roofline model attention_coarse_v1 '
        '(AI=2048 FLOP/byte, compute-bound), not from a measured peak',
        'the 8-case suite is a correctness scope, not 8 performance cases: L3 measures '
        'the primary case alone and no case_weights/max_regression are declared',
    ],
)]

#: 任务 → 计划编号。换任务必须新开计划, 不能原位改已冻结的那份。
PLAN_IDS = {'matmul-4096': PLAN_ID,
            'matmul-4096-fp32': 'operator-v08-increment-r2-fp32-v2',
            'attention-prefill-1024': 'operator-v08-increment-r2-attention-v1',
            'attention-prefill-4096': 'operator-v08-increment-r2-attention-v2'}

#: 撤回的编号必须被校验器**指名拒绝**, 而不是落进"编号不认识"那条通用消息里 ——
#: archived 的那份 JSON 不改（它在仓外证据树里）, 但它不再是可执行计划。
WITHDRAWN_PLAN_IDS = {
    'operator-v08-increment-r2-attention-v1':
        'primary case S=1024 is structurally degraded (launch floor 17.3% > 10%): every '
        'pair on it is non-decidable by construction. Superseded by '
        'operator-v08-increment-r2-attention-v2 (S=4096, measured valid), withdrawn before '
        'any pair ran',
    'operator-v08-increment-r2-fp32':
        'gate inherited the fp16 probe: command had no --dtype fp32, channel and 0.203% band '
        'were fp16\'s while the fp32 probe measured 0.10%; withdrawn before any pair ran, '
        'superseded by operator-v08-increment-r2-fp32-v2',
}

#: 门控里**不随任务变**的那几项：判据与处置，不是读数。
_GATE_POLICY = dict(
    start_only_if='action == "start"',
    on_hold=('do not start the pair and do not count it as a failure: a held window '
             'consumes no model or verification budget'),
    rerun_every_window=True,
)

_DETECTABILITY_RULE = ('the registered effect must clear the window band by '
                       'WINDOW_DEGRADATION_MARGIN; a pairing below that is not launched')

#: 每个任务**自己那次探针**的实测。command / channel / band / window 都属于那份回执，
#: 不许由另一个精度的任务继承：fp32 计划曾共用 fp16 的这份数据，于是登记的门控量的是
#: fp16（2026-11-07 审查）。band 取回执的 effective_band_pct（= max(nominal, empirical)）。
PROBES = {
    # 回执: output/v08-real-receipt-20261107/preflight-4096.json
    'matmul-4096': dict(
        command=('python -m operator_agent preflight --backend ssh --host <HOST> '
                 '--shape 4096,4096,4096 --reps 5 --repeat 8 --planned-effect-pct 2.5'),
        channel='triton-matmul-4096x4096x4096-cuda-event',
        observed_band_pct=0.203,
        observed_window='2026-11-07 13:36-14:10 CST, h20, 192 cores, load 5.35, 8 GPUs at 0%',
        min_detectable_effect_pct=2.5,
    ),
    # 回执: output/v08-candidates-20261107/gate-fp32.json
    'matmul-4096-fp32': dict(
        command=('python -m operator_agent preflight --backend ssh --host <HOST> '
                 '--shape 4096,4096,4096 --dtype fp32 --reps 5 --repeat 8 '
                 '--planned-effect-pct 2.5'),
        channel='matmul-4096x4096x4096-fp32-cuda-event',
        observed_band_pct=0.1,
        # 这次探针没留主机负载快照，所以不抄 fp16 那行的 cores/load/GPU 空闲数。
        observed_window='2026-11-07 16:36:51 CST (receipt created_at 08:36:51Z), h20, '
                        'SSHExecutor@h20(nvidia); host load not captured for this probe',
        min_detectable_effect_pct=2.5,
    ),
    # 回执: output/v08-attention-20261108/c5/gate-attention-20261108-nocausal.json
    # 同一张卡先量了 --causal（带宽 0.1%, 中位 0.09728 ms）再量这条 nocausal
    # （带宽 0.0998% → 0.1%, 中位 0.096224 ms）: 登记的是 nocausal 主用例，所以只能用这一份。
    'attention-prefill-1024': dict(
        command=('python -m operator_agent preflight --backend ssh --host <HOST> '
                 '--op attention --shape 1,16,1024,128 --dtype fp16 --reps 5 --repeat 8 '
                 '--planned-effect-pct 2.5'),
        channel='attention-1x16x1024x128-fp16-nocausal-cuda-event',
        observed_band_pct=0.1,
        observed_window='2026-11-08 16:42:56 CST (receipt created_at 08:42:56Z), h20 GPU 3 '
                        'shared with 1 other resident process, immediate util 0%; '
                        'median 0.096224 ms reproduces the same-window L3 0.09648 ms (0.3%)',
        min_detectable_effect_pct=2.5,
    ),
    # 回执: output/v08-attention-20261108/c5/gate-attention-4096-20261108.json
    # 探针中位 1.210976 ms 与同窗 L3 的 1.212096 ms 相差 0.09%。
    'attention-prefill-4096': dict(
        command=('python -m operator_agent preflight --backend ssh --host <HOST> '
                 '--op attention --shape 1,16,4096,128 --dtype fp16 --reps 5 --repeat 8 '
                 '--planned-effect-pct 2.5'),
        channel='attention-1x16x4096x128-fp16-nocausal-cuda-event',
        observed_band_pct=0.1,
        observed_window='2026-11-08 17:28:30 CST (receipt created_at 09:28:30Z), h20 GPU 3 '
                        'shared with 1 other resident process; median 1.210976 ms reproduces '
                        'the same-window L3 1.212096 ms (0.09%)',
        min_detectable_effect_pct=2.5,
    ),
}


def _gate(task_id):
    """门控与可判性按任务取值；margin_x 由 band 与 effect 算，不手填。"""
    probe = PROBES[task_id]
    band, effect = probe['observed_band_pct'], probe['min_detectable_effect_pct']
    gate = {name: probe[name] for name in
            ('command', 'channel', 'observed_band_pct', 'observed_window')}
    gate.update(_GATE_POLICY)
    detect = dict(measured_band_pct=band, min_detectable_effect_pct=effect,
                  margin_x=round(effect / band, 2), rule=_DETECTABILITY_RULE)
    return gate, detect


METRIC = dict(
    primary='paired difference in final speedup, only for pairs whose final L3 is valid',
    valid_requires=['full L2 passed', 'measurement_validity == "valid"',
                    'independent final evaluation on the same source hash'],
    secondary=['dispatches used', 'wall clock to first full L2', 'verifications used',
               'agreement between the optimize diagnosis and the measured binding axis'],
    never_report=['a degraded reading as a gain', 'an unpaired average as a comparison',
                  'cost inferred from token counters'],
)

STOP_RULES = dict(
    per_pair=['preflight action != "start" -> hold, do not launch',
              'provider output-cap violation -> stop the pair'],
    per_series=dict(no_valid_pairs_in_a_row=2,
                    on_stop='record the series as unknown; do not launch a third pair',
                    why=('two degraded pairs in a row is a systematic window or wiring '
                         'problem, and a third pair only spends money')),
    budget='stop at the frozen budget; never raise it mid-series to reach a threshold',
)

FINAL_EVALUATION = dict(independent=True, same_source_hash=True, reserve_per_pair=1,
                        note='one reserved independent evaluation per pair, never spent on exploration')

BOUNDARIES = [
    'both arms run the same plugin; only the optimize phase differs, so this compares the increment, not the workflow',
    'sampling_comparability is not established (see operator_agent usage-reconcile): report paired observations, not absolute gains',
    'the A arm has no equivalent model-call hard gate, so no A/B workflow claim is made from this series',
    'a task that never reaches a valid final L3 leaves the series unknown; that is a result about the window, not about the candidate',
]

BUDGET_FIELDS = ('pairs', 'max_model_calls', 'max_output_tokens_per_call',
                 'total_output_token_reservations', 'max_verifications', 'wall_clock_s',
                 'gpu_minutes')


def design(task_id='matmul-4096'):
    """The freezable part: no budget, no outcome, no credit."""
    tasks = [json.loads(json.dumps(row)) for row in TASKS if row['task_id'] == task_id]
    if not tasks:
        raise ValueError(f'unknown task {task_id!r}; known: {sorted(PLAN_IDS)}')
    gate, detect = _gate(task_id)
    body = dict(schema=SCHEMA, plan_id=PLAN_IDS[task_id], status='draft_awaiting_budget',
                tasks=tasks,
                measurement_gate=gate, detectability=detect,
                metric=dict(METRIC), stop_rules=json.loads(json.dumps(STOP_RULES)),
                final_evaluation=dict(FINAL_EVALUATION), boundaries=list(BOUNDARIES),
                budget={name: None for name in BUDGET_FIELDS},
                window_degradation_margin=WINDOW_DEGRADATION_MARGIN,
                # 这三项是"不许写进去的东西": 收益主张、正式对照信用、已完成的结果。
                formal_gain_claim=False, formal_comparison_credit=False, outcome=None)
    body['design_hash'] = digest(body)
    return body


def _positive(value):
    return type(value) in (int, float) and value == value and value > 0


def _command_dtype(command):
    """门控命令实际探的 dtype。`--dtype` 缺省时 preflight 探 fp16
    （operator_agent/app/cli.py），所以"没写"不等于"跟着任务走"。"""
    match = re.search(r'--dtype(?:\s+|=)([A-Za-z0-9_]+)', command)
    return match.group(1) if match else 'fp16'


def _command_op(command):
    """门控命令实际探的算子。`--op` 缺省即 matmul（operator_agent/app/cli.py），
    所以"没写"不等于"跟着任务走" —— 与 dtype 同一条纪律。"""
    match = re.search(r'--op(?:\s+|=)([A-Za-z0-9_]+)', command)
    return (match.group(1) if match else 'matmul').lower()


def _command_shape(command):
    """门控命令探的尺寸；没写就返回 None（等于没量目标尺寸，由调用方判）。"""
    match = re.search(r'--shape(?:\s+|=)([0-9,x]+)', command)
    if not match:
        return None
    return [int(part) for part in match.group(1).replace('x', ',').split(',') if part]


def _declared_kernel_type(declaration):
    """声明 → 算子的**同一个**解析器（不在这里另立一份映射）。"""
    from operator_agent.app.hitl_tools import build_task
    return (build_task(json.loads(json.dumps(declaration))).source.kernel_type or '').lower()


def validate(plan):
    """Return a list of problems; empty means the plan may be frozen/executed."""
    problems = []
    if not isinstance(plan, dict) or plan.get('schema') != SCHEMA:
        return ['not an R2 preregistration']
    if plan.get('plan_id') in WITHDRAWN_PLAN_IDS:
        problems.append(f'plan_id {plan["plan_id"]} was withdrawn: '
                        f'{WITHDRAWN_PLAN_IDS[plan["plan_id"]]}')
    elif plan.get('plan_id') not in set(PLAN_IDS.values()):
        problems.append(f'plan_id must be one of {sorted(set(PLAN_IDS.values()))}')
    body = {key: value for key, value in plan.items() if key != 'hash'}
    if plan.get('hash') is not None and digest(body) != plan.get('hash'):
        problems.append('plan hash does not match its content')
    if plan.get('formal_gain_claim') is not False or plan.get('formal_comparison_credit') is not False:
        problems.append('a preregistration cannot claim a gain or comparison credit')
    if plan.get('outcome') is not None:
        problems.append('a preregistration cannot carry an outcome')
    gate, detect = plan.get('measurement_gate') or {}, plan.get('detectability') or {}
    band = gate.get('observed_band_pct')
    effect = detect.get('min_detectable_effect_pct')
    if not _positive(band) or not _positive(effect):
        problems.append('the measured band and the registered effect must both be positive')
    else:
        required = float(band) * float(plan.get('window_degradation_margin') or 0)
        if float(effect) < required:
            problems.append(f'registered effect {effect}% does not clear {required:.3f}% '
                            f'({plan.get("window_degradation_margin")}x the {band}% band)')
    # 同一个带宽存了两份；重新冻结时只改一份就会让计划自相矛盾。
    if detect.get('measured_band_pct') != band:
        problems.append('detectability.measured_band_pct must equal '
                        'measurement_gate.observed_band_pct')
    series = (plan.get('stop_rules') or {}).get('per_series') or {}
    if series.get('no_valid_pairs_in_a_row') != 2:
        problems.append('the series stop rule must be "two consecutive pairs without a valid L3"')
    budget = plan.get('budget') or {}
    missing = [name for name in BUDGET_FIELDS
               if not (_positive(budget.get(name)) or budget.get(name) is None) ]
    if missing:
        problems.append(f'invalid budget fields: {missing}')
    unfilled = [name for name in BUDGET_FIELDS if budget.get(name) is None]
    if plan.get('status') == 'frozen' and unfilled:
        problems.append(f'a frozen plan needs every budget filled, missing: {unfilled}')
    tasks = plan.get('tasks')
    if not isinstance(tasks, (list, tuple)) or not tasks:
        problems.append('at least one task must be registered')
        tasks = ()
    command = gate.get('command')
    if not isinstance(command, str) or not command.strip():
        problems.append('the gate must declare the command that reproduces its reading')
    else:
        # 门控必须量**登记的那个任务**。fp32 计划曾登记 fp32 任务却配一条探 fp16 的
        # 命令（--dtype 缺省即 fp16），校验器当时放行了它。
        probed = _command_dtype(command)
        for task in tasks:
            declaration = task.get('declaration') if isinstance(task, dict) else None
            precision = declaration.get('precision') if isinstance(declaration, dict) else None
            if precision and probed != precision:
                problems.append(f'task {task.get("task_id")!r} declares precision={precision} '
                                f'but the gate command probes {probed}; the gate must measure '
                                'the registered task')
        # 与 dtype 同一条纪律，扩到算子、尺寸与 mask：一个从别的任务继承来的门控，
        # 量的是**另一个算子**的窗口，而它的读数看起来完全正常。
        probed_op = _command_op(command)
        probed_shape = _command_shape(command)
        channel = gate.get('channel') or ''
        for task in tasks:
            if not isinstance(task, dict):
                continue
            task_id = task.get('task_id')
            declaration = task.get('declaration') or {}
            try:
                kernel_type = _declared_kernel_type(declaration)
            except Exception as error:            # 声明本身不能建任务 ⇒ 计划不可冻结
                problems.append(f'task {task_id!r} declaration does not build a task: {error}')
                continue
            if kernel_type and probed_op != kernel_type:
                problems.append(f'task {task_id!r} is {kernel_type!r} but the gate command '
                                f'probes {probed_op!r}; the gate must measure the registered task')
            shapes = [[int(value) for value in shape] for shape in (declaration.get('shapes') or [])]
            if probed_shape is not None and shapes and probed_shape not in shapes:
                problems.append(f'task {task_id!r} declares {shapes} but the gate command '
                                f'probes {probed_shape}')
            mask = task.get('mask')
            if mask is not None:
                if mask not in ('causal', 'nocausal'):
                    problems.append(f'task {task_id!r} declares an unknown mask {mask!r}')
                elif bool(re.search(r'--causal\b', command)) != (mask == 'causal'):
                    problems.append(f'task {task_id!r} registers mask={mask} but the gate '
                                    'command disagrees about --causal')
                elif f'-{mask}-' not in channel:
                    problems.append(f'the gate channel {channel!r} does not name mask={mask}')
    return problems


def freeze(budget, output, task_id='matmul-4096'):
    """Write the plan once, with budgets filled. Refuses to overwrite or to guess."""
    plan = design(task_id)
    plan['budget'] = {name: (budget or {}).get(name) for name in BUDGET_FIELDS}
    plan['status'] = 'frozen'
    problems = validate(plan)
    if problems:
        raise ValueError('; '.join(problems))
    plan['hash'] = digest({key: value for key, value in plan.items() if key != 'hash'})
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(plan, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return plan