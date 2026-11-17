"""A/C workflow round: does DSH + the plugin deliver more independently verified kernels?

Why this is not `r2_plan`
-------------------------
The increment round (B/C) asks "does the optimize diagnosis buy speed", so its plan is
one task per plan id, its primary number is a paired speedup difference, and a plan is
only freezable when the registered effect clears ten times the measured window band.
None of those three fit the workflow question:

* the primary number here is a **count of tasks that deliver full L2**, so an arm that
  fails L2 is the signal, not a broken input (`r2_compare.arm_summary` raises on it);
* the minimum detectable unit is **+1 task across the round**, which is a property of
  the round and not of any single task, so one plan covers all cells;
* the primary needs no L3 at all, so a noise band cannot gate it. A window probe is
  required only for the secondary performance observation.

Nothing here is a claim about an outcome. `freeze()` refuses to write while any budget
is `None`, and both credit fields are pinned to `False`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from operator_agent.core.run_contract import digest

SCHEMA = 'v08-workflow-ac-preregistration-v1'

#: 计划编号：**冻结后不许原位改**。预算写错就新开编号、按名字撤回旧的。
PLAN_IDS = dict(v1='operator-v08-workflow-ac-v1',
                v2='operator-v08-workflow-ac-v2',
                v3='operator-v08-workflow-ac-v3',
                v4='operator-v08-workflow-ac-v4',
                v5='operator-v08-workflow-ac-v5',
                v6='operator-v08-workflow-ac-v6',
                v7='operator-v08-workflow-ac-v7',
                v8='operator-v08-workflow-ac-v8')
PLAN_ID = PLAN_IDS['v2']          # 历史默认；融合轮显式用 FUSED_PLAN_VERSION
FUSED_PLAN_VERSION = 'v8'          # 融合 A/C 轮：qualified delivery，2 对（27B/medium）

#: 撤回必须**指名**，并写清理由 —— 不能落进"编号不认识"那条通用消息里。
WITHDRAWN_PLAN_IDS = {
    'operator-v08-workflow-ac-v7':
        'sampling identity xhigh cannot submit inside the 32768 wire cap (2026-11-17: '
        'internal Coder 2/2 and a real A-arm cell 1/1 spent all 32768 output tokens in '
        'reasoning with zero tool calls and zero submissions). Superseded by '
        'operator-v08-workflow-ac-v8 at reasoning_effort=medium, decided by the user.',
    'operator-v08-workflow-ac-v5':
        'per-call output reservation 65536 is refused by the gateway guard '
        '(`s5_gateway_policy` rejects snapshot.max_output_tokens > 40960 with '
        'PolicyRefused -> HTTP 429), so no cell could start. The prompt fix that '
        'accompanies it (do not dictate the source inside reasoning) is carried forward '
        'in v6 at the original 40960 reservation. Superseded by '
        'operator-v08-workflow-ac-v6',
    'operator-v08-workflow-ac-v4':
        'output cap raised from 40960 to 65536 per call (2026-11-10) after observing the '
        'weak host dictating a whole kernel inside its reasoning and hitting the 32768 '
        'wire cap (attention/A, 654 s, turn_end=max-tokens). Superseded by '
        'operator-v08-workflow-ac-v5',
    'operator-v08-workflow-ac-v3':
        'rounds reduced from 3 to 2 by the user (2026-11-10) before any complete round '
        'was recorded: machine time is the constraint and two rounds still test '
        'reproduction under the strict rule (every round >= +1). Superseded by '
        'operator-v08-workflow-ac-v4',
    'operator-v08-workflow-ac-v1':
        'budget inherited the plugin host-budget defaults (8192 output tokens per call, '
        'total 245760) instead of the arm harness wire policy (max_tokens 32768, '
        'reservation 40960 per call). That truncated every model step at 8192 and one C-arm '
        'step died with reason=max-tokens. Withdrawn before a valid pair ran, superseded by '
        'operator-v08-workflow-ac-v2',
}

#: 臂名固定。A/C 与 B/C 是**两个不同的对照**，共用臂名会让回执被读成另一轮。
ARMS = ('A', 'C')

ARM_SURFACES = dict(
    A=dict(entry='dsh_native',
           plugin=False,
           ordinary_tools=True,
           verification='the same supervisor-owned oracle, reached through the public check.py client',
           note=('ordinary shell/edit/execute tools stay ENABLED. The archived B/C ablation '
                 'disabled tool-bash/tool-fs on both sides, which is right for an on/off '
                 'ablation and wrong for a bare-DSH arm')),
    C=dict(entry='dsh_plugin_host',
           plugin=True,
           plugin_version='v0.8',
           ordinary_tools=False,
           verification='operator_verify on the same frozen contract, one reserve kept for final',
           note=('the prompt prescribes the plugin workflow (prepare, submit the frozen seed '
                 'unchanged, repair/optimize by phase, reserve one final). Per the 2026-11-08 '
                 'decision that prescription is PART OF THE TREATMENT, so a gain is attributed '
                 'to "the plugin and its documented workflow", never to a single context field')),
)

#: `held_out` 说的是**相对于 v0.8 优化规则开发**的位置，不是"这个算子新"。
#: 实测约束：v0.8 诊断的三份真机回执是 matmul/relu/layernorm，attention 走完了 C5 与
#: optimize 闭环，所以 OPERATORS 里只有 softmax 整个族没参与过规则开发。第二个留出槽
#: 只能是**参与过的族里没被用过的配置**，必须如实标注，不能写成"未参与规则开发"。
TASKS = [
    dict(task_id='matmul-4096-fp32',
         declaration=dict(request='fp32 4096x4096x4096 matmul, triton',
                          shapes=[[4096, 4096, 4096]], precision='fp32', dsl='triton',
                          hardware='nvidia_gpu', min_speedup=1.0),
         held_out='no',
         why=('the calibrated anchor: frozen seed e52f637c..., IEEE fp32 baseline identity '
              'fixed on 2026-11-08, gate band 0.10%, naive triton at 0.3545x of baseline'),
         performance_probe=dict(required=True, measured_band_pct=0.1,
                                receipt='output/v08-candidates-20261107/gate-fp32.json')),
    dict(task_id='attention-prefill-4096',
         declaration=dict(request='fp16 attention prefill 1x16x4096x128',
                          shapes=[[1, 16, 4096, 128]], precision='fp16', dsl='triton',
                          hardware='nvidia_gpu', min_speedup=1.0),
         mask='nocausal',
         held_out='no',
         why=('the second anchor and the widest structural gap: v0.7.2 cannot even prepare '
              'attention, and a bare arm gets no reference, no 8-case suite and no Scope A '
              'geometry gate. S=4096 measured a VALID L3 (0.8608x, launch dominance 1.64%)'),
         performance_probe=dict(required=True, measured_band_pct=0.1,
                                receipt='output/v08-attention-20261108/c5/gate-attention-4096-20261108.json')),
    dict(task_id='softmax-16384x6144-fp16',
         declaration=dict(request='fp16 softmax 16384x6144', shapes=[[16384, 6144]],
                          precision='fp16', dsl='triton', hardware='nvidia_gpu',
                          min_speedup=1.0),
         held_out='family',
         why=('the only family in OPERATORS that no v0.8 optimize rule was developed against; '
              '192 MiB per fp16 tensor stays under the 256 MiB input cap'),
         performance_probe=dict(required=True, measured_band_pct=None,
                                note=('no probe exists for this shape. The primary metric does '
                                      'not need one; run preflight before the round and record '
                                      'the band, or this task contributes no performance '
                                      'observation. v0.7.2 softmax L3 was uninformative'))),
    dict(task_id='matmul-2048x16384x2048-fp16',
         declaration=dict(request='fp16 2048x16384x2048 matmul, triton',
                          shapes=[[2048, 16384, 2048]], precision='fp16', dsl='triton',
                          hardware='nvidia_gpu', min_speedup=1.0),
         held_out='configuration',
         why=('a tall-K configuration measured valid (1.0311x, 93.7% attainment) in the '
              '2026-11-07 candidate sweep but never used to develop a rule. The matmul '
              'FAMILY did participate, so this is a held-out configuration only'),
         performance_probe=dict(required=True, measured_band_pct=None,
                                note=('the sweep measured L3 but not a window band for this '
                                      'shape; the 0.203% band belongs to 4096 cubed and must '
                                      'not be inherited'))),
]

TASK_IDS = tuple(row['task_id'] for row in TASKS)

#: v8：融合任务 + 27B/medium。v7 原本登记 xhigh，但 2026-11-17 smoke 证实 xhigh 在 wire
#: 32768 下结构性不可用（内部 Coder 2/2、真实 A 臂 1/1 都把 32768 全写在思考里、零提交），
#: 用户据此决定降 medium ⇒ 新计划编号 v8，v7 不执行。
#: 融合任务。头寸门已过（手写核 2.229× eager），本轮问工作流能否**达标交付**。
#: 达标 = 完整 L2 + 独立 L3 valid + speedup >= 1.5 + 单核融合静态核验（见 fusion_check）。
FUSED_TASK = dict(
    task_id='fused-layernorm-residual-32768x6144-fp16',
    declaration=dict(request='fp16 32768x6144 residual add layernorm',
                     shapes=[[32768, 6144]], precision='fp16', dsl='triton',
                     hardware='nvidia_gpu', min_speedup=1.5),
    held_out='family',
    why=('newly registered residual+LayerNorm fusion family; no v0.8 optimize rule was '
         'developed against it, and the headroom gate confirmed a hand-written fused kernel '
         'runs at 2.229x the eager two-kernel baseline'),
    performance_probe=dict(required=True, measured_band_pct=None,
                           note=('window probe is fused_layernorm_residual on 32768x6144; '
                                 'measure it fresh for the round, do not inherit the headroom '
                                 'window')),
    acceptance=dict(full_l2=True, l3_validity='valid', min_speedup=1.5,
                    fusion_check='v08-fusion-static-v1'),
)

FUSED_METRIC = dict(
    primary=('number of cells whose final source is a QUALIFIED DELIVERY: full L2 in the '
             'independent final evaluation, final L3 measurement_validity=valid, '
             'speedup_vs_baseline >= 1.5, and the single-kernel fusion static check passes'),
    denominator='every dispatched cell; planned-but-unexecuted cells are listed separately, never dropped',
    minimum_detectable_unit='+1 task',
    minimum_detectable_unit_is_not_statistical_significance=True,
    secondary=['delivery that passes full L2 but misses the performance/fusion bar (correct_only)',
               'model dispatches used', 'verifications used',
               'wall clock to first full L2 and to the qualified final receipt'],
    never_report=['a correct-only delivery as a qualified delivery',
                  'an unpaired average as a comparison',
                  'cost inferred from token counters',
                  'a pass rate without its dispatched denominator'],
)

FUSED_EXECUTION = dict(
    gpu=dict(index=7, uuid='GPU-52ca6155-d8bc-a416-847a-6f0c71db1552', name='NVIDIA H20'),
    model='Qwen3.8-27B',
    model_endpoint_sha256=('67df113c7123ab1a97394b809da88efe35489ab43b6657850846c180f54d23d1'),
    sampling=dict(temperature=0.1, n=1, max_tokens=32768,
                  enable_thinking=True, reasoning_effort='medium'),
    wire_max_tokens=32768,
    output_reservation_per_call=40960,
    pairs=2,
    final_policy=dict(purpose='final', stop_after='L3',
                      final_run_verifications=2),
)

METRIC = dict(
    primary=('number of tasks whose submitted final source passes full L2 in the '
             'independent final evaluation, inside the frozen per-cell budget'),
    denominator='every dispatched cell; planned-but-unexecuted cells are listed separately, never dropped',
    minimum_detectable_unit='+1 task',
    minimum_detectable_unit_is_not_statistical_significance=True,
    secondary=['paired final speedup, only when both arms delivered full L2 AND both final '
               'L3 are valid AND both contract hashes match',
               'wall clock to first full L2', 'model dispatches used', 'verifications used'],
    never_report=['a degraded or uninformative reading as a gain',
                  'an unpaired average as a comparison',
                  'cost inferred from token counters',
                  'a pass rate without its dispatched denominator',
                  'the 8-case attention suite as "8 operators" or as 100% correctness'],
)

#: 每格预算。名字用插件 host_budget 的那一套；`arm_gate_budget()` 给出 A 臂硬门的映射，
#: 因为 ArmGate 读 max_calls / total_output_tokens_reservations，两套名字不能互相顶替。
BUDGET_FIELDS = ('cells', 'max_model_calls', 'max_output_tokens_per_call',
                 'total_output_token_reservations', 'max_verifications', 'wall_clock_s',
                 'final_reserve', 'gpu_minutes')

#: CLI 缺省是 --max-llm-calls 24 / --wall-clock-s 1800，比这里松紧不一致；
#: 每格都必须显式覆盖，否则"同预算"是假的。
#: 每格预算。**输出 token 用臂 harness 自己的口径，不是插件 host budget 的默认值**：
#: 线上 `sampling.POLICY.max_tokens=32768`，账本每次预留 40960（比线上上限略高，
#: 与 `phase1/connected.py:86` 同源），总预留 = 调用数 × 40960。
#: 2026-11-09 撤回的 v1 抄了插件的 8192/245760，把每一步的输出截在 8192。
WIRE_MAX_TOKENS = 32768
OUTPUT_RESERVATION_PER_CALL = 40960
FROZEN_PER_CELL = dict(max_model_calls=30,
                       max_output_tokens_per_call=OUTPUT_RESERVATION_PER_CALL,
                       total_output_token_reservations=30 * OUTPUT_RESERVATION_PER_CALL,
                       max_verifications=8, wall_clock_s=2400, final_reserve=1)

#: v7 的每格预算：20 次调用、8 次验证（6 开发 + 2 终评）、2400 s 含终评。
FROZEN_FUSED_PER_CELL = dict(max_model_calls=20,
                             max_output_tokens_per_call=OUTPUT_RESERVATION_PER_CALL,
                             total_output_token_reservations=20 * OUTPUT_RESERVATION_PER_CALL,
                             max_verifications=8, wall_clock_s=2400, final_reserve=1,
                             gpu_minutes=240)

STOP_RULES = dict(
    per_cell=['stop at the frozen budget; never raise it mid-round to reach a threshold',
              'a cell that never passes full L2 is recorded as FAILED, not unknown',
              'unknown is reserved for a broken record chain (missing receipt, drifted hash, '
              'unmetered dispatch), which is a wiring result and not a candidate result'],
    per_round=dict(on_null='report the null with its denominator and stop; do not add cells, '
                           'loosen tolerance or swap tasks to reach a positive',
                   replication='re-run the pre-selected key pair after the round and report '
                               'whether the effect reproduces'),
)

FINAL_EVALUATION = dict(
    independent=True, same_evaluator_version_for_both_arms=True, same_source_hash=True,
    reserve_per_cell=1,
    note=('A has no plugin, so its final receipt is produced by the supervisor: prepare a host '
          'task on A\'s submitted source, then verify with verification_purpose=final. '
          'hitl_tools.verify refuses a non-explore purpose without a prepared host task, and '
          'the CLI exposes no --verification-purpose, so this step needs its own driver'),
)

BOUNDARIES = [
    'this compares the WORKFLOW (bare DSH vs DSH + plugin), not the v0.8 increment; it is a new '
    'plan id and must not be filed under operator-v08-increment-r2-*',
    'the C prompt prescribes the plugin workflow, so attribution is to the plugin plus its '
    'documented workflow (decided 2026-11-08), not to any single context field',
    'sampling_comparability is not_established: contract 12 -> 13 was deliberately deferred '
    '(decided 2026-11-08), so this round reports paired OBSERVATIONS and no formal gain',
    'both arms reach the same oracle on the same frozen contract and the same per-cell budget; '
    'A is not given a weaker correctness check',
    'only softmax is a held-out FAMILY; the tall-K matmul is a held-out CONFIGURATION inside a '
    'family that participated in rule development',
    'this is not a reopening of the S5 24-cell series, which closed as insufficient evidence '
    'on 2026-10-30; it is a new round with its own identity',
]



#: v3：**复现计划**。只跑"弱宿主输掉的那三格"，用它判定 §2w–§3l 的 +3 是不是稳定信号。
#: 规则由用户在 2026-11-10 选定：①A 严格（每轮都要 ≥ +1，任一轮掉 0 就 null 且不追加）、
#: ②A 只复现这三格、③A 新增聚合器而不改 `ac_compare` 的单轮语义。
REPRODUCTION_TASKS = ('matmul-4096-fp32', 'attention-prefill-4096',
                      'softmax-16384x6144-fp16')

REPRODUCTION_ROUNDS = dict(v3=3, v4=2, v5=2, v6=2)

REPRODUCTION = dict(
    rounds=3,   # design() 会按版本覆盖；这里只是缺省
    rule='every_round_at_least_one_task',
    minimum_detectable_unit='+1 task',
    stop_on_null=True,
    subject=('weak host (Qwen3.8-27B, reasoning_effort=xhigh) on tasks that one '
             'generation can deliver'),
    tasks=list(REPRODUCTION_TASKS),
    note=('任一轮的配对差 < +1 ⇒ 整条判 null，且**不追加轮次**；'
          'formal_gain_claim 只有在每一轮都成立、且每轮 sampling_comparability=established '
          '时才为 true。范围永久限定在 subject 里写的那一类宿主与任务。'),
)

def arm_gate_budget(per_cell):
    """Host budget names -> the names `ArmGate` actually enforces."""
    return dict(max_calls=per_cell['max_model_calls'],
                max_output_tokens_per_call=per_cell['max_output_tokens_per_call'],
                total_output_tokens_reservations=per_cell['total_output_token_reservations'],
                wall_clock_s=per_cell['wall_clock_s'])


def _positive(value):
    return type(value) in (int, float) and value == value and value > 0


def _kernel_type(declaration):
    from operator_agent.app.hitl_tools import build_task
    return (build_task(json.loads(json.dumps(declaration))).source.kernel_type or '').lower()


def _supported_kernel_types():
    """从权威词表取，不再抄一份。OPERATORS 是 axis 生成器的词表，attention 单独注册
    （`docs/testing/v08-support.md` §1 就是按这两种身份区分它们的）。"""
    from operator_agent.decide.correctness_profiles import (OPERATORS,
                                                           REGISTERED_FUSED_KERNEL_TYPES)
    return set(OPERATORS) | {'attention'} | set(REGISTERED_FUSED_KERNEL_TYPES)


#: 融合计划的两个编号：v7（xhigh，未执行）与 v8（medium，当前可执行）。
FUSED_VERSIONS = ('v7', 'v8')


def design(version='v2'):
    """The freezable part: no budget, no outcome, no credit."""
    fused = version in FUSED_VERSIONS
    body = dict(schema=SCHEMA, plan_id=PLAN_IDS[version], status='draft_awaiting_budget',
                arms=list(ARMS), arm_surfaces=json.loads(json.dumps(ARM_SURFACES)),
                tasks=json.loads(json.dumps([FUSED_TASK] if fused else (
                    [task for task in TASKS if task['task_id'] in REPRODUCTION_TASKS]
                    if version in REPRODUCTION_ROUNDS else TASKS))),
                **({'reproduction': dict(json.loads(json.dumps(REPRODUCTION)),
                                         rounds=REPRODUCTION_ROUNDS[version])}
                   if version in REPRODUCTION_ROUNDS else {}),
                metric=dict(FUSED_METRIC if fused else METRIC),
                stop_rules=json.loads(json.dumps(STOP_RULES)),
                final_evaluation=dict(FINAL_EVALUATION),
                boundaries=list(BOUNDARIES) + ([
                    'v7 is the fused task only; its primary is qualified delivery, not the v6 '
                    'full-L2 delivery count, and its result must not be merged into v6 rounds',
                    'the hand-written headroom kernel is context, never an arm delivery',
                ] if fused else []),
                budget={name: None for name in BUDGET_FIELDS},
                formal_gain_claim=False, formal_comparison_credit=False, outcome=None)
    if fused:
        execution = json.loads(json.dumps(FUSED_EXECUTION))
        if version == 'v7':
            # v7 登记的是 xhigh；它的存在价值是记录"为什么没执行"，不是可执行计划。
            execution['sampling']['reasoning_effort'] = 'xhigh'
        body['execution'] = execution
        body['pairs'] = FUSED_EXECUTION['pairs']
    body['design_hash'] = digest(body)
    return body


def validate(plan):
    """Return a list of problems; empty means the plan may be frozen/executed."""
    if not isinstance(plan, dict) or plan.get('schema') != SCHEMA:
        return ['not an A/C workflow preregistration']
    problems = []
    # 复现类计划（v3/v4/v5…）都要求登记规则与轮数；新增版本不用再改这里。
    if plan.get('plan_id') in {PLAN_IDS[name] for name in REPRODUCTION_ROUNDS}:
        reproduction = plan.get('reproduction') or {}
        if reproduction.get('rule') not in ('every_round_at_least_one_task',):
            problems.append('复现规则未登记或不受支持（reproduction.rule）')
        if type(reproduction.get('rounds')) is not int or reproduction['rounds'] < 2:
            problems.append('至少要两轮才谈得上复现（reproduction.rounds）')
        want = set(reproduction.get('tasks') or ())
        have = {task.get('task_id') for task in plan.get('tasks') or ()}
        if want != have or not want:
            problems.append('tasks 必须正好是 reproduction.tasks 那几格')
    if plan.get('plan_id') in WITHDRAWN_PLAN_IDS:
        problems.append(f'plan_id {plan["plan_id"]} was withdrawn: '
                        f'{WITHDRAWN_PLAN_IDS[plan["plan_id"]]}')
    elif plan.get('plan_id') not in set(PLAN_IDS.values()):
        problems.append(f'plan_id must be one of {sorted(PLAN_IDS.values())}')
    body = {key: value for key, value in plan.items() if key != 'hash'}
    if plan.get('hash') is not None and digest(body) != plan.get('hash'):
        problems.append('plan hash does not match its content')
    if plan.get('formal_gain_claim') is not False or plan.get('formal_comparison_credit') is not False:
        problems.append('a preregistration cannot claim a gain or comparison credit')
    if plan.get('outcome') is not None:
        problems.append('a preregistration cannot carry an outcome')
    if tuple(plan.get('arms') or ()) != ARMS:
        problems.append(f'arms must be exactly {list(ARMS)}')
    metric = plan.get('metric') or {}
    if metric.get('minimum_detectable_unit') != '+1 task':
        problems.append('the minimum detectable unit must be "+1 task"')
    if metric.get('minimum_detectable_unit_is_not_statistical_significance') is not True:
        problems.append('the plan must state that +1 task is not statistical significance')
    tasks = plan.get('tasks')
    if not isinstance(tasks, (list, tuple)) or not tasks:
        problems.append('at least one task must be registered')
        tasks = ()
    seen = set()
    for task in tasks:
        if not isinstance(task, dict):
            problems.append('every task must be an object')
            continue
        task_id = task.get('task_id')
        if task_id in seen:
            problems.append(f'duplicate task_id {task_id!r}')
        seen.add(task_id)
        if task.get('held_out') not in ('no', 'family', 'configuration'):
            problems.append(f'task {task_id!r} must label held_out as no/family/configuration')
        declaration = task.get('declaration')
        if not isinstance(declaration, dict):
            problems.append(f'task {task_id!r} has no declaration')
            continue
        try:
            kernel_type = _kernel_type(declaration)
        except Exception as error:
            problems.append(f'task {task_id!r} declaration does not build a task: {error}')
            continue
        # 请求串里的算子名拼错时 build_task 不报错，而是给出 kernel_type='unknown'，
        # 而 OPERATORS 里没有这个族、也没有任何 profile ⇒ 这一格到开跑才会失败。
        if kernel_type not in _supported_kernel_types():
            problems.append(f'task {task_id!r} builds kernel_type {kernel_type!r}, which has '
                            f'no profile; supported: {sorted(_supported_kernel_types())}')
        probe = task.get('performance_probe')
        if not isinstance(probe, dict) or probe.get('required') is not True:
            problems.append(f'task {task_id!r} must say whether a window probe is required '
                            'for the secondary performance observation')
    budget = plan.get('budget') or {}
    invalid = [name for name in BUDGET_FIELDS
               if not (_positive(budget.get(name)) or budget.get(name) is None)]
    if invalid:
        problems.append(f'invalid budget fields: {invalid}')
    unfilled = [name for name in BUDGET_FIELDS if budget.get(name) is None]
    if plan.get('status') == 'frozen':
        if unfilled:
            problems.append(f'a frozen plan needs every budget filled, missing: {unfilled}')
        pairs = plan.get('pairs') or 1
        if type(pairs) is not int or pairs < 1:
            problems.append(f'pairs must be a positive int, got {pairs!r}')
        expected = len(tasks) * len(ARMS) * pairs
        if budget.get('cells') != expected:
            problems.append(f'cells must equal tasks x arms x pairs = {expected}, '
                            f'got {budget.get("cells")}')
    if plan.get('execution') is not None:
        ex = plan['execution']
        gpu = (ex or {}).get('gpu') or {}
        if (type(gpu.get('index')) is not int
                or not isinstance(gpu.get('uuid'), str)
                or not re.fullmatch(r'GPU-[0-9a-f-]{16,}', gpu.get('uuid') or '')
                or gpu.get('name') != 'NVIDIA H20'):
            problems.append('execution.gpu must pin index/uuid/name for NVIDIA H20')
        sampling = (ex or {}).get('sampling') or {}
        if set(sampling) != {'temperature', 'n', 'max_tokens', 'enable_thinking',
                             'reasoning_effort'}:
            problems.append('execution.sampling must carry the six frozen fields')
        if (ex or {}).get('wire_max_tokens') != 32768:
            problems.append('execution.wire_max_tokens must be the 32768 wire cap')
        if (ex or {}).get('output_reservation_per_call') != OUTPUT_RESERVATION_PER_CALL:
            problems.append('execution.output_reservation_per_call must be 40960')
        if (ex or {}).get('pairs') != plan.get('pairs'):
            problems.append('execution.pairs must match plan.pairs')
    return problems


def freeze(budget, output, version='v2'):
    """Write the plan once, with budgets filled. Refuses to overwrite or to guess."""
    plan = design(version)
    supplied = dict(FROZEN_FUSED_PER_CELL if version in FUSED_VERSIONS else FROZEN_PER_CELL)
    supplied.update(budget or {})
    plan['budget'] = {name: supplied.get(name) for name in BUDGET_FIELDS}
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