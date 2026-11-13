"""A/C 轮的两份 prompt 与"一格的记录链"（P0-4 ③ 的可验证部分）。

两份 prompt 是这一轮**归因**的载体：todo 里已拍板——插件的工作流规定算处理的一部分，
所以 C 臂照旧手把手（prepare → 原样提交冻结 seed → 按 phase 走 → 预留一次 final），
增益只能归到"插件及其配套工作流"，不能归到某个上下文字段。A 臂则相反：只给普通工具
与一条 `check.py` 提交路径。

`a_cell` 是**不含模型**的那一段：它把 A 臂 prompt 要求的动作序列，用同一个公开客户端
（`check.py` 背后就是 `hitl_tools.verify`）演一遍，证明"提交 → 选择 → 终评 → 落格"这条
记录链成立。DSH 子进程那一跳由 `start_headless` 负责，是这一层之上的薄调用。
"""
from __future__ import annotations

import json
from pathlib import Path

from .ac_driver import CELL_FIELDS, SurfaceError, final_evaluation

#: A 臂：普通工具 + 冻结声明 + `check.py` 提交。不出现任何 operator_* 词汇。
A_PROMPT = (
    'Implement the frozen operator using the ordinary shell, read and edit tools. '
    'Write the source yourself; no operator plugin, core CLI or generation delegation is provided. '
    'Submit each candidate by running `python check.py --code-file candidate.py` from the working '
    'directory. The client performs full verification through a supervisor-owned backend; use its '
    'feedback to revise the source. The final candidate is the LAST source submitted to check.py. '
    'Do not retry after the client reports budget exhaustion. Report the actual last result. '
    'The caller invokes `kernel(**inputs)` with exactly these parameters — do not rename '
    'the entry point, do not add parameters, and let the caller own the output buffer:\n'
    '{abi}\n'
    'Deliver code through the tools: write the source into a file (or pass it as the '
    'tool argument) and submit it. Do not dictate or re-derive the full source inside your '
    'reasoning or reply — that burns the output budget and produces no submission.\n'
    'Numeric contract (authoritative): {numeric}\n'
    'Rules you must obey (authoritative): {policy}\n'
    'Frozen common budget: {budget}. Frozen declaration: {declaration}.'
)

#: C 臂：插件工作流本身是处理的一部分，写死在 prompt 里（归因口径见 todo §1）。
C_PROMPT = (
    '你有一组 operator 工具，按下面的工作流完成这次算子开发，不要绕过它们：\n'
    '1. `operator_prepare` 用**下面这份声明原样**冻结任务，`host_budget` 用**下面这份预算原样**；'
    '不接受任何"随后再补"的预算。\n'
    '2. `operator_context(phase="generate")` 取权威事实与材料，然后**自己**写 kernel 源码。\n'
    '3. 用 `operator_verify` 提交源码；失败就按回包改，成功就进入优化。\n'
    '4. 优化阶段先 `operator_context(phase="optimize", verification_round=<n>)`，按它给的'
    '瓶颈轴与实验维度决定改什么（它是**假设**，不是指令），改完再 `operator_verify`。\n'
    '5. 选定终稿后，用**同一份源码**调一次 `operator_verify(verification_purpose="final")` ——'
    '这一份是独立终评，不能用来探索。\n'
    '提交方式：把源码写成文件（或作为工具参数）再提交；**不要**在思考或正文里整段复述源码——'
    '那会烧光输出预算却交不出东西。\n'
    '调用约定（ABI，权威）：\n{abi}\n数值契约（权威）：{numeric}\n规则（权威）：{policy}\n'
    '冻结预算：{budget}\n冻结声明：{declaration}\n'
    '如实报告最后一次提交的结果；预算用尽就停，不要新增调用去凑一个通过。'
)


def calling_convention(declaration):
    """这份声明的调用约定（ABI）——**与插件同源**，不手写。

    `build_task → freeze_from_task → contract.cases[0].tensors` 取到的就是插件在
    `operator_context.authoritative.interface` 里发布的那份：
    `wrapper_calling_convention(metadata)` 是同一个函数，所以两臂看到的是同一段事实。
    2026-11-08 的 A 臂失败正是缺了它：`L1: Missing 'kernel' function`。
    """
    from operator_agent.app.hitl_tools import build_task
    from operator_agent.core.contract import freeze_from_task
    from operator_agent.generate.generation_context import wrapper_calling_convention

    contract = freeze_from_task(build_task(declaration))
    if not contract.cases:
        raise SurfaceError('declaration derives no case: ABI cannot be established')
    case = contract.cases[0]
    # **标量也要进 ABI**：attention 的 `causal` 是按值传的，调用方会发
    # `kernel(K=..., Q=..., V=..., causal=False)`。只按 tensors 生成签名会得到
    # `def kernel(K=None, Q=None, V=None)`，模型照着写就必然 `TypeError`。
    # 这与插件 `host_context._case_metadata` 的做法一致（那里把 case.scalars 也加进去）。
    rows = {}
    for tensor in case.tensors:
        rows[tensor.name] = dict(shape=list(tensor.shape), dtype=tensor.dtype)
    for name, value in sorted((getattr(case, 'scalars', None) or {}).items()):
        rows[name] = dict(kind='scalar_by_value', py_type=type(value).__name__, value=value)
    return wrapper_calling_convention([rows])


def numeric_contract(declaration):
    """这份声明的**数值契约**：dtype 语义、容差与判据形式（与插件同源）。

    2026-11-09：A 臂两次都写 `tl.dot(a, b, acc)`（triton 对 fp32 默认走 tf32），
    误差 1.16e-3 / 2.73e-1，而契约要求 IEEE fp32 且 `atol=1e-5`。插件会通过
    `operator_context.authoritative.contract` 发布这条事实，A 臂没有任何等价物。
    """
    from operator_agent.app.hitl_tools import build_task
    from operator_agent.core.contract import freeze_from_task

    contract = freeze_from_task(build_task(declaration))
    import dataclasses
    return dict(dtype_semantics=dataclasses.asdict(contract.dtype_semantics),
                max_error=contract.max_error, target_metric=contract.target_metric,
                correctness_profile=contract.correctness_profile,
                comparison='elementwise: a mismatch must exceed BOTH atol and rtol*|reference|')


def arm_prompt(arm, *, declaration, budget):
    """(prompt 文本) —— 预算与声明**原样**插进去，不在这里改写口径。"""
    if arm not in ('A', 'C'):
        raise SurfaceError(f'unknown arm {arm!r}')
    from operator_agent.decide.anticheat import ANTICHEAT_POLICY
    template = A_PROMPT if arm == 'A' else C_PROMPT
    return template.format(policy=ANTICHEAT_POLICY,abi=json.dumps(calling_convention(declaration), ensure_ascii=False,
                                          sort_keys=True, indent=1),
                           numeric=json.dumps(numeric_contract(declaration), ensure_ascii=False,
                                              sort_keys=True),
                           budget=json.dumps(budget, ensure_ascii=False, sort_keys=True),
                           declaration=json.dumps(declaration, ensure_ascii=False, sort_keys=True))


def a_cell(task_id, declaration, code, *, host_budget, backend, run_dir, source_path=None,
           host=None, work_dir=None, stop_after='L2', timeout=300, persistent=False,
           model_dispatches=None, verifications=None, stop_reason=None):
    """A 臂一格的记录链（不含模型）：prepare → 提交（explore）→ final → 落格。

    `code` 是"模型最后提交的那份源码"（文本）；`source_path` 是它落盘的路径 —— 两件事
    必须分开：`ac_compare` 要按**文件**重算哈希，而判据要的是文本。没给路径就写一份
    到 run_dir 旁边。这里不选候选、不改源码：选择规则属于驱动，且必须落在"最后一次提交"上。
    """
    run_dir = Path(run_dir)
    if source_path is None:
        source_path = run_dir.parent / 'submission.py'
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(code, encoding='utf-8')
    result = final_evaluation(run_dir, code, declaration=declaration,
                              host_budget=host_budget, backend=backend,
                              stop_after=stop_after, host=host, work_dir=work_dir,
                              timeout=timeout, persistent=persistent)
    state = 'completed' if result['delivered'] else (
        'budget_exhausted' if stop_reason == 'budget_exhausted' else 'completed')
    body = dict(task_id=task_id, arm='A', state=state,
                final_receipt=result['final'], source=str(source_path),
                model_dispatches=model_dispatches, verifications=verifications,
                stop_reason=stop_reason)
    body.update(outcome='delivered' if result['delivered'] else 'failed',
                reason=result.get('reason'),
                levels=(result['final'] or {}).get('levels')
                if result['final'] else (result['explore'] or {}).get('levels'))
    unknown = set(body) - set(CELL_FIELDS) - {'outcome', 'reason', 'levels'}
    if unknown:
        raise SurfaceError(f'unknown cell fields: {sorted(unknown)}')
    return body