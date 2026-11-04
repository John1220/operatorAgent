"""Candidate v0.7 support boundary, separate from machine acceptance results."""
from operator_agent.app.hitl_tools import build_task
from operator_agent.core.contract import freeze_from_task

OPERATORS = ('matmul', 'relu', 'softmax', 'layernorm')
PRECISIONS = ('fp16', 'fp32')


def check(declaration, backend):
    task = build_task(declaration)
    if task.source.kernel_type not in OPERATORS or task.constraints.precision.value not in PRECISIONS:
        raise ValueError('v0.7 candidate scope: matmul/relu/softmax/layernorm, fp16/fp32')
    if backend == 'ssh' and task.target.dsl != 'triton':
        raise ValueError('connected GPU scope requires Triton source')
    if backend == 'fake' and task.target.dsl != 'python':
        raise ValueError('fake execution requires CPU Python source')
    if len(task.constraints.shapes) != 1:
        raise ValueError('one declared shape per connected cell is required')
    contract = freeze_from_task(task)
    if task.source.kernel_type == 'layernorm' and any(
            tuple(t.name for t in case.tensors) != ('X',) for case in contract.cases):
        raise ValueError('LayerNorm scope is non-affine, last axis, eps=1e-5')
    return task, contract


def matrix():
    return dict(product_version='v0.7', status='candidate_scope_not_release_acceptance',
        operators=list(OPERATORS), precision=list(PRECISIONS),
        gpu_backend='ssh', gpu_dsl='triton', offline_backend='fake', offline_dsl='python',
        layernorm=dict(affine=False, axis='last', eps=1e-5),
        correctness='builtin complete L2, including declared strided cases',
        gpu_identity='probe and pin index/name/UUID on every rented instance',
        runtime_compatibility='must be recorded and accepted on the actual instance',
        released=False)
