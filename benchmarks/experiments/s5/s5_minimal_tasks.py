"""Prospectively declared small tasks, not a model-training-unseen claim."""
from operator_agent.run_contract import digest

SPECS = (
 ('study-matmul-fp16-aligned','matmul',(96,128,64),'fp16'),
 ('study-matmul-fp32-tail','matmul',(37,53,23),'fp32'),
 ('study-layernorm-aligned','layernorm',(12,128),'fp32'),
 ('study-layernorm-tail','layernorm',(11,79),'fp32'),
 ('study-softmax-aligned','softmax',(10,128),'fp32'),
 ('study-softmax-tail','softmax',(13,83),'fp32'),
)

def manifest():
    rows=[dict(case_id=name,kernel_type=op,shape=list(shape),precision=dtype,
               layout='contiguous',exposure='prospective_project_task; model training exposure unknown')
          for name,op,shape,dtype in SPECS]
    body=dict(schema='s5-minimal-tasks-v1',tasks=rows,
        scope='outer source generation, supervised L2 feedback, independent final L2',
        evaluation='unchanged production builtin suite; no secret historical holdout claim',
        selection='declared before study outcomes; disjoint primary specs from recorded draft14')
    return dict(body,hash=digest(body))
