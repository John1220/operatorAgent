"""规则模板生成器 - Coder 全部失败时的最后回退
单一职责: 按算子类型 + DSL 返回保底可运行代码; 无覆盖算子返回空串

模板正文**不在这里** —— 它们有第二个读者（语料种子与标定量具），单一真源在
`kernel_templates.py`。这里只留「按算子族 + DSL 选哪一份」这件事，不再持有文本。
"""

from .kernel_templates import (
    MATMUL_NUMPY as _MATMUL_NUMPY,
    MATMUL_TRITON as _MATMUL_TRITON,
    SOFTMAX_NUMPY as _SOFTMAX_NUMPY,
    SOFTMAX_TRITON as _SOFTMAX_TRITON,
)
from .schemas import TaskSchema, UnifiedSketch


class RuleTemplateCoder:
    """规则模板回退: 按算子族 + DSL 返回保底实现"""

    def generate(self, task: TaskSchema, sketch: UnifiedSketch = None) -> str:
        kt = (task.source.kernel_type or "").lower()
        dsl = (task.target.dsl or "").lower()
        is_triton = dsl == "triton"

        if "matmul" in kt or "gemm" in kt:
            return _MATMUL_TRITON if is_triton else _MATMUL_NUMPY
        if "softmax" in kt:
            return _SOFTMAX_TRITON if is_triton else _SOFTMAX_NUMPY
        # 无覆盖算子: 返回空, 上层决定后续行为
        return ""
