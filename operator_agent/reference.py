"""可外挂参考实现 (pluggable reference)

为什么要有这一层: `verifier._build_reference_fn` 与 `remote_scripts/run_kernel.py`
的 `_reference_torch` 原先都按 `kernel_type` 字符串**写死** matmul/softmax/layernorm…
新架构算子 (DeepSeek-V4 的 CSA 压缩、HCA、MLA 吸收式注意力等) 没有库对手,
参考实现只能由任务侧带来。没有可插拔参考时 L2 会直接放行「无参考仅验证可执行性」,
L3 会「无可用基线」—— 也就是说**新算子整条闭环都不存在**。

参考源码的契约 (跟候选 kernel 一样是可传输的字符串):

    def reference(**inputs):
        \"\"\"inputs 是执行侧张量/标量; 返回单个张量\"\"\"
        ...
        return out

为什么用 `**inputs` 而不是固定参数名: 接口契约 (doc 41) 要求候选 kernel 的参数名
逐一等于输入名, 参考用同一约定才能被同一套 `kernel(**inputs)` 调用惯例共用。
"""
from __future__ import annotations

from typing import Callable, Optional

REFERENCE_FN_NAME = "reference"


class ReferenceSourceError(ValueError):
    """参考源码不符合契约 (没定义 reference / 返回不可调用)"""


def load_reference_source(source: str) -> Callable:
    """把一段参考源码编译成可调用的 `reference(**inputs)`。

    ⚠️ 这里 `exec` 与候选 kernel 的执行是**同一信任边界** (远端 `run_kernel.py`
    本来就会 exec 候选代码)。不要在控制端对不可信来源调用它。
    """
    if not source or not source.strip():
        raise ReferenceSourceError("reference_source 为空")
    ns: dict = {}
    try:
        exec(source, ns, ns)  # noqa: S102 — 与候选 kernel 同信任边界, 见模块 docstring
    except Exception as exc:  # noqa: BLE001
        raise ReferenceSourceError(f"reference_source 编译失败: {type(exc).__name__}: {exc}") from exc
    fn = ns.get(REFERENCE_FN_NAME)
    if not callable(fn):
        raise ReferenceSourceError(
            f"reference_source 必须定义 `def {REFERENCE_FN_NAME}(**inputs)`, "
            f"实际顶层绑定: {sorted(k for k in ns if not k.startswith('__'))}")
    return fn


def resolve_reference(task) -> Optional[Callable]:
    """从 TaskSchema 取外挂参考; 没有就返回 None (调用方回落到按 kernel_type 的内建参考)。

    判定只看 `source.reference_source` 是否非空 —— 空字符串与 None 同义,
    都是「这次没有外挂参考」, 不是「有但加载失败」。
    """
    src = getattr(getattr(task, "source", None), "reference_source", None)
    if not (src and str(src).strip()):
        return None
    return load_reference_source(str(src))
