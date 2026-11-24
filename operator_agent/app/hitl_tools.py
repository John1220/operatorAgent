"""HITL 工具壳：把判定层包成「一次一回合」的两个动作

为什么有这一层（`docs/dsh-plugin-evaluation.md` §7.3 / §9.5）：

本仓的 `Conductor` 是**无人值守**的确定性编排器 —— 一次调用跑完分钟~小时级的整段。
而「前期 HITL（人看得见每一步、能干预）」要的是另一种驱动方式：**外层每一回合只做一件事**，
人（或外层的模型）看到判定再决定下一步。

所以这一层只暴露两个动作，且都**只做判定、不做编排**：

    prepare(declaration, run_dir) -> 一份冻结契约 + 契约自带的输入 + 可复算的身份
    verify(run_dir, code, stop_after=...) -> 正交判定 + 处方 + 一份报告

三条设计纪律（都是被前面的实测逼出来的，见 §9.5）：

1. **不经过 Conductor**：只用 `IntentParser` / `freeze_from_task` / `Verifier` /
   `failure_route.classify` / `report.write_report`。`Conductor` 一行都不碰。
2. **处方与资格必须并列**：`failure_route.classify` 不读资格位（实测接缝），
   所以这里算一个 `conflict` 字段 —— 处方说「去搜索/算一轮」而资格说这个数不能驱动
   任何后果时，`action` 改成 `hold_for_measurement`，并给出人话的下一步。
   判定语义**不改**（`classify` 仍按它自己的口径判），改的是**呈现与建议**。
3. **不猜**：没有 `shapes` 就拒绝 prepare（契约拥有输入是这一层的前提）；后端只接线
   `mock`，其余显式失败，不静默降级。

安全面：参数走白名单（§6.3）。`max_error`、任何「跳过/放行/关库」类字段都**不在**
白名单里 —— 一旦出现，`ToolError('forbidden_parameter')`。
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from ..core.backend_caps import Cap
from ..core.contract import freeze_from_task
from ..core.task_declaration import (ALLOWED_DECLARATION_KEYS, DECLARATION_SCHEMA,
                                     DeclarationError, complexity_of, field_state,
                                     hardware_profile, parse_declaration, precision_of)
from ..decide.failure_route import Route, classify
from ..generate.generation_spec import generation_spec_identity
from ..generate.generation_context import candidate_summary
from ..core.run_contract import code_hash
from ..app.intent_parser import IntentParser
from ..core.report import build_report
from ..core.run_layout import write_once_artifact
from ..decide.tool_evidence import RoundEvidence, verification_data
from ..decide import verdict_cache
from ..core.execution_audit import install_level_recorder
from ..core.schemas import (ConstraintSpec, KernelArtifact, SourceSpec, TargetSpec,
                      TaskSchema, VerificationLevel, VerificationResult)
from ..decide.verifier import Verifier
from ..execute.fake_backend.config import tool_context
from ..execute.host_task_budget import FINAL_SELECTION_SCHEMA

#: 工具面的契约版本。**加/删工具或改回包形状就要 +1** —— 外层宿主会把这个数记进
#: 自己的配置与报告，同一份 `tool_version` 必须指同一套语义。
#: 2: 新增 `operator_bench`（探索级批量测量；回包刻意不含裁决字段）。
#: 3: `verify`/`repair` 的回包多了 `candidate_origin`（`repair` 另有
#:    `final_candidate_origin`）—— 候选来源的**归因格**，不进判据。
#: 4: repair 返回统一候选摘要、最终源码哈希，完整记录保存在 evidence。
#: 5: run 使用有限任务预算，status 返回预算用量、配置及停止原因。
#: 6: verify/repair/bench 独立留存原始源码与完整判定，回包引用不可覆盖的 evidence。
#: 7: invocation 绑定的取消入口、取消终态与可中断的 MCP 长调用。
#: 8: 资源预检按明确共享存储证据裁决，L1 保留未知/来源，不再从逻辑 tile 猜物理占用。
#: 9: repair 有限任务预算、独立模型/验证账本和监督截止；新增阶段/usage 审计元数据。
#: 10: 显式 fake 后端、run 场景/模型配置与 test_only 执行上下文。
#: 11: host workflow 的领域材料、跨调用验证预算和预留的独立终评。
#: 12: 宿主模型预算政策、收据与不重置原任务的恢复入口。
#: 13: `operator_prepare` 的 `operator` / `precision` 变成**必填**（判据层不再从
#:     `request` 推关键词）；新增 `operator_normalize`；`request` 降级为审计字段。
#: 14: host 模式新增 `operator_profile`：有界 NCU 采集、独立预算与 GPU 租约互斥。
#: 15: final 默认采用账本推荐源码；显式 override 与选择/交付投影随 journal 快照落账。
TOOL_VERSION = 15

#: 允许进 `declaration` 的键（§6.3 的白名单）。**没列在这里的一律拒绝** ——
#: 「先收下、以后再判断」是这一层最不能有的形状：它等于给未来开了一个没说出口的口子。
#: 声明白名单的**唯一权威**在 `core.task_declaration`。工具层再抄一份就是第二张会漂
#: 的表 —— 「工具说收 A、实现只认 B」正是这次要治的病。这里只做别名，语义在 core：
#:
#:   request        自然语言请求（必填）——**只进产物与审计，不参与判据推导**
#:   shapes         形状分布（必填）：matmul 是 [M,N,K]，逐元素/归一化是 [M,N]
#:   operator       算子类型（必填，闭合词表）——不再从 request 的关键词推断
#:   precision      精度（必填，fp32/fp16）——同上，不再从 request 推断
#:   hardware       默认 mock_gpu
#:   dsl            默认 python
#:   min_speedup    性能门槛（任务级决定，§10.2 同款）
#:   required_caps  执行侧必须有的能力（如 real_device）
#:   framework      默认 pytorch
#:   complexity     默认 medium（**成文默认**，不再是关键词推断的复杂度）
#:   schema_version 声明版本；认不出的版本直接拒绝
#:   notes          人给的备注，只进产物不进判据
ALLOWED_KEYS = ALLOWED_DECLARATION_KEYS

#: 明确禁止的键。列出来是为了让拒绝**可归因**（而不是只说「未知键」）：
#: 这些每一个都对应本仓「不许有开关」的清单（设计 §3.5 / §6.3）。
FORBIDDEN_KEYS = {
    "max_error": "容差是判据，不许由调用方放宽（§3.5 / §6.3）",
    "atol": "同 max_error",
    "rtol": "同 max_error",
    "skip_anticheat": "反作弊是安全不变量，没有开关",
    "require_double_confirm": "长期记忆的独立确认是**无条件**的（I79），不许配置",
    "skip_remeasure": "同上",
    "kernel_db": "语料库落点不是本工具的参数",
    "failure_db": "失败库落点不是本工具的参数",
    "disable_logging": "日志是安全不变量",
    "failure_gate": "失败门阈值不是本工具的参数",
}

#: 已接线的后端。`mock` = 控制机 numpy（最快、最诚实）；`local` = LocalExecutor（本机 CPU，
#: **没有输入回读协议** ⇒ 带正确性套件的任务 L2 会如实判 unknown，不要读成「候选错了」）；
#: `ssh` = 真机，需要 host（或 `~/.ssh/config` 别名）与网络。
_BACKENDS = ("mock", "local", "ssh", "fake")
_STOP_AFTER = {"L1": VerificationLevel.L1_COMPILE,
               "L2": VerificationLevel.L2_FUNCTION,
               "L3": VerificationLevel.L3_PERFORMANCE}

#: 候选来源（**闭合词表**）。判定层看不出源码是谁写的 ⇒ 这一格只能由调用方声明；
#: 但「没声明」本身是要记的事实，所以缺省值是 `undeclared` 而**不是省略这一格** ——
#: 省略就等于把「这个核走了哪条路」重新变成猜（`docs/design/hitl-generation-boundary.md` §4）。
#:
#: ⚠️ 它**不是判据**：任何取值都不改 `levels`/`route`/`eligibility`/`conflict`/`passed`，
#: 也不进契约（`prepare` 按 `unknown_parameter` 拒收，所以它不可能影响 `contract_hash`）。
#: 声明式的归因只和调用方的诚实一样可靠 —— 它治的是「看不见」，不治「说谎」。
CANDIDATE_ORIGINS = ("outer", "inner_fixer", "inner_coder")
#: 未声明时记进回包与报告的值。
UNDECLARED_ORIGIN = "undeclared"
_PROFILE_DTYPE_BYTES = {'fp16': 2, 'float16': 2, 'bf16': 2, 'bfloat16': 2,
                        'fp32': 4, 'float32': 4, 'int8': 1, 'int32': 4, 'int64': 8}


def _origin_of(value: Any) -> str:
    """把调用方给的来源收敛成词表里的一个值；不在词表里就拒（**可归因地**拒）。"""
    if value is None or value == "":
        return UNDECLARED_ORIGIN
    if value not in CANDIDATE_ORIGINS:
        raise ToolError(
            "unknown_candidate_origin",
            f"candidate_origin 只能是 {' / '.join(CANDIDATE_ORIGINS)} 之一"
            f"（或省略 = {UNDECLARED_ORIGIN}）；收到 {value!r}")
    return str(value)


def _legacy_task_only(run_dir):
    from ..execute.host_task_budget import host_task_enabled
    if host_task_enabled(run_dir):
        raise ToolError('host_workflow_only', 'Use context and budgeted verify for this host-owned task.')

#: 回给外层的 metrics 只留这几项 —— 把整份 metrics 丢出去会让「读哪个数」变成猜。
_METRIC_KEYS = (
    "max_error", "max_rel_error", "matched_ratio", "compared_elements",
    "first_bad_index", "coverage", "reference_nonfinite",
    "speedup_vs_baseline", "gflops", "mode", "median_time_ms",
    "baseline_type", "baseline_time_ms", "baseline_reused", "gpu_util_pct",
    "measurement_validity", "measurement_l3_pass_refused",
    "measurement_can_gate", "measurement_validity_reasons",
    "static_guards", "ruler_broken", "benchmark_contaminated",
    "test_only", "evidence_kind", "gpu_acceptance", "performance_evidence",
)


class ToolError(ValueError):
    """可归因的拒绝。`code` 是给机器读的稳定 token，`detail` 是给人读的原话。"""

    def __init__(self, code: str, detail: str = "", *, extra: Optional[dict] = None):
        self.code = code
        self.detail = detail or code
        # `extra` 是**给机器读的结构化补充**：缺哪些字段、哪些不受支持、哪些取了默认值。
        # 此前这些只折进 `detail` 那句中文里，宿主只能正字串 —— 而它本来就是结构化的，
        # 折成文本等于要求每个调用方另写一个解析器。
        self.extra = dict(extra or {})
        super().__init__(f"{code}: {self.detail}")

    def as_dict(self) -> dict:
        payload = {"ok": False, "error": self.code, "detail": self.detail}
        payload.update(self.extra)
        return payload


# ── 小工具 ────────────────────────────────────────────────────────────────
def _quiet(fn, *args, **kwargs):
    """跑一次判定，把库里的 stdout 收进缓冲区。

    判定层（`Verifier` / `IntentParser`）是**按人盯屏幕**写的，会给 stdout 打进度。
    这一层要保证 **stdout 只有一行 JSON**，所以进度必须被收走 —— 收走的不是日志，
    它会写进 run 目录（`verify-N.log`），只是不占 stdout。
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        value = fn(*args, **kwargs)
    return value, buf.getvalue()


def _write_json(path: Path, payload: Any) -> None:
    """原子写（先写临时文件再 rename）：半份 JSON 比没有更糟。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                   encoding="utf-8")
    os.replace(tmp, path)


def _check_keys(declaration: dict) -> None:
    if not isinstance(declaration, dict):
        raise ToolError("bad_declaration", "declaration 必须是一个对象")
    bad = sorted(k for k in declaration if k in FORBIDDEN_KEYS)
    if bad:
        raise ToolError(
            "forbidden_parameter",
            "; ".join(f"{k}: {FORBIDDEN_KEYS[k]}" for k in bad))
    unknown = sorted(k for k in declaration if k not in ALLOWED_KEYS)
    if unknown:
        raise ToolError("unknown_parameter",
                        f"不认识的键 {unknown}；允许的是 {sorted(ALLOWED_KEYS)}")
    request = declaration.get("request")
    if not isinstance(request, str) or not request.strip():
        raise ToolError("request_required", "declaration.request 必须是非空字符串")
    if not declaration.get("shapes"):
        # 不猜：契约拥有输入是这一层的前提；没有声明就没有「评的是哪组输入」。
        raise ToolError("shapes_required",
                        "必须声明 shapes —— 这一层不接受「输入由调用方临时给」的形态")


def _shapes(declaration: dict) -> tuple:
    raw = declaration.get("shapes")
    out = []
    for item in raw:
        if isinstance(item, (str, bytes)) or not isinstance(item, Iterable):
            raise ToolError("bad_shapes", f"shapes 里每一项必须是整数序列，实际 {item!r}")
        try:
            dims = tuple(int(d) for d in item)
        except (TypeError, ValueError) as exc:
            raise ToolError("bad_shapes", f"{item!r} 读不出整数维度: {exc}") from exc
        if any(d < 0 for d in dims):
            raise ToolError("bad_shapes", f"{item!r} 有负维度")
        out.append(dims)
    if not out:
        raise ToolError("shapes_required", "shapes 不能为空")
    return tuple(out)


def _required_caps(declaration: dict) -> tuple:
    raw = declaration.get("required_caps") or ()
    caps = []
    for name in raw:
        try:
            caps.append(Cap(str(name)))
        except ValueError as exc:
            raise ToolError(
                "unknown_cap",
                f"{name!r} 不在闭合词汇里；已知的是 {[c.value for c in Cap]}") from exc
    return tuple(sorted(set(caps), key=lambda c: c.value))


def build_task(declaration: dict) -> TaskSchema:
    """声明 → TaskSchema。**纯映射**：不读 `request` 的字面内容。

    这一层此前会拿 `request` 再跑一次规则版 `IntentParser`，再用声明覆盖其中几项；
    没被覆盖的字段就继承了关键词猜测 —— 「宿主以为声明的是 A、冻结契约实际是 B」。
    现在语义字段（`operator` / `precision`）必须显式，`request` 只进产物与审计。
    关键词解析退到 `normalize_declaration`，那里只产出**提案**，冻结不了任何东西。
    """
    from ..decide.correctness_profiles import (AttentionScopeError, unresolved_semantics,
                                               with_default_correctness)
    _check_keys(declaration)
    try:
        decl, _assumptions = parse_declaration(declaration)
    except DeclarationError as error:
        # 拒绝要**可行动**：缺哪些字段、哪些不支持，是结构化的，就按结构化交出去，
        # 不要只折进 detail 那句中文让宿主去正字串。
        extra = {key: value for key, value in error.as_dict().items()
                 if key not in ('ok', 'error', 'detail')}
        raise ToolError(error.code, error.detail, extra=extra) from error

    compute_units, warp_size = hardware_profile(decl.hardware)
    fused_ops, semantic_parameters = unresolved_semantics(decl.request, decl.operator)
    task = TaskSchema(
        source=SourceSpec(kernel_type=decl.operator,
                          complexity=complexity_of(decl),
                          framework=decl.framework,
                          fused_ops=list(fused_ops),
                          semantic_parameters=dict(semantic_parameters)),
        target=TargetSpec(hardware=decl.hardware, dsl=decl.dsl,
                          compute_units=compute_units, warp_size=warp_size,
                          required_caps=_required_caps(declaration)),
        constraints=ConstraintSpec(precision=precision_of(decl),
                                   min_speedup=decl.min_speedup,
                                   max_error=1e-5, shapes=decl.shapes),
    )
    try:
        return with_default_correctness(task)
    except AttentionScopeError as error:
        raise ToolError("attention_scope", str(error)) from error


def declaration_assumptions(declaration: dict) -> tuple:
    """哪些字段其实没被声明、取了成文默认值。`prepare` 把它回显给宿主。

    刻意**不**塞进 `TaskSchema`：那会改变契约哈希，让「默认值有没有被用到」污染判据。
    所以这份清单进响应与产物，不进判据。
    """
    try:
        _, assumptions = parse_declaration(declaration)
    except DeclarationError:
        return ()
    return assumptions


def draft_from_request(request: str, *, parser=None) -> dict:
    """从自然语言**起草**一份声明，供宿主确认。**不冻结任何东西。**

    这是关键词解析器唯一的落脚点，职责是「提案」而不是「权威」：返回的 dict 必须
    由调用方（宿主）核过、必要时改过，再以显式字段回填给 `parse_declaration`。
    没有回填这一步，声明就不完整 —— 提案不会自己变成契约。
    """
    if parser is None:
        parser = IntentParser()
    # 解析器是按「人盯屏幕」写的，会往 stdout 打进度。提案函数不该污染调用方的 stdout：
    # MCP 那一侧整个 stdout 必须是**一行 JSON**。
    task, _out = _quiet(parser.parse, request)
    draft = {
        "schema_version": DECLARATION_SCHEMA,
        "request": request,
        "operator": task.source.kernel_type,
        "precision": task.constraints.precision.value,
        "dsl": task.target.dsl,
        "hardware": task.target.hardware,
        "framework": task.source.framework,
        "min_speedup": task.constraints.min_speedup,
        "complexity": task.source.complexity.value,
    }
    if task.target.required_caps:
        draft["required_caps"] = [c.value for c in task.target.required_caps]
    return draft


def declaration_from_request(request: str, **explicit) -> dict:
    """**宿主侧**辅助：起草 + 应用显式覆盖 → 一份完整声明 dict。

    这是给「自己就拥有自然语言」的调用方用的 —— CLI、benchmark 驱动、实验脚本。
    推断发生在这里（宿主侧），产出的仍是**显式字段**，交给 `parse_declaration`。
    判据层看不到这个函数，也不需要在 `request` 里找任何东西。

    与 `normalize_declaration` 的区别：那个把提案交回给人核；这个直接把提案定下来，
    只用于「调用方本来就知道自己在干什么」的场景。
    """
    draft = draft_from_request(request)
    for key in ("operator", "precision", "dsl", "hardware", "framework",
                "min_speedup", "complexity", "shapes"):
        if explicit.get(key) not in (None, ""):
            draft[key] = explicit[key]
    return draft


def normalize_declaration(payload: dict) -> dict:
    """把自然语言**起草**成一份声明，交给宿主确认。**不冻结、不落盘。**

    这是关键词解析器唯一还活着的落脚点，职责从「生产契约」退成「提出候选值」：
    `proposed` 必须被宿主核过、必要时改过，再以显式字段回填给 `prepare`。没有回填
    这一步声明就不完整 —— 提案不会自己变成契约。
    """
    raw = (payload.get("declaration")
           if isinstance(payload, dict) and isinstance(payload.get("declaration"), dict)
           else payload)
    if not isinstance(raw, dict):
        raise ToolError("bad_declaration", "payload 必须是对象，或 {declaration: {...}}")
    request = raw.get("request")
    if not isinstance(request, str) or not request.strip():
        raise ToolError("request_required", "起草需要非空的 request")
    if not raw.get("shapes"):
        raise ToolError("shapes_required", "起草需要 shapes：契约拥有输入是这一层的前提")
    proposed = _quiet(draft_from_request, request)[0]
    proposed["shapes"] = raw["shapes"]
    for key in ("hardware", "dsl", "framework", "min_speedup", "notes"):
        # 宿主自己给了的字段以宿主为准；解析器只在缺失处提案。
        if raw.get(key) not in (None, ""):
            proposed[key] = raw[key]
    return {
        "ok": True,
        "schema_version": DECLARATION_SCHEMA,
        "proposed": proposed,
        "state": field_state(proposed),
        "next": ("核过 proposed 的每一个字段，再把这些显式字段回填给 operator_prepare；"
                 "提案本身不冻结任何东西"),
    }


def _case_summary(case) -> dict:
    return {"name": case.name, "key": case.key(), "seed": case.seed,
            "tensors": [{"name": t.name, "shape": list(t.shape), "dtype": t.dtype}
                        for t in case.tensors]}


def _backend(name: str, *, host: Optional[str] = None, port: Optional[int] = None,
             user: Optional[str] = None, key_path: Optional[str] = None,
             work_dir: Optional[str] = None, timeout: int = 300,
             persistent: bool = False):
    """按名字造后端。**显式失败，不静默降级**（doc 49 §3.3 S3 的三态语义）。"""
    if name not in _BACKENDS:
        raise ToolError("backend_not_wired",
                        f"backend={name!r} 未接线（已接线：{list(_BACKENDS)}）")
    if name == "fake":
        from ..execute.fake_backend import FakeBackend
        return FakeBackend()
    if name == "mock":
        from ..execute.mock_gpu import MockGPU
        return MockGPU()
    from ..execute.remote_executor import RemoteExecutor
    if name == "local":
        return RemoteExecutor(mode="local")
    if not host:
        # `mode="auto"` 会一路降到 CPU —— 那正是「以为在真机上、其实在控制机」的形状。
        raise ToolError("host_required",
                        "backend=ssh 需要 host（~/.ssh/config 的别名即可）；"
                        "本层不提供 auto 降级")
    from ..execute.backend_ownership import register_backend
    return register_backend(RemoteExecutor(mode="ssh", host=host, port=port, user=user,
                          key_path=key_path, work_dir=work_dir,
                          timeout=int(timeout), persistent=bool(persistent)),
                          persistent=bool(persistent))


def _make_backend(name: str, **options):
    """构造后端并把解析结果一起取出来。

    单独成一个函数是为了让**构造期的 print** 也落在 `_quiet` 的捕获范围里 ——
    `RemoteExecutor` 在解析时会打印「SSH 连接成功 / 使用本地 CPU 模式」，那几行跑进
    stdout 就会把回包弄成非 JSON（2026-10-18 真机第一次 L1 实测）。
    """
    backend_obj = _backend(name, **options)
    try:
        resolution = {
            "requested": name,
            "resolved": str(getattr(getattr(backend_obj, "resolution", None), "resolved", "")
                            or getattr(backend_obj, "name", name)),
            "degraded": bool(getattr(getattr(backend_obj, "resolution", None), "degraded", False)),
            "attempts": list(getattr(getattr(backend_obj, "resolution", None), "attempts", []) or []),
        }
    except Exception:                       # 解析失败留给 Verifier 那一层报
        resolution = {"requested": name, "resolved": "", "degraded": False, "attempts": []}
    return backend_obj, resolution


def _level_of(value) -> Optional[VerificationLevel]:
    if value is None:
        return None
    if isinstance(value, VerificationLevel):
        return value
    token = str(value).strip().upper()
    if token in _STOP_AFTER:
        return _STOP_AFTER[token]
    raise ToolError("bad_stop_after", f"stop_after 只认 {sorted(_STOP_AFTER)} 或 None")


#: 逐 case 只留这几格。**不带的**是 `metrics` / `repeat_metrics` —— 那里有完整的
#: 输入保真回读（每 case 好几 KB），2026-10-18 实测一次 L2 就 40 KB。
#: 判据是「工具输出是给模型读的」：明细留在 `report.md` 与 `verify-N.json` 里，读者按需去取。
_CASE_KEYS = ("case_id", "verdict", "reason", "shape_key", "pattern", "layout", "determinism")


def _selected_metrics(metrics: dict) -> dict:
    out = {k: metrics[k] for k in _METRIC_KEYS if k in metrics}
    correctness = metrics.get("correctness")
    if isinstance(correctness, dict):
        out["correctness"] = {k: correctness[k]
                              for k in ("verdict", "reason") if k in correctness}
        cases = correctness.get("cases")
        if isinstance(cases, list):
            out["correctness"]["cases_total"] = len(cases)
            out["correctness"]["cases"] = [
                {k: c[k] for k in _CASE_KEYS if k in c} for c in cases[:10]]
    return out


def _judgement_notes(contract, metrics: dict) -> list:
    """把「这个数该怎么读」里**可判**的那几条写出来，不替人做决定。

    第一条来自 2026-10-18 真机实测：契约写的是 `tf32=False`（fp32 全精度），而
    `verifier.baseline_plan()` 对 fp32 matmul 的**首选基线是 `pytorch_eager_tf32`**
    （理由写在那个函数里：Triton 的 `tl.dot` 在 Ampere+ 默认就走 TF32）。于是：
    候选若显式用 `input_precision="ieee"`，这个加速比**不是同数值语义的比较**。
    这里只把事实与两条出路写出来 —— 判定仍归 measurement/eligibility。
    """
    notes = []
    baseline = str(metrics.get("baseline_type") or "")
    tf32 = bool(getattr(getattr(contract, "dtype_semantics", None), "tf32", False))
    if baseline == "pytorch_eager_tf32" and not tf32:
        notes.append(
            "基线是 cuBLAS(TF32)，而契约声明 tf32=False：这个加速比不是同数值语义的比较。"
            "候选若用 IEEE fp32（tl.dot(input_precision=\"ieee\")），它天然跑不过 TF32 基线；"
            "要么让候选也用 TF32（同时容差要相应放宽），要么改用 fp16 任务比 cuBLAS fp16。")
    if metrics.get("ruler_broken"):
        notes.append("标定心跳判了 ruler_broken：这一窗的读数全部失效（数仍在，不许据此动候选）")
    return notes


def _eligibility_of(results: list) -> Optional[dict]:
    for r in results:
        m = getattr(r, "metrics", None) or {}
        elig = (m.get("observation") or {}).get("eligibility")
        if isinstance(elig, dict):
            return {k: bool(v) for k, v in elig.items()}
    return None


def _validity_of(results: list) -> Optional[str]:
    for r in reversed(results):
        m = getattr(r, "metrics", None) or {}
        if m.get("measurement_validity"):
            return str(m["measurement_validity"])
    return None


def _conflict(route: Route, eligibility: Optional[dict], validity: Optional[str]) -> dict:
    """处方与资格打架时的**显式**结论（§9.5 的接缝）。

    口径：处方说「这一轮算数 / 去搜索 / 去修」，而资格位说这个数不能驱动任何后果
    ⇒ 不许照着处方走。`hold_for_measurement` 是给人/外层看的下一步，不是新的判定。
    """
    if not eligibility or not route.counts_as_iteration:
        return {"present": False}
    usable = bool(eligibility.get("can_tune"))
    if usable:
        return {"present": False}
    return {
        "present": True,
        "why": (f"处方={route.route}（counts_as_iteration=True）而资格 can_tune=False"
                f"（validity={validity}）—— 这个数不能驱动搜索/调参"),
        "action": "hold_for_measurement",
        "next_step": ("先让读数有资格：声明/补齐 required_caps（例如 real_device）、"
                      "或换到能产出可判读数的执行侧；在资格位转正之前，"
                      "不要把 route 当下一步、也不要把它计入一轮。"),
    }


# ── 动作 1: prepare ──────────────────────────────────────────────────────
def prepare(declaration: dict, run_dir: str | Path, *, now: Optional[float] = None,
            host_budget: Optional[dict] = None) -> dict:
    from ..decide.correctness_profiles import AttentionScopeError
    from ..execute.host_task_budget import (HostTaskBudget, HostTaskBudgetError,
        HostTaskLimits, host_task_enabled, initialize_host_task)
    run = Path(run_dir)
    managed = host_budget is not None or host_task_enabled(run)
    if not managed:
        return _prepare_unlocked(declaration, run, now=now)
    from .host_workflow import prepare_lock, prepared_task
    try:
        limits = HostTaskLimits.from_dict(host_budget) if host_budget is not None else None
        with prepare_lock(run):
            if (run/'declaration.json').exists():
                _, _, existing = prepared_task(run)
                stored = json.loads((run/'declaration.json').read_text())
                if stored != declaration:
                    raise ToolError('host_task_binding', 'Use the frozen declaration for this host task.')
                if not host_task_enabled(run):
                    raise ToolError('host_task_binding', 'Start a host task in a fresh directory.')
                now = json.loads((run/'run.json').read_text()).get('created_at', now)
                if limits is None:
                    HostTaskBudget(run, existing.hash).snapshot()
            result = _prepare_unlocked(declaration, run, now=now)
            result['host_task'] = (initialize_host_task(run, result['contract_hash'], limits)
                if limits is not None else HostTaskBudget(run, result['contract_hash']).snapshot())
            result['workflow'] = 'host'
            for case in result['cases']:
                case.pop('seed', None)
            return result
    except HostTaskBudgetError as error:
        raise ToolError(error.code, error.detail) from error
    except AttentionScopeError as error:
        raise ToolError('attention_scope', str(error)) from error
    except (TypeError, ValueError) as error:
        if isinstance(error, ToolError):
            raise
        raise ToolError('invalid_host_budget', str(error)) from error


def _record_prepared(run: Path, task_id: str, contract_hash: str) -> None:
    """在任务账本里写下「这个目录被准备过」——**一次**，且只在这里写。

    为什么是 prepare 而不是别的写者：`task_prepared` 断言的是「这里有契约」。
    谁都可以在缺账本时补一条，那条流就不再能回答「这个目录当初被准备过吗」。
    所以重复 prepare 不追加（幂等），别人的 Job 事件也不许替它补。

    账本坏掉（半行、断号、不认的 schema）时**拒绝准备**：那种流已经不能作为
    这条任务的权威，继续往里写只会让「哪份是对的」更说不清。
    """
    from ..core import task_journal as journal
    try:
        events = journal.read_events(run)
    except journal.JournalError as error:
        raise ToolError('task_journal_unusable',
                        f'已有账本无法读取（{error.code}），拒绝在它上面继续准备') from error
    if not events:
        journal.append(run, 'task_prepared', task_id=task_id, contract_hash=contract_hash)
        return
    phase = journal.fold(events)['task']['phase']
    if phase == 'closed':
        raise ToolError('task_closed',
                        '这个任务目录已经封账（task_closed）；要继续推进请用新的任务目录')


def _prepare_unlocked(declaration: dict, run_dir: str | Path, *, now: Optional[float] = None) -> dict:
    """把一份声明冻结成「可复算的回合」：契约 + 输入 + 身份，落进 run 目录。"""
    task = build_task(declaration)
    try:
        contract = freeze_from_task(task)
    except ValueError as exc:                     # ContractError / BenchCaseError
        raise ToolError("contract_refused", str(exc)) from exc
    if not contract.cases:
        raise ToolError("contract_has_no_inputs",
                        "契约没有派生任何 case（形状分布没被接受）—— 拒绝开始")
    case = contract.cases[0]
    try:
        inputs = case.materialize()
    except ValueError as exc:
        raise ToolError("inputs_not_materializable", str(exc)) from exc

    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    payload = {
        "tool_version": TOOL_VERSION,
        "created_at": now if now is not None else time.time(),
        "declaration": declaration,
        "contract": contract.as_dict(),
    }
    _write_json(run / "declaration.json", declaration)
    _write_json(run / "contract.json", {"hash": contract.hash,
                                        "contract": contract.as_dict()})
    _write_json(run / "run.json", {
        "tool": "operator_prepare", "tool_version": TOOL_VERSION,
        "created_at": payload["created_at"],
        "task_id": getattr(task, "task_id", ""),
        "contract_hash": contract.hash,
    })
    _record_prepared(run, getattr(task, "task_id", ""), contract.hash)

    assumptions = declaration_assumptions(declaration)
    notes = []
    if contract.correctness_profile is None:
        notes.append("任务没有可信的正确性 profile（L2 会按未声明处理，可能判 unknown）")
    for assumption in assumptions:
        notes.append(f"{assumption.field}={assumption.value!r} 取自成文默认值（声明未给出）")

    return {
        "ok": True,
        "run_dir": str(run),
        "run_id": run.name,
        "contract_hash": contract.hash,
        # 哪些字段其实没被声明、取了成文默认值。默认值可以存在，但必须**可见** ——
        # 静默默认正是「宿主以为 A、契约是 B」的另一条通道。
        "assumptions": [a.as_dict() for a in assumptions],
        "generation_spec": generation_spec_identity(),
        "cases": [_case_summary(c) for c in contract.cases],
        "required_cases": list(contract.required_cases),
        # 输入里可能混着**按值直传的标量**（causal/eps）：它们没有 .shape，
        # 也不能被当成数组描述。分两处报，别让"标量"被读成"零维张量"。
        "inputs": {name: {"shape": list(arr.shape), "dtype": str(arr.dtype)}
                   for name, arr in inputs.items() if hasattr(arr, "shape")},
        **({"scalars": {name: value for name, value in inputs.items()
                        if not hasattr(value, "shape")}}
           if any(not hasattr(value, "shape") for value in inputs.values()) else {}),
        "correctness_profile": contract.correctness_profile,
        "judgement": {
            "kernel_type": contract.kernel_type,
            "dtype": contract.dtype_semantics.dtype,
            "max_error": contract.max_error,
            "min_speedup": contract.min_speedup,
            "required_caps": [c.value for c in contract.required_caps],
        },
        "task": {
            "kernel_type": task.source.kernel_type,
            "hardware": task.target.hardware,
            "dsl": task.target.dsl,
            "precision": task.constraints.precision.value,
            "shapes": [list(s) for s in task.constraints.shapes],
        },
        "notes": notes,
    }


def _recommended_final(snapshot, run, submitted_code, override_reason):
    best = snapshot.get('best_candidate') if isinstance(snapshot, dict) else None
    best = best if isinstance(best, dict) else {}
    recommended_hash = best.get('source_hash')
    recommended_round = best.get('round')
    recommended_source = best.get('source')
    if not recommended_source and best.get('evidence'):
        recommended_source = str(Path(best['evidence']).parent / 'candidate.py')
    submitted_hash = (code_hash(submitted_code)
                      if isinstance(submitted_code, str) and submitted_code.strip() else None)
    if not (recommended_hash and recommended_source and Path(recommended_source).is_file()):
        selection = dict(
            schema=FINAL_SELECTION_SCHEMA, policy='no_ledger_recommendation',
            recommended_source=recommended_hash, recommended_round=recommended_round,
            submitted_source=submitted_hash, actual_source=submitted_hash,
            override=False, override_reason=None, reason='no_recommendation',
            created_at=time.time())
        return submitted_code, selection
    recommended_code = Path(recommended_source).read_text(encoding='utf-8')
    if code_hash(recommended_code) != recommended_hash:
        raise ToolError('final_selection_source', 'recommended source no longer matches its ledger hash')
    if submitted_hash == recommended_hash:
        actual, override, reason = submitted_code, False, 'submitted_matches_recommendation'
    elif isinstance(override_reason, str) and override_reason.strip():
        actual, override, reason = submitted_code, True, 'explicit_override'
    else:
        actual, override, reason = recommended_code, False, 'ledger_recommended_default'
    selection = dict(
        schema=FINAL_SELECTION_SCHEMA, policy='ledger_recommended_default',
        recommended_source=recommended_hash, recommended_round=recommended_round,
        recommended_role=best.get('role'), submitted_source=submitted_hash,
        actual_source=code_hash(actual), override=override,
        override_reason=(override_reason if override else None), reason=reason,
        recommended_path=str(recommended_source), created_at=time.time())
    return actual, selection


# ── 动作 2: verify ───────────────────────────────────────────────────────
def verify(run_dir: str | Path, code: str, *, verification_purpose='explore',
           override_reason=None, **options) -> dict:
    from ..execute.host_task_budget import HostTaskBudget, HostTaskBudgetError, host_task_enabled
    run = Path(run_dir)
    if not host_task_enabled(run):
        if verification_purpose != 'explore':
            raise ToolError('host_task_required', 'final requires a prepared host task.')
        return _verify_candidate(run_dir, code, verification_purpose=verification_purpose,
                                 **options)
    from .host_workflow import prepared_task
    _, _, contract = prepared_task(run)
    budget = HostTaskBudget(run, contract.hash)
    reservation = None
    final_selection = None
    try:
        if not isinstance(code, str) or not code.strip():
            raise ToolError('code_required', 'verify needs candidate source.')
        if override_reason is not None and not isinstance(override_reason, str):
            raise ToolError('override_reason', 'override_reason must be a string.')
        if verification_purpose == 'final':
            code, final_selection = _recommended_final(
                budget.snapshot(readonly=True), run, code, override_reason)
        from .tool_calls import host_context
        owner = host_context(run)
        reservation = budget.reserve(code_hash(code), purpose=verification_purpose,
            owner_invocation_id=owner['invocation_id'] if owner else None,
            final_selection=final_selection)
        result = _verify_candidate(run_dir, code, verification_purpose=verification_purpose,
                                   **options)
        if final_selection is not None:
            result['final_selection'] = final_selection
        snapshot = budget.finish(reservation['id'], result=result)
        result['host_task'] = snapshot
        result['verification_purpose'] = verification_purpose
        return result
    except HostTaskBudgetError as error:
        raise ToolError(error.code, error.detail) from error
    except BaseException:
        if reservation is not None:
            budget.finish(reservation['id'], outcome='failed')
        raise


def _require_full_l2(verification: dict, contract, verification_round: int) -> None:
    verdicts = [row for row in verification.get('verdicts') or []
                if isinstance(row, dict) and row.get('level') == 'L2']
    levels = verification.get('levels') or {}
    if levels.get('L2') is not True or len(verdicts) != 1 or verdicts[0].get('passed') is not True:
        raise ToolError('profile_l2_required',
                        f'verification_round {verification_round} has no completed full L2 pass')
    cases = (verdicts[0].get('metrics') or {}).get('correctness', {}).get('cases')
    suite = getattr(contract, 'correctness_suite', None)
    if suite is not None:
        required = sorted(case.case_id for case in suite.cases)
        if (not isinstance(cases, list) or not required
                or any(not isinstance(case, dict) for case in cases)
                or sorted(case.get('case_id') for case in cases) != required
                or not all(case.get('verdict') == 'passed' for case in cases)):
            raise ToolError('profile_l2_required',
                            f'verification_round {verification_round} did not pass the full L2 suite')
    elif not isinstance(cases, list) or not all(
            isinstance(case, dict) and case.get('verdict') == 'passed' for case in cases):
        raise ToolError('profile_l2_required',
                        f'verification_round {verification_round} has no complete L2 case list')


def _profile_logical_bytes(case) -> int:
    total = 0
    for tensor in case.tensors:
        itemsize = _PROFILE_DTYPE_BYTES.get(str(tensor.dtype).lower())
        if itemsize is None:
            return 0
        count = 1
        for dim in tensor.shape:
            count *= int(dim)
        total += count * itemsize
    return total


def profile(run_dir: str | Path, verification_round: int, *, case_index: int = 0,
            backend: str = 'ssh', host: Optional[str] = None, port: Optional[int] = None,
            user: Optional[str] = None, key_path: Optional[str] = None,
            work_dir: Optional[str] = None, timeout: int = 300,
            ncu_path: Optional[str] = None) -> dict:
    """Capture one on-demand NCU profile for an already-L2-passed candidate."""
    from .host_workflow import _verification_materials, prepared_task
    from ..execute.host_task_budget import HostTaskBudget, host_task_enabled
    from ..execute.profile_budget import ProfileBudget, ProfileBudgetError
    from ..execute.profiling import ncu_profile
    from ..decide.bench_cases import case_spec
    if backend != 'ssh':
        raise ToolError('profile_backend',
                        'operator_profile currently requires backend=ssh (real GPU NCU)')
    if not isinstance(host, str) or not host.strip():
        raise ToolError('host_required',
                        'backend=ssh needs host (a ~/.ssh/config alias or hostname)')
    if type(timeout) is not int or not 1 <= timeout <= 300:
        raise ToolError('profile_timeout_setting', 'timeout must be an integer in [1, 300] seconds')
    run, task, contract = prepared_task(run_dir)
    if not host_task_enabled(run):
        raise ToolError('host_task_required', 'operator_profile requires a bounded host task')
    source, verification, directory = _verification_materials(
        run, contract, backend, verification_round)
    _require_full_l2(verification, contract, verification_round)
    budget = HostTaskBudget(run, contract.hash).snapshot()
    if budget['terminal']:
        raise ToolError('profile_task_terminal', 'the host task already reached a terminal phase')
    if budget['expired'] or budget['remaining_s'] < 1:
        raise ToolError('profile_deadline', 'the host task deadline leaves no room for a profile')
    if budget.get('unfinished'):
        raise ToolError('profile_task_busy',
                        'a verification is in flight on this task; wait for it to settle '
                        'before starting an NCU replay')
    if type(case_index) is not int or not 0 <= case_index < len(contract.cases):
        raise ToolError('profile_case',
                        f'case_index must be in [0, {len(contract.cases) - 1}]')
    case = contract.cases[case_index]
    spec = case_spec(case)
    logical_bytes = _profile_logical_bytes(case)
    l3 = [row for row in verification.get('verdicts') or []
          if isinstance(row, dict) and row.get('level') == 'L3']
    metrics = (l3[0].get('metrics') or {}) if len(l3) == 1 else {}
    gpu = metrics.get('selected_gpu')
    baseline_performance = {}
    baseline_path = run / 'baseline.json'
    if baseline_path.is_file():
        try:
            baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
            baseline_performance = (baseline.get('performance') or {}) if isinstance(baseline, dict) else {}
        except (OSError, ValueError, TypeError):
            baseline_performance = {}
    cache = ncu_profile.cache_conditions(metrics, baseline_performance)
    if type(gpu) is not int:
        raise ToolError('profile_gpu_unknown',
                        'the verification receipt has no selected GPU; cannot guarantee the same card')
    runtime = (((metrics.get('timing_receipt') or {}).get('calibration_identity') or {})
               .get('runtime') or {})
    gpu_uuid = runtime.get('gpu_uuid') if isinstance(runtime.get('gpu_uuid'), str) else None
    arm = (metrics.get('arm_identity') or {}).get('candidate') or {}
    gpu_name = (arm.get('name') if isinstance(arm.get('name'), str) and arm.get('name')
                else 'NVIDIA H20')
    ledger = ProfileBudget(run)
    try:
        profiles = ledger.snapshot()
    except ProfileBudgetError as error:
        raise ToolError(error.code, error.detail) from error
    if profiles['in_flight']:
        raise ToolError('profile_in_flight', 'another profile is still in flight')
    if profiles['remaining'] <= 0:
        raise ToolError('profile_budget', 'profile budget exhausted (2 per task)')
    if float(budget['remaining_s']) < 60:
        raise ToolError('profile_deadline',
                        'the remaining task wall clock is too short for a profile (minimum 60 s)')
    wall_cap = min(float(timeout), float(budget['remaining_s']))
    deadline = time.monotonic() + wall_cap
    from . import tool_calls as _tool_calls
    owner_context = _tool_calls.profile_context(run)
    owner = owner_context.get('invocation_id') if isinstance(owner_context, dict) else None
    try:
        reservation = ledger.reserve(timeout_s=wall_cap, owner_invocation_id=owner)
    except ProfileBudgetError as error:
        raise ToolError(error.code, error.detail) from error
    profile_id = reservation['id']
    profile_dir = run / f'profile-{profile_id}'
    started = time.time()
    required_env = {'OPERATOR_AGENT_REQUIRED_GPU': str(gpu),
                    'OPERATOR_AGENT_REQUIRED_GPU_UUID': gpu_uuid or '',
                    'OPERATOR_AGENT_REQUIRED_GPU_NAME': gpu_name or ''}
    previous_env = {key: os.environ.get(key) for key in required_env}
    settled = False
    backend_obj = None
    cleanup = None
    cleanup_s = None
    cleanup_grace_s = None

    def failed(code, detail):
        nonlocal settled
        if not settled:
            ledger.finish(profile_id, status='failed',
                          wall_s=time.time() - started, error=f'{code}: {detail}',
                          cleanup=cleanup, cleanup_s=cleanup_s,
                          cleanup_grace_s=cleanup_grace_s)
            settled = True
        return ToolError(code, detail)

    try:
        profile_dir.mkdir(parents=True, exist_ok=True)
        for key, value in required_env.items():
            if value:
                os.environ[key] = value
            else:
                os.environ.pop(key, None)
        (backend_obj, resolution), backend_log = _quiet(
            _make_backend, backend, host=host, port=port, user=user,
            key_path=key_path, work_dir=work_dir, timeout=int(wall_cap))
        if Cap.NCU_PROFILE not in backend_obj.capabilities:
            raise failed('profile_backend', 'the selected backend has no NCU profile path')
        result = backend_obj.profile(source, spec, artifact_dir=profile_dir,
                                     timeout=int(wall_cap), ncu_path=ncu_path,
                                     deadline=deadline)
        cleanup = result.get('cleanup')
        cleanup_s = result.get('cleanup_s')
        cleanup_grace_s = result.get('cleanup_grace_s')
        if not result.get('ok'):
            log = ((profile_dir / 'ncu.log').read_text(encoding='utf-8', errors='replace')
                   if (profile_dir / 'ncu.log').is_file() else '')
            if 'ERR_NVGPUCTRPERM' in log or 'permission' in log.lower():
                raise failed('ncu_permission', 'NCU cannot read GPU performance counters under this identity')
            if 'command not found' in log or 'No such file' in log:
                raise failed('ncu_missing', result.get('detail') or 'ncu executable not found')
            raise failed(result.get('error') or 'profile_failed',
                         result.get('detail') or log[:400] or 'NCU run failed')
        required_files = ('raw.csv', 'report.ncu-rep', 'reexport.csv', 'result.json', 'rc')
        missing = [name for name in required_files if not (profile_dir / name).is_file()]
        if missing:
            raise failed('profile_report_missing',
                         'missing NCU artifacts: ' + ', '.join(missing))
        try:
            launches = ncu_profile.parse_raw_csv((profile_dir / 'raw.csv').read_text(encoding='utf-8'))
            reexport = ncu_profile.parse_raw_csv(
                (profile_dir / 'reexport.csv').read_text(encoding='utf-8'))
        except (OSError, UnicodeError, ValueError) as error:
            raise failed('profile_report_unreadable', str(error))
        summary = ncu_profile.summarize(launches, range_name='candidate',
                                        logical_bytes=logical_bytes or None)
        if summary['decoy_seen']:
            raise failed('profile_decoy_captured',
                         f'decoy kernels reached the range: {summary["decoy_kernels"]}')
        if not summary['range_ok']:
            raise failed('profile_range_mismatch',
                         'some captured launches are outside the candidate NVTX range')
        comparison = ncu_profile.compare(launches, reexport)
        if not comparison['match']:
            raise failed('profile_reexport_mismatch',
                         '; '.join(comparison['reasons'][:4]) or 're-export differs')
        runner_path = profile_dir / 'result.json'
        runner = json.loads(runner_path.read_text(encoding='utf-8'))
        if runner.get('success') is not True:
            raise failed('profile_runner_failed', runner.get('error') or 'candidate did not run')
        files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted(profile_dir.iterdir()) if path.is_file()}
        receipt = dict(
            schema='operator-profile-v1', tool_version=TOOL_VERSION,
            created_at=time.time(), run_dir=str(run), contract_hash=contract.hash,
            verification_round=verification_round, source_hash=code_hash(source),
            source=str(directory / 'candidate.py'),
            source_evidence=str(directory / 'result.json'),
            case=dict(index=case_index, name=getattr(case, 'name', None),
                      seed=int(case.seed),
                      tensors=[tensor.to_dict() for tensor in case.tensors],
                      digest=spec.get('digest'), logical_bytes=logical_bytes),
            gpu=dict(index=gpu, uuid=gpu_uuid, name=gpu_name),
            ncu=dict(path=result.get('ncu'), version=result.get('ncu_version'),
                     args=result.get('command'), metrics=list(ncu_profile.METRICS),
                     replay='kernel', cache_control='none', clock_control='none',
                     nvtx_range='candidate'),
            cache_conditions=cache,
            launches=summary['launches'], totals=summary['totals'],
            decoy_seen=summary['decoy_seen'], range_ok=summary['range_ok'],
            reexport=dict(match=comparison['match'], reasons=comparison['reasons']),
            runner=dict(success=runner.get('success'), warmup=runner.get('warmup'),
                        decoy=runner.get('decoy'), checksum=runner.get('checksum')),
            wall_s=time.time() - started, cleanup=cleanup, cleanup_s=cleanup_s,
            cleanup_grace_s=cleanup_grace_s, profile_id=profile_id, hashes=files)
        receipt_path = profile_dir / 'receipt.json'
        write_once_artifact(receipt_path, json.dumps(receipt, ensure_ascii=False,
                                                     sort_keys=True).encode('utf-8'))
        members = dict(files)
        members['receipt.json'] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        binding = dict(source_hash=code_hash(source), contract_hash=contract.hash,
                       verification_round=verification_round,
                       case_digest=spec.get('digest'), profile_id=profile_id)
        complete_path = profile_dir / 'complete.json'
        write_once_artifact(complete_path, json.dumps(
            dict(schema='operator-profile-complete-v1', evidence_complete=True,
                 binding=binding, files_sha256=members),
            ensure_ascii=False, sort_keys=True).encode('utf-8'))
        ledger.finish(profile_id, status='completed', receipt=str(receipt_path),
                      wall_s=time.time() - started, cleanup=cleanup, cleanup_s=cleanup_s,
                      cleanup_grace_s=cleanup_grace_s)
        settled = True
        return dict(ok=True, workflow='host', run_dir=str(run),
                    contract_hash=contract.hash, verification_round=verification_round,
                    source_hash=code_hash(source), profile_id=profile_id,
                    receipt=str(receipt_path), evidence=str(complete_path),
                    case=receipt['case'], gpu=receipt['gpu'], ncu=receipt['ncu'],
                    cache_conditions=cache,
                    launches=summary['launches'], totals=summary['totals'],
                    range_ok=receipt['range_ok'], decoy_seen=receipt['decoy_seen'],
                    reexport=receipt['reexport'], hashes=members,
                    cleanup=cleanup, cleanup_s=cleanup_s,
                    cleanup_grace_s=cleanup_grace_s,
                    profiles=ledger.snapshot(),
                    host_task=HostTaskBudget(run, contract.hash).snapshot(),
                    backend_resolution=resolution, model_calls=0, context_owner='host')
    except ToolError as error:
        if not settled:
            ledger.finish(profile_id, status='failed', wall_s=time.time() - started,
                          error=f'{getattr(error, "code", "profile_failed")}: '
                                f'{getattr(error, "detail", str(error))}',
                          cleanup=cleanup, cleanup_s=cleanup_s,
                          cleanup_grace_s=cleanup_grace_s)
            settled = True
        raise
    except Exception as error:                   # noqa: BLE001 - receipt must stay truthful
        if not settled:
            ledger.finish(profile_id, status='failed', wall_s=time.time() - started,
                          error=f'profile_failed: {type(error).__name__}: {error}',
                          cleanup=cleanup, cleanup_s=cleanup_s,
                          cleanup_grace_s=cleanup_grace_s)
            settled = True
        raise ToolError('profile_failed', f'{type(error).__name__}: {error}') from error
    finally:
        if not settled:
            cleanup = 'not_started'
            cleanup_started = time.monotonic()
            if backend_obj is not None and callable(getattr(backend_obj, 'cleanup_profile', None)):
                try:
                    cleanup = backend_obj.cleanup_profile(profile_dir)
                except Exception as error:       # noqa: BLE001 - best effort
                    cleanup = f'failed:{type(error).__name__}'
            cleanup_s = time.monotonic() - cleanup_started
            try:
                ledger.finish(profile_id, status='cancelled',
                              wall_s=time.time() - started,
                              error=f'interrupted; cleanup={cleanup}',
                              cleanup=cleanup, cleanup_s=cleanup_s)
            except Exception:                    # noqa: BLE001 - keep the original failure
                pass
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _environment_of(backend_obj) -> dict:
    """缓存键用的环境身份。探测失败 ⇒ 空身份 ⇒ 本次缓存关闭（fail closed）。"""
    try:
        hw_info, _ = _quiet(backend_obj.get_hardware_info)
    except Exception:
        hw_info = {}
    return verdict_cache.environment_identity(
        hw_info, getattr(backend_obj, 'fingerprint_label', ''))


def _replay_cached(evidence, cached, *, source, contract, backend, resolution,
                   origin, level, environment, backend_log) -> dict:
    """命中缓存：新证据目录 + 原判定逐字回放，标清来源；不执行候选。

    预算门要求每次派发独立证据，所以回放也占一个完整证据目录；`cache_hit` 与
    `cached_from` 让审计能走回原始执行。预算照扣（单位是派发，不是 GPU 秒）。
    """
    full = cached["payload"]
    evidence.binding["cached_from"] = dict(
        round=cached["round"], evidence=cached["evidence"],
        source_hash=full.get("source_hash"), created_at=full.get("created_at"))
    verdicts = full.get("verdicts") or []
    compact = [{"level": v.get("level"), "passed": v.get("passed"),
                "message": (v.get("message") or "")[:600],
                "metrics": _selected_metrics(v.get("metrics") or {})} for v in verdicts]
    # 报告不照抄：归因是**本次提交**的声明，旧报告里的来源/轮次/生成时间属于另一轮。
    report_path = evidence.write_text("report.md", (
        "# 判定回放（cache hit）\n\n"
        f"- 候选来源: `{origin}` (调用方**声明**的归因, 不进判据)\n"
        f"- 本轮未执行候选；判定逐字回放自 verify-{cached['round']}\n"
        f"- 原始证据: {cached['evidence']}\n"
        f"- 原始报告: {full.get('report') or '(无)'}\n"
        f"- 契约哈希: `{contract.hash}`\n"
        f"- 源码哈希: `{source['source_hash']}`\n"
        f"- 判定: {json.dumps(full.get('levels'), ensure_ascii=False)}\n"))
    log = backend_log + f"[cache] 命中 verify-{cached['round']} 的判定，未执行候选\n"
    evidence.write_text("execution.log", log)
    payload = {
        "ok": True,
        **tool_context(backend),
        "run_id": evidence.run.name,
        "round": evidence.index,
        "contract_hash": contract.hash,
        "backend": backend,
        "backend_resolution": resolution,
        "stop_after": level.value if level else None,
        "candidate_origin": origin,
        "levels": full.get("levels"),
        "passed": full.get("passed"),
        "verdicts": compact,
        "route": full.get("route"),
        "eligibility": full.get("eligibility"),
        "validity": full.get("validity"),
        "conflict": full.get("conflict"),
        "report": report_path,
        "log": str(evidence.run / f"verify-{evidence.index}.log"),
        "evidence": str(evidence.path / "result.json"),
        **source,
        "environment": environment,
        "notes": [],
        "cache_hit": True,
        "cached_from": evidence.binding["cached_from"],
    }
    evidence.finish({**payload, "verdicts": verdicts},
                    legacy_payload=payload, legacy_log=log)
    return payload


def _verify_candidate(run_dir: str | Path, code: str, *, stop_after: Any = None,
           backend: str = "mock", target_gflops: float = 0.0,
           host: Optional[str] = None, port: Optional[int] = None,
           user: Optional[str] = None, key_path: Optional[str] = None,
           work_dir: Optional[str] = None, timeout: int = 300,
           persistent: bool = False,
           candidate_origin: Optional[str] = None,
           verification_purpose: str = 'explore',
           now: Optional[float] = None) -> dict:
    """对**这一回合**的候选源码做判定；不做候选生成、不做修复、不做搜索。

    `candidate_origin` 只是**归因**：记这份源码自称是谁写的（见 `CANDIDATE_ORIGINS`）。
    它不参与任何判定，省略时记为 `undeclared`。
    """
    if not isinstance(code, str) or not code.strip():
        raise ToolError("code_required", "verify 需要候选源码文本")
    origin = _origin_of(candidate_origin)
    run = Path(run_dir)
    decl_path = run / "declaration.json"
    if not decl_path.is_file():
        raise ToolError("run_not_prepared", f"{decl_path} 不存在；先 prepare")

    declaration = json.loads(decl_path.read_text(encoding="utf-8"))
    recorded = json.loads((run / "contract.json").read_text(encoding="utf-8"))["hash"]
    task = build_task(declaration)
    try:
        contract = freeze_from_task(task)
    except ValueError as exc:
        raise ToolError("contract_refused", str(exc)) from exc
    if contract.hash != recorded:
        # 与 `workload_codegen_h20` 的「执行必须等于被授权的计划」同一条纪律：
        # 判据在 prepare 之后被改过 ⇒ 这一回合不可比，拒绝而不是「按新的判据继续」。
        return {
            "ok": False, "error": "contract_drift",
            "detail": "run 目录里的契约与当前声明重新冻结出来的哈希不同 —— 判据在回合之间被改过",
            "recorded_hash": recorded, "current_hash": contract.hash,
            **tool_context(backend),
        }

    from ..execute.s5_verify_client import enabled, delegated_verify
    if enabled():
        try:
            return delegated_verify(run_dir, code, recorded_hash=recorded,
                stop_after=stop_after, backend=backend, host=host, port=port, user=user,
                key_path=key_path, work_dir=work_dir, persistent=persistent,
                target_gflops=target_gflops)
        except (OSError, ValueError) as exc:
            raise ToolError('s5_development_refused', str(exc)) from exc
    level = _level_of(stop_after)
    with RoundEvidence(run, "verify", declaration, contract, tool_version=TOOL_VERSION,
                       stop_after=level.value if level else None, now=now) as evidence:
        run = evidence.run
        source = evidence.source("candidate.py", code)
        evidence.binding["source_hash"] = source["source_hash"]
        inputs = contract.cases[0].materialize()
        # ⚠️ 后端构造/解析**也会 print**（`[RemoteExecutor] SSH 连接成功: …`）。2026-10-18
        # 第一次真机 L1 就撞上了：那行跑到 stdout，把整份 JSON 变成不可解析（真事）。
        # 所以这里与 verify 同一条纪律：库的 stdout 一律收进日志，不进协议流。
        (backend_obj, resolution), backend_log = _quiet(_make_backend,
                                                        backend, host=host, port=port,
                                                        user=user, key_path=key_path,
                                                        work_dir=work_dir, timeout=timeout,
                                                        persistent=persistent)
        environment = _environment_of(backend_obj)
        evidence.binding["environment"] = environment
        if verification_purpose == 'explore' and level is not None and level.value in ('L1', 'L2'):
            cached = verdict_cache.lookup(
                run, contract_hash=contract.hash, source_hash=source["source_hash"],
                stop_after=level.value, environment=environment,
                implementation=evidence.implementation)
            if cached is not None:
                return _replay_cached(evidence, cached, source=source, contract=contract,
                                      backend=backend, resolution=resolution, origin=origin,
                                      level=level, environment=environment,
                                      backend_log=backend_log)
        verifier = Verifier(backend_obj)
        install_level_recorder(verifier, evidence.path)
        results, verify_log = _quiet(verifier.verify, code, task, inputs, None,
                                     float(target_gflops), None, level)
        log = backend_log + verify_log

        route = classify(results)
        eligibility = _eligibility_of(results)
        validity = _validity_of(results)
        conflict = _conflict(route, eligibility, validity)

        index = evidence.index
        evidence.write_text("execution.log", log)

        l3 = next((r for r in results if r.level is VerificationLevel.L3_PERFORMANCE), None)
        artifact = KernelArtifact(
            kernel_id=f"{run.name}-v{index}",
            task_id=getattr(task, "task_id", ""),
            code=code,
            language="triton" if "import triton" in code else "python",
            verification_results=list(results),
            benchmark_data=dict(getattr(l3, "metrics", None) or {}) if l3 else {},
            iteration_history=[{
                "round": index,
                "action": "verify",
                "candidate_origin": origin,
                "levels_passed": {r.level.value: r.passed for r in results},
                "speedup": (getattr(l3, "metrics", None) or {}).get("speedup_vs_baseline")
                if l3 else None,
                "gflops": (getattr(l3, "metrics", None) or {}).get("gflops") if l3 else None,
                "stop_after": level.value if level else None,
            }],
        )
        report_path = evidence.write_text("report.md", build_report(artifact=artifact, task=task))

        payload = {
            "ok": True,
            **tool_context(backend),
            "run_id": run.name,
            "round": index,
            "contract_hash": contract.hash,
            "backend": backend,
            "backend_resolution": resolution,
            "stop_after": level.value if level else None,
            "candidate_origin": origin,
            "levels": {r.level.value: bool(r.passed) for r in results},
            "passed": all(r.passed for r in results),
            "verdicts": [{"level": r.level.value, "passed": bool(r.passed),
                          "message": (r.message or "")[:600],
                          "metrics": _selected_metrics(getattr(r, "metrics", None) or {})}
                         for r in results],
            "route": route.as_dict(),
            "eligibility": eligibility,
            "validity": validity,
            "conflict": conflict,
            "report": str(report_path),
            "log": str(run / f"verify-{index}.log"),
            "evidence": str(evidence.path / "result.json"),
            **source,
            "environment": environment,
            "notes": _judgement_notes(contract, getattr(l3, "metrics", None) or {}) if l3 else [],
        }
        evidence.finish({**payload, "verdicts": verification_data(results)},
                        legacy_payload=payload, legacy_log=log)
        return payload


# ── 动作 6: bench —— 探索级批量测量（**结构上不可裁决**）────────────────────
#: 回包里的分级标记。`verify` 的读数不带这个键；带了就说明它不能当裁决依据。
BENCH_GRADE = "exploration"
#: 一批最多几个候选。不是性能上限，是**窗口长度**上限：一块卡上批次越长，邻居插队的
#: 机会越多，而逐臂配对只能治「臂与臂之间」的漂移，治不了「整批一起漂」。
BENCH_MAX_CANDIDATES = 16


def _s5_development_only():
    from ..execute.s5_verify_client import enabled
    if enabled():
        raise ToolError('s5_development_only',
                        'This experiment permits prepare/verify only; revise source in the outer agent.')


def bench(run_dir: str | Path, candidates: list, *, backend: str = "mock",
          target_gflops: float = 0.0,
          host: Optional[str] = None, port: Optional[int] = None,
          user: Optional[str] = None, key_path: Optional[str] = None,
          work_dir: Optional[str] = None, timeout: int = 300,
          persistent: bool = True,
          max_candidates: int = BENCH_MAX_CANDIDATES,
          deadline_s: float = 120.0,
          now: Optional[float] = None) -> dict:
    """对**多个**候选只跑 L3 的批量测量。读数**探索级**，不可裁决。

    为什么需要它（`docs/design/hitl-bench-sweep.md` §0）：唯一能出数的 `verify` 是裁决级
    重调用 —— deploy + 环境自检 + 重测基线 + 15 条 L2 case，一次 40~60 s，还要求一个回合
    一个新 run_dir。而调优要几十到几百次测量，于是测量被迫搬到宿主 shell 里手搓
    （实测：79 次调用里插件只占 5 次，45 次是手搓的 ssh 扫描）。

    三条不变量，与 `verify` 共享、不许放宽：

    1. **契约不许漂**：`contract.json` 的 hash 必须等于按当前声明重算的 hash，否则本批
       拒绝（与 `verify` 同一条纪律：判据在回合之间被改过 ⇒ 不可比）。
    2. **每臂各自与基线同窗配对**：逐臂走 `Verifier.verify_l3_only`，它内部是
       `_bench_pair_interleaved` 的**一次窗口一条臂**。所以每臂都带自己的
       `validity`/`validity_reasons`，而不是「一批共用一个窗」—— 后者会把整批的漂移
       平摊到每臂身上，且看不出来源。
    3. **回包没有 levels / route / eligibility / passed**：这不是省略，是设计。
       资格位是「这个数能不能当依据」的闸门；一旦 `bench` 能出，调用方就会拿它替代
       `verify` —— 那正是本函数存在的动机，也正好是它必须挡住的那条路。

    `persistent=True` 默认：一次部署、一个常驻会话，逐臂之间只付测量本身。
    """
    _s5_development_only()
    _legacy_task_only(run_dir)
    run = Path(run_dir)
    decl_path = run / "declaration.json"
    if not decl_path.is_file():
        raise ToolError("run_not_prepared", f"{decl_path} 不存在；先 prepare")

    if not isinstance(candidates, list) or not candidates:
        raise ToolError("candidates_required", "bench 需要至少一个候选")
    cap = max(1, int(max_candidates))
    if len(candidates) > cap:
        raise ToolError(
            "too_many_candidates",
            f"一批最多 {cap} 个候选（收到 {len(candidates)}）—— 批次越长，"
            f"同一块卡上的测量窗口越长，邻居插队的机会越多；先粗筛，再对 survivors 细测")

    normalized = []
    for index, item in enumerate(candidates):
        if not isinstance(item, dict):
            raise ToolError("bad_candidate", f"第 {index} 个候选不是对象")
        code, code_file = item.get("code"), item.get("code_file")
        if bool(code) == bool(code_file):
            raise ToolError("code_source_ambiguous",
                            f"第 {index} 个候选必须给且只给 code / code_file 中的一个")
        if code_file:
            code = Path(code_file).read_bytes().decode("utf-8")
        if not isinstance(code, str) or not code.strip():
            raise ToolError("code_required", f"第 {index} 个候选源码为空")
        normalized.append({"name": str(item.get("name") or f"arm{index + 1}"), "code": code})

    declaration = json.loads(decl_path.read_text(encoding="utf-8"))
    recorded = json.loads((run / "contract.json").read_text(encoding="utf-8"))["hash"]
    task = build_task(declaration)
    try:
        contract = freeze_from_task(task)
    except ValueError as exc:
        raise ToolError("contract_refused", str(exc)) from exc
    if contract.hash != recorded:
        return {
            "ok": False, "error": "contract_drift",
            "detail": "run 目录里的契约与当前声明重新冻结出来的哈希不同 —— 判据在回合之间被改过",
            "recorded_hash": recorded, "current_hash": contract.hash,
            **tool_context(backend),
        }

    with RoundEvidence(run, "bench", declaration, contract,
                       tool_version=TOOL_VERSION, now=now) as evidence:
        run = evidence.run
        sources = [dict(name=item["name"], **evidence.source(f"candidates/{i+1}.py", item["code"]))
                   for i, item in enumerate(normalized)]
        evidence.binding["candidate_hashes"] = [source["source_hash"] for source in sources]
        inputs = contract.cases[0].materialize()
        (backend_obj, resolution), backend_log = _quiet(
            _make_backend, backend, host=host, port=port, user=user,
            key_path=key_path, work_dir=work_dir, timeout=timeout, persistent=persistent)
        verifier = Verifier(backend_obj)
        install_level_recorder(verifier, evidence.path)

        deadline = time.monotonic() + float(deadline_s)
        arms, logs, partial = [], [backend_log], False
        arm_evidence = []
        for candidate_index, item in enumerate(normalized):
            if time.monotonic() > deadline:
                # 截断的批次**不许**当完整结果：`measured < requested` 本身就是回包的一部分。
                partial = True
                break
            results = []
            try:
                results, arm_log = _quiet(verifier.verify_l3_only,
                                          item["code"], inputs, float(target_gflops), task,
                                          None, None, False, None, None)
                metrics = dict(getattr(results[0], "metrics", None) or {}) if results else {}
                arm_error = None
            except Exception as exc:
                # 一个臂炸了**不许毁掉整批**：候选在运行时抛异常（NameError、CUDA OOM、
                # 编译期崩溃）是扫描里的常态，它们该变成「这个臂没有数」，而不是让
                # 已经测完的那几个臂白测。异常原文留在臂上，别猜。
                metrics, arm_error = {}, f"{type(exc).__name__}: {exc}"
                arm_log = arm_error
            logs.append(f"### arm {item['name']}\n{arm_log}")
            arm_evidence.append({"name": item["name"], **sources[candidate_index],
                                 "verdicts": verification_data(results), "error": arm_error})
            arms.append({
                "name": item["name"],
                **tool_context(backend),
                "error": arm_error,
                "median_ms": metrics.get("median_time_ms"),
                "baseline_ms": metrics.get("baseline_time_ms"),
                "baseline_type": metrics.get("baseline_type"),
                "speedup": metrics.get("speedup_vs_baseline"),
                "gflops": metrics.get("gflops"),
                # 有效性是**测量**的性质，不是裁决：不给它，调用方就分不清「慢」和「没测准」。
                # 所以这两格必须在，而 `passed` 一定不在。
                "validity": metrics.get("measurement_validity"),
                "validity_reasons": list(metrics.get("measurement_validity_reasons") or []),
            })

        if not arms:
            raise ToolError("bench_produced_nothing",
                            "一个候选都没测到（deadline 太短或后端不可用）")

        usable = [a for a in arms if a["validity"] == "valid" and a["speedup"] is not None]
        best = max(usable, key=lambda a: a["speedup"])["name"] if usable else None

        index = evidence.index
        log = "\n".join(logs)
        evidence.write_text("execution.log", log)
        payload = {
            "ok": True,
            "run_id": run.name,
            "batch": index,
            **tool_context(backend),
            "contract_hash": contract.hash,
            "backend": backend,
            "backend_resolution": resolution,
            # ↓ 分级标记：有它 = 探索级
            "grade": BENCH_GRADE,
            "evidence": str(evidence.path / "result.json"),
            "arms": arms,
            "best": best,
            "requested": len(normalized),
            "measured": len(arms),
            "partial": partial,
            "note": ("探索级读数：不可裁决、不入库、不晋级。要交付结论，"
                     "请对 survivors 调 operator_verify"),
        }
        evidence.finish({**payload, "candidate_sources": sources, "arm_evidence": arm_evidence},
                        legacy_payload=payload, legacy_log=log)
        return payload


# ── 动作 0（只读）: capabilities ─────────────────────────────────────────
def _shape_probe_plan(op, dims, dtype, causal=False):
    """(输入占位, 通道标签) —— 让"探哪条通道"只有一处决定。

    标签进回执的 `channel`, 而预登记（R2）按 channel 绑定门控读数: 所以标签必须
    同时说清算子、尺寸、精度与 mask 形态, 否则 matmul 的带宽会被读成 attention 的。
    """
    import numpy as np
    element = np.float16 if dtype == 'fp16' else np.float32
    if op == 'layernorm':
        m, n = dims
        return {"X": np.empty((m, n), dtype=element)}, f'layernorm-{m}x{n}-{dtype}'
    if op == 'fused_layernorm_residual':
        m, n = dims
        return ({"X": np.empty((m, n), dtype=element), "R": np.empty((m, n), dtype=element)},
                f'fused-layernorm-residual-{m}x{n}-{dtype}')
    if op == 'attention':
        b, h, s, d = dims
        inputs = {"Q": np.empty((b, h, s, d), dtype=element),
                  "K": np.empty((b, h, s, d), dtype=element),
                  "V": np.empty((b, h, s, d), dtype=element),
                  "causal": bool(causal)}
        mask = 'causal' if causal else 'nocausal'
        return inputs, f'attention-{b}x{h}x{s}x{d}-{dtype}-{mask}'
    m, n, k = dims
    return ({"A": np.empty((m, k), dtype=element), "B": np.empty((k, n), dtype=element)},
            f'matmul-{m}x{n}x{k}-{dtype}')


def noise_preflight(*, backend='local', reps=5, planned_effect_pct=None, shape=None,
                    op='matmul', dtype='fp16', repeat=8, causal=False,
                    host=None, port=None, user=None, key_path=None, work_dir=None,
                    timeout=300, persistent=False) -> dict:
    """一条命令回答: 这个窗口现在能不能给出可判的性能读数。

    只量**这台机器围绕一个参照核的重复散布**, 不判任何候选、不发任何信用。回执里的
    `action` 三个值: start / hold / refuse —— hold 是「窗口太吵」而不是「候选不行」。

    `shape=[M,N,K]`（matmul）/ `[B,H,S,D]`（attention）给出时改用**目标尺寸的 triton** 探针, 走远端 CUDA event
    通道 (与候选核同一条计时路径)。不给则用小参照核, 那只覆盖计时通道本身 ——
    真机实测: 24×24 参照核在 SSH 上落在远端 `perf_counter` 分支, 量的是主机 CPU。
    """
    from ..decide import calibration
    from ..decide.noise_preflight import MIN_REPS, probe_source, run_preflight
    if type(reps) is not int or reps < MIN_REPS:
        raise ToolError("bad_reps", f"reps 必须是 >= {MIN_REPS} 的整数（当前 {reps!r}）")
    dims = None
    if op in ('attention', 'fused_layernorm_residual') and dtype != 'fp16':
        # attention 目前只有 fp16 profile（`builtin-attention-fp16-cuda-v1`）。
        # 探针可以量别的精度，但那会给出"这个窗口对这个算子可判"的印象，
        # 而那个算子今天根本判不了 —— 直接拒绝，比事后解释便宜。
        raise ToolError("preflight_dtype",
                        f"attention 探针只支持 fp16（当前 {dtype!r}）")
    if shape is not None:
        want = {'layernorm': 2, 'fused_layernorm_residual': 2, 'attention': 4}.get(op, 3)
        if not isinstance(shape, (list, tuple)) or len(shape) != want:
            raise ToolError("bad_shape",
                            f"shape 必须是 {want} 个整数 ("
                            f"{ {2: 'M,N', 4: 'B,H,S,D'}.get(want, 'M,N,K') }), 当前 {shape!r}")
        if any(type(value) is not int for value in shape):
            raise ToolError("bad_shape", f"shape 必须全是整数, 当前 {shape!r}")
        if op == 'attention':
            # 逐位给界: B=1、H=16 是 Scope A 的**正常**取值, 用 matmul 那条
            # [16, 65536] 会把它们判非法（第一次跑就被自己的校验拦住了）。
            b, h, s, d = shape
            if not (1 <= b <= 4096 and 1 <= h <= 4096 and 16 <= s <= 8192):
                raise ToolError("bad_shape",
                                f"attention 探针要求 1<=B<=4096, 1<=H<=4096, 16<=S<=8192"
                                f"（S² 的工作量: S=8192 已是被测任务的 4 倍), 当前 {shape!r}")
            if not (16 <= d <= 256 and d & (d - 1) == 0):
                # 探针用 tl.arange(0, D) 与 (BLOCK_M, D) 的累加器: D 必须是 2 的幂。
                raise ToolError("bad_shape",
                                f"attention 探针的 D 必须是 [16, 256] 内的 2 的幂, 当前 {d!r}")
        elif any(not 16 <= value <= 65536 for value in shape):
            raise ToolError("bad_shape",
                            f"shape 必须是 {want} 个 [16, 65536] 的整数, 当前 {shape!r}")
        dims = list(shape)
        if backend != "ssh":
            # 其它后端的计时不是这张卡的; 尺寸一上 4096 纯 Python 也跑不动。
            # **不静默换成小核** —— 那会把「没量到这张卡」说成合格。
            raise ToolError("preflight_shape_backend",
                            "shape 探针只支持 backend=ssh: 其它后端量不到这张卡的 CUDA event 通道")
    (backend_obj, resolution), _build_log = _quiet(
        _make_backend, backend, host=host, port=port, user=user, key_path=key_path,
        work_dir=work_dir, timeout=timeout, persistent=persistent)
    notes = ["measures the machine around one reference kernel, not any candidate"]
    if dims is None:
        bench = calibration.backend_bench_fn(backend_obj)
        call = lambda: bench(calibration.known_good_kernel, calibration.default_inputs())
        channel = f'backend:{resolution.get("resolved") or backend}'
        notes.append("small reference kernel: this exercises the timing channel, not the GPU; "
                     "pass shape=[M,N,K] to probe a real workload size")
    else:
        # 本地只占位拿 shape/dtype: SSH 侧按 spec 在远端生成数据, 不传几十 MB。
        # 标量（`causal`）走 `_serialize_inputs` 的 scalars 通道按值直传。
        inputs, label = _shape_probe_plan(op, dims, dtype, causal)
        # 每个 episode 内部采 `repeat` 个样本, 再用**各 episode 的中位**算跨轮散布:
        # 与 L3 的做法一致 (真机一次 L3 是 30~46 样本)。reps=5 且每个 episode 只采
        # 1 个样本时, iqr 只是 5 个点上的顺序统计量, 会把手抖放大成 6% 的判决带宽
        # —— 2026-11-07 真机现场就是这个数 (真值 ~0.1%)。
        bench = calibration.source_bench_fn(backend_obj, probe_source(op),
                                            warmup=2, repeat=repeat)
        call = lambda: bench(inputs)
        channel = f'{label}-cuda-event'
        notes.append("the remote runner times the kernel with CUDA events and flushes L2 per run")
    if backend in ("fake", "mock"):
        notes.append("test_only: 这个后端的计时来自模拟或控制机 CPU, 不构成真机窗口证据")
    try:
        receipt, _log = _quiet(
            run_preflight, call, reps=reps, planned_effect_pct=planned_effect_pct,
            channel=channel, identity=dict(resolution, backend=backend, shape=dims),
            notes=notes)
    except ValueError as error:
        raise ToolError("preflight_refused", str(error)) from error
    receipt["test_only"] = backend in ("fake", "mock")
    return receipt


def reconcile_usage(run_dir, arm_a_path, *, plugin_arm='B') -> dict:
    """对账两臂的模型计量: B 读插件账本, A 读声明的 arm log。

    这里只做**读数与判定**, 不推断费用、不把缺失计数当 0、也不因为两边数字接近就
    升级成「同口径」。`sampling_comparability` 要求采样身份逐项**已记录且相等**,
    未记录的字段会以 `unrecorded_sampling` 点名。
    """
    from ..decide.arm_metering import (MeteringError, compare, normalize_arm_document,
                                       normalize_host_ledger)
    run = Path(run_dir)
    policy_path, ledger_path = run/'host-task-policy.json', run/'host-model-budget.json'
    arm_path = Path(arm_a_path)
    for path in (policy_path, ledger_path, arm_path):
        if not path.is_file():
            raise ToolError('usage_input', f'缺少对账输入: {path}')
    try:
        policy = json.loads(policy_path.read_text(encoding='utf-8'))
        state = json.loads(ledger_path.read_text(encoding='utf-8'))
        document = json.loads(arm_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise ToolError('usage_input', f'对账输入读不出来: {error}') from error
    try:
        receipt = compare(normalize_arm_document(document),
                          normalize_host_ledger(state, policy, arm=plugin_arm))
    except MeteringError as error:
        raise ToolError(error.code, error.detail) from error
    receipt['inputs'] = dict(run_dir=str(run), policy=str(policy_path),
                             ledger=str(ledger_path), arm_a=str(arm_path))
    receipt['ok'] = True
    return receipt


def capabilities() -> dict:
    """不跑活、不联网：只回答「这台机器上这个壳能做什么」。

    SSH / 本地执行后端**只报存在性与是否接线**，不尝试连接 —— 探测会在无人值守的
    场景里变成一次真实的网络动作，而调用方只是想知道「能不能」。
    """
    from .. import __version__
    from ..decide.bench_cases import known_layouts
    from ..decide.correctness_profiles import (ATTENTION_FAMILY, FUSED_LN_RESIDUAL_FAMILY,
                                               OPERATORS, PROFILE_VERSION)

    def _spec(name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            return False

    return {
        "ok": True,
        "version": __version__,
        "tool_version": TOOL_VERSION,
        "backends": {
            "fake": {"wired": True, "requires": ["numpy"], "test_only": True,
                     "notes": ["始终离线；CPU 执行 Python 候选；固定性能值不具备 GPU 验收资格",
                               "run 默认使用 fake 模型；可用 fake.llm=live 检查真实模型接线"]},
            "mock": {"wired": True, "requires": [],
                     "notes": ["控制机 numpy；最快，也是唯一默认"]},
            "local": {"wired": True, "requires": [],
                      "limits": ["没有执行侧输入回读协议 ⇒ 带正确性套件的任务 L2 判 unknown"
                                 "（那不是「候选错了」）；要全量 L2 用 mock 或真机"]},
            "ssh": {"wired": True,
                    "requires": ["host（~/.ssh/config 别名即可）"],
                    "env_present": any(k.startswith("REMOTE_SSH_")
                                       for k in os.environ),
                    "limits": ["本函数不做连接探测；连接失败在 verify 时以 ToolError 报出"]},
        },
        "runtime": {
            "python": os.sys.version.split()[0],
            "numpy": bool(_spec("numpy")),
            "torch_installed": bool(_spec("torch")),
            "triton_installed": bool(_spec("triton")),
            "nvidia_smi": bool(shutil.which("nvidia-smi")),
        },
        "kernel_types_with_input_layout": list(known_layouts()),
        # `operators` 是**二维 axis 生成器**的词表，不是支持清单。别把两者混读：
        # 2026-11-08 一次会话开场看到这里只有四个算子，就以为 attention 不支持，
        # 而它当时已经有 profile、参考、8 条 case 与一套 L2/L3 判定。
        "correctness_profiles": {"version": PROFILE_VERSION,
                                 "operators": list(OPERATORS),
                                 "operators_scope": "二维 axis 生成器覆盖的算子；不是支持清单",
                                 "families": {"attention": dict(ATTENTION_FAMILY),
                                              "fused_layernorm_residual":
                                                  dict(FUSED_LN_RESIDUAL_FAMILY)}},
        "notes": [
            "capabilities 不做网络探测（含 SSH）",
            "backend=ssh 需要显式 host；本层不提供 auto 降级（「以为在真机上、其实在控制机」是这里最贵的误读）",
        ],
    }


# ── 动作 2.5: 给一份已有的核，自动修 ──────────────────────────────────────
def repair(run_dir: str | Path, code: str, *, max_rounds: int = 2,
           stop_after: Any = None, backend: str = "mock",
           host: Optional[str] = None, port: Optional[int] = None,
           user: Optional[str] = None, key_path: Optional[str] = None,
           work_dir: Optional[str] = None, timeout: int = 300,
           persistent: bool = False, target_gflops: float = 0.0,
           candidate_origin: Optional[str] = None, wall_clock_s: float = 1800.0,
           max_llm_calls: int = 24, oracle_total: int = 20) -> dict:
    """「判 → 读处方 → 调 Fixer → 再判」最多 N 轮。**判据全在既有模块里**。

    与 `verify` 的分工：`verify` 只判一次；这里会把处方喂给内层模型去改。
    与 `operator_run` 的分工：那条路是**从零生成**；这里是**从一份给定的核开始**。

    `candidate_origin` 描述**传进来的那份**源码。回包另有 `final_candidate_origin`：
    只要有哪一轮真的改了源码（`action="fixed"`），最终那份就是内层 Fixer 写的 ——
    这一格是**派生的**，不由调用方声明。
    """
    _s5_development_only()
    _legacy_task_only(run_dir)
    from ..execute.task_budget import TaskBudgetConfig, snapshot, stopped_reason
    from ..execute.task_clock import monotonic
    from ..app.tool_calls import repair_context
    from ..execute.budget import BudgetLedger, BudgetPlan
    from ..execute.s5_inner_budget import production_budget
    from ..execute.task_cancellation import partial_artifacts, publish_json
    try:
        policy = TaskBudgetConfig(wall_clock_s=wall_clock_s,
            max_llm_calls=max_llm_calls, oracle_total=oracle_total)
    except (TypeError, ValueError) as exc:
        raise ToolError('invalid_task_budget', str(exc)) from exc
    if type(max_rounds) is not int or max_rounds < 1:
        raise ToolError('invalid_task_budget', 'max_rounds must be a positive integer')
    context = repair_context(run_dir)
    if context and context['task_budget'] != policy.as_dict():
        raise ToolError('invalid_task_budget', 'Repair worker budget differs from its supervisor.')
    started_monotonic = context['started_monotonic'] if context else monotonic()
    deadline = context['deadline'] if context else started_monotonic + policy.wall_clock_s
    problem = _llm_problem()
    if problem:
        raise ToolError("llm_not_configured", problem)
    if not isinstance(code, str) or not code.strip():
        raise ToolError("code_required", "repair 需要候选源码文本")
    origin = _origin_of(candidate_origin)
    run = Path(run_dir)
    decl_path = run / "declaration.json"
    if not decl_path.is_file():
        raise ToolError("run_not_prepared", f"{decl_path} 不存在；先 prepare")
    declaration = json.loads(decl_path.read_text(encoding="utf-8"))
    recorded = json.loads((run / "contract.json").read_text(encoding="utf-8"))["hash"]
    task = build_task(declaration)
    try:
        contract = freeze_from_task(task)
    except ValueError as exc:
        raise ToolError("contract_refused", str(exc)) from exc
    if contract.hash != recorded:
        return {"ok": False, "error": "contract_drift",
                "detail": "判据在回合之间被改过 —— 拒绝在旧回合上继续",
                "recorded_hash": recorded, "current_hash": contract.hash,
                **tool_context(backend, model_mode='live')}

    level = _level_of(stop_after)
    with RoundEvidence(run, "repair", declaration, contract, tool_version=TOOL_VERSION,
                       stop_after=level.value if level else None) as evidence:
        run = evidence.run
        source = evidence.source("candidate.py", code)
        evidence.binding["source_hash"] = source["source_hash"]
        evidence.write_json('task-budget.json', dict(task_budget=policy.as_dict(),
            started_monotonic=started_monotonic, deadline=deadline,
            enforcement=('supervised_local_worker' if context else 'cooperative_library_deadline')))
        if context:
            publish_json(Path(context['directory'])/'repair-evidence.json', dict(
                invocation_id=context['invocation_id'], evidence=str(evidence.path)))
        validation_budget = BudgetLedger(BudgetPlan(oracle_total=policy.oracle_total,
                                                    wall_clock_s=policy.wall_clock_s))
        budget_stop = None

        def persist_validation_budget():
            value = validation_budget.as_dict()
            caps = validation_budget.caps
            value.update(unit='candidate_verification_dispatch_per_input_group',
                reservation_policy='pre_dispatch_non_refundable',
                raw_gpu_launch_cap=False, gpu_seconds_cap=False,
                remaining=dict(non_confirm=max(0, policy.oracle_total-caps['confirm']
                                                -validation_budget.spent_all),
                               confirm=max(0, caps['confirm']-validation_budget.spent['confirm'])))
            _write_json(evidence.path/'verification-budget.json', value)

        def budget_audit():
            value = snapshot(evidence.path, policy, started_monotonic)
            value['wall_clock']['scope'] = ('supervised_local_worker_lifetime' if context
                                            else 'repair_library_cooperative_deadline')
            return value

        def reserve_verification(index, current):
            nonlocal budget_stop
            phase = 'explore' if index == 1 else 'refine'
            if monotonic() >= deadline:
                budget_stop = 'budget_wall_clock'
                validation_budget.oracle_refuse(phase)
                persist_validation_budget()
                return False
            if index == 1 and pre_reserved:
                pre_reserved.clear()
                return True
            n = len(inputs) if isinstance(inputs, list) else 1
            if not validation_budget.oracle_may(phase, n):
                budget_stop = 'verification_budget_exhausted'
                validation_budget.oracle_refuse(phase)
                persist_validation_budget()
                return False
            validation_budget.oracle_spend(phase, n)
            persist_validation_budget()
            return True

        persist_validation_budget()

        def retain_verification(index, current, results, log):
            with RoundEvidence(evidence.path, "step", declaration, contract,
                               tool_version=TOOL_VERSION,
                               stop_after=level.value if level else None) as step:
                retained = step.source("candidate.py", current)
                step.binding["source_hash"] = retained["source_hash"]
                step.write_text("execution.log", log)
                l3 = next((r for r in results if r.level is VerificationLevel.L3_PERFORMANCE), None)
                report = step.write_text("report.md", build_report(artifact=KernelArtifact(
                    code=current, verification_results=results,
                    metadata=tool_context(backend, model_mode='live'),
                    benchmark_data=dict(l3.metrics or {}) if l3 else {}), task=task))
                payload = dict(round=index, **retained, **tool_context(backend, model_mode='live'),
                    levels={r.level.value: r.passed for r in results},
                    evidence=str(step.path / "result.json"), report=report,
                    log=str(step.path / "execution.log"))
                step.finish({**payload, "verdicts": verification_data(results)})
                return payload

        inputs = contract.cases[0].materialize()
        pre_reserved = set()
        backend_log, resolution, llm, runtime_error = '', None, None, None

        def stopped_result(reason):
            refusal = VerificationResult(level=VerificationLevel.L1_COMPILE, passed=False,
                message='修复验证未派发：' + reason,
                metrics=dict(budget_dispatch=False, budget_reason=reason))
            return dict(code=code, passed=False, rounds=[], results=[refusal], log='',
                        initial_full_l2_passed=None, final_full_l2_passed=None,
                        candidate_records=[], selected_source_hash=code_hash(code),
                        stopped_reason=reason)

        index = evidence.index
        if not reserve_verification(1, code):
            result = stopped_result(budget_stop)
        else:
            # Reserve before backend construction: resolving SSH is already an
            # external action, not a free step preceding the verification gate.
            pre_reserved.add(1)
            try:
                from ..generate.fixer import FixerAgent
                from ..generate.prompt_manager import PromptManager
                from ..execute.llm_client import LLMClient
                from ..generate.repair_loop import repair as run_repair
                ledger = production_budget(policy.llm_limits(min(120., policy.wall_clock_s)),
                                           evidence.path/'model-journal', deadline=deadline)
                llm = LLMClient(max_tokens=4096, timeout=min(120., policy.wall_clock_s),
                                call_budget=ledger)
                if monotonic() >= deadline:
                    budget_stop = 'budget_wall_clock'
                    result = stopped_result(budget_stop)
                else:
                    (backend_obj, resolution), backend_log = _quiet(_make_backend, backend,
                        host=host, port=port, user=user, key_path=key_path, work_dir=work_dir,
                        timeout=timeout, persistent=persistent)
                    verifier = Verifier(backend_obj)
                    install_level_recorder(verifier, evidence.path)
                    prompt_manager = PromptManager(
                        generation_instructions=backend_obj.generation_instructions)
                    result = run_repair(code, task, inputs, verifier,
                        FixerAgent(llm=llm, prompt_manager=prompt_manager),
                        max_rounds=max_rounds, target_gflops=float(target_gflops),
                        stop_after=level, on_verification=retain_verification,
                        before_verification=reserve_verification)
            except Exception as exc:
                from ..decide.measurement import safe_error_text
                runtime_error = dict(error=type(exc).__name__, detail=safe_error_text(exc))
                result = stopped_result('repair_failed')
        audit = budget_audit()
        budget_stop = (stopped_reason(audit) or budget_stop or result.get('stopped_reason'))
        if budget_stop:
            result['passed'] = False

        log = backend_log + (result.get("log") or "")
        final_code = result["code"]
        evidence.source("final.py", final_code)
        evidence.binding["final_source_hash"] = code_hash(final_code)
        evidence.write_text("execution.log", log)
        final_results = result.get("results", [])
        l3 = next((r for r in final_results if r.level is VerificationLevel.L3_PERFORMANCE), None)
        report_path = evidence.write_text("report.md", build_report(artifact=KernelArtifact(
            code=final_code, verification_results=final_results,
            metadata=tool_context(backend, model_mode='live'),
            benchmark_data=dict(l3.metrics or {}) if l3 else {}), task=task))
        # 派生而非声明：`action="fixed"` 的那一轮把 `current` 换成了 Fixer 的产出
        # （`repair_loop.repair` 的 anti-fake-fix 已经挡掉「原样返回/只改注释」）。
        final_origin = ("inner_fixer"
                        if any(e.get("action") == "fixed" for e in result["rounds"])
                        else origin)
        payload = {
            "ok": runtime_error is None,
            **tool_context(backend, model_mode='live'),
            "run_id": run.name,
            "attempt": index,
            "passed": result["passed"],
            "candidate_origin": origin,
            "final_candidate_origin": final_origin,
            "initial_full_l2_passed": result.get("initial_full_l2_passed"),
            "full_l2_passed": result.get("final_full_l2_passed"),
            "generation_spec": generation_spec_identity(),
            "rounds": result["rounds"],
            "candidate_records": result.get("candidate_records", []),
            "selected_source_hash": result.get("selected_source_hash"),
            "final_code": str(run / f"repair-{index}.py"),
            "log": str(run / f"repair-{index}.log"),
            "backend": backend,
            "backend_resolution": resolution,
            "contract_hash": contract.hash,
            "evidence": str(evidence.path / "result.json"),
            "report": report_path,
            "task_budget": policy.as_dict(),
            "budget": audit,
            "stopped_reason": budget_stop,
            "partial_artifacts": dict(partial_artifacts(run), repair_evidence=str(evidence.path)),
            "budget_enforcement": ('supervised_local_worker' if context
                                   else 'cooperative_library_deadline'),
            **source,
        }
        if runtime_error:
            payload.update(runtime_error)
        tracker = getattr(llm, 'cost_tracker', None)
        if tracker is not None and hasattr(tracker, 'summary'):
            payload['llm_cost'] = tracker.summary()
        evidence.finish({**payload, "verdicts": verification_data(final_results)},
                        legacy_payload=payload, legacy_log=log, legacy_source=final_code)
        # Preserve complete source sites and algebra witnesses on disk, while the
        # MCP response carries compact bindings and actionable repair obligations.
        compact = json.loads(json.dumps(payload, ensure_ascii=False))
        compact['candidate_records'] = [candidate_summary(r) for r in compact['candidate_records']]
        for row in compact["rounds"]:
            row["prescription"] = str(row.get("prescription") or "")[:240]
            row["fix_summary"] = str(row.get("fix_summary") or "")[:160] or None
            if 'candidate_records' in row:
                row['candidate_record_hashes'] = [r['record_hash'] for r in row.pop('candidate_records')]
            diagnosis = row.get("diagnosis")
            if isinstance(diagnosis, dict):
                row["diagnosis"] = {
                    key: diagnosis[key] for key in ("schema", "state", "reason", "hash", "bindings")
                    if key in diagnosis}
                row["diagnosis"]["findings"] = [
                    {key: finding[key] for key in ("kind", "repair_obligations") if key in finding}
                    for finding in diagnosis.get("findings", [])]
        return compact


# ── 动作 3: 整段自动化（Phase 3）───────────────────────────────────────────
SUMMARY_FILE = "conductor.json"
CONFIG_FILE = "autonomous.json"


def _llm_problem() -> Optional[str]:
    """无人值守那条路要**内层** LLM（`Conductor` 自己设计/写核），与外层宿主无关。

    密钥只认两种来源：环境变量，或 `LLM_API_KEY_FILE` 指向的文件（引用，不是值）。
    MCP server 的 env 被宿主洗过（`*KEY*` 全没了），所以要真跑，宿主得在它的 MCP
    配置行里显式传 `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY_FILE`。
    """
    if not os.environ.get("LLM_BASE_URL"):
        return "缺 LLM_BASE_URL（形如 http://host:port/v1）"
    if not (os.environ.get("LLM_API_KEY") or os.environ.get("LLM_API_KEY_FILE")):
        return "缺 LLM_API_KEY 或 LLM_API_KEY_FILE（建议用文件引用，别把值写进配置）"
    return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (TypeError, ValueError, OSError):
        return False


def run_autonomous(declaration: dict, run_dir: str | Path, *, max_rounds: int = 1,
                   backend: str = "mock", host: Optional[str] = None,
                   port: Optional[int] = None, user: Optional[str] = None,
                   key_path: Optional[str] = None, work_dir: Optional[str] = None,
                   timeout: int = 300, persistent: bool = False,
                   target_gflops: float = 0.0, wall_clock_s: float = 1800.0,
                   max_llm_calls: int = 24, oracle_total: int = 20,
                   fake: Optional[dict] = None) -> dict:
    """起一次**无人值守**的整段运行（`Conductor` 自己设计/写核/修/搜索）。

    与 `verify` 的分工：`verify` 判**外层给的**候选；这里把整条循环交回本仓。
    进程分离（`start_new_session`）的理由见 `operator_agent/app/autonomous.py` 的 docstring。
    """
    _s5_development_only()
    _legacy_task_only(run_dir)
    context = {}
    if backend not in _BACKENDS:
        raise ToolError('backend_not_wired', f'Unknown backend: {backend}')
    if backend == 'fake':
        from ..execute.fake_backend import FakeConfig, execution_context
        from ..execute.fake_backend.model import FIXTURES
        try:
            fake_config = FakeConfig.from_dict(fake)
        except (TypeError, ValueError) as exc:
            raise ToolError('invalid_fake_config', str(exc)) from exc
        if any(value is not None for value in (host, port, user, key_path, work_dir)) or persistent:
            raise ToolError('invalid_fake_config', 'fake backend does not accept remote connection options')
        task = build_task(declaration)
        if task.target.dsl != 'python':
            raise ToolError('fake_python_only', 'fake backend executes CPU Python candidates; use dsl=python')
        if task.target.required_caps - _backend('fake').capabilities:
            raise ToolError('fake_capability_missing', 'fake backend cannot satisfy real GPU requirements')
        if fake_config.llm == 'fake' and task.source.kernel_type not in FIXTURES:
            raise ToolError('fake_fixture_missing', 'fake LLM fixtures support relu and matmul')
        fake = fake_config.as_dict()
        context = dict(execution_context=execution_context(fake_config))
    elif fake is not None:
        raise ToolError('invalid_fake_config', 'fake options require backend=fake')
    from ..execute.task_budget import TaskBudgetConfig
    try:
        task_budget = TaskBudgetConfig(wall_clock_s=wall_clock_s,
            max_llm_calls=max_llm_calls, oracle_total=oracle_total).as_dict()
    except (TypeError, ValueError) as exc:
        raise ToolError('invalid_task_budget', str(exc)) from exc
    problem = None if backend == 'fake' and fake['llm'] == 'fake' else _llm_problem()
    if problem:
        raise ToolError("llm_not_configured", problem)
    from ..execute.task_cancellation import launch_identity
    run = Path(run_dir).resolve()
    if any((run/name).exists() for name in (CONFIG_FILE, SUMMARY_FILE,
                                           'worker-result.json', 'launch.json')):
        raise ToolError('run_already_started', 'Use a fresh run directory; existing runs are preserved.')
    run.mkdir(parents=True, exist_ok=True)
    invocation_id = uuid.uuid4().hex
    try:
        launch_identity(run, invocation_id, kind='autonomous', backend=backend,
                        task_budget=task_budget, **context)
    except FileExistsError as exc:
        raise ToolError('run_already_started', 'This run directory already has a launch identity.') from exc
    try:
        prepared = prepare(declaration, run)
    except Exception:
        # prepare 失败时**没有任务**：账本不存在，就没有可以记下的转移。
        # 伪造一条「已准备然后失败」的流，会让这个目录以后永远说不清自己是什么。
        raise
    config = {
        "backend": backend, "max_rounds": int(max_rounds),
        "host": host, "port": port, "user": user, "key_path": key_path,
        "work_dir": work_dir, "timeout": int(timeout),
        "persistent": bool(persistent), "target_gflops": float(target_gflops),
        "task_budget": task_budget,
        "invocation_id": invocation_id,
        "started_at": time.time(),
        **context,
    }
    if backend == 'fake':
        config['fake'] = fake
    _write_json(run / CONFIG_FILE, config)
    log_path = run / "autonomous.log"
    from ..execute.exec_guard import start_background_worker
    # 这里**不再**预先写一条「running / starting」：账本里的 `task_prepared` 已经回答
    # 了「准备好了、还没开跑」，监督者一起来就会自己接上 `task_activated`。
    try:
        proc = start_background_worker(
            [sys.executable, "-m", "operator_agent.app.autonomous", str(run)],
            cwd=str(Path(__file__).resolve().parents[2]), log_path=log_path,
        )
    except OSError as error:
        _record_task_end(run, invocation_id, 'failed', reason='supervisor_start_failed',
                         error='worker_start_failed', detail=str(error))
        raise
    from ..execute.task_cancellation import publish_json
    from ..execute.exec_guard import process_identity
    publish_json(run/'supervisor.json', dict(invocation_id=invocation_id, pid=proc.pid,
                                           pid_identity=process_identity(proc.pid)))
    return {
        "ok": True, "phase": "running", "pid": proc.pid, "run_id": run.name,
        "run_dir": str(run), "contract_hash": prepared["contract_hash"],
        "backend": backend, "max_rounds": int(max_rounds),
        "task_budget": task_budget,
        "invocation_id": invocation_id,
        "journal": str(run/'journal.jsonl'), "log": str(log_path),
        "hint": "用 operator_status 轮询；不要用 verify 的 run_dir 混用",
        **context,
    }


def _call_directories(run: Path) -> list:
    """`tool-calls/` 下已发出过的调用目录。

    用 `launch.json` 认，不用状态文件认：状态文件已经不是权威了，
    拿它当「这条调用存在过」的凭据，会让下一次迁移又把这条判据搬一遍。
    """
    directory = run/'tool-calls'
    return sorted(p for p in directory.glob('*')
                  if p.is_dir() and (p/'launch.json').is_file())


def _journal_call_status(directory: Path) -> Optional[dict]:
    """一次调用在任务账本里的样子，拼成与旧状态文件同形的 dict。

    账本里没有这条 invocation 时返回 None —— 那种目录要么是账本之前的老调用
    （仍可读它当年写的状态文件），要么根本没开始过。两种都不该由这里替它编。"""
    from ..execute.task_cancellation import read_json
    launch = read_json(directory/'launch.json') or {}
    invocation = launch.get('invocation_id') or directory.name
    from ..core import task_journal
    view = task_journal.invocation_view(directory, invocation)
    if view is None:
        return None
    status = dict(kind='tool_call', phase=view['phase'], stage=view['stage'],
                  tool=view.get('tool') or launch.get('tool'),
                  invocation_id=view.get('invocation') or invocation,
                  run_dir=str(task_journal.task_root(directory)),
                  pid=view.get('pid'), pid_identity=view.get('pid_identity'),
                  started_at=view.get('started_at'), finished_at=view.get('finished_at'),
                  stopped_reason=view.get('stopped_reason') or '',
                  cleanup=view.get('cleanup'), journal_phase=view['journal_phase'],
                  result_state=view.get('result') or 'none',
                  result_reason=view.get('result_reason') or '')
    if isinstance(launch.get('execution_context'), dict):
        status['execution_context'] = launch['execution_context']
    return status


def _record_task_end(run, invocation_id, outcome, *, reason='', **facts):
    """记下这次运行的终态。**不抛给调用方**：调用方正在报它自己的错。"""
    from ..core import task_journal as journal
    try:
        if not journal.read_events(run):
            return None
        return journal.append(run, 'task_closed', invocation=invocation_id,
                             outcome=outcome, reason=reason,
                             **{key: value for key, value in facts.items() if value is not None})
    except (journal.JournalError, ValueError, OSError):
        return None


def _live_budget(run: Path, view: dict):
    """预算的**现值**：从预算账本现算，不从任何状态文件抄。

    墙钟要跨进程算：事件里带着 `started_monotonic`（监督者那一侧的单调钟）与
    `started_at`（epoch）。两个时钟各自都不能跨进程比较，这个组合可以。
    """
    from ..execute.task_budget import TaskBudgetConfig, snapshot
    from ..execute.task_cancellation import read_json
    try:
        config = read_json(run/'autonomous.json') or {}
        if 'task_budget' not in config:
            return None          # 没有冻结的预算策略 ⇒ 没有可报的预算（不拿默认值凑）
        policy = TaskBudgetConfig.from_dict(config['task_budget'])
        started_monotonic = view.get('started_monotonic')
        started_at = view.get('started_at')
        if type(started_monotonic) not in (int, float) or type(started_at) not in (int, float):
            return None
        now = started_monotonic + max(0.0, time.time() - started_at)
        return snapshot(run, policy, started_monotonic, now=now)
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _journal_task_status(run: Path) -> Optional[dict]:
    """整段运行（autonomous）在任务账本里的样子，拼成与旧状态文件同形的 dict。

    账本里没有这次运行就返回 None —— 那要么是账本之前的老目录，要么根本没开始过。
    """
    from ..core import task_journal
    view = task_journal.task_view(run)
    if view is None:
        return None
    status = dict(kind=None, phase=view['phase'], stage=view.get('stage'),
                  invocation_id=view.get('invocation') or '',
                  pid=view.get('pid'), pid_identity=view.get('pid_identity'),
                  started_at=view.get('started_at'), finished_at=view.get('finished_at'),
                  duration_s=view.get('duration_s'),
                  backend=view.get('backend'), max_rounds=view.get('max_rounds'),
                  task_budget=view.get('task_budget'),
                  stopped_reason=view.get('reason') or '',
                  error=view.get('error'), detail=view.get('detail'),
                  cleanup=view.get('cleanup'),
                  generation_evidence=view.get('generation_evidence'),
                  execution_context=view.get('execution_context'))
    if view['terminal'] and isinstance(view.get('budget'), dict):
        # 结束时的现值**冻在事件里**（监督者收工那一刻算的），读的人不重算。
        status['budget'] = view['budget']
    else:
        status['budget'] = _live_budget(run, view)
    if (status['duration_s'] is None and status['finished_at'] is not None
            and status['started_at'] is not None):
        status['duration_s'] = round(status['finished_at'] - status['started_at'], 3)
    return status


def _no_journal_answer(run: Path) -> dict:
    """没有任务账本时怎么回答：**两种都不是「还在跑」**，而且都不猜。"""
    legacy = [name for name in ('autonomous.json', 'conductor.json', 'worker-result.json')
              if (run/name).exists()]
    if legacy:
        return dict(ok=False, error='legacy_run_directory', run_dir=str(run), read_only=True,
                    detail=('这个目录没有任务账本（journal.jsonl）—— 它在账本之前创建。'
                            '用产生它的那个版本查它的运行状态；本版本能读回的是契约、'
                            '两本账和交付源码（见 operator_resume）。'))
    return dict(ok=False, error='run_not_started',
                detail=f"{run/'journal.jsonl'} 不存在（这个目录还没被 prepare 过）")


def autonomous_status(run_dir: str | Path, invocation_id: Optional[str] = None) -> dict:
    """Query autonomous/foreground invocation; missing owners are explicitly lost."""
    run = Path(run_dir)
    from ..execute.host_task_budget import HostTaskBudget, host_task_enabled
    if invocation_id is None and host_task_enabled(run):
        from .host_workflow import prepared_task
        _, _, contract = prepared_task(run)
        budget = HostTaskBudget(run, contract.hash).snapshot()
        calls = _call_directories(run)
        latest = next((record for record in reversed(budget['records'])
            if record['status']=='completed' and record.get('source') and record.get('report')
            and record.get('accepted_by_supervisor') is not False), None)
        return dict(ok=True, workflow='host', phase=budget['phase'], run_dir=str(run),
            host_task=budget, calls=[autonomous_status(p) for p in calls],
            last_correct_candidate=budget.get('last_correct_candidate'),
            baseline=budget.get('baseline'), best_candidate=budget.get('best_candidate'),
            final_selection=budget.get('final_selection'),
            candidates=budget.get('candidates'), noise_band=budget.get('noise_band'),
            selection=budget.get('selection'), formal_delivery=budget.get('formal_delivery'),
            latest_verification=({key:latest.get(key) for key in ('id','source_hash','source','report',
                'evidence','full_l2_passed','validity')} if latest else None),
            context_owner='host', model_calls='host_owned_see_verified_model_budget_receipt')
    from ..execute.task_cancellation import read_json, cancellation, partial_artifacts, valid_invocation
    if invocation_id is not None:
        if not valid_invocation(invocation_id):
            return dict(ok=False, error='invocation_mismatch', detail='Invalid invocation identity.')
        if (read_json(run/'launch.json') or {}).get('invocation_id') != invocation_id:
            run = run/'tool-calls'/str(invocation_id)
            if (read_json(run/'launch.json') or {}).get('invocation_id') != invocation_id:
                return dict(ok=False, error='invocation_mismatch', detail='No matching invocation.')
    from ..core.task_journal import JournalError
    status = None
    try:
        if invocation_id is None:
            # 没点名要看哪一次：这里是一个装着多次调用的运行目录。列出来，不去
            # 冒充「整段运行」的相位 —— 那两个问题不一样。
            calls = _call_directories(run)
            if calls:
                return dict(ok=True, phase='tool_calls',
                            calls=[autonomous_status(p) for p in calls])
        if (read_json(run/'launch.json') or {}).get('kind') == 'tool_call':
            status = _journal_call_status(run)      # 一次调用：Job 层
        if status is None:
            status = _journal_task_status(run)      # 整段运行：任务层
    except JournalError as error:
        return dict(ok=False, error='task_journal_unusable',
                    detail=f'{error.code}: {error.detail}', run_dir=str(run))
    if status is None:
        return _no_journal_answer(run)
    if not status.get('pid'):
        owner = read_json(run/'supervisor.json') or {}
        if owner.get('invocation_id') == status.get('invocation_id'):
            status.update(pid=owner.get('pid'), pid_identity=owner.get('pid_identity'))
    phase = str(status.get("phase") or "unknown")
    alive = (_pid_alive(status.get("pid")) if phase == "running" and status.get('pid') else None)
    if alive and status.get('pid_identity'):
        from ..execute.exec_guard import process_identity
        actual = process_identity(status['pid'])
        if actual and actual != status['pid_identity']:
            alive = False  # reused PID is not this task's owner
    if phase == "running" and alive is False:
        # 进程没了但没写终态：**不许读成"还在跑"**，也不许猜成"失败" —— 单独一档。
        phase = "lost"
    summary = None
    if (run / SUMMARY_FILE).is_file():
        summary = json.loads((run / SUMMARY_FILE).read_text(encoding="utf-8"))
    result = None
    if status.get('kind') == 'tool_call' and (run / 'result.json').is_file():
        try:
            result = json.loads(json.loads(
                (run / 'result.json').read_text(encoding='utf-8'))['content'][0]['text'])
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            result = None
    tail = []
    log_path = run / ("execution.log" if status.get('kind') == 'tool_call' else "autonomous.log")
    if log_path.is_file():
        try:
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
        except OSError:
            tail = []
    return {
        "ok": True, "phase": phase, "process_alive": alive,
        "invocation_id": status.get('invocation_id'), "stage": status.get('stage'),
        "pid": status.get("pid"), "started_at": status.get("started_at"),
        "finished_at": status.get("finished_at"),
        "duration_s": status.get("duration_s"),
        "backend": status.get("backend"), "max_rounds": status.get("max_rounds"),
        "execution_context": status.get('execution_context') or (summary or {}).get('execution_context'),
        "error": status.get("error"), "detail": status.get("detail"),
        "generation_evidence": (status.get('generation_evidence')
                                or (summary or {}).get('generation_evidence')),
        "task_budget": status.get('task_budget') or (summary or {}).get('task_budget'),
        "budget": status.get('budget') or (summary or {}).get('budget'),
        "stopped_reason": status.get('stopped_reason') or (summary or {}).get('stopped_reason'),
        "cleanup": status.get('cleanup') or (dict(status='cleanup_unknown',
            remote_execution='unknown', provider_call='unknown') if phase == 'lost' else None),
        "partial_artifacts": status.get('partial_artifacts') or partial_artifacts(run),
        "cancellation": status.get('cancellation') or cancellation(run, status.get('invocation_id')),
        "summary": summary, "result": result, "log_tail": tail,
    }


def cancel_task(run_dir: str | Path, invocation_id: str) -> dict:
    from ..execute.task_cancellation import request_cancel
    try:
        return request_cancel(run_dir, invocation_id)
    except ValueError as exc:
        raise ToolError('cancellation_refused', str(exc)) from exc
