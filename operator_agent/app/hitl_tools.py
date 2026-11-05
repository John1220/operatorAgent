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
from ..decide.failure_route import Route, classify
from ..generate.generation_spec import generation_spec_identity
from ..generate.generation_context import candidate_summary
from ..core.run_contract import code_hash
from ..app.intent_parser import IntentParser
from ..core.report import build_report
from ..decide.tool_evidence import RoundEvidence, verification_data
from ..core.execution_audit import install_level_recorder
from ..core.schemas import (ConstraintSpec, KernelArtifact, Precision, TaskSchema,
                      VerificationLevel, VerificationResult)
from ..decide.verifier import Verifier
from ..execute.fake_backend.config import tool_context

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
TOOL_VERSION = 11

#: 允许进 `declaration` 的键（§6.3 的白名单）。**没列在这里的一律拒绝** ——
#: 「先收下、以后再判断」是这一层最不能有的形状：它等于给未来开了一个没说出口的口子。
ALLOWED_KEYS = frozenset({
    "request",        # 自然语言请求（必填）
    "shapes",         # 形状分布（必填）：matmul 是 [M,N,K]，逐元素/归一化是 [M,N]
    "hardware",       # 默认 mock_gpu
    "dsl",            # 默认 python
    "precision",      # 默认取 IntentParser 的规则结果
    "min_speedup",    # 性能门槛（任务级决定，§10.2 同款）
    "required_caps",  # 执行侧必须有的能力（如 real_device）
    "framework",      # 默认 pytorch
    "notes",          # 人给的备注，只进产物不进判据
})

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

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        self.detail = detail or code
        super().__init__(f"{code}: {self.detail}")

    def as_dict(self) -> dict:
        return {"ok": False, "error": self.code, "detail": self.detail}


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
    """声明 → TaskSchema。**纯函数**：同一份声明任何时候都得到同一份判据。"""
    _check_keys(declaration)
    parsed, _ = _quiet(IntentParser().parse, declaration["request"])   # 规则分支，零 LLM
    precision = declaration.get("precision")
    constraints = replace(
        parsed.constraints,
        precision=Precision(precision) if precision else parsed.constraints.precision,
        min_speedup=float(declaration.get("min_speedup", parsed.constraints.min_speedup)),
        shapes=_shapes(declaration),
    )
    target = replace(
        parsed.target,
        hardware=str(declaration.get("hardware") or "mock_gpu"),
        dsl=str(declaration.get("dsl") or "python"),
        required_caps=_required_caps(declaration),
    )
    source = replace(
        parsed.source,
        framework=str(declaration.get("framework") or parsed.source.framework),
    )
    # The public declaration may override precision/framework after rule parsing.
    # Recompute the parser's default suite against the final task, so FP16
    # LayerNorm receives its calibrated v2 profile rather than stale FP32 v1.
    from ..decide.correctness_profiles import with_default_correctness
    constraints = replace(constraints, correctness_profile=None, correctness_suite=None,
                          reference_contract=None)
    return with_default_correctness(replace(parsed, source=source, target=target, constraints=constraints))


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
    except (TypeError, ValueError) as error:
        if isinstance(error, ToolError):
            raise
        raise ToolError('invalid_host_budget', str(error)) from error


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

    notes = []
    if contract.correctness_profile is None:
        notes.append("任务没有可信的正确性 profile（L2 会按未声明处理，可能判 unknown）")
    if str(task.target.hardware) != str(declaration.get("hardware") or "mock_gpu"):
        notes.append("hardware 与声明不一致（不应发生）")

    return {
        "ok": True,
        "run_dir": str(run),
        "run_id": run.name,
        "contract_hash": contract.hash,
        "generation_spec": generation_spec_identity(),
        "cases": [_case_summary(c) for c in contract.cases],
        "required_cases": list(contract.required_cases),
        "inputs": {name: {"shape": list(arr.shape), "dtype": str(arr.dtype)}
                   for name, arr in inputs.items()},
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


# ── 动作 2: verify ───────────────────────────────────────────────────────
def verify(run_dir: str | Path, code: str, *, verification_purpose='explore', **options) -> dict:
    from ..execute.host_task_budget import HostTaskBudget, HostTaskBudgetError, host_task_enabled
    run = Path(run_dir)
    if not host_task_enabled(run):
        if verification_purpose != 'explore':
            raise ToolError('host_task_required', 'final requires a prepared host task.')
        return _verify_candidate(run_dir, code, **options)
    from .host_workflow import prepared_task
    _, _, contract = prepared_task(run)
    budget = HostTaskBudget(run, contract.hash)
    reservation = None
    try:
        if not isinstance(code, str) or not code.strip():
            raise ToolError('code_required', 'verify needs candidate source.')
        from .tool_calls import host_context
        owner = host_context(run)
        reservation = budget.reserve(code_hash(code), purpose=verification_purpose,
            owner_invocation_id=owner['invocation_id'] if owner else None)
        result = _verify_candidate(run_dir, code, **options)
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


def _verify_candidate(run_dir: str | Path, code: str, *, stop_after: Any = None,
           backend: str = "mock", target_gflops: float = 0.0,
           host: Optional[str] = None, port: Optional[int] = None,
           user: Optional[str] = None, key_path: Optional[str] = None,
           work_dir: Optional[str] = None, timeout: int = 300,
           persistent: bool = False,
           candidate_origin: Optional[str] = None,
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
def capabilities() -> dict:
    """不跑活、不联网：只回答「这台机器上这个壳能做什么」。

    SSH / 本地执行后端**只报存在性与是否接线**，不尝试连接 —— 探测会在无人值守的
    场景里变成一次真实的网络动作，而调用方只是想知道「能不能」。
    """
    from .. import __version__
    from ..decide.correctness_profiles import OPERATORS, PROFILE_VERSION
    from ..decide.bench_cases import known_layouts

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
        "correctness_profiles": {"version": PROFILE_VERSION,
                                 "operators": list(OPERATORS)},
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
STATUS_FILE = "run-status.json"
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
    if any((run/name).exists() for name in (CONFIG_FILE, STATUS_FILE, SUMMARY_FILE,
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
        _write_json(run / STATUS_FILE, dict(phase='failed', invocation_id=invocation_id,
            stopped_reason='prepare_failed', pid=None, finished_at=time.time(), **context))
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
    _write_json(run / STATUS_FILE, dict(phase='running', stage='starting', pid=None,
        invocation_id=invocation_id, started_at=config['started_at'], task_budget=task_budget, **context))
    try:
        proc = start_background_worker(
            [sys.executable, "-m", "operator_agent.app.autonomous", str(run)],
            cwd=str(Path(__file__).resolve().parents[2]), log_path=log_path,
        )
    except OSError:
        _write_json(run / STATUS_FILE, dict(phase='failed', invocation_id=invocation_id,
            stopped_reason='supervisor_start_failed', pid=None, finished_at=time.time(), **context))
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
        "status": str(run / STATUS_FILE), "log": str(log_path),
        "hint": "用 operator_status 轮询；不要用 verify 的 run_dir 混用",
        **context,
    }


def autonomous_status(run_dir: str | Path, invocation_id: Optional[str] = None) -> dict:
    """Query autonomous/foreground invocation; missing owners are explicitly lost."""
    run = Path(run_dir)
    from ..execute.host_task_budget import HostTaskBudget, host_task_enabled
    if invocation_id is None and host_task_enabled(run):
        from .host_workflow import prepared_task
        _, _, contract = prepared_task(run)
        budget = HostTaskBudget(run, contract.hash).snapshot()
        calls = sorted((run/'tool-calls').glob('*/run-status.json'))
        return dict(ok=True, workflow='host', phase=budget['phase'], run_dir=str(run),
            host_task=budget, calls=[autonomous_status(p.parent) for p in calls],
            last_correct_candidate=budget.get('last_correct_candidate'),
            context_owner='host', model_calls='host_owned_not_observed_by_plugin')
    from ..execute.task_cancellation import read_json, cancellation, partial_artifacts, valid_invocation
    if invocation_id is not None:
        if not valid_invocation(invocation_id):
            return dict(ok=False, error='invocation_mismatch', detail='Invalid invocation identity.')
        if (read_json(run/'launch.json') or {}).get('invocation_id') != invocation_id:
            run = run/'tool-calls'/str(invocation_id)
            if (read_json(run/'launch.json') or {}).get('invocation_id') != invocation_id:
                return dict(ok=False, error='invocation_mismatch', detail='No matching invocation.')
    status_path = run / STATUS_FILE
    if not status_path.is_file():
        calls = sorted((run/'tool-calls').glob('*/run-status.json'))
        if calls and invocation_id is None:
            return dict(ok=True, phase='tool_calls', calls=[autonomous_status(p.parent) for p in calls])
        return {"ok": False, "error": "run_not_started",
                "detail": f"{status_path} 不存在（还没调用 operator_run）"}
    status = json.loads(status_path.read_text(encoding="utf-8"))
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
        "summary": summary, "log_tail": tail,
    }


def cancel_task(run_dir: str | Path, invocation_id: str) -> dict:
    from ..execute.task_cancellation import request_cancel
    try:
        return request_cancel(run_dir, invocation_id)
    except ValueError as exc:
        raise ToolError('cancellation_refused', str(exc)) from exc
