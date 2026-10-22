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
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from .backend_caps import Cap
from .contract import freeze_from_task
from .failure_route import Route, classify
from .intent_parser import IntentParser
from .report import write_report
from .schemas import (ConstraintSpec, KernelArtifact, Precision, TaskSchema,
                      VerificationLevel, VerificationResult)
from .verifier import Verifier

TOOL_VERSION = 1

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
_BACKENDS = ("mock", "local", "ssh")
_STOP_AFTER = {"L1": VerificationLevel.L1_COMPILE,
               "L2": VerificationLevel.L2_FUNCTION,
               "L3": VerificationLevel.L3_PERFORMANCE}

#: 回给外层的 metrics 只留这几项 —— 把整份 metrics 丢出去会让「读哪个数」变成猜。
_METRIC_KEYS = (
    "max_error", "max_rel_error", "matched_ratio", "compared_elements",
    "first_bad_index", "coverage", "reference_nonfinite",
    "speedup_vs_baseline", "gflops", "mode", "median_time_ms",
    "baseline_type", "baseline_time_ms", "baseline_reused", "gpu_util_pct",
    "measurement_validity", "measurement_l3_pass_refused",
    "measurement_can_gate", "measurement_validity_reasons",
    "static_guards", "ruler_broken", "benchmark_contaminated",
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
    return replace(parsed, source=source, target=target, constraints=constraints)


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
    if name == "mock":
        from .mock_gpu import MockGPU
        return MockGPU()
    from .remote_executor import RemoteExecutor
    if name == "local":
        return RemoteExecutor(mode="local")
    if not host:
        # `mode="auto"` 会一路降到 CPU —— 那正是「以为在真机上、其实在控制机」的形状。
        raise ToolError("host_required",
                        "backend=ssh 需要 host（~/.ssh/config 的别名即可）；"
                        "本层不提供 auto 降级")
    return RemoteExecutor(mode="ssh", host=host, port=port, user=user,
                          key_path=key_path, work_dir=work_dir,
                          timeout=int(timeout), persistent=bool(persistent))


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
def prepare(declaration: dict, run_dir: str | Path, *, now: Optional[float] = None) -> dict:
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
def verify(run_dir: str | Path, code: str, *, stop_after: Any = None,
           backend: str = "mock", target_gflops: float = 0.0,
           host: Optional[str] = None, port: Optional[int] = None,
           user: Optional[str] = None, key_path: Optional[str] = None,
           work_dir: Optional[str] = None, timeout: int = 300,
           persistent: bool = False,
           now: Optional[float] = None) -> dict:
    """对**这一回合**的候选源码做判定；不做候选生成、不做修复、不做搜索。"""
    if not isinstance(code, str) or not code.strip():
        raise ToolError("code_required", "verify 需要候选源码文本")
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
        }

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
    level = _level_of(stop_after)
    results, verify_log = _quiet(verifier.verify, code, task, inputs, None,
                                 float(target_gflops), None, level)
    log = backend_log + verify_log

    route = classify(results)
    eligibility = _eligibility_of(results)
    validity = _validity_of(results)
    conflict = _conflict(route, eligibility, validity)

    index = len(list(run.glob("verify-*.json"))) + 1
    stamp = now if now is not None else time.time()
    (run / f"verify-{index}.log").write_text(log, encoding="utf-8")

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
            "levels_passed": {r.level.value: r.passed for r in results},
            "speedup": (getattr(l3, "metrics", None) or {}).get("speedup_vs_baseline")
            if l3 else None,
            "gflops": (getattr(l3, "metrics", None) or {}).get("gflops") if l3 else None,
            "stop_after": level.value if level else None,
        }],
    )
    report_path = write_report(run, artifact=artifact, task=task)

    payload = {
        "ok": True,
        "run_id": run.name,
        "round": index,
        "contract_hash": contract.hash,
        "backend": backend,
        "backend_resolution": resolution,
        "stop_after": level.value if level else None,
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
        "notes": _judgement_notes(contract, getattr(l3, "metrics", None) or {}) if l3 else [],
    }
    _write_json(run / f"verify-{index}.json", {**payload, "created_at": stamp})
    return payload


# ── 动作 0（只读）: capabilities ─────────────────────────────────────────
def capabilities() -> dict:
    """不跑活、不联网：只回答「这台机器上这个壳能做什么」。

    SSH / 本地执行后端**只报存在性与是否接线**，不尝试连接 —— 探测会在无人值守的
    场景里变成一次真实的网络动作，而调用方只是想知道「能不能」。
    """
    from . import __version__
    from .correctness_profiles import OPERATORS, PROFILE_VERSION
    from .bench_cases import known_layouts

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
           persistent: bool = False, target_gflops: float = 0.0) -> dict:
    """「判 → 读处方 → 调 Fixer → 再判」最多 N 轮。**判据全在既有模块里**。

    与 `verify` 的分工：`verify` 只判一次；这里会把处方喂给内层模型去改。
    与 `operator_run` 的分工：那条路是**从零生成**；这里是**从一份给定的核开始**。
    """
    problem = _llm_problem()
    if problem:
        raise ToolError("llm_not_configured", problem)
    if not isinstance(code, str) or not code.strip():
        raise ToolError("code_required", "repair 需要候选源码文本")
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
                "recorded_hash": recorded, "current_hash": contract.hash}

    inputs = contract.cases[0].materialize()
    backend_obj, resolution = _quiet(_make_backend, backend, host=host, port=port,
                                     user=user, key_path=key_path, work_dir=work_dir,
                                     timeout=timeout, persistent=persistent)[0]
    verifier = Verifier(backend_obj)
    from .fixer import FixerAgent
    from .llm_client import LLMClient
    from .repair_loop import repair as run_repair

    index = len(list(run.glob("repair-*.json"))) + 1
    llm = LLMClient(max_tokens=4096, timeout=120)
    result = run_repair(code, task, inputs, verifier, FixerAgent(llm=llm),
                        max_rounds=int(max_rounds),
                        target_gflops=float(target_gflops),
                        stop_after=_level_of(stop_after))

    (run / f"repair-{index}.log").write_text(result.pop("log") or "", encoding="utf-8")
    final_code = result["code"]
    (run / f"repair-{index}.py").write_text(final_code, encoding="utf-8")

    for entry in result["rounds"]:
        entry["prescription"] = str(entry.get("prescription") or "")[:240]
        entry["fix_summary"] = str(entry.get("fix_summary") or "")[:160] or None
    payload = {
        "ok": True,
        "run_id": run.name,
        "attempt": index,
        "passed": result["passed"],
        "rounds": result["rounds"],
        "final_code": str(run / f"repair-{index}.py"),
        "log": str(run / f"repair-{index}.log"),
        "backend": backend,
        "backend_resolution": resolution,
        "contract_hash": contract.hash,
    }
    _write_json(run / f"repair-{index}.json", payload)
    return payload


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
                   target_gflops: float = 0.0) -> dict:
    """起一次**无人值守**的整段运行（`Conductor` 自己设计/写核/修/搜索）。

    与 `verify` 的分工：`verify` 判**外层给的**候选；这里把整条循环交回本仓。
    进程分离（`start_new_session`）的理由见 `operator_agent/autonomous.py` 的 docstring。
    """
    problem = _llm_problem()
    if problem:
        raise ToolError("llm_not_configured", problem)
    prepared = prepare(declaration, run_dir)
    run = Path(run_dir)
    config = {
        "backend": backend, "max_rounds": int(max_rounds),
        "host": host, "port": port, "user": user, "key_path": key_path,
        "work_dir": work_dir, "timeout": int(timeout),
        "persistent": bool(persistent), "target_gflops": float(target_gflops),
        "started_at": time.time(),
    }
    _write_json(run / CONFIG_FILE, config)
    log_path = run / "autonomous.log"
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "operator_agent.autonomous", str(run)],
            cwd=str(Path(__file__).resolve().parents[1]),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True,
        )
    return {
        "ok": True, "phase": "running", "pid": proc.pid, "run_id": run.name,
        "run_dir": str(run), "contract_hash": prepared["contract_hash"],
        "backend": backend, "max_rounds": int(max_rounds),
        "status": str(run / STATUS_FILE), "log": str(log_path),
        "hint": "用 operator_status 轮询；不要用 verify 的 run_dir 混用",
    }


def autonomous_status(run_dir: str | Path) -> dict:
    """查一次无人值守运行的相位。**三态**：running / done / failed / lost。"""
    run = Path(run_dir)
    status_path = run / STATUS_FILE
    if not status_path.is_file():
        return {"ok": False, "error": "run_not_started",
                "detail": f"{status_path} 不存在（还没调用 operator_run）"}
    status = json.loads(status_path.read_text(encoding="utf-8"))
    phase = str(status.get("phase") or "unknown")
    alive = _pid_alive(status.get("pid")) if phase == "running" else None
    if phase == "running" and alive is False:
        # 进程没了但没写终态：**不许读成"还在跑"**，也不许猜成"失败" —— 单独一档。
        phase = "lost"
    summary = None
    if (run / SUMMARY_FILE).is_file():
        summary = json.loads((run / SUMMARY_FILE).read_text(encoding="utf-8"))
    tail = []
    log_path = run / "autonomous.log"
    if log_path.is_file():
        try:
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
        except OSError:
            tail = []
    return {
        "ok": True, "phase": phase, "process_alive": alive,
        "pid": status.get("pid"), "started_at": status.get("started_at"),
        "finished_at": status.get("finished_at"),
        "duration_s": status.get("duration_s"),
        "backend": status.get("backend"), "max_rounds": status.get("max_rounds"),
        "error": status.get("error"), "detail": status.get("detail"),
        "summary": summary, "log_tail": tail,
    }
