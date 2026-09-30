"""Oracle —— 评测器唯一收口点 (设计 §13.4 / §2 总体图的「脊柱」)

把四件脊柱收成**一次 evaluate**:

    契约哈希  →  心跳窗口  →  资格位  →  观测(值+溯源+资格)  →  可选重测确认

为什么是「唯一收口点」: 设计 §13.4 的三条硬性语义必须在 evaluate **内部**而不是
调用方 —— 否则每个调用点都会漏掉其中一条 (本仓已多次踩过「算了没人读」/「声明了
没人派发」那一族)。

    1. contract_hash 不匹配 → **拒评**, 不是降级
    2. 心跳失败窗口内的观测 → 全资格 False
    3. 资格决定「这个数被允许造成什么后果」, 不只是「过没过」

本模块**不**做 I/O、不连 GPU: 执行由调用方通过 `bench_fn` / `measure_fn` 注入,
保持纯函数可离线单测 (与 `measurement.py` / `tolerance.py` 同一条定位)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

from .calibration import HeartbeatResult, heartbeat_due
from .confirm import ConfirmRecord, confirm_persist
from .contract import ContractError, FrozenContract, assert_hash
from .eligibility import ALL_FALSE as ELIG_ALL_FALSE
from .eligibility import Eligibility, derive_eligibility
from .placement import pairing_conflict


@dataclass
class Observation:
    """设计 §13.3 的观测记录 (值 + 不确定度 + 溯源 + 资格)。"""
    metric: str
    value: Optional[float]
    unit: str
    validity: str                      # valid | degraded | uninformative
    eligibility: Eligibility
    evidence: list = field(default_factory=list)
    reasons: list = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    #: 拒评时这里放 ContractError 文案; 有值 = 本次**没有**有效观测
    refused: str = ""

    def as_dict(self) -> dict:
        return {
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "validity": self.validity,
            "eligibility": self.eligibility.as_dict(),
            "evidence": list(self.evidence),
            "reasons": list(self.reasons),
            "provenance": dict(self.provenance),
            "refused": self.refused,
        }


def evaluate(contract: FrozenContract,
             measure: Callable[[], Mapping[str, Any]],
             *,
             record_hash: Optional[str] = None,
             heartbeat: Optional[HeartbeatResult] = None,
             metric: str = "speedup_vs_baseline",
             unit: str = "x",
             remeasure: Optional[ConfirmRecord] = None,
             require_real_device: bool = True,
             peak: Optional[float] = None,
             candidate_device: Optional[str] = None,
             baseline_device: Optional[str] = None,
             candidate_window: Optional[str] = None,
             baseline_window: Optional[str] = None,
             ) -> Observation:
    """一次完整评测。`measure()` 返回的键至少含 `value` 与可选
    `validity` / `lib_fallback` / `cv_exceeded` / `launch_dominance_exceeded` /
    `partition_drift` / `l3_pass_is_refused` / `real_device`。

    **哈希校验是记录携带的** (spine-review P0-2): `record_hash` 是候选/记录
    在**生成时**烙进来的契约哈希, 与 `contract.hash` 比对 —— 不是「调用方
    同时给两侧」那种恒真。缺 `record_hash` 视为「记录没带哈希」→ 拒评
    (设计 §13.4: `rec.contract_hash != contract.hash → 拒评`)。

    ⚠️ 契约哈希不匹配时**不调用 measure()** —— 连测都不该测。
    """
    # ── 1. 契约哈希: 记录携带 vs 当前契约。改不了契约就改不了判据 ──
    expected = contract.hash or contract.compute_hash()
    if not record_hash:
        return Observation(
            metric=metric, value=None, unit=unit,
            validity="uninformative",
            eligibility=ELIG_ALL_FALSE,
            refused="记录未携带 contract_hash: 无法证明判据未被改写, 拒评",
            reasons=["missing_record_hash"],
            provenance={"contract_hash": expected},
        )
    try:
        assert_hash(contract, record_hash)
    except ContractError as exc:
        return Observation(
            metric=metric, value=None, unit=unit,
            validity="uninformative",
            eligibility=ELIG_ALL_FALSE,
            refused=str(exc),
            reasons=[str(exc)],
            provenance={"contract_hash": expected, "record_hash": record_hash},
        )

    # ── 2. 同卡配对 (设计 §11-2 / §13.4 硬性语义 2) ──
    # 声明了 same_device、实际跑到两张卡上的加速比**作废**。
    # spine-review P3-16: 这条检查必须在库里, 不能只活在测试文件里。
    if candidate_device is not None or baseline_device is not None:
        pair_why = pairing_conflict(candidate_device, baseline_device,
                                   candidate_window, baseline_window)
        if pair_why:
            return Observation(
                metric=metric, value=None, unit=unit,
                validity="uninformative",
                eligibility=ELIG_ALL_FALSE,
                refused=pair_why,
                reasons=[pair_why],
                provenance={"contract_hash": expected,
                            "candidate_device": candidate_device,
                            "baseline_device": baseline_device},
            )

    # ── 3. 心跳: 失败窗口内全资格 False, 但仍要记下数 (只是不让它造成后果) ──
    hb_ok = True
    hb_reasons = []
    if heartbeat is not None:
        hb_ok = bool(heartbeat.ok)
        hb_reasons = list(heartbeat.reasons or [])

    raw = dict(measure())
    value = raw.get("value")
    validity = str(raw.get("validity") or "valid")
    reasons = [str(r) for r in (raw.get("reasons") or [])]
    evidence = [str(e) for e in (raw.get("evidence") or [])]

    # ── 3. 资格位 (设计 §4.3 八条, 短路) ──
    elig = derive_eligibility(
        validity=validity,
        contract_hash_ok=True,               # 上面已拦
        heartbeat_ok=hb_ok,
        requires_real_device=require_real_device,
        real_device=raw.get("real_device"),
        lib_fallback=bool(raw.get("lib_fallback")),
        peak_exceeded=bool(raw.get("peak_exceeded"))
        or (peak is not None and isinstance(value, (int, float))
            and not isinstance(value, bool) and value > peak * 1.05),
        cv_exceeded=bool(raw.get("cv_exceeded")),
        launch_dominance_exceeded=bool(raw.get("launch_dominance_exceeded")),
        partition_drift=bool(raw.get("partition_drift")),
        l3_pass_is_refused=bool(raw.get("l3_pass_is_refused")),
    )
    if not hb_ok:
        reasons.append("RULER_BROKEN: 标定心跳失败, 同窗观测全资格 False")
        reasons.extend(hb_reasons)

    prov = {
        "contract_hash": expected,
        "record_hash": record_hash,
        "heartbeat_ok": hb_ok,
        "heartbeat_epoch": getattr(heartbeat, "epoch", None),
        "baseline": contract.baseline.name,
        "dtype_semantics": {
            "tf32": contract.dtype_semantics.tf32,
            "accum": contract.dtype_semantics.accum,
        },
    }
    prov.update(raw.get("provenance") or {})

    return Observation(
        metric=metric,
        value=value if isinstance(value, (int, float)) and not isinstance(value, bool) else None,
        unit=unit,
        validity="uninformative" if not elig.can_verdict else validity,
        eligibility=elig,
        evidence=evidence,
        reasons=reasons,
        provenance=prov,
    )


def admit_to_corpus(obs: Observation,
                    first_speedup: Optional[float],
                    first_validity: str,
                    remeasure: Optional[ConfirmRecord],
                    min_speedup: float = 1.0) -> tuple:
    """入库门 = 资格 `can_persist` **且** 两次确认 (设计 §10.5-4)。

    返回 (允许?, 理由)。任一关不过就拒 —— 没有「没有重测就当过了」。
    """
    if not obs.eligibility.can_persist:
        return False, (
            f"资格不足 (can_persist=False, validity={obs.validity}): "
            f"{'; '.join(obs.reasons) or '见 eligibility 规则'}")
    ok, why = confirm_persist(first_speedup if first_speedup is not None else obs.value,
                              first_validity or obs.validity,
                              remeasure,
                              min_speedup=min_speedup)
    return ok, why
