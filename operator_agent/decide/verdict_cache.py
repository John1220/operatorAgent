"""判定缓存：run 目录里的 write-once 证据就是缓存的读取面，不新建存储层。

缓存键 = 契约哈希 × 源码哈希 × 判定深度 × 环境身份 × 实现身份。五条全等才命中；
任何一条缺失（旧证据没有环境身份格）按不命中处理 —— 缺证据不等于比过了。

只回放 **L1/L2 的决定性判定**：
* L3 永不缓存 —— 性能读数是测量窗口绑定的，回放等于把上一窗的读数冒充本窗；
* unknown / 执行未完成 / 证据不完整永不缓存 —— 设施故障不该被冻成永久判定；
* `purpose=final` 的独立终评永不走缓存（调用方保证），否则终评失去独立性。

命中也写一份**新的**证据目录（预算门要求每次派发独立证据），标 `cache_hit` 与
`cached_from` 指回原始执行；预算照扣 —— 登记单位本来就是派发，不是 GPU 秒。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..core.environment import FINGERPRINT_KEYS, fingerprint_of
from .measurement import TRANSPORT_KEY, TRANSPORT_UNKNOWN

#: 指纹五键之外，缓存键**机会性**再收这三个执行身份量（hw_info 里有才收，不填默认值）。
#: SSH 后端当前不产出这三键（runner 与 executor 均未采集），故 SSH 上的有效环境身份是
#: **GPU 型号级**（name+device+cc+torch+triton+backend），不绑到具体卡 UUID —— 这是已接受
#: 的设计边界（2026-11-15 用户定）：缓存只回放决定性 L1/L2，同型号同栈正确性一致。要绑卡
#: 实例需在 runner+executor 两层补采集，届时这三键自动生效，无需改本模块。
_EXTRA_IDENTITY_KEYS = ("driver_version", "cuda_version", "gpu_uuid")

_CACHEABLE_LEVELS = ("L1", "L2")

_VERIFY_DIR = re.compile(r"verify-(\d+)")


def environment_identity(hw_info, backend_label: str = "") -> dict:
    """缓存用的环境身份：规范指纹 + 驱动/CUDA/GPU UUID（**后端产出才收**）。

    SSH 后端当前只产出型号级字段（name/device/cc/torch/triton），不产出驱动/CUDA/UUID，
    故 SSH 上的有效身份是 GPU 型号级、不绑卡实例（已接受的边界，见 _EXTRA_IDENTITY_KEYS）。
    探测失败（`hw_info` 空）时返回空字典或只剩 backend 键 —— 调用方据此**关闭**
    本次缓存，而不是拿一个残缺身份去匹配。
    """
    fp = fingerprint_of(hw_info, backend_label)
    for key in _EXTRA_IDENTITY_KEYS:
        if key in (hw_info or {}):
            fp[key] = hw_info[key]
    return fp


def _unknown(verdict: dict) -> bool:
    """判定是否被标为 unknown（设施缺口/传输没答话），而非候选的决定性胜负。

    标记来自 verifier 的各 unknown 生产者：`correctness.verdict` /
    `correctness_primary_verdict`（正确性预检/参考身份未确立）、`reference_identity_unknown`、
    `transport_outcome=outcome_unknown`（传输没答话）、`compile_status=transport_unknown`。
    这些回执的 `passed` 可能被写成 `False`，但语义是「不知道」——缓存绝不能把它冻成永久判定，
    否则同源码重试会回放那次 unknown 而不重新执行。
    """
    metrics = verdict.get("metrics") or {}
    correctness = metrics.get("correctness") or {}
    return (correctness.get("verdict") == "unknown"
            or metrics.get("correctness_primary_verdict") == "unknown"
            or metrics.get("reference_identity_unknown") is True
            or metrics.get(TRANSPORT_KEY) == TRANSPORT_UNKNOWN
            or metrics.get("compile_status") == "transport_unknown")


def _decisive(full: dict) -> bool:
    """判定是否「决定性」：ok、只有 L1/L2、每一级 passed 都是真 bool 且非 unknown。"""
    if full.get("ok") is not True:
        return False
    verdicts = full.get("verdicts")
    if not isinstance(verdicts, list) or not verdicts:
        return False
    return all(isinstance(v, dict) and v.get("level") in _CACHEABLE_LEVELS
               and type(v.get("passed")) is bool and not _unknown(v) for v in verdicts)


def _identified(environment: dict) -> bool:
    """环境身份是否含真实硬件指纹键（而非只剩 backend）。

    探测失败时 `fingerprint_of({}, label)` 只产出 `{backend: label}`；这种残缺身份
    必须 fail-closed（与 `environment_identity` docstring 一致），不能拿去匹配缓存。
    """
    return any(key in (environment or {}) for key in FINGERPRINT_KEYS)


def lookup(run_dir, *, contract_hash: str, source_hash: str, stop_after,
           environment: dict, implementation: dict):
    """在同一 run 目录里找可回放的判定；找不到返回 None。

    取**最新**一轮匹配（轮号大者优先）。读不动/读不懂的证据目录一律跳过，
    绝不让缓存层的故障变成验证层的故障。
    """
    if stop_after not in _CACHEABLE_LEVELS or not _identified(environment) or not implementation:
        return None
    run = Path(run_dir)
    candidates = []
    try:
        entries = list(run.iterdir())
    except OSError:
        return None
    for path in entries:
        match = _VERIFY_DIR.fullmatch(path.name)
        if match and path.is_dir():
            candidates.append((int(match.group(1)), path))
    for index, path in sorted(candidates, reverse=True):
        try:
            full = json.loads((path / "result.json").read_text(encoding="utf-8"))
            complete = json.loads((path / "complete.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        binding = full.get("binding") or {}
        if (full.get("evidence_complete") is not True
                or complete.get("evidence_complete") is not True
                or binding != complete.get("binding")
                or binding.get("contract_hash") != contract_hash
                or binding.get("source_hash") != source_hash
                or binding.get("stop_after") != stop_after
                or binding.get("environment") != environment
                or full.get("implementation") != implementation
                or not _decisive(full)):
            continue
        return dict(payload=full, evidence=str(path / "result.json"), round=index)
    return None