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

from ..core.environment import fingerprint_of

#: 指纹五键之外，缓存键再收这三个执行身份量（hw_info 里有才收，不填默认值）。
_EXTRA_IDENTITY_KEYS = ("driver_version", "cuda_version", "gpu_uuid")

_CACHEABLE_LEVELS = ("L1", "L2")

_VERIFY_DIR = re.compile(r"verify-(\d+)")


def environment_identity(hw_info, backend_label: str = "") -> dict:
    """缓存用的环境身份：规范指纹 + 驱动/CUDA/GPU UUID（有才收）。

    探测失败（`hw_info` 空）时返回空字典或只剩 backend 键 —— 调用方据此**关闭**
    本次缓存，而不是拿一个残缺身份去匹配。
    """
    fp = fingerprint_of(hw_info, backend_label)
    for key in _EXTRA_IDENTITY_KEYS:
        if key in (hw_info or {}):
            fp[key] = hw_info[key]
    return fp


def _decisive(full: dict) -> bool:
    """判定是否「决定性」：ok、只有 L1/L2、每一级 passed 都是真 bool。"""
    if full.get("ok") is not True:
        return False
    verdicts = full.get("verdicts")
    if not isinstance(verdicts, list) or not verdicts:
        return False
    return all(isinstance(v, dict) and v.get("level") in _CACHEABLE_LEVELS
               and type(v.get("passed")) is bool for v in verdicts)


def lookup(run_dir, *, contract_hash: str, source_hash: str, stop_after,
           environment: dict, implementation: dict):
    """在同一 run 目录里找可回放的判定；找不到返回 None。

    取**最新**一轮匹配（轮号大者优先）。读不动/读不懂的证据目录一律跳过，
    绝不让缓存层的故障变成验证层的故障。
    """
    if stop_after not in _CACHEABLE_LEVELS or not environment or not implementation:
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