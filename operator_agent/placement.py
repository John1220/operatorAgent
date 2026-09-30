"""Stage 2 · 多卡放置与身份分区 (设计 §11 / §13.1 / §13.5)

三件事, 刻意分成三种键:

  * `DeviceKey`     —— 物理/虚拟设备身份 (uuid + arch + 干扰域)
  * `PartitionKey`  —— 「记忆与测量」的分区 (驱动/运行时/后端版本)
  * `BuildKey`      —— 「编译产物」的分区 (arch + 源码/编译旗标)

设计 §9.3: 把编译缓存与测量缓存混成一个键, 是多卡系统最常见的错误 ——
编译产物同 arch 可共享, 测量结果**绝对不能** (时钟/温度/竞争不同)。

干扰域 (§11-1): 同 L2 / 同时钟域 / 同 NVLink 域 → **同域必须串行测量**,
否则 cv 会诚实地变大, 把观测打到 DEGRADED。
同卡配对 (§11-2): 候选与基线必须同卡同窗 —— A 卡基线 vs B 卡候选的加速比
**直接作废** (多卡下最容易犯、也最难发现的错)。
竞争检测 (§11-4): 租不到独占卡时记录 external_load, 观测降为 DEGRADED
—— **诚实降级优于静默污染**。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class DeviceKey:
    """设备身份。**用 uuid 当锚**, 不用 index —— index 在重启/热插拔后会漂。"""
    uuid: str
    node: str = ""
    index: int = -1
    arch: str = ""                   # sm90 / sm80 / sm75 / xpu-kl3
    mig: Optional[str] = None
    interference_domain: str = ""    # 同 L2/时钟/NVLink 域 → 域内不可并行测量

    def label(self) -> str:
        base = f"{self.uuid[:8]}@{self.node or '?'}"
        if self.mig:
            base += f"({self.mig})"
        return base


@dataclass(frozen=True)
class PartitionKey:
    """「记忆与测量」的分区。跨分区的 best/加速比**不可隐式共享** (§11-5)。"""
    device_uuid: str
    arch: str = ""
    driver: str = ""
    runtime: str = ""
    backend_ver: Tuple[str, str] = ("", "")

    def as_tuple(self) -> tuple:
        return (self.device_uuid, self.arch, self.driver, self.runtime,
                self.backend_ver)


@dataclass(frozen=True)
class BuildKey:
    """「编译产物」的分区 —— 与 PartitionKey **刻意不同** (§9.3)。"""
    backend: str
    arch: str
    source_hash: str
    build_flags_hash: str = ""


# ─── 同卡配对 (§11-2 / §13.4 硬性语义 2) ───────────────────────

def pairing_conflict(candidate_device, baseline_device,
                     candidate_window=None, baseline_window=None) -> str:
    """运行时配对检查。返回拒绝理由, 空串 = 通过。

    契约冻结时的 `same_device`/`same_window` 是**声明**; 这里是**观测复核** ——
    声明了同卡、实际跑到两张卡上的加速比必须作废。设计 §11-2 原句:
    「A 卡基线 vs B 卡候选的『加速比』直接作废」。
    """
    if candidate_device != baseline_device:
        return (f"候选与基线不在同一张卡: cand={candidate_device!r} "
                f"baseline={baseline_device!r} —— 跨卡加速比无指称对象, 作废")
    if (candidate_window is not None and baseline_window is not None
            and candidate_window != baseline_window):
        return (f"候选与基线不在同一时间窗: {candidate_window!r} vs "
                f"{baseline_window!r} —— 热/时钟状态污染, 作废")
    return ""


# ─── 放置 (§11-1 / §13.5) ──────────────────────────────────────

@dataclass(frozen=True)
class DeviceLease:
    device: DeviceKey
    exclusive: bool = True
    external_load: float = 0.0       # 0..1 探测到的外部占用

    @property
    def degraded_by_competition(self) -> bool:
        """§11-4: 非独占或有外部负载 → 观测应降 DEGRADED, 不是错误通过。"""
        return (not self.exclusive) or self.external_load > 0.0


def can_measure_parallel(a: DeviceKey, b: DeviceKey) -> bool:
    """§11-1: 同域串行, 异域并行。返回 False = 这两张卡不许同时测。"""
    if not a.interference_domain or not b.interference_domain:
        # 未知域 → 保守串行 (失败关闭: 不知道是不是同域就当同域)
        return False
    return a.interference_domain != b.interference_domain


def plan_leases(devices: Iterable[DeviceKey], n: int,
                loads: Optional[dict] = None) -> List[List[DeviceKey]]:
    """把 n 次测量排进「波次」: 同域的进同一波 (串行), 异域可并行。

    `loads`: {device.uuid: external_load} 竞争检测 (§11-4)。
    返回 list[wave], 每个 wave 是一组可并行的设备。
    """
    devs = list(devices or ())
    if n <= 0 or not devs:
        return []
    loads = loads or {}
    # 按干扰域分组
    by_domain: dict = {}
    unknown = []
    for d in devs:
        if d.interference_domain:
            by_domain.setdefault(d.interference_domain, []).append(d)
        else:
            unknown.append(d)
    # 未知域: 每个单独一波 (保守串行)
    waves: List[List[DeviceKey]] = []
    domains = list(by_domain.values())
    # 每域取轮转设备, 一波从每域取一个 (异域并行)
    for i in range(n):
        wave = []
        for group in domains:
            wave.append(group[i % len(group)])
        waves.append(wave)
    # 未知域追加为独立波次
    for d in unknown:
        waves.append([d])
    return waves[:n] if n <= len(waves) else waves + [waves[-1]] * (n - len(waves))


def lease_notes(lease: DeviceLease) -> Tuple[str, bool]:
    """`(evidence 串, 该不该降级)`。§11-4: 诚实降级优于静默污染。"""
    if lease.degraded_by_competition:
        ev = (f"external_load={lease.external_load:.2f}"
              f"{'' if lease.exclusive else ', non-exclusive'}")
        return ev, True
    return f"exclusive device {lease.device.label()}", False


# ─── 异构卡 (§11-5) ────────────────────────────────────────────

def partition_of(device: DeviceKey, driver: str = "", runtime: str = "",
                 backend_ver: Tuple[str, str] = ("", "")) -> PartitionKey:
    return PartitionKey(device_uuid=device.uuid, arch=device.arch,
                        driver=driver, runtime=runtime, backend_ver=backend_ver)


def same_partition(a: PartitionKey, b: PartitionKey) -> bool:
    """跨 arch **绝不隐式共享** best/记忆 (§11-5)。"""
    return a.as_tuple() == b.as_tuple()


def build_key(backend: str, arch: str, source_hash: str,
              flags_hash: str = "") -> BuildKey:
    return BuildKey(backend=backend, arch=arch, source_hash=source_hash,
                    build_flags_hash=flags_hash)
