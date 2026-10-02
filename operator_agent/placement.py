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
竞争检测 (§11-4): 租不到独占卡时**记录 external_load 并在 evidence 里显式标注**,
观测降为 DEGRADED —— **诚实降级优于静默污染**。

⚠️ 但「降级」这个判决**不在本模块**: 它是 `measurement.assess` 的职责 (读同一个
`gpu_util_pct`, 阈值也在那一处)。本模块只产出**证据**(`lease_evidence` 的三态)与
**观测记录**(`DeviceLease`)。原先这里的 `degraded_by_competition` / `lease_notes(…)->bool`
是同一判定的**第二处实现**, 而且阈值不同 (本模块 `>0.0`, 那边 `>5%`) —— 谁先去读它,
谁就成了尺子 (spine-review P1-6 就是这样发生的)。
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
        if self.uuid:
            base = f"{self.uuid[:8]}@{self.node or '?'}"
        else:
            # 没观测到 uuid 时**不许**印出一个像 uuid 的东西 (空串切片也是空串, 但
            # 若哪天有人塞了伪 uuid 就会冒充真锚)。如实说「只有索引」—— 而索引会漂
            # (§11: 「uuid 最稳的锚, 别用 index」), 所以这句话本身就是证据的一部分。
            base = f"index={self.index}@{self.node or '?'}"
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


# ─── 逐臂身份 → 可比较标签 (§11-2 配对检查的喂给者; CHECKLOG I47) ──────────
#
# 为什么需要这一层: `pairing_conflict` 比的是「两侧的设备标识」, 而远端报回来的是一个
# dict (`visible_index` / `name` / `cuda_visible_devices`)。压标签的口径必须只有一处 ——
# 否则「同卡」这件事会随调用方各自解释。v0.5 的 `metrics["measurement_scope"]` 正是
# 那句解释: 它写死成 `"same_remote_device"`, **键名就是 §11-2 要的那件事, 而当时没有
# 任何观测在背后** (全仓只有一个测试在读它)。现在它由观测算出来。
SCOPE_SAME = "same_remote_device"       #: 两臂观测到同一张卡 (有观测, 且一致)
SCOPE_CROSS = "cross_device_refused"    #: 观测到不同卡 ⇒ 该读数被拒评 (all-False)
SCOPE_UNOBSERVED = "unobserved"         #: 远端没报逐臂身份 ⇒ **不声称**同卡


def device_label(identity) -> Optional[str]:
    """一个逐臂身份 dict → 可比较标签。不可用 (没观测 / 读失败 / 不是 dict) ⇒ None。

    ⚠️ 标签必须同时含 `visible_index` 与 `cuda_visible_devices`: 只看 index 会把
    「`CUDA_VISIBLE_DEVICES=2` 下的 index 0」与「`=5` 下的 index 0」读成同一张卡 ——
    而「index 0 被当成卡号」正是 doc 63 那次事故的失效方式。
    """
    if not isinstance(identity, dict):
        return None
    idx = identity.get("visible_index")
    if idx is None:
        return None       # 读失败 ⇒ 没观测 (不是「约定成 0」)
    return "cuda:{i} {n} (visible={v})".format(
        i=idx, n=identity.get("name") or "unknown",
        v=identity.get("cuda_visible_devices") or "unset")


def observed_pairing(candidate_identity, baseline_identity):
    """远端两臂身份 → `(候选标签, 基线标签, scope)`。**缺观测返回 None** (不许空串冒充)。

    两个契约:
      * 标签不是判决 —— 「作废」这个决定仍然只有 `pairing_conflict` 一处。但两者读的是
        同一对标签, 所以不可能不一致 (标签相等 ⟺ `pairing_conflict` 的 `!=` 不成立)。
      * `None` 与 `""` 不是一回事: `None` = 没观测 (oracle 跳过这道检查, 与
        `real_device=None` 同一边界), `""` 会变成一次恒真的比较。
    """
    c = device_label(candidate_identity)
    b = device_label(baseline_identity)
    if c is None or b is None:
        return None, None, SCOPE_UNOBSERVED
    return c, b, (SCOPE_SAME if c == b else SCOPE_CROSS)


#: 逐臂**测量窗口**的三态 (§11-2 的「同一时间窗」那一半; CHECKLOG I47)。
#:
#: 与设备标签同形, 但回答的是另一个问题: 「这一对臂是不是在**同一次测量调用**里量的」。
#: `None` = 没观测 (oracle 跳过这道比较, 与 `real_device=None` 同一边界), `""` 会成为
#: 一次恒真的比较 —— 所以缺观测一律 `None`。
WINDOW_SAME = "same_window"                 #: 两臂同一次调用 ⇒ 同窗
WINDOW_DIFFERENT = "cross_window_refused"   #: 观测到不同窗 ⇒ 加速比作废 (§11-2)
WINDOW_UNOBSERVED = "unobserved"            #: 没有逐臂窗口观测 ⇒ 不声称同窗


def observed_windows(candidate_window, baseline_window):
    """两臂的窗口观测 → `(候选窗口, 基线窗口, scope)`。**缺观测返回 None**。

    窗口的意义 (§11-2): 热/时钟状态只有在**同一扇窗**里才可比。远端 runner 在一次
    `benchmark_with_baselines` 里按块交错量两臂 ⇒ 两臂的窗口 id 由**各自那一臂**的样本
    带着回来; 有一天某条臂的样本被复用 (缓存/短路), 带回来的就是**旧窗**, 于是这道比较
    就响。今天远端路径上「同窗」是构造出来的事实, 而这条观测让它是**被观测到的**, 不是
    被假设的。
    """
    if candidate_window is None or baseline_window is None:
        return None, None, WINDOW_UNOBSERVED
    return (candidate_window, baseline_window,
            WINDOW_SAME if candidate_window == baseline_window else WINDOW_DIFFERENT)


# ─── 放置 (§11-1 / §13.5) ──────────────────────────────────────

@dataclass(frozen=True)
class DeviceLease:
    """一次测量的**观测记录**: 这块卡是谁的、当时有没有别人在用 (§11-4)。

    三个字段都是**三态** (`缺 ≠ 0 ≠ 无上限` 那一族), 因为租约是证据:

      * `exclusive`: `None` = **没有观测到独占性**。今天恒为 `None` —— harness 不租卡,
        它只是挑一张空闲的卡, 没有任何观测能支撑「独占」这个声明。默认写 `True` 就是
        「没观测 ⇒ 声称独占」, 而下游会拿它当「干净」的证据。
      * `external_load`: `None` = 没观测到占用; `0.0` = 观测到**没有**占用。默认写 `0.0`
        会把「没测」印成「干净」。
      * `harness_peers` (CHECKLOG I68): `None` = 没观测 (旧 runner 不报这个键);
        `0` = 观测到没有; `>0` = **同卡还有别的 harness 进程** (claim 声明)。它与
        `external_load` 分工: 那个是任意负载的利用率 (时刻采样, 看不见还在 import 的
        对手), 这个的声明生命期 = 对手的整个进程 —— 一旦 >0, 竞争是**结构性**的。
      * `source`: 这条租约是从哪个观测装出来的 (空串 = 不知道出处 ⇒ 它只是一条声明)。

    ⚠️ **本类不做判决**: 降不降级只有 `measurement.assess` 说了算 (§11-4 的原话是
    「记录 external_load 并在 evidence 中显式标注; 观测降为 DEGRADED」—— 前半句是本类,
    后半句是测量侧)。
    """
    device: DeviceKey
    exclusive: Optional[bool] = None
    external_load: Optional[float] = None       # 0..1 观测到的外部占用
    harness_peers: Optional[int] = None         # 同卡其他 harness 进程 (claim 声明, I68)
    source: str = ""

    def as_dict(self) -> dict:
        return {"device": {"uuid": self.device.uuid, "node": self.device.node,
                           "index": self.device.index, "arch": self.device.arch,
                           "mig": self.device.mig,
                           "interference_domain": self.device.interference_domain},
                "device_label": self.device.label(),
                "exclusive": self.exclusive,
                "external_load": self.external_load,
                "harness_peers": self.harness_peers,
                "source": self.source}


#: 租约证据的三态 (§11-4 的「显式标注」)。**不是判决** —— 判决在 `measurement.assess`。
LEASE_CONTENDED = "contended"       #: 有观测显示这张卡在被别人用
LEASE_CLEAN = "clean"               #: 有观测显示当时没有别人用
LEASE_UNOBSERVED = "unobserved"     #: 没观测到 == 不许声称干净


def lease_evidence(lease: DeviceLease) -> Tuple[str, str]:
    """`(evidence 串, 三态)`。三态的优先级写死在这里, 免得两个调用方各解一次。"""
    bits = []
    if lease.exclusive is None:
        bits.append("exclusive 未观测")
    else:
        bits.append("exclusive" if lease.exclusive else "non-exclusive")
    if lease.external_load is None:
        bits.append("external_load 未观测")
    elif lease.external_load > 0:
        bits.append("external_load={:.2f}".format(lease.external_load))
    else:
        bits.append("external_load=0.00")
    # I68: 同卡 harness 声明只在**有对手**时占一格 —— `0` 与「未观测」都由
    # external_load 那一格负责表达「干净/没测」, 这里重复一遍只会稀释证据。
    if lease.harness_peers:
        bits.append(f"同卡 harness 进程 ×{lease.harness_peers} (claim 声明)")
    if lease.source:
        bits.append("source=" + lease.source)
    if (lease.exclusive is False or (lease.external_load or 0.0) > 0.0
            or (lease.harness_peers or 0) > 0):
        state = LEASE_CONTENDED
    elif lease.external_load is None:
        state = LEASE_UNOBSERVED
    else:
        state = LEASE_CLEAN
    return ", ".join(bits), state


def observed_lease(payload, *, device: Optional[DeviceKey] = None,
                   source: str = "") -> DeviceLease:
    """从**这次测量的载荷**装出租约。

    ⚠️ 读的键 (`gpu_util_pct` / `gpu_harness_peers`) 与 `measurement.assess` 读的是
    **同一份载荷**, 不另采一次 —— 「一处观测、一处判决」的前提是两个判定看同一份
    观测; 各采一次就会出现「租约说忙、档位说干净」这种谁也查不出来的分叉。
    """
    p = payload or {}
    util = p.get("gpu_util_pct")
    load = None
    if isinstance(util, (int, float)) and not isinstance(util, bool):
        load = float(util) / 100.0
    # I68: harness 声明是计数不是利用率, 单独一格; 键不在 (= 旧 runner) ⇒ None,
    # 「未观测」不许被读成「观测到没有」。
    peers = p.get("gpu_harness_peers")
    if isinstance(peers, bool) or not isinstance(peers, int):
        peers = None
    return DeviceLease(device=device if device is not None else device_of_payload(p),
                       exclusive=None,          # 今天没有任何观测能支撑独占性
                       external_load=load, harness_peers=peers, source=source)


def device_of_payload(payload) -> DeviceKey:
    """载荷 → 设备身份。只有索引级身份时**如实**记 `uuid=""`。

    `interference_domain` 用空串表示未观测 (`can_measure_parallel` 就是这么读的),
    uuid 沿用同一约定。**不许**拿 index 冒充 uuid (§11 原话: 「最稳的锚, 别用 index」)。
    """
    p = payload or {}
    idx = p.get("selected_gpu")
    if isinstance(idx, bool) or not isinstance(idx, int):
        idx = -1
    return DeviceKey(uuid="", node=str(p.get("device") or ""), index=idx, arch="")


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


def lease_waves(devices: Iterable[DeviceKey], n: int,
                loads: Optional[dict] = None) -> List[List[DeviceLease]]:
    """设计的 `Placement.lease(contract, n)` 形状: 波次里装的是**租约**。

    `loads` ({uuid: external_load}) 是今天**唯一**能观测到的竞争信号 ⇒ 记进租约;
    `exclusive` 仍然恒 `None` (不租卡就没有独占性观测 —— 见 `DeviceLease` 的三态说明)。

    ⚠️ **今天没有多卡消费者**: 真机是单卡串行、一次调用跑完两臂。这一条链路的诚实状态
    写在 doc 71 与设计 §15 的缺件表里, 不许读成「已经并行了」。
    """
    loads = loads or {}
    return [[DeviceLease(device=d, exclusive=None,
                         external_load=loads.get(d.uuid),
                         source="lease_waves")
             for d in wave] for wave in plan_leases(devices, n)]


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
