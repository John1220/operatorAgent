"""§13.5 的 `probe_peak`：**参考下界** —— 参考天花板出口唯一认的那个分母

设计的原话：

> `probe_peak` 必须实测：拿一个已知好 GEMM 打到真实峰值，而不是读 spec 表。
> 探测失败 → **天花板出口永久关闭**。

⚠️ 上面那句里的「打到真实峰值」是**器件峰值** (`peak_measured`) 那一半的字面口径 ——
它需要独立、compute-bound、按精度标定的 microbenchmark，今天**代码里没有**。本模块实现的是
设计 §13.5 三个概念里的第三个：`reference_lower_bound`（同一次 L3 里基线臂的实测可达值）。
两个名字不能互借：拿参考核的可达值说「贴着硬件峰值」，就是拿一个**下界**当上限用。

两问必须分开存放（CHECKLOG I49）：
① `peak_trusted`：表行**口径核过**没有（dense/sparse 没混、算术自洽）；
② `reference_lower_bound`：**真探测过一次**没有 —— 天花板出口只认这一个位。

本模块负责第 ② 问的**判据与发布**，并坚持三件事：

1. **探针自己也要过尺子的档位**：样本必须是 `valid` 档的读数。拿一个 degraded 的读数
   当分母，等于把「这张卡这一刻贴着参考实现的天花板了」建在一份自己都不可信的测量上 ——
   与 §10.4 那句「天花板出口只认有效档位的读数」是同一族纪律的下一层。
2. **样本必须真的打到算力峰值**：算术强度低于 ridge point 的样本量的是**带宽**，不是
   FLOPS。用它当分母会把「访存受限的快」误读成「算力接近到顶」。缺 AI / 缺 ridge
   （未知设备、表没命中）同样**不发布**（缺证据不是证据）。
3. **不许多报一个量**：实测量高出规格 5% 以上，说明采样买到的不是同一个量（稀疏/混合
   精度最常见的表现），如实拒绝而不是拿它当分母。

**今天的分母语义（要明写，别读成 datasheet）**：实测值来自同一次 L3 调用里**基线臂**
（同一 harness、同一形状、同一精度的已知好参考核）的实测可达值，即「这台机器上这个
精度**至少**能跑这么快」。它是**下界** ⇒ 拿它当分母，利用率会偏大、天花板出口会偏松。
这条风险靠两个条件约束：样本自己必须是 compute-bound（第 2 条）且是 valid 档（第 1 条）。
残余风险明写在此：若参考核自己没打到顶（形状太小、调度差），出口仍会偏松 —— 那时该修
的是参考核，不是放宽判据。

**成本是零**：探针复用同一次 L3 调用里已经付过钱的基线臂，不额外花 GPU 秒，因此不占
§10.2 那 5% 的标定额度（那个额度留给心跳）。这不是「省了钱」，而是「不重复测同一个东西」。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

#: 可以发布参考下界的精度键（与 verifier 里 `compute_used` 的取值一一对应）
PEAK_PRECISIONS = ("fp32", "tf32", "fp16", "bf16")

#: `verifier` 选分母时用的名字 → 精度键。**一个判据一处实现**：这里是唯一的映射表，
#: 谁要问「这一轮的分母是哪个精度」，都得走 `precision_key_of`。
COMPUTE_USED_TO_PRECISION = {
    "fp32_cuda_core": "fp32",
    "tf32_tensor_core": "tf32",
    "fp16_tensor_core": "fp16",
    "bf16_tensor_core": "bf16",
}

#: 实测量高于规格的容忍（比例）。稀疏/混精会把数字顶上去，而那说明采样不是同一个量。
MAX_ABOVE_SPEC = 1.05

#: 必须为真才算「真探测」的设备能力位名（走 caps 声明，不从设备名字符串猜）
REAL_DEVICE_CAP = "REAL_DEVICE"


def precision_key_of(compute_used: Optional[str]) -> Optional[str]:
    """`compute_used` → 精度键；认不出返回 `None`（不猜、不默认 fp32）"""
    if not compute_used:
        return None
    key = COMPUTE_USED_TO_PRECISION.get(str(compute_used))
    return key if key in PEAK_PRECISIONS else None


@dataclass(frozen=True)
class PeakSample:
    """一次「已知好 GEMM」的实测量，连同它自己那份测量的可信度证据"""

    tflops: Optional[float]                 # 实测可达值 (TFLOPS)
    precision: Optional[str]                # 精度键 (由 `precision_key_of` 给出)
    method: str                             # 这个数是怎么测出来的 (要能回指)
    real_device: bool                       # 走 caps 声明的能力位
    validity: Optional[str]                 # 这一次读数的档位
    arithmetic_intensity: Optional[float]
    ridge_point: Optional[float]
    #: 这一次读数的离散度（iqr/median，与测量侧同一个键）。`None` = 没上报 ⇒ 不发布：
    #: 分母要的是「这台机器能跑多快」，而不是「不知道稳不稳的某一次」。
    #: **刻意不再自设一个百分比阈值** —— 档位（`validity`）与「离散度超标了吗」
    #: （`dispersion_exceeded`）都是测量侧已经给出的**结论**，这里自己再算一遍就是
    #: 同一个判定两处实现（spine-review P1-6 的教训）。
    rel_dispersion: Optional[float] = None
    spec_tflops: Optional[float] = None     # 规格值 (只用于「不许多报」的对照)
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PeakProfile:
    """可以发布进 `HardwareCaps.reference_lower_bound_tflops` 的参考下界"""

    precision: str
    tflops: float
    method: str
    rel_dispersion: float
    evidence: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"precision": self.precision, "tflops": self.tflops,
                "method": self.method, "rel_dispersion": self.rel_dispersion,
                "evidence": dict(self.evidence)}


@dataclass(frozen=True)
class PeakProbeDecision:
    measured: bool
    profile: Optional[PeakProfile]
    refusals: Tuple[str, ...]
    detail: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"measured": self.measured,
                "profile": self.profile.as_dict() if self.profile else None,
                "refusals": list(self.refusals), "detail": dict(self.detail)}

    def refusal_note(self) -> str:
        return "; ".join(self.refusals)


def probe_peak(sample: PeakSample) -> PeakProbeDecision:
    """判据本体：这一次实测量能不能当 roofline 的分母。**任何一条不过就不发布**。"""
    refusals = []
    if sample.real_device is not True:
        refusals.append("不是真实设备 (能力位 REAL_DEVICE 未声明) —— "
                        "仿真读数可以驱动搜索, 但不能当参考下界")
    if sample.precision not in PEAK_PRECISIONS:
        refusals.append(f"精度键不可识别 ({sample.precision!r}) —— 分母按精度发布, "
                        f"认不出就不发布 (认得出的是 {', '.join(PEAK_PRECISIONS)})")
    if not sample.method:
        refusals.append("没有测量方法的出处 —— 分不出「实测」与「抄表」的数不许进分母")
    tflops = sample.tflops
    if (not isinstance(tflops, (int, float)) or isinstance(tflops, bool)
            or not math.isfinite(float(tflops)) or float(tflops) <= 0):
        refusals.append(f"实测量不是有限正数 ({tflops!r})")
    if sample.validity != "valid":
        refusals.append(f"这一次读数不是 valid 档 ({sample.validity!r}) —— "
                        "探针自己也得过尺子的档位")
    if sample.rel_dispersion is None:
        refusals.append("样本离散度未知 —— 不知道这个数稳不稳, 就不拿它当分母")
    elif not isinstance(sample.rel_dispersion, (int, float)) or sample.rel_dispersion < 0:
        refusals.append(f"样本离散度不是非负数 ({sample.rel_dispersion!r})")
    ai, ridge = sample.arithmetic_intensity, sample.ridge_point
    if ai is None or ridge is None or ridge <= 0:
        refusals.append(f"算术强度/脊点未知 (AI={ai!r}, ridge={ridge!r}) —— "
                        "分不出这个样本量的是算力还是带宽")
    elif ai < ridge:
        refusals.append(f"样本没打到算力峰值 (AI={ai:.4g} < 脊点 {ridge:.4g}) —— "
                        "访存受限的样本量的是带宽, 不是 FLOPS")
    spec = sample.spec_tflops
    if (isinstance(tflops, (int, float)) and not isinstance(tflops, bool)
            and isinstance(spec, (int, float)) and spec and spec > 0
            and float(tflops) > float(spec) * MAX_ABOVE_SPEC):
        refusals.append(f"实测量 {float(tflops):.4g} 高于规格 {float(spec):.4g} 的 "
                        f"{MAX_ABOVE_SPEC:.2f} 倍以上 —— 采样买到的不是同一个量 "
                        "(稀疏/混合精度最常见的表现)")

    detail = {
        "tflops": float(tflops) if isinstance(tflops, (int, float)) and not isinstance(tflops, bool) else None,
        "precision": sample.precision, "method": sample.method,
        "rel_dispersion": sample.rel_dispersion,
        "arithmetic_intensity": ai, "ridge_point": ridge,
        "spec_tflops": spec, "validity": sample.validity,
        "real_device": sample.real_device,
        "evidence": dict(sample.evidence),
    }
    if refusals:
        return PeakProbeDecision(measured=False, profile=None,
                                 refusals=tuple(refusals), detail=detail)
    profile = PeakProfile(precision=sample.precision, tflops=float(tflops),
                          method=sample.method,
                          rel_dispersion=float(sample.rel_dispersion),
                          evidence=dict(sample.evidence))
    return PeakProbeDecision(measured=True, profile=profile, refusals=(), detail=detail)


def sample_from_l3_metrics(metrics: Dict[str, Any], *,
                           real_device: bool, spec_tflops: Optional[float],
                           method: str, flops: Optional[float],
                           baseline_ms: Optional[float]) -> PeakSample:
    """从一次 L3 的产物里装出样本（**唯一**装配点：证据就是产物里记的那份）

    基线臂的实测可达值 = `flops / baseline_ms`。用 `flops` 而不是拿候选的 gflops 去按
    时间比换算: 后者把候选的 median/min 口径混进分母, 而这里要的是「同一个分子、换一个
    时间」—— 两个数都来自同一次调用, 不引入第二个真源。
    """
    roofline = metrics.get("roofline_analysis") or {}
    tflops = None
    if flops and baseline_ms and baseline_ms > 0:
        tflops = float(flops) / (float(baseline_ms) * 1e-3) / 1e12
    return PeakSample(
        tflops=tflops,
        precision=precision_key_of(roofline.get("compute_used")),
        method=method,
        real_device=real_device,
        validity=metrics.get("measurement_validity"),
        arithmetic_intensity=roofline.get("arithmetic_intensity"),
        ridge_point=roofline.get("ridge_point"),
        rel_dispersion=metrics.get("dispersion_iqr_over_median"),
        spec_tflops=spec_tflops,
        evidence={"flops": flops, "baseline_ms": baseline_ms,
                  "compute_used": roofline.get("compute_used"),
                  "candidate_gflops": metrics.get("gflops")},
    )