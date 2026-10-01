"""Oracle —— 评测器唯一收口点 (设计 §13.4 / §2 总体图的「脊柱」)

把脊柱收成**一次 evaluate**:

    契约哈希  →  心跳窗口  →  资格位  →  观测(值+溯源+资格)

入库则再叠一道（同模块的 `admit_to_corpus`）: 资格 `can_persist` **且** 两次确认。
两步分开是有意的 —— 重测要花机时，而它只在**要进语料**时才需要，不该由「每次评测」
触发。（`evaluate` 原先收着一个从不使用的 `remeasure` 形参，那个形状会让人以为它
在做这件事；形参已删。）

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

from .calibration import RULER_BROKEN, HeartbeatResult, ruler_broken_window
from .confirm import ConfirmRecord, confirm_persist
from .contract import ContractError, FrozenContract, assert_hash
from .eligibility import ALL_FALSE as ELIG_ALL_FALSE
from .eligibility import Eligibility, derive_eligibility, exceeds_measured_peak
from .measurement import persist_decision
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
    #: 契约哈希校验的**结论**, 机器可读 (spine-review P0-2):
    #: `verified` (记录携带的哈希与当前契约一致) / `fresh_unverified` (新观测, 没有旧
    #: 记录可校 —— 只盖了溯源戳, **没有**校验过) / `mismatch` / `missing`。
    #: 有它下游就不必从 `refused` 的文案里猜到底发生了什么。
    hash_check: str = ""
    #: 尺子状态: True = 本次读数落在**心跳失败**的窗口内 (§4.3 第 2 条 → 资格全 False),
    #: 或者心跳**到期却没跑**。False = 有心跳且通过, 或本次与心跳无关 —— 「没检查」与
    #: 「检查过且好」的两件事记在 provenance 的 `heartbeat_checked`/`heartbeat_expected`
    #: 两格里, 不共用一个 False。
    ruler_broken: bool = False

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
            "hash_check": self.hash_check,
            "ruler_broken": self.ruler_broken,
        }


def evaluate(contract: FrozenContract,
             measure: Callable[[], Mapping[str, Any]],
             *,
             record_hash: Optional[str] = None,
             fresh: bool = False,
             heartbeat: Optional[HeartbeatResult] = None,
             heartbeat_expected: bool = False,
             metric: str = "speedup_vs_baseline",
             unit: str = "x",
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
    同时给两侧」那种恒真。

    三种形态由调用方**明说**, 不许靠默认值糊过去:
      * `record_hash="…"` → 复用一条已存在的记录, 真比对; 不匹配拒评。
      * `fresh=True`      → 本次是**新观测**, 没有旧记录可校: 只盖溯源戳, 结论记
        `hash_check="fresh_unverified"` —— 明确表示**没有校验过**, 而不是假装校验通过。
      * 两者都没给        → 拒评 (fail-closed)。
    生产调用方原先用「从当前契约现算一个哈希再回喂」把第三种伪装成第一种,
    于是校验恒真。本函数拦不住伪造, 但「没带哈希」再也不可能被静默读成通过。

    ⚠️ 契约哈希不匹配时**不调用 measure()** —— 连测都不该测。
    """
    # ── 1. 契约哈希: 记录携带 vs 当前契约。改不了契约就改不了判据 ──
    expected = contract.hash or contract.compute_hash()
    if record_hash and fresh:
        return Observation(
            metric=metric, value=None, unit=unit,
            validity="uninformative",
            eligibility=ELIG_ALL_FALSE,
            refused="record_hash 与 fresh=True 同时给出: 无法判定这是新观测还是"
                    "复用旧记录, 拒评",
            reasons=["ambiguous_hash_mode"],
            provenance={"contract_hash": expected, "record_hash": record_hash},
            hash_check="missing",
        )
    if not record_hash and not fresh:
        return Observation(
            metric=metric, value=None, unit=unit,
            validity="uninformative",
            eligibility=ELIG_ALL_FALSE,
            refused="记录未携带 contract_hash: 无法证明判据未被改写, 拒评"
                    " (新观测请显式传 fresh=True —— 那条路不声称校验过)",
            reasons=["missing_record_hash"],
            provenance={"contract_hash": expected},
            hash_check="missing",
        )
    if record_hash:
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
                hash_check="mismatch",
            )
    # 「校验过」与「没得校」必须可区分 —— 前者才算收口, 后者只是溯源
    hash_check = "verified" if record_hash else "fresh_unverified"

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
                # 稳定分类 token + 人话各一份: `reasons` 是给**代码**读的分类
                # (上面两条拒评同样用 token: `ambiguous_hash_mode` / `missing_record_hash`),
                # `refused` 是给人读的原文。verifier 靠 token 决定「passed 有没有对象」,
                # 不靠中文措辞 —— 措辞改一个字不该让判决跟着变。
                reasons=["pairing_conflict"],
                provenance={"contract_hash": expected,
                            "candidate_device": candidate_device,
                            "baseline_device": baseline_device},
                hash_check=hash_check,
            )

    # ── 3. 心跳: 失败窗口内全资格 False, 但仍要记下数 (只是不让它造成后果) ──
    # 「没做心跳」不许读成「心跳通过」(spine-review P1-5)。但判据是**到期**而不是
    # 「永远要求」: 短跑 / 离线路径上心跳本就不到期, 一律要求会把每一次读数都降级
    # —— 那与打开 `require_double_confirm` 就把入库率归零是同一个陷阱。于是由调用方
    # 明说 `heartbeat_expected`: 到期而没跑 = 尺子状态**未知**, 无人值守下未知按坏处理。
    hb_checked = heartbeat is not None
    hb_window = str(getattr(heartbeat, "window_id", "") or "")
    # 只有**本窗**的心跳才算数 —— §13.5 的两面: 别窗坏了不牵连本次, 别窗好了也不给本次
    # 背书。「检查过」若按「收到过一张心跳」记, 递一张陈旧心跳就能把没被测过的窗口洗成
    # 已检查。构造上 Conductor 总是传当窗那张, 所以这条只在调用方递错时才响。
    if hb_checked and candidate_window and hb_window != candidate_window:
        hb_checked = False
    hb_reasons: list = []
    if hb_checked:
        # 用窗口判据而不是裸 `ok`: 一张卡/一扇窗的尺子坏了不该牵连别的窗口 (§13.5)。
        hb_broken = ruler_broken_window(heartbeat, candidate_window or "")
        hb_reasons = [str(r) for r in (heartbeat.reasons or [])]
    elif heartbeat_expected:
        hb_broken = True
        if heartbeat is not None:
            hb_reasons = [f"{RULER_BROKEN}: 心跳到期, 但手上这次参照读数是别窗 "
                          f"({hb_window}) 的, 不覆盖本次"]
        else:
            hb_reasons = [f"{RULER_BROKEN}: 标定心跳到期, 但本次没有参照读数"]
    else:
        # 不是本窗的读数 → 对本次不作任何声明 (既不判坏, 也不记成检查过)。
        hb_broken = False
    hb_ok = not hb_broken

    raw = dict(measure())
    value = raw.get("value")
    validity = str(raw.get("validity") or "valid")
    reasons = [str(r) for r in (raw.get("reasons") or [])]
    evidence = [str(e) for e in (raw.get("evidence") or [])]

    # ── 3. 资格位 (设计 §4.3 八条, 短路) ──
    # 先把规则输入收成一份, 再喂 derive_eligibility —— 同一份也进 provenance。
    # 为什么必须留下: 这一族反复出现的失效是「上游传了个错的值, 下游无从分辨」
    # (spine-review P1-6 就是 `cv_exceeded` 被传成「档位不是 VALID」的重编码,
    #  而资格位只回五个布尔, 从中看不出到底哪个输入说它不合格)。
    # 走资格层那一个判据 (`exceeds_measured_peak`), 不在这里再抄一遍阈值 ——
    # spine-review P2-9 的附带发现就是这里抄了字面量 `1.05`: 同一阈值两个真源, 会漂。
    peak_exceeded = bool(raw.get("peak_exceeded")) or exceeds_measured_peak(
        value if isinstance(value, (int, float)) and not isinstance(value, bool) else None,
        peak)
    rule_inputs = {
        "lib_fallback": bool(raw.get("lib_fallback")),
        "peak_exceeded": peak_exceeded,
        "cv_exceeded": bool(raw.get("cv_exceeded")),
        "launch_dominance_exceeded": bool(raw.get("launch_dominance_exceeded")),
        "partition_drift": bool(raw.get("partition_drift")),
        "l3_pass_is_refused": bool(raw.get("l3_pass_is_refused")),
        "real_device": raw.get("real_device"),
        "require_real_device": bool(require_real_device),
    }
    elig = derive_eligibility(
        validity=validity,
        contract_hash_ok=True,               # 上面已拦
        heartbeat_ok=hb_ok,
        requires_real_device=require_real_device,
        real_device=rule_inputs["real_device"],
        lib_fallback=rule_inputs["lib_fallback"],
        peak_exceeded=peak_exceeded,
        cv_exceeded=rule_inputs["cv_exceeded"],
        launch_dominance_exceeded=rule_inputs["launch_dominance_exceeded"],
        partition_drift=rule_inputs["partition_drift"],
        l3_pass_is_refused=rule_inputs["l3_pass_is_refused"],
    )
    if not hb_ok:
        reasons.append(f"{RULER_BROKEN}: 标定心跳失败, 同窗观测全资格 False")
        reasons.extend(hb_reasons)

    prov = {
        "contract_hash": expected,
        "record_hash": record_hash,
        "heartbeat_ok": hb_ok,
        "heartbeat_epoch": getattr(heartbeat, "epoch", None),
        # 「没检查」与「检查过且好」必须分得开: `heartbeat_ok=False` 一种值有两种成因
        # (心跳失败 / 到期没跑), 只看它读不出是哪种 —— 下面两格就是那个区别。
        "heartbeat_checked": hb_checked,
        "heartbeat_expected": bool(heartbeat_expected),
        "heartbeat_window_id": getattr(heartbeat, "window_id", ""),
        "heartbeat_reasons": list(hb_reasons),
        "heartbeat_ratios": {
            "good": getattr(heartbeat, "known_good_ratio", None),
            "slow": getattr(heartbeat, "known_slow_ratio", None),
        },
        "baseline": contract.baseline.name,
        "dtype_semantics": {
            "tf32": contract.dtype_semantics.tf32,
            "accum": contract.dtype_semantics.accum,
        },
    }
    prov.update(raw.get("provenance") or {})
    # 资格位的**原料** (见上面 rule_inputs 的说明): 五个布尔回推不出是哪个输入判的。
    # 写在 `prov.update(...)` **之后** —— 这是评测器自己的审计记录, 不该被载荷的
    # provenance 覆盖; 能改写「为什么判成这样」的记录等于没有记录。
    prov["rule_inputs"] = dict(rule_inputs)

    # 规则 1~5 命中时把档位一并记成 `uninformative`: 那些规则说的是「这个数不能支撑
    # **任何**结论」(出处不对 / 测的是别的库 / 超峰值), 而档位的定义正是「这个数能支撑
    # 到什么程度」。不这么做的话会出现 `can_verdict=False` 配 `validity="valid"` 的自相
    # 矛盾记录, 而 `confirm_persist` 那类读者只看档位 —— 两处说法必须一致。
    # 「为什么」没有被丢掉: 档位说的是结论强度, 成因在 `reasons` 与 `rule_inputs` 里。
    # 规则 6~7 (DEGRADED) **不**走这一支: `can_verdict` 仍为 True, 档位照实记 degraded
    # —— 低资格不等于没测准 (设计 §4.3 的两条设计意图)。
    return Observation(
        metric=metric,
        value=value if isinstance(value, (int, float)) and not isinstance(value, bool) else None,
        unit=unit,
        validity="uninformative" if not elig.can_verdict else validity,
        eligibility=elig,
        evidence=evidence,
        reasons=reasons,
        provenance=prov,
        hash_check=hash_check,
        ruler_broken=hb_broken,
    )


def admit_to_corpus(metrics: Mapping,
                    remeasure: Optional[ConfirmRecord],
                    *,
                    min_speedup: float = 1.0,
                    requires_remeasure: bool = True) -> tuple:
    """入库判据 —— **唯一**一处决定「这次读数能不能进长期记忆」(设计 §10.5-4)。

    三道门, 顺序即短路顺序:
      1. `persist_decision(metrics)` —— 这一次读数够不够格 (缺证据 / 不是真卡 / 被抢占)
      2. 资格位 `can_persist` —— DEGRADED 不许写长期记忆
      3. (可选) 两次独立确认 —— `requires_remeasure=True` 时由 `confirm_persist` 判

    返回 (允许?, 理由)。任一关不过就拒。

    **为什么吃 `metrics` 而不是 `Observation`** (2026-10-01): 生产路径手里只有**序列化过的**
    读数 (`VerificationResult.metrics` 搬的是能 JSON 化的东西), 而判据原先要一个 `Observation`
    对象 ⇒ 调用方只能**自己再写一遍**这三道门, 于是同一个决定有了两个实现、两种理由文本,
    而用例测的是没有生产调用者的那一个 —— 「判据不在链路上」的同族 (spine-review P0-1)。
    现在判据的输入就是生产路径真正持有的东西: 同一份 metrics, 同一处产出理由。
    形状与 `measurement.persist_decision` 一致 (它也是吃 metrics 的判据), 两道门不再各说各话。

    **两处三态, 都是刻意的** (缺键 ≠ 判过且不合格):
      - `metrics["eligibility"]` **缺失** = 这条读数没经过收口点 (老产物) ⇒ 不做追溯性重判;
        有键而 `can_persist` 缺失 ⇒ 判**不合格** (五位掩码缺一位就没有「能入库」的凭据);
      - `metrics["observation"]` 缺失 = 没有评测记录 ⇒ 两次确认那条判据拿不到判据戳,
        按 `confirm_persist` 的既有规矩「无从证明两次量的是同一条判据」时**没带就拒**
        (它只在首次那头给了哈希时才判同一性 —— 这里退化成不判, 不是判过)。

    ⚠️ 第 3 道门默认**开**, 但生产路径把它接到 `Conductor.require_double_confirm` (默认
    **False**) —— 多量一遍等于每个要入库的产物多付一次 L3, 那是机时承诺, 所以要显式打开。
    代价必须写清楚: **默认配置下单次读数即可入库**, 理由文本里会写明本次没要求两次确认,
    免得日后有人把「入库了」读成「两次确认通过了」。
    """
    allow, why = persist_decision(metrics)
    if not allow:
        return False, why

    elig = metrics.get("eligibility")
    if isinstance(elig, Mapping):
        if not elig.get("can_persist", False):
            obs = metrics.get("observation") or {}
            causes = [str(r) for r in (obs.get("reasons") or [])]
            # 资格位只回五个布尔, 从中读不出**是哪条规则**拒的 —— 而「资格位不够」这句话
            # 对着一屏日志等于没说。判据的原料在 `provenance.rule_inputs` 里 (P1-6 留它
            # 就是为了这件事), 把真正触发的那几项命名出来。
            # 只认「True 即拒」的六项: `real_device`/`require_real_device` 不能照套
            # —— 那两个为 True 表示**要求被满足**, 列成触发项会说反话。
            ri = (obs.get("provenance") or {}).get("rule_inputs") or {}
            hits = [k for k in ("lib_fallback", "peak_exceeded", "cv_exceeded",
                                "launch_dominance_exceeded", "partition_drift",
                                "l3_pass_is_refused") if ri.get(k) is True]
            if ri.get("require_real_device") and ri.get("real_device") is False:
                hits.append("real_device=缺失 (任务要求 REAL_DEVICE)")
            if hits:
                causes.append("触发项: " + ", ".join(hits))
            if not causes:
                causes.append("见 eligibility 规则")
            return False, (f"资格位 can_persist=False (validity="
                           f"{metrics.get('measurement_validity') or '未报'}): "
                           + "; ".join(causes))

    if not requires_remeasure:
        return True, (f"{why} —— 本次**未要求**两次确认 (requires_remeasure=False): "
                      f"单次读数即可入库, 这不是「两次确认已通过」")

    obs = metrics.get("observation") or {}
    prov = obs.get("provenance") or {}
    ok, why2 = confirm_persist(
        metrics.get("speedup_vs_baseline"),
        str(metrics.get("measurement_validity") or ""),
        remeasure,
        min_speedup=min_speedup,
        # 两次确认的**前提**是同一条判据。优先取评测器自己盖的那个戳 (它记录的是
        # 「这条读数在哪条契约下取得」), 没有评测记录时退回 metrics 上的那个 (P0-2 的
        # 同款理由: 能由记录携带的, 就不要让调用方随手给)。
        first_contract_hash=str(prov.get("contract_hash")
                                or metrics.get("contract_hash") or ""))
    if not ok:
        return False, why2
    return True, f"{why} —— 两次确认: {why2}"
