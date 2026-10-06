"""无人值守的两道收尾: 重测确认 + 「无法确立」一等公民报告

设计出处: `docs/design/autonomous-operator-agent-design.md`
  * §10.2 最后 10% 预算不可挪用 —— 任何进长期记忆的结论必须留一次独立重测
  * §10.5-4 永久结论需两次确认: 一次 valid 且 FAST + 一次独立重测仍不劣
  * §17 报告必须含「我无法确立的事」章节 —— 从不说测不准的系统, 所有数字都不可信

与既有机制的关系:
  * `persist_decision()` 问的是「这一次读数够不够格入库」
  * **本模块**问的是「确认之后还允不允许入库」—— 一次侥幸读数不能变成永久真理
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ConfirmRecord:
    """一次独立重测的结果。`n` 次 ConfirmRecord 里至少两次「不劣」才允许入库。"""
    ok: bool
    speedup: Optional[float] = None
    validity: str = ""
    reason: str = ""
    #: 是否与首次测量独立 (不同窗口 / 不同随机种子)。False = 不能当第二次确认
    independent: bool = True
    #: 这次重测是在**哪条契约**下取得的 (spine-review P0-2 的残留那个洞)。
    #: 两次确认的意义是「同一条判据下量了两次」; 判据在两次之间被改写, 就成了两条不同
    #: 断言各量一次 —— 那不是确认。空串 = 没带, 由 `confirm_persist` 按「无从证明」处理
    #: (只在首次那头给了哈希时才判, 见那里的说明)。
    contract_hash: str = ""


def confirm_persist(first_speedup: float,
                    first_validity: str,
                    remeasure: Optional[ConfirmRecord],
                    min_speedup: float = 1.0,
                    first_contract_hash: str = "") -> Tuple[bool, str]:
    """设计 §10.5-4: 永久结论需两次确认。

    规则:
      1. 首次必须 valid 且 speedup > min_speedup (已由 persist_decision 把关, 这里再挡一次)
      2. 必须有一次**独立**重测
      3. 重测不得比首次「明显劣化」—— 判为劣化的带宽用 validity 分档
         (DEGRADED 允许更宽的噪声, UNINFORMATIVE 直接不能当第二次)
      4. 重测必须与首次在**同一条契约**下取得 (给了 `first_contract_hash` 时才判)

    没有重测记录 → **拒绝入库** (而不是「没有重测就当过了」)。这是 10% 预算
    存在的意义: 没留预算就等于没做完。
    """
    if first_validity == "uninformative":
        return False, "首次测量 uninformative: 连第一次确认都不够格"
    if not isinstance(first_speedup, (int, float)) or isinstance(first_speedup, bool):
        return False, f"首次没有可用 speedup ({first_speedup!r})"
    if first_speedup <= min_speedup:
        return False, f"首次 speedup={first_speedup} 未过门槛 {min_speedup}"

    if remeasure is None:
        return False, "没有独立重测: 永久结论需要两次确认 (设计 §10.5-4)"

    if not remeasure.independent:
        return False, "重测不是独立测量 (同窗口/同种子), 不能当第二次确认"

    # 判据同一性 (spine-review P0-2 的残留): 两次确认的**前提**是同一条判据。首次那头
    # 给了哈希才判 —— 库里/调用方没有这个信息时不能凭空造一条, 否则就是「恒真」的反面:
    # 恒假。生产路径 (`_save_to_db`) 会给, 因为 L3 每条读数都盖了溯源戳。
    if first_contract_hash:
        second_hash = str(getattr(remeasure, "contract_hash", "") or "")
        if not second_hash:
            return False, ("重测没带契约哈希: 无从证明两次量的是同一条判据 "
                           "(判据被改写时, 「两次确认」就退化成两次不同的确认)")
        if second_hash != str(first_contract_hash):
            return False, (f"重测在另一条契约下取得 ({second_hash[:12]}… != "
                           f"{str(first_contract_hash)[:12]}…) —— 判据在两次测量之间变了")
    if not remeasure.ok:
        return False, f"重测失败: {remeasure.reason or 'ok=False'}"
    if remeasure.validity == "uninformative":
        return False, "重测 uninformative: 不能用坏尺子确认好结论"
    if not isinstance(remeasure.speedup, (int, float)) or isinstance(remeasure.speedup, bool):
        return False, f"重测没有可用 speedup ({remeasure.speedup!r})"

    # 「仍不劣」: 允许退到门槛之上一点点, 不允许掉回门槛下
    # (不需要与首次逐位相等 —— 噪声带内两次读数本来就会漂)
    if remeasure.speedup < min_speedup:
        return False, (f"重测 speedup={remeasure.speedup:.4f} 掉回门槛 {min_speedup} "
                       f"(首次 {first_speedup:.4f})")
    return True, (f"两次确认: 首次 {first_speedup:.4f}x / 重测 {remeasure.speedup:.4f}x "
                  f"(均 > {min_speedup})")


# ─── 「无法确立」报告 (设计 §17) ───────────────────────────────

@dataclass
class UnableItem:
    """一条没能确立的结论。故意不带「估计值」—— 报告里只写能不能确立。"""
    claim: str
    reason: str
    kind: str = "other"   # measurement | provenance | budget | ruler | contract
    evidence: str = ""


@dataclass
class UnableToEstablish:
    items: List[UnableItem] = field(default_factory=list)
    #: 「这一节为空」本身也是一个**断言**, 所以它必须带依据。
    #:
    #: 为什么不能像以前那样直接印一个 `(无)` 就过去: 空清单有两种来源 —— 「全部读数都
    #: 站得住」与「压根没收集」, 在产物里两者长得一模一样。这与 `measurement.assess()`
    #: 里「地板未知 ⇒ 记录『未做检查』而不是默认放行」是同一条纪律: 缺证据不是有效证据。
    #: 由 `report.collect_unable()` 在逐项核过之后填写; 手工构造的对象不填就会在 render
    #: 里被点名。
    absence_basis: Optional[str] = None

    def add(self, claim: str, reason: str, kind: str = "other", evidence: str = "") -> None:
        self.items.append(UnableItem(claim=claim, reason=reason, kind=kind, evidence=evidence))

    def extend_metrics(self, metrics: dict) -> None:
        """从一次评估的 metrics 里捞「说不清」的信号, 不猜数字。"""
        if metrics.get("measurement_can_gate") is False:
            self.add("该轮加速比不足以支撑结论",
                     "; ".join(metrics.get("measurement_validity_reasons") or ["can_gate=False"]),
                     kind="measurement",
                     evidence=f"validity={metrics.get('measurement_validity')}")
        if metrics.get("measurement_l3_pass_refused"):
            self.add("L3 通过/不通过无指称对象 (出处不对)",
                     "capability_shortfall/provenance_conflict", kind="provenance")
        if metrics.get("ruler_broken"):
            self.add("该窗口内所有数字不可用", "标定心跳失败 (RULER_BROKEN)",
                     kind="ruler",
                     evidence=str(metrics.get("ruler_reason") or ""))
        if metrics.get("contract_hash_mismatch"):
            self.add("契约被改写, 本次评测作废", "contract hash mismatch", kind="contract")
        if metrics.get("benchmark_contaminated"):
            self.add("测量窗口被抢占, 加速比不可比", "util 超阈值", kind="measurement")
        # 下面三条 (峰值未实测 / 该测的键缺了 / 资格位不许入库) 是 2026-10-01 补的:
        # 它们此前**没有任何读者** —— 判据算了、日志也打了, 但没人把它收进「说不清」
        # 这一节, 于是报告里读起来像是「这一轮一切正常」。
        _rl = metrics.get("roofline_analysis") or {}
        if _rl and _rl.get("reference_lower_bound") is not True:
            # 与 measurement.reference_ceiling_reached 同一个位: 分母没有参考下界 ⇒
            # 天花板出口关闭, 于是「已经贴着参考实现的天花板了」这句话这一轮**没有**
            # 获得资格 (设计 §13.5)。这句说的是**参考下界**有没有, 不是器件峰值:
            # 器件峰值那个量具 (`peak_measured`) 代码里还不存在。
            self.add("无法宣布已到参考天花板", "参考下界未取得 (同一次 L3 的基线臂没过探针判据)",
                     kind="measurement",
                     evidence=f"peak_source={_rl.get('peak_source', '')!r}")
        _elig = metrics.get("eligibility")
        if isinstance(_elig, dict) and _elig.get("can_persist") is False:
            self.add("该轮读数不足以支撑永久结论 (不进长期记忆)",
                     "资格位 can_persist=False",
                     kind="measurement",
                     evidence=f"eligibility={_elig}")
        # 三态: **缺键 ≠ 没加速**。只在「这明显是一份 L3 读数」时才这么问, 否则
        # 别的调用方 (传小字典进来的量具) 会被这句话淹没。
        # `can_gate is False` 那条**已经**说过这句话了 (「该轮加速比不足以支撑结论」),
        # 再加一条只是同一件事数两遍 —— 这一节一旦开始复读就会被跳过不读。
        if (metrics and metrics.get("measurement_can_gate") is not False
                and "speedup_vs_baseline" not in metrics
                and any(k in metrics for k in ("measurement_validity", "gflops",
                                               "measurement_can_gate", "roofline_analysis"))):
            self.add("这一轮没有可比加速比", "缺 speedup_vs_baseline 键 (缺键 ≠ 没加速)",
                     kind="measurement")

    def render(self) -> str:
        if not self.items:
            if self.absence_basis:
                return ("【我无法确立的事】(无)\n"
                        f"  依据: {self.absence_basis}\n"
                        "  ⚠️ 「无」是一个断言, 上面这条依据就是它的证据 —— "
                        "缺任何一项, 这一节就不该印成空的。")
            return ("【我无法确立的事】⚠️ 无法确立「没有无法确立的事」\n"
                    "  原因: 这一节是空的, 但没有任何依据说明它为什么是空的\n"
                    "  (空清单的两种来源 —— 「都站得住」与「压根没收集」—— 在产物里长得一样; "
                    "缺证据不是有效证据。)")
        lines = ["【我无法确立的事】(设计 §17 —— 一等公民, 不是附注)"]
        for i, it in enumerate(self.items, 1):
            lines.append(f"  {i}. [{it.kind}] {it.claim}")
            lines.append(f"     原因: {it.reason}")
            if it.evidence:
                lines.append(f"     证据: {it.evidence}")
        lines.append("  ⚠️ 以上条目**不许**被读成否定结论 (「没测出」≠「没有」)。")
        return "\n".join(lines)
