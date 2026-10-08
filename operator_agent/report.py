"""一次自动跑结束时的**报告产物** —— best-known-good + 证据链 + 「我无法确立的事」

设计 §17 列了四件交付物, 今天的状态是:

  1. best-known-good (代码 + 完整配置)      → `ArtifactStore` 有 (code.* + sketch.json)
  2. 不可变实验日志 (JSONL, 逐次读数)        → `ArtifactStore` 有 (result/verification/benchmark)
  3. **报告: 结论 + 证据链 + 「我无法确立的事」** → 此前**完全没有产物** ← 本模块
  4. 复现命令 + 硬件/驱动/库指纹              → 指纹有 (result.json), 复现命令散在文档里

第 3 件是设计里唯一被写成「**自动跑系统的信誉所在**」的一件:

    一个从不说「我测不准」的自动系统, 它的所有数字都不可信。

而它此前连一个调用点都没有: `confirm.UnableToEstablish` (收集器) 早就写好了, 但全仓
只有两个 benchmark 脚本调它 (`benchmarks/demo_spine_local.py` /
`harness_pass_rate.py`) —— 产品链路跑完一次真实验, 只留下 JSON 与一堆 print, 没有人
能打开一份文件回答「这一轮哪些事**没能**确立」。这与 CHECKLOG 里那一族「算了但没接上」
是同一种缺陷, 只是这次缺的是**整件交付物**。

两份纪律写在这里, 因为它们决定了本模块的形状:

**一、报告是产物, 不是日志。** 它必须能在 run 结束之后被单独打开; 所以它只吃
`KernelArtifact` + 少量显式传入的上下文 (逐轮观测 / 入库判据 / 已收集的 unable), 不
自己去读环境、不联网、不碰 GPU —— 也就不会因为「跑的时候顺手」而把一个不可复现的东西
写进来。

**二、「这一节为空」是一个断言, 不是省略。** 空清单有两种来源: 「所有读数都站得住」与
「压根没收集」, 在产物里两者长得一模一样 (`measurement.assess()` 里「地板未知 ⇒ 记录
『未做检查』而不是默认放行」是同一条纪律)。所以本节为空时, `_preconditions()` 逐项核过
的那张清单会作为 `absence_basis` 印出来; 核不过的项则变成条目本身。
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .confirm import UnableItem, UnableToEstablish

#: 产物目录里的文件名。固定名 (不是 `report_<ts>.md`): 一个 run 一个目录, 目录本身就是
#: 时间戳, 再加一层会让「哪份是这个 run 的」变成一个需要扫描才能回答的问题。
REPORT_NAME = "report.md"

#: 逐轮血缘里 L3 那一级在 `levels_passed` 里的键 (`VerificationLevel.value`)
_L3 = "L3"

#: 「这一份看起来是一轮 L3 读数」的标记键。用来区分 L3 metrics 与随手传进来的小字典,
#: 只有前者才该被问「加速比呢」。
_L3_MARKERS = ("measurement_validity", "gflops", "measurement_can_gate",
               "roofline_analysis")


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """dict 取键的容错版: 产物里的块可能是 None / 不是 dict (老记录、半成品 artifact)"""
    return obj.get(key, default) if isinstance(obj, dict) else default


def _last_metrics(artifact: Any) -> Dict[str, Any]:
    """最后一轮的 L3 读数。`benchmark_data` 只保留最后一轮 —— 逐轮的档位与加速比在
    `iteration_history` 里, 完整 metrics 只有这一份 (见下面的证据链表)。"""
    m = getattr(artifact, "benchmark_data", None)
    return m if isinstance(m, dict) else {}


def _rounds(artifact: Any) -> List[Dict[str, Any]]:
    hist = getattr(artifact, "iteration_history", None)
    return [h for h in hist if isinstance(h, dict)] if isinstance(hist, list) else []


def _uncovered_rules_from_lineage(artifact: Any) -> List[Tuple[str, str]]:
    """§21.4/§22.3 的覆盖缺口: 最后一轮裁决里「有对象但没量具」的判据 → (rule, detail)

    只取**最后一轮有裁决的**那一轮: 报告说的是「这次运行结束时还立不住的事」, 更早轮次的
    缺口若已经补上, 再印一遍就是复读（这一节一旦开始复读就会被跳过不读）。
    三态: 没有血缘 / 没有 `agent_action` / 没有 `coverage` ⇒ 返回空 —— 那表示**没判过**
    （老产物、没走 §21 的路径），不是「没有缺口」。

    ⚠️ 这里只读 `uncovered_rules`（= `unknown`: 有对象、没量具）。`not_applicable` 是
    「这一份候选没有这个对象」, **不是**这一节能说的事 —— 两者在 2026-10-07 之前被并成
    一个集合（CHECKLOG I90），读起来像「这一层的静态检查很弱」。
    """
    for h in reversed(_rounds(artifact)):
        pre = _get(_get(h, "agent_action", None), "precompile", None)
        cov = _get(pre, "coverage", None)
        if not isinstance(cov, dict):
            continue
        uncovered = cov.get("uncovered_rules") or []
        details = {o.get("rule"): o.get("detail", "")
                   for o in (_get(pre, "outcomes", None) or []) if isinstance(o, dict)}
        return [(str(r), str(details.get(r, ""))) for r in uncovered]
    return []


# ─── 「空清单的依据」: 逐项核过, 核不过就变成条目 ────────────────────


def _preconditions(artifact: Any) -> List[Tuple[str, bool, str, Optional[UnableItem]]]:
    """要印「(无)」必须先逐项核过这 8 件事。

    每一项返回 `(名称, 是否成立, 依据串, 不成立时的条目)`。**依据串在成立时也要给** ——
    它就是要印出来的那份证据 (「无」凭什么)。
    """
    m = _last_metrics(artifact)
    rl = _get(m, "roofline_analysis", {}) or {}
    elig = _get(m, "eligibility", None)
    drift = getattr(artifact, "environment_drift", None)
    checks: List[Tuple[str, bool, str, Optional[UnableItem]]] = []

    # 1. 至少有一条真值 (L3) 读数 —— 否则「最好」这个词没有指称对象
    has_l3 = bool(m) and any(k in m for k in _L3_MARKERS)
    checks.append((
        "有过 L3 真值读数", has_l3, f"benchmark_data 键数={len(m)}",
        None if has_l3 else UnableItem(
            claim="本次运行没有留下任何性能读数 (best-known-good 无从谈起)",
            reason="没有走到 L3, 或 L3 结果没有落进 benchmark_data",
            kind="measurement",
            evidence=f"iteration_history 轮数={len(_rounds(artifact))}"),
    ))

    # 2. 那个数能判事 (`can_gate`)。缺键不算通过: 缺证据不是有效证据。
    can_gate = _get(m, "measurement_can_gate", None)
    checks.append((
        "读数可判事 (can_gate=True)", can_gate is True, f"can_gate={can_gate!r}",
        None if can_gate is True else UnableItem(
            claim="该轮加速比不足以支撑任何结论",
            reason="measurement_can_gate 不是 True",
            kind="measurement",
            evidence=f"validity={_get(m, 'measurement_validity')!r} "
                     f"reasons={_get(m, 'measurement_validity_reasons')}"),
    ))

    # 3. 档位是 valid (degraded 时「数不可信」, uninformative 时判都判不了)
    validity = _get(m, "measurement_validity", None)
    checks.append((
        "档位 valid", validity == "valid", f"validity={validity!r}",
        None if validity == "valid" else UnableItem(
            claim="这一轮的读数不在 valid 档",
            reason=f"measurement_validity={validity!r} (degraded=数不可信 / "
                   "uninformative=判不了)",
            kind="measurement",
            evidence=f"reasons={_get(m, 'measurement_validity_reasons')}"),
    ))

    # 4. 标定心跳**检查过且是好的**。三态, 不许合并 —— 这是本模块最容易写错的一条:
    #      False = 检查过, 是好的   ⇒ 通过
    #      True  = 失败/到期没跑    ⇒ 该窗口数字不可用
    #      缺键  = **本次与心跳无关** ⇒ 不通过! 「没检查」不许伪装成「检查过且好」。
    #    这条纪律不是我加的, 是 `verifier.py:1290` 那三行注释的原话 (缺口 1 的失效方式:
    #    一个键只在坏的时候才写, 读它的人就永远只能看到坏消息, 于是「没跑」=「没消息」=
    #    「看起来没事」)。
    ruler_broken = _get(m, "ruler_broken", None)
    if ruler_broken is False:
        checks.append(("标定心跳检查过且是好的", True,
                       f"ruler_broken=False", None))
    elif ruler_broken is True:
        checks.append(("标定心跳检查过且是好的", False, "ruler_broken=True",
                       UnableItem(claim="该窗口内所有数字不可用",
                                  reason="标定心跳失败 (RULER_BROKEN)",
                                  kind="ruler",
                                  evidence=str(_get(m, "ruler_reason", "")))))
    else:
        checks.append(("标定心跳检查过且是好的", False, f"ruler_broken={ruler_broken!r}",
                       UnableItem(
                           claim="本次没有标定心跳的检查记录",
                           reason="没检查 ≠ 检查过且好 (设计 §10.5-1: 心跳是无人值守下"
                                  "唯一能发现「量具悄悄坏了」的机制)",
                           kind="ruler",
                           evidence="ruler_broken 键缺失: 本次与心跳无关, 或心跳没接上")))

    # 5. 测量窗口没被邻居抢占
    contaminated = _get(m, "benchmark_contaminated", None)
    checks.append((
        "窗口未被抢占", contaminated is not True, f"contaminated={contaminated!r}",
        None if contaminated is not True else UnableItem(
            claim="测量窗口被抢占, 加速比不可比",
            reason="util 超阈值 (benchmark_contaminated)",
            kind="measurement"),
    ))

    # 6. 峰值分母**有参考下界** —— 没有则参考天花板出口关闭 (设计 §13.5, CHECKLOG I49)。
    #    这条与 measurement.reference_ceiling_reached 读的是同一个位。
    reference_lower_bound = _get(rl, "reference_lower_bound", None)
    checks.append((
        "峰值分母有参考下界", reference_lower_bound is True, f"reference_lower_bound={reference_lower_bound!r}",
        # 条目文本与 `confirm.extend_metrics` 里那一条**逐字相同** —— 不是巧合:
        # 两处都会对同一份读数说话, 而措辞差一点就会被读成「两件事」(去重键是
        # (claim, reason))。同一条判据在两个地方发话, 说法就必须只有一种。
        None if reference_lower_bound is True else UnableItem(
            claim="无法宣布已到参考天花板",
            reason="参考下界未取得 (同一次 L3 的基线臂没过探针判据)",
            kind="measurement",
            evidence=f"peak_source={_get(rl, 'peak_source', '')!r}"),
    ))

    # 7. 环境漂移做过比较、且没漂。**没比较**同样不能算过: 「无从比较」与「比较了没问题」
    #    在产物里差别极大, 而这一条正是无人值守下唯一能发现「量具悄悄换了个世界」的地方。
    if not isinstance(drift, dict):
        checks.append((
            "环境漂移已比较", False, f"environment_drift={drift!r}",
            UnableItem(claim="本次无法比较环境漂移",
                       reason="没有上一次运行的指纹, 或本次没做比较",
                       kind="provenance"),
        ))
    else:
        comparable, drifted = drift.get("comparable"), drift.get("drifted")
        ok = comparable is True and drifted is not True
        checks.append((
            "环境漂移已比较且未漂", ok,
            f"comparable={comparable!r} drifted={drifted!r} "
            f"changed={drift.get('changed')}",
            None if ok else UnableItem(
                claim="跨漂移的数字不可比",
                reason=("环境指纹与上一次不可比" if comparable is not True
                        else "环境已漂移 (驱动/库/卡变了)"),
                kind="provenance",
                evidence=f"changed={drift.get('changed')} "
                         f"previous_from={drift.get('previous_from')}"),
        ))

    # 8. 契约没被改写 (设计 §13.4 的反 reward hacking 物理收口)。「没被改写」要有依据:
    #    收口点跑过 (它发布资格位) 且没有不匹配记录。**收口点没跑过**时不许印成通过 ——
    #    那正是真机 §14-8 的形态: 记录里说 A、判定按 B, 而报告读的是 A。
    mismatch = _get(m, "contract_hash_mismatch", None)
    if not isinstance(elig, dict):
        checks.append(("契约哈希核过", False, "eligibility 缺失 (收口点没跑)",
                       UnableItem(claim="这次评测没有经过收口点",
                                  reason="metrics 里没有 eligibility 块 —— "
                                         "契约哈希因此无从校验",
                                  kind="contract")))
    elif mismatch is True:
        checks.append(("契约哈希核过", False, "contract_hash_mismatch=True",
                       UnableItem(claim="契约被改写, 本次评测作废",
                                  reason="contract hash mismatch", kind="contract")))
    else:
        checks.append(("契约哈希核过", True,
                       f"收口点跑过且无不匹配记录 (mismatch={mismatch!r})", None))

    # 资格位不属于「核不过」那一类 —— can_persist=False 说的是**后果** (不许写长期记忆),
    # 不是「这个数测不准」。它由 `extend_metrics` 收成条目, 并在 best-known-good 一节
    # 明写「这条没有进长期记忆」。
    _ = elig
    return checks


# ─── 收集 ──────────────────────────────────────────────────────────


def collect_unable(*, artifact: Any, observations: Optional[Sequence[dict]] = None,
                   refusals: Optional[Sequence[str]] = None,
                   unable: Optional[UnableToEstablish] = None) -> UnableToEstablish:
    """把「这一轮说不清的事」收成一份 `UnableToEstablish`。

    `observations` 是逐轮的 L3 metrics (可选: 调用方手里有就传, 没有就只能靠最后一轮 ——
    这本身就是一条要在报告里说清的口径差别)。
    `refusals` 是入库判据的拒因 (`oracle.admit_to_corpus` 的 reason), 它决定「这条结论
    有没有进长期记忆」。
    """
    u = unable if unable is not None else UnableToEstablish()

    # 1. metrics → 条目 (收集器**只有一处**: `confirm.UnableToEstablish.extend_metrics`)
    docs: List[dict] = []
    last = _last_metrics(artifact)
    if last:
        docs.append(last)
    for o in observations or []:
        if isinstance(o, dict):
            docs.append(o)
    for d in docs:
        u.extend_metrics(d)

    for r in getattr(artifact, "verification_results", ()) or ():
        c = (getattr(r, "metrics", {}) or {}).get("correctness", {})
        if c.get("verdict") == "unknown":
            unknown = [row for row in c.get("cases", ()) if row.get("verdict") == "unknown"]
            for row in unknown:
                u.add(f"正确性检查 {row.get('case_id')} 无法确立", row.get("reason", ""),
                      kind="other", evidence=f"suite_hash={c.get('suite_hash')}")
            if not unknown:
                u.add("CorrectnessSuite 无法确立", c.get("reason", r.message), kind="other")

    # 2. 逐项前置 (核不过就是条目; 全过则成为空清单的依据)
    checks = _preconditions(artifact)
    for _name, ok, _evidence, item in checks:
        if not ok and item is not None:
            u.items.append(item)

    # 3. 入库判据的拒因。它以前只被 `print` 出来 (conductor `_save_to_db`), 产物里
    #    读不到 —— 「这条结论为什么没进语料」是报告里最要紧的一句之一。
    reasons = [str(x) for x in (refusals or [])]
    adm = _get(getattr(artifact, "metadata", None), "corpus_admission", None)
    if isinstance(adm, dict) and adm.get("kept") is False and adm.get("reason"):
        reasons.append(str(adm["reason"]))
    for reason in reasons:
        u.add("这条结论没有进入长期记忆", reason, kind="other",
              evidence="oracle.admit_to_corpus 的拒因")

    # 4. §21.4/§22.3 的覆盖缺口 («有对象但没量具» 的判据)。覆盖数据在血缘里留了痕,
    #    读它的只有这里 —— 一条判据没有量具是**这一轮没能确立**的事, 必须出现在这一节,
    #    否则读产物的人会以为源码级「幻觉 API」是查过的（真实代价: 曾白烧一轮 Fixer）。
    gaps = _uncovered_rules_from_lineage(artifact)
    if gaps:
        rules = ", ".join(sorted(r for r, _ in gaps))
        detail = "; ".join(d for _, d in gaps if d)
        u.add(f"这一层的静态审计有判据没有量具 ({rules})",
              detail or "该判据有对象但没有量具 (§21.4 / §22.3)",
              kind="measurement",
              evidence=("iteration_history[].agent_action.precompile.coverage"
                        ".uncovered_rules"))

    # 5. 去重: 同一条 (claim, reason) 可能同时来自 metrics 与前置项 —— 这一节一旦开始
    #    复读就会被跳过不读。
    seen, uniq = set(), []
    for it in u.items:
        key = (it.claim, it.reason)
        if key not in seen:
            seen.add(key)
            uniq.append(it)
    u.items = uniq

    if not u.items:
        u.absence_basis = "; ".join(
            f"{name} [{evidence}]" for name, ok, evidence, _ in checks if ok)
    return u


# ─── 渲染 ──────────────────────────────────────────────────────────


def _claim_level_lines(m: Dict[str, Any]) -> List[str]:
    """§4.4 的声明等级（一行 + 受限时的原因）。

    缺键 ⇒ **不印**：缺证据不是有效证据，而印一个默认等级会让「没算」读成「算出来是它」。
    这一行的量（`ceiling` / `coverage_mode`）与入库门读的是同一个 dict ——
    `metrics["claim_level"]` 由 `Conductor._save_to_db` 一处写入。
    """
    c = _get(m, "claim_level", None)
    if not isinstance(c, dict) or not c.get("level"):
        return []
    out = [f"- 声明等级 (§4.4): `{c['level']}` (上限 `{c.get('ceiling', '?')}`, "
           f"静态覆盖 `{c.get('coverage_mode') or '未判'}`)"]
    for reason in (c.get("reasons") or []):
        out.append(f"  - {reason}")
    return out


def _tri(value: Any) -> str:
    """三态显示：`True`/`False`/`None` ⇒ `是`/`否`/`未确立`（形状不对也按未确立印）

    与判据侧同一条纪律：`None` = **未确立**，不许印成「否」—— 报告里一个「否」会被读成
    「量过、没有越限」，而那是另一件事（`tests/test_implementation_decision.py` 钉住）。
    """
    if value is True:
        return "是"
    if value is False:
        return "否"
    return "未确立"


def _implementation_decision_lines(m: Dict[str, Any]) -> List[str]:
    """§17.1 的实现裁决（交付物第 1 件里那一个显式结论 + 它为什么是这个结论）。

    缺键 ⇒ **不印**（同 `_claim_level_lines`）：印一个默认裁决会让「没算」读成「算出来是它」，
    而这个字段的后果是**部署决定**（§17.1 原句：「候选比 baseline 快」不足以支撑它）。
    `metrics["implementation_decision"]` 由 `Conductor._save_to_db` 一处写入。
    """
    d = _get(m, "implementation_decision", None)
    if not isinstance(d, dict) or not d.get("decision"):
        return []
    out = [f"- 实现裁决 (§17.1): `{d['decision']}` "
           f"(声明等级 `{d.get('claim_level') or '未判'}`, "
           f"上限 `{d.get('claim_ceiling') or '未判'}`; "
           f"必判 case 全对 `{_tri(d.get('required_cases_ok'))}`, "
           f"回退越限 `{_tri(d.get('regression_over_limit'))}`)"]
    for reason in (d.get("reasons") or []):
        out.append(f"  - {reason}")
    return out


def _performance_verdict_lines(m: Dict[str, Any]) -> List[str]:
    """§16 收口那一格：这一次性能面的**成功标准是什么**（CHECKLOG I95）

    三态: 老产物没有这一格（`performance_verdict`）⇒ 一个字都不印 —— 不许拿「没这一格」
    去猜「按主 case 判的」。`basis` 是这一片唯一的新不变量: 契约声明了目标函数就判目标
    函数；没声明才退回主 case，并且**把「§16 没对它收口」印出来** —— 单形状成功标准在
    这台机器上仍然可能发生，但它必须是看得见的，不是默认。
    """
    v = m.get("performance_verdict")
    if not isinstance(v, Mapping):
        return []
    verdict = "过" if v.get("passed") is True else ("不过" if v.get("passed") is False else "未确立")
    if v.get("basis") == "objective":
        value = v.get("value")
        shown = f"{float(value):.4g}" if isinstance(value, (int, float)) else "未确立"
        floor = v.get("min_speedup")
        # 门槛只在**它真是一个数**的时候印: 老产物/局部载荷里可能没有这一格, 印一个
        # 字面的 `None` 会让人以为「门槛是 None」是一条判据。
        floor_text = f"(门槛 ≥ {floor})" if isinstance(floor, (int, float)) else ""
        out = [f"- 性能结论依据 (§16): **目标函数** = {shown} {floor_text} ⇒ {verdict}"
               f"（契约声明了目标函数 ⇒ 判的是分布, 不是单形状）"]
    else:
        out = [f"- 性能结论依据 (§16): **主 case 加速比** ⇒ {verdict}"
               f" —— ⚠️ 契约没声明目标函数 ⇒ 这一次的成功标准是**单形状**的（§16 未对它收口）"]
    if v.get("why"):
        out.append(f"  - {v['why']}")
    if v.get("cases") is not None:
        out.append(f"  - 本次账本里的 case 数: {v.get('cases')}"
                   + ("（逐 case 硬回退判据: " + _tri(v.get("regression_over_limit")) + "）"
                      if "regression_over_limit" in v else ""))
    return out

def _case_input_lines(m: Dict[str, Any]) -> List[str]:
    """§3.4 的**输入来源**与「契约↔本次」对账（三态：契约没声明 ⇒ 不印）

    `case_source` 空串 = 契约没声明具体输入（输入只能由调用方给）—— 那是老契约的形状，
    印一行「契约没声明」会把「没这一格」和「有但为空」混起来；`case_match` 只在契约声明了
    case 时才有这一格（没得对账 ≠ 对上了）。
    """
    source = _get(m, "case_source", None)
    if source not in ("contract", "caller"):
        return []
    if source == "contract":
        out = ["- 输入来源 (§3.4): **契约的 `cases`**（`materialize()` —— 评的就是契约里声明的那组）"]
    else:
        out = ["- 输入来源 (§3.4): 调用方给的数组（契约拥有输入那条路这次没走）"]
    if "case_match" in m:
        out.append(f"- 契约与本次对得上: {_tri(_get(m, 'case_match'))}"
                   f"（契约手里 {_get(m, 'case_declared')} 组具体输入，本次评的每一组都在其中）")
    return out

def _objective_lines(m: Dict[str, Any]) -> List[str]:
    """§6.2 的**目标函数**（权重 + 聚合口径 + 那个数）—— **声明了才印**

    缺键 ⇒ 不印（同 `_claim_level_lines` / `_implementation_decision_lines`）：印一个默认
    聚合会让「没声明」读成「等权几何平均」。这个数是**数据**不是门槛（准入今天仍看主
    case 的加速比），所以那条边界必须和数一起印 —— 读的人默认会把它当门槛。
    """
    obj = _get(m, "objective", None)
    if not isinstance(obj, dict):
        return []
    value = obj.get("speedup")
    out = [f"- 目标函数 (§6.2): `{obj.get('aggregate') or '未声明'}` = "
           f"{'未确立' if value is None else value} "
           f"({obj.get('evaluated_cases', 0)}/{obj.get('weighted_cases', 0)} 条 case 有权重)"]
    for reason in (obj.get("reason") or []):
        out.append(f"  - {reason}")
    if value is not None:
        out.append("  - ⚠️ 这个数今天**不进门槛**: 准入仍看主 case 的加速比 "
                   "(§16 反模式「用单形状的加速比作为成功标准」未收口, 见 CHECKLOG I93)")
    return out

def _best_known_good(artifact: Any, m: Dict[str, Any]) -> List[str]:
    out = ["## 1. best-known-good", ""]
    code = str(getattr(artifact, "code", "") or "")
    code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()[:12] if code else "(无代码)"
    out.append(f"- kernel_id: `{getattr(artifact, 'kernel_id', '')}`")
    out.append(f"- 代码: {len(code)} 字符, sha256[:12]=`{code_hash}`")
    sketch = getattr(artifact, "sketch", None)
    if sketch is not None:
        try:
            sk = sketch.to_dict()
        except Exception:
            sk = None
        if isinstance(sk, dict):
            tiling = sk.get("tiling") or {}
            out.append(f"- 配置: tiling={json.dumps(tiling, ensure_ascii=False)} "
                       f"precision={json.dumps(sk.get('precision'), ensure_ascii=False)}")
    if not m:
        out.append("- ⚠️ **没有性能读数**: 这次运行的「最好」无从谈起 (见第 3 节)")
        out.append("")
        return out
    out.append(f"- 最后一次 L3 读数: speedup={_get(m, 'speedup_vs_baseline')} "
               f"gflops={_get(m, 'gflops')} 档位={_get(m, 'measurement_validity')}")
    elig = _get(m, "eligibility", None)
    if isinstance(elig, dict):
        off = [k for k, v in sorted(elig.items()) if v is not True]
        out.append(f"- 资格位: {'全 True' if not off else '未开: ' + ', '.join(off)}")
        if elig.get("can_persist") is not True:
            out.append("- ⚠️ **这条没有进长期记忆** (can_persist=False): "
                       "它只活在本次运行与产物里, 下一次运行不会检索到它")
    remeasure = getattr(artifact, "remeasure", None)
    out.append(f"- 两次独立确认 (设计 §10.5-4): "
               f"{'有: ' + repr(remeasure) if remeasure is not None else '**没有**做重测'}")
    stop = _get(getattr(artifact, "metadata", None), "stop", None)
    if isinstance(stop, dict):
        out.append(f"- 循环停因: `{stop.get('reason')}` "
                   f"(第 {stop.get('round')} 轮) {stop.get('detail') or ''}".rstrip())
    out += _case_input_lines(m)
    out += _performance_verdict_lines(m)
    out += _objective_lines(m)
    out += _implementation_decision_lines(m)
    out.append("")
    return out


def _evidence_chain(artifact: Any) -> List[str]:
    out = ["## 2. 证据链 (逐轮)", ""]
    rounds = _rounds(artifact)
    if not rounds:
        out += ["(没有逐轮血缘 —— 这次运行不是 Conductor 循环跑出来的, 或血缘没写)", ""]
        return out
    out.append("| 轮 | 动作 | L1/L2/L3 | 加速比 | gflops | 档位 | 配方 |")
    out.append("|---|---|---|---|---|---|---|")
    for h in rounds:
        levels = h.get("levels_passed") or {}
        flags = "/".join(
            ("✅" if levels.get(lv) else ("❌" if lv in levels else "-"))
            for lv in ("L1", "L2", _L3))
        out.append("| {r} | {a} | {f} | {s} | {g} | {v} | {rec} |".format(
            r=h.get("round"), a=h.get("action"), f=flags,
            s=h.get("speedup"), g=h.get("gflops"), v=h.get("measurement_validity"),
            rec=",".join(h.get("recipes_applied") or []) or "-"))
    out += ["",
            "口径: 逐轮只留下 `iteration_history` 里的那几列 (档位/加速比/gflops); "
            "完整 metrics 只保留最后一轮, 所以这张表**不是**逐轮 metrics 的替代品。", ""]
    return out


def _collect_time(artifact: Any) -> List[str]:
    out = ["## 4. 复现与指纹", ""]
    fp = getattr(artifact, "environment_fingerprint", None)
    out.append(f"- 环境指纹: {json.dumps(fp, ensure_ascii=False) if fp else '(没有记录)'}")
    drift = getattr(artifact, "environment_drift", None)
    out.append(f"- 与上一次比较: {json.dumps(drift, ensure_ascii=False) if drift else '(没有比较)'}")
    rs = getattr(artifact, "resume_status", None)
    out.append(f"- 续跑: {json.dumps(rs, ensure_ascii=False) if rs else '(本次不是续跑)'}")
    sd = getattr(artifact, "saved_dir", None)
    if sd:
        out.append(f"- 落盘目录: `{sd}` (含本报告 / `result.json` / `execution_snapshot.json`)")
        out.append(f"- 续跑入口: `Conductor.execute(..., resume_from=\"{sd}\")`")
    out.append("- ⚠️ 具体 CLI 取决于调用方 (task schema 与后端); 本报告只记录产物侧的入口。")
    out.append("")
    return out


def build_report(*, artifact: Any, task: Any = None,
                 observations: Optional[Sequence[dict]] = None,
                 refusals: Optional[Sequence[str]] = None,
                 unable: Optional[UnableToEstablish] = None,
                 generated_at: Optional[str] = None) -> str:
    """渲染报告 (纯函数: 不读环境、不写盘、不联网)"""
    m = _last_metrics(artifact)
    kt = getattr(getattr(task, "source", None), "kernel_type", None) or "unknown"
    hw = getattr(getattr(task, "target", None), "hardware", None) or "unknown"
    dsl = getattr(getattr(task, "target", None), "dsl", None) or "unknown"
    ts = generated_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [f"# 自动跑报告 — {kt} / {hw} / {dsl}", "",
             f"- 生成时间: {ts}",
             f"- kernel_id: `{getattr(artifact, 'kernel_id', '')}`",
             f"- 轮数: {len(_rounds(artifact))}",
             ""]
    lines += _claim_level_lines(m)
    lines += _best_known_good(artifact, m)
    lines += _evidence_chain(artifact)
    lines += _correctness_lines(artifact)
    u = collect_unable(artifact=artifact, observations=observations,
                       refusals=refusals, unable=unable)
    lines += ["## 3. 我无法确立的事", "",
              "```", u.render(), "```", "",
              "口径: 这一节列的是**这一轮没能确立的事**, 不是否定结论 —— "
              "「没测出」≠「没有」。它为空时, 上面那份依据就是「为空」这个断言的证据。",
              ""]
    lines += _collect_time(artifact)
    return "\n".join(lines)


def _correctness_lines(artifact: Any) -> List[str]:
    """L2 evidence is not in benchmark_data, especially when L3 was blocked."""
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    out = []
    for r in getattr(artifact, "verification_results", ()) or ():
        c = (getattr(r, "metrics", {}) or {}).get("correctness")
        if not isinstance(c, dict):
            continue
        out += ["### CorrectnessSuite (§20)", "",
                f"- 结论: **{c.get('verdict', 'unknown')}**",
                f"- 套件哈希: `{c.get('suite_hash', '未确立')}`",
                f"- 范围: {c.get('scope', '声明无效或没有覆盖证据')}", "",
                "| Case | 值 / 布局 | 结论 | 证据 / 原因 |",
                "|---|---|---|---|"]
        for row in c.get("cases", ()):
            out.append("| " + " | ".join(cell(v) for v in (
                row.get("case_id"), f"{row.get('pattern', '-')} / {row.get('layout', '-')}",
                row.get("verdict"), row.get("reason"))) + " |")
        for row in c.get("cases", ()):
            m = row.get("metrics", {})
            receipt = m.get("input_fidelity", {})
            observed = receipt.get("observed", {})
            if not observed:
                continue
            out += ["", f"- Case `{cell(row.get('case_id'))}` 输入保真: "
                    f"{_tri(m.get('input_fidelity_verified'))}（执行侧 kernel 调用前回读）"]
            for name, spec in sorted(observed.items()):
                device = receipt.get("tensor_devices", {}).get(name)
                out.append(f"  - `{cell(name)}`: dtype={spec.get('dtype', spec.get('type'))}, "
                           f"shape={spec.get('shape')}, strides={spec.get('strides')}, "
                           f"offset={spec.get('offset')}, writeable={spec.get('writeable')}, "
                           f"value_hash=`{str(spec.get('value_digest', ''))[:12]}`"
                           + (f", device={cell(device)}" if device is not None else ""))
        if c.get("reason"):
            out.append(f"- 无法确立: {c['reason']}")
        out.append("")
    return out


def write_report(artifact_dir: Any, *, artifact: Any, task: Any = None,
                 observations: Optional[Sequence[dict]] = None,
                 refusals: Optional[Sequence[str]] = None,
                 unable: Optional[UnableToEstablish] = None) -> Path:
    """把报告写进产物目录, 返回路径。

    写失败**不抛**给调用方: 报告写不出来不该让一次跑完的实验失败 —— 但也不吞着
    (把异常原文写进那个文件), 否则读者拿到的是一份「看起来没有报告」的目录。
    """
    d = Path(artifact_dir)
    path = d / REPORT_NAME
    try:
        path.write_text(build_report(artifact=artifact, task=task,
                                     observations=observations, refusals=refusals,
                                     unable=unable), encoding="utf-8")
    except Exception as e:      # noqa: BLE001 — 见 docstring: 留痕, 不抛
        path.write_text(f"# 报告生成失败\n\n{e!r}\n", encoding="utf-8")
    return path
