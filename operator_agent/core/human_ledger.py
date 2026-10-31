"""人在环节上的出现记录（阶段 A 那句「谁在哪个环节花时间」的可测形式）

设计出处:
  * `docs/design/operator-development-product-roadmap.md` §5.1 把一条完整工作流写成六个环节，
    §6.3 要求「人工实际投入」记**活跃工作时间与干预原因**，且**不能用模型墙钟代替**
  * 台账 `docs/research/80-阶段A真实任务台账与基线.md` §2/§3.1：本仓从未记录过人时，
    而外部公开材料里也**没有任何一处给出按环节的人工时间百分比** —— 所以历史不可追，
    只能从现在起当场产生
  * 形态借 `arXiv 2607.17979` 附录 Table 5「Human and Agent Contribution Matrix」：
    **不追分钟，先追「谁做了哪类动作 + 为什么」**；分钟是可选的量级档，不是必填数值

与既有机制的关系:
  * `run_contract` / `corpus_admission` 记的是**机器侧事实**（身份、判定、耗时）
  * **本模块**只记人的出现与原因，两栏永不合算成一个数
  * 本模块**不**接入任何 MCP 工具的回包 —— 那会改工具契约，属另一件事，需单独设计与验收

三条不变量（都有对应用例，见 `tests/unit/test_human_ledger.py`）:
  1. **没问 ≠ 没人做**：`involved` 缺省是 `unknown`，不是 `no`，也不是 `False`
  2. **未答进分母**：完整度按「已答 / 全部环节」算，分母始终是全部环节，不是已答数
  3. **单位不折叠**：人自报的是量级档（字符串），机器是秒（数值），两者分开成两列，
     本模块不提供任何把它们相加的入口
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from .run_contract import atomic_json

SCHEMA = "operator-human-ledger-v1"

#: 六个环节，逐字取自路线 §5.1「过程：准备任务 → 检查与运行 → 分析反馈 →
#: 修改或修复 → 比较候选 → 最终验收」。**不要**在这里加第七个环节来塞"接入"：
#: 代码里没有接入阶段（状态机止于 `REPORT`），它应当记为 unknown 而不是被发明。
STAGES: Dict[str, str] = {
    "prepare": "准备任务",
    "check_run": "检查与运行",
    "analyze": "分析反馈",
    "modify": "修改或修复",
    "compare": "比较候选",
    "accept": "最终验收",
}

UNKNOWN = "unknown"
INVOLVED_VALUES = ("yes", "no", UNKNOWN)

#: 量级档，不是数值：要防止任何人把档取中值再与墙钟做算术。
MINUTE_BANDS = ("lt-5", "5-15", "15-60", "gt-60")


class LedgerError(ValueError):
    """拒绝消息必须自带可修正性：说清违反了哪条、以及合法取值是什么。"""


def _refuse(what: str, got, allowed) -> "LedgerError":
    return LedgerError(
        "%s refused: %r is not accepted; pass one of %s" % (what, got, ", ".join(allowed)))


def new_ledger(run_id: str) -> dict:
    """六个环节全部在场、全部 unknown —— 一张空表也是一张完整的表。"""
    return {
        "schema": SCHEMA,
        "run_id": run_id,
        "entries": {stage: {"involved": UNKNOWN, "action": None,
                            "reason": None, "minutes_band": None} for stage in STAGES},
    }


def record(ledger: dict, stage: str, *, involved: Optional[str] = None,
           action: Optional[str] = None, reason: Optional[str] = None,
           minutes_band: Optional[str] = None) -> dict:
    """填一格。**不传 involved 表示「没问过」**，它保持 unknown；要记「人没介入」必须显式写 `no`。"""
    entries = ledger.get("entries")
    if not isinstance(entries, dict) or ledger.get("schema") != SCHEMA:
        raise _refuse("record", ledger.get("schema"), [SCHEMA])
    if stage not in STAGES:
        raise _refuse("stage", stage, list(STAGES))
    if involved is not None and involved not in INVOLVED_VALUES:
        raise _refuse("involved", involved, list(INVOLVED_VALUES))
    if minutes_band is not None and minutes_band not in MINUTE_BANDS:
        raise _refuse("minutes_band", minutes_band, list(MINUTE_BANDS))
    current = entries[stage]["involved"]
    entry_asked = involved if involved is not None else current
    if minutes_band is not None and entry_asked != "yes":
        # 只有人确实介入过才谈得上量级；否则这一档会把"没问"伪装成"问过且很快"。
        raise LedgerError(
            "minutes_band refused on stage %r: minutes_band requires involved='yes', "
            "but this cell currently reads %r (unanswered counts as never asked, not as 'no')"
            % (stage, entry_asked))
    entry = entries[stage]
    if involved is not None:
        entry["involved"] = involved
    if action is not None:
        entry["action"] = action
    if reason is not None:
        entry["reason"] = reason
    if minutes_band is not None:
        entry["minutes_band"] = minutes_band
    return ledger


def counts(ledger: dict) -> dict:
    """分母恒为全部环节：`answered` 只数已答，`unknown` 单列，完整度由两者相除。"""
    entries = ledger["entries"]
    answered = sum(1 for e in entries.values() if e["involved"] != UNKNOWN)
    return {"total": len(STAGES), "answered": answered,
            "unknown": len(STAGES) - answered,
            "yes": sum(1 for e in entries.values() if e["involved"] == "yes"),
            "no": sum(1 for e in entries.values() if e["involved"] == "no")}


def as_report_rows(ledger: dict, machine_wall_s: Optional[dict] = None) -> List[dict]:
    """每个环节两列分开印：人自报的**量级档**（字符串）与机器**秒**（数值）。

    `machine_wall_s` 是按环节的秒数（缺则 None）。本函数刻意不返回任何合计字段 ——
    把「15-60 档」和「241.2 秒」折成一个数就是 §6.3 禁的那种事。
    """
    rows = []
    given = machine_wall_s or {}
    if not isinstance(given, dict):
        raise _refuse("machine_wall_s", type(given).__name__, ["dict keyed by stage"])
    for stage, label in STAGES.items():
        entry = ledger["entries"][stage]
        wall = given.get(stage)
        if isinstance(wall, bool) or (wall is not None and not isinstance(wall, (int, float))):
            raise _refuse("machine_wall_s value", wall, ["number or None"])
        rows.append({"stage": stage, "label": label,
                     "involved": entry["involved"], "action": entry["action"],
                     "reason": entry["reason"],
                     "human_minutes_band": entry["minutes_band"],
                     "machine_wall_s": wall})
    return rows


def write_ledger(path, ledger: dict) -> str:
    if ledger.get("schema") != SCHEMA:
        raise _refuse("write_ledger", ledger.get("schema"), [SCHEMA])
    atomic_json(path, ledger)
    return str(path)


def read_ledger(path) -> dict:
    """读回时校验 schema 与环节集合：少一格就是有人在偷偷删 unknown，而不是"数据简洁"。"""
    with open(path, encoding="utf-8") as handle:
        ledger = json.load(handle)
    if ledger.get("schema") != SCHEMA:
        raise _refuse("read_ledger", ledger.get("schema"), [SCHEMA])
    if set(ledger.get("entries", {})) != set(STAGES):
        raise LedgerError(
            "read_ledger refused: entries must cover exactly %s, got %s"
            % (sorted(STAGES), sorted(ledger.get("entries", {}))))
    return ledger
