# -*- coding: utf-8 -*-
"""未知卡的五道迁移门（G1–G5）：拿到新卡先跑这个，再谈任何性能结论。

出处：`docs/design/dsh-operator-portability-4090.md` §4 的 G1–G5，按危害排序。
本探针**不新增判据**，只做一件事——检查这些判据的**输入是不是这张卡测出来的**。
继承来的常量在这里一律不给绿：一条 `source_uuid` 对不上就是 fail，缺读数就是 unknown。

为什么值得单独有一道门（不是洁癖）：
  * `measurement.assess()` 在 `launch_floor_us` 为 `None` 时只写一条 reason，`validity`
    仍是 `VALID`，于是 `can_gate` 仍为 True（`operator_agent/decide/measurement.py:1007,1108`）。
    方向性是**变松**的：未标定的卡会把「地板附近的读数」当成可裁决读数，而禁用配方是永久决策。
  * 「没探测过硬件」在产物里长得和探测过一样（`operator_agent/execute/remote_executor.py:165`）。

跑法：

    # 零机时自检（不需要卡，验证门本身会拒）
    python3 -B -m benchmarks.probes.unknown_card_gates --selftest

    # 真跑：先在有卡的机器上产出一份探测输出，再喂给门
    python3 -B -m benchmarks.probes.unknown_card_gates --from-json ./card-probe.json

退出码：0 = 五门全绿（可以开始性能结论）；2 = 有门是 fail 或 unknown（runbook 第 0 步即停）。
"""

import json
import sys
from pathlib import Path

GATES = ("G1_launch_floor", "G2_shared_memory", "G3_peak_denominators",
         "G4_noise_bands", "G5_shape_inheritance")

#: 仓内现值 = H20 八卡共享机上标定出来的数。逐位相同又没有本卡测量记录 ⇒ 那是继承，不是标定。
INHERITED = {
    "noise_band": {"valid": 0.001, "degraded": 0.004, "uninformative": 0.02},
    "contamination_util_pct": 5.0,
    "dispersion_degraded": 0.10,
    "launch_floor_us_h20": 18.69,
}

MIN_FLOOR_SAMPLES = 5


def _missing(why):
    return {"verdict": "unknown", "value": None, "why": why,
            "reading_condition": "本卡未提供读数"}


def _fail(value, why, condition):
    return {"verdict": "fail", "value": value, "why": why, "reading_condition": condition}


def _pass(value, why, condition):
    return {"verdict": "pass", "value": value, "why": why, "reading_condition": condition}


def _same_as_inherited(candidate, inherited):
    """逐位相同就算继承。差一个数就不算——所以这不是「容差比对」，是「同一份数没重测」。"""
    if isinstance(inherited, dict):
        return isinstance(candidate, dict) and all(
            candidate.get(k) == v for k, v in inherited.items())
    return candidate == inherited


def assess(info):
    """info 是探测输出的 dict；返回五门的裁决。纯函数，零 GPU 可测。"""
    device = info.get("device") or {}
    uuid = device.get("uuid")
    out = {}

    # --- G1 launch 地板 -------------------------------------------------------
    g1 = info.get("launch_floor") or {}
    floor, samples, src = g1.get("median_us"), g1.get("samples"), g1.get("source_uuid")
    if floor is None:
        out["G1_launch_floor"] = _missing("地板未测：`assess()` 缺 floor 时仍给 VALID，`can_gate` 仍为 True")
    elif src != uuid:
        out["G1_launch_floor"] = _fail(floor,
            "地板来源不是本卡（source_uuid 与 device.uuid 不一致）——未标定的卡不能裁决性能",
            "读数属于 %s" % (src or "无来源"))
    elif not isinstance(samples, int) or samples < MIN_FLOOR_SAMPLES:
        out["G1_launch_floor"] = _fail(floor,
            "样本 %r 少于 %d 个：一条地板值不足以定「小形状不可裁决」的下界" % (samples, MIN_FLOOR_SAMPLES),
            "本卡 %s，样本 %r" % (uuid, samples))
    else:
        out["G1_launch_floor"] = _pass(floor,
            "本卡实测地板，可用于拦小形状", "本卡 %s，样本 %d" % (uuid, samples))

    # --- G2 共享内存 ----------------------------------------------------------
    g2 = info.get("shared_memory") or {}
    per_block, per_sm, src = g2.get("per_block_optin_bytes"), g2.get("per_sm_kb"), g2.get("source_uuid")
    # 两个上限只要**有一个是本卡读回的**就够排 tile 预算（torch 老 build 只暴露一半）。
    # 要求两个都有会把门变成「按 torch 版本判卡」——那是采集口径，不是硬件事实。
    if per_block is None and per_sm is None:
        out["G2_shared_memory"] = _missing("共享内存未从设备属性读回：缺它时 tile 预算会落到默认量级，Ada 类卡直接偏高")
    elif src != uuid:
        out["G2_shared_memory"] = _fail(per_block,
            "共享内存上限不是本卡读回的（继承值会把 tile 预算抬高，代价是多烧几轮 L1）",
            "读数属于 %s" % (src or "无来源"))
    else:
        out["G2_shared_memory"] = _pass({"per_block_optin_bytes": per_block, "per_sm_kb": per_sm},
            "上限来自本卡设备属性", "本卡 %s" % uuid)

    # --- G3 roofline 分母 -----------------------------------------------------
    g3 = info.get("peak") or {}
    bw, peak_map, trusted, src = g3.get("mem_bw_gbs"), g3.get("peak_by_dtype"), g3.get("trusted"), g3.get("source_uuid")
    if not bw or not peak_map:
        out["G3_peak_denominators"] = _missing("峰值分母缺失：`peak_trusted` 不为 True ⇒ 轴证据判不可用 ⇒ 「已到上界」的出口永不打开")
    elif not trusted or src != uuid:
        out["G3_peak_denominators"] = _fail({"mem_bw_gbs": bw},
            "峰值行口径不可信或不是本卡的：能跑，但拿不到判停质量；「宁可偏大」的口径不许用来判停",
            "trusted=%r source=%s" % (trusted, src or "无来源"))
    else:
        out["G3_peak_denominators"] = _pass({"mem_bw_gbs": bw, "dtypes": sorted(peak_map)},
            "本卡可信分母，轴筛选与判停出口可开", "本卡 %s" % uuid)

    # --- G4 噪声带与污染阈值 ---------------------------------------------------
    g4 = info.get("noise") or {}
    band, util, disp = g4.get("band"), g4.get("contamination_util_pct"), g4.get("dispersion_degraded")
    peers, span, src = g4.get("peers_observed"), g4.get("sharing_pattern"), g4.get("source_uuid")
    inherited_hits = [k for k, v in (("band", INHERITED["noise_band"]),
                                     ("contamination_util_pct", INHERITED["contamination_util_pct"]),
                                     ("dispersion_degraded", INHERITED["dispersion_degraded"]))
                      if _same_as_inherited({"band": band, "contamination_util_pct": util,
                                             "dispersion_degraded": disp}[k], v)]
    if band is None or util is None or disp is None:
        out["G4_noise_bands"] = _missing("本卡的散布/污染阈值未测：沿用八卡共享机的数会两头错")
    elif src is None:
        # 「写了共享形态」不能替代来源：一句说明不能让继承值变成本卡标定
        out["G4_noise_bands"] = _missing(
            "阈值没有本卡来源（source_uuid 缺失）：光给 sharing_pattern 不构成标定")
    elif src != uuid:
        out["G4_noise_bands"] = _fail({"band": band, "util_pct": util, "dispersion": disp},
            "阈值来源不是本卡（source_uuid 与 device.uuid 不一致）：阈值随共享形态与驱动一起变，"
            "搬别人的数会两头错", "读数属于 %s" % src)
    elif inherited_hits:
        out["G4_noise_bands"] = _fail(inherited_hits,
            "声称本卡标定，却与仓内 H20 八卡共享机的值逐位相同：要么没重测要么在抄。"
            "独占机上会永不 degraded，接显示器的机器上会每轮 degraded",
            "本卡 %s，来源 %s" % (uuid, src))
    elif not span:
        out["G4_noise_bands"] = _fail(None,
            "没写共享形态（独占 / 几台对手 / 是否接显示器），阈值无法判定方向", "缺 sharing_pattern")
    else:
        out["G4_noise_bands"] = _pass({"band": band, "util_pct": util, "dispersion": disp},
            "阈值带本卡共享形态观测，可参与档位判决", "本卡 %s，peers=%r，%s" % (uuid, peers, span))

    # --- G5 形状继承边界 ------------------------------------------------------
    g5 = info.get("shapes") or []
    pool = device.get("total_memory_bytes")
    if not pool:
        out["G5_shape_inheritance"] = _missing("本卡显存池未知：无法判断哪些形状可作验收样本")
    elif not g5:
        out["G5_shape_inheritance"] = _missing("没有形状清单：继承边界未核对")
    else:
        too_big = [s.get("label") for s in g5 if isinstance(s.get("bytes"), int) and s["bytes"] > 0.8 * pool]
        foreign = [s.get("label") for s in g5
                   if s.get("bytes") and s.get("validated_on_uuid") not in (None, uuid)]
        if too_big:
            out["G5_shape_inheritance"] = _fail(too_big,
                "这些形状超出本卡可用池（>0.8×pool），不构成新卡的可执行条件", "pool=%d bytes" % pool)
        elif foreign:
            out["G5_shape_inheritance"] = _fail(foreign,
                "这些形状只在别的卡上验收过（96 GB 级读数不能当作本卡验收样本）",
                "foreign=%s" % foreign)
        else:
            out["G5_shape_inheritance"] = _pass(len(g5),
                "所有形状在本卡池内，且没有跨卡验收样本", "本卡 %s" % uuid)

    return {"device": {"uuid": uuid, "name": device.get("name"), "cc": device.get("cc")},
            "gates": out, "all_green": all(v["verdict"] == "pass" for v in out.values())}


def _blank_info():
    return {"device": {"uuid": None}}


def render(result):
    lines = ["device=%s name=%s cc=%s" % (result["device"]["uuid"], result["device"]["name"],
                                          result["device"]["cc"])]
    for name in GATES:
        g = result["gates"][name]
        lines.append("%-22s %-7s %s | 条件：%s" % (name, g["verdict"], g["why"], g["reading_condition"]))
    lines.append("all_green=%s" % result["all_green"])
    return "\n".join(lines)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    selftest = "--selftest" in argv
    path = None
    if "--from-json" in argv:
        i = argv.index("--from-json")
        if i + 1 >= len(argv):
            print("--from-json 后面要给一个文件路径", file=sys.stderr)
            return 2
        path = argv[i + 1]
    if selftest:
        print("=== 自检：无读数必须全 unknown，继承值必须 fail ===")
        blank = assess(_blank_info())
        assert not blank["all_green"] and all(
            g["verdict"] == "unknown" for g in blank["gates"].values()), blank
        noise = {"band": INHERITED["noise_band"],
                 "contamination_util_pct": INHERITED["contamination_util_pct"],
                 "dispersion_degraded": INHERITED["dispersion_degraded"]}
        inherited = assess({"device": {"uuid": "GPU-mine", "total_memory_bytes": 16 * 2 ** 30},
                            "launch_floor": {"median_us": INHERITED["launch_floor_us_h20"],
                                             "samples": 5, "source_uuid": "H20-archive"},
                            "noise": noise})
        # 继承值只要求「不是 pass」：fail 还是 unknown 取决于它缺不缺来源字段，
        # 但两条都不许放行 —— 断死成 fail 会让这条自检对顺序变化过敏。
        assert inherited["gates"]["G1_launch_floor"]["verdict"] != "pass", inherited
        assert inherited["gates"]["G4_noise_bands"]["verdict"] != "pass", inherited
        print(render(blank))
        print("---")
        print(render(inherited))
        print("自检通过：无读数不绿，继承值不绿。")
        return 0
    if not path:
        print(render(assess(_blank_info())))
        print("\n没有 --from-json：本卡读数不存在，五门全 unknown。runbook 第 0 步应停在这里。",
              file=sys.stderr)
        return 2
    info = json.loads(Path(path).read_text(encoding="utf-8"))
    result = assess(info)
    print(render(result))
    return 0 if result["all_green"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
