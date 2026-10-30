"""设计 §14-9 的对照实验: **资格审计到底有没有收益** (离线比较器)

设计 §19.4 自己承认: 「§4 那层资格审计在上述任何工作中都没有对照」—— 既可能是真空白
(差异化机会), 也可能说明它在主流的同卡/短程/单次评测协议下并不必要。而 §14 的八条
验收**只测它工作不工作**, 没有一条测它对最终结果**有没有用**。于是设计里那句
「在 §14 通过之前不应对外声称它带来了收益」永远无法翻页: 谁也没设计过那个实验。

这个脚本就是那个实验的**比较器**: 它吃两份运行血缘, 吐一个判决。

    python3 benchmarks/experiments/misc/ablation_eligibility.py --arm-a <dir|result.json> --arm-b <...>
    python3 benchmarks/experiments/misc/ablation_eligibility.py --selftest      # 只看判据 (合成夹具)

两个 arm 的差别**只有一个**:

  A 组  照常 —— 资格位决定后果 (今天的产品行为)
  B 组  关掉资格审计的**后果** —— 判据照算照记, 只是不再拦任何事

⚠️ 那个「关掉」**不许做成产品开关**。设计 §10.5-5 说 agent 不能编辑自己的尺子, 而一个
「一键关掉安全门」的环境变量就是那把尺子 —— 它会先被用在无人值守的夜里。所以 B 组只能
在 harness 侧 monkeypatch 出来 (`oracle.admit_to_corpus` 忽略 `can_persist`, 控制器把
每个读数都当有效), 命令行里**没有**这个开关。脚本只负责比较, 不负责造 arm。

三个数, 全部从 `iteration_history` 离线可算 (所以这一节能复盘, 不必重跑):

  1. `best`        最终最好的加速比 (高者好)
  2. `measurements` 真值测量轮数 (低者省) —— 设计 §10.2: 「真值 oracle 次数是真正的货币」
  3. `dirty`        **读数没有资格造成永久后果, 却造成了**的轮数
                    (best 被覆盖 / 配方被永久禁用)

第 3 个数是这里的核心: 它是「资格审计在防什么」的可测形式。设计 §16 的反模式里
「让一个噪声大的数永久拉黑配置」说的就是它。

判据 (成对, 五态 + 一条无信息量的诚实出口):

  `无法比较`   任一 arm 的血缘不可用 (没有 L3 轮次 / 没有 best 读数)
  `没打到点上` B 组**一次都不曾**出现「本该被拦」的读数 ⇒ 这次对照无信息量。
               ⚠️ 这一态是刻意分开的: 从「门没被需要过」推出「门没用」是这一节最容易
               犯的错 (空清单的两种来源 —— 都站得住 / 压根没收集 —— 长得一样)。
  `没有收益`   审计有过用武之地 (dirty>0), 而 B 的 best 并不劣 ⇒ 白花机时 + 留下脏后果。
               **这一向必须写进报告** —— 设计缺的正是它。
  `证据不足`   出现过 dirty, 但两臂的 best 差落在噪声带内、而脏后果也不多 ⇒ 不下结论。
  `有收益`     B 的 best 明显更差 (超出噪声带)。

口径与边界 (不许被读成更强的结论):

  * 这里的 `best` 是**同一台机、同一批轮次**下的可比量; 跨机跑两个 arm 的数字**作废**
    (设计 §11-2 同卡配对: 线下对照同样要同卡)。
  * 噪声带借用 `measurement.NOISE_BAND[valid]`。它是**轮间**比较的带宽, 拿来比两臂的
    最终 best 是近似 —— 所以输出里永远带 n 轮的样本量, 并在 n 小时明确警告。
  * 「事后被推翻的长期记忆条数」这个更诚实的量需要**后续一次运行**来推翻, 单次对照
    算不出来 ⇒ 本脚本不假装能算 (边界写在这里, 而不是留一个恒 0 的假指标)。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from operator_agent.measurement import NOISE_BAND, Validity  # noqa: E402

#: **跨运行**带宽: 两臂最终 best 的差异在这个相对值内 = 「一样好」。
#:
#: 为什么不直接用 `NOISE_BAND[valid]` (0.1%): 那是**轮内**离散的带宽。真机实测 (doc 63)
#: 同一个核三次读数 0.643776/0.642928/0.642992 ms ⇒ 离散 0.13%, 轮内确实这么紧。但两个
#: arm 之间隔着两次完整运行: 时钟状态、邻居占用、温度 (doc 37 冷热上下文 / doc 40 DVFS)
#: 都会动, 而那几份文档量到的效应是**个百分点**级。
#:
#: 所以这里取 `NOISE_BAND[uninformative]` = **2%**: 那是「这个数自己就不太可信时, 判决
#: 该留多宽的带宽」—— 跨运行比较的可信度低于轮内比较, 借这一档比借 `valid` 那档更贴切。
#: 取偏大的方向也是安全的: 带宽越宽, 「有收益」越难声称 (需要更大的损失), 而设计 §19.4
#: 要的正是「不许在无对照的情况下声称收益」。轮内那 0.1% 留作参照, 供将来标定对比。
BAND = float(NOISE_BAND[Validity.UNINFORMATIVE])
ROUND_NOISE_BAND = float(NOISE_BAND[Validity.VALID])   #: 0.001, 参照用 (见上)

#: 样本量警告线。n 太小的时候上面那个带宽本身也估不准 —— 不是不能比, 是要说出来。
SMALL_SAMPLE = 5

VERDICTS = ("无法比较", "没打到点上", "没有收益", "证据不足", "有收益")


# ─── 读血缘 ────────────────────────────────────────────────────────


def load_history(path: Any) -> Tuple[Optional[List[dict]], str]:
    """从产物目录或 JSON 文件里取 `iteration_history`。返回 (血缘, 说明)"""
    p = Path(path)
    if p.is_dir():
        for name in ("result.json", "execution_snapshot.json"):
            cand = p / name
            if cand.exists():
                p = cand
                break
        else:
            return None, f"{path}: 目录里没有 result.json / execution_snapshot.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:      # noqa: BLE001
        return None, f"{p}: 读不出 JSON ({e!r})"
    hist = data.get("iteration_history") if isinstance(data, dict) else None
    if not isinstance(hist, list) or not hist:
        return None, f"{p}: 没有 iteration_history (旧产物? 或这次跑没进循环)"
    return [h for h in hist if isinstance(h, dict)], f"{p}"


# ─── 三个数 ────────────────────────────────────────────────────────


def _best(hist: List[dict]) -> Tuple[Optional[float], str]:
    """最终 best 的加速比。血缘里逐轮记的是 `speedup` (没有则退到 gflops)"""
    sp = [r.get("speedup") for r in hist]
    sp = [float(x) for x in sp if isinstance(x, (int, float)) and x > 0]
    if sp:
        return max(sp), "speedup_vs_baseline"
    gf = [r.get("gflops") for r in hist]
    gf = [float(x) for x in gf if isinstance(x, (int, float)) and x > 0]
    if gf:
        return max(gf), "gflops (血缘里没有加速比)"
    return None, "两个都没有"


def summarize(hist: List[dict]) -> Dict[str, Any]:
    """一份血缘 → 三个数 + 归因明细 (脏账逐轮列出来, 不只给个计数)"""
    best, which = _best(hist)
    measurements = sum(1 for r in hist
                       if (r.get("levels_passed") or {}).get("L3") is True)
    dirty: List[dict] = []
    unknown_elig = 0
    for r in hist:
        eff = r.get("permanent_effects")
        if not isinstance(eff, dict):
            # 老产物没有这一格: 数不出来就说数不出来, 不许当 0 (缺键 ≠ 没有后果)
            unknown_elig += 1
            continue
        effects = []
        if eff.get("best_updated"):
            effects.append("覆盖 best")
        if eff.get("recipes_disabled"):
            effects.append("永久禁用配方 " + ",".join(map(str, eff["recipes_disabled"])))
        if not effects:
            continue
        elig = r.get("eligibility")
        if not isinstance(elig, dict):
            unknown_elig += 1
            continue
        # 「有资格造成永久后果」在这套判据里 = can_tune (改配置) 且 can_persist (写记忆)。
        # 用 `is True` 而不是真值判断: 缺位就是没资格 (缺证据不是资格)。
        entitled = elig.get("can_tune") is True and elig.get("can_persist") is True
        if not entitled:
            dirty.append({"round": r.get("round"), "effects": effects,
                          "validity": r.get("measurement_validity"),
                          "off": sorted(k for k, v in elig.items() if v is not True)})
    return {"best": best, "best_from": which, "measurements": measurements,
            "dirty": dirty, "n_rounds": len(hist), "unknown_effects": unknown_elig}


# ─── 判决 ──────────────────────────────────────────────────────────


def judge(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[str, List[str]]:
    """两臂的三个数 → (判决, 理由行)"""
    why: List[str] = []
    if a.get("best") is None or b.get("best") is None:
        return "无法比较", [f"A best={a.get('best')} ({a.get('best_from')})",
                            f"B best={b.get('best')} ({b.get('best_from')})"]

    rel = (float(b["best"]) - float(a["best"])) / float(a["best"])
    dirty_b = len(b.get("dirty") or [])
    why.append(f"A best={a['best']:.4f} · B best={b['best']:.4f} "
               f"(相对差 {rel:+.2%}, 噪声带 ±{BAND:.1%})")
    why.append(f"真值测量轮数: A={a['measurements']} · B={b['measurements']}")
    why.append(f"B 的脏永久后果: {dirty_b} 轮"
               + ("" if not dirty_b else " —— " + "; ".join(
                   f"第 {d.get('round')} 轮 {','.join(d['effects'])} "
                   f"(档位={d.get('validity')}, 未开={d.get('off')})"
                   for d in b["dirty"])))
    for arm, name in ((a, "A"), (b, "B")):
        if arm.get("unknown_effects"):
            why.append(f"⚠️ {name} 组有 {arm['unknown_effects']} 轮缺 `permanent_effects`"
                       f"/`eligibility` (老产物), 这几轮**没被算进**脏账")
        if arm["n_rounds"] < SMALL_SAMPLE:
            why.append(f"⚠️ {name} 组只有 {arm['n_rounds']} 轮 (< {SMALL_SAMPLE}), "
                       f"噪声带本身也估不准")

    if dirty_b == 0:
        return "没打到点上", why + [
            "B 组一次都没出现「本该被拦」的读数 ⇒ 这次对照没打到点上。",
            "从「门没被需要过」推不出「门没用」—— 要造出困难场景 (邻居抢占 / 噪声 / "
            "降频) 或加轮次, 再来看。"]

    if rel >= -BAND:
        return "没有收益", why + [
            "审计有用武之地 (B 出现脏永久后果), 而 B 的 best 并不劣 ⇒ "
            "**这一段协议下资格审计没有带来收益**, 它只留下了脏后果"
            + ("" if b["measurements"] <= a["measurements"]
               else f", 而且 B 还多花了 {b['measurements'] - a['measurements']} 轮真值测量。"),
            "⚠️ 这一向必须写进报告 (设计 §19.4: 不许在无对照的情况下声称收益)。"]

    if rel < -3 * BAND:
        return "有收益", why + [
            f"B 的 best 低 {abs(rel):.1%} (远超跨运行带宽 ±{BAND:.0%}) ⇒ "
            f"关掉资格审计会损失质量, 审计防住的是真事。"]

    return "证据不足", why + [
        f"B 的 best 低 {abs(rel):.1%} —— 超出「一样好」的带宽但不足 3 倍, "
        f"脏后果 {dirty_b} 轮 ⇒ 不下结论, 加轮次或加场景重跑。"]


# ─── 自检夹具 (合成, 不是实测) ──────────────────────────────────────


def _fix_round(n: int, speedup: float, *, l3: bool = True, validity: str = "valid",
               can_tune: bool = True, can_persist: bool = True,
               best_updated: bool = False, disabled: Tuple[str, ...] = ()) -> dict:
    return {"round": n, "action": "optimize", "levels_passed": {"L1": True, "L2": True,
                                                                "L3": l3},
            "measurement_validity": validity if l3 else None,
            "speedup": speedup if l3 else None,
            "eligibility": ({"can_verdict": True, "can_tune": can_tune,
                             "can_persist": can_persist, "can_report": True,
                             "can_count_progress": can_persist} if l3 else None),
            "permanent_effects": {"best_updated": best_updated,
                                  "recipes_disabled": list(disabled),
                                  "stagnation_counted": False}}


def _fixtures() -> List[Tuple[str, List[dict], List[dict], str]]:
    """(名字, A 组血缘, B 组血缘, 期望判决)。**合成夹具** —— 只用来钉判据, 不是实测。"""
    a = [_fix_round(1, 1.10, best_updated=True), _fix_round(2, 1.30, best_updated=True)]
    clean_bad = [_fix_round(1, 1.10, best_updated=True),
                 _fix_round(2, 0.90, validity="degraded", can_tune=False, can_persist=False,
                            best_updated=True, disabled=("R-MATMUL-07",))]
    clean_same = [_fix_round(1, 1.10, best_updated=True),
                  _fix_round(2, 1.28, validity="uninformative", can_tune=False,
                             can_persist=False, best_updated=True)]
    never_needed = [_fix_round(1, 1.10, best_updated=True),
                    _fix_round(2, 1.32, best_updated=True)]
    no_l3 = [_fix_round(1, 0.0, l3=False)]
    borderline = [_fix_round(1, 1.10, best_updated=True),
                  _fix_round(2, 1.25, validity="degraded", can_tune=False,
                             can_persist=False, best_updated=True)]
    return [("有收益 (B 掉质量)", a, clean_bad, "有收益"),
            ("证据不足 (差在带外沿)", a, borderline, "证据不足"),
            ("没有收益 (B 一样好)", a, clean_same, "没有收益"),
            ("没打到点上 (门没被需要)", a, never_needed, "没打到点上"),
            ("无法比较 (B 没有 L3)", a, no_l3, "无法比较")]


def selftest() -> int:
    print("合成夹具 (不是实测): 只钉判据本身\n" + "=" * 78)
    bad = 0
    for name, ha, hb, expect in _fixtures():
        verdict, why = judge(summarize(ha), summarize(hb))
        ok = verdict == expect
        bad += 0 if ok else 1
        print(f"\n[{ '✓' if ok else '✗' }] {name}: 判决={verdict} (期望 {expect})")
        for line in why:
            print("    " + line)
    print("\n" + "=" * 78)
    print("判据自检: " + ("全过" if not bad else f"{bad} 条不符"))
    return 1 if bad else 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="设计 §14-9 资格审计对照实验的比较器")
    ap.add_argument("--arm-a", help="A 组 (照常) 的产物目录或 result.json")
    ap.add_argument("--arm-b", help="B 组 (关掉后果) 的产物目录或 result.json")
    ap.add_argument("--selftest", action="store_true", help="只跑合成夹具的判据自检")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出 (给脚本消费)")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if not (args.arm_a and args.arm_b):
        ap.error("需要 --arm-a 与 --arm-b (或 --selftest)")

    ha, sa = load_history(args.arm_a)
    hb, sb = load_history(args.arm_b)
    if ha is None or hb is None:
        print("无法比较: " + "; ".join(x for x in (sa if ha is None else None,
                                                  sb if hb is None else None) if x))
        return 2
    A, B = summarize(ha), summarize(hb)
    verdict, why = judge(A, B)
    if args.json:
        print(json.dumps({"verdict": verdict, "why": why, "a": A, "b": B},
                         ensure_ascii=False, indent=2))
    else:
        print(f"A: {sa}\nB: {sb}\n")
        print(f"判决: {verdict}")
        for line in why:
            print("  " + line)
        print("\n⚠️ 两个 arm 必须同机跑 (设计 §11-2): 跨机的数字直接作废。")
    return 0 if verdict in VERDICTS else 1


if __name__ == "__main__":
    sys.exit(main())