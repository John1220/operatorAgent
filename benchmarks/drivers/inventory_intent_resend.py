"""存量产物里「同一个 intent 被发了几次」的清点 (#102 第 3 片, **只量不改**)

doc 60 §四 给这一片定的形状是:

    `kernel_id` 的两种约定（uuid vs task_id）与「每次运行一个新 key」这件事清点：
    先量出「同一 intent 被重发」在存量运行里发生的频率，再谈要不要 pin key
    （按本仓纪律，观测先于判定）。

跑法:
    python3 -B benchmarks/drivers/inventory_intent_resend.py               # 读本机 output/
    python3 -B benchmarks/drivers/inventory_intent_resend.py --selftest     # 只跑合成树, 不需要本机产物
    python3 -B benchmarks/drivers/inventory_intent_resend.py --root <目录>

三态纪律 (这是本脚本唯一的新东西, 也是它存在的理由):
  · **完整产物** —— 目录里有 `result.json`。intent 只能从**文件里**的 `task_id` 读出来:
    目录名用的是 `kernel_id`, 而 `kernel_id` 每次运行都是一个新的 uuid
    (`schemas.py` 的 `default_factory`) —— 所以「同一个 intent」在**目录层是不可见的**。
  · **staging 快照** —— 目录里只有 `execution_snapshot.json`。那是 `conductor._snap()` 在
    「这一轮还没走到 save」时预创建的落点, 名字是 `snap_<秒级时间戳>_<task_id>`。
    ⚠️ 这个名字**每开一次就换一个时间戳**, 所以它既不能证「一轮一个目录」也不能证
    「一次运行一个目录」—— 于是这一片去快照里读 `round_done`, 把「同一个 task_id 多出来的
    那些目录」拆成「换了轮次」与「同一轮次再来一次」两格 (见 `render()` 那两句)。
  · **其他** —— 两种都不是（空目录、只有 `code.py` 的半截写入…). 这一格**不进任何分母**,
    但必须被印出来: 「分母之外还有东西」是本报表唯一能说自己没骗人的方式。
    同一族还有一条: 只看叶子目录这条**假设**自己也要被数一遍 (`nonleaf_with_result`)。

⚠️ 为什么这一片要重跑而不是引用 doc 60 里那个数: doc 60 §三 把 12,608 个目录归因成
「没有 per-intent key ⇒ 重复留在磁盘上」。这次把它拆开数之后，那个归因**大部分是错的** ——
而**本片自己第一版的解释也被自己的第二次读数推翻了一次** (第一版在 `render()` 里断言
「这些多出来的目录是一次运行里的不同轮」, 实测那两个来源各占多少之后发现占目录最多的那几个
task_id 的 `round_done` 全是 1)。两段更正都写在 doc 60 §三。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

RESULT = "result.json"
SNAP = "execution_snapshot.json"

KIND_FULL = "full"
KIND_STAGING = "staging"
KIND_OTHER = "other"


def classify(dir_path: str) -> str:
    """一个叶子目录属于三格里的哪一格 (判据只有文件名的存在性, 不读内容)"""
    try:
        names = set(os.listdir(dir_path))
    except OSError:
        return KIND_OTHER
    if RESULT in names:
        return KIND_FULL
    if names == {SNAP}:
        return KIND_STAGING
    return KIND_OTHER


def _read_json(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _snapshot_fields(dir_path: str) -> tuple:
    """一次读取取 `(task_id, round_done)` —— 分两次读同一份快照会造出两个互不一致的读数

    目录名是 `snap_<日期>_<时间>_<task_id>`, 但 task_id 自己可能含下划线 (驱动手写
    `driftcase` 那种), 所以名字不可靠; 快照里的字段才是出处。`round_done` 是这一格
    能不能把「一次运行的不同轮」与「同一个 intent 被重新发过」分开的唯一原料。
    """
    data = _read_json(os.path.join(dir_path, SNAP))
    if not isinstance(data, dict):
        return None, None
    tid = data.get("task_id")
    return (str(tid) if tid else None), data.get("round_done")


def inventory(root: str) -> dict:
    """走一遍 `root`, 数出三格与两类的重发频率。**只读**, 不写任何东西"""
    kinds = Counter()
    result_ids = Counter()          # task_id -> 完整产物数
    kernel_eq_task = 0
    kernel_ne_task = 0
    unreadable = 0
    staging_ids = Counter()         # task_id -> staging 目录数
    staging_rounds = Counter()      # (task_id, round_done) -> staging 目录数
    round_hist = Counter()          # round_done -> staging 目录数
    staging_no_id = 0
    nonleaf_with_result = 0         # 含 result.json 但**不是叶子** —— 这条路的目录今天不进分母
    for dirpath, dirnames, filenames in os.walk(root):
        if dirnames and dirpath != root and RESULT in filenames:
            nonleaf_with_result += 1
        if not dirnames and dirpath != root:
            kind = classify(dirpath)
            kinds[kind] += 1
            if kind == KIND_FULL:
                data = _read_json(os.path.join(dirpath, RESULT))
                if not isinstance(data, dict):
                    unreadable += 1
                    continue
                tid = data.get("task_id")
                kid = data.get("kernel_id")
                if tid:
                    result_ids[str(tid)] += 1
                if tid and kid:
                    if tid == kid:
                        kernel_eq_task += 1
                    else:
                        kernel_ne_task += 1
            elif kind == KIND_STAGING:
                tid, rnd = _snapshot_fields(dirpath)
                round_hist[str(rnd)] += 1
                if tid:
                    staging_ids[tid] += 1
                    staging_rounds[(tid, str(rnd))] += 1
                else:
                    staging_no_id += 1

    def _extra(counter: Counter) -> int:
        """多出来的次数 = 同一个 intent 第二次及以后的写入次数"""
        return sum(v - 1 for v in counter.values() if v > 1)

    # 把「同一个 task_id 多出来的那些快照目录」拆成两个来源 —— 这是一个恒等式而不是判断:
    #   总目录 - (task_id, round) 分组数 = 换轮多出来的 + 同一轮再来一次多出来的
    # 后者才是「同一个 intent 被重新发过」的证据, 前者才是「每一轮各占一个目录」的证据。
    # 写这一段之前, 这个脚本的 `render()` 里直接断言了「后者是每轮一个目录」—— 那句话
    # 在 2026-09-30 的实测里**被推翻** (占目录最多的那几个 task_id 的 `round_done` 全是 1)。
    extra_by_new_round = len(staging_rounds) - len(staging_ids)
    extra_by_same_round = sum(staging_rounds.values()) - len(staging_rounds)
    full_total = sum(result_ids.values())
    return {
        "root": root,
        "dirs": {KIND_FULL: kinds[KIND_FULL], KIND_STAGING: kinds[KIND_STAGING],
                 KIND_OTHER: kinds[KIND_OTHER], "total": sum(kinds.values())},
        "unreadable_result": unreadable,
        "nonleaf_with_result": nonleaf_with_result,
        "intents": len(result_ids),
        "kernel_id_equals_task_id": kernel_eq_task,
        "kernel_id_differs_from_task_id": kernel_ne_task,
        # 两份分母各算各的: 把 staging 混进「intent 重发率」会得出相反的量 ——
        # 而「staging 那侧到底是不是重发」这一问是这一片才量出来的, 不是先验。
        "resend_full": {"dirs": full_total, "intents": len(result_ids),
                        "extra_writes": _extra(result_ids),
                        "resended_intents": sum(1 for v in result_ids.values() if v > 1),
                        "max_per_intent": max(result_ids.values()) if result_ids else 0},
        "staging_per_intent": {"dirs": sum(staging_ids.values()),
                               "intents": len(staging_ids),
                               "no_task_id": staging_no_id,
                               "extra_dirs": _extra(staging_ids),
                               "extra_by_new_round": extra_by_new_round,
                               "extra_by_same_round": extra_by_same_round,
                               "round_hist": dict(round_hist),
                               "max_dirs_per_intent": max(staging_ids.values())
                               if staging_ids else 0},
        # 两个 id 集合的交集: 「写过快照」与「写完过整份产物」是两件事。这一格才是
        # doc 60 §三 那个归因真正该问的问题 —— 那 10,292 个快照目录来自**多少次没走完的运行**,
        # 而不是来自「同一个 intent 被重发了几次」。
        "id_sets": {"full_only": len(set(result_ids) - set(staging_ids)),
                    "both": len(set(result_ids) & set(staging_ids)),
                    "staging_only": len(set(staging_ids) - set(result_ids))},
    }


def render(rep: dict) -> str:
    d = rep["dirs"]
    f, s = rep["resend_full"], rep["staging_per_intent"]
    lines = [f"清点根目录: {rep['root']}"]
    lines.append(f"叶子目录 {d['total']} 个 = 完整产物 {d[KIND_FULL]} · "
                 f"只有快照的 staging {d[KIND_STAGING]} · **两种都不是 {d[KIND_OTHER]}**"
                 f"（这一格不进任何分母, 但必须先被看见）")
    nle = rep["nonleaf_with_result"]
    lines.append(f"「只看叶子」这条假设的盲区: 含 {RESULT} 又带子目录的目录 {nle} 个 —— "
                 + (f"⚠️ 上面的分母因此**少算了 {nle} 份**, 这个数是一个下限而不是全量"
                    if nle else "0 个 ⇒ 分母没被这条假设吃掉（这一句是逐目录数出来的, 不是假设）"))
    if rep["unreadable_result"]:
        lines.append(f"  ⚠️ 其中 {rep['unreadable_result']} 份 result.json 读不出 —— "
                     f"「读不出」不等于「没有」, 它们不进 intent 分母")
    lines.append(f"intent 在目录层不可见: {rep['kernel_id_differs_from_task_id']} 份产物的 "
                 f"kernel_id ≠ task_id, 相等的 {rep['kernel_id_equals_task_id']} 份 "
                 f"⇒ 目录名用的是**每次运行都换新值**的那个 id")
    rate = (100.0 * f["extra_writes"] / f["dirs"]) if f["dirs"] else 0.0
    lines.append(f"同一个 intent 被重发 (只数完整产物): {f['dirs']} 份产物 / "
                 f"{f['intents']} 个 intent ⇒ 多余写入 {f['extra_writes']} 次 = {rate:.1f}%, "
                 f"最多的一个 intent 占了 {f['max_per_intent']} 份")
    lines.append(f"另一件事 (staging 快照): {s['dirs']} 个目录 / {s['intents']} 个 task_id ⇒ "
                 f"多出来 {s['extra_dirs']} 个, 最多的一个 task_id 占了 {s['max_dirs_per_intent']} 个"
                 + (f"; 另有 {s['no_task_id']} 个 staging 目录连快照里的 task_id 都取不到"
                    if s["no_task_id"] else ""))
    top_rounds = sorted(s["round_hist"].items(), key=lambda kv: -kv[1])[:3]
    lines.append(f"  这 {s['extra_dirs']} 个拆成两个来源 (恒等式, 两边相加正好等于它): "
                 f"换了轮次多出来的 {s['extra_by_new_round']} 个 · "
                 f"**同一个轮次再来一次多出来的 {s['extra_by_same_round']} 个** —— 后者是"
                 f"「同一份 intent 被重新发过」, 不是「一轮一个目录」"
                 f"（快照里 `round_done` 的分布: "
                 + ", ".join(f"{k} 轮 {v} 个" for k, v in top_rounds) + "）")
    st = rep["id_sets"]
    lines.append(f"「写过快照」与「写完过整份产物」是两个集合, 不是一件事: 两处都有 {st['both']} 个 · "
                 f"只有完整产物 {st['full_only']} 个 · 只有快照而没有完整产物 {st['staging_only']} 个"
                 "（第三格是那些快照目录的出处; 它们**为什么**没走到 save 不在本片的读数里）")
    lines.append("⚠️ 这两个数**不能相加**: 前者是「同一件事被发了几次」(完整产物那一侧), "
                 "后者是「同一个 task_id 名下又开了几个快照目录」。doc 60 §三 那个「12,608 个目录 "
                 "= 没有 per-intent key ⇒ 重复留在磁盘上」的归因把两者当成了一件事, 而且它连"
                 "后者是哪种来源都没量。只观测: 本片不改 `artifact_store` 的命名, 也不 pin key。")
    return "\n".join(lines)


# ── 自检: 合成一棵树, 三格与两个数都必须按预期读出来 ──────────────────────────
def _build_fake_tree(base: str) -> dict:
    """返回「脚本应当数出什么」, 与 `inventory()` 的输出对拍 (判据只有一份实现)"""
    want = {"full": 0, "staging": 0, "other": 0}

    def mk(name, files):
        p = os.path.join(base, name)
        os.makedirs(p, exist_ok=True)
        for fn, body in files.items():
            full = os.path.join(p, fn)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8") as f:
                f.write(body)
        return p

    same_intent = json.dumps({"task_id": "case-A", "kernel_id": "u111"})
    other_intent = json.dumps({"task_id": "case-B", "kernel_id": "u222"})
    broken = "{不是 JSON"

    def snap(tid, rnd):
        return json.dumps({"task_id": tid, "round_done": rnd})
    mk("x/case-A-1", {RESULT: same_intent, "code.py": ""})
    mk("x/case-A-2", {RESULT: same_intent, "code.py": ""})
    want["full"] += 2
    mk("x/case-B-1", {RESULT: other_intent, "code.py": ""})
    want["full"] += 1
    mk("x/broken", {RESULT: broken})
    want["full"] += 1                       # 分类只看文件名: 读不出仍算完整产物那一格
    # case-A 名下故意三种都齐: 同一轮两次 (重发) + 换一轮 (一轮一目录) + 取不到 id 的那个
    mk("snap_20260101_101010_case-A", {SNAP: snap("case-A", 1)})
    mk("snap_20260101_101011_case-A", {SNAP: snap("case-A", 1)})
    mk("snap_20260101_101015_case-A", {SNAP: snap("case-A", 2)})
    mk("snap_20260101_101013_case-A", {SNAP: "{}"})   # 是 staging, 但快照里没有 task_id
    mk("snap_20260101_101012_case-A", {"task_id_missing.json": "{}"})
    mk("snap_20260101_101014_case-D", {SNAP: snap("case-D", 1)})
    want["staging"] += 5
    want["other"] += 1
    # 「只看叶子」这条假设的盲区: 一份完整产物, 但它带子目录 ⇒ 今天不进分母
    mk("x/case-C-1", {RESULT: json.dumps({"task_id": "case-C", "kernel_id": "u333"}),
                      "variants/v1.py": ""})
    want["other"] += 1              # 它那个子目录只有 code 文件, 落在第三格
    return want


def selftest() -> int:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        want = _build_fake_tree(td)
        rep = inventory(td)
        d = rep["dirs"]
        bad = []
        if d[KIND_FULL] != want["full"]:
            bad.append(f"完整产物数 {d[KIND_FULL]} != {want['full']}")
        if d[KIND_STAGING] != want["staging"]:
            bad.append(f"staging 数 {d[KIND_STAGING]} != {want['staging']}")
        if d[KIND_OTHER] != want["other"]:
            bad.append(f"第三格数 {d[KIND_OTHER]} != {want['other']}")
        if rep["unreadable_result"] != 1:
            bad.append(f"读不出的 result.json 应当单独数到 1, 实测 {rep['unreadable_result']}")
        f = rep["resend_full"]
        # case-A 两份 + case-B 一份 + 那份读不出的 (它不进 intent 分母) ⇒ 3 份 / 2 个 intent / 1 次多余
        if (f["dirs"], f["intents"], f["extra_writes"]) != (3, 2, 1):
            bad.append(f"重发读数 {(f['dirs'], f['intents'], f['extra_writes'])} != (3, 2, 1)")
        if f["max_per_intent"] != 2:
            bad.append(f"单个 intent 最多占的完整产物数 {f['max_per_intent']} != 2")
        s = rep["staging_per_intent"]
        # 带 task_id 的 staging 目录只有 4 个 (case-A×3 + case-D×1); 那个快照里没 id 的
        # 落进 `no_task_id`, 不进 `dirs` —— 两个数一起算就是把「取不到」当「没有」
        if (s["dirs"], s["intents"], s["extra_dirs"]) != (4, 2, 2):
            bad.append(f"staging 读数 {(s['dirs'], s['intents'], s['extra_dirs'])} != (4, 2, 2)")
        if s["max_dirs_per_intent"] != 3:
            bad.append(f"单个 task_id 最多占的快照目录数 {s['max_dirs_per_intent']} != 3")
        # 这一对是本片新加的那条恒等式: 多出来的目录必须能全量拆进两个来源, 一格都不许丢
        if (s["extra_by_new_round"], s["extra_by_same_round"]) != (1, 1):
            bad.append(f"「换轮」与「同轮再来一次」的拆分 {(s['extra_by_new_round'], s['extra_by_same_round'])} "
                       f"!= (1, 1) —— 这一拆就是 doc 60 §三 那个归因缺的那一步")
        if s["extra_by_new_round"] + s["extra_by_same_round"] != s["extra_dirs"]:
            bad.append("两个来源相加不等于 `extra_dirs`: 拆分丢了东西, 恒等式破了")
        if s["round_hist"].get("None") != 1:
            bad.append(f"快照里读不出 `round_done` 的那一格应当单独有一档, 实测 "
                       f"{s['round_hist'].get('None')} —— 它不许被当成某一轮")
        if s["no_task_id"] != 1:
            bad.append(f"「是 staging 但取不到 task_id」那一格应当单独数到 1, "
                       f"实测 {s['no_task_id']} —— 它不许被并进 `intents` 或 `dirs`")
        if rep["kernel_id_equals_task_id"] != 0:
            bad.append("合成树里 kernel_id 全都不等于 task_id; 数出相等说明读错了键")
        if rep["nonleaf_with_result"] != 1:
            bad.append(f"盲区计数应当数到那 1 份「有 result.json 又带子目录」的产物, "
                       f"实测 {rep['nonleaf_with_result']} —— 没有它, 上面的分母就是一个"
                       f"没人核过的假设")
        if rep["intents"] == 3:
            bad.append("case-C 那份产物不该进 intent 分母（它不是叶子）; "
                       "数到 3 说明分类与盲区计数用了两套判据")
        st = rep["id_sets"]
        # A 两处都有 · B 只有完整产物 · D 只有快照 ⇒ 三格各 1, 谁也不许被并入谁
        if (st["both"], st["full_only"], st["staging_only"]) != (1, 1, 1):
            bad.append(f"两个 id 集合的三格读数 {(st['both'], st['full_only'], st['staging_only'])} "
                       f"!= (1, 1, 1) —— 「写过快照」与「写完过产物」混成一件就是 doc 60 §三 犯的错")
    if bad:
        print("❌ 自检不通过:")
        for b in bad:
            print("   -", b)
        return 2
    print("[selftest] ok —— 三格分类、读不出那一格、两个分母都按预期对上")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="output")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", action="store_true", help="把整份读数按 JSON 打出来")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not os.path.isdir(a.root):
        print(f"❌ 根目录不存在: {a.root} —— 这些读数只在本机存量产物上可得, "
              f"没有它们就是「测不到」而不是 0")
        return 2
    rep = inventory(a.root)
    if a.json:
        print(json.dumps(rep, ensure_ascii=False, sort_keys=True))
    else:
        print(render(rep))
    total = rep["dirs"]["total"]
    if total == 0:
        print("❌ 一个叶子目录都没数到 —— 那是脚本没跑到该跑的地方, 不是「没有产物」")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
