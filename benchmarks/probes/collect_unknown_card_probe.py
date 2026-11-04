# -*- coding: utf-8 -*-
"""未知卡探测采集器：把「能机械读到的」读出来，把「没测的」明确留空。

与 `unknown_card_gates.py` 的关系：这个采集器**产出输入**，那道门**做判决**。
两个文件之间的字段契约由 `tests/unit/test_unknown_card_probe.py` 钉住——采集器写一个键、
门读另一个键，后果不是崩，而是五门全 unknown 且 nobody notices。

三条规矩：
  1. **不编默认值。** 共享形态（独占 / 接显示器 / 有对手）必须显式给，缺就拒绝。
     消费卡接显示器会让 util 长期 >5%，按现阈值每轮都 degraded——这是 4070 这类卡
     最可能的实际杀手，不能靠默认值糊过去。
  2. **来源戳只盖在自己真测过的段上。** `source_uuid` 只写进 device / shared_memory /
     noise 的形态观测；launch 地板、峰值分母、散布带这里一个都不填，门会把它们判 unknown。
  3. **不重造测量。** 地板用 `benchmarks/measurement/measure_launch_floor.py`，
     峰值与带宽用 `benchmarks/calibration/calibrate_achievable_peak.py`，散布带与
     污染阈值用 `benchmarks/calibration/calibrate_dispersion_thresholds.py`，形状清单用
     `benchmarks/probes/collect_shape_inventory.py`；四者尾巴上都会打一行
     `PROBE_JSON {...}`，用 `--from-probe` 喂回来。采集器只负责按门的 schema 组装，
     **而且只认读数里的 uuid 与本卡一致的那些**（`peak_trusted=True` 是 G3 的验收条件）。

跑法（在目标机上，torch 可用）：

    # 先在本卡上产四份实测（都会打 PROBE_JSON 行）
    python3 benchmarks/measurement/measure_launch_floor.py            > ./floor.txt
    python3 benchmarks/calibration/calibrate_achievable_peak.py       > ./peak.txt
    python3 benchmarks/calibration/calibrate_dispersion_thresholds.py > ./noise.txt
    python3 benchmarks/probes/collect_shape_inventory.py \
        --shape 1024,768 --shape 4096,4096 > ./shapes.txt
    # 再采集：把实测喂进门的 schema（--sharing 必须是**标定当时**的形态）
    python3 collect_unknown_card_probe.py --sharing exclusive \
        --from-probe ./floor.txt --from-probe ./peak.txt \
        --from-probe ./noise.txt --from-probe ./shapes.txt --out ./card-probe.json
    #  然后拿回本地判门：
    python3 -B -m benchmarks.probes.unknown_card_gates --from-json ./card-probe.json

  2026-11-01 之前 `noise` 与 `shapes` 两段**没有任何产出方**，于是 G4/G5 永远 unknown、
  runbook 第 0 步必然停住（一期清单里记成「结构性缺量具」）。两段现在各有来源，
  但纪律不变：**阈值必须是本卡的，形状必须在本卡上跑过**，缺一项就还是 unknown。
"""

import argparse
import json
import os
import sys
from pathlib import Path

SHARING_VALUES = ("exclusive", "desktop", "shared")

PROBE_PREFIX = "PROBE_JSON "   # 标定/地板脚本尾巴上的机器可读行

MEASURED_BY_ELSEWHERE = {
    "launch_floor": "benchmarks/measurement/measure_launch_floor.py（≥5 次同形状，取中位数；"
                    "尾巴打 PROBE_JSON 行，用 --from-probe 接进来，uuid 由脚本自己带）",
    "peak": "benchmarks/calibration/calibrate_achievable_peak.py（本卡实测峰值与 D2D 带宽，"
            "同样打 PROBE_JSON）；不要指望 CC 兜底表——CC 8.9 那行按 L40S 填，偏大",
    "noise.band/dispersion": "benchmarks/calibration/calibrate_dispersion_thresholds.py"
                             "（同一核连跑多轮的跨次散布 → 判决带宽；轮内 iqr/median 的 p95"
                             " → 散布门限；静止窗 util → 污染阈值。三者必须同时给出，"
                             "别沿用仓内 0.001/0.004/0.02 那组 H20 八卡共享机标定值）",
    "shapes": "benchmarks/probes/collect_shape_inventory.py（在**本卡**上逐个分配并跑通，"
              "字节数是实测分配量；H20 上跑过的形状不构成新卡的可执行条件）",
}

#: G4 要的三个阈值。**必须三个一起给**：只给一个会让「段存在」看起来像「已标定」，
#: 而那正是 `_still_missing` 当初偷懒会踩的坑（见那里 docstring）。
NOISE_THRESHOLD_KEYS = ("band", "contamination_util_pct", "dispersion_degraded")

#: `band` 是按测量档位分档的判决带宽，键名由 `measurement.NOISE_BAND` 定死。
#: 拼错一个键**不会崩**：`_same_as_inherited()` 拿它跟继承值比对会判「不等」，
#: 于是继承值反而被放行——这是这一格唯一的静默失败方向，所以下面显式校验键集。
BAND_KEYS = frozenset(("valid", "degraded", "uninformative"))


def _refuse(msg):
    raise SystemExit("collect refused: %s" % msg)


def load_probes(paths):
    """从测量脚本的输出里取 PROBE_JSON 行。取不到不算错，取到但没 uuid 才是问题。"""
    out = []
    for path in paths:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        line = [ln for ln in text.splitlines() if ln.startswith(PROBE_PREFIX)]
        if not line:
            _refuse("%s 里没有 %s 行：那份脚本没被改成可机器读，别手抄数字进来"
                    % (path, PROBE_PREFIX))
        out.append(json.loads(line[-1][len(PROBE_PREFIX):].strip()))
    return out


def _stamp(node, uuid, origin):
    """盖本卡的来源戳；脚本自报的卡号**对不上就停**，不悄悄覆盖。

    测量脚本自己也会打一个 `uuid`（它们在钉卡后仍可能报「可见索引 0」= 物理 0 号卡）。
    这里覆盖成采集器解析出的卡，但覆盖前先比一次：两份声明不一致说明有一份是错的，
    而错的那份正落在「来源」这一格上 —— 静默覆盖掉只会让错误消失，不会让错误不存在。
    """
    node = dict(node)
    claimed = node.get("uuid")
    if claimed and _norm_uuid(claimed) != _norm_uuid(uuid):
        _refuse("%s 自报的卡是 %s，而本次实测在 %s 上：先修那份脚本的卡号解析，"
                "别把它盖成本卡" % (origin, claimed, uuid))
    node["source_uuid"] = uuid
    node.setdefault("probed_by", origin)
    return node


def _still_missing(payload):
    """按**门实际读什么**列缺口，不按段名猜。

    `noise` 整段是采集器填的（共享形态是本卡观测），但 G4 要的是段里的三个阈值——
    所以「段存在」不等于「缺口已补」，这里若偷懒用 `split(".")[0] in payload`，
    `noise.band/dispersion` 会永远从 still_missing 里消失，看起来像已经标定过了。
    """
    out = []
    for key in sorted(MEASURED_BY_ELSEWHERE):
        if key == "noise.band/dispersion":
            noise = payload.get("noise") or {}
            if any(noise.get(f) is None for f in
                   ("band", "contamination_util_pct", "dispersion_degraded")):
                out.append(key)
        elif key.split(".")[0] not in payload:
            out.append(key)
    return out


def assemble(props, uuid, sharing, peers, probes=()):
    """把设备属性装配成门的输入。纯函数，零 GPU 可测。

    props: {"name","cc","total_memory_bytes","per_block_optin_bytes","per_sm_kb"}
    """
    if sharing not in SHARING_VALUES:
        _refuse("sharing must be one of %s, got %r — 共享形态决定污染阈值的方向，不给默认值"
                % (list(SHARING_VALUES), sharing))
    if not uuid:
        _refuse("device uuid 未知：没有来源戳的读数一律不许进表")
    if props.get("per_block_optin_bytes") is None and props.get("per_sm_kb") is None:
        _refuse("每块 opt-in 与每 SM 两个共享内存上限**都没**从设备属性读到（G2 要的正是这个，"
                "缺它 tile 预算会落到兜底值）—— torch build 只暴露一半是可以的，全没有不行")
    payload = {
        "device": {"uuid": uuid, "name": props.get("name"), "cc": props.get("cc"),
                   "total_memory_bytes": props.get("total_memory_bytes")},
        "shared_memory": {"per_block_optin_bytes": props.get("per_block_optin_bytes"),
                          "per_sm_kb": props.get("per_sm_kb"), "source_uuid": uuid},
        "noise": {"sharing_pattern": sharing, "peers_observed": peers, "source_uuid": uuid},
        "collected_from": "torch.cuda.get_device_properties + nvidia-smi",
    }
    # 实测段的来源核对：**别的卡的数不盖本卡的戳**。宁缺毋滥 —— 门会把缺的报 unknown，
    # 而错盖戳会让它在没测过的情况下判 pass，那比 unknown 危险得多。
    merged = []
    for probe in probes:
        p_uuid = probe.get("uuid")
        origin = probe.get("measured_by", "probe")
        if not p_uuid:
            _refuse("%s 的探测输出没有 uuid：拒绝把来历不明的读数当本卡标定" % origin)
        if p_uuid != uuid:
            _refuse("%s 的读数来自 %s，不是本机 %s —— 跨卡的地板与峰值不许混用"
                    % (origin, p_uuid, uuid))
        merged.append((probe, origin))
    for probe, origin in merged:
        if probe.get("launch_floor"):
            payload["launch_floor"] = _stamp(probe["launch_floor"], uuid, origin)
        if probe.get("peak_by_dtype") and probe.get("mem_bw_gbs"):
            payload["peak"] = _stamp({"mem_bw_gbs": probe["mem_bw_gbs"],
                                      "peak_by_dtype": probe["peak_by_dtype"],
                                      "trusted": True}, uuid, origin)
        if probe.get("noise"):
            payload["noise"].update(_noise_thresholds(probe["noise"], origin))
        if probe.get("shapes"):
            payload["shapes"] = _validated_shapes(probe["shapes"], uuid, origin)
    payload["still_missing"] = _still_missing(payload)
    return payload


def _noise_thresholds(node, origin):
    """从散布标定的输出里取 G4 要的三个阈值；**缺一个就拒**，不给半截标定。

    来源戳不在这里盖：`noise.source_uuid` 在 payload 里已经是本卡 uuid，且 `assemble()`
    的合并循环已经拒过 uuid 不一致的探针，再盖一次只会掩盖「这段是谁测的」这件事，
    所以血缘记在 `probed_by` 上。
    """
    band = node.get("band")
    if not isinstance(band, dict):
        _refuse("%s 的 noise.band 不是分档字典（G4 要 valid/degraded/uninformative 三档）"
                % origin)
    if set(band) != BAND_KEYS:
        _refuse("%s 的 noise.band 键集是 %s，必须是 %s —— 拼错的键会让继承值被判成"
                "「不等」而放行" % (origin, sorted(band), sorted(BAND_KEYS)))
    out = {}
    for key in NOISE_THRESHOLD_KEYS:
        value = node.get(key)
        if value is None:
            _refuse("%s 的 noise 缺 %s：三个阈值必须同一次标定里一起给出，"
                    "半截标定看起来像标定过" % (origin, key))
        if not isinstance(value, dict) and not isinstance(value, (int, float)):
            _refuse("%s 的 noise.%s 不是数值：%r" % (origin, key, value))
        out[key] = value
    out["probed_by"] = origin
    return out


def _validated_shapes(entries, uuid, origin):
    """校验形状清单：**只接受在本卡上跑过的条目**。

    `validated_on_uuid` 缺失 ⇒ 补成本卡（探针的 uuid 已经与本卡比对过，所以它确实是
    在本卡跑的）；写了别的卡 ⇒ 直接拒，因为「在 H20 上验收过的形状不构成新卡的可执行
    条件」，让门去 fail 也能拦住，但那是把该在入口处挡的东西推给下游。
    """
    if not isinstance(entries, list):
        _refuse("%s 的 shapes 不是列表：%r" % (origin, entries))
    out = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("label"):
            _refuse("%s 的 shapes 条目缺 label：%r" % (origin, entry))
        size = entry.get("bytes")
        if not isinstance(size, int) or size <= 0:
            _refuse("%s 的形状 %s 没给实测字节数：%r" % (origin, entry["label"], size))
        stamped_on = entry.get("validated_on_uuid")
        if stamped_on not in (None, uuid):
            _refuse("%s 的形状 %s 是在 %s 上验收的，不是本机 %s"
                    % (origin, entry["label"], stamped_on, uuid))
        item = dict(entry)
        item["validated_on_uuid"] = uuid
        item["probed_by"] = origin
        out.append(item)
    return out


def read_props():
    """在目标机上读设备属性；读不到就拒，不猜。"""
    try:
        import torch
    except ImportError:
        _refuse("目标机需要 torch（本地无 torch 就 scp 过去跑）")
    if not torch.cuda.is_available():
        _refuse("torch.cuda.is_available() 为假：这台机器上没有可探测的 CUDA 设备")
    idx = 0                                  # torch 的**可见**索引: 钉卡后物理卡不是 0
    props = torch.cuda.get_device_properties(idx)
    physical, uuid = resolve_device(os.environ.get("CUDA_VISIBLE_DEVICES"), gpu_rows())
    shown = getattr(props, "uuid", None)
    if shown and _norm_uuid(shown) != _norm_uuid(uuid):
        _refuse("torch 看见的卡 (%s) 与 nvidia-smi 解析出的卡 (%s) 不一致："
                "来源戳会挂错卡，先查 CUDA_VISIBLE_DEVICES" % (shown, uuid))
    return {
        "name": getattr(props, "name", None),
        "cc": "%d.%d" % (props.major, props.minor),
        "total_memory_bytes": int(getattr(props, "total_memory", 0)) or None,
        # opt-in 上限与每 SM 上限是两回事：G2 要的是前者（一个块能要到多少）
        "per_block_optin_bytes": int(getattr(props, "shared_memory_per_block_optin", 0)) or None,
        "per_sm_kb": int(getattr(props, "shared_memory_per_multiprocessor", 0)) // 1024 or None,
    }, idx, physical, uuid


def _norm_uuid(value):
    """UUID 归一: nvidia-smi 给 `GPU-xxx`，torch 给裸 `xxx`（或 `GPU-xxx`，随版本）。"""
    text = str(value).strip()
    return text[len("GPU-"):] if text.startswith("GPU-") else text


def resolve_device(cvd, rows):
    """把「实际被测的那张卡」解析成 nvidia-smi 的 `(physical_index, 'GPU-…')`。

    为什么必须解析而不是读 index 0: `CUDA_VISIBLE_DEVICES=3` 之后 torch 的 index 0 是
    **物理 3 号卡**，而 `nvidia-smi -i 0` 仍然指物理 0 号。共享机上钉住非 0 卡时直接读
    index 0，会把本卡的读数盖上**别人那张卡**的来源戳 —— 这种错戳不会崩、也不会让任何
    断言变红，只会让来源门以为测的就是它（2026-11-04 RTH H20 8 卡共享机实测踩到）。

    rows: [(physical_index, uuid)]，来自 `nvidia-smi --query-gpu=index,gpu_uuid`。
    """
    rows = [(int(index), str(uuid).strip()) for index, uuid in rows]
    if not rows:
        _refuse("nvidia-smi 没有返回任何 GPU 行")
    raw = (cvd or "").strip()
    if raw == "":                            # 未设置: CUDA 默认用序号最小那张
        return rows[0]
    first = raw.split(",")[0].strip()
    if first == "":
        _refuse("CUDA_VISIBLE_DEVICES=%r 表示没有可见 GPU，但 torch 看见了设备" % (raw,))
    if first.lstrip("-").isdigit():
        for row in rows:
            if row[0] == int(first):
                return row
        _refuse("CUDA_VISIBLE_DEVICES 指向物理卡 %s，nvidia-smi 没有这张卡（看到的：%s）"
                % (first, ", ".join(str(r[0]) for r in rows)))
    for row in rows:
        if _norm_uuid(row[1]) == _norm_uuid(first):
            return row
    _refuse("CUDA_VISIBLE_DEVICES=%r 不是任何一张卡的 UUID（看到的：%s）"
            % (first, ", ".join(r[1] for r in rows)))


def gpu_rows():
    """`nvidia-smi` 的 (index, uuid) 全表。读不到就拒 —— 不猜一个卡号。"""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,gpu_uuid",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15)
    except Exception as exc:
        _refuse("nvidia-smi 取 GPU 清单失败：%r" % (exc,))
    if out.returncode != 0:
        _refuse("nvidia-smi 返回 %d：%s" % (out.returncode, out.stderr.strip()[:120]))
    rows = []
    for line in out.stdout.splitlines():
        parts = [piece.strip() for piece in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            rows.append((int(parts[0]), parts[1]))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(prog="collect_unknown_card_probe", allow_abbrev=False)
    ap.add_argument("--sharing", choices=SHARING_VALUES,
                    help="exclusive=独占跑计算；desktop=接显示器；shared=有别的进程同卡")
    ap.add_argument("--peers", type=int, default=None, help="观测到的同卡对手数（unknown 就别填）")
    ap.add_argument("--out", help="写出路径；缺则打印到 stdout")
    ap.add_argument("--print-only", action="store_true", help="只打印不落盘")
    ap.add_argument("--from-probe", action="append", default=[],
                    help="测量脚本的输出（含 PROBE_JSON 行），可重复："
                         "calibrate_achievable_peak / measure_launch_floor")
    args = ap.parse_args(argv)
    props, idx, physical, uuid = read_props()
    print("本卡 = 物理 %d 号 / %s（CUDA_VISIBLE_DEVICES=%r, torch 可见索引 %d）"
          % (physical, uuid, os.environ.get("CUDA_VISIBLE_DEVICES", ""), idx),
          file=sys.stderr)
    payload = assemble(props, uuid, args.sharing, args.peers, load_probes(args.from_probe))
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out and not args.print_only:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print("wrote %s" % args.out)
    print(text)
    print("\n还缺这些段（门会把它们判 unknown，不开跑）：", file=sys.stderr)
    for section in payload["still_missing"]:
        print("  - %-22s %s" % (section, MEASURED_BY_ELSEWHERE[section]), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
