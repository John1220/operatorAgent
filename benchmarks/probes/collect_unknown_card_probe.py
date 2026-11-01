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
     峰值与带宽用 `benchmarks/calibration/calibrate_achievable_peak.py`；两者尾巴上都会
     打一行 `PROBE_JSON {...}`，用 `--from-probe` 喂回来。采集器只负责按门的 schema 组装，
     **而且只认读数里的 uuid 与本卡一致的那些**（`peak_trusted=True` 是 G3 的验收条件）。

跑法（在目标机上，torch 可用）：

    # 先在本卡上产两份实测（都会打 PROBE_JSON 行）
    python3 benchmarks/measurement/measure_launch_floor.py      > ./floor.txt
    python3 benchmarks/calibration/calibrate_achievable_peak.py  > ./peak.txt
    # 再采集：把实测喂进门的 schema
    python3 collect_unknown_card_probe.py --sharing exclusive \
        --from-probe ./floor.txt --from-probe ./peak.txt --out ./card-probe.json
    #  然后拿回本地判门：
    python3 -B -m benchmarks.probes.unknown_card_gates --from-json ./card-probe.json
"""

import argparse
import json
import sys
from pathlib import Path

SHARING_VALUES = ("exclusive", "desktop", "shared")

PROBE_PREFIX = "PROBE_JSON "   # 标定/地板脚本尾巴上的机器可读行

MEASURED_BY_ELSEWHERE = {
    "launch_floor": "benchmarks/measurement/measure_launch_floor.py（≥5 次同形状，取中位数；"
                    "尾巴打 PROBE_JSON 行，用 --from-probe 接进来，uuid 由脚本自己带）",
    "peak": "benchmarks/calibration/calibrate_achievable_peak.py（本卡实测峰值与 D2D 带宽，"
            "同样打 PROBE_JSON）；不要指望 CC 兜底表——CC 8.9 那行按 L40S 填，偏大",
    "noise.band/dispersion": "同一候选重复 ≥5 次的跨次散布（8 GB 档以上才有意义，"
                             "别沿用仓内 0.001/0.004/0.02 那组 H20 八卡共享机标定值）",
    "shapes": "在本卡上重跑一遍才算验收样本；H20 上跑过的形状不构成新卡的可执行条件",
}


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
    node = dict(node)
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
    payload["still_missing"] = _still_missing(payload)
    return payload


def read_props():
    """在目标机上读设备属性；读不到就拒，不猜。"""
    try:
        import torch
    except ImportError:
        _refuse("目标机需要 torch（本地无 torch 就 scp 过去跑）")
    if not torch.cuda.is_available():
        _refuse("torch.cuda.is_available() 为假：这台机器上没有可探测的 CUDA 设备")
    idx = 0
    props = torch.cuda.get_device_properties(idx)
    return {
        "name": getattr(props, "name", None),
        "cc": "%d.%d" % (props.major, props.minor),
        "total_memory_bytes": int(getattr(props, "total_memory", 0)) or None,
        # opt-in 上限与每 SM 上限是两回事：G2 要的是前者（一个块能要到多少）
        "per_block_optin_bytes": int(getattr(props, "shared_memory_per_block_optin", 0)) or None,
        "per_sm_kb": int(getattr(props, "shared_memory_per_multiprocessor", 0)) // 1024 or None,
    }, idx


def read_uuid(idx):
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=gpu_uuid", "-i", str(idx),
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15)
    except Exception as exc:
        _refuse("nvidia-smi 取 UUID 失败：%r" % (exc,))
    if out.returncode != 0:
        _refuse("nvidia-smi 返回 %d：%s" % (out.returncode, out.stderr.strip()[:120]))
    return out.stdout.strip().splitlines()[0].strip()


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
    props, idx = read_props()
    uuid = read_uuid(idx)
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
