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
     峰值用 `operator_agent/decide/peak_probe.py`（`peak_trusted=True` 是 G3 的验收条件），
     采集器只负责把结果按门的 schema 组装起来。

跑法（在目标机上，torch 可用）：

    python3 collect_unknown_card_probe.py --sharing exclusive --out ./card-probe.json
    #  然后拿回本地判门：
    python3 -B -m benchmarks.probes.unknown_card_gates --from-json ./card-probe.json
"""

import argparse
import json
import sys
from pathlib import Path

SHARING_VALUES = ("exclusive", "desktop", "shared")

#: 采集器会填的段；其余全部留给门去报 unknown。
FILLED_SECTIONS = ("device", "shared_memory", "noise")

MEASURED_BY_ELSEWHERE = {
    "launch_floor": "benchmarks/measurement/measure_launch_floor.py（≥5 次同形状，取中位数；"
                    "产出 median_us + samples 后按门的 schema 填 source_uuid=本卡 UUID）",
    "peak": "operator_agent/decide/peak_probe.py + 本卡实测带宽；"
            "验收是 `peak_trusted=True` 且出处字符串不以 ⚠️ 开头（CC 8.9 的兜底行按 L40S 填，偏大）",
    "noise.band/dispersion": "同一候选重复 ≥5 次的跨次散布（8 GB 档以上才有意义，"
                             "别沿用仓内 0.001/0.004/0.02 那组 H20 八卡共享机标定值）",
    "shapes": "在本卡上重跑一遍才算验收样本；H20 上跑过的形状不构成新卡的可执行条件",
}


def _refuse(msg):
    raise SystemExit("collect refused: %s" % msg)


def assemble(props, uuid, sharing, peers):
    """把设备属性装配成门的输入。纯函数，零 GPU 可测。

    props: {"name","cc","total_memory_bytes","per_block_optin_bytes","per_sm_kb"}
    """
    if sharing not in SHARING_VALUES:
        _refuse("sharing must be one of %s, got %r — 共享形态决定污染阈值的方向，不给默认值"
                % (list(SHARING_VALUES), sharing))
    if not uuid:
        _refuse("device uuid 未知：没有来源戳的读数一律不许进表")
    if props.get("per_block_optin_bytes") is None or props.get("per_sm_kb") is None:
        _refuse("共享内存上限没从设备属性读到（G2 要的正是这个，缺它 tile 预算会落到兜底值）")
    return {
        "device": {"uuid": uuid, "name": props.get("name"), "cc": props.get("cc"),
                   "total_memory_bytes": props.get("total_memory_bytes")},
        "shared_memory": {"per_block_optin_bytes": props["per_block_optin_bytes"],
                          "per_sm_kb": props["per_sm_kb"], "source_uuid": uuid},
        "noise": {"sharing_pattern": sharing, "peers_observed": peers, "source_uuid": uuid},
        "collected_from": "torch.cuda.get_device_properties + nvidia-smi",
        "still_missing": sorted(MEASURED_BY_ELSEWHERE),
    }


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
    args = ap.parse_args(argv)
    props, idx = read_props()
    uuid = read_uuid(idx)
    payload = assemble(props, uuid, args.sharing, args.peers)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out and not args.print_only:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print("wrote %s" % args.out)
    print(text)
    print("\n还缺这些段（门会把它们判 unknown，不开跑）：", file=sys.stderr)
    for section in sorted(MEASURED_BY_ELSEWHERE):
        print("  - %-22s %s" % (section, MEASURED_BY_ELSEWHERE[section]), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
