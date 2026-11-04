# -*- coding: utf-8 -*-
"""形状清单采集器：G5 要的「哪些形状可作本卡验收样本」的**来源**。

## 这个缺口原来长什么样

`unknown_card_gates.py` 的 G5 读 `info["shapes"]`，而全仓**没有任何脚本产出这一段**
（2026-11-01 实测 grep 命中只有门自己）。于是 G5 永远 unknown，runbook 第 0 步必然停住。
一段没有生产者的输入，在报告里长得和「已经核对过」几乎一样 —— 这是这个文件存在的理由。

## 它做的两件事，顺序不能换

1. **在本卡上真跑一遍。** 每个声明形状都分配输入与输出张量、跑一个 Triton 逐元素核、
   同步、再抽查一次数值。跑不通就拒，不写进产物。
2. **记实测分配字节数。** `bytes` 是这次实际占的显存（in + out 两块张量），不是按维度
   推算的理论工作集 —— 门的判据是「这个形状能不能在本卡的池子里跑」，那就该用跑的时候
   真的占了多少。

## 它**不**做的事

形状清单本身是**声明**，不是发现：用哪些形状由任务决定（`operator-agent prepare --shapes`
那一栏），脚本不猜。所以 `--shape` 是必填，缺了直接拒 —— 一个空清单会让
`g5 = info.get("shapes") or []` 落到 `not g5` 那一支判 unknown，看起来像「还没核对」，
而真相是「没人声明」。

跑法（目标机上，torch/triton 可用）：

    python3 collect_shape_inventory.py --shape 1024,768 --shape 4096,4096 \
        --dtype fp16 --out shapes.txt
    # 尾巴上的 PROBE_JSON 用 --from-probe 喂给 collect_unknown_card_probe.py

退出码：0 = 全部形状在本卡跑通并已写出；1 = 有形状跑不通（详情在 stderr）。
"""

import argparse
import json
import sys

PROBE_PREFIX = "PROBE_JSON "

DTYPES = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}


def _refuse(msg):
    raise SystemExit("shape inventory refused: %s" % msg)


def parse_shape(text):
    """`"1024,768"` → `(1024, 768)`。与 `operator-agent ... --shapes` 同一套写法。

    与 CLI 一样只收正整数；负数/零维会在分配前被拒，而不是让 torch 抛一个
    「size must be non-negative」—— 那个消息不会告诉你哪个形状错了。
    """
    parts = [p.strip() for p in str(text).split(",")]
    if not parts or any(not p for p in parts):
        _refuse("形状 %r 不是逗号分隔的整数列表" % text)
    try:
        dims = tuple(int(p) for p in parts)
    except ValueError:
        _refuse("形状 %r 里有非整数" % text)
    if any(d <= 0 for d in dims):
        _refuse("形状 %r 有非正维度" % text)
    return dims


def measure_shape(dims, dtype_name, device="cuda"):
    """分配 + 跑通 + 抽查，返回 (字节数, 说明)。跑不通抛异常，由调用方拒。"""
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _relu(X, Y, N, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < N
        tl.store(Y + offs, tl.maximum(tl.load(X + offs, mask=m), 0.0), mask=m)

    torch_dtype = getattr(torch, DTYPES[dtype_name])
    x = torch.randn(*dims, device=device, dtype=torch_dtype) - 0.5
    y = torch.empty_like(x)
    n = x.numel()
    _relu[(triton.cdiv(n, 1024),)](x, y, n, BLOCK=1024)
    torch.cuda.synchronize()
    # 抽查：抽一小段回 CPU 比对。跑通 ≠ 算对，而「能跑但结果是垃圾」同样是不可执行条件。
    probe = y.flatten()[:8].float().cpu()
    expect = torch.clamp(x.flatten()[:8].float().cpu(), min=0.0)
    if not torch.allclose(probe, expect, atol=1e-3, rtol=1e-3):
        raise AssertionError("relu 抽查不一致：%r vs %r" % (probe.tolist(), expect.tolist()))
    nbytes = (x.numel() + y.numel()) * x.element_size()
    del x, y
    torch.cuda.empty_cache()
    return nbytes, "in+out 两块张量实占；relu 跑通并抽查 8 个元素一致"


def collect(shapes, dtype_name, device="cuda"):
    """逐个形状实测。任何一个跑不通 ⇒ 整份清单作废（不给半份）。"""
    import torch
    if not torch.cuda.is_available():
        _refuse("torch.cuda.is_available() 为假：形状必须在有卡的机器上验")
    total = torch.cuda.get_device_properties(0).total_memory
    entries = []
    for dims in shapes:
        nbytes, note = measure_shape(dims, dtype_name, device=device)
        if nbytes > 0.8 * total:
            # 门会在 G5 判 fail；在这里就拒，是因为「>0.8×池」的形状连跑通都不该发生 ——
            # 真的分配到了说明池子比 self-report 的大，那本身就是另一条要先查的结论。
            raise AssertionError(
                "形状 %s 实占 %d 字节 > 0.8×池(%d)：它不构成这张卡的可执行条件"
                % (dims, nbytes, int(0.8 * total)))
        entries.append({"label": "%s %s" % ("x".join(map(str, dims)), dtype_name),
                        "bytes": int(nbytes), "note": note})
        print("  %-28s %12d bytes  %s" % (entries[-1]["label"], nbytes, note))
    return entries


def main(argv=None):
    ap = argparse.ArgumentParser(prog="collect_shape_inventory", allow_abbrev=False)
    ap.add_argument("--shape", action="append", default=[],
                    help='形状，可重复；如 --shape 1024,768（与 operator-agent --shapes 同写法）')
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    ap.add_argument("--out", help="写出路径；缺则只打印")
    args = ap.parse_args(argv)

    if not args.shape:
        _refuse("--shape 至少给一个：形状清单是**声明**，脚本不替任务猜")

    shapes = [parse_shape(s) for s in args.shape]
    print(f"形状清单（声明 %d 个，dtype={args.dtype}），逐个在本卡分配并跑通："
          % len(shapes))
    try:
        entries = collect(shapes, args.dtype)
    except AssertionError as exc:
        print("形状不合格：%s" % exc, file=sys.stderr)
        return 1

    import subprocess
    import torch
    uuid = None
    props = torch.cuda.get_device_properties(0)
    try:
        # 钉卡后可见索引 0 就是被测卡；`nvidia-smi` 的第一行是物理 0 号卡，不是它。
        value = getattr(props, "uuid", None)
        if value:
            value = str(value)
            uuid = value if value.startswith("GPU-") else "GPU-" + value
    except Exception:
        uuid = None
    if not uuid:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=gpu_uuid", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=10)
            if out.returncode == 0 and out.stdout.strip():
                uuid = out.stdout.strip().splitlines()[0].strip()
        except Exception:
            uuid = None
    if not uuid:
        # 没有 uuid 就不写产物：整份清单的价值在于「这些形状是在**这张卡**上验过的」。
        _refuse("取不到本卡 UUID（torch 与 nvidia-smi 都没给）：没有来源戳的形状清单不许进表")

    payload = {"uuid": uuid, "name": props.name, "cc": "%d.%d" % (props.major, props.minor),
               "device_total_memory_bytes": int(props.total_memory),
               "shapes": entries,
               "measured_by": "benchmarks/probes/collect_shape_inventory.py",
               "dtype": args.dtype}
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return emit(text, args.out)


def emit(text, out_path=None):
    """把载荷写成 `--from-probe` 直接能吃的形状，并打到 stdout。

    `--out` 先前只写裸 JSON，于是「写到文件里那份」喂不进采集器（`load_probes` 只认
    `PROBE_JSON ` 开头那行），实测当场踩到：`shapes.txt 里没有 PROBE_JSON 行`。
    现在文件与 stdout 是同一份带上前缀的东西 —— 少一次手工拼接就少一个坑。
    """
    line = PROBE_PREFIX + text
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(line + "\n")
        print("wrote %s（可直接 --from-probe）" % out_path)
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())