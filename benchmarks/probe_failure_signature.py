"""真机复核: 失败签名是从**远程真实 error_log** 里提出来的吗

离线单测用的是我手写的 traceback 文本 (`COMPILE_FAIL_A`)。那份文本的形状是按读代码的结果
推测的 —— 而「签名提不提得出东西」完全取决于 error_log 的**真实**形状, 所以必须拿真机的
真失败去量, 否则就是在自己的想象上做特征工程。

第一版只测「核体里用了未定义的 `tl`」, 结果**一条失败都没采到**: Triton 的核体是懒编译的,
`SSHExecutor.compile()` 只做远端 import, 那个 NameError 要等到调用才炸。这条否定结果是
本脚本存在的理由之一 (先确认量具能看见你要看的东西)。

跑法 (需要能连到 GPU 机; 只跑 import / 一次 L2 比对, 秒级、不烧卡):
    python3 benchmarks/probe_failure_signature.py

⚠️ 本脚本**不打印 error_log 原文**: 远程 Traceback 必然带那台机器的工作目录与用户名。
签名刻意不含路径 (那正是它的设计), 所以打印签名安全, 打印原文不安全。
"""
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from operator_agent._harness_controls import enable_for_harness
enable_for_harness()

os.environ.setdefault("OPERATOR_AGENT_LOG_DIR", "off")
os.environ.setdefault("OPERATOR_AGENT_KERNEL_DB", "off")
os.environ.setdefault("OPERATOR_AGENT_FAILURE_DB", "off")

import numpy as np  # noqa: E402

from operator_agent.kernel_db import failure_signature  # noqa: E402
from operator_agent.remote_executor import SSHExecutor  # noqa: E402
from operator_agent.verifier import failure_pair  # noqa: E402

# 三条, 各自对应一条真实存在过的失败路径
CASES = [
    ("L1", "远端 import 就炸 (模块级 NameError)",
     'import triton\nimport triton.language as tl\n\nBAD = tl.undefined_symbol\n\n'
     'def kernel(A, B):\n    return A + B\n'),
    ("L1", "真语法错",
     'import triton\nimport triton.language as tl\n\ndef kernel(A, B:\n    return A + B\n'),
    ("L2", "核体用了未定义的 tl (懒编译, 只在调用时炸)",
     'import triton\nimport triton.language as _wrong_name\n\n'
     'def kernel(A, B):\n    return tl.sum(A + B, axis=0)\n'),
]

_PATHISH = re.compile(r"(/home/|/data|/mnt|/root|/Users/|C:\\Users)")


def main() -> int:
    ex = SSHExecutor()
    print(f"[probe] 连接 {ex.host} (别名由 ~/.ssh/config 解析)")
    if not ex.deploy():
        print("[probe] ❌ runner 部署失败")
        return 1

    inputs = {"A": np.ones((64, 64), dtype=np.float32),
              "B": np.ones((64, 64), dtype=np.float32)}
    rows = []
    for level, label, code in CASES:
        if level == "L1":
            try:
                ex.compile(code)
                print(f"[probe] ⚠️ 「{label}」竟然通过了 L1 —— 这一条采不到失败")
                continue
            except Exception as e:  # noqa: BLE001 - 要的就是异常文本
                head, full = failure_pair(e)
        else:
            res = ex.verify_function(code, inputs, "matmul", max_error=1e-2)
            if res.get("success"):
                print(f"[probe] ⚠️ 「{label}」竟然通过了 L2 —— 这一条采不到失败")
                continue
            head, full = failure_pair(res.get("error", ""))

        sig = failure_signature(level, full)
        print(f"\n[probe] {level} · {label}")
        print(f"  首行: {head[:90]}")
        print(f"  error_log {len(full)} 字符; 原文含机器路径: "
              f"{bool(_PATHISH.search(full))}  ← 所以签名里不许有它")
        print(f"  签名: {sig}")
        rows.append((level, label, sig, full))

    if not rows:
        print("\n[probe] ❌ 一条真失败都没采到")
        return 1

    unknown = [r for r in rows if "|unknown|" in r[2]]
    pathly = [r for r in rows if _PATHISH.search(r[2])]
    distinct = len({r[2] for r in rows})
    print("\n=== 结论 ===")
    print(f"  采到 {len(rows)} 条真机失败, {distinct} 个不同签名; "
          f"落进 unknown 桶 {len(unknown)} 条; 签名里带机器路径 {len(pathly)} 条")
    for level, label, sig, _ in rows:
        print(f"    [{level}] {label[:34]:36s} → {sig}")
    ok = not unknown and not pathly
    print("\n✅ 真机 error_log 上签名提得出具体类名, 且不含任何机器路径" if ok else
          "❌ 上面的数字才是本轮的真结果 —— 按 doc 45 第八节的改判条件处理")
    print("""
判据 (doc 45 第八节):
  1. 真机签名的类名 = unknown  → 关键词表 / 异常名正则对不上真实文本, 先修签名再谈门;
  2. 真机签名里出现路径        → 分组会被机器差异碎成 1:1, 必须在提取前剥掉;
  3. 三条里出现同签名          → 说明「不同形状的失败被归成一类」, 要检查是不是过粗;
  4. 复发门在真机上开过没有    → 本脚本不判 (要 ≥3 个不同任务), 看运行结尾的 [FailureDB] 那句。
""")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
