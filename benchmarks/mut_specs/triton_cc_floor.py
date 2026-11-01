# -*- coding: utf-8 -*-
"""CC 下限预检（`require_triton_support` + Conductor 接线）的变异表。

跑法:
    python3 -B -m benchmarks.calibration.mutation_calibrate \
        --spec benchmarks/mut_specs/triton_cc_floor.py
    python3 -B -m benchmarks.calibration.mutation_calibrate --spec … --check-only

这一族守的形状和前几张表不同：**不是「把没测显示成测过」，而是「把机器的缺陷记成候选的缺陷」**。
一台编不出码的卡（P4 / P100 / TITAN X）跑完一轮，交出的是「这些候选全错」，
而真因写在 Triton 的那句报错里：`Triton only supports devices of CUDA Capability >= 7.0`。

**已知没有进表的一条**：Conductor 里 `cc_verdict == "sunset"` 那个分支只 `print` 一行提示。
把它改坏（比如改成 raise）今天没有任何用例会红——那是一段没有活入口的文字。
按「报缺陷前先证明有活入口」的纪律，这里**不假装覆盖**，只在 runbook 里把它记成
「提示性输出，判据不依赖它」；真要钉，得先给 7.x 一条端到端用例。
"""

from benchmarks.calibration.mutation_calibrate import Row

FILES = {
    "caps": "operator_agent/core/hardware_caps.py",
    "conductor": "operator_agent/app/conductor.py",
}

TARGETS = [
    "tests/unit/test_triton_cc_floor.py",
    "tests/unit/test_hardware_probe_honesty.py",
]

ROWS = [
    Row("M1 整道预检关掉（低于下限也放行）", "caps",
        '    if verdict == "below":',
        "    if False:",
        "存在具体输入可观察：`{'name': 'Tesla P4', 'compute_capability': '6.1'}` —— 原实现抛 "
        "`TritonUnsupported` 并在端到端那条用例里证明一次 LLM 都没发；关掉后它变成「正常开跑」，"
        "于是一整轮的钱照烧、读数全记成候选编译失败", True),

    Row("M2 把下限写成 0（门在清单上恒绿）", "caps",
        "TRITON_MIN_CC_MAJOR = 7",
        "TRITON_MIN_CC_MAJOR = 0",
        "存在具体输入可观察：同一份 cc 6.1 —— 原实现判 below，改成 0 后它落进 `sunset`（major<=7）"
        "⇒ 不抛。**这道门从此对任何卡都不响**，而它自己的用例集里那一族「below」断言正是唯一读者；"
        "按具体值列的门换一台机器就恒绿，这条是它的形状", True),

    Row("M3 「解析不出 cc」被当成低于下限（新的静默降级）", "caps",
        '    if not head.isdigit():\n        return "unknown", text',
        '    if not head.isdigit():\n        return "below", text',
        "存在具体输入可观察：`None` / `\"\"` / `\"gfx90a\"`，以及 `MockGPU().get_hardware_info()` 与 "
        "`LocalExecutor().get_hardware_info()`（两者都没有 cc）—— 原实现返回 unknown 并放行，"
        "改掉后**无 GPU 的开发路径与 CPU 回退一起被判死**。旧写法 `_nvidia_cc_major` 正是把解析失败"
        "折成 0，这条防的是同一个错被搬进新门", True),

    Row("M4 摘掉 Conductor 里的接线（门在但没人调）", "conductor",
        "        cc_verdict, cc_text = require_triton_support(hw_info, self.backend.label)",
        '        cc_verdict, cc_text = "ok", ""',
        "端到端可观察：一台 P4 走完 `Conductor._execute_inner` —— 原实现抛且 `llm.calls == 0`，"
        "摘掉接线后纯函数用例**照样全绿**（它们测的是没人调的那个函数）。这正是「静态差集拦不住"
        "写了走不到」那一族，所以配套有 AST 守卫与活性证明两条", True),
]
