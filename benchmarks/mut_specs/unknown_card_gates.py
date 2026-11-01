# -*- coding: utf-8 -*-
"""一期 W3（未知卡迁移门 + 人时表 CLI 入口）的变异表。

跑法:
    python3 -B -m benchmarks.calibration.mutation_calibrate \
        --spec benchmarks/mut_specs/unknown_card_gates.py
    python3 -B -m benchmarks.calibration.mutation_calibrate --spec … --check-only

七行全在行为分母。这一族要守的只有同一个形状：**把「没测」显示成「测过了」**。
门被关掉之后代码不会崩，只会更绿，所以每一行都必须有活用例把它拽红。

本表是**第一轮跑出来的产物**：初版六行里 M2、M3 曾 ESCAPED，暴露两件事——
用例没覆盖 G2 的来源核对，以及 G4 当时只要求「写了共享形态」就放行（继承值配一句说明就绿）。
先修门（G4 改为必须给 `source_uuid`）再补用例，才变成现在这七行。
``_missing()`` 那行是根：它一旦返回 pass，无读数的卡会得到五门全绿的表，
后面所有性能结论都建在没测过的地板上。
"""

from benchmarks.calibration.mutation_calibrate import Row

FILES = {
    "gates": "benchmarks/probes/unknown_card_gates.py",
    "cli": "operator_agent/app/cli.py",
}

TARGETS = [
    "tests/unit/test_unknown_card_gates.py",
    "tests/unit/test_cli_ledger.py",
]

ROWS = [
    Row("M1 G1 关掉「地板来源必须是本卡」", "gates",
        "    elif src != uuid:\n        out[\"G1_launch_floor\"] = _fail(floor,",
        "    elif False:\n        out[\"G1_launch_floor\"] = _fail(floor,",
        "一份 `launch_floor.source_uuid = 'H20-archive'` 的探测输出：原实现判 fail，关掉后落到 else 判 pass。后果是未标地板的卡把小形状读数当可裁决读数，而 `measurement.assess()` 缺 floor 时本来就给 VALID（方向性变松）", True),

    Row("M2 G2 关掉「共享内存必须由本卡读回」", "gates",
        "    elif src != uuid:\n        out[\"G2_shared_memory\"] = _fail(per_block,",
        "    elif False:\n        out[\"G2_shared_memory\"] = _fail(per_block,",
        "`shared_memory.source_uuid` 指向别的卡：关掉后继承上限直通 pass，Designer 的 tile 预算按偏大的池子排，代价是多烧几轮 L1（正是 4090 评估里 128 KiB 那个坑的形状）", True),

    Row("M3 G4 关掉「阈值来源必须是本卡」", "gates",
        "    elif src != uuid:\n        out[\"G4_noise_bands\"] = _fail({\"band\": band,",
        "    elif False:\n        out[\"G4_noise_bands\"] = _fail({\"band\": band,",
        "`noise.source_uuid = 'H20-archive'` 且值是自定义的：关掉后只剩「像不像仓内值」一道筛，自定义的外来数值直接算本卡标定", True),

    Row("M4 G4 关掉「与仓内标定值逐位相同即视为抄」", "gates",
        "    elif inherited_hits:",
        "    elif False:",
        "声称本卡测的、却与仓内 H20 八卡共享机的 `NOISE_BAND`/`CONTAMINATION_UTIL_PCT`/`DISPERSION_DEGRADED` 逐位相同：关掉后被当成本卡标定放行", True),

    Row("M5 G5 关掉显存池边界检查", "gates",
        "        too_big = [s.get(\"label\") for s in g5 if isinstance(s.get(\"bytes\"), int) and s[\"bytes\"] > 0.8 * pool]",
        "        too_big = []",
        "20 GiB 的形状进 16 GiB 的池：原实现判 fail（96 GB 级读数不构成新卡可执行条件），关掉后它成为本卡验收样本，形状集直接继承自另一台机器", True),

    Row("M6 无读数时 `_missing()` 返回 pass（门的根）", "gates",
        "def _missing(why):\n    return {\"verdict\": \"unknown\",",
        "def _missing(why):\n    return {\"verdict\": \"pass\",",
        "`assess({'device': {'uuid': 'GPU-mine'}})`（只有设备、没有读数）：原实现五门全 unknown、`all_green=False`；改成 pass 后这张表直接全绿，runbook 第 0 步形同不存在", True),

    Row("M7 `ledger-new` 允许覆盖已填写的表", "cli",
        "        if path.exists():",
        "        if False:",
        "先 new、填一格 analyze=yes、再 new 一次：原实现拒绝且已答格子原样还在；关掉后表被重置成"
        "全 unknown，「问过并答了」在报表里退化成「没问过」，分母看着干净其实是空的", True),
]
