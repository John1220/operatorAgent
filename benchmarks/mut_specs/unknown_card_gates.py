# -*- coding: utf-8 -*-
"""一期 W3（未知卡迁移门 + 人时表 CLI 入口）的变异表。

跑法:
    python3 -B -m benchmarks.calibration.mutation_calibrate \
        --spec benchmarks/mut_specs/unknown_card_gates.py
    python3 -B -m benchmarks.calibration.mutation_calibrate --spec … --check-only

廿一行（行为 18 / 结构 3）。这一族要守的只有同一个形状：**把「没测」显示成「测过了」**。
门被关掉之后代码不会崩，只会更绿，所以每一行都必须有活用例把它拽红。

本表是**第一轮跑出来的产物**：初版六行里 M2、M3 曾 ESCAPED，暴露两件事——
用例没覆盖 G2 的来源核对，以及 G4 当时只要求「写了共享形态」就放行（继承值配一句说明就绿）。
先修门（G4 改为必须给 `source_uuid`）再补用例，才变成现在这七行。
``_missing()`` 那行是根：它一旦返回 pass，无读数的卡会得到五门全绿的表，
后面所有性能结论都建在没测过的地板上。

M10–M19 是 2026-11-01「补掉 ①②」那一轮加的（`gpu_info` 自描述 + 标定→门的接面）：
十行首跑全 KILLED（行为 16/16、结构 3/3），但**不是因为覆盖天生就在**——M16 需要的
`peak.source_uuid` 用例、M11/M12 需要的 `LocalExecutor` 申报用例都是先补后进表的。
三行结构分母（M17/M18/M19）单列出来，是因为被改的三个文件都不在测试里执行：
runner 跑在 GPU 机上，两个标定脚本 scp 单文件过去、没法 import 门的 schema。
它们只由「从写者源码里解析键名」的静态守卫拦得住，这不等于行为覆盖。
"""

from benchmarks.calibration.mutation_calibrate import Row

FILES = {
    "gates": "benchmarks/probes/unknown_card_gates.py",
    "cli": "operator_agent/app/cli.py",
    "col": "benchmarks/probes/collect_unknown_card_probe.py",
    "agents": "operator_agent/generate/agents.py",
    "runner": "remote_scripts/run_kernel.py",
    "floor": "benchmarks/measurement/measure_launch_floor.py",
    "peak": "benchmarks/calibration/calibrate_achievable_peak.py",
}

TARGETS = [
    "tests/unit/test_unknown_card_gates.py",
    "tests/unit/test_cli_ledger.py",
    "tests/unit/test_unknown_card_probe.py",
    "tests/unit/test_smem_budget.py",
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

    Row("M8 采集器不再拒绝缺失共享形态（给个默认值）", "col",
        "    if sharing not in SHARING_VALUES:",
        "    if False:",
        "存在具体输入可观察：`assemble(props, uuid, None, 0)` —— 原实现直接拒并回显三个合法值；"
        "关掉后 payload 带着空 `sharing_pattern` 出门，G4 于是可以在**没问过机器形态**的情况下"
        "被当成已标定。消费卡接显示器就是靠这一步溜过去的", True),

    Row("M9 两个共享内存上限都没读到还照装", "col",
        '    if props.get("per_block_optin_bytes") is None and props.get("per_sm_kb") is None:',
        "    if False:",
        "存在具体输入可观察：`per_block_optin_bytes=None` **且** `per_sm_kb=None` 的 props —— 原实现拒绝"
        "（这正是 G2 门在 4090 评估里点名的兜底病灶），关掉后两个 None 被写进表、门判 unknown 而不是"
        "「采集失败」，**没人知道自己其实没测过**。只暴露一个上限不在此列：那是 torch 版本的差别", True),

    # ---- 以下八行是「补掉 ①②」那一轮加的面（2026-11-01）----

    Row("M10 Designer 不再优先采用实测的共享内存", "agents",
        '    if report.get("shared_memory_per_sm_kb"):',
        "    if False:",
        "存在具体输入可观察：cc 8.9 + `hardware_report.shared_memory_per_sm_kb=100` —— 原实现给 "
        "(100, 1536, 32)，关掉后 8.9 不在回退表里 ⇒ 直接抛。后果是「runner 报了也没用」，"
        "Ada 这类卡继续按一张不存在的池子排 tile", True),

    Row("M11 探测不到时回到 128KB 兜底（病灶本体）", "agents",
        '    raise HardwareProbeFailed(\n        "designer.tile_budget",\n        "共享内存上限既没探测到（hardware_report.shared_memory_per_sm_kb 缺失）、"\n'
        '        "也不在已知 cc 常量表里（cc=%r）：拒绝按 128KB 猜" % (cc or "unknown",),\n'
        '        endpoint_hint="要么升级远端 runner 让它自描述硬件，要么在契约里显式声明硬件事实")',
        "    return 128, 2048, 32",
        "存在具体输入可观察：`{}` 与 `LocalExecutor().get_hardware_info()`（`memory.local.size` 是 "
        '"N/A"）—— 原实现抛，改成 128 后 CPU 回退借到一台不存在的 GPU 池子，而产物里长得像探测过。'
        "这一行就是 4090 评估 §6 第 2 步点名的那一处", True),

    Row("M12 legacy 字段解析失败时又猜一个 128", "agents",
        '        raw = str(legacy_local["size"]).replace("KB", "").replace("/SM", "").strip()\n'
        "        if not raw.isdigit():\n            raise HardwareProbeFailed(\n"
        '                "designer.tile_budget",\n'
        '                "legacy 上报的共享内存 %r 解析不出整数 KB：拒绝按 128KB 猜" % (legacy_local["size"],))',
        '        raw = str(legacy_local["size"]).replace("KB", "").replace("/SM", "").strip()\n'
        '        raw = raw if raw.isdigit() else "128"',
        "存在具体输入可观察：`{'memory': {'local': {'size': 'N/A'}}}`（正是 LocalExecutor 的申报）——"
        " 原实现抛 HardwareProbeFailed，关掉后它变成 128。兜底病灶原来就住在这个分支里", True),

    Row("M13 采集器不再核对实测段的 uuid 是否存在", "col",
        "        if not p_uuid:",
        "        if False:",
        "存在具体输入可观察：`assemble(props, uuid, 'exclusive', 0, [{'launch_floor': {...}}])`（没有 uuid）"
        " —— 原实现拒绝，关掉后来历不明的地板被盖上**本卡**的戳，G1 于是能在没测过的情况下判 pass", True),

    Row("M14 采集器允许把别的卡的读数当本卡标定", "col",
        "        if p_uuid != uuid:",
        "        if False:",
        "存在具体输入可观察：`uuid='GPU-someone-elses'` 的地板读数 —— 原实现拒绝并回显两个 uuid，"
        "关掉后 H20 的 18.69µs 变成 4070 的地板，小形状全体被当成可裁决读数", True),

    Row("M15 still_missing 按段名猜，noise 的阈值缺口永远不出现在表里", "col",
        '        if key == "noise.band/dispersion":',
        "        if False:",
        "存在具体输入可观察：`assemble(props, uuid, 'exclusive', 0)`（没有任何实测段）—— 原实现列全四个缺口，"
        "关掉后只剩三个：`noise` 整段是采集器自己填的，用 `split('.')` 判存在就等于宣称「散布带标定过了」", True),

    Row("M16 G3 只认 trusted 不看读数属于哪张卡", "gates",
        "    elif not trusted or src != uuid:",
        "    elif not trusted or False:",
        "存在具体输入可观察：`peak.source_uuid='H20-archive'` 且 `trusted=True` —— 原实现判 fail，"
        "关掉后判 pass。`_PEAK_BY_CC['8.9']` 按 L40S 填、对 4070 偏大，拿它当本卡分母等于**永不判停**", True),

    Row("M17 runner 改报另一个键名，Designer 读不到", "runner",
        '                    ("shared_memory_per_sm_kb", "shared_memory_per_multiprocessor", 1024.0),',
        '                    ("smem_per_sm_kb", "shared_memory_per_multiprocessor", 1024.0),',
        "今天行为层与原文不可区分（runner 只在 GPU 机上执行，测试里不跑它），拦它的是结构守卫："
        "`test_smem_budget` 从 `get_gpu_info` 的 AST 里取键名核对。改名后 Designer 侧的"
        "`hardware_report.shared_memory_per_sm_kb` 恒缺 ⇒ 所有走老链路的任务全部停在探测失败", False),

    Row("M18 地板脚本的尾巴改了键名（PROBE_JSON 与采集器脱钩）", "floor",
        '    "launch_floor": {"median_us": round(med, 3), "min_us": round(mn, 3), "samples": REP},',
        '    "launch_floor": {"median_us": round(med, 3), "min_us": round(mn, 3), "n": REP},',
        "同样不可在行为层区分：这两个脚本 scp 单文件到 GPU 机上跑，没法 import 门的 schema，"
        "所以键名漂了没人报错——采集器读不到 `samples` 就少盖一个戳、门判 unknown，而输出照旧打印。"
        "拦它的是静态文本守卫（test_smem_budget 的 TestEmitterWireFormat）", False),

    Row("M19 峰值脚本的尾巴改了键名", "peak",
        '        "peak_by_dtype": _peak,',
        '        "peak_tflops_by_dtype": _peak,',
        "同上：改名的后果不是崩，而是 G3 永远 unknown（或更糟——`assemble` 里的 `probe.get('peak_by_dtype')` "
        "取不到 ⇒ 不填 peak 段），`--from-probe` 明明读了文件却不报任何异常。拦它的是静态文本守卫", False),

    Row("M20 Designer 不认「只报了每块 opt-in」这一种申报", "agents",
        '    elif report.get("shared_memory_per_block_optin_bytes"):',
        "    elif False:",
        "存在具体输入可观察：`hardware_report` 只有 `shared_memory_per_block_optin_bytes=99*1024`、cc 8.9 "
        "—— 原实现给 99（偏小、保守），关掉后落到「既没探测到也不在 cc 表里」⇒ 抛，**4070 上开不了跑**。"
        "`shared_memory_per_multiprocessor` 不是每个 torch build 都暴露，这条是采集口径不是硬件事实", True),

    Row("M21 G2 在两个上限都是 None 时判 pass", "gates",
        "    if per_block is None and per_sm is None:",
        "    if False:",
        "存在具体输入可观察：`shared_memory = {per_block_optin_bytes: None, per_sm_kb: None, source_uuid: 本卡}` "
        "—— 原实现 unknown，关掉后判 pass：一个都没读到的池子被当成读到了，tile 预算按不存在的共享内存排。"
        "`None` 与 `0` 都不是读数，这条守的是「段在 ≠ 测过」", True),
]
