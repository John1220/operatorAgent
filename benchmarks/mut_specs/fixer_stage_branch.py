"""`tests/test_fixer_stage_branch.py` 那张变异表的**可执行**形态 (每行带 kill witness)

为什么要在仓里有一份 (而不是一段 /tmp 脚本): 本会话刚撞过一次 —— doc 41 里写的「13 行、13/13」
有两行的锚点在任何文件里都查不到 ⇒ 那个数字不可复现。计数要连着产它的锚点一起存放。

witness 一栏按 doc 57 §二 第 1 条 (arXiv:2609.22220v1 行 246-251) 分两类:
  · behavioral=True 的行 —— 「在哪个具体输入下它真的改变可观察结果」;
  · behavioral=False 的行 —— 今天行为层与原文不可区分, 拦住它的是结构守卫, 于是**不进行为分母**。
    这三行 (M11/M16/M21) 以前和新行为行混在同一个「21/21」里, 那句话把两种完全不同的覆盖
    说成了一种。

跑: python3 -B -m benchmarks.mutation_calibrate --spec benchmarks/mut_specs/fixer_stage_branch.py
"""
import sys
from os.path import abspath, dirname

_ROOT = dirname(dirname(dirname(abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmarks.mutation_calibrate import Row  # noqa: E402

FILES = {
    "fixer": "operator_agent/fixer.py",
    "tpl": "operator_agent/prompts/fixer_repair.md",
    "kdb": "operator_agent/kernel_db.py",
    "gate": "operator_agent/interface_contract.py",
    "cond": "operator_agent/conductor.py",
}

TARGETS = [
    "tests/test_fixer_stage_branch.py",
    "tests/test_prompt_placeholder_contract.py",
    "tests/test_fixer.py",
    "tests/test_failure_memory.py",
    "tests/test_interface_contract.py",
    "tests/test_kernel_db_persistence.py",
]

ROWS = [
    # ── 第 1 片: 两态 (退让 vs 保持策略) ──
    Row("M1 恒返回编译取向", "fixer",
        '    return _L2_RETREAT if "L2" in levels else _L1_INPLACE',
        '    return _L1_INPLACE',
        "一条 L2 数值失败送进 run() ⇒ 取向段不再说「数值检查器/允许变慢」"),
    Row("M2 恒返回数值取向", "fixer",
        '    return _L2_RETREAT if "L2" in levels else _L1_INPLACE',
        '    return _L2_RETREAT',
        "一条 L1 编译失败送进 run() ⇒ 取向段不再许可「保持既有策略就地修」"),
    Row("M3 分支条件改成 L3", "fixer",
        '    return _L2_RETREAT if "L2" in levels else _L1_INPLACE',
        '    return _L2_RETREAT if "L3" in levels else _L1_INPLACE',
        "只有 L2 在报失败的输入 ⇒ 不再触发退让；而 L3 单独失败会触发 (外部 §4.3.1 说的正是反的)"),
    Row("M4 模板删掉取向占位符", "tpl", "{{stage_directive}}\n", "",
        "任何一次 run() 渲染出的 prompt 里都没有取向段 (值算了没人读, #67 那一族)"),
    Row("M5 模板把取向写成死文字", "tpl", "{{stage_directive}}", "请按最小改动修复。",
        "换输入级别 (L1 vs L2) 后 prompt 不再变 —— 分支失去读者"),
    Row("M6 模板恢复被合写的旧句子", "tpl",
        "以下代码未通过验证。本轮报失败的阶段: {{failure_stage}}",
        "以下代码在编译 (L1) 或功能 (L2) 验证中失败",
        "L1+L3 与 L2 两类输入渲染出同一句话 ⇒ 阶段上报消失"),
    Row("M7 调用处把级别写死", "fixer",
        'stage_directive=stage_directive(level_set, failure_cls,',
        'stage_directive=stage_directive({"L1"}, failure_cls,',
        "L2 失败送进 run() ⇒ 拿到的是编译取向"),
    Row("M8 删掉 L2 取向的优先级末句", "fixer",
        '    "若错误信息里同时出现性能未达标 —— 本轮**不要**为此优化。"',
        '    ""',
        "L2+L3 同框那份输入 (今天不可达, 由 fail-fast 的 AST 守卫钉着) 不再被禁止为性能优化"),
    Row("M9a 上下文检索里滤掉 L3", "fixer",
        '        error_matches: list = []',
        '        base = [r for r in base if r.level != "L3"]\n'
        '        error_matches: list = []',
        "库里有一条 L3 记录时, 历史栏不再出现它 ⇒ 末句失去对象"),
    Row("M9b 合并处滤掉 L3", "fixer",
        '            merged.append(r)',
        '            if r.level != "L3":\n                merged.append(r)',
        "同上, 但两条检索通道一起堵 ⇒ 证明「两条通道各自都算数」"),
    # ── 结构类: 今天行为不可区分, 只有结构守卫杀得掉 ──
    # Stage2 把轮次循环体整段搬进 _run_round (少一层缩进), 这条锚点跟着少 4 格 ——
    # 判据没变, 变的是它盯的那行代码现在住在哪儿。
    Row("M11 把 Fixer 入口那道门放宽到 L3", "cond",
        '            not r.passed and r.level in (VerificationLevel.L1_COMPILE, VerificationLevel.L2_FUNCTION)',
        '            not r.passed and r.level in (VerificationLevel.L1_COMPILE,\n'
        '                VerificationLevel.L2_FUNCTION, VerificationLevel.L3_PERFORMANCE)',
        "隔离理由: `verify()` fail-fast ⇒ 现役输入里构造不出「L3 单独失败还进了 Fixer」的活路径; "
        "杀它的是对 `l1_l2_failed` 那道门的 AST 断言 (它钉的是可达性, 不是当次输出)",
        False),
    # ── 第 2 片: 第三路 interface repair 的整条链 ──
    Row("M12 归类取整条签名 (本片原始 bug)", "fixer",
        '    return failure_class_of(levels, text)',
        '    return failure_signature(levels, text)',
        "门自己的错误原文送进 run() ⇒ 第三路永不触发 (取向非空、模板有读者, 所以只有行为对照能看见)"),
    Row("M13 调用处漏传归类", "fixer",
        'stage_directive=stage_directive(level_set, failure_cls,',
        'stage_directive=stage_directive(level_set,',
        "同上: 接口契约那条失败拿到的是通用 L1 取向"),
    Row("M14 第三路分支条件恒假", "fixer",
        '    if failure_cls == FAILURE_CLASS_INTERFACE:',
        '    if failure_cls == FAILURE_CLASS_INTERFACE and False:',
        "接口契约失败送进 run() ⇒ 走不到第三路"),
    Row("M15 第三路分支条件恒真", "fixer",
        '    if failure_cls == FAILURE_CLASS_INTERFACE:',
        '    if failure_cls is not None:',
        "普通 L1 编译失败与 L2 数值失败各自送进 run() ⇒ 都拿到接口取向 (互斥被打破)"),
    Row("M16 取向自己判文案、不走分类器", "fixer",
        '    return failure_class_of(levels, text)',
        '    return FAILURE_CLASS_INTERFACE if "签名" in text else "unknown"',
        "隔离理由: 今天两条通道对同一份门文本给出同一个桶名 ⇒ 没有任何输入能观察到差异; "
        "杀它的是「fixer 里不许有桶名的第二份字面量 / failure_class 必须调用那个分类器」的 AST 守卫 "
        "(词表一改两者就分家, 而那正是分家第一次发生前唯一能响的警)",
        False),
    Row("M17 failure_class_of 取错段", "kdb",
        '    return failure_signature(level, error_text).split("|", 2)[1]',
        '    return failure_signature(level, error_text).split("|", 2)[0]',
        "接口契约失败 ⇒ 分流键变成层级字符串, 第三路不触发"),
    Row("M18 归类词表里那条 marker 删掉", "kdb",
        '    ("签名与之对不上", FAILURE_CLASS_INTERFACE),\n',
        '',
        "门文本送进分类器 ⇒ 归成 TypeError (文本里那句举例外), 第三路与复发门的桶一起变"),
    Row("M19 门改文案一个字", "gate",
        '"kernel() 签名与之对不上 —— 真跑起来必然抛 "',
        '"kernel() 签名与它对不上 —— 真跑起来必然抛 "',
        "真门产出的文本不再命中词表 ⇒ 第三路哑掉 (这条同时证明词表与文案是两份字面量)"),
    Row("M20 删掉第三路那句改动范围锁", "fixer",
        '"取向: **本轮只改 `def kernel` 那一行的参数名**。',
        '"取向: ',
        "接口契约失败送进 run() ⇒ 取向段里不再有「只改那一行」的范围锁 —— "
        "第一版这条**逃逸**了, 因为那条用例只朝一个方向查缺席 (CHECKLOG I19)"),
    Row("M21 只改桶名常量值", "kdb",
        'FAILURE_CLASS_INTERFACE = "interface_mismatch"',
        'FAILURE_CLASS_INTERFACE = "interface_contract"',
        "隔离理由: 生产代码里分流键是单点, 改名两侧一起动 ⇒ 今天行为不可区分; 杀它的是那条把 "
        "`L1|interface_mismatch|-` 整个钉成期望值的用例 —— 桶名已写进已入库记录 id "
        "(`cx-{task}:{signature}`), 改名会让历史与新归类分家 (CHECKLOG I20)",
        False),
]
