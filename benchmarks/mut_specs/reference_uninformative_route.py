"""`tests/unit/test_reference_uninformative_route.py` 那张变异表的**可执行**形态 (#94 第 2 片)

与另两张 (`fixer_stage_branch.py` / `llm_sampling_knobs.py`) 同一个理由: 计数要连着产它的
锚点一起存放, 否则文档里那个「N/N」不可复现 (doc 41 的「13 行、13/13」有两行锚点查不到)。

witness 一栏按 doc 57 §二 第 1 条 (arXiv:2609.22220v1 行 246-251) 分两类:
  · behavioral=True —— 存在一个具体输入让这次改动能被观察到;
  · behavioral=False —— 今天在行为层与原文不可区分, 拦住它的是结构守卫, **不进行为分母**。
    本片有 M12 (只改桶名的值) 与 M13 (写库侧换成自己正则) 两行是这一类。
      M12 的先例是 `fixer_stage_branch.py` 的 M21 —— 桶名会被写进已入库记录 id
      (`cx-{task_id}:{signature}`), 「收成单点」只覆盖产它的那一半代码, 名字一旦被写进
      别的事实就不再是自由重构 (CHECKLOG I19/I20 那两次「只数产者、没数读者」)。

跑: python3 -B -m benchmarks.mutation_calibrate --spec benchmarks/mut_specs/reference_uninformative_route.py

⚠️ 这张表的靶子是**判决侧接上了没有**, 不是「坏参考被修好了」。后者是 #95 (任务准入要过
参考解自证可达那道门), 不在这一片。
"""
import sys
from os.path import abspath, dirname

_ROOT = dirname(dirname(dirname(abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmarks.calibration.mutation_calibrate import Row  # noqa: E402

FILES = {
    "kdb": "operator_agent/core/kernel_db.py",
    "fix": "operator_agent/generate/fixer.py",
    "cond": "operator_agent/app/conductor.py",
    "ver": "operator_agent/decide/verifier.py",
}

TARGETS = [
    "tests/unit/test_reference_uninformative_route.py",
    "tests/unit/test_fixer_stage_branch.py",
    "tests/unit/test_failure_memory.py",
    "tests/unit/test_reference_nonfinite.py",
    "tests/unit/test_kernel_db_persistence.py",
]

ROWS = [
    # ── 第 1 层: 分类器认得出那道门产的那句话 (两份字面量) ──
    Row("M1 词表里那条 marker 删掉", "kdb",
        '    ("本级判不出", FAILURE_CLASS_REFERENCE),\n',
        '',
        "门原文送进分类器 ⇒ 归成 unknown, 第四路与那两条写库守卫全部不触发"),
    Row("M2 门改措辞 (词表与文案脱钩)", "ver",
        'f"本级判不出, 这不是候选错; 该查的是 reference_source',
        'f"本级没有可比对象, 这不是候选错; 该查的是 reference_source',
        "真门产出的文本不再命中词表 ⇒ 夹具那条 `test_the_gate_still_owns_the_producer_side` "
        "红, 且整条分流哑掉 (与 fixer_stage_branch 的 M19 同一条纪律的另一侧)"),
    # ── 第 2 层: 取向分得到这一类 ──
    Row("M3 分流条件恒假", "fix",
        "    if failure_cls == FAILURE_CLASS_REFERENCE:",
        "    if failure_cls == FAILURE_CLASS_REFERENCE and False:",
        "坏参考那一轮拿回 `_L2_RETREAT` ⇒ prompt 里出现「correctness 优先于性能」"),
    Row("M4 分流条件恒真 (一切失败都算判不出)", "fix",
        "    if failure_cls == FAILURE_CLASS_REFERENCE:",
        "    if failure_cls == FAILURE_CLASS_REFERENCE or True:",
        "一条真的超容差失败也拿到第四路 ⇒ 本文件与 fixer_stage_branch 的数值侧对照全体红"),
    Row("M5 第四路整条不存在", "fix",
        "    if failure_cls == FAILURE_CLASS_REFERENCE:\n        return _REFERENCE_HOLD\n",
        '',
        "归类不再参与取向 ⇒ 坏参考那一轮按级别拿到 `_L2_RETREAT` (第 1 片之前的形状)"),
    # ── 第 3 层: 错误段不再给处方 ──
    Row("M6 hold 轮仍然印归因前言", "fix",
        "        head = [_ATTRIBUTION] if not hold else []",
        "        head = [_ATTRIBUTION]",
        "坏参考那一轮的 prompt 里出现「实测到的现象」—— 那是一句对没有发生的判定的断言"),
    Row("M7 hold 轮仍然好处方", "fix",
        "                if hold:\n                    lines.append(_HOLD_NOTE)\n"
        "                else:\n                    lines.extend(self._prescribe_lines(r))",
        "                lines.extend(self._prescribe_lines(r))",
        "坏参考那一轮拿到「要求 / 该改哪里」(或那句 «只依据执行侧原话定位根因» 的兜底) ⇒ "
        "一份指向错误对象的处方又回来了; 顺带把 `_HOLD_NOTE` 变成没人读的常量"),
    Row("M8 无 hold 的轮次也印 hold 注解", "fix",
        "                if hold:\n                    lines.append(_HOLD_NOTE)",
        "                if True:\n                    lines.append(_HOLD_NOTE)",
        "数值那一轮的 prompt 里同时出现「本轮不给处方」⇒ 那条反向对照红, 且「本轮没有判定」"
        "被稀释成一句通用文案 (抑制条件其实没在识别这一类)"),
    # ── 第 4 层: 语料不写 + 挡掉的量看得见 ──
    Row("M9 `_record_failure` 那道门恒假", "cond",
        "        if failure_class(failed_results) == FAILURE_CLASS_REFERENCE:",
        "        if False and failure_class(failed_results) == FAILURE_CLASS_REFERENCE:",
        "坏参考那一轮的兜底入库照常发生 ⇒ `db._records` 非空、计数不涨、结尾那行报 0 轮"),
    Row("M10 循环里那次 Reflexion 不再被挡", "cond",
        "                        if not reference_hold:",
        "                        if True:",
        "坏参考那一轮让模型编一份候选侧根因并 `add()` 进语料 ⇒ `reflexion_calls` 由 0 变 1、"
        "库里多一条假教训, 而后续轮次会把它当「同类错误的修复经验」检索回 prompt"),
    Row("M11 结尾那行账把轮数写死成 0", "cond",
        'f"·参考侧判不出 {self._reference_hold_rounds} 轮 (没入库)")',
        'f"·参考侧判不出 0 轮 (没入库)")',
        "坏参考那一轮的日志说「0 轮」而实际挡掉过 ⇒ 门在起作用但读不出来 (缺口 2 那族: "
        "看不见就等于没生效); 这条也是那两句断言为什么要打在**紧邻的轮数**上"),
    Row("M12 排除集合退回只挡 infra", "kdb",
        '_NOT_KERNEL_CLASSES = ("infra_no_baseline", FAILURE_CLASS_REFERENCE)',
        '_NOT_KERNEL_CLASSES = ("infra_no_baseline",)',
        "从文件重放回来的坏参考记录重新变成可注入的教训 (`usable()` 那条类别门放行)"),
    Row("M13 只改桶名的值 (改名)", "kdb",
        'FAILURE_CLASS_REFERENCE = "reference_uninformative"',
        'FAILURE_CLASS_REFERENCE = "reference_no_object"',
        "隔离理由: 分流键与签名都读同一个常量 ⇒ 今天行为不可区分; 杀它的是把 "
        "`L2|reference_uninformative|-` 整串钉成期望值那条 —— 桶名会被写进已入库记录 id "
        "(`cx-{task_id}:{signature}`), 改名会让历史与新归类分家 (CHECKLOG I19/I20 同族)",
        False),
    Row("M14 [结构] 写库侧换成自己正则看文案", "cond",
        "        if failure_class(failed_results) == FAILURE_CLASS_REFERENCE:",
        '        if "本级判不出" in str(failed_results):',
        "隔离理由: 对同一份门文本两种口径今天给出同一个答案 ⇒ 无输入可观察差异; 拦住它的是 "
        "`TestTheWiringIsStructural` 那条 AST 守卫 (词表一改两者就分家, 而那正是分家第一次 "
        "发生前唯一能响的警 —— fixer_stage_branch 的 M16 同一个形状)",
        False),
]
