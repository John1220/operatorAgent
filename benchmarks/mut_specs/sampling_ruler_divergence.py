"""`tests/unit/test_sampling_ruler_divergence.py` 那张变异表的可执行形态 (#80 第 1 片)

与前几张同一条理由: 计数要连着产它的锚点一起存放 (doc 41 那个「13/13」里有两行锚点查不到)。

分母按 #83 分两类: S5 是**结构行** —— 把共享判定式换成自己写的 `alt_bad == 0` 时, 今天在
ρ=1.0 的表上与原文给同一个答案 (采样窗口里没有「部分坏点仍算过」那一档), 所以它只能被那条
AST 守卫杀掉, 不进行为分母。

跑: python3 -B -m benchmarks.mutation_calibrate --spec benchmarks/mut_specs/sampling_ruler_divergence.py
"""
import sys
from os.path import abspath, dirname

_ROOT = dirname(dirname(dirname(abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmarks.calibration.mutation_calibrate import Row  # noqa: E402

FILES = {
    "ver": "operator_agent/decide/verifier.py",
    "tol": "operator_agent/core/tolerance.py",
}

TARGETS = [
    "tests/unit/test_sampling_ruler_divergence.py",
    "tests/unit/test_gate_single_ruler.py",
    "tests/unit/test_cross_round_ruler.py",
    "tests/unit/test_reference_nonfinite.py",
    "tests/unit/test_l2_tolerance.py",
]

ROWS = [
    # ── 尺的可见性 (缺陷 2: 这条路一个尺键都不报) ──
    Row("S1 采样那条不发布自己用过的 atol (退回本片之前)", "ver",
        '                "atol": max_allowed,\n',
        '',
        "这一窗明明按 atol=1e-2 判过, 而 `l2_ruler_reading` 仍读成 unreported ⇒ "
        "「没测」与「测了但尺不同」再次被压成同一件事"),
    Row("S2 发布的 atol 用容差表值而不是这一窗真正用的值", "ver",
        '                "atol": max_allowed,',
        '                "atol": alt_atol,',
        "两条红各打各的方向: 那把尺的真实值 (1e-2) 被换成表的 1e-4, 于是混轮对照里两把尺"
        "看起来是同一把 —— 本片的整个前提被反转"),
    Row("S3 把缺席的 rtol 补成值 (造证据)", "ver",
        '                "and_form_atol": alt_atol, "and_form_rtol": alt_rtol,',
        '                "rtol": alt_rtol,\n'
        '                "and_form_atol": alt_atol, "and_form_rtol": alt_rtol,',
        "这条尺没有相对项, 却报出一个 rtol ⇒ 读的人以为它在用「绝对+相对」; "
        "`assertNotIn(\"rtol\")` 那条就是钉这个的 (#53/#81 那一族)"),
    # ── 第二个读数 (缺陷 1 的测量半边) ──
    Row("S4 第二个读数整个不算", "ver",
        '            alt_atol, alt_rtol, alt_rho = tolerance.effective_atol(\n'
        '                task.constraints.precision.value, max_allowed)\n',
        '',
        "`and_form_*` 全部缺席 ⇒ 分歧率测不出来, 换尺那一步永远缺数据; 顺带 `both` 那句报警"
        "失去依据 (NameError 也算红, 但归属仍是这一条)"),
    Row("S5 [结构] 判定式自己写, 不问那个模块", "ver",
        "            alt_passed, alt_matched = tolerance.evaluate(alt_bad, n_compared, False, alt_rho)",
        "            alt_passed, alt_matched = (alt_bad == 0), 1.0 - alt_bad / max(n_compared, 1)",
        "隔离理由: ρ=1.0 时两种写法对同一窗给出同一个答案 ⇒ 今天行为不可区分; 拦住它的是那条 "
        "AST 守卫 (采样的块里必须出现 `tolerance.evaluate(` 而不许出现 `matched_ratio >=`) —— "
        "把规则抄第二遍之后, 表一改两边就分家 (fixer_stage_branch 的 M16 同形状)",
        False),
    Row("S6 min() 翻成 max() (任务能放宽判据)", "tol",
        "        atol = min(atol, float(task_max_error))",
        "        atol = max(atol, float(task_max_error))",
        "缩放 0.997 那份候选在两个方向上同时变: `and_form_atol` 从 1e-4 变 1e-2 ⇒ 第二个读数"
        "判过 ⇒ 分歧报警消失 —— 「任务只能收紧」那半句是这条尺的承重点"),
    # ── 运行级那两句 (聚合的 message 是重算的, 每组自己那句会被丢掉) ──
    Row("S7 聚合里不再点名是哪几组用了松尺", "ver",
        '                msg += (f" ⚠️ {len(abs_only_sets)}/{len(sets_with_reference)} 组比的是采样窗口那把"',
        '                msg += (f" ⚠️ 0/{len(sets_with_reference)} 组比的是采样窗口那把"',
        "日志里那句报警永远说 0 组, 而 metrics 里的组号列表还在 ⇒ 读日志的人看不见, 读产物的人"
        "要自己数 (缺口 2 那族: 门在起作用但读不出来)"),
    Row("S8 分歧收集条件反过来", "ver",
        '                if m.get("and_form_passed") is False:',
        '                if m.get("and_form_passed"):',
        "两个方向各一条红: 真分歧那一组不再进名单 (键缺席), 而**一致**的那一组反倒被列成"
        "「会被拦」"),
    Row("S9 一致时也照样印分歧", "ver",
        '            both = ("两把尺判得一致" if alt_passed == passed_abs else',
        '            both = ("⚠️ 分歧" if True else "两把尺判得一致" if alt_passed == passed_abs else',
        "报警被稀释成通用文案 ⇒ 「按这把尺判过、按那条判不过」这个真读数与噪音分不开 "
        "(与 llm_sampling_knobs 的 L6 同族: 报警到处印就等于没报警)"),
    # ── 这一片的边界: 只观测, 不改判决 ──
    Row("S10 顺手把判决换成那把紧尺 (本片明说没做的事)", "ver",
        "            passed_abs = max_error <= max_allowed",
        "            passed_abs = max_error <= max_allowed and alt_bad == 0",
        "缩放 0.997 那份候选当场判不过 ⇒ 「判决未换尺, 这一句是观测」那条用例红。"
        "这一行存在的理由: 换尺要带真机 A/B (它只会收紧, 但会改通过率), 谁顺手做了就必须响"),
    Row("S11 聚合不再发布用了哪几组的名单", "ver",
        '                metrics["absolute_only_sets"] = abs_only_sets',
        '                pass',
        "`probe_l2_sampling_gate.branch_of` 在聚合这一层只认这个键 ⇒ 判别键退回「认不出哪条路」, "
        "而读产物的人拿不到组号 (逐组的 `and_form_atol` 不会穿过聚合)"),
]
