"""§3.5「长期记忆的独立确认」不变量那张变异表的**可执行**形态

靶子是 `tests/unit/test_spine_wiring.py::TestTheDoubleConfirmationIsAnInvariant`（四条行为守卫
+ 一条源码守卫）与 `tests/unit/test_kernel_db_persistence.py`（那条门的结果决定落盘的线）。

为什么单独一张表: 这一片修的是「**默认关的安全装置等于不存在**」那类缺陷 (CHECKLOG I79)，
而这类缺陷的特点是**守卫会把错的行为钉成正确行为** —— 翻转前有三条用例正在断言
「默认不强制两次确认」。所以这张表要能同时证明两件事:
  · 不变量真的在链路上（把生产者挂回开关后面 / 判据侧关掉 ⇒ 必红）;
  · 它真的**没有开关**（把构造参数加回来 / 加一行环境变量 ⇒ 必红, 后者只有源码守卫杀得掉）。

witness 一栏按 doc 57 §二 第 1 条分两类:
  · behavioral=True —— 有**具体输入或条件**让这次改动能被观察到;
  · behavioral=False —— 今天在行为层与原文不可区分（没有人会去设那个环境变量），
    拦住它的只可能是源码守卫。

⚠️ `requires_remeasure` 这个判据参数**不是**产品开关: 它只服务 §14-9 的消融臂, docstring
明文"不许接到配置文件/环境变量/命令行/任务字段上"。所以 M2 打的是**生产调用点**, 不是
判据的默认值 —— 默认值那一栏今天在生产路径上根本走不到 (恒传 True)。

跑: python3 -B -m benchmarks.mutation_calibrate --spec benchmarks/mut_specs/double_confirm_invariant.py
"""
import sys
from os.path import abspath, dirname

_ROOT = dirname(dirname(dirname(abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmarks.calibration.mutation_calibrate import Row  # noqa: E402

FILES = {
    "cond": "operator_agent/conductor.py",
    "confirm": "operator_agent/confirm.py",
}

TARGETS = [
    "tests/unit/test_spine_wiring.py",
    "tests/unit/test_kernel_db_persistence.py",
]

ROWS = [
    Row("M1 生产者挂回开关后面 (默认关 ⇒ 重测从不发生)", "cond",
        "            self._produce_remeasure(artifact, task, test_inputs, target_gflops)",
        "            if getattr(self, \"require_double_confirm\", False):\n"
        "                self._produce_remeasure(artifact, task, test_inputs, target_gflops)",
        "**具体输入**: 默认构造的 Conductor 跑一次全程 (`LaunchDominatedGPU` + "
        "min_speedup=0.001) ⇒ `getattr` 回落 False ⇒ 产物里没有重测记录 "
        "⇒ `test_the_producer_runs_on_the_normal_path` 红 (`art.remeasure is None`), "
        "而它正是 I79 那个洞的原形"),
    Row("M2 生产调用点把第 3 道门关掉", "cond",
        "            requires_remeasure=True,",
        "            requires_remeasure=False,",
        "**具体输入**: 任意一次 `_save_to_db` ⇒ 判据收到的 `requires_remeasure` 是 False "
        "⇒ `test_the_double_confirmation_is_passed_as_an_invariant` 红 ([False, False] "
        "≠ [True, True]); 这也是「安全语义不许留开关」那条守卫的靶子"),
    Row("M3 把开关加回构造参数 (默认关)", "cond",
        "        run_root: str = \"\",",
        "        require_double_confirm: bool = False,\n        run_root: str = \"\",",
        "**具体输入**: `Conductor(require_double_confirm=False)` ⇒ 不再抛 TypeError "
        "⇒ `test_there_is_no_switch_to_turn_it_off` 红 (「没有开关」那半条)。"
        "⚠️ 只加参数不加消费者也照样红 —— 守卫判的是 **API 面上有没有这个入口**"),
    Row("M4 「没有重测就当过了」(判据反向)", "confirm",
        "    if remeasure is None:\n"
        "        return False, \"没有独立重测: 永久结论需要两次确认 (设计 §10.5-4)\"",
        "    if remeasure is None:\n"
        "        return True, \"没有独立重测也放行（变异）\"",
        "**具体输入**: 一份过了前两道门、但没有 `remeasure` 的产物 ⇒ "
        "`test_it_refuses_without_a_remeasure` 红; 同一条也把 "
        "`test_constructor_without_remeasure_refuses` 拉下水"),
    Row("M5 坏窗口里的重测也当有效确认", "cond",
        "        usable = bool(second is not None and not obs2.get(\"refused\")",
        "        usable = True or bool(second is not None and not obs2.get(\"refused\")",
        "**具体输入**: `_PinnedTimerGPU` 把参照核计时钉成坏值 ⇒ 重测落在坏窗口里, "
        "`usable` 却为真 ⇒ `test_a_remeasure_in_a_broken_window_cannot_confirm` 红 "
        "(「决定永久记忆的那一次测量不许是唯一没被检查过的那一次」)"),
    Row("M6 给不变量加一行环境变量开关", "cond",
        "        from .confirm import ConfirmRecord",
        "        import os\n"
        "        if os.environ.get(\"OPERATOR_AGENT_DOUBLE_CONFIRM\", \"1\") == \"0\":\n"
        "            return\n"
        "        from .confirm import ConfirmRecord",
        "**具体输入**: 没有 —— 今天没有任何用例会去设这个变量, 所以行为层与原文**不可区分** "
        "(§3.5 的字面是「无环境变量」, 而它的失效方式恰恰是静默的) ⇒ 只有 "
        "`test_no_environment_variable_or_cli_flag_can_turn_it_off` 这条源码守卫杀得掉它",
        behavioral=False),
]