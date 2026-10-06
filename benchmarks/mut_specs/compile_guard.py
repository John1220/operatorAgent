"""§22.2 第一项「编译超时」那张变异表的**可执行**形态（靶子：`tests/test_exec_compile_guard.py`）

这一片的目标只有一句话：**编译期那一步在受控子进程里跑，且阶段级上限真的到了那里**。
所以每一行都要能回答「去掉它，那件事还是真的吗」：

  · 编译档没有自己的默认值（M1）⇒ 子进程自己的 import 开销会被记成候选编译超时；
  · 编译模式顺手把核也跑了（M2）/ 子进程的读数被盾覆盖（M3）⇒ 编译的产物不再是编译的产物；
  · 异常跨进程重建时丢掉字段（M4）⇒ 行号列号静默消失（`str()` 一样，所以不会有别的证人）；
  · 把「执行器拦下」并进「候选自己炸了」（M5）⇒ 超时被读成候选的语法错（I65/I11 那一族）；
  · 后端 / Verifier / 签名表 / 语料探针 / Conductor 任一处断开（M6–M11）⇒ 链条或产物少一段；
  · 编译路径里塞回进程内 `exec`（M12/M13）⇒ 只能靠**结构证人**：那两行是死代码，
    行为层与原文不可区分。

⚠️ 为什么 M12/M13 写成「死代码」而不是直接把 `run_compile` 换回 `exec`：后者的行为证人是
「悬挂候选被墙钟拦下」，而变异之后那份用例会**永远挂住**（校准器卡在那里，不是变红）。
「编译又回到进程内」这一条因此由 AST 断言与那份「只减不增」的账本承担。
"""

import sys
from os.path import abspath, dirname

_ROOT = dirname(dirname(dirname(abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmarks.mutation_calibrate import Row  # noqa: E402

FILES = {
    "guard": "operator_agent/exec_guard.py",
    "gpu": "operator_agent/mock_gpu.py",
    "local": "operator_agent/remote_executor.py",
    "ver": "operator_agent/verifier.py",
    "kdb": "operator_agent/kernel_db.py",
    "cond": "operator_agent/conductor.py",
    "vch": "benchmarks/verify_corpus_honesty.py",
}

TARGETS = [
    "tests/test_exec_compile_guard.py",
    "tests/test_exec_guard.py",
]

ROWS = [
    # ── 上限：编译那一档必须有自己的数 ──
    Row("M1 编译档的默认值退回运行档", "guard",
        "COMPILE_LIMITS = Limits(wall_ms=30000, cpu_s=60)",
        "COMPILE_LIMITS = Limits(wall_ms=5000, cpu_s=10)",
        "**具体输入**: 两个模式的默认墙钟相等 ⇒ "
        "`TestCompileMode::test_one_default_per_mode` 的 assertNotEqual 红 "
        "(子进程自己 import 的开销会被记成候选编译超时)"),

    # ── 编译模式：只 exec，不跑核 ──
    Row("M2 编译模式顺手把核也跑了", "guard",
        '                return _compile_only()',
        '                return _exec_and_call()',
        "**具体输入**: `TestCompileMode::test_compile_mode_does_not_call_the_kernel` 的代码在 "
        "`kernel` 里 raise ⇒ run_compile 变成 crashed（应当是 ok）"),
    Row("M3 子进程的读数被盾的读数覆盖", "guard",
        '        result["metrics"] = {**(result.get("metrics") or {}), **lib_fallback_metrics()}',
        '        result["metrics"] = lib_fallback_metrics()',
        "**具体输入**: 编译成功那一刻 `has_kernel` / `kernel_callable` 被清空 ⇒ "
        "`test_compile_mode_does_not_call_the_kernel` 与 "
        "`test_missing_kernel_is_reported_not_invented` 红"),

    # ── 异常跨进程重建：类型与字段一起活下来 ──
    Row("M4 语法错的字段不带回来", "guard",
        '            result["syntax_fields"] = {',
        '            result["syntax_fields_unused"] = {',
        "**具体输入**: 一份语法错候选 ⇒ `candidate_error` 重建出的 SyntaxError 没有 "
        "lineno/offset（`str()` 仍然一样，只有逐字段比对的用例会红）⇒ "
        "`test_the_candidate_exception_type_survives_the_boundary` 红"),
    Row("M5 执行器拦下被并进候选自己炸了", "gpu",
        "        if guard.status == STATUS_CRASHED:",
        "        if True:",
        "**具体输入**: 悬挂候选 + 400ms 墙钟 ⇒ 超时被重建成候选异常（不再是 GuardRefusal）⇒ "
        "`test_mock_gpu_passes_its_compile_limits` 红"),

    # ── 接线：五处必须连着 ──
    Row("M6 后端不再传自己的编译上限", "gpu",
        "                            limits=limits_for(MODE_COMPILE, self.compile_limits))",
        "                            limits=None)",
        "**具体输入**: `MockGPU(compile_limits=Limits(wall_ms=400))` + 悬挂候选 ⇒ "
        "按默认 30s 跑（用例的 10s 上界与异常类型都会红）"),
    Row("M7 LocalExecutor 不再传自己的编译上限", "local",
        "                            limits=limits_for(MODE_COMPILE, self.compile_limits))",
        "                            limits=None)",
        "**具体输入**: 同上，只是走本地降级梯子那个后端 ⇒ "
        "`test_local_executor_passes_its_compile_limits` 红"),
    Row("M8 Verifier 不发布这一档", "ver",
        '                metrics["compile_status"] = f"compile_{e.status}"',
        '                metrics["compile_status_unused"] = f"compile_{e.status}"',
        "**具体输入**: 一份超时候选走 L1 ⇒ 产物里没有 compile_status ⇒ "
        "`test_a_compile_timeout_is_published_as_its_own_status` 红"),
    Row("M9 签名表不认这一族", "kdb",
        '    ("被 §22.2 拦下", "exec_guard_refused"),',
        '    ("被 §22.2 拦下(已停用)", "exec_guard_refused"),',
        "**具体输入**: 一句超时文案进 failure_class_of ⇒ 落回 `unknown` ⇒ "
        "`test_the_failure_class_is_not_unknown` 红"),
    Row("M10 语料探针把它读成远端", "vch",
        '    if str(metrics.get("compile_status") or "").startswith("compile_"):',
        '    if False:',
        "**具体输入**: 一条 compile_timeout 的 L1 结果 ⇒ origin_of 返回 remote ⇒ "
        "`test_the_corpus_honesty_probe_reads_it_as_local` 红"),
    Row("M11 Conductor 不把上限交给自建后端", "cond",
        "            exec_limits=execution_limits, compile_limits=compile_limits)",
        "            )",
        "**具体输入**: `Conductor(compile_limits=X, execution_limits=Y)` ⇒ 后端拿到 None ⇒ "
        "`test_the_conductor_hands_its_limits_to_the_backend_it_builds` 红"),

    # ── 结构证人：编译路径里不许出现进程内 exec（写成死代码，行为不可区分）──
    Row("M12 MockGPU.compile 里出现进程内 exec", "gpu",
        "                has_kernel, callable_kernel = self._compile_in_guard(kernel_code)",
        "                if False:\n"
        "                    exec(kernel_code, {})\n"
        "                has_kernel, callable_kernel = self._compile_in_guard(kernel_code)",
        "**结构证人**: 这一行是死代码 ⇒ 行为层与原文逐字不可区分；拦住它的只能是 "
        "`TestTheInProcessExecInventory::test_the_two_compile_paths_have_no_in_process_exec` "
        "的 AST 断言",
        behavioral=False),
    Row("M13 LocalExecutor.compile 里出现进程内 exec", "local",
        '        guard = run_compile(kernel_code,\n'
        '                            limits=limits_for(MODE_COMPILE, self.compile_limits))',
        "        if False:\n"
        "            exec(kernel_code, {})\n"
        '        guard = run_compile(kernel_code,\n'
        '                            limits=limits_for(MODE_COMPILE, self.compile_limits))',
        "**结构证人**: 同 M12，靶子是降级梯子的终点那一格；死代码 ⇒ 只有 AST 账本会红",
        behavioral=False),
]
