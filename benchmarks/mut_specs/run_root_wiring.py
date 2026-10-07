"""「§22.1 的布局根真的有生产调用点、过门的候选真的落盘」那张变异表的**可执行**形态

靶子是 `tests/test_run_root_wiring.py`（切片 C 的守卫：根的唯一性 / 落盘内容与权限位 /
被拒的不留痕 / 唯一写入口有调用者 / 真机驱动在传 run_root）。

为什么单独一张表: `benchmarks/mut_specs/run_layout_doors.py` 钉的是**权限位与角色判据**
（§22.1 的执行层），而这一片钉的是**接线**——「判据指着的那个路径」与「盘上真实存在的
产物」是不是同一个文件、驱动是不是真的在建布局。两类失效方式不同, 各配一张表。

跑: python3 -B -m benchmarks.mutation_calibrate --spec benchmarks/mut_specs/run_root_wiring.py
"""
import sys
from os.path import abspath, dirname

_ROOT = dirname(dirname(dirname(abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmarks.mutation_calibrate import Row  # noqa: E402

FILES = {
    "rl": "operator_agent/run_layout.py",
    "cu": "operator_agent/conductor.py",
    "hp": "benchmarks/harness_pass_rate.py",
}

TARGETS = ["tests/test_run_root_wiring.py"]

ROWS = [
    Row("M1 「一次运行一个新根」退回第一版 (秒级时间戳 + pid)", "rl",
        '    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")\n'
        '    seq = next(_RUN_SEQ)\n'
        '    return os.path.join(os.path.abspath(os.path.expanduser(base_dir)),\n'
        '                        f"{safe}-{stamp}-{os.getpid()}-{seq}")',
        '    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")\n'
        '    return os.path.join(os.path.abspath(os.path.expanduser(base_dir)),\n'
        '                        f"{safe}-{stamp}-{os.getpid()}")',
        "**具体输入**: 同一进程内连续两次 `new_run_root(base)` —— 时钟粒度不保证两次拿到"
        "不同的秒值, 实测 hz 上两次都落 `run-20261007T070329Z-49156` ⇒ "
        "`test_two_runs_get_two_roots_under_the_base` 红。这正是**落地当天自己写出来**的"
        "缺陷: 函数 docstring 明写「同一根跨运行复用会让上一轮的候选被覆盖」, 而第一版正是"
        "那样"),
    Row("M2 落盘的候选内容不是交给后端的那一份", "cu",
        "        return self.layout.write_for(ROLE_CANDIDATE, rel, code)",
        '        return self.layout.write_for(ROLE_CANDIDATE, rel, "")',
        "**具体输入**: 带 `run_root` 跑一次 `execute` ⇒ `candidate/code.py` 是空串, 而"
        "同样一份码送到了后端 ⇒ `test_the_file_is_the_code_that_reached_the_backend` 的"
        "`assertIn(text, backend.seen)` 红。§21 的 `Patch.path` 与 §22.1 的产物必须是"
        "同一个文件 —— 这条钉的就是「判据指着的路径≠盘上内容」"),
    Row("M3 落盘绕过唯一写入口 (裸写)", "cu",
        "        return self.layout.write_for(ROLE_CANDIDATE, rel, code)",
        '        path = self.layout.path(rel)\n'
        '        with open(path, "w", encoding="utf-8") as _fh:\n'
        '            _fh.write(code)\n'
        '        return path',
        "**具体输入**: 同 M2 的一次运行 —— 裸写落出的文件权限位是 `0644`（`open` 的默认"
        "模式）而 §22.1 的 artifact 一律 `0400` ⇒ `test_the_file_is_the_code_that_reached_"
        "the_backend` 的权限位断言红; 同时 `write_for(` 在 `conductor.py` 里数成 0 处 ⇒ "
        "`test_conductor_calls_write_for_exactly_once` 红。两条守卫各钉一层 (内容 vs 门)"),
    Row("M4 「没有 run_root 就什么都不做」拆掉", "cu",
        "        if not self.run_root or self.layout is None:",
        "        if False:",
        "**具体输入**: 不带 `run_root` 构造 `Conductor` 后调 `_materialize_candidate` —— "
        "旧行为是返回 `None`（那是「声明」那一档, 不是「默认建一个相对 CWD 的布局」）, "
        "拆掉后 `self.layout` 是 `None` ⇒ `AttributeError` ⇒ "
        "`test_no_run_root_means_no_write` 红"),
    Row("M5 真机驱动不再传 run_root", "hp",
        'run_root=new_run_root(str(out_dir), label)',
        'run_root=""',
        "**没有具体输入**: 驱动少传一个根, 行为上只是**不建布局**（不落候选、权限位保护"
        "零个产物）—— 今天没有任何行为断言覆盖「这个驱动传了根」, 拦住它的只有源码守卫 "
        "`test_the_real_machine_drivers_pass_a_run_root`。这正是本仓记过多次的形态: "
        "「声明 + 无生产调用点」在离线全绿时与「已接线」长得一样 (I82)",
        behavioral=False),
    Row("M6 被 §21 拒掉的候选也落盘", "cu",
        '            if not _action_verdict.accepted:\n'
        '                print(f"\\n[§21 Runtime] 动作被拒 (零 GPU 秒): "',
        '            if not _action_verdict.accepted:\n'
        '                self._materialize_candidate(current_code, task)\n'
        '                print(f"\\n[§21 Runtime] 动作被拒 (零 GPU 秒): "',
        "**具体输入**: 两轮的脚本化运行（第 1 轮过门、第 2 轮带 `import torch` 被拒）—— "
        "被拒那份覆盖 `candidate/code.py` ⇒ 盘上留下 `REFUSED_MARK` ⇒ "
        "`test_a_refused_candidate_leaves_nothing`（内容级否证）红。与「被拒的候选一个字符"
        "都没到后端」是同一条纪律, 只是这一面看的是**盘上**"),
]
