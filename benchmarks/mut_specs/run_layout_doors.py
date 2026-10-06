"""「§22.1 目录权限位：越权写入必须在写的那一刻失败」那张变异表的**可执行**形态

靶子是 `tests/test_run_layout.py`（形状 / 角色矩阵 / 裸写 / 同源 / 单门 / 接线）。

为什么单独一张表: §22.1 的主张不是「有七个目录」，而是**违反时写入必须失败**。
这句话有两层（权限位 + 唯一入口的角色判据），变异表每行拆掉一层，看守卫是不是真的
在守它 —— 只守一层的话，另一层就是「文档里写着、代码里没有」。

跑: python3 -B -m benchmarks.mutation_calibrate --spec benchmarks/mut_specs/run_layout_doors.py
"""
import sys
from os.path import abspath, dirname

_ROOT = dirname(dirname(dirname(abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmarks.mutation_calibrate import Row  # noqa: E402

FILES = {
    "rl": "operator_agent/run_layout.py",
    "ar": "operator_agent/agent_runtime.py",
    "cu": "operator_agent/conductor.py",
}

TARGETS = ["tests/test_run_layout.py"]

ROWS = [
    Row("M1 唯一入口不再判角色 (Python 那层拆掉)", "rl",
        "        if declared != role:",
        "        if False:",
        "**具体输入**: `write_for(\"candidate\", \"measurements/l3.json\", b\"cheat\")` —— "
        "注意写窗是**入口自己**开的（chmod 0700），所以角色判据一拆，越权写就真的落进 "
        "`measurements/` ⇒ `test_the_candidate_cannot_write_any_protected_dir` 与角色矩阵红"),
    Row("M2 受保护格的权限位改成可写 (内核那层拆掉)", "rl",
        "PROTECTED_DIR_MODE = 0o500",
        "PROTECTED_DIR_MODE = 0o700",
        "**具体输入**: 裸 `open(root/measurements/new.txt, \"w\")` ⇒ 不再抛 "
        "`PermissionError` ⇒ `test_creating_a_new_entry_in_a_protected_dir_fails` 与"
        "形状用例（权限位读数）红"),
    Row("M3 写者换人 (measurements 归 candidate)", "rl",
        '    "measurements": ROLE_ORACLE,',
        '    "measurements": ROLE_CANDIDATE,',
        "**具体输入**: 与 M1 同一个调用，但这次是**声明表**被改 —— 角色判据仍然在跑、"
        "仍然放行 ⇒ 角色矩阵用例（7 合法 / 28 拒）与 `test_the_candidate_cannot_write_"
        "any_protected_dir` 红。这一行说明判据的强度取决于那张表，而表本身要有守卫"),
    Row("M4 路径逃逸不再拦 (`..` 放行)", "rl",
        '    if any(p == ".." for p in parts):',
        '    if False:',
        "**具体输入**: `layout.path(\"../x.py\")` / `write_for(\"oracle\", "
        "\"measurements/../../x\", …)` ⇒ 不再抛 `LayoutError` ⇒ "
        "`test_paths_cannot_escape_the_layout` 红（越出 `run_root` 的路径重新变得可表达）"),
    Row("M5 文件写位拆掉 (artifact 可被覆盖)", "rl",
        "            os.chmod(tmp, ARTIFACT_MODE)",
        "            os.chmod(tmp, 0o600)",
        "**具体输入**: 先经入口写一份 `measurements/l3.json`，再裸 `open(path, \"w\")` ⇒ "
        "覆盖成功 ⇒ `test_overwriting_an_existing_artifact_fails` 与"
        "`TestTheSanctionedPathWorks` 的权限位断言红。目录写位管「新建条目」、"
        "文件写位管「覆盖内容」—— 这一行拆的是后一半"),
    Row("M9 `ensure_dir` 一律收 0500 (可写格的子目录被收成只读)", "rl",
        "            want = PROTECTED_DIR_MODE if protected else WRITABLE_DIR_MODE",
        "            want = PROTECTED_DIR_MODE",
        "**具体输入**: `ensure_dir(\"candidate\", \"candidate/scratch/tmp\")` ⇒ 子目录变 "
        "`0500`, 在候选的 scratch 里写文件抛 `PermissionError` ⇒ "
        "`test_ensure_dir_keeps_a_writable_slot_writable` 红。这一行钉的是**落地当天自己"
        "写出来又收掉的**那个缺陷: 一层 `if` 少了一半, 行为读数只在这条新用例上现形"),
    Row("M6 §21 判据与 §22.1 布局各存一份名单", "ar",
        "PROTECTED_DIR_NAMES: Tuple[str, ...] = _LAYOUT_PROTECTED_DIR_NAMES",
        'PROTECTED_DIR_NAMES: Tuple[str, ...] = ("contract", "harness", "baseline",\n'
        '                                       "measurements", "report")',
        "**没有具体输入**: 两份名单今天**逐字相同**, 任何行为读数都一样（判据跑出来的"
        "四个状态一字不差）—— 拦住它的只有 `test_the_protected_names_are_the_same_object` "
        "里的 `assertIs`。这正是「同源」必须写成身份断言而不是内容断言的理由: "
        "内容比对在它们各自漂走的那一天才红, 而那天已经晚了",
        behavioral=False),
    Row("M7 `candidate_path` 又把绝对路径的根吃掉", "ar",
        '    return layout_path_of(run_root, f"{CANDIDATE_DIR}/code.{ext}")',
        '    return "/".join(p for p in (str(run_root).strip("/"), CANDIDATE_DIR,\n'
        '                                f"code.{ext}") if p)',
        "**具体输入**: `candidate_path(\"/a/b\")` ⇒ 返回 `a/b/candidate/code.py`（相对路径）"
        "⇒ `test_the_absolute_path_keeps_its_root` 与 `test_the_candidate_path_comes_"
        "from_the_layout_module` 红。这条是老实现的回归, 记在 CHECKLOG I82"),
    Row("M8 `execute` 不再建布局 (接线拆掉)", "cu",
        "        self.layout = run_layout(self.run_root) if self.run_root.strip() else None",
        "        self.layout = None",
        "**具体输入**: 带 `run_root` 跑一次 `execute` ⇒ 内层跑起来时 `conductor.layout` "
        "仍是 `None`, 而 artifact 已经在写了 ⇒ `test_execute_builds_it_before_the_"
        "inner_run` 红; 源码守卫（接线那三行）也红"),
]