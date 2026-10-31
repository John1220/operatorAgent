"""环境自检与依赖管理
职责: 远程 GPU 机器的链路自检 + 依赖分析 + 自动配置 + 依赖清单输出
     + 环境指纹的构造与跨运行漂移比较 (doc 49 §3.5 第 1 处)

链路: SSH → uv → Python → torch → triton → CUDA → GPU 硬件
依赖分析: 从生成的 kernel 代码静态提取 import, 与已装包对比, 输出缺失清单
"""

import ast
import re
from dataclasses import dataclass, field
from typing import Optional

from ..decide.measurement import safe_error_text


# ─── 环境指纹 (doc 49 §3.5 第 1 处「算了没人读」) ────────────────────────────────
#
# `conductor.py` 一直算着一个 `fingerprint` (name/device/cc/torch/triton), 旁边那行
# 注释写着「用于环境漂移自检与结果追溯」, 然后 `print` 一次就丢了 —— 全仓 3 次命中
# 全在那 6 行里, 不入库、不比较。**没有环境漂移自检**, 注释描述的是一个不存在的机制。
#
# 这一节是那个机制本身: 构造 (fingerprint_of) + 比较 (compare_fingerprints)。放这个
# 模块是因为它就是「环境自检」的归宿, 而判据的**后果** (降到哪一档) 在
# `measurement.assess()` 里 —— 两件事分开, 各自可离线单测。

#: 指纹只收**跨运行可比的身份量**: 器件名与三个版本号。
#: 刻意不收 `compute_units`/`memory` 这类规格 —— 它们在 `hardware_caps` 的表里, 而
#: doc 49 §八 第 4 条记着「探测值因为字段名不匹配被丢掉」, 收进来会让漂移报告混进
#: 一个本来就不稳定的量。
FINGERPRINT_KEYS = ("name", "device", "compute_capability",
                    "torch_version", "triton_version")

#: 执行侧身份单独立一个键。「同一台机器、换了个后端」是最要紧的一种漂移: 门面
#: `RemoteExecutor(mode="ssh")` 连不上时会解析到 `LocalExecutor`, 器件名与版本号
#: 一项都不会变, 只有这个键会变 (doc 49 §3.1 A1 / §八 第 1 条)。
FINGERPRINT_BACKEND_KEY = "backend"


def fingerprint_of(hw_info: Optional[dict], backend_label: str = "") -> dict:
    """从 `Backend.get_hardware_info()` 抽出环境指纹 (只留可比的键)

    `hw_info` 里缺的键就**不出现在指纹里**, 不填默认值: 「上一次报了 triton_version、
    这一次没报」本身就是一种漂移, 用默认值填平会把它抹掉 (与 doc 49 §3.3 那类
    「把不知道翻译成一个像样的值」是同一个错误)。
    """
    fp = {k: hw_info[k] for k in FINGERPRINT_KEYS if k in (hw_info or {})}
    if backend_label:
        fp[FINGERPRINT_BACKEND_KEY] = backend_label
    return fp


def compare_fingerprints(current: Optional[dict], previous: Optional[dict],
                         previous_from: str = "") -> "EnvironmentDrift":
    """比较两份指纹, 返回差异。任一侧为空 ⇒ `comparable=False`, **不是**「没漂移」

    「没有上一次可比」与「比过了、一样」是两件事, 混成一个空列表就会把前者读成后者 ——
    那正是 `measurement.assess()` 里「launch 地板未知」那条已经付过学费的坑
    (缺证据不等于检查过了)。
    """
    if not current or not previous:
        return EnvironmentDrift(changed=(), comparable=False,
                                previous_from=previous_from)
    changed = []
    for k in sorted(set(current) | set(previous)):
        if k not in previous:
            changed.append(f"{k}: 上一次没有 → {current[k]!r}")
        elif k not in current:
            changed.append(f"{k}: {previous[k]!r} → 本次没报")
        elif current[k] != previous[k]:
            changed.append(f"{k}: {previous[k]!r} → {current[k]!r}")
    return EnvironmentDrift(changed=tuple(changed), comparable=True,
                            previous_from=previous_from)


@dataclass(frozen=True)
class EnvironmentDrift:
    """一次漂移比较的结果

    `changed` 是给人读的句子 (它会被打进日志、写进产物、并作为
    `measurement.assess(environment_drift=…)` 的输入, 于是同一份措辞在三个地方一致)。
    """

    changed: tuple = ()
    comparable: bool = False
    previous_from: str = ""

    @property
    def drifted(self) -> bool:
        return self.comparable and bool(self.changed)

    def as_dict(self) -> dict:
        """落进 result.json / execution_snapshot.json 的形状"""
        return {"comparable": self.comparable,
                "drifted": self.drifted,
                "changed": list(self.changed),
                "previous_from": self.previous_from}

    def report_lines(self) -> list:
        """日志里的几行。没漂移时也报一句 —— 只报坏消息的话, 「机制没跑」与「跑了没问题」
        在日志里长得一样。"""
        if not self.comparable:
            return ["[Env] 环境漂移: 没有可比的上一次运行 (首次跑 / 旧产物没记指纹)"]
        if not self.drifted:
            return [f"[Env] 环境漂移: 无 (与 {self.previous_from or '上一次运行'} 一致)"]
        out = [f"[Env] ⚠️ 环境漂移: 与 {self.previous_from or '上一次运行'} 相比有 "
               f"{len(self.changed)} 项变了 —— 本轮读数与上一次不可比"]
        out += [f"[Env]   - {c}" for c in self.changed]
        return out


@dataclass
class CheckItem:
    """单项检查结果"""
    name: str            # 检查项名 (如 "ssh", "uv", "torch")
    passed: bool
    detail: str = ""     # 版本号或错误信息
    auto_fixable: bool = False   # 是否可自动配置
    fix_hint: str = ""           # 人工配置提示 (auto_fixable=False 时)


@dataclass
class EnvironmentReport:
    """环境自检报告"""
    items: list = field(default_factory=list)
    all_passed: bool = True
    missing_deps: list = field(default_factory=list)   # 缺失的 pip 依赖名

    def add(self, item: CheckItem):
        self.items.append(item)
        if not item.passed:
            self.all_passed = False
            if item.auto_fixable and item.name not in self.missing_deps:
                self.missing_deps.append(item.name)

    def summary(self) -> str:
        lines = []
        for it in self.items:
            tag = "OK" if it.passed else "FAIL"
            lines.append(f"  [{tag}] {it.name}: {it.detail}" if it.detail
                         else f"  [{tag}] {it.name}")
        return "\n".join(lines)

    def dependency_manifest(self) -> str:
        """输出供用户人工配置的依赖清单"""
        if self.all_passed:
            return ""
        lines = ["环境自检未通过, 需配置以下依赖:"]
        for it in self.items:
            if not it.passed:
                lines.append(f"  - {it.name}: {it.fix_hint or it.detail}")
        return "\n".join(lines)


class DependencyAnalyzer:
    """从 kernel 代码静态提取 import, 输出所需 pip 依赖名"""

    # 模块名 → pip 包名映射 (导入名与包名不一致的)
    _MODULE_TO_PIP = {
        "torch": "torch",
        "triton": "triton",
        "numpy": "numpy",
        "einops": "einops",
        "flash_attn": "flash-attn",
        "xformers": "xformers",
        "cupy": "cupy-cuda12x",
        "numba": "numba",
    }

    @classmethod
    def extract_imports(cls, code: str) -> list:
        """提取代码中 import 的顶层模块名"""
        modules = set()
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        modules.add(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        modules.add(node.module.split(".")[0])
        except SyntaxError:
            # 语法错误时回退正则
            for m in re.finditer(r"^\s*(?:import|from)\s+([a-zA-Z_]\w*)", code, re.MULTILINE):
                modules.add(m.group(1).split(".")[0])
        return sorted(modules)

    @classmethod
    def to_pip_names(cls, modules: list) -> list:
        """将模块名转为 pip 包名; 内置模块排除"""
        builtins = {"os", "sys", "time", "math", "re", "json", "typing",
                    "dataclasses", "pathlib", "functools", "itertools",
                    "collections", "abc", "io", "copy", "struct"}
        pips = []
        for mod in modules:
            if mod in builtins:
                continue
            pips.append(cls._MODULE_TO_PIP.get(mod, mod))
        return sorted(set(pips))


class EnvironmentChecker:
    """
    远程环境链路自检

    用法:
        checker = EnvironmentChecker(ssh_exec_fn, uv_path)
        report = checker.check()
        if not report.all_passed:
            print(report.dependency_manifest())
    """

    def __init__(self, ssh_exec_fn, uv_path: str = "~/.local/bin/uv",
                 work_dir: str = "."):
        self._ssh = ssh_exec_fn          # callable: (cmd) -> stdout, raises on fail
        self._uv = uv_path
        self._work_dir = work_dir

    def check(self, required_pips: Optional[list] = None) -> EnvironmentReport:
        """执行全链路自检; required_pips 为代码所需依赖(按需追加检查)"""
        report = EnvironmentReport()

        # 1. SSH 连通
        try:
            self._ssh("echo ok")
            report.add(CheckItem("ssh", True, "连接正常"))
        except Exception as e:
            # 与执行侧四个出口同一条规矩 (#102 第 2 片): 这句会进环境报告 → `result.json`,
            # 而 `str(TimeoutExpired)` 带着整条 ssh argv (端口、密钥路径、user@host)。
            report.add(CheckItem("ssh", False, safe_error_text(e)[:120],
                                 fix_hint="检查 SSH 免密登录与网络"))
            return report   # SSH 不通, 后续无意义

        # 2. uv
        try:
            ver = self._ssh(f"{self._uv} --version")
            report.add(CheckItem("uv", True, ver))
        except Exception:
            report.add(CheckItem("uv", False, "未安装", auto_fixable=True,
                                fix_hint="curl -LsSf https://astral.sh/uv/install.sh | sh"))

        # 3. Python
        try:
            ver = self._ssh("python3 --version")
            report.add(CheckItem("python", True, ver))
        except Exception as e:
            report.add(CheckItem("python", False, str(e)[:80], fix_hint="安装 Python 3.10+"))

        # 4. torch
        try:
            ver = self._ssh(
                f"cd {self._work_dir} && {self._uv} run python -c 'import torch; print(torch.__version__)'"
            )
            report.add(CheckItem("torch", True, ver))
        except Exception:
            report.add(CheckItem("torch", False, "未安装", auto_fixable=True,
                                fix_hint="uv add torch (或按 CUDA 版本选择对应 wheel)"))

        # 5. triton
        try:
            ver = self._ssh(
                f"cd {self._work_dir} && {self._uv} run python -c 'import triton; print(triton.__version__)'"
            )
            report.add(CheckItem("triton", True, ver))
        except Exception:
            report.add(CheckItem("triton", False, "未安装", auto_fixable=True,
                                fix_hint="uv add triton"))

        # 6. CUDA 可用性 (torch 视角)
        try:
            avail = self._ssh(
                f"cd {self._work_dir} && {self._uv} run python -c "
                "'import torch; print(torch.cuda.is_available())'"
            )
            if "True" in avail:
                report.add(CheckItem("cuda", True, "torch.cuda 可用"))
            else:
                report.add(CheckItem("cuda", False, "torch.cuda 不可用",
                                     fix_hint="检查 NVIDIA 驱动与 CUDA toolkit 版本匹配"))
        except Exception as e:
            report.add(CheckItem("cuda", False, str(e)[:80], fix_hint="检查 NVIDIA 驱动"))

        # 7. GPU 硬件
        try:
            name = self._ssh(
                f"cd {self._work_dir} && {self._uv} run python -c "
                "'import torch; print(torch.cuda.get_device_name(0))'"
            )
            report.add(CheckItem("gpu", True, name))
        except Exception as e:
            report.add(CheckItem("gpu", False, str(e)[:80], fix_hint="检查 GPU 设备是否就绪"))

        # 8. 额外所需依赖 (从生成代码提取)
        for pip_name in (required_pips or []):
            mod_name = pip_name.replace("-", "_")
            try:
                self._ssh(
                    f"cd {self._work_dir} && {self._uv} run python -c 'import {mod_name}'"
                )
                report.add(CheckItem(f"dep:{pip_name}", True, "已安装"))
            except Exception:
                report.add(CheckItem(f"dep:{pip_name}", False, "未安装",
                                     auto_fixable=True, fix_hint=f"uv add {pip_name}"))

        return report
