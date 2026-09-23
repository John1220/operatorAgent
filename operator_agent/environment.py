"""环境自检与依赖管理
职责: 远程 GPU 机器的链路自检 + 依赖分析 + 自动配置 + 依赖清单输出

链路: SSH → uv → Python → torch → triton → CUDA → GPU 硬件
依赖分析: 从生成的 kernel 代码静态提取 import, 与已装包对比, 输出缺失清单
"""

import ast
import re
from dataclasses import dataclass, field
from typing import Optional


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
            report.add(CheckItem("ssh", False, str(e)[:80], fix_hint="检查 SSH 免密登录与网络"))
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
