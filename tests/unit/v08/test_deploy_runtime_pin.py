"""A1: 部署模板必须钉 torch 版本，装了还要回读核对。

背景（2026-11-08，H20 真机）：`_setup_uv_env` 原来跑 `uv add numpy torch triton`，
没有版本约束。隔天在新目录部署时 torch 从 2.14.0 被升到 2.14.1，参考身份门随即
fail closed（`native_libraries, torch_config_sha256, torch_git, torch_version`），
attention 与 layernorm v2 都停在 L2 之前。

这两条测试钉的是**行为**，不是字符串：
  1. 安装命令里的 torch 带版本，且版本来自 `core.runtime_pin`（单一真源）；
  2. 回读到的 `torch.__version__` 不等于受信版本时，函数返回 False 且**不**把环境
     标记成就绪 —— 宁可部署失败，也不要三分钟后一个 identity mismatch。
"""
from pathlib import Path

import pytest

from operator_agent.core.runtime_pin import TORCH_VERSION, matches, torch_requirement
from operator_agent.execute.remote_executor import SSHExecutor


class _ScriptedSSH(SSHExecutor):
    """No network: answers the fixed command sequence `_setup_uv_env` issues."""

    def __init__(self, installed):
        super().__init__(host="offline")
        self.work_dir = "/tmp/oa-deploy-probe"
        self.installed = installed
        self.commands = []

    def _ssh_exec(self, command, *args, **kwargs):
        self.commands.append(command)
        if "import torch" in command:
            return self.installed + "\n"
        if "sys.version_info" in command:
            return "3.12\n"
        return "ok\n"


def test_the_single_source_is_the_attested_build():
    assert TORCH_VERSION == "2.14.0+cu130"
    # 发行版本不带本地标记；`+cu130` 由 wheel 构建写进 `torch.__version__`
    assert torch_requirement() == "torch==2.14.0"
    assert matches("2.14.0+cu130") and not matches("2.14.1+cu130")
    assert not matches("") and not matches(None)


def test_deploy_pins_the_requirement_and_checks_what_actually_landed():
    peer = _ScriptedSSH(installed="2.14.0+cu130")
    assert peer._setup_uv_env() is True
    add = [c for c in peer.commands if "uv add" in c]
    assert len(add) == 1, peer.commands
    assert "torch==2.14.0" in add[0], add[0]
    assert "torch triton" not in add[0] and not add[0].rstrip().endswith("uv add numpy torch triton")
    assert peer._uv_setup is True


def test_a_drifted_build_is_refused_instead_of_marked_ready():
    peer = _ScriptedSSH(installed="2.14.1+cu130")
    assert peer._setup_uv_env() is False
    assert getattr(peer, "_uv_setup", False) is False

def test_the_two_literals_are_one_value():
    """`core.runtime_pin` 与 `reference_contract.PINNED_RUNTIME` 必须逐字相同。

    为什么不 import 成一个：`reference_contract.py` 在远端是**平铺的顶层模块**
    （部署 = 逐文件 scp 到工作目录根），任何 `from ..` 都会让身份回执变空
    —— 2026-11-08 当场踩过。所以是两份字面量 + 这一条守卫。
    """
    from operator_agent.decide.reference_contract import PINNED_RUNTIME

    assert PINNED_RUNTIME["torch_version"] == TORCH_VERSION


def test_the_reference_module_still_imports_flat(tmp_path):
    """远端怎么加载它，这里就怎么加载：拷成一个裸文件，`import reference_contract`。"""
    import shutil
    import subprocess
    import sys

    source = Path(__file__).resolve().parents[3] / "operator_agent" / "decide" / "reference_contract.py"
    shutil.copy(source, tmp_path / "reference_contract.py")
    result = subprocess.run([sys.executable, "-c", "import reference_contract"],
                            cwd=str(tmp_path), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-500:]
