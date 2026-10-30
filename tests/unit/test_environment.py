"""环境自检与依赖管理的单元测试

覆盖:
- DependencyAnalyzer: import 提取 (ast/正则回退)、模块→pip 名映射、内置排除
- EnvironmentChecker: 链路自检 (ssh/uv/python/torch/triton/cuda/gpu)、缺失项报告
- EnvironmentReport: summary、dependency_manifest 输出
- SSHExecutor.deploy: 按清单逐文件 hash 同步、依赖先于 runner 推送
- RemoteExecutor 门面: 可选能力由提供方**声明**, 缺失时抛而不是报「已就绪」
- Conductor: 环境自检未通过时提前返回 (不进入生成流程)

运行: python3 tests/unit/test_environment.py  或  python3 -m unittest tests.test_environment -v
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from operator_agent.environment import (
    DependencyAnalyzer, EnvironmentChecker, EnvironmentReport, CheckItem,
)
from operator_agent.backend_caps import OPTIONAL_CAP_METHODS, CapabilityMissing, Cap
from operator_agent.remote_executor import RemoteExecutor
from operator_agent.schemas import (
    TaskSchema, SourceSpec, TargetSpec, ConstraintSpec,
    TaskType, TaskComplexity, Precision, VerificationLevel,
)

CODE_WITH_IMPORTS = """import torch
import triton
import triton.language as tl
import numpy as np
import einops
import os
import sys

def kernel(A, B):
    return A @ B
"""

CODE_SYNTAX_ERROR = "import torch\ndef kernel(A, B:\n    return A @ B"


# ─── DependencyAnalyzer ──────────────────────────────────────

class TestDependencyAnalyzer(unittest.TestCase):

    def test_extract_imports(self):
        mods = DependencyAnalyzer.extract_imports(CODE_WITH_IMPORTS)
        self.assertIn("torch", mods)
        self.assertIn("triton", mods)
        self.assertIn("numpy", mods)
        self.assertIn("einops", mods)

    def test_builtins_excluded_from_pip(self):
        pips = DependencyAnalyzer.to_pip_names(
            DependencyAnalyzer.extract_imports(CODE_WITH_IMPORTS)
        )
        self.assertNotIn("os", pips)
        self.assertNotIn("sys", pips)

    def test_module_to_pip_mapping(self):
        pips = DependencyAnalyzer.to_pip_names(["flash_attn", "torch", "numpy"])
        self.assertIn("flash-attn", pips)   # 下划线→连字符
        self.assertIn("torch", pips)
        self.assertIn("numpy", pips)

    def test_syntax_error_fallback_regex(self):
        mods = DependencyAnalyzer.extract_imports(CODE_SYNTAX_ERROR)
        self.assertIn("torch", mods)

    def test_empty_code(self):
        self.assertEqual(DependencyAnalyzer.extract_imports(""), [])

    def test_from_import(self):
        code = "from einops.layers.torch import Rearrange\nimport torch"
        mods = DependencyAnalyzer.extract_imports(code)
        self.assertIn("einops", mods)
        self.assertIn("torch", mods)


# ─── EnvironmentReport ───────────────────────────────────────

class TestEnvironmentReport(unittest.TestCase):

    def test_all_passed(self):
        r = EnvironmentReport()
        r.add(CheckItem("ssh", True, "ok"))
        r.add(CheckItem("torch", True, "2.0.0"))
        self.assertTrue(r.all_passed)
        self.assertEqual(r.dependency_manifest(), "")

    def test_failed_items_in_manifest(self):
        r = EnvironmentReport()
        r.add(CheckItem("ssh", True, "ok"))
        r.add(CheckItem("torch", False, "未安装", fix_hint="uv add torch"))
        r.add(CheckItem("cuda", False, "不可用", fix_hint="检查驱动"))
        self.assertFalse(r.all_passed)
        manifest = r.dependency_manifest()
        self.assertIn("torch", manifest)
        self.assertIn("uv add torch", manifest)
        self.assertIn("检查驱动", manifest)

    def test_summary_format(self):
        r = EnvironmentReport()
        r.add(CheckItem("ssh", True, "ok"))
        r.add(CheckItem("torch", False, "未安装"))
        s = r.summary()
        self.assertIn("[OK] ssh", s)
        self.assertIn("[FAIL] torch", s)


# ─── EnvironmentChecker ─────────────────────────────────────

class TestEnvironmentChecker(unittest.TestCase):

    def _checker(self, ssh_fn):
        return EnvironmentChecker(ssh_fn, uv_path="uv", work_dir=".")

    def test_all_pass(self):
        def ssh_fn(cmd):
            if "echo ok" in cmd: return "ok"
            if "uv --version" in cmd: return "uv 0.5.0"
            if "python3 --version" in cmd: return "Python 3.12.0"
            if "import torch; print(torch.__version__)" in cmd: return "2.4.0"
            if "import triton; print(triton.__version__)" in cmd: return "3.1.0"
            if "cuda.is_available" in cmd: return "True"
            if "get_device_name" in cmd: return "NVIDIA A100"
            return ""
        r = self._checker(ssh_fn).check()
        self.assertTrue(r.all_passed, r.summary())
        self.assertEqual(len(r.items), 7)

    def test_ssh_fail_returns_early(self):
        def ssh_fn(cmd):
            raise RuntimeError("connection refused")
        r = self._checker(ssh_fn).check()
        self.assertFalse(r.all_passed)
        self.assertEqual(len(r.items), 1)   # 只有 ssh 一项
        self.assertEqual(r.items[0].name, "ssh")

    def test_torch_missing(self):
        def ssh_fn(cmd):
            if "echo ok" in cmd: return "ok"
            if "--version" in cmd: return "uv 0.5.0"
            if "python3 --version" in cmd: return "Python 3.12.0"
            raise RuntimeError("import failed")
        r = self._checker(ssh_fn).check()
        self.assertFalse(r.all_passed)
        torch_item = next(i for i in r.items if i.name == "torch")
        self.assertFalse(torch_item.passed)
        self.assertTrue(torch_item.auto_fixable)

    def test_required_pip_dep_checked(self):
        def ssh_fn(cmd):
            if "echo ok" in cmd: return "ok"
            if "--version" in cmd or "python3 --version" in cmd: return "v1"
            if "import torch" in cmd and "cuda" not in cmd: return "2.0"
            if "import triton" in cmd: return "3.0"
            if "cuda.is_available" in cmd: return "True"
            if "get_device_name" in cmd: return "A100"
            if "import einops" in cmd: raise RuntimeError("no module")
            return ""
        r = self._checker(ssh_fn).check(required_pips=["einops"])
        dep_item = next(i for i in r.items if i.name == "dep:einops")
        self.assertFalse(dep_item.passed)
        self.assertTrue(dep_item.auto_fixable)


# ─── SSHExecutor.deploy hash 同步 ────────────────────────────

class TestDeployHashSync(unittest.TestCase):
    """deploy() 按 hash 决定是否 scp (替代'存在即跳过')"""

    @patch("operator_agent.remote_executor.subprocess")
    def test_remote_outdated_triggers_scp(self, mock_subproc):
        from operator_agent.remote_executor import SSHExecutor
        # 远程 hash 与本地不同 -> 触发 scp
        mock_subproc.run.return_value = MagicMock(returncode=0)
        mock_subproc.run.return_value.stdout = "remote_hash_old"
        # ssh_exec 返回不匹配的 hash
        ex = SSHExecutor(host="h", user="u")
        ex._uv_path = "uv"
        ex._uv_setup = True  # 跳过 _setup_uv_env

        # mock _ssh_exec: deploy 的 hash 检查返回不匹配
        call_count = {"n": 0}
        def ssh_exec(cmd, stdin_data=None):
            call_count["n"] += 1
            if "hashlib" in cmd:
                return "old_hash_abc"   # 远程旧 hash
            return "ok"
        ex._ssh_exec = ssh_exec

        result = ex.deploy()
        self.assertTrue(result)
        # scp 应被调用 (本地 vs 远程 hash 不匹配)
        self.assertGreater(mock_subproc.run.call_count, 0)

    @patch("operator_agent.remote_executor.subprocess")
    def test_remote_uptodate_skips_scp(self, mock_subproc):
        from operator_agent.remote_executor import SSHExecutor
        import hashlib

        ex = SSHExecutor(host="h", user="u")
        ex._uv_path = "uv"
        ex._uv_setup = True

        files = ex._remote_files()
        if not all(local.exists() for local, _ in files):
            self.skipTest("待部署文件缺失")
        # 远端 hash 查询现在是一次往返返回多行 "<路径> <hash>", 按 _remote_files()
        # 派生而不是写死 run_kernel.py —— 将来往清单里加文件时本用例不必再改。
        listing = "\n".join(
            f"{remote} {hashlib.sha256(local.read_bytes()).hexdigest()[:12]}"
            for local, remote in files
        )

        def ssh_exec(cmd, stdin_data=None):
            if "hashlib" in cmd:
                return listing   # 全部匹配
            return "ok"
        ex._ssh_exec = ssh_exec

        result = ex.deploy()
        self.assertTrue(result)
        # hash 匹配, scp 不应被调用
        mock_subproc.run.assert_not_called()

    @patch("operator_agent.remote_executor.subprocess")
    def test_dependencies_pushed_before_runner(self, mock_subproc):
        """runner 的同级依赖必须先于 runner 推送

        部署中途失败时, 这个顺序决定结果是「新依赖 + 旧 runner」(能跑) 还是
        「新 runner + 缺依赖」(import 就炸)。远端 hash 查询失败/文件缺失都会
        触发全量推送, 所以这里让远端报空。
        """
        from operator_agent.remote_executor import SSHExecutor

        ex = SSHExecutor(host="h", user="u")
        ex._uv_path = "uv"
        ex._uv_setup = True
        ex._ssh_exec = lambda cmd, stdin_data=None: "" if "hashlib" in cmd else "ok"

        self.assertTrue(ex.deploy())

        pushed = [c.args[0][-1].split(":")[-1] for c in mock_subproc.run.call_args_list]
        self.assertEqual(pushed, [remote for _, remote in ex._remote_files()])
        self.assertIn("tolerance.py", pushed[0],
                      "L2 数值门的容差表必须先于 runner 到位, 否则新 runner 一 import 就炸")


# ─── Conductor 环境前置自检 ─────────────────────────────────

class TestConductorEnvCheck(unittest.TestCase):

    def test_env_fail_returns_early_without_generation(self):
        """环境自检失败 -> 跳过 Coder/Verifier, 返回带清单的失败 artifact"""
        from operator_agent.conductor import Conductor
        from operator_agent.mock_gpu import MockGPU

        # MockGPU 不声明 ENV_PROVISION, 不会触发; 用声明了该能力且自检失败的后端。
        # 能力必须**声明**: conductor 读的是 `Cap.ENV_PROVISION in backend.capabilities`,
        # 光把方法挂上去不会让它跑起来 —— 那正是旧 hasattr 守卫的病 (见下一个类)。
        class FailingBackend(MockGPU):
            @property
            def capabilities(self):
                return frozenset(super().capabilities | {Cap.ENV_PROVISION})

            def ensure_environment(self, required_pips=None):
                return False, "需配置: torch (uv add torch)"
            def get_hardware_info(self):
                return {"name": "mock"}

        import numpy as np
        task = TaskSchema(
            task_type=TaskType.OPERATOR_GENERATION,
            source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.LOW),
            target=TargetSpec(hardware="mock", dsl="python"),
            constraints=ConstraintSpec(precision=Precision.FP32,
                                       min_speedup=0.5, max_error=1e-5),
        )
        # 需 LLM 生成 sketch, 但环境失败应在此之前
        llm = MagicMock()
        sketch = {"algorithm": "matmul", "tiling": {"block_m": 64, "block_n": 64, "block_k": 32},
                  "memory_plan": {}, "pipeline": [], "optimization_hints": []}
        llm.chat_json.return_value = sketch
        llm.chat.return_value = ""
        conductor = Conductor(backend=FailingBackend(), llm=llm)
        rng = np.random.default_rng(0)
        inputs = {"A": rng.standard_normal((4, 4)).astype(np.float32),
                  "B": rng.standard_normal((4, 4)).astype(np.float32)}

        with patch.object(conductor.designer, "run") as mock_designer:
            artifact = conductor.execute(task, inputs, target_gflops=0.0)
        # 未进入 Designer (环境失败提前返回)
        mock_designer.assert_not_called()
        self.assertFalse(artifact.all_passed)
        self.assertIn("环境自检", artifact.verification_results[0].message)


class TestCapabilityIsDeclaredNotSniffed(unittest.TestCase):
    """能力由提供方**声明**, 调用方按声明决定要不要调 (doc 49 §4.1 第 1/3 条)

    缺陷实例 (2026-09-25 于 H20 验证 L2 数值门时撞见): conductor.py:233 的守卫是
    `if hasattr(self.backend, "ensure_environment")`, 而三个 driver 传进 Conductor
    的都是 RemoteExecutor 门面。门面没有这个方法 → 守卫恒为 False → 「任务开始前
    的环境自检」在真机路径上从来没跑过。不报错, 只是不执行, 所以一直没被发现,
    而 docs/问题识别与技术方案.md 的 P1-3 把它记为 ✅ 已实施。

    修法**不是**「把方法补到门面上」—— 那只是把 hasattr 从恒假变成恒真, 静默 no-op
    照旧。修法是让可选方法在 `Backend` 基类上恒存在且一调就抛, 于是 hasattr 再也
    得不到可用的答案, 判断只能改由 `Cap.X in backend.capabilities` 承担。
    """

    def test_optional_methods_exist_on_every_backend_so_hasattr_is_useless(self):
        """这条为真, 才说明「用 hasattr 判能力」这种写法已经不可表达"""
        backend = RemoteExecutor(mode="local")
        for name in OPTIONAL_CAP_METHODS.values():
            self.assertTrue(hasattr(backend, name), f"{name} 必须在基类上存在")

    def test_backend_without_env_provisioning_raises_instead_of_reporting_ready(self):
        """LocalExecutor 没有要 provision 的远端环境 —— 但「没有」不等于「已就绪」

        旧写法在这里返回 `(True, "")`, 理由是「没有要 provision 的东西就是就绪的诚实
        答案」。对 LocalExecutor 诚实, 对 HTTPExecutor 不诚实: 它把「这个后端没法准备
        自己的环境」上报成了「环境已就绪」(doc 49 §3.2 B1)。
        """
        ex = RemoteExecutor(mode="local")
        self.assertNotIn(Cap.ENV_PROVISION, ex.capabilities)
        with self.assertRaises(CapabilityMissing) as ctx:
            ex.ensure_environment()
        self.assertIn("LocalExecutor", str(ctx.exception),
                      "异常必须带着**内层**后端的身份, 不是门面的 (postmortem 0004)")

    def test_facade_delegates_rather_than_stubbing(self):
        """委托必须真的落到底层实现, 不能在门面上就地返回 True"""
        ex = RemoteExecutor(mode="ssh")
        fake = MagicMock()
        fake.ensure_environment.return_value = (False, "需配置: torch (uv add torch)")
        ex._backend = fake

        self.assertEqual(ex.ensure_environment(required_pips=["torch"]),
                         (False, "需配置: torch (uv add torch)"))
        fake.ensure_environment.assert_called_once_with(required_pips=["torch"])

    def test_facade_deploy_delegates(self):
        ex = RemoteExecutor(mode="ssh")
        fake = MagicMock()
        fake.deploy.return_value = True
        ex._backend = fake
        self.assertTrue(ex.deploy())
        fake.deploy.assert_called_once_with()


if __name__ == "__main__":
    unittest.main(verbosity=2)
