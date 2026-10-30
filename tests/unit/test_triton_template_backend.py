"""§12 真模板后端（Triton matmul）：点 → **真能跑**的源码（批次审查 No.2 的第二半）

关键判据不是「字符串长得像」，而是**这份源码能不能过评测器那几道静态门**
（接口契约 / 反作弊 / L1 语法）—— 能过，模板路径才真的接上了端到端。
"""

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from operator_agent.anticheat import static_audit  # noqa: E402
from operator_agent.candidates import PruneResult, prune_by_smem  # noqa: E402
from operator_agent.interface_contract import check_kernel_interface  # noqa: E402
from operator_agent.resource_check import MemCaps  # noqa: E402
from operator_agent.triton_matmul_template import (  # noqa: E402
    INSTANTIATOR_VER, TEMPLATE_ID, TritonMatmulTemplateBackend,
    TritonTemplateSSHExecutor, instantiate, param_space)

POINT = {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "stages": 2, "warps": 8}


class TestTheSpaceIsARealLattice(unittest.TestCase):
    def test_the_space_enumerates_deterministically(self):
        sp = param_space()
        self.assertEqual(TEMPLATE_ID, sp.template_id)
        self.assertEqual(INSTANTIATOR_VER, sp.instantiator_ver)
        self.assertEqual(2 * 2 * 2 * 3 * 2, sp.size())
        first = [c.params for c in sp.points()][:2]
        self.assertEqual([{"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32,
                           "stages": 1, "warps": 4},
                          {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32,
                           "stages": 1, "warps": 8}], first,
                         "轴序 = 插入序, 最后一轴变化最快")

    def test_a_smem_limit_only_shrinks_the_lattice(self):
        small = param_space(smem_limit_bytes=48 * 1024)
        self.assertLessEqual(small.size(), param_space().size())
        for values in small.axes.values():
            self.assertTrue(values, "缩格不许把某一轴缩成空集 (那会让空间恒空)")

    def test_the_prune_accepts_the_real_space(self):
        """真空间必须被现有剪枝吃下 —— 假后端换成真后端不该改剪枝的接口"""
        res = prune_by_smem(param_space().points(), MemCaps(228))
        self.assertIsInstance(res, PruneResult)
        self.assertTrue(res.kept)


class TestTheEmittedSourceIsExecutable(unittest.TestCase):
    def test_the_point_becomes_source_that_passes_the_static_gates(self):
        src = instantiate(POINT).source
        violations = static_audit(src, dsl="triton").violations
        self.assertEqual([], violations, f"反作弊不该有违规: {violations}")
        verdict = check_kernel_interface(src, ("A", "B"), language="triton")
        self.assertIsNone(verdict.problem, "接口契约要求 kernel(**inputs)")
        self.assertTrue(verdict.applicable)
        compile(src, "<template>", "exec")     # L1 的语法那道门

    def test_the_multi_buffer_axis_reaches_the_launch_not_the_text(self):
        """stages/warps 是**启动参数**: 源码文本里只有它们进 launch 的那一处"""
        src = instantiate(dict(POINT, stages=3, warps=4)).source
        self.assertIn("num_stages=3", src)
        self.assertIn("num_warps=4", src)
        self.assertNotIn("BLOCK_M: tl.constexpr = 128", src,
                         "tile 是编译期常量参数, 不是写死的字面量")

    def test_the_tile_actually_changes_with_the_point(self):
        a = instantiate(dict(POINT, BLOCK_M=64)).source
        b = instantiate(dict(POINT, BLOCK_M=128)).source
        self.assertNotEqual(a, b, "不同点位必须产出不同源码 (否则整条搜索是假的)")

    def test_the_template_pins_the_precision_it_claims(self):
        """doc 79 观察① 的教训写进模板: 默认 TF32 在 fp32 容差 1e-4 下过不了 L2"""
        self.assertIn('input_precision="ieee"', instantiate(POINT).source)

    def test_a_point_with_a_missing_axis_is_refused(self):
        for bad in ({k: v for k, v in POINT.items() if k != "stages"},
                    dict(POINT, warps=0), dict(POINT, BLOCK_K="32"),
                    dict(POINT, stages=True)):
            with self.subTest(point=bad):
                with self.assertRaises(ValueError):
                    instantiate(bad)

    def test_the_evidence_says_the_launch_parameters_are_not_in_the_text(self):
        ev = instantiate(POINT).evidence
        self.assertIn("stages/warps 是启动参数", ev)
        self.assertIn("BM=128", ev)


class TestTheProtocolIsHonoured(unittest.TestCase):
    def test_the_backend_serves_the_space_and_the_instantiator(self):
        # 不再需要 _concrete 补桩: 本类不是执行后端, 不继承 Backend (I71 的修复面)
        be = TritonMatmulTemplateBackend(smem_limit_bytes=228 * 1024)
        self.assertEqual(be.param_space().template_id, TEMPLATE_ID)
        self.assertEqual(INSTANTIATOR_VER, be.instantiate(TEMPLATE_ID, POINT).instantiator_ver)
        with self.assertRaises(ValueError):
            be.instantiate("nope", POINT)

    def test_the_default_is_still_code_only_for_a_hosted_backend(self):
        """反向: 基类默认「代码型」—— 这一片不许把「所有后端都有参数空间」变成默认。

        用一个只补上执行侧抽象方法的宿主类, 验证**继承来的**两个默认:
        `param_space()` 返 None（有意义的答案）、`instantiate()` raise（没有默认实例化）。
        """
        from operator_agent.backend_caps import Backend

        class Hosted(Backend):
            @property
            def label(self): return "hosted"

            @property
            def capabilities(self): return frozenset()

            def compile(self, *a, **k): raise NotImplementedError
            def execute(self, *a, **k): raise NotImplementedError
            def benchmark(self, *a, **k): raise NotImplementedError
            def get_hardware_info(self, *a, **k): raise NotImplementedError

        self.assertIsNone(Hosted().param_space())
        with self.assertRaises(NotImplementedError):
            Hosted().instantiate(TEMPLATE_ID, POINT)
        # 基类默认: 指纹身份 = label (只有组合后端覆写它 —— I72)
        self.assertEqual("hosted", Hosted().fingerprint_label)


class TestTheComposedBackendServesBothHalves(unittest.TestCase):
    """生产里 `Conductor.backend` **一个对象**要同时是可执行后端与模板后端"""

    class _FakeExec:
        label = "fake_exec"

        @property
        def capabilities(self):
            return frozenset({"real_device"})

        def compile(self, *a, **k): return {"ok": True}

    def test_both_halves_come_from_the_right_place(self):
        be = TritonTemplateSSHExecutor(self._FakeExec(), smem_limit_bytes=228 * 1024)
        self.assertEqual(TEMPLATE_ID, be.param_space().template_id)
        self.assertEqual(INSTANTIATOR_VER, be.instantiate(TEMPLATE_ID, POINT).instantiator_ver)
        self.assertEqual({"ok": True}, be.compile("src"), "执行那一半按名字转发")
        self.assertEqual(frozenset({"real_device"}), be.capabilities)
        self.assertIn("triton_matmul_v1", be.label)
        # 访问器也必须指对那一半 —— 诊断面拿错对象就说谎 (变异矩阵 M7 的证人)
        self.assertEqual(TEMPLATE_ID, be.template_backend.param_space().template_id)

    def test_a_missing_attribute_still_raises(self):
        """fail-closed: 转发不许把「少了个方法」变成 None"""
        be = TritonTemplateSSHExecutor(self._FakeExec())
        with self.assertRaises(AttributeError):
            be.not_a_method_at_all

    def test_the_smem_limit_reaches_the_template_half(self):
        """A cap alone does not establish a physical layout or justify deleting axes."""
        tiny = TritonTemplateSSHExecutor(self._FakeExec(), smem_limit_bytes=1)
        self.assertEqual(tiny.param_space().size(), param_space().size())
        for values in tiny.param_space().axes.values():
            self.assertTrue(values, "缩格不许把某一轴缩成空集 (那会让空间恒空)")

    def test_the_fingerprint_takes_the_executor_identity(self):
        """I72 的证人: 指纹的 backend 维度取**执行侧**身份 —— 换模板壳是候选来源
        变化, 不是执行环境变化。不覆写时每个新组合后端首跑都被判「环境漂移」
        (degraded, can_persist=False), 第一份读数永远不可入库 (变异矩阵 M9 的证人)。"""
        be = TritonTemplateSSHExecutor(self._FakeExec())
        self.assertEqual("fake_exec", be.fingerprint_label)
        self.assertNotEqual(be.fingerprint_label, be.label,
                            "组合 label 带模板后缀, 正是它不许进指纹的原因")

    def test_the_composed_backend_builds_in_a_fresh_process(self):
        """I71 的证人: 组合后端曾只在「同文件早先用例已把模板类污染成具体类」的
        进程里可构造 —— 全新进程 (生产的形状) __init__ 直接 TypeError, 而全套件
        绿把地雷埋住了。证人必须是**子进程**: 类级污染跨不出本进程。
        """
        code = (
            "from operator_agent.triton_matmul_template import ("
            "TEMPLATE_ID, TritonTemplateSSHExecutor)\n"
            "class Exec:\n"
            "    label = 'exec'\n"
            "    def compile(self, *a, **k): return True\n"
            "be = TritonTemplateSSHExecutor(Exec())\n"
            "assert be.param_space().template_id == TEMPLATE_ID\n"
            "assert be.compile('x') is True\n"
        )
        proc = subprocess.run([sys.executable, "-B", "-c", code],
                              capture_output=True, text=True, cwd=str(ROOT))
        self.assertEqual(0, proc.returncode,
                         f"全新进程里构造/使用组合后端失败:\n{proc.stderr}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
