"""P1: 可外挂参考 + CSA 压缩算子 (DeepSeek-V4 式 11–12) 的离线标定

为什么离线: 这一层只改「参考从哪来」与一个纯函数算子, 不需要 GPU/LLM。
真机试打是下一步 (拿 REFERENCE_SOURCE 喂 Conductor), 不在本文件。
"""
import ast
import textwrap
import unittest
from pathlib import Path

import numpy as np

from operator_agent.ops import csa_compress
from operator_agent.decide.reference import (ReferenceSourceError, load_reference_source,
                                      resolve_reference)
from operator_agent.core.schemas import ConstraintSpec, Precision, SourceSpec, TargetSpec, TaskSchema, TaskType

REPO_ROOT = Path(__file__).resolve().parents[2]


def _task(reference_source=None, kernel_type="csa_compress"):
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type=kernel_type, reference_source=reference_source),
        target=TargetSpec(hardware="mock_gpu", dsl="python"),
        constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=1.0),
    )


class TestReferenceSourceContract(unittest.TestCase):
    def test_loads_a_minimal_reference(self):
        fn = load_reference_source("def reference(**inputs):\n    return inputs['X'] * 2\n")
        self.assertEqual(4.0, fn(X=np.array([2.0], dtype=np.float32))[0])

    def test_missing_reference_name_is_refused_not_silently_none(self):
        with self.assertRaises(ReferenceSourceError):
            load_reference_source("def other(**inputs):\n    return 1\n")

    def test_broken_source_is_refused(self):
        with self.assertRaises(ReferenceSourceError):
            load_reference_source("def reference(**inputs):\n    return (\n")

    def test_empty_source_is_refused(self):
        with self.assertRaises(ReferenceSourceError):
            load_reference_source("   ")

    def test_resolve_returns_none_when_absent(self):
        self.assertIsNone(resolve_reference(_task(None)))
        self.assertIsNone(resolve_reference(_task("  ")))

    def test_resolve_loads_when_present(self):
        fn = resolve_reference(_task("def reference(**inputs):\n    return inputs['X']\n"))
        self.assertTrue(callable(fn))

    def test_resolve_refuses_broken_source_rather_than_falling_back(self):
        """加载失败必须炸 —— 回落到内建参考等于「新算子被当成 matmul 去验」"""
        with self.assertRaises(ReferenceSourceError):
            resolve_reference(_task("def reference(**inputs):\n    return (\n"))


class TestCsaCompressMatchesThePaper(unittest.TestCase):
    """式 11–12: 2m 窗 softmax + Hadamard 加权和, i=0 时前窗 Z=−∞、C=0"""

    def test_shape_is_n_over_m(self):
        xs = csa_compress.make_inputs(n=16, c=8, m=4, seed=1)
        out = csa_compress.reference(**xs)
        self.assertEqual((4, 8), out.shape)

    def test_first_window_ignores_the_nonexistent_previous_block(self):
        """i=0 时 S^b 全 0, C^Comp_0 应只依赖当前窗 —— 扰动「前一窗」不该有影响

        前一窗在 n=8,m=4 时就是不存在的那一段; 我们用「改 B_b」来扰动 i=0
        的虚拟前窗路径: B_b 只出现在 Z^b, 而 i=0 的 Z^b 是 −∞ 常数, 与 B_b 无关。
        但 i≥1 会用 B_b。所以比 i=0 与 i=1 对 B_b 的敏感性。
        """
        xs = csa_compress.make_inputs(n=8, c=4, m=4, seed=2)
        out0 = csa_compress.reference(**xs)
        xs2 = dict(xs)
        xs2["B_b"] = xs["B_b"] + 3.0
        out1 = csa_compress.reference(**xs2)
        np.testing.assert_allclose(out0[0], out1[0], rtol=1e-6, atol=1e-6,
                                   err_msg="i=0 不该依赖 B_b")
        self.assertGreater(float(np.max(np.abs(out0[1] - out1[1]))), 1e-3,
                           "i=1 该依赖 B_b —— 整条路径对 B_b 都不敏感说明公式写错了")

    def test_softmax_is_over_the_stacked_2m_rows_per_channel(self):
        """独立第二实现: 手写 2m×c 的按列 softmax, 与主参考必须逐元素一致"""
        xs = csa_compress.make_inputs(n=8, c=5, m=2, seed=3)
        C, Z, B_a, B_b = (np.asarray(xs[k], dtype=np.float64) for k in ("C", "Z", "B_a", "B_b"))
        n, c = C.shape
        m = B_a.shape[0]
        out = np.zeros((n // m, c), dtype=np.float64)
        for i in range(n // m):
            ca = C[i * m:(i + 1) * m]
            za = Z[i * m:(i + 1) * m] + B_a
            if i == 0:
                cb, zb = np.zeros((m, c)), np.full((m, c), -np.inf)
            else:
                cb = C[(i - 1) * m:i * m]
                zb = Z[(i - 1) * m:i * m] + B_b
            logits = np.concatenate([za, zb], axis=0)   # (2m, c)
            logits = logits - logits.max(axis=0, keepdims=True)
            ex = np.exp(logits)
            s = ex / ex.sum(axis=0, keepdims=True)      # 按列归一
            out[i] = (s[:m] * ca).sum(0) + (s[m:] * cb).sum(0)
        np.testing.assert_allclose(csa_compress.reference(**xs), out.astype(np.float32),
                                   rtol=1e-5, atol=1e-5)

    def test_overlapping_windows_share_the_middle_block(self):
        """相邻窗的 C^b 与 C^a 重叠: n=12,m=4 时块 4..8 同时是 i=1 的 C^a 与 i=2 的 C^b"""
        xs = csa_compress.make_inputs(n=12, c=4, m=4, seed=5)
        a = csa_compress.reference(**xs)
        xs2 = {k: v.copy() for k, v in xs.items()}
        xs2["C"][4:8] += 0.5
        b = csa_compress.reference(**xs2)
        # i=0 只用块 0..4, 不该动
        np.testing.assert_allclose(a[0], b[0], rtol=1e-6, atol=1e-6)
        self.assertGreater(float(np.max(np.abs(a[1] - b[1]))), 1e-3,
                           "i=1 的 C^a 是块 4..8, 改它必须影响 i=1")
        self.assertGreater(float(np.max(np.abs(a[2] - b[2]))), 1e-3,
                           "i=2 的 C^b 是块 4..8 (重叠), 改它必须影响 i=2")

    def test_n_not_divisible_by_m_is_refused(self):
        with self.assertRaises(ValueError):
            csa_compress.make_inputs(n=6, c=4, m=4)

    def test_reference_source_reimplements_the_same_math(self):
        """REFRENCE_SOURCE (torch) 与本地 NumPy 参考在已知输入上必须一致"""
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("本机没有 torch")
        fn = load_reference_source(csa_compress.REFERENCE_SOURCE)
        xs = csa_compress.make_inputs(n=8, c=6, m=2, seed=7)
        out_src = fn(**{k: np.asarray(v) for k, v in xs.items()})
        out_local = csa_compress.reference(**xs)
        np.testing.assert_allclose(np.asarray(out_src, dtype=np.float32), out_local,
                                   rtol=1e-5, atol=1e-5)


class TestVerifierPicksUpTheExternalReference(unittest.TestCase):
    """`_build_reference_fn` 必须先认 reference_source, 而不是查内建表"""

    def _verifier(self):
        from operator_agent.execute.mock_gpu import MockGPU
        from operator_agent.decide.verifier import Verifier
        return Verifier(MockGPU())

    def test_external_reference_wins_over_kernel_type_lookup(self):
        v = self._verifier()
        src = "def reference(**inputs):\n    return inputs['X'] + 1\n"
        task = _task(src, kernel_type="matmul")   # 故意写成 matmul: 不能走内建 matmul 分支
        fn, kind = v._build_reference_fn(task, {"X": np.zeros((2, 2), dtype=np.float32)})
        self.assertEqual("reference_source", kind)
        out = fn()
        np.testing.assert_allclose(np.asarray(out), np.ones((2, 2), dtype=np.float32))

    def test_broken_external_reference_is_an_error_not_a_fallback(self):
        v = self._verifier()
        with self.assertRaises(ValueError):
            v._build_reference_fn(_task("def reference(**inputs):\n    return (\n"),
                                  {"X": np.zeros(4)})

    def test_absent_external_reference_still_uses_builtin(self):
        v = self._verifier()
        fn, kind = v._build_reference_fn(_task(None, kernel_type="matmul"),
                                         {"A": np.ones((2, 2)), "B": np.ones((2, 2))})
        self.assertEqual("pytorch_eager", kind)
        self.assertIsNotNone(fn)


class TestRemotePayloadCarriesReferenceSource(unittest.TestCase):
    """控制端发出的 verify / benchmark_with_baselines 必须带 reference_source 键"""

    def test_ssh_executor_sends_the_key(self):
        src = (REPO_ROOT / "operator_agent" / "execute" / "remote_executor.py").read_text(encoding="utf-8")
        self.assertIn('"reference_source"', src)

    def test_both_runners_consume_the_key(self):
        for name in ("run_kernel.py", "run_kernel_xpu.py"):
            src = (REPO_ROOT / "remote_scripts" / name).read_text(encoding="utf-8")
            self.assertIn("reference_source", src, f"{name} 没读 reference_source")
            self.assertIn("_load_reference_from_source", src, f"{name} 没有外挂参考加载器")

    def test_runner_falls_back_to_builtin_when_source_empty(self):
        """空 source = 旧行为; 不能因键存在就跳过内建参考"""
        src = (REPO_ROOT / "remote_scripts" / "run_kernel.py").read_text(encoding="utf-8")
        # 加载器之后必须还有一行回落 _reference_torch
        self.assertIn("_reference_torch(kernel_type", src)

    def test_verifier_passes_task_reference_into_remote_calls(self):
        src = (REPO_ROOT / "operator_agent" / "decide" / "verifier.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(src.count("reference_source=task.source.reference_source"), 2,
                                "L2/L3 至少两处远程调用都要带上 reference_source")


class TestRunnerContractStillCloses(unittest.TestCase):
    """加键之后两个 runner 的键集合必须仍然对得上 (守卫自己别被改红)"""

    def test_runner_contract_suite_still_passes(self):
        import os
        import subprocess
        import sys
        # 子进程要清掉 OPERATOR_AGENT_* 否则 conftest 的日志目录哨兵会撞红
        # (与 tests/conftest.py 同一条纪律: 测试日志不能写回仓库 logs/)
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("OPERATOR_AGENT_")}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        r = subprocess.run(
            [sys.executable, "-B", "-m", "pytest",
             "tests/unit/test_runner_contract.py", "-q", "--tb=line"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, env=env)
        self.assertEqual(0, r.returncode, r.stdout[-1500:] + r.stderr[-500:])


if __name__ == "__main__":
    unittest.main()
