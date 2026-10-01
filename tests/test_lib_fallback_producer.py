"""lib_fallback 生产者 (spine-review P1-4)

WHY: eligibility 规则 4 / oracle / verifier 都读 `lib_fallback`, 但此前全仓
**没有任何写入点** → 该护栏恒为 False, 报表上却显示「已检查」。恒 False 的护栏
比没有更危险。本文件钉住生产者: 反作弊命中必须把 `lib_fallback=True` 写进 metrics,
且由此推导的 eligibility 必须全 False (规则 4: 它在测别的东西)。
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from operator_agent.anticheat import (
    CheatDetectedError, lib_fallback_from_error, lib_fallback_metrics, shield_torch,
    static_audit,
)
from operator_agent.eligibility import ALL_FALSE, derive_eligibility
from operator_agent.mock_gpu import MockGPU
from operator_agent.schemas import (
    ConstraintSpec, Precision, SourceSpec, TargetSpec, TaskComplexity,
    TaskSchema, TaskType, VerificationLevel,
)
from operator_agent.verifier import Verifier

from backend_fakes import DeclaredFakeBackend

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


#: 作弊核: wrapper 里直接调用 torch.matmul —— Atrex 实测「正确率被 PyTorch
#: fallback 抬高」的最小复现 (spine-review §P1-4 / §19.3)。
CHEAT_PYTHON = """import torch

def kernel(A, B):
    return torch.matmul(A, B)
"""

#: 静态正则/AST 可能漏过的动态拼接形式, 只能靠 shield_torch 运行时拦。
CHEAT_EVASIVE = """import torch

def kernel(A, B):
    fn = getattr(torch, "mat" + "mul")
    return fn(A, B)
"""

CLEAN_PYTHON = """import numpy as np

def kernel(A, B):
    return A @ B
"""


def _task(dsl: str = "python") -> TaskSchema:
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.LOW),
        target=TargetSpec(hardware="mock_gpu", dsl=dsl,
                          compute_units=64, warp_size=32),
        constraints=ConstraintSpec(precision=Precision.FP32,
                                   min_speedup=0.5, max_error=1e-5),
    )


#: 真机上回来的那句 (2026-10-01 §14-3 原样摘录): 远端进程里的盾抛的 CheatDetectedError
#: 经 `Remote verification error:` 前缀回到控制端。
SHIELD_ERROR = ("Remote verification error: CheatDetectedError: "
                "反作弊: 验证执行期间禁止调用原生算子 torch.matmul()")


def _remote_verifier(error_text: str) -> Verifier:
    """一个**声明了远端比对**的假件 —— 它的失败文本就是真机回来的那一句。

    用 `DeclaredFakeBackend` 而不是 monkeypatch: 门面的能力派生自内层后端,
    打上去的方法会在 `Cap.REMOTE_VERIFY in caps` 那一道判据上静默失效。
    """
    fake = DeclaredFakeBackend(
        verify_function=lambda *a, **kw: {"success": False, "error": error_text})
    return Verifier(fake)


def _inputs(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {"A": (rng.standard_normal((8, 4)) * 0.1).astype(np.float32),
            "B": (rng.standard_normal((4, 8)) * 0.1).astype(np.float32)}


class TestLibFallbackProducer(unittest.TestCase):
    """反作弊命中 → metrics["lib_fallback"] is True → eligibility 全 False"""

    def test_static_audit_hit_writes_lib_fallback(self):
        """静态审计发现 torch.matmul → metrics 必须带 lib_fallback=True"""
        violations = static_audit(CHEAT_PYTHON, "python")
        self.assertTrue(violations, "静态审计应拦下 torch.matmul")
        metrics = lib_fallback_metrics(violations=violations)
        self.assertIs(metrics.get("lib_fallback"), True,
                      "static_audit 命中没有生产 lib_fallback —— 规则 4 恒 False 的老毛病")

    def test_verifier_l2_static_hit_metrics(self):
        """走 Verifier 全路径: L2 失败的 metrics 里必须有 lib_fallback=True"""
        verifier = Verifier(MockGPU())
        results = verifier.verify(CHEAT_PYTHON, _task(), _inputs(),
                                  target_gflops=0.0)
        self.assertGreaterEqual(len(results), 2)
        l2 = results[1]
        self.assertEqual(l2.level, VerificationLevel.L2_FUNCTION)
        self.assertFalse(l2.passed)
        self.assertIs(l2.metrics.get("lib_fallback"), True,
                      f"L2 metrics 未写 lib_fallback: {l2.metrics}")
        elig = derive_eligibility(
            lib_fallback=l2.metrics["lib_fallback"], validity="valid")
        self.assertEqual(elig, ALL_FALSE,
                         "lib_fallback=True 时 eligibility 必须全 False (规则 4)")

    def test_verifier_l2_runtime_shield_hit_metrics(self):
        """动态拼接避开静态审计 → shield_torch 运行时拦 → metrics 仍要有 lib_fallback"""
        verifier = Verifier(MockGPU())
        results = verifier.verify(CHEAT_EVASIVE, _task(), _inputs(),
                                  target_gflops=0.0)
        self.assertGreaterEqual(len(results), 2)
        l2 = results[-1]
        self.assertEqual(l2.level, VerificationLevel.L2_FUNCTION)
        self.assertFalse(l2.passed)
        self.assertIs(l2.metrics.get("lib_fallback"), True,
                      f"运行时命中没有生产 lib_fallback: {l2.metrics}")
        elig = derive_eligibility(
            lib_fallback=l2.metrics["lib_fallback"], validity="valid")
        self.assertEqual(elig, ALL_FALSE)

    def test_mock_gpu_execute_produces_metrics(self):
        """mock_gpu 执行路径本身就是生产者 (不必经过 Verifier)"""
        if not HAS_TORCH:
            self.skipTest("torch 未安装")
        result = MockGPU().execute(CHEAT_PYTHON, _inputs())
        self.assertFalse(result.success)
        self.assertIs(result.metrics.get("lib_fallback"), True,
                      f"mock_gpu 未写 metrics['lib_fallback']: {result.metrics}")
        self.assertIs(result.lib_fallback, True)
        elig = derive_eligibility(
            lib_fallback=result.metrics["lib_fallback"], validity="valid")
        self.assertEqual(elig, ALL_FALSE)

    def test_remote_l2_failure_carries_lib_fallback(self):
        """远端 L2 失败文本里认回运行时盾的命中 (2026-10-01 真机 §14-3)

        WHY: 上面那条 `..._runtime_shield_hit_metrics` 走的是 **MockGPU 本地执行** ——
        盾与 Verifier 在同一个进程里, `take_lib_fallback()` 读得到。真机上盾跑在**远端
        进程**里, 本地那个全局变量永远是空的, 于是同一次库回退只剩一句错误文本、
        `lib_fallback` 没进 metrics —— 而资格规则 4 恰恰是给「静态审计漏了、盾拦住了」
        这条路径准备的。真机实测: 拼名字的 `torch.matmul` 被远端盾拦下 (`CheatDetectedError`),
        L2 如实失败, 但 `lib_fallback=None`。
        """
        results = _remote_verifier(SHIELD_ERROR).verify(
            CHEAT_EVASIVE, _task(), _inputs(), target_gflops=0.0)
        l2 = results[-1]
        self.assertEqual(l2.level, VerificationLevel.L2_FUNCTION)
        self.assertFalse(l2.passed)
        self.assertIs(l2.metrics.get("lib_fallback"), True,
                      f"远端盾命中没有生产 lib_fallback: {l2.metrics}")
        self.assertIn("runtime:torch.matmul",
                      l2.metrics.get("lib_fallback_evidence", ""),
                      "证据要指得出被拦的那个符号")
        elig = derive_eligibility(lib_fallback=l2.metrics["lib_fallback"],
                                  validity="valid")
        self.assertEqual(elig, ALL_FALSE)

    def test_a_remote_error_without_the_shield_text_is_not_claimed(self):
        """反方向: 认文本只认自家文案, 否则任何含 matmul 的报错都会被读成库回退"""
        results = _remote_verifier(
            "Remote verification error: RuntimeError: matmul 输入维度不匹配"
        ).verify(CHEAT_EVASIVE, _task(), _inputs(), target_gflops=0.0)
        l2 = results[-1]
        self.assertFalse(l2.passed)
        self.assertNotIn("lib_fallback", l2.metrics,
                         f"不是盾的报错被读成了库回退: {l2.metrics}")

    def test_parser_shape(self):
        """解析器只认前缀 + 取符号; 认不出就是 None (不猜)"""
        self.assertEqual(lib_fallback_from_error(SHIELD_ERROR), "torch.matmul")
        self.assertEqual(
            lib_fallback_from_error("反作弊: 验证执行期间禁止调用原生算子 torch.mm()"),
            "torch.mm")
        self.assertIsNone(lib_fallback_from_error(""), "空串不算命中")
        self.assertIsNone(lib_fallback_from_error(None), "None 不算命中")
        self.assertIsNone(lib_fallback_from_error("torch.matmul failed"), "没前缀不算命中")

    @unittest.skipUnless(HAS_TORCH, "torch 未安装")
    def test_shield_torch_records_hit(self):
        """shield_torch 拦到调用后, lib_fallback_metrics() 必须能取到"""
        a = torch.randn(2, 2)
        with self.assertRaises(CheatDetectedError):
            with shield_torch():
                torch.matmul(a, a)
        metrics = lib_fallback_metrics()
        self.assertIs(metrics.get("lib_fallback"), True)
        self.assertIn("torch.matmul", metrics.get("lib_fallback_evidence", ""))

    def test_clean_kernel_does_not_set_lib_fallback(self):
        """反例: 干净核不得误报 lib_fallback"""
        metrics = lib_fallback_metrics()
        self.assertEqual(metrics, {}, "没有命中时生产者必须返回空 dict")
        verifier = Verifier(MockGPU())
        results = verifier.verify(CLEAN_PYTHON, _task(), _inputs(),
                                  target_gflops=0.0)
        l2 = results[1]
        self.assertTrue(l2.passed, l2.message)
        self.assertIs(l2.metrics.get("lib_fallback"), None,
                      f"干净核误报 lib_fallback: {l2.metrics}")


if __name__ == "__main__":
    unittest.main()
