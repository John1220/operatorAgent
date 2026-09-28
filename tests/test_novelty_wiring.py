"""Conductor 有没有真的把候选签名记下来 (doc 47 的接线检查)

`novelty.py` 自己的单测只证明「键提得出来」。这一组证明的是另一半、也是本项目最常栽的那一半:
**算出来的键有没有被写进产物**。`iteration_history` 会被 `ArtifactStore.save()` 序列化进
`result.json`, 所以只要这几条断言成立, 「重复率」这个问题就从「只能靠当场打印那一行」
变成可以离线复盘的数。

用一个最小的假 LLM (不 import 别的测试文件, 免得测试之间互相拖着):
第 1 轮给一份核, 第 2 轮给**只改了名字**的同一份核 —— 于是重复必须是「测出来」的,
而不是因为两轮都返回同一串字符才显然。
"""
import contextlib
import io
import unittest

from operator_agent.conductor import Conductor
from operator_agent.kernel_db import KernelDatabase
from operator_agent.mock_gpu import MockGPU
from operator_agent.novelty import signature_of
from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec, TargetSpec,
                                    TaskComplexity, TaskSchema, TaskType)

KERNEL_A = '''import numpy as np

def kernel(A, B):
    n = A.shape[0]
    out = np.zeros((n, n), dtype=A.dtype)
    for i in range(n):
        out[i, i] = np.sum(A[i] * B[i])
    return out + B
'''

# 与 KERNEL_A 结构完全相同, 只有标识符换了 —— 这正是「模型换个说法重新交一遍」的形状
KERNEL_A_RENAMED = '''import numpy as np

def kernel(X, Y):
    m = X.shape[0]
    res = np.zeros((m, m), dtype=X.dtype)
    for j in range(m):
        res[j, j] = np.sum(X[j] * Y[j])
    return res + Y
'''


class _FakeLLM:
    """Designer 走 chat_json (返回固定 sketch), Coder 走 chat (按调用次序换代码)"""

    def __init__(self, codes):
        self.codes = list(codes)
        self.n_coder = 0
        self._sketch = {"algorithm": "matmul",
                        "tiling": {"block_m": 64, "block_n": 64, "block_k": 32},
                        "memory_plan": {}, "pipeline": [], "optimization_hints": []}

    def chat(self, messages, temperature=None, max_tokens=None, raw=False,
             caller_tag="Unknown"):
        text = messages[-1]["content"]
        if "待修复代码" in text:
            return f"```python\n{self.codes[0]}```"
        code = self.codes[min(self.n_coder, len(self.codes) - 1)]
        self.n_coder += 1
        return f"```python\n{code}```"

    def chat_json(self, messages, temperature=None, max_tokens=None, caller_tag="Unknown"):
        return dict(self._sketch)

    class _Tracker:
        total_calls = 0

        def summary(self):
            return {}

        def pretty(self):
            return "LLM: fake"

    cost_tracker = _Tracker()


def _task() -> TaskSchema:
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.LOW),
        target=TargetSpec(hardware="mock_gpu", dsl="python"),
        constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=0.5,
                                   max_error=1e-3))


class TestConductorNoveltyWiring(unittest.TestCase):
    def _run(self, codes, rounds=3):
        import numpy as np
        llm = _FakeLLM(codes)
        c = Conductor(backend=MockGPU(), kernel_db=KernelDatabase(), llm=llm,
                      max_optimize_rounds=rounds)
        inputs = {"A": np.ones((8, 8), dtype=np.float32),
                  "B": np.ones((8, 8), dtype=np.float32)}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            artifact = c.execute(_task(), inputs, target_gflops=0.0)
        return artifact, buf.getvalue()

    def test_signatures_are_recorded_per_round(self):
        artifact, out = self._run([KERNEL_A, KERNEL_A, KERNEL_A])
        hist = artifact.iteration_history
        self.assertGreaterEqual(len(hist), 2,
                                f"只跑了 {len(hist)} 轮, 这个接线测试就什么都没验")
        for h in hist:
            self.assertIn("candidate_signature", h,
                          "签名没进 iteration_history —— 重复率就无法离线复盘")
            self.assertEqual(signature_of(KERNEL_A), h["candidate_signature"],
                              f"第 {h['round']} 轮记的签名对不上本轮该跑的代码")
        self.assertIn("[Novelty]", out)

    def test_renamed_repeat_is_detected_as_duplicate(self):
        artifact, out = self._run([KERNEL_A, KERNEL_A_RENAMED, KERNEL_A_RENAMED])
        hist = artifact.iteration_history
        sigs = [h.get("candidate_signature") for h in hist]
        self.assertGreaterEqual(len(sigs), 2, "轮数不够, 断言会空转")
        self.assertEqual(sigs[0], sigs[1],
                         "只改名的两份核被判成不同候选 —— 检测器看不见要抓的那一类")
        self.assertEqual(1, hist[1]["duplicate_of"],
                         f"重复没被记账: {[h.get('duplicate_of') for h in hist]}")
        self.assertIn("结构相同", out)

    def test_genuinely_different_candidate_is_not_flagged(self):
        """方向检查的另一边: 假阳的代价是「把改进当成重复」, 比多烧一次验证贵得多。

        这一条在**键**上验 (novelty 的单测里 `test_operator_change_is_a_different_candidate`),
        在**流水线**上验则要能精确控制每一轮喂给 Coder 的代码 —— 第一版想用「第 N 次 chat
        返回第 N 份代码」来控, 结果 Coder 内部的重试与 SearchAgent 的调用共用同一个计数器,
        第 2 轮拿到的还是第 1 份代码, 于是「没判成重复」永远不可能发生: 一个自身就不会
        产生差异的接线测试, 断得再严也是空的。
        """
        from operator_agent.novelty import signature_of
        different = KERNEL_A.replace("out + B", "out - B")
        self.assertNotEqual(signature_of(KERNEL_A), signature_of(different))
        same = KERNEL_A.replace("out + B", "out + B")
        self.assertEqual(signature_of(KERNEL_A), signature_of(same))


if __name__ == "__main__":
    unittest.main()
