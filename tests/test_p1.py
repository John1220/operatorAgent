"""P1 级问题优化的单元测试 (M2: 性能可信与效率)

覆盖:
- P1-1: CoderAgent 生成后预检 (截断/语法/结构) + 带反馈重试
- P1-2: KernelDatabase 评分式检索, FailureDatabase 失败案例库,
        Fixer 历史案例注入, Conductor 负样本沉淀, 产物血缘
- P1-3: Verifier 编译缓存 (code hash), RemoteExecutor 主机键控串行锁
- P1-4: Router 多特征路由

运行: python3 tests/test_p1.py  或  python3 -m unittest tests.test_p1 -v
"""

import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from operator_agent.agents import CoderAgent
from operator_agent.conductor import Conductor, Router
from operator_agent.fixer import FixerAgent
from operator_agent.kernel_db import KernelDatabase, FailureDatabase, FailureRecord
from operator_agent.mock_gpu import MockGPU
from operator_agent.prompt_manager import PromptManager
from operator_agent.remote_executor import RemoteExecutor, _get_exec_lock
from operator_agent.verifier import Verifier
from operator_agent.schemas import (
    TaskSchema, SourceSpec, TargetSpec, ConstraintSpec, UnifiedSketch,
    TaskType, TaskComplexity, Precision, RouteType, VerificationLevel,
    TilingStrategy, TritonCompileConfig, PipelineStage, DSLEnum,
)


PYTHON_MATMUL = """import numpy as np

def kernel(A, B):
    return A @ B
"""

TRUNCATED_MATMUL = """import numpy as np

def kernel(A, B):
    return (A @ B"""

BROKEN_MATMUL = """import numpy as np

def kernel(A, B:
    return A @ B
"""

FENCED_VALID = f"```python\n{PYTHON_MATMUL}```"
FIXER_MARKER = "待修复代码"


class MockLLM:
    def __init__(self, chat_handler=None):
        self.chat_calls: list[list[dict]] = []
        self._chat_handler = chat_handler or (lambda prompt: "")
        self._sketch = {
            "algorithm": "matmul",
            "tiling": {"block_m": 64, "block_n": 64, "block_k": 32},
            "memory_plan": {},
            "pipeline": [],
            "optimization_hints": [],
        }

    def chat(self, messages, temperature=None, max_tokens=None, raw=False, caller_tag="Unknown"):
        self.chat_calls.append(messages)
        return self._chat_handler(messages[-1]["content"])

    def chat_json(self, messages, temperature=None, max_tokens=None, caller_tag="Unknown"):
        return dict(self._sketch)


def _make_task(dsl="python", hw="mock_gpu", kernel="matmul",
               complexity=TaskComplexity.LOW, fused=None) -> TaskSchema:
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type=kernel, complexity=complexity,
                          fused_ops=fused or []),
        target=TargetSpec(hardware=hw, dsl=dsl, compute_units=64, warp_size=32),
        constraints=ConstraintSpec(precision=Precision.FP32,
                                   min_speedup=0.5, max_error=1e-5),
    )


def _make_inputs(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {"A": (rng.standard_normal((8, 4)) * 0.1).astype(np.float32),
            "B": (rng.standard_normal((4, 8)) * 0.1).astype(np.float32)}


# ─── P1-1: Coder 预检 + 重试 ─────────────────────────────────

class TestValidateCode(unittest.TestCase):

    def test_valid_code_passes(self):
        ok, issues = CoderAgent._validate_code(PYTHON_MATMUL, "python")
        self.assertTrue(ok, issues)

    def test_truncated_code_detected(self):
        ok, issues = CoderAgent._validate_code(TRUNCATED_MATMUL, "python")
        self.assertFalse(ok)
        self.assertTrue(any("截断" in i or "语法" in i for i in issues))

    def test_missing_kernel_detected(self):
        ok, issues = CoderAgent._validate_code("import numpy as np\nx = 1", "python")
        self.assertFalse(ok)
        self.assertTrue(any("kernel" in i for i in issues))

    def test_missing_jit_detected_for_triton(self):
        ok, issues = CoderAgent._validate_code(PYTHON_MATMUL, "triton")
        self.assertFalse(ok)
        self.assertTrue(any("triton.jit" in i for i in issues))

    def test_empty_code(self):
        ok, _ = CoderAgent._validate_code("", "python")
        self.assertFalse(ok)


class TestCoderRetry(unittest.TestCase):

    def _coder(self, handler) -> tuple:
        llm = MockLLM(chat_handler=handler)
        coder = CoderAgent(KernelDatabase(), llm=llm, prompt_manager=PromptManager())
        return coder, llm

    def test_truncated_then_valid_retries_with_feedback(self):
        responses = [TRUNCATED_MATMUL, PYTHON_MATMUL]
        calls = {"n": 0}

        def handler(prompt):
            r = responses[calls["n"]]
            calls["n"] += 1
            return r

        coder, llm = self._coder(handler)
        sketch = UnifiedSketch(algorithm="matmul", version=1, target_dsl="python")
        code = coder.run(sketch, _make_task())

        self.assertIn("def kernel", code)
        self.assertEqual(len(llm.chat_calls), 2)
        # 第二次请求必须携带预检反馈
        self.assertIn("未通过预检", llm.chat_calls[1][-1]["content"])

    def test_attempts_exhausted_returns_last(self):
        coder, llm = self._coder(lambda p: TRUNCATED_MATMUL)
        sketch = UnifiedSketch(algorithm="matmul", version=1, target_dsl="python")
        code = coder.run(sketch, _make_task())
        self.assertEqual(len(llm.chat_calls), CoderAgent.MAX_GENERATE_ATTEMPTS)
        self.assertTrue(code)  # 返回最后一次结果 (交由 L1 判定)

    def test_valid_first_try_no_retry(self):
        coder, llm = self._coder(lambda p: PYTHON_MATMUL)
        sketch = UnifiedSketch(algorithm="matmul", version=1, target_dsl="python")
        coder.run(sketch, _make_task())
        self.assertEqual(len(llm.chat_calls), 1)


# ─── P1-2: 评分式检索 ────────────────────────────────────────

class TestKernelDBSearch(unittest.TestCase):

    def setUp(self):
        self.db = KernelDatabase()

    def test_exact_dsl_and_algorithm_ranks_first(self):
        results = self.db.search(algorithm="matmul", hardware="nvidia_gpu", dsl="triton")
        self.assertTrue(results)
        self.assertEqual(results[0].dsl, "triton")
        self.assertIn("matmul", results[0].algorithm)

    def test_token_overlap_ranks_attention(self):
        results = self.db.search(algorithm="flash attention", hardware="nvidia_gpu", dsl="triton")
        self.assertTrue(results)
        self.assertEqual(results[0].algorithm, "attention")

    def test_irrelevant_query_returns_empty(self):
        results = self.db.search(algorithm="exotic_op_xyz", hardware="nvidia_gpu", dsl="triton")
        self.assertEqual(results, [])

    def test_mock_gpu_python_fallback(self):
        results = self.db.search(algorithm="matmul", hardware="mock_gpu", dsl="python")
        self.assertEqual(results[0].dsl, "python")

    def test_limit_respected(self):
        results = self.db.search(hardware="nvidia_gpu", dsl="triton", limit=2)
        self.assertLessEqual(len(results), 2)


# ─── P1-2: 失败案例库 ────────────────────────────────────────

class TestFailureDatabase(unittest.TestCase):

    def _record(self, fid, kernel="matmul", dsl="python", fixed=False,
                error="编译失败: 语法错误", fix="修正语法",
                lessons=None, hardware="mock_gpu", keywords=None, root_cause="",
                applicable_hardware=None):
        return FailureRecord(failure_id=fid, kernel_type=kernel, hardware=hardware,
                             dsl=dsl, level="L1", error_summary=error,
                             fix_summary=fix if fixed else "", fixed=fixed,
                             lessons=lessons or [],
                             error_keywords=keywords or [],
                             root_cause=root_cause,
                             applicable_hardware=applicable_hardware or [])

    def test_search_fixed_first(self):
        db = FailureDatabase()
        db.add(self._record("a", fixed=False))
        db.add(self._record("b", fixed=True))
        results = db.search(kernel_type="matmul", dsl="python")
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].fixed)

    def test_dsl_filter(self):
        db = FailureDatabase()
        db.add(self._record("a", dsl="python"))
        self.assertEqual(db.search(kernel_type="matmul", dsl="triton"), [])

    def test_fixer_injects_history_cases(self):
        db = FailureDatabase()
        db.add(self._record("a", fixed=True,
                            lessons=["docstring 起止引号必须同时使用占位符"],
                            root_cause="docstring 起始引号丢失导致语法错误"))
        llm = MockLLM(chat_handler=lambda p: FENCED_VALID)
        fixer = FixerAgent(llm=llm, failure_db=db)
        from operator_agent.schemas import VerificationResult
        fixer.run(BROKEN_MATMUL, _make_task(), [
            VerificationResult(level=VerificationLevel.L1_COMPILE,
                               passed=False, message="编译失败"),
        ])
        prompt = llm.chat_calls[0][-1]["content"]
        # Reflexion 新增字段应出现在 prompt 中
        self.assertIn("可复用教训", prompt)
        self.assertIn("docstring", prompt)
        self.assertIn("根因", prompt)

    # ---------- Reflexion 新测试 ----------
    def test_search_by_error_jaccard_match(self):
        """search_by_error 应按错误词元相似度排序"""
        db = FailureDatabase()
        db.add(self._record("mismatch",
                            error="RuntimeError: shape mismatch at tl.dot [BLOCK_M, BLOCK_K]",
                            lessons=["在 tl.dot 前校验 A/B 的 tile 维度匹配"]))
        db.add(self._record("syntax",
                            error="SyntaxError: unexpected EOF while parsing 括号不配对",
                            lessons=["检查括号配对; 防止 max_tokens 截断"]))
        # 查 shape mismatch 相关错误
        results = db.search_by_error("tl.dot 出现 shape mismatch BLOCK_M",
                                     kernel_type="matmul", dsl="python")
        self.assertTrue(results, "应命中 shape 那条")
        self.assertEqual(results[0].failure_id, "mismatch")
        self.assertIn("tl.dot", results[0].lessons[0])

    def test_search_by_context_hw_priority(self):
        """search_by_context 应优先 applicable_hardware 匹配的记录"""
        db = FailureDatabase()
        db.add(self._record("generic", fixed=True, hardware="mock_gpu"))
        db.add(self._record("nvidia_specific", fixed=True, hardware="mock_gpu",
                            applicable_hardware=["nvidia_gpu"]))
        results = db.search_by_context(kernel_type="matmul", dsl="python",
                                       hardware="nvidia_gpu", limit=2)
        self.assertEqual(results[0].failure_id, "nvidia_specific",
                         "应优先 applicable_hardware 命中 nvidia_gpu 的记录")

    def test_reflexion_fields_preserved_in_record(self):
        """Reflexion 新字段在 FailureRecord 构建中不丢失"""
        r = FailureRecord(
            failure_id="r1", kernel_type="matmul", hardware="nvidia_gpu", dsl="triton",
            level="L1", error_summary="name 'tl' is not defined",
            fix_summary="添加 import triton.language as tl", fixed=True,
            root_cause="LLM 生成代码遗漏了 import triton.language",
            lessons=["使用 tl.* 前必须先 import triton.language as tl"],
            error_keywords=["nameerror", "tl", "import"],
            applicable_hardware=["nvidia_gpu"],
        )
        self.assertTrue(r.fixed)
        self.assertIn("import triton.language", r.lessons[0])
        self.assertEqual(r.error_keywords[0], "nameerror")
        self.assertEqual(r.applicable_hardware, ["nvidia_gpu"])
        self.assertIn("遗漏了 import", r.root_cause)


class TestConductorFailureRecording(unittest.TestCase):

    def _conductor(self, llm, max_rounds=3) -> Conductor:
        return Conductor(backend=MockGPU(), kernel_db=KernelDatabase(),
                         llm=llm, max_optimize_rounds=max_rounds)

    def test_repair_success_recorded_as_fixed(self):
        def handler(prompt):
            return FENCED_VALID if FIXER_MARKER in prompt else BROKEN_MATMUL
        conductor = self._conductor(MockLLM(chat_handler=handler))
        artifact = conductor.execute(_make_task(), _make_inputs(), target_gflops=0.0)
        self.assertTrue(artifact.all_passed)
        records = conductor.failure_db.search(kernel_type="matmul", dsl="python")
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0].fixed)
        self.assertEqual(records[0].level, "L1")

    def test_budget_exhausted_recorded_as_unfixed(self):
        conductor = self._conductor(MockLLM(chat_handler=lambda p: BROKEN_MATMUL))
        artifact = conductor.execute(_make_task(), _make_inputs(), target_gflops=0.0)
        self.assertFalse(artifact.all_passed)
        records = conductor.failure_db.list_all()
        self.assertTrue(records)
        self.assertFalse(records[0].fixed)

    def test_iteration_history_lineage(self):
        def handler(prompt):
            return FENCED_VALID if FIXER_MARKER in prompt else BROKEN_MATMUL
        conductor = self._conductor(MockLLM(chat_handler=handler))
        artifact = conductor.execute(_make_task(), _make_inputs(), target_gflops=0.0)
        self.assertEqual(len(artifact.iteration_history), 2)
        self.assertEqual(artifact.iteration_history[0]["action"], "generate")
        self.assertEqual(artifact.iteration_history[1]["action"], "fix")
        self.assertEqual(artifact.iteration_history[1]["all_passed"], True)


# ─── P1-3: 编译缓存 ──────────────────────────────────────────

class CountingMockGPU(MockGPU):
    def __init__(self):
        super().__init__()
        self.compile_calls = 0

    def compile(self, kernel_code: str) -> bool:
        self.compile_calls += 1
        return super().compile(kernel_code)


class TestCompileCache(unittest.TestCase):

    def test_same_code_compiles_once(self):
        backend = CountingMockGPU()
        verifier = Verifier(backend)
        task = _make_task()
        inputs = _make_inputs()
        verifier.verify(PYTHON_MATMUL, task, inputs, target_gflops=0.0)
        results2 = verifier.verify(PYTHON_MATMUL, task, inputs, target_gflops=0.0)
        self.assertEqual(backend.compile_calls, 1)
        self.assertEqual(results2[0].metrics["compile_status"], "cached")
        self.assertTrue(results2[0].passed)


# ─── P1-3: 执行串行锁 ────────────────────────────────────────

class TestExecLock(unittest.TestCase):

    def test_execute_blocked_by_host_lock(self):
        executor = RemoteExecutor(mode="local")
        lock = _get_exec_lock(executor._lock_key())
        done = threading.Event()

        lock.acquire()
        t = threading.Thread(
            target=lambda: (executor.execute(PYTHON_MATMUL, _make_inputs()), done.set()),
            daemon=True,
        )
        t.start()
        time.sleep(0.3)
        self.assertFalse(done.is_set())   # 锁持有期间 execute 被阻塞
        lock.release()
        t.join(timeout=5)
        self.assertTrue(done.is_set())


# ─── P1-4: 多特征路由 ────────────────────────────────────────

class TestRouter(unittest.TestCase):

    def setUp(self):
        self.router = Router()
        self.db = KernelDatabase()

    def test_fused_ops_route_to_agent(self):
        task = _make_task(complexity=TaskComplexity.MEDIUM, fused=["matmul", "relu"])
        self.assertEqual(self.router.route(task, self.db), RouteType.AGENT_GENERATION)

    def test_low_complexity_translation(self):
        task = _make_task(complexity=TaskComplexity.LOW)
        self.assertEqual(self.router.route(task, self.db), RouteType.CODE_TRANSLATION)

    def test_medium_with_history_uses_backend(self):
        task = _make_task(dsl="triton", hw="nvidia_gpu", kernel="matmul",
                          complexity=TaskComplexity.MEDIUM)
        self.assertEqual(self.router.route(task, self.db), RouteType.COMPILER_BACKEND)

    def test_medium_without_history_uses_agent(self):
        task = _make_task(dsl="triton", hw="nvidia_gpu", kernel="exotic_op_xyz",
                          complexity=TaskComplexity.MEDIUM)
        self.assertEqual(self.router.route(task, self.db), RouteType.AGENT_GENERATION)

    def test_medium_no_db_backward_compatible(self):
        task = _make_task(complexity=TaskComplexity.MEDIUM)
        self.assertEqual(self.router.route(task), RouteType.COMPILER_BACKEND)

    def test_high_complexity_agent(self):
        task = _make_task(complexity=TaskComplexity.HIGH)
        self.assertEqual(self.router.route(task, self.db), RouteType.AGENT_GENERATION)


class TestSchemaValidation(unittest.TestCase):
    """对 schemas.py 新增的 validate() 方法进行单元测试 (P1 Schema 校验)

    覆盖: TilingStrategy, TritonCompileConfig, SourceSpec/TargetSpec/ConstraintSpec,
    TaskSchema, UnifiedSketch 的合法/非法边界。
    """

    # ---------- TilingStrategy ----------
    def test_tiling_legal_values(self):
        for m in (16, 32, 64, 128, 256, 512):
            for n in (16, 32, 64, 128, 256):
                for k in (16, 32, 64, 128):
                    self.assertEqual(
                        TilingStrategy(block_m=m, block_n=n, block_k=k).validate(),
                        [],
                        f"Tiling({m},{n},{k}) 应通过"
                    )

    def test_tiling_illegal_non_power(self):
        errs = TilingStrategy(block_m=100, block_n=100, block_k=32).validate()
        self.assertTrue(any("不在允许集合" in e for e in errs))

    def test_tiling_illegal_zero(self):
        errs = TilingStrategy(block_m=0, block_n=32, block_k=32).validate()
        self.assertTrue(any("必须为正整数" in e for e in errs))

    def test_tiling_illegal_k(self):
        errs = TilingStrategy(block_m=64, block_n=64, block_k=7).validate()
        self.assertTrue(any("block_k=7" in e for e in errs))

    # ---------- TritonCompileConfig ----------
    def test_triton_config_legal(self):
        self.assertEqual(TritonCompileConfig(num_warps=4, num_stages=3).validate(), [])

    def test_triton_config_non_power_warps(self):
        errs = TritonCompileConfig(num_warps=3, num_stages=2).validate()
        self.assertTrue(any("2 的幂" in e for e in errs))

    def test_triton_config_big_stages_warns(self):
        errs = TritonCompileConfig(num_warps=4, num_stages=9).validate()
        self.assertTrue(any("[WARNING]" in e for e in errs))
        # Warning 类不视为 hard error
        self.assertFalse(any("必须" in e and "2 的幂" not in e for e in errs))

    # ---------- Source/Target/Constraint ----------
    def test_source_unknown_kernel_warns_but_passes(self):
        s = SourceSpec(kernel_type="my_custom_op")
        errs = s.validate()
        # 仅 WARNING, 允许探索新算子
        self.assertTrue(all(e.startswith("[WARNING]") for e in errs), str(errs))

    def test_target_invalid_dsl_hard_error(self):
        t = TargetSpec(hardware="mock_gpu", dsl="ruby")
        errs = t.validate()
        self.assertTrue(any("不在允许集合" in e and "dsl" in e for e in errs))

    def test_constraints_negative_speedup(self):
        errs = ConstraintSpec(min_speedup=-1.0).validate()
        self.assertTrue(any("不能为负" in e and "min_speedup" in e for e in errs))

    # ---------- UnifiedSketch ----------
    def test_sketch_empty_algorithm_error(self):
        s = UnifiedSketch(algorithm="", tiling=TilingStrategy(64, 64, 32), target_dsl="python")
        errs = s.validate()
        self.assertTrue(any("algorithm 不能为空" in e for e in errs))

    def test_sketch_pipeline_instance_error(self):
        s = UnifiedSketch(
            algorithm="matmul", tiling=TilingStrategy(64, 64, 32), target_dsl="python"
        )
        s.pipeline = [{"stage": "load"}]  # 不是 PipelineStage 实例
        errs = s.validate()
        self.assertTrue(any("PipelineStage" in e for e in errs))

    def test_sketch_hint_not_str(self):
        s = UnifiedSketch(
            algorithm="softmax", tiling=TilingStrategy(64, 64, 32), target_dsl="triton"
        )
        s.optimization_hints = ["ok", 123]
        errs = s.validate()
        self.assertTrue(any("optimization_hints" in e for e in errs))

    def test_sketch_bad_target_dsl(self):
        s = UnifiedSketch(
            algorithm="softmax", tiling=TilingStrategy(64, 64, 32), target_dsl="golang"
        )
        errs = s.validate()
        self.assertTrue(any("target_dsl" in e for e in errs))

    # ---------- TaskSchema (全链路契约) ----------
    def test_task_contract_legal(self):
        task = _make_task()
        self.assertEqual(task.validate(), [])

    def test_task_contract_bad_target_compute_units(self):
        task = _make_task()
        task.target.compute_units = 0
        errs = task.validate()
        self.assertTrue(any("compute_units" in e for e in errs))

    def test_task_warning_only_passes(self):
        """新硬件+新算子只 Warning, 不阻塞"""
        task = _make_task(hw="amd_gpu", kernel="my_unknown_op")
        errs = task.validate()
        self.assertTrue(errs, "应存在 warnings")
        self.assertTrue(all(e.startswith("[WARNING]") for e in errs), str(errs))


if __name__ == "__main__":
    unittest.main(verbosity=2)
