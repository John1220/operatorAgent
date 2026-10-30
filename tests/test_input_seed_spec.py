"""按 seed 物化输入（L2 传描述而不是传字节）的守卫

背景（评估文档 §9.8.1）：L2 的输入原先整份 base64 过线，fp16 4096³ 两个张量 ≈ 90 MB 文本，
实测链路 ~0.28 MB/s ⇒ 一次 L2 光传输就 570 s。现在控制端只传 seed/shape/dtype + digest，
远端按同一规则物化后**再核一次** digest；核不过就退回传字节。

这里钉四件事：

1. **两侧物化逐字节相同**（同一个实现，`bench_cases` 委托给 `remote_input_fidelity`）；
2. **digest 是真的在核**：改一个数、换一个 dtype、换一个 seed 都要变；
3. **控制端只在「完全一致」时才允许替换字节**（`Verifier._seed_spec_for`）；
4. **远端核不过要回 `input_seed_mismatch`，控制端必须回退传字节**（不静默使用近似值）。

运行: python3 -m pytest tests/test_input_seed_spec.py -q
"""

import os
import unittest

import numpy as np

from operator_agent.bench_cases import BenchmarkCase, TensorSpec, case_spec, cases_from_shapes
from operator_agent.remote_input_fidelity import (SEED_PROTOCOL, materialize_seed_inputs,
                                                  seed_input_digest)
from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec, TargetSpec,
                                    TaskSchema, TaskType)


def _task(shapes=((64, 64, 64),), precision=Precision.FP16, kernel_type="matmul"):
    return TaskSchema(task_type=TaskType.OPERATOR_GENERATION,
                      source=SourceSpec(kernel_type=kernel_type),
                      target=TargetSpec(hardware="nvidia_gpu", dsl="triton"),
                      constraints=ConstraintSpec(precision=precision,
                                                 shapes=shapes, min_speedup=0.5))


class TestBothSidesMaterializeTheSameBytes(unittest.TestCase):

    def test_bench_case_delegates_to_the_shared_materializer(self):
        case = cases_from_shapes("matmul", (((32, 32, 4), 1.0),), "fp16")[0]
        a = case.materialize()
        b = materialize_seed_inputs(case.seed_spec())
        self.assertEqual(set(a), set(b))
        for name in a:
            self.assertEqual(a[name].dtype, b[name].dtype, name)
            self.assertEqual(a[name].shape, b[name].shape, name)
            self.assertEqual(a[name].tobytes(), b[name].tobytes(), name)

    def test_digest_covers_values_dtype_and_shape(self):
        case = cases_from_shapes("matmul", (((8, 8, 4), 1.0),), "fp16")[0]
        base = case.materialize()
        spec = case_spec(case)
        self.assertEqual(seed_input_digest(base), spec["digest"])

        changed = {k: v.copy() for k, v in base.items()}
        changed["A"][0, 0] = changed["A"][0, 0] + 1
        self.assertNotEqual(seed_input_digest(changed), spec["digest"])

        retyped = {k: v.astype(np.float32) for k, v in base.items()}
        self.assertNotEqual(seed_input_digest(retyped), spec["digest"])

    def test_another_seed_is_another_digest(self):
        one = cases_from_shapes("matmul", (((8, 8, 4), 1.0),), "fp16", seed=1)[0]
        two = cases_from_shapes("matmul", (((8, 8, 4), 1.0),), "fp16", seed=2)[0]
        self.assertNotEqual(case_spec(one)["digest"], case_spec(two)["digest"])

    def test_the_wire_payload_is_tiny(self):
        import json
        spec = case_spec(cases_from_shapes("matmul", (((4096, 4096, 4096), 1.0),), "fp16")[0])
        self.assertEqual(spec["protocol"], SEED_PROTOCOL)
        self.assertLess(len(json.dumps(spec)), 512, "描述必须只有几百字节")


class TestControlSideOnlyReplacesBytesWhenIdentical(unittest.TestCase):

    def _verifier(self):
        from operator_agent.mock_gpu import MockGPU
        from operator_agent.verifier import Verifier
        return Verifier(MockGPU())

    def test_matching_inputs_get_a_spec(self):
        task = _task()
        case = cases_from_shapes("matmul", (((64, 64, 64), 1.0),), "fp16")[0]
        inputs = case.materialize()
        spec = self._verifier()._seed_spec_for(task, inputs)
        self.assertIsNotNone(spec)
        self.assertEqual(spec["protocol"], SEED_PROTOCOL)
        self.assertEqual(spec["seed"], case.seed)

    def test_rows_case_without_affine_inputs_gets_a_spec(self):
        """layernorm 的内建 profile 只发 X；不能因为 rows-case 声明了 weight/bias 就回退字节。"""
        task = _task(shapes=((64, 32),), precision=Precision.FP32, kernel_type="layernorm")
        case = cases_from_shapes("layernorm", (((64, 32), 1.0),), "fp32")[0]
        inputs = {"X": case.materialize()["X"]}
        spec = self._verifier()._seed_spec_for(task, inputs)
        self.assertIsNotNone(spec)
        self.assertEqual(spec["protocol"], SEED_PROTOCOL)
        self.assertEqual([t["name"] for t in spec["tensors"]], ["X"])

    def test_rows_case_with_affine_inputs_still_gets_a_spec(self):
        task = _task(shapes=((64, 32),), precision=Precision.FP32, kernel_type="layernorm")
        case = cases_from_shapes("layernorm", (((64, 32), 1.0),), "fp32")[0]
        spec = self._verifier()._seed_spec_for(task, case.materialize())
        self.assertIsNotNone(spec)
        self.assertEqual(spec["protocol"], SEED_PROTOCOL)

    def test_rows_case_changed_value_still_disables_the_shortcut(self):
        task = _task(shapes=((64, 32),), precision=Precision.FP32, kernel_type="layernorm")
        case = cases_from_shapes("layernorm", (((64, 32), 1.0),), "fp32")[0]
        inputs = {"X": case.materialize()["X"].copy()}
        inputs["X"][0, 0] += 1.0
        self.assertIsNone(self._verifier()._seed_spec_for(task, inputs))

    def test_a_single_changed_value_disables_the_shortcut(self):
        task = _task()
        inputs = cases_from_shapes("matmul", (((64, 64, 64), 1.0),), "fp16")[0].materialize()
        inputs["A"][0, 0] += 0.5
        self.assertIsNone(self._verifier()._seed_spec_for(task, inputs))

    def test_scalars_disable_the_shortcut(self):
        task = _task()
        inputs = cases_from_shapes("matmul", (((64, 64, 64), 1.0),), "fp16")[0].materialize()
        inputs["scale"] = 2.5                      # 标量：spec 表示不了
        self.assertIsNone(self._verifier()._seed_spec_for(task, inputs))

    def test_no_declared_shapes_means_no_shortcut(self):
        task = _task(shapes=())
        inputs = cases_from_shapes("matmul", (((64, 64, 64), 1.0),), "fp16")[0].materialize()
        self.assertIsNone(self._verifier()._seed_spec_for(task, inputs))

    def test_the_decision_is_cached(self):
        task = _task()
        inputs = cases_from_shapes("matmul", (((64, 64, 64), 1.0),), "fp16")[0].materialize()
        v = self._verifier()
        first = v._seed_spec_for(task, inputs)
        self.assertIn("A=64x64|B=64x64", v._seed_spec_cache)
        self.assertEqual(v._seed_spec_for(task, inputs), first)


class TestTheTransportFallsBackToBytes(unittest.TestCase):
    """真机过一次 90 MB 太贵；这里用替身把两条分支都跑到。"""

    def _executor(self, responses):
        from operator_agent.remote_executor import SSHExecutor
        ex = SSHExecutor.__new__(SSHExecutor)
        ex.platform = "nvidia"
        ex._persistent = False
        ex._deployed = True
        ex._platform = "nvidia"
        ex.calls = []

        def fake(payload):
            ex.calls.append(payload)
            return responses[min(len(ex.calls) - 1, len(responses) - 1)]

        ex._ssh_exec_json = fake
        return ex

    def _inputs(self):
        return cases_from_shapes("matmul", (((32, 32, 4), 1.0),), "fp16")[0].materialize()

    def test_seeded_request_carries_no_bytes(self):
        spec = case_spec(cases_from_shapes("matmul", (((32, 32, 4), 1.0),), "fp16")[0])
        ex = self._executor([{"success": True, "max_error": 0.0}])
        out = ex.verify_function("def kernel(): pass", self._inputs(), "matmul", 1e-5,
                                 precision="fp16", input_seed=spec)
        self.assertTrue(out["success"])
        self.assertEqual(len(ex.calls), 1)
        self.assertIn("inputs_spec", ex.calls[0])
        self.assertNotIn("inputs", ex.calls[0], "给了 spec 就**不许**再传字节")

    def test_a_digest_mismatch_falls_back_to_bytes_and_leaves_a_trace(self):
        spec = case_spec(cases_from_shapes("matmul", (((32, 32, 4), 1.0),), "fp16")[0])
        ex = self._executor([{"success": False, "input_seed_mismatch": True},
                             {"success": True, "max_error": 0.0}])
        out = ex.verify_function("def kernel(): pass", self._inputs(), "matmul", 1e-5,
                                 precision="fp16", input_seed=spec)
        self.assertTrue(out["success"])
        self.assertTrue(out.get("input_seed_fallback"), "回退这件事必须留在回包里")
        self.assertEqual(len(ex.calls), 2)
        self.assertNotIn("inputs", ex.calls[0])
        self.assertIn("inputs", ex.calls[1], "第二次必须是字节")

    def test_benchmark_also_sends_a_description_not_bytes(self):
        """L3 那条路（benchmark_with_baselines）走同一套：否则钱只是从 L2 挪到 L3。"""
        spec = case_spec(cases_from_shapes("matmul", (((32, 32, 4), 1.0),), "fp16")[0])
        ex = self._executor([{"success": True, "candidate": {"min_time_ms": 1.0}}])
        out = ex.benchmark_with_baselines("def kernel(): pass", self._inputs(), "matmul",
                                          precision="fp16", input_seed=spec)
        self.assertTrue(out["success"])
        self.assertEqual(len(ex.calls), 1)
        self.assertIn("inputs_spec", ex.calls[0])
        self.assertNotIn("inputs", ex.calls[0])

    def test_benchmark_falls_back_and_records_it(self):
        spec = case_spec(cases_from_shapes("matmul", (((32, 32, 4), 1.0),), "fp16")[0])
        ex = self._executor([{"success": False, "input_seed_mismatch": True},
                             {"success": True}])
        out = ex.benchmark_with_baselines("def kernel(): pass", self._inputs(), "matmul",
                                          precision="fp16", input_seed=spec)
        self.assertTrue(out["success"])
        self.assertTrue(out.get("input_seed_fallback"))
        self.assertEqual(len(ex.calls), 2)
        self.assertIn("inputs", ex.calls[1])

    def test_without_a_spec_it_behaves_as_before(self):
        ex = self._executor([{"success": True}])
        ex.verify_function("def kernel(): pass", self._inputs(), "matmul", 1e-5,
                           precision="fp16")
        self.assertEqual(len(ex.calls), 1)
        self.assertIn("inputs", ex.calls[0])
        self.assertNotIn("inputs_spec", ex.calls[0])


if __name__ == "__main__":
    unittest.main()


class TestProtocolIdentitySurvivesAPackagedLayout(unittest.TestCase):
    """`protocol_identity()` 必须只看**包目录**，不看 cwd、也不假设仓库根。

    这条是「打包之后还认得出自己」的前提：它哈希的协议源码里既有
    `operator_agent/*`，也有 `remote_scripts/*` —— 前者来自包，后者来自那个
    `remote_scripts/__init__.py` 撑起来的兄弟包。任何一处路径假设错了，续跑对账
    都会静默失效（换一台机器读同一份产物，身份对不上）。
    """

    def test_it_hashes_package_relative_files_from_any_cwd(self):
        import os
        import tempfile
        from operator_agent import run_contract

        here = run_contract.protocol_identity()
        cwd = os.getcwd()
        try:
            os.chdir(tempfile.mkdtemp(prefix="pid-check-"))
            elsewhere = run_contract.protocol_identity()
        finally:
            os.chdir(cwd)
        self.assertEqual(here, elsewhere, "换 cwd 之后身份变了 —— 说明路径不是包相对的")
        self.assertIn("remote_scripts/run_kernel.py", here)
        self.assertIn("operator_agent/verifier.py", here)

    def test_every_hashed_file_lives_under_the_package_parent(self):
        from pathlib import Path
        from operator_agent import run_contract

        root = Path(run_contract.__file__).resolve().parent.parent
        self.assertTrue((root / "operator_agent").is_dir())
        self.assertTrue((root / "remote_scripts").is_dir())
        identity = run_contract.protocol_identity()
        self.assertEqual(len(identity), 73)
        self.assertIn('operator_agent/s5_verify_client.py', identity)
        self.assertTrue({"operator_agent/generation_spec.py", "operator_agent/prompt_manager.py",
                         "operator_agent/fixer.py", "operator_agent/repair_loop.py",
                         "operator_agent/repair_stride_diagnoser.py", "operator_agent/repair_scalar_diagnoser.py",
                         "operator_agent/compile_evidence.py", "operator_agent/generation_context.py",
                         "operator_agent/task_budget.py", "operator_agent/autonomous.py",
                         "operator_agent/tool_evidence.py", "operator_agent/hitl_tools.py",
                         "operator_agent/report.py", "operator_agent/run_layout.py",
                         "operator_agent/task_cancellation.py", "operator_agent/tool_calls.py",
                         "operator_agent/task_clock.py", "operator_agent/candidates.py",
                         "operator_agent/coder_pipeline.py", "operator_agent/triton_matmul_template.py",
                         "operator_agent/agents.py", "operator_agent/execution_audit.py",
                         "operator_agent/backend_ownership.py", "operator_agent/s5_shared_budget.py",
                         "operator_agent/s5_inner_budget.py"} <= set(identity))
        self.assertTrue(all((root / name).is_file() for name in identity))
