"""_execute_inner 的行为不变量守卫 —— 拆分它之前的安全网。

为什么先要它：conductor._execute_inner 是 1620 行 / 232 个判定点 / 123 个局部变量的一块
逻辑，本仓最大。拆它之前必须先有一条判据能把「行为变了」与「结构变了」分开 —— 否则拆完
只剩「测试还是绿的」，而绿的测试并不知道自己该盯什么：4128 行里挑得出几十种拆法，每一种
都能让 2500+ 个测试保持绿。

判据是**一次确定性端到端运行的可观测行为摘要**（MockGPU + 固定 LLM + 关掉两个库），不是
「函数多长」。记下来的是正交事实各自的值：验证等级序列、轮次、命中的配方数、产物文件
清单、measurement 键集、路由决策、环境指纹、恢复状态。

刻意**不**记的两位，以及理由（不写清楚就会变成随机红）：
  * kernel_id —— 它按构造就是随机的（schemas.py:674 = str(uuid.uuid4())[:8]），
    不是被测对象的函数；
  * resume_status.root —— 一个临时目录路径。

失败时打印逐键 diff，而不是只丢一句 assertEqual 失败。

变异标定：只把 max_optimize_rounds 从 1 改成 2（真实的行为改变，结构没动）⇒ 本文件必须红。
这条标定写在文件里（test_the_guard_is_not_a_no_op），不靠文档记。

**覆盖边界（Stage1 标定时量出来的，写下来免得被当成全覆盖）**：本文件是**单轮**成功路径
的摘要，它**不**覆盖多轮分支 —— 停滞判定、升档、候选耗尽、预算退避。标定证据：把循环里
一处 `state.best_gflops` 漏改回裸局部（于是读到构造时的过期值，静默错），本文件 3 条全绿，
而 tests/ 全量红 18 条（test_conductor_stagnation / test_exhaustion / test_escalation /
test_budget_gate）。所以：改这段循环时本文件只是快速反馈，**每一段收口都必须再跑一次全量**。

运行: python -m pytest tests/unit/test_execute_inner_behavior.py -q
"""
import contextlib
import io
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from operator_agent import Conductor, KernelDatabase, MockGPU
from operator_agent.core.schemas import (ConstraintSpec, Precision, SourceSpec,
                                    TargetSpec, TaskComplexity, TaskSchema,
                                    TaskType)

#: 固定 LLM：三个温度都返回同一份能过 L1/L2/L3 的代码，于是整条链路可复现。
_CODE = "import numpy as np" + chr(10) + chr(10) + "def kernel(A, B):" + chr(10) + "    return A @ B" + chr(10)
_SKETCH = {"algorithm": "matmul",
           "tiling": {"block_m": 16, "block_n": 16, "block_k": 16},
           "memory_plan": {}, "pipeline": [], "optimization_hints": []}


class _Llm:
    def chat(self, messages, **kw):
        return _CODE

    def chat_json(self, messages, **kw):
        return dict(_SKETCH)


def _task():
    return TaskSchema(task_id="digest-1", task_type=TaskType.OPERATOR_GENERATION,
                      source=SourceSpec(kernel_type="matmul",
                                        complexity=TaskComplexity.MEDIUM),
                      target=TargetSpec(hardware="mock_gpu", dsl="python"),
                      constraints=ConstraintSpec(precision=Precision.FP32,
                                                 min_speedup=0.0, max_error=1e-3))


def digest(max_optimize_rounds=1) -> dict:
    """跑一遍确定性管线，返回可观测行为摘要。"""
    tmp = tempfile.TemporaryDirectory()
    try:
        c = Conductor(backend=MockGPU(), kernel_db=KernelDatabase(jsonl_path="off"),
                      llm=_Llm(), max_optimize_rounds=max_optimize_rounds)
        c.store.output_dir = Path(tmp.name)
        rng = np.random.default_rng(0)
        inputs = {"A": rng.standard_normal((32, 32)).astype(np.float32),
                  "B": rng.standard_normal((32, 32)).astype(np.float32)}
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            art = c.execute(_task(), inputs, None, target_gflops=0.0)
        out = buf.getvalue()
        resume = dict(art.resume_status)
        resume.pop("root", None)
        return {
            "route": re.findall(r"\[Router\] 路由决策: (\S+)", out),
            "round_headers": [list(x) for x in re.findall(r"轮次 (\d+)/(\d+)", out)],
            "recipe_counts": re.findall(r"\[Planner\] 选中 (\d+) 个优化配方", out),
            "verifier_events": [list(x) for x in
                                re.findall(r"\[Verifier\] (L\d) (PASSED|FAILED)", out)],
            "levels": [[str(getattr(r.level, "value", r.level)), bool(r.passed)]
                       for r in art.verification_results],
            "iterations": len(art.iteration_history),
            "language": art.language,
            "files": sorted(p.name for p in Path(art.saved_dir).iterdir()),
            "metadata_keys": sorted(art.metadata),
            "compile_keys": sorted(art.compile_info),
            "benchmark_keys": sorted(art.benchmark_data),
            "env_fingerprint": art.environment_fingerprint,
            "env_drift_keys": sorted(art.environment_drift),
            "resume_status": resume,
        }
    finally:
        tmp.cleanup()


#: 记录下来的行为。改这一份 = 承认行为变了，必须同时说清为什么。
GOLDEN = {   'benchmark_keys': [   'arithmetic_intensity',
                          'avg_time_ms',
                          'backend',
                          'backend_real_device',
                          'bandwidth_utilization_pct',
                          'baseline_reused',
                          'baseline_time_ms',
                          'baseline_type',
                          'benchmark_contaminated',
                          'bound_type',
                          'case_key',
                          'case_source',
                          'claim_level',
                          'contract_hash',
                          'correctness_verified',
                          'device',
                          'device_lease',
                          'device_lease_text',
                          'eligibility',
                          'gflops',
                          'gflops_utilization_pct',
                          'hw_compute_capability',
                          'hw_name',
                          'hw_peak_bandwidth_gbs',
                          'hw_peak_gflops',
                          'implementation_decision',
                          'language',
                          'max_time_ms',
                          'measurement_can_count_stagnation',
                          'measurement_can_disable_recipe',
                          'measurement_can_gate',
                          'measurement_capability_shortfall',
                          'measurement_environment_drift',
                          'measurement_intruder_suspected',
                          'measurement_l3_pass_refused',
                          'measurement_provenance_conflict',
                          'measurement_validity',
                          'measurement_validity_reasons',
                          'median_time_ms',
                          'memory_bandwidth_gbs',
                          'min_speedup_required',
                          'min_time_ms',
                          'mode',
                          'objective',
                          'observation',
                          'peak_probe',
                          'perf_criterion',
                          'performance_verdict',
                          'primary_metric',
                          'ridge_point',
                          'roofline_analysis',
                          'roofline_bound',
                          'run_binding',
                          'runs',
                          'shared_mem_per_sm_kb',
                          'speedup_vs_baseline',
                          'static_guards'],
    'compile_keys': ['resource_precheck', 'run_binding', 'static_guards', 'static_guards_lang'],
    'env_drift_keys': ['changed', 'comparable', 'drifted', 'previous_from'],
    'env_fingerprint': {'backend': 'MockGPU', 'name': 'MockGPU'},
    'files': [   'benchmark.json',
                 'code.py',
                 'confirmation.json',
                 'contract.json',
                 'execution_snapshot.json',
                 'report.md',
                 'result.json',
                 'run_manifest.json',
                 'sketch.json',
                 'verification.json'],
    'iterations': 1,
    'language': 'python',
    'levels': [['L1', True], ['L2', True], ['L3', True]],
    'metadata_keys': [   'bandit',
                         'budget',
                         'candidate_records',
                         'claim_level',
                         'corpus_admission',
                         'coverage',
                         'escalation',
                         'remeasure_binding',
                         'run_binding',
                         'selected_source_hash',
                         'selection_policy',
                         'stop'],
    'recipe_counts': ['9'],
    'resume_status': {   'candidates': 0,
                         'requested': False,
                         'status': 'not_requested',
                         'unreadable': 0},
    'round_headers': [['1', '1']],
    'route': ['compiler_backend'],
    'verifier_events': [   ['L1', 'PASSED'],
                           ['L2', 'PASSED'],
                           ['L3', 'PASSED'],
                           ['L3', 'PASSED']]}


def _diff(got: dict, want: dict) -> str:
    keys = sorted(set(got) | set(want))
    return chr(10).join("  %-18s golden=%r  got=%r" % (k, want.get(k), got.get(k))
                        for k in keys if got.get(k) != want.get(k))


class TestTheRunBehavesAsRecorded(unittest.TestCase):

    def test_the_digest_is_reproducible(self):
        """先证判据本身是确定的 —— 否则下面的不变量会被当成 flaky 忽略掉"""
        self.assertEqual(digest(), digest(), "摘要本身不可复现, 先修判据再谈不变量")

    def test_the_run_behaves_exactly_as_recorded(self):
        got = digest()
        if got != GOLDEN:
            self.fail("端到端行为与记录不符 (逐键 diff):" + chr(10) + _diff(got, GOLDEN))

    def test_the_guard_is_not_a_no_op(self):
        """反向标定: 真实的行为改变必须让判据变红, 而结构改变不该。"""
        other = digest(max_optimize_rounds=2)
        self.assertNotEqual(GOLDEN, other,
                            "改了优化轮数摘要却没变 ⇒ 这条判据没有鉴别力, 拆了也白拆")
        self.assertNotEqual(GOLDEN["round_headers"], other["round_headers"],
                            "轮数变了而轮次摘要没变: 判据漏掉了最该记的那一位")


if __name__ == "__main__":
    unittest.main(verbosity=2)
