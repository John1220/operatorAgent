"""L3 读数的**出处**终于有了读者 (doc 49 §3.5 第 2/3 处, 验收见 §五 第 4 片)

那一节点名的三处「算了没人读」里, 这里是后两处:

* **`"mode": "simulated"`** —— `MockGPU` 无 CUDA 时按硬件规格**估**出来的那份基准。
  搬运链路一直把它带进 metrics (`verifier.py` 的 extra_key 白名单), 而全仓读者数为 0。
  后果: 一份纯模拟的基准与一份真机实测的基准**在产物里权威性等价**。
* **`device == "cpu"`** —— 四个提供方会打这个标, 唯一的消费者是「选基线的显示名」
  (`cuBLAS(FP32)` vs `PyTorch Eager(CUDA)`)。没有任何地方拒绝记录一份在 CPU 上测出来
  的 L3 成绩。

接上的读者是 `measurement.py` 既有的那套词汇 (三档 Validity + 四态 Verdict), **不是**
另立一道门: 出处不对 ⇒ UNINFORMATIVE ⇒ `can_gate=False`, 于是配方禁用、停滞计数、
入库门 (`persist_decision`)、参考天花板出口一起停止把它当证据; 任务在
`target.required_caps` 里声明了 `REAL_DEVICE` 时再进一步 —— L3 不许 `passed=True`。

判据走 `Cap` 的**声明**, 不从 `device`/`name` 的字符串猜能力 (doc 49 §3.1 记过三处
名字匹配的毛病)。载荷自报的 `mode`/`device` 只用作与声明**对照**的第二手证据。

纯本地: 不需要 torch、不需要 GPU、不碰网络。

运行: python3 -m unittest tests.test_l3_provenance -v
"""

import ast
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np

from operator_agent.backend_caps import Cap
from operator_agent.measurement import (Validity, assess, persist_decision,
                                        provenance_conflicts)
from operator_agent.mock_gpu import MockGPU, _has_cuda
from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec, TargetSpec,
                                    TaskComplexity, TaskSchema, TaskType,
                                    VerificationLevel)
from operator_agent.verifier import Verifier
from tests.backend_fakes import DeclaredFakeBackend

PYTHON_MATMUL = """import numpy as np

def kernel(A, B):
    return A @ B
"""

# 真 Triton 形状 (wrapper 名叫 `kernel`, 契约检查与 MockGPU 的模拟执行都认它)。
# 用来驱动**真提供方** `MockGPU`: 没有 CUDA 时它按规格估性能并报 `mode="simulated"`,
# 正是 doc 49 §3.5 记的那个形状 —— 用真件而不是假件, 免得守卫测的是我的假件。
TRITON_MATMUL = """import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, BLOCK_M: tl.constexpr):
    acc = tl.zeros((BLOCK_M, BLOCK_M), dtype=tl.float32)
    acc += tl.dot(a_ptr, b_ptr)
    tl.store(c_ptr, acc)

def kernel(A, B):
    M, K = A.shape
    C = torch.empty((M, M), device=A.device, dtype=torch.float32)
    matmul_kernel[(1,)](A, B, C, BLOCK_M=64)
    return C
"""

H20_HW = {"name": "NVIDIA H20", "device": "cuda", "compute_capability": "9.0"}
# matmul 4096³ fp16 在 H20 上的真机净时间 (docs/research/32): 落在 `valid` 档,
# 于是「档位掉下来」只可能是出处那两条信号造成的, 不会与 launch 地板混在一起。
CANDIDATE_MS = 0.97992
BASELINE_MS = 1.10


def _inputs(n=64):
    rng = np.random.default_rng(0)
    return {"A": rng.standard_normal((n, n)).astype(np.float32),
            "B": rng.standard_normal((n, n)).astype(np.float32)}


def _bench(mode=None, device=None, candidate_ms=CANDIDATE_MS, baseline_ms=BASELINE_MS):
    """同机同卡基准协议的载荷形状 (远端只测一个基线并回报 `baseline_selected`)"""
    payload = {
        "success": True, "gflops": 140.0, "memory_bandwidth_gbs": 500.0,
        "candidate": {"min_time_ms": candidate_ms, "median_time_ms": candidate_ms,
                      "avg_time_ms": candidate_ms, "max_time_ms": candidate_ms,
                      "runs": 30},
        "baselines": {"pytorch_eager": {"median_time_ms": baseline_ms,
                                        "min_time_ms": baseline_ms}},
        "baseline_selected": "pytorch_eager",
        "paired": {},
    }
    if mode is not None:
        payload["mode"] = mode
    if device is not None:
        payload["device"] = device
    return payload


def _verifier(caps, payload, hw=None):
    """`DeclaredFakeBackend` 而不是 monkeypatch: 能力是声明出来的 (doc 49 §7.4)"""
    backend = DeclaredFakeBackend(
        caps=set(caps),
        benchmark_with_baselines=lambda *a, **kw: payload,
        get_hardware_info=lambda: dict(hw if hw is not None else H20_HW),
    )
    return Verifier(backend)


def _task(required=(), min_speedup=1.0, dsl="python", hardware="nvidia_gpu",
          kernel_type="matmul"):
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type=kernel_type, complexity=TaskComplexity.MEDIUM),
        target=TargetSpec(hardware=hardware, dsl=dsl, required_caps=tuple(required)),
        constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=min_speedup,
                                   max_error=1e-2),
    )


def _l3(verifier, task=None, code=PYTHON_MATMUL, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = verifier._verify_performance(code, _inputs(), 0.0, task or _task(), **kw)
    return r, buf.getvalue()


REAL = {Cap.REAL_DEVICE}
NOT_REAL = {Cap.EMULATED, Cap.SIMULATED_PERF}


# ─── 一、判据本体 ────────────────────────────────────────────────

class TestProvenanceConflicts(unittest.TestCase):
    """`provenance_conflicts` 的两个方向: 声明层缺口 / 声明与载荷打架"""

    def test_a_required_cap_the_backend_lacks_is_a_shortfall(self):
        shortfall, conflicts = provenance_conflicts(NOT_REAL, {Cap.REAL_DEVICE})
        self.assertEqual(("real_device",), shortfall)
        self.assertEqual((), conflicts, "缺能力不是「自相矛盾」, 两个方向不许混")

    def test_no_shortfall_when_the_declaration_covers_the_requirement(self):
        self.assertEqual((), provenance_conflicts(REAL, {Cap.REAL_DEVICE})[0])

    def test_no_requirement_means_no_shortfall(self):
        """默认不要求真卡: 本地 mock 运行本来就没有真卡, 无条件要求等于把离线链路判死"""
        self.assertEqual((), provenance_conflicts(NOT_REAL, frozenset())[0])

    def test_simulated_payload_contradicts_a_real_device_claim(self):
        """声明里同时给 REAL_DEVICE 与 SIMULATED_PERF, 好把这一条**单独**隔离出来

        只给 REAL_DEVICE 的话会同时触发另一条 (载荷说在模拟、后端却没声明
        `SIMULATED_PERF` = 未声明的降级), 两条都是真缺陷, 但要分开钉。
        """
        _, conflicts = provenance_conflicts(REAL | {Cap.SIMULATED_PERF}, frozenset(),
                                            reported_mode="simulated")
        self.assertEqual(1, len(conflicts))
        self.assertIn("real_device", conflicts[0])
        self.assertIn("simulated", conflicts[0])

    def test_both_directions_fire_at_once_when_both_defects_are_present(self):
        _, conflicts = provenance_conflicts(REAL, frozenset(), reported_mode="simulated")
        self.assertEqual(2, len(conflicts),
                         "声明与载荷打架 + 模拟没被声明, 是两件不同的事, 都要说出来")

    def test_cpu_payload_contradicts_a_real_device_claim(self):
        _, conflicts = provenance_conflicts(REAL, frozenset(), reported_device="cpu")
        self.assertEqual(1, len(conflicts))
        self.assertIn("cpu", conflicts[0])

    def test_simulated_payload_without_the_declaration_is_an_undeclared_downgrade(self):
        """反向: 载荷老实说了在模拟, 后端却没声明 `SIMULATED_PERF` (doc 49 §3.3 那一族)"""
        _, conflicts = provenance_conflicts(frozenset(), frozenset(),
                                            reported_mode="simulated")
        self.assertEqual(1, len(conflicts))
        self.assertIn("simulated_perf", conflicts[0])

    def test_a_conservative_declaration_is_not_a_contradiction(self):
        """没声明 REAL_DEVICE、载荷写 cpu —— 两边一致, 不许报矛盾

        这条是**反向对照**: 把它写成「device 里有 cpu 就报」的话, 本地每一次 mock
        运行都会被标成自相矛盾, 信号当场失效 (与 measurement 里 util>5% 只取 util、
        不取「有别的进程」是同一个理由)。
        """
        shortfall, conflicts = provenance_conflicts(NOT_REAL, frozenset(),
                                                    reported_mode="actual",
                                                    reported_device="cpu")
        self.assertEqual(((), ()), (shortfall, conflicts))

    def test_device_is_matched_exactly_not_by_substring(self):
        """`"cuda (fallback)"` / `"cpu0"` 这类串一律不认

        认了子串就等于把能力判断又搬回字符串匹配 (doc 49 §3.1 记的三处名字匹配)。
        """
        for device in ("cuda (fallback)", "cpu0", "CUDA", "xpu", "", None):
            _, conflicts = provenance_conflicts(REAL, frozenset(), reported_device=device)
            self.assertEqual((), conflicts, f"device={device!r} 不该被判成矛盾")

    def test_absent_payload_fields_are_not_evidence(self):
        """载荷没报 mode/device 时不许猜 (远端 `benchmark_with_baselines` 就不报 mode)"""
        self.assertEqual(((), ()), provenance_conflicts(REAL, frozenset()))


class TestTheJudgeIsNotVacuous(unittest.TestCase):
    """已知答案对照: 判据必须能认出它被造出来要认的那个形状

    没有这一组, 一个恒返回 `((), ())` 的实现会让上面所有 `assertEqual((), …)`
    **永真地绿着** —— 而那个绿正是全部安全感的来源。
    """

    def test_the_documented_defect_shape_is_caught(self):
        """doc 49 §3.5 第 2 处的原形状: MockGPU 无 CUDA + 任务要真卡

        响的是**缺口**那一路而不是矛盾那一路 —— 因为 MockGPU 的声明与载荷是一致的
        (它老实说了自己在模拟: `SIMULATED_PERF` ↔ `mode="simulated"`)。两个方向能
        分开, 正是「声明层缺口」与「声明与载荷打架」分成两个字段的全部理由。
        """
        shortfall, conflicts = provenance_conflicts(
            NOT_REAL, {Cap.REAL_DEVICE}, reported_mode="simulated",
            reported_device="mock (MockGPU)")
        self.assertEqual(("real_device",), shortfall)
        self.assertEqual((), conflicts)

    def test_local_executor_shape_is_caught(self):
        """doc 49 §3.5 第 3 处的原形状: `LocalExecutor` 声明空集, 在控制机 CPU 上真测"""
        shortfall, _ = provenance_conflicts(frozenset(), {Cap.REAL_DEVICE},
                                            reported_device="cpu")
        self.assertEqual(("real_device",), shortfall)

    def test_a_real_device_measurement_is_left_alone(self):
        """方向对照: 真机那份必须干净, 否则判据只是「一律说不行」"""
        self.assertEqual(((), ()),
                         provenance_conflicts(REAL, {Cap.REAL_DEVICE},
                                              reported_mode="cuda",
                                              reported_device="cuda:2"))


class TestAssessGradesProvenance(unittest.TestCase):
    """出处信号进的是既有那套三档, 不是另立的一套"""

    def test_shortfall_is_uninformative_and_cannot_gate(self):
        a = assess(median_time_ms=CANDIDATE_MS, launch_floor_us=20.24,
                   backend_caps=frozenset(NOT_REAL), required_caps={Cap.REAL_DEVICE})
        self.assertIs(Validity.UNINFORMATIVE, a.validity)
        self.assertFalse(a.can_gate)
        self.assertFalse(a.can_disable_recipe)
        self.assertFalse(a.can_count_stagnation)
        self.assertTrue(a.l3_pass_is_refused)

    def test_contradiction_is_uninformative(self):
        a = assess(median_time_ms=CANDIDATE_MS, launch_floor_us=20.24,
                   backend_caps=frozenset(REAL), reported_mode="simulated")
        self.assertIs(Validity.UNINFORMATIVE, a.validity)
        self.assertIn("自相矛盾", " ".join(a.reasons))

    def test_contradiction_alone_does_not_refuse_the_pass(self):
        """任务没要真卡时, 矛盾只降级、不改判: 把观测当判据是另一条错路"""
        a = assess(median_time_ms=CANDIDATE_MS, launch_floor_us=20.24,
                   backend_caps=frozenset(REAL), reported_device="cpu")
        self.assertIs(Validity.UNINFORMATIVE, a.validity)
        self.assertFalse(a.l3_pass_is_refused)

    def test_clean_provenance_stays_valid(self):
        a = assess(median_time_ms=CANDIDATE_MS, launch_floor_us=20.24,
                   backend_caps=frozenset(REAL), required_caps={Cap.REAL_DEVICE},
                   reported_mode="cuda", reported_device="cuda:2")
        self.assertIs(Validity.VALID, a.validity)
        self.assertTrue(a.can_gate)
        self.assertFalse(a.l3_pass_is_refused)
        self.assertIs(True, a.as_metrics()["backend_real_device"])

    def test_no_capability_information_is_not_rejudged(self):
        """`backend_caps=None` (老路径 / 直接调 assess) 与「声明了空集」必须分开

        缺证据不等于无效, 但必须显式说出来 —— 与 launch 地板未知那条同一处置。
        """
        a = assess(median_time_ms=CANDIDATE_MS, launch_floor_us=20.24)
        self.assertIs(Validity.VALID, a.validity)
        self.assertIsNone(a.backend_real_device)
        self.assertNotIn("backend_real_device", a.as_metrics(),
                         "缺键与 False 是两件事: 老产物不该被追溯性重判")
        b = assess(backend_caps=None, required_caps={Cap.REAL_DEVICE})
        self.assertIs(Validity.VALID, b.validity, "没给能力信息时不猜")
        self.assertIn("未做出处检查", " ".join(b.reasons),
                      "但必须留一句「没检查」, 否则会被读成「检查过了」")

        c = assess(median_time_ms=CANDIDATE_MS, launch_floor_us=20.24,
                   backend_caps=frozenset(), required_caps={Cap.REAL_DEVICE})
        self.assertIs(Validity.UNINFORMATIVE, c.validity,
                      "而**声明了**空集就是要判 —— 两者混起来门就没了")

    def test_worse_signal_wins_when_provenance_and_dispersion_both_fire(self):
        a = assess(median_time_ms=CANDIDATE_MS, launch_floor_us=20.24, dispersion=0.2,
                   backend_caps=frozenset(NOT_REAL), required_caps={Cap.REAL_DEVICE})
        self.assertIs(Validity.UNINFORMATIVE, a.validity)
        self.assertTrue(a.l3_pass_is_refused)


class TestCorpusGateReadsIt(unittest.TestCase):
    """`persist_decision` 是 `backend_real_device` 的读者: 模拟成绩进不了跨运行语料"""

    BASE = {"measurement_can_gate": True, "benchmark_contaminated": False,
            "speedup_vs_baseline": 1.35, "measurement_validity": "valid"}

    def test_simulated_numbers_are_refused(self):
        keep, why = persist_decision(dict(self.BASE, backend_real_device=False))
        self.assertFalse(keep)
        self.assertIn("REAL_DEVICE", why)

    def test_real_device_numbers_are_still_admitted(self):
        keep, _ = persist_decision(dict(self.BASE, backend_real_device=True))
        self.assertTrue(keep, "方向对照: 门不能写成「一律不入库」")

    def test_old_artifacts_without_the_key_are_not_rejudged(self):
        keep, _ = persist_decision(dict(self.BASE))
        self.assertTrue(keep, "缺键走 `is False`, 老产物不被追溯性重判")

    def test_missing_speedup_is_not_reported_as_no_speedup(self):
        """2026-09-29 真机报表里 `speedup=None` 被印成「相对同机基线没有加速」——
        那是把**没有读数**说成**读数说不快**。拒收照旧, 但理由必须是缺陷信号而不是判定。
        """
        for bad in (None, "n/a", True):
            keep, why = persist_decision(dict(self.BASE, speedup_vs_baseline=bad))
            self.assertFalse(keep, f"非法 speedup={bad!r} 不能当合格样本入库")
            self.assertIn("没有可用的 speedup 读数", why, why)
            self.assertNotIn("没有加速", why, f"不能把缺读数讲成判定: {why}")

    def test_a_measured_slow_run_is_still_called_a_verdict(self):
        """方向对照: 拆成三态之后, 「测到了、确实不快」那条理由不能丢"""
        keep, why = persist_decision(dict(self.BASE, speedup_vs_baseline=0.97))
        self.assertFalse(keep)
        self.assertIn("相对同机基线没有加速 (speedup=0.97)", why)


# ─── 二、Verifier 的三条 L3 返回路径 ──────────────────────────────

class TestVerifierRefusesSimulatedL3(unittest.TestCase):
    """§五 第 4 片的验收条件: `mode == "simulated"` 的 L3 不许 `passed=True`"""

    def test_simulated_l3_does_not_pass_when_the_task_declared_real_device(self):
        v = _verifier(NOT_REAL, _bench(mode="simulated", device="mock (MockGPU)"))
        r, _ = _l3(v, _task(required=(Cap.REAL_DEVICE,)))
        self.assertFalse(r.passed, r.message)
        self.assertFalse(r.metrics["measurement_can_gate"])
        self.assertEqual(Validity.UNINFORMATIVE.value, r.metrics["measurement_validity"])
        self.assertIn("real_device", r.metrics["measurement_validity_reasons"][0]
                      if r.metrics["measurement_validity_reasons"] else "")

    def test_the_eligibility_published_by_the_closure_point_is_also_all_false(self):
        """任务要真卡 ⇒ **收口点发布的资格**也必须全 False, 不只是 `passed=False`

        ⚠️ 覆盖缺口 (2026-10-08 变异标定抓到): 上面几条钉的都是 `passed` / `validity`,
        而「拒绝通过」与「资格不许自称有资格」是两件事 —— 后者才决定这个数能不能进
        长期记忆 (`metrics["eligibility"]` 正是入库判据读的那一份, I46/I91 那一族)。
        实测: 把收口点的 `require_real_device=Cap.REAL_DEVICE.value in required` 硬写成
        `False`, **全量 2582 条一条都不红**; 补上这条之后那条变异变红。
        """
        v = _verifier(NOT_REAL, _bench(mode="simulated", device="mock (MockGPU)"))
        r, _ = _l3(v, _task(required=(Cap.REAL_DEVICE,)))
        self.assertFalse(r.passed, r.message)
        elig = r.metrics["eligibility"]
        self.assertTrue(elig and all(x is False for x in elig.values()),
                        "收口点没把资格压成全 False: %s" % (elig,))
        # 触发这条规则的是**喂进收口点的两个输入** (只认这一对): 要求为真、事实为假。
        # 钉住它们 —— `require_real_device=False` 这种「把要求抹掉」的写法会当场变红
        # (它不是拒评: "拒评" 是 refused 非空, 这里 observation 照产, 只是资格位不许为真)。
        ri = r.metrics["observation"]["provenance"]["rule_inputs"]
        self.assertIs(True, ri["require_real_device"])
        self.assertIs(False, ri["real_device"])

    def test_the_same_numbers_pass_on_a_backend_that_declares_real_device(self):
        """方向对照: 只有换掉能力声明, 其余一字不改

        没有这一条, 「一律不通过」也能让上面那条绿 —— 那正是判据方向写反的样子。
        """
        payload = _bench(mode="cuda", device="cuda:2")
        refused = _l3(_verifier(NOT_REAL, payload), _task(required=(Cap.REAL_DEVICE,)))[0]
        accepted = _l3(_verifier(REAL, payload), _task(required=(Cap.REAL_DEVICE,)))[0]
        self.assertFalse(refused.passed)
        self.assertTrue(accepted.passed, accepted.message)
        self.assertEqual(Validity.VALID.value, accepted.metrics["measurement_validity"])

    def test_cpu_measured_l3_does_not_pass_when_the_task_declared_real_device(self):
        """doc 49 §3.5 第 3 处: 没人拒绝记录一份在 CPU 上测出来的 L3 —— 现在有了"""
        v = _verifier(frozenset(), _bench(device="cpu"))
        r, _ = _l3(v, _task(required=(Cap.REAL_DEVICE,)))
        self.assertFalse(r.passed, r.message)

    def test_a_real_device_claim_contradicted_by_the_payload_is_refused(self):
        """声明有真卡、载荷却说这一轮在 CPU 上测的: 任务要的是真卡 ⇒ 不记通过

        真有其事: `MockGPU` 有 CUDA 时声明 `REAL_DEVICE`, 而跑 python 核走的是
        `_benchmark_python`, 载荷老实写着 `device="cpu"`。
        """
        v = _verifier(REAL, _bench(mode="actual", device="cpu"))
        r, _ = _l3(v, _task(required=(Cap.REAL_DEVICE,)))
        self.assertFalse(r.passed, r.message)
        self.assertIn("自相矛盾", r.message)

    def test_the_message_says_why_it_was_refused(self):
        """message 会进 Fixer / SearchAgent 的 prompt, 是唯一保证被下游看到的通道"""
        v = _verifier(NOT_REAL, _bench(mode="simulated"))
        r, out = _l3(v, _task(required=(Cap.REAL_DEVICE,)))
        self.assertIn("不记为通过", r.message)
        self.assertIn("real_device", r.message)
        self.assertIn("FAILED", out)

    def test_metrics_record_that_the_numbers_are_not_from_a_real_device(self):
        """产物里两份读数不再权威性等价: 这个键就是它们的权威性标记"""
        v = _verifier(NOT_REAL, _bench(mode="simulated"))
        r, _ = _l3(v, _task())
        self.assertIs(False, r.metrics["backend_real_device"])
        self.assertEqual("simulated", r.metrics["mode"], "载荷自报的 mode 仍在产物里")

    def test_no_requirement_keeps_the_old_verdict_but_records_the_provenance(self):
        """没声明要真卡 ⇒ 照旧判 (离线链路不能被判死), 但产物里记清楚出处"""
        v = _verifier(NOT_REAL, _bench(mode="simulated"))
        r, _ = _l3(v, _task())
        self.assertTrue(r.passed, r.message)
        self.assertIs(False, r.metrics["backend_real_device"])

    def test_the_absolute_gflops_fallback_path_is_gated_too(self):
        """第三条返回路径 (无基线, 按绝对 GFLOPS 判) 同样要过门

        `_verify_performance` 有三条 return, 只 gate 其中一条的话, 走到另一条的模拟
        读数照样记成通过 —— 「N 个同型分支只覆盖了一个」那一族。
        """
        backend = DeclaredFakeBackend(
            caps=set(NOT_REAL),
            get_hardware_info=lambda: dict(H20_HW),
            benchmark=lambda *a, **kw: {"avg_time_ms": CANDIDATE_MS,
                                        "min_time_ms": CANDIDATE_MS,
                                        "gflops": 500.0, "mode": "simulated"},
            execute=lambda code, inputs: _exec_result(),
        )
        v = Verifier(backend)
        # 无 baseline 参考的算子 → 落到 legacy 绝对目标那一支
        r, _ = _l3(v, _task(required=(Cap.REAL_DEVICE,), kernel_type="my_custom_op"),
                   code=PYTHON_MATMUL)
        self.assertFalse(r.passed, r.message)
        self.assertIn("无 baseline 参考", r.message)


def _exec_result():
    from operator_agent.mock_gpu import ExecutionResult
    return ExecutionResult(output=np.zeros((64, 64), dtype=np.float32),
                           execution_time_ms=CANDIDATE_MS, gflops=500.0,
                           memory_bandwidth_gbs=100.0, device="cpu")


# ─── 三、拒绝这件事在产物里读得出来 ────────────────────────────────

class TestRefusalIsReadableInMetrics(unittest.TestCase):
    """`capability_shortfall` / `provenance_conflict` / 拒绝位此前只活在内存里的
    `MeasurementAssessment` 上, **从没进 `metrics`** —— 也就是进不了 `result.json`。
    下游想区分「核太慢所以没过」与「压根没测所以没资格过」就只能去 string-match
    message 里那个 ⛔。这是 doc 49 §3.5 那一族的最后一个现场: 这一片修「算了没人读」,
    自己又留了一个算了没人读的。

    发现方式值得记一次: 是在真机上**试图读这几个键**时撞出来的
    (`benchmarks/verify_provenance_gate.py`), 不是读代码读出来的。
    """

    def test_a_refusal_is_readable_without_string_matching_the_mark(self):
        v = _verifier(NOT_REAL, _bench(mode="simulated"))
        r, _ = _l3(v, _task(required=(Cap.REAL_DEVICE,)))
        self.assertFalse(r.passed, r.message)
        self.assertTrue(r.metrics["measurement_l3_pass_refused"])
        self.assertEqual(["real_device"], r.metrics["measurement_capability_shortfall"])

    def test_a_slow_kernel_fails_without_raising_the_refusal_flag(self):
        """控制变量: 同一份核、能力齐全, 只是门槛高 —— 那必须是「慢」而不是「没测」"""
        v = _verifier(REAL, _bench())
        r, _ = _l3(v, _task(required=(Cap.REAL_DEVICE,), min_speedup=5.0))
        self.assertFalse(r.passed, "这份载荷的加速比约 1.12x, 5.0x 门槛应当没过")
        self.assertFalse(r.metrics["measurement_l3_pass_refused"],
                         "慢不是出处问题 —— 拒绝位必须区分这两件事")
        self.assertEqual([], r.metrics["measurement_capability_shortfall"])

    def test_the_keys_are_absent_when_no_capability_info_was_given(self):
        """缺键 ≠ 空列表: 没做过出处检查时不许下游读成「检查过、干净的」"""
        a = assess(median_time_ms=CANDIDATE_MS)   # 不传 backend_caps
        m = a.as_metrics()
        self.assertNotIn("measurement_capability_shortfall", m)
        self.assertNotIn("measurement_l3_pass_refused", m)
        self.assertNotIn("backend_real_device", m)

    def test_drift_reaches_metrics_as_a_list(self):
        a = assess(median_time_ms=CANDIDATE_MS,
                   environment_drift=["torch_version: '2.14' → '2.15'"])
        self.assertEqual(["torch_version: '2.14' → '2.15'"],
                         a.as_metrics()["measurement_environment_drift"])


# ─── 四、真提供方: 不用假件的那一组 ────────────────────────────────

@unittest.skipIf(_has_cuda(), "本机有 CUDA 时 MockGPU 声明 REAL_DEVICE, 这组前提不成立")
class TestRealProviderMockGPU(unittest.TestCase):
    """守卫不许只测我自己的假件 (doc 48 §6.5 同源的那条教训)

    `MockGPU` 在控制机上就是 doc 49 §3.5 记的那个形状: 无 CUDA ⇒ 声明
    `{EMULATED, SIMULATED_PERF}`, 而 triton 核的 `benchmark()` 按硬件规格估、
    报 `"mode": "simulated"`。这一组走的是**产品代码**。
    """

    def test_mockgpu_declares_simulated_perf_and_reports_the_matching_mode(self):
        """前提核对: 提供方那两半仍然对得上, 否则下面两条用例失去意义"""
        gpu = MockGPU()
        self.assertIn(Cap.SIMULATED_PERF, gpu.capabilities)
        self.assertNotIn(Cap.REAL_DEVICE, gpu.capabilities)
        bench = gpu.benchmark(TRITON_MATMUL, _inputs(32))
        self.assertEqual("simulated", bench["mode"])

    def test_a_simulated_triton_benchmark_never_passes_a_real_device_task(self):
        task = _task(required=(Cap.REAL_DEVICE,), hardware="mock_gpu", dsl="triton",
                     min_speedup=0.0)
        r, _ = _l3(Verifier(MockGPU()), task, code=TRITON_MATMUL)
        self.assertEqual("simulated", r.metrics["mode"])
        self.assertFalse(r.passed, r.message)
        self.assertFalse(r.metrics["measurement_can_gate"])

    def test_the_same_run_without_the_requirement_still_passes(self):
        """方向对照: 拒绝来自**任务的要求**, 不是来自「一律不信 mock」"""
        task = _task(hardware="mock_gpu", dsl="triton", min_speedup=0.0)
        r, _ = _l3(Verifier(MockGPU()), task, code=TRITON_MATMUL)
        self.assertTrue(r.passed, r.message)

    def test_python_kernels_on_mockgpu_are_tagged_cpu(self):
        """第 3 处的另一半: python 核在 MockGPU 上就是 CPU 计时 (`mode="actual"`)"""
        bench = MockGPU().benchmark(PYTHON_MATMUL, _inputs(32))
        self.assertEqual("cpu", bench["device"])
        self.assertEqual("actual", bench["mode"])


# ─── 四、Conductor 真的停止把它当证据 ─────────────────────────────

class _ScriptedLLM:
    SKETCH = {"algorithm": "matmul",
              "tiling": {"block_m": 64, "block_n": 64, "block_k": 32},
              "memory_plan": {}, "pipeline": [], "optimization_hints": []}

    def __init__(self, code=PYTHON_MATMUL):
        self._code = code

    def chat(self, messages, **kw):
        return self._code

    def chat_json(self, messages, **kw):
        return dict(self.SKETCH)


class TestConductorStopsTreatingItAsEvidence(unittest.TestCase):
    """端到端: 一份模拟读数不得驱动配方禁用 / 停滞计数 / 入库, 也不得记成任务成功"""

    maxDiff = 600

    def _run(self, required, rounds=3, min_speedup=5.0):
        from operator_agent.conductor import Conductor
        from operator_agent.kernel_db import KernelDatabase
        backend = DeclaredFakeBackend(
            caps=set(NOT_REAL),
            verify_function=lambda *a, **kw: {"success": True, "max_error": 0.0,
                                              "compared_elements": 4096},
            benchmark_with_baselines=lambda *a, **kw: _bench(mode="simulated",
                                                             device="mock (MockGPU)"),
            get_hardware_info=lambda: dict(H20_HW),
        )
        # 独占一个产物目录: `Conductor` 现在会把本次指纹与**同 (算子, 硬件) 的上一次
        # 运行**比较, 而整个测试会话共用 conftest 那一个 tmp 目录 —— 不隔开的话,
        # 本文件写的 matmul/nvidia_gpu 产物会让别的文件 (以及别的文件让它) 报出一次
        # 与用例无关的环境漂移。
        with tempfile.TemporaryDirectory() as out_dir:
            with mock.patch.dict(os.environ, {"OPERATOR_AGENT_OUTPUT_DIR": out_dir}):
                conductor = Conductor(backend=backend, kernel_db=KernelDatabase(),
                                      llm=_ScriptedLLM(), max_optimize_rounds=rounds)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    art = conductor.execute(_task(required=required,
                                                  min_speedup=min_speedup),
                                            _inputs(64), target_gflops=0.0)
        return art, buf.getvalue()

    def test_a_simulated_run_is_never_recorded_as_passed(self):
        art, out = self._run(required=(Cap.REAL_DEVICE,))
        self.assertFalse(art.all_passed)
        self.assertEqual(3, len({h["round"] for h in art.iteration_history}),
                         "没有一轮记成通过 ⇒ 循环该跑满预算, 而不是第 1 轮就 break")
        self.assertNotIn("所有验证通过", out)

    def test_a_simulated_run_never_disables_a_recipe_or_counts_stagnation(self):
        art, out = self._run(required=(Cap.REAL_DEVICE,))
        # ⚠️ 不能断言 `assertNotIn("禁用配方", out)`: 缺口 1 那句处置说明本身就叫
        # 「不禁用配方 / 不计入停滞轮次」, 子串会把「机制生效了」读成「机制失效了」。
        # 断言的是**唯一会禁用配方的那条路径**没走 (性能退步 = SLOW), 与停滞出口没开。
        self.assertNotIn("性能退步", out)
        self.assertNotIn("提前终止", out)
        self.assertIn("不禁用配方 / 不计入停滞轮次", out,
                      "缺口 1 那句话得照样打出来: 判据换了来源, 处置没变")
        self.assertTrue(all(h["measurement_validity"] == Validity.UNINFORMATIVE.value
                            for h in art.iteration_history))

    def test_the_refusal_reason_reaches_the_persisted_artifact(self):
        art, _ = self._run(required=(Cap.REAL_DEVICE,))
        l3 = next(r for r in art.verification_results
                  if r.level == VerificationLevel.L3_PERFORMANCE)
        self.assertIs(False, l3.metrics["backend_real_device"])
        self.assertIn("不记为通过", l3.message)
        self.assertIn("real_device", " ".join(l3.metrics["measurement_validity_reasons"]))

    def test_without_the_requirement_the_loop_is_not_refused(self):
        """方向对照: 同一份模拟载荷, 任务没要真卡 ⇒ 不因出处拒评 (停滞另说)

        ⚠️ 这里原来断言的是「照旧提前终止」。§10.4 的第二半 (CHECKLOG I64) 之后,
        代码型路径的停滞**无法确立** ⇒ 不再只因轮数停, 而这个用例要对照的本来就是
        「拒评那一条没有触发」, 不是「停在第几轮」—— 所以观察面换成「跑满轮次且没有
        拒评的那条消息」, 判据本身一个字没动。
        """
        _, out = self._run(required=(), rounds=4)
        self.assertNotIn("拒评", out)
        self.assertIn("覆盖**无法确立**", out)


# ─── 五、AST 守卫: 三条返回路径都过了门 ──────────────────────────

def _strip_docstrings(tree):
    """把 docstring 换成 `Pass` (保住行号), 于是判据只看**可执行语句**

    必须剥: 说明文字里恰恰会引用被禁的写法来解释为什么不能再那么写, 而判据若连
    说明文字一起读就会假阳性 —— 假阳性的样子是「测试变红」, 看起来像真抓到了什么,
    于是下一个人会去改说明文字而不是改判据 (doc 49 §7.2 / tests/unit/test_backend_caps.py)。
    """
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(body, list) and body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            body[0] = ast.Pass()
    return tree


def _l3_gate_report(trees):
    """判据本体: 每个「L3 + `passed=passed`」的 return, 前面有没有过 `_refuse_l3_pass`

    `trees` 可换成一段已知答案的代码 —— 对照必须走**同一条**判据, 否则标定的是判据的
    副本, 不是判据。
    """
    report = []
    for label, tree in trees:
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            gates, returns = [], []
            for node in ast.walk(fn):
                if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Attribute)
                        and node.value.func.attr == "_refuse_l3_pass"):
                    gates.append(node.lineno)
                if not (isinstance(node, ast.Return) and isinstance(node.value, ast.Call)):
                    continue
                kw = {k.arg: k.value for k in node.value.keywords}
                level = kw.get("level")
                is_l3 = (isinstance(level, ast.Attribute)
                         and level.attr == "L3_PERFORMANCE")
                passed_is_bare = isinstance(kw.get("passed"), ast.Name)
                if is_l3 and passed_is_bare and kw["passed"].id == "passed":
                    returns.append(node.lineno)
            for lineno in sorted(returns):
                report.append((f"{label}:{fn.name}", lineno,
                               any(g < lineno for g in gates)))
    return report


_PRODUCTION = [(p.name, _strip_docstrings(ast.parse(p.read_text(encoding="utf-8"))))
               for p in sorted((REPO / "operator_agent").glob("*.py"))]


class TestEveryL3ReturnPathIsGated(unittest.TestCase):
    """`_verify_performance` 有三条 L3 返回路径, 少 gate 一条就漏一条"""

    def test_every_l3_return_in_production_is_gated(self):
        report = _l3_gate_report(_PRODUCTION)
        self.assertGreaterEqual(len(report), 3,
                                "认出的 L3 返回点少于 3 个 —— 判据在空转, 全绿没有意义")
        self.assertEqual([], [f"{name}:{line} 没过 _refuse_l3_pass"
                              for name, line, gated in report if not gated])

    def test_the_guard_sees_an_ungated_return(self):
        """已知答案对照 (走同一条判据): 没有它, 上面那条在判据写错时安静地绿"""
        src = ("class V:\n"
               "    def f(self):\n"
               "        passed = True\n"
               "        return VerificationResult(\n"
               "            level=VerificationLevel.L3_PERFORMANCE,\n"
               "            passed=passed, message='', metrics={})\n"
               "    def g(self):\n"
               "        passed = True\n"
               "        passed = self._refuse_l3_pass(passed, a)\n"
               "        return VerificationResult(\n"
               "            level=VerificationLevel.L3_PERFORMANCE,\n"
               "            passed=passed, message='', metrics={})\n")
        report = _l3_gate_report([("probe", _strip_docstrings(ast.parse(src)))])
        self.assertEqual([("probe:f", 4, False), ("probe:g", 10, True)], report)

    def test_the_guard_is_not_fooled_by_a_docstring(self):
        """说明文字里写着 `passed = self._refuse_l3_pass(...)` **不算**过了门

        这条是「剥 docstring」那一步的已知答案对照: 前提 (说明文字里确实引用了被要求的
        写法) 一旦不成立, 这条对照就该退休 —— 与 test_backend_caps.py 里那条同型。
        """
        src = ('class V:\n'
               '    def f(self):\n'
               '        """改这里之前先读: passed = self._refuse_l3_pass(passed, a)"""\n'
               '        passed = True\n'
               '        return VerificationResult(\n'
               '            level=VerificationLevel.L3_PERFORMANCE,\n'
               '            passed=passed, message="", metrics={})\n')
        raw = ast.parse(src)
        self.assertIn("_refuse_l3_pass", ast.unparse(raw),
                      "前提变了: 说明文字里已经不再引用这道门, 这条对照失去意义")
        report = _l3_gate_report([("probe", _strip_docstrings(raw))])
        self.assertEqual([("probe:f", 5, False)], report)

    def test_a_non_l3_return_is_not_counted(self):
        """反向对照: 不能写成「什么都拦」—— L1/L2 的 return 与这道门无关"""
        src = ("class V:\n"
               "    def f(self):\n"
               "        return VerificationResult(\n"
               "            level=VerificationLevel.L2_FUNCTION,\n"
               "            passed=passed, message='')\n")
        self.assertEqual([], _l3_gate_report([("probe", _strip_docstrings(ast.parse(src)))]))


class TestReadersAreNotJustMentions(unittest.TestCase):
    """doc 49 §3.5 是手工 grep 出来的; 这一组把「有读者」变成机械守卫

    判据只认 `.get("<字段>")` / `["<字段>"]` 这两种**读**的形式。搬运用的白名单
    (`for extra_key in ("mode", "device", …)`) 是元组里的字面量, 不算读者 —— 它正是
    那个「搬了但没人读」的现场。
    """

    @staticmethod
    def _read_sites(field, trees=None):
        sites = []
        for label, tree in (trees if trees is not None else _PRODUCTION):
            for node in ast.walk(tree):
                if isinstance(node, ast.Subscript):
                    key = node.slice
                    if isinstance(key, ast.Constant) and key.value == field:
                        sites.append(f"{label}:{node.lineno}")
                elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                      and node.func.attr == "get" and node.args
                      and isinstance(node.args[0], ast.Constant)
                      and node.args[0].value == field):
                    sites.append(f"{label}:{node.lineno}")
        return sites

    @staticmethod
    def _call_sites(name, trees=None):
        out = []
        for label, tree in (trees if trees is not None else _PRODUCTION):
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    fn = node.func
                    ident = fn.id if isinstance(fn, ast.Name) else (
                        fn.attr if isinstance(fn, ast.Attribute) else "")
                    if ident == name:
                        out.append(f"{label}:{node.lineno}")
        return out

    def test_mode_has_a_reader_outside_the_transport_whitelist(self):
        sites = self._read_sites("mode")
        self.assertTrue(sites, "`mode` 又变成零读者了 (doc 49 §3.5 第 2 处)")
        self.assertTrue(any(s.startswith("verifier.py") for s in sites), sites)

    def test_device_has_more_readers_than_just_the_display_name(self):
        """改前 `device` 在 verifier 里只被读来选显示名 (两处 startswith), 现在 >= 3"""
        sites = [s for s in self._read_sites("device") if s.startswith("verifier.py")]
        self.assertGreaterEqual(len(sites), 3,
                                f"`device` 的读者又只剩显示名了: {sites}")

    def test_the_refusal_flag_has_a_production_reader(self):
        """写进 metrics 却没人读 = 又造一个 §3.5。Conductor 的汇总必须读它"""
        sites = [s for s in self._read_sites("measurement_l3_pass_refused")
                 if s.startswith("conductor.py")]
        self.assertTrue(sites,
                        "`measurement_l3_pass_refused` 又变成零读者了 (doc 49 §3.5)")

    def test_the_fingerprint_helpers_are_called_by_the_conductor(self):
        for name in ("fingerprint_of", "compare_fingerprints", "previous_environment"):
            sites = [s for s in self._call_sites(name) if s.startswith("conductor.py")]
            self.assertTrue(sites, f"{name} 没有调用方 —— 又一处「算了没人读」")

    def test_the_reader_detector_is_not_vacuous(self):
        """已知答案对照: 拿一个**确定有读者**的字段喂同一条判据

        没有这条, 一个把 region 找错、于是恒返回空列表的判据会让上面三条
        `assertTrue` 全部变红得莫名其妙, 而 `assertGreaterEqual(…, 3)` 之外的
        断言根本分不清「判据坏了」与「机制被删了」。
        """
        self.assertTrue(self._read_sites("gflops"), "判据连 gflops 都认不出来 ⇒ 它在空转")
        self.assertTrue(self._call_sites("assess"))
        src = 'def f(metrics):\n    "docstring: metrics.get(\'mode\') 没人读"\n    return 0\n'
        self.assertEqual([], self._read_sites("mode", [("probe", _strip_docstrings(ast.parse(src)))]),
                         "说明文字里的 `.get('mode')` 不该被当成读者")


# ─── 契约哈希: 记录携带的才叫「校验过」 (spine-review P0-2) ──────

class TestContractHashIsRecordCarried(unittest.TestCase):
    """拆掉恒真: 载荷带来的旧契约哈希才比对, 没带就明说「没得校」

    原先 `verifier.py` 写的是 `metrics.setdefault("contract_hash", stamp)` 再把
    同一个值当 `record_hash` 回喂 —— 比较的两侧都由那一行自己造, `assert_hash`
    恒真, 「唯一收口点」名存实亡。现在两侧来源分开:
      * 载荷带的旧哈希 → 真比对 (不匹配 → 拒评);
      * 没带           → `fresh=True`, 结论记 `fresh_unverified`。
    """

    def _stale(self, **kw):
        payload = _bench(mode="cuda", device="cuda:2", **kw)
        payload["contract_hash"] = "0" * 64          # 记录携带的旧契约, 与当前不符
        return payload

    def test_stale_carried_hash_refuses_and_the_conclusion_becomes_readable(self):
        """过期契约 → 拒评, 且 `contract_hash_mismatch` 真的被写出来

        这个键此前**没有任何生产写入者**, 而 `confirm.py` 的 UnableReport 一直在读
        它 —— 于是「契约被改写, 本次评测作废」永远不会出现在报告里 (死读键)。
        """
        r, _ = _l3(_verifier(REAL, self._stale()),
                   _task(required=(Cap.REAL_DEVICE,)))
        obs = r.metrics["observation"]
        self.assertEqual("mismatch", obs["hash_check"])
        self.assertTrue(obs["refused"])
        self.assertFalse(obs["eligibility"]["can_verdict"],
                         "契约不匹配还能判决 —— 收口没接上")
        self.assertIs(True, r.metrics["contract_hash_mismatch"],
                      "结论没落到 confirm.py 读的那个键上 ⇒ 报告里永远看不到它")

    def test_no_carried_hash_is_marked_fresh_not_verified(self):
        """载荷没带旧哈希 = 新观测: 照常可判决, 但结论写明**没有**校验过

        生产路径上每个 L3 读数都是新观测 —— 这里若按拒评处理, 整条闭环直接死;
        所以要紧的不是拒, 而是不许把「没得校」说成「校验通过」。
        """
        r, _ = _l3(_verifier(REAL, _bench(mode="cuda", device="cuda:2")),
                   _task(required=(Cap.REAL_DEVICE,)))
        obs = r.metrics["observation"]
        self.assertEqual("fresh_unverified", obs["hash_check"])
        self.assertFalse(obs["refused"])
        self.assertTrue(r.passed, r.message)
        self.assertEqual(Validity.VALID.value, r.metrics["measurement_validity"])
        self.assertNotIn("contract_hash_mismatch", r.metrics)

    def test_a_carried_hash_matching_the_contract_is_verified(self):
        """带的旧哈希与当前契约一致 → 这才是真的校验过 (方向对照)"""
        from operator_agent.contract import freeze_from_task
        payload = _bench(mode="cuda", device="cuda:2")
        task = _task(required=(Cap.REAL_DEVICE,))
        payload["contract_hash"] = freeze_from_task(task).hash
        r, _ = _l3(_verifier(REAL, payload), task)
        self.assertEqual("verified", r.metrics["observation"]["hash_check"])
        self.assertNotIn("contract_hash_mismatch", r.metrics)

    def test_the_mismatch_reaches_the_unable_to_establish_report(self):
        """生产者→消费者: verifier 写的键正是报告在读的那个键

        两边各自的单测都在 (verifier 写 / confirm 读), 但**跨文件**的词汇漂移只有
        这样一条端到端断言才拦得住 —— 而这个键此前根本没有生产者, 于是报告里那条
        「契约被改写, 本次评测作废」永远不会出现。
        """
        from operator_agent.confirm import UnableToEstablish
        r, _ = _l3(_verifier(REAL, self._stale()),
                   _task(required=(Cap.REAL_DEVICE,)))
        u = UnableToEstablish()
        u.extend_metrics(r.metrics)
        self.assertIn("contract", {i.kind for i in u.items},
                      f"契约作废没进报告: {u.render()}")

    def test_the_stamp_is_always_the_live_contract(self):
        """溯源戳永远写**当前**契约, 载荷带的那个另存在 observation 里

        两者混作一个键, 产物里就读不出「这数在哪个尺子下取得」与「它自称是哪个
        尺子下取得的」的区别 —— 而这条不符正是要报出来的东西。
        """
        from operator_agent.contract import freeze_from_task
        task = _task(required=(Cap.REAL_DEVICE,))
        live = freeze_from_task(task).hash
        r, _ = _l3(_verifier(REAL, self._stale()), task)
        self.assertEqual(live, r.metrics["contract_hash"])
        self.assertEqual("0" * 64,
                         r.metrics["observation"]["provenance"]["record_hash"])


# ─── 尺子: 心跳的结论必须落到报告读的那两个键上 (spine-review P1-5) ──

def _hb(ok, window="hb-1", slow=3.0, reasons=()):
    from operator_agent.calibration import HeartbeatResult
    return HeartbeatResult(ok=ok, epoch=1, reasons=tuple(reasons),
                           known_good_ratio=1.0, known_slow_ratio=slow,
                           window_id=window)


class TestRulerVerdictReachesTheReport(unittest.TestCase):
    """`ruler_broken` / `ruler_reason` 此前是**死读键** (`confirm.py` 的「该窗口内所有
    数字不可用」与 `harness_pass_rate` 都在读, 全仓没有生产写入者)。这条把它接上。
    """

    def _reading(self, task=None, **kw):
        return _l3(_verifier(REAL, _bench(mode="cuda", device="cuda:2")),
                   task or _task(required=(Cap.REAL_DEVICE,)), **kw)[0]

    def test_a_broken_ruler_invalidates_the_reading_and_reaches_the_report(self):
        from operator_agent.confirm import UnableToEstablish
        r = self._reading(heartbeat=_hb(False, slow=1.0,
                                        reasons=["RULER_BROKEN: 分离度不足"]),
                          heartbeat_expected=True)
        self.assertIs(True, r.metrics["ruler_broken"])
        self.assertIn("RULER_BROKEN", r.metrics["ruler_reason"])
        self.assertFalse(r.metrics["observation"]["eligibility"]["can_verdict"],
                         "尺子坏了这条读数仍然可判决")
        u = UnableToEstablish()
        u.extend_metrics(r.metrics)
        self.assertIn("ruler", {i.kind for i in u.items}, u.render())

    def test_a_healthy_ruler_says_it_was_checked(self):
        """「检查过且好」要写成 False, 不能缺席 —— 缺席与 False 是两件事"""
        r = self._reading(heartbeat=_hb(True), heartbeat_expected=True)
        self.assertIs(False, r.metrics["ruler_broken"])
        self.assertTrue(r.metrics["observation"]["eligibility"]["can_verdict"])

    def test_due_but_missing_is_written_as_broken(self):
        """到期没跑 ≠ 没到期: 前者必须写 True, 后者一个键都不写"""
        due = self._reading(heartbeat=None, heartbeat_expected=True)
        self.assertIs(True, due.metrics["ruler_broken"])
        self.assertIn("心跳", due.metrics["ruler_reason"])
        not_due = self._reading()
        self.assertNotIn("ruler_broken", not_due.metrics,
                         "没检查的读数写了个 ruler_broken —— 那是「检查过且好」的位子")


if __name__ == "__main__":
    unittest.main(verbosity=2)
