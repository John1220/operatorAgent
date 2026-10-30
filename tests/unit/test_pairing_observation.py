"""§11-2 同卡配对检查的**喂给者** (CHECKLOG I47) —— 观测 → 标签 → 收口点

I47 的形状是「判据实现了、生产上没有喂给者」: `placeage.pairing_conflict` 与
`oracle.evaluate` 里的那道拒评都在, 但控制端从不传 `candidate_device`/`baseline_device`,
于是 §14-7「把候选和基线放到不同卡 ⇒ 配对检查拒绝该观测」在生产链路上**不会发生**。
更难看的是旁边那句话: `metrics["measurement_scope"]` 写死成 `"same_remote_device"` ——
键名就是 §11-2 要的那件事, 而背后没有任何观测 (全仓只有一个测试在读它)。

喂给者补上之后, 这一格的失效方式有三种, 都得有守卫:
  * 观测到**同一张卡** ⇒ 照旧 (不该因为多了个检查就改判决);
  * 观测到**不同卡** ⇒ 拒评 (资格全 False) **且** `passed` 不许立 (跨卡加速比没有指称对象);
  * **没有观测** ⇒ 也不许自称同卡 ("没得校" 不是 "校验过")。

另有两条性质值得单独钉住:
  * `visible_index` 不是卡号: `CUDA_VISIBLE_DEVICES=2` 下它是 0。只看 index 会把
    「=2 下的 0」与「=5 下的 0」读成同一张卡 —— 这正是 doc 63 那次事故的失效方式。
  * 标签与判决**不可能分叉**: 两者读同一对标签, 所以
    `bool(pairing_conflict(c, b)) == (scope == SCOPE_CROSS)` 必须是恒等式, 用性质测试钉。

运行: python3 -m unittest tests.test_pairing_observation -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from operator_agent.placement import (SCOPE_CROSS, SCOPE_SAME, SCOPE_UNOBSERVED,
                                     device_label, observed_pairing,
                                     pairing_conflict)
from operator_agent.schemas import (
    TaskSchema, SourceSpec, TargetSpec, ConstraintSpec,
    TaskType, TaskComplexity, Precision,
)
from operator_agent.verifier import Verifier
from operator_agent.mock_gpu import MockGPU
from tests.backend_fakes import fake_backend

PYTHON_MATMUL = """import numpy as np

def kernel(A, B):
    return A @ B
"""


def _ident(index=0, name="NVIDIA H20", visible="2", count=1):
    return {"visible_index": index, "name": name, "count": count,
            "cuda_visible_devices": visible}


# ─── A. 观测 → 标签 (纯函数, 成对) ──────────────────────────────

class TestDeviceLabel(unittest.TestCase):

    def test_the_three_fields_are_all_in_the_label(self):
        lab = device_label(_ident())
        self.assertIn("cuda:0", lab)
        self.assertIn("NVIDIA H20", lab)
        self.assertIn("visible=2", lab)

    def test_an_unusable_identity_is_none_not_empty_string(self):
        """空串会变成一次**恒真**的比较 —— 缺观测必须是 None (与 real_device=None 同边)"""
        for bad in (None, {}, {"visible_index": None}, "cuda:0", 42, []):
            self.assertIsNone(device_label(bad), f"{bad!r} 不该压出标签")

    def test_a_read_failure_is_not_device_zero(self):
        """远端读卡失败时报的是 None + 原因 —— 不许被读成「约定成 0 号卡」"""
        self.assertIsNone(device_label(_ident(index=None, name="读取失败: RuntimeError")))


class TestObservedPairing(unittest.TestCase):

    def test_same_device_is_same(self):
        c, b, scope = observed_pairing(_ident(), _ident())
        self.assertEqual(SCOPE_SAME, scope)
        self.assertEqual(c, b)

    def test_a_different_index_is_cross(self):
        _, _, scope = observed_pairing(_ident(index=0), _ident(index=1))
        self.assertEqual(SCOPE_CROSS, scope)

    def test_the_same_index_under_a_different_visible_map_is_cross(self):
        """doc 63 那个陷阱: index 是可见集内的序号, `=2` 下的 0 与 `=5` 下的 0 不是同一张卡"""
        _, _, scope = observed_pairing(_ident(visible="2"), _ident(visible="5"))
        self.assertEqual(SCOPE_CROSS, scope)

    def test_a_missing_observation_is_unobserved_and_has_no_labels(self):
        for pair in ((None, None), (_ident(), None), (None, _ident()),
                     (_ident(index=None), _ident(index=None))):
            c, b, scope = observed_pairing(*pair)
            self.assertEqual(SCOPE_UNOBSERVED, scope)
            self.assertIsNone(c)
            self.assertIsNone(b)

    def test_the_label_and_the_verdict_cannot_disagree(self):
        """性质测试: 标签相等 ⟺ `pairing_conflict` 不拒 —— 两处读同一对标签, 不许分叉"""
        idents = [_ident(), _ident(index=1), _ident(visible="5"),
                  _ident(name="NVIDIA A100"), _ident(index=None)]
        for ci in idents:
            for bi in idents:
                c, b, scope = observed_pairing(ci, bi)
                if scope == SCOPE_UNOBSERVED:
                    continue
                self.assertEqual(bool(pairing_conflict(c, b)), scope == SCOPE_CROSS,
                                 f"标签与判决分叉了: {c!r} vs {b!r}")


# ─── B. 生产链路: 假远端载荷 → L3 读数 ──────────────────────────

def _make_task(min_speedup=1.2) -> TaskSchema:
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.LOW),
        target=TargetSpec(hardware="mock_gpu", dsl="python",
                          compute_units=64, warp_size=32),
        constraints=ConstraintSpec(precision=Precision.FP32,
                                   min_speedup=min_speedup, max_error=1e-5),
    )


def _matmul_inputs(seed=0) -> dict:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((64, 64)).astype(np.float32)
    b = rng.standard_normal((64, 64)).astype(np.float32)
    return {"A": a, "B": b}


def _payload(arm_identity=None, paired_extra=None) -> dict:
    """远端 `benchmark_with_baselines` 的回报形状 (只留 L3 判定要用的键)"""
    paired = {"speedup_paired_median": 1.30, "speedup_paired_iqr_over_median": 0.02,
              "paired_runs": 30, "paired_block_samples": 5,
              "block_ratio_medians": [1.28, 1.31]}
    if arm_identity is not None:
        paired["arm_identity"] = arm_identity
    if paired_extra:
        paired.update(paired_extra)
    return {"success": True, "device": "cuda:0", "gflops": 1000.0,
            "memory_bandwidth_gbs": 500.0,
            # 候选 1.0ms vs 基线 1.6ms = 1.6x (>= 要求 1.2x): 这样「判失败」只可能
            # 来自配对那件事, 不会与速度门混在一起
            "candidate": {"min_time_ms": 1.0, "median_time_ms": 1.0,
                          "avg_time_ms": 1.02, "max_time_ms": 1.2, "runs": 30},
            "baselines": {"pytorch_eager_tf32": {"median_time_ms": 1.6,
                                                 "min_time_ms": 1.55}},
            "baseline_selected": "pytorch_eager_tf32",
            "paired": paired}


class TestThroughTheRealL3Path(unittest.TestCase):
    """走生产的 `_verify_performance` (与真机同一条路), 不是直接调 oracle"""

    def _run(self, payload):
        backend = fake_backend(benchmark_with_baselines=lambda *a, **k: payload)
        v = Verifier(backend)
        v._measure_baseline = lambda *_: self.fail("控制端 baseline 不得参与远程评测")
        return v._verify_performance(PYTHON_MATMUL, _matmul_inputs(), 0.0, _make_task())

    def test_an_observed_same_card_still_passes_and_says_so(self):
        r = self._run(_payload({"candidate": _ident(), "baseline": _ident(),
                                "after": _ident()}))
        self.assertEqual(SCOPE_SAME, r.metrics["measurement_scope"])
        self.assertTrue(r.passed, r.message)
        self.assertIn("同机同卡", r.message)
        # 观测原样落盘 (可审计), 不是只落一个结论位
        self.assertEqual(0, r.metrics["arm_identity"]["candidate"]["visible_index"])
        self.assertEqual({"candidate": device_label(_ident()),
                          "baseline": device_label(_ident())},
                         r.metrics["arm_device_labels"])
        self.assertTrue(r.metrics["eligibility"]["can_persist"],
                        "同卡观测不该动资格位")

    def test_a_cross_card_observation_voids_the_number(self):
        """§14-7 的判据: 配对检查拒绝该观测 (资格全 False) 且 passed 不许立"""
        r = self._run(_payload({"candidate": _ident(index=0), "baseline": _ident(index=3),
                                "after": _ident(index=3)}))
        self.assertEqual(SCOPE_CROSS, r.metrics["measurement_scope"])
        self.assertFalse(r.passed, "跨卡加速比没有指称对象, 不许判通过")
        self.assertEqual("uninformative", r.metrics["measurement_validity"])
        elig = r.metrics["eligibility"]
        self.assertTrue(all(v is False for v in elig.values()), elig)
        self.assertIn("pairing_conflict", r.metrics["observation"]["reasons"])
        self.assertIn("跨卡", r.metrics["observation"]["refused"])
        self.assertIn("作废", r.message)

    def test_without_an_observation_it_does_not_claim_the_same_card(self):
        """老载荷 (没有逐臂身份) ⇒ 「未观测」, 既不拒评也不自称同卡"""
        r = self._run(_payload(None))
        self.assertEqual(SCOPE_UNOBSERVED, r.metrics["measurement_scope"])
        self.assertEqual({"candidate": None, "baseline": None},
                         r.metrics["arm_device_labels"])
        self.assertTrue(r.passed, "缺观测不等于冲突 —— 不许拒评")
        self.assertIn("配对未观测", r.message)
        self.assertNotIn("同机同卡", r.message)

    def test_two_cpu_arms_are_paired_but_the_number_is_still_not_a_real_device(self):
        """CPU 上两臂确实在同一「设备」⇒ 配对成立; 「是不是真卡」是**另一条**判据

        这两件事共用一个键会让读者把「同 CPU」读成「在真卡上测过」—— 所以 scope 只回答
        配对, real_device 由 oracle 的规则 3 回答。
        """
        cpu = _ident(index=-1, name="cpu", visible="", count=0)
        r = self._run(_payload({"candidate": cpu, "baseline": cpu, "after": cpu}))
        self.assertEqual(SCOPE_SAME, r.metrics["measurement_scope"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

# ─── C. 喂给者的**源头**: 远端脚本真的按臂采了身份, 且带得回来 ──────────

def _load_remote_runner():
    """按路径加载 remote_scripts/run_kernel.py (该目录不是包)"""
    import importlib.util
    path = Path(__file__).resolve().parents[2] / "remote_scripts" / "run_kernel.py"
    spec = importlib.util.spec_from_file_location("run_kernel_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)      # 模块顶层不 import torch, 所以控制机上可加载
    return mod


class TestTheRemoteSideReportsIt(unittest.TestCase):
    """控制侧全绿而远端悄悄不报身份 = 喂给者在源头断了, 所以源头也要有牙"""

    def test_cpu_identity_is_explicit_not_a_guessed_zero(self):
        ident = _load_remote_runner()._arm_identity(False)
        self.assertEqual(-1, ident["visible_index"], "CPU 上的身份不许写成 0 号卡")
        self.assertEqual("cpu", ident["name"])
        self.assertIn("cuda_visible_devices", ident)

    def test_a_read_failure_never_invents_an_index(self):
        """没 CUDA 的机器上 `current_device()` 会抛 —— 那时必须是 None + 原因。

        这条在有 GPU 的机器上走成功分支, 所以断言写成「要么是真索引, 要么是 None+原因,
        永远不许是猜出来的值」—— 两边都成立, 都挡住了「读失败就写 0」。
        """
        ident = _load_remote_runner()._arm_identity(True)
        if str(ident["name"]).startswith("读取失败"):
            self.assertIsNone(ident["visible_index"])
        else:
            self.assertIsInstance(ident["visible_index"], int)

    def test_the_identity_is_read_per_arm_before_that_arm_is_timed(self):
        """顺序也是判据: 每臂各在自己**计时前**读一次 (同一个 while 迭代内)。

        只在开局读一次全局的话, 两臂之间若有人改了当前设备, 对比就看不见 —— 而
        「index 0 被当成卡号」正是 doc 63 那次事故的失效方式。
        """
        import ast
        src = (Path(__file__).resolve().parents[2] / "remote_scripts"
               / "run_kernel.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        fn = next(n for n in tree.body
                  if isinstance(n, ast.FunctionDef) and n.name == "_bench_pair_interleaved")
        body = ast.get_source_segment(src, fn)
        seq = ["cand_ident = _arm_identity(", "cand_times.extend(",
               "base_ident = _arm_identity(", "base_times.extend("]
        pos = [body.find(tok) for tok in seq]
        self.assertNotIn(-1, pos, f"远端没按臂采身份: {dict(zip(seq, pos))}")
        self.assertEqual(pos, sorted(pos),
                         f"采身份的时机不对 (必须在各臂计时之前): {dict(zip(seq, pos))}")

    def test_the_identity_survives_the_return_trip(self):
        """算出来还要带得回来: 放进 `paired` 并且 `paired` 必须在回报里 ——
        「值算出来了, 死在搬运路上」是这个仓最常见的一族缺陷。"""
        src = (Path(__file__).resolve().parents[2] / "remote_scripts"
               / "run_kernel.py").read_text(encoding="utf-8")
        self.assertIn('"arm_identity": {"candidate": cand_ident, "baseline": base_ident,',
                      src)
        self.assertIn('"paired": paired', src)
