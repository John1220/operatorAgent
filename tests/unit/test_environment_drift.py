"""环境指纹终于有了读者 (doc 49 §3.5 第 1 处「算了没人读」)

`conductor.py` 一直算着一个 `fingerprint` (name/device/cc/torch/triton), 旁边那行注释
写着「用于**环境漂移自检**与结果追溯」, 然后 `if fingerprint: print(...)` 就没了 ——
全仓 3 次命中全在那 6 行里, 不入库、不比较、不持久化。**没有环境漂移自检**, 那句注释
描述的是一个不存在的机制。

接上之后它有三个读者:

1. `result.json` / `execution_snapshot.json` —— 落盘。读者是**下一次运行**: 不落盘就
   无从比较。
2. 日志 —— 没漂移时也报一句, 否则「机制没跑」与「跑了没问题」在日志里长得一样。
3. `measurement.assess(environment_drift=…)` —— 判据。按项目「观测先于判定」的处置:
   **只降到 DEGRADED** (不禁用配方、不计停滞), 不据此判核失败, 也**没有阈值** ——
   「torch 从 2.14 变成 2.15 该不该让一次运行作废」这个问题, 本机还没有分布能回答。

纯本地: 不需要 torch、不需要 GPU、不碰网络。

运行: python3 -m unittest tests.test_environment_drift -v
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np

from operator_agent.core.artifact_store import ArtifactStore
from operator_agent.core.backend_caps import Cap
from operator_agent.core.environment import (FINGERPRINT_BACKEND_KEY, FINGERPRINT_KEYS,
                                        EnvironmentDrift, compare_fingerprints,
                                        fingerprint_of)
from operator_agent.decide.measurement import Validity, assess
from operator_agent.execute.mock_gpu import MockGPU
from operator_agent.core.schemas import (ConstraintSpec, KernelArtifact, Precision, SourceSpec,
                                    TargetSpec, TaskComplexity, TaskSchema, TaskType)

PYTHON_MATMUL = """import numpy as np

def kernel(A, B):
    return A @ B
"""

H20_HW = {"name": "NVIDIA H20", "device": "cuda", "compute_capability": "9.0",
          "torch_version": "2.14.0", "triton_version": "3.8.0",
          # 这三个不该进指纹: 它们是规格表里的量, 不是「这次跑在什么环境上」
          "compute_units": 78, "warp_size": 32, "vector_width": 128}


class _VersionedMockGPU(MockGPU):
    """真 `MockGPU` + 两个版本号: 漂移要有东西可漂

    刻意用**子类**而不是 `DeclaredFakeBackend`: 这一组要验的是 Conductor 与真提供方
    之间那条链路, 假件可以让守卫变绿, 但不能被守卫当成被测对象 (doc 48 §6.5)。
    """

    def __init__(self, torch_version="2.14.0", triton_version="3.8.0"):
        super().__init__()
        self._torch_version = torch_version
        self._triton_version = triton_version

    def get_hardware_info(self) -> dict:
        info = super().get_hardware_info()
        info["torch_version"] = self._torch_version
        info["triton_version"] = self._triton_version
        return info


class _RelabelledMockGPU(_VersionedMockGPU):
    """器件名与版本号一字不变, 只有**执行侧身份**变了 (doc 49 §3.1 A1 的形状)

    不能直接改 `self.name`: `MockGPU.get_hardware_info()` 的 `name` 就取自它, 改了
    就等于同时制造两种漂移, 那条用例就分不清是哪一半在起作用。
    """

    LABEL = "LocalExecutor (CPU)"

    @property
    def label(self) -> str:
        return self.LABEL


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


def _task(task_id="driftcase", min_speedup=0.5):
    return TaskSchema(
        task_id=task_id,
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.MEDIUM),
        target=TargetSpec(hardware="mock_gpu", dsl="python"),
        constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=min_speedup,
                                   max_error=1e-3),
    )


def _inputs(n=32):
    rng = np.random.default_rng(0)
    return {"A": rng.standard_normal((n, n)).astype(np.float32),
            "B": rng.standard_normal((n, n)).astype(np.float32)}


def _run_conductor(backend, task, out_dir, rounds=1, resume_from=None):
    """在独占的产物目录里跑一次 Conductor, 返回 (artifact, stdout)"""
    from operator_agent.app.conductor import Conductor
    from operator_agent.core.kernel_db import KernelDatabase
    from unittest import mock
    with mock.patch.dict(os.environ, {"OPERATOR_AGENT_OUTPUT_DIR": out_dir}):
        conductor = Conductor(backend=backend, kernel_db=KernelDatabase(),
                              llm=_ScriptedLLM(), max_optimize_rounds=rounds)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            art = conductor.execute(task, _inputs(), target_gflops=0.0,
                                    resume_from=resume_from)
    return art, buf.getvalue()


def _result_json(artifact_dir):
    return json.loads((Path(artifact_dir) / "result.json").read_text(encoding="utf-8"))


# ─── 一、指纹本身 ────────────────────────────────────────────────

class TestFingerprintIsComparable(unittest.TestCase):

    def test_only_the_comparable_keys_are_kept(self):
        fp = fingerprint_of(H20_HW, "SSHExecutor@alias")
        self.assertEqual({"name": "NVIDIA H20", "device": "cuda",
                          "compute_capability": "9.0", "torch_version": "2.14.0",
                          "triton_version": "3.8.0",
                          FINGERPRINT_BACKEND_KEY: "SSHExecutor@alias"}, fp)

    def test_the_backend_identity_is_part_of_the_fingerprint(self):
        """「同一台机器换了个执行侧」是最要紧的一种漂移, 器件名与版本号一项都不会变

        doc 49 §3.1 A1: 门面连不上时静默降级到 LocalExecutor, 而报表表头照旧写着真机。
        """
        self.assertEqual("LocalExecutor (CPU)",
                         fingerprint_of(H20_HW, "LocalExecutor (CPU)")[FINGERPRINT_BACKEND_KEY])
        self.assertNotEqual(fingerprint_of(H20_HW, "SSHExecutor@alias"),
                            fingerprint_of(H20_HW, "LocalExecutor (CPU)"))

    def test_a_missing_key_stays_missing(self):
        """缺的键**不填默认值**: 「这次没报 triton_version」本身就是一种漂移

        填平它就等于把 doc 49 §3.3 那类「把不知道翻译成一个像样的值」搬进指纹。
        """
        fp = fingerprint_of({"name": "MockGPU"})
        self.assertEqual({"name": "MockGPU"}, fp)
        for k in FINGERPRINT_KEYS[1:]:
            self.assertNotIn(k, fp)

    def test_an_empty_probe_yields_an_empty_fingerprint(self):
        self.assertEqual({}, fingerprint_of({}, ""))
        self.assertEqual({}, fingerprint_of(None, ""))


class TestDriftComparisonIsNotVacuous(unittest.TestCase):
    """比较器必须能算出非空差异 —— 否则「没漂移」的断言全是永真的"""

    def test_identical_fingerprints_are_comparable_and_unchanged(self):
        d = compare_fingerprints({"name": "H20"}, {"name": "H20"}, "上一次")
        self.assertTrue(d.comparable)
        self.assertFalse(d.drifted)
        self.assertEqual((), d.changed)

    def test_a_changed_value_is_reported_with_both_sides(self):
        d = compare_fingerprints({"torch_version": "2.15.0"}, {"torch_version": "2.14.0"})
        self.assertTrue(d.drifted)
        self.assertEqual(1, len(d.changed))
        self.assertIn("2.14.0", d.changed[0])
        self.assertIn("2.15.0", d.changed[0])

    def test_a_key_appearing_or_disappearing_is_drift(self):
        """「上一次没报 triton_version」与「这一次不报了」都是漂移, 不是「少比一项」

        两侧都非空才算可比 (空的一侧见下面那条), 但**键集合**变了必须报出来。
        """
        appeared = compare_fingerprints({"name": "H20", "triton_version": "3.8.0"},
                                        {"name": "H20"})
        self.assertTrue(appeared.drifted)
        self.assertIn("上一次没有", appeared.changed[0])
        vanished = compare_fingerprints({"name": "H20"},
                                        {"name": "H20", "triton_version": "3.8.0"})
        self.assertTrue(vanished.drifted)
        self.assertIn("本次没报", vanished.changed[0])

    def test_no_previous_is_not_the_same_as_no_drift(self):
        """「没有上一次可比」与「比过了、一样」必须分得开

        混成一个空列表, 一份从来没有比较过的产物就会被读成「环境稳定」。这与
        `measurement.assess` 里「launch 地板未知 ⇒ 显式记一句没检查」是同一处置。
        """
        for current, previous in (({"name": "H20"}, {}), ({}, {"name": "H20"}),
                                  ({"name": "H20"}, None), (None, None)):
            d = compare_fingerprints(current, previous)
            self.assertFalse(d.comparable, f"{current} vs {previous}")
            self.assertFalse(d.drifted)
            self.assertEqual((), d.changed)

    def test_report_lines_distinguish_the_three_states(self):
        """日志里三种状态必须长得不一样, 否则「机制没跑」看起来像「跑了没问题」"""
        incomparable = compare_fingerprints({"name": "H20"}, None).report_lines()
        stable = compare_fingerprints({"name": "H20"}, {"name": "H20"}).report_lines()
        drifted = compare_fingerprints({"name": "H200"}, {"name": "H20"}).report_lines()
        self.assertIn("没有可比的上一次运行", incomparable[0])
        self.assertIn("无", stable[0])
        self.assertIn("⚠️", drifted[0])
        self.assertEqual(2, len(drifted), "漂移那一项本身也要逐条列出来")
        self.assertNotEqual(incomparable[0], stable[0])

    def test_as_dict_is_json_shaped(self):
        d = compare_fingerprints({"name": "B"}, {"name": "A"}, "上一次运行的产物 (x)")
        self.assertEqual({"comparable": True, "drifted": True,
                          "changed": ["name: 'A' → 'B'"],
                          "previous_from": "上一次运行的产物 (x)"}, d.as_dict())
        self.assertEqual(d.as_dict(), json.loads(json.dumps(d.as_dict())))

    def test_the_default_constructed_drift_is_incomparable(self):
        self.assertFalse(EnvironmentDrift().comparable)
        self.assertFalse(EnvironmentDrift().drifted)


class TestDriftDegradesWithoutJudging(unittest.TestCase):
    """漂移进的是既有那套三档, 而且**只降一档**"""

    CLEAN = dict(median_time_ms=0.97992, launch_floor_us=20.24)

    def test_drift_is_degraded_not_uninformative(self):
        a = assess(environment_drift=["torch_version: '2.14.0' → '2.15.0'"], **self.CLEAN)
        self.assertIs(Validity.DEGRADED, a.validity)
        self.assertTrue(a.can_gate, "漂移不该把「能不能判通过/不通过」也关掉")
        self.assertFalse(a.can_disable_recipe)
        self.assertFalse(a.can_count_stagnation)

    def test_drift_does_not_refuse_the_l3_pass(self):
        """观测先于判定: 环境变了是**这一轮与上一轮不可比**, 不是这个核不行"""
        a = assess(environment_drift=["name: 'H20' → 'H200'"], **self.CLEAN)
        self.assertFalse(a.l3_pass_is_refused)

    def test_the_reason_says_it_is_not_a_verdict_on_the_kernel(self):
        a = assess(environment_drift=["device: 'cuda' → 'cpu'"], **self.CLEAN)
        text = " ".join(a.reasons)
        self.assertIn("环境指纹", text)
        self.assertIn("不据此判核失败", text)

    def test_no_drift_leaves_the_tier_alone(self):
        self.assertIs(Validity.VALID, assess(environment_drift=[], **self.CLEAN).validity)
        self.assertIs(Validity.VALID, assess(**self.CLEAN).validity)

    def test_drift_does_not_upgrade_an_already_worse_signal(self):
        a = assess(dispersion=0.5, environment_drift=["name: 'A' → 'B'"], **self.CLEAN)
        self.assertIs(Validity.UNINFORMATIVE, a.validity, "取更差的一档, 不是覆盖")

    def test_drift_and_provenance_are_independent_signals(self):
        """两者同时成立时都要说出来 —— 它们的处置不同 (一个降级, 一个拒绝记通过)"""
        a = assess(backend_caps=frozenset(), required_caps={Cap.REAL_DEVICE},
                   environment_drift=["name: 'A' → 'B'"], **self.CLEAN)
        self.assertIs(Validity.UNINFORMATIVE, a.validity)
        self.assertTrue(a.l3_pass_is_refused)
        self.assertEqual(1, len(a.environment_drift))
        self.assertEqual(("real_device",), a.capability_shortfall)


# ─── 二、产物库: 「上一次」从哪儿来 ──────────────────────────────

class TestStoreFindsThePreviousRun(unittest.TestCase):

    def _save(self, store, kernel_type, hardware, fingerprint, task_id="t",
              kernel_id="k1"):
        art = KernelArtifact(kernel_id=kernel_id, task_id=task_id, code="# x",
                             language="python", environment_fingerprint=fingerprint)
        task = TaskSchema(task_id=task_id,
                          source=SourceSpec(kernel_type=kernel_type),
                          target=TargetSpec(hardware=hardware, dsl="python"))
        return store.save(art, task)

    def test_the_most_recent_fingerprinted_run_is_returned(self):
        with tempfile.TemporaryDirectory() as d:
            store = ArtifactStore(output_dir=d)
            older = self._save(store, "matmul", "nvidia_gpu", {"torch_version": "2.14.0"},
                               kernel_id="kOld")
            newer = self._save(store, "matmul", "nvidia_gpu", {"torch_version": "2.15.0"},
                               kernel_id="kNew")
            self.assertNotEqual(str(older), str(newer),
                                "两次落盘撞进了同一个目录, 这条用例就什么也没验")
            got = store.previous_environment("matmul", "nvidia_gpu")
            self.assertEqual({"torch_version": "2.15.0"}, got["fingerprint"])
            self.assertEqual(str(newer), got["artifact_dir"])

    def test_it_reports_where_the_previous_came_from(self):
        with tempfile.TemporaryDirectory() as d:
            store = ArtifactStore(output_dir=d)
            newer = self._save(store, "matmul", "nvidia_gpu", {"name": "H20"})
            got = store.previous_environment("matmul", "nvidia_gpu")
            self.assertEqual(str(newer), got["artifact_dir"])
            self.assertIn("saved_at", got)

    def test_other_buckets_are_not_compared(self):
        """跨算子/跨硬件比较会把「换了一个任务」误报成「换了环境」"""
        with tempfile.TemporaryDirectory() as d:
            store = ArtifactStore(output_dir=d)
            self._save(store, "matmul", "nvidia_gpu", {"name": "H20"})
            self.assertIsNone(store.previous_environment("layernorm", "nvidia_gpu"))
            self.assertIsNone(store.previous_environment("matmul", "mock_gpu"))

    def test_artifacts_without_a_fingerprint_are_skipped_not_reported_as_empty(self):
        """仓库里 2872 份历史 result.json 一份指纹都没有 —— 找不到就如实说找不到

        返回一个 `{}` 的「上一次」会被 `compare_fingerprints` 读成「上一次什么都没报」,
        于是每一个键都变成漂移。
        """
        with tempfile.TemporaryDirectory() as d:
            store = ArtifactStore(output_dir=d)
            art = KernelArtifact(kernel_id="legacy", task_id="t", code="# x")
            task = TaskSchema(task_id="t", source=SourceSpec(kernel_type="matmul"),
                              target=TargetSpec(hardware="mock_gpu", dsl="python"))
            store.save(art, task)
            self.assertIsNone(store.previous_environment("matmul", "mock_gpu"))

    def test_a_corrupt_result_json_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            store = ArtifactStore(output_dir=d)
            good = self._save(store, "matmul", "nvidia_gpu", {"name": "H20"})
            bad = Path(d) / "matmul" / "nvidia_gpu" / "29990101_000000_zzz"
            bad.mkdir(parents=True)
            (bad / "result.json").write_text("{ not json", encoding="utf-8")
            got = store.previous_environment("matmul", "nvidia_gpu")
            self.assertEqual(str(good), got["artifact_dir"])

    def test_the_lookup_is_not_vacuous(self):
        """已知答案对照: 空目录 / 不存在的桶都必须返回 None, 而不是抛或返回 {}"""
        with tempfile.TemporaryDirectory() as d:
            store = ArtifactStore(output_dir=d)
            self.assertIsNone(store.previous_environment("matmul", "nvidia_gpu"))
            self.assertIsNone(store.previous_environment("", ""))


# ─── 三、端到端: 落盘 + 比较 + 报告 + 只降级 ─────────────────────

class TestConductorPersistsAndCompares(unittest.TestCase):

    maxDiff = 400

    def test_first_run_reports_that_there_is_nothing_to_compare_against(self):
        with tempfile.TemporaryDirectory() as d:
            art, out = _run_conductor(_VersionedMockGPU(), _task(), d)
            self.assertIn("没有可比的上一次运行", out)
            self.assertFalse(art.environment_drift["comparable"])
            self.assertEqual("MockGPU", art.environment_fingerprint["name"])
            self.assertEqual("2.14.0", art.environment_fingerprint["torch_version"])

    def test_the_fingerprint_reaches_result_json(self):
        """落盘是全部机制的前提: 读者是**下一次运行**, 不落盘就等于没算"""
        with tempfile.TemporaryDirectory() as d:
            art, _ = _run_conductor(_VersionedMockGPU(), _task(), d)
            data = _result_json(art.saved_dir)
            self.assertEqual(art.environment_fingerprint, data["environment_fingerprint"])
            self.assertEqual(art.environment_drift, data["environment_drift"])

    def test_the_snapshot_carries_it_too(self):
        """续跑走的是快照, 不是 result.json —— 两边都得有"""
        with tempfile.TemporaryDirectory() as d:
            art, _ = _run_conductor(_VersionedMockGPU(), _task(), d)
            snap = json.loads((Path(art.saved_dir) / "execution_snapshot.json")
                              .read_text(encoding="utf-8"))
            self.assertEqual(art.environment_fingerprint, snap["environment_fingerprint"])
            self.assertEqual(art.environment_drift, snap["environment_drift"])

    def test_an_unchanged_environment_is_reported_as_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            first, _ = _run_conductor(_VersionedMockGPU(), _task(), d)
            art, out = _run_conductor(_VersionedMockGPU(), _task("driftcase2"), d)
            self.assertTrue(art.environment_drift["comparable"])
            self.assertFalse(art.environment_drift["drifted"])
            self.assertIn("环境漂移: 无", out)
            # 「上一次」必须是**第一次**那个目录: 比较发生在本次落盘之前, 所以本次
            # 不可能是自己的上一次。这条钉的就是那个时序。
            self.assertIn(Path(first.saved_dir).name, art.environment_drift["previous_from"])
            self.assertNotIn(Path(art.saved_dir).name, art.environment_drift["previous_from"])

    def test_a_changed_environment_is_reported_persisted_and_degrades(self):
        with tempfile.TemporaryDirectory() as d:
            _run_conductor(_VersionedMockGPU(torch_version="2.14.0"), _task(), d)
            art, out = _run_conductor(_VersionedMockGPU(torch_version="2.15.0"),
                                      _task("driftcase3"), d)
            self.assertTrue(art.environment_drift["drifted"])
            self.assertEqual(1, len(art.environment_drift["changed"]))
            self.assertIn("torch_version", art.environment_drift["changed"][0])
            # 日志
            self.assertIn("⚠️ 环境漂移", out)
            self.assertIn("torch_version", out)
            # 产物
            self.assertTrue(_result_json(art.saved_dir)["environment_drift"]["drifted"])
            # 判据: 降到 DEGRADED, 而**不是** UNINFORMATIVE
            l3 = art.benchmark_data
            self.assertEqual(Validity.DEGRADED.value, l3["measurement_validity"])
            self.assertTrue(l3["measurement_can_gate"])
            self.assertFalse(l3["measurement_can_disable_recipe"])
            self.assertFalse(l3["measurement_can_count_stagnation"])
            self.assertIn("环境指纹", " ".join(l3["measurement_validity_reasons"]))

    def test_drift_does_not_by_itself_fail_the_kernel(self):
        """观测先于判定: 换了环境是「这两轮不可比」, 不是「这个核不行」"""
        with tempfile.TemporaryDirectory() as d:
            _run_conductor(_VersionedMockGPU(torch_version="2.14.0"), _task(), d)
            art, out = _run_conductor(_VersionedMockGPU(torch_version="9.99.9"),
                                      _task("driftcase4"), d)
            self.assertTrue(art.environment_drift["drifted"])
            self.assertTrue(art.all_passed, f"漂移把核判死了: {out[-800:]}")
            self.assertIn("所有验证通过", out)

    def test_a_backend_switch_is_drift_even_when_the_device_looks_identical(self):
        """doc 49 §3.1 A1 的形状: 器件名与版本号一项没变, 只有执行侧换了"""
        with tempfile.TemporaryDirectory() as d:
            _run_conductor(_VersionedMockGPU(), _task(), d)
            art, out = _run_conductor(_RelabelledMockGPU(), _task("driftcase5"), d)
            self.assertTrue(art.environment_drift["drifted"])
            changed = " ".join(art.environment_drift["changed"])
            self.assertIn("backend", changed)
            self.assertIn("LocalExecutor (CPU)", changed)
            self.assertEqual(1, len(art.environment_drift["changed"]),
                             f"只该报执行侧这一项: {art.environment_drift['changed']}")
            self.assertIn("⚠️ 环境漂移", out)

    def test_resume_compares_against_the_snapshot_first(self):
        """续跑优先比快照: 那条路径上 `best_gflops` 与已禁用配方全是从上一次继承来的

        环境一变, 继承下来的那个数就不再与本轮可比 —— 这是漂移唯一会直接污染判决的
        场合, 所以它排在产物库前面。
        """
        with tempfile.TemporaryDirectory() as d:
            task = _task("resumecase", min_speedup=5.0)     # 故意达不到 ⇒ 第 1 轮不通过
            first, _ = _run_conductor(_VersionedMockGPU(torch_version="2.14.0"),
                                      task, d, rounds=1)
            self.assertIsNotNone(first.saved_dir)
            resumed, out = _run_conductor(_VersionedMockGPU(torch_version="2.15.0"),
                                          task, d, rounds=2,
                                          resume_from=first.saved_dir)
            self.assertIn("unknown/HOLD", out)
            self.assertTrue(resumed.environment_drift["drifted"])
            self.assertIn("断点续跑的快照", resumed.environment_drift["previous_from"])
            self.assertIn("torch_version", " ".join(resumed.environment_drift["changed"]))
            self.assertTrue(resumed.verification_results[0].metrics["run_identity_unknown"])
            self.assertEqual("identity_mismatch", resumed.resume_status["status"])
            self.assertFalse(resumed.benchmark_data)

    def test_a_snapshot_without_a_fingerprint_holds_without_store_fallback(self):
        """缺恢复证据不能借另一条产物的指纹补齐。"""
        with tempfile.TemporaryDirectory() as d:
            task = _task("oldsnap", min_speedup=5.0)
            first, _ = _run_conductor(_VersionedMockGPU(), task, d, rounds=1)
            snap_path = Path(first.saved_dir) / "execution_snapshot.json"
            snap = json.loads(snap_path.read_text(encoding="utf-8"))
            self.assertIn("environment_fingerprint", snap, "前提: 新快照确实带指纹")
            snap.pop("environment_fingerprint")
            snap_path.write_text(json.dumps(snap, ensure_ascii=False), encoding="utf-8")
            resumed, out = _run_conductor(_VersionedMockGPU(), task, d, rounds=2,
                                          resume_from=first.saved_dir)
            self.assertTrue(resumed.verification_results[0].metrics["run_identity_unknown"])
            self.assertEqual("identity_mismatch", resumed.resume_status["status"])
            self.assertIsNone(resumed.environment_drift)
            self.assertNotIn("上一次运行的产物", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
