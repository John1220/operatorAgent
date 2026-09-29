"""经验库的持久化 (缺口 2) —— 第 2 次运行必须比第 1 次聪明

以前的状态: `kernel_db.py` 零文件 I/O, 语料永远是 `_init_seed_data()` 里那 6 条字面量,
所以「第 236 次运行和第 1 次一样聪明」。设计取舍与外部依据见 docs/research/43:
存储是**追加式 JSONL 事件日志** (不用向量库/embedding —— 检索键是结构化的, 而外部三个
用 embedding 的系统都依赖外部 embedding 服务), 入库门比 Voyager 的 `if success` 多两条
(测量可信 + 真的有加速)。

⚠️ 每个用例都要求「换一个进程/换一个实例后仍然成立」或「两个分支结果不同」——
本项目在测试上栽过的两次逃逸都是造在两条分支一样的地方 (见 doc 41/43 的变异记录)。
"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from operator_agent.kernel_db import KernelDatabase, KernelRecord, _perf_weight
from operator_agent.measurement import persist_decision


def _rec(kid="k1", algorithm="matmul", hardware="nvidia_gpu", dsl="triton",
         *, performance=0.0, score=0.0, validity="valid"):
    return KernelRecord(kernel_id=kid, algorithm=algorithm, hardware=hardware,
                        dsl=dsl, code=f"# {kid}", performance=performance,
                        score=score, measurement_validity=validity,
                        tags=[algorithm, "verified"])


class TestPersistence(unittest.TestCase):
    def test_promoted_record_survives_into_a_new_instance(self):
        """缺口 2 的正命题: 一个进程入库, 另一个实例必须读得到"""
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "kernel_db.jsonl")
            db1 = KernelDatabase(jsonl_path=p)
            self.assertTrue(db1.promote(_rec("kA", score=1.42)))
            db2 = KernelDatabase(jsonl_path=p)
            got = db2.get("kA")
            self.assertIsNotNone(got, "入库后换个实例就读不到 —— 等于没有持久化")
            self.assertAlmostEqual(got.score, 1.42, places=6)
            self.assertEqual(1, db2.loaded_from_disk)
            # seed 之外只剩这一条历史, 说明它确实来自磁盘而不是内存串台
            self.assertIn("seed 6 + 历史 1", db2.summary())

    def test_plain_add_does_not_reach_disk(self):
        """反向: `add()` 仍然只是内存登记。少这条断言, 「把 add 也写成落盘」这种
        改动会让不可信记录悄悄进语料, 而上面那条用例照样绿。"""
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "kernel_db.jsonl")
            db = KernelDatabase(jsonl_path=p)
            db.add(_rec("kB"))
            self.assertEqual(0, db.loaded_from_disk)
            self.assertIsNone(KernelDatabase(jsonl_path=p).get("kB"),
                              "add() 不该落盘 —— 落盘的是 promote(), 它才过门")

    def test_retrieval_is_counted_and_persisted(self):
        """「RAG 命中 N 个案例」以前只等于「语料里有 N 条」。现在取用次数会被攒下来,
        因为它是将来任何退役规则唯一的输入 (本轮刻意不设阈值)。

        ⚠️ 用一个新的 algorithm 名: seed 里的 mock_gpu 记录被 search() 当通用回退,
        混进来会让「命中 1 次」变成「命中 3 次」而看不出机制没变。
        """
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "kernel_db.jsonl")
            db = KernelDatabase(jsonl_path=p)
            db.promote(_rec("kR", algorithm="countprobe", score=1.2))
            db.search(algorithm="countprobe", hardware="nvidia_gpu", dsl="triton")
            self.assertEqual(1, db.retrievals_this_run, "本次运行的取用次数要可见")
            reloaded = KernelDatabase(jsonl_path=p)
            self.assertEqual(1, reloaded.get("kR").retrieved_count)
            self.assertIn("曾被检索 1 条", reloaded.summary())

    def test_truncated_last_line_is_skipped_not_fatal(self):
        """追加式日志被 Ctrl-C 截断最后一行是常态, 不能因此丢掉整库"""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "kernel_db.jsonl"
            db = KernelDatabase(jsonl_path=str(p))
            db.promote(_rec("kGood"))
            with p.open("a", encoding="utf-8") as f:
                f.write('{"event": "kernel", "record": {"kernel_id": "kTru')
            reloaded = KernelDatabase(jsonl_path=str(p))
            self.assertIsNotNone(reloaded.get("kGood"))
            self.assertIsNone(reloaded.get("kTru"))
            self.assertEqual(1, reloaded.disk_skipped_lines)
            self.assertIn("跳过坏行 1", reloaded.summary())

    def test_promote_does_not_downgrade_an_existing_record(self):
        """同一个核的更差测量不该覆盖更好的那条 (crowding 式替换的最小版本)"""
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "kernel_db.jsonl")
            db = KernelDatabase(jsonl_path=p)
            self.assertTrue(db.promote(_rec("kS", score=1.8)))
            self.assertFalse(db.promote(_rec("kS", score=1.1)))
            self.assertAlmostEqual(1.8, db.get("kS").score, places=6)
            self.assertAlmostEqual(1.8, KernelDatabase(jsonl_path=p).get("kS").score,
                                   places=6, msg="拒绝的那次不该被写进日志")

    def test_ranking_prefers_trusted_speedup_over_raw_gflops(self):
        """`score` 的读者: 排序。旧公式只看 gflops, 对 memory-bound 算子恰好是
        缺口 4 记过的那个失真 —— 所以这条断言区分新旧两种实现。"""
        db = KernelDatabase(jsonl_path=None)
        # 算法名刻意用 seed 里没有的: mock_gpu 的 seed 会被当通用回退混进结果,
        # 那样断言的就不是排序而是「谁碰巧也在列表里」。
        db.add(_rec("loud", algorithm="rankprobe", performance=90.0, score=1.05))
        db.add(_rec("quiet", algorithm="rankprobe", performance=10.0, score=2.50))
        hits = db.search(algorithm="rankprobe", hardware="nvidia_gpu", dsl="triton")
        self.assertEqual(["quiet", "loud"], [r.kernel_id for r in hits])
        self.assertAlmostEqual(0.0, _perf_weight(_rec("z", performance=0.0, score=0.0)))

    def test_measured_records_without_a_baseline_still_rank_by_gflops(self):
        """回退分支留给**实测过但没有同机基线**的那类: gflops 是这台卡上量出来的"""
        lo = _rec("lo", performance=5.0, score=0.0, validity="valid")
        hi = _rec("hi", performance=80.0, score=0.0, validity="valid")
        self.assertGreater(_perf_weight(hi), _perf_weight(lo))

    def test_a_never_measured_prior_gets_no_bonus(self):
        """缺口 4 剩下的那一半: 6 条手写 seed 的 performance 是抄进来的先验数, 没测过

        旧写法把它们换算成 0.05–0.85 的加分, 于是**全库权重最高的一条是
        `seed_triton_001` 的 0.85**, 而真机跑出来、带 valid 档位的 8 条只有 0.35–0.44 ——
        没测过的数字在同一条轴上赢了测过的数字 (2026-09-29 实测的这两个数, 不是假想)。
        """
        prior = _rec("prior", performance=90.0, score=0.0, validity="")
        self.assertAlmostEqual(0.0, _perf_weight(prior))
        # 差分守卫: 期望值取**模块自己的种子**, 不抄数字 —— 以后给 seed 补实测值时这条会跟着动
        seeded = KernelDatabase(jsonl_path=None)
        weights = [_perf_weight(r) for r in seeded._records.values()]
        self.assertTrue(weights, "种子库是空的, 这条断言在空转")
        self.assertEqual(0.0, max(weights),
                         "手写种子又拿到加分了 —— 先验数不许和测量数排在同一条轴上")

    def test_an_unmeasured_prior_cannot_be_preferred_over_a_measurement(self):
        """消费方视角: 同等相关性下, 实测 1.05× 的那条必须排在先验 gflops=90 的那条前面"""
        db = KernelDatabase(jsonl_path=None)
        db.add(_rec("prior", algorithm="rankprobe2", performance=90.0, score=0.0,
                    validity=""))
        db.add(_rec("measured", algorithm="rankprobe2", performance=1.0, score=1.05))
        hits = db.search(algorithm="rankprobe2", hardware="nvidia_gpu", dsl="triton")
        self.assertEqual(["measured", "prior"], [r.kernel_id for r in hits])


class TestGate(unittest.TestCase):
    """入库门: 门本身没有读者就等于没有, 所以四条拒绝各给一个不同的原因。"""

    def test_accepts_a_trusted_real_speedup(self):
        keep, why = persist_decision({"measurement_can_gate": True,
                                      "benchmark_contaminated": False,
                                      "speedup_vs_baseline": 1.35,
                                      "measurement_validity": "valid"})
        self.assertTrue(keep)
        self.assertIn("1.350x", why)

    def test_rejects_untrustworthy_measurement(self):
        for m in ({"measurement_can_gate": False, "speedup_vs_baseline": 9.0},
                  {"benchmark_contaminated": True, "speedup_vs_baseline": 9.0}):
            keep, why = persist_decision(m)
            self.assertFalse(keep, f"不该入库: {m} → {why}")

    def test_rejects_no_speedup_and_missing_number(self):
        base = {"measurement_can_gate": True, "benchmark_contaminated": False}
        for sp in (1.0, 0.4, None, "1.5"):
            keep, _ = persist_decision(dict(base, speedup_vs_baseline=sp))
            self.assertFalse(keep, f"speedup={sp!r} 不该入库")


class TestGateWiring(unittest.TestCase):
    """门本身有单测不等于循环读了它 —— 这一类断言专为「判据算出来没人用」而写。

    直接调 `Conductor._save_to_db` (不走整条流水线), 因为要验的是「门的结果有没有决定
    落盘」这一根线, 而不是流水线本身。
    """

    def _conductor(self, path):
        from operator_agent.conductor import Conductor
        from operator_agent.kernel_db import KernelDatabase
        c = Conductor.__new__(Conductor)          # 绕开 LLM/后端的构造成本
        c.kernel_db = KernelDatabase(jsonl_path=path)
        return c

    def _artifact(self, metrics):
        from operator_agent.schemas import KernelArtifact
        # all_passed 是 KernelArtifact 上的只读 property, 由 verification_results 推出来;
        # _save_to_db 不看它 (调用方才看), 所以这里只塞 benchmark_data。
        return KernelArtifact(kernel_id="kw", task_id="t", code="# x",
                              language="triton", benchmark_data=metrics)

    def _task(self):
        from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec,
                                            TargetSpec, TaskComplexity, TaskSchema,
                                            TaskType)
        return TaskSchema(task_type=TaskType.OPERATOR_GENERATION,
                          source=SourceSpec(kernel_type="matmul",
                                            complexity=TaskComplexity.MEDIUM),
                          target=TargetSpec(hardware="nvidia_gpu", dsl="triton"),
                          constraints=ConstraintSpec(precision=Precision.FP16))

    def test_untrusted_measurement_is_not_written_to_disk(self):
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "kernel_db.jsonl")
            c = self._conductor(p)
            kept = c._save_to_db(self._artifact({
                "gflops": 140.0, "speedup_vs_baseline": 3.1,
                "measurement_can_gate": False, "measurement_validity": "uninformative"}),
                self._task())
            self.assertFalse(kept, "门拒了却仍然入库")
            self.assertFalse(Path(p).exists(), "不可信的测量落盘了")
            self.assertIsNotNone(c.kernel_db.get("kw"), "内存里可以留, 否则这次运行没有示范")

    def test_trusted_measurement_is_written_and_reloaded(self):
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "kernel_db.jsonl")
            c = self._conductor(p)
            self.assertTrue(c._save_to_db(self._artifact({
                "gflops": 140.0, "speedup_vs_baseline": 1.35,
                "measurement_can_gate": True, "measurement_validity": "valid"}),
                self._task()))
            reloaded = KernelDatabase(jsonl_path=p).get("kw")
            self.assertIsNotNone(reloaded)
            self.assertAlmostEqual(1.35, reloaded.score, places=6)
            self.assertEqual("valid", reloaded.measurement_validity)


class TestHermetic(unittest.TestCase):
    def test_off_leaves_no_file_and_repo_corpus_untouched(self):
        """`off` 关掉持久化。⚠️ 观察面写全: 仓库 output/kernel_db.jsonl 的字节数
        前后一致 (以前栽过一次: 只对比我以为它会写的地方, 结果它写到别处还全绿)。"""
        repo_db = Path(__file__).resolve().parent.parent / "output" / "kernel_db.jsonl"
        before = repo_db.stat().st_size if repo_db.exists() else None
        with tempfile.TemporaryDirectory() as cwd:
            old = os.getcwd()
            os.chdir(cwd)
            try:
                with mock.patch.dict(os.environ, {"OPERATOR_AGENT_KERNEL_DB": "off"}):
                    db = KernelDatabase()
                    db.promote(_rec("kOff"))
                    db.search(algorithm="matmul")
                self.assertEqual([], sorted(os.listdir(cwd)),
                                 "off 时不该在任何地方建目录或落文件")
                self.assertIn("持久化关闭", db.summary())
            finally:
                os.chdir(old)
        after = repo_db.stat().st_size if repo_db.exists() else None
        self.assertEqual(before, after, "仓库里的 output/kernel_db.jsonl 被写动了")

    def test_artifact_store_honours_the_output_dir_env(self):
        """产物目录也要能重定向, 否则「历史语料」里 99% 是单元测试现场
        (doc 43 第三节量到的正是这件事)。"""
        from operator_agent.artifact_store import ArtifactStore
        from operator_agent.schemas import KernelArtifact, SourceSpec, TargetSpec, \
            ConstraintSpec, TaskSchema, TaskType, TaskComplexity, Precision
        with tempfile.TemporaryDirectory() as d:
            task = TaskSchema(
                task_type=TaskType.OPERATOR_GENERATION,
                source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.LOW),
                target=TargetSpec(hardware="mock_gpu", dsl="python"),
                constraints=ConstraintSpec(precision=Precision.FP32))
            art = KernelArtifact(kernel_id="kx", task_id="tx",
                                 code="def kernel():\n    return 1\n")
            with mock.patch.dict(os.environ, {"OPERATOR_AGENT_OUTPUT_DIR": d}):
                saved = ArtifactStore().save(art, task)
            self.assertEqual(Path(d), Path(saved).parents[2],
                             "env 没生效的话就会写回仓库的 output/")
            self.assertTrue((Path(saved) / "result.json").exists())


if __name__ == "__main__":
    unittest.main()
