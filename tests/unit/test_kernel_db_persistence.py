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

from operator_agent.core.kernel_db import KernelDatabase, KernelRecord, _perf_weight
from operator_agent.decide.measurement import persist_decision


def _rec(kid="k1", algorithm="matmul", hardware="nvidia_gpu", dsl="triton",
         *, performance=0.0, score=0.0, validity="valid", tags=None):
    return KernelRecord(kernel_id=kid, algorithm=algorithm, hardware=hardware,
                        dsl=dsl, code=f"# {kid}", performance=performance,
                        score=score, measurement_validity=validity,
                        tags=tags if tags is not None else [algorithm, "verified"])


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

    def test_group_best_is_not_flooded_out_by_relevance(self):
        """P1-2 (generation-side-review): 组内实测 best 硬排最前, 不被词元高分淹没

        旧排序把相关性 (2.0/词元) 与实测加分 (上限 1.0) 放同一条轴上 —— 这条
        用例里 noisy 的词元重叠让它总分 9.0, 压过实测 1.4× 的 best (总分 5.5):
        「历史最好」永远输给「描述写得热闹」。新排序下 best 是第 0 级。
        """
        db = KernelDatabase(jsonl_path=None)
        db.add(_rec("best", algorithm="rankprobe3", score=1.4, tags=["rankprobe3"]))
        db.add(_rec("noisy", algorithm="rankprobe3 alpha beta", score=0.0,
                    validity="", tags=["rankprobe3", "alpha", "beta"]))
        hits = db.search(algorithm="rankprobe3", hardware="nvidia_gpu", dsl="triton",
                         tags=["alpha", "beta"])
        self.assertEqual(["best", "noisy"], [r.kernel_id for r in hits],
                         "组内实测 best 被词元高分淹没 —— 两级排序没生效")

    def test_out_of_group_measurement_never_jumps_the_group(self):
        """组边界: 别的算子就算实测过 (score 再高), 也不许插进本组前面

        反向守卫: 若有人把第 0 级的条件从「本组且实测」放宽成「只要实测」,
        off (score=3.0 但算法不同) 会升到第 0 级压过 on —— 这条就红。
        """
        db = KernelDatabase(jsonl_path=None)
        db.add(_rec("on", algorithm="rankprobe4", score=0.0, validity="",
                    tags=["rankprobe4", "gamma"]))
        db.add(_rec("off", algorithm="elsewhere gamma", score=3.0,
                    tags=["elsewhere", "gamma"]))
        hits = db.search(algorithm="rankprobe4", hardware="nvidia_gpu", dsl="triton",
                         tags=["gamma"])
        ids = [r.kernel_id for r in hits]
        self.assertIn("on", ids)
        self.assertIn("off", ids)
        self.assertLess(ids.index("on"), ids.index("off"),
                        "组外实测记录跳过了组边界 —— 第 0 级条件被放宽了")

    def test_equal_scores_tiebreak_by_total(self):
        """同组同级同分: 总分 (相关性 + _perf_weight) 高的在前, 与旧序一致"""
        db = KernelDatabase(jsonl_path=None)
        db.add(_rec("b1", algorithm="rankprobe5", score=1.2,
                    tags=["rankprobe5", "extra"]))
        db.add(_rec("b2", algorithm="rankprobe5", score=1.2, tags=["rankprobe5"]))
        hits = db.search(algorithm="rankprobe5", hardware="nvidia_gpu", dsl="triton",
                         tags=["extra"])
        self.assertEqual(["b1", "b2"], [r.kernel_id for r in hits])


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
        from operator_agent.app.conductor import Conductor
        from operator_agent.core.kernel_db import KernelDatabase
        c = Conductor.__new__(Conductor)          # 绕开 LLM/后端的构造成本
        c.kernel_db = KernelDatabase(jsonl_path=path)
        return c

    def _artifact(self, metrics):
        from operator_agent.core.schemas import KernelArtifact
        # all_passed 是 KernelArtifact 上的只读 property, 由 verification_results 推出来;
        # _save_to_db 不看它 (调用方才看), 所以这里只塞 benchmark_data。
        return KernelArtifact(kernel_id="kw", task_id="t", code="# x",
                              language="triton", benchmark_data=metrics)

    def _task(self, dsl="triton"):
        from operator_agent.core.schemas import (ConstraintSpec, Precision, SourceSpec,
                                            TargetSpec, TaskComplexity, TaskSchema,
                                            TaskType)
        return TaskSchema(task_type=TaskType.OPERATOR_GENERATION,
                          source=SourceSpec(kernel_type="matmul",
                                            complexity=TaskComplexity.MEDIUM),
                          target=TargetSpec(hardware="nvidia_gpu", dsl=dsl),
                          constraints=ConstraintSpec(precision=Precision.FP16))

    #: 一份「除了来源形态之外一切都好」的读数 —— 用来把 §22.3 的覆盖那一条单独照亮
    _ALL_TRUE_MASK = {"can_verdict": True, "can_tune": True, "can_persist": True,
                      "can_report": True, "can_count_progress": True}

    def _good_metrics(self, **extra):
        m = {"gflops": 140.0, "speedup_vs_baseline": 1.35,
             "measurement_can_gate": True, "measurement_validity": "valid",
             "eligibility": dict(self._ALL_TRUE_MASK)}
        m.update(extra)
        return m

    def _confirmed(self, metrics):
        from operator_agent.core.confirm import ConfirmRecord
        art = self._artifact(metrics)
        art.remeasure = ConfirmRecord(ok=True, speedup=1.4, validity="valid",
                                      independent=True)
        return art

    def test_the_implementation_decision_is_published_next_to_the_claim(self):
        """§17.1 的裁决是交付物第 1 件的一部分 ⇒ 必须由**生产路径**写进产物

        今天两条证据（`required_cases_ok` / `regression_over_limit`）都**没有生产者**
        （契约里没有那两个字段; 逐形状证据要 §20.1）⇒ 裁决只能是 `inconclusive` ——
        这条同时钉住「不许塞 False 假装量过」（CHECKLOG I91）。
        """
        metrics = self._good_metrics()
        with tempfile.TemporaryDirectory() as d:
            c = self._conductor(str(Path(d) / "kernel_db.jsonl"))
            c._save_to_db(self._confirmed(metrics), self._task())
        dec = metrics["implementation_decision"]
        self.assertEqual("inconclusive", dec["decision"], dec["reasons"])
        self.assertIsNone(dec["required_cases_ok"])
        self.assertIsNone(dec["regression_over_limit"])
        self.assertEqual(metrics["claim_level"]["level"], dec["claim_level"],
                         "裁决记的等级必须与产物里发布的那一次读数是同一个")

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
        """可信读数 + 那条独立重测 ⇒ 落盘并可重载

        ⚠️ 2026-10-03 起「可信」还包含「有过一次独立重测」: 两次确认是设计 §3.5 的
        不变量 (CHECKLOG I79), 所以这条用例的产物必须像生产路径那样带上 `remeasure`
        —— 生产路径由 `_produce_remeasure` 无条件补上 (`tests/unit/test_spine_wiring.py`
        有端到端用例钉住), 这里直接给记录是为了只验「门的结果决定落盘」这一根线。
        """
        from operator_agent.core.confirm import ConfirmRecord
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "kernel_db.jsonl")
            c = self._conductor(p)
            art = self._artifact({
                "gflops": 140.0, "speedup_vs_baseline": 1.35,
                "measurement_can_gate": True, "measurement_validity": "valid"})
            art.remeasure = ConfirmRecord(ok=True, speedup=1.4, validity="valid",
                                          independent=True)
            self.assertTrue(c._save_to_db(art, self._task()))
            reloaded = KernelDatabase(jsonl_path=p).get("kw")
            self.assertIsNotNone(reloaded)
            self.assertAlmostEqual(1.35, reloaded.score, places=6)
            self.assertEqual("valid", reloaded.measurement_validity)


    def test_the_claim_level_is_produced_and_published(self):
        """§4.4/§22.3 的**生产者**: `_save_to_db` 一处写入 claim_level

        判据读的那一格与产物里留下的那一格必须是同一个 dict（与 eligibility /
        observation 同一条纪律）—— 否则「报告说它在 validated、门却按另一个上限判」
        会在产物上读不出来。
        """
        with tempfile.TemporaryDirectory() as d:
            c = self._conductor(str(Path(d) / "kernel_db.jsonl"))
            art = self._confirmed(self._good_metrics())
            self.assertTrue(c._save_to_db(art, self._task()))
            self.assertEqual("persisted", art.metadata["claim_level"]["level"])
            self.assertEqual("persisted", art.benchmark_data["claim_level"]["ceiling"])
            self.assertEqual("ast", art.benchmark_data["claim_level"]["coverage_mode"])
            self.assertEqual("persisted", art.benchmark_data["claim_level"]["level"],
                             "入库之后产物理该是 persisted（§4.4 的最后一跳）")

    def test_a_regex_only_source_is_refused_even_with_a_confirmation(self):
        """§22.3 的下游那一半: 覆盖不足 ⇒ 上限 < promotable ⇒ 不许入库

        它拦的是**候选的来源形态**（cuda_c 只有文本规则），不是读数质量 ——
        所以两次确认过了也不放行，而拒因必须自己说出「§22.3」这件事。
        """
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "kernel_db.jsonl")
            c = self._conductor(p)
            art = self._confirmed(self._good_metrics())
            kept = c._save_to_db(art, self._task(dsl="cuda_c"))
            self.assertFalse(kept, "覆盖只有 regex_only 的源码进了长期记忆")
            self.assertEqual("validated", art.metadata["claim_level"]["ceiling"])
            self.assertEqual("regex_only", art.metadata["claim_level"]["coverage_mode"])
            self.assertIn("§22.3", art.metadata["corpus_admission"]["reason"])
            self.assertFalse(Path(p).exists())
            self.assertIsNotNone(c.kernel_db.get("kw"), "内存里仍可留作示范")

    def test_a_triton_source_with_the_same_metrics_is_kept(self):
        """只差来源形态这一格: 同一份读数、同一次确认 ⇒ ast 那一边照常入库

        这一对是**对照臂** —— 少了它, 上面那条拒因可能来自任何别的东西。
        """
        with tempfile.TemporaryDirectory() as d:
            c = self._conductor(str(Path(d) / "kernel_db.jsonl"))
            art = self._confirmed(self._good_metrics())
            self.assertTrue(c._save_to_db(art, self._task(dsl="triton")))
            self.assertEqual("ast", art.metadata["claim_level"]["coverage_mode"])


class TestHermetic(unittest.TestCase):
    def test_off_leaves_no_file_and_repo_corpus_untouched(self):
        """`off` 关掉持久化。⚠️ 观察面写全: 仓库 output/kernel_db.jsonl 的字节数
        前后一致 (以前栽过一次: 只对比我以为它会写的地方, 结果它写到别处还全绿)。"""
        repo_db = Path(__file__).resolve().parents[2] / "output" / "kernel_db.jsonl"
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
        from operator_agent.core.artifact_store import ArtifactStore
        from operator_agent.core.schemas import KernelArtifact, SourceSpec, TargetSpec, \
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


class TestInjectedCorpusIsTheOneWritten(unittest.TestCase):
    """注入的语料必须**就是**被写的那一个 (doc 49 §7.17, 真机探针撞出来的)

    `FailureDatabase` 定义了 `__len__` ⇒ 一个刚建好、还没有记录的实例是 **falsy** 的,
    于是 `Conductor.__init__` 里的 `failure_db or FailureDatabase()` 把调用方给的实例
    换成一个新默认实例: 内存里两份, 落盘写到仓库的 `output/failure_db.jsonl`, 而调用方
    读的那个临时文件永远是空的。真机上这个探针往仓库语料里写了 7 条探针记录, 而它自己的
    所有记账判据**全绿** —— 因为它读的就是那个被换上去的实例。

    ⚠️ 观察面按上面那条 hermetic 用例的教训写全: 既查身份 (`is`), 也查落点 (临时文件
    有行 / 仓库文件字节数不变)。只查身份会漏掉「写对了实例但写到别的路径」, 只查落点会
    漏掉「探针读的是另一个对象所以看不见」。
    """

    def _record(self, conductor, task_id="t-inject"):
        from operator_agent.core.schemas import (ConstraintSpec, Precision, SourceSpec,
                                             TargetSpec, TaskComplexity, TaskSchema,
                                             TaskType, VerificationLevel, VerificationResult)
        task = TaskSchema(
            task_type=TaskType.OPERATOR_GENERATION,
            source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.LOW),
            target=TargetSpec(hardware="mock_gpu", dsl="triton"),
            constraints=ConstraintSpec(precision=Precision.FP32))
        failed = VerificationResult(level=VerificationLevel.L1_COMPILE, passed=False,
                                    message="编译失败: NameError: name 'x' is not defined",
                                    error_log="NameError: name 'x' is not defined")
        conductor._record_failure(task, [failed], "用例: 只测落点", fixed=False)
        return task_id

    def test_empty_injected_failure_db_survives_conductor(self):
        from operator_agent.app.conductor import Conductor
        from operator_agent.core.kernel_db import FailureDatabase
        repo_db = Path(__file__).resolve().parents[2] / "output" / "failure_db.jsonl"
        repo_before = repo_db.stat().st_size if repo_db.exists() else None
        with tempfile.TemporaryDirectory() as d:
            fpath = str(Path(d) / "f.jsonl")
            db = FailureDatabase(jsonl_path=fpath)
            self.assertEqual(0, len(db))
            self.assertFalse(bool(db), "前提: 空语料因为 __len__ 而是 falsy 的")
            conductor = Conductor(kernel_db=KernelDatabase(jsonl_path=str(Path(d) / "k.jsonl")),
                                  failure_db=db, llm=None)
            self.assertIs(db, conductor.failure_db, "注入的实例被 `or` 换成默认实例了")
            self._record(conductor)
            self.assertEqual(1, len(db), "写进了另一个对象 —— 注入的语料一条都没收到")
            self.assertTrue(Path(fpath).exists(), "落点不在调用方给的路径上")
            self.assertEqual(1, len(Path(fpath).read_text(encoding="utf-8").splitlines()))
        repo_after = repo_db.stat().st_size if repo_db.exists() else None
        self.assertEqual(repo_before, repo_after, "仓库里的 output/failure_db.jsonl 被写动了")

    def test_passing_none_still_gets_a_working_default(self):
        """对照: 不注入时仍然要有一个可用的默认实例, 否则这条修复改坏了默认分支"""
        from operator_agent.app.conductor import Conductor
        with mock.patch.dict(os.environ, {"OPERATOR_AGENT_FAILURE_DB": "off"}):
            conductor = Conductor(kernel_db=KernelDatabase(jsonl_path="none"), llm=None)
            self._record(conductor)
            self.assertEqual(1, len(conductor.failure_db))
            self.assertIn("持久化关闭", conductor.failure_db.summary())


if __name__ == "__main__":
    unittest.main()
