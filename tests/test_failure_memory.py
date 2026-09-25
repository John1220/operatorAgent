"""负样本记忆 (doc 45) —— 失败要有文件, 而且「同一种失败」要数得出任务数

改之前的状态: `FailureDatabase` 零文件 I/O (缺口 2 的「仍未做」那一半), 每条失败在进程
退出时蒸发; 而它内部那两个 id 方案 (conductor 用 `task_id_{库里条数}`、fixer 用
`sha1(kernel_type|dsl|level|错误前200字)_{随机4位}`) 都**不是**「这类失败」的标识 ——
前者随插入顺序漂移, 后者把 kernel_type 和错误原文写进了键, 于是「跨任务复发」这条判据
在旧结构里根本没法算。

外部依据 (逐条核过源码/原文, 见 doc 45):
  · Ecdysis (arXiv 2609.11677) 的分组键 = 终止原因 + 失败的 reward 分量, 刻意不含任务;
  · AdaExplore (arXiv 2604.16625) 的频率门出厂默认 3, 且把带 max_difference 的日志整条丢掉;
  · arXiv 2503.12374 用正则 + 关键词把 92 种原始变体归成 32 类, 门槛是「≥5 个任务」。
"""
import tempfile
import unittest
from pathlib import Path

from operator_agent.kernel_db import (FailureDatabase, FailureRecord,
                                      failure_signature, is_promotable_failure)

# 一段真实的 Triton 远程编译失败的样子: 远端把异常名放在**开头**, 中间回显源码
COMPILE_FAIL_A = (
    "Compile error: CompilationError: at 12:0:\n"
    "def kernel(A, B, C, M, N, K):\n"
    '  File "/remote/work/run_kernel.py", line 351, in _load_module\n'
    "    mod = importlib.reload(mod)\n"
    "NameError: name 'tl' is not defined\n"
)
# 同一个缺陷, 换一台机器、行号不同、算子类型不同
COMPILE_FAIL_B = (
    "Compile error: CompilationError: at 9:4:\n"
    "def kernel(X, Y):\n"
    '  File "/home/other/deploy/run_kernel.py", line 402, in _load_module\n'
    "    mod = importlib.reload(mod)\n"
    "NameError: name 'tl' is not defined\n"
)

NAME_RE = "L1|NameError|_load_module"


def _rec(fid="f1", kernel_type="matmul", dsl="python", level="L1",
         error=COMPILE_FAIL_A, *, fixed=False, task_id="t1", signature=None,
         lessons=None, root_cause=""):
    return FailureRecord(failure_id=fid, kernel_type=kernel_type, hardware="nvidia_gpu",
                         dsl=dsl, level=level, error_summary=error, fixed=fixed,
                         task_id=task_id,
                         signature=signature if signature is not None
                         else failure_signature(level, error),
                         lessons=lessons or [], root_cause=root_cause)


def _write(path, records):
    """先把记录落进一个 jsonl, 再交给另一个实例重放 —— 复现「下一次运行」"""
    db = FailureDatabase(jsonl_path=str(path))
    for r in records:
        db.add(r)
    return str(path)


class TestFailureSignature(unittest.TestCase):
    """签名只认「哪一类失败」, 不认这次是哪个任务、哪台机器、第几行"""

    def test_same_defect_across_different_kernels_shares_one_signature(self):
        """C3 的立足点: 不同算子上的同一类失败必须落进同一个桶"""
        a = failure_signature("L1", COMPILE_FAIL_A)
        b = failure_signature("L1", COMPILE_FAIL_B)
        self.assertEqual(a, b, "行号/路径/算子源码的变化把同一缺陷拆成了两个签名")
        self.assertEqual(NAME_RE, a)

    def test_signature_ignores_kernel_type_entirely(self):
        """旧 sha1 把 kernel_type 写进键里, 于是跨算子复发永远数不到"""
        for kt in ("matmul", "layernorm", "attention"):
            r = _rec(kernel_type=kt)
            self.assertEqual(NAME_RE, r.signature)

    def test_different_exception_is_a_different_signature(self):
        self.assertNotEqual(failure_signature("L1", COMPILE_FAIL_A),
                            failure_signature("L1", "Import error (triton): ModuleNotFoundError: no triton"))

    def test_level_is_part_of_the_signature(self):
        self.assertNotEqual(failure_signature("L1", COMPILE_FAIL_A),
                            failure_signature("L2", COMPILE_FAIL_A))

    def test_our_own_fixed_sentence_wins_over_a_nested_exception_name(self):
        """远端把「每个基线各试了一次」的错误列表整个塞进 error_log, 里面几乎总能
        匹配到一个异常类名 —— 那个名字属于基础设施的某次尝试, 不是这次失败的诊断。"""
        text = ("No usable remote baseline for matmul; attempts="
                "['pytorch_eager_tf32: AttributeError: 'NoneType' object has no attribute 'matmul'']")
        self.assertEqual("L3|infra_no_baseline|-", failure_signature("L3", text))

    def test_cheat_and_tolerance_markers_are_recognised(self):
        self.assertEqual("L2|cheat_detected|-",
                         failure_signature("L2", "反作弊: 检测到禁用的原生算子回退调用 (1 处)"))
        self.assertEqual("L2|numeric_tolerance|-",
                         failure_signature("L2", "执行失败: 数值不一致, 最大误差=3.10e-03"))

    def test_unrecognised_text_is_an_explicit_unknown_bucket(self):
        """兜底桶必须是**看得见**的, 不是静默归到某个类里: 语料里 unknown 占比高
        就说明关键词表该补了 (这张表只收仓库里能找到的字面量)"""
        self.assertEqual("L2|unknown|-", failure_signature("L2", "Something nobody predicted"))

    def test_innermost_frame_is_used_not_the_outermost(self):
        text = ('Traceback:\n'
                '  File "a.py", line 1, in run\n'
                '  File "a.py", line 2, in kernel\n'
                "ZeroDivisionError: division by zero")
        self.assertEqual("L2|ZeroDivisionError|kernel", failure_signature("L2", text))


class TestPromotable(unittest.TestCase):
    """AdaExplore 的 filter_max_difference, 换成本项目的说法"""

    def test_numeric_tolerance_failures_are_not_reusable_lessons(self):
        """从「误差 3.1e-3 超了门限」里能提炼出的自然教训是「这类误差可以接受」——
        那正是数值门要防的事。让语料教模型放宽检查 = 给 reward hacking 建库。"""
        r = _rec(level="L2", error="执行失败: 数值不一致, 最大误差=3.10e-03")
        self.assertFalse(is_promotable_failure(r))

    def test_infra_failures_are_not_the_kernels_fault(self):
        r = _rec(level="L3",
                 error="No usable remote baseline for matmul; attempts=['torch_compile: AttributeError: x']")
        self.assertFalse(is_promotable_failure(r))

    def test_a_compile_error_is_promotable(self):
        self.assertTrue(is_promotable_failure(_rec()))

    def test_a_real_exception_on_l2_is_still_promotable(self):
        """只有「容差」措辞才被判为放宽检查的诱惑; L2 上的真异常不是"""
        self.assertTrue(is_promotable_failure(
            _rec(level="L2", error="Triton error: OutOfResources: shared memory")))


class TestPersistence(unittest.TestCase):
    def test_failure_survives_into_a_new_instance(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "failure_db.jsonl"
            _write(p, [_rec("fA", task_id="t1")])
            db2 = FailureDatabase(jsonl_path=str(p))
            self.assertEqual(1, db2.loaded_from_disk)
            self.assertEqual("t1", db2.list_all()[0].task_id)
            self.assertEqual(NAME_RE, db2.list_all()[0].signature,
                             "签名没跟着落盘 —— 重放后复发计数无从算起")

    def test_state_change_on_the_same_id_survives_replay(self):
        """同一件事从 fixed=False 演进到 fixed=True: 写侧不去重、后写覆盖先写在重放时做
        —— 若写侧去重, 这次演进就永远落不了盘"""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "failure_db.jsonl"
            db = FailureDatabase(jsonl_path=str(p))
            db.add(_rec("fA", task_id="t1"))
            db.add(_rec("fA", task_id="t1", fixed=True))
            db2 = FailureDatabase(jsonl_path=str(p))
            self.assertEqual(1, len(db2.list_all()), "同一 id 重放后成了两条, 复发数会虚高")
            self.assertTrue(db2.list_all()[0].fixed, "第二次 (fixed=True) 没覆盖第一次")
            self.assertEqual(1, db2.loaded_from_disk)

    def test_replaying_the_same_file_twice_does_not_double_the_corpus(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "failure_db.jsonl"
            _write(p, [_rec("fA", task_id="t1")])
            db = FailureDatabase(jsonl_path=str(p))
            db = FailureDatabase(jsonl_path=str(p))
            self.assertEqual(1, len(db.list_all()))
            self.assertEqual(1, db.recurrence()[NAME_RE])

    def test_truncated_last_line_is_skipped_and_counted(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "failure_db.jsonl"
            _write(p, [_rec("fA", task_id="t1")])
            with p.open("a", encoding="utf-8") as f:
                f.write('{"event": "failure", "record": {"failure_id": "cut"')  # Ctrl-C 现场
            db = FailureDatabase(jsonl_path=str(p))
            self.assertEqual(1, db.loaded_from_disk)
            self.assertEqual(1, db.disk_skipped_lines)
            self.assertIn("跳过坏行 1", db.summary())

    def test_off_leaves_no_file(self):
        with tempfile.TemporaryDirectory() as d:
            cwd = Path(d)
            p = cwd / "nothing"
            db = FailureDatabase(jsonl_path=str(p))
            db._jsonl = None
            db.add(_rec())
            self.assertFalse(p.exists())


class TestRecurrenceGate(unittest.TestCase):
    """复发门只挡**从文件重放回来**的记录: 挡本次进程的记录是把 Fixer 现在就能
    看到的自身经验关掉, 那是回归不是收紧。"""

    def test_single_occurrence_history_is_not_injected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "failure_db.jsonl"
            _write(p, [_rec("fA", task_id="t1", lessons=["补 import"])])
            db = FailureDatabase(jsonl_path=str(p))
            self.assertEqual([], db.search_by_context(kernel_type="matmul", dsl="python"))
            self.assertEqual(1, len(db.suppressed_by_gate))

    def test_same_signature_on_three_tasks_is_injected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "failure_db.jsonl"
            _write(p, [_rec(f"f{i}", task_id=f"t{i}") for i in range(3)])
            db = FailureDatabase(jsonl_path=str(p))
            got = db.search_by_context(kernel_type="matmul", dsl="python", limit=3)
            self.assertEqual(3, len(got), "跨 3 个任务复发仍被挡住 —— 门永远开不了")

    def test_three_records_on_one_task_do_not_open_the_gate(self):
        """这是整条判据的全部要点: 同一任务里连烧三轮修复 = 同一次失败被观测三次"""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "failure_db.jsonl"
            _write(p, [_rec(f"f{i}", task_id="same") for i in range(3)])
            db = FailureDatabase(jsonl_path=str(p))
            self.assertEqual(1, db.recurrence()[NAME_RE],
                             "复发按记录数而不是任务数计数 —— C3 的判据就没了")
            self.assertEqual([], db.search_by_context(kernel_type="matmul", dsl="python"))

    def test_in_process_records_are_never_gated(self):
        db = FailureDatabase(jsonl_path=None)
        db.add(_rec("fA", task_id="t1", lessons=["补 import"]))
        self.assertEqual(1, len(db.search_by_context(kernel_type="matmul", dsl="python")),
                         "本次进程自己写的教训被挡住了 —— Fixer 丢了它原有的经验")

    def test_tolerance_history_stays_out_even_when_recurrent(self):
        """类别门优先: 复发三次也不该把「放宽容差」的教训抬进 prompt"""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "failure_db.jsonl"
            _write(p, [_rec(f"f{i}", level="L2", dsl="python",
                            error="执行失败: 数值不一致, 最大误差=3.10e-03",
                            task_id=f"t{i}") for i in range(4)])
            db = FailureDatabase(jsonl_path=str(p))
            self.assertEqual([], db.search_by_context(dsl="python"))
            self.assertEqual(4, len(db.suppressed_by_kind))
            self.assertEqual(0, len(db.suppressed_by_gate))

    def test_gate_can_be_turned_down_by_env(self):
        """3 是一篇论文调出来的常数, 没有消融支撑 (doc 45 第三节) —— 所以必须可调"""
        import os
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "failure_db.jsonl"
            _write(p, [_rec("fA", task_id="t1")])
            old = os.environ.get("OPERATOR_AGENT_FAILURE_GATE")
            os.environ["OPERATOR_AGENT_FAILURE_GATE"] = "1"
            try:
                db = FailureDatabase(jsonl_path=str(p))
                self.assertEqual(1, len(db.search_by_context(kernel_type="matmul", dsl="python")))
            finally:
                if old is None:
                    os.environ.pop("OPERATOR_AGENT_FAILURE_GATE", None)
                else:
                    os.environ["OPERATOR_AGENT_FAILURE_GATE"] = old

    def test_summary_makes_the_gate_visible(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "failure_db.jsonl"
            _write(p, [_rec("fA", task_id="t1"), _rec("fB", task_id="t2"),
                       _rec("fC", task_id="t3"),
                       _rec("fD", task_id="t4", level="L2",
                            error="反作弊: 检测到禁用的原生算子回退调用 (1 处)")])
            db = FailureDatabase(jsonl_path=str(p))
            s = db.summary()
            self.assertIn("2 个签名", s, "签名数是「分组有没有碎成 1:1」的唯一可见指标")
            self.assertIn("≥3 的 1 个", s, "门阈值与够格的组数看不见就等于没生效")


class TestConductorWiring(unittest.TestCase):
    """判据算出来没人用是本项目的传统失效方式, 所以这根线单独验"""

    def _conductor(self, path):
        from operator_agent.conductor import Conductor
        c = Conductor.__new__(Conductor)
        c.failure_db = FailureDatabase(jsonl_path=path)
        return c

    def _task(self):
        from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec,
                                            TargetSpec, TaskComplexity, TaskSchema,
                                            TaskType)
        return TaskSchema(task_type=TaskType.OPERATOR_GENERATION,
                          source=SourceSpec(kernel_type="matmul",
                                            complexity=TaskComplexity.MEDIUM),
                          target=TargetSpec(hardware="nvidia_gpu", dsl="triton"),
                          constraints=ConstraintSpec(precision=Precision.FP16))

    def _failed(self, levels):
        from operator_agent.schemas import VerificationLevel, VerificationResult
        by_name = {"L1": VerificationLevel.L1_COMPILE,
                   "L2": VerificationLevel.L2_FUNCTION,
                   "L3": VerificationLevel.L3_PERFORMANCE}
        return [VerificationResult(level=by_name[l] if isinstance(l, str) else l,
                                   passed=False, message="失败",
                                   error_log="NameError: name 'tl' is not defined")
                for l in levels]

    def test_record_carries_signature_and_task_id_and_is_on_disk(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "failure_db.jsonl"
            c = self._conductor(p)
            task = self._task()
            c._record_failure(task, self._failed(["L1"]), "改法", fixed=False)
            self.assertEqual(1, len(c.failure_db.list_all()))
            r = c.failure_db.list_all()[0]
            self.assertTrue(r.signature.startswith("L1|"), f"签名没建: {r.signature}")
            self.assertEqual(task.task_id, r.task_id, "复发要按任务数, 而任务 id 没进记录")
            fresh = FailureDatabase(jsonl_path=str(p))
            self.assertEqual(1, fresh.loaded_from_disk, "失败只留在内存, 进程一退就蒸发")
            self.assertEqual(r.signature, fresh.list_all()[0].signature)

    def test_level_order_does_not_fragment_the_signature(self):
        """旧写法 `"+".join(...)` 跟着 failed_results 的顺序走: 同一份失败可以写成
        "L1+L2" 也可以写成 "L2+L1", 于是两个桶, 复发永远数不满 3"""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "failure_db.jsonl"
            task = self._task()
            a = self._conductor(p)
            a._record_failure(task, self._failed(["L1", "L2"]), "x", False)
            b = self._conductor(p)
            b._record_failure(task, self._failed(["L2", "L1"]), "y", False)
            ids = [r.failure_id for r in FailureDatabase(jsonl_path=str(p)).list_all()]
            self.assertEqual(1, len(ids), f"同一任务同一类失败被记成两条: {ids}")

    def test_three_runs_of_the_same_defect_open_the_gate(self):
        """生产端 → 落盘 → 重放 → 复发门开, 一根线走到底。

        只验门的单测会放过「生产端忘了写 signature」这种断法: 那扇门永远不开, 而
        所有门相关的用例照样全绿 (语料是空的)。
        """
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "failure_db.jsonl"
            for _ in range(3):
                c = self._conductor(p)
                c._record_failure(self._task(), self._failed(["L1"]), "补 import", fixed=False)
            fresh = FailureDatabase(jsonl_path=str(p))
            self.assertEqual(3, fresh.loaded_from_disk)
            self.assertEqual(3, fresh.recurrence()["L1|NameError|-"])
            self.assertEqual(3, len(fresh.search_by_context(kernel_type="matmul", dsl="triton")))


if __name__ == "__main__":
    unittest.main()
