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

    def test_contract_prescription_is_not_classified_by_the_exception_it_predicts(self):
        """真机探针 (doc 49 §7.17) 撞出来的**过度合并**, 与碎分相反的另一个方向

        接口契约那条处方文案自己写着「真跑起来必然抛 `TypeError: kernel() got an
        unexpected keyword argument …`」—— 那是**对未来的预言**, 不是本次失败的诊断,
        可 `_ERROR_CLASS_RE` 分不清, 于是静态契约检查与一条真的模块级 TypeError 拿到
        同一个签名。因为 id 是内容派生的, 同任务下这两条不同根因**共用一个 id**,
        后者把前者直接覆盖掉: 语料少了一条, 复发门还把两者数成同一类。

        期望值取 `check_kernel_interface` **自己的输出**, 不抄文案 —— 处方文案改了措辞
        时这条会红, 而那正是该重新看一次归类的时候。
        """
        from operator_agent.interface_contract import check_kernel_interface
        text = check_kernel_interface("def kernel(X, weight):\n    return X\n",
                                      {"X", "weight", "bias"}).problem
        self.assertIsNotNone(text, "前提: 这份代码确实与调用契约不符")
        self.assertEqual("L1|interface_mismatch|-", failure_signature("L1", text))
        self.assertNotEqual(
            failure_signature("L1", text),
            failure_signature("L1", "Compile error: TypeError: can only concatenate str"))

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


class TestMarkerTableIsNotRotting(unittest.TestCase):
    """`_FAILURE_CLASS_MARKERS` 的那句纪律需要一个读者 (doc 45 第二节, doc 49 §八 第 15 条)

    表的注释写着「只收**本仓库里能找到的字面量**」。这句话本身没有读者 —— 有人改了处方文案
    的措辞，对应 marker 就变成一条**从不发火**的规则，归类静默退回「取最后一个异常名」，
    于是 §7.17 那种过并 (预言里的异常名被当成诊断) 随时可能复发，而没有任何东西会红。
    本类把那句话变成一个会红的检查。

    ⚠️ 两个**反空转**自核不是装饰：
      · 不排除 `kernel_db.py` 的话，这张表自己就把五个字面量全提供了 —— 守卫会在
        字面量从生产文案里消失的那一天照样绿 (本项目记过的「假件被当被测对象」同型)。
      · 不剥 docstring 的话，「文案里举了一个从未发出的异常名」这种注释也能喂饱它。
    """

    @staticmethod
    def _emitted_strings(exclude_kernel_db: bool = True) -> str:
        import ast
        root = Path(__file__).resolve().parents[2]
        files = sorted(list((root / "operator_agent").glob("*.py"))
                       + list((root / "remote_scripts").glob("*.py")))
        out = []
        for f in files:
            if exclude_kernel_db and f.name == "kernel_db.py":
                continue
            tree = ast.parse(f.read_text(encoding="utf-8"))
            docstring_nodes = set()
            for n in ast.walk(tree):
                if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                  ast.AsyncFunctionDef)):
                    body = getattr(n, "body", [])
                    if (body and isinstance(body[0], ast.Expr)
                            and isinstance(body[0].value, ast.Constant)
                            and isinstance(body[0].value.value, str)):
                        docstring_nodes.add(id(body[0].value))
            for n in ast.walk(tree):
                if (isinstance(n, ast.Constant) and isinstance(n.value, str)
                        and id(n) not in docstring_nodes):
                    out.append(n.value)
        return "\n".join(out)

    def test_every_marker_literal_is_still_emitted_by_production_code(self):
        from operator_agent.kernel_db import _FAILURE_CLASS_MARKERS
        corpus = self._emitted_strings()
        self.assertTrue(corpus.strip(), "取字面量这一步本身空转了")
        for literal, cls in _FAILURE_CLASS_MARKERS:
            self.assertIn(
                literal, corpus,
                f"marker {cls!r} 的字面量 {literal!r} 已经不在任何会被发出去的文本里了 —— "
                f"它现在是死规则，归类会退回异常名那条路径 (§7.17 的过并就是这么来的)")

    def test_guard_is_not_satisfied_by_its_own_table(self):
        """反空转 ①: 表自己所在的那个文件确实会把字面量喂进来, 排除它不是装饰

        判据不是「排除后取不到」(那当然取不到, 生产文案才是来源)，而是**出现次数会变少** ——
        这条不成立就意味着 `kernel_db.py` 里其实一份字面量都没有，那排除动作是白写的，
        而哪天有人把它加回扫描列表时也无人会红。
        """
        from operator_agent.kernel_db import _FAILURE_CLASS_MARKERS
        with_table = self._emitted_strings(exclude_kernel_db=False)
        without = self._emitted_strings(exclude_kernel_db=True)
        dropped = [(lit, with_table.count(lit) - without.count(lit))
                   for lit, _ in _FAILURE_CLASS_MARKERS]
        self.assertTrue(all(n >= 1 for _, n in dropped),
                        f"排除 kernel_db.py 后这些字面量的出现次数没有减少: {dropped} "
                        f"⇒ 「表自己会喂饱守卫」这件事没被验证")

    def test_guard_is_not_satisfied_by_a_docstring(self):
        """反空转 ②: 只存在于 docstring 里的短语必须取不到 (剥 docstring 真的在剥)"""
        corpus = self._emitted_strings()
        # hardware_caps.py 里这句只在函数 docstring 中出现
        self.assertNotIn("把 compute_capability 字符串解析成主版本号", corpus)


class TestErrorSimilarityRetrievalDoesNotFabricateEvidence(unittest.TestCase):
    """`search_by_error` 末尾那个 `or candidates[:limit]` 兜底: 「没找到」被写成「找到了」

    发现路径不是想出来的, 是变异标定撞出来的 (#76③ 第 1 片, doc 41 §七):
    给 `tests/unit/test_fixer_stage_branch.py` 注入 M9a「把上下文检索里的 L3 过滤掉」时本该红,
    却全绿 —— 当前那一轮的错误文案是**纯中文**, 而 `_tokenize` 只认 `[a-z0-9_]+`, 一个词元都
    取不出来 ⇒ 旧代码走进 `if not error_tokens: return candidates[:limit]`, 把整套同类历史
    原样带回。`fixer.py` 的标签是按「在不在这份返回列表里」算的, 于是这些记录全被印成
    `[错误相似度命中]` —— 检索自己声明的依据是 Jaccard>0, 那个标签是假的。
    另一条同族的兜底是 `... or candidates[:limit]` (有英文词元但零重叠时走它),
    下面 `test_zero_overlap_query_returns_nothing` 钉的是这第二条。

    ⚠️ 同一条用例第一次写在 `FailureDatabase()` 上是**不会红的**: 默认构造会重放
    `<repo>/output/failure_db.jsonl` (2026-09-29 那一份里有 2 条含「加速比」的真机记录),
    磁盘语料把断言喂饱了。所以这里全部显式 `jsonl_path="off"`。
    """

    PERF_REC = dict(kernel_type="matmul", hardware="nvidia_gpu", dsl="python",
                    level="L3", error_summary="speedup below gate: achieved 0.97 needs 1.00",
                    fix_summary="加大 BLOCK_K", fixed=True)
    # 当前这一轮的错误。**中文错误文案在这条检索里等于「没有词元」**: `_tokenize` 只认
    # `[a-z0-9_]+`, 所以纯中文文案一个词元都取不出来 —— 旧代码在这一点上直接
    # `return candidates[:limit]`, 把整套同类历史当「相似度命中」塞进 prompt。
    CJK_ONLY = "最大绝对误差超出容差"
    # 有词元、但与上面那条**零重叠**的英文错误 (另一条走到 `if _ > 0` 那道筛)
    NO_OVERLAP = "shape mismatch in tl.dot BLOCK_M"

    def _db(self):
        from operator_agent.kernel_db import FailureDatabase, FailureRecord
        db = FailureDatabase(jsonl_path="off")
        self.assertIsNone(db._jsonl, "持久化没关掉 ⇒ 用例会读到仓库自己的真语料")
        db.add(FailureRecord(failure_id="h-l3", task_id="t9", **self.PERF_REC))
        return db

    def test_zero_overlap_query_returns_nothing(self):
        db = self._db()
        self.assertEqual([], db.search_by_error(self.NO_OVERLAP, kernel_type="matmul",
                                                dsl="python", limit=2),
                         "Jaccard=0 却被带回 —— 调用方会把它标成「错误相似度命中」")

    def test_query_without_any_token_returns_nothing(self):
        """两条「没有可比的东西」都算: 空串, 和**只有中文**的真实错误文案"""
        db = self._db()
        self.assertEqual([], db.search_by_error("", kernel_type="matmul", dsl="python"))
        self.assertEqual([], db.search_by_error(self.CJK_ONLY, kernel_type="matmul",
                                                dsl="python"),
                         "取不出词元时旧代码返回整套候选 ⇒ 中文报错的任务永远带着假命中")

    def test_positive_overlap_still_returns_it(self):
        """反空转: 上面两条不许是靠「这条检索永远返回空」做到的"""
        db = self._db()
        hits = db.search_by_error("speedup below gate: achieved 0.98 needs 1.00",
                                  kernel_type="matmul", dsl="python", limit=2)
        self.assertEqual(["h-l3"], [r.failure_id for r in hits])

    def test_the_context_route_still_covers_cold_start(self):
        """删掉兜底的代价必须不是「历史栏整段消失」—— 同类算子的覆盖由上下文检索提供"""
        db = self._db()
        base = db.search_by_context(kernel_type="matmul", dsl="python", limit=2)
        self.assertEqual(["h-l3"], [r.failure_id for r in base])

    def test_the_fixer_labels_arrival_route_not_a_fabricated_similarity(self):
        """活路径: 中文报错的那一轮里, 历史栏仍可出现那条 L3, 但只能标「上下文命中」"""
        from operator_agent.fixer import FixerAgent
        from operator_agent.schemas import (TaskSchema, SourceSpec, TargetSpec, ConstraintSpec,
                                            TaskType, TaskComplexity, Precision,
                                            VerificationResult, VerificationLevel)

        class _LLM:
            def __init__(self):
                self.prompts = []

            def chat(self, messages, **kw):
                self.prompts.append(messages[-1]["content"])
                return ("```python\nimport numpy as np\n\n"
                        "def kernel(A, B):\n    return A @ B\n```")

        task = TaskSchema(
            task_id="t-fab", task_type=TaskType.OPERATOR_GENERATION,
            source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.LOW),
            target=TargetSpec(hardware="nvidia_gpu", dsl="python"),
            constraints=ConstraintSpec(precision=Precision.FP32, max_error=1e-5))
        llm = _LLM()
        FixerAgent(llm=llm, failure_db=self._db()).run(
            "def kernel(A, B:\n    return A @ B\n", task,
            [VerificationResult(level=VerificationLevel.L2_FUNCTION, passed=False,
                                message=self.CJK_ONLY, error_log=self.CJK_ONLY)])
        p = llm.prompts[0]
        self.assertIn("[L3] speedup below gate", p, "上下文检索的覆盖不该被这次修改弄丢")
        self.assertNotIn("[错误相似度命中]", p,
                         "零相似的记录被标成「错误相似度命中」= 把「没找到」说成「找到了」")


if __name__ == "__main__":
    unittest.main()
