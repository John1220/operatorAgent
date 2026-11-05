"""HITL 工具壳的守卫：`operator_agent/app/hitl_tools.py` 的两条纪律必须一直成立

这一层被包给外层宿主（DSH 插件 / MCP / 脚本）调用，所以它的**失败形态**比成功形态更重要。
这里钉住四件事：

1. **判据不许被调用方放宽**：白名单与黑名单不相交，且 `max_error` / 任何「跳过」类参数
   一出现就 `forbidden_parameter`（设计 §3.5 / §6.3 的「不许有开关」）。
2. **回合身份可复算、且改判据会被抓住**：同一份声明两次冻结出同一个哈希；prepare 之后
   改 `declaration.json` ⇒ verify 返回 `contract_drift`，不按新判据继续。
3. **处方与资格打架时必须显式**（§9.5 实测撞出来的接缝）：`route.counts_as_iteration`
   与 `can_tune=False` 同时成立 ⇒ `conflict.present=True` + `action=hold_for_measurement`。
4. **stdout 只有一行 JSON**：判定层的进度输出被收进 run 目录的 `verify-N.log`，
   不进 stdout —— 否则任何解析这一行的宿主都会挂。

运行: python3 -m pytest tests/unit/test_hitl_tools.py -q
"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from operator_agent.app import hitl_tools as ht
from operator_agent.app.cli import main as cli_main

CORRECT = "def kernel(A, B):\n    return A @ B\n"
WRONG = "def kernel(A, B):\n    return (A @ B) * 0.9\n"


def declaration(**over):
    base = {
        "request": "把 64x64x64 的 fp32 矩阵乘法写成算子",
        "shapes": [[64, 64, 64]],
        "min_speedup": 0.5,       # 64³ 在本机 CPU 上的比值在噪声带里，门槛只为让判定稳定
    }
    base.update(over)
    return base


class TestTheSurfaceRefusesToGrowSwitches(unittest.TestCase):
    """白名单/黑名单是**结构**约束，不是文档约束。"""

    def test_allowed_and_forbidden_never_overlap(self):
        self.assertEqual(ht.ALLOWED_KEYS & set(ht.FORBIDDEN_KEYS), set())

    def test_the_judgement_levers_are_not_parameters(self):
        # 这些名字出现即等于开了一个「放宽判据」的口子。
        for name in ("max_error", "atol", "rtol", "require_double_confirm",
                     "skip_anticheat", "failure_gate", "kernel_db", "failure_db"):
            self.assertNotIn(name, ht.ALLOWED_KEYS, name)
            self.assertIn(name, ht.FORBIDDEN_KEYS, name)

    def test_a_forbidden_key_is_refused_with_the_reason(self):
        with self.assertRaises(ht.ToolError) as ctx:
            ht.prepare(declaration(max_error=1.0), tempfile.mkdtemp())
        self.assertEqual(ctx.exception.code, "forbidden_parameter")
        self.assertIn("max_error", ctx.exception.detail)

    def test_an_unknown_key_is_refused(self):
        with self.assertRaises(ht.ToolError) as ctx:
            ht.prepare(declaration(whatever=1), tempfile.mkdtemp())
        self.assertEqual(ctx.exception.code, "unknown_parameter")

    def test_shapes_are_required(self):
        with self.assertRaises(ht.ToolError) as ctx:
            ht.prepare({"request": "把 fp32 矩阵乘法写成算子"}, tempfile.mkdtemp())
        self.assertEqual(ctx.exception.code, "shapes_required")

    def test_an_unknown_capability_is_refused(self):
        with self.assertRaises(ht.ToolError) as ctx:
            ht.prepare(declaration(required_caps=["telepathy"]), tempfile.mkdtemp())
        self.assertEqual(ctx.exception.code, "unknown_cap")

    def test_an_unknown_backend_is_refused(self):
        run = Path(tempfile.mkdtemp()) / "r"
        ht.prepare(declaration(), run)
        with self.assertRaises(ht.ToolError) as ctx:
            ht.verify(run, CORRECT, backend="auto")
        self.assertEqual(ctx.exception.code, "backend_not_wired")

    def test_ssh_without_a_host_is_refused_instead_of_degrading_to_cpu(self):
        """`mode="auto"` 会一路降到 LocalExecutor —— 那正是「以为在真机上」的形状。"""
        run = Path(tempfile.mkdtemp()) / "r"
        ht.prepare(declaration(), run)
        with self.assertRaises(ht.ToolError) as ctx:
            ht.verify(run, CORRECT, backend="ssh")
        self.assertEqual(ctx.exception.code, "host_required")
        self.assertIn("不提供 auto 降级", ctx.exception.detail)


class TestPrepareFreezesARound(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="hitl-tools-"))
        cls.round_dir = cls.tmp / "run"
        cls.payload = ht.prepare(declaration(), cls.round_dir)

    def test_it_writes_the_round_identity_into_the_run_dir(self):
        for name in ("declaration.json", "contract.json", "run.json"):
            self.assertTrue((self.round_dir / name).is_file(), name)
        recorded = json.loads((self.round_dir / "contract.json").read_text(encoding="utf-8"))
        self.assertEqual(recorded["hash"], self.payload["contract_hash"])

    def test_the_contract_owns_reproducible_inputs(self):
        self.assertEqual(self.payload["cases"][0]["key"], "A=64x64|B=64x64")
        self.assertEqual(self.payload["inputs"]["A"]["shape"], [64, 64])
        self.assertEqual(self.payload["inputs"]["A"]["dtype"], "float32")
        # 同一份声明再冻结一次：哈希必须一样（判据可复算，不依赖进程状态/时间）
        again = ht.prepare(declaration(), self.tmp / "run2")
        self.assertEqual(again["contract_hash"], self.payload["contract_hash"])

    def test_a_profile_is_declared_for_a_supported_operator(self):
        self.assertEqual(self.payload["correctness_profile"], "builtin-l2-v1")
        self.assertEqual(self.payload["judgement"]["kernel_type"], "matmul")
        self.assertEqual(self.payload["judgement"]["max_error"], 1e-05)

    def test_an_operator_without_a_layout_is_refused(self):
        with self.assertRaises(ht.ToolError) as ctx:
            ht.prepare(declaration(request="把 4x8x2048x64 的 fp32 attention 写成算子",
                                  shapes=[[4, 8, 2048, 64]]), self.tmp / "attn")
        self.assertEqual(ctx.exception.code, "contract_refused")


class TestVerifyReturnsOrthogonalFacts(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="hitl-verify-"))
        cls.round_dir = cls.tmp / "run"
        ht.prepare(declaration(), cls.round_dir)
        cls.strict = cls.tmp / "strict"
        ht.prepare(declaration(required_caps=["real_device"]), cls.strict)

    def test_a_correct_kernel_passes_and_carries_no_conflict(self):
        out = ht.verify(self.round_dir, CORRECT)
        self.assertTrue(out["ok"])
        self.assertEqual(out["levels"], {"L1": True, "L2": True, "L3": True})
        self.assertEqual(out["route"]["route"], "hold")
        self.assertEqual(out["conflict"], {"present": False})
        self.assertEqual(out["validity"], "valid")
        self.assertEqual(set(out["eligibility"]),
                         {"can_verdict", "can_tune", "can_persist",
                          "can_report", "can_count_progress"})

    def test_a_wrong_kernel_stops_at_l2_with_a_fix_prescription(self):
        out = ht.verify(self.round_dir, WRONG, stop_after="L2")
        self.assertEqual(out["levels"], {"L1": True, "L2": False})
        self.assertEqual(out["route"]["route"], "debug_fix")
        self.assertTrue(out["route"]["fix_not_rewrite"])
        self.assertIn("形态", out["route"]["reason"] + str(out["verdicts"][-1]["message"]))
        self.assertIn("max_error", out["verdicts"][-1]["metrics"])

    def test_one_reading_under_two_contracts_gets_two_consequences(self):
        """§9.5 的 B vs B'：同一个核，只改了契约里的一格 required_caps。"""
        free = ht.verify(self.round_dir, CORRECT)
        strict = ht.verify(self.strict, CORRECT)
        self.assertEqual(free["levels"]["L3"], True)
        self.assertEqual(strict["levels"]["L3"], False)
        self.assertEqual(free["validity"], "valid")
        self.assertNotEqual(free["validity"], strict["validity"])
        self.assertTrue(all(free["eligibility"].values()))
        self.assertFalse(any(strict["eligibility"].values()))
        self.assertEqual(strict["contract_hash"] != free["contract_hash"], True)

    def test_the_route_vs_eligibility_seam_is_reported_as_a_conflict(self):
        """处方说「去搜索/算一轮」，而资格说这个数不能驱动任何后果 —— 必须显式。"""
        strict = ht.verify(self.strict, CORRECT)
        self.assertTrue(strict["route"]["counts_as_iteration"])
        self.assertFalse(strict["eligibility"]["can_tune"])
        self.assertTrue(strict["conflict"]["present"])
        self.assertEqual(strict["conflict"]["action"], "hold_for_measurement")
        self.assertIn("can_tune=False", strict["conflict"]["why"])
        self.assertIn("required_caps", strict["conflict"]["next_step"])

    def test_the_local_backend_can_do_l1_but_not_the_declared_suite(self):
        """`local` 是**弱后端**：没有执行侧输入回读协议 ⇒ 带套件的 L2 只能判 unknown。

        这条守住「不要读成候选错了」：L1 通过、L2 的失败理由必须指向**能力缺口**。
        """
        run = Path(tempfile.mkdtemp(prefix="hitl-local-")) / "run"
        ht.prepare(declaration(), run)
        l1 = ht.verify(run, CORRECT, backend="local", stop_after="L1")
        self.assertEqual(l1["levels"], {"L1": True})
        self.assertIn("Local", l1["backend_resolution"]["resolved"])

        full = ht.verify(run, CORRECT, backend="local")
        self.assertFalse(full["levels"]["L2"])
        joined = " ".join(str(v.get("message", "")) for v in full["verdicts"])
        self.assertTrue("回读" in joined or "unknown" in joined, joined)

    def test_ssh_without_host_never_silently_falls_back_to_cpu(self):
        run = Path(tempfile.mkdtemp(prefix="hitl-ssh-")) / "run"
        ht.prepare(declaration(), run)
        with self.assertRaises(ht.ToolError) as ctx:
            ht.verify(run, CORRECT, backend="ssh")
        self.assertEqual(ctx.exception.code, "host_required")

    def test_the_baseline_semantics_note_is_derived_not_guessed(self):
        """真机实测：fp32 matmul 的首选基线是 cuBLAS(TF32)，而契约写 tf32=False。

        这条只钉「把事实说出来」：`baseline_type` 出现在回包里，且当它是 TF32 而契约不是
        TF32 时给一句人话。判定本身仍归 measurement / eligibility。
        """
        out = ht.verify(self.round_dir, CORRECT, stop_after="L2")
        self.assertEqual(out["notes"], [])          # 只跑到 L2：没有基线可谈

        from operator_agent.core.contract import freeze_from_task
        contract = freeze_from_task(ht.build_task(declaration()))
        notes = ht._judgement_notes(contract, {"baseline_type": "pytorch_eager_tf32"})
        self.assertTrue(notes and "不是同数值语义的比较" in notes[0])
        self.assertIn("input_precision", notes[0])
        # 契约自己声明 tf32=True 时不该再说这句
        self.assertEqual(
            ht._judgement_notes(
                type(contract)(**{**contract.__dict__,
                                  "dtype_semantics": type(contract.dtype_semantics)(
                                      dtype=contract.dtype_semantics.dtype, tf32=True,
                                      accum=contract.dtype_semantics.accum,
                                      deterministic=contract.dtype_semantics.deterministic)}),
                {"baseline_type": "pytorch_eager_tf32"}),
            [])

    def test_the_payload_stays_small_enough_for_a_model_to_read(self):
        """外层的模型是**直接读**这个 JSON 的：逐 case 的输入保真明细不能进来。

        2026-10-18 实测：不裁的时候一次 L2 回包 40 KB（`correctness.cases[*].metrics`
        里是完整的 storage/view 回读）；裁到「只留 verdict/reason/身份」后 4.8 KB。
        完整指标留在 `evidence` 指向的独立 result.json 中，读者按需读取。
        """
        out = ht.verify(self.round_dir, CORRECT)
        size = len(json.dumps(out, ensure_ascii=False))
        self.assertLess(size, 12_000, f"回包 {size} B 太大（模型要直接读它）")
        correctness = next(v["metrics"]["correctness"] for v in out["verdicts"]
                           if v["level"] == "L2")
        self.assertLessEqual(len(correctness["cases"]), 10)
        self.assertIn("cases_total", correctness)
        for case in correctness["cases"]:
            self.assertNotIn("metrics", case, "逐 case 的明细不该进工具回包")
            self.assertIn("case_id", case)

    def test_a_report_and_a_log_are_written_next_to_the_round(self):
        out = ht.verify(self.round_dir, CORRECT)
        self.assertTrue(Path(out["report"]).is_file())
        self.assertTrue(Path(out["log"]).is_file())
        self.assertGreater(Path(out["report"]).stat().st_size, 0)
        self.assertIn("Verifier", Path(out["log"]).read_text(encoding="utf-8"))

    def test_rounds_are_numbered_and_never_overwrite_each_other(self):
        first = ht.verify(self.round_dir, CORRECT, stop_after="L1")
        second = ht.verify(self.round_dir, CORRECT, stop_after="L1")
        self.assertGreater(second["round"], first["round"])
        self.assertTrue(Path(first["log"]).is_file())
        self.assertTrue(Path(second["log"]).is_file())
        self.assertNotEqual(first["log"], second["log"])


class TestTheRoundIsTamperEvident(unittest.TestCase):

    def test_editing_the_declaration_after_prepare_is_caught(self):
        run = Path(tempfile.mkdtemp(prefix="hitl-drift-")) / "run"
        ht.prepare(declaration(), run)
        path = run / "declaration.json"
        edited = json.loads(path.read_text(encoding="utf-8"))
        edited["shapes"] = [[32, 32, 32]]                  # 判据被改过
        path.write_text(json.dumps(edited), encoding="utf-8")

        out = ht.verify(run, CORRECT)
        self.assertFalse(out["ok"])
        self.assertEqual(out["error"], "contract_drift")
        self.assertNotEqual(out["recorded_hash"], out["current_hash"])

    def test_verify_without_prepare_is_refused(self):
        with self.assertRaises(ht.ToolError) as ctx:
            ht.verify(tempfile.mkdtemp(), CORRECT)
        self.assertEqual(ctx.exception.code, "run_not_prepared")

    def test_empty_source_is_refused(self):
        run = Path(tempfile.mkdtemp(prefix="hitl-empty-")) / "run"
        ht.prepare(declaration(), run)
        with self.assertRaises(ht.ToolError) as ctx:
            ht.verify(run, "   ")
        self.assertEqual(ctx.exception.code, "code_required")


class TestCapabilitiesDoesNoWork(unittest.TestCase):

    def test_it_reports_without_probing_the_network(self):
        caps = ht.capabilities()
        self.assertTrue(caps["ok"])
        self.assertTrue(caps["backends"]["mock"]["wired"])
        # 两个都接线了，但都带着**限制说明**：local 没有输入回读协议，ssh 需要 host。
        self.assertTrue(caps["backends"]["local"]["wired"])
        self.assertTrue(caps["backends"]["ssh"]["wired"])
        self.assertTrue(caps["backends"]["local"]["limits"])
        self.assertTrue(caps["backends"]["ssh"]["limits"])
        self.assertIn("matmul", caps["kernel_types_with_input_layout"])
        self.assertNotIn("attention", caps["kernel_types_with_input_layout"])
        self.assertTrue(any("不做网络探测" in n for n in caps["notes"]))


class TestTheCliKeepsStdoutParseable(unittest.TestCase):
    """外层宿主只读 stdout，所以「库里的 print」一行都不许漏出来。"""

    def _run(self, argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cli_main(argv)
        return code, buf.getvalue()

    def test_prepare_and_verify_emit_one_json_object(self):
        tmp = Path(tempfile.mkdtemp(prefix="hitl-cli-"))
        run = tmp / "run"
        code, out = self._run([
            "prepare", "--request", "把 64x64x64 的 fp32 矩阵乘法写成算子",
            "--shapes", "64,64,64", "--run-dir", str(run), "--min-speedup", "0.5",
        ])
        self.assertEqual(code, 0)
        payload = json.loads(out)                 # 整份 stdout 必须是合法 JSON
        self.assertTrue(payload["ok"])

        cand = tmp / "cand.py"
        cand.write_text(CORRECT, encoding="utf-8")
        code, out = self._run(["verify", "--run-dir", str(run),
                               "--code-file", str(cand), "--stop-after", "L2"])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["levels"], {"L1": True, "L2": True})

    def test_a_noisy_backend_cannot_pollute_stdout(self):
        """真机第一次 L1 实测：`[RemoteExecutor] SSH 连接成功…` 跑进 stdout，JSON 解析失败。

        这里用一个**会 print 的替身后端**钉住那条纪律，不需要真的连 GPU。
        """
        import sys
        from unittest import mock
        from operator_agent.app import hitl_tools as real, tool_calls
        from operator_agent.execute import exec_guard
        # S2 executes tools in a child: inject the noisy backend in that child,
        # so this still checks the actual CLI/worker/log boundary.
        script = '''
import sys
from operator_agent.app import hitl_tools as ht, tool_calls
class Noisy:
    name = "NoisyBackend"
    capabilities = frozenset()
def make(name, **options):
    print("  [RemoteExecutor] SSH 连接成功: 假后端")
    return Noisy(), dict(requested=name, resolved="NoisyBackend", degraded=False, attempts=[])
ht._make_backend = make
tool_calls.main([sys.argv[1]])
'''
        def start(directory):
            return exec_guard.start_task_worker([sys.executable, '-c', script, str(directory)],
                cwd=str(Path(__file__).resolve().parents[2]), log_path=directory/'execution.log')

        tmp = Path(tempfile.mkdtemp(prefix="hitl-noisy-"))
        run = tmp / "run"
        real.prepare(declaration(), run)
        cand = tmp / "c.py"
        cand.write_text(CORRECT, encoding="utf-8")
        with mock.patch.object(tool_calls, '_start_call_worker', start):
            code, out = self._run(["verify", "--run-dir", str(run),
                                   "--code-file", str(cand), "--stop-after", "L1"])
        payload = json.loads(out)                 # 有污染就会在这里炸
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["backend_resolution"]["resolved"], "NoisyBackend")
        log = Path(payload["log"]).read_text(encoding="utf-8")
        self.assertIn("SSH 连接成功", log, "后端构造期的输出应当进日志，而不是丢掉")

    def test_a_refusal_is_exit_code_two_with_a_reason(self):
        code, out = self._run([
            "prepare", "--request", "把 64x64x64 的 fp32 矩阵乘法写成算子",
            "--run-dir", tempfile.mkdtemp(prefix="hitl-cli-bad-")])
        self.assertEqual(code, 2)
        payload = json.loads(out)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"], "shapes_required")

    def test_capabilities_is_json(self):
        code, out = self._run(["capabilities"])
        self.assertEqual(code, 0)
        self.assertIn("backends", json.loads(out))


class TestBenchIsExplorationGrade(unittest.TestCase):
    """`bench` 读数只能用来选方向，不能当依据 —— 这条在**结构上**守住。

    设计稿：docs/design/hitl-bench-sweep.md。为什么不是「有但为 false」：一旦回包里
    出现 `eligibility`，调用方就会拿它替代 `verify`（那正是它存在的动机），
    所以这里钉的是「键根本不存在」。
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="hitl-bench-"))
        cls.round_dir = cls.tmp / "run"
        ht.prepare(declaration(), cls.round_dir)

    def test_it_reports_numbers_per_arm_and_no_verdict_fields(self):
        out = ht.bench(self.round_dir, [
            {"name": "base", "code": CORRECT},
            {"name": "slow", "code": WRONG},
        ])
        self.assertTrue(out["ok"])
        self.assertEqual(out["grade"], "exploration")
        self.assertEqual(len(out["arms"]), 2)
        self.assertEqual(out["measured"], 2)
        self.assertFalse(out["partial"])
        # 裁决字段**一个都不许有**（顶层；逐臂的 validity 是测量的性质，允许）。
        for banned in ("levels", "route", "eligibility", "passed", "verdicts", "validity"):
            self.assertNotIn(banned, out, f"bench 回包里出现了裁决字段 {banned}")
        arm = out["arms"][0]
        self.assertEqual(arm["name"], "base")
        for key in ("median_ms", "speedup", "gflops", "validity"):
            self.assertIn(key, arm)
        self.assertIn(out["best"], [a["name"] for a in out["arms"]])

    def test_it_writes_bench_files_and_never_a_verify_round(self):
        out = ht.bench(self.round_dir, [{"name": "only", "code": CORRECT}])
        self.assertTrue((self.round_dir / f"bench-{out['batch']}.json").is_file())
        self.assertTrue((self.round_dir / f"bench-{out['batch']}.log").is_file())
        # 这是「不入库/不晋级」的落盘半边：verify 的轮次文件一个都不该多出来。
        self.assertEqual(list(self.round_dir.glob("verify-*.json")), [])

    def test_a_broken_candidate_gets_no_best_and_no_fake_numbers(self):
        """L1 都过不了的候选：拿不到 speedup，也就不该被选成 best（不猜）。"""
        broken = "def kernel(A, B):\n    this is not python\n"
        out = ht.bench(self.round_dir, [{"name": "broken", "code": broken},
                                        {"name": "ok", "code": CORRECT}])
        self.assertTrue(out["ok"])
        broken_arm = next(a for a in out["arms"] if a["name"] == "broken")
        self.assertIsNone(broken_arm["speedup"])
        self.assertTrue(broken_arm["error"], "炸掉的臂必须带原因，不许猜")
        # 一个臂炸了不毁整批：另一个臂照样有数，且 best 不会指向炸掉的那个。
        self.assertNotEqual(out["best"], "broken")
        self.assertEqual(out["measured"], 2)

    def test_the_batch_cap_is_explicit(self):
        with self.assertRaises(ht.ToolError) as ctx:
            ht.bench(self.round_dir, [{"name": f"a{i}", "code": CORRECT} for i in range(3)],
                     max_candidates=2)
        self.assertEqual(ctx.exception.code, "too_many_candidates")

    def test_a_drifted_contract_is_refused(self):
        run = Path(tempfile.mkdtemp(prefix="hitl-bench-drift-")) / "run"
        ht.prepare(declaration(), run)
        contract_path = run / "contract.json"
        frozen = json.loads(contract_path.read_text(encoding="utf-8"))
        frozen["hash"] = "0" * 64
        contract_path.write_text(json.dumps(frozen), encoding="utf-8")
        out = ht.bench(run, [{"name": "a", "code": CORRECT}])
        self.assertFalse(out["ok"])
        self.assertEqual(out["error"], "contract_drift")


class TestCandidateOriginIsAttributionNotJudgement(unittest.TestCase):
    """`candidate_origin` 是**归因格**：记「这份源码自称是谁写的」，仅此而已。

    这一层要钉住的不是「它能记」，而是三条**不许**：
    ① 不许因为没声明就省略这一格（省略 = 把「走了哪条路」重新变成猜，
      见 `docs/design/hitl-generation-boundary.md` §4）；
    ② 不许影响任何判定字段（否则它就成了判据，而判据不许由调用方给）；
    ③ 不许进契约（否则声明一下来源就会改 `contract_hash`，把回合判成 drift）。
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="hitl-origin-"))
        cls.round_dir = cls.tmp / "run"
        ht.prepare(declaration(), cls.round_dir)

    #: 判定字段：来源变了这些必须一格都不动。
    JUDGEMENT_KEYS = ("levels", "passed", "verdicts", "route", "eligibility",
                      "validity", "conflict", "contract_hash", "notes")

    def test_a_declared_origin_lands_in_the_payload_and_the_round_file(self):
        out = ht.verify(self.round_dir, CORRECT, stop_after="L1",
                        candidate_origin="outer")
        self.assertEqual(out["candidate_origin"], "outer")
        on_disk = json.loads(Path(out["log"]).with_suffix(".json")
                             .read_text(encoding="utf-8"))
        self.assertEqual(on_disk["candidate_origin"], "outer")

    def test_an_undeclared_origin_is_recorded_not_omitted(self):
        """**核心那条**：没声明 ⇒ 记 `undeclared`，而不是没有这一格。"""
        out = ht.verify(self.round_dir, CORRECT, stop_after="L1")
        self.assertIn("candidate_origin", out)
        self.assertEqual(out["candidate_origin"], ht.UNDECLARED_ORIGIN)
        self.assertEqual(ht.UNDECLARED_ORIGIN, "undeclared")

    def test_every_value_of_the_closed_vocabulary_is_accepted(self):
        for value in ht.CANDIDATE_ORIGINS:
            out = ht.verify(self.round_dir, CORRECT, stop_after="L1",
                            candidate_origin=value)
            self.assertEqual(out["candidate_origin"], value, value)

    def test_an_unknown_origin_is_refused_naming_the_vocabulary(self):
        with self.assertRaises(ht.ToolError) as ctx:
            ht.verify(self.round_dir, CORRECT, stop_after="L1",
                      candidate_origin="inner")
        self.assertEqual(ctx.exception.code, "unknown_candidate_origin")
        for value in ht.CANDIDATE_ORIGINS:
            self.assertIn(value, ctx.exception.detail)

    def test_the_origin_moves_no_judgement_field(self):
        base = ht.verify(self.round_dir, WRONG, stop_after="L2")
        for value in list(ht.CANDIDATE_ORIGINS) + [None]:
            out = ht.verify(self.round_dir, WRONG, stop_after="L2",
                            candidate_origin=value)
            for key in self.JUDGEMENT_KEYS:
                self.assertEqual(out[key], base[key],
                                 f"{key} 被 candidate_origin={value!r} 改了")

    def test_the_origin_cannot_enter_the_contract(self):
        """来源不是声明的一部分：混进 prepare 必须按 `unknown_parameter` 拒。"""
        with self.assertRaises(ht.ToolError) as ctx:
            ht.prepare(declaration(candidate_origin="outer"),
                       self.tmp / "poisoned")
        self.assertEqual(ctx.exception.code, "unknown_parameter")
        self.assertNotIn("candidate_origin", ht.ALLOWED_KEYS)
        self.assertNotIn("candidate_origin", ht.FORBIDDEN_KEYS)

    def test_the_report_names_the_origin(self):
        out = ht.verify(self.round_dir, CORRECT, stop_after="L1",
                        candidate_origin="inner_coder")
        self.assertIn("候选来源: `inner_coder`",
                      Path(out["report"]).read_text(encoding="utf-8"))

    def test_a_report_whose_rounds_lack_the_key_gains_no_line(self):
        """`Conductor` 那条路的 artifact 不带这个键 ⇒ 它的报告必须**一字不变**。

        无条件加行会改动既有报告（含 golden 产物），而那条路的「谁写的核」
        不是这一格要回答的问题。
        """
        from operator_agent.core.report import build_report
        from operator_agent.core.schemas import KernelArtifact

        def report(history):
            return build_report(artifact=KernelArtifact(
                kernel_id="k", task_id="t", code=CORRECT, language="python",
                verification_results=[], benchmark_data={},
                iteration_history=history))

        self.assertNotIn("候选来源", report([{"round": 1, "action": "verify"}]))
        self.assertIn("候选来源",
                      report([{"round": 1, "action": "verify",
                               "candidate_origin": "outer"}]))

    def test_the_schema_enum_and_the_runtime_vocabulary_are_one_list(self):
        """schema 与实现各写一份词表 = 「工具说收 A、实现只认 B」。"""
        from operator_agent.app.mcp_server import _tool_definitions
        tools = {t["name"]: t for t in _tool_definitions()}
        for name in ("operator_verify", "operator_repair"):
            enum = tools[name]["inputSchema"]["properties"]["candidate_origin"]["enum"]
            self.assertEqual(enum, list(ht.CANDIDATE_ORIGINS), name)
        self.assertNotIn("candidate_origin",
                         tools["operator_prepare"]["inputSchema"]["properties"])


class TestRepairAttributesTheFinalCodeNotTheInput(unittest.TestCase):
    """`repair` 的 `final_candidate_origin` 是**派生**的，不接受声明。

    内层 Fixer 真改过源码（`repair_loop` 的 `action="fixed"`）⇒ 最终那份就是内层写的，
    哪怕调用方声称传进来的是 `outer`。这一格存在的意义就是让「裸写的核」与
    「内层改过的核」事后可分。
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hitl-repair-origin-"))
        self.run = self.tmp / "run"
        ht.prepare(declaration(), self.run)

    def _repair(self, rounds, origin=None):
        from unittest import mock
        from types import SimpleNamespace

        from operator_agent.generate import fixer as fixer_module, repair_loop
        from operator_agent.execute import llm_client

        def fake_loop(code, *args, **kwargs):
            return {"passed": False, "rounds": rounds, "code": code, "log": "",
                    "initial_full_l2_passed": False, "final_full_l2_passed": False}
        backend = SimpleNamespace(generation_instructions="")

        # `_llm_problem` 与 `LLMClient` 都要挡：前者是配置门，后者一构造就去读
        # `LLM_BASE_URL`（没有就抛），而这条用例测的是**来源怎么派生**，不是凭据。
        with mock.patch.object(ht, "_llm_problem", lambda: None), \
             mock.patch.object(llm_client, "LLMClient", lambda **k: object()), \
             mock.patch.object(ht, "_make_backend", lambda *a, **k: (backend, {})), \
             mock.patch.object(ht, "Verifier", lambda backend: object()), \
             mock.patch.object(fixer_module, "FixerAgent", lambda **k: object()), \
             mock.patch.object(repair_loop, "repair", fake_loop):
            return ht.repair(self.run, CORRECT, backend="mock",
                             candidate_origin=origin)

    def test_a_round_that_really_fixed_makes_the_final_code_inner(self):
        out = self._repair([{"round": 1, "action": "fixed", "source_hash": "a"}],
                           origin="outer")
        self.assertEqual(out["candidate_origin"], "outer")
        self.assertEqual(out["final_candidate_origin"], "inner_fixer")

    def test_no_fix_means_the_caller_still_owns_the_code(self):
        for rounds in ([{"round": 1, "action": "done"}],
                       [{"round": 1, "action": "stop", "why": "hold"}],
                       [{"round": 1, "action": "repair_unchanged_source"}],
                       []):
            out = self._repair(rounds, origin="outer")
            self.assertEqual(out["final_candidate_origin"], "outer", rounds)

    def test_an_undeclared_input_stays_undeclared_when_nothing_was_fixed(self):
        out = self._repair([{"round": 1, "action": "budget_exhausted"}])
        self.assertEqual(out["candidate_origin"], ht.UNDECLARED_ORIGIN)
        self.assertEqual(out["final_candidate_origin"], ht.UNDECLARED_ORIGIN)

    def test_an_unknown_origin_is_refused_with_the_vocabulary(self):
        # `repair` 先查内层 LLM 配置再查参数，所以这里把 LLM 那一关放行，
        # 单独钉「词表外的值被拒」—— 否则这条用例测的是 llm_not_configured。
        from unittest import mock
        with mock.patch.object(ht, "_llm_problem", lambda: None):
            with self.assertRaises(ht.ToolError) as ctx:
                ht.repair(self.run, CORRECT, candidate_origin="inner")
        self.assertEqual(ctx.exception.code, "unknown_candidate_origin")
        for value in ht.CANDIDATE_ORIGINS:
            self.assertIn(value, ctx.exception.detail)


class TestTheRepairCliPathIsAlive(unittest.TestCase):
    """`python -m operator_agent repair` 必须能走到 `repair()`。

    WHY 有这条：repair 子命令曾经没有 `--code-stdin`，而 `_read_code` 对三个来源都做
    `bool(...)` ⇒ 这条**文档里写着的**路径（`dsh-plugin-implementation.md` §0.5.1 工作流 D）
    一次都跑不起来，直接 `AttributeError`。少一个选项不是「功能少一点」，是整条路死掉。
    """

    def test_repair_reaches_the_tool_instead_of_dying_in_argument_reading(self):
        from unittest import mock

        run = Path(tempfile.mkdtemp(prefix="hitl-repair-cli-")) / "run"
        ht.prepare(declaration(), run)
        source = Path(run).parent / "cand.py"
        source.write_text(CORRECT, encoding="utf-8")

        stdout = io.StringIO()
        # 不配内层 LLM：这条用例要证明的是「走到了 repair」，不是「修好了」。
        import os
        with mock.patch.dict(os.environ, dict(LLM_BASE_URL='', LLM_API_KEY='', LLM_API_KEY_FILE='')), \
             contextlib.redirect_stdout(stdout):
            code = cli_main(["repair", "--run-dir", str(run),
                             "--code-file", str(source),
                             "--candidate-origin", "outer"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(stdout.getvalue())["error"], "llm_not_configured")


if __name__ == "__main__":
    unittest.main()
