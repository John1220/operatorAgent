"""续跑的完整性 (doc 50) —— 「请求了续跑」与「这次从头跑」不许长得一样

为什么单独有这一组: `execution_snapshot.json` 是每轮都写的**唯一**可续跑状态, 而它的
写与读两侧各有一条静默降级 (2026-09-29 实测, 全部本地可复现):

* 写侧 `open(path,"w") + json.dump` 不是原子操作 —— 那个 `open` 当场把文件截成 0 字节,
  于是「写到一半崩了」留下的是一份**截断的 JSON**, 而不是上一个可用版本。
* 读侧把**任何**异常都吞成一行 print 然后从头跑。合起来的代价是: 一次续跑从 1 次 LLM
  调用变成 4 次, 而产物里没有任何地方说明「这次没续上、为什么」—— 事后无法复盘,
  因为 stdout 不归档。
* 按 task_id 找快照时 (③ 那条分支) 还有两处更静的错: 搜索根写死成 CWD 相对的 `"output"`,
  不认 `OPERATOR_AGENT_OUTPUT_DIR` (设了重定向就找不到, 或者捡到**别的配置**写的快照);
  多个候选时取 `os.walk` 的第一个命中而不是最新的那个 —— 实测两个候选 (round_done 1 与 3)
  **永远选中 round 1**, 也就是把已完成的轮次扔掉重来。

外部依据 (逐条一手核过, 见 doc 50 第二节): POSIX `rename()` 的原子性要求 +
「dst 已存在则先删除再改名」⇒ 「同目录临时文件 + `os.replace`」把「半个快照」这个状态
从可能变成不可能。

⚠️ 这一组刻意**不用**声明式假件当被测对象: 走的是真 `MockGPU` + 真 `Conductor.execute`
+ 真 `ArtifactStore` 落盘 (doc 48 §6.5 那条纪律)。

运行: python3 -m unittest tests.test_resume_integrity -v
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np

from operator_agent.conductor import Conductor
from operator_agent.kernel_db import KernelDatabase
from operator_agent.mock_gpu import MockGPU
from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec, TargetSpec,
                                    TaskComplexity, TaskSchema, TaskType)

SNAP = "execution_snapshot.json"
PYTHON_MATMUL = "import numpy as np\n\ndef kernel(A, B):\n    return A @ B\n"


class _ScriptedLLM:
    SKETCH = {"algorithm": "matmul",
              "tiling": {"block_m": 64, "block_n": 64, "block_k": 32},
              "memory_plan": {}, "pipeline": [], "optimization_hints": []}

    def __init__(self):
        self.calls = 0

    def chat(self, messages, **kw):
        self.calls += 1
        return PYTHON_MATMUL

    def chat_json(self, messages, **kw):
        self.calls += 1
        return dict(self.SKETCH)


def _task(task_id="resumecase", min_speedup=5.0):
    return TaskSchema(
        task_id=task_id,
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(kernel_type="matmul", complexity=TaskComplexity.MEDIUM),
        target=TargetSpec(hardware="mock_gpu", dsl="python"),
        constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=min_speedup,
                                   max_error=1e-3))


def _inputs(n=32):
    rng = np.random.default_rng(0)
    return {"A": rng.standard_normal((n, n)).astype(np.float32),
            "B": rng.standard_normal((n, n)).astype(np.float32)}


def _conductor(out_dir, llm=None, rounds=2):
    c = Conductor(backend=MockGPU(), kernel_db=KernelDatabase(),
                  llm=llm or _ScriptedLLM(), max_optimize_rounds=rounds)
    c.store.output_dir = Path(out_dir)
    return c


def _run(out_dir, task, rounds=1, resume_from=None, llm=None):
    """跑一次完整流水线, 返回 (artifact, stdout, LLM 调用次数)"""
    c = _conductor(out_dir, llm=llm, rounds=rounds)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        art = c.execute(task, _inputs(), target_gflops=0.0, resume_from=resume_from)
    return art, buf.getvalue(), c.llm.calls


class TestTornCheckpointIsNeverInvisible(unittest.TestCase):
    """① 写侧: 崩溃不许留下半个快照; ② 读侧: 读不动不许只印一行"""

    def test_failure_midway_through_the_write_leaves_the_previous_snapshot_readable(self):
        """原子替换的意义就这一条: 读者永远只能看到**完整**的某个版本

        注入的是「序列化到一半抛了」(等价于进程被杀 / 磁盘满)。修前 `open(path,"w")` 已经
        把原文件截空, 于是上一轮那个可用快照没了 —— 而下一次的读侧只会说「改为从头开始」。
        """
        with tempfile.TemporaryDirectory() as d:
            art, _, _ = _run(d, _task("atomic1"), rounds=1)
            snap_path = Path(art.saved_dir) / SNAP
            before = json.loads(snap_path.read_text(encoding="utf-8"))
            with mock.patch("operator_agent.conductor.json.dump",
                            side_effect=TypeError("模拟写到一半崩了")):
                _run(d, _task("atomic1"), rounds=2, resume_from=art.saved_dir)
            self.assertTrue(snap_path.exists(), "写失败把原快照删掉了")
            after = json.loads(snap_path.read_text(encoding="utf-8"))
            self.assertEqual(before["round_done"], after["round_done"],
                             "旧快照被半截写入毁了")
            self.assertEqual([], list(Path(art.saved_dir).glob(SNAP + ".tmp")),
                             "临时文件没被清掉")

    def test_unreadable_snapshot_is_recorded_in_the_artifact(self):
        """截断的快照: 必须**落盘一个状态**, 而不是只在 stdout 说一句"""
        with tempfile.TemporaryDirectory() as d:
            art, _, _ = _run(d, _task("torn1"), rounds=1)
            snap_path = Path(art.saved_dir) / SNAP
            full = snap_path.read_bytes()
            snap_path.write_bytes(full[: len(full) // 2])
            resumed, out, calls = _run(d, _task("torn1"), rounds=2,
                                       resume_from=art.saved_dir)
            status = resumed.resume_status or {}
            self.assertEqual("unreadable_snapshot", status.get("status"),
                             f"读不动的快照没被区分出来: {status}")
            self.assertEqual("JSONDecodeError", status.get("reason"),
                             f"只该存异常类名 (str(e) 可能带路径): {status}")
            self.assertEqual(len(full) // 2, status.get("size_bytes"),
                             "落盘字节数就是那次崩溃的形状, 不存它等于没证据")
            self.assertIn("从头", out)
            # 从产物里读回来: 单测读内存对象看不见落盘边界 (doc 49 §7.9 那条教训)
            result = json.loads((Path(resumed.saved_dir) / "result.json")
                                .read_text(encoding="utf-8"))
            self.assertEqual("unreadable_snapshot", result["resume"]["status"])
            self.assertGreater(calls, 1, "没续上就是从第 1 轮重跑, 这条断言钉住代价可见")

    def test_truncation_at_every_boundary_is_either_whole_or_loud(self):
        """属性式扫描: 任何截断长度上, 要么完整恢复, 要么显式判定读不动

        「半份状态被当成完整状态用」是这一族唯一的真危险 —— 所以不许只测一个截断点。
        """
        with tempfile.TemporaryDirectory() as d:
            art, _, _ = _run(d, _task("sweep1"), rounds=1)
            snap_path = Path(art.saved_dir) / SNAP
            full = snap_path.read_bytes()
            c = _conductor(d)
            loaded_ok = 0
            step = max(1, len(full) // 40)
            cuts = list(range(0, len(full), step)) + [len(full)]   # 不补最后这一刀就永远测不到完整那份
            for cut in cuts:
                snap_path.write_bytes(full[:cut])
                try:
                    data = c._load_snapshot(str(snap_path))
                except (OSError, ValueError) as e:
                    self.assertTrue(type(e).__name__, "异常必须有类名可存 (落盘只存类名)")
                    continue
                loaded_ok += 1
                self.assertEqual(data.get("task_id"), "sweep1",
                                 f"截到 {cut} B 还能解析, 但内容不完整 —— 这就是半份状态")
                self.assertIn("round_done", data)
            self.assertGreater(loaded_ok, 0, "一条都没解析成功 ⇒ 这个快照根本不是 JSON")

    def test_not_requested_and_no_snapshot_are_different_states(self):
        """缺证据不许写成否定证据: 「没请求续跑」与「请求了但找不到」是两个状态"""
        with tempfile.TemporaryDirectory() as d:
            first, _, _ = _run(d, _task("state1"), rounds=1)
            self.assertEqual("not_requested", first.resume_status["status"])
            missing, out, _ = _run(d, _task("state1"), rounds=2,
                                   resume_from=str(Path(d) / "does_not_exist"))
            self.assertEqual("no_snapshot", missing.resume_status["status"])
            self.assertIn("没有 task_id 匹配的完整快照", out)

    def test_task_id_mismatch_is_a_state_of_its_own(self):
        """快照属于别的任务: 那是「找到了但不对」, 不是「没有快照」"""
        with tempfile.TemporaryDirectory() as d:
            art, _, _ = _run(d, _task("ownerA"), rounds=1)
            resumed, out, _ = _run(d, _task("ownerB"), rounds=2,
                                   resume_from=art.saved_dir)
            self.assertEqual("task_mismatch", resumed.resume_status["status"])
            self.assertEqual("ownerA", resumed.resume_status["snapshot_task_id"])
            self.assertIn("不一致", out)


class TestSnapshotLookupFollowsTheConfiguration(unittest.TestCase):
    """③ 按 task_id 找快照: 根目录跟着配置走, 多个候选取最新的那一个"""

    def _write_snap(self, base, name, task_id, round_done):
        sub = Path(base) / name
        sub.mkdir(parents=True, exist_ok=True)
        (sub / SNAP).write_text(json.dumps({"schema_version": 1, "task_id": task_id,
                                            "round_done": round_done}), encoding="utf-8")
        return sub

    def test_lookup_searches_the_configured_dir_not_a_cwd_relative_one(self):
        """设了 `OPERATOR_AGENT_OUTPUT_DIR` 之后, 写死 `"output"` 的搜索会去找一个
        本次配置根本没写过的目录: 要么找不到 (静默从头跑), 要么捡到别的运行的快照。

        这里两个候选都摆出来 (配置里的 round_done=4 与 CWD 相对的 `output/` 里的 7),
        断言拿到的必须是**配置里**那个 —— 反过来只断言「找到了」会让它读错目录也照样绿。
        """
        with tempfile.TemporaryDirectory() as d:
            self._write_snap(Path(d) / "artifacts" / "matmul", "run_real", "p3case", 4)
            self._write_snap(Path(d) / "cwd" / "output" / "matmul", "decoy", "p3case", 7)
            c = _conductor(str(Path(d) / "artifacts"))
            old = os.getcwd()
            os.chdir(str(Path(d) / "cwd"))
            try:
                got, scan = c._resolve_snapshot_path("p3case")
            finally:
                os.chdir(old)
            self.assertIsNotNone(got, "配置里的快照没被找到")
            self.assertEqual(4, json.loads(Path(got).read_text())["round_done"],
                             "捡到的是本次配置根本没写过的目录里那个快照")
            self.assertEqual(str(c.store.output_dir), scan["root"],
                             "搜索根与实际落盘根不同源")

    def test_newest_round_wins_when_two_snapshots_exist_for_one_task(self):
        """修前是 `os.walk` 的第一个命中: 目录名把旧的排在前面时, 续跑会丢掉已完成的轮次"""
        with tempfile.TemporaryDirectory() as d:
            # 刻意让**字典序**与轮次反着: 旧的 round_done=1 目录名排在前面
            self._write_snap(d, "aaa_run_round1", "p2case", 1)
            newer = self._write_snap(d, "zzz_run_round3", "p2case", 3)
            c = _conductor(d)
            got, scan = c._resolve_snapshot_path("p2case")
            self.assertEqual(str(newer / SNAP), got,
                             "选中了不是最新的那一个 —— 已完成的轮次被扔掉")
            self.assertEqual(3, json.loads(Path(got).read_text())["round_done"])
            self.assertEqual(2, scan["candidates"])

    def test_unreadable_candidates_are_counted_not_silently_skipped(self):
        """坏的那条不许无声消失: 计数进 scan, 于是「找不到」能说清是**为什么**找不到"""
        with tempfile.TemporaryDirectory() as d:
            bad = self._write_snap(d, "run_bad", "p4case", 5)
            (bad / SNAP).write_text('{"schema_version": 1, "task_id": "p4c',
                                    encoding="utf-8")
            c = _conductor(d)
            got, scan = c._resolve_snapshot_path("p4case")
            self.assertIsNone(got)
            self.assertEqual(1, scan["unreadable"])
            self.assertEqual(0, scan["candidates"])

    def test_staging_snapshot_lands_under_the_configured_root(self):
        """第 1 轮还没 save 时的 staging 目录也不许绕开配置 (同一条 env 的两个分支)"""
        with tempfile.TemporaryDirectory() as d:
            from operator_agent.schemas import KernelArtifact, VerificationResult
            from operator_agent.schemas import VerificationLevel
            c = _conductor(d)
            got = c._flush_snapshot(task=_task("stage1"), artifact_dir=None, round_done=1,
                                    current_sketch=None, current_code=PYTHON_MATMUL,
                                    pending_fix=None, iteration_history=[],
                                    best_gflops=0.0, rounds_without_improvement=0,
                                    optimization_history=[], applied_recipes=set(),
                                    disabled_recipes=set(),
                                    all_verification_results=[
                                        VerificationResult(level=VerificationLevel.L1_COMPILE,
                                                           passed=False)])
            self.assertIsNotNone(got)
            self.assertTrue(Path(got).is_relative_to(Path(d)),
                            f"staging 快照落到了配置之外: {got}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
