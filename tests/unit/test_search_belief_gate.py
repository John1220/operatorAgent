"""`can_tune` 门在**生产链路**上的守卫 (CHECKLOG I52)

设计的伪码就是判据 (设计 §10.4):

    obs = oracle.evaluate(c, contract)
    if obs.eligibility.can_tune: belief.update(c, obs)          # ← 这里
    if obs.eligibility.can_count_progress: stagnation.update(belief)

所以 `can_tune=False` 管的是 **belief**: 那个数不许进信念, 于是下一轮的提议只能从上一次
**有资格**的读数出发。它**不是**「停下来」—— 同族策略另有真机用例守着 (`uninformative`
轮次必须跑满预算、不许提前终止), 这条边界在这里也要成对测。

I52 之前的状态: 这个分支里**没有任何门**, 一个五位全 False 的读数照样被拼成 `perf_data`
喂进邻居排序与 LLM prompt。更细的一层: `SearchAgent._llm_optimize` 用 `.get(k, 0)`, 于是
「没有读数」在 prompt 里长得像 `0.00x` —— 一个不存在的事实。

运行: python3 -m unittest tests.test_search_belief_gate -v
"""

import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from operator_agent.generate.agents import SearchAgent
from operator_agent.generate.bandit import Bandit
from operator_agent.app.conductor import Conductor
from operator_agent.core.kernel_db import KernelDatabase
from operator_agent.execute.mock_gpu import MockGPU
from operator_agent.core.schemas import (
    TaskSchema, SourceSpec, TargetSpec, ConstraintSpec, UnifiedSketch,
    TilingStrategy, TaskType, TaskComplexity, Precision,
)
from tests.unit.test_measurement_validity import (
    H20_HW_INFO, LaunchDominatedGPU, NET_MS_LAYERNORM_SMALL, NET_MS_MATMUL_4096,
)

GOOD_KERNEL = "import numpy as np\n\ndef kernel(A, B):\n    return A @ B\n"

FULL_PERF = {
    "gflops": 1234.5, "speedup": 0.42, "target_speedup": 5.0,
    "bound_type": "memory", "avg_time_ms": 1.5, "memory_bandwidth_gbs": 321.0,
    "hw_peak_gflops": 74000.0, "hw_peak_bandwidth_gbs": 4000.0,
    "gflops_utilization_pct": 1.7, "bandwidth_utilization_pct": 8.0,
    "arithmetic_intensity": 10.6667, "ridge_point": 18.5,
    "roofline_bound": "memory", "name_bound": "compute",
    "hw_name": "nvidia h20", "hw_compute_capability": "9.0",
    "optimization_history": [],
}


class _CapturingLlm:
    """记下 Search 那次 chat_json 的 prompt (prompt 是唯一保证被模型看到的通道)"""

    def __init__(self):
        self.prompts = []

    def chat_json(self, messages, **kw):
        self.prompts.append(messages[0]["content"])
        return {}

    def chat(self, messages, **kw):
        return GOOD_KERNEL


def _sketch():
    # 只给 tiling: `memory_plan`/`pipeline` 留默认对象 (传 {} 会在 to_dict 里炸)
    return UnifiedSketch(algorithm="matmul", version=2, target_dsl="triton",
                         tiling=TilingStrategy(64, 64, 32))


class TestNoReadingNeverBecomesANumber(unittest.TestCase):
    """prompt 层: 没有读数时不许印出任何数字 (成对)"""

    def _prompt(self, perf_data):
        llm = _CapturingLlm()
        SearchAgent(llm=llm)._llm_optimize(_sketch(), perf_data)
        return llm.prompts[0]

    def test_a_refused_reading_prints_no_numbers(self):
        txt = self._prompt({"target_speedup": 5.0, "tuning_refused": "can_tune=False",
                            "optimization_history": []})
        self.assertIn("无可用读数", txt)
        self.assertIn("本轮没有任何可用的性能读数", txt)
        for invented in ("0.00x", "0.0%", "1234.50", "321.00", "74000.0"):
            self.assertNotIn(invented, txt, f"{invented}: 这是编出来的事实")

    def test_a_real_reading_still_prints_the_numbers(self):
        """反向: 门不是「永远说没有读数」"""
        txt = self._prompt(dict(FULL_PERF))
        self.assertIn("1234.50", txt)
        self.assertIn("0.42x", txt)
        self.assertNotIn("无可用读数", txt)
        self.assertNotIn("本轮没有任何可用的性能读数", txt)

    def test_the_target_is_a_task_constant_and_stays(self):
        """`target_speedup` 不是读数 ⇒ 即使没有读数也要印 (它是标尺, 不是观测)"""
        txt = self._prompt({"target_speedup": 5.0, "tuning_refused": "can_tune=False"})
        self.assertIn("目标 5.00x", txt)


class _RecordingSearch:
    """替掉真 Search: 记下它收到的 perf_data, 不产队列 (于是每轮都会再问一次)"""

    def __init__(self):
        self.perf_seen = []
        self.last_alternatives = []

    def run(self, sketch, perf_data):
        # `optimization_history` 要**拷一份**: 原样存引用的话, 后面那个「沿用同一份信念」
        # 的场景里首轮快照会跟着一起长, 于是「有没有偷偷累加」这件事就观测不到了。
        snap = dict(perf_data)
        if "optimization_history" in snap:
            snap["optimization_history"] = list(snap["optimization_history"])
        self.perf_seen.append(snap)
        new = UnifiedSketch(algorithm=sketch.algorithm, version=sketch.version + 1,
                            target_dsl=sketch.target_dsl,
                            tiling=TilingStrategy(128, 64, 32))
        return new, "记录用"


class _FlippingGPU(LaunchDominatedGPU):
    """第 1 轮给一个有效读数, 之后给落在地板上的读数 (用来验「沿用上一次信念」)"""

    def __init__(self, first_ms, later_ms):
        super().__init__(first_ms)
        self._first_ms = first_ms
        self._later_ms = later_ms
        self._calls = 0

    def benchmark(self, kernel_code, inputs, warmup=3, repeat=10):
        self._calls += 1
        self._candidate_ms = self._first_ms if self._calls == 1 else self._later_ms
        return super().benchmark(kernel_code, inputs, warmup, repeat)


def _task(min_speedup=5.0):
    return TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(framework="pytorch", kernel_type="matmul", fused_ops=[],
                          complexity=TaskComplexity.MEDIUM),
        target=TargetSpec(hardware="mock_gpu", dsl="python", compute_units=78,
                          warp_size=32),
        constraints=ConstraintSpec(precision=Precision.FP32, min_speedup=min_speedup,
                                   max_error=1e-5),
    )


def _inputs():
    rng = np.random.default_rng(0)
    return {"A": rng.standard_normal((64, 64)).astype(np.float32),
            "B": rng.standard_normal((64, 64)).astype(np.float32)}


class TestTheGateOnTheRealLoop(unittest.TestCase):
    """走生产 `Conductor.execute`, 看 Search 真正收到了什么"""

    maxDiff = 600

    def _run(self, backend, max_rounds=3):
        c = Conductor(backend=backend, kernel_db=KernelDatabase(jsonl_path=None),
                      llm=_LoopLlm(), max_optimize_rounds=max_rounds)
        fake = _RecordingSearch()
        c.search = fake
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            art = c.execute(_task(), _inputs(), None, target_gflops=0.0)
        return fake, art, buf.getvalue()

    def test_an_uninformative_round_hands_search_no_numbers(self):
        fake, art, out = self._run(LaunchDominatedGPU(NET_MS_LAYERNORM_SMALL))
        self.assertTrue(fake.perf_seen, "前提: 优化分支跑到了")
        self.assertTrue(all(p.get("tuning_refused") for p in fake.perf_seen),
                        f"没资格的读数被喂进了 Search: {fake.perf_seen}")
        for p in fake.perf_seen:
            for numeric in ("gflops", "speedup", "roofline_bound", "bound_type"):
                self.assertNotIn(numeric, p, f"{numeric}: 没资格的数不许进信念")
        self.assertIn("不进信念", out)
        # 「不许停下来」: 跑满预算这条同族策略不能被这次改动带走
        self.assertEqual(3, sum(1 for r in (1, 2, 3) if f"轮次 {r}/3" in out))
        self.assertNotIn("提前终止", out)

    def test_a_valid_round_hands_search_the_real_numbers(self):
        """反向: 门不是「永远剥掉数字」"""
        fake, _, _ = self._run(LaunchDominatedGPU(NET_MS_MATMUL_4096))
        self.assertTrue(fake.perf_seen)
        first = fake.perf_seen[0]
        self.assertNotIn("tuning_refused", first)
        self.assertIn("gflops", first)
        self.assertIn("speedup", first)
        self.assertIn("roofline_bound", first)

    def test_the_belief_is_reused_instead_of_rewritten(self):
        """第 1 轮有效、之后落在噪声里 ⇒ 后续轮沿用**第 1 轮**那份信念 (伪码的 belief)

        这个场景是唯一能把三条门同时钉住的地方: 第 1 轮让 `best_gflops > 0` (于是后面
        的轮次**算得出**回报), 后面几轮没资格 ⇒ 回报必须被挡在统计之外、history 不许
        累加、信念不许被改写。
        """
        # 4 轮: Search 在第 1/2/3 轮的末尾各被问一次, 于是「没资格的那一轮有没有往
        # history 里塞东西」在第 3 次问话时看得见 (3 轮时只有 2 次问话, 追加发生在
        # 问话之后 ⇒ 观测不到)。
        fake, art, _ = self._run(_FlippingGPU(NET_MS_MATMUL_4096, NET_MS_LAYERNORM_SMALL),
                                 max_rounds=4)
        self.assertGreaterEqual(len(fake.perf_seen), 2, "至少要有两轮才谈得上沿用")
        first = fake.perf_seen[0]
        for later in fake.perf_seen[1:]:
            self.assertNotIn("tuning_refused", later, "有旧信念可用时不该退回空读数")
            self.assertEqual(first.get("roofline_bound"), later.get("roofline_bound"),
                             "沿用旧信念 ⇒ 轴证据必须与旧的那份一致")
            self.assertEqual(first.get("speedup"), later.get("speedup"),
                             f"没资格的读数覆盖了信念: {first} → {later}")
            # `optimization_history` 会印进下一轮的 prompt ⇒ 它同样是信念的一部分。
            # 注意第 1 轮自己那条**有资格**的记录会正常出现在后面几轮的信念里 (它是
            # 第 1 轮末尾追加的), 所以要钉的不是「长度不变」而是「不再长」:
            # 第 1 轮之后每一轮拿到的 history 长度都必须一样。
        later_lens = {len(p.get("optimization_history") or []) for p in fake.perf_seen[1:]}
        self.assertEqual({1}, later_lens,
                         f"没资格的那一轮数字混进了信念 (history 长度 {later_lens})")
        # 回报门: 这一跑里后面几轮**算得出**回报 (baseline 已在第 1 轮建立, gflops 也报了),
        # 但它们没资格 ⇒ 一个都不许进统计。只看 `total_scored==0` 是抓不住这条的
        # (纯噪声跑里 `best_gflops` 恒为 0, 回报本来就是 None) —— 必须让回报可算。
        arms = ((art.metadata or {}).get("bandit") or {}).get("arms") or {}
        # 第 2 轮起量的是 Search 给的那个 tiling (128x64x32), 读数没资格 ⇒
        # 「试过」要记 (pulls≥1), 回报一个都不许进统计 (scored==0)。这就是 §10.5-3
        # 在搜索侧唯一的实现 —— 变异点: 把 eligible 硬写成 True, 这里立刻变红。
        self.assertEqual(0, arms.get("128x64x32", {}).get("scored"),
                         f"没资格的回报被算进了统计: {arms}")
        self.assertGreaterEqual(arms.get("128x64x32", {}).get("pulls", 0), 1,
                                "「试过」必须记 —— 否则会重复烧机时")
        # 反向: 第 1 轮有资格 (valid) ⇒ 它的回报必须进了统计, 否则这道门是「一律不收」
        self.assertGreaterEqual(arms.get("64x64x32", {}).get("scored", 0), 1,
                                f"有资格的回报反而没进统计: {arms}")

    def test_the_bandit_state_lands_in_the_artifact(self):
        """产物里要能看见「有几轮的数进了统计」, 否则排序是没有出处的黑箱"""
        _, art, _ = self._run(LaunchDominatedGPU(NET_MS_LAYERNORM_SMALL))
        st = (art.metadata or {}).get("bandit")
        self.assertIsNotNone(st, "bandit 状态没落盘")
        self.assertEqual(0, st["total_scored"], "没资格的轮次不许把回报算进统计")
        self.assertGreaterEqual(st["unscored_updates"], 1)

    def test_an_eligible_round_does_feed_the_bandit(self):
        """反向: 有资格的轮次必须真的进统计 (否则 bandit 永远在空转)"""
        _, art, _ = self._run(LaunchDominatedGPU(NET_MS_MATMUL_4096))
        st = (art.metadata or {}).get("bandit") or {}
        self.assertGreaterEqual(st.get("total_scored", 0), 1, f"统计没吃进回报: {st}")


class _ReversingBandit(Bandit):
    """只覆写 `order` 的真 Bandit: §10.3 的选择器要读它的证据 (`pulls` 等), 而「顺序来自
    排序器」这条接线**恰恰**要由一个会反转的排序器来证 —— 子类化而不是打桩全部方法。"""
    """桩 (只验接线): 队列真的过了 `bandit.order` 的话, 消费顺序就该被它反转。

    为什么不拿真 bandit 测端到端顺序: 一个 3 轮的运行里, 臂要**再次**被量到才拿得到
    可用的回报 (奖励口径是「相对上一轮最好的 gflops 的改进」, 首轮没有 baseline),
    那样的排队在短跑里看不出来 —— 硬凑只会造出一个脆弱的用例。所以顺序由单元用例守
    (`tests/unit/test_bandit.py`), 这里只守「Conductor 有没有把它接上」。
    """

    instances = []

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        # `total_scored` 在真 Bandit 里是**只读属性** ⇒ 这个桩不去覆写它, 而是拿一次
        # 真回报把「有证据」这条状态造出来 (桩只覆写 `order`, 其余口径仍是真件的)。
        self.update("__warm__", 0.1, eligible=True)
        self.unscored_updates = 0
        self.ordered = []
        _ReversingBandit.instances.append(self)

    def order(self, items, key=None):
        out = list(reversed(list(items)))
        self.ordered.append([key(x) for x in out])
        return out

    def as_dict(self):
        return {"stub": True}


class _QueueSearch(_RecordingSearch):
    """第一次被问时排两个候选 (先验顺序 128 在前), 之后不再产队列"""

    def run(self, sketch, perf_data):
        self.perf_seen.append(dict(perf_data))
        if not self.last_alternatives:
            self.last_alternatives = [
                UnifiedSketch(algorithm="matmul", version=3, target_dsl="triton",
                              tiling=TilingStrategy(128, 64, 32)),
                UnifiedSketch(algorithm="matmul", version=4, target_dsl="triton",
                              tiling=TilingStrategy(256, 64, 32)),
            ]
        new = UnifiedSketch(algorithm=sketch.algorithm, version=sketch.version + 1,
                            target_dsl=sketch.target_dsl,
                            tiling=TilingStrategy(64, 64, 32))
        return new, "队列用"


class TestTheQueueGoesThroughTheBandit(unittest.TestCase):

    def test_the_consumed_order_is_the_bandits_output(self):
        from unittest import mock
        _ReversingBandit.instances = []
        with mock.patch("operator_agent.app.conductor.Bandit", _ReversingBandit):
            c = Conductor(backend=LaunchDominatedGPU(NET_MS_MATMUL_4096),
                          kernel_db=KernelDatabase(jsonl_path=None),
                          llm=_LoopLlm(), max_optimize_rounds=4)
            c.search = _QueueSearch()
            seen = []
            real_run = c.coder.run

            def spy(sketch, task, **kw):
                seen.append(sketch.tiling.block_m)
                return real_run(sketch, task, **kw)

            c.coder.run = spy
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                c.execute(_task(), _inputs(), None, target_gflops=0.0)
        self.assertTrue(_ReversingBandit.instances, "Conductor 没建 Bandit")
        stub = _ReversingBandit.instances[0]
        self.assertTrue(stub.ordered, "队列没有过 bandit.order")
        # 先验是 [128, 256], 桩反转 ⇒ 消费顺序必须是 256 先
        self.assertIn(["256x64x32", "128x64x32"], stub.ordered,
                      f"order 拿到的不是那批候选: {stub.ordered}")
        # 队列里的候选在后续轮次被消费 (pop 发生在上一轮末尾)。停滞上限 (局部常量 2)
        # 可能让循环提前收尾, 所以只钉**头一个**必须是 bandit 排出来的那个 —— 那正是
        # 「顺序跟不跟 bandit」这件事; 后面还有就顺带钉住。
        tail = [m for m in seen if m in (128, 256)]
        self.assertTrue(tail, f"队列里的候选一个都没被消费: {seen}")
        self.assertEqual(256, tail[0], f"消费顺序没跟着 bandit 走: {seen}")
        if len(tail) > 1:
            self.assertEqual(128, tail[1], f"第二个候选的顺序不对: {seen}")
        self.assertIn("队列按证据重排", buf.getvalue())


class _LoopLlm:
    """Designer/Planner/Coder 共用的一台假 LLM (L1/L2 恒过)"""

    def chat(self, messages, **kw):
        return GOOD_KERNEL

    def chat_json(self, messages, **kw):
        return {"algorithm": "matmul",
                "tiling": {"block_m": 64, "block_n": 64, "block_k": 32},
                "memory_plan": {}, "pipeline": [], "optimization_hints": []}


if __name__ == "__main__":
    unittest.main(verbosity=2)