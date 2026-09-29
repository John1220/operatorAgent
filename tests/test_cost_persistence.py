"""LLM 成本读数过落盘边界 (doc 53 §五.1 / #70)。

现状：写侧 `conductor.py:926` 把账单放进 `artifact.metadata["llm_cost"]`，而
`artifact_store.py:71-74` 的注释自己写着「`metadata` 整个不进 result.json (llm_cost 也一样
只在内存里)」⇒ **每次生产运行的 `result.json` 里没有成本**。

⚠️ 这里本会话犯过一次数错，值得留在文件里：第一版写的是「全仓读者数 0」，那是我按手里那个
**局部变量名** `cost_summary` 去 grep 的结果。按接口上的键名 `llm_cost` 重数，读者其实有
两处 —— `harness_pass_rate.py:176` 的 `cost_from_metadata()` 与 `golden_regression.py:357`，
只是两处读的都是**同一次调用链里的内存对象**（后者再把两列抄进
`tests/golden_registry.json`，那是 golden 路径自己的出口，不是运行产物）。
⇒ 通则：**清点读者要按落盘/接口上的键名数，不能按自己手上那个变量名数**；
否则会把「跨运行没人读」写成「没人读」，既夸大也漏掉真正要修的那一半。

本片补的是跨运行那一半：下一次运行与事后复盘没有内存对象可读。

这与本仓已经修过的两处同族（环境指纹 `artifact_store.py:71-74`、`resume_status`
`:78-81`），所以修法也照那条先例：**单独立顶层键，不整包倒 metadata**。
`metadata` 里还放着「执行快照路径」这类含真实路径的外挂，整包落盘会同时踩两条既有纪律
（落盘键白名单、派生量先剥路径）。

本片**只接观测**：没有任何判据、没有阈值、没有「预算用完就停」那条出口。
阈值要等 #48 那批分布（`rounds-to-pass` 的跨次散布）才定得出来，先加判据＝编一个门。
为什么现在就落盘：第四出口（doc 53 §五.1）的原料是这个数，而它得**跨过落盘边界**
才能被下一次运行读到 —— 只在内存里等于没算。

运行: python3 -B -m unittest tests.test_cost_persistence -v

变异标定 6 条，**0 逃逸**（`operator_agent/artifact_store.py` 与 `conductor.py` 各改坏一次，
期望集合是按「这条变异让哪条断言的前提消失」推出来的，不是跑一遍抄的）：
整块不搬→落盘等式/键名同源/证据质量位/AST 契约四条；`if True`（缺账单也写键）→
`test_missing_cost_writes_no_key`；整包倒 `metadata`→上面三条 + 边界收窄 + AST 契约；
落盘键改名→前三条（AST 契约不响，因为 `.get("llm_cost")` 这个调用还在）；
丢 `n_estimated_usage`→前三条；**写侧**键名改成 `cost_usd`→只有 AST 契约那条响
（单测自己构造 metadata，看不见生产写侧改名 —— 这正是契约要两侧一起守的理由）。
"""

import ast
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from operator_agent.artifact_store import ArtifactStore  # noqa: E402
from operator_agent.llm_client import CostTracker  # noqa: E402
from operator_agent.schemas import KernelArtifact  # noqa: E402


def _tracker() -> CostTracker:
    """两笔调用：一笔真报 usage、一笔估算 —— 后者是证据质量位，必须一起活下来。"""
    t = CostTracker()
    t.record(model="deepseek-chat", prompt_tokens=1000, completion_tokens=500,
             usage_source="api", caller_tag="Coder")
    t.record(model="deepseek-chat", prompt_tokens=700, completion_tokens=200,
             usage_source="estimated", caller_tag="Fixer")
    return t


class _StoreCase(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="costp-"))
        self.store = ArtifactStore(output_dir=str(self.dir))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _save_and_read(self, artifact: KernelArtifact) -> dict:
        saved = self.store.save(artifact)
        # 观察面取本次的权威句柄, 不按业务键去 glob (量具先标定那条纪律的第 25 条)
        with io.open(saved / "result.json", encoding="utf-8") as fh:
            return json.load(fh)


class TestCostReachesResultJson(_StoreCase):
    def test_the_cost_block_crosses_the_persistence_boundary(self):
        t = _tracker()
        art = KernelArtifact(task_id="t1", code="x", metadata={"llm_cost": t.summary()})
        result = self._save_and_read(art)
        self.assertIn("llm_cost", result, "账单又只在内存里活了一次")
        # 同源对照: 落盘的这份必须与生产者的 summary() **整块相等**, 不是挑几个键比
        self.assertEqual(t.summary(), result["llm_cost"])

    def test_the_reader_sees_the_producer_key_names(self):
        """键名与生产者同源。

        CHECKLOG H3 那条 (`cost_usd` vs 真键 `total_cost_usd` ⇒ 全程 $0.0) 的形态是
        「读侧写了一个生产者没有的名字」。这里期望的键集合**从生产者取**, 所以任何一侧
        改名都会被抓到; 另加三个绝对锚点, 防止两边同时坏掉而自洽。
        """
        want = set(CostTracker().summary())
        art = KernelArtifact(task_id="t1", code="x", metadata={"llm_cost": _tracker().summary()})
        got = self._save_and_read(art)["llm_cost"]
        self.assertEqual(sorted(want), sorted(got))
        self.assertEqual(2, got["total_calls"])
        self.assertEqual(1700, got["total_prompt_tokens"])
        self.assertEqual(2400, got["total_tokens"])
        self.assertGreater(got["total_cost_usd"], 0.0, "记账位不许被写成 0")

    def test_the_estimated_usage_flag_survives(self):
        """「这笔钱是报的还是估的」是判据方向的一部分, 不能只留在内存。"""
        art = KernelArtifact(task_id="t1", code="x", metadata={"llm_cost": _tracker().summary()})
        got = self._save_and_read(art)["llm_cost"]
        self.assertEqual(1, got["n_estimated_usage"])
        self.assertIn("breakdown_by_caller", got)
        self.assertEqual({"Coder", "Fixer"}, set(got["breakdown_by_caller"]))

    def test_missing_cost_writes_no_key(self):
        """缺证据不许当否定证据: 没有账单时不造一个 `{"total_cost_usd": 0}`。"""
        self.assertNotIn("llm_cost", self._save_and_read(KernelArtifact(task_id="t", code="x")))
        art = KernelArtifact(task_id="t2", code="x",
                             metadata={"remote_workdir": "/somewhere/keep/private"})
        self.assertNotIn("llm_cost", self._save_and_read(art))


class TestTheBoundaryStaysNarrow(_StoreCase):
    def test_the_rest_of_metadata_is_not_dumped(self):
        """只搬登记过的键。整包倒出来时未登记的键会跟着过界。

        ⚠️ 那两个诱饵是**合成的**：今天 `artifact.metadata` 只有 `stop` 与 `llm_cost`
        两个写者（`CHECKLOG I8` 数过），本条钉的是那道门本身，不是对当前内容的描述。
        """
        art = KernelArtifact(task_id="t1", code="x", metadata={
            "llm_cost": _tracker().summary(),
            "remote_workdir": "/somewhere/keep/private",
            "execution_snapshot": "/somewhere/snap.tar",
        })
        result = self._save_and_read(art)
        self.assertNotIn("metadata", result)
        for key in ("remote_workdir", "execution_snapshot"):
            self.assertNotIn(key, json.dumps(result), f"{key} 跟着整包落盘了")

    def test_writer_and_reader_use_the_same_key(self):
        """装配环节的契约要两侧一起守: 写侧 `metadata["llm_cost"]` 与读侧 `.get("llm_cost")`
        必须是同一个字面量。只查一侧 = 半扇门 (改名时另一侧静默读到空)。"""
        writer = io.open(ROOT / "operator_agent" / "conductor.py", encoding="utf-8").read()
        reader = io.open(ROOT / "operator_agent" / "artifact_store.py", encoding="utf-8").read()

        def assigned_literal(src: str, dict_expr: str) -> set:
            out = set()
            for node in ast.walk(ast.parse(src)):
                if (isinstance(node, ast.Assign) and len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Subscript)
                        and dict_expr in ast.dump(node.targets[0].value)):
                    sl = node.targets[0].slice
                    key = sl.value if isinstance(sl, ast.Constant) else getattr(sl, "value", None)
                    if isinstance(key, str):
                        out.add(key)
            return out

        written = assigned_literal(writer, "metadata")
        self.assertIn("llm_cost", written, f"conductor 不再往 metadata['llm_cost'] 写账单: {written}")

        read_keys = {n.args[0].value for n in ast.walk(ast.parse(reader))
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                     and n.func.attr == "get" and n.args
                     and isinstance(n.args[0], ast.Constant)
                     and isinstance(n.args[0].value, str)}
        self.assertIn("llm_cost", read_keys, "artifact_store 不再按 'llm_cost' 读")
        # 反向: 读侧读的那个键必须就是写侧写的那个, 两边各自存在但不同名同样坏
        self.assertIn("llm_cost", written & read_keys)


if __name__ == "__main__":
    unittest.main(verbosity=2)
