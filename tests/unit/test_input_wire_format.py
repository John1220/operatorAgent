"""输入线的格式: 标量必须**按值**到达远端 (2026-10-01 真机实测抓到)

`SSHExecutor._serialize_inputs` 与 `remote_scripts/run_kernel.py:_deserialize_inputs`
是一对, 中间是 SSH stdin 上的一段 JSON。数组走「shape+dtype+seed, 远端重新生成」
(刻意如此, 避免 67MB 的 JSON); 而**标量也走了同一条路**, 于是它的值被一个 N(0,1)
抽样顶掉再按 dtype 截断。algorithm-h20 上的探针实测:

    传入 causal=False  →  远端 np.True_         (布尔被翻转)
    传入 eps=1e-5      →  远端 np.float64(-0.0223)
    传入 repeats=8     →  远端 np.int64(0)
    传入 scale=2.5     →  远端 np.float64(0.0771)

这不是「数不准」而是**验的不是同一个算子**, 且 L2 会照样给出通过/不通过。这个接缝
此前全仓没有一个测试, 所以它活了很久 —— 下面三条同时钉住: 标量按值、数组按描述、
以及载荷本身能过 JSON。
"""
import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np

from operator_agent.execute.remote_executor import SSHExecutor

ROOT = Path(__file__).resolve().parents[2]


def _runner():
    """加载远端脚本本体 (它与 tolerance/anticheat 同级依赖, 直接在仓库里可导入)"""
    path = ROOT / "remote_scripts" / "run_kernel.py"
    spec = importlib.util.spec_from_file_location("_runner_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestScalarsSurviveTheWire(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = _runner()

    def _round_trip(self, sent):
        wire = SSHExecutor._serialize_inputs(sent)
        json.dumps(wire)          # 它要真的过 SSH stdin 上的 JSON
        return self.runner._deserialize_inputs(json.loads(json.dumps(wire)))

    def test_the_real_machine_regression(self):
        """真机上那四个观测值, 一个都不许复现"""
        A = np.zeros((4, 4), dtype=np.float32)
        got = self._round_trip({"A": A, "B": A, "causal": False, "eps": 1e-5,
                                "repeats": 8, "scale": 2.5})
        self.assertIs(False, bool(got["causal"]), "布尔标量被翻转了 (真机实测: 成了 True)")
        self.assertAlmostEqual(1e-5, float(got["eps"]), places=12)
        self.assertEqual(8, int(got["repeats"]))
        self.assertAlmostEqual(2.5, float(got["scale"]), places=12)

    def test_scalars_are_carried_not_regenerated(self):
        """判据是「按值」而不是「恰好对上」: 同一个键换个值必须跟着变

        旧的重新生成只由 `键|shape|dtype` 派生 seed, 与**值**无关 —— 所以
        `repeats=8` 与 `repeats=3` 会得到同一个数。这条正是那个性质的反面。
        """
        a = self._round_trip({"repeats": 8})["repeats"]
        b = self._round_trip({"repeats": 3})["repeats"]
        self.assertNotEqual(int(a), int(b), "标量值没有参与传输")

    def test_arrays_still_go_by_description(self):
        """数组**刻意**不传值: 传的是 shape+dtype+seed (避免大矩阵 JSON)"""
        A = np.ones((8, 8), dtype=np.float32)
        wire = SSHExecutor._serialize_inputs({"A": A})
        self.assertNotIn("data", json.dumps(wire), "数组被整份传了 —— 那会退化成 67MB JSON")
        self.assertEqual([[8, 8]], [s["shape"] for s in wire["specs"].values()])
        got = self.runner._deserialize_inputs(wire)
        self.assertEqual((8, 8), got["A"].shape)
        self.assertEqual("float32", str(got["A"].dtype))
        # 生成的是正态 × 0.1 的**随机**数据, 不是我们自己给的那份 (这是有意的取舍):
        # 候选与基线各自跑同一套描述, 输入一致, 所以比较仍然公平。
        self.assertFalse(np.allclose(got["A"], A))

    def test_strings_and_none_pass_through(self):
        """字符串/None 也过同一条线 (装不进 «重新生成» 的模具)"""
        got = self._round_trip({"mode": "causal", "flag": None})
        self.assertEqual("causal", got["mode"])
        self.assertIsNone(got["flag"])

    def test_a_scalar_value_must_be_json_native(self):
        """np 标量要落成 Python 标量, 否则 json 序列化就炸 —— 别在 SSH 上才发现"""
        wire = SSHExecutor._serialize_inputs({"r": np.int64(8), "b": np.bool_(False)})
        self.assertIsInstance(wire["scalars"]["r"], int)
        self.assertIsInstance(wire["scalars"]["b"], bool)
        self.assertNotIsInstance(wire["scalars"]["r"], np.generic)


if __name__ == "__main__":
    unittest.main(verbosity=2)