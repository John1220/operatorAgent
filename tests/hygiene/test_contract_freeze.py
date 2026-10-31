"""契约冻结的离线标定 (design/autonomous-operator-agent-design.md §3 / §10.5-5)

四条要命的失效, 各配一个方向检查:
  1. 内容变一个字, 哈希必须变
  2. 同内容两份拷贝, 哈希必须相同 (不是 mtime)
  3. 缺数值语义 / 基线不同卡同窗 —— 拒绝冻结, 不猜默认
  4. 哈希不匹配时评测器拒评
"""
import unittest

from operator_agent.core.contract import (BaselineDef, ContractError, DTypeSemantics,
                                     FrozenContract, ShapeDist, assert_hash,
                                     freeze, load_dict)


def _ok_contract(**kw):
    args = dict(
        kernel_type="matmul",
        dtype_semantics=DTypeSemantics(dtype="fp16", tf32=False, accum="fp32"),
        baseline=BaselineDef(name="cuBLAS_fp16", same_device=True,
                             same_window=True, same_dtype_semantics=True),
        shapes=ShapeDist(entries=(((4096, 4096, 4096), 1.0),)),
        max_error=1e-2,
        min_speedup=1.0,
    )
    args.update(kw)
    return freeze(**args)


class TestHashIsContentAddressed(unittest.TestCase):
    def test_same_content_same_hash(self):
        a, b = _ok_contract(), _ok_contract()
        self.assertEqual(a.hash, b.hash, "同内容哈希不同 —— 不是内容哈希")

    def test_one_field_change_changes_hash(self):
        a = _ok_contract()
        b = _ok_contract(max_error=1e-3)
        self.assertNotEqual(a.hash, b.hash)

    def test_dtype_semantics_change_changes_hash(self):
        """TF32 开关差 2x 加速比 (设计 §3.2) —— 必须进哈希"""
        a = _ok_contract()
        b = _ok_contract(dtype_semantics=DTypeSemantics(
            dtype="fp16", tf32=True, accum="fp32"))
        self.assertNotEqual(a.hash, b.hash)

    def test_baseline_change_changes_hash(self):
        a = _ok_contract()
        b = _ok_contract(baseline=BaselineDef(
            name="pytorch_eager", same_device=True, same_window=True,
            same_dtype_semantics=True))
        self.assertNotEqual(a.hash, b.hash)

    def test_notes_do_not_affect_hash(self):
        """说明性文字不该让「同一把尺子」变成两把"""
        a = _ok_contract(notes="foo")
        b = _ok_contract(notes="bar")
        self.assertEqual(a.hash, b.hash)


class TestFreezeRefusesUnjudgeable(unittest.TestCase):
    def test_empty_kernel_type_refused(self):
        with self.assertRaises(ContractError):
            _ok_contract(kernel_type="  ")

    def test_nonpositive_tol_refused(self):
        with self.assertRaises(ContractError):
            _ok_contract(max_error=0)
        with self.assertRaises(ContractError):
            _ok_contract(min_speedup=0)

    def test_baseline_must_be_same_device_window_semantics(self):
        with self.assertRaises(ContractError) as cm:
            _ok_contract(baseline=BaselineDef(
                name="cuBLAS", same_device=False, same_window=True,
                same_dtype_semantics=True))
        self.assertIn("同卡", str(cm.exception))

    def test_empty_shape_dist_refused(self):
        with self.assertRaises(ContractError):
            _ok_contract(shapes=ShapeDist(entries=()))

    def test_bad_accum_refused(self):
        with self.assertRaises(ContractError):
            _ok_contract(dtype_semantics=DTypeSemantics(dtype="fp32", accum="float4"))


class TestAssertHashRejectsRewrittenRuler(unittest.TestCase):
    def test_matching_hash_passes(self):
        c = _ok_contract()
        assert_hash(c, c.hash)   # 不抛即通过

    def test_wrong_hash_raises(self):
        c = _ok_contract()
        with self.assertRaises(ContractError) as cm:
            assert_hash(c, "0" * 64)
        self.assertIn("拒", str(cm.exception))

    def test_empty_expected_hash_raises(self):
        with self.assertRaises(ContractError):
            assert_hash(_ok_contract(), "")


class TestRoundTrip(unittest.TestCase):
    def test_load_dict_reproduces_same_hash(self):
        c = _ok_contract()
        c2 = load_dict(c.as_dict())
        self.assertEqual(c.hash, c2.hash)
        self.assertEqual(c.content_dict(), c2.content_dict())

    def test_load_dict_of_tampered_payload_is_refused(self):
        """spine-review P3-15: 载荷自带哈希与内容重算不一致 → 拒绝重建

        原先忽略载荷里的 hash 重新 freeze(), 于是「被改过的载荷」拿到一个合法
        **新**哈希, 改写检测不出来。现在 load_dict 拒绝它。
        """
        d = _ok_contract().as_dict()
        d["min_speedup"] = 2.0
        with self.assertRaises(ContractError) as cm:
            load_dict(d)
        self.assertIn("改写", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
