"""§12 的候选联合类型、参数空间、便宜代理剪枝与那个开关

设计 §12 的原话是「`Candidate` 必须是**联合类型**」，开关是「`param_space()` 返回
`None` 的后端 = **代码型后端**」。这里钉三件事：两类候选的**去重口径必须分开**
（模板候选没有源码可签名）、参数空间的枚举**顺序确定**、剪枝的三态与「不饿死这一轮」。

运行: python3 -m unittest tests.test_candidates -v
"""

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operator_agent.candidates import (
    ParamSpace, SourceCandidate, TemplateCandidate, as_lineage, dedup_key,
    is_template, param_space_of, prune_by_smem, require_supported_space,
    smem_ok, source_candidate, space_origin, tile_smem_bytes,
)
from operator_agent.backend_caps import Backend
from operator_agent.mock_gpu import MockGPU
from operator_agent.resource_check import SMEM_SLACK, smem_limit_bytes


@dataclass
class _Caps:
    shared_mem_per_sm_kb: int = 228


@dataclass
class _NoSmemCaps:
    """没有 smem 这个字段的设备描述 —— 「不知道」必须与「0」分开"""

    launch_floor_us: float = 3.0


_KERNEL_A = '''import triton
import triton.language as tl

@triton.jit
def kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M: tl.constexpr):
    pid = tl.program_id(0)
'''

#: 与 `_KERNEL_A` **α-等价**（只换了名字）—— 去重口径必须把它们认成同一个候选
_KERNEL_A_RENAMED = _KERNEL_A.replace("a_ptr", "x_ptr").replace("b_ptr", "y_ptr") \
    .replace("c_ptr", "z_ptr").replace("M, N, K", "M2, N2, K2")

#: 结构不同（多了一个 program_id 轴）
_KERNEL_B = _KERNEL_A.replace("pid = tl.program_id(0)", "pid = tl.program_id(0) + tl.program_id(1)")


class TestTheUnionIsATypeNotAString(unittest.TestCase):

    def test_source_candidate_carries_the_alpha_equivalent_signature(self):
        c = source_candidate("triton", _KERNEL_A)
        self.assertEqual(c.struct_sig, source_candidate("triton", _KERNEL_A).struct_sig)
        self.assertTrue(c.struct_sig, "构造时必须当场算好签名 (去重口径的唯一来源)")

    def test_alpha_equivalent_sources_are_the_same_candidate(self):
        a = source_candidate("triton", _KERNEL_A)
        b = source_candidate("triton", _KERNEL_A_RENAMED)
        self.assertEqual(dedup_key(a), dedup_key(b), "改了变量名仍是同一个结构")
        self.assertEqual(1, len({a, b}), "候选要能进 set —— 去重口径即身份")
        self.assertNotEqual(dedup_key(a), dedup_key(source_candidate("triton", _KERNEL_B)))

    def test_dedup_key_never_falls_back_to_none(self):
        """三种情况都要有键: 有签名 / 没有签名但有源码 / 两个都没有"""
        with_sig = SourceCandidate("triton", _KERNEL_A, {}, "sig")
        no_sig = SourceCandidate("triton", _KERNEL_A, {}, "")
        empty = SourceCandidate("triton", "", {}, "")
        self.assertEqual(("source", "triton", "sig"), dedup_key(with_sig))
        self.assertIn(_KERNEL_A, dedup_key(no_sig))
        self.assertEqual(("source", "triton", ""), dedup_key(empty))
        self.assertEqual(3, len({dedup_key(with_sig), dedup_key(no_sig), dedup_key(empty)}))

    def test_template_params_order_does_not_change_identity(self):
        a = TemplateCandidate("cutlass", "sm80", {"BM": 128, "BN": 64}, "1.0")
        b = TemplateCandidate("cutlass", "sm80", {"BN": 64, "BM": 128}, "1.0")
        self.assertEqual(dedup_key(a), dedup_key(b), "字典顺序不该决定是不是同一个点")
        self.assertEqual(1, len({a, b}))

    def test_instantiator_version_is_part_of_identity(self):
        a = TemplateCandidate("cutlass", "sm80", {"BM": 128}, "1.0")
        b = TemplateCandidate("cutlass", "sm80", {"BM": 128}, "1.1")
        self.assertNotEqual(dedup_key(a), dedup_key(b),
                            "同一个点位由不同版本的实例化器产出 = 不同的构建")

    def test_the_two_kinds_never_collide_and_lineage_says_which(self):
        s = SourceCandidate("x", "code", {}, "sig")
        t = TemplateCandidate("x", "sig", {}, "")          # 故意让 id 与签名同串
        self.assertNotEqual(dedup_key(s), dedup_key(t), "两类候选的键空间不许交叉")
        self.assertFalse(is_template(s))
        self.assertTrue(is_template(t))
        self.assertEqual("source", as_lineage(s)["kind"])
        self.assertEqual("template", as_lineage(t)["kind"])
        self.assertEqual("sig", as_lineage(t)["template_id"],
                         "template_id 是第二个位置参数 (上面故意与源码签名同串)")

    def test_template_detection_is_by_type_not_by_attribute(self):
        """带 `source` 属性的模板候选仍是模板候选 (按类型断)"""
        s = SourceCandidate("x", "code", {}, "sig")
        self.assertFalse(is_template(s))
        self.assertNotIn("source", as_lineage(s), "血缘里源码不进快照 (太大且可重算)")


class TestParamSpaceEnumeratesDeterministically(unittest.TestCase):

    def _space(self):
        return ParamSpace(template_id="sm80_gemm",
                          axes={"BM": (64, 128), "BN": (32, 64), "BK": (16,)},
                          instantiator_ver="1.0", backend="cutlass")

    def test_order_is_axis_order_with_the_last_axis_fastest(self):
        pts = self._space().points()
        self.assertEqual(4, len(pts))
        self.assertEqual(4, self._space().size())
        self.assertEqual([(64, 32, 16), (64, 64, 16), (128, 32, 16), (128, 64, 16)],
                         [(p.params["BM"], p.params["BN"], p.params["BK"]) for p in pts])
        self.assertEqual([p.params for p in self._space().points()],
                         [p.params for p in pts], "两次枚举必须逐点相同")

    def test_points_are_template_candidates_with_the_version(self):
        p = self._space().points()[0]
        self.assertTrue(is_template(p))
        self.assertEqual("1.0", p.instantiator_ver)
        self.assertEqual("sm80_gemm", p.template_id)

    def test_empty_space_is_an_error_not_an_empty_list(self):
        """「空间是空的」不许与「空间没配」共用一个返回值"""
        with self.assertRaises(ValueError):
            ParamSpace(template_id="t", axes={})
        with self.assertRaises(ValueError):
            ParamSpace(template_id="t", axes={"BM": ()})
        with self.assertRaises(ValueError):
            ParamSpace(template_id="", axes={"BM": (64,)})

    def test_as_dict_is_snapshot_ready(self):
        d = self._space().as_dict()
        self.assertEqual(4, d["size"])
        self.assertEqual([64, 128], d["axes"]["BM"])


class TestTheSwitchIsOneJudgement(unittest.TestCase):

    def test_a_code_type_backend_inherits_the_none_answer(self):
        b = MockGPU()
        self.assertIsNone(param_space_of(b), "代码型后端 ⇒ None (不是空空间)")
        self.assertEqual("inherited", space_origin(b))
        self.assertIsNone(require_supported_space(None, b), "代码型这条路必须放行")

    def test_a_declared_space_is_refused_loudly_not_degraded(self):
        class _TemplateBackend(MockGPU):
            def param_space(self, contract=None):
                return ParamSpace(template_id="cutlass_sm80", axes={"BM": (64,)},
                                  backend="cutlass")

        b = _TemplateBackend()
        space = param_space_of(b)
        self.assertIsNotNone(space)
        self.assertEqual("overridden", space_origin(b))
        with self.assertRaises(NotImplementedError) as ctx:
            require_supported_space(space, b)
        msg = str(ctx.exception)
        self.assertIn("cutlass_sm80", msg, "拒绝必须点名那个模板")
        self.assertIn("§12", msg, "拒绝必须指向设计里那一条")

    def test_the_base_class_provides_the_protocol_method(self):
        """协议方法立在基类上 ⇒ 「没有这个方法」按设计不可表达 (backend_caps 第 1 条)"""
        self.assertTrue(hasattr(Backend, "param_space"))
        self.assertIsNone(Backend.param_space(MockGPU()))


class TestProxyPrune(unittest.TestCase):

    def _points(self, *sizes):
        return [TemplateCandidate("t", "tiling",
                                  {"BLOCK_M": m, "BLOCK_N": n, "BLOCK_K": k})
                for m, n, k in sizes]

    def test_smem_estimate_follows_the_resource_check_formula(self):
        # (BM*BK + BK*BN + BM*BN) * elem = (64*32 + 32*64 + 64*64) * 4
        self.assertEqual((64 * 32 + 32 * 64 + 64 * 64) * 4,
                         tile_smem_bytes(64, 64, 32, 4))
        self.assertIsNone(tile_smem_bytes(0, 64, 32, 4), "缺一维 ⇒ 不知道, 不是 0")

    def test_nothing_is_dropped_when_it_fits(self):
        res = prune_by_smem(self._points((64, 64, 32)), _Caps())
        self.assertEqual(1, len(res.kept))
        self.assertEqual([], res.dropped)
        self.assertFalse(res.no_limit)
        self.assertEqual("", res.as_note(), "没摘任何点 ⇒ 没有可报的事")

    def test_the_oversized_point_is_named_and_dropped(self):
        res = prune_by_smem(self._points((64, 64, 32), (512, 512, 128)), _Caps(64))
        self.assertEqual([(64, 64, 32)], [(c.params["BLOCK_M"], c.params["BLOCK_N"],
                                           c.params["BLOCK_K"]) for c in res.kept])
        self.assertEqual(1, len(res.dropped))
        self.assertIn("512x512x128", res.as_note())

    def test_the_slack_is_shared_with_the_l1_precheck(self):
        """宽松系数只有**一处**: 剪枝与 L1 预检用的是同一个对象, 不是两个 1.2

        (边界的**逐字节**行为由 `test_resource_precheck` 那条钉着 —— 它曾在我重构
        上界取整时当场变红, 这里只钉「同一个系数」与「留有余量」这两件事。)
        """
        import operator_agent.resource_check as rc
        self.assertIs(SMEM_SLACK, rc.SMEM_SLACK, "两个 1.2 会漂成两套阈值")
        est = tile_smem_bytes(64, 64, 32, 4)     # 32768 B
        self.assertTrue(smem_ok(est, _Caps(30)), "留出余量 ⇒ 放行 (交给远端判定)")
        self.assertFalse(smem_ok(est, _Caps(26)), "明显放不下 ⇒ 摘")
        # ⚠️ 剪枝那条路**也要**钉这个边界, 否则「两处共用一个系数」只是句口号:
        # 单测 `smem_ok` 时把 prune 里的系数改成 1.0, 上面两条断言全绿 (第一版就是这样)。
        # 28 KB: 预检放行 (28×1.2 = 33.6 KB ≥ 32 KB), 若剪枝用 1.0 就会摘掉它。
        small = TemplateCandidate("t", "tiling",
                                  {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16})
        edge = TemplateCandidate("t", "tiling",
                                 {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32})
        res = prune_by_smem([small, edge], _Caps(28))
        self.assertEqual(2, len(res.kept),
                          "1.0~1.2 之间: 剪枝必须与预检用同一条宽松边界")
        res2 = prune_by_smem([small, edge], _Caps(26))
        self.assertEqual(1, len(res2.kept), "明显放不下 ⇒ 摘; 小点留着 ⇒ 不饿死这一轮")
        self.assertIn("64x64x32", res2.as_note())

    def test_missing_limit_is_a_third_state_not_a_pass(self):
        res = prune_by_smem(self._points((256, 256, 128)), _NoSmemCaps())
        self.assertTrue(res.no_limit)
        self.assertEqual(1, len(res.kept), "不知道 ⇒ 不剪")
        self.assertIn("未生效", res.as_note())
        self.assertIsNone(smem_ok(4096, _NoSmemCaps()), "三态: None ≠ True")
        self.assertIsNone(smem_limit_bytes(_NoSmemCaps()))

    def test_pruning_never_starves_the_round(self):
        """全被摘光 ⇒ 一个都不摘 (剪枝是省冤枉钱, 不是让这一轮没得跑)"""
        pts = self._points((256, 256, 128), (512, 512, 128))
        res = prune_by_smem(pts, _Caps(1))
        self.assertEqual(2, len(res.kept), "全超上界也要留下待远端判定")
        self.assertTrue(res.kept_all_to_avoid_starvation)
        self.assertIn("全部保留", res.as_note())

    def test_unknown_axes_are_not_pruned(self):
        """认不出的参数轴 ⇒ 不剪 (宁可漏拦, 与预检同一条保守原则)"""
        pts = [TemplateCandidate("t", "tiling", {"WARP_SPECIALIZE": True})]
        res = prune_by_smem(pts, _Caps(1))
        self.assertEqual(1, len(res.kept))
        self.assertEqual([], res.dropped)


if __name__ == "__main__":
    unittest.main(verbosity=2)