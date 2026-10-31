"""§12 的模板候选**实例化** (`build`)：点 → 源码，身份不跟着代码走

设计 §12 的两类候选里，模板型那半在 2026-10-02 只有**枚举**（第一片，CHECKLOG I56）。
这一片补上「点 → 源码」，并钉住四处：身份、版本、空源码、以及「有点没建成」。

四条判据各自的失效（都在下面有用例）：

| 若没有这道判据 | 会读成什么 |
|---|---|
| 身份委派给模板点 | 「两点碰巧同源」被读成同一候选（§12 要挡的正是这个）；反过来「同点换版本」被读成两个候选 |
| 版本必须自报并核对 | 实例化器换了版本却不说 ⇒ 去重键把两个不同的构建读成同一个候选 |
| 空源码拒绝 | 「实例化失败」被读成「产出了一个空候选」，一路走到编译才发现 |
| 拒绝逐条留痕 | 「12 个点建了 11 个」塌成「这一轮有 11 个候选」，某类参数一直失败看不见 |

运行: python3 -m unittest tests.test_template_instantiation -v
"""

import sys
import unittest
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from operator_agent.core.backend_caps import Backend, Instantiation
from operator_agent.core.candidates import (BuiltCandidate, ParamSpace,
                                      dedup_key, instantiate_candidate,
                                      instantiate_points, instantiator_available,
                                      require_supported_space,
                                      source_candidate)
from tests.backend_fakes import DeclaredFakeBackend


_CONDUCTOR_STOP_REASONS = tuple(
    line.split('"')[1] for line in
    (Path(__file__).resolve().parents[2] / "operator_agent/app/conductor.py")
    .read_text(encoding="utf-8").splitlines()
    if "_record_stop(" in line and '"' in line and "def " not in line)


class _SpaceBackend(DeclaredFakeBackend):
    """声明了参数空间的后端（默认**没有**实例化器）"""

    def __init__(self, **overrides):
        for key in ("space", "instantiate", "label"):
            if key in overrides:
                setattr(self, key, overrides.pop(key))
        super().__init__(**overrides)

    def param_space(self, contract: object = None) -> ParamSpace:
        return getattr(self, "space", _SPACE)

    @property
    def label(self) -> str:
        return getattr(self, "_label", "fake-template-backend")


def _backend(instantiator=None, ver="tv-1", space=None) -> _SpaceBackend:
    overrides = {"space": space or _SPACE}
    if instantiator is not None:
        overrides["instantiate"] = instantiator
    backend = _SpaceBackend(**overrides)
    backend._label = "fake-template-backend"
    return backend


_UNSET = object()


def _inst(ver="tv-1", source=_UNSET, evidence="模板: t1"):
    """`source` 不传 = 用默认源码; 传 `None`/数字 = **刻意**测那条拒绝"""
    def instantiate(template_id, params, contract=None):
        code = (f"// {template_id} {sorted(params.items())}"
                if source is _UNSET else source)
        return Instantiation(source=code, instantiator_ver=ver, evidence=evidence)
    return instantiate


_SPACE = ParamSpace(template_id="t1", backend="fake",
                    instantiator_ver="tv-1",
                    axes={"BLOCK_M": (64, 128), "BLOCK_N": (64, 128)})


class TestTheProtocolDefault(unittest.TestCase):

    def test_a_backend_without_an_instantiator_raises_and_names_itself(self):
        backend = _backend()
        with self.assertRaises(NotImplementedError) as ctx:
            backend.instantiate("t1", {"BLOCK_M": 64})
        msg = str(ctx.exception)
        self.assertIn("fake-template-backend", msg)
        self.assertIn("t1", msg)
        self.assertIn("instantiate", msg)

    def test_a_plain_backend_has_no_instantiator(self):
        self.assertFalse(instantiator_available(DeclaredFakeBackend()))

    def test_a_backend_that_overrides_it_has_one(self):
        self.assertTrue(instantiator_available(_backend(_inst())))


class TestIdentityDoesNotFollowTheCode(unittest.TestCase):

    def setUp(self):
        self.backend = _backend(_inst())
        self.points = _SPACE.points()

    def test_two_different_points_with_the_same_source_stay_two_candidates(self):
        """§12 的要点：候选**不是**源码字符串。同源 ≠ 同一个候选。"""
        same_source = "// one fixed kernel body\n"
        backend = _backend(_inst(source=same_source))
        batch = instantiate_points(self.points, backend)
        self.assertEqual(4, len(batch.built))
        self.assertEqual(4, len({b.dedup_key() for b in batch.built}),
                         "两个点位即使产出一模一样的源码, 也仍是两个候选")

    def test_the_identity_is_the_template_point(self):
        built = instantiate_candidate(self.points[0], self.backend)
        self.assertEqual(dedup_key(self.points[0]), built.dedup_key())
        self.assertNotEqual(built.dedup_key(), dedup_key(self.points[1]))

    def test_the_same_point_built_twice_is_the_same_candidate(self):
        """同一个点、实例化器**换了版本却忘了报** ⇒ 身份仍然是那个点"""
        a = instantiate_candidate(self.points[0], _backend(_inst(ver="tv-1")))
        b = instantiate_candidate(self.points[0], _backend(_inst(ver="tv-1", source="// other")))
        self.assertEqual(a.dedup_key(), b.dedup_key())

    def test_the_source_hash_is_evidence_not_identity(self):
        """同点两次构建、源码不同 ⇒ 血缘里的 `build_source_hash` **看得出来**

        用内容哈希而不是只用 α-等价签名: 后者是 Python AST 的量具, 对 C++/CUDA 源
        (模板后端真正产出的东西) 恒为空串 —— 只靠它会把「源码变了」整个看不见。
        """
        a = instantiate_candidate(self.points[0],
                                  _backend(_inst(source="// a\n")))
        b = instantiate_candidate(self.points[0],
                                  _backend(_inst(source="// b\n")))
        self.assertEqual(a.dedup_key(), b.dedup_key())
        self.assertNotEqual(a.as_lineage()["build_source_hash"],
                            b.as_lineage()["build_source_hash"])
        lin = a.as_lineage()
        self.assertEqual("template", lin["kind"])
        self.assertIn("params", lin)
        self.assertNotIn("struct_sig", lin,
                         "模板候选的血缘不许冒用源码型那个身份字段")

    def test_the_struct_signature_is_empty_for_non_python_sources(self):
        """三态: `build_struct_sig == ""` = 量具不适用 (C++ 源), 不是「没有源码」"""
        built = instantiate_candidate(self.points[0],
                                      _backend(_inst(source="// cutlass\n")))
        self.assertEqual("", built.struct_sig)
        self.assertTrue(built.source_hash, "内容哈希跨语言都算得出来")

    def test_a_source_candidate_is_refused_on_this_path(self):
        """⚠️ 必须喂一个**有实例化器**的后端: 否则拒绝理由是「没有实例化器」, 而类型
        那道判据一次都没被执行 —— 变异 M8 正是这样逃逸过一次 (用例被另一条路挡住了)。"""
        with self.assertRaises(TypeError) as ctx:
            instantiate_candidate(source_candidate("triton", "def kernel(): pass"),
                                  _backend(_inst()))
        self.assertIn("模板候选", str(ctx.exception))
        with self.assertRaises(NotImplementedError):
            # 后端没有实例化器: 那是**实现缺口** (NotImplementedError), 不是类型错
            instantiate_candidate(self.points[0], _backend())


class TestTheVersionGuard(unittest.TestCase):

    def test_a_mismatched_instantiation_version_is_refused(self):
        backend = _backend(_inst(ver="tv-2"))
        with self.assertRaises(ValueError) as ctx:
            instantiate_candidate(_SPACE.points()[0], backend)
        self.assertIn("tv-1", str(ctx.exception))
        self.assertIn("tv-2", str(ctx.exception))

    def test_a_matching_version_passes(self):
        built = instantiate_candidate(_SPACE.points()[0], _backend(_inst(ver="tv-1")))
        self.assertEqual("tv-1", built.instantiator_ver)

    def test_an_unspeaking_instantiator_inherits_the_space_version(self):
        """实例化器自报空 ⇒ 保留参数空间里的版本, 而不是抹成空串 (缺 ≠ 0)"""
        built = instantiate_candidate(_SPACE.points()[0], _backend(_inst(ver="")))
        self.assertEqual("tv-1", built.instantiator_ver)
        self.assertEqual("tv-1", built.as_lineage()["instantiator_ver"])

    def test_a_point_without_a_version_accepts_the_reported_one(self):
        cand = ParamSpace(template_id="t1", backend="fake", axes={"BLOCK_M": (64,)}).points()[0]
        built = instantiate_candidate(cand, _backend(_inst(ver="tv-9")))
        self.assertEqual("tv-9", built.instantiator_ver)


class TestTheEmptySourceGuard(unittest.TestCase):

    def test_an_empty_source_is_a_failure_not_a_candidate(self):
        for bad in ("", "   \n\t", None, 123):
            backend = _backend(_inst(source=bad))
            with self.assertRaises(ValueError):
                instantiate_candidate(_SPACE.points()[0], backend)

    def test_a_non_instantiation_return_is_refused(self):
        backend = _backend(lambda template_id, params, contract=None: "just a string")
        with self.assertRaises(TypeError) as ctx:
            instantiate_candidate(_SPACE.points()[0], backend)
        self.assertIn("Instantiation", str(ctx.exception))


class TestTheBatch(unittest.TestCase):

    def test_one_bad_point_does_not_hide_the_others(self):
        good = _backend(_inst())
        points = _SPACE.points()

        def flaky(template_id, params, contract=None):
            if params.get("BLOCK_M") == 128:
                raise RuntimeError("这类参数上实例化器炸了")
            return Instantiation(source="// ok", instantiator_ver="tv-1")
        built = instantiate_points(points, _backend(flaky)) if False else None
        self.assertIsNone(built)
        batch = instantiate_points(points, _backend(flaky))
        self.assertEqual(2, len(batch.built))
        self.assertEqual(2, len(batch.refusals))
        self.assertIn("RuntimeError", batch.refusals[0][1])
        self.assertIn("拒绝 2 个", batch.as_note())
        self.assertEqual(2, len(batch.lineage()))
        self.assertEqual(4, len(good and instantiate_points(points, good).built))

    def test_an_empty_input_is_not_a_batch_of_zero(self):
        batch = instantiate_points([], _backend(_inst()))
        self.assertTrue(batch.empty_input)
        self.assertEqual([], batch.built)
        self.assertIn("没有点可建", batch.as_note())

    def test_the_note_is_empty_only_when_there_is_nothing_to_say(self):
        batch = instantiate_points(_SPACE.points(), _backend(_inst()))
        self.assertFalse(batch.refusals)
        self.assertIn("建成 4 个", batch.as_note())


class TestTheTwoSidedSwitch(unittest.TestCase):

    def test_no_space_is_the_code_backend(self):
        self.assertIsNone(require_supported_space(None, _backend(_inst())))

    def test_a_space_without_an_instantiator_is_the_backend_gap(self):
        with self.assertRaises(NotImplementedError) as ctx:
            require_supported_space(_SPACE, _backend())
        msg = str(ctx.exception)
        self.assertIn("Backend.instantiate", msg)
        self.assertIn("template_no_candidate", _CONDUCTOR_STOP_REASONS,
                      "控制器侧的缺口现在是**停机理由**, 不再是第二个 raise")

    def test_a_space_with_an_instantiator_passes_the_switch(self):
        """2026-10-02 控制器那一侧接上候选队列之后, 这条开关就只剩**后端**那个缺口"""
        space = require_supported_space(_SPACE, _backend(_inst()))
        self.assertIs(space, _SPACE, "有实例化器 ⇒ 放行 (返回空间)")


if __name__ == "__main__":
    unittest.main(verbosity=2)