"""静态守卫的**适用性**：把「查不了」与「查过没问题」分开（CHECKLOG I65）

设计 §19.3 要求反作弊静态守卫**必须保留**，而两道守卫
(`anticheat.static_audit` / `interface_contract.check_kernel_interface`) 都是
**Python 形状**的：AST + `def kernel`。对一份 C++/CUDA 模板源它们都判不了，而旧返回值
只有一个「有/没有问题」的通道 —— 空违规列表与 `None` 同时表示「查过了没问题」与
「根本没查成」。这是 fail-open，也是本仓基线纪律（**信任全部来自验证侧**）最不能接受的
一种：信任会凭空产生。

本文件钉住四件事：
1. 判据只有一处 (`source_lang.classify`)，三态不许被压成布尔；
2. 两道守卫消费同一个判据（`cuda_c` 是**真实但更弱**的文本覆盖，与「没有覆盖」分开）；
3. 适用性**随结果走**并进资格层：明确不适用 ⇒ 可判/可调/可报，但不得持久化、不得计进度；
   「未观测」⇒ 不改判（缺键 ≠ 判过且不合格）；
4. 生产路径真的把它传到了收口点（Verifier → oracle 的 `rule_inputs`，以及 Conductor
   从层级结果里取最保守的一档）。
"""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from operator_agent.anticheat import static_audit  # noqa: E402
from operator_agent.eligibility import derive_eligibility  # noqa: E402
from operator_agent.interface_contract import check_kernel_interface  # noqa: E402
from operator_agent.remote_executor import SSHExecutor  # noqa: E402
from operator_agent.source_lang import (AST_FAMILY, COVERAGE_AST, COVERAGE_NONE,  # noqa: E402
                                        COVERAGE_REGEX, DOES_NOT_APPLY, UNKNOWN,
                                        classify, flag_from_metrics)

CUDA = "__global__ void kern(const float* A, float* B) { B[0] = A[0]; }\n"
PY_SRC = "def kernel(A, B):\n    return A @ B\n"


class TestTheJudgementIsThreeState(unittest.TestCase):
    """判据：`applicable=None` 不许被读成任何一边"""

    def test_a_declared_python_family_language_applies(self):
        for name in AST_FAMILY:
            with self.subTest(dsl=name):
                v = classify(CUDA, name)      # 声明是权威: 不看源码
                self.assertIs(True, v.applicable)
                self.assertEqual(COVERAGE_AST, v.coverage)
                self.assertEqual("applies", v.state)

    def test_a_declared_language_with_only_textual_rules_is_weak_not_absent(self):
        """`cuda_c` 的源码是 C++, 但禁用清单的**正则规则就是为它写的** (且被用例钉着)

        这一格是「真实但更弱」的覆盖: 与「没有覆盖」混在一起, 要么把能查的说成不能查,
        要么把不能查的说成能查。
        """
        v = classify(CUDA, "cuda_c")
        self.assertIs(True, v.applicable)
        self.assertEqual(COVERAGE_REGEX, v.coverage)
        self.assertIn("文本模式", v.reason)

    def test_a_declared_language_with_no_rules_does_not_apply(self):
        v = classify(CUDA, "hip_c")
        self.assertIs(False, v.applicable)
        self.assertEqual(DOES_NOT_APPLY, v.state)
        self.assertIn("hip_c", v.reason)
        self.assertIn("判不了", v.reason)

    def test_undeclared_and_not_python_is_unknown_not_false(self):
        """「没声明」与「声明了别的」是两件事 —— 两者的理由不同, 结论也不同"""
        v = classify(CUDA, "")
        self.assertIsNone(v.applicable, "「无法确立」不许被读成 False")
        self.assertEqual(UNKNOWN, v.state)
        self.assertEqual(COVERAGE_NONE, v.coverage)

    def test_undeclared_but_python_is_inferred_and_says_so(self):
        """没声明就跑**一条**推断 (ast.parse), 且推断这件事本身要留在结论里"""
        v = classify(PY_SRC, "")
        self.assertIs(True, v.applicable)
        self.assertEqual("语言没声明, 但源码就是 Python (解析得过) ⇒ AST 审计覆盖", v.reason)

    def test_the_three_states_are_distinguishable_in_the_record(self):
        seen = {json.dumps(classify(src, dsl).as_dict(), sort_keys=True)
                for src, dsl in ((PY_SRC, "python"), (CUDA, "hip_c"), (CUDA, ""))}
        self.assertEqual(3, len(seen), "三态在产物里必须是三份不同的记录")


class TestBothGuardsConsumeTheOneJudgement(unittest.TestCase):
    """两个守卫都只消费 `classify` 的结论, 不许各自再嗅一次"""

    def test_the_audit_says_it_could_not_judge(self):
        verdict = static_audit(CUDA, "hip_c")
        self.assertEqual([], list(verdict), "不适用的源码不会有违规列表")
        self.assertIs(False, verdict.applicable)
        self.assertIs(False, bool(verdict), "空列表仍然是 falsy (旧读法不变)")

    def test_the_audit_still_catches_what_it_covers(self):
        """反向保护: 收紧适用性时最容易顺手打死的那一条"""
        cheat = "import torch\ndef kernel(A, B):\n    return torch.matmul(A, B)\n"
        self.assertTrue(static_audit(cheat, "triton").violations)
        self.assertIs(True, static_audit(cheat, "triton").applicable)

    def test_a_language_with_no_rules_is_not_scanned_by_another_languages_rules(self):
        """早期返回**有**观察面: C++ 注释里出现 `torch.matmul` 不该被当违规

        没有这一条, 「不适用就早退」与「拿别的语言的规则硬扫」在用例上不可区分
        (那份 CUDA 样本里本来就没有禁用符号 —— 变异能活下来)。
        """
        cxx = "// 这里不能用 torch.matmul(\n__global__ void k() {}\n"
        self.assertEqual([], static_audit(cxx, "hip_c"))
        self.assertIs(False, static_audit(cxx, "hip_c").applicable)
        # 反向: 同一个声明下真的被覆盖的语言照旧拦
        self.assertTrue(static_audit("import torch\ndef kernel(A):\n"
                                     "    return torch.matmul(A, A)\n", "triton"))

    def test_the_audit_result_is_still_a_list(self):
        """跨进程与旧读法: 远端 `run_kernel.py` 会把它塞进 JSON, 也会切片取前三条"""
        verdict = static_audit("import torch\ndef kernel(A):\n    return torch.softmax(A, 0)\n",
                               "triton")
        self.assertIsInstance(verdict, list)
        self.assertEqual(len(verdict), len(list(verdict)))
        self.assertEqual(list(verdict)[:3], verdict[:3])
        self.assertEqual(list(verdict), json.loads(json.dumps(list(verdict))))

    def test_the_signature_check_does_not_claim_ok_when_it_cannot_tell(self):
        """`problem=None` 原来既是「对得上」也是「看不出来」"""
        ok = check_kernel_interface(PY_SRC, ["A", "B"], language="python")
        self.assertIsNone(ok.problem)
        self.assertIs(True, ok.applicable)

        blind = check_kernel_interface(CUDA, ["A", "B"], language="cuda_c")
        self.assertIsNone(blind.problem, "判不了 ≠ 判出问题")
        self.assertIs(False, blind.applicable, "但它也 ≠ 签名对得上")

    def test_a_python_source_without_a_kernel_def_is_not_called_a_pass(self):
        src = "def helper(A):\n    return A\n"
        v = check_kernel_interface(src, ["A"], language="python")
        self.assertIsNone(v.problem)
        self.assertIsNone(v.applicable, "找不到入口 ⇒ 无法确立, 不是适用")


class TestTheFlagReachesTheEligibilityLayer(unittest.TestCase):
    """明确不适用 ⇒ 降级 (可判/可调/可报, 不持久化/不计进度); 未观测 ⇒ 不改判"""

    def test_not_applicable_degrades_exactly_the_two_permanent_bits(self):
        e = derive_eligibility(static_guards=False)
        self.assertTrue(e.can_verdict, "数字本身照旧可判")
        self.assertTrue(e.can_tune)
        self.assertTrue(e.can_report)
        self.assertFalse(e.can_persist, "「查不了」不许进长期记忆")
        self.assertFalse(e.can_count_progress, "也不许当收敛的证据")

    def test_unobserved_changes_nothing(self):
        self.assertEqual(derive_eligibility().as_dict(),
                         derive_eligibility(static_guards=None).as_dict())

    def test_applicable_changes_nothing(self):
        self.assertEqual(derive_eligibility().as_dict(),
                         derive_eligibility(static_guards=True).as_dict())

    def test_a_worse_rule_still_wins(self):
        """规则优先级: 规则 3/4/5 在先, 适用性不许把「没有指称对象」救回来"""
        self.assertFalse(derive_eligibility(static_guards=False,
                                            lib_fallback=True).can_verdict)

    def test_the_flag_is_taken_from_the_published_metrics(self):
        """读的是产物里记的那一份, 取最保守的一档"""
        self.assertIs(False, flag_from_metrics({"static_guards": True},
                                               {"static_guards": False}))
        self.assertIs(True, flag_from_metrics({"static_guards": True}, {}))
        self.assertIsNone(flag_from_metrics({}, None),
                          "一条都没记 ⇒ 未观测 (不是 False)")
        self.assertIsNone(flag_from_metrics({"static_guards": None}))


class TestTheReceiptPointReadsIt(unittest.TestCase):
    """收口点 (oracle) 真的读了它, 并且三态在 provenance 里分得开"""

    def _oracle(self, **raw_extra):
        from operator_agent.contract import freeze_from_task
        from operator_agent.oracle import evaluate
        from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec,
                                            TargetSpec, TaskSchema, TaskType)
        task = TaskSchema(task_id="guard-app", task_type=TaskType.CODE_TRANSLATION,
                          source=SourceSpec(kernel_type="matmul"),
                          target=TargetSpec(hardware="mock", dsl="python"),
                          constraints=ConstraintSpec(precision=Precision.FP32,
                                                     min_speedup=1.0, max_error=1e-3))
        raw = {"value": 1.5, "validity": "valid", "lib_fallback": False,
               "real_device": None, "l3_pass_is_refused": False, "cv_exceeded": False,
               "launch_dominance_exceeded": False, "partition_drift": False}
        raw.update(raw_extra)
        return evaluate(freeze_from_task(task), lambda: raw, fresh=True,
                        metric="speedup_vs_baseline", unit="x",
                        require_real_device=False)

    def test_not_applicable_is_in_the_rule_inputs_and_the_reasons(self):
        obs = self._oracle(static_guards=False)
        self.assertIs(False, obs.provenance["rule_inputs"]["static_guards"])
        self.assertFalse(obs.eligibility.can_persist)
        self.assertTrue(obs.eligibility.can_verdict)
        self.assertTrue(any("不适用" in r for r in obs.reasons), obs.reasons)

    def test_unobserved_says_so_without_changing_the_verdict(self):
        obs = self._oracle()
        self.assertIsNone(obs.provenance["rule_inputs"]["static_guards"])
        self.assertTrue(obs.eligibility.can_persist)
        self.assertTrue(any("未观测" in r for r in obs.reasons), obs.reasons)

    def test_a_stringified_false_is_not_read_as_false(self):
        """布尔化的老坑: `bool("False")` 是 True, `bool("")` 是 False —— 两个都不许"""
        for bogus in ("False", "", "false", 0, 1):
            with self.subTest(value=bogus):
                obs = self._oracle(static_guards=bogus)
                self.assertIsNone(obs.provenance["rule_inputs"]["static_guards"],
                                  f"{bogus!r} 不是布尔 ⇒ 只能记「未观测」")


class TestTheVerifierPublishesItAndTheLoopPassesItOn(unittest.TestCase):
    """生产路径: L1 → (Conductor) → L3 的资格判定, 以及部署链"""

    def test_the_verifier_publishes_the_judgement(self):
        from operator_agent.mock_gpu import MockGPU
        from operator_agent.schemas import (ConstraintSpec, Precision, SourceSpec,
                                            TargetSpec, TaskSchema, TaskType)
        from operator_agent.verifier import Verifier
        task = TaskSchema(task_id="pub", task_type=TaskType.CODE_TRANSLATION,
                          source=SourceSpec(kernel_type="matmul"),
                          target=TargetSpec(hardware="mock", dsl="python"),
                          constraints=ConstraintSpec(precision=Precision.FP32,
                                                     min_speedup=1.0, max_error=1e-3))
        import numpy as np
        inputs = {"A": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                  "B": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)}
        v = Verifier(MockGPU())
        results = v.verify(PY_SRC, task, inputs)
        l1 = results[0]
        self.assertIs(True, l1.metrics.get("static_guards"),
                      "Python 源码的两道守卫都适用 ⇒ True (不是缺键)")
        self.assertIn("static_guards_lang", l1.metrics)
        # 明确不适用的那一档: 走到 L3 的资格判定里必须是「不持久化」
        l3 = v.verify_l3_only(PY_SRC, inputs, 10.0, task, static_guards=False)[0]
        self.assertIs(False, l3.metrics["observation"]["provenance"]
                      ["rule_inputs"]["static_guards"])
        self.assertFalse(l3.metrics["eligibility"]["can_persist"])
        self.assertTrue(l3.metrics["eligibility"]["can_verdict"])

    def test_the_deploy_chain_carries_the_dependency_before_its_reader(self):
        """远端 `import anticheat` 会跟着 import source_lang ⇒ 它必须排在前面"""
        names = [Path(remote).name for _, remote in SSHExecutor(host="h", user="u")._remote_files()]
        self.assertIn("source_lang.py", names)
        self.assertLess(names.index("source_lang.py"), names.index("anticheat.py"),
                        "依赖排在读者之后 ⇒ 半途失败会得到「新 anticheat + 缺依赖」")


if __name__ == "__main__":
    unittest.main(verbosity=2)