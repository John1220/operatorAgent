"""A source hash is not the identity of a hidden launch/configuration search.

All receipts are synthetic harness fixtures. Source is parsed, never executed;
these tests require no Triton import, model, SSH connection or GPU.
"""
import json
import unittest

from operator_agent.resource_check import (
    MemCaps,
    SharedBuffer,
    SharedMemoryEvidence,
    assess_shared_memory,
    check_resource_feasibility,
    source_resource_binding,
)


HEAD = """import triton
import triton.language as tl
@triton.jit
def _k(X, BLOCK: tl.constexpr):
    pass
"""
LITERAL = HEAD + """
def kernel(X):
    _k[(1,)](X, BLOCK=64, num_stages=2)
    return X
"""


def receipt(source, kind, byte_count=128 * 1024):
    fields = dict(binding=source_resource_binding(source), kind=kind,
                  producer="s3-source-scope-test-harness")
    if kind == "native_compiler":
        fields["native_bytes"] = byte_count
    else:
        fields["buffers"] = (SharedBuffer("shared", (byte_count,), "int8"),)
    return SharedMemoryEvidence(**fields)


class TestOnlyOneEstablishedLaunchCanConsumeSourceReceipts(unittest.TestCase):
    def unknown_for_both_kinds(self, source):
        for kind in ("native_compiler", "trusted_shared_layout"):
            with self.subTest(kind=kind, source=source):
                evidence = receipt(source, kind)
                report = check_resource_feasibility(
                    source, MemCaps(64), resource_evidence=evidence)
                self.assertEqual("unknown", report.assessment)
                self.assertTrue(report.feasible)
                self.assertIsNone(report.estimated_smem_bytes)
                self.assertEqual([], report.buffers)
                self.assertTrue(report.notes)
                json.dumps(report.as_dict(), allow_nan=False)
                # The primitive still accepts an independently established
                # launch binding. The source entry must not extend its scope.
                primitive = assess_shared_memory(evidence, MemCaps(64),
                                                  binding=evidence.binding)
                self.assertEqual("over_limit", primitive.assessment)

    def test_literal_single_launch_still_accepts_explicit_evidence(self):
        for kind in ("native_compiler", "trusted_shared_layout"):
            with self.subTest(kind=kind):
                report = check_resource_feasibility(
                    LITERAL, MemCaps(64), resource_evidence=receipt(LITERAL, kind))
                self.assertEqual("over_limit", report.assessment)
                self.assertFalse(report.feasible)
                self.assertEqual(128 * 1024, report.estimated_smem_bytes)

    def test_loop_stages_do_not_share_one_binary_receipt(self):
        self.unknown_for_both_kinds(HEAD + """
def kernel(X):
    for stages in (1, 4):
        _k[(1,)](X, BLOCK=64, num_stages=stages)
    return X
""")

    def test_even_a_literal_launch_inside_a_loop_has_unknown_invocation_scope(self):
        self.unknown_for_both_kinds(HEAD + """
def kernel(X):
    while X:
        _k[(1,)](X, BLOCK=64, num_stages=2)
    return X
""")

    def test_comprehensions_cannot_hide_repeated_or_deferred_launches(self):
        expressions = (
            "[_k[(1,)](X, BLOCK=64, num_stages=stage) for stage in (1, 4)]",
            "{_k[(1,)](X, BLOCK=64, num_stages=2) for stage in (1, 4)}",
            "{stage: _k[(1,)](X, BLOCK=64, num_stages=2) for stage in (1, 4)}",
            "(_k[(1,)](X, BLOCK=64, num_stages=2) for stage in (1, 4))",
        )
        for expression in expressions:
            self.unknown_for_both_kinds(HEAD + "\ndef kernel(X):\n    return " + expression + "\n")

    def test_dynamic_launch_resource_keywords_are_unknown(self):
        for expression in (
            "BLOCK=choose_block(), num_stages=2",
            "BLOCK=BLOCK, num_stages=2",
            "BLOCK=64, num_stages=choose_stages()",
            "BLOCK=64, num_stages=(2 if X else 4)",
            "BLOCK=64, num_stages=2, num_warps=warps",
            "BLOCK=64, num_stages=2, enable_fp_fusion=choose_fusion()",
        ):
            self.unknown_for_both_kinds(HEAD + "\ndef kernel(X):\n    _k[(1,)](X, "
                                        + expression + ")\n    return X\n")

    def test_launch_keyword_expansion_is_not_evaluated(self):
        for expression in ("**meta", "**{'BLOCK': 64, 'num_stages': 2}"):
            self.unknown_for_both_kinds(HEAD + "\ndef kernel(X):\n    _k[(1,)](X, "
                                        + expression + ")\n    return X\n")

    def test_launch_positional_expansion_is_not_evaluated(self):
        self.unknown_for_both_kinds(HEAD + """
def kernel(X):
    _k[(1,)](*args, BLOCK=64, num_stages=2)
    return X
""")

    def test_positional_constexpr_parameter_must_be_literal(self):
        self.unknown_for_both_kinds(HEAD + """
def kernel(X):
    _k[(1,)](X, choose_block(), num_stages=2)
    return X
""")

    def test_literal_positional_constexpr_parameter_is_one_config(self):
        source = LITERAL.replace("X, BLOCK=64", "X, 64")
        report = check_resource_feasibility(
            source, MemCaps(64), resource_evidence=receipt(source, "native_compiler"))
        self.assertEqual("over_limit", report.assessment)

    def test_internal_jit_loop_does_not_make_the_host_launch_ambiguous(self):
        source = LITERAL.replace("    pass", "    for k in range(4):\n        pass")
        report = check_resource_feasibility(
            source, MemCaps(64), resource_evidence=receipt(source, "native_compiler"))
        self.assertEqual("over_limit", report.assessment)

    def test_no_launch_is_not_a_single_binary_identity(self):
        self.unknown_for_both_kinds(HEAD + "\ndef kernel(X):\n    return X\n")

    def test_another_host_helper_can_be_invoked_more_than_once(self):
        self.unknown_for_both_kinds(HEAD + """
def run(X):
    _k[(1,)](X, BLOCK=64, num_stages=2)
    return X
def kernel(X):
    run(X)
    return run(X)
""")

    def test_unresolved_constexpr_annotation_alias_does_not_hide_dynamic_config(self):
        source = LITERAL.replace("BLOCK: tl.constexpr", "TILE: ce")
        source = source.replace("BLOCK=64", "TILE=choose_tile()")
        self.unknown_for_both_kinds(source)


if __name__ == "__main__":
    unittest.main()
