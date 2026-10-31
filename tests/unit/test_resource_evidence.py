"""S3: source hints are not physical shared-memory allocation evidence.

All resource receipts in this module are synthetic harness contracts. The fixture
reconstructs a retained historical FP16/FP32 pattern and is never executed here.
No test imports Triton, calls a model, uses SSH, or touches a GPU.
"""
from __future__ import annotations

import contextlib
from dataclasses import FrozenInstanceError
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from operator_agent.decide.resource_check import (
    MemCaps,
    SharedBuffer,
    SharedMemoryEvidence,
    assess_shared_memory,
    check_resource_feasibility,
    source_resource_binding,
)
from operator_agent.decide.verifier import Verifier
from tests.backend_fakes import DeclaredFakeBackend


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "s3_fp16_storage_fp32_accumulator.txt"
FP16_FP32_ACC = FIXTURE.read_text(encoding="utf-8")
PRODUCER = "s3-test-harness:shared-layout-v1"


def layout(*buffers, code=FP16_FP32_ACC, **changes):
    fields = dict(binding=source_resource_binding(code),
                  kind="trusted_shared_layout", producer=PRODUCER,
                  buffers=tuple(buffers))
    fields.update(changes)
    return SharedMemoryEvidence(**fields)


def native(byte_count, code=FP16_FP32_ACC, **changes):
    fields = dict(binding=source_resource_binding(code), kind="native_compiler",
                  producer="s3-test-harness:native-compiler-v1",
                  native_bytes=byte_count)
    fields.update(changes)
    return SharedMemoryEvidence(**fields)


def assess(evidence, cap_kb=228, code=FP16_FP32_ACC):
    return assess_shared_memory(evidence, MemCaps(cap_kb),
                                binding=source_resource_binding(code))


class ResourceAssertions(unittest.TestCase):
    def assert_unknown(self, report):
        self.assertEqual("unknown", report.assessment)
        self.assertTrue(report.feasible, "unknown allows native checking; it is not proof of fit")
        self.assertIsNone(report.estimated_smem_bytes,
                          "missing or invalid evidence must not manufacture physical bytes")
        self.assertTrue(report.notes or report.reason, "unknown needs an explanation")
        payload = report.as_dict()
        self.assertEqual("unknown", payload["assessment"])
        self.assertIsNone(payload["estimated_smem_bytes"])
        json.dumps(payload, allow_nan=False)


class TestHistoricalStorageAccumulatorRegression(ResourceAssertions):
    def test_fixture_labels_its_reconstruction_and_retained_source(self):
        self.assertIn("not the original 2026-10-19 kernel", FP16_FP32_ACC)
        self.assertIn("history-matmul4096-autonomous.tar.gz", FP16_FP32_ACC)
        self.assertIn("dtype=tl.float32", FP16_FP32_ACC)
        self.assertIn("BLOCK_M=256, BLOCK_N=256, BLOCK_K=64", FP16_FP32_ACC)
        compile(FP16_FP32_ACC, str(FIXTURE), "exec")  # Syntax only; no imports or execution.

    def test_fp32_accumulator_does_not_invent_384_kib_of_shared_storage(self):
        report = check_resource_feasibility(FP16_FP32_ACC, MemCaps(228), "matmul")
        self.assert_unknown(report)
        self.assertNotEqual("within_limit", report.assessment)

    def test_trusted_half_staging_layout_keeps_storage_separate_from_accumulator(self):
        # This is a synthetic harness layout, not a claim about the original
        # Triton allocation. No FP32 accumulator shared buffer is declared.
        evidence = layout(SharedBuffer("A_staging", (256, 64), "float16"),
                          SharedBuffer("B_staging", (64, 256), "float16"),
                          SharedBuffer("C_staging", (256, 256), "float16"))
        report = check_resource_feasibility(
            FP16_FP32_ACC, MemCaps(228), "matmul", resource_evidence=evidence)
        self.assertEqual("within_limit", report.assessment)
        self.assertTrue(report.feasible)
        self.assertEqual(192 * 1024, report.estimated_smem_bytes)
        self.assertEqual("trusted_shared_layout", report.evidence_kind)
        self.assertEqual(PRODUCER, report.evidence_producer)
        self.assertEqual(source_resource_binding(FP16_FP32_ACC), report.evidence_binding)

    def test_comments_and_docstrings_cannot_establish_a_storage_dtype(self):
        for prose in ("# float32 BLOCK_M=8192 BLOCK_N=8192 num_stages=99\n",
                      '"""float32 BLOCK_M=8192 shared memory=999999999"""\n'):
            with self.subTest(prose=prose):
                report = check_resource_feasibility(prose + FP16_FP32_ACC,
                                                    MemCaps(1), "matmul")
                self.assert_unknown(report)

    def test_tl_zeros_dot_and_launch_stages_remain_logical_source_hints(self):
        for changed in (FP16_FP32_ACC.replace("tl.float32", "tl.float64"),
                        FP16_FP32_ACC.replace("num_stages=3", "num_stages=99"),
                        FP16_FP32_ACC.replace("BLOCK_M=256", "BLOCK_M=8192")):
            with self.subTest(source=changed[-180:]):
                self.assert_unknown(check_resource_feasibility(changed, MemCaps(1), "matmul"))


class TestSourceConfigurationsAreKeptIntact(ResourceAssertions):
    AUTOTUNE = """import triton
@triton.autotune(configs=[
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=2),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=5),
], key=['M', 'N', 'K'])
def kernel(A, B):
    return A
"""

    def test_autotune_records_each_group_without_a_cross_config_maximum(self):
        report = check_resource_feasibility(self.AUTOTUNE, MemCaps(1), "matmul")
        self.assert_unknown(report)
        groups = [item for item in report.configurations
                  if item.get("source") == "triton.Config"]
        self.assertEqual(2, len(groups))
        self.assertEqual([{"BLOCK_M": 256, "BLOCK_N": 32, "BLOCK_K": 16},
                          {"BLOCK_M": 32, "BLOCK_N": 256, "BLOCK_K": 64}],
                         [item["params"] for item in groups])
        self.assertEqual([2, 5], [item["num_stages"] for item in groups])
        self.assertTrue(all(item["assessment"] == "unknown" for item in groups))
        self.assertNotIn({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64},
                         [item["params"] for item in groups])

    def test_dynamic_autotune_values_are_unknown_and_are_never_evaluated(self):
        code = self.AUTOTUNE.replace("'BLOCK_M': 256", "'BLOCK_M': choose_block()")
        code = code.replace("num_stages=2", "num_stages=choose_stages()")
        report = check_resource_feasibility(code, MemCaps(1), "matmul")
        self.assert_unknown(report)
        groups = [item for item in report.configurations
                  if item.get("source") == "triton.Config"]
        self.assertEqual(2, len(groups))
        self.assertIsNone(groups[0]["num_stages"])
        self.assertTrue(groups[0]["notes"])
        self.assertEqual({"BLOCK_M": 32, "BLOCK_N": 256, "BLOCK_K": 64},
                         groups[1]["params"])

    def test_dynamic_launch_dimensions_do_not_become_a_zero_byte_allocation(self):
        code = FP16_FP32_ACC.replace("BLOCK_M=256", "BLOCK_M=choose_block(A)")
        self.assert_unknown(check_resource_feasibility(code, MemCaps(1), "matmul"))

    def test_one_source_hash_receipt_cannot_cover_an_autotune_configuration_set(self):
        evidence_types = (
            native(1024, code=self.AUTOTUNE),
            layout(SharedBuffer("staging", (512,), "float16"), code=self.AUTOTUNE),
        )
        for evidence in evidence_types:
            with self.subTest(kind=evidence.kind):
                # Primitive evidence validation has a single binding scope.
                self.assertEqual("within_limit", assess(evidence, code=self.AUTOTUNE).assessment)
                # Source-side autotune has multiple compile configurations, so a
                # source hash alone cannot establish a resource receipt for all.
                report = check_resource_feasibility(
                    self.AUTOTUNE, MemCaps(228), "matmul", resource_evidence=evidence)
                self.assert_unknown(report)

    def test_autotune_import_aliases_do_not_avoid_configuration_scope_checks(self):
        variants = (
            self.AUTOTUNE.replace("import triton", "import triton as tr").replace("triton.", "tr."),
            self.AUTOTUNE.replace("import triton", "import triton\nfrom triton import autotune as tune")
            .replace("@triton.autotune", "@tune"),
        )
        for code in variants:
            with self.subTest(decorator=code.splitlines()[1]):
                report = check_resource_feasibility(
                    code, MemCaps(228), "matmul", resource_evidence=native(1024, code=code))
                self.assert_unknown(report)

    def test_one_source_hash_receipt_cannot_pick_a_kernel_from_multiple_jit_helpers(self):
        code = FP16_FP32_ACC + "\n@triton.jit\ndef _other_kernel(X):\n    return X\n"
        report = check_resource_feasibility(
            code, MemCaps(228), "matmul", resource_evidence=native(1024, code=code))
        self.assert_unknown(report)

    def test_one_source_hash_receipt_cannot_choose_between_multiple_launches(self):
        second_launch = ("    _matmul_kernel[grid](A, B, C, M, N, K,\n"
                         "                         BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,\n"
                         "                         num_stages=2)\n")
        code = FP16_FP32_ACC.replace("    return C\n", second_launch + "    return C\n")
        report = check_resource_feasibility(
            code, MemCaps(228), "matmul", resource_evidence=native(1024, code=code))
        self.assert_unknown(report)


class TestExplicitBufferEvidence(ResourceAssertions):
    def test_buffers_keep_independent_dtype_and_pipeline_copy_counts(self):
        a = SharedBuffer("A", (64, 32), "float16", copies=3)
        b = SharedBuffer("B", (32, 64), "float16", copies=3)
        scratch = SharedBuffer("reduction_scratch", (64, 64), "float32", copies=1)
        report = assess(layout(a, b, scratch), cap_kb=64)
        expected = (64 * 32 * 2 * 3) + (32 * 64 * 2 * 3) + (64 * 64 * 4)
        self.assertEqual(40 * 1024, expected)
        self.assertEqual(expected, report.estimated_smem_bytes)
        self.assertEqual("within_limit", report.assessment)
        buffers = report.as_dict()["buffers"]
        self.assertEqual([3, 3, 1], [item["copies"] for item in buffers])
        self.assertEqual(["float16", "float16", "float32"],
                         [item["dtype"] for item in buffers])
        self.assertEqual([12288, 12288, 16384], [item["bytes"] for item in buffers])

    def test_changing_one_buffers_copies_does_not_multiply_every_buffer(self):
        for copies in (1, 2, 5):
            with self.subTest(copies=copies):
                evidence = layout(SharedBuffer("input", (64, 32), "float16", copies=copies),
                                  SharedBuffer("scratch", (64, 64), "float32"))
                report = assess(evidence)
                self.assertEqual(4096 * copies + 16384, report.estimated_smem_bytes)

    def test_declared_fp32_shared_storage_is_not_narrowed_by_fp16_source(self):
        evidence = layout(SharedBuffer("fp32_shared_intermediate", (256, 256), "float32", copies=2))
        report = check_resource_feasibility(
            FP16_FP32_ACC, MemCaps(228), "matmul", resource_evidence=evidence)
        self.assertEqual(512 * 1024, report.estimated_smem_bytes)
        self.assertEqual("over_limit", report.assessment)
        self.assertFalse(report.feasible)
        self.assertIn("资源预检", report.as_message())

    def test_slack_boundary_remains_byte_accurate(self):
        evidence = layout(SharedBuffer("bytes", (1536,), "int8"))
        self.assertEqual("within_limit", assess(evidence, 1300 / 1024).assessment)
        bad = assess(evidence, 1200 / 1024)
        self.assertEqual("over_limit", bad.assessment)
        self.assertFalse(bad.feasible)

    def test_unknown_buffer_prevents_a_partial_sum_from_becoming_physical_total(self):
        evidence = layout(SharedBuffer("known", (64, 64), "float16"),
                          SharedBuffer("dynamic", (64, None), "float32"))
        self.assert_unknown(assess(evidence))

    def test_buffer_and_evidence_contracts_are_immutable(self):
        buffer = SharedBuffer("A", (64, 32), "float16")
        evidence = layout(buffer)
        with self.assertRaises(FrozenInstanceError):
            buffer.copies = 3
        with self.assertRaises(FrozenInstanceError):
            evidence.binding = "another-source"


class TestNativeResourceReceipts(ResourceAssertions):
    def test_native_bytes_are_not_recomputed_from_source_tile_or_stages(self):
        report = check_resource_feasibility(
            FP16_FP32_ACC, MemCaps(64), "matmul", resource_evidence=native(32 * 1024))
        self.assertEqual("within_limit", report.assessment)
        self.assertEqual(32 * 1024, report.estimated_smem_bytes)
        self.assertEqual("native_compiler", report.evidence_kind)
        self.assertEqual("native_compiler", report.as_dict()["evidence_kind"])
        self.assertEqual([], report.as_dict()["buffers"])
        self.assertIn("native-compiler", report.evidence_producer)

    def test_native_zero_bytes_is_known_instead_of_missing(self):
        report = assess(native(0))
        self.assertEqual("within_limit", report.assessment)
        self.assertEqual(0, report.estimated_smem_bytes)

    def test_native_oversized_allocation_is_still_refused(self):
        report = assess(native(384 * 1024))
        self.assertEqual("over_limit", report.assessment)
        self.assertEqual(384 * 1024, report.estimated_smem_bytes)
        self.assertFalse(report.feasible)

    def test_non_integral_native_byte_counts_do_not_become_measurements(self):
        for value in (None, -1, True, False, 1024.0, "1024", float("nan"),
                      float("inf"), -float("inf"), [], {}):
            with self.subTest(native_bytes=value):
                self.assert_unknown(assess(native(value)))

    def test_a_receipt_cannot_mix_native_bytes_with_a_declared_layout(self):
        buffer = SharedBuffer("staging", (64, 64), "float16")
        for evidence in (native(1024, buffers=(buffer,)), layout(buffer, native_bytes=1024)):
            with self.subTest(kind=evidence.kind):
                self.assert_unknown(assess(evidence))


class TestInvalidEvidenceStaysUnknown(ResourceAssertions):
    def test_binding_is_the_sha256_of_the_exact_source(self):
        self.assertEqual(hashlib.sha256(FP16_FP32_ACC.encode("utf-8")).hexdigest(),
                         source_resource_binding(FP16_FP32_ACC))
        self.assertNotEqual(source_resource_binding(FP16_FP32_ACC),
                            source_resource_binding(FP16_FP32_ACC + "\n"))

    def test_a_receipt_for_another_source_cannot_reject_this_source(self):
        evidence = native(2 ** 30, code="def kernel(A, B): return A\n")
        self.assert_unknown(assess(evidence))

    def test_source_changes_including_prose_invalidate_the_receipt_binding(self):
        evidence = layout(SharedBuffer("A", (64, 32), "float16"))
        report = check_resource_feasibility(
            "# source changed\n" + FP16_FP32_ACC, MemCaps(228), "matmul",
            resource_evidence=evidence)
        self.assert_unknown(report)

    def test_an_unsupported_kind_cannot_turn_source_prose_into_a_contract(self):
        for kind in ("", "source_guess", "llm_claim", None, True, [], {}):
            with self.subTest(kind=kind):
                self.assert_unknown(assess(native(2 ** 30, kind=kind)))

    def test_missing_or_malformed_producer_is_not_attributable_evidence(self):
        for producer in ("", " ", None, True, 123, [], {}):
            with self.subTest(producer=producer):
                self.assert_unknown(assess(native(2 ** 30, producer=producer)))

    def test_missing_or_malformed_bindings_cannot_match_by_coercion(self):
        for binding in ("", None, True, 123, [], {}):
            with self.subTest(binding=binding):
                self.assert_unknown(assess(native(2 ** 30, binding=binding)))

    def test_arbitrary_objects_and_mappings_are_not_harness_contracts(self):
        for evidence in (None, {}, {"kind": "native_compiler", "native_bytes": 2 ** 30},
                         "native_compiler", True, [], object()):
            with self.subTest(evidence_type=type(evidence).__name__):
                self.assert_unknown(assess(evidence))

    def test_unknown_dtype_is_not_assumed_to_be_task_or_accumulator_dtype(self):
        for dtype in ("", "mystery_dtype", None, True, 2, [], {}):
            with self.subTest(dtype=dtype):
                self.assert_unknown(assess(layout(SharedBuffer("A", (64, 32), dtype))))

    def test_invalid_dimensions_are_not_truncated_or_coerced(self):
        for dim in (None, 0, -1, True, False, 64.0, "64", float("nan"),
                    float("inf"), -float("inf"), [], {}):
            with self.subTest(dimension=dim):
                self.assert_unknown(assess(layout(SharedBuffer("A", (dim, 32), "float16"))))

    def test_shapes_must_be_explicit_nonempty_tuples(self):
        for shape in ((), None, [64, 32], "64,32", 64, True):
            with self.subTest(shape=shape):
                self.assert_unknown(assess(layout(SharedBuffer("A", shape, "float16"))))

    def test_invalid_copy_counts_are_not_silently_replaced_with_one(self):
        for copies in (None, 0, -1, True, False, 3.0, "3", float("nan"),
                       float("inf"), -float("inf"), [], {}):
            with self.subTest(copies=copies):
                self.assert_unknown(assess(layout(SharedBuffer("A", (64, 32), "float16", copies))))

    def test_buffers_need_names_and_the_declared_buffer_type(self):
        for item in (None, True, {}, {"name": "A", "shape": (64, 32), "dtype": "float16"},
                     SharedBuffer("", (64, 32), "float16"),
                     SharedBuffer(None, (64, 32), "float16")):
            with self.subTest(buffer_type=type(item).__name__):
                self.assert_unknown(assess(layout(item)))

    def test_no_layout_buffers_is_unknown_instead_of_a_zero_byte_proof(self):
        self.assert_unknown(assess(layout()))

    def test_duplicate_buffer_names_cannot_describe_distinct_allocations(self):
        evidence = layout(SharedBuffer("staging", (64, 32), "float16"),
                          SharedBuffer("staging", (64, 64), "float32"))
        self.assert_unknown(assess(evidence))

    def test_oversized_physical_integers_do_not_overflow_or_escape_json_bounds(self):
        enormous = 1 << 20000
        evidence_cases = {
            "native bytes": native(enormous),
            "shape dimension": layout(SharedBuffer("A", (enormous,), "float16")),
            "copies": layout(SharedBuffer("A", (64, 32), "float16", enormous)),
            "buffer product": layout(SharedBuffer("A", (2 ** 32, 2 ** 32), "float16")),
            "buffer sum": layout(SharedBuffer("A", (2 ** 62,), "float16"),
                                 SharedBuffer("B", (2 ** 62,), "float16")),
        }
        for label, evidence in evidence_cases.items():
            with self.subTest(field=label):
                self.assert_unknown(assess(evidence))

    def test_oversized_hardware_limit_is_unknown_without_losing_valid_native_bytes(self):
        for limit in (1 << 20000, float(2 ** 64)):
            with self.subTest(limit_type=type(limit).__name__):
                report = assess_shared_memory(
                    native(1024), SimpleNamespace(shared_mem_per_sm_kb=limit),
                    binding=source_resource_binding(FP16_FP32_ACC))
                self.assertEqual("unknown", report.assessment)
                self.assertTrue(report.feasible)
                self.assertEqual(1024, report.estimated_smem_bytes)
                self.assertIsNone(report.limit_bytes)
                json.dumps(report.as_dict(), allow_nan=False)

    def test_invalid_caps_keep_known_bytes_but_cannot_establish_fit(self):
        evidence = native(32 * 1024)
        for limit in (None, 0, -1, True, False, "228", float("nan"),
                      float("inf"), -float("inf"), [], {}):
            with self.subTest(limit=limit):
                report = assess_shared_memory(
                    evidence, SimpleNamespace(shared_mem_per_sm_kb=limit),
                    binding=source_resource_binding(FP16_FP32_ACC))
                self.assertEqual("unknown", report.assessment)
                self.assertTrue(report.feasible)
                self.assertEqual(32 * 1024, report.estimated_smem_bytes,
                                 "known native bytes survive an unknown hardware threshold")
                self.assertIsNone(report.limit_bytes)
                self.assertTrue(report.notes or report.reason)
                json.dumps(report.as_dict(), allow_nan=False)


class TestUnknownResourcesStillReachNativeCompilation(ResourceAssertions):
    def test_unknown_hardware_limit_is_published_after_backend_compile(self):
        calls = []

        def compile_successfully(code):
            calls.append(code)
            return True

        verifier = Verifier(DeclaredFakeBackend(compile=compile_successfully))
        verifier._l1_caps = None
        with contextlib.redirect_stdout(io.StringIO()):
            result = verifier._verify_compile(FP16_FP32_ACC, dsl="triton",
                                              input_key_sets=[{"A", "B"}])
        self.assertEqual([FP16_FP32_ACC], calls)
        self.assertTrue(result.passed)
        report = result.metrics["resource_precheck"]
        self.assertEqual("unknown", report["assessment"])
        self.assertIsNone(report["estimated_smem_bytes"])
        self.assertIsNone(report["limit_bytes"])

    def test_unknown_resource_hints_cannot_hide_a_native_compile_failure(self):
        calls = []

        def reject_at_native_compile(code):
            calls.append(code)
            raise RuntimeError("synthetic native compiler: invalid shared layout")

        verifier = Verifier(DeclaredFakeBackend(compile=reject_at_native_compile))
        verifier._l1_caps = MemCaps(1)
        with contextlib.redirect_stdout(io.StringIO()):
            result = verifier._verify_compile(FP16_FP32_ACC, dsl="triton",
                                              input_key_sets=[{"A", "B"}])
        self.assertEqual([FP16_FP32_ACC], calls,
                         "resource unknown must reach the backend, even for a large source tile")
        self.assertFalse(result.passed, "continuing past resource hints is not an L1 pass")
        self.assertNotEqual("resource_infeasible", result.metrics.get("compile_status"))
        self.assertEqual("unknown", result.metrics["resource_precheck"]["assessment"])
        self.assertIsNone(result.metrics["resource_precheck"]["estimated_smem_bytes"])
        self.assertIn("synthetic native compiler", result.error_log)


if __name__ == "__main__":
    unittest.main()
