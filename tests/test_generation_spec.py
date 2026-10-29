"""Offline guards for the guidance actually rendered into every generation path.

No model or GPU execution: these checks protect shared rules and dynamic caller
inputs, while the existing numerical suites remain the acceptance authority.
"""
import hashlib

import numpy as np
import pytest

from operator_agent import generation_spec as spec
from operator_agent import single_candidate_coder as bounded
from operator_agent.interface_contract import broadcast_calling_convention
from operator_agent.prompt_manager import PromptManager


FAMILIES = ("designer_", "planner_", "coder_", "fixer_", "search_")
TEMPLATES = sorted(name for name in PromptManager().list_prompts()
                   if name.startswith(FAMILIES))


def test_identity_versions_the_exact_shared_text_and_is_not_shared_mutable_state():
    expected = {"version": spec.GENERATION_SPEC_VERSION,
                "sha256": hashlib.sha256(spec.GENERATION_SPEC.encode("utf-8")).hexdigest()}
    assert spec.generation_spec_identity() == expected
    identity = spec.generation_spec_identity()
    identity["version"] = "changed-by-caller"
    assert spec.generation_spec_identity() == expected


def test_identity_changes_when_the_shared_rules_change(monkeypatch):
    before = spec.generation_spec_identity()
    monkeypatch.setattr(spec, "GENERATION_SPEC", spec.GENERATION_SPEC + "\nchanged guidance")
    assert spec.generation_spec_identity()["sha256"] != before["sha256"]


@pytest.mark.parametrize("name", TEMPLATES)
def test_all_generation_templates_get_the_same_rules_without_role_or_skill_opt_in(name):
    rendered = PromptManager().render(name)
    assert rendered.startswith(spec.GENERATION_SPEC + "\n\n")
    assert rendered.count(spec.GENERATION_SPEC) == 1


def test_prompt_family_inventory_covers_every_production_template_except_intent_parsing():
    manager = PromptManager()
    assert set(manager.list_prompts()) - set(TEMPLATES) == {"intent_parse"}
    assert "fixer_repair" in TEMPLATES and "coder_repair" in TEMPLATES
    assert spec.GENERATION_SPEC not in manager.render("intent_parse")


@pytest.mark.parametrize("role", ("designer", "planner", "coder", "fixer", "search"))
def test_custom_template_for_a_generation_role_gets_rules_without_a_new_placeholder(tmp_path, role):
    (tmp_path / "custom.md").write_text("Local task: {{task}}", encoding="utf-8")
    rendered = PromptManager(prompts_dir=str(tmp_path)).render(
        "custom", _agent_role=role, task="caller-owned-input")
    assert rendered == spec.GENERATION_SPEC + "\n\nLocal task: caller-owned-input"


def test_bounded_generation_uses_the_identical_spec_without_an_implementation_template():
    assert bounded.PROMPT.count(spec.GENERATION_SPEC) == 1
    assert "No retries or alternative candidates" in bounded.PROMPT
    assert "tl.load" not in bounded.PROMPT and "tl.store" not in bounded.PROMPT
    assert "strict JSON object" in bounded.PROMPT.replace("\n", " ")


@pytest.mark.parametrize("name", ("coder_generate", "coder_generate_triton",
                                  "coder_generate_cuda_c", "coder_from_plan"))
@pytest.mark.parametrize("inputs", ({"X": np.zeros((1, 3), dtype=np.float32)},
                                    {"A": np.zeros((2, 3), dtype=np.float16),
                                     "B": np.zeros((3, 2), dtype=np.float16)}))
def test_the_actual_caller_convention_is_preserved_once_in_every_coder_path(name, inputs):
    convention = broadcast_calling_convention(inputs)
    rendered = PromptManager().render(name, calling_convention=convention,
                                      _agent_role="coder", _skill_name="triton_skill")
    assert rendered.count(convention) == 1
    assert rendered.count(spec.GENERATION_SPEC) == 1
    assert "{{calling_convention}}" not in rendered


@pytest.mark.parametrize("name", ("designer_sketch", "coder_generate_triton", "coder_generate_cuda_c",
                                  "coder_from_plan", "coder_repair", "fixer_repair"))
def test_rendered_prompts_do_not_restore_the_known_conflicting_rules(name):
    role = "designer" if name.startswith("designer_") else "coder"
    rendered = PromptManager().render(name, _agent_role=role, _skill_name="triton_skill")
    for obsolete in ("FP16 累加器 (tl.float16)", "使用 BLOCK_SIZE=256 分块处理",
                     "单个 tile 不超过 256 个 fp32 元素", "@triton.jit def kernel",
                     "单 tile 不超过 256 个 fp32 元素", "sum 和 sum_sq",
                     "Required: Single-pass mean/variance computation",
                     "Use fp16/bf16 for accumulation where numerical error is acceptable"):
        assert obsolete not in rendered
    assert "LayerNorm:" in rendered and "Softmax:" in rendered
    assert "reduction identity after all transformations" in rendered
    assert "logical reduction length for normalization" in rendered
    assert "serialized NumPy strides are in bytes" in rendered
    assert "fp16/bf16 inputs do not imply fp16/bf16" in rendered


def test_shared_guidance_is_not_a_verification_or_tolerance_switch():
    text = spec.GENERATION_SPEC
    assert "Preserve all correctness, measurement, confirmation and admission gates" in text
    assert "Static checks and plausible rationale do not prove" in text
    assert "supported zero strides" in text
    assert "no universal 256-element tile" in text


@pytest.mark.parametrize("name", ("coder_generate_triton", "coder_repair", "fixer_repair"))
def test_large_offset_guidance_reaches_generation_and_repair_without_changing_the_contract(name):
    rendered = PromptManager().render(name, _agent_role="coder")
    # A centered-square variance alone does not protect the earlier mean from
    # cancellation. Both generation and repair must receive the more precise rule.
    assert "sum(x)/N followed by" in rendered
    assert "centered residual in that shifted domain for both variance and" in rendered
    assert "Mask shifted values and centered squares" in rendered
    assert "empty-input behavior before loading an" in rendered
    # The recipe is conditional on the reference contract, never a new oracle or
    # an allowance to hide instability with a variance/tolerance adjustment.
    assert "When algebraically equivalent statistics are permitted by the frozen contract" in rendered
    assert "population/sample variance convention" in rendered
    assert "Do not clamp variance\nor loosen tolerances" in rendered
    assert "only when the frozen task declares them" in rendered
    assert "eps=1e-5" not in spec.GENERATION_SPEC


def test_scalar_and_scope_guidance_is_shared_with_the_bounded_coder():
    # Scalar conversion must not assume all dimensions are runtime tl.tensor
    # objects; allocation must not refer to names local to a separate JIT scope.
    for rendered in (PromptManager().render("coder_generate_triton"), bounded.PROMPT):
        assert "dimension equal to one" in rendered
        assert "tl.cast(N, tl.float32) rather than N.to(...)" in rendered
        assert "Loaded tensors may use tensor .to(...)" in rendered
        assert "declared single-column and tail cases" in rendered
        assert "without hard-coded input-value branches" in rendered
        assert "names visible in the Python wrapper" in rendered
        assert "local tile constant inside a JIT helper is not a wrapper variable" in rendered
