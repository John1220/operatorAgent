"""The attention reference is registered on its own profile, with calibrated tolerance."""
import pytest

from operator_agent.decide.reference_contract import (PROFILE_ATTENTION, PROFILE_V2,
                                                      ReferenceIdentityError,
                                                      as_reference, registered_reference)


def test_attention_has_its_own_contract_and_layernorm_is_untouched():
    layer = registered_reference(PROFILE_V2)
    attn = registered_reference(PROFILE_ATTENTION)
    assert layer.profile == PROFILE_V2 and attn.profile == PROFILE_ATTENTION
    assert layer.definition.startswith('native-pytorch-layernorm')
    assert attn.definition == 'native-pytorch-sdpa-mha-prefill-fp16-v1'
    assert layer.digest() != attn.digest()
    # layernorm 的 digest 是冻结锚点：改它等于作废所有已归档回执
    assert layer.digest().startswith('8f779954d3d2b3af')


def test_attention_semantics_say_what_it_is_and_is_not():
    attn = registered_reference(PROFILE_ATTENTION)
    assert attn.epsilon == 0.0            # 没有 eps 归一化
    assert attn.normalized_axis == -1     # softmax 沿末轴
    assert attn.affine is False
    assert attn.fp_fusion == 'forbid_native_sdpa'
    assert attn.deterministic_outputs is True
    assert attn.rtol == 1e-2 and attn.required_matched_ratio == 1.0


def test_an_unregistered_profile_fails_closed():
    with pytest.raises(ReferenceIdentityError, match='no registered reference'):
        registered_reference('builtin-attention-fp16-cuda-v99')
    broken = registered_reference(PROFILE_ATTENTION).content_dict()
    broken['profile'] = 'builtin-attention-fp16-cuda-v99'
    with pytest.raises(ReferenceIdentityError):
        as_reference(broken)


@pytest.mark.parametrize('field,value', [('rtol', 0.5), ('required_matched_ratio', 0.9),
                                         ('fp_fusion', 'allowed'), ('definition', 'invented')])
def test_a_relaxed_or_renamed_attention_contract_is_refused(field, value):
    tampered = registered_reference(PROFILE_ATTENTION).content_dict()
    tampered[field] = value
    with pytest.raises(ReferenceIdentityError):
        as_reference(tampered)


def test_the_registry_round_trips_every_entry():
    for profile in (PROFILE_V2, PROFILE_ATTENTION):
        spec = registered_reference(profile)
        assert as_reference(spec.content_dict()) == spec


def test_the_registered_absolute_tolerance_is_per_profile():
    from operator_agent.decide.reference_contract import (PROFILE_FUSED_LN_RESIDUAL,
                                                          REGISTERED_ATOL, registered_atol)
    assert REGISTERED_ATOL == {PROFILE_V2: 1e-5, PROFILE_ATTENTION: 3e-4,
                               PROFILE_FUSED_LN_RESIDUAL: 5e-3}
    for profile, expected in ((PROFILE_V2, 1e-5), (PROFILE_ATTENTION, 3e-4),
                              (PROFILE_FUSED_LN_RESIDUAL, 5e-3)):
        assert registered_atol(registered_reference(profile).content_dict()) == expected


def test_an_unknown_profile_has_no_absolute_tolerance_to_borrow():
    """L3/dispatch 用 registered_atol 填 atol 槽; 未知 profile 必须 fail closed, 不兜默认。"""
    from operator_agent.decide.reference_contract import registered_atol
    broken = registered_reference(PROFILE_ATTENTION).content_dict()
    broken['profile'] = 'builtin-attention-fp16-cuda-v99'
    with pytest.raises(ReferenceIdentityError):
        registered_atol(broken)


def _bare_fp16_task(kernel_type):
    from operator_agent.core.schemas import (ConstraintSpec, Precision, SourceSpec,
                                             TargetSpec, TaskSchema)
    return TaskSchema(source=SourceSpec(kernel_type=kernel_type),
                      target=TargetSpec(dsl='triton', hardware='nvidia_gpu'),
                      constraints=ConstraintSpec(precision=Precision.FP16, min_speedup=1.0))


@pytest.mark.parametrize('kernel_type,profile,expected', [
    ('attention', PROFILE_ATTENTION, 3e-4),
    ('layernorm', PROFILE_V2, 1e-5),
])
def test_attaching_a_trusted_reference_binds_its_calibrated_tolerance(kernel_type, profile, expected):
    """绑定必须发生在挂参考那一步：调用方无权设容差（max_error/atol/rtol 都在
    hitl_tools.FORBIDDEN_KEYS 里），所以系统不绑就没人能绑。

    layernorm 那一格是**不变式**：它的标定值恰好等于旧全局默认 1e-5 ⇒ 绑定是 no-op ⇒
    契约内容不变 ⇒ 已冻结的 layernorm 回执一份都不作废（实测哈希前后逐字相同）。
    attention 的标定值是 3e-4，所以它的契约哈希会变，必须按新编号重冻。
    """
    from operator_agent.decide.correctness_profiles import with_default_correctness
    task = _bare_fp16_task(kernel_type)
    assert task.constraints.max_error == 1e-5        # ConstraintSpec 的全局默认
    bound = with_default_correctness(task)
    assert bound.constraints.correctness_profile == profile
    assert bound.constraints.max_error == expected
    assert bound.constraints.reference_contract is not None


def test_a_task_without_a_trusted_reference_keeps_its_own_tolerance():
    """没有受信任参考的任务（axis 生成器那套）不该被塞一个 profile 的标定值。"""
    from operator_agent.decide.correctness_profiles import with_default_correctness
    bound = with_default_correctness(_bare_fp16_task('matmul'))
    assert bound.constraints.reference_contract is None
    assert bound.constraints.max_error == 1e-5


# ---------------------------------------------------------------------------
# C5 现场（2026-11-08）：v2 回执校验不许假设"第一个输入叫 X"
# ---------------------------------------------------------------------------

import copy

import numpy as np

from operator_agent import Verifier
from operator_agent.app.hitl_tools import build_task
from operator_agent.decide.correctness_profiles import expected_output_elements
from operator_agent.decide.reference_contract import check_observed
from operator_agent.execute.remote_executor import SSHExecutor

_ATTENTION_DECLARATION = {
    "request": "fp16 attention prefill 1x16x1024x128",
    "shapes": [[1, 16, 1024, 128]],
    "hardware": "nvidia_gpu", "dsl": "triton", "precision": "fp16", "min_speedup": 1.0,
}


def _identity_receipt():
    spec = registered_reference(PROFILE_ATTENTION)
    return check_observed(spec, spec.execution.content_dict())


class _AttentionPeer(SSHExecutor):
    """Deliberately fabricated peer: proves the controller branch, never an oracle."""

    def __init__(self):
        super().__init__(host="offline")
        self._deployed = True
        self.requests = []

    def _ssh_exec_json(self, payload):
        from operator_agent.decide.remote_input_fidelity import (decode_json_inputs,
                                                                 to_torch_inputs,
                                                                 torch_input_readback)
        self.requests.append(copy.deepcopy(payload))
        if payload["action"] == "gpu_info":
            return dict(success=True, reference_receipt=_identity_receipt())
        values = decode_json_inputs(payload["inputs"])
        reading = torch_input_readback(to_torch_inputs(values, "cpu"))
        reading["tensor_devices"] = {k: "cuda:0" for k in values}
        return dict(success=True, reference_receipt=_identity_receipt(),
                    output_dtype="torch.float16", input_fidelity=reading,
                    max_error=1.0e-4, max_rel_error=1.0e-3, matched_ratio=1.0,
                    bad_elements=0, compared_elements=values["Q"].size, coverage=1.0,
                    atol=payload["max_error"], rtol=1.0e-2, required_matched_ratio=1.0,
                    reference_nonfinite=0, output_digest="f" * 64, device="cuda",
                    reference_type="pytorch_eager")


def test_expected_output_elements_is_owned_by_the_profile_not_by_a_key_name():
    """输出有多大由 profile 说，不看输入键名；问不出来就返回 None，由调用方判 unknown。"""
    attn = as_reference(registered_reference(PROFILE_ATTENTION))
    q = np.zeros((1, 2, 8, 4), np.float16)
    assert expected_output_elements(attn, {"Q": q, "K": q, "V": q}) == q.size
    assert expected_output_elements(attn, {"K": q}) is None          # 没有 Q ⇒ 不猜
    layer = as_reference(registered_reference(PROFILE_V2))
    x = np.zeros((3, 5), np.float16)
    assert expected_output_elements(layer, {"X": x}) == x.size
    assert expected_output_elements(type("S", (), {"profile": "unknown-p"})(), {"X": x}) is None


def test_the_v2_receipt_gate_accepts_qkv_inputs_without_an_x_tensor():
    """没有 `X` 的输入集也要能过 v2 回执校验 —— 旧写法在这里 KeyError 到 worker 崩溃。"""
    task = build_task(dict(_ATTENTION_DECLARATION))
    peer = _AttentionPeer()
    # 只给三张量: 这条测的是"没有 X 的输入集"，标量路径由 test_case_scalars 与真机回执覆盖。
    # 形状必须是 Scope A 内的（A5 之后 S=256 会在 prepare 处就被拒）——
    # 这条用例测的是 v2 回执校验，不是几何边界。
    inputs = {"Q": np.zeros((1, 16, 1024, 128), np.float16),
              "K": np.zeros((1, 16, 1024, 128), np.float16),
              "V": np.zeros((1, 16, 1024, 128), np.float16)}
    result = Verifier(peer)._verify_single_set("def kernel(Q, K, V, causal):\n    return Q\n",
                                               inputs, None, task)
    assert result.passed, result.message
    assert result.metrics["compared_elements"] == inputs["Q"].size
