"""A4/A5（2026-11-08）：attention 的支持清单要读得到，Scope A 之外要被拒绝。

两条缺口的共同点是"静默"：
  * A4：`capabilities()` 只报二维 axis 生成器的四个算子，调用方（含本会话开场那次）
    会以为 attention 不支持 —— 而它当时已经有 profile、参考、8 条 case 与 L2/L3 判定。
  * A5：`S=3072`、`B=8/H=8/D=64` 都能**静默冻结契约**，于是一份标定过的参考与容差
    被拿去判它从没标定过的形状。这不叫"支持"，叫"没拦住"。
"""
import contextlib
import io
import json
import os

import pytest

from operator_agent.app.hitl_tools import ToolError, capabilities, prepare


def declaration(*shape, precision='fp16'):
    """按**字面**形状造声明 —— 不要给参数默认值：`declaration(1, 16, 1024)` 会
    被默认的 d=128 补齐，于是"维度数不对"那条用例根本没测到它想测的东西。"""
    dims = [int(value) for value in shape]
    label = "x".join(str(value) for value in dims)
    return {"request": f"{precision} attention prefill {label}",
            "shapes": [dims], "hardware": "nvidia_gpu", "dsl": "triton",
            "precision": precision, "min_speedup": 1.0}


def test_capabilities_publish_the_attention_family_and_its_edges():
    profiles = capabilities()["correctness_profiles"]
    family = profiles["families"]["attention"]
    assert family["profile"] == "builtin-attention-fp16-cuda-v1"
    assert family["dtype"] == ["fp16"] and family["correctness_cases"] == 8
    assert "B=1" in family["scope"] and "S in {1024,2048,4096}" in family["scope"]
    assert any("GQA" in row for row in family["excluded"])
    assert any("decode" in row for row in family["excluded"])
    # 二维 axis 生成器的词表不动，但它必须被标明"不是支持清单"
    assert profiles["operators"] == ["relu", "matmul", "softmax", "layernorm"]
    assert "不是支持清单" in profiles["operators_scope"]


@pytest.mark.parametrize("shape", [[1, 16, 1024, 128], [1, 16, 2048, 128], [1, 16, 4096, 128]])
def test_scope_a_shapes_still_freeze(shape, tmp_path):
    result = prepare(declaration(*shape), tmp_path / "run")
    assert result["ok"] is True
    assert result["correctness_profile"] == "builtin-attention-fp16-cuda-v1"
    assert result["judgement"]["max_error"] == pytest.approx(3e-4)


@pytest.mark.parametrize("shape,needle", [
    ([1, 16, 3072, 128], "S 只支持"),          # Scope A 之外的规模
    ([1, 16, 1024, 64], "head 维只支持"),       # D=64
    ([1, 8, 1024, 128], "只支持 MHA"),          # GQA/MQA 之外的 H
    ([8, 16, 1024, 128], "batch 只支持"),       # 批量
    ([1, 16, 1024], "必须是 (B,H,S,D)"),        # 维度数不对
])
def test_out_of_scope_shapes_are_refused_by_name(shape, needle, tmp_path):
    with pytest.raises(ToolError) as caught:
        prepare(declaration(*shape), tmp_path / "run")
    assert caught.value.code == "attention_scope"
    assert "Scope A" in caught.value.detail and needle in caught.value.detail


def test_the_refusal_reaches_the_cli_as_one_json_receipt(tmp_path):
    from operator_agent.app.cli import main as cli_main

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli_main(["prepare", "--request", "fp16 attention prefill 1x16x3072x128",
                         "--shapes", "1,16,3072,128", "--hardware", "nvidia_gpu",
                         "--dsl", "triton", "--precision", "fp16",
                         "--run-dir", str(tmp_path / "run")])
    assert code == 2
    payload = json.loads(buf.getvalue())
    assert payload["error"] == "attention_scope"
    assert "3072" in payload["detail"]
    # 拒绝发生在冻结之前：不许留下半份契约
    assert not os.path.exists(tmp_path / "run" / "contract.json")


def test_the_other_profiles_are_not_touched():
    """只收紧 attention：layernorm / matmul 的默认套件与容差一个字节都不动。"""
    for request, profile in (("layernorm fp16 triton nvidia", "builtin-layernorm-fp16-cuda-v2"),
                             ("fp16 4096x4096x4096 matmul, triton", "builtin-l2-v1")):
        from operator_agent.app.hitl_tools import build_task
        task = build_task({"request": request, "hardware": "nvidia_gpu", "dsl": "triton",
                           "precision": "fp16", "shapes": [[4096, 4096, 4096]]
                           if "matmul" in request else [[16384, 6144]]})
        assert task.constraints.correctness_profile == profile