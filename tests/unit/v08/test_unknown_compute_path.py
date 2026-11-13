"""候选计算路径未确立时，prompt 里只许出现"没有读数"，不许出现编造出来的 0。

2026-11-08：IEEE FP32 修复之后，verifier 在拿不到候选 PTX 证据时把
`gflops_utilization_pct` 置为 None（不再拿设备支持的精度冒充候选实际用的精度）。
那是对的——但 None 一路走到 `_llm_optimize` 的格式化器里抛 TypeError，被上层吞成
"LLM 优化失败 ⇒ 回退规则优化"：结构搜索静默消失，日志里看起来只是一句回退。
"""
from types import SimpleNamespace

from operator_agent.core.schemas import UnifiedSketch
from operator_agent.generate.agents import SearchAgent


def perf_data(**overrides):
    body = dict(gflops=1000.0, speedup=0.88, target_speedup=1.0, avg_time_ms=0.096,
                memory_bandwidth_gbs=3000.0,
                # verifier 在计算路径未确立时的真实形态：利用率为 None、峰值为 0、轴 unknown
                hw_peak_gflops=0.0, hw_peak_bandwidth_gbs=4000.0,
                gflops_utilization_pct=None, bandwidth_utilization_pct=4.3,
                arithmetic_intensity=512.0, ridge_point=None, roofline_bound='unknown',
                name_bound='compute', hw_name='nvidia h20', hw_compute_capability='9.0',
                shared_mem_per_sm_kb=228, elem_bytes=2, optimization_history=[],
                structural_escalation_denied=[])
    body.update(overrides)
    return body


def prompt_for(perf):
    prompts = []
    llm = SimpleNamespace(chat_json=lambda messages, **kw:
                          prompts.append(messages[0]['content']) or {})
    SearchAgent(llm=llm)._llm_optimize(UnifiedSketch(algorithm='matmul', target_dsl='python'), perf)
    return prompts[0]


def test_a_missing_attainment_renders_as_no_reading_instead_of_crashing():
    prompt = prompt_for(perf_data())
    assert '无可用读数' in prompt
    # 不许编一个 0.0% 出来：0.0% 会被读成"这个核一点算力都没用上"
    assert '0.0%' not in prompt
    # 也不许把 0 峰值印成"这个卡没有算力"
    assert '峰值' in prompt and '0.0 |' not in prompt


def test_a_known_attainment_still_renders_as_a_number():
    prompt = prompt_for(perf_data(hw_peak_gflops=148000.0, gflops_utilization_pct=60.2,
                                  ridge_point=37.0, roofline_bound='compute'))
    assert '60.2%' in prompt and '148000.0' in prompt
    assert '无可用读数' not in prompt


def test_the_host_projection_carries_unknown_rather_than_a_zero():
    """arm C 拿到的是投影后的事实：路径未确立时它必须**缺席**利用率，而不是 0。"""
    from operator_agent.generate.host_context import _performance_projection

    metrics = dict(measurement_validity='degraded', gflops=1000.0,
                   speedup_vs_baseline=0.88, median_time_ms=0.096,
                   gflops_utilization_pct=None, bandwidth_utilization_pct=4.3,
                   roofline_analysis=dict(compute_used='unknown', bound_type='unknown',
                                          roofline_bound='unknown', peak_trusted=False,
                                          reference_lower_bound=False,
                                          gflops_utilization_pct=None,
                                          hw_peak_gflops=0.0,
                                          arithmetic_intensity=512.0))
    perf = _performance_projection('L3', metrics)
    assert perf['roofline']['compute_used'] == 'unknown'
    assert perf['roofline']['bound_type'] == 'unknown'
    assert 'gflops_utilization_pct' not in perf['roofline']
    assert 'binding_attainment_pct' not in perf          # 没有可算的达成率就不写这一格
    assert perf['reference_ceiling_reached'] is False    # 天花板出口保持关闭


# ---------------------------------------------------------------------------
# 上下文字节上限：超限时**裁最不承重的一块并登记**，不是把上限调大
# ---------------------------------------------------------------------------

def test_an_oversized_context_drops_template_text_and_says_so(monkeypatch):
    """2026-11-08：attention 在 S=4096 的 optimize 体是 41.3 KB，超过 40 KB 上限 334 字节，
    arm C 直接拿不到上下文。裁的是角色模板正文（宿主自己组装消息），
    名字/角色/sha256 全留着——按哈希仍能取回原文。"""
    from operator_agent.app.hitl_tools import build_task
    from operator_agent.core.contract import freeze_from_task
    from operator_agent.generate import host_context as H

    # generate 体 34.1 KB，其中模板正文 8.9 KB：上限压到 30 KB 就只会裁这一块。
    monkeypatch.setattr(H, "MAX_CONTEXT_BYTES", 30_000)
    task = build_task({"request": "fp16 4096x4096x4096 matmul, triton",
                       "hardware": "nvidia_gpu", "dsl": "triton",
                       "precision": "fp16", "shapes": [[4096, 4096, 4096]]})
    ctx = H.build_host_context(task, freeze_from_task(task), backend="ssh", phase="generate")
    integrity = ctx["material_integrity"]
    assert integrity["truncated"] is True
    assert integrity["dropped"] == ["materials.templates[].text"]
    assert "context byte budget" in integrity["truncation"]
    assert all("text" not in row for row in ctx["materials"]["templates"])
    assert all(row.get("sha256") for row in ctx["materials"]["templates"])   # 身份不能丢
    assert ctx["materials"]["skills"]                                        # 领域材料不许裁


def test_a_context_that_cannot_fit_after_trimming_still_fails_closed(monkeypatch):
    """裁到没有可裁的还超限 ⇒ 抛错。**不许**为了通过上限把领域材料也悄悄扔掉。"""
    import pytest
    from operator_agent.app.hitl_tools import build_task
    from operator_agent.core.contract import freeze_from_task
    from operator_agent.generate import host_context as H

    monkeypatch.setattr(H, "MAX_CONTEXT_BYTES", 8_000)
    task = build_task({"request": "fp16 4096x4096x4096 matmul, triton",
                       "hardware": "nvidia_gpu", "dsl": "triton",
                       "precision": "fp16", "shapes": [[4096, 4096, 4096]]})
    with pytest.raises(ValueError, match="byte limit"):
        H.build_host_context(task, freeze_from_task(task), backend="ssh", phase="generate")


def test_a_normal_context_is_not_marked_truncated():
    from operator_agent.app.hitl_tools import build_task
    from operator_agent.core.contract import freeze_from_task
    from operator_agent.generate.host_context import build_host_context

    task = build_task({"request": "fp16 4096x4096x4096 matmul, triton",
                       "hardware": "nvidia_gpu", "dsl": "triton",
                       "precision": "fp16", "shapes": [[4096, 4096, 4096]]})
    ctx = build_host_context(task, freeze_from_task(task), backend="ssh", phase="generate")
    assert ctx["material_integrity"]["truncated"] is False
    assert all("text" in row for row in ctx["materials"]["templates"])
