"""P2 方案②：固定轮数 optimize 诊断开关实验入口（受控驱动，R2 同构 prompt 接缝）。

问题：同一份冻结起点核上，ON 臂额外收 roofline 诊断提示 vs OFF 臂不收，给定**同样 N 次
LLM 优化尝试机会**，诊断能帮核再快多少。两臂同测速反馈（perf_data）、同模型、同预算，
唯一差别是 search_optimize prompt 里的 roofline 诊断块。

为什么不用 Conductor 原循环（2026-11-16 勘查定）：插件 optimize 整套（升级门/停滞/邻域
bandit 轮换）围绕「L3 未过」设计。种子 L3 已达 1.46×≥1.0，强制续跑会撞——
conductor.py:2660 升级门缺省拒绝、真重算 2766 只在 2678（L3 fail）分支内 ⇒ L3 过的轮
门停在拒绝 ⇒ agents.py:645 两臂都退规则搜索、prompt 差分根本不渲染；邻域队列存在时
bandit 轮换还会跳过 LLM。故本驱动**直接调插件的 SearchAgent.run N 次**（保留 LLM 优化
+邻域排序），绕过 Conductor 的 L3-gate/升级门/bandit 轮换——这些正是方案②已接受偏离的
生产停止机制（须声明口径）。每轮强制走 LLM 路径（perf_data 不带 escalation_denied/
budget_exhausted），保证两臂各 N 次优化调用；回退由 SearchAgent.last_path 观测落盘。

两臂用同一份 P2 搜索模板。开启臂在模板中加入 Roofline 诊断块，任务说明、硬件信息、测速反馈、
通用优化指导和 JSON 输出格式共用。PromptManager 给两臂渲染同样的 GENERATION_SPEC 和 skill。

step1 本地 mock 验接线（generate/verify 回调可注入）；step3 真机接 CoderAgent+Verifier。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from operator_agent.generate.agents import SearchAgent
from operator_agent.generate.prompt_manager import PromptManager

PROMPTS_DIR = Path(__file__).resolve().parent / 'p2_prompts'
DIAGNOSIS_SLOT = '\n\n{{roofline_diagnosis}}\n\n'


class P2PromptManager(PromptManager):
    """从同一模板渲染两臂的搜索提示，记录模型收到的提示哈希。"""

    def __init__(self, *, include_diagnosis: bool):
        super().__init__()
        template = (PROMPTS_DIR / 'search_optimize.md').read_text(encoding='utf-8').strip()
        if template.count(DIAGNOSIS_SLOT) != 1:
            raise ValueError('P2 搜索模板必须有一个 Roofline 诊断插入位置')
        diagnosis = ((PROMPTS_DIR / 'roofline_diagnosis.md').read_text(encoding='utf-8').strip()
                     if include_diagnosis else '')
        insertion = f'\n\n{diagnosis}\n\n' if diagnosis else '\n\n'
        self._search_template = template.replace(DIAGNOSIS_SLOT, insertion)
        self.last_search_prompt_sha256 = None

    def get_raw(self, name: str) -> str:
        if name == 'search_optimize':
            return self._search_template
        return super().get_raw(name)

    def render(self, name: str, **kwargs) -> str:
        rendered = super().render(name, **kwargs)
        if name == 'search_optimize':
            self.last_search_prompt_sha256 = hashlib.sha256(rendered.encode('utf-8')).hexdigest()
        return rendered


def make_searcher(llm, arm: str) -> SearchAgent:
    arm = arm.lower()
    if arm not in ('on', 'off'):
        raise ValueError(f"arm 必须是 on/off: {arm!r}")
    pm = P2PromptManager(include_diagnosis=arm == 'on')
    return SearchAgent(llm=llm, prompt_manager=pm)


def build_perf_data(metrics: dict, *, min_speedup: float, history: list,
                    elem_bytes: int = 2) -> dict:
    """从 L3 metrics 构造 perf_data，逐键镜像 conductor.py:1870-1896（两臂同源）。

    不带 structural_escalation_denied / llm_budget_exhausted ⇒ SearchAgent.run 走 LLM 路径。
    诊断数据（roofline）照常进 perf_data：OFF 臂的差异只在 prompt 不渲染它，邻域排序仍按
    同一份 perf_data 跑（两臂相同，非混淆）。
    """
    return {
        'gflops': metrics.get('gflops', 0),
        'speedup': metrics.get('speedup_vs_baseline', 0),
        'target_speedup': min_speedup,
        'bound_type': metrics.get('bound_type', 'unknown'),
        'avg_time_ms': metrics.get('avg_time_ms', 0),
        'memory_bandwidth_gbs': metrics.get('memory_bandwidth_gbs', 0),
        'hw_peak_gflops': metrics.get('hw_peak_gflops', 0),
        'hw_peak_bandwidth_gbs': metrics.get('hw_peak_bandwidth_gbs', 0),
        'gflops_utilization_pct': metrics.get('gflops_utilization_pct', 0),
        'bandwidth_utilization_pct': metrics.get('bandwidth_utilization_pct', 0),
        'arithmetic_intensity': metrics.get('arithmetic_intensity', 0),
        'ridge_point': metrics.get('ridge_point', 0),
        'roofline_bound': metrics.get('roofline_bound', 'unknown'),
        'name_bound': metrics.get('name_bound', 'unknown'),
        'hw_name': metrics.get('hw_name', 'unknown'),
        'hw_compute_capability': metrics.get('hw_compute_capability', 'unknown'),
        'shared_mem_per_sm_kb': metrics.get('shared_mem_per_sm_kb', 0),
        'elem_bytes': elem_bytes,
        'optimization_history': history,
    }


def search_round(searcher: SearchAgent, sketch, perf_data: dict) -> tuple:
    """跑一轮插件搜索（LLM 优化+邻域排序），返回 (new_sketch, suggestion, record)。

    record 落盘归因：path（llm/rule_*）、llm_error、prompt 哈希（验两臂差分）。
    """
    new_sketch, suggestion = searcher.run(sketch, perf_data)
    record = dict(path=searcher.last_path, llm_error=searcher.last_llm_error,
                  prompt_sha256=searcher.pm.last_search_prompt_sha256)
    return new_sketch, suggestion, record


def run_optimize_arm(*, seed_sketch, seed_metrics: dict, min_speedup: float, arm: str,
                     n_rounds: int, llm, generate, verify, elem_bytes: int = 2) -> dict:
    """固定 N 轮 optimize（方案②入口）。两臂同 seed/同 perf_data 构造/同模型/同预算。

    generate(sketch, suggestion, round_idx) -> code：Coder 重生成回调（step3 接 CoderAgent）。
    verify(code) -> dict(L3 metrics)：验证回调（step3 接 Verifier，含 L2）。
    每轮：search → generate → verify → 用新 metrics 构下一轮 perf_data。回退逐轮落盘。
    """
    sketch = seed_sketch
    metrics = dict(seed_metrics)
    history: list = []
    rounds = []
    code = None
    for idx in range(1, n_rounds + 1):
        perf_data = build_perf_data(metrics, min_speedup=min_speedup,
                                    history=history, elem_bytes=elem_bytes)
        searcher = make_searcher(llm, arm)
        sketch, suggestion, record = search_round(searcher, sketch, perf_data)
        code = generate(sketch, suggestion, idx)
        metrics = verify(code)
        record.update(round=idx, arm=arm, suggestion=suggestion,
                      speedup=metrics.get('speedup_vs_baseline'),
                      source_sha256=hashlib.sha256(code.encode('utf-8')).hexdigest())
        rounds.append(record)
        history.append(dict(round=idx, tiling=sketch.tiling.to_dict()
                            if hasattr(sketch.tiling, 'to_dict') else str(sketch.tiling),
                            speedup=metrics.get('speedup_vs_baseline'),
                            outcome=record['path']))
    return dict(arm=arm, n_rounds=n_rounds, rounds=rounds, final_code=code,
                final_metrics=metrics,
                llm_optimize_calls=sum(1 for r in rounds if r['path'] == 'llm'),
                fallbacks=[r for r in rounds if r['path'] != 'llm'])
