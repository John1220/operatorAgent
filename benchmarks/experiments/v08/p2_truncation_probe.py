"""P2 起点核截断成分探针：查清 xhigh Coder 撞 40960 顶时，思考 vs 最终代码各占多少。

四次起点核生成（output/v08-p2-seed{2,3,4,5}）Coder 都写满 40960 截断、核 JSON 没发出。
v6 prompt 修正（别在思考里复述源码）也压不住。要决定「降推理档位」还是「提内层上限」，
先拿一次**真实 Coder prompt** 的截断响应，量 ``reasoning_content``（思考）与 ``content``
（最终代码）的占比：思考占绝大多数 ⇒ 降档位有效；代码本身就长 ⇒ 需提上限。

忠实重建 CoderAgent._llm_generate_code 的 prompt（build_generation_context +
coder_generate_triton 模板），对 vLLM 发**原始** chat completion（绕过 LLMClient 的截断即抛，
才能拿到完整响应体）。一次模型调用，不跑 Conductor、不占 GPU 验证。
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

from operator_agent.app.hitl_tools import build_task
from operator_agent.core.contract import freeze_from_task
from operator_agent.core.schemas import UnifiedSketch, TilingStrategy
from operator_agent.core.interface_contract import broadcast_calling_convention
from operator_agent.generate.generation_context import build_generation_context
from operator_agent.generate.prompt_manager import PromptManager

DECL = dict(operator='layernorm', request='fp16 16384x6144 layernorm', shapes=[[16384, 6144]],
            precision='fp16', dsl='triton', hardware='nvidia_gpu', min_speedup=1.0)


def build_coder_prompt():
    task = build_task(DECL)
    contract = freeze_from_task(task)
    inputs = contract.cases[0].materialize()
    sketch = UnifiedSketch(algorithm='layernorm', target_dsl='triton', version=1,
                           tiling=TilingStrategy(block_m=16, block_n=512, block_k=32))
    pm = PromptManager()
    context = build_generation_context(task, sketch, inputs,
        producer='CoderAgent', strategy='single-with-precheck-retry',
        policy=dict(max_attempts=3, render_dsl='triton'),
        prompt_manager=pm, agent_role='coder', skill_name='triton_skill')
    calling_convention = context['interface']['broadcast']
    prompt = pm.render('coder_generate_triton',
        _agent_role='coder', _skill_name='triton_skill',
        kernel_type=context['task']['kernel_type'], hardware=context['task']['hardware'],
        dsl='triton', precision=context['task']['precision'],
        declared_shapes=json.dumps(task.constraints.shapes),
        sketch=json.dumps(sketch.to_prompt_dict(), ensure_ascii=False, indent=2),
        ref_code='无', optimization_hints='无',
        optimization_recipes='（本轮没有新增配方，按上述提示优化即可）',
        calling_convention=calling_convention)
    return prompt


def analyze(result, *, reasoning_effort, max_tokens, wall):
    """vLLM qwen3 reasoning-parser 把思考放在 ``message.reasoning``（不是 reasoning_content）；
    token 拆分以 ``usage.completion_tokens_details.reasoning_tokens`` 为准。"""
    choice = (result.get('choices') or [{}])[0]
    message = choice.get('message') or {}
    reasoning = message.get('reasoning') or message.get('reasoning_content') or ''
    content = message.get('content') or ''
    finish = choice.get('finish_reason')
    usage = result.get('usage') or {}
    completion_tokens = usage.get('completion_tokens')
    details = usage.get('completion_tokens_details') or {}
    reasoning_tokens = details.get('reasoning_tokens')
    content_tokens = (completion_tokens - reasoning_tokens
                      if completion_tokens is not None and reasoning_tokens is not None else None)
    reasoning_token_frac = (reasoning_tokens / completion_tokens
                            if completion_tokens and reasoning_tokens is not None else None)
    has_kernel = 'def kernel' in content and '@triton.jit' in content
    return dict(
        reasoning_effort=reasoning_effort, max_tokens=int(max_tokens),
        finish_reason=finish, wall_s=wall,
        completion_tokens=completion_tokens, reasoning_tokens=reasoning_tokens,
        content_tokens=content_tokens, reasoning_token_frac=reasoning_token_frac,
        prompt_tokens=usage.get('prompt_tokens'),
        reasoning_chars=len(reasoning), content_chars=len(content),
        content_has_complete_kernel=has_kernel,
        content_tail=content[-400:], reasoning_tail=reasoning[-400:], usage=usage)


def probe(*, base_url, model_id, reasoning_effort, max_tokens, root):
    prompt = build_coder_prompt()
    body = dict(model=model_id, messages=[dict(role='user', content=prompt)],
                temperature=0.1, max_tokens=int(max_tokens), stream=False,
                enable_thinking=True, reasoning_effort=reasoning_effort, n=1)
    data = json.dumps(body).encode('utf-8')
    req = urllib.request.Request(base_url.rstrip('/') + '/chat/completions',
                                 data=data, headers={'Content-Type': 'application/json'},
                                 method='POST')
    print(f'[p2-trunc] calling vLLM reasoning_effort={reasoning_effort} '
          f'max_tokens={max_tokens} prompt_chars={len(prompt)}', flush=True)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        result = json.loads(resp.read().decode('utf-8'))
    wall = round(time.time() - t0, 1)
    report = analyze(result, reasoning_effort=reasoning_effort, max_tokens=max_tokens, wall=wall)
    out = Path(root)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'truncation-report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (out / 'raw-response.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (out / 'coder-prompt.txt').write_text(prompt, encoding='utf-8')
    print(f'[p2-trunc] finish={report["finish_reason"]} completion={report["completion_tokens"]} '
          f'reasoning_tokens={report["reasoning_tokens"]} content_tokens={report["content_tokens"]} '
          f'wall={wall}s', flush=True)
    print(f'[p2-trunc] reasoning_token_frac={report["reasoning_token_frac"]} '
          f'content_has_complete_kernel={report["content_has_complete_kernel"]}', flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog='p2_truncation_probe', description=__doc__.splitlines()[0])
    parser.add_argument('--base-url', default='http://127.0.0.1:18003/v1')
    parser.add_argument('--model-id', default='Qwen3.8-27B')
    parser.add_argument('--reasoning-effort', default='xhigh')
    parser.add_argument('--max-tokens', type=int, default=40960)
    parser.add_argument('--root', default='output/v08-p2-trunc-20261116')
    args = parser.parse_args(argv)
    report = probe(base_url=args.base_url, model_id=args.model_id,
                   reasoning_effort=args.reasoning_effort, max_tokens=args.max_tokens,
                   root=args.root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())