"""Fused A/C pre-freeze smoke: does the real Coder prompt fit the 32768-token wire cap?

The frozen execution identity is ``Qwen3.8-27B / xhigh / enable_thinking / max_tokens=32768``
(40960 is the ledger reservation, not the wire cap). P2 learned the hard way that
``xhigh`` plus the full 11-recipe Coder prompt can write the whole kernel inside its
reasoning and run to the cap; when that happens the frozen source is never emitted.

This smoke runs the **real** Coder prompt for the *fused* task twice and records, per
attempt: finish_reason, completion/reasoning/content tokens, and whether the content
holds a complete ``@triton.jit`` + ``def kernel`` pair.

Exit code 0 only when every attempt finished with ``stop`` **and** emitted a complete
kernel. Anything else means: do not freeze the A/C round; decide (drop to medium, or
tighten the prompt) and record that decision as a new plan identity.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

from operator_agent.app.hitl_tools import build_task
from operator_agent.core.contract import freeze_from_task
from operator_agent.core.interface_contract import broadcast_calling_convention
from operator_agent.core.schemas import TilingStrategy, UnifiedSketch
from operator_agent.generate.generation_context import build_generation_context
from operator_agent.generate.prompt_manager import PromptManager

DECL = dict(operator='fused_layernorm_residual', request="fp16 32768x6144 residual add layernorm", shapes=[[32768, 6144]],
            precision="fp16", dsl="triton", hardware="nvidia_gpu", min_speedup=1.5)
WIRE_MAX_TOKENS = 32768


def build_coder_prompt():
    task = build_task(DECL)
    contract = freeze_from_task(task)
    inputs = contract.cases[0].materialize()
    sketch = UnifiedSketch(algorithm="fused_layernorm_residual", target_dsl="triton", version=1,
                           tiling=TilingStrategy(block_m=16, block_n=512, block_k=32))
    pm = PromptManager()
    context = build_generation_context(task, sketch, inputs,
                                       producer="CoderAgent",
                                       strategy="single-with-precheck-retry",
                                       policy=dict(max_attempts=3, render_dsl="triton"),
                                       prompt_manager=pm, agent_role="coder",
                                       skill_name="triton_skill")
    return pm.render("coder_generate_triton",
                     _agent_role="coder", _skill_name="triton_skill",
                     kernel_type=context["task"]["kernel_type"],
                     hardware=context["task"]["hardware"],
                     dsl="triton", precision=context["task"]["precision"],
                     declared_shapes=json.dumps(task.constraints.shapes),
                     sketch=json.dumps(sketch.to_prompt_dict(), ensure_ascii=False, indent=2),
                     ref_code="无", optimization_hints="无",
                     optimization_recipes="（本轮没有新增配方，按上述提示优化即可）",
                     calling_convention=broadcast_calling_convention(
                         context["interface"]["broadcast"]))


def analyze(result, *, attempt, wall):
    choice = (result.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    content = message.get("content") or ""
    usage = result.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    reasoning_tokens = details.get("reasoning_tokens")
    completion_tokens = usage.get("completion_tokens")
    complete = ("@triton.jit" in content and "def kernel" in content
                and "return" in content)
    return dict(attempt=attempt, finish_reason=choice.get("finish_reason"), wall_s=wall,
                completion_tokens=completion_tokens,
                reasoning_tokens=reasoning_tokens,
                content_tokens=(completion_tokens - reasoning_tokens
                                if completion_tokens is not None
                                and reasoning_tokens is not None else None),
                prompt_tokens=usage.get("prompt_tokens"),
                content_has_complete_kernel=complete,
                content_chars=len(content), reasoning_chars=len(reasoning),
                content_tail=content[-300:])


def run(*, base_url, model_id, reasoning_effort, attempts, root):
    prompt = build_coder_prompt()
    out = Path(root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "coder-prompt.txt").write_text(prompt, encoding="utf-8")
    reports = []
    for index in range(1, attempts + 1):
        body = dict(model=model_id, messages=[dict(role="user", content=prompt)],
                    temperature=0.1, max_tokens=WIRE_MAX_TOKENS, stream=False,
                    enable_thinking=True, reasoning_effort=reasoning_effort, n=1)
        request = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        print(f"[fused-smoke] attempt {index}: effort={reasoning_effort} "
              f"max_tokens={WIRE_MAX_TOKENS} prompt_chars={len(prompt)}", flush=True)
        started = time.time()
        with urllib.request.urlopen(request, timeout=1800) as response:
            result = json.loads(response.read().decode("utf-8"))
        report = analyze(result, attempt=index, wall=round(time.time() - started, 1))
        (out / f"raw-attempt-{index}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        reports.append(report)
        print(f"[fused-smoke] attempt {index}: finish={report['finish_reason']} "
              f"completion={report['completion_tokens']} "
              f"reasoning={report['reasoning_tokens']} "
              f"complete_kernel={report['content_has_complete_kernel']} "
              f"wall={report['wall_s']}s", flush=True)
    passed = all(report["finish_reason"] == "stop"
                 and report["content_has_complete_kernel"] for report in reports)
    summary = dict(schema="v08-fused-ac-smoke-v1", passed=passed,
                   reasoning_effort=reasoning_effort, wire_max_tokens=WIRE_MAX_TOKENS,
                   model=model_id, attempts=reports)
    (out / "smoke.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    print(json.dumps(dict(passed=passed,
                          finishes=[r["finish_reason"] for r in reports],
                          complete_kernel=[r["content_has_complete_kernel"] for r in reports]),
                     ensure_ascii=False))
    return 0 if passed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:18003/v1")
    parser.add_argument("--model-id", default="Qwen3.8-27B")
    parser.add_argument("--reasoning-effort", default="xhigh")
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--root", required=True)
    args = parser.parse_args(argv)
    return run(base_url=args.base_url, model_id=args.model_id,
               reasoning_effort=args.reasoning_effort, attempts=args.attempts,
               root=args.root)


if __name__ == "__main__":
    raise SystemExit(main())
