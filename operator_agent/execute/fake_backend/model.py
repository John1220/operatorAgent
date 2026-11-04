"""Scripted fixtures using the real reservation journal, with zero API traffic."""
import json
import time

from ..llm_budget import ProductionLLMCallBudget
from ..llm_client import CostTracker


FIXTURES = {
    "relu": ("X", "np.maximum(X, 0)"),
    "matmul": ("A, B", "np.matmul(A.astype(np.float32), B.astype(np.float32)).astype(A.dtype)"),
}


class FakeLLM:
    test_only = True
    model = "fake-fixture-v1"
    active_model = model
    fallback_triggered = False
    temperature = .1
    max_tokens = 4096
    enable_thinking = False

    def __init__(self, kernel_type, *, call_budget, timeout=120.):
        if kernel_type not in FIXTURES:
            raise ValueError("fake LLM fixtures support only: " + ", ".join(FIXTURES))
        if not isinstance(call_budget, ProductionLLMCallBudget):
            raise TypeError("fake LLM requires the ordinary production task budget")
        self.kernel_type = kernel_type
        self.call_budget = call_budget
        self.timeout = timeout
        self.cost_tracker = CostTracker()
        self._code_calls = 0
        self._identity = dict(model=self.model, transport="none", test_only=True,
                              kernel_type=kernel_type, timeout=timeout)
        call_budget.bind(self._identity)

    def _respond(self, messages, *, caller_tag, temperature=None, max_tokens=None, json_mode=False):
        body = dict(model=self.model, messages=messages,
                    temperature=self.temperature if temperature is None else temperature,
                    max_tokens=self.max_tokens if max_tokens is None else max_tokens)
        started = time.monotonic()
        call, _ = self.call_budget.reserve(self._identity, json.dumps(body, ensure_ascii=False),
                                           caller_tag, self.timeout)
        outcome, response = "failed", None
        try:
            if json_mode:
                value = dict(algorithm='elementwise' if self.kernel_type == 'relu' else 'matmul',
                             tiling=dict(block_m=16, block_n=16, block_k=16),
                             memory_plan={}, pipeline=[], optimization_hints=[])
                response = json.dumps(value)
            else:
                self._code_calls += 1
                params, expression = FIXTURES[self.kernel_type]
                # A changed implementation lets the actual repair gate verify
                # progress; identical/comment-only source still stops normally.
                returned = ("result" if self._code_calls == 1 else
                            "np.add(result, np.zeros((), dtype=result.dtype))")
                response = ("import numpy as np\n\ndef kernel(" + params + "):\n"
                            "    result = " + expression + "\n    return " + returned + "\n")
            self.cost_tracker.record(model=self.model, prompt_tokens=0, completion_tokens=0,
                usage_source="test_fixture", caller_tag=caller_tag,
                temperature=body["temperature"], max_tokens=body["max_tokens"],
                enable_thinking=False)
            outcome = "answered"
            return json.loads(response) if json_mode else response
        finally:
            # Do not mark_dispatched: no provider transport handoff occurred.
            self.call_budget.finish(call, response=response, outcome=outcome,
                wall_seconds=time.monotonic()-started, usage_source="test_fixture")

    def chat(self, messages, *, caller_tag="Unknown", temperature=None, max_tokens=None, **kwargs):
        return self._respond(messages, caller_tag=caller_tag, temperature=temperature,
                             max_tokens=max_tokens)

    def chat_json(self, messages, *, caller_tag="Unknown", temperature=None, max_tokens=None, **kwargs):
        return self._respond(messages, caller_tag=caller_tag, temperature=temperature,
                             max_tokens=max_tokens, json_mode=True)
