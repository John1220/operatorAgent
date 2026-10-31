"""Fixer receipts describe proposals and never replace independent verification.

All replies are local test doubles. No model, candidate or GPU is executed.
"""
import copy
import json

import numpy as np
import pytest

from operator_agent.generate.agent_runtime import ActionEnvelope, ENVELOPE_ABSENT
from operator_agent.generate.fixer import FixOutcome, FixerAgent
from operator_agent.generate.generation_spec import GENERATION_SPEC
from operator_agent.core.run_contract import code_hash, digest
from operator_agent.core.schemas import (ConstraintSpec, Precision, SourceSpec,
                                   TargetSpec, TaskSchema, VerificationLevel,
                                   VerificationResult)


ORIGINAL = "def kernel(X):\n    return X - 1"
FIXED = "def kernel(X):\n    return X + 1"
INVALID = "def kernel(X):\n    return X +"


class Replies:
    model = "local-receipt-fixture"
    api_key = "secret-test-key"
    base_url = "https://user:private-endpoint@model.invalid/v1"

    def __init__(self, replies):
        self.replies = iter(replies)
        self.sent = []

    def chat(self, messages, **kwargs):
        self.sent.append((copy.deepcopy(messages), copy.deepcopy(kwargs)))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply


def task(dsl="python"):
    return TaskSchema(task_id="fixer-records", source=SourceSpec(kernel_type="relu"),
        target=TargetSpec(hardware="mock_gpu", dsl=dsl),
        constraints=ConstraintSpec(precision=Precision.FP32, shapes=((2, 3),)))


def failures():
    return [VerificationResult(VerificationLevel.L2_FUNCTION, False,
        message="数值误差超出容差", metrics={"bad_elements": 6})]


def reply(code):
    return "```python\n" + code + "\n```"


def assert_proposal_only(record):
    for name in ("candidate_execution", "correctness_proven", "restored_eligibility"):
        assert record[name] is False
    assert record["gpu_measurements"] == 0
    assert "passed" not in record and "eligibility" not in record


def test_exact_source_parent_sent_request_and_input_metadata_are_bound():
    model = Replies([reply(FIXED)])
    fixer = FixerAgent(llm=model)
    inputs = {"X": np.arange(12, dtype=np.float32).reshape(2, 6)[:, ::2]}
    outcome = fixer.run_with_action(ORIGINAL, task(), failures(), test_inputs=inputs)
    assert outcome.code == FIXED and len(model.sent) == 1
    assert len(outcome.candidate_records) == len(fixer.candidate_records) == 1
    record = outcome.candidate_records[0]
    assert record == fixer.last_candidate_record
    assert record["source"] == FIXED and record["source_hash"] == code_hash(FIXED)
    assert record["parent_source_hash"] == code_hash(ORIGINAL)
    assert record["producer"] == "Fixer" and record["strategy"] == "repair"
    assert record["status"] == "prechecked" and record["static_precheck"] == "passed"
    assert record["attempt"] == 1
    context = record["context"]
    assert context["task"]["dsl"] == "python"
    assert context["input_metadata"] == [{"X": {"shape": [2, 3],
        "dtype": "float32", "strides": [24, 8]}}]
    assert context["knowledge"]["skill"] is None
    messages, options = model.sent[0]
    assert record["request"]["messages_hash"] == digest(messages)
    assert record["request"]["prompt_hashes"] == [code_hash(messages[0]["content"])]
    assert options == dict(temperature=0.1, max_tokens=4096, caller_tag="Fixer")
    assert messages[0]["content"].count(GENERATION_SPEC) == 1
    assert "> SKILL SPECIFICATION (CODER role)" not in messages[0]["content"]
    serialized = json.dumps(record)
    assert model.api_key not in serialized and model.base_url not in serialized
    assert "values" not in context["input_metadata"][0]["X"]
    assert_proposal_only(record)


def test_triton_repair_receipt_tracks_existing_skill_without_extra_calls():
    source = "import triton\n@triton.jit\ndef helper(X):\n    pass\ndef kernel(X):\n    return X"
    model = Replies([reply(source)])
    result = FixerAgent(llm=model).run_with_action(ORIGINAL, task("triton"), failures())
    assert len(model.sent) == 1 and result.code == source
    record = result.candidate_records[0]
    assert record["context"]["knowledge"]["skill"]["name"] == "triton_skill"
    assert record["context"]["knowledge"]["skill"]["role"] == "coder"
    assert record["context"]["input_metadata"] is None
    assert "SKILL SPECIFICATION (CODER role)" in model.sent[0][0][0]["content"]
    assert_proposal_only(record)


def test_retry_records_keep_failed_attempt_and_do_not_change_existing_limit():
    model = Replies([reply(INVALID), reply(FIXED)])
    fixer = FixerAgent(llm=model)
    result = fixer.run_with_action(ORIGINAL, task(), failures())
    assert result.code == FIXED and len(model.sent) == 2
    assert [r["attempt"] for r in result.candidate_records] == [1, 2]
    assert [r["status"] for r in result.candidate_records] == ["precheck_failed", "prechecked"]
    assert [r["source"] for r in result.candidate_records] == [INVALID, FIXED]
    assert all(r["parent_source_hash"] == code_hash(ORIGINAL) for r in result.candidate_records)
    assert "上次修复未通过预检" not in model.sent[0][0][0]["content"]
    assert "上次修复未通过预检" in model.sent[1][0][0]["content"]
    assert fixer.MAX_FIX_ATTEMPTS == 3
    for record, (messages, _) in zip(result.candidate_records, model.sent):
        assert record["request"]["messages_hash"] == digest(messages)
        assert_proposal_only(record)


def test_failed_precheck_exhaustion_still_returns_the_same_last_source():
    model = Replies([reply(INVALID)] * 3)
    result = FixerAgent(llm=model).run_with_action(ORIGINAL, task(), failures())
    assert result.code == INVALID and len(model.sent) == 3
    assert len(result.candidate_records) == 3
    assert all(r["status"] == "precheck_failed" for r in result.candidate_records)
    assert all(r["static_precheck"] == "failed" for r in result.candidate_records)


@pytest.mark.parametrize("with_source", [False, True])
def test_ask_only_reply_never_credits_a_candidate_even_when_code_was_in_reply(with_source):
    raw = '```agent-action\n{"kind":"explain_failure","rationale":"Need clarification"}\n```'
    if with_source:
        raw += "\n" + reply(FIXED)
    model = Replies([raw])
    result = FixerAgent(llm=model).run_with_action(ORIGINAL, task(), failures())
    assert result.asked and result.code == "" and len(model.sent) == 1
    record = result.candidate_records[0]
    assert record["status"] == "no_candidate"
    assert record["source"] is record["source_hash"] is None
    assert record["static_precheck"] == "not_checked"
    assert record["reason"] == "ask_without_patch"
    assert_proposal_only(record)


def test_model_exception_is_recorded_and_reraised_without_implicit_retry():
    error = TimeoutError("private-server-detail")
    model = Replies([error])
    fixer = FixerAgent(llm=model)
    with pytest.raises(TimeoutError) as caught:
        fixer.run_with_action(ORIGINAL, task(), failures())
    assert caught.value is error and len(model.sent) == 1
    record = fixer.last_candidate_record
    assert record["status"] == "model_failed" and record["reason"] == "TimeoutError"
    assert record["source"] is record["source_hash"] is None
    assert "private-server-detail" not in json.dumps(record)
    assert_proposal_only(record)


def test_missing_code_attempts_remain_absent_and_are_all_retained():
    model = Replies(["No candidate available."] * 3)
    fixer = FixerAgent(llm=model)
    with pytest.raises(ValueError, match="未能从 LLM 回复中提取有效代码"):
        fixer.run_with_action(ORIGINAL, task(), failures())
    assert len(model.sent) == len(fixer.candidate_records) == 3
    assert all(r["status"] == "no_candidate" and r["source_hash"] is None
               for r in fixer.candidate_records)


def test_precheck_exception_retains_the_returned_source_without_retry(monkeypatch):
    model = Replies([reply(FIXED)])
    fixer = FixerAgent(llm=model)
    error = RuntimeError("private-validator-detail")
    def refused(*args):
        raise error
    monkeypatch.setattr(fixer, "_validate_code", refused)
    with pytest.raises(RuntimeError) as caught:
        fixer.run_with_action(ORIGINAL, task(), failures())
    assert caught.value is error and len(model.sent) == 1
    record = fixer.last_candidate_record
    assert record["source"] == FIXED and record["status"] == "precheck_failed"
    assert record["reason"] == "RuntimeError"
    assert "private-validator-detail" not in json.dumps(record)
    assert_proposal_only(record)


def test_records_reset_per_call_and_returned_outcome_keeps_its_original_receipts():
    model = Replies([reply(FIXED), reply(ORIGINAL)])
    fixer = FixerAgent(llm=model)
    first = fixer.run_with_action(ORIGINAL, task(), failures())
    frozen = copy.deepcopy(first.candidate_records)
    second = fixer.run_with_action(FIXED, task(), failures())
    assert first.candidate_records == frozen
    assert first.candidate_records[0]["source"] == FIXED
    assert second.candidate_records[0]["source"] == ORIGINAL
    assert len(fixer.candidate_records) == 1 and fixer.last_candidate_record["attempt"] == 1
    fixer.candidate_records[0]["status"] = "changed-by-observer"
    assert second.candidate_records[0]["status"] == "prechecked"


def test_legacy_tuple_and_three_argument_outcome_stay_compatible():
    model = Replies([reply(FIXED)])
    fixer = FixerAgent(llm=model)
    code, summary = fixer.run(ORIGINAL, task(), failures())
    assert code == FIXED and isinstance(summary, str) and len(model.sent) == 1
    legacy = FixOutcome("", "legacy", ActionEnvelope(status=ENVELOPE_ABSENT))
    assert legacy.candidate_records == ()
    assert set(legacy.as_dict()) == {"code_chars", "summary", "asked", "envelope"}
    current = FixOutcome(FIXED, summary, ActionEnvelope(status=ENVELOPE_ABSENT),
                         tuple(fixer.candidate_records))
    compact = current.as_dict()["candidate_records"][0]
    assert "source" not in compact and "context" not in compact
    assert compact["source_hash"] == code_hash(FIXED)


def test_no_model_configuration_clears_previous_call_receipts():
    fixer = FixerAgent(llm=Replies([reply(FIXED)]))
    fixer.run_with_action(ORIGINAL, task(), failures())
    fixer.llm = None
    with pytest.raises(RuntimeError, match="LLM 未配置"):
        fixer.run_with_action(ORIGINAL, task(), failures())
    assert fixer.candidate_records == [] and fixer.last_candidate_record is None
