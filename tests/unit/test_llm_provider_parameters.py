"""Synthetic HTTP responses verify provider parameters without network or GPU work."""

import hashlib
import json
from unittest.mock import Mock

import pytest

from operator_agent.execute import llm_client as lc
from operator_agent.execute.llm_budget import LLMCallBudget, LLMCallLimits, LLMCallRefused
from operator_agent.execute.llm_client import LLMClient


BASE_URL = "https://provider.invalid/v1"
MODEL = "synthetic-provider-model"
MESSAGES = [{"role": "user", "content": "Return a small JSON object."}]


@pytest.fixture(autouse=True)
def isolated_provider_environment(monkeypatch):
    for key in ("LLM_API_KEY", "LLM_API_KEY_FILE", "LLM_THINKING_PARAMETER",
                "LLM_FALLBACK_MODEL", "LLM_FALLBACK_BASE_URL", "HY_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(lc.urllib.request, "urlopen", Mock(
        side_effect=AssertionError("provider parameter tests must not access the network")))
    monkeypatch.setattr(lc, "bounded_dispatch", Mock(
        side_effect=AssertionError("provider parameter tests must not dispatch a model")))


class Response:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps({
            "model": MODEL,
            "choices": [{"message": {"content": '{"ok":true}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7,
                      "prompt_tokens_details": {"cached_tokens": 3}},
        }).encode("utf-8")


def client(**options):
    return LLMClient(base_url=BASE_URL, model=MODEL,
                     api_key="synthetic-provider-credential", **options)


@pytest.mark.parametrize("parameter", ["chat_template_kwargs", "enable_thinking"])
@pytest.mark.parametrize("thinking", [False, True])
def test_provider_body_and_usage_keep_the_same_thinking_semantics(monkeypatch, parameter, thinking):
    sent = []

    def capture(request, timeout):
        assert request.full_url == BASE_URL + "/chat/completions"
        assert request.get_method() == "POST"
        assert timeout == 120
        sent.append(json.loads(request.data))
        return Response()

    monkeypatch.setattr(lc.urllib.request, "urlopen", capture)
    selected = client(thinking_parameter=parameter, enable_thinking=thinking)
    assert selected.chat_json(MESSAGES, temperature=0.1, max_tokens=64,
                              caller_tag="Fixer") == {"ok": True}

    expected = {"model": MODEL, "messages": MESSAGES, "temperature": 0.1,
                "max_tokens": 64, "stream": False}
    if parameter == "chat_template_kwargs":
        expected["chat_template_kwargs"] = {"enable_thinking": thinking}
    else:
        expected["enable_thinking"] = thinking
    assert sent == [expected]

    assert selected.cost_tracker.total_calls == 1
    recorded = selected.cost_tracker.records[0]
    assert recorded.enable_thinking is thinking
    assert recorded.temperature == 0.1
    assert recorded.max_tokens == 64
    assert recorded.model == recorded.reported_model == MODEL
    assert recorded.caller_tag == "Fixer"
    assert recorded.prompt_tokens == 11
    assert recorded.completion_tokens == 7
    assert recorded.total_tokens == 18
    assert recorded.cached_prompt_tokens == 3
    assert recorded.usage_source == "api_usage"


def test_unconfigured_parameter_preserves_the_legacy_body(monkeypatch):
    captured = []

    def capture(request, timeout):
        captured.append(json.loads(request.data))
        return Response()

    monkeypatch.setattr(lc.urllib.request, "urlopen", capture)
    selected = client()
    assert selected.chat(MESSAGES) == '{"ok":true}'
    assert selected.thinking_parameter == "chat_template_kwargs"
    assert captured == [{"model": MODEL, "messages": MESSAGES, "temperature": 0.3,
                         "max_tokens": 4096, "stream": False,
                         "chat_template_kwargs": {"enable_thinking": False}}]


@pytest.mark.parametrize("parameter", ["chat_template_kwargs", "enable_thinking"])
def test_environment_selects_the_provider_parameter(monkeypatch, parameter):
    monkeypatch.setenv("LLM_THINKING_PARAMETER", parameter)
    assert client().thinking_parameter == parameter


@pytest.mark.parametrize("environment,explicit", [
    ("enable_thinking", "chat_template_kwargs"),
    ("chat_template_kwargs", "enable_thinking"),
    ("invalid-environment-value", "chat_template_kwargs"),
])
def test_explicit_parameter_takes_precedence_over_environment(monkeypatch, environment, explicit):
    monkeypatch.setenv("LLM_THINKING_PARAMETER", environment)
    assert client(thinking_parameter=explicit).thinking_parameter == explicit


@pytest.mark.parametrize("value", ["", "thinking", "Enable_Thinking"])
@pytest.mark.parametrize("source", ["environment", "explicit"])
def test_invalid_parameter_is_rejected_without_dispatch(monkeypatch, value, source):
    options = {}
    if source == "environment":
        monkeypatch.setenv("LLM_THINKING_PARAMETER", value)
    else:
        monkeypatch.setenv("LLM_THINKING_PARAMETER", "enable_thinking")
        options["thinking_parameter"] = value
    with pytest.raises(ValueError, match="LLM_THINKING_PARAMETER"):
        client(**options)
    lc.urllib.request.urlopen.assert_not_called()
    lc.bounded_dispatch.assert_not_called()


def bounded_client(root, **options):
    budget = LLMCallBudget(LLMCallLimits(
        max_calls=1, max_output_tokens=32, total_output_token_reservations=32,
        allowed_callers=("Coder",)), root)
    return client(max_tokens=32, call_budget=budget, **options)


def test_bounded_identity_keeps_default_compatible_and_binds_the_nondefault_parameter(tmp_path):
    legacy = bounded_client(tmp_path / "legacy")
    top_level = bounded_client(tmp_path / "top-level", thinking_parameter="enable_thinking")
    expected = {"endpoint_sha256": hashlib.sha256(BASE_URL.encode()).hexdigest(),
                "model": MODEL, "credential_present": True, "enable_thinking": False,
                "timeout": 120, "default_temperature": 0.3, "default_max_tokens": 32}
    assert legacy.call_budget.summary()["identity"] == expected
    assert top_level.call_budget.summary()["identity"] == dict(
        expected, thinking_parameter="enable_thinking")
    for selected in (legacy, top_level):
        persisted = json.loads((selected.call_budget.root / "identity.json").read_text())
        assert persisted == selected.call_budget.summary()["identity"]


@pytest.mark.parametrize("before,after", [
    ("chat_template_kwargs", "enable_thinking"),
    ("enable_thinking", "chat_template_kwargs"),
])
def test_bounded_parameter_drift_is_refused_before_reservation_or_dispatch(tmp_path, before, after):
    selected = bounded_client(tmp_path / "calls", thinking_parameter=before)
    selected.thinking_parameter = after
    with pytest.raises(LLMCallRefused, match="drift"):
        selected.chat(MESSAGES, caller_tag="Coder")
    assert selected.call_budget.summary()["reserved_calls"] == 0
    lc.urllib.request.urlopen.assert_not_called()
    lc.bounded_dispatch.assert_not_called()
