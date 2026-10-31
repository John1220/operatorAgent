"""Credential references resolve at client construction without model/GPU calls.

Use synthetic credentials only. The run/repair checks keep the real LLMClient
constructor and replace execution, so they also catch entry points bypassing
the shared file-reference resolution or publishing secrets into os.environ.
"""

import json
import os
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest

from operator_agent.app import autonomous, hitl_tools
from operator_agent.execute.llm_client import LLMClient


BASE_URL = "https://model.invalid/v1"
FILE_CREDENTIAL = "synthetic-file-credential"


@pytest.fixture(autouse=True)
def isolated_credentials(monkeypatch):
    for name in ("LLM_API_KEY", "LLM_API_KEY_FILE", "LLM_BASE_URL", "LLM_MODEL"):
        # Register restoration even when originally absent: autonomous.main
        # used to add LLM_API_KEY directly, bypassing monkeypatch.setenv.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name, raising=False)

    def no_request(*args, **kwargs):
        raise AssertionError("credential tests must not send model requests")

    monkeypatch.setattr(LLMClient, "chat", no_request)
    monkeypatch.setattr(LLMClient, "chat_json", no_request)


def credential_file(tmp_path, content=FILE_CREDENTIAL):
    path = tmp_path / "credential.txt"
    path.write_text(content, encoding="utf-8")
    return path


def configure_file(monkeypatch, path):
    monkeypatch.setenv("LLM_BASE_URL", BASE_URL)
    monkeypatch.setenv("LLM_MODEL", "synthetic-model")
    monkeypatch.setenv("LLM_API_KEY_FILE", str(path))


def assert_key_not_exported():
    # On failure pytest should render a boolean, never the full environment.
    exported = "LLM_API_KEY" in os.environ
    assert not exported


def test_explicit_key_wins_without_reading_environment_file(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_API_KEY", "synthetic-environment-credential")
    monkeypatch.setenv("LLM_API_KEY_FILE", str(tmp_path / "absent.txt"))
    client = LLMClient(base_url=BASE_URL, api_key="synthetic-explicit-credential")
    assert client.api_key == "synthetic-explicit-credential"


def test_environment_key_wins_without_reading_file(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_API_KEY", "synthetic-environment-credential")
    monkeypatch.setenv("LLM_API_KEY_FILE", str(tmp_path / "absent.txt"))
    client = LLMClient(base_url=BASE_URL)
    assert client.api_key == "synthetic-environment-credential"


def test_file_reference_expands_home_and_strips_whitespace(monkeypatch, tmp_path):
    credential_file(tmp_path, " \n\t" + FILE_CREDENTIAL + "\r\n ")
    # Patch the expansion rather than mutating the process's HOME.
    original_expanduser = Path.expanduser

    def expand_reference(path):
        if str(path) == "~/credential.txt":
            return tmp_path / "credential.txt"
        return original_expanduser(path)

    monkeypatch.setattr(Path, "expanduser", expand_reference)
    configure_file(monkeypatch, "~/credential.txt")
    client = LLMClient()
    assert client.api_key == FILE_CREDENTIAL
    assert_key_not_exported()


def test_empty_environment_key_falls_back_to_file(monkeypatch, tmp_path):
    path = credential_file(tmp_path)
    configure_file(monkeypatch, path)
    monkeypatch.setenv("LLM_API_KEY", "")
    client = LLMClient()
    assert client.api_key == FILE_CREDENTIAL
    assert os.environ["LLM_API_KEY"] == ""


def test_unconfigured_key_remains_available_for_unauthenticated_endpoints():
    assert LLMClient(base_url=BASE_URL).api_key == ""


@pytest.mark.parametrize("content", ["", " \t\r\n"])
def test_empty_file_is_a_clear_configuration_error(monkeypatch, tmp_path, content):
    configure_file(monkeypatch, credential_file(tmp_path, content))
    with pytest.raises(ValueError, match="LLM_API_KEY_FILE"):
        LLMClient()
    assert_key_not_exported()


def test_missing_file_is_a_clear_configuration_error(monkeypatch, tmp_path):
    configure_file(monkeypatch, tmp_path / "absent.txt")
    with pytest.raises(ValueError, match="LLM_API_KEY_FILE"):
        LLMClient()
    assert_key_not_exported()


def test_file_read_error_does_not_expose_underlying_error_content(monkeypatch, tmp_path):
    path = credential_file(tmp_path)
    configure_file(monkeypatch, path)
    original_read_text = Path.read_text

    def fail_read(selected, *args, **kwargs):
        if selected == path:
            raise OSError("synthetic error containing " + FILE_CREDENTIAL)
        return original_read_text(selected, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_read)
    with pytest.raises(ValueError) as error:
        LLMClient()
    assert "LLM_API_KEY_FILE" in str(error.value)
    assert FILE_CREDENTIAL not in str(error.value)
    rendered = "".join(traceback.format_exception(type(error.value), error.value,
                                                 error.value.__traceback__))
    assert FILE_CREDENTIAL not in rendered
    assert_key_not_exported()


def test_invalid_utf8_is_a_clear_configuration_error(monkeypatch, tmp_path):
    path = tmp_path / "credential.txt"
    path.write_bytes(b"\xff\xfe")
    configure_file(monkeypatch, path)
    with pytest.raises(ValueError, match="LLM_API_KEY_FILE"):
        LLMClient()


def test_client_representation_does_not_include_file_credential(monkeypatch, tmp_path):
    configure_file(monkeypatch, credential_file(tmp_path))
    client = LLMClient()
    assert client.api_key == FILE_CREDENTIAL
    assert FILE_CREDENTIAL not in repr(client)
    assert FILE_CREDENTIAL not in str(client)


def prepare_run(tmp_path):
    run = tmp_path / "run"
    hitl_tools.prepare({"request": "matmul fp32", "shapes": [[2, 2, 2]]}, run)
    return run


def test_autonomous_worker_constructs_client_from_file_without_exporting_key(
        monkeypatch, tmp_path):
    from operator_agent.app import conductor as conductor_module

    configure_file(monkeypatch, credential_file(tmp_path))
    run = prepare_run(tmp_path)
    (run / autonomous.CONFIG).write_text(json.dumps({"backend": "mock", "max_rounds": 1}),
                                        encoding="utf-8")
    for name in ("OPERATOR_AGENT_OUTPUT_DIR", "OPERATOR_AGENT_LOG_DIR",
                 "OPERATOR_AGENT_KERNEL_DB", "OPERATOR_AGENT_FAILURE_DB"):
        # main() changes these directly; register restoration with monkeypatch.
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    received = []

    class NoExecutionConductor:
        def __init__(self, **kwargs):
            from operator_agent.execute.budget import BudgetLedger
            self._budget = BudgetLedger(kwargs['budget'])
            received.append(kwargs["llm"].api_key)
            assert_key_not_exported()

        def execute(self, *args, **kwargs):
            return SimpleNamespace(all_passed=False, benchmark_data={}, verification_results=[],
                                   iteration_history=[])

    monkeypatch.setattr(conductor_module, "Conductor", NoExecutionConductor)
    monkeypatch.setattr(autonomous, "_backend", lambda *args: object())
    import time
    started = time.monotonic()
    assert autonomous._worker_main(run, started+30., started) == 0
    assert received == [FILE_CREDENTIAL]
    assert_key_not_exported()
    status = (run / autonomous.WORKER_RESULT).read_text(encoding="utf-8")
    summary = (run / autonomous.SUMMARY).read_text(encoding="utf-8")
    assert FILE_CREDENTIAL not in status + summary


def test_repair_constructs_client_from_file_without_model_or_gpu(
        monkeypatch, tmp_path):
    from operator_agent.generate import fixer as fixer_module, repair_loop

    configure_file(monkeypatch, credential_file(tmp_path))
    run = prepare_run(tmp_path)
    received = []
    source = "def kernel(A, B):\n    return A @ B\n"

    def capture_fixer(*, llm):
        received.append(llm.api_key)
        assert_key_not_exported()
        return SimpleNamespace(llm=llm)

    def no_execution_repair(code, *args, **kwargs):
        return {"passed": False, "rounds": [], "code": code, "log": ""}

    monkeypatch.setattr(hitl_tools, "_make_backend", lambda *args, **kwargs: (object(), {}))
    monkeypatch.setattr(hitl_tools, "Verifier", lambda backend: object())
    monkeypatch.setattr(fixer_module, "FixerAgent", capture_fixer)
    monkeypatch.setattr(repair_loop, "repair", no_execution_repair)
    result = hitl_tools.repair(run, source, backend="mock")
    assert result["ok"] is True
    assert received == [FILE_CREDENTIAL]
    assert_key_not_exported()
    assert FILE_CREDENTIAL not in json.dumps(result)
