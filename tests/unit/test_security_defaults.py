"""Wave 0 security-default guards.

The test suite enables process-local harness controls in conftest so it can keep
its fixtures hermetic.  These tests temporarily remove that control and prove
that ordinary environment values cannot disable production safeguards.
"""

from pathlib import Path

import pytest

from operator_agent.generate import agent_logger
from operator_agent.app import _harness_controls
from operator_agent.generate.agent_logger import capture_execution
from operator_agent.app.conductor import Conductor, MAX_HEARTBEAT_INTERVAL
from operator_agent.core.kernel_db import FailureDatabase, KernelDatabase


def test_kernel_db_off_env_cannot_disable_persistence(monkeypatch):
    monkeypatch.setattr(_harness_controls, "_ENABLED", False)
    monkeypatch.setenv("OPERATOR_AGENT_KERNEL_DB", "off")
    path = KernelDatabase._resolve_jsonl(None)
    assert path is not None
    assert path.name == "kernel_db.jsonl"


def test_failure_db_off_env_cannot_disable_persistence(monkeypatch):
    monkeypatch.setattr(_harness_controls, "_ENABLED", False)
    monkeypatch.setenv("OPERATOR_AGENT_FAILURE_DB", "off")
    path = FailureDatabase._resolve_jsonl(None)
    assert path is not None
    assert path.name == "failure_db.jsonl"


def test_failure_gate_env_cannot_lower_recurrence_threshold(monkeypatch):
    monkeypatch.setattr(_harness_controls, "_ENABLED", False)
    monkeypatch.setenv("OPERATOR_AGENT_FAILURE_GATE", "0")
    assert FailureDatabase._resolve_gate(None) == 3


def test_log_dir_off_env_cannot_disable_audit(monkeypatch, tmp_path):
    monkeypatch.setattr(_harness_controls, "_ENABLED", False)
    monkeypatch.setenv("OPERATOR_AGENT_LOG_DIR", "off")
    # Redirect the module's default project root into tmp_path so this assertion
    # never touches the repository logs directory.
    fake_file = tmp_path / "operator_agent" / "generate" / "agent_logger.py"
    monkeypatch.setattr(agent_logger, "__file__", str(fake_file))
    with capture_execution(task_id="security-default") as log_path:
        print("audit remains enabled")
    assert log_path is not None
    assert Path(log_path).exists()
    assert Path(log_path).parent == tmp_path / "logs"


def test_oversized_heartbeat_is_rejected_outside_harness(monkeypatch):
    monkeypatch.setattr(_harness_controls, "_ENABLED", False)
    with pytest.raises(ValueError, match="heartbeat_interval 不能超过"):
        Conductor(heartbeat_interval=MAX_HEARTBEAT_INTERVAL + 1)
