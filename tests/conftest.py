"""让单元测试不改动仓库状态。

`Conductor.run()` 无条件走 `capture_execution()`, 而它会
(a) 往 `<repo>/logs/` 写一个文件, (b) 顺带触发 `cleanup_old_logs(keep_count=10)`。
于是**跑一次测试集 = 往 logs/ 灌十几个 mock 现场, 并把真实跑出来的现场轮掉**。
这不是假想病: `docs/research/35` 第九节给 C1 重测方案挑的素材就是「logs/ 里真实
修 3 轮仍失败的 case」, 而那些文件已经被测试批次删掉了 (2026-09-26 核实: 现存 11 个
日志全部带 `LLM: <MagicMock ...>` 头)。

对策: 测试期内把日志目录指到 tmp, 或者干脆关掉。真实运行不受影响 —— 不设变量时
行为与从前一致。
"""
import os
import tempfile

import pytest

# 整个测试会话共享一个临时日志目录; 用 env 而不是 monkeypatch, 因为它必须在任何
# Conductor 被构造之前就生效。
_SESSION_LOG_DIR = tempfile.mkdtemp(prefix="operator_agent_tests_logs_")


def pytest_configure(config):
    os.environ.setdefault("OPERATOR_AGENT_LOG_DIR", _SESSION_LOG_DIR)
    # 长周期实验里 keep=10 会删掉要复查的现场; 测试会话里干脆放宽 (目录本就是 tmp)
    os.environ.setdefault("OPERATOR_AGENT_LOG_KEEP", "10000")


@pytest.fixture(scope="session", autouse=True)
def _assert_logging_is_redirectable():
    """守住这条机制本身: 变量若被改掉, 测试又会开始写仓库的 logs/。"""
    assert os.environ.get("OPERATOR_AGENT_LOG_DIR") == _SESSION_LOG_DIR, (
        "OPERATOR_AGENT_LOG_DIR 没生效 —— 单元测试会重新污染仓库的 logs/ 目录")
    yield
