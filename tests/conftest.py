"""让单元测试不改动仓库状态。

两个曾经真实发生过的污染:

1. `Conductor.run()` 无条件走 `capture_execution()`, 而它会
   (a) 往 `<repo>/logs/` 写一个文件, (b) 顺带触发 `cleanup_old_logs(keep_count=10)`。
   于是**跑一次测试集 = 往 logs/ 灌十几个 mock 现场, 并把真实跑出来的现场轮掉**。
   这不是假想病: `docs/research/35` 第九节给 C1 重测方案挑的素材就是「logs/ 里真实
   修 3 轮仍失败的 case」, 而那些文件已经被测试批次删掉了 (2026-09-26 核实: 现存 11 个
   日志全部带 `LLM: <MagicMock ...>` 头)。

2. `ArtifactStore` 无条件写 `<repo>/output/`, `KernelDatabase` 现在也会追加 JSONL。
   同一型问题的另一面, 而且已经量到了后果 (2026-09-26 盘点): 仓库里 5326 份
   `execution_snapshot.json` 有 **5316 份**、2872 份 `result.json` 有 **2822 份** 是
   mock_gpu 跑出来的单元测试现场 —— 「历史语料」这个概念在仓库里原本不成立
   (docs/research/43 第三节)。

对策: 测试期内把目录指到 tmp (`OPERATOR_AGENT_LOG_DIR` / `OPERATOR_AGENT_OUTPUT_DIR`),
或干脆关掉 (`OPERATOR_AGENT_KERNEL_DB` / `OPERATOR_AGENT_FAILURE_DB` = off)。
真实运行不受影响 —— 不设变量时行为与从前一致。
"""
import os
import tempfile

import pytest

# 整个测试会话共享临时目录; 用 env 而不是 monkeypatch, 因为它们必须在任何 Conductor /
# ArtifactStore 被构造之前就生效。
_SESSION_LOG_DIR = tempfile.mkdtemp(prefix="operator_agent_tests_logs_")
_SESSION_OUTPUT_DIR = tempfile.mkdtemp(prefix="operator_agent_tests_output_")


def pytest_configure(config):
    os.environ.setdefault("OPERATOR_AGENT_LOG_DIR", _SESSION_LOG_DIR)
    # 长周期实验里 keep=10 会删掉要复查的现场; 测试会话里干脆放宽 (目录本就是 tmp)
    os.environ.setdefault("OPERATOR_AGENT_LOG_KEEP", "10000")
    os.environ.setdefault("OPERATOR_AGENT_OUTPUT_DIR", _SESSION_OUTPUT_DIR)
    # 经验库是「跨运行的记忆」: 单元测试既不该共享它, 也不该往仓库写 JSONL。
    # 真要测持久化的用例自己传 jsonl_path=<tmp>。
    os.environ.setdefault("OPERATOR_AGENT_KERNEL_DB", "off")
    # 负样本库同一条纪律 (doc 45)。它比正样本库更危险: 每条失败都写, 不做入库门,
    # 一次测试批次就能把语料灌满 mock 的假失败。
    os.environ.setdefault("OPERATOR_AGENT_FAILURE_DB", "off")


@pytest.fixture(scope="session", autouse=True)
def _assert_side_effects_are_redirectable():
    """守住这条机制本身: 变量若被改掉, 测试又会开始写仓库的 logs/ 与 output/。"""
    assert os.environ.get("OPERATOR_AGENT_LOG_DIR") == _SESSION_LOG_DIR, (
        "OPERATOR_AGENT_LOG_DIR 没生效 —— 单元测试会重新污染仓库的 logs/ 目录")
    assert os.environ.get("OPERATOR_AGENT_OUTPUT_DIR") == _SESSION_OUTPUT_DIR, (
        "OPERATOR_AGENT_OUTPUT_DIR 没生效 —— 单元测试会重新往仓库 output/ 灌 mock 现场")
    yield
