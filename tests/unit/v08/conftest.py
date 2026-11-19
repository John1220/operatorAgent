"""不让 A/C 驱动的全局补丁泄漏到别的测试。

`ac_live._widen_stream_deadline()` 会把 `host.upstream_deadline_s` **替换成模块级函数**
（线上确实需要：流式截止要随输出上限缩放）。但 pytest 在同一个进程里跑所有测试，
于是 v08 的测试跑完之后，`test_study_sampling` 里“4096 应保持 120 s”的断言会看到
被换掉的实现而失败（2026-11-19 实测：单独跑过、合并跑挂）。

这里在**每个 v08 测试之后**把那个函数还原成原始实现，恢复测试间的隔离。
"""
import pytest


@pytest.fixture(autouse=True)
def _restore_upstream_deadline():
    from benchmarks.experiments.phase1 import host
    original = host.upstream_deadline_s
    try:
        yield
    finally:
        host.upstream_deadline_s = original
