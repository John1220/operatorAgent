"""验收组整体标记 slow。

这里的用例会起真实的 DSH / MCP / 沙箱 / SSH 子进程（部分需要特定宿主安装），
单个文件几十秒, 整组约 200 s。日常开发循环用 `-m "not slow"` 跳过;
封版前用 `-m slow` 单独跑，并保留完整日志。

用目录级 hook 而不是逐个文件加 pytestmark: 验收文件很多, 逐个标记容易漏,
而且新增文件默认就该归入这一组。

注意: `pytest_collection_modifyitems` 是全局 hook, 子目录 conftest 里定义它也会收到
**全部** item，所以这里必须自己按路径过滤，否则整个测试集都会被标成 slow。
"""
from pathlib import Path

import pytest

ACCEPTANCE = Path(__file__).resolve().parent


def pytest_collection_modifyitems(config, items):
    for item in items:
        if Path(str(item.path)).resolve().is_relative_to(ACCEPTANCE):
            item.add_marker(pytest.mark.slow)