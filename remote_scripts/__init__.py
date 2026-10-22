"""远端执行载荷（`run_kernel.py` / `runner_server.py` / …）。

这个 `__init__.py` 存在的唯一理由：让 setuptools 把 `remote_scripts/` 作为**顶层包**
装进 site-packages，从而保留 `operator_agent/../remote_scripts` 这个相对布局 ——
`remote_executor.deploy()` 与 `run_contract.protocol_identity()` 都是按「包目录的上一级」
找它们的。没有它，`pip install` 出来的包在 ssh 后端的第一次部署就会找不到 runner。
"""
