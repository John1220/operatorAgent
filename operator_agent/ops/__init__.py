"""外挂算子参考实现集合。

每个模块导出:
  - `REFERENCE_SOURCE`: 可随任务传输的 Python 源码, 契约见 `operator_agent.reference`
  - `reference(**inputs)`: 本地可直接调用的同一实现 (与源码等价)
  - `make_inputs(...)` / 契约常量: 造输入用
"""
