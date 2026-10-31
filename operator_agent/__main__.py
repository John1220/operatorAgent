"""`python -m operator_agent` —— 委托给 app 层的 CLI。

2026-10-31: 包分五层后真正的入口在 `operator_agent/app/`，但 `python -m operator_agent`
是**公开**调用形态（README / user_guide / 插件测试都在用），所以包根留这一个委托文件。
它不是第二份实现：只 import 并调用 app.cli 的 main。
"""
import sys

from .app.cli import main

if __name__ == "__main__":
    sys.exit(main())
