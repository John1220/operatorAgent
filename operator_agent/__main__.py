"""`python -m operator_agent` → 与 `python -m operator_agent.cli` 同一个入口。"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
