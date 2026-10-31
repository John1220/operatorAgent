"""`python -m operator_agent.app` → 与 `python -m operator_agent` / `operator-agent` 同一入口。"""

import sys

from ..app.cli import main

if __name__ == "__main__":
    sys.exit(main())
