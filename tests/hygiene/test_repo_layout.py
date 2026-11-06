"""仓库布局守卫：根目录只放该放的，平铺目录不许无限长，缓存与产物不许入库。

为什么需要这一条：本仓 2.0 GB 的工作树里，1.0 GB 是 `output/`（审计证据）、
905 MB 是 `triton/`（本地检出），两者都在 `.gitignore` 里 —— 也就是说**没有任何
一次提交会提醒你**它们正在和源码同级膨胀，而 `git status` 永远是干净的。2026-10-30
的处置是把证据目录搬出仓库、仓内只留符号链接；这里把那件事的**结果**钉成检查，
顺便给三个平铺目录（`operator_agent/`、`benchmarks/`、`tests/`）设条目预算：
新增文件必须显式抬预算，而抬预算是一次可评审的 diff。
"""
import json
import os
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: 仓库根允许出现的东西。加新条目要连同这份清单一起改 —— 这正是目的：
#: 根目录是人的第一屏，它长出新东西应该是一次有意识的选择，不是 `_tmp.py` 随手一放。
ALLOWED_ROOT = {
    # 版本控制与宿主
    ".git", ".gitignore", ".github",
    # 构建与元数据
    "README.md", "LICENSE", "NOTICE", "MANIFEST.in", "THIRD_PARTY_NOTICES.md",
    "pyproject.toml", ".python-version", "config.example.yaml",
    # 源码与文档
    "operator_agent", "remote_scripts", "benchmarks", "tests", "docs",
    "integrations", "operator-dev-agent-design",
    # 真机 driver（Phase 3 计划搬进 benchmarks/drivers/，届时从这里删）
    "test_nvidia_matmul.py", "test_nvidia_layernorm.py", "test_nvidia_attention.py",
    # 被 .gitignore 覆盖的运行期/本地目录：允许存在，但不许入库
    "output", "logs", "debug", "triton", "config.yaml", "__pycache__",
    ".pytest_cache", ".uploads", ".trae-html-share-packages", ".DS_Store",
    ".venv", ".workbuddy",
}

#: **单层**条目预算：一个目录里直接放多少个 tracked 文件就到头了。
#: 2026-10-31 把 tests/（264 个平铺）与 benchmarks/（142 个平铺）按用途分了目录，
#: 这条守卫的职责从「总量别涨」变成「别再摊回一层」。
#: `operator_agent/` 还是 99 个平铺模块 —— 目标是 60 以内，分包的方案在
#: docs/design/repo-slimming-20261031.md 里，尚未执行。
DIRECTORY_BUDGETS = {
    "tests": 6,                  # 只留 conftest/fixtures/golden 与三个子目录
    "tests/unit": 220,
    "tests/acceptance": 60,
    "tests/hygiene": 10,
    "benchmarks": 10,            # __init__ + 私有辅助 + 子目录
    "benchmarks/experiments/s5": 70,
    "benchmarks/drivers": 25,
    "operator_agent": 4,         # 只剩 __init__.py + ops/ prompts/ skills/ 与五个层目录
    "operator_agent/core": 30,
    "operator_agent/decide": 35,
    "operator_agent/execute": 20,
    "operator_agent/generate": 26,
    "operator_agent/app": 12,
    "docs/design": 62,
    "docs/research": 75,
    "docs/status": 51,            # v0.7.2：预算、生命周期、自动入口、封版与待办记录
    "docs/phase1": 8,               # 一期文档：清单 + 两份记录 + 目录索引
}

#: 绝不该出现在 `git ls-files` 里的形态：缓存、构建产物、被搬走的证据目录。
FORBIDDEN_TRACKED = (
    re.compile(r"(^|/)__pycache__/"),
    re.compile(r"(^|/)\.pytest_cache/"),
    re.compile(r"(^|/)\.trae-html-share-packages/"),
    re.compile(r"\.pyc$"),
    re.compile(r"(^|/)\.DS_Store$"),
    re.compile(r"(^|/)dist/"),
    re.compile(r"(^|/)output/"),
    re.compile(r"(^|/)logs/"),
    re.compile(r"(^|/)triton/"),
    re.compile(r"(^|/)node_modules/"),
)


def tracked_files():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=str(ROOT),
                         capture_output=True, text=True, check=True).stdout
    return [name for name in out.split("\0") if name]


class TestRepoLayout(unittest.TestCase):

    def test_the_root_only_holds_known_entries(self):
        unknown = sorted(set(os.listdir(ROOT)) - ALLOWED_ROOT)
        self.assertEqual(unknown, [], (
            "根目录多了清单外的条目：%s\n"
            "要么它该进某个子目录，要么把它加进 ALLOWED_ROOT 并说明为什么该放根目录"
            % unknown))

    def test_cache_build_and_evidence_paths_are_never_tracked(self):
        hits = [name for name in tracked_files()
                if any(pattern.search(name) for pattern in FORBIDDEN_TRACKED)]
        self.assertEqual(hits, [], "这些路径不该被版本控制：%s" % hits[:10])

    def test_evidence_stays_out_of_the_worktree(self):
        """`output/` 只能是符号链接，或者干脆不存在。

        搬走的理由不是「占地方不好看」：1.0 GB 的证据与源码同级会让每次
        `git status`/`rg`/`find` 都扫过它，而这些证据本身需要的是**离线备份**，
        不是版本控制。搬走后仓内留链接，是为了让 94 处 `../../output/...` 文档链接
        和四个按绝对路径读证据的测试继续可用。
        """
        target = ROOT / "output"
        if not target.exists():
            return                      # fresh clone：没有证据目录是正常状态
        self.assertTrue(target.is_symlink(), (
            "output/ 又是实体目录了：证据应放在仓库外并在这里留符号链接；"
            "若确实要临时重跑实验，先迁移再跑，别把 1.0 GB 重新长回工作树"))

    def test_no_directory_piles_files_back_into_one_layer(self):
        """按**直接子文件数**算，不按递归总数 —— 摊回一层正是这次要防的退化。"""
        files = tracked_files()
        direct = {}
        for name in files:
            directory = name.rsplit("/", 1)[0] if "/" in name else ""
            direct[directory] = direct.get(directory, 0) + 1
        over = ["%s: %d > %d" % (d, direct.get(d, 0), budget)
                for d, budget in DIRECTORY_BUDGETS.items() if direct.get(d, 0) > budget]
        self.assertEqual(over, [], (
            "这些目录又摊平了：%s\n正确反应是分组，而不是把预算调大" % over))

    def test_the_evidence_pointer_is_documented(self):
        """搬走之后必须留下「证据在哪、怎么重建」的说明，否则下一个人找不到。"""
        note = ROOT / "docs" / "research" / "evidence-location.json"
        self.assertTrue(note.is_file(), "缺少证据位置说明：%s" % note)
        data = json.loads(note.read_text(encoding="utf-8"))
        self.assertIn("path", data)
        self.assertIn("why", data)
        self.assertTrue(data["path"].startswith("~"), "证据路径请写成家目录相对形式，别写死本机绝对路径")


#: 层间依赖方向（2026-10-31 分包时定）：`app → generate → decide → execute → core`。
#: 这条表的用意不是「已经完美」——**当前有 23 处向上依赖**，几乎全是函数内的延迟导入
#: （`from ..app.autonomous import ...` 是为了避免循环导入）或类型注解。把它做成
#: **棘轮**：新增一处就红，拆掉一处也要来改表，于是「又往上依赖了」不会静默发生。
LAYER_ORDER = ("core", "decide", "execute", "generate", "app")
UPWARD_IMPORTS = {
    "core/candidates.py -> decide",
    "core/confirm.py -> decide",
    "core/contract.py -> decide",
    "core/corpus_admission.py -> decide",
    "core/environment.py -> decide",
    "core/execution_audit.py -> decide",
    "core/execution_audit.py -> execute",
    "core/execution_audit.py -> generate",
    "core/kernel_db.py -> app",
    "core/run_contract.py -> decide",
    "core/run_contract.py -> generate",
    "core/schemas.py -> decide",
    "core/workload.py -> decide",
    "decide/oracle.py -> execute",
    "decide/quality_evaluation.py -> app",
    "decide/quality_evaluation.py -> execute",
    "decide/quality_evaluation.py -> generate",
    "decide/quality_pairing.py -> execute",
    "decide/quality_portfolio.py -> generate",
    "decide/quality_preflight.py -> execute",
    "decide/verifier.py -> execute",
    "execute/task_cancellation.py -> app",
    "generate/agent_logger.py -> app",
}


def _upward_imports():
    import ast
    rank = {layer: index for index, layer in enumerate(LAYER_ORDER)}
    found = set()
    for layer in LAYER_ORDER:
        for path in sorted((ROOT / "operator_agent" / layer).glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                targets = []
                if isinstance(node, ast.ImportFrom):
                    if node.level and node.module:
                        targets.append(node.module.split(".")[0])
                    elif node.module and node.module.startswith("operator_agent."):
                        parts = node.module.split(".")
                        if len(parts) > 2:
                            targets.append(parts[2])
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("operator_agent."):
                            parts = alias.name.split(".")
                            if len(parts) > 2:
                                targets.append(parts[2])
                for target in targets:
                    if target in rank and rank[target] > rank[layer]:
                        found.add("%s/%s -> %s" % (layer, path.name, target))
    return found


class TestLayerDirection(unittest.TestCase):
    """`app → generate → decide → execute → core`：向上依赖只能减少，不能静默增加。"""

    def test_upward_imports_do_not_grow(self):
        found = _upward_imports()
        new = sorted(found - UPWARD_IMPORTS)
        self.assertEqual([], new, (
            "新增了向上依赖：%s\n"
            "层的方向是 app → generate → decide → execute → core；"
            "要用上层的函数请反转依赖（回调/参数注入），或把这条登记进 UPWARD_IMPORTS 并写明理由"
            % new))

    def test_the_ratchet_table_has_no_stale_rows(self):
        """拆掉依赖也要来改表 —— 否则这张表会变成一份越来越假的历史。"""
        stale = sorted(UPWARD_IMPORTS - _upward_imports())
        self.assertEqual([], stale, "这些行已经不再是向上依赖了，请从 UPWARD_IMPORTS 删掉: %s" % stale)


if __name__ == "__main__":
    unittest.main()
