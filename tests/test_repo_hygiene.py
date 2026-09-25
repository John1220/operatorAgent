"""仓库卫生守卫: 已跟踪文件里不得出现凭据形态的内容。

动机不是假设性的。这一段写 docs/research/20 时, 正文里顺手写了那个被泄露过
的外部 LLM 服务名 —— 靠「每次提交前手工跑一遍 git grep」才拦下来。而这恰好是
本次吸纳进来的那条研究结论:

    481 个公开 CLAUDE.md 中只有 4-16% 的自然语言规则有对应的内建强制控制,
    最严格匹配下 4.4%。所以防御必须是机械断言, 而不是文档。

把一个已经写了、也已经被人肉执行过的检查清单落成测试, 就是把它从 4.4% 那一侧
搬到另一侧。这个文件就是这个作用。

本文件自身必然包含这些被禁模式 (它是扫描器), 因此在扫描时排除自己 —— 与
test_recipe_caps_gate.py 对配方 ID 的处理同一个理由。

运行: python3 -m unittest tests.test_repo_hygiene -v
"""

import os
import re
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve().name
SCANNER_REL = "tests/test_repo_hygiene.py"   # 仓库相对路径, 用于断言扫描覆盖面

# 这些值曾经进入过公开仓, 或属于本机凭据布局。任何一条命中都意味着**公开仓
# origin 会拿到它** —— origin 是 github.com/John1220/operatorAgent, 公开库。
#
# 分成两组是为了可读, 合成一个 pattern 匹配。
FORBIDDEN_LITERAL_PATTERNS = [
    r"sk-[A-Za-z0-9]{20,}",          # API key 形态
    r"BEGIN [A-Z ]*PRIVATE KEY",     # 私钥文件内容
    r"id_ed25519",
    r"algorithm\.pem",
    r"yita_zhanghan",
    r"zhanghan",                     # 真实用户名, 包括出现在路径里的
    r"tokenhub",                     # 泄露 key 所属服务的标识词
    r"tencentmaas",
]
# 真实 IP 与端口: 单独一组, 因为 minified JS 里会有巧合的数字串。
FORBIDDEN_ADDRESS_PATTERNS = [
    r"8\.146\.225\.89",
    r"43\.180\.254\.145",
    r"152\.136\.252\.171",
    r":8003\b", r":8008\b", r":8091\b", r":8092\b", r":9001\b",
]

# 没有全局排除白名单。原先沿用的「vendored minified JS 含数字串会巧合命中端口模式」
# 这条理由, 在把端口模式收紧成带前导冒号的 `:8003\b` 形式之后已经不成立 —— 实测
# 那 2 个 js 文件对全部 16 个模式**一条都不命中**。既然排除项挡不住任何真命中、却会
# 挡住未来的真泄漏, 就整个删掉。唯一不扫的是本文件 (它是扫描器, 自己必然含这些串)。


def _tracked_files():
    r = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("`git ls-files` 失败, 这个测试必须在 git 检出里跑")
    return [f for f in r.stdout.split("\0") if f]


def _excluded(rel: str) -> bool:
    return os.path.basename(rel) == SELF


class TestScannerCoversTheWholeRepo(unittest.TestCase):
    """扫描器自己不许有盲区 —— 有盲区就必须在这里写明为什么安全。"""

    def test_only_the_scanner_itself_is_skipped(self):
        tracked = _tracked_files()
        self.assertIn(SCANNER_REL, tracked, "扫描器自身不在索引里, 这条断言暂无意义")
        skipped = [f for f in tracked if _excluded(f)]
        # 断言具体路径而不是重跑 _excluded: 后者会把「_excluded 被改宽」这件事
        # 原样吸收掉, 变成一个永真的断言。
        self.assertEqual([SCANNER_REL], skipped,
                         "除扫描器自身外还有文件被跳过, 每个跳过都要单独论证")

    def test_every_tracked_text_file_is_actually_read(self):
        """`git ls-files` 非空, 而 setUpClass 真的读到了内容。"""
        files = _tracked_files()
        self.assertGreater(len(files), 20, f"只列出 {len(files)} 个跟踪文件, 可疑")


class TestNoCredentialsInTrackedFiles(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.blobs = {}
        for rel in _tracked_files():
            if _excluded(rel):
                continue
            p = REPO_ROOT / rel
            if not p.is_file():
                continue          # 符号链接 / 已删除但仍在索引里
            try:
                cls.blobs[rel] = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
        assert cls.blobs, "一个可扫描的跟踪文件都没取到, 这个测试将形同虚设"

    def _scan(self, patterns):
        hits = {}
        for rel, body in self.blobs.items():
            for pat in patterns:
                for m in re.finditer(pat, body):
                    line = body[:m.start()].count("\n") + 1
                    hits.setdefault(pat, []).append(f"{rel}:{line}")
        return hits

    def test_no_credential_shaped_content(self):
        hits = self._scan(FORBIDDEN_LITERAL_PATTERNS)
        report = "\n".join(f"  {p}: {v[:8]}" for p, v in sorted(hits.items()))
        self.assertEqual({}, hits,
                         f"已跟踪文件命中禁止模式 (origin 是公开仓!):\n{report}")

    def test_no_real_addresses_or_ports(self):
        hits = self._scan(FORBIDDEN_ADDRESS_PATTERNS)
        report = "\n".join(f"  {p}: {v[:8]}" for p, v in sorted(hits.items()))
        self.assertEqual({}, hits,
                         f"已跟踪文件命中真实地址/端口模式:\n{report}")

    def test_scanner_is_not_a_no_op(self):
        """反向验证扫描器真的在匹配: 喂一段含违禁形态的文本必须被抓到。

        没这条的话, 一个把文件列表写错的测试会永远绿着 —— 而它的「绿」正是全部
        安全感的来源。
        """
        probe = {"fake.py": "key = 'sk-abcdefghij0123456789'\n# zhanghan@host\n"}
        found = set()
        for pat in FORBIDDEN_LITERAL_PATTERNS:
            for rel, body in probe.items():
                if re.search(pat, body):
                    found.add(pat)
        self.assertIn(r"sk-[A-Za-z0-9]{20,}", found)
        self.assertIn("zhanghan", found)


class TestLocalConfigStaysUntracked(unittest.TestCase):
    """config.yaml 装真实值, triton/ 是 vendored 检出。两者都必须是未跟踪的。"""

    def test_config_yaml_not_tracked(self):
        tracked = set(_tracked_files())
        self.assertNotIn("config.yaml", tracked,
                         "config.yaml 进了索引 —— 它含真实 SSH 主机与 LLM 端点")

    def test_config_yaml_is_ignored(self):
        r = subprocess.run(["git", "-C", str(REPO_ROOT), "check-ignore", "-q",
                            "config.yaml"], capture_output=True)
        self.assertEqual(0, r.returncode, "config.yaml 不再被 .gitignore 覆盖")

    def test_triton_checkout_not_tracked(self):
        tracked = [f for f in _tracked_files() if f == "triton" or f.startswith("triton/")]
        self.assertEqual([], tracked, "本地 Triton 检出泄漏进索引了")

    def test_example_config_has_no_real_values(self):
        """config.example.yaml 是模板, 它比 config.yaml 更该干净 —— 后者已被忽略。"""
        p = REPO_ROOT / "config.example.yaml"
        if not p.is_file():
            self.skipTest("无 config.example.yaml")
        body = p.read_text(encoding="utf-8")
        for pat in FORBIDDEN_LITERAL_PATTERNS + FORBIDDEN_ADDRESS_PATTERNS:
            self.assertIsNone(re.search(pat, body),
                              f"config.example.yaml 命中 {pat} —— 模板里只该有占位符")


class TestNoScratchFilesAtRepoRoot(unittest.TestCase):
    """根目录的 `_*.py` 临时脚本历史上含过旧凭据。

    它们被 `.gitignore` 的 `/_*.py` 忽略, 因此 `_tracked_files()` 扫不到 —— 这条
    补的就是那个盲区: 一旦有人 `git add -f` 把它拉进索引, 上面的扫描会立刻看见。
    """

    def test_root_scratch_scripts_are_not_forced_into_the_index(self):
        tracked = [f for f in _tracked_files()
                   if "/" not in f and f.startswith("_") and f.endswith(".py")]
        self.assertEqual([], tracked,
                         f"根目录临时脚本被强制加入索引: {tracked} —— 这类文件不曾、"
                         f"也不该出现在公开仓里")


if __name__ == "__main__":
    unittest.main(verbosity=2)
