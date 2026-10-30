"""打包形态的守卫：别让「装出来和跑起来不是同一个东西」

这里只钉三件**零成本**的事：

1. `pyproject.toml` 的版本号和 `operator_agent.__version__` 一致；
2. 装包必须带上的东西都在（`remote_scripts/__init__.py`、`prompts/`、`skills/`）；
3. npm 壳的 `dsh.bundle.patch` 指向的文件存在，且 `engines.dsh` 有值；
4. npm 壳的「脸」（locale / icon / `./package.json` 子路径导出）符合宿主规则。

前两条防的是「源码树能跑、装完就找不到 runner / 模板」；第三条防的是
「发布出去的包装不上」——而那两种失效都不会在单测里自己叫。第四条防的是
「宿主静默退化」（标题变成裸包名、图标没有），只在 UI 上看得出来。
"""

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPythonPackaging(unittest.TestCase):

    def test_pyproject_version_matches_the_package(self):
        from operator_agent import __version__
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'^version = "([^"]+)"', text, re.MULTILINE)
        self.assertIsNotNone(match, "pyproject.toml 里没有 version")
        self.assertEqual(match.group(1), __version__)

    def test_the_remote_runner_ships_with_the_package(self):
        # 没有这个文件，setuptools 不会把 remote_scripts 当顶层包装，
        # ssh 后端第一次 deploy 就会找不到 runner（源码树里却一切正常）。
        self.assertTrue((ROOT / "remote_scripts" / "__init__.py").is_file())
        self.assertTrue((ROOT / "remote_scripts" / "run_kernel.py").is_file())

    def test_templates_and_skills_are_declared_as_package_data(self):
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        for needle in ('"prompts/*.md"', '"skills/*.md"'):
            self.assertIn(needle, text, f"package-data 缺 {needle}")
        self.assertTrue((ROOT / "operator_agent" / "prompts").is_dir())
        self.assertTrue((ROOT / "operator_agent" / "skills").is_dir())

    def test_runtime_directories_never_enter_the_distribution(self):
        text = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
        for name in ("output", "logs", "triton"):
            self.assertIn(f"prune {name}", text, f"MANIFEST 没排除 {name}/")


class TestLicensing(unittest.TestCase):
    """许可是一处「声明」与三处「字段」必须互相指认的东西。

    这类漂移（改了 `package.json` 忘了 `LICENSE`，或反过来）不会让任何测试变红，
    只会让发布物在法律上自相矛盾 —— 所以钉住。
    """

    SPDX = "Apache-2.0"

    def test_the_license_file_is_the_apache_text(self):
        text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("Apache License", text)
        self.assertIn("Version 2.0, January 2004", text)
        self.assertIn("http://www.apache.org/licenses/", text)

    def test_every_manifest_declares_the_same_spdx_id(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn(f'license = {{ text = "{self.SPDX}" }}', pyproject)
        pkg = json.loads((ROOT / "integrations" / "dsh-operator" / "package.json").read_text())
        self.assertEqual(pkg["license"], self.SPDX)
        for name in ("pyproject.toml", "package.json"):
            blob = (ROOT / ("pyproject.toml" if name == "pyproject.toml"
                            else "integrations/dsh-operator/package.json")).read_text()
            self.assertNotIn("UNLICENSED", blob, f"{name} 还留着 UNLICENSED")

    def test_third_party_assets_keep_their_own_licenses(self):
        text = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        self.assertIn("echarts.min.js", text)
        self.assertIn("mermaid.min.js", text)
        self.assertIn("Apache-2.0", text)
        self.assertIn("MIT", text)

    def test_the_notice_carries_the_copyright_holder(self):
        text = (ROOT / "NOTICE").read_text(encoding="utf-8")
        self.assertIn("operator-agent", text)
        self.assertIn("Copyright 2026 the operator-agent authors", text)
        for placeholder in ("[yyyy]", "[name of copyright owner]", "TODO", "待填"):
            self.assertNotIn(placeholder, text, f"NOTICE 里还留着占位符 {placeholder}")
        self.assertIn("THIRD_PARTY_NOTICES.md", text, "第三方声明要被 NOTICE 指到")

    def test_the_readme_points_at_the_license(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("LICENSE", text)
        self.assertIn(self.SPDX, text)


class TestTheDshShim(unittest.TestCase):

    SHIM = ROOT / "integrations" / "dsh-operator"

    def test_manifest_points_at_an_existing_patch_file(self):
        pkg = json.loads((self.SHIM / "package.json").read_text(encoding="utf-8"))
        patch = pkg["dsh"]["bundle"]["patch"]
        self.assertTrue((self.SHIM / patch).is_file(), patch)
        self.assertTrue(pkg.get("engines", {}).get("dsh"), "engines.dsh 必须有值")

    def test_the_patch_registers_the_mcp_server(self):
        import yaml
        rows = yaml.safe_load((self.SHIM / "cordis.patch.yml").read_text(encoding="utf-8"))
        row = rows[0]["insert"][0]
        self.assertEqual(row["name"], "@deepseek-ai/dsh-mcp-client")
        config = row["config"]
        self.assertEqual(config["serverName"], "operator")
        self.assertEqual(config["transport"], "stdio")
        # 发布形态不写死绝对路径：command 是 pip 装出来的 console script
        self.assertEqual(config["command"], "operator-agent-mcp")

    def test_every_patch_row_that_is_not_official_points_at_this_package(self):
        """按包名安装时，patch 行里的 `name:` 必须**就是本包名**。

        写错的后果不是报错而是「装上了但 boot 不起来」：宿主按 `name` 去 profile /
        安装锚点解析，解析不到就跳过这一行（`skipping profile bundle …` 只在启动时打印）。
        官方包（`@deepseek-ai/*`）由安装侧提供，不在此列。
        """
        import yaml
        pkg = json.loads((self.SHIM / "package.json").read_text(encoding="utf-8"))
        rows = yaml.safe_load((self.SHIM / "cordis.patch.yml").read_text(encoding="utf-8"))
        names = [row["name"] for entry in rows for row in entry.get("insert", [])]
        self.assertTrue(names, "patch 里没有 insert 行？")
        for name in names:
            if name.startswith("@deepseek-ai/"):
                continue
            self.assertEqual(name, pkg["name"],
                             f"patch 行 name={name} 与包名 {pkg['name']} 不一致")

    def test_the_mcp_call_timeout_covers_a_real_machine_verdict(self):
        """一次真机完整判定的实测成本，必须是这一格的**下界**。

        实测 866.8 s（docs/dsh-plugin-evaluation.md §9.8，L2 的 15 个 case 是大头），
        而 MCP 默认只有 60 s。这里钉住「别把它调小回默认量级」：超时不是优雅退出 ——
        server 是同步逐行读，客户端 abort 后它仍会算完再写一个没人认领的响应。
        """
        import yaml
        config = yaml.safe_load(
            (self.SHIM / "cordis.patch.yml").read_text(encoding="utf-8"))[0]["insert"][0]["config"]
        self.assertGreaterEqual(
            config["toolCallTimeoutMs"], 1_800_000,
            "真机一次完整判定约 866.8 s、repair 默认 2 轮约 30 min；别低于 30 min")

    def test_the_fields_the_host_shell_reads_still_exist(self):
        """`index.mjs` 是静态壳：payload 改名它只会打成「?」，不会崩。

        所以契约钉在**产出侧**（这里），而不是只靠 JS 侧的 fixture。
        """
        from operator_agent.hitl_tools import autonomous_status, capabilities

        cap = capabilities()
        for key in ("ok", "version", "tool_version", "backends", "runtime",
                    "kernel_types_with_input_layout"):
            self.assertIn(key, cap)
        for name in ("mock", "local", "ssh"):
            self.assertIn("wired", cap["backends"][name], name)
        for key in ("python", "torch_installed", "triton_installed"):
            self.assertIn(key, cap["runtime"])

        # 终态下 process_alive 是 None（不是 False）—— 壳必须按 phase 判死活，
        # 否则一个跑完的 run 会显示成「进程 在跑」（JS 侧有对应回归测试）。
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "run-status.json").write_text(
                json.dumps({"phase": "done", "pid": 999999, "started_at": 1760000000.0}),
                encoding="utf-8")
            status = autonomous_status(tmp)
        self.assertEqual(status["phase"], "done")
        self.assertIsNone(status["process_alive"])
        for key in ("ok", "phase", "process_alive", "started_at", "duration_s",
                    "summary", "log_tail"):
            self.assertIn(key, status)


class TestTheDshPluginFace(unittest.TestCase):
    """宿主给插件配「脸」的三个资源：`<包名>/package.json`、`<包名>/locale/*.json`、`icon`。

    规则是从 DSH 源码（`dsh-app-boot` 的 `readPluginMeta` / `iconOf` / `dictionariesOf`）
    抄下来镜像一份。为什么值得钉：宿主读不到时的表现**不是报错，而是静默退化** ——
    - 少了 `"./package.json"` 这条 exports：manifest 读不到 ⇒ 标题退化成裸包名
      `dsh-operator-agent`（而且 `manifest.icon` 这条路也一起断了）；
    - 少了 `locale/en.json`：那是宿主**列目录枚举语言**的入口，等于整个 locale 不存在；
    - 少了 `files` 里的条目：源码树里一切正常，发出去的包里没有。
    这三种都只在 UI 上看得出来，所以在这里（以及 `test/host-contract.test.mjs`
    里用真 `readPluginMeta`）各钉一遍。
    """

    SHIM = ROOT / "integrations" / "dsh-operator"
    #: 宿主 `LANGUAGE_ID`：locale 文件名必须是语言 id。
    LANGUAGE_ID = re.compile(r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$")
    ICON_TYPES = {".svg", ".png", ".jpg", ".jpeg", ".webp"}
    MAX_ICON_BYTES = 256 * 1024

    def manifest(self):
        return json.loads((self.SHIM / "package.json").read_text(encoding="utf-8"))

    def test_the_manifest_subpath_is_exported(self):
        exports = self.manifest()["exports"]
        self.assertIn("./package.json", exports,
                      "宿主用真模块解析器读 <包名>/package.json；不导出就等于没有元数据")

    def test_the_locale_files_are_well_formed_exported_and_packed(self):
        pkg = self.manifest()
        locales = sorted((self.SHIM / "locale").glob("*.json"))
        self.assertTrue(locales, "至少要有 locale/en.json + locale/zh.json")
        self.assertTrue((self.SHIM / "locale" / "en.json").is_file(),
                        "en.json 是宿主枚举语言的入口，缺了等于没有 locale")
        for path in locales:
            self.assertRegex(path.stem, self.LANGUAGE_ID, path.name)
            meta = json.loads(path.read_text(encoding="utf-8")).get("meta")
            self.assertIsInstance(meta, dict, f"{path.name}: 需要 meta 对象")
            for field in ("title", "description"):
                value = meta.get(field)
                self.assertIsInstance(value, str, f"{path.name}: meta.{field}")
                self.assertTrue(value.strip(), f"{path.name}: meta.{field} 不能是空串")
        self.assertIn("./locale/*.json", pkg["exports"], "locale 子路径必须导出")
        self.assertIn("locale/*.json", pkg["files"], "locale 必须进 files")

    def test_the_icon_obeys_the_hosts_rules(self):
        pkg = self.manifest()
        icon = pkg["icon"]
        self.assertTrue(icon.startswith("./"), f"宿主只收包内相对路径：{icon}")
        target = (self.SHIM / icon).resolve()
        self.assertTrue(target.is_relative_to(self.SHIM.resolve()),
                        "图标必须留在包目录内（宿主会 realpath 后校验）")
        self.assertTrue(target.is_file(), icon)
        self.assertIn(target.suffix.lower(), self.ICON_TYPES)
        self.assertLessEqual(target.stat().st_size, self.MAX_ICON_BYTES)
        self.assertIn(icon[2:], pkg["files"], "图标必须进 files，否则发出去没有")


class TestTheDshReleaseShape(unittest.TestCase):
    """对外发布形态：许可随包、包可发、版本号与内容一一对应。

    这三条都是「源码树里看不出问题、发出去才违约」的那一类：
    npm 只打包 `files` 里列到的东西，`license` 字段声明 Apache-2.0 而包里没有条文，
    安装方就拿不到 Apache-2.0 §4 要求的许可副本；`private: true` 则让 `npm publish`
    直接失败。同号不同内容会让「按版本号引用工件」变成假引用 —— 0.1.9 已有两份内容
    不同的本地构建，靠 SHA256 区分，所以发布版的号必须往前挪。
    """

    SHIM = ROOT / "integrations" / "dsh-operator"
    #: 必须随包分发、在包内是仓库根同名文件的逐字节副本。
    LEGAL = ("LICENSE", "NOTICE")

    def manifest(self):
        return json.loads((self.SHIM / "package.json").read_text(encoding="utf-8"))

    def test_the_shipped_legal_files_are_byte_copies_of_the_root_ones(self):
        for name in self.LEGAL:
            packaged = self.SHIM / name
            self.assertTrue(packaged.is_file(), f"{name} 不在包目录里：npm 发不出去")
            self.assertEqual(packaged.read_bytes(), (ROOT / name).read_bytes(),
                             f"{name} 与仓库根不一致：改一处忘了另一处")

    def test_the_legal_files_are_declared_in_files(self):
        files = self.manifest()["files"]
        for name in self.LEGAL:
            self.assertIn(name, files, f"{name} 没进 files，npm pack 里不会有")

    def test_the_package_is_publishable(self):
        pkg = self.manifest()
        self.assertIs(pkg.get("private"), False,
                      "private 不为 false 时 npm publish 直接拒绝")
        self.assertEqual(pkg["license"], "Apache-2.0")
        self.assertRegex(pkg["version"], r"^\d+\.\d+\.\d+$")
        self.assertNotEqual(pkg["version"], "0.1.9",
                            "0.1.9 已被两份本地构建占用，发布版必须换号")

    def test_the_changelog_opens_with_the_manifest_version(self):
        text = (self.SHIM / "CHANGELOG.md").read_text(encoding="utf-8")
        first = next(line for line in text.splitlines() if line.startswith("## "))
        self.assertEqual(first.strip(), "## " + self.manifest()["version"])


if __name__ == "__main__":
    unittest.main()
