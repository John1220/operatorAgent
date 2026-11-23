"""构建期清单守卫：**一份手抄的名单**比没有名单更危险。

背景：JS 的模型预算门（`host-model-budget.mjs`）与 Python 的工具面各存一份同样的
七条工具名。Python 改了工具面，JS 那份不会响 —— 而它是判据。落后的名单不会让任何
东西编译失败，只会让那道门守着一个已经不存在的工具集，并且看起来一切正常。

现在：Python 在构建期生成 `integrations/dsh-operator/manifest.json`，JS 读它。
这个文件钉三件事：

1. **清单没腐烂** —— 提交在仓里的那份必须等于现算出来的那份。改了工具面忘了重新
   生成，这里就红。重新生成：`python -m operator_agent.app.mcp_server --emit-manifest`
2. **清单自洽** —— 声明的 host 工具必须真的被定义，标记与清单不许分家。
3. **没有第二份名单** —— JS 里不许再出现手抄的工具名数组；npm 必须把清单打进去。
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLUGIN = ROOT / "integrations" / "dsh-operator"
MANIFEST = PLUGIN / "manifest.json"

from operator_agent.app import mcp_server                                  # noqa: E402

REGENERATE = ("重新生成：python -m operator_agent.app.mcp_server --emit-manifest"
              "（改了任何工具 schema 都必须重跑）")


class TestTheManifestDoesNotRot:
    def test_the_committed_manifest_matches_a_fresh_generation(self):
        committed = json.loads(MANIFEST.read_text(encoding="utf-8"))
        fresh = mcp_server.build_manifest()
        assert committed == fresh, (
            "清单与工具面不一致 —— 工具改了但清单没重新生成。" + REGENERATE)

    def test_the_manifest_is_self_consistent(self):
        committed = json.loads(MANIFEST.read_text(encoding="utf-8"))
        assert mcp_server.manifest_problems(committed) == []

    def test_every_defined_tool_is_in_the_manifest(self):
        """反向也要成立：定义了却没进清单的工具，宿主永远看不到它。"""
        committed = json.loads(MANIFEST.read_text(encoding="utf-8"))
        defined = {tool["name"] for tool in mcp_server._tool_definitions(host_mode=True)}
        assert defined == {row["name"] for row in committed["tools"]}

    def test_the_host_tool_list_has_one_source(self):
        """清单里的 host_tools 就是 `HOST_TOOLS`，不是它的副本。"""
        committed = json.loads(MANIFEST.read_text(encoding="utf-8"))
        assert committed["host_tools"] == sorted(mcp_server.HOST_TOOLS)

    def test_the_manifest_describes_the_production_surface_only(self):
        """清单按**正式路径**（host 模式）枚举，legacy 三支不在里面。

        它们没有消失 —— 非 host 模式的运行时工具面里仍然有它们（研究/兼容用），
        但正式路径不暴露。这与 `_guard_host_tool` 是同一条纪律的两处表达：
        一处是「不给清单」，一处是「给了也拒」。
        """
        committed = json.loads(MANIFEST.read_text(encoding="utf-8"))
        names = {row["name"] for row in committed["tools"]}
        assert names == set(mcp_server.HOST_TOOLS)
        legacy = {"operator_run", "operator_repair", "operator_bench"}
        assert names & legacy == set()
        research = {tool["name"] for tool in mcp_server._tool_definitions(host_mode=False)}
        assert legacy <= research, "legacy 工具被删了而不是被藏起来 —— 那是另一件事"

    def test_the_read_only_list_is_in_the_manifest(self):
        """只读集合也随清单发 —— 它是 JS 模型门的第三个手抄副本。"""
        committed = json.loads(MANIFEST.read_text(encoding="utf-8"))
        assert committed["read_only_tools"] == sorted(mcp_server.READ_ONLY_TOOLS)
        assert set(committed["read_only_tools"]) <= set(committed["host_tools"])


class TestTheDescriptionsOnlyNameToolsThatExist:
    """工具描述里提到的工具必须真的存在。

    这条是**我自己踩出来的**：`operator_prepare` 的 schema 曾经写着「拿不准就先调
    `operator_normalize` 起草一份」，而那个工具并不存在。宿主照着描述做会直接失败，
    而且没有任何东西会响 —— 描述是写给模型读的契约，读到不存在的工具名是一种
    不会被执行的错误。
    """

    #: 描述里出现、但**不是**工具名的 `operator_*` 词。
    NOT_TOOLS = {"operator_agent"}

    @staticmethod
    def _strings(node):
        """schema 里所有字符串（描述、字段描述、枚举……）。"""
        if isinstance(node, str):
            yield node
        elif isinstance(node, dict):
            for value in node.values():
                yield from TestTheDescriptionsOnlyNameToolsThatExist._strings(value)
        elif isinstance(node, list):
            for item in node:
                yield from TestTheDescriptionsOnlyNameToolsThatExist._strings(item)

    def test_every_named_tool_can_actually_be_called(self):
        """整个 inputSchema 一起扫，不止工具描述 —— 字段描述同样会被模型读到。"""
        tools = mcp_server._tool_definitions(host_mode=True)
        # 「提到」不等于「可调」：`candidate_origin` 的描述要解释「inner_fixer 是
        # operator_repair 的内层模型改的」，那是在讲归因词汇，不是叫模型去调。
        # 所以判据是「这个名字在**某个**模式里真的被定义过」，而不是「在 host 面里」。
        defined = {tool["name"] for mode in (True, False)
                   for tool in mcp_server._tool_definitions(host_mode=mode)}
        mentioned = set()
        for tool in tools:
            for text in self._strings(tool.get("inputSchema", {})):
                mentioned.update(re.findall(r"\boperator_[a-z_]+", text))
            mentioned.update(re.findall(r"\boperator_[a-z_]+", tool.get("description", "")))
        dangling = sorted(mentioned - defined - self.NOT_TOOLS)
        assert dangling == [], (
            f"工具描述里提到了不存在的工具 {dangling} —— 模型会照着调，然后失败")

    def test_the_instructions_point_at_the_drafter(self):
        """`instructions` 是宿主读的第一段契约，同样会提到工具名，同样会漂。"""
        imported = set(re.findall(r"\boperator_[a-z_]+", mcp_server.INSTRUCTIONS))
        defined = {tool["name"] for mode in (True, False)
                   for tool in mcp_server._tool_definitions(host_mode=mode)}
        dangling = sorted(imported - defined - self.NOT_TOOLS)
        assert dangling == [], f"instructions 里提到了不存在的工具 {dangling}"

    def test_the_operator_field_points_at_the_drafter(self):
        """缺 `operator` / `precision` 时该去哪儿起草，字段描述里得说清楚。

        宿主是在读 `operator` 这一格时才发现自己不知道填什么的，入口就该在那里；
        只写在工具描述里，读者已经错过那一屏了。
        """
        prepare = next(tool for tool in mcp_server._tool_definitions(host_mode=True)
                       if tool["name"] == "operator_prepare")
        for field in ("operator", "precision"):
            described = prepare["inputSchema"]["properties"][field]["description"]
            assert "operator_normalize" in described, f"{field} 的描述没给起草入口"


class TestThereIsNoSecondCopyOfTheToolList:
    def test_the_js_budget_gate_reads_the_manifest(self):
        loader = (PLUGIN / "manifest.mjs").read_text(encoding="utf-8")
        assert "manifest.json" in loader, "加载器没有读清单文件"
        gate = (PLUGIN / "host-model-budget.mjs").read_text(encoding="utf-8")
        assert "MANIFEST.host_tools" in gate, "预算门没有从清单派生工具名单"

    def test_the_js_budget_gate_has_no_hand_written_tool_array(self):
        """手抄形状的守卫：出现连续两个工具名当字面量，就是第二份名单回来了。"""
        source = (PLUGIN / "host-model-budget.mjs").read_text(encoding="utf-8")
        hand_written = re.search(r'\[\s*"operator_\w+"\s*,\s*"operator_\w+"', source)
        assert hand_written is None, (
            "host-model-budget.mjs 里又出现了手写的工具名数组 —— "
            "那是第二份会漂的真相，应该从 MANIFEST.host_tools 派生")

    def test_the_npm_package_ships_the_manifest(self):
        """清单不进包，装了就等于没有 —— 而且会在装载期直接炸。"""
        package = json.loads((PLUGIN / "package.json").read_text(encoding="utf-8"))
        assert "manifest.json" in package["files"]

    def test_no_js_module_keeps_its_own_tool_array(self):
        """全仓扫：`test/` 之外任何 `.mjs` 都不许出现两个以上工具名的字面量数组。

        这是**唯一**拦得住「以后又抄一份」的形状。单点断言只能守住已经被收掉的那四处，
        拦不住第五处；逐个文件写断言又会随文件增减而腐烂。
        """
        pattern = re.compile(r'\[\s*"operator_[a-z_]+"\s*,\s*"operator_[a-z_]+')
        offenders = sorted(
            str(path.relative_to(ROOT))
            for path in PLUGIN.rglob("*.mjs")
            if "/test/" not in path.as_posix() and pattern.search(path.read_text(encoding="utf-8")))
        assert offenders == [], (
            f"这些 JS 模块还在手抄工具名单：{offenders} —— 应该从 manifest.mjs 派生")

    def test_every_js_consumer_imports_the_shared_loader(self):
        """四处曾经的手抄副本，现在都必须走 `manifest.mjs`。"""
        for name in ("host-model-budget.mjs", "host-session-lifecycle.mjs", "bin/setup.mjs"):
            source = (PLUGIN / name).read_text(encoding="utf-8")
            assert "manifest.mjs" in source, f"{name} 没有走共享加载器"

    def test_the_manifest_loader_is_shipped(self):
        package = json.loads((PLUGIN / "package.json").read_text(encoding="utf-8"))
        assert "manifest.mjs" in package["files"]

    def test_bind_tools_are_in_the_manifest(self):
        committed = json.loads(MANIFEST.read_text(encoding="utf-8"))
        assert committed["bind_tools"] == sorted(mcp_server.BIND_TOOLS)
        assert set(committed["bind_tools"]) <= set(committed["host_tools"])

    def test_the_bundle_actually_selects_the_mode_the_manifest_describes(self):
        """清单描述的是**host 模式**的工具面，捆包就必须真的选 host。

        `OPERATOR_AGENT_WORKFLOW` 的 Python 默认是 `legacy`（独立 CLI 保留原行为，
        这是有意的）。所以「清单说七支」只在捆包把环境设成 host 时才成立 —— 这一条
        把两件事钉在一起：谁把捆包改成 legacy，工具面与清单立刻分家。
        """
        import yaml
        patch = yaml.safe_load((PLUGIN / "cordis.patch.yml").read_text(encoding="utf-8"))

        def find_env(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "OPERATOR_AGENT_WORKFLOW":
                        return value
                    found = find_env(value)
                    if found is not None:
                        return found
            elif isinstance(node, list):
                for item in node:
                    found = find_env(item)
                    if found is not None:
                        return found
            return None

        assert find_env(patch) == "host", (
            "捆包没有把 OPERATOR_AGENT_WORKFLOW 设成 host —— "
            "清单描述的是 host 模式工具面（七支），捆包若跑 legacy 就会暴露另一套")

    def test_a_manifest_from_another_build_is_refused_loudly(self):
        """schema 不认识就必须当场炸，不许降级成空集合（空集合 = 谁也不拦）。"""
        loader = (PLUGIN / "manifest.mjs").read_text(encoding="utf-8")
        assert "dsh-operator-manifest-v1" in loader
        assert "throw new Error" in loader
        # 拒绝的形状也要钉住：不许出现「读失败就退回空集合」这种降级。
        assert "?? []" not in loader and "|| []" not in loader, (
            "加载器出现了降级到空集合的写法 —— 空集合意味着谁也不拦")