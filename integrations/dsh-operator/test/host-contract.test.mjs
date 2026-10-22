// 与**真实宿主**的契约：替身 ctx 只会照单全收，验不出 normalizeDefinition /
// normalizeResult 的规矩。这条用装在机器上的 DSH 自己的 `@deepseek-ai/cordis` 与
// `@deepseek-ai/dsh-commands` 起一个真服务，把插件注册进去、再真执行一次命令。
//
// 找不到 DSH 安装就**跳过**（不假装通过）：设 `DSH_APP_ROOT` 指到 DSH 的
// `Contents/Resources/app`（或在 `node_modules` 里）即可。刻意不依赖 Python ——
// 成功路径用一个假 CLI，所以这些在任何装了 DSH 的机器上都能跑；只有最后一条
// （instructions 端到端）需要真 Python，找不到就自己跳过。
import { test } from "node:test";
import assert from "node:assert/strict";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, symlinkSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { apply } from "../index.mjs";

/** DSH 的安装根（app 目录）；macOS 桌面版给一个默认值，别的形态用环境变量指。 */
function findDshApp() {
  const candidates = [
    process.env.DSH_APP_ROOT,
    "/Applications/DSH Desktop.app/Contents/Resources/app",
  ].filter(Boolean);
  return candidates.find((dir) =>
    existsSync(join(dir, "node_modules", "@deepseek-ai", "cordis", "lib", "index.js")));
}

/** 本包目录：宿主的元数据读取器就是拿裸包名从这里解析子路径的。 */
const PACKAGE_DIR = dirname(dirname(fileURLToPath(import.meta.url)));
/** 判定层仓库根（`integrations/dsh-operator` 的上两级）：起真 server 时要拿它当 cwd。 */
const REPO_ROOT = dirname(dirname(PACKAGE_DIR));

const APP = findDshApp();
const skip = APP ? false : "找不到 DSH 安装（设 DSH_APP_ROOT 后可跑）";

/** 从宿主安装里直接按文件路径导入：裸包名解析不在本仓的 node_modules 里。 */
const fromApp = (specifier) => import(pathToFileURL(join(APP, "node_modules", specifier)).href);

async function realCommandsContext() {
  const [{ Context }, Commands] = await Promise.all([
    fromApp("@deepseek-ai/cordis/lib/index.js"),
    fromApp("@deepseek-ai/dsh-commands/lib/index.js"),
  ]);
  const root = new Context();
  root.plugin(Commands.default);            // 服务是**类**插件（export default）
  await new Promise((resolve) => setTimeout(resolve, 300));
  assert.ok(root.commands, "宿主没有起 commands 服务");
  return root;
}

/** 真 commands 服务 + 最小替身：插件只用 systemPrompt.section 与 logger。 */
function ctxShim(root, onSection = () => {}) {
  return {
    commands: root.commands,
    systemPrompt: { section: onSection },
    logger: { info: () => {}, warn: () => {}, error: () => {} },
  };
}

function fakeCli(body) {
  const dir = mkdtempSync(join(tmpdir(), "host-contract-"));
  const file = join(dir, "operator-agent");
  writeFileSync(file, `#!/bin/sh\n${body}\n`, "utf8");
  chmodSync(file, 0o755);
  return file;
}

test("真宿主的 commands 服务收下 /op：名字、definitionId、hint 都活在服务侧", { skip }, async () => {
  const root = await realCommandsContext();
  const sections = [];
  apply(ctxShim(root, (s) => sections.push(s)), { command: "operator-agent" });

  const listed = await root.commands.list({});
  assert.equal(listed.length, 1);
  assert.equal(listed[0].name, "op");
  assert.equal(listed[0].definitionId, "dsh-operator-agent");
  assert.equal(listed[0].input.hint, "capabilities | status <run_dir> | help");
  // 真宿主上同样**不该**有自注册的提示段：说明来自 server 的 instructions，
  // 由 mcp-client 注册成 `mcp:operator`（见 dsh-mcp-client 的 registerServerContext）。
  assert.deepEqual(sections, []);
});

test("命令真的执行：宿主过一遍 normalizeResult，结果与生命周期事件都成立", { skip }, async () => {
  const root = await realCommandsContext();
  const cli = fakeCli(`cat <<'JSON'
{"ok":true,"version":"0.6.0","tool_version":1,
 "backends":{"mock":{"wired":true}},
 "runtime":{"python":"3.9.12","torch_installed":true,"triton_installed":true},
 "kernel_types_with_input_layout":["matmul"]}
JSON`);
  apply(ctxShim(root), { command: cli });

  const log = [];
  const agent = { session: { append: (type, data) => log.push(type) } };
  const out = await root.commands.execute(agent, "/op capabilities", [], new AbortController().signal);

  assert.equal(out.result.kind, "success");
  assert.ok(out.result.text.includes("算子判定层 0.6.0"), out.result.text);
  assert.deepEqual(log, ["command/run", "command/done"]);
});

test("取消由宿主在 handler 之前就挡下（真 AbortError）", { skip }, async () => {
  const root = await realCommandsContext();
  apply(ctxShim(root), { command: "/definitely/not/installed" });
  const controller = new AbortController();
  controller.abort();
  const agent = { session: { append: () => {} } };
  await assert.rejects(
    () => root.commands.execute(agent, "/op capabilities", [], controller.signal),
    (err) => err.name === "AbortError");
});

test("不认识的子命令回 error，且 error 文案非空（宿主会校验）", { skip }, async () => {
  const root = await realCommandsContext();
  apply(ctxShim(root), { command: "operator-agent" });
  const agent = { session: { append: () => {} } };
  const out = await root.commands.execute(agent, "/op nope", [], new AbortController().signal);
  assert.equal(out.result.kind, "error");
  assert.ok(out.result.text.trim().length > 0);
});
test("真宿主把 agent 交给 handler：含空格的相对路径整段到 CLI", { skip }, async () => {
  const root = await realCommandsContext();
  // 假 CLI 把收到的第三个参数（--run-dir 的值）原样回进 payload。
  const cli = fakeCli(`printf '{"ok":true,"phase":"%s"}' "$3"`);
  apply(ctxShim(root), { command: cli });
  const agent = { session: { append: () => {} } };
  const out = await root.commands.execute(agent, "/op status my runs/x", [], new AbortController().signal);

  assert.equal(out.result.kind, "success", out.result.text);
  assert.ok(out.result.text.includes("相位：my runs/x"),
    `路径没整段到达 CLI（被切碎或被改写了）：${out.result.text}`);
});

test("engines.dsh 与**装在这台机器上的** DSH 版本相容", { skip }, async () => {
  const { evaluatePluginCompatibility, getDshRuntimeVersion, pluginCompatibilityWarning } =
    await fromApp("@deepseek-ai/dsh-app-boot/lib/index.js");
  const pkg = JSON.parse(readFileSync(join(PACKAGE_DIR, "package.json"), "utf8"));
  // 用宿主自己的判据 + 宿主自己的版本：这个范围是人手写的，装了新宿主会悄悄失配，
  // 而失配的后果是**装不上**（`dsh plugin` 会直接拒），所以别等那一刻才发现。
  const issue = evaluatePluginCompatibility(pkg, {});
  assert.equal(issue, undefined,
    `engines.dsh=${JSON.stringify(pkg.engines?.dsh)} 不满足装在这台机器上的 dsh `
    + `${getDshRuntimeVersion()}：${issue ? pluginCompatibilityWarning(issue) : ""}`);
});

test("宿主读得到这张「脸」：标题/描述/图标（少一条 exports 就红）", { skip }, async () => {
  const { readPluginMeta } = await fromApp("@deepseek-ai/dsh-app-boot/lib/index.js");
  // 复刻 `link:` 安装形态：一个 node_modules/dsh-operator-agent 指向本包 —— 宿主就是
  // 拿裸包名 + 真模块解析器去读 `<包名>/package.json`、`<包名>/locale/en.json` 的，
  // 所以这条同时钉住了 `exports` 里必须有的那两个子路径。
  const home = mkdtempSync(join(tmpdir(), "plugin-meta-"));
  mkdirSync(join(home, "node_modules"), { recursive: true });
  symlinkSync(PACKAGE_DIR, join(home, "node_modules", "dsh-operator-agent"), "dir");

  const meta = readPluginMeta("dsh-operator-agent", pathToFileURL(join(home, "profile.mjs")).href);
  assert.ok(meta && meta.error === undefined, `读元数据失败：${JSON.stringify(meta)}`);

  // 期望值直接取 locale 原文：说明读到的是**我们的**字典，而不是退化的包名/描述。
  const of = (language) => JSON.parse(
    readFileSync(join(PACKAGE_DIR, "locale", `${language}.json`), "utf8")).meta;
  assert.deepEqual(meta.title, { en: of("en").title, zh: of("zh").title });
  assert.deepEqual(meta.description, { en: of("en").description, zh: of("zh").description });
  assert.notDeepEqual(meta.title, { en: "dsh-operator-agent" }, "标题退化成了裸包名");
  assert.match(meta.icon, /^data:image\/svg\+xml;base64,/);
});

/** 找到一个**能 import 判定层**的 python（显式指定 > PATH 上的 python3）。 */
function findPython() {
  const candidates = [process.env.OPERATOR_AGENT_PYTHON, "python3", "/usr/bin/python3"].filter(Boolean);
  return candidates.find((python) => {
    const probe = spawnSync(python, ["-c", "import operator_agent.mcp_server"],
                            { cwd: REPO_ROOT, timeout: 30_000 });
    return probe.status === 0;
  });
}

test("server 的 instructions 真的到达 MCP 客户端（宿主据此渲染 mcp:operator 段）", async (t) => {
  if (!APP) return t.skip("找不到 DSH 安装（设 DSH_APP_ROOT 后可跑）");
  const python = findPython();
  if (!python) return t.skip("找不到能 import 判定层的 python（设 OPERATOR_AGENT_PYTHON）");

  // 用**宿主自己依赖的那个** SDK 客户端起真 server：这条验的是「说明真的过线」，
  // 而不是「我们在源码里写了 instructions」。
  const [{ Client }, { StdioClientTransport }] = await Promise.all([
    fromApp("@modelcontextprotocol/client/dist/index.mjs"),
    fromApp("@modelcontextprotocol/client/dist/stdio.mjs"),
  ]);
  const client = new Client({ name: "contract", version: "0" }, { capabilities: {} });
  await client.connect(new StdioClientTransport({
    command: python,
    args: ["-m", "operator_agent.mcp_server"],
    cwd: REPO_ROOT,
    env: { PATH: process.env.PATH, HOME: process.env.HOME },
  }));
  try {
    const text = client.getInstructions() ?? "";
    assert.ok(text.includes("eligibility"), `instructions 没到客户端：${text.slice(0, 120)}`);
    assert.ok(text.includes("hold_for_measurement"), "冲突处置那条丢了");
    // DSH 默认 maxInstructionBytes = 32768：超了宿主会直接拒绝连接（不是截断）。
    assert.ok(Buffer.byteLength(text) < 32 * 1024, `instructions 超了宿主上限：${Buffer.byteLength(text)}`);
    const { tools } = await client.listTools();
    assert.equal(tools.length, 6, tools.map((x) => x.name).join(","));
  } finally {
    await client.close();
  }
});
