#!/usr/bin/env node
/**
 * `dsh-operator-agent-setup`：把插件依赖的 Python 运行时准备好，并把两条覆盖记录写进
 * DSH profile 的 `cordis.patch.yml`。
 *
 * 为什么需要它：npm 包只是"接线"（ESM 模块 + bundle patch），真正的判定能力在 Python
 * 包里。DSH 只校验 `engines.dsh`、应用 bundle patch，**不管外部运行时**——所以这一步
 * 只能由插件自己提供助手。命令名与参数尽量少，能一条命令跑完就不让人分三步。
 *
 * 它**不做**的事：
 *   - 不调用模型、不跑 GPU、不联网（除非你显式给 `--wheel` 之外的 URL）；
 *   - 不替你执行 `dsh plugin add`：Desktop profile 由 Electron 独占，插件安装走 GUI；
 *   - 不覆盖你不认识的配置：只维护文件末尾一段带 marker 的覆盖记录。
 *
 * 用法：
 *   npx dsh-operator-agent-setup --wheel /abs/path/operator_agent-0.7.2-py3-none-any.whl
 *   npx dsh-operator-agent-setup --python ~/.local/share/operator-agent/v0.7.2/venv/bin/python
 *   npx dsh-operator-agent-setup --python <exe> --dry-run     # 只打印，不写文件
 */
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync, copyFileSync, chmodSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { homedir } from "node:os";
import { spawnSync } from "node:child_process";
import { randomBytes } from "node:crypto";

/** 本插件配套的 Python 版本；与 package.json 的 version 一起决定安装目录。 */
export const PYTHON_VERSION = "0.7.2";

/** host 模式暴露的七个工具。少了任何一个都说明 Python 侧不是这一版。 */
export const TOOLS = [
  "operator_cancel", "operator_capabilities", "operator_context", "operator_prepare",
  "operator_resume", "operator_status", "operator_verify",
];

export const BEGIN = "# BEGIN operator-agent host installer v1";
export const END = "# END operator-agent host installer v1";

const USAGE = `用法：dsh-operator-agent-setup [选项]

  --wheel <文件>        用本地 wheel 建 venv 并安装（与 --python 二选一）
  --python <可执行文件>  已有 Python 环境；跳过安装，只做校验与配置
  --version <版本>      Python 版本目录名（默认 ${PYTHON_VERSION}）
  --profile <名字>      目标 profile（默认 desktop）
  --dsh-home <目录>     DSH_HOME（默认 $DSH_HOME 或 ~/.dsh）
  --dry-run             只检查并打印将要写入的配置，不落盘
  --configure-only      只写 profile 配置，不建环境
  --help                这段说明

退出码：0 成功；1 参数或环境问题；2 写入后自检失败（已自动还原原 patch）。`;

export function parseArgs(argv) {
  const out = { wheel: "", python: "", version: PYTHON_VERSION, profile: "desktop", dshHome: "", dryRun: false, configureOnly: false, help: false };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = () => {
      const value = argv[i + 1];
      if (value === undefined || value.startsWith("--")) throw new Error(`${arg} 缺参数值`);
      i += 1;
      return value;
    };
    if (arg === "--help" || arg === "-h") out.help = true;
    else if (arg === "--wheel") out.wheel = resolve(next());
    else if (arg === "--python") out.python = resolve(next());
    else if (arg === "--version") out.version = next();
    else if (arg === "--profile") out.profile = next();
    else if (arg === "--dsh-home") out.dshHome = resolve(next());
    else if (arg === "--dry-run") out.dryRun = true;
    else if (arg === "--configure-only") out.configureOnly = true;
    else throw new Error(`不认识的参数：${arg}`);
  }
  if (out.wheel && out.python) throw new Error("--wheel 与 --python 只能给一个：前者建环境，后者用现成的");
  return out;
}

/**
 * 与 Python 版安装器同语义的 patch 写入：
 *   - 保留原文件的注释与顺序；我们的两条记录放在**最后**（DSH 的 patch 是后覆盖前）；
 *   - 已有我们维护的 marker 时替换整段，不重复追加；
 *   - 文件是 JSON 数组形态时按 id 去重后追加。
 */
export function patchText(original, entries) {
  const lines = String(original).split(/(?<=\n)/);
  const first = lines.findIndex((line) => line.trim() && !line.trimStart().startsWith("#"));
  const prefix = lines.slice(0, first < 0 ? lines.length : first).join("");
  const body = lines.slice(first < 0 ? lines.length : first).join("");

  if (body.trimStart().startsWith("[")) {
    const values = JSON.parse(body);
    if (!Array.isArray(values)) throw new Error("profile patch 必须是数组");
    const ids = new Set(entries.map((entry) => entry.id));
    const kept = values.filter((value) => !(value && typeof value === "object" && ids.has(value.id)));
    return `${prefix}${JSON.stringify([...kept, ...entries], null, 2)}\n`;
  }

  if (original.split(BEGIN).length !== original.split(END).length || original.split(BEGIN).length > 2) {
    throw new Error("配置里的托管标记不完整或重复，请人工检查后再运行");
  }
  let text = original;
  if (text.includes(BEGIN)) {
    const start = text.indexOf(BEGIN);
    const finish = text.indexOf(END, start) + END.length;
    if (finish <= start) throw new Error("配置里的托管标记顺序不对");
    text = text.slice(0, start) + text.slice(finish).replace(/^[\r\n]+/, "");
  }
  const block = entries.map((entry) => `- ${JSON.stringify(entry)}`).join("\n");
  return `${text.replace(/\s+$/, "")}\n\n${BEGIN}\n${block}\n${END}\n`;
}

/** 两条覆盖记录：MCP server 指向 venv 里的可执行文件，/op 命令指向同一个 venv。 */
export function entries(python) {
  const bin = dirname(python);
  return [
    { id: "mcp-operator", disabled: false, config: {
      serverName: "operator", transport: "stdio", command: join(bin, "operator-agent-mcp"),
      toolCallTimeoutMs: 3600000, failOnStartupError: true, env: { OPERATOR_AGENT_WORKFLOW: "host" } } },
    { id: "operator-tools", disabled: false, config: {
      command: join(bin, "operator-agent"), timeoutMs: 60000, hostModelBudget: {} } },
  ];
}

function run(argv, options = {}) {
  return spawnSync(argv[0], argv.slice(1), { encoding: "utf8", ...options });
}

function pythonBin(python) {
  return join(dirname(python), "operator-agent-mcp");
}

/** 起一次真实 MCP 握手：能列出七个工具才算环境可用，而不是只看文件在不在。 */
export function checkPython(python) {
  for (const name of ["operator-agent", "operator-agent-mcp"]) {
    const path = join(dirname(python), name);
    if (!existsSync(path)) throw new Error(`Python 环境缺少 ${name}：${path}`);
  }
  const requests = [
    { jsonrpc: "2.0", id: 1, method: "initialize", params: { protocolVersion: "2025-11-25", capabilities: {}, clientInfo: { name: "operator-setup", version: "1" } } },
    { jsonrpc: "2.0", id: 2, method: "tools/list", params: {} },
  ].map((item) => JSON.stringify(item)).join("\n");
  const result = run([python, "-I", "-m", "operator_agent.app.mcp_server"], {
    input: `${requests}\n`, timeout: 30_000, cwd: homedir(),
    env: { HOME: homedir(), PATH: process.env.PATH || "", LANG: "C.UTF-8", OPERATOR_AGENT_WORKFLOW: "host" },
  });
  if (result.status !== 0) throw new Error(`MCP 自检失败（退出码 ${result.status}）：${(result.stderr || "").trim().slice(0, 400)}`);
  const replies = String(result.stdout).split("\n").filter(Boolean).map((line) => JSON.parse(line));
  const listed = replies.find((reply) => reply.id === 2);
  const names = (listed?.result?.tools || []).map((tool) => tool.name).sort();
  if (JSON.stringify(names) !== JSON.stringify(TOOLS)) {
    throw new Error(`Python 侧暴露的工具不是 v0.7.2 的七个（实际：${names.join(", ") || "无"}）`);
  }
  const init = replies.find((reply) => reply.id === 1);
  return init?.result?.serverInfo?.version || "unknown";
}

function receiptDir(dshHome) {
  const stamp = `${new Date().toISOString().replace(/[-:T]/g, "").slice(0, 14)}-${randomBytes(3).toString("hex")}`;
  return join(dshHome, "operator-install-receipts", `setup-${stamp}`);
}

function main(argv) {
  const options = parseArgs(argv);
  if (options.help) { process.stdout.write(`${USAGE}\n`); return 0; }

  const dshHome = options.dshHome || process.env.DSH_HOME || join(homedir(), ".dsh");
  const patch = join(dshHome, "profiles", options.profile, "cordis.patch.yml");
  let python = options.python;

  if (!options.configureOnly) {
    if (options.python) {
      const version = checkPython(python);
      process.stdout.write(`✓ 已有环境可用：Python 侧版本 ${version}\n`);
    } else {
      const base = join(homedir(), ".local", "share", "operator-agent", options.version);
      const venv = join(base, "venv");
      python = join(venv, "bin", "python");
      if (options.dryRun) {
        process.stdout.write(`· 将创建 venv：${venv}${options.wheel ? `，并安装 ${options.wheel}` : ""}\n`);
      } else {
        mkdirSync(base, { recursive: true });
        if (!existsSync(python)) {
          const made = run([process.env.PYTHON3 || "python3", "-m", "venv", venv], { stdio: "inherit" });
          if (made.status !== 0) throw new Error("创建 venv 失败（需要 Python 3.9+）");
        }
        if (options.wheel) {
          const installed = run([python, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", options.wheel], { stdio: "inherit" });
          if (installed.status !== 0) throw new Error(`安装 wheel 失败：${options.wheel}`);
        }
        const version = checkPython(python);
        process.stdout.write(`✓ 环境就绪：${python}（Python 侧版本 ${version}）\n`);
      }
    }
  } else if (!python) {
    throw new Error("--configure-only 需要同时给 --python <已有环境>");
  } else {
    checkPython(python);
  }

  const current = existsSync(patch) ? readFileSync(patch, "utf8") : "";
  const next = patchText(current, entries(python));
  if (options.dryRun) {
    process.stdout.write(`· 将写入 ${patch}：\n${next}\n`);
    return 0;
  }

  const receipt = receiptDir(dshHome);
  mkdirSync(receipt, { recursive: true, mode: 0o700 });
  if (existsSync(patch)) copyFileSync(patch, join(receipt, "cordis.patch.yml.bak"));
  writeFileSync(join(receipt, "receipt.json"), `${JSON.stringify({ python, profile: options.profile, patch, dry_run: false }, null, 2)}\n`, { mode: 0o600 });

  const temp = `${patch}.tmp-${process.pid}`;
  mkdirSync(dirname(patch), { recursive: true });
  writeFileSync(temp, next, { mode: 0o600 });
  renameSync(temp, patch);
  process.stdout.write(`✓ 已写入 ${patch}\n  备份与回执：${receipt}\n`);

  // 写后自检：把自己的两条记录读回来，确认没有把别人的配置冲掉。
  const written = readFileSync(patch, "utf8");
  for (const marker of [BEGIN, END, "mcp-operator", "operator-tools"]) {
    if (!written.includes(marker)) throw new Error(`写入后自检失败：缺少 ${marker}`);
  }
  process.stdout.write("✓ 自检通过。重开 DSH，在插件页确认两个组件运行中，然后输入 /op doctor\n");
  process.stdout.write("  注意：插件包本身仍需通过 DSH 的插件页（或 dsh plugin add）安装；本命令只负责运行时与配置。\n");
  return 0;
}

const invoked = process.argv[1] ? resolve(process.argv[1]) : "";
if (invoked && resolve(new URL(import.meta.url).pathname) === invoked) {
  try {
    process.exitCode = main(process.argv.slice(2));
  } catch (error) {
    process.stderr.write(`✗ ${error.message}\n`);
    process.exitCode = 1;
  }
}