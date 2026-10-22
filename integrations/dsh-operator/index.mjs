/**
 * DSH 宿主插件：斜杠命令 + 系统提示段。
 *
 * 它**不执行判定**——判定永远在 Python 侧（`operator-agent` CLI / MCP server）。
 * 这个模块只做两件宿主侧的事：
 *   1. `/op`：让人不经过模型就能查状态/能力（输出直接给人看，不进模型上下文）；
 *   2. 一段 system prompt：告诉模型 `mcp__operator__*` 工具什么时候该用、回来怎么读。
 *
 * 为什么命令走 CLI 而不是 MCP：斜杠命令在**宿主进程**里执行，拿不到 MCP 的客户端连接；
 * 而 `operator-agent` 是同一个判定层的进程入口，参数与 MCP 工具同源。
 */

import { execFile } from "node:child_process";
import { homedir } from "node:os";
import { join, resolve as resolvePath } from "node:path";

export const name = "operator-tools";
export const inject = ["commands", "systemPrompt"];

const USAGE = [
  "用法：",
  "  /op capabilities            这台机器能做什么（只读，不跑活）",
  "  /op status <run_dir>        查一次无人值守运行的相位（running/done/failed/lost）",
  "  /op help                    这段说明",
  "",
  "run_dir 可以相对写（按**会话工作区**解析），也可以绝对写；含空格请加引号。",
  "",
  "判定/修复/无人值守请用模型工具 mcp__operator__*：",
  "  prepare → verify（外层写核）· repair（给一份已有的核）· run + status（无人值守）",
].join("\n");

const PROMPT_SECTION = [
  "算子判定层（MCP 工具 `mcp__operator__*`）用于 GPU 算子（Triton/CUDA）的编译、正确性、性能判定与修复。",
  "什么时候用：用户要把一个算子写成/改好/测快，或问「这个核对不对/快不快/读数可不可信」时。",
  "什么时候**不要**用：纯 CPU/非算子任务；没有 GPU 却要做性能结论（`backend=mock` 的读数不代表加速卡成绩）。",
  "典型顺序：`operator_prepare` 先冻结契约（声明 shapes/precision/required_caps），再 `operator_verify` 判候选；",
  "已有核要修就用 `operator_repair`；从头自动化用 `operator_run` + `operator_status` 轮询。",
  "读回包时：`levels` 看判到哪一级；**`route` 与 `eligibility` 必须一起读** —— 两者冲突时 `conflict.action=hold_for_measurement`，",
  "此时不要照着 `route` 迭代，先修判据（形状/精度/required_caps）。`validity=degraded/uninformative` 的读数不驱动调参或入库。",
].join("");

/**
 * 解析 `/op` 后面的输入。**`rest` 是原样的剩余整段**（不是按空白切开的数组）：
 * run_dir 可能是含空格的路径，切开就再也拼不回来。
 */
function parseInput(rawInput) {
  const text = String(rawInput || "").trim();
  const match = /^(\S+)\s*([\s\S]*)$/.exec(text);
  if (!match) return { sub: "help", rest: "" };
  return { sub: match[1].toLowerCase(), rest: match[2].trim() };
}

/** 去掉一对匹配的引号（含空格路径的写法），不做其它转义处理。 */
function stripQuotes(text) {
  const pair = /^(["'])([\s\S]*)\1$/.exec(text);
  return pair ? pair[2] : text;
}

/**
 * 洗掉敏感环境变量，规则**与宿主给 MCP 子进程用的那套一致**：
 * 名字含 `KEY`/`PASSWORD`/`SECRET`/`TOKEN`，或带 `DSH_` 前缀。
 *
 * 为什么 `/op` 也要洗：`command` 是可配置的，等于"把宿主环境交给一个可替换的二进制"。
 * 显式 `config.env` 在清洗**之后**合并，所以真要传什么仍传得进去（见 `normalizeConfig`）。
 */
export function scrubEnv(source = process.env) {
  const SENSITIVE = /KEY|PASSWORD|SECRET|TOKEN/i;
  const out = {};
  for (const [key, value] of Object.entries(source || {})) {
    if (key.startsWith("DSH_") || SENSITIVE.test(key)) continue;
    out[key] = value;
  }
  return out;
}

const KNOWN_CONFIG_KEYS = new Set(["command", "args", "timeoutMs", "env"]);

/**
 * 配置校验。**不校验的代价是静默**：`comand:` 拼错会让 `/op` 悄悄换成另一个二进制，
 * `timeoutMs: "30s"` 会悄悄变回默认值——两种都只在事后从行为上才看得出来。
 *
 * 只对"写了但不对"说话；没写的键走默认值，不吵。`warn` 由调用方注入（测试直接收）。
 */
export function normalizeConfig(config = {}, warn = () => {}) {
  const raw = config && typeof config === "object" && !Array.isArray(config) ? config : {};
  if (raw !== config) warn("配置必须是对象，已全部改用默认值");
  for (const key of Object.keys(raw)) {
    if (KNOWN_CONFIG_KEYS.has(key)) continue;
    const hint = key.toLowerCase() === "timeoutms" ? "；正确写法是 timeoutMs" : "";
    warn(`配置项 ${key} 不认识（拼错了？），已忽略${hint}`);
  }

  let command = "operator-agent";
  if (raw.command !== undefined) {
    if (typeof raw.command === "string" && raw.command.trim()) command = raw.command;
    else warn(`command 必须是非空字符串（收到 ${JSON.stringify(raw.command)}），改用 ${command}`);
  }

  let args = [];
  if (raw.args !== undefined) {
    if (Array.isArray(raw.args) && raw.args.every((a) => typeof a === "string")) {
      args = [...raw.args];
    } else warn(`args 必须是字符串数组（收到 ${JSON.stringify(raw.args)}），已忽略`);
  }

  let timeoutMs = 60_000;
  if (raw.timeoutMs !== undefined) {
    if (typeof raw.timeoutMs === "number" && Number.isFinite(raw.timeoutMs) && raw.timeoutMs > 0) {
      timeoutMs = raw.timeoutMs;
    } else warn(`timeoutMs 必须是正的毫秒数（收到 ${JSON.stringify(raw.timeoutMs)}），改用 ${timeoutMs}`);
  }

  let env;
  if (raw.env !== undefined) {
    const values = raw.env && typeof raw.env === "object" && !Array.isArray(raw.env)
      ? Object.values(raw.env) : undefined;
    if (values && values.every((v) => typeof v === "string")) env = { ...raw.env };
    else warn(`env 必须是「字符串 → 字符串」的对象（收到 ${JSON.stringify(raw.env)}），已忽略`);
  }

  return { command, args, timeoutMs, env };
}

/** 跑一次 CLI，**永不抛**：命令不存在/超时/被取消/非零退出都折成结构化结果。 */
export function runCli(command, argv, timeoutMs = 60_000, env = undefined, signal = undefined) {
  return new Promise((resolve) => {
    const done = (err, stdout, stderr) => {
      const text = String(stdout || "").trim();
      const errText = String(stderr || "").trim();
      // 取消要排在超时前面：abort 也会把子进程杀掉，`err.killed` 同样为 true。
      if (err && err.name === "AbortError") {
        resolve({ ok: false, reason: "cancelled", text, stderr: errText,
                  detail: "命令被取消" });
        return;
      }
      if (err && err.code === "ENOENT") {
        resolve({ ok: false, reason: "not_installed", text: "", stderr: errText,
                  detail: `找不到命令 ${command}（pip install operator-agent？）` });
        return;
      }
      if (err && err.killed) {
        resolve({ ok: false, reason: "timeout", text, stderr: errText,
                  detail: `命令超时（${timeoutMs} ms）` });
        return;
      }
      const code = err && typeof err.code === "number" ? err.code : (err ? -1 : 0);
      resolve({ ok: true, reason: "", code, text, stderr: errText, detail: "" });
    };
    try {
      execFile(command, argv, { timeout: timeoutMs, maxBuffer: 8 << 20, env, signal }, done);
    } catch (err) {
      // 已经 abort 的 signal 可能在这里同步抛 —— 同样折成 cancelled，不外泄。
      done(err, "", "");
    }
  });
}

/**
 * 判定层的**拒绝**回包（`ok:false`）不是成功：原因进正文，kind 走 error。
 * 不这样折的话，`capabilities` 会打出一堆「?」，`status` 会把「查不到」显示成绿行。
 */
function rejection(payload) {
  return { kind: "error", text: `判定层拒绝：${payload.detail || payload.error || "未知原因"}` };
}

/** 相位与进程死活是**两件事**：只有 `running` 才谈「在跑」。 */
function processNote(phase, alive) {
  if (phase === "running") return alive === false ? "进程已退出" : "进程在跑";
  if (phase === "lost") return "进程已退出（没了但没写终态）";
  if (phase === "done" || phase === "failed") return "进程已结束";
  return "进程状态未知";
}

export function formatCapabilities(payload) {
  const backends = payload.backends || {};
  const wired = Object.keys(backends).filter((k) => backends[k] && backends[k].wired);
  const runtime = payload.runtime || {};
  const layouts = (payload.kernel_types_with_input_layout || []).join(", ");
  return [
    `算子判定层 ${payload.version || "?"}（tool ${payload.tool_version ?? "?"}）`,
    `已接线后端：${wired.join(" / ") || "无"}`,
    `运行时：python ${runtime.python || "?"} · torch ${runtime.torch_installed ? "有" : "无"} · triton ${runtime.triton_installed ? "有" : "无"}`,
    `有输入布局的算子：${layouts || "无"}`,
  ].join("\n");
}

/**
 * 只吃**成功**回包（`ok:true`）；拒绝回包由 {@link rejection} 折，别在这里猜。
 * `runDir` 可选：传了就回显**实际读的那个路径**——相对路径解析出问题时，这是唯一线索。
 */
export function formatStatus(payload, runDir = undefined) {
  // 终态下 Python 侧给的 process_alive 是 null（不是 false）——按相位判，别按死活判，
  // 否则一个跑完的 run 会显示成「进程 在跑」。
  const lines = [];
  if (runDir) lines.push(`目录：${runDir}`);
  lines.push(`相位：${payload.phase}（${processNote(payload.phase, payload.process_alive)}）`);
  if (payload.started_at) lines.push(`开始：${new Date(payload.started_at * 1000).toISOString()}`);
  if (payload.duration_s != null) lines.push(`耗时：${payload.duration_s} s`);
  const summary = payload.summary;
  if (summary) {
    const levels = Object.entries(summary.levels || {}).map(([k, v]) => `${k}${v ? "✓" : "✗"}`).join(" ");
    lines.push(`结果：${summary.all_passed ? "全部通过" : "未通过"}（${levels || "未判"}）`);
    if (summary.speedup != null) lines.push(`加速比：${Number(summary.speedup).toFixed(4)}x · 档位 ${summary.validity || "未知"}`);
  }
  if (payload.error) lines.push(`错误：${payload.error} ${payload.detail || ""}`.trim());
  const tail = (payload.log_tail || []).filter(Boolean).slice(-1)[0];
  if (tail) lines.push(`日志尾：${tail}`);
  return lines.join("\n");
}

export function apply(ctx, config = {}) {
  // 宿主日志里留痕（`ctx.logger` 是 cordis 核心服务，不用 inject）。**同一件事只说一次**：
  // 配置错一次就够，命令没装/超时会被反复撞上，都不该刷屏；`said` 随插件实例生灭。
  const said = new Set();
  const say = (level, key, message) => {
    if (said.has(key)) return;
    said.add(key);
    ctx.logger[level](`operator-tools: ${message}`);
  };

  const { command, args: prefix, timeoutMs, env: extraEnv } = normalizeConfig(
    config, (message) => say("warn", `config:${message}`, message));
  // 清洗在**前**、显式 `env` 在**后**：默认不把宿主密钥交给子进程，但配置里写了就照传。
  // `env` 也用于「没 pip install、直接从源码树跑」的形态（PYTHONPATH 指到仓库）。
  const env = { ...scrubEnv(process.env), ...(extraEnv || {}) };
  const run = (argv, signal) => runCli(command, [...prefix, ...argv], timeoutMs, env, signal);

  // 取消是人的正常操作，不进日志；「命令不存在」是插件根本没接线，按 error 记。
  const fail = (out) => {
    if (out.reason !== "cancelled") {
      const level = out.reason === "not_installed" ? "error" : "warn";
      say(level, out.reason, `${out.detail}（同类失败只记这一次）`);
    }
    return { kind: "error", text: out.detail };
  };

  /**
   * 相对路径按**会话工作区**解析。宿主进程的 cwd 不是人的工作区（GUI 进程多半在 `/`），
   * 不解析就会报「run-status.json 不存在」，看着像 run 丢了。
   * `workingDirectory` 没装/查不到时原样传下去——不因为可选依赖把整条命令变成不可用。
   */
  const resolveRunDir = (raw, agent) => {
    const text = stripQuotes(String(raw || "").trim());
    if (!text) return "";
    if (text === "~" || text.startsWith("~/")) return join(homedir(), text.slice(2));
    if (text.startsWith("/")) return text;
    const wd = typeof ctx.get === "function" ? ctx.get("workingDirectory") : undefined;
    const cwd = wd && agent ? wd.get(agent.session) : undefined;
    return cwd ? resolvePath(cwd, text) : text;
  };

  const handle = async (rawInput, signal, agent) => {
    const { sub, rest } = parseInput(rawInput);
    if (sub === "help" || sub === "-h" || sub === "--help") {
      return { kind: "success", text: USAGE };
    }
    if (sub === "capabilities") {
      const out = await run(["capabilities"], signal);
      if (!out.ok) return fail(out);
      try {
        const payload = JSON.parse(out.text);
        if (payload.ok === false) return rejection(payload);
        return { kind: "success", text: formatCapabilities(payload) };
      } catch {
        say("warn", "bad_output", `返回的不是 JSON：${out.text.slice(0, 120)}`);
        return { kind: "error", text: `返回的不是 JSON：\n${out.text.slice(0, 400)}` };
      }
    }
    if (sub === "status") {
      if (!rest) return { kind: "error", text: "用法：/op status <run_dir>" };
      const runDir = resolveRunDir(rest, agent);
      const out = await run(["status", "--run-dir", runDir], signal);
      if (!out.ok) return fail(out);
      try {
        const payload = JSON.parse(out.text);
        if (payload.ok === false) return rejection(payload);
        return { kind: "success", text: formatStatus(payload, runDir) };
      } catch {
        say("warn", "bad_output", `返回的不是 JSON：${out.text.slice(0, 120)}`);
        return { kind: "error", text: `返回的不是 JSON：\n${out.text.slice(0, 400)}` };
      }
    }
    return { kind: "error", text: `不认识的子命令 ${sub}\n\n${USAGE}` };
  };

  // 生命周期归 cordis：service tracker 让 register/section 的 effect 挂在**调用方 ctx** 上，
  // 所以卸载/HMR 会自动回滚，这里不需要（也不该）自己攒 disposer。
  ctx.commands.register({
    definitionId: "dsh-operator-agent",
    name: "op",
    description: "算子判定层：查能力 / 查无人值守状态（不经过模型）",
    input: { hint: "capabilities | status <run_dir> | help" },
    // `agent` 是宿主给的（invocation = {commandId, agent, rawInput, attachments, signal}），
    // 只有它带着会话工作区，相对路径要靠它才能落到人想去的地方。
    handler: async ({ rawInput, signal, agent }) => handle(rawInput, signal, agent),
  });

  // `inject` 是**必需**语义（缺了 fiber 根本不启动），所以这里不做「拿不到就跳过」的兜底 ——
  // 那种分支永远不会被执行，只会让人以为 systemPrompt 是可选的。
  ctx.systemPrompt.section({ name: "tool:operator", order: 160, text: PROMPT_SECTION });
}
