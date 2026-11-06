/**
 * DSH 宿主插件：斜杠命令 `/op`。
 *
 * 它**不执行判定**——判定永远在 Python 侧（`operator-agent` CLI / MCP server）。
 * 默认提供人读的能力、状态和取消命令；启用 modelRelay 时，宿主也可以代发内层模型请求。
 *
 * 「什么时候该用 `operator_*` 工具、回来怎么读」的那段说明**不在这里**：它是 MCP
 * server 的 `initialize.instructions`（`operator_agent/mcp_server.py`），宿主会自动
 * 渲染成 `mcp:operator` 提示段。放那儿的好处是只有一处来源、换任何 MCP 客户端都在；
 * 放这儿就得跟着工具描述一起手动同步。
 *
 * 为什么命令走 CLI 而不是 MCP：斜杠命令在**宿主进程**里执行，拿不到 MCP 的客户端连接；
 * 而 `operator-agent` 是同一个判定层的进程入口，参数与 MCP 工具同源。
 */

import { execFile } from "node:child_process";
import { homedir } from "node:os";
import { join, resolve as resolvePath } from "node:path";
import { startModelRelay } from "./model-relay.mjs";
import { installHostModelBudget } from "./host-model-budget.mjs";

export const name = "operator-tools";
export const inject = ["commands"];

const USAGE = [
  "✓ 已执行 · /op help",
  "用法：",
  "  /op capabilities            这台机器能做什么（只读，不跑活）",
  "  /op new [run_dir]            在本会话显式开启下一项独立任务（不调用模型）",
  "  /op status                  查本会话的任务/讨论状态",
  "  /op status <run_dir> [invocation_id]  查任务/工具调用（含 cancelled）",
  "  /op cancel <run_dir> <invocation_id>  请求取消，再查 status 确认",
  "  /op help                    这段说明",
  "",
  "run_dir 可以相对写（按**会话工作区**解析），也可以绝对写；含空格请加引号。",
  "",
  "输出直接显示在对话里（一个可点开折叠的节点），**不进模型上下文**；",
  "折叠时只显示首行（以 ✓ 开头的那行），点它展开看完整明细。这是旁路查询，不产生轮次。",
  "",
  "开发请用模型工具 mcp__operator__*：",
  "  host: prepare → context → 外层写核 → verify；resume 恢复原任务",
  "  legacy: repair（已有核）· run + status（内层无人值守）",
].join("\n");


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

const KNOWN_CONFIG_KEYS = new Set(["command", "args", "timeoutMs", "env", "modelRelay", "hostModelBudget"]);

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
 *
 * `✗` 前缀是**回执**：折叠态只显示首行，没有它的话「拒绝」和「成功」在收起时长得一样，
 * 都是命令名加一段摘要。成功用 `✓`（在各自的 format* 里），失败用 `✗`，一眼能分。
 */
function rejection(payload) {
  return { kind: "error", text: `✗ 判定层拒绝 · ${payload.detail || payload.error || "未知原因"}` };
}

/** 相位与进程死活是**两件事**：只有 `running` 才谈「在跑」。 */
function processNote(phase, alive) {
  if (phase === "running") return alive === false ? "进程已退出" : "进程在跑";
  if (phase === "lost") return "进程已退出（没了但没写终态）";
  if (phase === "done" || phase === "failed" || phase === "cancelled") return "进程已结束";
  return "进程状态未知";
}

export function formatCapabilities(payload) {
  const backends = payload.backends || {};
  const wired = Object.keys(backends).filter((k) => backends[k] && backends[k].wired);
  const runtime = payload.runtime || {};
  const kernelTypes = payload.kernel_types_with_input_layout || [];
  const layouts = kernelTypes.join(", ");
  // 首行是**折叠态唯一的可见内容**：宿主 GenericCommandCard 的摘要是
  // `white-space:nowrap; text-overflow:ellipsis` 的单行（不是折行），所以首行放不下的
  // 部分会被直接截掉、连省略号之后都看不见。两个约束因此同时成立：
  //   1) 回执必须在前（`✓ ` 开头），否则「跑没跑」这个判断又被挤到第二行看不见的地方；
  //   2) 摘要要短，且**把低频信息（tool 号）挪到明细**，给回执腾位置。
  // 明细留在后面的行里，点开才看（见 README §5.5）。
  return [
    `✓ 已执行 · 算子判定层 ${payload.version || "?"} · 后端 ${wired.join("/") || "无"} · ${kernelTypes.length} 个算子有输入布局`,
    `工具版本：tool ${payload.tool_version ?? "?"}（回包契约版本，与判定层版本不是一回事）`,
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
  if (payload.workflow === "host") {
    const budget = payload.host_task || {};
    const count = (value) => Number.isFinite(value) && value >= 0 ? value : "未知";
    const lines = [`✓ 查询完成 · 宿主任务 ${payload.phase}`];
    if (runDir) lines.push(`目录：${runDir}`);
    lines.push(`验证派发：${count(budget.reserved)} / ${count(budget.limits?.max_verifications)}`);
    lines.push(`探索剩余：${count(budget.remaining_exploration)} · 独立终评剩余：${count(budget.remaining_final)}`);
    lines.push(`任务剩余：${count(budget.remaining_s)} s`);
    if (budget.development_closed) lines.push("开发已关闭，以独立终评证据确认结果");
    const correct = payload.last_correct_candidate;
    if (correct) lines.push(`最近完整 L2 候选：${correct.source_hash}（不代表性能最佳）`);
    const model = budget.model_budget;
    if (model?.receipt_verified && model.enforcement === "host_enforced_receipt") {
      lines.push(`模型派发：${count(model.calls?.reserved)} / ${count(model.calls?.limit)} · 剩余 ${count(model.calls?.remaining)}`);
      lines.push(`输出 token 预留：${count(model.tokens?.output_reserved)} / ${count(model.tokens?.reservation_limit)}`);
      lines.push("范围：任务绑定后的主模型请求；准备/辅助调用不在此限额，费用未知");
    } else {
      lines.push(`模型预算：${model?.enforcement ?? "未绑定或未知"}（没有有效宿主收据）`);
    }
    if (correct?.source) lines.push(`源码：${correct.source}`);
    if (correct?.report) lines.push(`报告：${correct.report}`);
    if (!correct && payload.latest_verification) {
      lines.push("最近验证尚未取得完整 L2 资格，保留源码与失败报告");
      if (payload.latest_verification.source) lines.push(`源码：${payload.latest_verification.source}`);
      if (payload.latest_verification.report) lines.push(`报告：${payload.latest_verification.report}`);
    }
    for (const call of payload.calls || []) {
      if (["running", "cancelling", "lost"].includes(call.phase) && call.invocation_id) {
        lines.push(`工具实例：${call.invocation_id} · ${call.phase}`);
      }
    }
    return lines.join("\n");
  }
  if (payload.phase === "tool_calls") {
    return [`✓ 查询完成 · ${payload.calls?.length ?? 0} 次工具调用`,
      ...(payload.calls || []).map((call) => `${call.invocation_id} · ${call.phase}`)].join("\n");
  }
  // 终态下 Python 侧给的 process_alive 是 null（不是 false）——按相位判，别按死活判，
  // 否则一个跑完的 run 会显示成「进程 在跑」。
  //
  // 首行同 `formatCapabilities`：折叠态只显示这一行（摘要 nowrap 截断），所以回执 +
  // 相位必须在最前。「查到了」和「查到什么」是两件事，首行两件都说。
  // 相位只写一次：首行已经带着它了，下面不再重复（重复会让展开态读起来像回声）。
  const summary = payload.summary;
  const stoppedReason = payload.stopped_reason || summary?.stopped_reason;
  const budgetStopped = payload.phase === "done" && typeof stoppedReason === "string"
    && (stoppedReason.startsWith("budget_") || stoppedReason === "wall_clock_exhausted");
  const phaseNote = `相位 ${payload.phase}（${processNote(payload.phase, payload.process_alive)}）`;
  const lines = [`✓ 查询完成 · ${phaseNote}${budgetStopped ? " · 预算停止" : ""}`];
  if (runDir) lines.push(`目录：${runDir}`);
  if (payload.invocation_id) lines.push(`任务实例：${payload.invocation_id}`);
  if (payload.cancellation) lines.push("取消：已请求（以相位确认最终结果）");
  if (payload.cleanup) {
    lines.push(`本机清理：${payload.cleanup.status ?? "未知"}`);
    lines.push(`远端执行：${payload.cleanup.remote_execution ?? "未知"} · 服务商调用：${payload.cleanup.provider_call ?? "未知"}`);
  }
  if (payload.started_at) lines.push(`开始：${new Date(payload.started_at * 1000).toISOString()}`);
  if (payload.duration_s != null) lines.push(`耗时：${payload.duration_s} s`);
  if (stoppedReason) lines.push(`停止原因：${stoppedReason}`);
  const budget = payload.budget ?? summary?.budget;
  const taskBudget = payload.task_budget ?? summary?.task_budget ?? budget?.limits;
  if (budget != null || taskBudget != null) {
    // Missing/inconsistent journals are unknown usage, never zero credit.
    const count = (value) => Number.isInteger(value) && value >= 0 ? String(value) : "未知";
    const seconds = (value) => typeof value === "number" && Number.isFinite(value) && value >= 0
      ? String(value) : "未知";
    const wall = budget?.wall_clock;
    const calls = budget?.model_calls;
    const verification = budget?.verification;
    lines.push(`任务墙钟：剩余 ${seconds(wall?.remaining_s)} s / 上限 ${seconds(wall?.limit_s ?? taskBudget?.wall_clock_s)} s`);
    lines.push(`模型调用：预留 ${count(calls?.reserved)} / 上限 ${count(calls?.limit ?? taskBudget?.max_llm_calls)} · 剩余 ${count(calls?.remaining)}`);
    lines.push(`验证输入组：预留 ${count(verification?.spent_all)} / 上限 ${count(verification?.plan?.oracle_total ?? taskBudget?.oracle_total)} · 剩余 ${count(verification?.remaining?.total)}`);
    lines.push(`最终确认：预留 ${count(verification?.plan?.phase_caps?.confirm)} · 已用 ${count(verification?.spent?.confirm)} · 剩余 ${count(verification?.remaining?.confirm)}`);
  }
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
  let hostBudget;
  if (config?.hostModelBudget !== undefined && config.hostModelBudget !== false) {
    ctx.inject(["llm", "tools", "agents", "sessionQuery"], async function operatorHostBudget(hostCtx) {
      await hostCtx.effect(() => {
        const controller = installHostModelBudget(hostCtx, config.hostModelBudget);
        hostBudget = controller;
        return () => { if (hostBudget === controller) hostBudget = undefined; controller(); };
      }, "operator-tools.hostModelBudget");
    });
  }
  const relayConfig = config?.modelRelay;
  if (relayConfig !== undefined) {
    // The opt-in relay follows the host LLM service lifecycle. /op only requires commands.
    ctx.inject(["llm"], async function operatorModelRelay(relayCtx) {
      await relayCtx.effect(() => startModelRelay(relayCtx, relayConfig), "operator-tools.modelRelay");
    });
  }

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
  // `✗` 前缀同 rejection：折叠态只显示首行，失败回执必须自己说出来（取消不算失败，不挂 ✗）。
  const fail = (out) => {
    if (out.reason !== "cancelled") {
      const level = out.reason === "not_installed" ? "error" : "warn";
      say(level, out.reason, `${out.detail}（同类失败只记这一次）`);
    }
    const mark = out.reason === "cancelled" ? "已取消" : `✗ 未执行 · ${out.reason}`;
    return { kind: "error", text: `${mark}\n${out.detail}` };
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

  const handle = async (rawInput, signal, agent, commandId) => {
    const { sub, rest } = parseInput(rawInput);
    if (sub === "help" || sub === "-h" || sub === "--help") {
      return { kind: "success", text: USAGE };
    }
    if (sub === "new" || (sub === "status" && !rest && config.hostModelBudget !== undefined && config.hostModelBudget !== false)) {
      if (!hostBudget) return { kind: "error", text: "✗ 任务入口尚未就绪：请确认 hostModelBudget 已启用，或等待插件加载后再试。" };
      try {
        const text = sub === "new"
          ? await hostBudget.newTask({ agent, commandId, signal, ...(rest ? { runDir: resolveRunDir(rest, agent) } : {}) })
          : await hostBudget.sessionStatus({ agent, signal });
        return { kind: "success", text };
      } catch (error) { return { kind: "error", text: `✗ 未切换任务 · ${error.message}` }; }
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
        return { kind: "error", text: `✗ 读数不可用 · 返回的不是 JSON\n${out.text.slice(0, 400)}` };
      }
    }
    if (sub === "status" || sub === "cancel") {
      const match = /^([\s\S]+?)\s+([0-9a-f]{32})$/.exec(rest);
      if (!rest || (sub === "cancel" && !match)) {
        return { kind: "error", text: `✗ 缺参数 · 用法：/op ${sub} <run_dir>${sub === "cancel" ? " <invocation_id>" : " [invocation_id]"}` };
      }
      const runDir = resolveRunDir(match ? match[1] : rest, agent);
      const argv = [sub, "--run-dir", runDir];
      if (match) argv.push("--invocation-id", match[2]);
      const out = await run(argv, signal);
      if (!out.ok) return fail(out);
      try {
        const payload = JSON.parse(out.text);
        if (payload.ok === false) return rejection(payload);
        if (sub === "cancel") return { kind: "success", text:
          payload.accepted ? `✓ 取消已请求 · ${payload.invocation_id}\n用 /op status 确认终态与清理结果。`
            : payload.phase === "lost" ? `✓ 监督者不可用 · lost\n本机清理结果未知，请查 /op status。\n任务实例：${payload.invocation_id}`
              : `✓ 任务已结束 · ${payload.phase}\n任务实例：${payload.invocation_id}` };
        return { kind: "success", text: formatStatus(payload, runDir) };
      } catch {
        say("warn", "bad_output", `返回的不是 JSON：${out.text.slice(0, 120)}`);
        return { kind: "error", text: `✗ 读数不可用 · 返回的不是 JSON\n${out.text.slice(0, 400)}` };
      }
    }
    return { kind: "error", text: `✗ 未知子命令 · ${sub}\n\n${USAGE}` };
  };

  // 生命周期归 cordis：service tracker 让 register/section 的 effect 挂在**调用方 ctx** 上，
  // 所以卸载/HMR 会自动回滚，这里不需要（也不该）自己攒 disposer。
  ctx.commands.register({
    definitionId: "dsh-operator-agent",
    name: "op",
    description: "算子判定层：查能力 / 查无人值守状态（不经过模型）",
    input: { hint: "new [run_dir] | capabilities | status [run_dir] [invocation_id] | cancel <run_dir> <invocation_id> | help" },
    // `agent` 是宿主给的（invocation = {commandId, agent, rawInput, attachments, signal}），
    // 只有它带着会话工作区，相对路径要靠它才能落到人想去的地方。
    handler: async ({ rawInput, signal, agent, commandId }) => handle(rawInput, signal, agent, commandId),
  });
}
