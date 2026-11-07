/**
 * `/op doctor` 与安装助手的最小守卫：
 *   - 自检读数必须**先说事实**，失败项带可复制的修复命令；
 *   - patch 写入必须保留用户原配置、可重复执行、不吞掉 marker 前的注释。
 * 这里不碰网络、不碰 GPU、不调模型；只测纯函数。
 */
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync, chmodSync, symlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import test from "node:test";

import { formatDoctor, locate, PAIRED_PYTHON, EXPECTED_TOOL_VERSION } from "../index.mjs";
import { patchText, entries, entriesUvx, parseArgs, BEGIN, END } from "../bin/setup.mjs";

const healthy = () => ({
  pluginVersion: "0.2.3", command: "operator-agent", cliFound: true, cliPath: "/opt/venv/bin/operator-agent",
  mcpFound: true, mcpPath: "/opt/venv/bin/operator-agent-mcp", workflow: "host",
  capabilitiesOk: true, capabilities: {
    version: PAIRED_PYTHON, tool_version: EXPECTED_TOOL_VERSION,
    runtime: { python: "3.12.16", torch_installed: false, triton_installed: false },
    backends: { mock: { wired: true }, fake: { wired: true } },
  },
});

test("healthy install reports every fact and no problem", () => {
  const text = formatDoctor(healthy());
  assert.match(text.split("\n")[0], /^✓ 安装自检通过/);
  assert.match(text, /dsh-operator-agent 0\.2\.3/);
  assert.match(text, /工具契约：12/);
  assert.match(text, /判定层：0\.7\.2/);
  assert.match(text, /python 3\.12\.16/);
  assert.match(text, /已接线后端：mock \/ fake/);
});

test("missing CLI names the command and the one-liner fix", () => {
  const text = formatDoctor({ ...healthy(), cliFound: false });
  assert.match(text.split("\n")[0], /^✗ 安装自检发现 1 项问题/);
  assert.match(text, /CLI 未找到/);
  assert.match(text, /dsh-operator-agent-setup --wheel/);
});

test("contract mismatch is a failure, not a warning", () => {
  const facts = healthy();
  facts.capabilities.tool_version = 11;
  const text = formatDoctor(facts);
  assert.match(text, /工具契约 11 ≠ 期望 12/);
  assert.match(text, /pip install --upgrade dsh-operator-agent==0\.7\.2/);
});

test("unset workflow env is informational, because the patch injects it", () => {
  const text = formatDoctor({ ...healthy(), workflow: "" });
  assert.match(text.split("\n")[0], /^✓ 安装自检通过/);
  assert.match(text, /插件进程未设置/);
});

test("locate resolves PATH entries and refuses missing absolute paths", () => {
  const dir = mkdtempSync(join(tmpdir(), "op-doctor-"));
  const bin = join(dir, "operator-agent");
  writeFileSync(bin, "#!/bin/sh\nexit 0\n");
  chmodSync(bin, 0o755);
  assert.equal(locate("operator-agent", dir), bin);
  assert.equal(locate("/definitely/not/here/operator-agent"), null);
  assert.equal(locate("operator-agent", dir), bin);
});

test("patchText appends a managed block and keeps the user's own lines", () => {
  const original = "# 我的主题配置\n- insert:\n    - id: my-theme\n      name: theme\n";
  const once = patchText(original, entries("/opt/venv/bin/python"));
  assert.ok(once.startsWith(original.slice(0, original.indexOf("# 我的主题配置"))));
  assert.ok(once.includes("# 我的主题配置"));
  assert.ok(once.includes(BEGIN) && once.includes(END));
  assert.ok(once.includes('"id":"mcp-operator"'));
  assert.ok(once.includes("/opt/venv/bin/operator-agent-mcp"));
  assert.ok(once.includes('"OPERATOR_AGENT_WORKFLOW":"host"'));

  const twice = patchText(once, entries("/opt/venv/bin/python"));
  assert.equal(twice, once, "重复执行必须幂等，不叠加第二段");
});

test("patchText replaces an outdated managed block instead of duplicating it", () => {
  const first = patchText("", entries("/old/bin/python"));
  const second = patchText(first, entries("/new/bin/python"));
  assert.ok(second.includes("/new/bin/operator-agent"));
  assert.ok(!second.includes("/old/bin/operator-agent"));
  assert.equal(second.split(BEGIN).length, 2);
});

test("patchText handles the JSON-array profile format with id-based replacement", () => {
  const original = '[{"id":"mcp-operator","disabled":false,"config":{"command":"old"}},{"id":"other","name":"x"}]\n';
  const out = patchText(original, entries("/opt/venv/bin/python"));
  const values = JSON.parse(out.trim());
  assert.equal(values.length, 3);
  assert.equal(values.filter((v) => v.id === "mcp-operator").length, 1);
  assert.ok(values.some((v) => v.id === "other"));
});

test("patchText refuses duplicated or unbalanced managed markers", () => {
  const doubled = `${BEGIN}\n- {}\n${END}\n${BEGIN}\n- {}\n${END}\n`;
  assert.throws(() => patchText(doubled, entries("/opt/venv/bin/python")), /托管标记/);
});

test("bin 通过符号链接调用时仍然执行（npm .bin 就是符号链接）", () => {
  const dir = mkdtempSync(join(tmpdir(), "op-bin-"));
  const link = join(dir, "dsh-operator-agent-setup");
  symlinkSync(resolve("bin/setup.mjs"), link);
  const out = spawnSync(process.execPath, [link, "--help"], { encoding: "utf8" });
  assert.equal(out.status, 0);
  assert.match(out.stdout, /用法：dsh-operator-agent-setup/);
});

test("uvx 形态把版本钉死，并同时配置 MCP 与 /op 两条记录", () => {
  const rows = entriesUvx("0.7.2", "/opt/homebrew/bin/uvx");
  assert.equal(rows.length, 2);
  assert.deepEqual(rows[0].config.args, ["--from", "dsh-operator-agent==0.7.2", "operator-agent-mcp"]);
  assert.deepEqual(rows[1].config.args, ["--from", "dsh-operator-agent==0.7.2", "operator-agent"]);
  assert.equal(rows[0].config.env.OPERATOR_AGENT_WORKFLOW, "host");
  assert.equal(rows[1].config.hostModelBudget && Object.keys(rows[1].config.hostModelBudget).length, 0);
});

test("uvx 与 --python/--wheel 互斥", () => {
  assert.throws(() => parseArgs(["--runtime", "uvx", "--python", "/x/python"]), /不能同时使用/);
  assert.throws(() => parseArgs(["--runtime", "docker"]), /path 或 uvx/);
});
