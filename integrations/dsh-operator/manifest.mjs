/**
 * 构建期清单的唯一加载点。
 *
 * 这份文件存在的理由只有一条：**工具名单只允许有一个来源**。它此前散在四处手抄：
 *
 *   host-model-budget.mjs   TASK_TOOLS（七条）
 *   host-model-budget.mjs   READ_ONLY（四条）
 *   host-session-lifecycle.mjs  BINDS（两条）
 *   bin/setup.mjs           TOOLS（七条，用来校验 Python 侧的工具面）
 *
 * 手抄的后果不是编译不过，而是**判据静默失效**：名单落后了，模型门就守着一个已经
 * 不存在的工具集，而一切看起来正常。所以这里做两件事 —— 读，并且**认真地不读成功**：
 * 清单缺失或 schema 不认识就当场抛错，绝不降级成空集合（空集合意味着谁也不拦）。
 */
import { readFileSync } from "node:fs";

const MANIFEST_SCHEMA = "dsh-operator-manifest-v1";

function load() {
  const url = new URL("./manifest.json", import.meta.url);
  let body;
  try {
    body = JSON.parse(readFileSync(url, "utf8"));
  } catch (error) {
    throw new Error(
      `operator-agent: 读不到构建期清单 ${url.pathname} (${error.message})。` +
      "装了不完整的包，或 Python 侧没有跑 --emit-manifest。");
  }
  if (body?.schema !== MANIFEST_SCHEMA) {
    throw new Error(
      `operator-agent: 清单 schema 不受支持 (${body?.schema})，期望 ${MANIFEST_SCHEMA}。` +
      "Python 与 JS 不是同一套构建，拒绝继续。");
  }
  for (const key of ["host_tools", "read_only_tools", "bind_tools", "tools"]) {
    if (!Array.isArray(body[key])) {
      throw new Error(`operator-agent: 清单缺 ${key} 数组，拒绝继续。`);
    }
  }
  return body;
}

export const MANIFEST = load();

/** 加 `mcp__operator__` 前缀，得到宿主看到的工具全名。 */
export const toolNames = (key) => MANIFEST[key].map(name => `mcp__operator__${name}`);