# DSH profile 与组合包：为什么插件装上了却在 GUI 里看不见

日期：2026-10-25。目标：说清 **profile / 组合包 / 手工条目** 三者的关系，
以及「插件明明在跑，GUI 插件页却看不到它」这个坑的成因与两种装法。

**这不是什么**：不是 DSH 的完整配置手册（那是宿主的事，见
`@deepseek-ai/dsh-app-boot` 与 `dsh-plugin-manager` 的包内文档）；也不教怎么写插件。
本文只回答：**装成什么样，GUI 才认**。

起因是本仓 `integrations/dsh-operator` 的一次实际踩坑：`/op` 命令一直在正常工作，
日志里每条 `command/done` 都是 `success`，但侧栏插件页里找不到它 —— 因为它被装成了
**profile 里的一条手工条目**，而不是一个**组合包**。

---

## 1. 三个概念

### 1.1 profile = 一份「加载哪些插件」的清单

官方定义（`@deepseek-ai/dsh-app-boot` README §profiles）：

> profile 是同一套 dsh 安装提供不同应用界面的方式：`web`、`headless`、`acp`、`sdk` 与
> `sdk-minimal` 从同一 launcher 启动不同组合。

同一份 DSH 程序，换一份清单就是另一个应用形态。profile 位于
`$DSH_HOME/profiles/<name>`（本机是 `~/.dsh/profiles/`），**物理上就是三个文件**：

| 文件 | 作用 |
|---|---|
| `package.json` | `dependencies`（装了哪些包）+ `dsh.profile.bundles`（启用哪些组合包） |
| `cordis.patch.yml` | 你的调整层：覆盖配置、插入条目、禁用条目 |
| `cordis.yml` | 主题配置（常为 `[]`） |

本机现有 `desktop`（GUI 在用）与 `web` 两个。**`~/.dsh/profiles/node_modules/` 不是
profile**，是所有 profile 共享的包仓库；各 profile 自己的包在 `<profile>/node_modules/`。

### 1.2 组合包（bundle）= 自带 patch 的 npm 包

组合包是一个 npm 包，其 `package.json` 声明：

```json
"dsh": { "bundle": { "patch": "./cordis.patch.yml" } }
```

这份 patch 说明「我要往应用里插哪些条目」。宿主启动时把各组合包的 patch 层按
`dsh.profile.bundles` 的顺序拼接，**在你自己的 `cordis.patch.yml` 之前**应用
（所以你的 patch 优先级更高，能覆盖它们的默认值）。

### 1.3 手工条目 = 直接写进 patch 的 `insert`

`cordis.patch.yml` 里可以写：

```yaml
- insert:
    - id: operator-tools
      name: /abs/path/to/index.mjs
```

这条能让插件**正常工作**——宿主照样加载它、注册它。但它**不是组合包**，
所以在 GUI 插件页里不可见、不可开关、没有图标与描述。

### 1.4 关系

```
profile: desktop
  └─ dsh.profile.bundles            ← 清单里列的组合包
       ├─ @deepseek-ai/dsh-base
       ├─ @deepseek-ai/dsh-web-app
       ├─ beauticode-dsh            ← 第三方组合包
       └─ dsh-operator-agent        ← 本仓的插件
            └─ cordis.patch.yml     ← 组合包自带，插入两行：
                 ├─ mcp-operator     （八个工具 + instructions）
                 └─ operator-tools   （/op 斜杠命令）
```

**这两行是组合包插进来的**，不是 profile 手工写的 —— 这是关键区别。

---

## 2. 两种装法，以及为什么一种看不见

| | 手工条目（路径 `insert`） | 组合包（`dependencies` + `bundles`） |
|---|---|---|
| 插件能跑 | ✓ | ✓ |
| GUI 插件页可见 | ✗ | ✓ |
| 可开关 / 显示图标与描述 | ✗ | ✓ |
| 装法 | 手写 `cordis.patch.yml` | `dsh plugin add` 或 GUI「安装第三方插件」 |
| 适用 | 临时代码、绝对路径试验 | **正式使用（应该用这个）** |

插件页只读 `listBundles`，也就是 profile 的**依赖 + 组合包选择**；`insert` 进来的
条目不在那个视图里。所以「在跑」与「可见」是两件独立的事。

---

## 3. 从手工条目转成组合包（本仓的实际做法）

### 3.1 前置：插件包本身要满足组合包的形状

`integrations/dsh-operator/package.json` 需要：

```json
"name": "dsh-operator-agent",
"type": "module",
"main": "./index.mjs",
"exports": { ".": { "import": "./index.mjs" } },
"dsh": { "bundle": { "patch": "./cordis.patch.yml" } },
"engines": { "dsh": "^0.2.1-rc.2", "node": ">=22" },
"icon": "./icon.svg"
```

- `dsh.bundle.patch` —— 没有它就不是组合包，`inspect` 会报 `not-a-bundle`。
- `engines.dsh` —— **失配不会在安装时拦住**，而是在 boot 时把整个组合包跳过，
  只在 stderr 留一行 `dsh: skipping profile bundle "<name>": <reason>`。
  表现是「装好了但没生效」，比报错更难查。
  注意比较对象是 `getDshRuntimeVersion()`（本机 **`0.2.1-rc.2`**），
  **不是 Desktop 外壳的版本号**（那是 `package.json` 里的 `2.0.18`）。

自检：

```bash
node -e '
import {readFileSync} from "node:fs";
const boot = await import("/Applications/DSH Desktop.app/Contents/Resources/app/node_modules/@deepseek-ai/dsh-app-boot/lib/index.js");
const pkg = JSON.parse(readFileSync("<profile>/node_modules/dsh-operator-agent/package.json","utf8"));
const issue = boot.evaluatePluginCompatibility(pkg, {});
console.log(issue ? boot.pluginCompatibilityWarning(issue) : "兼容");
' --input-type=module
```

### 3.2 登记成依赖与组合包

本地路径装（改代码能直接同步，适合开发）：

```bash
cd ~/.dsh/profiles/desktop
pnpm add file:<path-to-repo>/integrations/dsh-operator
```

或在 GUI：侧栏**插件 → 添加插件 → 安装第三方插件**，输入那个绝对路径。
之后把包名加进 `dsh.profile.bundles`（GUI 安装默认启用，会自己加）。

### 3.3 删掉旧的手工条目，但把 config 转到覆盖项

**这一步最容易出错。** 组合包自带的 patch 用的是**通用默认值**
（`command: operator-agent`，假定 `pip install` 过），而本机是从源码树跑的：

```yaml
# 插件自带 cordis.patch.yml 的默认值
- id: operator-tools
  config:
    command: operator-agent     # ← 本机没有这个命令
```

直接迁移会让 `/op` 变成「找不到命令」。所以在 profile 的 `cordis.patch.yml`
里按 id **覆盖**它的 config：

```yaml
- id: operator-tools
  config:
    command: <abs-path-to-python3>
    args: ["-m", "operator_agent"]
    env:
      PYTHONPATH: <path-to-repo>
    timeoutMs: 60000
```

两条硬规则：

1. **覆盖用 `id` + `config`，不要再用 `insert`**。同 id 的 `insert` 会让条目
   注册两遍；`mcp-operator` 重复注册的后果是 MCP server 插两次。
2. **`config` 覆盖是整体替换**，不是深合并 —— 要保留的字段必须重述一遍
   （所以上面的 `serverName`、`transport`、`toolCallTimeoutMs` 都写全了）。

### 3.4 重启

`patchReload: "live"` 只负责 **patch 层**的热更新。**新增组合包**要参与启动期的
runtime resolution，必须重启 Desktop。

---

## 4. 怎么确认装对了

| 检查 | 命令 / 位置 | 期望 |
|---|---|---|
| 包能解析 | `node -e "import('dsh-operator-agent')"`（在 profile 目录） | 打印导出列表 |
| 兼容性 | §3.1 的自检 | `兼容` |
| 组合包在清单里 | `package.json` 的 `dsh.profile.bundles` | 含包名 |
| 无重复 insert | `grep -c "insert:" cordis.patch.yml` | 0（若全部改成覆盖）。注意零匹配时 `grep` 退出码是 1，别当成命令失败 |
| boot 没跳过 | 重启后的 stderr | 无 `skipping profile bundle` |
| GUI 可见 | 侧栏插件页 | 有图标与描述 |
| 功能生效 | `/op capabilities` | 首行以 `✓ 已执行` 开头 |

---

## 5. `file:` 依赖是硬链接，不是符号链接

pnpm 对 `file:` 依赖默认建**硬链接**（可用 `stat -f "%i %N"` 比对 inode）。
影响：

- 改源码后，`node_modules` 里的内容**跟着变**（同一份数据块），不用重装。
- 但若编辑器用「写临时文件再 rename」保存，inode 会变、链接断开，需再跑 `pnpm install`。
- 想彻底避免可换成 `link:` 前缀（符号链接），但会改变该包的依赖解析规则，需另行验证。

---

## 6. 回滚

`package.json` 与 `cordis.patch.yml` 的备份在改之前落盘，形如
`package.json.bak-<时间戳>`、`cordis.patch.yml.bak-<时间戳>`。回滚：

```bash
cd ~/.dsh/profiles/desktop
cp package.json.bak-<ts> package.json
cp cordis.patch.yml.bak-<ts> cordis.patch.yml
pnpm install        # 让 node_modules 跟随清单
# 重启 Desktop
```