# v0.7.2 发布手册（本地封版 + 三个公开通道）

更新：2026-11-07。适用于当前已封版的 `v0.7.2`；本文件是**操作手册**，事实与回执见
[封版记录](../status/v072-release-20261106.md) 与交付目录 `release.json`。

## 0. 版本对应（三处必须一致）

| 位置 | 身份 |
|---|---|
| Python 分布（PyPI） | `dsh-operator-agent==0.7.2`（import 名仍是 `operator_agent`） |
| DSH 插件（npm） | `dsh-operator-agent@0.2.5` |
| 工具契约 | `12`（插件与判定层必须配对，`/op doctor` 会查） |
| 宿主兼容 | `engines.dsh: ^0.2.1-rc.2` |

> 原名 `operator-agent` 在 PyPI 不可用（JSON 404 但上传 403：已注册未发布或过于相似）。
> 改名只影响分布名，命令 `operator-agent` / `operator-agent-mcp` 与 import 名不变。

## 1. 发布前检查（每项都要过）

```bash
python -m pytest -q -m "not slow and not serial" -n 4 --dist loadfile   # ~80 s
python -m pytest -q -m "serial"                                         # ~20 s
python -m pytest -q tests/hygiene                                       # 59 项
cd integrations/dsh-operator && node --test test/*.test.mjs             # 151 项，7 skip 需 DSH Desktop 布局
```

工件侧（**用封版原件，不要重新构建**，除非有意换身份）：

```bash
cd "<交付目录>/v0.7.2" && shasum -a 256 -c SHA256SUMS
uvx twine check artifacts/*.whl artifacts/*.tar.gz        # 均需 PASSED
```

## 2. GitHub

```bash
git push origin main
git push origin v0.7.0 v0.7.1 v0.7.2
```

- 标签必须是**已验提交**；`v0.7.2` 指向封版提交 `ce8deab`。
- Release 页面附 `SHA256SUMS` 与三个工件；正文写清"本地封版 + 已发布 npm/PyPI"。

## 3. PyPI（Python 判定层）

凭据：`~/.pypi-token`（`chmod 600`）。**首包必须用 account 级 token**，项目级 token 不能创建新项目。

```bash
# 需要新工件时（否则跳过，直接用封版原件）
uv build --out-dir /tmp/pypi-out

uv publish --token "$(cat ~/.pypi-token)" \
  dsh_operator_agent-0.7.2-py3-none-any.whl dsh_operator_agent-0.7.2.tar.gz
```

验证：

```bash
pip install --index-url https://pypi.org/simple dsh-operator-agent==0.7.2
operator-agent capabilities            # version 0.7.2 / tool 12
uvx --from dsh-operator-agent==0.7.2 operator-agent-mcp   # MCP 握手，host 模式 7 个工具
```

两个坑：

- 本机默认源是 aliyun/tuna 镜像，**新包同步有延迟**；刚发完装不到时加 `--index-url https://pypi.org/simple`。
- `uv build` 会生成 `dsh_operator_agent.egg-info/`，`tests/hygiene/test_repo_layout.py` 会报；提交前删掉。

## 4. npm（DSH 插件）

凭据：`~/.npm-token`（`chmod 600`）。账号开了 2FA 时，普通 granular token 不能发布，
需要 **Classic Automation token**（天然绕过 2FA）或 `--otp=<6 位码>`。

```bash
cd integrations/dsh-operator && npm pack        # 产出 dsh-operator-agent-0.2.5.tgz
cp dsh-operator-agent-0.2.5.tgz "<交付目录>/dsh-plugin-0.2.5/"
cd "<交付目录>/dsh-plugin-0.2.5" && shasum -a 256 *.tgz | tee SHA256SUMS

npm publish dsh-operator-agent-0.2.5.tgz \
  --registry https://registry.npmjs.org/ --access public --tag latest \
  --//registry.npmjs.org/:_authToken="$(cat ~/.npm-token)"
```

- **新包首发会进 staged review**：registry 先出现占位版本 `0.0.0-stage`，`npm install` 会 `notarget`；
  等 `dist-tags.latest` 变成目标版本即可（实测几分钟）。
- 本机 `~/.npmrc` 指向 npmmirror，**发布必须显式 `--registry https://registry.npmjs.org/`**。
- 发坏的版本用 `npm deprecate dsh-operator-agent@<ver> "<原因>"`；`0.2.3` 就是这么处理的
  （`.bin` 符号链接下入口判定失效，已由 0.2.4 修复）。
- 包内 `.bin/*` 是符号链接：入口判定必须 `realpath` 后比较，否则 `npx` 会静默退出。

## 5. 发布后验证清单

| 检查 | 命令 | 期望 |
|---|---|---|
| PyPI 版本 | `curl -s https://pypi.org/pypi/dsh-operator-agent/json \| jq -r .info.version` | `0.7.2` |
| 文件哈希一致 | 对比 `urls[].digests.sha256` 与本地 | 逐字相同 |
| npm 标签 | `npm view dsh-operator-agent dist-tags --registry https://registry.npmjs.org/` | `latest: 0.2.5` |
| 插件完整性 | `npm view dsh-operator-agent@0.2.5 dist.integrity` | 与本地 tgz 的 sha512 一致 |
| 端到端 | 干净 venv 装 PyPI + 插件页装 npm → `npx dsh-operator-agent-setup --runtime uvx` → `/op doctor` | 自检 7 项通过 |

## 6. 回滚

- **PyPI**：版本不能删除；用 `yank`（`pip` 默认跳过 yanked），并发布补丁版；文档注明原因。
- **npm**：`npm deprecate` 为首选（保留可追溯）；`unpublish` 仅 72h 内且无依赖时可用。
- **GitHub**：标签保留不动，新增记录说明；回退用 `v0.7.1` 标签与交付目录。
- 任何回滚都不改写已验工件的哈希与历史回执。

## 7. 凭据与安全

| 文件 | 权限 | 用途 |
|---|---|---|
| `~/.pypi-token` | 600 | PyPI 上传 |
| `~/.npm-token` | 600 | npm 发布（Automation token） |
| `~/.config/pypi-recovery/` | 700 | PyPI 恢复码（曾误落仓库根目录，已移出） |

- token 只放文件、由 `~/.zshrc` 读取导出；**不要把明文贴进聊天或提交历史**。
- 泄露后立即在对应仓库轮换；仓库卫生测试会拦 `PyPI-Recovery-Codes-*` 一类文件名。

## 8. 本版已知边界

Desktop profile 由 Electron 独占，插件包必须走 GUI 安装（CLI 只能配置）；单主会话单任务；
性能读数为 `uninformative`，不证明收益；费用与服务商停止状态 `unknown`。