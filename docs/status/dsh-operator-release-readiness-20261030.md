# 算子开发插件发布形态补齐（2026-10-30）

状态：**发布形态已补齐并通过本机验证；效果主张的边界不变**。补齐四项工程缺口（许可随包、可发布标志、版本号、CI 守卫），并修复一处让模型无法自我纠正的拒绝消息。工具契约仍为 **9**，工具行为、schema 与指令未变。本次没有真实模型调用，也没有 GPU 验证；不改变 [S5 结项](dsh-operator-s5-closeout-20261030.md) 的结论：**插件整体优于原生 DSH 未建立**。

## 1. 修了什么

| # | 问题 | 修前事实 | 修法 | 守卫 |
|---|---|---|---|---|
| 1 | 发布的 npm 包不带许可正文 | 旧 `dsh-operator-agent-0.1.9.tgz` 文件表 8 项，无 `LICENSE`/`NOTICE`；`license` 字段声明 Apache-2.0，而 Apache-2.0 §4 要求随附许可副本 | 仓库根两份文件放入包目录并列入 `files` | `tests/test_packaging.py::TestTheDshReleaseShape`（逐字节相等、在 `files` 里）、CI `release-shape` 作业 |
| 2 | `private: true` 让 `npm publish` 直接失败 | `integrations/dsh-operator/package.json` | 改为 `false`，版本 `0.1.9` → `0.1.10` | 同上（`private` 必须为 `false`） |
| 3 | 版本号与内容不一一对应 | `0.1.9` 已被两份内容不同的本地构建占用（靠 SHA256 区分）；Python `0.6.0` 同理 | 插件 `0.1.10`，Python `0.6.1`（`pyproject.toml` 与 `operator_agent/__init__.py` 同步） | 同上 + `test_pyproject_version_matches_the_package` |
| 4 | 仓库完全没有 CI | 无 `.github/`，所有验证靠手工 | 新增 `.github/workflows/ci.yml`：Python 3.9/3.12 全量 pytest、插件 `node --test`、发布物形态（wheel 含 prompts/skills 与许可、`npm pack` 文件表含许可） | 工作流本身（`release-shape` 作业） |
| 5 | 受控验证面的拒绝消息不可修正 | 2026-10-30 Matmul C 两次把 `run_dir` 指向 `/work`（应为 `/work/task`），两次都只收到 `Frozen development surface refused this request`，模型无从纠正、耗尽调用额度 | [s5_plugin_rpc.py](../../benchmarks/s5_plugin_rpc.py) 的拒绝回 `reason + 可接受取值`，一次列出**全部**违规项，并写入账本 `refusal` 字段 | `tests/test_s5_plugin_rpc.py` 两条新用例：命名违规规则与可接受值、多违规同时列出、账本留痕，且断言消息里**不出现** supervisor 路径 |

第 5 项修的是受控实验的验证面（`benchmarks/`），不是已发布插件本身——产品侧的 `hitl_tools` 早就返结构化的 `ToolError`（`code_required` / `run_not_prepared` / `contract_refused`）。所以它不改工具契约，也不进 wheel。

拒绝消息现在长这样（同一回包列出两条违规）：

```
Frozen development surface refused this request: run path refused
(run_dir must resolve to the prepared task directory; pass 'task');
verification identity refused (operator_verify accepts stop_after='L2',
backend='mock' and 1 <= timeout <= 600)
```

提示里只出现**工作区相对**位置与冻结取值，不出现 supervisor 的绝对路径——隔离边界不变。

## 2. 工件

证据目录：[20261030-release-v1](../../output/plugin_completion/20261030-release-v1/)（Git 忽略，须单独备份），
`artifacts-manifest.json` 记逐文件大小与 SHA256。

| 工件 | 大小 | SHA256 |
|---|---:|---|
| `operator_agent-0.6.1-py3-none-any.whl` | 962,373 | `097918d395d08bc2859b07edd3db81cd543a3c10b9b588856fd3ea8100df6de4` |
| `dsh-operator-agent-0.1.10.tgz` | 31,022 | `8245f1689c9c5b8145986248afd71631c1d6a50a958e194be7a4d222e65bab0d` |

构建条件：wheel 从 `rsync` 干净导出（排除 `output/`、`logs/`、`triton/`、`.git/`）构建，规避
README §8 记的 `pip < 21.3` + `0400` 产物坑；`npm pack` 用临时缓存（默认 `~/.npm` 不可写，同 S6 记录）。

内容核对：

- wheel 141 文件，`dist-info/licenses/{LICENSE,NOTICE}` 齐备，11 个 prompts、2 个 skills。
- tgz 10 文件：`index.mjs`、`cordis.patch.yml`、`icon.svg`、`locale/{en,zh}.json`、`README.md`、`CHANGELOG.md`、`LICENSE`、`NOTICE`、`package.json`；包内 `private=false`、`version=0.1.10`、`license=Apache-2.0`。
- 旧 `0.6.0`/`0.1.9` 工件与其 SHA256 原样保留，不覆盖、不复用验收。

## 3. 本机验证

| 项 | 结果 |
|---|---|
| 隔离 venv 安装 wheel（不继承 site-packages，离线 NumPy） | `pip check` 无冲突；`operator_agent.__version__ == 0.6.1`；`operator-agent capabilities` 正常返回 |
| 临时 npm 工程安装 tgz | 安装成功；`import('dsh-operator-agent')` 导出 `apply/inject/name/normalizeConfig/runCli/scrubEnv/format*`；包内含许可两份 |
| 全量 pytest | **6159 通过、1 跳过**；15 失败 + 4 错误全部是需要 `ps`／`sandbox-exec`／真 DSH 的集成用例，在本会话沙箱内被拒（`Operation not permitted`），已复现根因（见下）。同一命令在并行改动进入工作区后又跑一次，总数不变（6159/1/15+4） |
| 插件 node 测试 | **58 通过、0 跳过**（49 + 9，含真宿主契约、真 `evaluatePluginCompatibility`） |
| CI 定义 | YAML 可解析、3 个作业；**未在 GitHub 上真实跑过**（仓库此前无 CI 历史） |

沙箱外签核**仍未完成**：我发起的一次提权重跑（`danger-full-access`）长时间未返回，已终止，没有结果。
因此「这 17 条在真实机器上通过」目前只有 2026-10-29/30 各轮记录（当时总回归全绿）作为旁证，本次没有新增读数。
发布签核前应在普通终端跑一次 `python -m pytest tests/ -q` 并留存日志。

## 4. 仍不能对外声称的

以下都是**已接受的范围**，不是本次遗留缺陷：

- **插件整体优于原生 DSH**：未建立。S5 按「研究结项、贡献未建立」收口。
- **性能**：只有单卡单形状的合格读数（Matmul 4096³/8192³ `1.0569×`/`1.0239×`，LayerNorm 16384×6144 `1.4135×`，均对各自基线、未做独立确认）；同条件 A/C pilot 的 LayerNorm 只快 1.34%，低于预注册 1.05× 目标。
- **安全边界**：`exec_guard` 不是沙箱；MCP 工具不过 DSH 审批层；网络唯一出口与恶意候选沙箱已后移。
- **证据持久性**：私有原始证据为单副本（用户 2026-10-30 决定接受），H20 与 P800 已下线，读数不可再生。

## 5. 还差的人工动作

1. **真正的发布**：`npm publish`（需账号/2FA）与 PyPI 上传（`twine upload`）；两者都会产生不可撤回的公开版本号，本次只到「可发布」为止，没有上传。
2. **沙箱外全量回归**：一次 `python -m pytest tests/ -q` 留档（见 §3）。
3. **CI 首跑**：工作流要推到远端仓库才会执行；首次运行的红/绿应记录在案。

只想在本机用的话，不需要上面任何一步：按 [插件 README §2.2](../../integrations/dsh-operator/README.md) 装
`operator_agent-0.6.1` wheel + `dsh-operator-agent-0.1.10.tgz` 即可。

## 6. 怎么复核

```bash
# 工件哈希
shasum -a 256 output/plugin_completion/20261030-release-v1/artifacts/*

# 发布形态守卫（许可副本、private、版本号）
python -m pytest tests/test_packaging.py -q

# 拒绝消息可修正性
python -m pytest tests/test_s5_plugin_rpc.py -q

# 插件宿主契约
cd integrations/dsh-operator && npm test
```