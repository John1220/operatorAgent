# 交付身份重绑：平铺布局与分层布局不再是同一身份（2026-10-31）

| 项目 | 说明 |
|---|---|
| 文档性质 | 身份断链的事实记录与失效清单；不是验收报告，不新增能力 |
| 依据 | [第一期清单 T1.1](operator-development-phase1-execution-20261031.md)、[仓库瘦身与分包](../design/repo-slimming-20261031.md) |
| 读数条件 | 本机 `/usr/bin/python3` 3.9.6（numpy 2.0.2）；`HEAD = 24f4fc2`（分支 `feat/phase1-delivery-identity`）；旧身份从在盘 wheel 解包到**独立目录**后 import 计算，新身份从仓库根计算 |
| 本期动作 | 重绑与记录：不重算旧哈希、不迁移历史证据、不回退布局、不跑 GPU、不叫模型；另在**仓外**重建 `0.6.2`/`0.1.11` 工件并核对身份（§5） |

## 1. 断链的两端

| 维度 | 旧（在盘 `operator_agent-0.6.1` wheel） | 新（`HEAD = 24f4fc2`） |
|---|---|---|
| **协议身份聚合摘要** | `78993c983003b9a86b3953f1828198f6e6cbab2201f9c787d3a21a8e357eb555` | `e8c7f2c6ffa995769a607b60e9ce6eaf6a1afc25ff11345422aa98b6ded6683b` |
| 清单条目 | 73 | 73（文件名集合完全相同） |
| 其中纯搬运（内容哈希不变、仅路径键变） | — | **14** |
| 其中内容也变了（import 重写） | — | **59** |
| `remote_scripts/` 三件 | 原样 | **三件全变**（远端部署集合的身份也随之改变） |
| 包布局 | 包根平铺 `operator_agent/*.py` | 五层子包，包根只剩 `__init__.py` 与 `__main__.py` |
| console script | `operator-agent = operator_agent.cli:main`、`operator-agent-mcp = operator_agent.mcp_server:main`（wheel 内 `entry_points.txt`） | `operator_agent.app.cli:main`、`operator_agent.app.mcp_server:main`（[pyproject.toml:21](../../pyproject.toml#L21-L22)） |
| 版本号 | `0.6.1` | `0.6.1`（**没变**，见 §4） |

聚合摘要用仓库自己的 `digest()` 归约，不另造口径。两侧 `protocol_identity()` 的根目录解析式不同（旧用 `Path(__file__).parent.parent`，新用 `parents[2]`，[core/run_contract.py:70](../../operator_agent/core/run_contract.py#L70)），所以复算时必须分别在独立解包目录与仓库根下进行，否则拿到的是"缺文件"提示而不是哈希。

## 2. 这不是文档对不上，是门会拒

分层后的身份与分层前**不能混用**，机制在三处，全是硬失败：

| 位置 | 行为 |
|---|---|
| [core/corpus_admission.py:36-L42](../../operator_agent/core/corpus_admission.py#L36-L42) | 读存量证据目录时比对 `manifest['measurement_protocol'] != rc.protocol_identity()`，不等即 `raise ValueError("admission manifest/contract/evaluator drift")`。⇒ **分层前落盘的经验库/已确认条目，在 HEAD 上会被直接拒收** |
| [core/run_contract.py:149](../../operator_agent/core/run_contract.py#L149) | 冻结契约把 `measurement_protocol: protocol_identity()` 写进清单——旧契约存的是旧身份 |
| `benchmarks/drivers/workload_*_h20.py`、`benchmarks/experiments/s5/s5_perf_evaluator.py` | 派发前比对 `protocol_identity() != plan['measurement_protocol']` 即停。⇒ 旧批次冻结计划不能"接着新一轮跑" |

另外 [decide/tool_evidence.py:103-L110](../../operator_agent/decide/tool_evidence.py#L103-L110) 每次落盘都存 `implementation` 与 `implementation_stable`，历史 evidence 里存的是旧身份，按同一字段与新读数对照时必然不等。

## 3. 失效清单：以下冻结记录从此只读、不可当新身份引用

| 记录 | 它引用的旧身份 | 现在 |
|---|---|---|
| [S5 安装工件验收](../status/dsh-operator-s5-installed-gpu-acceptance-20261030.md) | wheel `0.6.0`（`8cd75a38…`）、协议身份 73 文件（平铺键） | 只读。不得写成分层后身份的验收 |
| [S6 交付验收](../status/dsh-operator-s6-acceptance-20261030.md) | wheel `0.6.0` + 插件 `0.1.9` | 只读 |
| [S6 准备](../status/dsh-operator-s6-preparation-20261027.md) | 「已安装包的协议身份含 70 文件」 | 只读。70 与 73 是不同时期的清单，别再互相换算 |
| [GPU 中断恢复](../status/gpu-interrupt-recovery-20261030.md) | 本轮源码（平铺布局，109 个冻结文件） | 只读。四项真机验收结论本身不变，但绑定的是旧布局身份 |
| [发布形态](../status/dsh-operator-release-readiness-20261030.md) | wheel `0.6.1`（`097918d3…`）、插件 `0.1.10`（`8245f168…`） | 只读。这两个工件保留原样，不覆盖、不复用其验收 |

在盘两个工件的 SHA256（本日实算，与发布形态记录一致）：

```
097918d395d08bc2859b07edd3db81cd543a3c10b9b588856fd3ea8100df6de4  operator_agent-0.6.1-py3-none-any.whl
8245f1689c9c5b8145986248afd71631c1d6a50a958e194be7a4d222e65bab0d  dsh-operator-agent-0.1.10.tgz
```

## 4. 本期的选择

- **接受断链**。分层与[层依赖棘轮守卫](../../tests/hygiene/test_repo_layout.py)是本轮定下的方向；旧身份不可复现，要复原只能回滚布局，本期不回滚。
- **不迁移、不重算**。给历史证据目录改写 `measurement_protocol` 字段等于把旧批次伪装成新身份，属于伪造同源，本期禁止。
- **版本号必须换**。`0.6.1` 现在同时指平铺 wheel 与分层 HEAD，正是[发布形态 §1 第 3 项](../status/dsh-operator-release-readiness-20261030.md)刚修掉又被打回来的缺陷；随 T1.2 升 `0.6.2` / `0.1.11`，本记录即新身份的起算点。
- **报数不得跨身份混表**。引用历史读数时按旧摘要 `78993c98…` 注明不可与新批次同列。

## 5. 重绑后的新工件（T1.3）

| 工件 | 大小 | SHA256 | 身份核对 |
|---|---:|---|---|
| `operator_agent-0.6.2-py3-none-any.whl` | 968,687 | `11683240a7e78db2cf941c9425a710b1be6c115eb64a71e070d3d2cbf5894941` | 解包到独立目录后 `digest(protocol_identity())` = `e8c7f2c6…`，**与 HEAD 相同** |
| `dsh-operator-agent-0.1.11.tgz` | 31,751 | `0ce665a72b28f603b23c72b77d0e02fb565acd995de7ee047aac3e16c6817b8b` | 包内 10 文件，含 `LICENSE`/`NOTICE` |

布局与入口断言（在 wheel 解包目录内实际查看）：包内是 `core/decide/execute/generate/app` 五层 + `ops` + `prompts`/`skills`，`prompts` 11 份、`skills` 2 份、`dist-info/licenses/` 两份齐备；`entry_points.txt` 指向 `operator_agent.app.cli:main` 与 `operator_agent.app.mcp_server:main`。

构建与留档条件：

- 从**仓外克隆**构建（`git clone` 到临时目录 → `uv build --wheel`；`npm pack` 用临时 cache 与目标目录），避免与共享工作树互相污染。
- 一条附带读数：克隆根目录里留下的 `build/` 与 `operator_agent.egg-info` 会让 [布局守卫](../../tests/hygiene/test_repo_layout.py) 的 `test_the_root_only_holds_known_entries` 变红。这正是"构建必须落在仓外"的可复现证据，不是守卫缺陷；同一时刻真实仓内 `tests/hygiene` 保持 **59 passed**。
- 证据落 `output/plugin_completion/20261031-phase1-identity-v1/`（经符号链接解析到仓外证据树），含 `artifacts/` 与 `artifacts-SHA256.txt`。
- 旧 `0.6.1`/`0.1.10` 两个工件与其哈希原样保留，不覆盖、不复用其验收。

### 5.1 一期文档迁入 `docs/phase1/` 之后的再建（2026-11-01）

搬动改了**打进包里的文字**——插件 README、插件 CHANGELOG，以及 wheel 内 `human_ledger.py` 的文档路径引用。按 `0.1.10` 立下的「版本号 ↔ 内容一一对应」规矩，这必须换号：

| 工件 | 大小 | SHA256 | 身份核对 |
|---|---:|---|---|
| `operator_agent-0.6.3-py3-none-any.whl` | 972,283 | `108fd527add299ddfd2e0d2ba5436c2ccfdb1cc8a732845b50878e5eb31749b3` | 克隆自 `6f1d583` 重建；解包后 `digest(protocol_identity())` 仍是 `e8c7f2c6…` ⇒ 升号与文档路径改动**不触及那 73 个协议身份文件** |
| `dsh-operator-agent-0.1.12.tgz` | 32,010 | `2eb97dc42c4d1d3a88c1733915cf65aba1c326c825addfe8e97a323e81482809` | 包内 10 文件，含 `LICENSE`/`NOTICE` |

`0.6.2`/`0.1.11` 的哈希（`11683240…` / `0ce665a7…`，全值见 §5）与工件留在
`output/plugin_completion/20261031-phase1-identity-v1/`；本次新一组在
`output/plugin_completion/20261101-phase1-docindex-v1/`（含 `manifest.json`）。
一条测量教训也记在 manifest 里：**基线断链数不能在临时工作树里扫**——那里没有 `output/`
符号链接，`../../output/...` 会被整批判成断链（实测一份 closeout 记录就虚报 19 条）。

### 5.2 入口 docstring 修完之后（2026-11-01，同日再升一次）

T1.8 改的 `operator_agent/__init__.py` 模块清单也在 wheel 里，所以按同一条规矩再升号：

| 工件 | SHA256 | 身份核对 |
|---|---|---|
| `operator_agent-0.6.4-py3-none-any.whl` | `2b3742a61e52fb40b12a3eec77ec172115d0b69949b52b0aae3876e0135cab30` | 克隆自 `ba29424` 重建；解包内 `digest(protocol_identity())` 仍是 `e8c7f2c6…` |
| `dsh-operator-agent-0.1.13.tgz` | `b10b19af893b5df7c420e5afb4c18d4a8948c95599cefe32c3512273beb9ecac` | 包内 10 文件，含许可两份 |

三代 Python 工件（`0.6.2` / `0.6.3` / `0.6.4`）与三代插件包（`0.1.11` / `0.1.12` / `0.1.13`）的哈希逐件实算、
互不覆盖，分别留在 `output/plugin_completion/20261031-phase1-identity-v1/` 与
`.../20261101-phase1-docindex-v1/`。**协议摘要一路不变**——版本号与文档路径都不在那 73 个文件里，
这本身就是「身份清单只覆盖评测面」的一次现场印证。

### 5.3 CLI 与迁移门入口落进 wheel 之后（2026-11-01，同日第三次）

| 工件 | SHA256 |
|---|---|
| `operator_agent-0.6.5-py3-none-any.whl` | `460c56a0f37a92fa8938fb3029f8df7063f106dcc4ba01bdfb3ee0c240cf8cc9` |
| `dsh-operator-agent-0.1.14.tgz` | `0da759ee4c43b418d9744def077169d57848ed256b255ac40c340168e71cbc64` |

克隆自 `2e41cf9` 重建（`app/cli.py` 加了 `ledger-new|fill|report` 三条子命令，`core/human_ledger.py`
边界说明同步）；解包内 `digest(protocol_identity())` **仍是 `e8c7f2c6…`**，解包后 CLI 三个子命令在位。

至此四代 Python 工件（`0.6.2`/`0.6.3`/`0.6.4`/`0.6.5`）与四代插件包并存、逐件哈希、互不覆盖，
**协议摘要一路没动**。这不是巧合而是定义使然：身份清单只覆盖评测面那 73 个文件，
版本号、文档路径、CLI 入口都不在其中。**所以「协议身份相同」不等于「工件相同」，
「工件不同」也不等于「身份变了」**——两者要分开引用，这也是 §5 那张四代表格的用处。

## 6. 怎么复算

```bash
# 新身份（仓库根，本机 python3 需带 numpy）
/usr/bin/python3 -c "import sys; sys.path.insert(0, '.'); from operator_agent.core.run_contract import digest, protocol_identity; print(digest(protocol_identity()))"

# 旧身份（先解包到独立目录，在该目录内 import）
tmp=$(mktemp -d) && unzip -q output/plugin_completion/20261030-release-v1/artifacts/operator_agent-0.6.1-py3-none-any.whl -d "$tmp"
( cd "$tmp" && /usr/bin/python3 -c "import sys; sys.path.insert(0, '.'); from operator_agent.run_contract import digest, protocol_identity; print(digest(protocol_identity()))" )

# 守卫基线（本日读数：59 passed）
/usr/bin/python3 -m pytest tests/hygiene -q
```

两条 `digest` 输出应分别等于 `78993c98…` 与 `e8c7f2c6…`。不等就先查解释器与解包位置，不要先查判据。

## 修订记录

| 日期 | 变更 | 依据 |
|---|---|---|
| 2026-10-31 | 创建：记录平铺 → 五层造成的协议身份断链、入口与布局差异、三处硬失败机制、五份冻结记录失效清单 | 本日实算的两侧 `protocol_identity()` 与 `digest()`、wheel `entry_points.txt`、73 条目逐项比对、`tests/hygiene` 59 通过基线 |
| 2026-10-31 | 新增 §5：`0.6.2` wheel 与 `0.1.11` 插件包在仓外重建完成，解包后身份摘要等于 HEAD 的 `e8c7f2c6…`，工件与哈希落证据树；旧工件未被覆盖 | `uv build --wheel` 与 `npm pack` 实跑；解包目录内的布局/入口/prompts-skills 逐项查看；构建残留触发布局守卫变红一次（已记为约束证据而非缺陷） |
| 2026-11-01 | 本文件与一期清单、台账一起迁入 `docs/phase1/`；新增 §5.1 记录文档搬动导致的再建（`0.6.3` / `0.1.12`），并核对「升号与文档路径改动不触及那 73 个协议身份文件」。一处哈希前缀笔误（写成 `0ce65a…`）在写入后立即与证据树的 `artifacts-SHA256.txt` 对回并更正——记错的哈希前缀会伪装成「证据被篡改」，所以不复算就不写 | `git clone` 自 `6f1d583` 重建 + `npm pack`；解包内 `digest(protocol_identity())` 复算；`shasum -a 256` 逐件实算 |
| 2026-11-01 | 加 §5.3：CLI 人时子命令与迁移门入口进 wheel ⇒ 第三次升号 `0.6.5`/`0.1.14`，解包复算协议摘要仍 `e8c7f2c6…`；写明「身份相同 ≠ 工件相同」这条区分 | `git clone` 自 `2e41cf9` + `uv build` + `npm pack`；`shasum -a 256` 逐件实算；解包内 `digest(protocol_identity())` 与 CLI 子命令逐项核对 |

