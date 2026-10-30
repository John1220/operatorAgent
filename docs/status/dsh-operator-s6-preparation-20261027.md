# S6 安装、证据包和演示准备（2026-10-27）

状态（2026-10-30）：**按用户调整后的交付范围完成**。用户确认不录制视频，以安装说明、脱敏案例包和演示脚本作为材料。当前 wheel/DSH 工件仓库外干净安装、宿主契约 9/9 及真实 DSH CLI 隔离 profile 的添加/配置发现/卸载生命周期均通过；脱敏案例包外部离线复核通过。详见 [S6 交付验收](dsh-operator-s6-acceptance-20261030.md)。本地真实 H20 MCP L2 12/12 另见[安装工件验收](dsh-operator-s5-installed-gpu-acceptance-20261030.md)。没有修改个人默认 profile、发布包或录制视频；S5 整体 A/C 贡献结论未建立。

## 干净环境安装验证

从 S4 冻结的 486 文件源码归档提取实现与测试，并复制现有构建元数据后，在临时目录离线构建 `operator_agent-0.6.0-py3-none-any.whl` 和 `dsh-operator-agent-0.1.9.tgz`。归档未包含顶层构建元数据，因此将另行复制的 `pyproject.toml`、README 和许可证等哈希记入验收摘要；不能将它们误称为 S4 的 486 文件快照内容。构建不修改用户 DSH profile。

新建 Python 3.9 venv，**不继承系统 site-packages**；安装本地 Python wheel 和单独从 PyPI 获取的 NumPy 2.0.2 wheel。在 `/tmp`、移除 `PYTHONPATH` 且不提供模型凭据的条件下：

- `pip check` 通过；已安装包的协议身份含 70 文件，其哈希与 S4 真实任务身份一致。
- wheel 包含 11 个 prompts、2 个 skills、20 个 remote_scripts 文件；CLI `capabilities` 正常。
- 从仓库外使用已安装 CLI 做 `prepare` + mock backend 的 L1/L2 验证通过；这是本机 mock，不是 GPU 成绩。
- 已安装的 MCP server 完成真实 stdio 初始化和 `tools/list`，返回 8 个工具。
- DSH 包安装到临时 npm prefix；通过已安装 DSH 宿主的 9 项契约测试（0 跳过），包含命令注册、插件元数据、版本兼容、真实 MCP 客户端和相对工作区路径。测试从临时安装的包加载 `index.mjs`，使用隔离 venv 的 Python；没有往用户的真实 DSH profile 安装或更改配置。

首次 `npm pack` 因默认 npm 缓存目录不可写失败；改用临时缓存后离线打包通过。刻意先用 `pip --no-deps` 安装而不提供 NumPy 时，`pip check` 准确指出缺依赖，但 CLI 启动显示 Python `ModuleNotFoundError` 堆栈；正常安装应安装所声明的 NumPy。日志保留，当前没有为缺依赖增加友好提示。

wheel、npm 包、NumPy wheel、摘要和检查日志在 [20261027-s6](../../output/plugin_completion/20261027-s6/)；目录被 Git 忽略，需单独备份。`clean-install-summary.json` 保留工件和构建输入哈希。临时 venv 和 npm prefix 只用于此轮验证，不是可发布的交付物。

## 脱敏案例包与演示

主要演示材料现为 [S5 脱敏插件案例 ZIP](../../output/plugin_completion/20261030-s5-closeout-v1/shareable/s5-plugin-cases.zip)和[三分钟演示/简历脚本](../design/dsh-operator-s5-resume-demo-20261030.md)，覆盖真实插件入口、受控缺陷修复及证据边界。ZIP 已在仓库外通过标准库核验器；当前文件清单与案例数见 [S6 交付验收](dsh-operator-s6-acceptance-20261030.md)。不录制视频是本轮用户确认的范围调整。

此前的 [S4 脱敏案例包](../../output/plugin_completion/20261027-s4/portable-case-bundle-v2.zip)及[补充讲解脚本](../design/dsh-operator-s6-case-demo-20261027.md)继续保留，内容是 CLI/core 任务，不冒充 MCP 插件调用或插件增益。

该 ZIP 是脱敏投影和内部一致性证据，未提供签名、GPU 重演能力或完整原始日志，不能冒充完整可移植原始证据包；原始材料必须继续私密保存。S5 三组对照未执行，不作插件整体贡献声明。

## 边界与后续事项

源码、构建工件与历史原始材料有独立归档和 SHA256 清单，位置见 [S5 结项](dsh-operator-s5-closeout-20261030.md)。归档仍在同一本机文件系统，Linux runner 整体环境未制作离线安装镜像；私有原始日志不作为公开分享包。插件安装生命周期已在隔离 DSH profile 验证；个人默认 `desktop` profile 保持未修改。若今后要在个人 profile 中试装，应作为单独的用户环境变更操作，不属于本轮 S6 验收。
