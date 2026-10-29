# S6 安装、证据包和演示准备（2026-10-27）

状态：**进行中**。S4 的真实任务可以作为演示材料；S5 三组对照尚未执行，不能声称插件的整体贡献已评估。

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

已准备 [S4 脱敏案例包](../../output/plugin_completion/20261027-s4/portable-case-bundle-v2.zip)及 [三分钟演示脚本](../design/dsh-operator-s6-case-demo-20261027.md)，覆盖错误被拦住、修复后完整 L2 通过，以及性能读数未知三种情况。包内白名单投影与独立标准库验证器已在仓库外运行；候选和汇总篡改、敏感 URL 注入均有拒绝测试。额外用本机现有两项凭据值检查 ZIP，未发现匹配；数据文件未发现本机个人路径或 URL。这只是针对已知值及所检查模式，不保证发现所有敏感信息。

该 ZIP 是脱敏投影和内部一致性证据，未提供签名、GPU 重演能力或完整原始日志，不能冒充完整可移植原始证据包；原始材料必须继续私密保存。S5 三组对照未执行，不作插件整体贡献声明。

## 尚未完成

完整可移植原始证据包、真实用户 profile 的安装/卸载验收和视频或交互演示尚未完成；S6 在这些交付完成前不勾选整体完成。
