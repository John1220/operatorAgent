# DSH 算子开发插件 S6 交付验收（2026-10-30）

状态：**按用户调整后的范围完成。** 用户确认暂不录制视频，以可复核安装说明、脱敏案例包和演示脚本作为演示材料。本报告验收本地交付工件、真实 DSH CLI 的 profile 安装生命周期和离线案例复核；未修改个人默认 DSH profile，也未发布到包仓库。

## 安装工件

- 当前交付包含 Python wheel `operator_agent-0.6.0-py3-none-any.whl`，SHA256 `8cd75a3864d7a300fd71f33360f81c3ba92d240b8889cb16289a525497898906`，以及 DSH 插件 `dsh-operator-agent-0.1.9.tgz`，SHA256 `881e8384598a3ff360b16eea901eed56ed31cb9cbeb6c69c604239bb5ecb6af6`。这两个本地构建版本号沿用现有元数据；哈希用于区分旧构建，未上传发布仓库。
- 已有仓库外干净安装记录：隔离 Python 3.9 环境、`pip check`、CLI capabilities、mock 完整 L2、MCP 握手及 8 工具发现通过；真实 DSH runtime/SDK 宿主契约 **9 passed、0 skipped**。详见 [S6 准备记录](dsh-operator-s6-preparation-20261027.md)与[当前工件 H20 验收](dsh-operator-s5-installed-gpu-acceptance-20261030.md)。
- 本次另以当前 `.tgz` 在临时 `DSH_HOME` 下通过 DSH CLI `0.1.1-rc.2` 实测 profile 生命周期（pnpm `11.10.0`）：添加后，包依赖与 `dsh.profile.bundles` 均包含插件；`dsh --profile acceptance --dump-config` 合成配置包含 `mcp-operator`（命令 `operator-agent-mcp`）和 `operator-tools`（命令 `operator-agent`）；移除后，依赖、bundle 注册和安装包均消失。整个生命周期验证未调用模型或 GPU。此项验证的是隔离 profile，不声称已安装到个人 `desktop` profile。
- 初始探查曾触发 DSH 在默认 `$DSH_HOME` 下创建空的 `temporary` profile；确认只含自动生成的初始配置后已移除。个人 `desktop` profile 未被操作。
- 当前生命周期的添加/移除 stdout、合成配置片段、前后 profile 状态 JSON、工件 SHA256 和离线校验结果另存于 [S6 原始验收输出说明](../../output/plugin_completion/20261030-s6-acceptance/README.md)，文件完整性索引为同目录的 `evidence-manifest.json`。

## 证据与演示材料

- [交付包](../../output/plugin_completion/20261030-s5-closeout-v1/)的 159 个清单文件逐项 SHA256 校验通过。交付包还包含源码快照、历史私有证据归档、安装日志和 P1 真机记录；私有归档仅供本机持有人核验，不作为公开分享内容。
- [脱敏案例 ZIP](../../output/plugin_completion/20261030-s5-closeout-v1/shareable/s5-plugin-cases.zip) SHA256 为 `e74e0e621f2fbbad87de6488b1fa43bedb2efce26c5b0991a4b7e7375929bb8d`。在仓库外解压后，以 `python3 -I -S verify.py` 通过离线检查：2 个历史 LayerNorm 插件案例，`verified=true`；不联网、不调用模型/GPU、不执行候选。该校验仅证明包内一致性，不提供历史真实性签名。
- [三分钟演示与简历口径](../design/dsh-operator-s5-resume-demo-20261030.md)可直接按脱敏 ZIP 逐项讲解；另有[S4 CLI 补充案例脚本](../design/dsh-operator-s6-case-demo-20261027.md)。二者都标注历史回放及性能证据边界，本轮按用户决定不录制视频。
- [安装指南](../../integrations/dsh-operator/README.md)已补充使用本地 wheel 与 `.tgz` 工件的安装步骤，并保留 profile 卸载方式。

## 结论边界

S6 的安装与材料交付在上述范围内完成。GPU H20 完整 L2 12/12 的含义及条件见独立验收报告；它不构成 A/C 性能或开发效率优势。原 S5 24 格重复对照仍未完成，插件整体优于原生 DSH 的结论未建立。

本次交付不包括包仓库发布、Linux runner 离线镜像、异地备份或个人默认 profile 的改动。历史私有证据仅保存在本机同一文件系统，不存在第二份副本；**单副本是用户 2026-10-30 明确接受的决定**（见[S5 结项报告 §5](dsh-operator-s5-closeout-20261030.md)），不列为待补项。只分享明确标记的脱敏案例 ZIP。
