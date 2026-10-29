# S5：三组启动路径与统一验证接线（2026-10-28）

状态：**接线骨架完成，一次三组预检通过；不是正式 S5 对照，不计入正式分母。**

本轮按用户决定执行：只做支撑「1 个任务 × 三组试运行」所必需的接线，用假模型/假上游/mock L2 验证，不发起真实模型请求、不占 GPU；且该试运行**提前确定不计入正式分母**（写入冻结计划，运行前即存在，不能看结果后再选）。

## 新增组件

| 文件 | 作用 | 边界 |
| --- | --- | --- |
| `benchmarks/s5_runtime.py` | 冻结 B/C 入口所需 numpy 依赖目录并计算树哈希；`PYTHONNOUSERSITE=1` 下显式可见 | 本机依赖冻结，不是 Linux 依赖清单或安装证明 |
| `benchmarks/s5_preflight_plan.py` | 冻结一个任务、A/B/C 固定顺序、调用/输出/墙钟预算、验证后端；写一次、哈希可复核、声明不计分 | 计划只约束本进程的接线，不是防篡改执行环境 |
| `benchmarks/s5_arm_env.py` | 三组安装面：A 裸 DSH；B 加 core CLI 到 `PATH`；C 加同一 CLI + 插件 + MCP 客户端；并做环境可达性探针 | 探针只说明环境可见性，不是文件/网络隔离 |
| `benchmarks/s5_verify_service.py` | 监督器侧统一提交入口：先落盘候选哈希再评测，绑定计划的任务/契约/suite 身份，拒绝未计划组与重复提交 | 只有 mock 后端接通；`h20` 后端返回 unknown，不模拟结果 |
| `benchmarks/s5_three_group_run.py` | 串行跑三组、逐组记录执行状态、共享账本事实、环境探针、验证结论，并做纯函数汇总 | 单机 macOS fixture，假模型，无 GPU/性能结论 |

`tests/test_s5_three_group_preflight.py` 覆盖计划冻结、安装面差异、探针事实、统一验证的拒绝与 unknown 路径、汇总纯度与 `unknown` 保留、单组失败降级，以及一次真实 DSH headless 三组端到端（15 项；S5 全部专项测试 **228 项通过**）。

## 预检结果

计划 `layernorm-tail`，固定任务文本与前序纵向试验一致（有测试保证），模型为进程内脚本化假模型，预算每组最多 2 次调用 / 单次 1024 输出 token / 总 2048 / 墙钟 300 秒。

| 组 | 安装面 | 状态 | 执行 | 墙钟 | 账本 | 探针 |
| --- | --- | --- | --- | --- | --- | --- |
| A | 裸 DSH | passed | finished | 1.798 s | 1 预留 / 1 完成 / usage unknown | 不可导入 core、`PATH` 无 CLI |
| B | DSH + core CLI | passed | finished | 2.647 s | 1 预留 / 1 完成 / usage unknown | 可导入 core、`PATH` 有 CLI |
| C | DSH + 同一 CLI + 插件 | passed | finished | 2.990 s | 1 预留 / 1 完成 / usage unknown | 可导入 core、`PATH` 有 CLI、插件/MCP 入口加载 |

- 三组候选同哈希（190 字节 fixture），各自 1 次假模型请求；交付统一验证入口后得到**同一** verdict 哈希 `02c5acd210cf0623397e04dba2a1a4ee17acf2909cb891c6efe6f2be0dc98f44`，均为完整 12 case suite 的 mock L2 通过。墙钟只反映启动与 fixture 开销，不构成性能数据。
- C 组 stderr 为空、无 entry 激活失败，说明插件与 MCP 入口可加载；本轮**没有**调用 `/op` 或 MCP 工具，`tool-bash`/`tool-fs` 等在三组均关闭，因此三组的工具使用差异未被本轮覆盖。
- `repo_canary_readable=True` 在三组均成立：同机同 UID 下 A 仍能读本仓，A 的“裸 DSH”仍**未被证明**；探针如实记录了这一点而不是升级成隔离结论。
- 计划哈希 `e3f590480b8475662c6954c07cc72879db1a4cf2d44492f8a261c08814bf05a7`；汇总哈希 `009c179b965d5ec48950eb1ce13cf5d255583d4c7c112c47277b188d5a9b1c8e`；运行时依赖树 931 文件、`6df07688809d09848bd013da66778acfbd294bbb511d652b577c04443947c748`。
- 原始记录 `output/plugin_completion/20261028-s5-three-group-preflight-v1/`（约 11 MiB、608 文件，Git 忽略，需单独备份），含冻结计划、启动标记、每组 launch/probe/事件/账本/候选/提交/结论与汇总；汇总可由三份记录重算得到同一哈希（有测试）。

## 过程中发现并修正的问题

- 之前的 MCP 与工具纵向试验在当前环境下失败：`PYTHONNOUSERSITE=1` 隐藏了用户 site-packages 中的 numpy，MCP worker 与 core CLI 无法导入。现改为显式冻结依赖目录（`output/plugin_completion/20261027-s5/runtime-deps/site`，含 931 文件树哈希），MCP/工具/三组入口共用；相关纵向测试恢复。这是环境漂移暴露的真实缺陷，不是本轮接线引入。
- 失败组现在也必写账本快照（原来 headless 抛错时 `budget.json` 会缺失），保证中断组同样进入分母。

## 仍然未知（正式试运行前必须闭合）

1. **真实模型路径**：本轮只经过回环假上游与 SQLite 账本；真实 DSH 默认 Responses 路径、真实 provider 记账、不可绕行的唯一网关均未验证。
2. **统一真机验证**：`h20` 后端未接，统一入口对真机会返回 unknown；需要把 SSH `RemoteExecutor` 的 runner 版本、输入保真、同卡执行、停止与 owned 会话清理接进同一提交入口，且不允许把 mock 结果当真机证据。
3. **隔离**：A 组仍可读本仓与插件；需要容器/命名空间级别的文件与网络边界，并用绕行测试证明 A 不能读 core/插件、Docker socket、宿主凭据，三组也无法互相读取。
4. **工具面差异**：本轮关闭了 shell/文件工具，三组没有真正使用各自工具；正式试运行需要按组打开一致的基础工具，并让 B/C 实际调用 core CLI、C 实际调用 `/op` 或 MCP 工具。
5. **预算权威**：仍然只有“配合的调用”受账本约束；`participant` 标签不是进程身份，同机可绕过。
6. **任务与预算封版**：14 项草案未封版，正式每组调用/验证次数、时间与 GPU 时段未定。

按上述顺序，第 1、2、3 项决定一次真实三组试运行是否值得做；在第 3 项之前，即使跑通，A 也只能记为“环境变量层面无 core”，不能记为裸 DSH。