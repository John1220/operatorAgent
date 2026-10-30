# S5 最小计划 P1 验收（2026-10-29）

状态：**P1 驱动与共同判定已完成，P2 差距审计已完成，正式 24 格未放行、未执行。** 本轮真实模型调用 0、GPU 执行 0、真实 SSH 调用 0。

## 实现结果

- `s5_plugin_workflow` 新增 `study` 模式和 `s5-plugin-study-plan-v1`：固定前瞻六任务 × A/C × 两次重复，共 24 格。原有预检、C-only、入口及修复模式仍独立保留。
- 按任务/重复/组分离工作目录与账本身份；组顺序按任务和重复交替，同一格或执行目录重派拒绝。加载时核对任务、提示、模型/预算、服务契约，以及主计划观察设备与实际验证服务的 GPU/后端身份一致性。
- 新增 `s5_candidate_extraction.extract_final_source`，A/C 在 study 模式共用：保留原始答复，接受纯 Python 或唯一明确的 Python 代码块；源码字节及换行保留，拒绝歧义围栏、重复公开 kernel 定义及无有效源码。只提取最终答复，不读取历史已通过候选作为替代。无开发验证历史但有有效终稿时仍可独立终评。
- 交付格式拒绝记为普通未提交，保留已发生成本，不消耗 GPU 终评，不误触发设施停止。验证返回的源码 SHA256 与契约必须匹配；其余意外错误保留 unknown 并停止后续派发。
- 新增 study 汇总与离线 CLI：每组 12 格、12 个重复配对、胜平负/unknown、全分母成本及双方均成功时的成本差；缺失 token、首次正确调用水位、L3 及费用不补造。
- `s5_study_readiness` 在 `execute` 和直接 `run_one` 前拒绝正式资格不足的请求。删改 mode 不能绕过 study schema 的门；人工报告自称全部通过也不能补出当前实现不存在的网络/候选隔离能力。当前 gate 故意没有成功派发路径，需真正补齐 P2 实现与探针后更新代码及计划身份。

## 回归与模拟证据

最终相关回归 **252 项通过，49.25 秒**。覆盖工作流、study/历史汇总、共同提取、资格拒绝、实际 DSH 假模型/MCP 接线、共同配额、共享账本和模拟远端验证。不是全仓回归或真实模型/GPU 成绩。

首轮统一回归误设了尚不存在的临时目录父路径，导致 pytest fixture 初始化错误；随后改用仓库外独立临时目录，先通过 247 项，再补设备一致性拒绝用例后通过最终 252 项。前次日志及最终日志分别保留，不把初始化错误归因为产品失败。

| 模拟场景 | 固定格数 | 实际模拟派发 | 结果及分母 |
|---|---:|---:|---|
| 全批完成 | 24 | 24 | 预置 passed 8、failed 8、unknown 8；其中普通调用耗尽继续后续格 |
| 设施停止 | 24 | 7 | passed 2、failed 2、unknown 20；后续 17 格未执行，不算模型失败 |

上述排程测试显式替换了 `run_one` 与资格门，只验证调度和汇总，并非 24 次真实 DSH 模型会话。已有实际 DSH 假模型接线由本轮其他相关测试覆盖。两个模拟场景的 24 份逐格记录、计划、汇总已复制到持久证据目录，从记录重算哈希一致。

另用实际生产 `prepare`、前瞻任务语义和既有无模型 readiness 离线冻结/加载了 24 格候选计划，哈希 `f4372bd28bce6094b843b4abb5b92e133485171d7bfcf4987de895f6975a4b2b`，状态 `frozen_study_candidate`，没有正式报告。调用真实 `execute` 入口立即得到 `FormalReadinessRefused`，执行目录未创建，没有读取 provider 凭据或进行资源查询。候选计划中的 GPU 身份来自历史配置，仅用于离线一致性核验，不是当前 GPU 可用快照或正式放行凭据。

证据目录：`output/plugin_completion/20261029-s5-minimum-p1-v1/`。`final-regression.log`、`final-test-invocation.json`、`offline-scheduling/`、`offline-candidate-plan/`、`formal-dispatch-refusal.json` 及 `frozen-source/` 均已保存。验收摘要 `acceptance-summary.json` 哈希为 `bf1fa3113dd48fe1917d714ba95d54b15d3d969510a4f91bba401ff96ed2d9cb`；Git 忽略目录需随证据包单独保留。

离线重算示例（输出文件必须未存在，不调用模型/GPU）：

```sh
python3.9 -m benchmarks.s5_plugin_study_summary \
  --plan output/plugin_completion/20261029-s5-minimum-p1-v1/offline-scheduling/complete/frozen/plan.json \
  --execution output/plugin_completion/20261029-s5-minimum-p1-v1/offline-scheduling/complete/execution \
  --output /tmp/s5-study-recomputed-NEW.json
```

## P2 的实际阻断与代价

详见[正式执行条件审计](dsh-operator-s5-formal-readiness-audit-20261029.md)。本机无模型实测已确认 A/C 文件围栏下 Python 和 shell 子进程能连接非本轮网关的合成回环端口。远端候选仍在 SSH 用户 runner 同一解释器加载，没有受验证的文件/网络/参考材料和完整后代清理边界。二者均未修复。

最小后续可保留 macOS DSH，分两项改造：G1 为仅 study 启用的精确回环端点白名单及正常/拒绝探针；G2 为 Linux 远端验证实例的受限候选 worker，明确候选与监督器/参考的分离，验证 CUDA/Triton 依赖、设备和缓存访问及取消清理。无需先迁移整套 DSH，但 G2 已是执行适配改造，不能按改参数估计成本。

建议下一轮只先定稿 G2 的 worker/数据交换/参考分离协议，并做无模型、无 GPU 的文件、网络和后代清理原型；原型通过后单独验证实际 GPU 路径。是否可用现有命名空间而无需额外权限、GPU 兼容和参考分离成本尚未知，不承诺工期或金额，不自动部署特权服务或改变共享宿主。该范围须在真实改造前明确，不用放宽正式标准来推进 P3。

P3/P4 仍未完成，S5 尚未封版，不进入 S6。
