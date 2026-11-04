# 一期正式对照：租卡前预登记与实例绑定

准备采用两阶段冻结。`preregister` 固定本地设计、模型元数据、源码及完整任务契约；拿到实例后才采集并绑定真实 GPU、运行时和 runner。预登记本身不读模型密钥、不调用模型、不连接 SSH，也不物化四个大任务的主输入数组。

计划编号继续为 `phase1-agent-comparison-v2`。它与原来的小任务离线演练、`phase1-entry-integration-v1` 接入检查分别保存，不能拼接成正式结果。

## 1. 当前固定设计

| 项目 | 冻结值 |
|---|---|
| 任务 | FP16 Triton：Matmul 4096³、ReLU 16384×6144、Softmax 4096²、ReLU 1024×768 |
| 顺序 | 同卡串行，A→B、B→A、A→B、B→A；八格预登记，失败与未执行也保留 |
| A | 原生 DSH 的 shell/edit/read，自行写核，通过公共 `check.py` 调用完整验证 |
| B | 相同 DSH 宿主，通过 operator 插件委派给生产自主 worker |
| 模型请求 | 同一 provider/model，温度 0.1、输出上限 32768、`n=1`、顶层 `enable_thinking=true`、`reasoning_effort=medium` |
| 每格预算 | 30 次模型调用、2400 秒墙钟、20 次 oracle 验证预留；失败不退款、confirm 保留不可用于开发 |
| B 策略参数 | max_rounds=8；它不是 A 的提交次数，A 只受共同验证预算约束 |
| 独立终评 | 终稿/契约哈希绑定、新验证目录、最多 1 次验证、660 秒独立截止 |
| 结果条件 | 完整 L2；至少一次 valid L3 且 speedup≥1.0；另报噪声与带内打平 |

所有主 case、辅助正确性 case、种子、dtype、布局/stride、参考实现来源及摘要写入预登记。硬件目标为尚未绑定具体型号的 NVIDIA GPU；GPU UUID/name/index 属于第二阶段的实测绑定。

A 的远端执行经可信验证服务，不提供任意 SSH 会话或密钥。A 没有 operator 插件、生成委派或 core CLI；B 的提示词、知识、工具面和上下文分解是工作流差异，均保留身份，不把比较外推为任意 Agent 的总体收益或某一个组件的独立增益。

## 2. 冻结模型与本地身份

正式模型沿用用户提供的阿里云 DeepSeek 桌面配置。实际 ID 和专有 base URL 只写入仓外预登记包。公开示例不包含真实密钥或专有地址：

```json
{
  "base_url": "https://YOUR_ALIYUN_ENDPOINT/compatible-mode/v1",
  "id": "YOUR_CONFIGURED_DEEPSEEK_ID",
  "api_key_env": "LLM_API_KEY",
  "temperature": 0.1,
  "transport": "live"
}
```

在 Python 3.12 项目环境运行：

```sh
python -B -m benchmarks.experiments.phase1 preregister \
  --model-config /absolute/model-metadata.json \
  --output /absolute/new-preregistration
python -B -m benchmarks.experiments.phase1 check-study \
  --preregistration /absolute/new-preregistration/preregister.json
```

输出包含 `preregister.json` 和待填写的 `instance-config.example.json`。不提供 `--model-config` 时允许登记未绑定模型，但不能进行实例绑定。

身份包括全部控制面/远端源码、提示词与 skills、部署 payload、量具/校准/终评来源、DSH 与插件、Python/依赖版本和 git 工作树事实。未提交源码按内容哈希固定；不能仅用旧 HEAD 重建。后续代码、提示资源、依赖、DSH 或任务变更会拒绝绑定/继续派发，需要重新登记。预登记不宣称服务端权重版本或实际遵守思考开关已经得到验证。

## 3. 拿到卡后采集与绑定

先准备专用远端运行时、部署匹配 runner，完成已有的 G1–G5 探针。本入口不会租卡、上传 runner 或安装远端依赖。将实例配置填入仓外 JSON：

```json
{
  "connection": {"host": "GPU_SSH_ALIAS", "work_dir": "/absolute/prepared/runtime", "timeout": 300},
  "gpu": {"index": 0, "uuid": "MEASURED_UUID", "name": "MEASURED_NAME"},
  "probe": "/absolute/card-probe.json",
  "instance_receipt": "/absolute/instance-receipt.json"
}
```

以下采集命令会实际连接所指定的 SSH 实例，读取版本、NVIDIA 身份/驱动及所有 runner 哈希；不生成候选、不执行 kernel、不调用模型：

```sh
python -B -m benchmarks.experiments.phase1 collect-instance \
  --preregistration /absolute/new-preregistration/preregister.json \
  --instance-config /absolute/instance.json --output /absolute/instance-receipt.json
python -B -m benchmarks.experiments.phase1 bind-study \
  --preregistration /absolute/new-preregistration/preregister.json \
  --instance-config /absolute/instance.json --output /absolute/new-bound-plan
```

五门不全绿、GPU 不一致、运行时版本缺失、runner 有一个文件不匹配，均拒绝绑定。采集回执与预登记哈希关联；纯 `bind-study` 只读已有回执。正式派发每格及终评之前还会重新读取该实例，运行时/驱动/GPU/runner 漂移立即停止该格，不能只相信旧回执。

```sh
# 会调用配置的实际模型及 SSH 实例。运行时安全注入 LLM_API_KEY
# 或已有的 LLM_API_KEY_FILE，不能将密钥值写进预登记 JSON。
python -B -m benchmarks.experiments.phase1 run-connected \
  --plan /absolute/new-bound-plan/plan.json --output /absolute/new-execution
python -B -m benchmarks.experiments.phase1 summary --execution /absolute/new-execution
python -B -m benchmarks.experiments.phase1 cancel --execution /absolute/new-execution
```

## 4. 采样与观测审计

采样统一发生在 LLM 预算预留和请求哈希之前。实验 wrapper 覆盖 B 各角色的原有温度/上限，并通过实际自主 supervisor/worker 的子进程传递；DSH 网关也在预留之前统一请求。普通 CLI/MCP 的默认行为保持原样。

`sampling-inner.jsonl`、`sampling-outer.jsonl` 保留请求原参数、实际参数和请求摘要；这些是采样意图，不是额外计费调用。汇总把它们与共享预算、生产模型账本和外层实际请求逐一关联，缺一项不能报 `sampling_aligned=true`。配额包含全部外层/内层模型调用，不按角色重置。

保存原始宿主轨迹和验证来源，规范化工具观察仍为 partial；人工消息/外部编辑未完整接入，人工总数与分钟维持 unknown。API usage 仅认已校验计数，DSH 外层 usage 可能缺失，费用不能由调用预留估算为事实。外层文件边界不覆盖可信内层或 SSH 候选，不宣称对敌意代码隔离。

租卡前的本地验证使用实际安装的 DSH、CPU FakeBackend 与本地 HTTP 模型 fixture；它可以证明 A 自行写代码、B 经插件委派和实际 wire 参数一致。大任务性能、真实 provider 行为、正式八格及 GPU 资格均继续待执行，准备完成不等于收益已经成立。

## 5. 初版 4K 预登记与本地验证（2026-11-04）

已按当前桌面配置登记阿里云模型 `deepseek-v4.1-flash-1017`；专有地址仅保留在仓外私有资料中。持久目录为 `/Users/zh/op-agent-delivery/phase1-study-20261104/`，其中 `preregister.json` 是冻结对象，`verification.json` 是本地验证汇总，归档包含 378 份源码/资源及逐文件摘要。

预登记摘要为 `190909f60efc04f653de558c84b36f2367d6d106b07787b638200b9fafecc17a`。状态为 `preregistered_pending_instance`：模型元数据已配置，GPU、远端运行时、实际部署及 provider 行为均未伪造为已验证。

Python 3.12 相关范围 `tests/unit/phase1 tests/hygiene tests/acceptance/phase1` 为 **472 passed / 1 skipped**，101.26 秒；跳过项是未显式提供工件环境的独立安装验收。其后补充传递控制脚本的身份覆盖，冻结相关子集 **25 passed**，15.44 秒。实际两臂 DSH 本地服务已经观察到全部请求的统一采样参数、共享预算核对、完整 L2 独立终评及 test_only/L3 无资格标识，未调用阿里云、SSH 或 GPU。

源码快照用于审查与恢复，当前预登记绑定原本机解释器、DSH 和工作树身份；搬到新环境后应重新登记，不将快照冒充已安装产品或已完成八格对照。既有 v0.7 候选 wheel/tgz 未重打，其 145 个运行文件与本轮登记源码已逐字节核对一致。

## 6. 32K 采样复验与新预登记（2026-11-04）

A、B 共用 DSH 输出上限和流空闲超时。32K 的单次请求上限为约 524 秒，并受该格剩余墙钟约束。B 的委派进程在启动前接收相同上限。网关允许最多 65,536 个流事件、32 MiB 响应，以容纳细分 SSE 流。

预算核对把生产账本的 `prompt_tokens_details.cached_tokens` 与共享账本的 `cache_read_tokens` 按同一计数比较。采样对齐同时检查实际外层请求、精确的 token 预留，以及内层请求摘要对应的采样记录。

实际 DSH 本地验收中，A 发出 2 次外层请求，B 发出 9 次外层和 2 次内层请求。13 次请求都符合冻结参数，每笔预留 32768 token，请求摘要与共享账本逐笔对应。两臂独立完整 L2 均通过，B 的缓存计数也完成对账。

本轮新预登记沿用 v7 的阿里云模型元数据，历史 v5、v6 的真机试跑和 v7 的实例绑定继续保存在原目录。新预登记的实例身份待重新核对和绑定。

相关本地回归共 571 项通过，另有一次实际 DSH 双臂验收。回归覆盖普通接入、32K 长流、账本核对，以及探针和标定脚本。v8 的私有预登记、源码摘要和验证回执保存在 `/Users/zh/op-agent-delivery/phase1-study-20261104-v8/`。
