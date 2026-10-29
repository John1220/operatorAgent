# LayerNorm 列跨距诊断（2026-10-23）

已将 `missing_input_column_stride` 接入现有 `operator_repair` → Fixer 链路，补充此前[真实 H20 配对验收](../research/operator-quality-real-paired-20261023.md)未覆盖的布局错误。线上诊断策略为 `candidate-repair-diagnoser-policy-v2`，最多输出一个 finding，优先检查有保真证据的跨距问题，再检查原有尾列方差模式。离线单候选诊断保持原有方差识别器。

## 支持条件

新识别器只在三个条件同时成立时输出 `supported_hypothesis`：

1. **冻结用例与数值失败一致。** 完整 suite 身份、失败 case ID、shape、pattern、layout、dtype 必须与任务声明一致。完整数值比对的元素数、坏点整数、匹配比例、参考有限性、覆盖率和容差必须完整且相互一致。外层状态未知、矛盾或编译失败不支持该假设。
2. **真实布局回读一致。** 除 `input_fidelity_verified is True` 外，还用既有 `check_readback` 重新比对完整 expected/observed 描述。要求二维 FP16/FP32、正行跨距、列数大于一且列元素跨距大于一；字节跨距必须能整除元素宽度，view 不得越出 storage。当前仅支持可写的 `torch-storage-v1` CUDA 输入，设备回读必须存在。dtype、offset、storage 与内容摘要均进入布局绑定。
3. **源码寻址链明确。** 只识别 canonical imports、单个 JIT helper、`kernel(X)`、二维 `X.shape`、`torch.empty_like(X)`、whole-row grid 和单次 masked load/store。追踪 helper 参数与 launch 实参，确立原始 X、运行时 `X.stride(0)`、row index、column lane，解析出 `X + row * row_stride + column`，且加载值接入返回输出的 store 数据流。

不存在字符串匹配式的「缺少 `stride(1)` 就判错」。正确双轴寻址、显式列偏移因子、输入先复制为连续布局、复杂别名、其他 grid、分支和不支持的源码形态均返回 `unknown`。模块或参数遮蔽、局部重绑定、先使用后定义、死分支 load、短路表达式、推导式与生成器均不在支持域内。除原有 32 KiB / 4096 AST 节点限制外，地址和依赖展开分别受 32 层、1024 步限制。

## 诊断内容和接线

诊断包含源码 statement 的精确 span/excerpt/hash、冻结失败用例身份、字节及元素跨距、offset、storage 大小、expected/observed 摘要和输入设备；这些事实进入 feedback hash。返回前及 Fixer 发送前均从源码、契约和失败结果重算，重新 hash 的篡改也无法通过。成功的 L1 结果不进入绑定，因此修复链路传全部结果、Fixer 只传失败结果时仍一致。

标量地址见证以逻辑列 1 比较相对偏移。本次真实 FP32 失败的元素跨距为 `[64, 2]`：源码列偏移为 1 元素 / 4 字节，输入声明要求 2 元素 / 8 字节。它证明受支持 AST 寻址模式与输入布局不符，不证明所有数值误差均由此引起，也不是 GPU 执行或正确性资格。

修复义务要求传递原始输入的运行时列跨距、在输入 load 地址中使用它、独立派生输出跨距，并保留 wrapper、mask、参考、容差和完整 suite。诊断不执行或改写候选，不增加调用预算；新源码仍须 fresh L1 和完整 L2。修复未改源码、只有格式或注释改动时，原有守卫继续停止。

新模块 `operator_agent/repair_stride_diagnoser.py` 纳入生产协议身份。实现变更后必须冻结新评测计划；历史验收结论和旧额度不被重新解释为当前执行资格。

## 保留证据复分析

用新逻辑静态读取旧验收的两组首发与修复后候选，以及对应验证 JSON：两组首发均输出同一个 `missing_input_column_stride` 假设，修复后均不输出该假设。两份首发诊断 hash 相同，源码/失败反馈在 Fixer 输入方式下也能通过重算。

证据保存在[复分析摘要](/private/tmp/operator-stride-quality-20261023-ehl721ji/retained-interpretation.json)。原始验收文件未修改；当时的诊断仍为 `unknown`。本次复分析新增模型、SSH、GPU 调用均为 **0**，不能当作更新后的真实修复通过率或额外收益。

## 验证

新增 `tests/test_stride_repair_diagnosis.py` 共 **110 项**，与原在线诊断测试合跑 **138 项通过**。包括真实失败投影、合法寻址及复制反例、冻结身份和布局漂移、设备与类型混淆、死分支、受限展开、诊断发送前重算和 fresh 完整 L2 重验。

冻结生产源码后的完整回归为 **4707 通过、1 跳过**；原有非有限参考测试产生 2 条 warning。首轮发现的打包协议文件计数断言已由 51 更新为 52，并显式检查新模块；6 项后台测试所需的 localhost HTTP 假服务在允许本地监听的环境下通过，未调用真实模型服务。Golden 串行比较 **4 PASS / 0 REVIEW / 0 FAIL**，DSH 插件与宿主契约 **39 项通过**；未修改基线或阈值。

另对公开 `operator_repair` MCP 路由做假件回放：一次模型假响应、两轮验证假结果，紧凑响应 **4293 bytes**，完整源码位置与地址见证落在证据文件；它验证 DSH 接线，不是新数值实测。完整日志、源码身份和复分析摘要保存在 `/private/tmp/operator-stride-quality-20261023-ehl721ji/`。

随后完成[固定缺陷的真实修复对照](../research/operator-quality-seeded-repair-20261023.md)：跨距诊断真实进入模型请求，两组均一次修复通过完整 L2，额外修复收益为 0；诊断组使用 7,309 tokens，基线 5,448 tokens。原始历史报告保持不变。
