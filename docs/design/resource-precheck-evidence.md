# S3：资源预检的证据边界

工具契约 **8**、DSH 插件 **0.1.8**。本次修复的是确定拒绝的依据；参考实现、容差、硬件上限、任务预算与默认生成策略保持。

## 为什么不能只把 4 字节改成任务 dtype

历史预检看见任意 `tl.float32` 就把全部 tile 按 4 字节计算，将 `(BM×BK + BK×BN + BM×BN)` 当作共享存储。FP16 输入、FP32 累加的源码会触发 384 KiB 代理值；同一公式改用 2 字节会得 192 KiB。

这两个数都不是物理占用证明：累加器可能不是 shared，输入可以在中间转型，流水参数也不直接证明共享缓冲副本。非 matmul 的 `BLOCK_*` 乘积同样不能当作共享存储下界。S3 因此不把所有存储统一改成任务 dtype。

10-19 原始误拒源码与日志缺失，S0 H04 仅保留文档记录。新增 fixture 从保留的 H07 FP16/FP32 累加模式提炼，并把 tile 改为文档中的 256×256×64 来复现旧算法；它是**重建回归**，不是找回的原核，也不是已在 H20 编译成功的证明。

## 三态与来源

`ResourceReport.assessment` 有三种值：

| 值 | 含义 |
|---|---|
| `unknown` | 缺少有效共享存储证据、作用域/身份不匹配，或硬件上限未知；继续后续原生检查 |
| `within_limit` | 已确立字节未超过既有宽松拒绝阈值（limit × 1.2），不是 launch fit/occupancy 证明 |
| `over_limit` | 已确立共享字节超过该阈值，预检拒绝 |

`feasible=True` 只表示本门未拒绝。资源未知不会自动产生 L1 通过；后端编译成功、缓存命中或失败仍按原路径记录。已知字节但上限未知时可保留字节，assessment 仍为 unknown。

只有 trusted harness 能提供 `SharedMemoryEvidence`：

- `trusted_shared_layout`：每个实际共享缓冲区独立声明 name、shape、存储 dtype、同时存活关系与 copies。评估器按各自 dtype/copies 计算；不把累加器、输出或任务 dtype 自动加入，也不从 num_stages 补出副本。
- `native_compiler`：当前具体编译产物的共享字节回执，保留其来源；不与声明 layout 混种或再相加。

资源证据的数据、来源和绑定必须由 harness 在调用前确立。类型检查及哈希不是来源认证，候选源码自报字典、注释、MCP/CLI 参数或用户 `perf_data` 不能创建这种权威。当前没有从 runner 的第一个 cache kernel/raw metrics 自动拼接全候选资源回执；原生编译/launch 的实际错误仍由既有执行路径保存。

维度、副本、原生字节必须为有界整数；布尔值、非有限/超大数值、未知 dtype、动态维度、重复缓冲区名或混合证据保持 unknown。解析不执行源码，不导入 Triton，也不调用候选函数。

## 配置和绑定

源码元数据按 AST 逐组保留可解析的 `triton.Config` 参数、stage 和其他选项；动态或不完整配置显式未知。不再用不同组的各轴最大值构造一组新配置。兼容的 `block_shapes` 只展示字面量摘要，不作为物理形状。

源码绑定使用完整 SHA256；只有明确的单 jit helper、单次直接 launch 和字面量编译配置可接收单份源码资源证据。autotune、多 helper、多 launch、循环/推导式中的 launch、动态 constexpr/编译参数和参数展开都不能由单份 native 回执代表。

模板资源绑定包含 backend、template_id、instantiator_ver 和全部带类型参数，独立于去重键。非法/超界参数保持未知，错版本或错参数的证据不剪枝。这是 **harness 当前执行上下文**的资源契约：架构、编译 flags、任务输入 dtype 等仍须由 harness 在传入前核验；不能仅靠该 hash 跨上下文缓存/复用原生回执。当前生产路径没有新增此类缓存或用户注入入口。

## 接线与效果范围

- L1 保存 `metrics.resource_precheck`，包含 assessment、字节、来源、缓冲区与配置；S1 的完整结果文件保留这些事实。
- CoderPipeline 与 L1 共用资源评估；未知不触发资源修复，不改变语法门原结果。
- 模板和搜索候选没有已确立资源证据时保留；Triton 参数空间不再仅按逻辑公式缩格。显式证据下的超限、同一 1.2 系数和全摘光回退继续有正反例。
- 逻辑代理不能再进入 `smem-bound` / occupancy 硬限制归因，不能据此购买结构升级的额外 LLM 调用。

未知时交原生检查可能增加单次编译尝试，但不增加预算上限、不使候选取得 L2/L3 或入库资格。旧逻辑公式作为兼容计算函数保留，禁止用于确定拒绝、剪枝或硬限制证明。

协议清单由 **64 → 68**，增加 candidates、coder_pipeline、triton_matmul_template 和 agents 搜索入口；resource_check、Verifier、Conductor 已在清单中。旧实验按原有源码漂移规则识别，历史包及判定不改写。

本地回归和重建复现不代表 GPU 性能或编译资源实测；正式成功率和真实资源验收应使用修正后的冻结版本。S4 另行冻结真实任务与预算，Q1 暂停决定保持。
