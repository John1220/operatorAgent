# 冻结参考身份与精度语义：LayerNorm fp16 CUDA v2

日期：2026-10-08。接续 [独立标定](fp16-layernorm-calibration.md)。
这是窄范围 profile 的准入闭环，不代表全域正确性、全部设备支持或性能收益。

## 1. 决策与兼容性

量尺选择：**与指定 CUDA native PyTorch 的数值行为一致**，不是用数学零值、
CPU eager、fp32 提升输入或 fp64 oracle 替换 native fp16 输出。

新版本 `builtin-layernorm-fp16-cuda-v2`：新任务的普通 PyTorch LayerNorm fp16
入口默认选它；显式旧 `builtin-l2-v1` 的该组合继续 unknown，其他七个 v1 组合不变。
未声明 profile 的程序化旧任务不强制迁移。自定义参数、affine、融合、外挂参考等
变体不能借用它。CPU、XPU、未登记的 Torch/CUDA/驱动构建不能冒充通过。

```python
from operator_agent import IntentParser
task = IntentParser().parse("layernorm fp16 triton nvidia")
# correctness_profile = builtin-layernorm-fp16-cuda-v2
# reference_contract = 可信注册表的完整声明，不接受 LLM 自定义
# correctness_suite = 全部 32 组标定用例
```

## 2. 冻结内容

| 项目 | v2 语义与限制 |
|---|---|
| 输入/输出 | 二维 CUDA fp16；输出必须 fp16，且在输入所在 CUDA device |
| 参考调用 | `F.layer_norm(X, X.shape[-1:], weight=None, bias=None, eps=1e-5)` |
| 轴/方差/affine | 末轴、总体方差、无 affine |
| 候选中间精度 | fp32 **声明**；不把有限样本通过解释为内部算术 attestation |
| 参考中间精度 | `native_implementation_pinned`，不从输出 dtype 推断内部累加 |
| 浮点融合 | 允许，但结果必须通过同一容差；不伪造任意 launch 配置的检测 |
| 确定性 | 每个 suite case 重复执行，完整输出摘要一致 |
| 数值门 | fp32 逐元素原 AND 比较，全部有限、完整覆盖 |
| 容差 | atol=1e-5、rtol=1e-2、matched ratio=1.0；任务只能进一步收紧 atol |
| L3 基线 | 仅 native `pytorch_eager`，不回退 torch_compile/CPU/另一参考 |

关闭 fusion 的固定 Triton fp32-centered 候选是标定正例，不是新规则要求。
默认 fusion 候选在 `large_varying` 连续/strided 两组仍失败，**保留失败，不改容差**。
native 单列/非二进制常量的残差也保留，不“纠正”为数学零。

`ReferenceContract` 嵌入任务与 `FrozenContract`，全部内容进入 hash。
缺字段、未知键、bool 冒充数值、登记外定义/构建都拒绝；不能重算任意自定义 hash
就获得可信身份。旧任务无该字段时不写 null，旧契约 hash 不变。

## 3. 执行身份：不只版本字符串

登记构建：NVIDIA H20 / compute capability 9.0、Torch `2.14.0+cu130`、CUDA runtime
`13.0`、驱动 `580.105.08`，并固定：

- Torch git revision、build config 内容 SHA256；
- `torch.nn.functional` 文件与实际 `layer_norm` callable 源码 SHA256；
- 本项目独立 native reference primitive 文件 SHA256；
- libtorch_cpu/cuda/python 三份原生库内容 SHA256；
- 当前逻辑 CUDA device、设备型号/能力、确定性算法开关。

这些指纹来自已有环境的隔离只读探测，不从候选或失败后“自动发现并信任”。
执行回执包含 observed、其内容 hash、登记 reference hash；控制侧重核全部字段。
初次原生库 hashing 较重；只在文件 device/inode/size/mtime/ctime 均相同时缓存摘要，
每个请求仍重查执行环境，不缓存“某次通过”。

**信任边界**：检测正常构建/源码/设备/开关漂移，不是恶意同 UID 进程的安全沙箱；
不声称所有传递依赖、内存 bytecode 或每条 GPU 指令已证明。
逻辑 `cuda:0` 与物理 GPU pin 是两层身份；既有 GPU_PINNING、同卡/同窗证据仍需成立。

## 4. 闭环与 unknown

1. Conductor 在 Router/环境准备/Designer/Coder 前冻结完整契约，并在 Verifier 内
   按 task_id 绑定本次 run 的 hash。参考声明随任务进入正式 result.json，L2/L3
   保留执行回执，L3 保留绑定 hash；生成前绑定是内存状态，不宣称生产已新增独立
   contract.json 落盘步骤。验收 harness 另外保存完整 FrozenContract 到 contract/。
2. 身份探测复用 `gpu_info` action 的可选 reference_contract。后端必须兑现
   REFERENCE_IDENTITY、真实设备、GPU pin、远程 L2 和同设备基线能力。
3. L2 携带完整声明；执行侧在加载候选前、候选调用后且参考使用前核对身份。
   控制侧核对回执、输出 dtype 与原容差；输入仍由实际 Torch storage 回读证明。
4. L3 携带原性能输入的 exact storage/view；测量前后校验身份，原生参考 timed
   callback 不插入指纹探测，不把探测耗时算成基线时间。
   控制侧要求 L2 凭据、native eager 选择、精确输入回读与身份回执。
5. 主输入、确认与次级性能 case 共用门；未知身份不触发本地 fallback、测量重试、
   另一 finalist、Fixer/search、进展计数或正/负语料准入。

绑定后修改容差/套件/参考或移除它们会拒绝，不退回 legacy。
这是**本次 run** 的 hash 绑定；新一次显式 run 可重新绑定，不宣称跨进程/全部历史
resume 的不可变审计日志。

身份无法确立：correctness unknown + reference_identity_unknown；轮内出口
reference_identity_hold。L3 不发布可晋级性能值，资格全 False。
身份成立而数值/输出 dtype 错误是候选 failed；输入保真缺口仍是 unknown。
XPU 明确拒绝可选 reference_contract，不静默丢字段；两 runner 的 action/key 守卫同步保留。

## 5. 不删标定反例

原 12 组自动边界 + 单列五种值 × 两种布局 + 五种低方差/幅值模式 × 两种布局，
共 **32 组**。单列 strided 用多行真实 row gap，不把 contiguous 叫 strided。
扩展模式：非二进制常量、近常量、大值变化、小值变化、大值正负交替。
离线按 dtype/shape/stride/value 多重集核对与标定样本完全相同。

除 policy hash/lint 外，还要求完整 suite hash 与注册表一致，删一个难样本也不行。
L2 套件不替换原 L3 workload、性能 case 或权重。

## 6. 验证入口

```sh
python -m pytest tests -q
python tests/golden_regression.py compare
python benchmarks/reference_profile_h20.py --gpu 2
```

H20 驱动只 stub 候选生产，实际跑 Conductor/L1/L2/L3/确认/报告/资格门。
正例、fusion 差异、dense stride、tail、fp16 intermediate、错误输出 dtype 都真实执行。
漂移试验只改**自有私有目录**中的 primitive：先确认 owned peer quiescent，再追加注释；
要求在生成前 unknown，恢复后重验正例。无安装、共享源码修改或广域 kill。
清理先于易失败的报告写入，并核对 quiescent/removed。

离线 fabricated CUDA 回执仅用于协议/路由反例，不计为 GPU 数值证据。
小张量 L3 可真实执行但无测量资格，不能为了入库而放宽门。
失败验收也保留，不能把“六个数值试验完成”记成全部验收完成。

## 7. 验收结果

- 全量：**3067 passed / 1 skipped / 2 warnings**。skip 为本地 CUDA 不可用；
  warnings 是已有非有限 oracle 诊断，不改 unknown 判决。
- 新增参考身份守卫 **74 条通过**；私有副本 clean 通过，**8/8 mutants 被断言杀死**，
  覆盖登记外定义、执行漂移、删 suite、改绑定 hash、忽略 L2 回执、替换 L3 基线、
  复用过期文件摘要、unknown 进入 Fixer。未靠收集错误计数。
- 串行 Golden：**PASS 4 / REVIEW 0 / FAIL 0 / SKIP 0**，未改 baseline 或容忍度。
- H20 验收 **8/8**：两次正例全 32 组与重复摘要通过；dense/tail/fp16 intermediate
  真实数值拒绝，fp32 输出在对齐主输入即 failed；默认 fusion 两组仍报
  max_abs=5.829334e-5、各 88 个坏点。
- 私有 primitive 漂移触发 primitive_sha256 mismatch，**零生成、零编译、零 verify、零 L3**；
  恢复后正例再次通过。共 **370 次 verify、7 次 compile、4 次真实配对 L3**；
  小张量测量资格仍全部 False，**所有试验均未正向持久化**。
- 最终 owned runner quiescent、私有目录 removed、无清理信号、无 forbidden action。

成功运行编号：`reference-profile-h20-20261008T123337011603Z-29614-1`。
参考身份 hash：`8f779954d3d2b3af67f0cdd12aa677a4bbeceaf2304c67159a1154e9b2d9d18d`。
此前运行 `reference-profile-h20-20261008T121629942800Z-22282-1` 的六个数值试验通过，
但漂移步骤的辅助脚本漏输出 JSON，整体明确 completed=False，且仍完成清理；
修复后两次完整验收通过。最终强化的成功响应完整性守卫另核对了 348 份实际 CUDA 成功读数。
