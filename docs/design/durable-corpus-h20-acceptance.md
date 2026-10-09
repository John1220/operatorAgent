# 持久化入库协议的 H20 端到端验收

日期：2026-10-09。接续 [本地语料持久化提交与幂等恢复](durable-corpus-admission-and-recovery.md)。

## 1. 范围与身份

本轮验收生产代码提交 **`c92ed01`**：在新 H20 观测上，经正常资格门完成实际语料提交，
然后只恢复入库副作用，证明不会重复追加。没有修改生产源码、性能阈值、精度、参考、
量具 policy、协议输入上限、默认 H20 地板或既有 Golden registry。

固定的 LayerNorm FP16 Triton 候选与上一轮相同，冻结 v2 参考语义也不变：

- 候选 SHA：`ea4e9fe9363d79d0b75b3be7fabcbe51f161fa6b1e4d9d0b84968765e5ba55c8`。
- 参考 hash：`8f779954d3d2b3af67f0cdd12aa677a4bbeceaf2304c67159a1154e9b2d9d18d`。
- 当前完整 evaluator hash：`7aba71b57476a144684f906994912dbab89a91594d66b344a2647c40cb102ba3`。
- 预声明验收 plan hash：`9412688ad5fe36e1352bca6407bed6f7473b723cb5ad5cb449a4a0850024f1c6`。
- 实际 workload/profile plan hash：`6e19e63809fa7f5879a305a123a870f45e7af7ace3e7124ba684f3f18daa01ad`。

量具 profile 绑定计时器部署源码、设备与 runtime，不等于完整控制端源码身份。
仅改入库源码时，同一仪器的旧 profile 可能仍能校验；但旧 run 的完整 evaluator 已不同，
不能据此声称新版本已验收。本次**主动重新独立预检、使用新 profile、新验证和新读数**，
没有复用旧性能观测或确认回执。

实验在首次远端执行前冻结：两次固定预检；240/192 MiB 两个 workload；合法 resume；
输入值/阈值漂移两个负例；每个 positive 的两次新进程恢复及错误 corpus 路径负例。
没有按测量结果扩大 workload、重试直到 valid，或挑最快窗口。

最初源码上传因缺少目的地明确授权被审查拦截，未执行校准。用户明确允许上传后，
才将 runner/支持源码送入随机 0700 owned scratch；不传配置/密钥，不安装或 sync 依赖，
不改共享 runtime，不终止其他进程。GPU 2 启动前 util=0%，空闲显存 39334 MiB；
仍有其他常驻进程，**不是独占卡实验**。

生成阶段为固定 Designer/Coder 假件；后端、L1/L2、完整数值 suite、L3、心跳、
独立确认、Oracle、持久化提交与恢复均走生产实现。本轮不声称验收了 LLM 搜索能力。

## 2. 两次新量具预检

预检目录：
`output/cuda_calibration_h20/cuda-calibration-h20-20261009T090016768479Z-82464-1`。

| 项目 | audit 1 | audit 2 |
|---|---:|---:|
| 生产 paired window | `ad5ed545985c` | `42516bef9771` |
| 空核 flush median | 4.576 µs | 4.544 µs |
| 空核 flush P95 | 4.640 µs | 4.608 µs |
| 生产好核 median | 10.400 µs | 10.400 µs |
| 好核 IQR/median | 1.54% | 1.54% |
| 保守慢/好分离度 | 6.7100× | 6.3879× |
| 同 batch CUDA/host 比，3 次 | 0.99121 / 0.99387 / 0.99433 | 0.99279 / 0.99419 / 0.99442 |

按既有固定 policy 派生，非手填：

- profile：`74de0d9fcf8a0c0bb2c89a23e21c10a83131bb927381b8fcaeaf94eae717a7c5`。
- calibration：`1a931e52f658533f97508a34185f7359136207a1355cba884d65ea6edfc32003`。
- 仪器 identity：`2b524238408372ca5b2da5a6b2f62f552af32abd2d7ab168818278d95a462838`。
- floor：**4.639999940991402 µs**，expected_good：**0.010400 ms**。

仪器 identity 与上一轮相同符合实际：计时器部署源码、同一 GPU/runtime 没变；新 profile
来自本次两窗口证据，而不是新填一个更低数字。默认 hardware_caps 地板仍为 **20.24 µs**，
本次仅显式使用该通道 profile。预检 shutdown ACK、owned PID absent、scratch removed
均已确认；signals=[]、forbidden_actions=[]。

## 3. 当前 evaluator 的真实观测与入库

运行目录：
`output/durable_profile_h20/durable-profile-h20-20261009T090252738658Z-82808-1`。

主输入形状、FP16 原值模式 `[-1,0,1,2]`、候选与参考保持预声明内容。
240/192 MiB 都在既有 256 MiB storage+scalar 上限内；未放宽协议。
性能比较为同机同卡 PyTorch Eager(CUDA) baseline，门槛仍 1.05×。

| 过程 | 主测 median | 主测 / 确认比值 | 主测地板占比 | 主测 / 确认有效性 | kernel ID |
|---|---:|---:|---:|---|---|
| 8192×15360，240 MiB | 149.104 µs | 2.0182× / 2.0179× | 3.02% | valid / valid | `e125ec21` |
| 8192×12288，192 MiB | 117.024 µs | 1.9560× / 1.9553× | 3.81% | valid / valid | `0e12cd2a` |
| 合法 resume，240 MiB | 149.184 µs | 2.0169× / 2.0182× | 3.02% | valid / valid | `e4982407` |

以上 median 按实际 L3 产物口径报告。本轮没有优化候选 kernel；这些数字证明当前提交
链路能保存合格真机观测，**不是入库改动使 kernel 提速**，也不证明所有 shape 都加速。

每个 positive 实际重新执行 1 compile、65 verify、2 benchmark、2 native heartbeat。
主输入与完整 32-case suite/重复确定性通过；确认是新的 L3 性能窗口，不声称再次独立
跑了一遍完整 suite。合法 resume 的 Designer/Coder 调用为零，没有恢复旧验证资格。

六次 native 心跳的好核/独立锚点比均为 1.00308，慢/好分离度为 6.6535..7.0062。
两次预检、六次心跳和六次主测/确认的 **14 个计时 window 全部不同**。
每次实际 timing receipt 的通道、源码/设备/runtime/default stream、样本与 paired identity
均重新核对；三个 manifest 的完整源码身份匹配当前 evaluator，controls 冻结同一 profile。

保留了长尾诊断：240 MiB 主测 max/median 约 2.88；192 MiB 确认约 4.54。
六窗口 IQR/median 为 0.22%..0.77%，按既有有效性规则仍为 valid；未删样本或改门槛。
这里的预检/锚点是粗粒度量具失效检查，不是独占环境、锁频或精确时钟认证。

两个身份漂移负例均为 HOLD：输入值改变、max_error 减半。均无该 trial 的 RPC、
Designer/Coder、语料副作用；不是借用旧数据通过。

## 4. 实际提交权威与重复恢复

实际隔离语料是运行目录下 `corpus/kernel_db.jsonl`：**3 条 kernel event、3 个合法 seal**，
不是看 admission flag 或文件变大就算成功。独立审计逐项读取：

1. 实际 `admission.json` 的完整 request hash、固定 corpus 绝对路径、record_hash；
2. actual manifest/contract、首测 benchmark、候选源码与 confirmation；
3. 两次验证 binding 的公共身份与不同 verification ID，重新推导 confirmed-observation key；
4. journal 的 admission ID/key/record_hash、完整 event_hash；
5. 实际 stored record、fresh KernelDatabase replay 与原主测源码/字段/score 精确一致；
6. 原 run 的回执为 committed，result/benchmark 声明等级为 persisted。

| kernel ID | admission ID |
|---|---|
| `e125ec21` | `856a21ddac64506565056267339dc7d753943d0bfa8ed615914a7f8b698cbf97` |
| `0e12cd2a` | `946a9eab6fa9f294388968e1ecc2258e658f761fdd6da9e657c1cc927d041142` |
| `e4982407` | `0d15990abc963e58289144305b40000f310f71a0d53f903ce65df3aba96058c2` |

同候选的三次合法观测不是全局源码重复：两个不同 workload；合法 resume 复用原 frozen
run 身份，但重新验证/测量，有新的主测与确认 verification ID，因此有新 admission。

确认远端 runner 停止、scratch 删除之后，对每个 positive 调用两次新的 Python 进程：

```sh
python -m operator_agent.corpus_admission \
  --artifact /absolute/path/to/original/artifact \
  --corpus /absolute/path/to/original/kernel_db.jsonl
```

**6/6 返回 already_committed**，admission ID/record_hash 与原 seal 一致，整个 corpus
文件字节完全不变，没有第 4 条事件。更换 corpus 路径的 **3/3 recovery_hold**，
错误路径不创建文件，原 corpus 不变。

CLI 只核对并完成入库副作用，不创建 backend/LLM，不跑 GPU，不重放预算/bandit。
恢复将登记投影更新为 already_committed；原 run 的 evidence/report 仍保留当时 committed
的事实，二者不矛盾。没有把历史结果升级为新任务的 live credential。

恢复后 corpus SHA：`dd964afb528c86c30143b6b00575f2e13ffe52bf1932568f241507c93ef307db`。
fresh replay 无坏行、无重复 admission；failure corpus 零写入。主测 runner 的 shutdown ACK、
owned PID absent/quiescent 与私有 scratch removed 全部确认，无需发送终止信号。

## 5. 回归与证据目录

- 当前专项：**158 passed**。
- 当前全量重跑：**3376 passed，1 skipped，2 warnings**；两条既有 nonfinite-reference warning 保留。
- 当前 Golden：**4/4 PASS**，未刷新 registry。
- 入库故障的 7 个实际进程死亡点与 15/15 mutation 结果仍见上一轮入库协议文档。
  本轮在真实 H20 事件上检验重复恢复，不冒称在 GPU 运行中又执行了这些故障注入。

`output/durable_corpus_h20_validation` 保存 frozen plan、只读启动检查、两次预检原始证据
的摘要、主测日志、独立 audit、六个 positive CLI 回执、三个错误 corpus 回执及回归日志。
实际原始 audit/trace/manifest/confirmation/intent/ledger 保留在上面的两个运行目录。
这些含运行环境/本机路径的证据均不进 Git；本文只记录结论与摘要身份。

完整复现需要新的匹配预检，而不是复制本次 profile 去别的设备/runtime：

```sh
python benchmarks/cuda_calibration_h20.py --gpu 2 --ruler-profile
python benchmarks/durable_profile_h20.py --gpu 2 \
  --workload 8192x15360 --workload 8192x12288 \
  --cuda-ruler-profile /absolute/path/to/new/ruler-profile.json
python -m pytest tests -q
python tests/golden_regression.py compare
```

总体验收：5/5 trial 无错误，3 个 positive durable admission，6 次精确重复恢复与 3 个
路径漂移负例通过，owned cleanup 已确认。性能结论仅限上述预声明主输入、当前 frozen
evaluator 与本次窗口；不提供分布式 exactly-once、物理断电实验或全产物目录事务保证。
`persisted` 是长期记忆的提交成功声明，不自动等价于全工作负载的 `use_custom` 部署决策。
