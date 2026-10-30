# H20 冻结验收（2026-10-30）

状态：**完成。H20 资源于本轮结束后释放，之后不再新增真机证据。** 本轮在 `v0.6.0`
（`45ec960`）上完成了三件 GPU 才能做的事：4096³ fp16 matmul 的完整 L1/L2/L3、
§14 真机冒烟、launch floor 与 L2 冲刷口径复标。S5 正式 P3 24 格**未启动**：
P2 门（网络唯一出口、远端候选隔离）未通过，且 GPU 即将释放，按证据不足归档。
受控合作环境的探索性 24 格已另行派发 13 格后停批（12 通过 / 1 设施 unknown），
见[探索性实测](dsh-operator-s5-cooperative-study-20261029.md)；本轮不改变其分母。

证据目录（Git 忽略，需单独备份）：
`output/plugin_completion/20261030-gpu-freeze/`。

## 1. 环境身份

| 项 | 值 |
|---|---|
| 代码 | `45ec9609420bae2da87043efc42f85ac27ef2cfd`，`v0.6.0`；§2.7 的 seed 修复跑在 `ffe471a` |
| 控制机 | macOS，Python 3.9.6 |
| 远端 | `algorithm-h20`（Linux 5.15，x86_64） |
| GPU | **GPU 7**，`NVIDIA H20`，UUID `GPU-52ca6155-d8bc-a416-847a-6f0c71db1552` |
| 驱动 | `580.105.08` |
| runner 环境 | Python 3.10.12，torch `2.14.0+cu130`，CUDA 13.0，triton `3.8.0` |
| 系统 python（只读探测用） | Python 3.10.12，torch `2.13.0+cu130`，triton `3.7.1` |

⚠️ 两个环境**不是同一个**：测量与编译走 `~/operator_agent` 的 uv 项目环境
（torch 2.14 / triton 3.8）；`python3 -c` 的只读探测走系统环境（torch 2.13 /
triton 3.7）。引用版本时以 runner 环境为准。完整快照见
`environment.txt`，远端 runner 哈希见同文件末尾。

## 2. 结果

### 2.1 4096³ fp16 matmul：完整 L1/L2/L3，`valid`

| 项 | 值 |
|---|---|
| 候选 | `matmul4096/candidate.py`（`matmul_best.py` + `kernel()` 包装，手写核） |
| 来源标记 | `candidate_origin=outer`（**不是**模型生成，不构成插件增益） |
| L1 / L2 / L3 | 通过 / 通过 / 通过 |
| L2 套件 | `builtin-l2-v1`，15/15 case，最大误差 `0.00e+00` |
| 加速比 | **1.0569×**（`0.9876/1.044 ms`） vs cuBLAS(FP16)，同机同卡 |
| 吞吐 | **139,157.831 GFLOPS ≈ 139.16 TFLOPS**，H20 fp16 峰值 148 TFLOPS 的 **94.0%** |
| Roofline | `compute` 界，ridge point 37.0；峰值出处 `flash-attention dense 表 148/4.0` |
| launch 占比 | **2.01%**（floor 20.24 µs / 候选 median 0.9876 ms） |
| 测量档位 | `valid`；`measurement_can_gate=true` |
| 资格五位 | 全 `True` |
| 设备租约 | `exclusive=未观测`，`external_load=0.00`，无 harness 邻居，`gpu_other_procs=0` |
| 固化设备 | `CUDA_VISIBLE_DEVICES=7`，候选与基线均 `cuda:0 NVIDIA H20 (visible=7)` |

这是项目**第一个可裁决的 L3 性能读数**（S4 的四项 L3 全部 `uninformative`）。
边界：契约没有声明目标函数，报告按**主 case** 加速比 >= 1.0 判定，因此这是
**单形状**结论，不是算子全域性能证明；也没有做独立重测与入库。

### 2.1b 8192³ fp16 matmul（扩展点）

同一候选、同一 GPU 7，只把 shape 放大到 8192³，走 `_LARGE_TILE`（128×128）分支：

| 项 | 值 |
|---|---|
| L1 / L2 / L3 | 通过 / 通过 / 通过（15/15 case，最大误差 0） |
| 加速比 | **1.0239×**（`7.822/8.010 ms`） vs cuBLAS(FP16) |
| 吞吐 | **140,559 GFLOPS ≈ 140.56 TFLOPS**，峰值 **95.0%** |
| launch 占比 | **0.26%** |
| 档位 / 资格 | `valid` / `can_gate=true`，五位全 True |

这证明性能链路在更大 shape 上仍给出 `valid`，但仍是 **compute-bound matmul**，
不能替代 memory-bound 算子的证据。

### 2.2 §14 真机冒烟

`benchmarks/smoke_spine_h20.py --skip-contention`，固定 GPU 7，**0 失败**：
46 项 ✅、8 项 ➖、9 项 ⚠️（提示性）。覆盖：

- §14-1 已知好核：真基线下的 `valid` + 跨次稳定；
- §14-2 故意写慢：`slower`，没有记成通过；
- §14-3 库回退假候选：静态审计 + 运行时盾两层拦下；
- §14-7 跨卡观测被拒（检测器）；运行时检查在生产上没有喂给者；
- §14-8 坏心跳：该窗口资格全 False、拒入库；对照臂真心跳可用。

边界：§14-4 因共享机器用 `--skip-contention` 未跑；§14-5 kill 续跑需要一次
真实 LLM 跑，未覆盖；§14-6 纯离线。

### 2.3 标定复标

| 项 | 本轮 | 既有口径 | 结论 |
|---|---|---|---|
| 空核 launch floor | median **20.42 µs**，min 18.94 µs | 20.24 µs | 一致，未改常量 |
| 空核（不冲刷 / 冲刷） | **20.00 / 4.64 µs** | 20.03 / 4.64 µs | 一致 |
| layernorm 1024×768 | 21.98 / 7.33 µs | 32.5→7.2 那一组 | 量级一致 |
| matmul 4096³ | 1002.62 / 989.68 µs | — | 冲刷影响 ~1.3% |

冲刷脚本报告的 torch/triton 是 `2.14.0/3.8.0`，与 runner 环境一致。launch floor
的 20.42 µs 与代码里的 20.24 µs 差约 0.9%，**没有**改常量；这次只作为复标证据。

离散度复标（30 轮 × 30 样本）：合并 `iqr/median` p95=**0.0261**，单核最大 p95=0.0324；
`DISPERSION_DEGRADED=0.10` 与 `DISPERSION_UNINFORMATIVE=0.30` 都高于观测到的最大值
0.0374，本轮未触发，**未改常量**。`INTRUDER_MAX_OVER_MEDIAN=2.0` 低于小 shape 的
`max/median` p95=10.34，但该阈值只作诊断、不改 validity；记为后续调整项，不在本轮
改代码。

### 2.4 64³ L1 冒烟

`matmul4096/sanity64`：L1 通过。用于确认部署、固定 GPU 7、编译链路在正式跑之前
是通的；不构成 L2/L3 证据。

### 2.5 取消与清理（当前 HEAD）

`cancellation/`：启动一次 2048³ verify（GPU 7），在 L2 阶段通过 CLI
`cancel --invocation-id` 取消。结果：

- CLI 退出码 **130**，回包 `phase=cancelled`、`stopped_reason=user_cancelled`、
  `error=task_cancelled`；
- `cleanup.worker_reaped=true`；`remote_execution` / `provider_call` 均为 `unknown`，
  `all_descendants_proven=false` —— 只声明本机 worker 已回收，**不**声明远端或服务商已停；
- 未生成 `complete.json`；`verify-1/` 保留部分候选与 `started.json`，用于复算；
- 取消后远端无 `runner_server` / `run_kernel.py` 残留，GPU 7 `0% / 4 MiB`。

这补的是 v0.6.0（S4 之后代码有变）的取消路径证据，不构成正确性或性能结论。

### 2.6 Memory-bound 尝试（未成立）

目标是给 LayerNorm 一个大 shape 的 `valid` L3；三次尝试都被**输入传输**挡住，
不是 GPU 算力问题：

| shape（fp32） | 字节 | 结果 |
|---|---:|---|
| 16384×6144 | 402,653,184 | L2 拒绝：`logical tensor 超过输入字节上限`（单 tensor 上限 256 MiB） |
| 16384×4096 | 268,435,456 | 恰好等于 256 MiB；L2 拒绝：`JSON 输入超过字节上限`（张量 + 标量超过同一预算） |
| 16384×3800 | 249,036,800 | 通过上限，但 seed 未被采用，回落成 249 MB base64 JSON；5.5 分钟后取消（exit 130） |

第三条的根因已核对：`Verifier._seed_spec_for` 用 `cases_from_shapes` 的
`_rows_case`，它构造 `X/weight/bias`；而本契约与 correctness suite 的输入只有
`X` —— 集合不相等 ⇒ `_seed_spec_for` 返回 `None` ⇒ 走整份 JSON 传输。
对 249 MB 的输入，这条路不可用。

结论：seed bug 已修（`ffe471a`），第三条不再传 249 MB；但 memory-bound 的 `valid`
仍没拿到，见 §2.7。

### 2.7 Seed 修复后的重跑：能跑通，但只到 `degraded`

修 `ffe471a` 后重跑 16384×3800（同一候选、同一 GPU 7）：

| 项 | 值 |
|---|---|
| L1 / L2 / L3 | 通过 / 通过 / 通过（12/12 case，最大误差 `1.43e-06`） |
| 加速比 | **1.4954×**（`0.1608/0.2405 ms`） vs PyTorch Eager(FP32) |
| 吞吐 | **2,322.6 GFLOPS** |
| launch 占比 | **11.2%**（> 10%） |
| 档位 | **`degraded`**；`can_gate=true`、`can_tune=true`、`can_verdict=true`，但 `can_persist=false`、`can_count_progress=false` |
| 原因 | `launch 地板占实测时间 11.2% (> 10%)：判定可用，但不得据此禁用配方或累计停滞` |

为什么不是 `valid`：256 MiB/tensor 上限把最大 shape 卡在约 249 MB，候选只有
160.8 µs；要把 launch 占比压到 10% 以下需要 >202 µs。更大的 16384×6144（402 MB）
会超上限；把核故意写慢去跨阈值不是证据，所以**没有**这么做。

结论：**memory-bound 现在端到端可跑，但本轮只有 `degraded`，没有 `valid` L3**。
若要 `valid`，需要放开 seed 路径的单 tensor 上限（例如允许 512 MiB），或换一个
天然更慢/更大的 memory-bound 形状。

补充：seed 修好后也重试了 16384×6144（402 MB），L2 仍在传输前被 256 MiB/tensor
上限拒绝（`logical tensor 超过输入字节上限`）。所以这个上限不是 JSON-only 的，
16384×3800 是当前 cap 内能跑的最大档；要 `valid` 必须改 cap，而不是再调 seed。

## 3. 未执行与永久边界

- **S5 正式 24 格（P3）未启动**。P2 仍未证明模型网络唯一出口与远端候选执行隔离；
  GPU 释放后无法完成冻结与执行。`formal_s5_credit=false`、`inference_allowed=false`，
  不新增样本、不重派旧 unknown 格。受控合作环境的探索性 24 格是另一条记录：已派发
  13/24 格后因 DSH 流空闲超时停批，12 格通过、1 格设施 unknown，正式分母仍未完成。
- **跨卡验证未做**：本轮只有 H20；没有 A100/H100 读数，因此不能声称“跨 GPU 通用”。
- **离散度已复标**（§2.3）：`DISPERSION_*` 未触发、未改；`INTRUDER_MAX_OVER_MEDIAN=2.0`
  对小于 1 ms 的核偏紧，但它只作诊断，不影响 validity —— 记为后续调整项。
- **v2 LayerNorm 参考未跑**：环境 torch 2.14/driver 580 与 `PINNED_RUNTIME` 表面一致，
  但没有执行 attestation；不能声称 v2 可用。
- **Memory-bound valid L3 未取得**：seed 修复后 16384×3800 能完成（§2.7），但 launch
  11.2% → `degraded`；16384×6144 被 256 MiB/tensor 上限在传输前拒绝。没有靠放慢核
  去凑 `valid`。
- **多卡/并发/UI**：未做。

H20 释放后**永久不可做**：新增真机证据、memory-bound 算子的 valid L3、常量再标定、
S5 正式批次、跨卡对照。

## 4. 清理

- 远端 `runner_server` / `run_kernel.py` 残留：**无**。
- GPU 7：`0%` 利用率、`4 MiB` 占用（清理后快照）。
- 上传到 `~/` 的标定脚本已删除；`~/operator_agent` 是运行器工作目录，保留。
- 不声称整机全局静默；只声明本次检查未见自有残留。

## 5. 证据

| 文件 | SHA256 |
|---|---|
| `matmul4096/verify.json` | （见 `SHA256SUMS`） |
| `matmul4096/verify-1/result.json` | `54051a051b314e159e093310e3579d541b5dbd6e2b29d786085de561028d4258` |
| `matmul4096/verify-1/complete.json` | `a4fb705fc53275673c999a2d255ce8e572cd3b0ce6f5c5a659a3f51a7e888f0b` |
| `matmul4096/verify-1/report.md` | `1f2ebd3b319a192b28de420ddce795ec4d45596951d6f82b5ebd617ee520d1bc` |
| `smoke.log` | `effcb1641b28d25d3f03096b6f489f1a19d1042547291097dea7299947b996fb` |
| `launch-floor.log` | `2dc4aeee85972d7913fc618aa4c3bfba17a48e7f0700b18c5040d0822ab3f385` |
| `l2-flush-effect.log` | `664939cea6100335ec83f01e28cf58430160c2c924ae3ab286015170de0cc6a4` |
| `dispersion.log` | `49998bae441ca35c7bf8fadd347d6be7b0438d29fc016f077470b772a53e8b2e` |
| `matmul8192/verify-1/result.json` | `4f1b4632f363f9c21269f1d2ab1b2076aef14b8b9166e9faf3cfbb8ffa8fbab6` |
| `matmul8192/verify-1/report.md` | `6423bb2b652a084d0bcb2018b8d7c03f6ebdaf58ac6f08dcd7545ac89fca1859` |
| `matmul8192/summary.json` | `e0f0fb5a6d42b68e844234124f479dd65d33d9e70ea4985e457a81405b8461dc` |
| `layernorm16384x6144/verify-1/result.json` | `15097de1720f45406fe86b7ddfaa0e6b89ad9816d4f57932f4b02f6340c67be5` |
| `layernorm16384x4096/verify-1/result.json` | `bd72ed2335d32c732715787a86ec07359a5d6902bc33255cde826b6804657fe1` |
| `layernorm16384x3800/status.json` | `9aae93a0133d42b316d566147381995df46234648b2c0ac557bf8981102b4826` |
| `layernorm16384x3800-v2/verify-1/result.json` | `cb559dbdb828b3a021f6738ebaca9ae813f74d216c9709095a47e5ccd24e1300` |
| `layernorm16384x3800-v2/verify-1/report.md` | `0c0e0f052503204ef40d4a8cd131c74249792b40684047acd3e3a5004864ca9b` |
| `layernorm16384x3800-v2/summary.json` | `73df202ac7786c9f7adf3a309de4f3c81f2f22301d66353f0aaf1045b5088a62` |
| `layernorm16384x6144-v2/verify-1/result.json` | `ce579f30ab6adf57303b68d44160c7c2b2f960c1cb5704008ab26e2c8a481752` |
| `summary.json` | `60c09b6e2c826e5841b2f959064ad1b02ee0e094252598b5d7134d8c81ded098` |
| `cancellation/cancel.json` | `8bb39852d9791bfd6176e1724ba7a79b95e9bf43dca738fa66f1f7e82eab7396` |
| `cancellation/status.json` | `9c5471ceb2130bd97cc764a6b3ba13354be64b68c6d908437783c2854b44f70c` |
| `environment.txt` | `5a37dfb43fbb80aef62cb162d0300b6a787dfe0768240cae3056a564fbefd4ce` |

打包：`output/plugin_completion/20261030-gpu-freeze-evidence.tar.gz`
（SHA256 `c0745d0b6e7e7434e6c960f682e2d76831bf8c2e85cf4ccfa6ccd598a01f98cd`）。

完整清单：`output/plugin_completion/20261030-gpu-freeze/SHA256SUMS`。复核：

```bash
cd /path/to/operatorAgent
shasum -a 256 -c output/plugin_completion/20261030-gpu-freeze/SHA256SUMS
```

重算 4096 关键事实：

```bash
python3 - <<'PY'
import json
r=json.load(open("output/plugin_completion/20261030-gpu-freeze/matmul4096/verify-1/result.json"))
l3=[v for v in r["verdicts"] if v["level"]=="L3"][0]["metrics"]
print(l3["measurement_validity"], l3["measurement_can_gate"],
      l3["speedup_vs_baseline"], l3["gflops"], l3["launch_dominance"])
PY
```

## 6. 与 S5 的关系

本轮**不**补齐 [S5 最小完成计划](../design/s5-minimum-completion-plan-20261029.md)
的 24 格，也不改变其 P1/P2 结论。受控合作环境的探索性 24 格已另行派发 13 格后停批，
见[实测记录](dsh-operator-s5-cooperative-study-20261029.md)；本轮不补跑、不改写其分母。
10-29 的插件入口验收与失败修复验收仍是独立的功能证据，不计正式 S5 分母；本轮新增的
4096 读数属于**判定层性能链路**证据，同样不计 S5。S5 按“设施与资源不足，正式对照
未完成”归档。
