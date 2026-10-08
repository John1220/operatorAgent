# fp16 LayerNorm：参考与精度语义标定

## 1. 决策

2026-10-08 已完成本地 CPU 与隔离 H20 标定。

**`builtin-l2-v1` 的 fp16 LayerNorm 继续为 `unknown`，不启用默认套件。**
这不是没有跑完：同一 Torch 版本内的 CPU/CUDA 参考存在超过当前门限的差异；
现有契约也没有冻结这类参考实现身份与中间舍入/融合语义。

没有放宽容差、删除单列或常量边界、替换生产参考、更新 Golden baseline，
没有把参考问题自动当成候选问题修复，也没有新增生产 runner action。

新增实现：

- `benchmarks/calibrate_layernorm.py`：CPU / 私有 H20 标定驱动、覆盖与证据检查。
- `benchmarks/layernorm_numerics.py`：共享诊断量具、参考身份与输出量化分离。
- `tests/test_layernorm_calibration.py`：标定自身的反例与失败关闭守卫。
- `operator_agent/correctness_profiles.py`：未知原因更新为已发现的阻碍。

## 2. 标定的精度口径

固定语义为二维、末轴归一化、无 affine、`eps=1e-5`、总体方差（除以列数）。

| 项目 | 口径 |
|---|---|
| 输入 | **先量化到 fp16**，所有臂读取这份输入，不与量化前随机数比较 |
| 候选 | 显式 fp32 归约与计算，写回 fp16；另测关闭浮点融合的编译对照 |
| 生产参考 | 原样调用本地 `Verifier._build_reference_fn` / 远端 `_reference_torch` 的闭包 |
| 提升精度臂 | `F.layer_norm(X.float()).half()`，**不同执行路径，只诊断、不替换参考** |
| 数学诊断 | NumPy centered fp64，在已经量化的输入上计算 |
| 舍入后诊断 | 数学诊断输出再转 fp16，与未舍入 fp64 真值分开报告 |
| 第二份 fp64 读数 | Torch fp64 LayerNorm；与数学诊断相互检查，不参与生产 L2 改判 |
| L2 量具 | 原规则 fp32 比较，`atol=1e-5, rtol=1e-2, rho=1.0`，绝对/相对 **AND** |

两份 fp64 诊断的 `1e-10` 一致性界限仅确认这份小规模标定的量具成立，
不是新 L2 容差。非有限结果、缺失参考、设备/输入回读缺失、重复输出不一致、
诊断臂缺失或门限漂移，都不能报告标定完成。

**`accum="fp32"` 的契约声明不是 native Torch 内部实现证明。**
native 参考内部累加路径明确记录为 `implementation_defined_not_attested`。
提升输入精度不能倒推原 fp16 路径用了什么归约算法。

参考身份记录算子语义、Torch/NumPy/CUDA runtime 版本、执行设备与型号、
CPU 指令能力/线程数、确定性开关、参考构造器名与源码 SHA256，再做内容哈希。
实际导入的诊断量具、容差、输入保真/回读四个共享模块也逐一与控制端 SHA256 匹配；
私有部署优先导入已校验的同级模块，不能被远端已安装的旧包遮蔽。
这是诊断出处，不是已接入 FrozenContract 的执行身份约束，也不是 Torch 原生二进制证明。

## 3. 样本与量具

完整保留拟议边界集的 **12 cases / 21 rules**：

- rows `1/7/8/9`，cols `1/31/32/33`，seed 17；
- random / zeros / constant / large / small；
- 连续与真实 strided 输入。

该集合只是一份诊断提案：从 fp32 LayerNorm 的可信 recipe 获取形状和模式，
明确换成 fp16，**没有把它注册/冒充成受支持的 v1 profile**。

提案 policy hash：`e1cc4d4909a71461eb7f1ce13b768f622ed69a6c459a12b66f9882b093df946c`。
suite hash：`0453674dc3138df396e7455a784e27a0f885a398fba55834d6073f0fbb706ecd`。

另加 20 个连续/strided 成对样本，总计 **32 组**：

- 单列 `0 / 1 / 100 / 1e-4 / 0.1`；
- 33 列非二进制常量、近常量、大值低方差、小幅变化、正负大值。

每组张量不超过 1024 字节。设备张量按 storage/offset/stride/dtype/值摘要回读，
不会先转连续再声称验证了 stride；参考保留原生产行为，另有保留 view 的 eager 对照。
H20 还在**同一安装的 Torch 2.14** 中运行 CPU eager 臂，排除本地与远端版本不同的干扰。

## 4. 实测结果

### 本地 Torch 2.8 CPU

32 组参考诊断完整，centered fp32 与 native fp16 参考有 **3 组**触发门限：

| 样本 | 数学输出 | CPU native 参考最大残差 |
|---|---:|---:|
| 原始 `(8,1)` random | 0 | `2.384185791015625e-5` |
| 单列常量 100，连续 / strided | 0 | 各 `9.765625e-4` |

提升为 fp32 的 eager 臂与 centered 数学臂在这些样本上输出零。
只更改生产参考为提升精度臂就会改变判决，因此没有做这种“修复”。

普通随机样本 `(8,32)` 的 native 输出距**未舍入** fp64 真值约 `9.50703e-4`，
但距**舍入后** fp16 真值为零。前者是该样本的 fp16 输出量化误差，
不能误写成归约精度不足，也不能据此放宽零点的绝对门限。

### H20：Torch 2.14.0+cu130，CUDA 13.0

在自动选择的空闲物理 GPU 2 上固定到 `cuda:0`。
32 组参考诊断完整；CUDA native 与 centered fp32 对照全部通过当前 AND 门。

**同一 Torch 2.14 的 CPU 臂与 CUDA native 有 6 组不一致**：

| 样本 | CPU 与 CUDA 最大差 | 坏元素数 |
|---|---:|---:|
| 原始单列 random | `2.384185791015625e-5` | 1/8 |
| 原始 `(8,32)` large 常量 100 | `9.765625e-4` | 256/256 |
| 单列常量 100，连续 / strided | 各 `9.765625e-4` | 各 8/8 |
| 大值低方差，连续 / strided | 各 `1.52587890625e-5` | 各 88/264 |

远端 CPU 指令能力为 AVX512。本地 CPU 没有在 32 列 large 样本上出现同一残差，
因此不能推广成“所有 CPU 都有同一个误差”，也不能仅由 dtype 冻结参考数值行为。

### Triton 编译融合与错误对照

共有 **67 条候选判读**：两种编译配置各 32 组，每组重复一次，外加 3 个错误反例。
判读均具备真实 CUDA 输入回读、原参考/门限、全输出覆盖和候选输出摘要。

| 候选配置 | 当前生产 L2 判决 |
|---|---|
| centered fp32，默认浮点融合 | 30/32 通过；大值低方差连续/strided 两组失败 |
| 同一源码，`enable_fp_fusion=False` | **32/32 通过**，各组重复摘要一致 |
| 错误 dense stride | 拒绝；256 坏元素 |
| 错误尾部写零 | 拒绝；8 坏元素 |
| 将方差归约和显式舍入为 fp16 | 拒绝；176 坏元素 |

默认融合版在参考零点产生 `-5.829334259033203e-5` 残差，两组各 88 个坏点；
仅关闭融合就消失。对照定位到编译融合相关的差异，
**未检查反汇编，不宣称已经证明某条具体 FMA 是原因**。
没有更改历史 H20 acceptance 的候选源码或其期望结果。

“标定完成”表示读数/覆盖/重复/反例/清理证据完整，
**不表示所有候选通过，不产生默认 profile 的正确性资格，也不产生性能或语料准入结论。**

## 5. 复现与安全边界

在仓库根目录，使用已安装 NumPy/Torch 的 Python：

```sh
python benchmarks/calibrate_layernorm.py
python benchmarks/calibrate_layernorm.py --h20 --host <已有SSH别名>
# 可选 --gpu N：该卡不空闲则拒绝，不强占。
```

- 使用已有 runtime，无依赖安装或同步、无 L3/性能 benchmark、无生成 LLM。
- 生产源文件与私有诊断 server/module 部署到随机 0700 目录，逐文件校验 SHA256。
- 诊断 action 仅加在私有 server 中，生产 action 表不变。
- 复用生产 RPC 截止时间、启动前身份发布、owner token、pidfd 身份清理。
- `finally` 先确认本次 owned runner 静默，再删除本次私有目录，最后写证据；
  不能确认静默就保留目录并报告未完成。没有广域 `pkill`，不处理他人任务。
- 这不是敌意候选沙箱或 GPU watchdog 证明。

证据在 ignored `output/layernorm_calibration/`，包含逐组读数、输入回读、环境、
源文件快照与最终清理回执；真实远端用户/runtime/路径不写入版本库。

最终记录：

- CPU：`layernorm-calibration-20261008T113225352699Z-10079-1`。
- H20：`layernorm-calibration-20261008T113241456910Z-10726-1`。
- H20 最终 `quiescent=true, removed=true`，正常 shutdown，无需发送信号。

## 6. 下一步准入条件

不在 v1 中静默换参考。先为新版 profile / FrozenContract 增加**可信参考身份与执行能力匹配**：

1. 明确“数学 LayerNorm 允许误差”还是“与指定 native eager 数值行为一致”；
   生成候选前确定参考定义，失败后 Agent 不可换尺子。
2. 冻结输入/中间/输出精度、eps、归一化轴、方差定义、affine、舍入/融合要求及参考来源；
   不能只靠 `accum=fp32` 表达完整精度语义。
3. 指定参考实现/版本/后端的支持范围，执行前校验，不匹配就 `unknown`；
   不把 H20 CUDA 的单次成功当成其他设备或 Torch 版本的资格。
4. 在该身份下保留单列、常量、低方差、大/小值、真实 stride 与错误反例，
   通过正式 L2 套件后再走 Conductor / L3 / 报告闭环。

现有 v1 标记、其余七个受支持组合、显式套件/历史任务与契约哈希保持原行为。

## 7. 回归与量具反例

- 全量 `python -m pytest tests -q`：**2993 passed / 1 skipped / 2 warnings**。
  skip 为本地 CUDA 不可用；两条 warning 是已有非有限参考 oracle 的诊断样本，仍判 unknown。
- 新增标定守卫 **27 条通过**；不把离线守卫当作 CUDA 证据。
- 串行 Golden compare：**PASS 4 / REVIEW 0 / FAIL 0 / SKIP 0**，未改 baseline 或容忍度。
- 在私有临时副本中变异：干净副本通过，**8/8 mutants 被断言杀死**，没有靠收集错误计数。

| 变异 | 守卫 |
|---|---|
| AND 改 OR | 小残差不应被相对项单独拒绝 |
| 放宽 atol | 单列参考残差必须仍超原门限 |
| 用未舍入真值冒充舍入后真值 | 输出量化与归约误差读数必须分离 |
| 移除实际导入源码校验 | 内容哈希重算后，错误共享源仍被拒绝 |
| 移除输入回读校验 | 同 shape/dtype 但不同输入值不能冒充原样本 |
| 标定完成就启用 profile | 全部 GPU 通过仍不能绕过参考身份冻结 |
| 忽略候选重复输出 | 不同摘要不能计作完整重复证据 |
| 接受未冻结 fp16 默认 profile | unknown 必须在编译/L3 前停止 |
