# Operator Agent 用户使用文档

## 目录

1. [项目简介与核心功能概述](#1-项目简介与核心功能概述)
2. [安装部署与环境配置](#2-安装部署与环境配置)
3. [基础功能使用指南与操作示例](#3-基础功能使用指南与操作示例)
4. [高级配置参数说明](#4-高级配置参数说明)
5. [常见问题与故障排查指引](#5-常见问题与故障排查指引)
6. [验证链路上的门与判决档位](#6-验证链路上的门与判决档位)

---

## 1. 项目简介与核心功能概述

### 1.1 项目简介

**Operator Agent** 是一个基于大语言模型（LLM）驱动的 GPU 算子自动开发框架，支持 Triton DSL 生成高性能矩阵乘法、Softmax、Attention 等 GPU 算子。

**版本**: v0.5.0

### 1.2 核心功能

| 功能模块 | 说明 |
|---------|------|
| **LLM 驱动生成** | Designer Agent 生成设计草图，Coder Agent 生成代码，Search Agent 优化性能 |
| **三级验证体系** | L1 编译验证、L2 功能验证、L3 性能验证 |
| **远程执行支持** | 支持 SSH/HTTP/Local 三种模式远程执行代码 |
| **自动日志管理** | 所有执行过程自动记录到日志文件，自动清理旧日志（保留最新10个） |
| **产物持久化** | 生成的算子代码、Sketch、验证结果自动保存到 `output/` 目录 |
| **RAG 知识库** | 算子知识库支持历史案例检索和复用 |

### 1.3 架构设计

```
用户请求
    ↓
IntentParser (意图解析)
    ↓
Conductor (调度编排)
    ├── Router (路由决策)
    ├── DesignerAgent (设计草图)
    ├── CoderAgent (代码生成)
    ├── Verifier (三级验证)
    │   ├── L1: 编译验证
    │   ├── L2: 功能验证
    │   └── L3: 性能验证
    └── SearchAgent (性能优化)
    ↓
ArtifactStore (产物存储)
    ↓
output/ 目录
```

### 1.4 支持的硬件与 DSL

| 硬件 | 支持的 DSL |
|------|-----------|
| nvidia_gpu | triton, cuda_c, python |
| mock_gpu | python, triton |
| ascend | python |

---

## 2. 安装部署与环境配置

### 2.1 环境要求

- Python >= 3.9
- NVIDIA GPU（远程验证时）或 MockGPU（本地测试）
- SSH 访问权限（远程执行时）

### 2.2 安装步骤

```bash
# 克隆项目
git clone <repository-url>
cd operatorAgent

# 安装依赖
pip install -r requirements.txt

# 安装 Triton（如果需要本地编译）
pip install triton torch numpy
```

### 2.3 目录结构

```
operatorAgent/
├── operator_agent/           # 核心模块
│   ├── __init__.py
│   ├── schemas.py           # 数据结构定义
│   ├── kernel_db.py         # 算子知识库
│   ├── mock_gpu.py          # Mock GPU 后端
│   ├── remote_executor.py   # 远程执行器
│   ├── llm_client.py        # LLM 客户端
│   ├── prompt_manager.py    # Prompt 模板管理
│   ├── agents.py            # Designer/Coder/Search Agent
│   ├── verifier.py          # 验证层
│   ├── conductor.py         # 调度层
│   ├── intent_parser.py     # 意图解析
│   ├── artifact_store.py    # 产物存储
│   ├── agent_logger.py      # 日志管理
│   └── prompts/             # Prompt 模板目录
├── remote_scripts/          # 远程执行脚本
│   └── run_kernel.py
├── output/                  # 生成的算子产物
├── logs/                    # 执行日志
├── config.yaml              # 配置文件
├── test_nvidia_matmul.py    # 测试脚本
└── tests/                   # 单元测试
```

---

## 3. 基础功能使用指南与操作示例

### 3.1 快速开始

#### 方式一：运行预定义测试脚本

```bash
cd /Users/zh/repos/operatorAgent
python test_nvidia_matmul.py
```

#### 方式二：手动构建任务

```python
import numpy as np
from operator_agent import (
    LLMClient, Conductor, RemoteExecutor, MockGPU,
    TaskSchema, SourceSpec, TargetSpec, ConstraintSpec,
    TaskType, TaskComplexity, Precision,
)

# 1. 初始化 LLM 客户端
llm = LLMClient(
    base_url="https://your-llm-endpoint.example.com/v1",
    model="your-model-name",
    api_key="your-api-key",
    temperature=0.3,
    max_tokens=4096,
)

# 2. 构造任务
task = TaskSchema(
    task_type=TaskType.OPERATOR_GENERATION,
    source=SourceSpec(
        framework="pytorch",
        kernel_type="matmul",
        fused_ops=[],
        complexity=TaskComplexity.MEDIUM,
    ),
    target=TargetSpec(
        hardware="nvidia_gpu",
        dsl="triton",
        compute_units=108,
        warp_size=32,
    ),
    constraints=ConstraintSpec(
        precision=Precision.FP32,
        min_speedup=1.0,
        max_error=1e-3,
    ),
)

# 3. 初始化后端（远程 SSH 或本地 MockGPU）
# host 传 ~/.ssh/config 里的 Host 别名即可; port/user/key_path 不传,
# 由该 Host 块的 Port / User / IdentityFile 解析, 仓库与文档中不出现真实值
backend = RemoteExecutor(mode="ssh", host="my-gpu-box")
# 或使用 MockGPU: backend = MockGPU(name="MockA100", compute_units=108)

# 4. 生成测试输入
M, K, N = 256, 256, 128
A = np.random.randn(M, K).astype(np.float32) * 0.1
B = np.random.randn(K, N).astype(np.float32) * 0.1
test_inputs = {"A": A, "B": B}
expected_output = (A @ B).astype(np.float32)

# 5. 执行任务
conductor = Conductor(backend=backend, llm=llm, max_optimize_rounds=3)
artifact = conductor.execute(
    task=task,
    test_inputs=test_inputs,
    expected_output=expected_output,
    target_gflops=1.0,
)

# 6. 查看结果
print(f"验证结果: {'通过' if artifact.all_passed else '失败'}")
print(f"产物目录: {artifact.kernel_id}")
```

### 3.2 使用 IntentParser（自然语言输入）

```python
from operator_agent import IntentParser, LLMClient

llm = LLMClient(...)
parser = IntentParser(llm=llm)

# 解析自然语言请求
user_request = "我需要为 Nvidia GPU 用 Triton 实现一个矩阵乘法算子，要求fp32精度"
task = parser.parse(user_request)

print(f"解析结果: kernel_type={task.source.kernel_type}, dsl={task.target.dsl}")
```

### 3.3 使用日志管理

```python
from operator_agent import capture_execution, get_latest_logs, read_log

# 方式一：使用上下文管理器
with capture_execution(task_id="my_task") as log_path:
    print("这行会同时输出到控制台和日志文件")

# 方式二：获取最新日志
logs = get_latest_logs(n=5)
for log_path in logs:
    print(log_path)

# 方式三：读取日志内容
content = read_log(log_path)
print(content[:1000])
```

### 3.4 使用产物存储

```python
from operator_agent import ArtifactStore

store = ArtifactStore()

# 列出所有产物
artifacts = store.list_all(kernel_type="matmul", hardware="nvidia_gpu")
for art in artifacts:
    print(f"{art['kernel_id']} - {art['all_passed']}")

# 加载指定产物
artifact = store.load("/path/to/artifact_dir")
print(artifact.code)
```

---

## 4. 高级配置参数说明

### 4.1 配置文件结构

配置文件路径：`config.yaml`

```yaml
# LLM 配置
llm:
  base_url: "https://your-llm-endpoint.example.com/v1"  # LLM API 地址
  model: "Qwen/Qwen3.5-9B"                        # 模型名称
  temperature: 0.3                                # 温度参数（0-1，越低越确定性）
  max_tokens: 4096                                # 最大生成 token 数
  timeout: 120                                    # 请求超时时间（秒）
  # api_key 从环境变量 LLM_API_KEY 读取

# 远程执行配置
remote:
  mode: "auto"                                    # 连接模式: ssh/http/local/auto
  work_dir: "~/operator_agent"                    # 远程工作目录
  ssh:
    host: "my-gpu-box"                            # SSH 主机或 ~/.ssh/config 别名
    port: null                                    # 留空则由 ssh config 的 Port 解析
    user: null                                    # 留空则由 ssh config 的 User 解析
    key_path: null                                # 留空则由 IdentityFile 解析
    runner_path: "~/operator_agent/run_kernel.py"
  http:
    url: ""                                       # HTTP 接口地址（不用 http 模式则留空）

# Prompt 模板目录
prompts_dir: "operator_agent/prompts"

# 优化配置
optimization:
  max_rounds: 3                                   # 最大优化轮次
  target_gflops: 10.0                             # 目标 GFLOPS

# 硬件预设
hardware_presets:
  nvidia_gpu:
    compute_units: 128
    warp_size: 32
    vector_width: 128
  mock_gpu:
    compute_units: 64
    warp_size: 32
    vector_width: 64
  ascend:
    compute_units: 64
    warp_size: 16
    vector_width: 128
```

### 4.2 环境变量

| 环境变量 | 说明 | 默认值 |
|---------|------|-------|
| `LLM_API_KEY` | LLM API 密钥 | - |
| `REMOTE_SSH_HOST` | SSH 主机或 ssh config 别名 | algorithm-h20 |
| `REMOTE_SSH_PORT` | SSH 端口 | 不设，由 ssh config 解析 |
| `REMOTE_SSH_USER` | SSH 用户名 | 不设，由 ssh config 解析 |
| `REMOTE_SSH_KEY` | SSH 私钥路径 | 不设，由 IdentityFile 解析 |
| `REMOTE_WORK_DIR` | 远程工作目录 | ~/operator_agent |
| `REMOTE_RUNNER_PATH` | 远程脚本路径 | {work_dir}/run_kernel.py |
| `REMOTE_HTTP_URL` | HTTP 接口地址 | - |

运行期开关（全部走环境变量，无需改代码）：

| 环境变量 | 说明 | 默认值 |
|---------|------|-------|
| `OPERATOR_AGENT_GPU` | 固定 L1/L2/L3 用哪张卡。**共享机器上必须固定** —— 不固定时三级验证各自重选、落在不同卡上，加速比就不可比 | 不设 → 按 (低利用率, 高显存余量) 自动选；利用率读不到时按最坏的 100% 处理 |
| `OPERATOR_AGENT_KERNEL_DB` | 正样本经验库（成功且可信的核）的 JSONL 路径。取 `off`/`none`/`0`/`false` 时整个关闭持久化，只留本次进程内存 | `<repo>/output/kernel_db.jsonl` |
| `OPERATOR_AGENT_FAILURE_DB` | 负样本经验库（失败签名）的 JSONL 路径，关闭语义同上 | `<repo>/output/failure_db.jsonl` |
| `OPERATOR_AGENT_FAILURE_GATE` | 一条失败签名要在**多少个不同任务**上复发，才升级为可复用教训（见 6.5） | `3` |
| `OPERATOR_AGENT_OUTPUT_DIR` | 产物快照的根目录。跑批量实验或单元测试时指到仓库外，否则历史语料会被测试现场淹没（见 6.7 的盘点） | `<repo>/output` |
| `OPERATOR_AGENT_LOG_DIR` | 日志目录。取 `off`/`none`/`0`/`false` 时**完全不落文件**，只透传标准输出 | `<repo>/logs` |
| `OPERATOR_AGENT_LOG_KEEP` | 自动清理保留最近多少个日志。做长周期实验时默认值会把你还要复查的现场删掉 | `10` |

> 这三个「关闭 / 改址」开关（`KERNEL_DB`、`OUTPUT_DIR`、`LOG_DIR`）存在的理由是同一件事：
> Conductor 无条件走落盘路径，所以**跑一次测试就会往仓库里灌一批产物、并顺带把真实实验
> 现场轮掉**。单元测试靠它们保持仓库无副作用。

### 4.3 Conductor 参数

```python
Conductor(
    backend=None,                # 验证后端（MockGPU 或 RemoteExecutor）
    kernel_db=None,              # 算子知识库
    llm=None,                    # LLM 客户端
    prompt_manager=None,         # Prompt 管理器
    max_optimize_rounds=3,       # 最大优化轮次
    failure_db=None,             # 失败案例数据库
)
```

### 4.4 LLMClient 参数

```python
LLMClient(
    base_url="http://...",       # API 地址
    model="model-name",          # 模型名称
    api_key=None,                # API 密钥
    temperature=0.3,             # 温度参数
    max_tokens=4096,             # 最大 token 数
    timeout=120,                 # 超时时间
)
```

### 4.5 日志自动清理配置

日志自动清理在 [agent_logger.py](file:///Users/zh/repos/operatorAgent/operator_agent/agent_logger.py#L188) 中配置：

```python
def cleanup_old_logs(log_dir=None, keep_count=10):
    """清理旧日志，只保留最新的 keep_count 个"""
```

**默认行为**：每次创建新日志时自动清理，只保留最新的 10 个日志文件。
保留个数可用 `OPERATOR_AGENT_LOG_KEEP` 放宽 —— 做长周期实验时默认值会把你还要复查的现场删掉。
只想把输出透传到控制台、完全不落文件时，设 `OPERATOR_AGENT_LOG_DIR=off`（见 4.2）。

---

## 5. 常见问题与故障排查指引

### 5.1 LLM 连接问题

**问题**：LLM 连通测试失败

**解决方案**：
1. 检查网络连接
2. 确认 API 密钥正确
3. 验证 base_url 是否可达
4. 检查防火墙设置

```python
# 测试 LLM 连接
from operator_agent import LLMClient

llm = LLMClient(
    base_url="https://your-llm-endpoint.example.com/v1",
    model="your-model-name",
    api_key="your-api-key",
)
try:
    resp = llm.chat([{"role": "user", "content": "回复OK"}], max_tokens=256)
    print("LLM 连接成功")
except Exception as e:
    print(f"LLM 连接失败: {e}")
```

### 5.2 SSH 远程执行问题

**问题**：SSH 连接失败或远程执行报错

**解决方案**：

1. **检查 SSH 连接**：
   ```bash
   ssh my-gpu-box          # 别名由 ~/.ssh/config 提供 host/port/user/key
   ```

2. **检查远程 uv 环境**：
   ```bash
   ssh my-gpu-box "cd ~/operator_agent && uv run python -c \"import triton; print('Triton OK')\""
   ```

3. **检查远程脚本是否存在**：
   ```bash
   ssh my-gpu-box "ls ~/operator_agent/run_kernel.py"
   ```

### 5.3 编译验证失败（L1 FAIL）

**常见错误**：

| 错误信息 | 原因 | 解决方案 |
|---------|------|---------|
| `invalid syntax` | LLM 生成的代码有语法错误 | 检查代码清理逻辑，更换 LLM 模型 |
| `name 'triton' is not defined` | 缺少 import 语句 | 确保代码包含 `import triton` |
| `expected an indented block` | 函数体缩进错误 | 检查代码缩进 |
| `@jit functions should be defined in a Python file` | Triton 代码必须在文件中定义 | 使用临时文件导入模块 |

### 5.4 功能验证失败（L2 FAIL）

**问题**：最大误差超过阈值

**解决方案**：
1. 调整 `max_error` 参数（默认 1e-3）
2. 将累加器精度从 float32 改为 float64
3. 在代码清理逻辑中添加精度转换：
   ```python
   code = code.replace(
       'acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)',
       'acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)'
   )
   ```

### 5.5 性能验证失败（L3 FAIL）

**问题**：GFLOPS 未达到目标

**解决方案**：
1. 降低 `target_gflops` 参数
2. 增加 `max_optimize_rounds` 参数
3. 使用 SearchAgent 进行多轮优化

### 5.6 日志文件过多

**问题**：logs/ 目录下日志文件过多

**解决方案**：
- 系统会自动清理，只保留最新 10 个日志文件
- 可手动调用清理函数：
  ```python
  from operator_agent import cleanup_old_logs
  deleted = cleanup_old_logs(keep_count=10)
  print(f"删除了 {deleted} 个旧日志文件")
  ```

### 5.7 产物目录命名问题

**目录格式**：`timestamp_kernel_id`

**示例**：`20260726_203150_af206fdd`

**结构**：
```
output/matmul/nvidia_gpu/20260726_203150_af206fdd/
├── code.py           # 生成的算子代码
├── sketch.json       # 设计草图
├── result.json       # 完整产物信息
├── verification.json # 验证结果
└── benchmark.json    # 性能基准数据（如果有）
```

### 5.8 常见错误码

| 错误码 | 说明 | 解决方向 |
|-------|------|---------|
| `Errno 7` | Argument list too long | 使用 stdin 传递数据，避免命令行参数过长 |
| `ModuleNotFoundError` | 缺少依赖 | 安装所需依赖（numpy, torch, triton） |
| `AttributeError` | 对象属性不存在 | 检查对象初始化和方法调用 |
| `ValueError` | JSON 解析失败 | 检查 LLM 返回格式是否正确 |

### 5.9 「跑成功了但好像没生效」的五种情况

这五种都不是 bug，是第 6 章那几道门在起作用。按你会看到的输出对号：

| 你看到的 | 原因 | 去处 |
|---------|------|------|
| L3 PASSED，但 `[KernelDB] 未入库` | 入库门拒收（档位不可信 / 被抢占 / 没有真的加速） | 6.5 |
| 日志里带 `⚠️ 测量有效性=degraded/uninformative`，且本轮**没有**禁用任何配方 | 该档位没有驱动闭环的资格，这是设计行为 | 6.3 |
| 连续两轮性能几乎一样，却没有禁用配方 | `SAME` 不禁用（只有 `SLOW` 才禁），但仍计入停滞 | 6.4 |
| L1 一进来就被退回，错误消息提到参数名 | 接口契约门在动 GPU 之前拦的 | 6.2 |
| 跑测试之后 `output/`、`logs/` 里多了几百个目录 | 测试现场与生产现场没分开 | 设 `OPERATOR_AGENT_OUTPUT_DIR` / `LOG_DIR=off`，见 4.2 |

---

## 6. 验证链路上的门与判决档位

本章讲的是「**为什么一个通过了的核没有生效 / 没有被记住**」。v0.6.0 起链路上多了五道门，
它们的存在理由都是同一个：早期版本里 verifier 算出来的东西（污染标记、性能档位、失败类别）
被下游直接当成可信信号用了，而其中一些读数在真机上被证明是错的。

阅读顺序建议：先看 6.1 的总表，哪道门挡了你按表查小节。

### 6.1 总表：什么信号有资格驱动什么决定

| 信号 | 可以驱动 | **不可以**驱动 |
|------|---------|---------------|
| 测量档位 `valid` | 全部：本轮判定、禁用配方、停滞计数、入库、判「已到 roofline」 | — |
| 测量档位 `degraded` | 本轮通过 / 不通过的判定 | 禁用配方、停滞计数、入库 |
| 测量档位 `uninformative` | 仅作为诊断信息打印出来 | **任何**闭环决定（判决一律转 `AMBIGUOUS`） |
| L2 数值容差类失败 | 本轮 Fixer 修代码 | 成为跨任务的可复用教训（被类别门挡） |
| 基础设施类失败（无基线 / 无入口 / ssh 断） | 本轮重试与告警 | 同上 —— 它不是模型的错 |
| 候选结构重复（同一份核改了名字） | 写进血缘记录与日志 | 跳过任何一级验证（见 6.7） |

一句话规则：**「通过了」不等于「测准了」，「测准了」不等于「值得被记住」。**

### 6.2 接口契约门（在 L1 里、动 GPU 之前）

调用方固定以 `kernel(**输入)` 的**全关键字**方式调用核函数。这条契约以前只写在实现里没写下来，
违反它的代价是一整轮 Fixer（真实案例：attention 连烧 3 轮在
`TypeError: kernel() got an unexpected keyword argument 'causal'`）。

现在它在 [`interface_contract.py`](../operator_agent/interface_contract.py) 里被静态检查，
调用点是 `verifier._verify_compile`（即 L1，不占 GPU）。

**你会看到什么**：一段处方式说明，含「本次会传入的输入名 / `kernel()` 声明的参数 /
具体哪几项不匹配 / 改法 / `def kernel` 的行号」。

**改法**：让 `def kernel(...)` 的参数名**逐一等于**输入名；确实用不到的参数也照样写进签名、
在函数体里忽略。**不要改输入名** —— 那是任务侧给的。

**看不准就放行**（宁可漏拦不误拦，误拦会把能跑的核打回去重画）：签名里有 `**kwargs`、
缺的参数有默认值、代码有语法错（那是预检的活）、找不到 `def kernel` —— 一律不拦。

**成本**：拦一次 0.04 s，走完整验证 15.5 s。

### 6.3 测量有效性三态

一次 L3 读数先被分档，档位决定它有资格参与哪些决定（表见 6.1）。三个信号来源：

| 来源 | 阈值 | 落到哪档 |
|------|------|---------|
| launch 地板占实测时间 | > 30% | `uninformative` |
| | > 10% | `degraded` |
| 样本主体离散度 `iqr/median` | > 0.30 | `uninformative` |
| | > 0.10 | `degraded` |
| 测量前 GPU util > 5%（判定被抢占） | — | `degraded` |

三点要解释清楚的行为：

- **为什么 util 污染只判 `degraded` 不判 `uninformative`**：基线与候选是在同机同卡交错计时的，
  邻居插队对两臂同等影响，比值仍部分可用。
- **util 的采法**是「隔 1.2 s 采两次取较小值」。因为 `nvidia-smi` 的 `utilization.gpu` 是过去约
  1 秒窗口的占比，紧挨着上一轮测量采样会读到**你自己上一轮的残留** —— 较大的那次可能全是自己，
  较小者才接近「别人占了多少」。
- **这些阈值是经验分界，不是推导值**（代码注释里明写），改动需要回归验证。

### 6.4 四态判决与 roofline 收敛出口

两个优化轮次之间不再做「`current > best`」这种无带宽的标量比较，而是四态：

```
FAST        提升超过噪声带宽
SAME        落在带宽内   → 不禁用配方
SLOW        退步超过带宽  → 禁用本轮新配方
AMBIGUOUS   没有可比值，或档位是 uninformative（带宽比信号本身还宽）
```

噪声带宽按档位取 `0.10% / 0.40% / 2.00%`。**只有 `SLOW` 才禁用配方** —— 禁用是永久决策，
「这次没测出更好」的证据强度不够（打平与差 0.0001% 在旧规则下都会拉黑一个实际有效的配方）。

**roofline 收敛出口**：绑定轴上的达成率 ≥ `90%` 时判定「已经贴着上界」，直接停止继续换配方
（继续换只会白烧 GPU 与 token）。达成率取 compute 轴与 memory 轴的 `max`。
三个前置条件任一不满足就**不开**这个出口：测量档位不可信、窗口被抢占、
**以及峰值分母自己未被核过口径**（硬件表里标 ⚠️ 的行可能偏小一倍，那会把还能优化的核提前判成到顶）。

### 6.5 入库门：通过 ≠ 被记住

`persist_decision()`（[`measurement.py:112`](../operator_agent/measurement.py)）决定这次运行的核
值不值得写进跨运行经验库。三条拒收，任一命中就不入库、并打印理由：

| 拒收条件 | 理由 |
|---------|------|
| `measurement_can_gate` 为假 | 「通过」可以是被 launch 地板喂出来的假信号；持久化会把它**固化成下一次检索的样本**，比一次性误判更贵 |
| 窗口被抢占 | 同上，读数不可比 |
| 相对同机基线没有加速（`speedup <= 1.0`） | 比基线慢的核不该当示范 |

分数用 `speedup_vs_baseline` 而不是 GFLOPS —— 对 memory-bound 算子 GFLOPS 整个失真。

### 6.6 负样本记忆：签名、复发门、类别门

失败现在会落盘（`OPERATOR_AGENT_FAILURE_DB`），并归一化成一个**与任务无关**的签名：

```
层级 | 失败类名 | 最内层栈帧
```

`失败类名` 取自异常类名或一组已知标记（`numeric_tolerance`、`cheat_detected`、
`infra_no_baseline`、`no_kernel_entry` 等）；`最内层栈帧` 取 traceback 里**最后**一帧的函数名，
因为那才是根因所在，最外层只是包装者。签名刻意**不含**任务 id、kernel_type、行号、文件路径
—— 否则「同一种失败」永远数不出跨了几个任务。

两道门决定一条失败记录能不能成为可复用教训：

- **复发门**（默认 3，`OPERATOR_AGENT_FAILURE_GATE`）：这条签名要在 ≥3 个**不同任务**上出现过。
  只挡从文件重放回来的记录；本次进程内新产生的记录一定放行（那是首次遭遇，不是复发）。
- **类别门**：**L2 数值容差类**与**基础设施类**失败永远不作为教训。理由很硬 ——
  从「误差超过门限」能学到的只有「把检查放宽」，那是给 reward hacking 建库。
  所以也别为了让 L2 通过去调大 `max_error`：数值门是防线本身（曾经有一个整体缩小 14.93% 的
  核被判为通过）。

运行结束时会打一行账：`[FailureDB] … 本次被复发门挡住 N 条·被类别门挡住 M 条`。

> ⚠️ **已知未接的最后一跳**：复发计数目前**只打印**，还没有反过来影响 Planner/Designer 的选择。
> 「跨任务复发 ⇒ 是 harness 缺陷」这条升级规则尚未接通。

### 6.7 候选结构键：只观测，不跳过

每轮的候选代码会算一个 α-等价 AST 结构键（标识符按首次出现顺序重命名，**常量保留**），
于是「第 3 轮只是第 1 轮改了变量名」第一次看得见。结束时打：

```
[Novelty] 3 轮候选, 3 个不同结构, 其中 0 轮与更早的轮次结构相同
```

**重复轮照常走完整三级验证**，一个都不跳。理由是量过的：重复轮会判 `SAME`，
`SAME` 不禁配方但**计入停滞**，所以最坏情况是每任务多烧约 2 次 L3；
而在档位不可信时想省的更多 —— 那恰恰是唯一不敢复用历史读数的场合。

---

## 附录：核心模块参考

### A.1 Agent 模块

| Agent | 职责 | 文件 |
|-------|------|------|
| DesignerAgent | 生成设计草图（算法选择、tiling策略） | [agents.py](file:///Users/zh/repos/operatorAgent/operator_agent/agents.py) |
| CoderAgent | 根据 Sketch 生成代码 | [agents.py](file:///Users/zh/repos/operatorAgent/operator_agent/agents.py) |
| SearchAgent | 分析性能瓶颈并优化 | [agents.py](file:///Users/zh/repos/operatorAgent/operator_agent/agents.py) |
| FixerAgent | 修复代码错误 | [fixer.py](file:///Users/zh/repos/operatorAgent/operator_agent/fixer.py) |

### A.2 验证等级

| 等级 | 名称 | 验证内容 |
|------|------|---------|
| L1 | 编译验证 | 代码可编译、语法正确 |
| L2 | 功能验证 | 输出正确，误差在容许范围内 |
| L3 | 性能验证 | 性能达到预期加速比（GFLOPS） |

### A.3 路由策略

| 条件 | 路由类型 |
|------|---------|
| 融合算子 / 高复杂度 | AGENT_GENERATION |
| 目标硬件不支持该 DSL | AGENT_GENERATION |
| 低复杂度 | CODE_TRANSLATION |
| 中复杂度 + 知识库命中 | COMPILER_BACKEND |
| 中复杂度 + 无知识库命中 | AGENT_GENERATION |

---

**文档版本**: v0.6.0+  
**最后更新**: 2026-09-28（补第 6 章「验证链路上的门与判决档位」、4.2 的运行期开关、5.9 排查表；
这三块对应的是 v0.6.0 之后 `research/agent-design-2026` 分支引入的机制，此前手册停留在 v0.5.0）