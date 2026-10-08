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
| **三级验证体系** | L1 编译验证（含静态接口契约门）、L2 功能验证 + 反作弊、L3 交错 A/B 性能验证；L3 读数先分「测量有效性」档位再决定它能驱动哪些决策 |
| **远程执行支持** | 支持 SSH/HTTP/Local 三种模式远程执行代码；共享 GPU 机器上可固定卡号（见 4.2） |
| **自动日志管理** | 所有执行过程自动记录到日志文件，自动清理旧日志（默认保留最新 10 个，`OPERATOR_AGENT_LOG_KEEP` 可调，`OPERATOR_AGENT_LOG_DIR=off` 可完全不落文件） |
| **产物持久化** | 生成的算子代码、Sketch、验证结果自动保存到 `output/` 目录；每轮的动作为结果（含结构签名）记进血缘 `iteration_history` |
| **两级跨运行记忆** | 正样本 `KernelDatabase`（Designer/Coder 检索）与负样本 `FailureDatabase`（Fixer 检索），都是追加式本机 JSONL；写入要过入库门，负样本还要过复发门与类别门（见 6.5、6.6） |

### 1.3 架构设计

```
用户请求
    ↓
IntentParser (意图解析 → TaskSchema)
    ↓
Conductor (调度编排, 最多 max_optimize_rounds 轮)
    ├── Router (路由决策)
    ├── Planner (选优化配方)              ← 配方禁用 / 停滞计数受第 6 章的门约束
    ├── DesignerAgent (设计草图)          ← 读正样本经验库 (KernelDatabase) 做 RAG
    ├── CoderAgent (代码生成 + 预检重试)   ← 同上
    ├── Verifier (三级验证)
    │   ├── L1: 接口契约门 (静态, 不占 GPU) → 语法 → 远程编译
    │   ├── L2: 功能验证 + 反作弊          ← 绝对/相对 AND 容差, 按精度分档
    │   └── L3: 交错 A/B 性能 + Roofline   ← 输出先过「测量有效性」分档
    ├── FixerAgent (L1/L2 失败时修复)      ← 读负样本经验库 (FailureDatabase)
    └── SearchAgent (L3 失败时调 tiling/hints)
    ↓
ArtifactStore (产物快照) → output/<kernel_type>/<hardware>/<时间戳>_<id>/
    ↓
入库门 persist_decision() → 只有「可信 + 出处对 + 未污染 + 真的加速」才写进跨运行经验库
```

四条读图要点：

1. **通过 ≠ 生效**。每轮的性能读数先被分档（`valid / degraded / uninformative`），
   档位决定它有没有资格去禁用配方、累计停滞、或写进经验库 —— 详见第 6 章。
2. **两级记忆**。正样本（成功且可信的核，Designer/Coder 会检索）与负样本
   （失败签名，Fixer 会检索）都是本机 JSONL 文件，默认落在 `output/` 下且不进仓库。
3. **候选代码的每次执行都在反作弊屏蔽区内**（静态 AST 审计 + 运行时 `shield_torch` 两层），
   见 6.8。
4. **远端失败的原因不带连接信息**。传输层异常（超时、scp 失败）的原文里其实**就是整条命令**，
   含 `-p <端口>`、`-i <密钥路径>` 与 `<用户>@<主机>`；这些文本会进 `result.json`、Fixer 的 prompt
   和失败签名，所以落盘前先过一道掩码（`measurement.safe_error_text`）：保留「是哪一类失败、
   哪一步、超时多久」，摘掉地址类信息。**候选代码自己的报错原样保留** —— 那是被修对象的信息。

英文/源码视角的同一张图见设计规范 [`docs/design/autonomous-operator-agent-design.md`](design/autonomous-operator-agent-design.md) §2.1（一次运行的流水线）与 §2.2（五平面目标架构）；当前实现状态与守卫见 [`docs/status/autonomous-operator-agent-status.md`](status/autonomous-operator-agent-status.md)。

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

# 安装依赖 —— 本仓库**没有 requirements.txt**（依赖极少，直接装）
pip install numpy            # 必需：参考实现 / MockGPU
pip install torch triton     # 真机执行才需要（跑 MockGPU 与单测不需要）
```

**开发与自测**（都不需要 GPU / LLM）：

```bash
# 全部单元测试（~2000 条，一分钟内）
python -m pytest tests/ -q

# 单个文件 / 单个用例
python -m pytest tests/test_p1.py -v
python -m pytest tests/test_coder_pipeline.py::TestPlanStage::test_planner_outputs_structured_plan -v

# 真机端到端（需要 GPU 与 SSH）
python test_nvidia_matmul.py

# 冒烟：不碰 GPU 与 LLM
python -c "
from operator_agent import MockGPU, KernelDatabase
print(KernelDatabase().search(algorithm='matmul', hardware='nvidia_gpu', dsl='triton')[0].kernel_id)
"
```

> 跑测试前若担心往仓库里灌产物，用 `OPERATOR_AGENT_OUTPUT_DIR` / `OPERATOR_AGENT_LOG_DIR`
> 指到仓库外（理由见 §4.2 与 §6.7）。

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
cd ~/repos/operatorAgent
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

### 4.1 配置文件结构（`config.yaml` 是**索引**，不是生效配置）

> ⚠️ **`config.yaml` 目前不被任何模块读取**（全仓无 `import yaml`）。改它不会有任何效果。
> 真正生效的配置面是**三处**，换机器时请同步这三处：
>
> 1. `operator_agent/remote_executor.py` 的 `SSHExecutor.__init__` 默认值
> 2. `test_nvidia_*.py` / `test_kunlunxin_matmul.py` 的 `build_llm()` 与 `ssh_config()`
> 3. **环境变量**（见 §4.2）—— 唯一受支持的运行时开关面
>
> 下面这份结构只是这些默认值的**可读索引**（与仓库根目录 `config.example.yaml` 顶部那段
> 说明同一个口径）。`config.yaml` 已被 `.gitignore` 覆盖：它含真实 SSH 主机与 LLM 端点，
> **不要提交**。

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

**LLM 客户端**（`operator_agent/llm_client.py`，排查 LLM 侧问题时先读这三条）：

- **零依赖**：只用标准库 `urllib` 的 OpenAI 兼容客户端，不装 `openai` 包；
- **关掉了 Qwen3 的思考模式**（`chat_template_kwargs={"enable_thinking": False}`）——
  否则 JSON / 代码会被截断；
- **JSON 抽取带自动修复**：模型输出的常见格式错误（围栏、尾逗号、单引号）会被就地修掉，
  修不掉才报错。

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
| `OPERATOR_AGENT_GPU` | ⚠️ **只在远端 runner 那一侧生效**：手工直接跑 `remote_scripts/run_kernel.py` 时用它指定卡号。**走 agent 流水线时它不生效** —— `SSHExecutor` 从不读这个变量，而是本进程按 (低利用率, 高显存余量) 自动选一张卡，再把**自己选中的号**作为前缀下发（覆盖你设的值）。查实际用了哪张卡：量具报表的 `device` 字段（来源是后端，不是环境变量）或日志里的「固定使用 GPU N」。目前**没有**受支持的「钉死某张卡」开关，要改 `_pinned_gpu()` | 不设 → 自动选；利用率读不到时按最坏的 100% 处理 |
| `OPERATOR_AGENT_KERNEL_DB` | 正样本经验库（成功且可信的核）的 JSONL 路径。取 `off`/`none`/`0`/`false` 时整个关闭持久化，只留本次进程内存 | `<repo>/output/kernel_db.jsonl` |
| `OPERATOR_AGENT_FAILURE_DB` | 负样本经验库（失败签名）的 JSONL 路径，关闭语义同上 | `<repo>/output/failure_db.jsonl` |
| `OPERATOR_AGENT_FAILURE_GATE` | 一条失败签名要在**多少个不同任务**上复发，才升级为可复用教训（见 6.5） | `3` |
| `OPERATOR_AGENT_OUTPUT_DIR` | 产物快照的根目录。跑批量实验或单元测试时指到仓库外，否则历史语料会被测试现场淹没（见 6.7 的盘点） | `<repo>/output` |
| `OPERATOR_AGENT_LOG_DIR` | 日志目录。取 `off`/`none`/`0`/`false` 时**完全不落文件**，只透传标准输出 | `<repo>/logs` |
| `OPERATOR_AGENT_LOG_KEEP` | 自动清理保留最近多少个日志。做长周期实验时默认值会把你还要复查的现场删掉 | `10` |

> 这三个「关闭 / 改址」开关（`KERNEL_DB`、`OUTPUT_DIR`、`LOG_DIR`）存在的理由是同一件事：
> Conductor 无条件走落盘路径，所以**跑一次测试就会往仓库里灌一批产物、并顺带把真实实验
> 现场轮掉**。单元测试靠它们保持仓库无副作用。

> ⚠️ **`kernel_db=None` 不等于「没有经验库」**。`Conductor.__init__` 写的是
> `kernel_db if kernel_db is not None else KernelDatabase()`（`conductor.py`），
> 所以传 `None` 拿到的是**默认那个真库**；`KernelDatabase(jsonl_path=None)` 同理，
> 会落回 `<repo>/output/kernel_db.jsonl`。实测：
>
> ```text
> KernelDatabase(jsonl_path="off")._jsonl   -> None                                  # 真的关掉
> KernelDatabase(jsonl_path=None)._jsonl    -> <repo>/output/kernel_db.jsonl         # 默认库
> ```
>
> 想「这一趟不读也不写语料」，**设环境变量**，别传 `None`。这条与 doc 69 里
> 「`KernelDatabase(jsonl_path=None)` 不是『无库』，而是『用默认库』」是同一条。

### 4.3 Conductor 参数

```python
Conductor(
    backend=None,                # 验证后端（MockGPU 或 RemoteExecutor）
    kernel_db=None,              # 算子知识库 —— ⚠️ None = 用**默认那个库**, 不是空库 (见 4.2 的 ⚠️)
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

日志自动清理在 [agent_logger.py](../operator_agent/agent_logger.py#L188) 中配置：

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

**先理解门的形式**：L2 不是纯绝对容差，而是**绝对 + 相对两项 AND**，按精度分档取阈值
（[`tolerance.py`](../operator_agent/tolerance.py)）：

| precision | atol | rtol |
|-----------|------|------|
| `fp32` | 1e-4 | 1e-4 |
| `tf32` | 1e-3 | 1e-3 |
| `fp16` / `bf16` | 1e-2 | 1e-2 |
| `int8` | 0.0 | 0.0（整数输出应当逐元素相等） |

AND 是关键：近零参考处相对误差会爆，但只要绝对误差在 `atol` 内就不算坏点。

**⚠️ 但 L2 其实有三条比对路，只有两条走上面那把尺**（2026-09-30 核实的，见
[`docs/research/12`](research/12-正确性验证与reward-hacking防护.md) §八）：

| 路 | 什么时候走 | 判据 | 报的尺键 |
|---|---|---|---|
| 全量 | 后端声明 `FULL_OUTPUT_TENSOR`（本地 numpy / MockGPU） | 绝对 + 相对 AND + ρ | `atol` `rtol` `required_matched_ratio` |
| 真机 | 后端声明 `REMOTE_VERIFY`（SSH 上 `run_kernel.py`） | 同一个 `tolerance` 模块 | 同上三键（逐键透传） |
| **采样窗口** | 执行侧只回样本（两个能力都没声明） | **纯绝对** `max_error <= 任务自带值` | 只有 `atol`（= 那个任务值） |

`effective_atol` 取的是 `min(表, 任务值)`，所以第三条那把尺**只会同松或更松**：探针臂 4 里一份
常数偏移 3e-3 的候选配大幅值参考，纯绝对 `atol=5.00e-01` 判过、AND 那条 `atol=1.00e-04` 拦下。
2026-09-30 那一步做的是**观测**（不改判决）：这条路现在如实发布自己用过的 `atol`，并把同一条
AND 判据在同一窗上的第二个读数发布成 `and_form_*`，日志与 `metrics` 里都能看出哪几组用的不是
同一把尺（`absolute_only_sets` / `and_form_would_reject_sets`）。换成同一把尺要带真机 A/B（#80 第 2 片）。

**⚠️ 「调大 `max_error`」这条路现在是无效的**，别再按它排查。任务自带的 `max_error` 与容差表
合并时取的是 **min**（`effective_atol()`）——**任务只能收紧，不能放宽**。历史上三个 driver 都传过
`max_error=5e-1`，那不是疏忽：在只有绝对项的门禁下，输出幅值到 3.35 时确实需要很大的 `atol`
才能让正确的核通过。加上相对项之后，实测合法 fp16 累加核的 `max_abs` 只有 1.2e-2，这个量级
不再需要 —— 而 0.5 恰好等于「允许结果整体缩小 14.93%」仍能判过。所以传宽松值现在是 no-op，
传更严的值（如 `1e-5`）仍然生效。

**确实需要改容差时**：改 `TOLERANCE_BY_PRECISION` 并写明理由，**不要在调用处放宽**。
表里 `int8` 那行的注释就是这个原则的出处。

**正确的排查方向**（按命中率高到低）：

1. **累加器精度**：fp16/bf16 的 matmul 累加必须走 `fp32` 累加器。写 `fp16` 累加器是精度 bug
   （配方库早期版本教过这条，已改）。把 float32 换成 float64 不是解法 —— 累加器本来就该是
   float32，而 Triton 的 fp64 路径在张量核上没有性能可言。
2. **参考实现的口径**：L2 与一个 PyTorch/NumPy 参考比。先确认参考用的精度与任务声明的
   `precision` 一致（曾有过 L2 按 fp32 判精度、L3 却按 TF32 给算力的一侧不一致，见 doc 30 第六节）。
   注意一个正确的 Triton fp16 核对 cuBLAS 参考的实测误差是 **0.0** —— 余量约 4 个数量级，
   所以「差一点点」几乎总是上面两条之一，不是容差太紧。
3. **边界与 mask**：shape 不是块尺寸整数倍时，尾块的 mask/累加范围错会表现为数值超差。
4. **反作弊拦的是「用了库函数」而不是「算错了」**：如果 error_log 里是 `cheat_detected`，
   那与容差无关，见 6.8。

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

### 5.9 「跑成功了但好像没生效」的六种情况

这六种都不是 bug，是第 6 章那几道门在起作用。按你会看到的输出对号：

| 你看到的 | 原因 | 去处 |
|---------|------|------|
| L3 PASSED，但 `[KernelDB] 未入库` | 入库门拒收（档位不可信 / 被抢占 / 没有真的加速） | 6.5 |
| 日志里带 `⚠️ 测量有效性=degraded/uninformative`，且本轮**没有**禁用任何配方 | 该档位没有驱动闭环的资格，这是设计行为 | 6.3 |
| 连续两轮性能几乎一样，却没有禁用配方 | `SAME` 不禁用（只有 `SLOW` 才禁），但仍计入停滞 | 6.4 |
| L1 一进来就被退回，错误消息提到参数名 | 接口契约门在动 GPU 之前拦的 | 6.2 |
| `CheatDetectedError`，但代码里没看到 `torch.matmul` | 静态审计之外还有一层运行时屏蔽，间接调用在这里暴露 | 6.8 |
| 跑测试之后 `output/`、`logs/` 里多了几百个目录 | 测试现场与生产现场没分开 | 设 `OPERATOR_AGENT_OUTPUT_DIR` / `LOG_DIR=off`，见 4.2 |

---

## 6. 验证链路上的门与判决档位

本章讲的是「**为什么一个通过了的核没有生效 / 没有被记住**」。v0.6.0 起链路上多了几道门，
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
| L2「本级判不出」（参考侧没有一个可比元素） | 本轮日志里那句归因 + 第四路取向（见 6.6） | **改代码**、进负样本库、产 Reflexion —— 这一轮对候选没有作出任何判定 |
| 远端没有可用基线（`_NOT_KERNEL_CLASSES` 里的两条之一） | 本轮重试与告警 | 同上 —— 它不是核的错，记进教训库会变成「上次这个算子失败过」的假先验 |
| `cheat_detected`、`no_kernel_entry` | 本轮拒绝 + **可以**成为跨运行教训 | 放宽屏蔽清单（见 6.8） |
| 候选结构重复（同一份核改了名字） | 写进血缘记录与日志 | 跳过任何一级验证（见 6.7） |

一句话规则：**「通过了」不等于「测准了」，「测准了」不等于「值得被记住」。**

### 6.2 接口契约门（在 L1 里、动 GPU 之前）

调用方固定以 `kernel(**输入)` 的**全关键字**方式调用核函数。这条契约以前只写在实现里没写下来，
违反它的代价是一整轮 Fixer（真实案例：attention 连烧 3 轮在
`TypeError: kernel() got an unexpected keyword argument 'causal'`）。

现在它在 [`interface_contract.py`](../operator_agent/interface_contract.py) 里被静态检查，
调用点是 `verifier._verify_compile`（即 L1，不占 GPU）。

**生成之前也说一遍**（2026-09-29）：同一个模块的 `broadcast_calling_convention()` 会把「本次实际
会传入的输入名」作为一节放进 Coder 的 prompt，四条生成路径（Plan 采样 + 三份 legacy 模板）都带。
在此之前 prompt 里唯一的签名是 Planner 按算子类型猜出来的 `def kernel(A, B)`，而真机读数是
**16 轮里 4 轮被这道门拦下**（[`docs/research/48`](research/48-端到端成功率量具.md) §六B、
[`docs/research/41`](research/41-核函数接口契约静态检查.md) §七）。任务没带输入名时这段仍然会写，
并明说「harness 没有拿到输入名」—— 一句消失的约束比一句「不知道」更容易被模型当成「没有约束」。

**你会看到什么**：一段处方式说明，含「本次会传入的输入名 / `kernel()` 声明的参数 /
具体哪几项不匹配 / 改法 / `def kernel` 的行号」。

**修的那一轮也只改这一行**（2026-09-29）：Fixer 的修复取向以前只按级别分（L2 允许退回保守写法 /
L1 就地修一处误用），而这一类失败虽然落在 L1，却是 L1 里唯一不看代码内容的一道 —— 取向现在由
负样本库那一个分类器归类后单独成一路：只改 `def kernel` 的参数名，函数体、`BLOCK_*` 取值与已应用
的优化配方一律不动。为什么不反过来改输入名：远端固定 `kernel(**输入)`，输入名是任务给的。
见 [`docs/research/55`](research/55-算子agent新论文一手核验与吸纳.md) §3.3。

> 同一个分类器在 2026-09-30 又分出**第四路**：L2 报「本级判不出」（参考自己没有可比元素）时，
> 取向不再是「退回保守写法」而是「本轮不改你的核」，且这一轮不进负样本库、不产 Reflexion
> —— 理由与那行账都在 6.6。

**改法**：让 `def kernel(...)` 的参数名**逐一等于**输入名；确实用不到的参数也照样写进签名、
在函数体里忽略。**不要改输入名** —— 那是任务侧给的。

**看不准就放行**（宁可漏拦不误拦，误拦会把能跑的核打回去重画）：签名里有 `**kwargs`、
缺的参数有默认值、代码有语法错（那是预检的活）、找不到 `def kernel` —— 一律不拦。

**成本**：拦一次 0.04 s，走完整验证 15.5 s。

### 6.3 测量有效性三态

一次 L3 读数先被分档，档位决定它有资格参与哪些决定（表见 6.1）。五个信号来源：

| 来源 | 阈值 | 落到哪档 |
|------|------|---------|
| launch 地板占实测时间 | > 30% | `uninformative` |
| | > 10% | `degraded` |
| 样本主体离散度 `iqr/median` | > 0.30 | `uninformative` |
| | > 0.10 | `degraded` |
| 测量前 GPU util > 5%（判定被抢占） | — | `degraded` |
| **出处**：任务声明的 `required_caps` 执行侧没有 | — | `uninformative` |
| **出处**：执行侧声明与载荷自报的 `mode`/`device` 互相矛盾 | — | `uninformative` |
| **环境漂移**：器件名 / torch / triton / 执行侧身份与上一次运行不一致 | — | `degraded` |

后两行问的不是「测得准不准」而是「**测的是不是这个东西**」：一份按硬件规格**估**出来的
加速比（`MockGPU` 的 `mode="simulated"`）与一份在控制机 CPU 上真测的读数（`LocalExecutor`），
都不是「这张卡上这个核有多快」的证据。判据走后端的能力声明（`Cap`），**不**从 `device`/
`name` 的子串去猜；载荷自报的那两个字段只用来**与声明对照**。

任务侧的开关是 `TargetSpec.required_caps`（默认空集，写字符串也行、会被归一成 `Cap`，
拼错的名字当场炸）。默认空集是有意的：本地 mock 运行本来就没有真卡，无条件要求
`REAL_DEVICE` 等于把整条离线链路判死。**要真卡的是量具与真机驱动**，由它们显式声明。

五点要解释清楚的行为：

- **为什么 util 污染只判 `degraded` 不判 `uninformative`**：基线与候选是在同机同卡交错计时的，
  邻居插队对两臂同等影响，比值仍部分可用。
- **util 的采法**是「隔 1.2 s 采两次取较小值」。因为 `nvidia-smi` 的 `utilization.gpu` 是过去约
  1 秒窗口的占比，紧挨着上一轮测量采样会读到**你自己上一轮的残留** —— 较大的那次可能全是自己，
  较小者才接近「别人占了多少」。
- **出处不对会把 `passed` 拉成 False，而其他档位不会**。档位是「这个数字撑不住结论」，
  历史上它不改写通过与否（launch 地板支配那一档的核可能确实是对的）；而任务声明了要真卡、
  执行侧给不出时，「通过」这个结论**没有对象**。这条只在任务真的声明了 `REAL_DEVICE`
  时才拒绝 —— 单有「声明与载荷矛盾」只降档、不拒绝，否则一份本地 mock 会因为载荷老实
  写了 `device="cpu"` 而被判死，那是把观测当判据。
- **环境漂移不设阈值**。「torch 从 2.14 变成 2.15 该不该让一次运行作废」本机还没有分布能
  回答，所以漂移只做到「变了哪些键」可见：不禁用配方、不累计停滞，**也不据此判核失败**。
  「没有上一次可比」与「比过了、一样」是两件事，产物里分开记（`comparable` 字段）。
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

每轮的记录里带着**那次判决真正用过的那条带宽**（`l3_improvement_band`；没用过带宽的那两支记
`null`，与「这一轮没走到判决」的键缺席分开）。跨轮读它时把「带宽换了」拆成两种归因：档位也换了
且值对得上表 ⇒ 那是**测量变差**（`measurement_validity` 已另有一格记着）；档位没换而值变了 ⇒
那才是**规格漂移**（`NOISE_BAND` 被改过，现场通常是续跑跨过了一次代码改动）。conductor 只在后
两者印一行 `[Band]`，端到端报表有【改善判据那条带宽】一栏 —— **只观测，不改判决**。

**roofline 收敛出口**：绑定轴上的达成率 ≥ `90%` 时判定「已经贴着上界」，直接停止继续换配方
（继续换只会白烧 GPU 与 token）。达成率取 compute 轴与 memory 轴的 `max`。
三个前置条件任一不满足就**不开**这个出口：测量档位不可信、窗口被抢占、
**以及峰值分母自己未被核过口径**（硬件表里标 ⚠️ 的行可能偏小一倍，那会把还能优化的核提前判成到顶）。

### 6.5 入库门：通过 ≠ 被记住

`persist_decision()`（[`measurement.py:112`](../operator_agent/measurement.py)）决定这次运行的核
值不值得写进跨运行经验库。四条拒收，任一命中就不入库、并打印理由：

| 拒收条件 | 理由 |
|---------|------|
| `measurement_can_gate` 为假 | 「通过」可以是被 launch 地板喂出来的假信号；持久化会把它**固化成下一次检索的样本**，比一次性误判更贵 |
| `backend_real_device` 为假 | 执行侧没声明 `REAL_DEVICE`：数字要么是按规格**估**的，要么是控制机 CPU 上真测的。语料是按 (算法, 硬件) 检索的，一条 mock 成绩会被下一次真机运行当示范取出来 |
| 窗口被抢占 | 同上，读数不可比 |
| 相对同机基线没有加速（`speedup <= 1.0`） | 比基线慢的核不该当示范 |

`backend_real_device` 用 `is False` 而不是 `not`：这个键之前不存在，老产物不做追溯性重判。

分数用 `speedup_vs_baseline` 而不是 GFLOPS —— 对 memory-bound 算子 GFLOPS 整个失真。

### 6.6 负样本记忆：签名、复发门、类别门

失败现在会落盘（`OPERATOR_AGENT_FAILURE_DB`），并归一化成一个**与任务无关**的签名：

```
层级 | 失败类名 | 最内层栈帧
```

`失败类名` 取自异常类名或一组已知标记（`numeric_tolerance`、`cheat_detected`、
`infra_no_baseline`、`no_kernel_entry`、`reference_uninformative` 等）；`最内层栈帧` 取 traceback 里**最后**一帧的函数名，
因为那才是根因所在，最外层只是包装者。签名刻意**不含**任务 id、kernel_type、行号、文件路径
—— 否则「同一种失败」永远数不出跨了几个任务。

词表里那条 `("本级判不出", reference_uninformative)` **排在最前**，那是刻意的优先级：它说的
是「本级没有对象可比」，也就是一次判定**没有发生**；一轮里若同时拼进别的失败（兜底入库那次
拿的是所有轮次的未通过结果），「这一轮什么都没证明」盖过「这一轮判出了坏结果」。

两道门决定一条失败记录能不能成为可复用教训：

- **复发门**（默认 3，`OPERATOR_AGENT_FAILURE_GATE`）：这条签名要在 ≥3 个**不同任务**上出现过。
  只挡从文件重放回来的记录；本次进程内新产生的记录一定放行（那是首次遭遇，不是复发）。
- **类别门**：**只有三类**失败永远不作为可复用教训 ——
  ① L2 的数值容差类（签名类名 `numeric_tolerance`），② 远端没有可用基线，③ L2「本级判不出」
  （`reference_uninformative`，与 ② 同在一个集合 `_NOT_KERNEL_CLASSES` 里；名字从
  `_INFRA_CLASSES` 改过来的，因为第二条进来之后「infra」不再描述这个集合的共同性质）。理由很硬 ——
  从「误差超过门限」能学到的只有「把检查放宽」，那是给 reward hacking 建库；
  而「这台机当时没有基线」与「参考自己没有可比元素」都不是核的错，记进去会变成
  「上次这个算子失败过」的假先验。③ 还更硬一层：那一条至少观测到了「这次跑没有基线可比」，
  而这一条**什么也没观测到**，沉淀它就等于让模型编一份候选侧根因。
  其余类别（`cheat_detected`、`no_kernel_entry` 等）**照常成为教训** —— 「不要调库算子」、
  「必须有 `def kernel` 入口」都是正确的教训。
  所以也别为了让 L2 通过去调大 `max_error`：数值门是防线本身（曾经有一个整体缩小 14.93% 的
  核被判为通过），而且它对 L2 失败本来就无效（见 5.4）。

⚠️ 类别门只管**从文件重放回来**的记录（复发门同理），所以「本级判不出」这一类在写的一侧也被
挡掉：`conductor._record_failure` 直接 return、循环里那一次 `produce_reflexion()` 不发起。
不挡写的一侧会造出两个读者拿到相反处理 —— 同一次运行里新产生的那条照样被检索回 prompt。

运行结束时会打一行账：`[FailureDB] … 本次被复发门挡住 N 条·被类别门挡住 M 条·参考侧判不出 K 轮 (没入库)`。

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

### 6.8 反作弊：这道门为什么是两层

这一节此前在手册里完全缺失，而它是本项目**最严重缺陷**的对策：曾经有一个只写
`return A @ B` 的假核，在真机上通过了 L1/L2/L3 全部三级，并报出 1.00x vs cuBLAS、
95.3% 峰值利用率（doc 03）。LLM 生成的核「调用库算子冒充自己的实现」是结构性风险，
不是偶发 bug。

两层，共用**同一份**禁用清单（`anticheat.py` 里刻意合并，避免两处漂移）：

| 层 | 位置 | 做法 |
|---|---|---|
| 静态审计 `static_audit` | L2 之前 | AST 遍历被调用者，拦 `torch.matmul` / `mm` / `bmm`、`F.softmax` / `F.linear` 等函数式调用、`np.matmul` / `dot` / `einsum` / `tensordot` / `vdot`、以及张量上的同名方法。早期版本是正则，被证明有 5 种绕过写法（含写在 docstring 里的「参考结果用 torch.matmul 验证」这种说明文字），所以改成按 AST 解析 |
| 运行时屏蔽 `shield_torch` | **包住候选的每一次执行** | 执行期间把 torch 原生算子临时替换成抛 `CheatDetectedError` 的桩，退出时恢复。这是第二道防线：静态看不住的间接调用（lambda 绑定、动态取属性）在这里暴露 |

要点：

- **`CheatDetectedError` 不是编译错误也不是数值错误**，它意味着「这个核没有自己算」。
  修法是让核真的用 Triton 原语实现该运算，而不是换个写法绕开屏蔽。
- **白名单例外**：目标硬件是昇腾/寒武纪这类自研加速卡时**不做任何 torch 屏蔽** ——
  那类硬件只能通过 `torch.matmul` 等高层 API 激活厂商的张量核实现，屏蔽了等于禁掉
  唯一的高性能路径。所以「拦不拦」取决于 `target.hardware`，不是常量。
- **基线与参考实现在屏蔽之外**：屏蔽区只包候选的执行。把参考实现也包进去会让比对永远失败，
  而漏包某一次候选执行就是给绕过开口 —— 新增任何候选调用点时，它必须在屏蔽区内。
- 失败会进负样本库，签名类名是 `cheat_detected`；它属于**可复用教训**（不被类别门挡），
  因为「不要调库算子」是一条正确的教训，而「容差太紧」不是。

---

## 附录：核心模块参考

### A.1 Agent 模块

| Agent | 职责 | 文件 |
|-------|------|------|
| DesignerAgent | 生成设计草图（算法选择、tiling策略） | [agents.py](../operator_agent/agents.py) |
| CoderAgent | 根据 Sketch 生成代码 | [agents.py](../operator_agent/agents.py) |
| SearchAgent | 分析性能瓶颈并优化 | [agents.py](../operator_agent/agents.py) |
| FixerAgent | 修复代码错误 | [fixer.py](../operator_agent/fixer.py) |

### A.2 验证等级

| 等级 | 名称 | 验证内容 |
|------|------|---------|
| L1 | 编译验证 | 先过**接口契约门**（静态，参数名与 `kernel(**输入)` 的调用约定一致，不占 GPU），再语法检查、远程编译 |
| L2 | 功能验证 | 输出与参考实现比对，用**绝对 + 相对两项 AND** 的容差、按精度分档（见 5.4）；同时执行**反作弊**两层（见 6.8） |
| L3 | 性能验证 | 与**选定的基线**做块交错的 A/B 计时，判据是**加速比** `speedup_vs_baseline ≥ min_speedup`，并给出 Roofline 达成率；同时产出峰值显存、逐块比值序列、时钟旁证。⚠️ 这说的是三条比对路里的**两条**：既拿不到远端基线、本机又没有可用参考实现时，第三条（legacy）判的是**绝对 GFLOPS 目标** `gflops ≥ target_gflops × min_speedup` —— 同一个 `min_speedup` 在那里是系数不是地板。哪一条由 `metrics.perf_criterion` 逐轮声明 |

⚠️ 三个常见误解：

- **GFLOPS 通常不是 L3 的通过判据，但有一条路上它是**。有基线时它是 Roofline 报告的输入之一，
  跨运行经验库的分数用 `speedup_vs_baseline`，因为对 memory-bound 算子 GFLOPS 整个失真（缺口 4）；
  而 legacy 那条（无基线、无参考实现）判的正是它。这就是 `perf_criterion` 存在的理由：
  `min_speedup=1.5` 这个数在两条路上**同值不同义**，只比数值会把一次换了量纲的漂移读成
  「尺没变」（#87 第 3 片 / doc 58 §3.4）。
- **L3 的数字先分档再决定用途**。`uninformative` 档的读数只作诊断，不参与任何闭环决定（6.3）。
- **「通过」不等于「入库」**。不可信、被抢占、或没真加速的运行会被入库门拒收（6.5）。

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