# Operator Agent 用户使用文档

## 目录

1. [项目简介与核心功能概述](#1-项目简介与核心功能概述)
2. [安装部署与环境配置](#2-安装部署与环境配置)
3. [基础功能使用指南与操作示例](#3-基础功能使用指南与操作示例)
4. [高级配置参数说明](#4-高级配置参数说明)
5. [常见问题与故障排查指引](#5-常见问题与故障排查指引)

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

**文档版本**: v0.5.0  
**最后更新**: 2026-07-26