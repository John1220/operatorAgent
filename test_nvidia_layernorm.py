#!/usr/bin/env python3
"""
Nvidia GPU LayerNorm 算子开发 - 端到端测试

完整流程:
  用户自然语言 -> LLM 意图解析 -> Conductor 调度
  -> Designer (LLM) 生成 Sketch -> Coder (LLM) 生成代码
  -> Verifier L1/L2/L3 (远程/本地) -> Search (LLM) 优化
  所有执行过程自动写入日志文件 (logs/ 目录)

运行 (仓库根目录下):
  export LLM_BASE_URL=http://<llm-host>:<port>/v1   # 必填, 缺失则启动即报错
  export LLM_MODEL=<model-name>                     # 可选
  python test_nvidia_layernorm.py

SSH 默认走 ~/.ssh/config 里的 Host 别名 (见 remote_executor.py)。需要临时指向
别的机器时再设 REMOTE_SSH_HOST / REMOTE_SSH_PORT / REMOTE_SSH_USER / REMOTE_SSH_KEY。
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from operator_agent import (
    LLMClient,
    Conductor, RemoteExecutor,
    TaskSchema, SourceSpec, TargetSpec, ConstraintSpec,
    TaskType, TaskComplexity, Precision,
    get_latest_logs,
)
from operator_agent.backend_caps import Cap


# ─── LLM 配置 (唯一来源) ─────────────────────────────────────

# ─── 连接配置 (唯一来源) ─────────────────────────────────────
# 真实端点 / 主机 / 密钥一律不进仓库, 全部来自环境变量与 ~/.ssh/config:
#   LLM_BASE_URL (必填) / LLM_MODEL / LLM_API_KEY
#   HY_BASE_URL / HY_MODEL / HY_API_KEY    可选, 主模型失败时降级
#   REMOTE_SSH_HOST / REMOTE_SSH_PORT / REMOTE_SSH_USER / REMOTE_SSH_KEY
# 不设 REMOTE_SSH_* 时 SSHExecutor 用它自己的默认 ssh 别名, 端口 / 用户名 /
# 密钥由 ~/.ssh/config 的 Host 块解析。

def build_llm() -> LLMClient:
    return LLMClient(
        temperature=0.3,
        max_tokens=4096,
        fallback_model=os.environ.get("HY_MODEL") or None,
        fallback_base_url=os.environ.get("HY_BASE_URL") or None,
        fallback_api_key=os.environ.get("HY_API_KEY") or None,
    )


def llm_label(llm: LLMClient) -> str:
    return f"{llm.model} @ {llm.base_url}"


def ssh_config():
    """SSH 验证后端配置。三项全为 None 时交给 SSHExecutor 的别名默认值。"""
    port = os.environ.get("REMOTE_SSH_PORT")
    return (os.environ.get("REMOTE_SSH_HOST") or None,
            os.environ.get("REMOTE_SSH_USER") or None,
            int(port) if port else None)


def ssh_label(host, port) -> str:
    """显示用: 未显式配置时主机/端口来自 ~/.ssh/config, 不能打印成 None"""
    if host and port:
        return f"{host}:{port}"
    return host or "~/.ssh/config 别名"


def test_nvidia_layernorm_llm(llm: LLMClient, ssh_host, ssh_user, ssh_port):
    """
    测试: 用户需要适配 Nvidia GPU 的 LayerNorm 算子
    全 LLM 驱动 + 远程验证 (远程不可用时自动回退本地)
    """
    print("\n" + "#" * 60)
    print("# Nvidia GPU LayerNorm - LLM 驱动全流程测试")
    print("#" * 60)

    # ── 1. LLM 连通性自检 ──
    print(f"\n[Step 0] LLM 客户端: {llm_label(llm)}")
    try:
        resp = llm.chat([{"role": "user", "content": "回复OK"}], max_tokens=256)
        print(f"  LLM 连通测试: OK (回复: {resp[:30]}...)")
    except Exception as e:
        print(f"  ⚠️  LLM 连通测试失败: {e}")
        print("  继续尝试, 下游将报告真实错误")

    # ── 2. 用户自然语言请求 ──
    user_request = "我需要为 Nvidia GPU 用 Triton 实现一个 LayerNorm 算子，要求fp16精度，输入形状为 1024x768"
    print(f"\n[Step 1] 用户请求: \"{user_request}\"")

    # ── 3. 快速通道: 直接构造 TaskSchema ──
    print("\n[Step 2] 快速通道 - 直接构造 TaskSchema...")
    task = TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(
            framework="pytorch",
            kernel_type="layernorm",
            fused_ops=[],
            complexity=TaskComplexity.LOW,
        ),
        target=TargetSpec(
            hardware="nvidia_gpu",
            dsl="triton",
            compute_units=78,   # H20 实测 78 SM (A100 为 108)
            warp_size=32,
        ),
        constraints=ConstraintSpec(
            precision=Precision.FP16,
            min_speedup=0.8,
            max_error=5e-1,  # FP16 精度放宽
        ),
    )
    print(f"  TaskSchema 构造完成: kernel_type={task.source.kernel_type}, dsl={task.target.dsl}")

    # ── 4. 初始化验证后端 ──
    print(f"\n[Step 3] 初始化验证后端 (SSH @ {ssh_label(ssh_host, ssh_port)})...")
    remote = RemoteExecutor(
        mode="ssh",
        host=ssh_host,
        port=ssh_port,
        user=ssh_user,
        timeout=600,
    )
    # SSH 连不上就**失败退出**, 不再凭空造一台 SM 数是真的假 H20 (doc 49 §3.1 A3)。
    # 旧写法两个毛病叠在一起: ① 用显示名子串 `"LocalExecutor" in remote.name` 猜到底连上
    # 没有; ② 猜错时造 MockGPU(name="MockH20", compute_units=78) —— 名字与 SM 数都是真的,
    # 只有「它是模拟器」这件事是假的, 于是跑出来的性能数字看起来完全像 H20 的。
    # mode="ssh" 现在是**承诺**: 连不上时 require() 抛 BackendUnavailable (带每一级探测的
    # 失败原因), 由 main 的异常出口报出来 —— 驱动不该替执行侧编一个能用的替身。
    remote.require(Cap.REAL_DEVICE, Cap.SAME_DEVICE_BASELINE)
    backend = remote
    print(f"  远程连接成功: {remote.name}")

    # ── 5. 生成测试输入 (FP16) ──
    print("\n[Step 4] 生成测试输入...")
    M, N = 1024, 768
    total_elements = M * N
    dtype_bytes = 2  # FP16 = 2 bytes
    mem_x_gb = total_elements * dtype_bytes / 1e9
    print(f"  输入: X({M}x{N}), 元素数={total_elements}, 计算量≈{7 * total_elements / 1e9:.1f} GFLOP (每元素~7 ops)")
    print(f"  显存占用(FP16): X={mem_x_gb:.4f}GB")
    np.random.seed(42)
    X = (np.random.randn(M, N) * 0.1).astype(np.float32)
    # LayerNorm 参数
    weight = np.ones(N, dtype=np.float32)
    bias = np.zeros(N, dtype=np.float32)
    test_inputs = {"X": X, "weight": weight, "bias": bias}
    # 参考输出: PyTorch F.layer_norm
    import torch
    X_t = torch.from_numpy(X)
    expected_output = torch.nn.functional.layer_norm(X_t, (N,),
                                                      torch.from_numpy(weight),
                                                      torch.from_numpy(bias)).numpy().astype(np.float32)
    print(f"  期望输出 Y: {expected_output.shape}, dtype=float32")

    # ── 6. Conductor 执行完整流水线 ──
    print(f"\n[Step 5] Conductor 执行完整流水线 (LLM: {llm.model})...")

    max_retries = 1
    artifact = None
    for attempt in range(1, max_retries + 1):
        print(f"\n  尝试 {attempt}/{max_retries}...")
        conductor = Conductor(
            backend=backend,
            llm=llm,
            max_optimize_rounds=3,
        )
        try:
            artifact = conductor.execute(
                task=task,
                test_inputs=test_inputs,
                expected_output=expected_output,
                target_gflops=1.0,
            )
            if artifact.all_passed:
                print(f"  ✅ 测试通过!")
                break
            else:
                print(f"  ⚠️  测试未完全通过，重试中...")
        except Exception as e:
            print(f"  ❌ 执行异常: {e}")
            if attempt < max_retries:
                print(f"  重试中...")

    return artifact


def print_report(artifacts):
    """打印测试报告"""
    print("\n" + "=" * 60)
    print("                    测试报告")
    print("=" * 60)
    for i, art in enumerate(artifacts):
        status = "PASSED" if art.all_passed else "FAILED"
        print(f"\n  测试{i+1}: [{status}]")
        print(f"    Kernel ID:  {art.kernel_id}")
        print(f"    语言:        {art.language}")
        print(f"    代码长度:    {len(art.code)} chars")
        print(f"    Sketch 版本:  v{art.sketch.version if art.sketch else 'N/A'}")
        for v in art.verification_results:
            tag = "PASS" if v.passed else "FAIL"
            print(f"    {v.level.value}: [{tag}] {v.message}")
        if art.benchmark_data:
            gflops = art.benchmark_data.get("gflops", 0)
            backend = art.benchmark_data.get("backend", "unknown")
            mode = art.benchmark_data.get("mode", "unknown")
            speedup = art.benchmark_data.get("speedup_vs_baseline")
            print(f"    性能: GFLOPS={gflops:.2f}", end="")
            if speedup is not None:
                print(f", 加速比={speedup:.2f}x vs baseline", end="")
            print()
            print(f"    验证后端: {backend}")
            print(f"    执行模式: {mode}")
            # ---- Roofline 指标 ----
            hw_peak = art.benchmark_data.get("hw_peak_gflops", 0)
            gflops_util = art.benchmark_data.get("gflops_utilization_pct", 0)
            bw_util = art.benchmark_data.get("bandwidth_utilization_pct", 0)
            ai = art.benchmark_data.get("arithmetic_intensity", 0)
            roofline_bound = art.benchmark_data.get("roofline_bound", "unknown")
            if hw_peak > 0:
                print(f"    Roofline: {roofline_bound} (AI={ai:.2f} FLOP/Byte, "
                      f"GFLOPS利用率={gflops_util:.1f}%, 带宽利用率={bw_util:.1f}%)")
        if art.compile_info:
            print(f"    编译信息: {art.compile_info}")
    total = len(artifacts)
    passed = sum(1 for a in artifacts if a.all_passed)
    print(f"\n{'=' * 60}")
    print(f"  通过率: {passed}/{total}")
    print(f"{'=' * 60}")

    print("\n最近的执行日志:")
    logs = get_latest_logs(n=3)
    for log_path in logs:
        print(f"  - {log_path}")


def main():
    llm = build_llm()
    ssh_host, ssh_user, ssh_port = ssh_config()

    print("=" * 60)
    print("  算子开发Agent v0.5.0 - Nvidia GPU LayerNorm 测试")
    print(f"  LLM: {llm_label(llm)}")
    print(f"  验证: SSH @ {ssh_label(ssh_host, ssh_port)} (不可用时直接失败, 不降级到模拟)")
    print("  日志: logs/ 目录")
    print("=" * 60)

    artifacts = []
    try:
        art1 = test_nvidia_layernorm_llm(llm, ssh_host, ssh_user, ssh_port)
        artifacts.append(art1)
    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()

    if artifacts:
        valid = [a for a in artifacts if a is not None]
        if valid:
            print_report(valid)
        else:
            print("\n[ERROR] 所有测试均失败，无有效产物")

    return artifacts


if __name__ == "__main__":
    main()
