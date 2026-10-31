#!/usr/bin/env python3
"""
Nvidia GPU 融合注意力 (Flash Attention) 算子开发 - 端到端测试

测试算子开发 Agent 在远程 NVIDIA GPU 上使用 Triton 实现：
  1. Designer Agent → 生成 tiling strategy / memory plan / sketch
  2. Coder Agent → 生成 Triton kernel (Flash Attention v2)
  3. Verifier → L1编译 / L2正确性 / L3性能基准
  4. Search Agent → 性能优化 (tile size, pipeline stages)

完整流水线:
  用户自然语言 → LLM 意图解析 → Conductor 调度
  → Designer (LLM) 生成 Sketch → Coder (LLM) 生成代码
  → Verifier L1/L2/L3 (远程/本地) → Search (LLM) 优化

运行 (仓库根目录下):
  export LLM_BASE_URL=http://<llm-host>:<port>/v1   # 必填, 缺失则启动即报错
  export LLM_MODEL=<model-name>                     # 可选
  python test_nvidia_attention.py

SSH 默认走 ~/.ssh/config 里的 Host 别名 (见 remote_executor.py)。需要临时指向
别的机器时再设 REMOTE_SSH_HOST / REMOTE_SSH_PORT / REMOTE_SSH_USER / REMOTE_SSH_KEY。
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from operator_agent import (
    LLMClient,
    Conductor, RemoteExecutor, Verifier,
    TaskSchema, SourceSpec, TargetSpec, ConstraintSpec,
    TaskType, TaskComplexity, Precision,
    get_latest_logs,
)
from operator_agent.core.backend_caps import Cap


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
    """统一 LLM 显示标签"""
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


def test_nvidia_attention_llm(llm: LLMClient, ssh_host, ssh_user, ssh_port):
    """
    测试: Agent 在 Nvidia GPU 上实现 Flash Attention 算子
    """
    print("\n" + "#" * 70)
    print("# Nvidia GPU 融合注意力 (Flash Attention) - LLM Agent 全流程测试")
    print("#" * 70)

    # ── 1. LLM 连通性自检 ──
    print(f"\n[Step 0] LLM 客户端: {llm_label(llm)}")
    try:
        resp = llm.chat([{"role": "user", "content": "回复OK"}], max_tokens=256)
        print(f"  LLM 连通测试: OK (回复: {resp[:30]}...)")
    except Exception as e:
        print(f"  ⚠️  LLM 连通测试失败: {e}")

    # ── 2. 用户自然语言请求 ──
    # 不写具体卡型: 硬件由远端探测 (get_hardware_info -> hardware_caps), 在这里点名
    # 一张卡只会让 Designer 拿到与实际不符的 SM 数 / 脊点。
    user_request = (
        "我需要为 Nvidia GPU 用 Triton 实现一个融合注意力算子 "
        "(Flash Attention风格)，要求 fp16 精度，batch=4, heads=8, seq_len=2048, head_dim=64"
    )
    print(f"\n[Step 1] 用户请求: \"{user_request}\"")

    # ── 3. 构造 TaskSchema (Attention) ──
    print("\n[Step 2] 构造 TaskSchema...")
    BATCH, N_HEADS, N_CTX, HEAD_DIM = 4, 8, 2048, 64
    task = TaskSchema(
        task_type=TaskType.OPERATOR_GENERATION,
        source=SourceSpec(
            framework="pytorch",
            kernel_type="attention",  # → 触发 kernel_db RAG 匹配 "flash_attention"
            fused_ops=["online_softmax", "qk_matmul", "pv_matmul"],
            complexity=TaskComplexity.HIGH,  # High → AGENT_GENERATION 路径
        ),
        target=TargetSpec(
            hardware="nvidia_gpu",
            dsl="triton",
            compute_units=78,  # H20 实测 78 SM (A100 SXM4 为 108)
            warp_size=32,
        ),
        constraints=ConstraintSpec(
            precision=Precision.FP16,
            min_speedup=0.5,  # Agent 达到 PyTorch SDPA 的 50% 算过
            max_error=5e-1,  # Flash Attention online softmax 精度稍低
        ),
    )
    print(f"  kernel_type={task.source.kernel_type}, dsl={task.target.dsl}")
    print(f"  complexity={task.source.complexity.value}, precision={task.constraints.precision.value}")

    # ── 4. 初始化验证后端 ──
    print(f"\n[Step 3] 初始化验证后端 (SSH @ {ssh_label(ssh_host, ssh_port)})...")
    remote = RemoteExecutor(
        mode="ssh",
        host=ssh_host,
        port=ssh_port,
        user=ssh_user,
        timeout=900,  # Attention 编译 + 大矩阵计算较慢
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

    # ── 5. 生成 Attention 测试输入 (FP16, B×H×S×D) ──
    print("\n[Step 4] 生成测试输入...")
    S, D = N_CTX, HEAD_DIM
    # Attention FLOPs: 4 × B × H × S² × D (QK^T + PV matmuls, ignore softmax)
    total_gflop = 4.0 * BATCH * N_HEADS * S * S * D / 1e9
    dtype_bytes = 2  # FP16 = 2 bytes
    # Flash Attention 只读写 Q, K, V, O (各 BHS × D bytes)
    qkv_bytes = 3 * BATCH * N_HEADS * S * D * dtype_bytes
    o_bytes = BATCH * N_HEADS * S * D * dtype_bytes
    total_hbm_bytes = qkv_bytes + o_bytes
    # Arithmetic Intensity: FLOPs / Bytes
    ai = total_gflop * 1e9 / total_hbm_bytes
    print(f"  Attention shape: B={BATCH}, H={N_HEADS}, S={S}, D={D}")
    print(f"  计算量: {total_gflop:.1f} GFLOPs (4×B×H×S²×D)")
    print(f"  数据传输 (Flash): ~{total_hbm_bytes / 1e9:.2f} GB (Q+K+V read, O write)")
    # 脊点不写死: 原先硬编码 A100 的 153, 在 H20 上会把 memory-bound 判成
    # compute-bound (H20 fp16 脊点 ≈ 37.6, 差 4x), 优化方向整个反过来。
    # 复用 Verifier 的同一张峰值表, 免得 driver 与验证器各说一套。
    peak = Verifier(backend)._detect_theoretical_peak()
    bw = peak.get("memory_bandwidth_gbs") or 0
    ridge = (peak.get("fp16_tflops", 0) * 1000 / bw) if bw else 0.0
    dev = peak.get("name") or peak.get("device") or "未识别硬件"
    print(f"  算术强度 (AI): {ai:.0f} FLOPs/Byte ({dev} fp16 脊点 ≈ {ridge:.0f})")
    if not ridge:
        print(f"  → 未取到硬件峰值, 跳过 bound 判定 (以 Verifier 的 roofline 为准)")
    elif ai > ridge:
        print(f"  → Compute-bound: 优化目标 = Tensor Core 利用率")
    else:
        print(f"  → Memory-bound: 优化目标 = 带宽利用率")

    # 生成 Q, K, V (先在 CPU 用 FP32 生成高精度参考，再转 FP16 给 kernel)
    np.random.seed(42)
    Q_fp32 = (np.random.randn(BATCH, N_HEADS, S, D) * 0.05).astype(np.float32)
    K_fp32 = (np.random.randn(BATCH, N_HEADS, S, D) * 0.05).astype(np.float32)
    V_fp32 = (np.random.randn(BATCH, N_HEADS, S, D) * 0.05).astype(np.float32)

    test_inputs = {"Q": Q_fp32, "K": K_fp32, "V": V_fp32}
    # Optional: causal=False, sm_scale=1/sqrt(D)
    test_inputs["causal"] = False
    test_inputs["sm_scale"] = 1.0 / np.sqrt(D)

    # 计算参考输出 (优先用 PyTorch，本地没有时用纯 NumPy 的手动实现)
    print(f"  计算参考输出...")
    expected_output = None
    try:
        import torch
        import torch.nn.functional as F
        print(f"  使用 PyTorch SDPA (FP32)...")
        Qt = torch.from_numpy(Q_fp32)
        Kt = torch.from_numpy(K_fp32)
        Vt = torch.from_numpy(V_fp32)
        with torch.no_grad():
            O_ref = F.scaled_dot_product_attention(
                Qt, Kt, Vt,
                is_causal=False,
                scale=1.0 / np.sqrt(D),
            ).numpy()
        expected_output = O_ref.astype(np.float32)
        print(f"  期望输出: shape={expected_output.shape}, dtype=float32 (PyTorch SDPA)")
    except ImportError:
        # 本地无 torch，用 NumPy 手动实现 (准确但慢)
        print(f"  本地无 PyTorch，使用纯 NumPy 手动计算...")
        # (B, H, S, D)
        B, H, S, D_shape = Q_fp32.shape
        scale = 1.0 / np.sqrt(D)
        expected_list = []
        for b in range(B):
            for h in range(H):
                Q_h = Q_fp32[b, h]  # (S, D)
                K_h = K_fp32[b, h]
                V_h = V_fp32[b, h]
                # QK^T (S, D) @ (D, S) = (S, S)
                scores = Q_h @ K_h.T * scale  # (S, S)
                # online softmax (numerically stable)
                scores_max = scores.max(axis=-1, keepdims=True)
                exp_scores = np.exp(scores - scores_max)
                attn = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
                # PV (S, S) @ (S, D) = (S, D)
                O_h = attn @ V_h
                expected_list.append(O_h)
        expected_np = np.stack(expected_list).reshape(B, H, S, D_shape)
        expected_output = expected_np.astype(np.float32)
        print(f"  期望输出: shape={expected_output.shape}, dtype=float32 (NumPy manual)")

    # ── 6. Conductor 执行完整流水线 ──
    print(f"\n[Step 5] Conductor 执行完整流水线 (LLM: {llm.model})")
    print(f"  max_optimize_rounds=3, min_speedup=0.5×PyTorch SDPA")
    print()

    max_retries = 1
    artifact = None
    for attempt in range(1, max_retries + 1):
        print(f"\n  ── 尝试 {attempt}/{max_retries} ──")
        conductor = Conductor(
            backend=backend,
            llm=llm,
            max_optimize_rounds=3,  # Designer → Coder → Verifier ×3 优化
        )
        try:
            artifact = conductor.execute(
                task=task,
                test_inputs=test_inputs,
                expected_output=expected_output,
                target_gflops=100.0,  # A100 参考 ~160 TFLOPS
            )
            if artifact.all_passed:
                print(f"\n  ✅ 测试通过! L1/L2/L3 全部 OK")
                break
            else:
                print(f"\n  ⚠️  测试未完全通过 (L1/L2 或 L3 失败)")
        except Exception as e:
            print(f"\n  ❌ 执行异常: {e}")
            import traceback
            traceback.print_exc()
            if attempt < max_retries:
                print(f"  重试中...")

    return artifact


def print_report(artifacts):
    """打印测试报告，包含与 Triton 官方 Attention / PyTorch SDPA 对比分析"""
    print("\n" + "=" * 80)
    print("  " + "算子开发 Agent - Nvidia Attention 测试报告".center(76))
    print("=" * 80)

    for i, art in enumerate(artifacts):
        if art is None:
            print(f"\n  测试{i+1}: [NOT AVAILABLE] 无有效产物")
            continue
        status = "PASSED" if art.all_passed else "FAILED"
        print(f"\n  测试{i+1}: [{status}]")
        print(f"  {'─'*76}")
        print(f"    Kernel ID:    {art.kernel_id}")
        print(f"    语言:         {art.language}")
        print(f"    代码长度:     {len(art.code)} chars")
        print(f"    Sketch 版本:  v{art.sketch.version if art.sketch else 'N/A'}")

        print(f"\n    验证结果 (L1/L2/L3):")
        for v in art.verification_results:
            tag = "✅ PASS" if v.passed else "❌ FAIL"
            print(f"      {v.level.value:<8}: [{tag}]  {v.message[:120]}")

        if art.benchmark_data:
            print(f"\n    性能数据:")
            gflops = art.benchmark_data.get("gflops", 0)
            speedup = art.benchmark_data.get("speedup_vs_baseline")
            gflops_util = art.benchmark_data.get("gflops_utilization_pct", 0)
            bw_util = art.benchmark_data.get("bandwidth_utilization_pct", 0)
            ai = art.benchmark_data.get("arithmetic_intensity", 0)
            bound = art.benchmark_data.get("roofline_bound", "unknown")
            hw_peak = art.benchmark_data.get("hw_peak_gflops", 0)

            print(f"      GFLOPS:          {gflops:.1f} / {hw_peak:.0f} peak ({gflops_util:.1f}% util)")
            if speedup is not None:
                print(f"      加速比 (vs baseline): {speedup:.3f}x")
                # 对比基线
                if speedup >= 1.0:
                    print(f"      → 优于 PyTorch SDPA! ✨")
                elif speedup >= 0.8:
                    print(f"      → 接近官方 Triton 实现水平 (90%-100% SDPA)")
                elif speedup >= 0.6:
                    print(f"      → 达到目标 (0.62x SDPA = 80% Triton 官方) ✅")
                else:
                    print(f"      → 低于目标 (Agent target = 0.62x)")
            print(f"      Roofline: {bound} (AI={ai:.1f} FLOP/B)")
            print(f"      带宽利用率: {bw_util:.1f}%")

            # 对比之前的 benchmarks/drivers/test_attention.py 官方结果
            # B=4,H=8,S=4096,D=64: Triton 官方 ~144 TFLOPS / 0.858x SDPA
            # B=4,H=8,S=2048,D=64: Triton 官方 ~? TFLOPS
            tflops = gflops / 1000
            print(f"\n    对比参考 (benchmarks/drivers/test_attention.py 官方 Triton):")
            print(f"      测试形状: B=4,H=8,S=4096,D=64")
            print(f"      官方 Triton 官方峰值: 144 TFLOPS (0.858x SDPA)")
            print(f"      Agent 产出: {tflops:.1f} TFLOPS" + 
                  f" ({tflops/144*100:.0f}% 官方实现)")

        if art.sketch:
            print(f"\n    设计草图 (UnifiedSketch):")
            print(f"      Algorithm: {art.sketch.algorithm}")
            if hasattr(art.sketch, 'tiling') and art.sketch.tiling:
                tiling = art.sketch.tiling
                print(f"      Tiling: BLOCK_M={getattr(tiling, 'block_m', '?')}, "
                      f"BLOCK_N={getattr(tiling, 'block_n', '?')}, "
                      f"BLOCK_K={getattr(tiling, 'block_k', '?')}")
            if hasattr(art.sketch, 'memory_plan') and art.sketch.memory_plan:
                print(f"      Memory Plan: {str(art.sketch.memory_plan)[:80]}")
            if hasattr(art.sketch, 'optimization_hints') and art.sketch.optimization_hints:
                hints = art.sketch.optimization_hints
                if isinstance(hints, list):
                    print(f"      Opt Hints: {len(hints)} items")
                    for h in hints[:5]:
                        print(f"        - {str(h)[:80]}")
                else:
                    print(f"      Opt Hints: {str(hints)[:120]}")

        # 代码质量指标
        if art.code:
            n_lines = art.code.count('\n') + 1
            n_def = art.code.count('def ')
            n_triton_jit = art.code.count('@triton.jit')
            n_autotune = art.code.count('@triton.autotune')
            print(f"\n    代码质量:")
            print(f"      行数: {n_lines}, def 数: {n_def}")
            print(f"      @triton.jit: {n_triton_jit}, @triton.autotune: {n_autotune}")

            # 检查是否有 Flash Attention 关键组件
            code_lower = art.code.lower()
            has_online_softmax = 'online' in code_lower or 'l_i' in code_lower or 'm_i' in code_lower
            has_tile_qk = 'tl.dot' in code_lower or 'tl.load' in code_lower
            has_mask = 'causal' in code_lower or 'mask' in code_lower
            print(f"      Online Softmax: {'✅' if has_online_softmax else '❌'} (l_i/m_i accumulators)")
            print(f"      Tiled QK/PV: {'✅' if has_tile_qk else '❌'} (tl.load / tl.dot)")
            print(f"      Causal Mask: {'✅' if has_mask else '❌'} (causal support)")

    total = len(artifacts)
    passed = sum(1 for a in artifacts if a is not None and a.all_passed)
    print(f"\n{'='*80}")
    print(f"  总通过率: {passed}/{total}")
    print(f"{'='*80}")

    # Agent 效能评分
    if artifacts:
        valid = [a for a in artifacts if a is not None and a.all_passed and a.benchmark_data]
        if valid:
            a = valid[0]
            gflops = a.benchmark_data.get("gflops", 0)
            spd = a.benchmark_data.get("speedup_vs_baseline") or 0
            agent_tflops = gflops / 1000
            # Grade
            if gflops >= 120:  # >= 80% of 144 TFLOPS official
                grade = "⭐⭐⭐⭐⭐ 优秀"
            elif gflops >= 100:
                grade = "⭐⭐⭐⭐ 良好"
            elif gflops >= 70:
                grade = "⭐⭐⭐ 中等"
            elif gflops > 0:
                grade = "⭐⭐ 及格"
            else:
                grade = "⭐ 需改进"
            print(f"\n  Agent 性能评分: {grade}")
            print(f"    Agent 性能 = {agent_tflops:.1f} TFLOPS (官方实现 {144.4} TFLOPS)")
            print(f"    Agent 相对 = {agent_tflops / 144.4 * 100:.0f}%")
            if spd:
                print(f"    vs PyTorch SDPA = {spd:.3f}x (目标 ≥ 0.62x)")

    # 最近日志
    print("\n最近的执行日志:")
    logs = get_latest_logs(n=3)
    for log_path in logs:
        print(f"  - {log_path}")


def main():
    llm = build_llm()
    ssh_host, ssh_user, ssh_port = ssh_config()

    print("=" * 80)
    print("  算子开发 Agent v0.5.0 - Nvidia GPU 融合注意力测试")
    print("=" * 80)
    print(f"  LLM: {llm_label(llm)}")
    print(f"  验证: SSH @ {ssh_label(ssh_host, ssh_port)} (不可用时直接失败, 不降级到模拟)")
    print(f"  目标: B=4, H=8, S=2048, D=64 FP16 Attention")
    print(f"  指标: L1/L2通过 + L3 ≥ 50% PyTorch SDPA 性能")
    print(f"  对比基准: benchmarks/drivers/test_attention.py Triton官方实现")
    print(f"  日志: logs/ 目录 (文件名含 attention_nvidia_gpu)")
    print("=" * 80)

    artifacts = []
    try:
        art1 = test_nvidia_attention_llm(llm, ssh_host, ssh_user, ssh_port)
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
