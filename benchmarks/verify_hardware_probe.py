"""真机标定「硬件探测的诚实性」 (doc 49 §3.3 的 S1 / S2 / S7 + 第 4、5 处)

在控制端运行 (会触发一次真 ssh 往返, 需要能连到 GPU 机器):
    python3 benchmarks/verify_hardware_probe.py

要还的债: #46 的 23 条守卫**全部离线**, 靠 mock 驱动真实现。而这一族改的是真机路径上
「探测成功时返回什么」与「探测失败时抛不抛」, 所以两条都要在真机上量:
① 成功的读数必须来自真探测 (而不是表里抄的规格), ② 那处改成**致命**的
`.python-version` 写入 (第 5 处) 在真机上确实一直成功 —— 否则我会把一次能跑的报告
变成一次跑不了的回退。

注入的「已知会被看见」的信号是**两条独立来源交叉核对**, 而不是自我一致:
`get_gpu_info()` 的 name/显存要和远端 `nvidia-smi` 的直接问法对上 —— 本项目的表里
写多少都不算数。

| 检查 | 内容 | 期望 |
|---|---|---|
| 1 | 探测值 vs 另一条路径 (nvidia-smi) 的器件名与显存 | 一致 |
| 2 | `compute_units` == 探测到的 SM 数; `local.size` 是 H20 的 228 而不是编造的 128 | 成立 |
| 3 | 第 4 处: `memory.local.bandwidth` | **unknown** (两个真源里都没有这个字段) |
| 4 | 把 runner 打桩成抛异常 (模拟 runner 坏) | 抛 `HardwareProbeFailed`, 不返回字典 |
| 5 | host 指向不可达别名 (传输层真的坏) | 同样抛 —— 证明第 4 条不是只有打桩才能触发 |
| 6 | 真机 `_setup_uv_env()` | True, 且远端 `.python-version` 与远端解释器版本**读回来**一致 |

⚠️ 只打印派生值 (布尔、数字、版本号)。远端原文必然带那台机器的工作目录与用户名,
按项目约定不进仓库、也不进本脚本的输出。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from operator_agent.remote_executor import (  # noqa: E402
    HardwareProbeFailed,
    SSHExecutor,
)


def _nvidia_smi_pair(ex):
    """绕开本项目的一切代码, 直接问远端 nvidia-smi: (器件名, 总显存 GB)

    ⚠️ 单位不是同一套: `nvidia-smi --query-gpu=memory.total` 给的是 **MiB**,
    而 runner 里的 `props.total_memory / 1e9` 是**字节除以 10^9**。96GB 的 H20 报
    97871 MiB = 102.6e9 字节 —— 不按 1048576 换算就必然「不一致」, 而那个不一致是
    我造的, 不是探测错了。
    """
    out = ex._ssh_exec(
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits "
        "| head -1").strip().splitlines()[0]
    name, mem_mib = [p.strip() for p in out.split(",")[:2]]
    return name, round(float(mem_mib) * 1048576 / 1e9, 1)


def _same_card(size_str, smi_gb, tol=1.0) -> bool:
    try:
        return abs(float(str(size_str).rstrip("GBgB ")) - float(smi_gb)) <= tol
    except ValueError:
        return False


def main() -> int:
    ex = SSHExecutor()
    print(f"后端: {type(ex).__name__}({ex.platform})  能力 {len(ex.capabilities)} 项")

    ok_setup = ex._setup_uv_env()
    info = ex.get_gpu_info()
    smi_name, smi_mem = _nvidia_smi_pair(ex)

    print("\n探测到的 (派生值, 不含路径):")
    print(f"  name={info.get('name')!r}  compute_units={info.get('compute_units')}")
    print(f"  global.size={info['memory']['global']['size']!r} "
          f"global.bandwidth={info['memory']['global']['bandwidth']!r}")
    print(f"  local.size={info['memory']['local']['size']!r} "
          f"local.bandwidth={info['memory']['local']['bandwidth']!r}")
    print(f"  nvidia-smi 独立问法: name={smi_name!r} total={smi_mem}GB")

    probe = ex._ssh_exec_json({"action": "gpu_info"})
    fabricated = ("40GB", "2TB/s", "128KB/SM", "19TB/s", "NVIDIA GPU (Remote")
    blob = repr(info)

    # 检查 4: 已知会被看见的失败注入 —— runner 抛异常时必须响, 不许返回字典
    stubbed = _probe_failure_raises(_stubbed_executor(ex))

    # 检查 5: 传输层真的坏 (同一个类, 但不是打桩出来的)
    broken_host = SSHExecutor(host="probe.invalid", work_dir=ex.work_dir)
    transport = _probe_failure_raises(broken_host)

    # 检查 6: 远端 pin 的**实际内容**读回来核对 (第 5 处改成致命之后的关键对照)
    pin = ex._ssh_exec(f"cat {ex.work_dir}/.python-version").strip().splitlines()[-1]
    interp = ex._ssh_exec(
        "python3 -c 'import sys;print(\"%d.%d\"%sys.version_info[:2])'"
    ).strip().splitlines()[-1]
    print(f"  远端 .python-version={pin!r}  远端解释器={interp!r}")

    checks = [
        ("1 探测值与 nvidia-smi 的独立问法一致 (器件名 + 显存)",
         str(info.get("name")).lower() in smi_name.lower()
         or smi_name.lower() in str(info.get("name")).lower(),
         ),
        ("1b 显存与 nvidia-smi 在同一台卡的换算内一致 (不是表里抄的 40GB)",
         _same_card(info["memory"]["global"]["size"], smi_mem)),
        ("2 SM 数来自探测载荷 (与载荷逐字相等, 不是查表)",
         info.get("compute_units") == probe.get("multiprocessor_count")),
        ("2b 共享内存是这台卡的 228KB/SM 量级, 不是编造的 128KB/SM",
         "128" not in info["memory"]["local"]["size"]
         and info["memory"]["local"]["size"].endswith("KB/SM")),
        ("3 第 4 处: L1/共享内存带宽标成 unknown (没有真源)",
         info["memory"]["local"]["bandwidth"] == "unknown"),
        ("3b 整个字典里不含任何编造值",
         not any(t in blob for t in fabricated)),
        ("4 runner 抛异常时抛 HardwareProbeFailed 而不是返回字典", stubbed),
        ("5 传输层真的坏时也抛 (证明第 4 条不只靠打桩)", transport),
        ("6 真机 _setup_uv_env() 仍然成功 (第 5 处改成致命没有把路堵死)", ok_setup),
        ("6b 远端 pin 与实际解释器一致", pin == interp),
    ]
    print(f"\n{'=' * 66}\n标定判据\n{'=' * 66}")
    bad = 0
    for name, passed_check in checks:
        print(f"  {'✅' if passed_check else '❌'} {name}")
        bad += 0 if passed_check else 1
    return 1 if bad else 0


def _stubbed_executor(ex):
    """复用**同一台真机**的 executor, 只把最外层那次 I/O 换成必抛

    (假件可以被守卫喂绿, 但不能当被测对象 —— 见 doc 49 §7.8 那条教训:
    这里被测的是真 `get_gpu_info` 的 except 分支, 桩只负责让那次往返失败。)
    """
    def boom(payload):
        raise RuntimeError("注入: runner 输出不是 JSON")
    ex._ssh_exec_json = boom          # 实例级替换, 不碰类
    ex._gpu_info_cache = None
    return ex


def _probe_failure_raises(candidate) -> bool:
    try:
        candidate._deployed = True
        info = candidate.get_gpu_info()
    except HardwareProbeFailed as e:
        return not any(t in str(e) for t in ("40GB", "2TB/s", "19TB/s", "128KB/SM"))
    print(f"  ❌ 没抛, 返回了 {info!r}")
    return False


if __name__ == "__main__":
    sys.exit(main())
