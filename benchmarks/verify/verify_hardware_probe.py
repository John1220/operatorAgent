"""真机标定「硬件探测的诚实性」 (doc 49 §3.3 的 S1 / S2 / S7 + 第 4、5 处)

在控制端运行 (会触发一次真 ssh 往返, 需要能连到 GPU 机器):
    python3 benchmarks/verify/verify_hardware_probe.py

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
| 7 | 报表读的卡号 (#52) | == 远端 runner 自报的 `selected_gpu`; 读它**不**再花一次往返; 注入一个**必然不等于实卡**的号时 `conflicts` 必须为真而 `gpu` 仍是实卡 |
| 8 | SM 数被字段名丢掉 (§3.4 第 4 处) | 把真机字典里的 `compute_units` 注入成 99 ⇒ `detect_hardware_caps()` 必须跟着变 99 (旧写法读不到那个键, 会回表里的 78) |
| 9 | 推断路径的宽松默认 (§八 第 3 条) | 同一份真机字典三种问法: 表命中→与表逐字段相等; 抹掉器件身份→六个 `has_*` 全 False、DSL 空、但 SM 读数保留; 只补 `device="cuda"`→那六项收回 |

⚠️ 只打印派生值 (布尔、数字、版本号)。远端原文必然带那台机器的工作目录与用户名,
按项目约定不进仓库、也不进本脚本的输出。
"""
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from operator_agent.core.hardware_caps import (  # noqa: E402
    HardwareCaps,
    _KNOWN_HARDWARE,
    detect_hardware_caps,
)
from operator_agent.execute.remote_executor import (  # noqa: E402
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


#: dataclass 里默认为 **True** 的六个 has_*。§八 第 3 条的缺陷本体就是: 推断路径原来
#: 直接吃这份默认, 于是一台「认不出」的卡也被宣称支持共享内存 / warp shuffle / CUDA C。
#: 列表从 dataclass 现场算, 不在脚本里抄 —— 抄的那份会漂。
DEFAULTED_TRUE = tuple(
    f.name for f in dataclasses.fields(HardwareCaps)
    if f.name.startswith("has_") and f.default is True)


def _table_hit_claims_unchanged(info) -> bool:
    """生产路径: 真机字典命中查表时, caps 必须与表里那一条**逐字段**相等"""
    caps = detect_hardware_caps(dict(info))
    entry = _KNOWN_HARDWARE["h20"]
    a, b = caps.to_dict(), entry.to_dict()
    diff = {k: (a[k], b[k]) for k in a if a[k] != b[k]}
    if diff:
        print(f"  ❌ 表命中的 caps 与表不等: {diff}")
        return False
    if not all(getattr(caps, f) for f in DEFAULTED_TRUE):
        print(f"  ❌ H20 上 {DEFAULTED_TRUE} 有不是 True 的")
        return False
    return True


def _unrecognized_claims_nothing(info) -> bool:
    """抹掉器件身份 = 一台认不出的卡: 六项承诺全收, DSL 空, 但**探测到的 SM 数要留下**"""
    caps = detect_hardware_caps({"compute_units": info.get("compute_units")})
    lied = [f for f in DEFAULTED_TRUE if getattr(caps, f)]
    bad = []
    if caps.vendor != "unknown":
        bad.append(f"vendor={caps.vendor!r}")
    if lied:
        bad.append(f"仍然宣称 {lied}")
    if caps.supported_dsls:
        bad.append(f"仍然宣称支持 DSL {sorted(caps.supported_dsls)}")
    if bad:
        print(f"  ❌ 认不出的卡拿到了承诺: {'; '.join(bad)}")
        return False
    # 承诺收回不等于读数收回: 78 个计算单元这一件事是探测来的, 与「这是什么卡」无关
    return caps.sm_count == info.get("compute_units")


def _cuda_branch_still_claims(info) -> bool:
    """反向对照: 只补一句 `device="cuda"` (知道它是 CUDA 设备) 就该把 ISA 保证的收回来。

    没有这一条, 上面那条红也可能是「我把函数改死了」而不是「口径变了」。
    """
    caps = detect_hardware_caps({"compute_units": info.get("compute_units"),
                                 "device": "cuda"})
    missing = [f for f in DEFAULTED_TRUE if not getattr(caps, f)]
    if missing:
        print(f"  ❌ 已知是 CUDA 设备却没宣称 {missing}")
        return False
    return "cuda_c" in caps.supported_dsls


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
    ssh_calls = []
    _real_ssh_exec = ex._ssh_exec

    def _counting(cmd, *a, **k):
        ssh_calls.append(cmd)
        return _real_ssh_exec(cmd, *a, **k)

    ex._ssh_exec = _counting                    # 只数往返次数, 不改行为
    pin = ex._ssh_exec(f"cat {ex.work_dir}/.python-version").strip().splitlines()[-1]
    interp = ex._ssh_exec(
        "python3 -c 'import sys;print(\"%d.%d\"%sys.version_info[:2])'"
    ).strip().splitlines()[-1]
    print(f"  远端 .python-version={pin!r}  远端解释器={interp!r}")

    # 检查 7 (#52): 报表读的卡号必须与**远端自报**的卡号同源, 且读它不许再花一次往返
    from benchmarks.drivers.harness_pass_rate import device_record
    before = len(ssh_calls)
    actual = ex.selected_gpu
    no_round_trip = len(ssh_calls) == before
    # 7c 的冲突要由判据**自己造出来**: 上一版注入的是硬编码 "2", 而实卡是自动选的,
    # 于是这条绿不绿取决于那一次恰好选到哪张卡 (2026-09-29 真机就撞上: 这次选中的
    # 正好是 GPU 2, 判据立刻假红)。注入一个必然不等于实卡的号。
    other = next(str(i) for i in range(8) if str(i) != str(actual))
    rec = device_record(ex, env={"OPERATOR_AGENT_GPU": other})
    print(f"  执行侧 selected_gpu={actual}  远端自报={probe.get('selected_gpu')}  "
          f"报表 state={rec['state']!r} conflicts={rec['conflicts']} "
          f"(注入的请求卡号={other})")

    checks = [
        ("1 探测值与 nvidia-smi 的独立问法一致 (器件名 + 显存)",
         str(info.get("name")).lower() in smi_name.lower()
         or smi_name.lower() in str(info.get("name")).lower()),
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
        ("7 报表读的卡号与远端自报的卡号同源 (#52)",
         ex.selected_gpu is not None and ex.selected_gpu == probe.get("selected_gpu")),
        ("7b 读 selected_gpu 不再花一次往返", no_round_trip),
        ("7c 请求的卡号与实卡不同时冲突必须响, 而 gpu 仍是实卡",
         rec["conflicts"] is True and rec["gpu"] == actual
         and rec["requested_env"] == other),
        # 检查 8 (§3.4 第 4 处): 已知幅度的信号注入 —— 把**真机字典**里的 SM 数改成 99,
        # 派生出的 caps 必须跟着变 99。旧写法读的是 `multiprocessor_count` (字典里没这个键),
        # 于是会返回表里的 78 ⇒ 这一条在旧代码上必然红, 而真机数值恰好 78=78 本身看不出区别。
        ("8 注入 SM 数=99 后 caps 跟着变 (探测值没被字段名丢掉)",
         detect_hardware_caps(dict(info, compute_units=99)).sm_count == 99),
        ("8b 未注入时真机读数不变 (78, 与载荷逐字相等)",
         detect_hardware_caps(dict(info)).sm_count == 78 == probe.get("multiprocessor_count")),
        # 检查 9 (§八 第 3 条): 同一份**真机字典**的三种问法。旧写法在「去掉器件名」那一问
        # 上会照旧宣称六个 has_* 全 True + 支持 CUDA C (吃 dataclass 的宽松默认), 而它此时
        # 对这台卡一无所知。表命中那一问必须**全字段**不变 —— 那是生产路径。
        ("9 真机字典原样 (表命中) 时 caps 与表逐字段相等, 六个 has_* 仍全 True (生产路径没动)",
         _table_hit_claims_unchanged(info)),
        ("9b 同一份字典抹掉器件身份 (= 一台认不出的卡) 后六个 has_* 全 False 且不宣称 CUDA C",
         _unrecognized_claims_nothing(info)),
        ("9c 只补一句 device=cuda 就把 ISA 保证的那六项收回来 (修复不是一律 False)",
         _cuda_branch_still_claims(info)),
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
