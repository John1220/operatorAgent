"""「静态反作弊的白名单不许被任务字段改写」那张变异表的**可执行**形态

靶子是 `tests/unit/test_anticheat_bypass.py`（白名单那一组 + 运行时屏蔽的结构守卫）、
`tests/unit/test_lib_fallback_producer.py` 与 `tests/unit/test_m1_hardening.py`（运行时盾的行为）。

为什么单独一张表: CHECKLOG I80 的硬伤一是**可被非人化输入关掉的安全门** ——
`verifier` 把 `task.target.hardware` 传给 `anticheat.static_audit`, 而白名单当时是
子串匹配 (名单里含 `dc/ls/ft/rv`), 于是一个任务声明 `hardware="Ascend910"` 就能让
静态审计判断零违规。修法 = 白名单的输入换成**执行侧声明** (`VENDOR_TORCH_OK`),
静态与运行时两侧读同一个声明。

这张表要证明的是:
  · 那条通道真的没了（把任务字段接回去 / 把 `hardware=` 参数加回来 ⇒ 必红）;
  · 白名单本身还在（恒开 ⇒ 必红, 说明「恒开」不是安全的方向）;
  · 运行时那一侧仍然恒开（把 shield 的声明传 True / 让 shield 恒不装 ⇒ 必红）。

witness 一栏按 doc 57 §二 第 1 条分两类:
  · behavioral=True —— 有**具体输入或条件**让这次改动能被观察到;
  · behavioral=False —— 今天在行为层与原文不可区分, 拦住它的只可能是结构守卫。

跑: python3 -B -m benchmarks.mutation_calibrate --spec benchmarks/mut_specs/vendor_whitelist_not_from_task.py
"""
import sys
from os.path import abspath, dirname

_ROOT = dirname(dirname(dirname(abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmarks.calibration.mutation_calibrate import Row  # noqa: E402

FILES = {
    "ver": "operator_agent/verifier.py",
    "ac": "operator_agent/anticheat.py",
    "mgpu": "operator_agent/mock_gpu.py",
}

TARGETS = [
    "tests/unit/test_anticheat_bypass.py",
    "tests/unit/test_lib_fallback_producer.py",
    "tests/unit/test_m1_hardening.py",
]

ROWS = [
    Row("M1 生产调用点又去读任务字段 (洞的原形)", "ver",
        "                             vendor_torch_ok=VENDOR_TORCH_OK)",
        "                             vendor_torch_ok=bool(task.target.hardware))",
        "**具体输入**: 任务 `target.hardware=\"Ascend910\"` + 一份 `torch.matmul` 候选 ⇒ "
        "静态审计零违规、L2 放行 ⇒ "
        "`TestVendorTorchWhitelist::test_the_verifier_never_reads_the_task_hardware` 红; "
        "同一条也把源码守卫那半拉下水 (它要求调用点写的是 `VENDOR_TORCH_OK`)"),
    Row("M2 白名单恒开 (不需要任何声明)", "ac",
        "    whitelisted = bool(vendor_torch_ok)",
        "    whitelisted = True",
        "**具体输入**: 不带声明调一次 `static_audit(torch_code, \"python\")` ⇒ "
        "`test_the_task_field_channel_is_gone` 最后那句「没有声明却放行」红; "
        "同时 `test_the_verifier_never_reads_the_task_hardware` 也红 (形同永远白名单)"),
    Row("M3 `static_audit` 把 `hardware` 参数加回来", "ac",
        "def static_audit(code: str, dsl: str = \"python\",\n"
        "                 vendor_torch_ok: bool = False) -> AuditVerdict:",
        "def static_audit(code: str, dsl: str = \"python\",\n"
        "                 hardware: Optional[str] = None,\n"
        "                 vendor_torch_ok: bool = False) -> AuditVerdict:",
        "**具体输入**: `static_audit(code, \"python\", hardware=\"Ascend910\")` 不再抛 "
        "`TypeError` ⇒ `test_the_task_field_channel_is_gone` 的签名断言红 "
        "(判 API 面而不是判某一次行为: 行为用例证明不了「没有别的传法」)"),
    Row("M4 运行时盾恒不装 (白名单短路)", "ac",
        "    # ── 厂商卡白名单: 同一个声明, 静态与运行时两侧一致 ──\n"
        "    if vendor_torch_ok:\n"
        "        yield\n"
        "        return",
        "    # ── 厂商卡白名单: 同一个声明, 静态与运行时两侧一致 ──\n"
        "    if True:\n"
        "        yield\n"
        "        return",
        "**具体输入**: `with shield_torch():` 里调一次 `torch.matmul` ⇒ 不再抛 "
        "`CheatDetectedError` ⇒ `tests/unit/test_lib_fallback_producer.py` 与 "
        "`tests/unit/test_m1_hardening.py` 里那几条 `assertRaises(CheatDetectedError)` 红"),
    Row("M5 生产调用点给 shield 传了声明 (python 执行路径)", "mgpu",
        "            with shield_torch():\n                local_ns = {}",
        "            with shield_torch(vendor_torch_ok=True):\n                local_ns = {}",
        "**具体输入**: 一份在 kernel 里 `torch.matmul` 的候选走 MockGPU 的 python 路径 "
        "⇒ 运行时盾不再拦它 ⇒ `test_mock_gpu_execute_produces_metrics` 与 "
        "`test_verifier_l2_runtime_shield_hit_metrics` 红 (实测 3 条红, 另有一条是 AST 守卫)"),
    Row("M6 生产调用点给 shield 传了声明 (CUDA 执行路径 —— 本机走不到)", "mgpu",
        "            with shield_torch():\n                exec(kernel_code, ns)",
        "            with shield_torch(vendor_torch_ok=True):\n                exec(kernel_code, ns)",
        "**具体输入**: 没有 —— 这一处在 `_execute_triton_cuda` 里, 而控制机上没有 CUDA "
        "⇒ 那条路径走不到, 行为用例**全绿** (实测 57 passed); 拦住它的只有 "
        "`test_every_production_shield_call_is_bare` 那条 AST 断言。"
        "这正是结构守卫存在的理由: 「运行时那半在生产路径上恒开」不能只在能跑的那条路上成立",
        behavioral=False),
]