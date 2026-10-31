"""测量有效性判定 (docs/research/00 缺口 1, 设计见 docs/research/01 第五节)

L3 的数字有三种「不可信」, 而它们都不该被当成「性能不达标」:

1. **launch 地板支配** —— 实测时间几乎全是 kernel launch 开销, 描述的是驱动派发
   而不是核的质量。实测: layernorm 1024×768 读写 3.15 MB, roofline 下界 0.86 µs,
   而 launch 地板 18.69 µs → 地板占比 95.6%。那一轮报出的「0.65x」「带宽利用率
   5.4%」「memory-bound 需优化」全部是在描述 launch 开销。决定性旁证是
   torch_compile 反而比 eager 慢 2.5× —— 编译开销在这个尺寸根本摊不平。
2. **基准被污染** —— 测量窗口内 GPU 上有别的进程在跑, 加速比不可比。证据是测量
   **前**的 util 快照 (且须排掉自己上一轮的残留, 见 docs/research/33)。最慢样本/
   中位样本本来是想补的第二条证据 (窗口**中途**的抢占, util 快照看不见), 但实测
   它被样本数系统性抬高, 目前只作诊断。
3. **样本自身在抖** —— 30 个样本的 iqr/median 大到超过被测差异的量级, median
   看着权威其实是噪声。这条以前完全没检查: runner 跑了 n 个样本, 只报 min/median/
   avg/max, 从不看它们散不散 (决策与出处: docs/research/20 §3.1 D1)。

上面三条问的都是「**测得准不准**」。2026-09-28 又补了两条问「**测的是不是这个东西**」
(doc 49 §3.5 第 1/2/3 处, 那一节记的失效方式叫「算了没人读」):

4. **出处不对** —— 数字可能压根不是测出来的: `MockGPU` 无 CUDA 时按硬件规格**估**
   (`"mode": "simulated"`), `LocalExecutor` 在控制机 CPU 上真测 (那不是加速卡的成绩)。
   两者以前都在产物里与真机读数**权威性等价**。判据走 `backend_caps` 的声明而不是
   字符串匹配 `device`/`name`: 任务在 `target.required_caps` 里声明它要什么, 执行侧
   声明它有什么, 缺就是 UNINFORMATIVE 且 L3 不许 `passed=True`。载荷自报的
   `mode`/`device` 只用作**与声明对照**的第二手证据 (两者打架说明连声明都不可信)。
5. **环境漂移** —— 器件名/torch/triton 版本/执行侧身份与上一次运行不一致时, 这一轮的
   加速比与上一轮不可比。按本项目「观测先于判定」的处置: **只降到 DEGRADED**
   (不禁用配方、不计停滞), 不据此判核失败, 也没有阈值 —— 一份「变了哪些键」的清单
   还没有分布支撑, 现在定阈值就是编一个门。

`VerificationResult.passed` 只有两态, 于是前三种情况都被记成失败, 再被 Conductor
拿去禁用配方、累计停滞轮次。已记录的后果: matmul 那轮触发污染 (测量前 util=14%)
→ gflops 被压低 → 判「未改善」→ 把**实际有效**的配方拉黑; 而
`max_stagnant_rounds=2`, 两次误判就终止整个搜索。这台机器上恒有 7 个其他进程,
不是边缘情况。

本模块只做判定: 不含硬件数据 (launch 地板由调用方从 hardware_caps 取), 不做 I/O,
不依赖 torch。定位与 `tolerance.py` 相同 —— 单一权威判据 + 可离线单测。

上面五条件之外, 这里还住着一族问「**判据本身用的是哪把尺**」的规则 (缺口 1 的另一半,
#87): `GATE_PARAM_KEYS` / `single_ruler` (一场之内几组输入之间) 与 `l2_ruler_reading` /
`cross_round_ruler` / `ruler_note` (一次运行之内几轮之间)。它们放在这里而不是留在
`verifier.py` 里, 是因为读者有两个 (`conductor` 每轮那句 `[Ruler]` 与端到端报表那一栏),
而两个读者必须读同一个集合 —— 与上面 `CORRECTNESS_LEVELS` 那条是同一个理由。
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from ..core.backend_caps import Cap
from ..core.schemas import VerificationLevel


class Validity(str, Enum):
    """测量有效性的三个档位 (docs/research/01 表, 第五节第 2 条)"""

    VALID = "valid"                    # 可参与全部闭环决策
    DEGRADED = "degraded"              # 判定可用, 但不得驱动配方禁用/停滞计数
    UNINFORMATIVE = "uninformative"    # 只作诊断信息呈现, 不参与任何闭环决策


class Verdict(str, Enum):
    """两轮性能之间的四态判决 (吸纳自测量方法论一系, docs/research/20 第三节 D4)

    二元的「改善 / 没改善」把两件不同的事压成一件:
      - 真的更慢 (有证据说明这轮的配方有害)
      - 落在噪声里 (没有证据, 只是这次没测出差别)
    前者才可以禁用配方 —— 那是**永久**决策; 后者只该累计停滞轮次。原来的写法是
    `current > best` 的严格比较, 于是「完全打平」和「差 0.0001%」都会去禁用配方。
    """

    FAST = "faster"            # 超出带宽的改善
    SAME = "same"              # 落在带宽内: 计停滞, 但不据此禁用配方
    SLOW = "slower"            # 超出带宽的退步: 唯一可以禁用配方的一态
    AMBIGUOUS = "ambiguous"    # 证据不足 (没有可比前值, 或测量本身不可信)


# 判决带宽 = 3 × 实测的「同一份代码连跑多次的加速比散布」。
#
# 实测值来自 H20 GPU 2、无注入、每档连跑 8 次 L3 (docs/research/34 第四节表):
#   valid (matmul 4096³)         跨次 iqr 0.02%
#   degraded (layernorm 16384×6144)   0.13%
#   uninformative (layernorm 1024×768) 0.66% (范围宽到 2.1%)
#
# 外部 (SOL-ExecBench 一系) 的 0.5%/2% 是锁频专用机上的值, 不能照抄: 0.5% 在 valid
# 档是真实噪声底的 25 倍 (会把真的小改善判成「没变」), 在 uninformative 档又比实测
# 范围 2.1% 还窄 (一切都落 SAME/AMBIGUOUS) —— 这正是 doc 20 当初暂缓 D4 的理由,
# 现在有了本机的数就按档分开设。
NOISE_BAND = {
    Validity.VALID: 0.001,
    Validity.DEGRADED: 0.004,
    Validity.UNINFORMATIVE: 0.02,
}

# 绑定轴上达到 参考天花板的这个比例, 就认为「再优化也不会有 measurable 收益」。
# 缺口 4 的正解在这里: 判据该看**离上界还有多远**, 而不是继续拿一个标量比大小。
# 0.90 这个数以前写的理由是「真机上 matmul 4096³ fp16 实测 101% GFLOPS 利用率,
# 峰值表本身有 ±5% 余量」—— 那条**已经不成立了** (2026-09-26 核实): 101% 不是余量,
# 是分母用错了对象 (拿 cuBLAS 8192³ 的一次读数当上界)。分母换成 dense 规格 148 TFLOPS
# 后同一个核真机复测 **94.7%**, 于是 90 的含义从「比库实现的成绩还差一点」变成
# 「离规格上限还差 10%」。
#
# ⚠️ 两条轴的可达性因此**不对称**, 别把它当一个通用常数读 (2026-09-26 真机):
#   - compute 轴: 本机实测最好的 fp16 核 140.2 TFLOPS = 规格的 94.7%, 出口打得开,
#     但只剩 4.7% 余量 (对着旧的 138.2 分母是「随便就打满」)。
#   - memory 轴: 本机 D2D 拷贝实测最好 3357 GB/s = 4000 规格的 **84%** (带宽阶梯见
#     docs/research/42), 低于 90 —— 这条轴上出口**永不打开**, memory-bound 核会把
#     轮次跑满 (由停滞出口兜底)。改之前分母是 3680, 同一个读数 91.2% 是能出口的。
#   这是这次吸纳**故意付出的代价**: 分母偏小 (旧 138.2 更偏小) 会把还能优化的核**提前**
#   判成到顶, 而多烧几轮是可恢复的。正解是按 shape 查这台卡的可达上界 (缺口 2 的持久化
#   记录), 而不是把阈值调回 84% 去迁就一个更好的分母。
# ⚠️ 参照 KernelBench-Hard 的一句话: 人写的 Hopper 核常见只到峰值的 60~85% —— 按
# 这个量级, 90 是一个**偏乐观**的出口, 打不开它才是常态 (docs/research/42)。
REFERENCE_CEILING_PCT = 90.0


# ── 「正确性已验证」的单点定义 ────────────────────────────────────────────────
# 这个集合现在有**两个读者**, 而且它们读的必须是同一件事:
#   · `conductor.correctness_verified` —— 修订起点退回哪一版 (arXiv:2609.35140v1 行 214);
#   · `benchmarks/harness_pass_rate`   —— 端到端报表里「摘掉真机那一栏」的通过率 (#77,
#     外部一手 arXiv:2609.26777v1 行 99-101 的双记分口径)。
# 两处各写一遍 `("L1", "L2")` 就是两套口径 —— 那是本仓反复登记过的那一族 (CHECKLOG I22/H4)。
#
# L3 **不在**这里面: 一个「正确但不够快」的核是被验证过的, 只是没到性能门槛。
CORRECTNESS_LEVELS: tuple = (VerificationLevel.L1_COMPILE.value,
                             VerificationLevel.L2_FUNCTION.value)


def correctness_verified_levels(levels: dict) -> bool:
    """`levels` 是 `{VerificationLevel.value: passed}` —— 血缘与报表里的那种形状

    「到场」和「通过」都要查: `Verifier.verify()` 是 fail-fast, L1 挂时 L2 那一格**根本不存在**,
    而 `all(...)` 在这种残缺列表上照样会给 True —— 缺证据不是通过 (doc 49 §八 那一族)。
    所以这里逐项 `is True`: 值只可能来自一个真的布尔, 缺键与 False 都不算数。
    """
    return all(levels.get(k) is True for k in CORRECTNESS_LEVELS)


# ── 「用的是哪把尺」的单点定义 (缺口 1 / #87 第 2 片) ─────────────────────────
# 外部一手 (arXiv:2609.35140v1 行 150, 逐字):
#     "𝒮 is the **immutable** evaluation specification ... It keeps correctness and
#      performance criteria fixed across attempts."
# 那句话问的不是「算得对不对」, 而是**整条判据里是不是同一把尺**。第 1 片 (commit 2b6a390)
# 管的是「一场之内、几组输入之间」; 本片管「一次运行之内、几轮之间」。
#
# 这三键的名字与「同值才配发布单值」那条规则**只在这里定义一次**。它们原先住在
# `verifier.GATE_PARAM_KEYS` / `verifier._single_ruler`, 而跨轮那半必须用同一个集合 ——
# 两处各写一遍就是两套口径 (CHECKLOG I22/H4 那一族), 而第 1 片立的规矩是「逐组收集后统一
# 判定」, 搬到这里才配被轮级与运行级两个读者共用。
GATE_PARAM_KEYS: tuple = ("atol", "rtol", "required_matched_ratio")

# ── 性能那一半 (#87 第 3 片) ─────────────────────────────────────────────────
# 上面那三键只覆盖 correctness。外部那句话问的是**整个** 𝒮 ("It keeps correctness and
# performance criteria fixed across attempts"), 所以「一次运行里换没换尺」这个问题今天只
# 问了一半: L3 的通过门 (`speedup >= min_speedup`) 跨轮是不是同一把, 之前**没有读数**。
#
# 这三个键都是 L3 判决真正读到的量, 而且都是**今天就会漂**的那种漂:
#   * `min_speedup_required` —— 门槛本身。它来自 `task.constraints`, 一场运行里通常不动,
#     但配方热重载 / 意图解析重跑都会动它。
#   * `perf_criterion` —— **判的是哪个量**。`verifier._verify_performance` 有三条比对路,
#     前两条比 `speedup >= min_speedup`, 第三条 (legacy 无基线) 比的是
#     `gflops >= target_gflops * min_speedup` —— 同一个 `min_speedup=1.5` 在前者是
#     「快 1.5 倍」的地板, 在后者是「把绝对目标乘 1.5」的系数。所以只看数值会把一次
#     **换了量纲**的漂移读成「尺没变」。这一键就是为了让那种情况变成「尺变了」。
#   * `bound_type` —— roofline 的绑定轴。天花板出口那条 (`reference_ceiling_reached`) 比的是
#     「这条轴上达到上界的百分比」, 轴从 compute 翻到 memory 时, 同一个 90% 意思是两件事
#     (本机两条轴的可达性实测不对称: compute 轴最好 94.7% 打得开口, memory 轴最好 84%
#     永不开, 见上面 `REFERENCE_CEILING_PCT` 那段)。
#: 绑定轴那个键的名字与它的占位值。**这里定义一次**, 因为「`unknown` 不算一个读数」这条
#: 规则必须和键名待在一起 —— 抄两份字符串, 将来改键名时那条规则会安静地失去作用。
#: ⚠️ 本模块因此成为 `bound_type` 的一个**读者** (tests/unit/test_bound_axis_gate.py 那张读者
#: 清单会因此报红, 那是它该做的); 这里拿它做什么 = **只把它当尺的身份比较, 不据此选轴**,
#: 而 `"unknown"` 正是它必须认识的那个第三态 (见 `l3_ruler_reading`)。
BOUND_AXIS_KEY = "bound_type"
BOUND_AXIS_UNKNOWN = "unknown"

PERF_GATE_PARAM_KEYS: tuple = ("min_speedup_required", "perf_criterion", BOUND_AXIS_KEY)

#: 三条比对路的**身份**。产它的 (`verifier`) 与读它的 (跨轮判定、报表、用例) 都从这里取
#: 字符串, 否则就是两套口径 (CHECKLOG I22/H4 那一族)。
PERF_CRITERION_SPEEDUP = "speedup_vs_baseline>=min_speedup"
PERF_CRITERION_ABSOLUTE_GFLOPS = "gflops>=target_gflops*min_speedup"


#: 每轮的尺读数写进 `iteration_history` 时用的键。**产它的 (conductor) 与读它的
#: (conductor 的跨轮判定、端到端报表) 都从这里取名字**, 免得两处字符串飘开之后就只剩
#: 「报表那一栏永远是空的」这种静默失效。
LINEAGE_RULER_KEY = "l2_ruler"

#: 性能那一半在血缘里的兄弟键。**产它的 (conductor) 与读它的 (跨轮判定、端到端报表) 同样
#: 都从这里取名字** —— 上一格断在「血缘一直在写但没人读」上, 这一格不能重犯 (doc 43 第三节)。
LINEAGE_PERF_RULER_KEY = "l3_ruler"

#: 两份读数各自印出来的标签。**只放名字**: 键集合从各自的 `*_GATE_PARAM_KEYS` 读, 抄进
#: 标签字符串里就等于加一个键要有第二处记着改 (I20 那一族)。
RULER_LABEL_CORRECTNESS = "正确性那把尺"
RULER_LABEL_PERFORMANCE = "性能那把尺"

# ── 执行侧那一格: 「没听到回答」不等于「对方答了失败」 (缺口 1 的执行侧, #102 第 1 片) ──
#: 一手依据: Limbo `2609.29095v1` 行 226-232 —— Definition 1 (观测等价) 之后那句
#: "A timeout is observation-equivalent under absent, committed (response lost) and late
#: (request still in flight). A recovery policy therefore cannot condition on the outcome
#: state directly"。**一次超时在「没执行 / 执行了但回执丢了 / 还在飞」三种世界里对客户端
#: 完全同形**, 所以客户端只有两条路: 把它单独记成一格「不知道」, 或者选一个在所有等价世界
#: 里都安全的动作。我方此前两条都没走 —— `SSHExecutor.execute` 的 `except Exception` 把它
#: 收成 `success=False`, `Verifier._verify_single_set` 再收成 `message="执行失败: …"`,
#: 于是「未知」与「跑完了但输出不对」在判决层同形, Fixer 会对着前者给候选开处方 (doc 60 §四)。
TRANSPORT_KEY = "transport_outcome"
TRANSPORT_ANSWERED = "answered"          #: 听到了回答 —— 哪怕回答的内容是「失败」
TRANSPORT_UNKNOWN = "outcome_unknown"    #: 没听到回答 —— 远端那次执行可能已经发生
TRANSPORT_STATES: tuple = (TRANSPORT_ANSWERED, TRANSPORT_UNKNOWN)

#: SSH 保留给「ssh 自己失败」的退出码 (远端命令的返回码不会被它覆盖): 255 意味着命令
#: 可能根本没跑, 也可能跑完了但连接断了 —— 正落在上面那三种等价世界里。
SSH_TRANSPORT_EXIT_CODE = 255

#: 只列「**确定**没听到回答」的那一族。方向刻意保守: 误报一次 unknown 只是多问一句,
#: 漏报一次是把「这张卡可能已经跑过一遍」擦掉 —— 而 #102 第 2/3 片要拿它做幂等清点。
_NO_RESPONSE_EXCEPTIONS: tuple = (
    "SessionError",           # timeout/EOF/desync/broken resident RPC: no attributable response
    "TimeoutExpired",         # `subprocess.run(timeout=…)` 打的正是这一格
    "timeout",                # socket.timeout (py3.10+ 起是 TimeoutError 的别名)
    "TimeoutError",
    "ConnectionError", "ConnectionResetError", "ConnectionAbortedError",
    "BrokenPipeError", "BrokenResourceError",
    "RemoteDisconnected", "IncompleteRead",   # 回答断在半路 = 没听到完整的回答
    "URLError",               # urllib: 连不上与中途断都算, 请求也可能已经发出去了
)


def transport_outcome(exc, remote_rc=None) -> str:
    """把一次远程调用的异常归到「听到回答」/「没听到回答」这一格。

    按 **MRO 上的类名**匹配而不是 `isinstance`: 本模块的纪律是不做 I/O、不 import 传输层
    (见模块头), 而 `subprocess` / `http.client` / `socket` 全都是传输层。名字匹配的代价是
    「别人仿了一个同名异常也会被判 unknown」—— 这里接受, 因为 unknown 是**不报警**的那一侧。

    `remote_rc` 由调用方从自己的传输层带上 (SSH 的 255): 那个码本身不是异常类型的区分,
    而是「命令没跑 / 跑了但回执丢了」的区分。
    """
    if remote_rc == SSH_TRANSPORT_EXIT_CODE:
        return TRANSPORT_UNKNOWN
    for klass in type(exc).__mro__:
        if klass.__name__ in _NO_RESPONSE_EXCEPTIONS:
            return TRANSPORT_UNKNOWN
    return TRANSPORT_ANSWERED


def transport_outcome_of_result(result) -> str:
    """从一份**执行结果对象**里读那一格。

    没有这一格的结果按 `answered` 读, 这不是兜一个像样的默认值, 而是一条推理: 手里既然
    有一份 result 对象, 传输层就**给回了东西** —— 它落在 Limbo `2609.29095v1` 行 220-225
    那三种等价世界之外, 因为「没答话」那一侧从来不表现为一份结果, 只表现为异常。
    `MockGPU` 的 `ExecutionResult` 与测试里的假后端都不带这个字段, 而它们能走到这里只
    有一种方式 (本机执行失败) —— 直接 `result.transport_outcome` 会把「候选算错了」
    这个判决路径换成一个 `AttributeError`。
    """
    return getattr(result, TRANSPORT_KEY, TRANSPORT_ANSWERED)


# ── 远端异常的**文本**过落盘边界之前的一道掩码 (#102 第 2 片) ──
#
# 与 `transport_outcome` 放在同一个模块的理由是一样的: 那是异常的结构侧、这是它的文本侧,
# 而**写它的执行侧与读它的报表/环境自检必须共用一套规矩** (I22 那一族) —— 各写一份就会长成
# 「有一路在脱敏、另一路没有」，而那恰好是本仓最难发现的一类失效。
#
# ⚠️ 为什么 `str(e)` 不能直接进产物: `subprocess` 那两类异常 (`TimeoutExpired` /
# `CalledProcessError`) 的 `str()` 里带着**整条 argv** —— ssh/scp 那条命令含 `-p <端口>`、
# `-i <密钥路径>` 与 `<用户>@<主机>`。这句文本今天有三个读者: 落盘的 `result.json`、发给外部
# LLM 的 Fixer prompt、以及**跨运行持久化**的失败签名 (`FixerAgent` 从错误文本里取一段做签名,
# 见 `fixer.py:209`)。`_truncate_error` 保头保尾也救不了 —— argv 的秘密正好在**头部**。
# 现在没出事只因为 `_build_ssh_base` 默认把 port/key/user 留给 `~/.ssh/config` 解析:
# 那是「缺席」, **不是一道掩码** (doc 60 §四 第 2 条原文就是这个意思)。
_MASKED = "[已隐去]"
# 「旗标 + 值」这一对要先摘, 再摘 `user@host` 与 IPv4: 反过来 `ssh: connect to host 1.2.3.4
# port 2222` 会先被 IPv4 那条吃掉一半而留下裸端口, 而 `-p 2222` 会变成留在文本里的 `2222`。
# 最后那条 `port <数字>` 不是重复: 它是 ssh **自己**写的措辞 (不是 argv 的 `-p`),
# 第一版就是只摘了 IPv4, 于是真机上那句 `connect to host [已隐去] port 2222` 里的端口活了下来。
_ARGV_SECRET_FLAG = re.compile(r"(?i)\s(-i|-p|-l|-F|--identity-file|--port)\s+\S+")
_USER_AT_HOST = re.compile(r"\b[\w.+-]+@[\w.-]+\b")
_IPV4_WITH_PORT = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d{1,5})?\b")
_PORT_WORD = re.compile(r"(?i)\b(port)\s+\d{1,5}\b")


def mask_transport_secrets(text: str) -> str:
    """把一段**传输层**文本里的地址类信息换成占位符; 只给远端那一路用。

    刻意只在远程出口调用: 候选代码自己的 traceback 里出现 `@` 或点分数字是常见的
    (`@triton.jit`、版本号), 对它做同样的替换会吃掉 Fixer 需要的诊断信息 —— 那一路
    该原样进 prompt (#102 第 2 片的范围就是「远端写副作用的那几个出口」)。
    """
    if not text:
        return text
    # 保留**旗标**、只摘它的**值**: `-p`/`-i` 本身是「这一路带过端口/密钥」的线索,
    # 而值才是秘密。整对吃掉会让掩码顺手把诊断能力一起掩掉 (用例里那条 M3 的反向对照)。
    out = _ARGV_SECRET_FLAG.sub(r" \1 " + _MASKED, text)
    out = _USER_AT_HOST.sub(_MASKED, out)
    out = _IPV4_WITH_PORT.sub(_MASKED, out)
    return _PORT_WORD.sub(r"\1 " + _MASKED, out)


def _cmd_shape(cmd) -> str:
    """argv 的**形状**而不是内容: 程序名 + 参数个数。

    远程命令的内容在这里没有诊断价值 (载荷走 stdin、不在 argv 里, 而 argv 剩下的部分正是
    端口/密钥/`user@host`); 程序名才是「哪一步」的线索: `ssh` = 执行/探测, `scp` = 部署上传。
    """
    if isinstance(cmd, (list, tuple)):
        head = str(cmd[0]).rsplit("/", 1)[-1] if cmd else "?"
        return f"{head} +{max(len(cmd) - 1, 0)} 个参数"
    return mask_transport_secrets(str(cmd))[:40]


def safe_error_text(exc, with_class: bool = False) -> str:
    """一次远程调用失败时, 那句**可以**进产物 / LLM prompt / 失败签名的文本。

    带 `cmd` 的异常 (subprocess 那两类) 不再用 `str()` —— 整条 argv 就是它的 `str()`;
    改成「异常类名 + argv 的形状 + 超时秒数或返回码」。其余异常走 `mask_transport_secrets`,
    因为 ssh 自己写的 stderr (`connect to host <IP> port <N>`) 也在这条路上。

    `with_class=True` 给的是**只有一句 reason**的那些读者 (连通性探测的
    `Resolution.attempts`): 那里要分得开「DNS / 认证 / 超时」是哪一类, 而 `socket.gaierror`
    的原话里没有类名。默认关着是因为四个出口那边的文本已经自带出处
    (`SSH exec failed [rc=…]`), 前面再加一个 `RuntimeError:` 是噪声。
    """
    cmd = getattr(exc, "cmd", None)
    if cmd is not None:
        parts = [type(exc).__name__, _cmd_shape(cmd)]
        rc = getattr(exc, "returncode", None)
        timeout = getattr(exc, "timeout", None)
        if rc is not None:
            parts.append(f"rc={rc}")
        if timeout is not None:
            parts.append(f"超时 {timeout}s")
        return " ".join(parts)
    text = mask_transport_secrets(str(exc))
    if with_class and text and type(exc).__name__ not in text:
        return f"{type(exc).__name__}: {text}"
    return text


RULER_REPORTED = "reported"            #: 这一轮各组同一把尺, 值照抄
RULER_CONFLICT = "self_conflict"       #: 这一轮自己就不是同一把尺
RULER_UNREPORTED = "unreported"        #: 一个尺键都没报 (没走到 L2 / 没做参考比对)


def single_ruler(gates: list[dict]) -> tuple[dict, dict]:
    """只有「每一组参考比对都用同一把尺」时, 聚合值才配发布一个单值

    返回 `(可发布的单值, 冲突表)`。冲突表形如 `{键: {组下标: 那一组实际用的值}}`,
    读的人不必重跑就能看出是哪几组、哪一把尺不一致。

    旧写法是 `if not gate_params: gate_params = ...` —— 第一组赢, 而它下面聚合的是跨组的量
    (`max_error` 取 max、`matched_ratio` 取 min)。于是产物里会印着「atol=1e-3」, 而那个混合值
    其实是几把不同的尺各自判过之后再混起来的 —— 读它的人 (含 `benchmarks/probes/probe_l2_sampling_gate.py`,
    它正是拿这几个键核「绝对+相对」那条门的) 会把一个不存在的统一口径当成前提。

    冲突时那个键**直接不发布**, 而不是挑一个: 与 #64「分母不可信时 `bound_type` 不许替它判」
    同形。一组**缺**这个键同样算冲突 (不能假设它与别的组同值), 所以这里的判据是
    「每一组都在场且每一组同值」。

    ⚠️ 「每一组都在场」那半是本片真正的牙齿, 它**今天真能发生**而不需要谁出错: 远端那路的载荷
    是按 `verifier._REMOTE_L2_METRIC_KEYS` 逐键过滤 (`if k in remote_verify`) 透传上来的,
    所以部署在远端的 `run_kernel.py` 是旧版本、或某组走了缺字段的分支时, 就会出现「A 组报了
    `required_matched_ratio`、B 组没报」。
    """
    if not gates:
        return {}, {}
    single: dict = {}
    conflict: dict = {}
    for key in GATE_PARAM_KEYS:
        vals = {i: g[key] for i, g in enumerate(gates) if key in g}
        if not vals:
            # 没有任何一组报过这个键 ⇒ 既不发布, 也**不谎报冲突**。写漏这一支时冲突表里
            # 会出现 `{"rtol": {}}` 这种空证人, 而 message 会照着念「这几组用的不是同一把尺
            # ['rtol']」—— 拿一个谁都没报过的键去报警, 与本仓「缺证据不是否定证据」同族。
            continue
        if len(vals) == len(gates) and len({repr(v) for v in vals.values()}) == 1:
            single[key] = next(iter(vals.values()))
        else:
            conflict[key] = vals
    return single, conflict


def _ruler_reading(metrics, keys: tuple, *,
                   not_a_value: Optional[dict] = None) -> dict:
    """`l2_ruler_reading` / `l3_ruler_reading` 的共同实现 —— 口径只有一份。

    分成两个薄壳而不是给 `l2_ruler_reading` 加参数: 那两个名字的**键集合不同义** (一把是
    数值容差, 一把是性能门槛与判的是哪个量), 混在一个带默认值的签名上, 调用方漏传 keys 就
    会静默拿 L2 的键集合去读 L3 的 metrics —— 那正好制造本函数要避免的那种错。

    `not_a_value` 是 `{键: 一组「不是读数的值」}`。这一族最阴的一种错法是**两个都没有数被
    读成「两轮一致」**: `bound_type` 的 `"unknown"` 就是一个占位值 (`verifier._roofline_analysis`
    在既推不出脊点、也认不出算子名时报它), 连续两轮都 `unknown` 时跨轮判据会给该键 `same`,
    于是「这把尺说的是哪条轴」这件谁都不知道的事被印成已确认。所以这类值在收读数时就被
    剔掉 —— 该键随后进不了 `vals_by_key`, 与「这一轮没报」同处, 而不是与「报了且相同」同处。
    """
    m = metrics if isinstance(metrics, dict) else {}
    values = {k: m[k] for k in keys
              if k in m and m[k] not in (not_a_value or {}).get(k, ())}
    conflict = m.get("gate_params_conflict")
    if conflict:
        # 打架的键名也带上: 读者要能分清「atol 说不清但 rtol 说得清」和「全都说不清」
        return {"state": RULER_CONFLICT,
                "conflicted_keys": sorted(str(k) for k in conflict), **values}
    if values:
        return {"state": RULER_REPORTED, **values}
    return {"state": RULER_UNREPORTED}


def l2_ruler_reading(l2_metrics) -> dict:
    """把一轮 L2 的 metrics 收成「这一轮用的哪把尺」那一份读数, 供血缘与跨轮判定用

    三态, 每条都对应一种**不同的**下游该读出来的意思:
      * `reported` —— 这一轮各组同一把尺, 能报的键都报出来了;
      * `self_conflict` —— 第 1 片那道守卫在这一轮内部就报了警 (`gate_params_conflict`
        在场)。**这不等于「换了一把尺」**: 它是「这一轮自己说不清用了哪把」, 两件事的
        修法不同 (前者查部署, 后者查多组输入的口径);
      * `unreported` —— 一个尺键都没有: 那轮死在 L1/L2 之前, 或 L2 只验可执行性没做参考比对。

    「没有读数」与「读数说尺变了」必须分开 —— 后者是证据, 前者只是缺陷信号 (本仓 #53 那一族:
    把 `speedup=None` 印成「没有加速」)。
    """
    return _ruler_reading(l2_metrics, GATE_PARAM_KEYS)


def l3_ruler_reading(l3_metrics) -> dict:
    """一轮 L3 的「性能那一半用的是哪把尺」—— 与 `l2_ruler_reading` 同一个口径、另一组键。

    键集合见 `PERF_GATE_PARAM_KEYS`。这一侧**今天只有两态** (`reported` / `unreported`),
    第三态 `self_conflict` 结构上到不了: 那一态的意思是「一场之内的几组输入各用一把尺」,
    而 L3 按 `verifier._verify_performance` 的写法**只拿第一组输入做基准** (源码里那句注释:
    「多组输入套件时，性能基准使用第一组输入」), 没有跨组的东西可打架。
    这不是「暂时没实现」, 是这一级今天的形状 —— 所以跨轮那侧印 `0 轮内部就不一致` 时,
    那句是**真话**, 不是一格没接上的空位。

    ⚠️ **`bound_type` 的 `"unknown"` 不算一个读数**: 那条轴推不出来时 Verifier 报的就是这个
    占位值, 而连续两轮都 `unknown` 会被逐键判据读成「这一键各轮同值 ⇒ 尺没变」—— 那是把
    「谁都不知道是哪条轴」印成「已确认是同一把尺」, 与本仓 #53 / I10 那一族同形。所以这种
    值在收读数时就被剔掉, 该键随后进不了可比对, 与「这一轮没报」同处。

    ⚠️ 一个已知盲区, 与 #89 同一条纪律: 这一侧测得出「这一轮没报尺」, 测不出「这一轮
    走的其实是三条比对路之外的第四条」—— 因为 `perf_criterion` 是**各条路自己声明**的。
    新增一条比对路而忘了声明, 那一轮就退成 `unreported` (看得见), 但如果它**照抄了别人的
    字符串** 就看不见。所以那几个字符串住在本模块、由用例逐条钉住 (不放在 verifier 里就地
    写), 而 `tests/unit/test_perf_ruler_across_rounds.py` 里有一条 AST 守卫: `_verify_performance`
    里**凡报 `min_speedup_required` 的那个字面量**都必须同时报 `perf_criterion`, 且值必须是
    `PERF_CRITERION_*` 常量 (查字面量而不是查 `return` 是因为三条路都在 `metrics.update`
    里填名字, `return` 那一段里根本没有它 —— 第一版按 return 查, 五个 L3 出口一个都不命中)。
    """
    return _ruler_reading(l3_metrics, PERF_GATE_PARAM_KEYS,
                          not_a_value={BOUND_AXIS_KEY: (BOUND_AXIS_UNKNOWN,)})


def cross_round_ruler(iteration_history: list, *,
                      keys: tuple = GATE_PARAM_KEYS,
                      lineage_key: str = LINEAGE_RULER_KEY) -> dict:
    """一次运行之内, 各轮的数值门用的是不是同一把尺。

    `keys` / `lineage_key` 让同一套判据服务两半 (#87 第 3 片): 默认值是 correctness 那一半
    (`GATE_PARAM_KEYS` + `l2_ruler`), 传 `PERF_GATE_PARAM_KEYS` + `LINEAGE_PERF_RULER_KEY`
    就是性能那一半。**判据只有一份**是这个函数存在的理由 —— 复制一份去判 L3, 就会有两套
    「什么算同值」的口径 (本仓 I22/H4 那一族)。

    入参是 `artifact.iteration_history` (或报表里那份原样搬运的 `rounds`), 每轮取
    `lineage_key` 那一项。**没有那一项**与「有, 但是 `unreported`」一起算「这一轮
    没报」—— 前者是这次运行跑在写这个键之前的代码上 (续跑旧快照就会遇到), 血缘里没有原料,
    所以只能如实计入分母, 不能倒推成「没漂移」。

    判据是**逐键**的, 与第 1 片同形 (不因一个键打架就让别的键陪着不说话):
      * 某个键被 **≥2 轮**报过且值全同 ⇒ 该键 `same`;
      * 被 ≥2 轮报过且值不同 ⇒ 进 `ruler_changed`, 并带每轮的取值当证人;
      * 只被 1 轮报过 ⇒ 没有可比对象, 既不说同也不说变 —— 这类键进 `ruler_keys_partial`。

    `ruler_same` 因此是三态: `True` 只在「至少有一个键可比, 且可比键都不打架」时成立;
    `False` 是有证人; `None` 是**没有任何可比对** (少于两轮报过尺)。把 None 折成 True
    就是「拿缺证据当通过」, 折成 False 就是「拿缺证据当反证」, 两个方向都错过。
    """
    rounds = list(iteration_history or ())
    readings: list = []
    for idx, h in enumerate(rounds):
        if not isinstance(h, dict):
            continue
        reading = h.get(lineage_key)
        if not isinstance(reading, dict):
            continue
        label = h.get("round", idx + 1)
        readings.append((label, reading))

    states = [r.get("state") for _, r in readings]
    vals_by_key: dict = {}
    for key in keys:
        per_round = {label: r[key] for label, r in readings if key in r}
        if per_round:
            vals_by_key[key] = per_round
    comparable = {k: v for k, v in vals_by_key.items() if len(v) >= 2}
    changed = {k: v for k, v in comparable.items()
               if len({repr(x) for x in v.values()}) > 1}
    # 「报了几轮」按**覆盖最广的那个键**来算, 因为各轮报的键集合可以不同 (远端那路逐键过滤)。
    # 拿轮数当分母会把「这一轮压根没报 rtol」算成「rtol 上出现了不一致」的反面 —— 那是
    # 第 1 片那条 M2 (缺组当同值) 的跨轮版本。
    coverage = max((len(v) for v in vals_by_key.values()), default=0)

    return {
        "ruler_rounds_total": len(rounds),
        # 血缘里**带了尺读数**的轮数。为 0 时这一栏是「测不到」而不是「一致」——
        # 与 `_duplicate_rate` 那个 `duplicate_measurable` 同一个位置。
        "ruler_rounds_recorded": len(readings),
        "ruler_rounds_reported": states.count(RULER_REPORTED),
        "ruler_rounds_self_conflict": states.count(RULER_CONFLICT),
        "ruler_rounds_unreported": states.count(RULER_UNREPORTED),
        "ruler_key_coverage": coverage,
        "ruler_same": (None if not comparable else not changed),
        "ruler_changed": changed,
        # 顺序跟着 `GATE_PARAM_KEYS` 走 (不是字母序): `vals_by_key` 就是按那个顺序建的,
        # 而报表会把这份名单原样印出来 —— 名单的顺序也该是可预期的。
        "ruler_keys_partial": tuple(
            k for k, v in vals_by_key.items() if len(v) < coverage),
    }


def ruler_note(agg: dict, label: str = "", advice: str = "去查部署是否中途换过版本") -> str:
    """把 `cross_round_ruler` 的结果写成给人读的一行 —— **只报不判**

    `label` / `advice` 是给第 3 片那两半用的: 现在有两份这样的读数 (correctness 的
    `atol/rtol/ρ` 与 performance 的 `min_speedup/判的是哪个量/绑定轴`), 两行都印在同一份
    日志里。不带标签时「各轮的尺一致」这句**读不出说的是哪一把** —— 那正是本仓反复登记的
    那一族: 一条正确的观测被写成两句一模一样的话, 读者只能猜。
    而且那两半的**修法不同**: 数值尺换了几乎一定是部署中途换过版本, 性能尺里的 `bound_type`
    换了通常只是候选自己换了形状 (那是事实, 不是待查的漂移)。所以连那句建议都参数化,
    默认值与两片之前逐字相同 —— 既有读者 (与钉它们的用例、报表那一栏) 一个字节都不动。

    与 `verifier._oracle_note` 同定位: 这一句不改任何判决。它存在的理由是「跨轮换了尺」
    今天**没有阈值可判** —— 按本项目「观测先于判定」的处置 (同一位置上的另一条是环境漂移
    只降到 DEGRADED), 先把这件事在日志与报表里变得可见, 并让**分母**跟着一起出来。

    三个分支都要能单独被测出来, 所以措辞里不出现「正常」: `ruler_same is None` 说的是
    「无从比对」, 把它印成「各轮一致」就是拿缺证据当通过。
    """
    if not agg.get("ruler_rounds_recorded"):
        return ""
    same = agg.get("ruler_same")
    counts = (f"报了尺的轮 {agg['ruler_rounds_reported']}"
              f" (另有 {agg['ruler_rounds_self_conflict']} 轮内部就不一致、"
              f"{agg['ruler_rounds_unreported']} 轮没报; "
              f"共 {agg['ruler_rounds_total']} 轮)")
    if same is True:
        head = "各轮的尺一致"
    elif same is False:
        head = ("⚠️ 各轮的尺**不一致**: 变了的是 "
                f"{dict(agg['ruler_changed'])} —— 不据此判核失败, "
                f"{advice}")
    else:
        head = "各轮的尺**无从比对** (可比键不到两轮报过)"
    if agg.get("ruler_keys_partial") and same is not False:
        head += f" (只被部分轮报过: {list(agg['ruler_keys_partial'])})"
    return (f"{label}: " if label else "") + head + "; " + counts


# ── 跨组聚合: 只 fold「报了这个键」的那些组 ────────────────────────────────
#
# 上面那一族问的是「几组用的是不是同一把尺」(`single_ruler`), 这一族问的是另一件事:
# 「聚合出来的那个数底下是**几组**」。现场是 doc 57 §三 第 3 条, 由
# `benchmarks/probes/probe_l2_sampling_gate.py` 的臂 0b 打出来: 旧写法把三个键用
# `m.get("max_rel_error", 0.0)` / `m.get("bad_elements", 0)` / `m.get("matched_ratio", 1.0)`
# 兜底, 于是**从没算过这三个量的那条比对路径** (声明里没有 `Cap.FULL_OUTPUT_TENSOR` 的那条,
# 判的是纯绝对阈值) 在产物里变成了「无相对误差、零坏点、全部元素匹配」。三处细节让它比
# 看上去更贵:
#   ① 只发生在**通过**那一侧 (失败时提前 return 原始 metrics) ⇒ 被污染的正好是通过率的分母;
#   ② 兜出来的值与真值长得一样 (`matched_ratio=1.0` 是个完全合理的读数) ⇒ 从 `result.json`
#      读不出这两条路的区别;
#   ③ 「min over 几组」这件事本来就没写 —— 即使每组都报了, 读者也不知道那个 min 覆盖了几组。
# 规则因此是三条, 各对应一种已经犯过的读法 (缺证据当证据、把像样的默认值当读数、把分母压平)。
L2_AGGREGATIONS: tuple = (
    ("max_error", max),
    ("max_rel_error", max),
    ("matched_ratio", min),
    ("bad_elements", sum),
    ("compared_elements", sum),
    ("coverage", min),
)


def aggregate_l2_readings(sets: list[dict]) -> tuple[dict, dict]:
    """把逐组 L2 读数各按各的算子聚合成一份, 并把「这个键是从几组聚出来的」一起返回

    `(值, 出处)`; 出处形如 `{键: [报过的组数, 参与聚合的组数]}` (用 list 而不是 tuple ——
    这份东西要进 `result.json`)。三条规则:
      * 只 fold **报了这个键**的那些组, 缺席就是缺席, 不当 0 也不当 1.0;
      * 一组都没报的键**不发布** (写 0 会被读成「查过了, 零坏点」, 与本仓 #53/#81 那一族同形);
      * 出处照实上报: `matched_ratio` 从 1/3 组聚出来与从 3/3 组聚出来是两个不同的数,
        产物里得看得出来, 否则「min over what」又被压平了一层。
    """
    total = len(sets)
    if not total:
        # 一组都没进来时返回两份**空**东西: 此时「这些量 0/0 组算过」不是信息而是噪音,
        # 而调用方 (`verifier._verify_function`) 本来也不会进 `has_reference` 那一支
        return {}, {}
    values: dict = {}
    origin: dict = {}
    for key, op in L2_AGGREGATIONS:
        vals = [s[key] for s in sets if key in s]
        origin[key] = [len(vals), total]
        if vals:
            values[key] = op(vals)
    return values, origin


def aggregate_note(origin: dict) -> str:
    """把「哪些键一条都没报过 / 哪些只从部分组聚出来」说成一句人话; 全都齐时返回空串"""
    missing = sorted(k for k, (got, _) in origin.items() if got == 0)
    partial = sorted(k for k, (got, tot) in origin.items() if 0 < got < tot)
    if not missing and not partial:
        return ""
    parts = []
    if missing:
        parts.append(f"这些量**没有任何一组算过**, 因此不发布: {missing}")
    if partial:
        parts.append("这些量只从部分组聚出来: "
                     + ", ".join(f"{k} {origin[k][0]}/{origin[k][1]} 组" for k in partial))
    return " ⚠️ " + "; ".join(parts)


def binding_attainment(metrics: dict) -> Optional[float]:
    """绑定轴上的 roofline 达成率 (%), 拿不到返回 None

    memory-bound 看带宽利用率、compute-bound 看 GFLOPS 利用率。取两者的 **max**
    而不是按 bound 分支选一个: roofline 的语义就是「离上界多远由紧的那条轴决定」,
    分支选指标会在两条轴接近时选错 (而 `roofline_bound` 这个字符串本身还有过
    memory/compute 与 name/bound 两套拼法飘掉的历史, 见 docs/research/15 第五节)。
    """
    vals = []
    for key in ("gflops_utilization_pct", "bandwidth_utilization_pct"):
        v = metrics.get(key)
        if isinstance(v, (int, float)) and v >= 0:
            vals.append(float(v))
    return max(vals) if vals else None


def persist_decision(metrics: dict) -> tuple:
    """这次运行的核值不值得写进跨运行经验库 (缺口 2)。返回 (入库?, 理由)

    写成判据而不是写在 conductor 里, 是为了让它有单测与变异守卫 —— 缺口 2 原先的
    失效方式正是「快照一直在写, 但没人读」(docs/research/43 第三节)。

    门比 Voyager 的 `if success` 严三条, 理由都来自本机实测:
      - `measurement_can_gate` (缺口 1 的产物): 「通过」可以是被 launch 地板喂出来的
        假信号 (doc 31/32: 同一个 layernorm 核换个计时口径, 加速比从 0.895x 翻到 1.346x)。
        持久化会把这种假信号**固化成下一次检索的样本**, 比一次性误判更贵。
      - `backend_real_device` (doc 49 §3.5 第 2/3 处): 执行侧没声明 `Cap.REAL_DEVICE`
        时, 那个加速比要么是按规格**估**的 (`MockGPU` 的 `mode="simulated"`), 要么是
        控制机 CPU 上真测的 (`LocalExecutor`)。两者都不是「这张卡上这个核有多快」的
        证据, 而语料是按 (algorithm, hardware) 检索的 —— 一条 mock 成绩会被下一次
        真机运行当示范取出来。用 `is False` 而不是 `not`: 老产物没这个键, 不做追溯性重判。
      - score 用 speedup 而不是 gflops: 对 memory-bound 算子 gflops 整个失真 (缺口 4)。
    """
    if metrics.get("measurement_can_gate") is False:
        return False, "测量不可信 (can_gate=False): 不能把假信号固化成语料"
    if metrics.get("backend_real_device") is False:
        return False, ("执行侧未声明 REAL_DEVICE: 数字来自模拟或 CPU, "
                       "不是这张卡上的成绩, 不能当语料")
    if metrics.get("benchmark_contaminated"):
        return False, "窗口被抢占污染 (util 超阈值)"
    from ..decide.eligibility import decision_speedup
    speedup = decision_speedup(metrics)
    verdict = metrics.get("performance_verdict") or {}
    if verdict.get("basis") == "objective" and verdict.get("passed") is not True:
        return False, "声明的 workload 目标/回退判据未通过，不能退回主 case 入库"
    # `bool` 显式排除: `isinstance(True, int)` 在 Python 里是真的, 而 `True <= 1.0` 为假,
    # 于是一个被写成 `True` 的 speedup 会被这道门当成「1x 且合格」放过去。schemas 校验
    # `min_speedup` 时同一条规矩 (`schemas.py:257`)。
    if isinstance(speedup, bool) or not isinstance(speedup, (int, float)):
        # 三态而不是两态: 「没有这个读数」与「读数说不快」不是一回事。前者是缺陷信号
        # (该轮没走到 L3, 或载荷没把 speedup 搬回来), 报成「没有加速」就把缺证据当成了
        # 否定证据 —— 与 `EnvironmentDrift.comparable`、`assess()` 里「地板未知」、
        # `_duplicate_rate` 里「没有签名 ≠ 0%」是同一条纪律 (2026-09-29 真机报表里
        # `speedup=None` 被印成「相对同机基线没有加速」, 见 doc 48 §六B)。
        return False, (f"没有可用的 speedup 读数 (speedup={speedup!r}) —— "
                       f"这不是「没加速」的判定, 是缺陷信号: 该轮没走到 L3 或载荷没搬回来")
    if speedup <= 1.0:
        return False, f"相对同机基线没有加速 (speedup={speedup})"
    return True, (f"speedup={speedup:.3f}x, "
                  f"validity={metrics.get('measurement_validity', '未知')}")


def reference_ceiling_reached(metrics: dict) -> bool:
    """已经贴着参考实现的天花板 —— 继续换配方只会白烧 GPU 与 LLM token"""
    if (metrics.get("measurement_can_gate") is False
            or metrics.get("benchmark_contaminated")):
        return False          # 不可信的测量不能说「已经到顶」
    # 分母自己不可信时同样不能说「到顶」: 峰值表里没核过口径的行可能偏小 2 倍
    # (sparse/dense 混过), 那会把还能优化的核提前判成已到上限。用 `is False` 而不是
    # `not ...`: 老记录里没这个键时保持原有行为 (默认开), 新链路一律带 True/False。
    if (metrics.get("roofline_analysis") or {}).get("peak_trusted") is False:
        return False
    # 分母是**参考下界**不是器件峰值 (设计 §13.5 的三个概念: peak_trusted / peak_measured /
    # reference_lower_bound)。「口径核过」不等于「有下界」: 表里的规格/推导行给的是
    # datasheet 推算, 与「这次测量的上限」之间隔了时钟/温度/占用率。
    # ⚠️ 因此本出口的名字是 `reference_ceiling_reached`(贴着**参考实现**的天花板), 不是
    # `roofline_converged`: 参考下界是下界 ⇒ 拿它当分母利用率偏大、出口偏松, 它不配说
    # 「贴着硬件峰值」。真正的硬件峰值出口只认 `peak_measured`, 而今天代码里没有那个量具。
    # 缺这个键的老记录同样不开 (fail-closed): 缺证据不是有效证据 —— 与「地板未知」
    # 「峰值表行未核」同一条纪律, 而且方向是不对称的 (关着只会白烧几轮, 还有停滞出口兜底;
    # 开着会把还能优化的核提前判成到顶)。
    if (metrics.get("roofline_analysis") or {}).get("reference_lower_bound") is not True:
        return False
    a = binding_attainment(metrics)
    return a is not None and a >= REFERENCE_CEILING_PCT


def noise_band_for(validity: Validity) -> Optional[float]:
    """这一档的改善判决带宽; `UNINFORMATIVE` 返回 `None` 而不是表里那个 0.02。

    那不是我懒得查表: `verdict()` 在 consult 表**之前**就对 uninformative 返回 AMBIGUOUS
    ("判决带宽比信号本身宽"), 所以那一轮**没有任何判决用过这条带宽**。把它记成 0.02 会让
    血缘里出现一个从未参与过决定的数 —— 与 #81「通过侧聚合用默认值造证据」同一族,
    方向反过来 (那是没测的键被兜上值, 这是没用上的带宽被当成用过了)。
    """
    if validity is Validity.UNINFORMATIVE:
        return None
    return NOISE_BAND[validity]


def verdict(current: Optional[float], previous: Optional[float],
            validity: Validity) -> tuple[Verdict, str, Optional[float]]:
    """两轮性能值的四态判决。current/previous 是同一轴上的标量 (越大越好)。

    打平不算退步: `SAME` 与 `SLOW` 的区别就是「能不能禁用配方」的分界。

    返回值第三个元素是**这次判决真正用过的那条带宽** (`None` = 这一支没有 consult 表)。
    它必须从判决这里出, 不能由调用方自己查一次表 (`LINEAGE_BAND_KEY` 那格记的就是它):
    「有没有用过带宽」是判决自己的分支结构 (缺值与 uninformative 两支都没用过), 调用方要
    复刻这个就是第二套判据 (I22 那一族); 而分两次查同一张表之间表可以被换掉 —— 真机现场:
    续跑跨过一次代码改动, 于是「记下的带宽」与「判决用过的带宽」不是同一个数。
    """
    if current is None or previous is None or current <= 0 or previous <= 0:
        return Verdict.AMBIGUOUS, "无可比的性能值 (缺 current 或 previous)", None
    if validity is Validity.UNINFORMATIVE:
        return Verdict.AMBIGUOUS, "测量档位 uninformative: 判决带宽比信号本身宽", None
    band = noise_band_for(validity)
    delta = (current - previous) / previous
    if delta > band:
        return Verdict.FAST, f"+{delta:.4%} > 带宽 {band:.2%}", band
    if delta < -band:
        return Verdict.SLOW, f"{delta:.4%} < -带宽 {band:.2%}", band
    return Verdict.SAME, f"{delta:+.4%} 落在带宽 ±{band:.2%} 内", band


# ── 跨轮的改善判决带宽 (#87 第 4 片, **只观测**) ──
#: 落在 `iteration_history` 每一轮上的键: 这一轮的四态判决**真正用过**的那条带宽。
#: `None` = 这一轮走到了判决但没用带宽 (AMBIGUOUS 那两支); 键**缺席** = 这一轮压根没到
#: L3 判决 (L1/L2 就死了) 或产物写在这项之前 —— 两者在 `cross_round_band` 里分开数。
LINEAGE_BAND_KEY = "l3_improvement_band"

BAND_SAME = "same"                        #: 用过带宽的轮数 ≥2 且值全同
BAND_TIER_EXPLAINED = "tier_explained"    #: 带宽变了, 而档位也变了且对得上表 —— 那是**测量变差**
BAND_SPEC_DRIFT = "spec_drift"            #: 带宽变了, 档位却没变 —— 那才是**规格漂移**
BAND_UNEXPLAINED = "unexplained"          #: 带宽变了, 而档位那一格缺失/对不上表, 无从归因
BAND_UNMEASURABLE = "unmeasurable"        #: 用过带宽的轮不足 2, 这个问句没有输入
BAND_STATES: tuple = (BAND_SAME, BAND_TIER_EXPLAINED, BAND_SPEC_DRIFT,
                      BAND_UNEXPLAINED, BAND_UNMEASURABLE)

#: #87 第 4 片当初没做的那件事, 在这里有了答案: 「带宽跨轮换过」同时是**两件事**的读数 ——
#: 档位降级本身就按表把带宽换宽 (那是测量质量那一维, `measurement_validity` 已经记着),
#: 而**表本身**被换过才是规格漂移 (现场: 续跑跨过一次代码改动, 前一半轮用的是旧表)。
#: 把两件事折成一句「尺变了」会把降级这件事报成漂移, 而它是我们已经知道并另有一格记录的事;
#: 反过来把「档位一致」当「尺一致」则永远看不见换表。所以这里按 `(band, tier)` 成对判,
#: 而且 tier 缺失时给第三态 `unexplained` —— 既不归因成表漂移, 也不归因成降级 (#53 那一族)。
def _band_of_tier(tier):
    """档位字符串 → 该档在表里的带宽; 认不出来 (旧产物的别的写法 / 非枚举值) 返回 `None`。

    血缘里那一格存的是 `Validity.value`, 但产物可能来自写过另一套值的版本 —— 让
    `Validity("…")` 直接抛会把一次离线复盘变成崩溃, 而这条判据要的只是「能不能归因」。
    """
    try:
        return NOISE_BAND[Validity(tier)]
    except (ValueError, KeyError, TypeError):
        return None


def cross_round_band(history) -> dict:
    """逐轮读 `(带宽, 档位)` 这一对, 把「测量变差」与「规格漂移」分开归因。**不改判决**。

    三态在两层上: 结论本身可能是 `None` 意义上的没有输入 (`BAND_UNMEASURABLE`), 而每一对
    读数可能是「带宽没用过」(`None`) 或「这一轮没走到判决」(键缺席) —— 后者与前者在
    `rounds_no_band` / `rounds_no_key` 里分开数, 不许合并。
    """
    rounds = list(history or ())
    used = []            # (round_no, band, tier) —— 只有真用过带宽的轮进来
    no_band = 0          # 走到了判决但没用带宽 (AMBIGUOUS)
    no_key = 0           # 这一轮压根没到判决 / 旧产物没有这一项
    missing_tier = 0
    for i, rec in enumerate(rounds, 1):
        if LINEAGE_BAND_KEY not in rec:
            no_key += 1
            continue
        band = rec[LINEAGE_BAND_KEY]
        tier = rec.get("measurement_validity")
        if band is None:
            no_band += 1
            continue
        used.append((rec.get("round", i), band, tier))
        if tier is None:
            missing_tier += 1

    out = {"rounds": len(rounds), "rounds_with_band": len(used),
           "rounds_no_band": no_band, "rounds_no_key": no_key,
           "missing_tier": missing_tier,
           "bands": [{"round": r, "band": b, "tier": t} for r, b, t in used],
           "band_state": BAND_UNMEASURABLE, "band_changed": {}, "drift_rounds": [],
           "tier_rounds": []}
    if len(used) < 2:
        out["note"] = ("用过带宽的轮不足 2 —— 这个问句没有输入, 不是「跨轮同一条带宽」")
        return out

    distinct = {b for _, b, _ in used}
    if len(distinct) == 1:
        out["band_state"] = BAND_SAME
        return out
    out["band_changed"] = {"band": {r: b for r, b, _ in used}}

    # 带宽换了: 逐对看「档位这一维能不能解释它」。能解释 ⇒ 那是测量变差 (已有另一格记着);
    # 档位在而解释不了 ⇒ 规格漂移; 档位不在 (缺失) ⇒ 无从归因, 两个都不许说。
    explainable = True
    drifted_by_tier = False
    for (r1, b1, t1), (r2, b2, t2) in zip(used, used[1:]):
        if b1 == b2:
            continue
        if t1 is None or t2 is None:
            explainable = False
            continue
        if t1 == t2:
            out["band_state"] = BAND_SPEC_DRIFT
            out["drift_rounds"].append((r1, r2))
            continue
        e1, e2 = _band_of_tier(t1), _band_of_tier(t2)
        if e1 is None or e2 is None or e1 != b1 or e2 != b2:
            explainable = False   # 连表都对不上 (或那一格根本不是个档位) —— 那就不只是降级
            continue
        drifted_by_tier = True
    if out["band_state"] == BAND_SPEC_DRIFT:
        return out
    out["band_state"] = (BAND_TIER_EXPLAINED if (drifted_by_tier and explainable)
                         else BAND_UNEXPLAINED)
    if drifted_by_tier:
        out["tier_rounds"] = [r for r, _, _ in used]
    return out


def band_note(agg: dict) -> str:
    """`cross_round_band` 的那句人话 (两个读者共用一个实现, 与 `ruler_note` 同姿势)"""
    state = agg.get("band_state")
    if state == BAND_SAME:
        return (f"{agg['rounds_with_band']} 轮用过带宽的那几轮, 带宽是同一把 "
                f"{_band_head(agg)} —— 各轮同带宽是默认前提, 不占日志 (只观测不判决)")
    if state == BAND_UNMEASURABLE:
        return (f"测不到: 用过带宽的轮只有 {agg.get('rounds_with_band', 0)} 轮 "
                f"(不足 2), 而 {agg.get('rounds_no_band', 0)} 轮走到了判决却没用带宽、"
                f"{agg.get('rounds_no_key', 0)} 轮压根没这一项 —— **这不是**「跨轮同一条带宽」")
    if state == BAND_TIER_EXPLAINED:
        return ("带宽换了, 但那是**测量档位降级**按表换的宽 (不是规格漂移): "
                + _band_head(agg) + f" · 轮 {agg.get('tier_rounds')}; "
                "另一格 `measurement_validity` 已经记着档位, 这里只补归因")
    if state == BAND_UNEXPLAINED:
        return ("带宽换了而**归因不了**: 有轮次缺 `measurement_validity` 或对不上表 ("
                f"缺档位的轮 {agg.get('missing_tier')}) —— 既不说「只是降级」也不说「规格漂移」"
                f" · {_band_head(agg)}")
    if state == BAND_SPEC_DRIFT:
        return ("带宽在**同一档位**下换了值 ⇒ 那是规格漂移, 不是测量变差。现场通常是"
                "续跑跨过了一次代码改动 (`NOISE_BAND` 被改过), 或两侧读的不是同一份表 · "
                + _band_head(agg) + f" · 变在轮 {agg.get('drift_rounds')}")
    return f"未知的带宽状态 {state!r} —— 判据与读者不同源 (I22 那一族)"


def _band_head(agg: dict) -> str:
    return " / ".join(f"轮 {x['round']} → {x['band']:.2%} ({x['tier']})"
                      for x in (agg.get("bands") or [])[:4])


# launch_dominance = 地板 / 候选 median 时间。
# 0.10 与 0.30 是 docs/research/01 第二节的经验分界 (<10% 时实测/下界已收敛到
# 1.1×; >30% 时曲线仍完全平坦), **不是推导值**, 改动需回归验证。
LAUNCH_DOMINANCE_VALID = 0.10
LAUNCH_DOMINANCE_UNINFORMATIVE = 0.30

# 沿用 verifier 里既有的硬污染阈值 (>5%), 不另立新数。
# 只统计 util 而不统计「有其他进程」: 这台机器上恒有 7 个其他进程, 若把「有进程」
# 也算污染, 每次运行都会被标记, 信号就失效了。
CONTAMINATION_UTIL_PCT = 5.0

# 样本离散度 iqr/median 的门限 (阈值由真机标定得出: docs/research/20 第四节)。
#
# 0.10 取自 PyTorch `torch.utils.benchmark` 的 `_IQR_WARN_THRESHOLD` (✅ 一手核实
# `torch/utils/benchmark/utils/common.py`), 它是「结果可以报到几位有效数字」的判据。
# NVBench 的 `same.relative_dispersion_ceiling` 是 0.02 (✅ 默认 TOML), 但那是**锁频
# 专用机**的值; 本机是 8 卡共享、恒有 7 个其他进程, 直接套 0.02 会让几乎所有测量
# 落到最严的一档, 信号同样失效 —— 与上面 util 阈值只取 5% 是同一个理由。
#
# 0.30 这一档没有外部锚点, 是按「与 launch dominance 的 0.10/0.30 保持同一节奏」
# 取的, 属经验值, 改动需回归验证。
DISPERSION_DEGRADED = 0.10
DISPERSION_UNINFORMATIVE = 0.30

# 「测量窗口内发生过抢占」的旁证: 最慢样本 / 中位样本。
#
# iqr/median 对这个完全不敏感 (29 个 1.0 + 1 个 50.0 → iqr = 0), 而当决策统计量是
# median 时这正是想要的性质。问题在于 gpu_util 只在测量**前**采一次快照, 窗口中途
# 插进来的邻居进程它同样看不见 —— 于是「测量期间发生过抢占」这件事以前没有任何
# 证据通道。max/median 是唯一的廉价旁证。
#
# ⚠️ 比值同时依赖核的绝对时长和样本数, 因此目前只作诊断、不改 validity 档位。
# 真机数据 (H20, 2026-09-26, docs/research/33):
#
#     核                    median   实际样本数 runs   max/median
#     layernorm 1024×768    7.2 µs        750         19.17
#     layernorm 16384×6144  164 µs        239          1.86   ← util=0%, 无邻居插队
#     matmul 4096³ fp16     980 µs         48          1.12
#
# 原先 benchmarks/calibration/calibrate_dispersion_thresholds.py 标定时刻意固定 30 样本/轮,
# 所以只看见第 1 层偏差 (绝对时长), 没看见第 2 层 (顺序统计量随 n 抬升) —— 那是
# 标定脚本自身的方法缺陷。阈值保留在此, 供 reasons 里那句观察引用; 要重新赋予判据
# 地位, 须先换成对 n 稳健的统计量。
INTRUDER_MAX_OVER_MEDIAN = 2.0


# ─── 数字的出处 (doc 49 §3.5 第 2/3 处) ────────────────────────────────────────
#
# 两个字符串是**载荷自报**的值, 只用来与执行侧的能力声明对照, 不用来推断能力:
#   `"simulated"` = `MockGPU._benchmark_triton_simulated` 按硬件规格估出来的那份;
#   `"cpu"`       = `LocalExecutor` / `MockGPU._benchmark_python` / runner 的
#                   `"device": str(device) if lang == "torch" else "cpu"` 三处会打的标。
# 判据本身走 `Cap` (声明), 见 `provenance_conflicts` 的 docstring。
SIMULATED_MODE = "simulated"
CPU_DEVICE = "cpu"


def provenance_conflicts(backend_caps, required_caps,
                         reported_mode: Optional[str] = None,
                         reported_device: Optional[str] = None) -> tuple:
    """这批数字的**出处**站不站得住。返回 `(能力缺口, 自相矛盾)`, 都是字符串 tuple

    为什么要两份而不是一份: 它们的后果不同。

    * **能力缺口** = 任务在 `target.required_caps` 里声明了它要什么, 执行侧的
      `capabilities` 里没有。这是 doc 49 §五 第 4 片的验收条件所要拦的那件事:
      `mode == "simulated"` 的 L3 不许 `passed=True` (当任务声明了 `REAL_DEVICE`)。
      数字不是「测得不准」, 是**根本没测** —— 一份按规格估出来的加速比, 它的
      「通过」与「不通过」都没有对象。
    * **自相矛盾** = 声明与载荷自己报的对不上。声明 `REAL_DEVICE` 却报
      `mode="simulated"` / `device="cpu"`, 说明提供方要么在说谎、要么中途回退了
      (真有其事: `MockGPU` 有 CUDA 时声明 `REAL_DEVICE`, 而跑 python 核走的是
      `_benchmark_python`, 载荷老老实实写着 `device="cpu"`); 反过来, 载荷自报
      `simulated` 而后端没声明 `SIMULATED_PERF`, 是一次**未声明的降级**
      (doc 49 §3.3 那一族)。

    ⚠️ 刻意**不**用 `device`/`name` 的子串去推断能力 (那正是 doc 49 §3.1 记的三处
    名字匹配的毛病): 这里只判「等于 `cpu`」这一个精确值, 且只在它**与声明打架**时
    才算证据。`"cuda (fallback)"` 这类串一律不认 —— 认了就等于把能力判断又搬回
    字符串匹配。
    """
    caps = frozenset(backend_caps or ())
    required = frozenset(required_caps or ())
    shortfall = tuple(sorted(c.value for c in (required - caps)))

    declares_real = Cap.REAL_DEVICE in caps
    mode = str(reported_mode or "").strip().lower()
    device = str(reported_device or "").strip().lower()

    conflicts: List[str] = []
    if declares_real and mode == SIMULATED_MODE:
        conflicts.append(f"声明了 {Cap.REAL_DEVICE.value}, 载荷却自报 "
                         f"mode={reported_mode!r} (性能是按规格估的)")
    if declares_real and device == CPU_DEVICE:
        conflicts.append(f"声明了 {Cap.REAL_DEVICE.value}, 载荷却自报 "
                         f"device={reported_device!r} (这一轮在 CPU 上测的)")
    if mode == SIMULATED_MODE and Cap.SIMULATED_PERF not in caps:
        conflicts.append(f"载荷自报 mode={reported_mode!r}, 后端却没声明 "
                         f"{Cap.SIMULATED_PERF.value} (未声明的降级)")
    return shortfall, tuple(conflicts)


def _worse(a: "Validity", b: "Validity") -> "Validity":
    """取更差的一档。三个信号各自独立判定, 合成时不做加权 —— 任一项不可信,
    整个数字就不该拿去做闭环决策。"""
    order = {Validity.VALID: 0, Validity.DEGRADED: 1, Validity.UNINFORMATIVE: 2}
    return a if order[a] >= order[b] else b


@dataclass
class MeasurementAssessment:
    """一次 L3 测量的有效性判定结果

    三个 can_* 是给 Conductor 用的: 它不需要知道地板和 util 是多少, 只需要知道
    「这个数字能不能拿去做这个决定」。把用途而不是档位暴露出去, 是为了让调用方
    不必在自己的代码里重新解释一遍分档表 (那正是缺口 1 原来的失效方式:
    verifier 算出了 contaminated, 却没有任何下游读它)。
    """

    validity: Validity
    reasons: List[str] = field(default_factory=list)
    launch_dominance: Optional[float] = None     # None = 地板未知, 无从判定
    launch_floor_us: Optional[float] = None
    contaminated: bool = False
    dispersion: Optional[float] = None           # iqr/median, None = 样本不足或未上报
    max_over_median: Optional[float] = None      # 窗口内抢占的旁证, 与 dispersion 互补
    #: 离散度是否**真的**越过了阈值 (`> DISPERSION_DEGRADED`, 含 UNINFORMATIVE 那档)。
    #: None = 未上报 / 样本不足, 无从判定。
    #: 这是离散度判据的**唯一**来源 (spine-review P1-6): 调用方不许拿
    #: `validity is not VALID` 重编码一遍 —— 那个式子从不与任何 CV 阈值比较, 于是
    #: 污染 / 出处矛盾 / 漂移造成的降级都会被贴上「离散度超标」的标签, 让诊断文本
    #: 指向一个没被检查过的东西。
    dispersion_exceeded: Optional[bool] = None
    intruder_suspected: bool = False             # 仅诊断; 为何不作判据见 assess() 内说明
    #: 任务要求、而执行侧没声明的能力 (`Cap.value` 字符串, 已排序)
    capability_shortfall: tuple = ()
    #: 声明与载荷自报互相打架的地方 (doc 49 §3.5 第 2/3 处)
    provenance_conflict: tuple = ()
    #: 任务声明要的能力, 用来决定「矛盾」够不够格拒绝 L3 通过
    required_caps: tuple = ()
    #: 执行侧是否声明了 `Cap.REAL_DEVICE`。None = 调用方没给能力信息 (老路径)
    backend_real_device: Optional[bool] = None
    #: 与上一次运行相比变了哪些环境项; 空 = 比过了没漂移或无从比较 (见 EnvironmentDrift)
    environment_drift: tuple = ()
    #: 这份候选的**正确性没有凭据** (设计 §14-10 / §20.4): 只有 `verify_l3_only` 那条
    #: 绕开 L2 的路径可能带出 `True`。它不是"测得准不准", 而是"这个数有没有指称对象" ——
    #: 所以与 `capability_shortfall` 同族: 拒 `passed` (见 `l3_pass_is_refused`)。
    correctness_unverified: bool = False
    ruler_identity_error: str = ""

    @property
    def can_gate(self) -> bool:
        """能否用于 L3 的通过/不通过判定"""
        return self.validity is not Validity.UNINFORMATIVE

    @property
    def can_disable_recipe(self) -> bool:
        """能否据此禁用配方 —— 只有 VALID 可以"""
        return self.validity is Validity.VALID

    @property
    def can_count_stagnation(self) -> bool:
        """能否据此累计停滞轮次 / 触发提前终止 —— 只有 VALID 可以"""
        return self.validity is Validity.VALID

    @property
    def l3_pass_is_refused(self) -> bool:
        """L3 的 `passed` 不许为 True —— 与 `can_gate` 刻意分成两条

        `can_gate=False` 说的是「这个数字不足以判定通过/不通过」, 它**不**把 passed
        拉成 False: launch 地板支配 / 离散度超标的那一档, 核可能确实是对的, 只是数字
        撑不住结论, 而既有链路 (含真机标定过的 layernorm 1024×768) 一直按这个语义跑。

        出处不对是另一回事。任务声明了要真卡, 执行侧给不出 (或给了却自报在 CPU 上测),
        这时「通过」这个结论**没有对象** —— 那份加速比不是测得不准, 是压根没测。
        doc 49 §五 第 4 片的验收条件就是这一条: `mode == "simulated"` 的 L3 结果
        不许 `passed=True` (当任务声明了 `REAL_DEVICE`)。
        """
        if self.capability_shortfall:
            return True
        # 正确性没凭据 (设计 §14-10 / §20.4) 与出处缺失同族: 一份"我们还没确认它算得对"
        # 的核, 它的速率读数不是测得不准, 而是**没有指称对象** ⇒「通过」不许为 True。
        if self.ruler_identity_error or self.correctness_unverified:
            return True
        # 声明与载荷打架时, 只有「任务真的要过真卡」才够格拒绝通过: 否则一份本地
        # mock 运行会因为载荷老实写了 device="cpu" 而被判失败, 那是把观测当判据。
        return bool(self.provenance_conflict) and Cap.REAL_DEVICE.value in self.required_caps

    @property
    def l3_pass_refusal_causes(self) -> tuple:
        """`l3_pass_is_refused` 的**成因**, 人可读, 给 message 用

        判定本身只有一个是非题 (passed 收不收), 而读者要的是"为什么不收"。三个成因的
        处置完全不同 —— 混成一句"能力没拿到"就是 I46 那一族: 判定对了, 说出来的却是
        另一件事 (真机 §14-8 抓到的正是这个形状)。

        分支顺序与 `l3_pass_is_refused` 一致; 两者**同时有内容/同时为空**由用例钉住
        (`tests/unit/test_measurement_validity.py`), 免得将来加了第四个成因只改一处。
        """
        causes = []
        if self.ruler_identity_error:
            causes.append("计时量具身份/锚点不成立: " + self.ruler_identity_error)
        if self.correctness_unverified:
            causes.append("这个候选的正确性没有凭据 "
                          "(只有 `verify_l3_only` 这条绕开 L2 的路会这样)")
        if self.capability_shortfall:
            causes.append("任务声明要的执行侧能力没拿到 (缺 "
                          + ", ".join(self.capability_shortfall) + ")")
        if self.provenance_conflict and Cap.REAL_DEVICE.value in self.required_caps:
            causes.append("出处自相矛盾: " + "; ".join(self.provenance_conflict))
        return tuple(causes)

    def as_metrics(self) -> dict:
        """摊平进 VerificationResult.metrics 的部分"""
        out = {
            "measurement_validity": self.validity.value,
            "measurement_validity_reasons": list(self.reasons),
            "measurement_can_gate": self.can_gate,
            "measurement_can_disable_recipe": self.can_disable_recipe,
            "measurement_can_count_stagnation": self.can_count_stagnation,
        }
        if self.launch_dominance is not None:
            out["launch_dominance"] = round(self.launch_dominance, 4)
        if self.launch_floor_us is not None:
            out["launch_floor_us"] = self.launch_floor_us
        if self.ruler_identity_error:
            out["ruler_identity_error"] = self.ruler_identity_error
        if self.dispersion is not None:
            out["dispersion_iqr_over_median"] = self.dispersion
        if self.dispersion_exceeded is not None:
            # 结论而不是原料: 下游要用「离散度超标了吗」时读这个键, 不许自己拿
            # validity 再推一遍 (spine-review P1-6 正是那样发生的)。
            out["dispersion_exceeded"] = self.dispersion_exceeded
        if self.max_over_median is not None:
            out["dispersion_max_over_median"] = self.max_over_median
        out["measurement_intruder_suspected"] = self.intruder_suspected
        # `backend_real_device` 是**判据的输入**而不是装饰: `persist_decision` 读它决定
        # 「这份成绩能不能进跨运行语料」。有了它, 一份纯模拟的基准与一份真机实测的基准
        # 在产物里不再权威性等价 (doc 49 §3.5 第 2 处)。
        # 只在调用方给了能力信息时才写 —— 缺键与 False 是两件事, 老产物走 `is False`
        # 判定, 不会被追溯性重判。同一批里的 shortfall/矛盾/拒绝是同一个理由: 没做过
        # 出处检查时不写, 免得下游把「没检查」读成「检查过、干净的」。
        if self.backend_real_device is not None:
            out["backend_real_device"] = self.backend_real_device
            out["measurement_capability_shortfall"] = list(self.capability_shortfall)
            out["measurement_provenance_conflict"] = list(self.provenance_conflict)
            # 这个布尔是**判据的输出**: 没有它, 下游想区分「核太慢所以没过」与
            # 「压根没测所以没资格过」就只能去 string-match message 里那个 ⛔。
            out["measurement_l3_pass_refused"] = self.l3_pass_is_refused
        # 漂移这里落成列表: 空 = 本轮没有漂移信号。注意「比过了没变」与「没有上一次可比」
        # 在这一层分不开 (conductor 两条都传 None) —— 那个区分在产物级的
        # `environment_drift.comparable` 里, 不在轮级 metrics 里。
        out["measurement_environment_drift"] = list(self.environment_drift)
        return out


def launch_dominance(launch_floor_us: Optional[float], net_time_ms: Optional[float]) -> Optional[float]:
    """端到端时间里 launch 开销占的比例; 任一侧缺失或非正则返回 None (无从判定)

    runner 报的是**冲刷 L2 后**的净执行时间 (Triton `do_bench` 口径), 事件窗口里
    已经不含 GPU 空等 launch 命令的那一段。而用户真正付出的端到端时间是
    「launch 开销 + 净执行」, 所以占比是 `floor / (floor + net)`。

    不能用 `floor / net`: 真机标定 (benchmarks/calibration/calibrate_l2_flush_effect.py, H20,
    torch 2.14.0+cu130 / triton 3.8.0) 显示同一个空 Triton 核

        不冲刷 median = 20.03 µs      冲刷 median = 4.64 µs

    也就是说「20.24 µs 的 launch 地板」量的其实是**空闲 GPU 上事件窗口吃进的
    launch 延迟**, 不是一个恒定的执行开销 —— 用 do_bench 的 256 MB 冲刷把 GPU
    填满后, 同一个空核只剩 4.64 µs。冲刷前后是两个口径, 拿旧口径的地板去除新
    口径的净时间会算出 >1 的值 (实测 layernorm 1024×768 得到 2.81), 那正是口径
    混用的信号。
    """
    if not launch_floor_us or launch_floor_us <= 0:
        return None
    if not net_time_ms or net_time_ms <= 0:
        return None
    floor_ms = launch_floor_us / 1000.0
    return floor_ms / (floor_ms + net_time_ms)


def assess(median_time_ms: Optional[float] = None,
           launch_floor_us: Optional[float] = None,
           gpu_util_pct: Optional[float] = None,
           dispersion: Optional[float] = None,
           max_over_median: Optional[float] = None,
           backend_caps: Optional[frozenset] = None,
           required_caps: Optional[frozenset] = None,
           reported_mode: Optional[str] = None,
           reported_device: Optional[str] = None,
           environment_drift: Optional[List[str]] = None,
           gpu_harness_peers: Optional[int] = None,
           correctness_verified: Optional[bool] = None,
           ruler_identity_error: str = "") -> MeasurementAssessment:
    """综合判定测量有效性: 信号各自独立判定, 合成时取更差的一档

    问「测得准不准」的三个: launch 地板支配 (`median_time_ms` vs `launch_floor_us`)、
    测量前 util 污染 (`gpu_util_pct`)、样本主体离散度 (`dispersion` = iqr/median)。

    问「测的是不是这个东西」的两个 (doc 49 §3.5): 出处 (`backend_caps` 对
    `required_caps`, 加上载荷自报的 `reported_mode`/`reported_device` 作对照) 与
    环境漂移 (`environment_drift`, 由 `environment.compare_fingerprints` 算出)。

    污染判为 DEGRADED 而不是 UNINFORMATIVE: `benchmark_with_baselines` 是同机同卡
    依次测候选与各基线的, 两者落在同一个污染窗口里, 比值仍保留部分意义; 但绝对
    gflops 与比值都被扰动过, 不足以支撑「这个配方没用」这种结论。

    离散度同理: iqr/median 超过 0.10 说明这台机器此刻在抖, median 本身仍是无偏的,
    但它的精度撑不住「加速比 1.02x」这种量级的结论。

    环境漂移也停在 DEGRADED, 而且**没有阈值**: 「torch 从 2.14 变成 2.15」到底该不该
    让一次运行作废, 本机还没有分布能回答。按项目「观测先于判定」的处置, 先把
    「变了哪些键」这件事变得可见 (产物 + 日志), 并且不让它去驱动永久决策。

    max_over_median **不改变档位**, 只写进 reasons 并置 `intruder_suspected` —— 它被
    顺序统计量效应污染, 实测干净测量就已打到 1.86。详见 INTRUDER_MAX_OVER_MEDIAN
    注释与 docs/research/33。
    """
    reasons: List[str] = []
    validity = Validity.VALID

    dom = launch_dominance(launch_floor_us, median_time_ms)
    if dom is None:
        # 地板未知 (hardware_caps 里没有这台设备的实测值) 时不猜。缺证据不等于无效,
        # 但必须显式说出来, 否则「没检查」会被读成「检查过了」。
        if launch_floor_us is None:
            reasons.append("launch 地板未知, 未做地板支配检查")
    elif dom > LAUNCH_DOMINANCE_UNINFORMATIVE:
        validity = Validity.UNINFORMATIVE
        reasons.append(
            f"launch 地板占实测时间 {dom * 100:.1f}% (> {LAUNCH_DOMINANCE_UNINFORMATIVE * 100:.0f}%): "
            f"数字描述的是驱动派发开销, 不是核质量; 换更大的 shape")
    elif dom > LAUNCH_DOMINANCE_VALID:
        validity = Validity.DEGRADED
        reasons.append(
            f"launch 地板占实测时间 {dom * 100:.1f}% (> {LAUNCH_DOMINANCE_VALID * 100:.0f}%): "
            f"判定可用, 但不得据此禁用配方或累计停滞")

    contaminated = gpu_util_pct is not None and gpu_util_pct > CONTAMINATION_UTIL_PCT
    if contaminated:
        reasons.append(f"测量前 GPU util={gpu_util_pct}%, 基准被污染, 加速比不可比")
        validity = _worse(validity, Validity.DEGRADED)

    # I68: 同卡还有 harness 进程 (claim 声明)。与 util 污染同档 (DEGRADED 而不是
    # UNINFORMATIVE): interleaved A/B 的比值仍保留部分意义, 但实测时间可整体偏移
    # (I68 实测 2.1× 膨胀而档位曾仍 valid), 不足以支撑「禁用配方/累计停滞」这类
    # 永久决策。时刻采样 (util / compute-apps) 看不见还在 import 的对手, 所以这个
    # 信号来自**进程级** claim 声明, 生命期覆盖对手整个进程。
    peers = (gpu_harness_peers if isinstance(gpu_harness_peers, int)
             and not isinstance(gpu_harness_peers, bool) else None)
    if peers:
        reasons.append(f"同卡还有 {peers} 个 harness 进程 (claim 声明) —— 跨进程共卡, "
                       f"实测时间可整体偏移 (I68 实测 2.1×)")
        validity = _worse(validity, Validity.DEGRADED)

    disp_exceeded: Optional[bool] = None
    if dispersion is not None and dispersion >= 0:
        # 「越界」= 至少落进 DEGRADED 档。**判据就在这一行**, 别处不许拿 validity 重推。
        disp_exceeded = dispersion > DISPERSION_DEGRADED
        if dispersion > DISPERSION_UNINFORMATIVE:
            validity = _worse(validity, Validity.UNINFORMATIVE)
            reasons.append(
                f"样本离散度 iqr/median={dispersion:.3f} "
                f"(> {DISPERSION_UNINFORMATIVE:.2f}): 抖动量级已超过被测差异, "
                f"这一轮的数字无法支撑任何结论; 重测或换更长的 shape")
        elif dispersion > DISPERSION_DEGRADED:
            validity = _worse(validity, Validity.DEGRADED)
            reasons.append(
                f"样本离散度 iqr/median={dispersion:.3f} "
                f"(> {DISPERSION_DEGRADED:.2f}): median 仍无偏, 但精度撑不住小差异, "
                f"不得据此禁用配方或累计停滞")

    # max_over_median **只作观察, 不改变 validity 档位**。
    #
    # 真机验证 (docs/research/33) 推翻了把它当判据的设计。两层偏差同向叠加:
    #   1. 绝对时长 —— 一次 ~30 µs 的宿主抖动对 4.5 µs 的核是 18 倍, 对 1 ms 的核
    #      只有 1.04 倍;
    #   2. 顺序统计量 —— runner 按毫秒预算自动加采样 (实测 runs = 750/239/48),
    #      样本越多 max 越靠尾部。而能通过「核足够长」这个前提的核必然净时间
    #      >= 9x 地板 ~= 182 µs, 那个长度正好换来 ~274 个样本 —— 用得上这条判据的
    #      区间, 恰好是 max 被抬得最高的区间。
    # 实测 layernorm 16384x6144 在 util=0%、完全没有邻居插队时就已报出 1.86,
    # 逼近阈值 2.0。拿它当判据会系统性地把干净测量判成污染, 即把闭环静默关掉 ——
    # 那正是缺口 1 最初的病。
    #
    # 仍写进 reasons: 缺口 1 的教训是「算了没人读」, reasons 会进 Fixer /
    # SearchAgent 的 prompt。但措辞必须自我声明不影响判定, 否则下游会当判据读。
    # 想救回这条判据得换成对 n 稳健的形式 (如「> 2xmedian 的样本占比」)。
    intruder = (max_over_median is not None
                and max_over_median > INTRUDER_MAX_OVER_MEDIAN)
    if intruder:
        reasons.append(
            f"最慢样本是中位样本的 {max_over_median:.1f} 倍 (阈值 "
            f"{INTRUDER_MAX_OVER_MEDIAN:.0f}, 实测样本数越多该值天然越高): "
            f"窗口内可能有抢占, 也可能只是长尾; **仅诊断, 不影响本轮判定**")

    # ── 出处 (doc 49 §3.5 第 2/3 处) ──────────────────────────────────
    # `backend_caps is None` 表示调用方压根没提供能力信息 (老路径 / 直接调 assess 的
    # 单测), 与「提供了空集」必须分开: 后者是 `LocalExecutor` 如实声明的样子, 该判;
    # 前者无从判, 只能显式记一句。
    shortfall: tuple = ()
    conflicts: tuple = ()
    real_device: Optional[bool] = None
    if backend_caps is None:
        if required_caps:
            reasons.append("任务声明了 required_caps 但调用方没提供后端能力, 未做出处检查")
    else:
        real_device = Cap.REAL_DEVICE in frozenset(backend_caps)
        shortfall, conflicts = provenance_conflicts(
            backend_caps, required_caps, reported_mode, reported_device)
        if shortfall:
            validity = _worse(validity, Validity.UNINFORMATIVE)
            reasons.append(
                f"任务要求的能力执行侧没有: {', '.join(shortfall)} —— 这批数字不是"
                f"在满足要求的设备上测出来的, 「通过/不通过」都没有对象")
        for c in conflicts:
            validity = _worse(validity, Validity.UNINFORMATIVE)
            reasons.append(f"出处自相矛盾: {c}")

    # ── 环境漂移 (doc 49 §3.5 第 1 处) ────────────────────────────────
    drift = tuple(environment_drift or ())
    if drift:
        validity = _worse(validity, Validity.DEGRADED)
        reasons.append(
            "环境指纹与上一次运行不一致 (" + "; ".join(drift) + "): "
            "本轮加速比与上一次不可比, 不得据此禁用配方或累计停滞; "
            "**不据此判核失败** (观测先于判定)")

    # ── 正确性凭据 (设计 §14-10 / §20.4) ──────────────────────────────
    # 与出处那一条同族: 问的不是"测得准不准", 而是"这个数说的是不是这个东西"。
    # 三态: 只有明确的 False 才拒 (`is False`), 缺键 = 老产物没带这一格 ⇒ 不改判 ——
    # 与 `real_device=None`、`peak` 那几处同一边界 (measurement.py 里的既有写法)。
    if correctness_verified is False:
        validity = _worse(validity, Validity.UNINFORMATIVE)
        reasons.append(
            "这个候选的正确性**没有凭据** (correctness_verified=False): 它的速率读数是"
            "关于另一个对象的, 「通过」没有指称对象 —— 只有 `verify_l3_only` 这条绕开 "
            "L2 的路径会走到这里, 调用方应当把手里那份 L2 结果作为凭据传进来")

    if ruler_identity_error:
        validity = Validity.UNINFORMATIVE
        reasons.append("RULER_BROKEN: " + ruler_identity_error)

    return MeasurementAssessment(
        validity=validity,
        reasons=reasons,
        launch_dominance=dom,
        launch_floor_us=launch_floor_us,
        ruler_identity_error=ruler_identity_error,
        contaminated=contaminated,
        dispersion=dispersion,
        max_over_median=max_over_median,
        dispersion_exceeded=disp_exceeded,
        intruder_suspected=intruder,
        capability_shortfall=shortfall,
        provenance_conflict=conflicts,
        required_caps=tuple(sorted(c.value for c in frozenset(required_caps or ()))),
        backend_real_device=real_device,
        environment_drift=drift,
        correctness_unverified=(correctness_verified is False),
    )

# ── 可归因证据门 (#103 第 1 片, 外部依据 HarnessLens `2608.27311v1` 行 99/715/878-880) ──
#: 一手原话 (行 98-99): "Existing propose-and-verify methods typically score every candidate on a
#: fixed task set, ... **allowing aggregate scores to obscure specific regressions**"。
#: 那条门 (行 878-880): "A verification stage advances only if it contains **attributable positive
#: evidence and no attributable regression**. Positive evidence is either an attributed recovery or
#: an attributed stable success with an increased pass count; **preservation alone is insufficient**."
#:
#: 我方今天两份报表数字都**只比聚合数**: `compare_runs` 看任务级成功率的 Fisher 与 Δ 的符号,
#: 从不说「哪一条 case 变坏了」。2026-09-30 在存量两份报表上量到的形状比论文那句更糟一档:
#: `0a4f0fd` 的 5/6 与 `4e80b13` 的 3/6 之间, 统计门印的是「underpowered —— 不知道」,
#: 而逐 case 那一层其实是 **1 条修好 + 3 条变坏** (见 doc 60 §六)。缺的不是更严的阈, 是
#: **点名**这件事。
#:
#: 这条门只**观测**: 它不参与任何生产判决 (闭环的停止判据在 `conductor`, 不动)。它做的两件事
#: 与行 1050 那句 "Nothing is re-executed" 同形状 —— 把两条规则**算在同一份存量证据上**:
#: metric-only (Δ>0 就接受) 与可归因门 (要有正证据且零回归), 然后报它们的分歧与变坏的 case 名。
ATTRIBUTABLE_RECOVERED = "recovered"          #: 这侧从不过到过 —— 正证据的一种
ATTRIBUTABLE_REGRESSED = "regressed"          #: 从过到不过 —— 门要零, 不看总数
ATTRIBUTABLE_STABLE_PASS = "stable_pass"      #: 两侧都过 = preservation, **单独不构成正证据**
ATTRIBUTABLE_STABLE_FAIL = "stable_fail"      #: 两侧都不过
ATTRIBUTABLE_OUTCOMES: tuple = (ATTRIBUTABLE_RECOVERED, ATTRIBUTABLE_REGRESSED,
                                ATTRIBUTABLE_STABLE_PASS, ATTRIBUTABLE_STABLE_FAIL)


def attributable_evidence(cases) -> dict:
    """把「两批之间逐 case 的得与失」按两条规则各算一遍, 并给出它们的分歧。

    输入 `cases`: `(名字, 上一批的结果, 这一批的结果)` 的序列, 每个结果 ∈ {True, False, None}。
    `None` = **这一侧没有可用读数** (那次运行崩溃了 / 这批里没有这条 case / 同名出现了两次) ——
    它既不算回归也不算通过, 只进 `unmatched`。把缺席读成「没变坏」是 #53 那一族的反面
    (缺证据当通过), 把缺席读成「变坏了」是同一族的正面, 两条都不许。

    两条规则的口径 (行 878-880 逐句对应):
      · `accepted_metric_only` = 配对样本上 `pass_b > pass_a` —— 「聚合数涨了就算好」。
      · `accepted_gate` = 有正证据 **且** 回归数为零。正证据只有「至少一条 recovered」这一种:
        原文还给了第二种 ("an attributed stable success with an increased pass count"), 但在
        按 case 配对的口径下它**蕴含在**第一种里 (两侧都过的 case 不动任何一侧的计数), 于是
        那个分支会是永远走不到的代码 —— 这句偏差写在下面实现处, 不假装照抄。
        **只有 preservation 不构成正证据** (行 880 "preservation alone is insufficient")。
    `disagreement` 只有在两条都有布尔结论时才是布尔; 一条都没配上时三者都是 `None`,
    而**不是** `accepted_gate=True` —— 没有配对样本时「零回归」是空的真, 这正是
    I25 那一族「输入是空气时判据永真」的形状。
    """
    counts = {k: 0 for k in ATTRIBUTABLE_OUTCOMES}
    recovered: list = []
    regressed: list = []
    unmatched: list = []
    pass_a = pass_b = 0
    for case in cases:
        name, a, b = case[0], case[1], case[2]
        if a is None or b is None:
            unmatched.append(str(name))
            continue
        if a:
            pass_a += 1
        if b:
            pass_b += 1
        if not a and b:
            counts[ATTRIBUTABLE_RECOVERED] += 1
            recovered.append(str(name))
        elif a and not b:
            counts[ATTRIBUTABLE_REGRESSED] += 1
            regressed.append(str(name))
        elif a and b:
            counts[ATTRIBUTABLE_STABLE_PASS] += 1
        else:
            counts[ATTRIBUTABLE_STABLE_FAIL] += 1

    matched = sum(counts.values())
    out = {
        "matched": matched, "counts": counts,
        "pass_a": pass_a, "pass_b": pass_b, "delta": pass_b - pass_a,
        "recovered_cases": recovered, "regressed_cases": regressed,
        "unmatched_cases": unmatched,
        "accepted_metric_only": None, "accepted_gate": None,
        "disagreement": None, "positive_evidence": None,
    }
    if not matched:
        out["note"] = ("没有一条配上的 case —— 三条结论都是 `None`, 不是「没有回归」也不是"
                       "「都变坏了」(缺证据既不当通过也不当反证)")
        return out
    if recovered:
        out["positive_evidence"] = "recovery"
    else:
        # 原文给了**两种**正证据: "an attributed recovery **or** an attributed stable success
        # with an increased pass count" (行 879-880)。按我方「只在配上的 case 上数」这个口径,
        # 第二种**蕴含在**第一种里: 配对之后 `pass_b > pass_a` 必然意味着至少一条从不过到过
        # (两侧都过的 case 不动任何一侧的计数)。所以这里只可能有一种正证据 —— 这不是漏抄:
        # 他们的通过数在**固定任务集**上算, 「同一批里换了 case」也会让它涨, 那在我们看来
        # 恰好不是正证据。留着那个分支会变成一条永远走不到的代码 + 一张杀不掉的变异行。
        out["positive_evidence"] = None
    out["accepted_metric_only"] = pass_b > pass_a
    out["accepted_gate"] = bool(out["positive_evidence"]) and not regressed
    out["disagreement"] = out["accepted_metric_only"] != out["accepted_gate"]
    return out
