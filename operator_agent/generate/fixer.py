"""Fixer Agent 模块
职责单一: 根据 L1(编译)/L2(功能) 验证失败信息修复 kernel 代码

设计要点:
- 输入: 失败代码 + TaskSchema + 验证失败结果列表 (级别/消息/错误日志)
- 输出: (修复后的代码, 修复说明, Reflexion 结构化教训)
- 错误日志经 _truncate_error 截断压缩，避免撑爆 LLM 上下文
- 代码提取 (_extract_code) 与 Coder 的生成清理逻辑解耦，
  同时兼容 Triton 与纯 Python kernel，不做任何语义改写
- Reflexion: 修复成功后调用 produce_reflexion() 产出 {root_cause, lessons,
  error_keywords, applicable_hardware} 结构化教训供 FailureDatabase 沉淀
"""

import re
import json
import textwrap
import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional
from ..generate.agent_runtime import (  # §21.1 的生产者侧: 动作信封的解析只有一处
    ActionEnvelope,
    ENVELOPE_ABSENT,
    parse_action_envelope,
)
from ..core.schemas import TaskSchema, VerificationResult
from ..generate.prompt_manager import PromptManager
from ..generate.generation_context import (build_generation_context, candidate_summary,
                                 generation_request, make_candidate_record)
from ..core.kernel_db import (  # 用于回写 Reflexion 教训 + 本轮取向的归类
    FAILURE_CLASS_INTERFACE,
    FAILURE_CLASS_REFERENCE,
    FailureRecord,
    failure_class_of,
    failure_signature,
)


# ── 处方式错误反馈 (docs/research/20 C1) ─────────────────────────
#
# 依据是 SPARC (arXiv:2607.09803, ✅ 读过全文): **同一个错误, 归因为外部来源
# (编译器/硬件/GPU) 时模型能修, 归因为它自己的输出时修不了**; 其理论条件是误差
# 传播算子的谱半径 >= 1。它测的四个骨干是 Llama-3-8B / Qwen2.5-7B /
# DeepSeek-R1-Distill-7B / QwQ-32B —— 正是本项目自建 Qwen 27B 的档位, 所以这条
# 证据不是外推来的。另外该文报告加一个 `Wait` 类修正标记可把盲区降 89.3%。
#
# 所以这里做三件事, 顺序固定:
#   1. 措辞上把错误的来源指给编译器与数值检查器, 不写「你上一版写错了」;
#   2. 每条错误给「要求什么 / 该改哪一处」, 而不是只把 traceback 丢过去;
#   3. 开头放一个 Wait 标记, 强制先复述直接原因再生成代码。
#
# 处方表有两个入口, 少一个就会有半张表永远没人读: `_format_failures` 吃
# VerificationResult (Verifier / anticheat / 远程执行), 重试 feedback 吃
# `_validate_code` 的 issue 原文 —— 截断、结构缺失、跨硬件禁令三条只在那里出现。
#
# 表里的签名全部来自本项目真实出现过的失败 (logs/ 里的 28 条 L1/预检失败) 或
# doc 15/30 一手核实的 Triton 3.8 行为, 不是凭空的 Triton 错误百科。
_ATTRIBUTION = (
    "下面每一条都是编译器、GPU 或数值检查器在执行时实测到的现象, 与上一版是谁写的无关。"
    "Wait —— 动手改之前, 先按输出要求用 `根因:` 一行说明每条错误的直接原因, "
    "再照「要求 / 该改哪里」修。"
)

# 「本级判不出」那一轮 `_ATTRIBUTION` 与处方都不出现, 但错误段不能空着 —— 空着会让模型
# 自己去补一个「要求」, 而它补出来的方向只能是改代码。这一行顶掉那两条, 并把「没有判定」
# 这件事本身当成读到的东西。
_HOLD_NOTE = (
    "   本轮不给处方: 上面那句不是这一版的症状, 这一级没有对这段代码作出任何判定。"
    "要动的是参考那一侧 —— 范围见上一段取向。"
)

_PRESCRIPTIONS: list[tuple[str, str, str]] = [
    # ── 来自 _validate_code 本地预检 (重试 feedback 走 prescribe) ──
    (r"疑似输出截断|不配对",
     "上一版在某个结构闭合之前就被截断了, 所以它不是完整可解析的程序",
     "这一次必须从 import 到 return 输出完整函数, 不许续写片段; 若长度是瓶颈, "
     "删掉冗余的 BLOCK 常量组或注释来缩短, **不要**靠省略结尾"),
    (r"未能从 LLM 回复中提取有效代码|代码为空",
     "上一版里没有可编译的代码: 整段被解释文字占掉, 或只回到一半",
     "这次回复必须有且只有一份完整代码, 以 `import` 开头, 到 `return` 结束"),
    (r"expected an indented block",
     "某个 def/if/for 之后没有缩进块",
     "在报错那一行下面补至少一行 4 空格缩进的语句; 注意 Triton 核体不能是空的"),
    (r"invalid syntax|SyntaxError|unexpected EOF|unmatched|was never closed|语法错误",
     "编译器在这一行停止解析; 真正缺的闭合符号往往在**上面几行**",
     "从报错行往上找第一个未闭合的 ( [ { 并补上; 别只改报错那一行"),
    (r"缺少 @triton\.jit|缺少 kernel 入口|缺少 load_inline|缺少 C\+\+ 源码",
     "这份代码缺了跑起来必需的结构, 预检不会放行, 远程编译也不会开始",
     "按 SKILL 的最小骨架补齐: import → `@triton.jit` 的核函数 → `def kernel(...)` "
     "入口调用核并 return; 只补缺的结构, 不要顺手引入新的抽象层"),
    (r"但代码里出现了 Triton API|严禁使用任何 Triton",
     "目标平台上没有 Triton 运行时, 这份代码即使编译通过也跑不起来",
     "彻底重写为纯 Python/PyTorch: 删掉 import triton / @triton.jit / 所有 tl.* 行, "
     "矩阵乘法用 torch.matmul 或分块 torch.mm, 不要只注释掉报错的那几行"),
    # ── 来自 VerificationResult (Verifier / anticheat / 远程执行) ──
    (r"CheatDetected|反作弊|禁止调用|shield",
     "候选核里出现了被屏蔽的高层算子, 运行时拒绝执行",
     "参考实现由 harness 自己跑, 候选里不要调 torch 的高层算子 (matmul/softmax/"
     "layer_norm 等); 用显式分块循环 + tl.load/tl.store 把计算写出来"),
    (r"has been removed|no attribute|\bAttributeError\b|\bNameError\b",
     "这个标识符在当前 Triton 版本里不存在, 照记忆补 API 一定会再错一次",
     "只允许用 SKILL SPECIFICATION 与优化配方里明确出现的 tl.* 写法; 想用的 API "
     "不在其中就换成等价实现, 不要猜名字"),
    (r"max_abs|最大绝对误差|超出容差|数值",
     "输出与参考实现不一致, 超出了该精度档的容差",
     "按顺序查: 累加器是否 fp32 (R-UNIV-03) → tl.dot 的 input_precision → "
     "边界 mask 与 other= 的取值 → stride 是否从 tensor.stride() 取"),
]


def prescribe(err_text: str) -> Optional[tuple[str, str]]:
    """把一个错误文本映射到 (要求, 该改哪里); 认不出来返回 None。"""
    for pat, requirement, action in _PRESCRIPTIONS:
        if re.search(pat, err_text, re.IGNORECASE):
            return requirement, action
    return None


# ── 修复取向按「哪一级报的失败 + 库里怎么归类」分支 (docs/research/55,
#    CUDA-Harness §4.3.1 的两路 + Fabrica 行 521-522 的第三路 interface repair) ──
#
# 原文 (arXiv:2609.00058v1, 一手核过):
#   "The refinement branches based on which validation stage reports the failure.
#    If the generated kernel fails to compile, the agent is steered towards a
#    specific repair based on the error details provided by the compilation tool.
#    When the kernel fails functional validation, the agent is instructed to
#    retreat to a more conservative implementation, prioritizing correctness
#    above all else."
#   理由那一句才是重点: "When optimization is built on a kernel that looks
#   high-performance but is functionally flawed, subsequent refinement typically
#   remains trapped in the same faulty pattern."
#
# 我方此前的形状: 一个模板、一句话 («在编译 (L1) 或功能 (L2) 验证中失败»),
# 取向只有一条「修好它」, 从来没有许可过「退回去」。分支 buys 的是原文那条理由 ——
# 优化叠在有数值缺陷的核上会困在同一个错误模式里, 而**本轮谁优先**必须由 harness 规定。
#
# ⚠️ 末句「若错误信息里同时出现性能未达标」这**两句纠正过两次, 两次都是我没核入口**:
#   ① 原先写「`conductor.py:780` 把所有未通过结果交给 Fixer, 所以同一份 prompt 里会同时
#      有『误差超标』和『还不够快』」—— 读回 `verifier.verify()` (`:233-249`): 每级失败就
#      `return`, **fail-fast ⇒ 同轮最多只有一个未通过的级别**。
#   ② 改成「跨轮经 `history_cases` 并存」也不成立: Fixer 那条支路被
#      `conductor.py:763` 的 `l1_l2_failed` 门挡着, 而循环外那处兜底入库
#      (`:952-955`) 明写 `if any(level in (L1, L2))` —— **今天的代码一条 L3 记录都不产**。
#      (仓库现存 `output/failure_db.jsonl` 里那 2 条 L3 是更早版本/探针留下的, 现在没有
#       写者: 全仓 grep 不到任何生产代码写出这句 error_summary。)
#   所以这一句今天是**死文字**, 留着是因为它两条通道各自只被一条结构约束挡着, 而那两条
#   约束都各有一条守卫钉住 (`tests/unit/test_fixer_stage_branch.py`: `verify()` 早退的 AST 用例
#   + `conductor` 那道 `l1_l2_failed` 门的用例)。任一松掉, 性能文本就会进到取向要裁决的
#   地方, 这句是唯一说清「本轮不为它优化」的位置。这不是「算了没人读」的观测 —— 它是
#   一条明标的保险, 且它没有替任何未通过的判定说话。

_L2_RETREAT = (
    "本轮报失败的是**数值检查器** (L2): 这一版能编译、能跑, 但输出与参考实现不一致。"
    "取向: **correctness 优先于性能, 允许这一版比上一版慢**。做法是撤销可疑的激进改动"
    "(累加精度、向量化/tl.dot 的 input_precision、mask 与 other= 的取值、stride), "
    "回到最保守的正确写法, 而不是在同一个错误模式上再叠一个补丁。"
    "若错误信息里同时出现性能未达标 —— 本轮**不要**为此优化。"
)

# §19.3 修正 2 (Don't Regenerate, Debug, arXiv:2608.02712): 编译通过、仅数值失败时
# **修 > 重写**, 省 92.8% token, 且方法效应 (+20.4pp) 大于模型效应 (+13.0pp)。
# 这条比 _L2_RETREAT 更紧: 不仅「别叠补丁」, 而是**只许改错误邻域**。
_MINIMAL_FIX = (
    "本轮是**近失 (near-miss)**: 编译已通过、能跑, 只有数值/边界不对。"
    "取向: **最小改动修复, 禁止推倒重写**。只许改与错误直接相关的那几行"
    "(看 first_bad_index / 错误信息指到的公式、边界、索引), 其余一律不动。"
    "外部对照 (Don't Regenerate, Debug): 修比重写省 92.8% token, 且通过率更高。"
    "重写会把「还没被任何检查器看过的结构」丢掉, 用一次数值错误换一轮新的未知失败。"
)

_L1_INPLACE = (
    "本轮报失败的是**编译器 / 预检** (L1): 这一版还没有跑到能被判数值的地步。"
    "取向: **就地修这一处结构或 API 误用, 保持既有的优化策略不变**。"
    "编译失败不代表策略错, 退回保守写法只会丢掉已经有效的加速。"
)

# Fabrica (arXiv:2608.25124v1, 行 521-522) 的三路分流里第三路是 interface repair:
#   "The reviewer then routes architectural, implementation, and host/layout failures
#    to redesign, code repair, or interface repair, respectively."
# 我方今天这两路 (退让 / 就地修) 都是按**级别**分的, 而接口契约那一类失败在 L1 里
# 是最特殊的: 它不看代码内容, 只看 `def kernel(...)` 的名字与调用方要传的那几个。
# 按「就地修一处 API 误用」的口径去处理它, 模型常见两种走偏: 反过来改输入名 (那是任务侧
# 给的, 远端 `kernel(**输入)` 不会跟着改), 或整段重写 —— 把一份还没被任何检查器看过的
# 结构丢掉, 等于用一次签名错误换一轮新的未知失败。
# 基率 (doc 48 §六B): 16 轮里 4 轮死在这道门上, 且全部 L1 失败都归它 —— 这不是边角。
#
# 与那道门自己的文案的关系 (写之前核过 `interface_contract.py:184-197`): 门已经把
# 「缺哪几个、多哪几个」和「改 `def kernel` 的参数名」讲清楚了, 所以这段**不复述那份清单**;
# 它补的是门不知道的那一半 —— 本轮的**改动范围** (函数体/取值/配方一律不动) 和不动的理由。
_L1_INTERFACE = (
    "本轮报失败的是**入口契约门**: L1 里唯一只看名字、不看代码内容的一道, 由 harness 在动 "
    "GPU 之前拦下。函数体还没有被任何检查器看过, 不匹配的只有 `def kernel(...)` 的参数名 "
    "(缺哪几个、多哪几个, 见下面错误原文里「本次会传入的输入名」与「不匹配」那两行)。"
    "取向: **本轮只改 `def kernel` 那一行的参数名**。函数体、BLOCK_* 取值与已应用的优化配方"
    "一律不动 —— 签名错误不代表策略错, 而这一版的写法还没被验证过一次, 整段重写等于用一次"
    "已知的签名错误换一轮未知的失败。"
    "输入名不是你那一侧: 远端固定以 `kernel(**输入)` 全关键字调用, 输入名由任务给定。"
)


# KernelZero (arXiv:2609.33074v1, 行 147) 那条「先验 Spec」在本项目落成了两片的第 1 片
# (doc 58 §4.2): L2 现在显式算「比得上几个元素」, 参考自己产 NaN/Inf 时报「本级判不出」。
# 第 1 片只把**归因**写对了, 判决那一侧原样不变 —— 这句话仍然作为一条候选失败送进 Fixer,
# 于是拿到的是「比对报了不一致 ⇒ correctness 优先于性能, 撤销激进改动退回保守写法」,
# 而这一轮**对这张核没有作出任何判定**。一份指向错误对象的处方比沉默更坏: 模型照它改完,
# 下一轮仍然判不出, 而那一次改动已经把唯一一份还没被否证过的实现丢掉了。
_REFERENCE_HOLD = (
    "本轮停在 L2 那道比对上, 但报的原因不是「输出与参考不一致」: 参考自身在这批输入上"
    "没有一个元素可比 (全是 NaN/Inf) —— **本级没有可比对象**, 这一轮对这张核既没说它错、"
    "也没说它对。"
    "取向: **本轮不改你的核**。撤销激进改动、退回保守写法、换累加精度或 mask 与 other= "
    "的取值在这里都没有依据 —— 那些动作修的是「判出来的不一致」, 而这一轮没有不一致可修; "
    "把一份还没被否证过的实现换掉, 只是用一次未知替换掉一次已知。"
    "该查的是参考那一侧 (`reference_source` 或内建参考在这批输入上的算法), 那不是你能改、"
    "也不是本轮该改的东西。只有当下面的错误原文里另有与你代码直接相关的一条 (语法、缺 "
    "import、签名) 时, 才就地修那一条。"
)


def failure_class(failed_results) -> str:
    """本轮失败属于哪一类 —— 用**负样本库那一个分类器**, 不看文案措辞

    为什么不让取向自己判正则: 取向与库里的归类一旦用两套口径, 就会出现「prompt 说是接口
    问题、库里记成 TypeError」这种分裂 —— 而复发门、检索与本轮取向读的是同一个桶。
    入参口径与 `conductor._record_failure` 逐字一致: 排序去重的级别 + 每条
    `_truncate_error(…, 300)` 用 " | " 拼起来的文本。取的是签名三段里的**类名**那一段,
    整条签名拿去向 `"interface_mismatch"` 永远不相等 (层级与最内层帧也在里面) ——
    那会是一条写了走不到的路。
    """
    levels = "+".join(sorted({r.level.value for r in failed_results}))
    text = " | ".join(FixerAgent._truncate_error(r.error_log or r.message, 300)
                      for r in failed_results)
    return failure_class_of(levels, text)


def stage_directive(levels: set, failure_cls: str = "",
                    fix_not_rewrite: bool = False) -> str:
    """由「哪些级别报了失败」+「库里那个分类器怎么归类这次失败」得出本轮修复取向。

    四路 = Fabrica 的 interface repair / code repair / 保守退让 (arXiv:2608.25124v1
    行 521-522), 加第四路「本级没有判定」—— 那一路 Fabrica 没有, 因为它的 reviewer 假设
    被审的那一次运行是**一次观测**; 我方第 1 片 (#94) 之后 L2 会报「没有可比对象」,
    那一条不是候选的症状, 不该领任何针对候选的动作。
    归类排在级别之前: 接口契约那一类必然落在 L1 里, 只看级别会把它并进「就地修一处 API
    误用」那条通用取向, 而那条恰恰不限制改动范围。「本级判不出」又排在归类之前: 它同样是
    L2 里的一条, 只看级别 (甚至只看类名之外的文案) 都会把它并进「退回保守写法」那条。
    `fix_not_rewrite=True` (§8 路由的 near-miss) 压过上面的级别取向: 只许最小改动。
    不认识的归类 (含 `unknown`、真异常名) 一律退到级别那条老路径, 所以取向永不为空。
    """
    if failure_cls == FAILURE_CLASS_REFERENCE:
        return _REFERENCE_HOLD
    if fix_not_rewrite:
        return _MINIMAL_FIX
    if failure_cls == FAILURE_CLASS_INTERFACE:
        return _L1_INTERFACE
    return _L2_RETREAT if "L2" in levels else _L1_INPLACE


@dataclass(frozen=True)
class FixOutcome:
    """一次修复尝试的**完整**产物：候选 + 说明 + 模型交上来的动作信封。

    为什么信封要**随返回值**走、而不是存在实例属性上：本仓 I77 那一族缺陷的形状正是
    「留痕成了普通变量」—— 写进去、下一个读的人不知道该在什么时候读、于是它悄悄失效。
    随返回值走 ⇒ 「谁产它、谁读它」在类型上就写死了。

    `code` 为空**不是错误**：`envelope.asks_without_patch` 为真时这一轮是**提问**
    （解释 / 请求观测），而不是一次失败的修复 —— 两者在调用方要分开处理。
    """

    code: str
    summary: str
    envelope: ActionEnvelope
    candidate_records: tuple = ()

    @property
    def asked(self) -> bool:
        """这一轮是不是「不改候选」的提问。"""
        return not self.code and self.envelope.usable and self.envelope.asks_without_patch

    def as_dict(self) -> Dict[str, Any]:
        result = {"code_chars": len(self.code or ""), "summary": self.summary,
                  "asked": self.asked, "envelope": self.envelope.as_dict()}
        if self.candidate_records:
            result["candidate_records"] = [candidate_summary(record)
                                           for record in self.candidate_records]
        return result


class FixerAgent:
    """
    Fixer Agent (错误修复)
    职责: 当 L1/L2 验证失败时，分析错误原因并生成修复后的代码
    基于 LLM 驱动修复
    """

    MAX_ERROR_CHARS = 1500  # 单条错误日志注入 prompt 的最大长度
    MAX_FIX_ATTEMPTS = 3    # 修复代码的预检重试次数
    #: 修复那一次 LLM 调用的 token 上限。**唯一一份**：§21.1 的动作预算声明读的就是它
    #: （`token_budget`）。写成两处（一处调用、一处声明）会让「声明 1000、实际花 4096」
    #: 这种漂移无法被发现, 而那道门看起来永远能过。
    FIX_MAX_TOKENS = 4096

    def __init__(self, llm=None, prompt_manager: Optional[PromptManager] = None,
                 failure_db=None):
        self.llm = llm
        self.pm = prompt_manager or PromptManager()
        self.failure_db = failure_db  # 失败案例库 (可选): 检索历史修复经验
        self.name = "Fixer"
        # 修复代码时必须遵守与 Coder 相同的 SKILL 禁令/必需模式
        self._agent_role = "coder"
        self._skill_name = "triton_skill"
        self.candidate_records: list[dict] = []
        self.last_candidate_record = None

    def _remember_candidate(self, context, source, *, request, attempt, status,
                            parent_source, reason=None):
        record = make_candidate_record(context, source, request=request,
            attempt=attempt, status=status, source_author="model",
            parent_source=parent_source, reason=reason)
        self.candidate_records.append(record)
        self.last_candidate_record = record
        return record

    def run(
        self,
        code: str,
        task: TaskSchema,
        failed_results: list[VerificationResult],
        fix_not_rewrite: bool = False,
        structured_evidence: str = "",
        error_shape: str = "",
        ask_answer: str = "",
        diagnosis: Optional[dict] = None,
        diagnosis_results: Optional[list[VerificationResult]] = None,
        test_inputs=None,
    ) -> tuple[str, str]:
        """`run_with_action` 的**两格投影**（唯一的差别是丢掉动作信封）。

        保留这个入口有两个理由：① 与接线前逐字同形，于是「接线有没有改修复行为」
        可以只差一个返回值就判开；② 不关心 §21.1 的调用方（与既有用例）不必知道信封。
        守卫钉住「它就是投影」：同一个替身 LLM 下 `run(...) == run_with_action(...)` 的前两格。
        """
        outcome = self.run_with_action(
            code, task, failed_results, fix_not_rewrite=fix_not_rewrite,
            structured_evidence=structured_evidence, error_shape=error_shape,
            ask_answer=ask_answer,
            diagnosis=diagnosis,
            **({'diagnosis_results': diagnosis_results} if diagnosis_results is not None else {}),
            **({'test_inputs': test_inputs} if test_inputs is not None else {}),
        )
        return outcome.code, outcome.summary

    def run_with_action(
        self,
        code: str,
        task: TaskSchema,
        failed_results: list[VerificationResult],
        fix_not_rewrite: bool = False,
        structured_evidence: str = "",
        error_shape: str = "",
        ask_answer: str = "",
        diagnosis: Optional[dict] = None,
        diagnosis_results: Optional[list[VerificationResult]] = None,
        test_inputs=None,
    ) -> "FixOutcome":
        """修复失败代码 (带预检+重试闭环，和 CoderAgent._llm_generate_code 一致)

        `fix_not_rewrite=True` (§8 路由: 编译过、仅数值失败): 把取向钉成
        **最小改动** —— 外部实测 (Don't Regenerate, Debug) 修比重写省 92.8% token,
        且方法效应大于模型效应。此时 prompt 不许推倒重来。

        `structured_evidence` (§21.3): Conductor 现算并渲染好的**结构化证据**那一段
        (契约摘要 / 父候选 diff / 失败分类 / 首个错误与形态 / profiler 事实 / 可用工具 /
        剩余预算 / 可写路径)。空串 = 这一次不给这一段 (A/B 对照的关闭侧)。
        ⚠️ 本方法**不自己拼**这一段: 拼装只有 `AgentRuntime.context_for(...).render()`
        一处 —— 在这里再拼一遍就是第二个注入通道, 而「只有这八项」正是靠只有一个拼装点
        成立的。

        **动作信封 (§21.1 的生产者侧)**: 回复里可以带一段 ```agent-action 的意图
        (种类 / 理由 / 预期瓶颈 / 想用的工具)。解析在这里做、**并且在这里就把这一段从
        文本里切掉** —— 于是代码提取看到的只有源码, 信封里写什么都不可能被当成候选。
        读不出来 (`absent` / `malformed`) 不是错误: 沿用接线前的行为 (由 harness 拼装动作)。

        关键防御:
        - 当 task.target.dsl != "triton" (昇腾等非 NVIDIA 硬件) 时, 不注入 triton_skill
          否则 LLM 看到 SKILL 里的 tl.* 指令会继续输出 Triton -> 编译无限失败
        - 修复生成后走与 Coder 同一套 _validate_code 预检 (含"非 triton 禁止 Triton API")
          有问题就带反馈最多重试 MAX_FIX_ATTEMPTS 次
        """
        self.candidate_records = []
        self.last_candidate_record = None
        if self.llm is None:
            raise RuntimeError("LLM 未配置，无法执行修复")
        if diagnosis is not None:
            from ..generate.repair_diagnoser import validate_candidate_diagnosis
            evidence = failed_results if diagnosis_results is None else diagnosis_results
            if [r for r in evidence if not r.passed] != failed_results:
                raise ValueError('candidate diagnosis failure-set drift')
            validate_candidate_diagnosis(diagnosis, code, task, evidence)

        self._last_task = task
        self._last_failed_results = failed_results
        self._last_original_code = code

        error_details = self._format_failures(failed_results)
        # Reflexion 增强: 先按 kernel_type/dsl 上下文取, 再用 error_summary 相似度补充
        history_cases = self._build_history_cases(task, failed_results)

        # ── SKILL 选择策略 ──
        effective_skill = (
            self._skill_name if (task.target.dsl or "").lower() == "triton" else None
        )
        dsl = task.target.dsl
        context = build_generation_context(task, test_inputs=test_inputs,
            producer="Fixer", strategy="repair", parent_source=code,
            policy=dict(max_attempts=self.MAX_FIX_ATTEMPTS,
                        max_output_tokens=self.FIX_MAX_TOKENS,
                        fix_not_rewrite=bool(fix_not_rewrite),
                        diagnosis_hash=diagnosis.get("hash") if diagnosis is not None else None),
            prompt_manager=self.pm, skill_name=effective_skill,
            agent_role=self._agent_role)
        facts = context["task"]

        # 渲染前两样都要算好: 级别进模板当「本轮是谁报的失败」, 归类当第三路的分流键,
        # 取向由这两者一起决定。归类在这里算而不是让 `stage_directive` 自己去看文案:
        # 它走的是 `conductor._record_failure` 那一个分类器, 于是 prompt 里的取向与负样本
        # 库里的归类必然是同一个桶。
        levels = ", ".join(r.level.value for r in failed_results)
        level_set = {r.level.value for r in failed_results}
        failure_cls = failure_class(failed_results)

        base_prompt = self.pm.render(
            "fixer_repair",
            _agent_role=self._agent_role,
            _skill_name=effective_skill,
            kernel_type=facts["kernel_type"],
            hardware=facts["hardware"],
            dsl=dsl,
            precision=facts["precision"],
            failure_stage=levels or "未知",
            stage_directive=stage_directive(level_set, failure_cls,
                                            fix_not_rewrite=fix_not_rewrite),
            error_details=error_details,
            history_cases=history_cases or "无",
            code=code,
            structured_evidence=structured_evidence,
            error_shape=error_shape or "(这一轮没有确立误差形态)",
            ask_answer=ask_answer,
        )
        if diagnosis is not None:
            from ..generate.repair_diagnoser import DIAGNOSED_REPAIR_INSTRUCTION
            base_prompt += ('\n\n## Source-bound repair diagnosis\n'
                            + DIAGNOSED_REPAIR_INSTRUCTION
                            + json.dumps(diagnosis, ensure_ascii=False, allow_nan=False))

        feedback = ""
        fixed = ""
        succeeded = False
        envelope = ActionEnvelope(status=ENVELOPE_ABSENT, reason="一次 LLM 调用都没成功")
        asked = False
        for attempt in range(1, self.MAX_FIX_ATTEMPTS + 1):
            messages = [{"role": "user", "content": base_prompt + feedback}]
            request = generation_request(self.llm, messages, caller="Fixer",
                temperature=0.1, max_tokens=self.FIX_MAX_TOKENS)
            try:
                raw = self.llm.chat(
                    messages,
                    temperature=0.1,
                    max_tokens=self.FIX_MAX_TOKENS,
                    caller_tag="Fixer",
                )
            except Exception as exc:
                self._remember_candidate(context, None, request=request, attempt=attempt,
                    status="model_failed", parent_source=code, reason=type(exc).__name__)
                raise
            # 先取信封再提代码: 信封那一段**被切掉**了, 于是代码提取只看得到源码
            try:
                envelope, body = parse_action_envelope(raw)
                fixed = self._extract_code(body)
            except Exception as exc:
                self._remember_candidate(context, None, request=request, attempt=attempt,
                    status="no_candidate", parent_source=code, reason=type(exc).__name__)
                raise
            if envelope.usable and envelope.asks_without_patch:
                # 「不改候选」的那几类 (解释 / 请求观测)。**这一轮没有候选** ——
                # 不重试、也不抛: 它不是一次失败的修复, 是一次提问。响应它 (答/拒)
                # 是 Conductor 的事 (`_answer_ask`), 而它决定要不要再问一次。
                asked = True
                if fixed:
                    print(f"  [Fixer] 信封是 {envelope.kind}, 同时带了代码 —— "
                          f"按声明**不收**这份代码 (借请求之名改代码正是 §21.1 要挡的)")
                fixed = ""
                self._remember_candidate(context, None, request=request, attempt=attempt,
                    status="no_candidate", parent_source=code, reason="ask_without_patch")
                break
            if not fixed:
                issues = ["Fixer 未能从 LLM 回复中提取有效代码"]
            else:
                try:
                    _, issues = self._validate_code(fixed, dsl)
                except Exception as exc:
                    self._remember_candidate(context, fixed, request=request, attempt=attempt,
                        status="precheck_failed", parent_source=code, reason=type(exc).__name__)
                    raise
            self._remember_candidate(context, fixed or None, request=request, attempt=attempt,
                status="prechecked" if not issues else ("precheck_failed" if fixed else "no_candidate"),
                parent_source=code, reason=None if not issues else (
                    "static_precheck_failed" if fixed else "missing_code"))
            if not issues:
                succeeded = True
                if attempt > 1:
                    print(f"  [Fixer] 第 {attempt} 次修复通过预检")
                break
            print(f"  [Fixer] 第 {attempt} 次修复预检未通过: {'; '.join(issues)}")
            # 预检问题同样给处方: 这些正是表里"截断/结构/跨硬件禁令"三条的产出方,
            # 只把 issue 原文丢回去等于让模型自己猜预检想听什么
            bullets = []
            for i in issues:
                bullets.append(f"- {i}")
                fx = prescribe(i)
                if fx is not None:
                    bullets.append(f"  要求: {fx[0]}\n  该改哪里: {fx[1]}")
            feedback = (
                f"\n\n## 上次修复未通过预检 (尝试 {attempt}/{self.MAX_FIX_ATTEMPTS})\n"
                + "\n".join(bullets) + "\n"
                f"请重新输出完整修复代码: 以 `import` 开头，包含 kernel 入口函数，"
                f"括号配对完整；除 `根因:` 那几行外不要输出解释；"
                f"特别注意当前 dsl={dsl!r} 的 API 禁令。"
            )
        if not fixed and not asked:
            raise ValueError("Fixer 未能从 LLM 回复中提取有效代码")

        if asked:
            # 提问轮: 不进 Reflexion 上下文 (`_last_fixed_code` 留空 ⇒ `produce_reflexion`
            # 返回 None) —— 那一份是「修复成功后的教训」, 而这一轮根本没有候选。
            summary = (f"基于 {len(failed_results)} 个失败结果提问 "
                       f"({levels}): {envelope.kind}")
            print(f"  [Fixer] 收到动作信封 {envelope.kind} (无候选): "
                  f"{envelope.rationale[:80] or '(未写理由)'}")
            return FixOutcome(code="", summary=summary, envelope=envelope,
                              candidate_records=tuple(copy.deepcopy(self.candidate_records)))

        summary = f"基于 {len(failed_results)} 个失败结果修复 ({levels})"
        print(f"  [Fixer] LLM 修复完成: {len(fixed)} chars")
        # 仅当修复通过预检时才留下可 Reflexion 的上下文 (避免"假修复"入库)
        if succeeded:
            self._last_fixed_code = fixed
            self._last_fix_summary = summary
        return FixOutcome(code=fixed, summary=summary, envelope=envelope,
                          candidate_records=tuple(copy.deepcopy(self.candidate_records)))

    def produce_reflexion(self) -> Optional[FailureRecord]:
        """Reflexion: 修复后让 LLM 生成结构化自我反思并封装为 FailureRecord。

        返回 FailureRecord (带 lessons/error_keywords/root_cause) 供上游 Conductor 写入
        FailureDatabase 沉淀经验。若无 last_* 上下文或 LLM 未配置则返回 None。

        ⚠️ `fixed` 这里**不填** (留 dataclass 默认的 False): 被调用的那一刻下一轮验证还没跑,
        而字段的意思是「该失败**最终**是否被修复」。原先它无条件写 True —— 于是每条反思都在
        结果未知时替结果写了答案, 而检索侧 (kernel_db 的两处排序) 就按「已修复」优先用它。
        补写由 Conductor 在任务结束时做 (那时才知道结果)。
        """
        if not hasattr(self, '_last_fixed_code') or not self._last_fixed_code:
            return None
        if self.llm is None:
            return None

        task = getattr(self, '_last_task', None)
        failed = getattr(self, '_last_failed_results', [])
        original = getattr(self, '_last_original_code', '')
        if task is None:
            return None

        error_details = self._format_failures(failed, prescriptive=False)
        levels = "+".join(sorted({r.level.value for r in failed})) or "L?"

        prompt = self.pm.render(
            "fixer_reflexion",
            _agent_role="coder",
            _skill_name=None,  # 反思不注入 skill
            kernel_type=task.source.kernel_type,
            hardware=task.target.hardware,
            dsl=task.target.dsl,
            error_details=error_details,
            fix_summary=self._last_fix_summary,
        )

        try:
            result = self.llm.chat_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
                caller_tag="Fixer.Reflexion",
            )
        except Exception as e:
            print(f"  [Fixer] Reflexion LLM 调用失败, 跳过沉淀: {e}")
            return None

        # 构造 FailureRecord。原先这里写的是「固定 failure_id, 避免重复写入」, 却在 id
        # 末尾拼了 `uuid4().hex[:4]` —— 每次反思都生成一个新 id, 同一份教训因此可以无限
        # 重复入库, 那句注释描述的正是代码没有做的事。现在 id 完全由内容决定
        # (任务 + 签名), 同一任务同一类失败重放/重写都是覆盖同一条。
        import time
        signature = failure_signature(levels, error_details)

        lessons = result.get("lessons") or []
        if isinstance(lessons, str):
            lessons = [lessons.strip()]
        lessons = [str(x).strip() for x in lessons if str(x).strip()]

        kw = result.get("error_keywords") or []
        if isinstance(kw, str):
            kw = [k.strip() for k in re.split(r"[,;，；]", kw) if k.strip()]
        kw = [str(x).strip() for x in kw if str(x).strip()]

        hw = result.get("applicable_hardware") or [task.target.hardware]
        if isinstance(hw, str):
            hw = [x.strip() for x in re.split(r"[,;，；]", hw) if x.strip()]
        hw = [str(x).strip() for x in hw if str(x).strip()]

        return FailureRecord(
            failure_id=f"rx-{task.task_id}:{signature}",
            kernel_type=task.source.kernel_type,
            hardware=task.target.hardware,
            dsl=task.target.dsl,
            level=levels,
            error_summary=error_details[:500],
            fix_summary=self._last_fix_summary,
            created_at=time.time(),
            root_cause=str(result.get("root_cause", "")).strip()[:300],
            lessons=lessons[:10],
            error_keywords=kw[:15],
            applicable_hardware=hw[:10],
            signature=signature,
            task_id=task.task_id,
        )

    @staticmethod
    def _validate_code(code: str, dsl: str = "python") -> tuple:
        """
        修复后预检 —— 与 CoderAgent._validate_code 保持完全一致的规则集,
        确保修复后的代码至少能过静态关卡 (截断/语法/入口/跨硬件 API 禁令)
        """
        if not code or not code.strip():
            return False, ["代码为空"]
        issues = []

        for open_ch, close_ch in (("(", ")"), ("[", "]"), ("{", "}")):
            if code.count(open_ch) != code.count(close_ch):
                issues.append(f"括号 {open_ch}{close_ch} 不配对 (疑似输出截断)")
                break

        stripped = code.rstrip()
        if stripped.endswith(("(", ",", ":", "\\", "+", "-", "*", "/", "=")):
            issues.append("代码结尾悬空 (疑似输出截断)")

        if "def kernel" not in code:
            issues.append("缺少 kernel 入口函数")

        # 跨硬件 API 禁令: dsl != "triton" 时禁止 Triton (昇腾等非 NVIDIA 新硬件核心防线)
        if (dsl or "").lower() != "triton":
            if "import triton" in code or "from triton" in code or "@triton.jit" in code:
                issues.append(
                    f"禁止: dsl={dsl!r} 但代码里出现了 Triton API (import triton / @triton.jit 等)。"
                    " 昇腾/其他非 Nvidia 硬件必须使用纯 Python/PyTorch/NumPy，不能用 Triton/CUDA。"
                )

        if (dsl or "").lower() == "triton" and "@triton.jit" not in code:
            issues.append("缺少 @triton.jit 装饰器")

        dsl_lower = (dsl or "").lower()
        if dsl_lower == "cuda_c":
            # C++ Extension 预检: 必须含 load_inline 和 C++ 源码
            if "load_inline" not in code:
                issues.append("cuda_c 代码缺少 load_inline 调用 (PyTorch C++ Extension JIT 编译)")
            if "_CPP_SOURCE" not in code and "cpp_sources" not in code:
                issues.append("cuda_c 代码缺少 C++ 源码 (_CPP_SOURCE 或 cpp_sources)")
        elif dsl_lower in ("python", "triton"):
            try:
                compile(code, "<generated>", "exec")
            except SyntaxError as e:
                issues.append(f"语法错误: {e.msg} (第 {e.lineno} 行)")

        return (not issues, issues)

    def _build_history_cases(self, task: TaskSchema,
                             failed_results: Optional[list[VerificationResult]] = None) -> str:
        """从失败案例库检索相似历史案例 (已修复案例优先)，注入 prompt。

        Reflexion 增强:
          1. search_by_context(kernel_type/dsl/hardware) 取基础同类案例
          2. 若有 failed_results, search_by_error(error_text) 再取 "错误词元相似" 案例
          3. 去重合并后, 每条案例输出 error_summary + fix_summary + Reflexion lessons
        """
        if not self.failure_db:
            return ""
        base = self.failure_db.search_by_context(
            kernel_type=task.source.kernel_type,
            dsl=task.target.dsl,
            hardware=task.target.hardware,
            limit=2,
        )
        error_matches: list = []
        if failed_results:
            err_text = "\n".join(r.error_log or r.message for r in failed_results)
            error_matches = self.failure_db.search_by_error(
                err_text,
                kernel_type=task.source.kernel_type,
                dsl=task.target.dsl,
                limit=2,
            )
        # 合并去重 (按 failure_id)
        seen = set()
        merged: list = []
        for r in base + error_matches:
            if r.failure_id in seen:
                continue
            seen.add(r.failure_id)
            merged.append(r)
        if not merged:
            return ""
        lines = []
        for i, c in enumerate(merged):
            tag = ("[错误相似度命中]" if c in error_matches else "[上下文命中]")
            lines.append(f"历史案例{i + 1} {tag}: [{c.level}] {c.error_summary}")
            if c.root_cause:
                lines.append(f"  根因: {c.root_cause}")
            if c.fix_summary:
                lines.append(f"  修复方式: {c.fix_summary}")
            if c.lessons:
                lines.append("  可复用教训:")
                for l in c.lessons[:5]:
                    lines.append(f"    - {l}")
            if c.applicable_hardware:
                lines.append(f"  适用硬件: {', '.join(c.applicable_hardware)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 错误信息组装与压缩
    # ------------------------------------------------------------------

    def _format_failures(self, failed_results: list[VerificationResult],
                         prescriptive: bool = True) -> str:
        """将失败结果格式化为注入 prompt 的错误描述 (日志已截断)

        `prescriptive=False` 给 Reflexion 用: 那一步只是回看这次失败, 把处方混进去会
        让沉淀的教训变成抄我们的模板; 而且这段文本直接喂给 `failure_signature()`,
        固定归因前言会让不同错误的签名互相靠拢 —— 签名要分的是「哪一类失败」,
        不是「我们的 prompt 长什么样」。

        「本级判不出」那一类同样不走归因前言 (`_ATTRIBUTION`) 与处方 (`_prescribe_lines`):
        前言断言「下面每一条都是…实测到的现象」, 而这一轮的现象是**没有发生任何一次判定**;
        处方那条兜底文案 («只依据上面的执行侧原话定位根因») 更直接 —— 它要模型在自己的
        代码里找一个并不存在的根因。归类在这里重算而不是由调用方传标志: 同一个纯函数、
        同样入参, 多传一个标志只多出一个「两处的标志可能不一致」的面。
        """
        if not failed_results:
            return _ATTRIBUTION if prescriptive else "无详细错误信息"

        hold = failure_class(failed_results) == FAILURE_CLASS_REFERENCE
        parts = []
        for r in failed_results:
            lines = [f"[{r.level.value}] {r.message}"]
            if r.error_log:
                lines.append("   执行侧原话:" if prescriptive else "错误日志:")
                lines.append(self._truncate_error(r.error_log))
            if prescriptive:
                if hold:
                    lines.append(_HOLD_NOTE)
                else:
                    lines.extend(self._prescribe_lines(r))
            parts.append("\n".join(lines))

        if not prescriptive:
            return "\n\n".join(parts)
        head = [_ATTRIBUTION] if not hold else []
        return "\n\n".join(head + ["── " + p for p in parts])

    @staticmethod
    def _prescribe_lines(r: VerificationResult) -> list[str]:
        # 用未截断的全文匹配: 签名可能正好落在 _truncate_error 省略掉的中间段
        fx = prescribe(f"{r.message}\n{r.error_log or ''}")
        if fx is None:
            return ["   要求: (无现成处方 —— 这不是已知的失败形态, 只依据上面的执行侧"
                    "原话定位根因, 不要假设项目之外的限制)"]
        return [f"   要求: {fx[0]}", f"   该改哪里: {fx[1]}"]

    @classmethod
    def _truncate_error(cls, log: str, max_chars: Optional[int] = None) -> str:
        """
        截断过长的错误日志: 保留头部 (首个 error 上下文) 与尾部 (最终报错行)，
        中间以省略标记连接，避免长编译错误栈撑爆上下文
        """
        if not log:
            return ""
        limit = max_chars or cls.MAX_ERROR_CHARS
        log = log.strip()
        if len(log) <= limit:
            return log
        head = limit // 2
        tail = limit - head
        omitted = len(log) - limit
        return f"{log[:head]}\n... [省略 {omitted} 字符] ...\n{log[-tail:]}"

    # ------------------------------------------------------------------
    # 代码提取 (语言无关，不做语义改写)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_code(raw: str) -> str:
        """
        从 LLM 回复中提取代码:
        1. 剥离 thinking 标签
        2. 优先提取 ``` 代码块
        3. 无代码块时定位代码起始行 (import/from/@triton.jit/def/#include)
        返回空串表示提取失败 (由调用方决定重试或终止)
        """
        code = raw.strip()

        for end_tag in ("</think>", "</thinking>"):
            idx = code.find(end_tag)
            if idx != -1:
                code = code[idx + len(end_tag):].strip()
                break

        m = re.search(
            r"```(?:python|triton|cpp|cuda|c\+\+)?\s*\n(.*?)```",
            code,
            re.DOTALL | re.IGNORECASE,
        )
        if m:
            # 去除代码块整体缩进 (LLM 常把 fence 放在列表/引用内导致行首带缩进)
            return textwrap.dedent(m.group(1)).strip()

        lines = code.split("\n")
        start = None
        for i, line in enumerate(lines):
            if re.match(r"^(import |from |@triton\.jit|def |#include)", line.strip()):
                start = i
                break
        if start is None:
            return ""

        body = "\n".join(lines[start:])
        if body.rstrip().endswith("```"):
            body = body.rstrip()[:-3]
        return textwrap.dedent(body).strip()

    def __repr__(self) -> str:
        return f"FixerAgent(llm={self.llm is not None})"
