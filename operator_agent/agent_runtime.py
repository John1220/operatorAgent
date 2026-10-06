"""§21 Agent Runtime：结构化动作、动作权限、结构化证据上下文、进编译前的独立检查

设计 `docs/design/autonomous-operator-agent-design.md` §21 的四条在这里各有实现物：

| 设计节 | 本模块的实现 |
|---|---|
| §21.1 Agent 只提交结构化动作 | `AgentAction` / `Patch` + `AgentRuntime.admit` 的三道校验（schema / 权限 / 预算） |
| §21.2 结构性变更与局部修复的边界 | `ActionContext` + `permission_problems`（路由 → 允许的动作种类） |
| §21.3 上下文只使用结构化证据 | `ProposalContext` + `render()`（白名单 + 截断留痕） |
| §21.4 进编译之前的独立检查 | `precompile_checks()`（七条 → 六条判据，映射见 `DESIGN_CHECKS`）+ `AuditCoverage` |

## 为什么要有这个模块（而不是继续「LLM 直接改字符串」）

今天的链路上，LLM 的产物是**一整份源码字符串**：Coder 返回 str、Fixer 返回 str、
Conductor 把 str 送进 L1/L2/L3。这条形状能跑，但它让设计 §21 的三件事无处落脚：

1. **可审计**：没有「谁、因为什么、把父代的哪一份改成了什么」这个对象，产物里只剩
   最后一轮的代码；
2. **权限**：拒绝一份候选要有一个**判据面**（哪一条不过、凭什么是这一条），
   否则拒绝只能靠调用方自觉；
3. **覆盖**：七条检查里今天有量具的和没有量具的必须分得开 —— 「查不了」不许被读成
   「没问题」（本仓 I65 那条纪律）。

## 三条刻意保守的口径（误拦截比误放行贵，但「没查」不许当「通过」）

* **只有 `violation` 才拒绝**。`unknown` / `not_applicable` 是**结论**，进产物、进报告，
  但不改判决 —— 它们表示「这条判据今天对这份候选没有量具」，不是「它有问题」。
  两者的方向相反：把 `unknown` 读成拒绝会让整条生成路径停摆，读成通过就是 fail-open。
* **判据不重造**。七条里能复用的全部复用现成的单一真源：接口契约用
  `interface_contract.check_kernel_interface`、重复用 `novelty.signature_of`、
  依赖用 `environment.DependencyAnalyzer`、瓶颈量具用 `escalation.GAUGES`、
  能力声明用 `backend_caps.Cap`。本模块**只**新增两件今天没有的东西：
  `Patch`（候选改动的对象）与路径范围判据。
* **今天没有的东西明写**。`spill-bound` / `bank-conflict` 没有量具、
  `profiler` / `shape_sweep` 没有消费者、源码级「幻觉 API」没有 API 面清单 ——
  三处都进 `AuditCoverage.uncovered_rules` / 动作拒绝理由，而不是留空。

## 接在哪一点上（位置是判据的一部分）

`Conductor._execute_inner` 的 **Step 4 之前**：所有候选来源（Coder / 模板实例化器 / Fixer /
决赛选择）都已经汇合成 `current_code`，而**任何** GPU 秒与 SSH 往返都还没花。放到 producer
那一侧会漏掉三条路，放进 `Verifier` 则会把「动作协议」塞进评测器。

被拒的后果是**定义好的**：本轮不进 L1/L2/L3，由 Conductor 合成一条 L1 失败
（`metrics["compile_status"] == "agent_action_refused"`，`agent_action_refused=True`），
于是 §8 的路由与 Fixer 的处方照旧工作 —— 拒绝不是「静默丢掉这个候选」。
依赖分析那一段会跳过被拒的候选（它根本没进编译，为它跑一次远端往返是纯浪费）。

⚠️ 「进编译之前」指的是**进验证链路的编译之前**：生成侧流水线自己还有一次本地编译剪枝
（`CoderPipeline._gate_check` 的采样筛选，§9 的便宜代理），那是采样器的事，不在本层管辖内
—— 本层管的是「这一轮的候选要不要被提交上去」，那是唯一的提交点。

⚠️ **位置对三道校验的含义不同**：路径 / ABI / 依赖 / 重复 / 能力这五轴拒绝发生在 GPU 秒
之前，省下的是机时；**token 预算那一轴拒绝省下的是 0**（候选是「先生成、后提交」，裁决时
那次 `chat` 已经付过钱）。所以预算那一格由 `ActionContext.spend_committed` 分档：事前动作
（请求工具 / 结构升级）默认 `False` ⇒ 拒绝；候选类动作在真链路上写 `True` ⇒ 预算问题照记
进产物但不据此拒绝（否则会把 §10.2「超了退化成纯数值搜索」打成死循环，见 CHECKLOG I76）。

## 本模块**不做**的事（边界必须一并读）

* **它不是沙箱**。§22 的目录权限、进程/资源隔离、编译与运行超时都不在这里；
  本模块判的是**声明的路径**，真正的权限位是 §22（今天未落地 ⇒ 没有内核级强制）。
  路径判据因此有一个诚实的身份：**判据成立、输入端今天只有声明**。
* **它不生成候选**，也不改候选；拒绝之后走哪条路（回退父代 / 转修复）由 Conductor 决定。
* **它不写事件日志**（§23.1 的 append-only 日志是另一片）；本模块的产物是
  `ActionVerdict.as_dict()`，由调用方落进 `iteration_history`。
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from . import escalation, interface_contract, novelty, tolerance
from .run_layout import (PROTECTED_DIR_NAMES as _LAYOUT_PROTECTED_DIR_NAMES,
                         CANDIDATE_DIR as _LAYOUT_CANDIDATE_DIR,
                         layout_path as layout_path_of)
from .backend_caps import Cap
from .environment import DependencyAnalyzer
from .source_lang import COVERAGE_NONE, classify

# ─── 1. 闭合词汇（§21.1） ─────────────────────────────────────

#: 动作种类。**设计 §21.1 那一份**，一个字都不许自己加：加一个词要同时给出
#: 「谁产它」与「谁读它」，否则它就是死词汇（与 `backend_caps.Cap` 同一条纪律）。
KIND_PROPOSE_TEMPLATE = "propose_template"
KIND_PROPOSE_SOURCE_PATCH = "propose_source_patch"
KIND_REPAIR_COMPILE = "repair_compile"
KIND_REPAIR_CORRECTNESS = "repair_correctness"
KIND_REQUEST_PROFILER = "request_profiler"
KIND_REQUEST_NEW_SHAPE = "request_new_shape"
KIND_EXPLAIN_FAILURE = "explain_failure"

ACTION_KINDS: Tuple[str, ...] = (
    KIND_PROPOSE_TEMPLATE,
    KIND_PROPOSE_SOURCE_PATCH,
    KIND_REPAIR_COMPILE,
    KIND_REPAIR_CORRECTNESS,
    KIND_REQUEST_PROFILER,
    KIND_REQUEST_NEW_SHAPE,
    KIND_EXPLAIN_FAILURE,
)

#: 会**改变候选**的那几类：必须带 `Patch`。
PATCH_KINDS: Tuple[str, ...] = (
    KIND_PROPOSE_TEMPLATE, KIND_PROPOSE_SOURCE_PATCH,
    KIND_REPAIR_COMPILE, KIND_REPAIR_CORRECTNESS,
)
#: 只请求观测/只解释的那几类：**不许**带 `Patch`（带了就是「借解释之名改代码」）。
SIDE_EFFECT_FREE_KINDS: Tuple[str, ...] = (
    KIND_REQUEST_PROFILER, KIND_REQUEST_NEW_SHAPE, KIND_EXPLAIN_FAILURE,
)
#: 局部修复（§21.2 的「修 > 重写」那一侧）。
REPAIR_KINDS: Tuple[str, ...] = (KIND_REPAIR_COMPILE, KIND_REPAIR_CORRECTNESS)
#: 可能做结构改动的那几类（§10.3 的三条件门禁管的就是它们）。
STRUCTURAL_KINDS: Tuple[str, ...] = (KIND_PROPOSE_SOURCE_PATCH,)

#: 设计 §21.2 的边界形态（尾块 / 整行 / 整列）：这几类**允许回退父代**，
#: 而不是硬修回一个可能越修越远的版本。`None`（形态未确立）不在此列 ——
#: 判不了形态时不许声称「这是边界错」，与 `failure_route` 同一条纪律。
BOUNDARY_ERROR_SHAPES: Tuple[str, ...] = (
    tolerance.ERROR_SHAPE_TAIL_BLOCK,
    tolerance.ERROR_SHAPE_WHOLE_ROW,
    tolerance.ERROR_SHAPE_BROADCAST_AXIS,
)

# ─── 2. 工具注册表（「引用不存在的能力」的判据面） ──────────────

TOOL_DEPENDENCIES = "analyze_dependencies"
TOOL_VERIFY_NUMERICS = "verify_numerics"
TOOL_MEASURE_SAME_DEVICE = "measure_same_device"
TOOL_PROFILER = "profiler"
TOOL_SHAPE_SWEEP = "shape_sweep"

#: 可用工具 → 兑现它的**后端能力**（`None` = 今天没有实现物）。
#: 这是 §21.4 第 7 条「引用不存在的能力」在本层的判据：动作点名一个 `None` 那一档的工具，
#: 或者后端 `capabilities` 里没有那个 `Cap`，都在**零 GPU 时间**上判拒。
#: `None` 的两档各有出处，不许当成「还没填」：
#:   * `profiler` —— 全仓无 NCU/nsight 量具（`grep profiler` 只命中一处注释）；
#:   * `shape_sweep` —— L3 只测 `test_inputs[0]`（设计 §3.4 的缺件，状态文档有登记）。
#: 工具的开销**不进** token 预算：profiler / 测量花的是机时，那是 §10.2 的
#: oracle 次数预算管的货币，两本账不合并（合并会让「钱」这个词有两个含义）。
TOOLS: Dict[str, Optional[Cap]] = {
    TOOL_DEPENDENCIES: Cap.DEP_ANALYSIS,
    TOOL_VERIFY_NUMERICS: Cap.REMOTE_VERIFY,
    TOOL_MEASURE_SAME_DEVICE: Cap.SAME_DEVICE_BASELINE,
    TOOL_PROFILER: None,
    TOOL_SHAPE_SWEEP: None,
}

#: 动作种类**自带**的工具需求（§21.1 的那两个 request_* 词就是这件事）。
KIND_REQUIRED_TOOLS: Dict[str, Tuple[str, ...]] = {
    KIND_REQUEST_PROFILER: (TOOL_PROFILER,),
    KIND_REQUEST_NEW_SHAPE: (TOOL_SHAPE_SWEEP,),
}

# ─── 3. §21.4 的七条 → 本模块的六条判据 ───────────────────────

#: 设计 §21.4 的原文清单（按序）→ 兑现它的判据名。
#: 为什么是 7 → 6 而不是 7 个判据：清单的第 1、2 条是同一件事的两种说法
#: （「写入范围」与「只修改授权路径」在**单文件候选**上是同一个包含关系），
#: 把它们写成两个判据只会让人以为有两道防线。第 3 条的「契约有没有被篡改」那一半
#: 已有归属（`contract.assert_hash` 每轮校验、`oracle` 拒评），本层只补路径那一半。
DESIGN_CHECKS: Tuple[Tuple[str, str], ...] = (
    ("文件写入范围", "write_scope"),
    ("patch 是否只修改授权路径", "write_scope"),
    ("contract / harness / baseline 是否被修改", "protected_paths"),
    ("依赖是否新增", "new_dependency"),
    ("ABI 是否变化", "abi"),
    ("候选是否与父代重复", "duplicate_of_parent"),
    ("是否引用不存在的能力", "capability"),
)

#: 判据名的闭合词汇（`DESIGN_CHECKS` 的值集必须**恰好**是它）。
CHECK_RULES: Tuple[str, ...] = (
    "write_scope", "protected_paths", "new_dependency", "abi",
    "duplicate_of_parent", "capability",
)

#: §22.1 的受保护目录名：候选**不能**写进去（§21.4 第 3 条点名前三个）。
#: 判据按**路径的每一段**比 —— `x/report/y.py` 与 `report/y.py` 一样命中。
#:
#: ⚠️ 这**不是**本模块自己的一份名单：判据与 §22.1 的目录权限位读同一个元组
#: （定义在 `run_layout.py`，这里再导出）。两份名单迟早出现「判据说 protected、
#: 文件系统说 writable」或反过来 —— 那正是 §22.1 要求「违反时写入必须失败」而不是
#: 「事后被审计发现」的理由：**审计和执行必须同源**。守卫断言两边是同一个对象。
PROTECTED_DIR_NAMES: Tuple[str, ...] = _LAYOUT_PROTECTED_DIR_NAMES

#: §22.1 的运行布局里「Agent 只写」的那一格。
CANDIDATE_DIR = _LAYOUT_CANDIDATE_DIR


def candidate_path(run_root: str = "", ext: str = "py") -> str:
    """候选源码的角色路径 / 真路径（§21 ↔ §22.1 的接缝）

    * 没有 `run_root`：返回**角色路径** `candidate/code.py` —— §21 的动作协议今天
      活在这一档（判据成立，输入端只有「声明」）；
    * 有 `run_root`：返回**真实绝对路径** `<run_root>/candidate/code.py`，与
      `run_layout(...).path("candidate/code.py")` 逐字相同（同一个路径构造点）。

    以前这里是 `"/".join(... strip("/") ...)` —— 它把绝对路径的根悄悄吃掉
    （`/a/b` → `a/b/candidate/code.py`，根那一段被吃掉），所以「把真实 run_root 传进来即可、
    判据一个字都不用改」这句话当时**不成立**；现在由布局模块唯一地构造路径。
    """
    ext = ext.lstrip(".")
    if not str(run_root or "").strip():
        return f"{CANDIDATE_DIR}/code.{ext}"
    return layout_path_of(run_root, f"{CANDIDATE_DIR}/code.{ext}")


#: 判据的四种结论。**三态 + 违约**，四者不许压成布尔：
#:   * `ok`            这条判据在这份候选上跑过，没有违规；
#:   * `violation`     跑过、且违规（唯一会拒绝的一档）；
#:   * `not_applicable`这条判据对这份候选**没有对象/没有量具**；
#:   * `unknown`       有对象但**判不了**（缺输入），既不是通过也不是不适用。
STATE_OK = "ok"
STATE_VIOLATION = "violation"
STATE_NOT_APPLICABLE = "not_applicable"
STATE_UNKNOWN = "unknown"
CHECK_STATES: Tuple[str, ...] = (STATE_OK, STATE_VIOLATION,
                                 STATE_NOT_APPLICABLE, STATE_UNKNOWN)

#: 上下文里每段的字符上限（§21.3「不要把完整历史日志无条件塞入上下文」）。
#: 超限**截断并留痕**，不静默 —— 截断这件事本身有读者（「模型看到的错误信息是不是
#: 只剩开头 200 字」）。
MAX_ERROR_CHARS = 600
MAX_DIFF_CHARS = 2000
MAX_CONTRACT_CHARS = 1200
MAX_PROFILER_KEYS = 12


# ─── 4. Patch：候选改动的对象（今天没有的那个东西） ────────────


@dataclass(frozen=True)
class Patch:
    """一次候选改动 = 目标路径 + 新源码 + 父代源码

    为什么不是「diff 文本」：这条链路上从来没有人产出过 diff。LLM 给的是**整份源码**
    （`CoderAgent` / `FixerAgent` / 模板实例化器都是），而「改了什么」是可以**当场算出来**的
    派生量 —— 反过来要求生产者自己写 diff，就等于让模型用它最不擅长的形式交付。
    于是：审计与回滚需要的那两个量（`diff()`、`parent_source`）在这里派生，**不由生产者声明**。

    `parent_source` 是回滚点。空串 = 没有父代（首轮生成），此时「回滚」无从谈起，
    `rollback_source()` 返回 `None` 而不是空源码 —— 「没有父代」与「父代是空文件」必须分得开。
    """

    path: str
    new_source: str
    parent_source: str = ""

    @property
    def has_parent(self) -> bool:
        return bool(self.parent_source)

    def source_hash(self) -> str:
        """候选源码的内容哈希（前 16 位）。跨语言都能算（与 `candidates.source_hash` 同口径）"""
        return hashlib.sha256(self.new_source.encode("utf-8")).hexdigest()[:16]

    def parent_hash(self) -> str:
        return (hashlib.sha256(self.parent_source.encode("utf-8")).hexdigest()[:16]
                if self.parent_source else "")

    def diff(self, *, context_lines: int = 2) -> str:
        """unified diff（审计用）。没有父代 ⇒ 空串（不是「与空文件比」的巨大 diff）"""
        if not self.parent_source:
            return ""
        return "".join(difflib.unified_diff(
            self.parent_source.splitlines(keepends=True),
            self.new_source.splitlines(keepends=True),
            fromfile=f"a/{self.path}", tofile=f"b/{self.path}",
            n=context_lines,
        ))

    def changed_lines(self) -> Tuple[int, int]:
        """(新增行数, 删除行数)。三态无关：没有父代 ⇒ (0, 0)（等于「没有可比对象」）"""
        if not self.parent_source:
            return (0, 0)
        added = removed = 0
        for line in self.diff(context_lines=0).splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                added += 1
            elif line.startswith("-") and not line.startswith("---"):
                removed += 1
        return (added, removed)

    def rollback_source(self) -> Optional[str]:
        """回滚点。**`None` 表示没有父代可回** —— 不许被读成「回滚到空候选」"""
        return self.parent_source if self.parent_source else None

    def as_dict(self) -> Dict[str, Any]:
        added, removed = self.changed_lines()
        return {"path": self.path, "source_hash": self.source_hash(),
                "parent_hash": self.parent_hash(), "has_parent": self.has_parent,
                "added_lines": added, "removed_lines": removed}


# ─── 5. AgentAction：Agent 唯一能提交的东西（§21.1） ────────────


@dataclass(frozen=True)
class AgentAction:
    """设计 §21.1 的那个结构。字段与顺序**照抄设计**，只多了 `as_dict`。

    与设计的唯一一处改写：`requested_tools` / `expected_bottleneck` 的取值都被**闭合词汇**
    管住（工具见 `TOOLS`，瓶颈见 `escalation.gauged_bottlenecks()`）。设计说它们分别是
    「可用工具列表」与「预期瓶颈」，而一个不在表里的名字在今天的链路上等于一句
    「引用不存在的能力」—— 那正是 §21.4 第 7 条要在零 GPU 时间上拦下的东西。
    """

    action_id: str
    kind: str
    rationale: str
    parent_candidate: Optional[str] = None
    patch: Optional[Patch] = None
    expected_bottleneck: Optional[str] = None
    requested_tools: Tuple[str, ...] = ()
    token_budget: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id, "kind": self.kind,
            "parent_candidate": self.parent_candidate,
            "rationale": self.rationale,
            "expected_bottleneck": self.expected_bottleneck,
            "requested_tools": list(self.requested_tools),
            "token_budget": self.token_budget,
            "patch": self.patch.as_dict() if self.patch is not None else None,
        }


@dataclass(frozen=True)
class ActionContext:
    """裁决一个动作需要的**全部**输入。每个字段都可缺 —— 缺就是「这一条判不了」

    ⚠️ 这里刻意**没有**「历史日志」这一类字段。§21.3 的八项就是上下文的全集；
    想加一项必须先回答「它进 prompt 的哪一段、谁产它」，否则它就是「把日志塞进去」。
    """

    #: `failure_route.classify(...).kind`：`"compile"` / `"numeric"` / `"performance"` /
    #: `"infra"` / `"silent"`。`""` = 这一轮**没有失败在驱动它**（首轮生成 / 搜索驱动的重生成）。
    route_kind: str = ""
    #: §8.2 的误差形态；`None` = 判不了（三态，不许被读成「没有形态」）。
    error_shape: Optional[str] = None
    #: `failure_route` 给的那条纪律（编译过、仅数值失败 ⇒ 修不重写）。
    fix_not_rewrite: bool = False
    #: §10.3 三条件升级门**最近一次**的裁决（结构性变更的许可证）。
    escalation_allowed: bool = False
    escalation_refusals: Tuple[str, ...] = ()
    #: 上一轮 L1 是否判过「接口与调用契约不符」（`compile_status == interface_mismatch`）。
    #: 与判据 `abi` 分工不同：那一条判**这份 patch**，这一格记**已知的坏状态**。
    abi_broken: bool = False
    #: 授权写入根（`Patch.path` 必须落在这里面）。空 = 没有声明 ⇒ 判据报 `unknown`。
    authorized_roots: Tuple[str, ...] = ()
    #: 调用方这一轮会以关键字传进去的输入名，**每组一份**
    #: （`interface_contract.input_key_sets` 的产物）。空 = 没有可判的对象 ⇒ `abi` 报
    #: `not_applicable`，而不是「签名对得上」。分组而不是取并集：某一组少传一个必填参数
    #: 同样会 TypeError，取并集恰好把这一组漏掉（与 L1 逐组判的做法一致）。
    input_key_sets: Tuple[Tuple[str, ...], ...] = ()
    #: 这份源码声明的语言/DSL（`""` = 没声明，由 `source_lang` 推一条）。
    language: str = ""
    #: 已验证父代可用吗（§21.2 的「回退父代」那一半的可选出路）。
    parent_verified: bool = False
    #: 剩余 LLM token；`None` = 没设上限（不许被读成「用完了」）。
    tokens_left: Optional[int] = None
    #: **这次动作的钱花了没有**。它是「拒绝能买到什么」的判据：
    #: `False`（事前提议：请求工具 / 结构升级）⇒ 预算不过就拒绝，**省得下钱**；
    #: `True`（事后提交：本仓的候选是「先生成、后提交」）⇒ 预算问题照记但**不据此拒绝**
    #: —— 钱已经花了，拒绝只能毁掉一份已经付过钱、且可能通过评测的候选，
    #: 而 §10.2 给这一档的后果是「退化成纯数值搜索」，不是杀进程。
    spend_committed: bool = False
    #: 「新增依赖」的**基线源码**。空 = 用 patch 自己的父代（`Patch.parent_source`）。
    #:
    #: 为什么这一条要独立于 `Patch.parent_source`: 「新增」的本意是「这份候选往流程里
    #: 塞了一个之前没进过的依赖」——而**被拒的候选从没进过流程**，它不该成为基线。
    #: 真机一手（CHECKLOG I78）: 修复产物夹带 `import torch`，第 2 轮被拒（零远端调用），
    #: 但第 3 轮 Fixer 又交了同一份 —— 那时 patch 的父代是**上一轮那份被拒的候选**，
    #: 于是 `torch` 不再是「新增」，门放行、真机编译真的发生了（L1 通过，最后靠 L2 的
    #: 反作弊静态审计才拦住）。基线改成「上一次**过门**的候选」之后，同一份违规候选
    #: 每一轮都被拒，远端调用恒为 0。
    dep_baseline_source: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "route_kind": self.route_kind, "error_shape": self.error_shape,
            "fix_not_rewrite": self.fix_not_rewrite,
            "escalation_allowed": self.escalation_allowed,
            "escalation_refusals": list(self.escalation_refusals),
            "abi_broken": self.abi_broken,
            "authorized_roots": list(self.authorized_roots),
            "input_key_sets": [list(s) for s in self.input_key_sets],
            "language": self.language,
            "parent_verified": self.parent_verified, "tokens_left": self.tokens_left,
            "spend_committed": self.spend_committed,
            "dep_baseline_source": self.dep_baseline_source,
        }


# ─── 5b. 动作信封：LLM 交上来的那一份（§21.1 的生产者侧） ─────────
#
# 在接线之前，`AgentAction` 是**harness 把候选包出来的**：模型只交一整份源码字符串，
# `kind` / `rationale` / `expected_bottleneck` / `requested_tools` 全由 harness 填。
# 那让 §21.1 的七个动作种类里有两个（`request_profiler` / `request_new_shape`）
# 与一个（`explain_failure`）**没有生产者** —— 判据与读者都在岗，没人产它们。
# 这一节就是那个生产者：模型在源码之外先交一段**意图**，harness 把它接到同一套判据上。
#
# 两条刻意划下的界线（不划就会被读成「模型可以自己发权限」）：
#   * 模型写**意图**（要做什么、为什么、想用什么工具），harness 写**数字与路径**
#     （`patch.path` 由 §22.1 布局给，`token_budget` 与真正花掉的那个上限同源）——
#     交给模型写预算，就是让被判的人自己填判据里的数；
#   * 信封**读不出来**与**读出来了但判据不过**是两件事：前者退回 harness 拼装的动作
#     （`absent` / `malformed`），后者进 §21.1 的拒绝路径（`parsed` + 判据红）。
#     压成一个布尔，「模型没按协议交」与「模型交了一个越权的动作」在产物里长得一样。

#: 信封的围栏标签。刻意与代码围栏（```python）分开：解析器先把这一段**切掉**，
#: 代码提取只看剩下的文本 ⇒ 信封里出现任何东西都不会被当成候选源码。
ENVELOPE_FENCE = "agent-action"

#: 信封三态（见上）。
ENVELOPE_PARSED = "parsed"
ENVELOPE_ABSENT = "absent"
ENVELOPE_MALFORMED = "malformed"
ENVELOPE_STATUSES: Tuple[str, ...] = (ENVELOPE_PARSED, ENVELOPE_ABSENT, ENVELOPE_MALFORMED)

#: 动作是**谁**写的。进血缘，于是「这一轮的 rationale 是模型写的还是 harness 拼的」
#: 事后可判 —— 没有这一格，两件事在产物里长得一样（本仓 I77 那一族）。
ACTION_SOURCE_LLM = "llm"
ACTION_SOURCE_HARNESS = "harness"
ACTION_SOURCES: Tuple[str, ...] = (ACTION_SOURCE_LLM, ACTION_SOURCE_HARNESS)

#: 信封里**认识**的键。多出来的键不静默丢弃：进 `unknown_keys` 并存进血缘
#: （模型以为自己能声明 `path` 却被忽略，是「协议没写清」的证据，不是噪音）。
ENVELOPE_KEYS: Tuple[str, ...] = (
    "kind", "rationale", "expected_bottleneck", "requested_tools", "token_budget",
)

#: 信封的围栏（**恰好一段**；多于一段就是「不猜哪一段是它」）。
_ENVELOPE_RE = re.compile(
    r"```[ \t]*" + re.escape(ENVELOPE_FENCE) + r"[ \t]*\r?\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


@dataclass(frozen=True)
class ActionEnvelope:
    """模型交上来的**动作意图**（§21.1 的生产者侧）。它**不是** `AgentAction`。

    差的那两格是刻意的：`patch.path`（写哪个文件由 §22.1 的布局给，今天只有一个产者
    `candidate_path()`）与 `token_budget`（判它的数必须与真正花掉的同源）。信封只有意图。
    """

    status: str
    reason: str = ""
    kind: str = ""
    rationale: str = ""
    expected_bottleneck: Optional[str] = None
    requested_tools: Tuple[str, ...] = ()
    declared_token_budget: Optional[int] = None
    unknown_keys: Tuple[str, ...] = ()
    raw: str = ""

    @property
    def usable(self) -> bool:
        """能不能拿它去构造动作。`absent` / `malformed` 都退回 harness 拼装。"""
        return self.status == ENVELOPE_PARSED

    @property
    def kind_known(self) -> bool:
        """种类在不在闭合词汇表里。**不在也算 `parsed`** —— 那是判据的事，不是解析的事。"""
        return self.kind in ACTION_KINDS

    @property
    def asks_without_patch(self) -> bool:
        """这是「不改候选」的那一类意图（解释 / 请求观测）。"""
        return self.kind in SIDE_EFFECT_FREE_KINDS

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status, "reason": self.reason, "kind": self.kind,
            "rationale": self.rationale,
            "expected_bottleneck": self.expected_bottleneck,
            "requested_tools": list(self.requested_tools),
            "declared_token_budget": self.declared_token_budget,
            "unknown_keys": list(self.unknown_keys),
        }


def parse_action_envelope(text: str) -> Tuple[ActionEnvelope, str]:
    """从模型的回复里取动作信封，返回 `(信封, 去掉信封后的文本)`。

    第二个返回值是**必须**的：调用方要在剩下的文本上做代码提取，而信封里可能有
    任何东西（包括看起来像代码的行）。让两件事共用一个正则 = 迟早把信封当候选。

    三条判据，顺序固定：
      1. 没有围栏段 ⇒ `absent`（**不是**错：接线前模型就是这么回话的）；
      2. 多于一段 ⇒ `malformed`（不猜哪一段是它）；
      3. 有且只有一段 ⇒ 按 `ENVELOPE_KEYS` 严格解析：JSON 错 / 不是对象 / `kind` 缺失
         或不是非空串 / 某个已知键的**类型**不对 ⇒ 整段 `malformed`。**半信半疑地取
         一半字段比整段退回更难审计**，所以类型不对就是整段不认。
    种类**不在**闭合词汇表里不算解析失败（`parsed`）：模型说了它想干什么，只是那件事
    不被允许 ⇒ 让 §21.1 的判据去拒，并留下「被拒的理由」。这就是判据第一次遇到一个
    真正不可靠的生产者。
    """
    raw = str(text or "")
    blocks = _ENVELOPE_RE.findall(raw)
    body = _ENVELOPE_RE.sub("", raw)
    if not blocks:
        return ActionEnvelope(status=ENVELOPE_ABSENT,
                              reason=f"回复里没有 ```{ENVELOPE_FENCE} 段"), body
    if len(blocks) > 1:
        return ActionEnvelope(status=ENVELOPE_MALFORMED,
                              reason=f"回复里有 {len(blocks)} 段 "
                                     f"```{ENVELOPE_FENCE} —— 不猜哪一段是它"), body
    block = blocks[0].strip()
    try:
        payload = json.loads(block)
    except ValueError as exc:
        return ActionEnvelope(status=ENVELOPE_MALFORMED,
                              reason=f"信封不是合法 JSON: {exc}", raw=block), body
    if not isinstance(payload, dict):
        return ActionEnvelope(status=ENVELOPE_MALFORMED,
                              reason=f"信封不是 JSON 对象而是 {type(payload).__name__}",
                              raw=block), body
    unknown = tuple(sorted(k for k in payload if k not in ENVELOPE_KEYS))
    kind = payload.get("kind")
    if not isinstance(kind, str) or not kind.strip():
        return ActionEnvelope(
            status=ENVELOPE_MALFORMED, raw=block, unknown_keys=unknown,
            reason="信封里没有 kind —— 没说它要做什么的信封读不出意图"), body
    rationale = payload.get("rationale", "")
    if not isinstance(rationale, str):
        return ActionEnvelope(status=ENVELOPE_MALFORMED, raw=block, unknown_keys=unknown,
                              reason=f"rationale 必须是字符串: {type(rationale).__name__}"), body
    bottleneck = payload.get("expected_bottleneck")
    if bottleneck is not None and not isinstance(bottleneck, str):
        return ActionEnvelope(status=ENVELOPE_MALFORMED, raw=block, unknown_keys=unknown,
                              reason="expected_bottleneck 必须是字符串或 null"), body
    tools = payload.get("requested_tools", [])
    if not isinstance(tools, list) or any(not isinstance(t, str) for t in tools):
        return ActionEnvelope(status=ENVELOPE_MALFORMED, raw=block, unknown_keys=unknown,
                              reason="requested_tools 必须是字符串数组"), body
    budget = payload.get("token_budget")
    if budget is not None and (isinstance(budget, bool) or not isinstance(budget, int)):
        return ActionEnvelope(status=ENVELOPE_MALFORMED, raw=block, unknown_keys=unknown,
                              reason="token_budget 必须是整数或 null"), body
    return ActionEnvelope(
        status=ENVELOPE_PARSED, kind=kind.strip(), rationale=rationale.strip(),
        expected_bottleneck=(bottleneck.strip() if isinstance(bottleneck, str)
                             and bottleneck.strip() else None),
        requested_tools=tuple(t.strip() for t in tools if t.strip()),
        declared_token_budget=budget, unknown_keys=unknown, raw=block,
    ), body


# ─── 6. 裁决产物 ───────────────────────────────────────────────


@dataclass(frozen=True)
class CheckOutcome:
    """一条判据在**这一份候选**上的结论。`state` 取 `CHECK_STATES` 之一。"""

    rule: str
    state: str
    detail: str = ""

    def __post_init__(self) -> None:
        if self.rule not in CHECK_RULES:
            raise ValueError(f"判据名不在词汇表里: {self.rule!r} (表 {CHECK_RULES})")
        if self.state not in CHECK_STATES:
            raise ValueError(f"判据结论不在词汇表里: {self.state!r} (表 {CHECK_STATES})")

    @property
    def is_violation(self) -> bool:
        return self.state == STATE_VIOLATION

    def as_dict(self) -> Dict[str, Any]:
        return {"rule": self.rule, "state": self.state, "detail": self.detail}


@dataclass(frozen=True)
class AuditCoverage:
    """§22.3 的形状：审计的覆盖情况是**数据**，不是「有/没有问题」

    §22.3 原本是给「不同语言/形态的静态检查能力不同」写的（Python AST / 正则 / 完全没有），
    这里先按同一个形状把**进编译前这七条**的覆盖表达出来 —— 于是「这一层今天能判什么」
    不需要读者去读代码，`covered_rules` / `uncovered_rules` 两个集合就是答案。

    ⚠️ `mode` 说的是**源码**那一半的覆盖强度（`ast` / `regex_only` / `none`），
    它**不**代表每条判据的适用性 —— 后者在 `PrecompileVerdict.outcomes` 里逐条给出。
    """

    language: str
    mode: str
    covered_rules: frozenset
    uncovered_rules: frozenset

    def as_dict(self) -> Dict[str, Any]:
        return {"language": self.language, "mode": self.mode,
                "covered_rules": sorted(self.covered_rules),
                "uncovered_rules": sorted(self.uncovered_rules)}


@dataclass(frozen=True)
class PrecompileVerdict:
    """§21.4 的七条在这一份候选上的全部结论 + 覆盖

    `outcomes` 的顺序与 `DESIGN_CHECKS` 一致（报告按设计原文的序读）。
    `coverage` 由 `outcomes` 派生 —— 不另存一份，免得两处答案漂移。
    """

    outcomes: Tuple[CheckOutcome, ...]
    language_verdict: Any = None            #: `source_lang.LangVerdict`；None = 没判（无源码）
    design_items: Tuple[Tuple[str, str], ...] = DESIGN_CHECKS

    @property
    def violations(self) -> Tuple[CheckOutcome, ...]:
        return tuple(o for o in self.outcomes if o.is_violation)

    @property
    def accepted(self) -> bool:
        """只有**违约**才拒绝。`unknown` / `not_applicable` 是结论，不是拒绝理由。"""
        return not self.violations

    def state_of(self, rule: str) -> Optional[str]:
        return next((o.state for o in self.outcomes if o.rule == rule), None)

    @property
    def coverage(self) -> AuditCoverage:
        lang = self.language_verdict
        covered = {o.rule for o in self.outcomes
                   if o.state in (STATE_OK, STATE_VIOLATION)}
        uncovered = {o.rule for o in self.outcomes
                     if o.state in (STATE_NOT_APPLICABLE, STATE_UNKNOWN)}
        if lang is None:
            mode, name = COVERAGE_NONE, ""
        else:
            mode, name = lang.coverage, lang.language
        return AuditCoverage(language=name, mode=mode,
                             covered_rules=frozenset(covered),
                             uncovered_rules=frozenset(uncovered))

    def as_dict(self) -> Dict[str, Any]:
        return {"accepted": self.accepted,
                "outcomes": [o.as_dict() for o in self.outcomes],
                "violations": [o.detail for o in self.violations],
                "coverage": self.coverage.as_dict()}


@dataclass(frozen=True)
class ActionVerdict:
    """一次动作裁决的完整产物。§21.1 的三道校验 + §21.4 的独立检查，**都留痕**"""

    action: AgentAction
    accepted: bool
    schema_problems: Tuple[str, ...] = ()
    permission_problems: Tuple[str, ...] = ()
    budget_problems: Tuple[str, ...] = ()
    #: 裁决时看到的那份上下文。**必须进产物**：只留「拒绝了」而不留「当时看到什么」，
    #: 复盘时无法区分「判据拦住了一个坏候选」与「上下文装配错了」。
    context: Optional[ActionContext] = None
    #: `None` = **没跑**（schema 没过 / 没有 patch）—— 不是「跑过了没问题」。
    precompile: Optional[PrecompileVerdict] = None
    #: 结构性变更有没有拿到 §10.3 的许可证（只观测：放行的那条路也要看得出来源）。
    structural_licensed: bool = False
    #: token 预算的问题**报了但没有据此拒绝**时的说明（`""` = 没有这种事）。
    #: 读者：复盘「这一轮钱花超了、候选还是进了评测」时要能看出那是**定义好的**行为，
    #: 而不是门坏了。见 `ActionContext.spend_committed`。
    budget_deferred_note: str = ""
    #: 回退父代是不是一条可选出路（§21.2 的「或回退父代」那一半）。今天只是**证据**：
    #: 走不走是 Conductor 的决定，不由本模块替它走。
    fallback_available: bool = False

    @property
    def problems(self) -> Tuple[str, ...]:
        """所有拒绝理由，按校验层次排列（读者是日志与修复提示）"""
        return self.schema_problems + self.permission_problems + self.budget_problems \
            + tuple(o.detail for o in (self.precompile.violations if self.precompile else ()))

    def refusal_note(self) -> str:
        return "; ".join(self.problems)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.as_dict(),
            "accepted": self.accepted,
            "context": self.context.as_dict() if self.context is not None else None,
            "schema_problems": list(self.schema_problems),
            "permission_problems": list(self.permission_problems),
            "budget_problems": list(self.budget_problems),
            "precompile": self.precompile.as_dict() if self.precompile else None,
            "structural_licensed": self.structural_licensed,
            "budget_deferred_note": self.budget_deferred_note,
            "fallback_available": self.fallback_available,
        }


def ask_unavailable_reason(kind: str) -> str:
    """一个「不改候选」的动作今天**答不答得出**。答得出 ⇒ 空串；答不出 ⇒ 原因。

    它读的是与 §21.4 第 7 条**同一个**注册表（`KIND_REQUIRED_TOOLS` × `TOOLS`），
    这不是重复实现而是同源：两处各写一份「profiler 没有实现物」，哪天 profiler 落地了，
    必然有一处忘了改 —— 于是「请求了却没有」与「请求了就有」会同时成立。
    守卫钉住同源：对每个 `SIDE_EFFECT_FREE_KINDS`，`ask_unavailable_reason` 判空
    **当且仅当** §21.4 的 `_check_capability` 不在这个动作上报工具/瓶颈的问题。

    答案是**一句事实**（不是布尔）：答不出来这件事必须回到 prompt 里，
    否则模型会以为「观测结果是空」（两者在产物里长得一样 —— 本仓 I65 那条纪律）；
    而「这条请求答不出来」那种对模型说的话由响应方去拼（`Conductor._answer_ask_and_retry`），
    判据详情里只留事实句。
    """
    missing = [tool for tool in KIND_REQUIRED_TOOLS.get(kind, ()) if TOOLS.get(tool) is None]
    if not missing:
        return ""
    # 只回**事实句**，「这条请求答不出来」那种对模型说的话由响应方去拼
    # （判据详情里出现「答不出来」会让读产物的人以为这是给人看的答复）。
    return (f"{kind} 需要的工具 {', '.join(missing)} 今天没有实现物 "
            f"(TOOLS 注册表里是 None)")


def contract_digest(contract: Any) -> str:
    """§21.3 第一项「当前契约摘要」。**派生**自冻结契约本身, 不另立一份口径

    为什么是派生而不是手写一段: 尺子的内容只有 `FrozenContract.content_dict()` 一个真源
    (`contract.py`), 而 prompt 里那句摘要是**模型看到的尺子** —— 两处手写必然漂, 漂的
    方向恰好是「prompt 里写的容差比评测用的宽」。这里把 `content_dict()` 里与「生成侧要
    遵守什么」有关的那几项摊平, 并带上**短哈希**: 模型能看出这一轮是哪一份契约在管它,
    复盘时也能把 prompt 与判据对上是同一份。

    `contract` 为 `None` 时返回空串 (调用方按「未提供」处理, 由 `render()` 说出来)。
    """
    if contract is None:
        return ""
    try:
        d = contract.content_dict()
    except AttributeError:
        return ""
    sem = d.get("dtype_semantics") or {}
    base = d.get("baseline") or {}
    shapes = d.get("shapes") or []
    shape_txt = ", ".join(
        f"{'x'.join(str(i) for i in s)}@{w:g}" for s, w in shapes[:4]) or "未声明"
    caps = "、".join(d.get("required_caps") or ()) or "无"
    stamp = getattr(contract, "hash", "") or ""
    if not stamp:
        try:
            stamp = contract.compute_hash()
        except Exception:            # noqa: BLE001 — 算不出就明说算不出, 不许编一个
            stamp = ""
    return (
        f"kernel_type={d.get('kernel_type')} · dtype={sem.get('dtype')}"
        f"(acc={sem.get('accum')}, tf32={sem.get('tf32')})\n"
        f"max_error={d.get('max_error')} · min_speedup={d.get('min_speedup')}"
        f" · target_metric={d.get('target_metric')}\n"
        f"baseline={base.get('name')}(同设备={base.get('same_device')}, "
        f"同窗口={base.get('same_window')})\n"
        f"shapes={shape_txt} · required_caps={caps}\n"
        f"契约短哈希={stamp[:12] or '(算不出)'}"
    )


# ─── 7. §21.3 的结构化证据上下文 ───────────────────────────────


@dataclass(frozen=True)
class ProposalContext:
    """每次 LLM 调用**只**提供这八项（设计 §21.3）。字段名 → 设计原文：

    | 字段 | §21.3 |
    |---|---|
    | `contract_summary` | 当前契约摘要 |
    | `parent_diff` | 父候选 diff |
    | `last_failure_kind` | 最近一次失败分类 |
    | `first_error` / `error_shape` | 首个错误与误差形态 |
    | `profiler_metrics` | 相关 profiler 指标 |
    | `available_tools` | 可用工具列表 |
    | `budget_left` | 剩余预算 |
    | `writable_paths` | 允许修改的文件 |

    渲染由 `render()` 负责，它是**白名单**：这个方法里没有第九个字段，也就没有第二个
    注入通道 —— 「不要把完整历史日志无条件塞入上下文」在这里是一条结构性质，
    而不是一句要靠调用方自觉的约定。
    """

    contract_summary: str = ""
    parent_diff: str = ""
    last_failure_kind: str = ""
    first_error: str = ""
    error_shape: Optional[str] = None
    profiler_metrics: Mapping[str, Any] = field(default_factory=dict)
    available_tools: Tuple[str, ...] = ()
    budget_left: Optional[int] = None
    writable_paths: Tuple[str, ...] = ()

    #: 被截断的段（读者：复盘「模型看到的证据是不是只剩开头」）
    def truncations(self) -> Tuple[str, ...]:
        out = []
        if len(self.first_error) > MAX_ERROR_CHARS:
            out.append("first_error")
        if len(self.parent_diff) > MAX_DIFF_CHARS:
            out.append("parent_diff")
        if len(self.contract_summary) > MAX_CONTRACT_CHARS:
            out.append("contract_summary")
        if len(self._scalar_metrics()) > MAX_PROFILER_KEYS:
            out.append("profiler_metrics")
        return tuple(out)

    def _scalar_metrics(self) -> Dict[str, Any]:
        """只留标量。指标是个 dict / list 时**整条不渲染**（它是日志，不是读数）"""
        out = {}
        for k, v in (self.profiler_metrics or {}).items():
            if isinstance(v, bool) or isinstance(v, (int, float, str)):
                out[str(k)] = v
        return out

    @staticmethod
    def _clip(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return (text[:limit] +
                f"\n… (已截断: 原文 {len(text)} 字符, 只给了前 {limit} 字符)")

    def render(self) -> str:
        """渲染成可以直接进 prompt 的一段文字。**永远非空**（没有证据时也说出来）"""
        metrics = self._scalar_metrics()
        lines = ["## 结构化证据 (由 harness 注入, 不是历史日志)"]

        lines.append("\n### 当前契约摘要\n" +
                     (self._clip(self.contract_summary, MAX_CONTRACT_CHARS)
                      if self.contract_summary else "(未提供 —— 按任务描述里的契约来)"))
        lines.append("\n### 父候选 diff\n" +
                     (self._clip(self.parent_diff, MAX_DIFF_CHARS)
                      if self.parent_diff else "(没有父代: 这是首次提议, 没有可比对象)"))
        lines.append("\n### 最近一次失败分类\n" +
                     (self.last_failure_kind or "(还没有失败: 没有可分类的对象)"))
        err = self._clip(self.first_error, MAX_ERROR_CHARS) if self.first_error else \
            "(没有错误可给: 这不是一次失败驱动的动作)"
        shape = self.error_shape or "(误差形态未确立 —— 不许猜一个)"
        lines.append(f"\n### 首个错误与误差形态\n{err}\n误差形态: {shape}")
        if len(metrics) > MAX_PROFILER_KEYS:
            shown = dict(list(metrics.items())[:MAX_PROFILER_KEYS])
            dropped = len(metrics) - MAX_PROFILER_KEYS
        else:
            shown, dropped = metrics, 0
        body = ("\n".join(f"- {k}: {v}" for k, v in shown.items())
                if shown else "(没有相关指标: 本轮不是测量驱动的动作)")
        if dropped:
            body += f"\n… (+{dropped} 项未渲染 —— 全量指标塞进上下文是负收益)"
        lines.append("\n### 相关 profiler 指标\n" + body)
        lines.append("\n### 可用工具列表\n" +
                     (", ".join(self.available_tools) if self.available_tools
                      else "(一个都没有: 后端没有声明任何能力)"))
        budget = "未设上限" if self.budget_left is None else str(self.budget_left)
        lines.append(f"\n### 剩余预算\nLLM token: {budget}")
        lines.append("\n### 允许修改的文件\n" +
                     ("\n".join(f"- {p}" for p in self.writable_paths)
                      if self.writable_paths else "(没有声明授权路径)"))
        return "\n".join(lines) + "\n"

    def as_dict(self) -> Dict[str, Any]:
        """进产物的形状。**与 render 同源**（同一份字段），但数字不被截断。

        `truncated_sections` 是刻意的一位：上下文被截断这件事在产物里必须有名字，
        否则「模型看到的证据」与「我们手里的证据」是两份东西而没人知道。
        """
        return {"contract_summary": self.contract_summary,
                "parent_diff": self.parent_diff,
                "last_failure_kind": self.last_failure_kind,
                "first_error": self.first_error,
                "error_shape": self.error_shape,
                "profiler_metrics": dict(self.profiler_metrics or {}),
                "available_tools": list(self.available_tools),
                "budget_left": self.budget_left,
                "writable_paths": list(self.writable_paths),
                "truncated_sections": list(self.truncations())}


# ─── 8. 校验与判据 ─────────────────────────────────────────────


def _norm(path: str) -> str:
    """规范化一条路径用于**包含关系**判定。反斜杠统一成正斜杠（产物里也这么记）"""
    return os.path.normpath(str(path).replace("\\", "/")).replace("\\", "/")


def _under_any(path: str, roots: Sequence[str]) -> bool:
    p = _norm(path)
    for root in roots or ():
        r = _norm(root)
        if not r:
            continue
        if p == r or p.startswith(r.rstrip("/") + "/"):
            return True
    return False


def schema_problems(action: AgentAction) -> Tuple[str, ...]:
    """§21.1 的第一道：形状。**不碰任何后端、不花一秒机时**"""
    out = []
    if not str(action.action_id or "").strip():
        out.append("action_id 为空 —— 一个没有身份的动作无法被审计或回滚")
    if action.kind not in ACTION_KINDS:
        out.append(f"动作种类不在闭合词汇表里: {action.kind!r} (表 {ACTION_KINDS})")
    if not str(action.rationale or "").strip():
        out.append("rationale 为空 —— 没有理由的动作不入账 (可审计的前提)")
    if isinstance(action.token_budget, bool) or not isinstance(action.token_budget, int) \
            or action.token_budget <= 0:
        out.append(f"token_budget 必须是正整数: {action.token_budget!r}")
    for tool in action.requested_tools or ():
        if tool not in TOOLS:
            out.append(f"请求的工具不在注册表里: {tool!r} (表 {tuple(TOOLS)})")
    if action.kind in PATCH_KINDS:
        if action.patch is None:
            out.append(f"{action.kind} 必须带 patch —— 它是要改候选的那一类动作")
        elif not str(action.patch.new_source or "").strip():
            out.append(f"{action.kind} 的 patch.new_source 为空")
        elif not str(action.patch.path or "").strip():
            out.append(f"{action.kind} 的 patch.path 为空 —— 不知道改的是哪个文件")
    if action.kind in SIDE_EFFECT_FREE_KINDS and action.patch is not None:
        out.append(f"{action.kind} **不许**带 patch —— 它不改候选 "
                   "(带 patch 就是借解释/请求之名改代码)")
    return tuple(out)


def _allowed_kinds(ctx: ActionContext) -> Tuple[str, ...]:
    """§21.2 的表。返回**这一轮允许**的动作种类

    `SIDE_EFFECT_FREE_KINDS`（解释 / 请求观测）**恒允许** —— 它们不改候选，因此没有
    任何一条路由有资格禁止它们；真正拦它们的是 `capability` 判据（今天没有 profiler
    与多 shape 的量具）。两条判据各说各的话，读者才不会把「这个路由不许要 profiler」
    误读成「profiler 不可用」。

    §21.2 原表与本函数的对应关系：
    ```
    编译错误        → repair_compile（局部修复优先）
    数值错误        → repair_correctness（局部优先，必须带误差形态）
    边界错误        → repair_correctness，或回退父代
    性能不达标      → 进入搜索，不默认改代码 ⇒ 只有拿到 §10.3 的许可证才许改源码
    算法瓶颈        → 允许结构性变更（仍受 §10.3 三条件门禁）
    接口 / ABI 破坏 → 拒绝候选，回退父代
    ```
    表**外**的路由（`infra` / `silent` / 空）不据此裁剪权限：那两种路由自己已经决定了
    「不许学 / 不动候选」，而「这一轮要不要走 Fixer」是 §8 的选择（`ROUTE_HOLD` 仍然走
    Fixer 是刻意的控制流）。在这里再裁一刀 = 同一个决定两处实现，且方向相反。
    """
    base = SIDE_EFFECT_FREE_KINDS
    kind = (ctx.route_kind or "").strip().lower()
    if kind == "compile":
        return base + (KIND_REPAIR_COMPILE,)
    if kind == "numeric":
        return base + (KIND_REPAIR_CORRECTNESS,)
    if kind == "performance":
        # 「搜索失败不是 bug」：参数搜索（模板点）才是默认出路；改源码要许可证。
        allow = base + (KIND_PROPOSE_TEMPLATE,)
        if ctx.escalation_allowed:
            allow = allow + STRUCTURAL_KINDS
        return allow
    if kind in ("infra", "silent"):
        return ACTION_KINDS                       # 表外：不裁剪
    # `""` = 没有失败在驱动它（首轮生成 / 搜索驱动的重生成）。这不是「编译错误」，
    # 但也不是「性能不达标」：它是**提议**，不是对失败的响应。
    return base + (KIND_PROPOSE_TEMPLATE, KIND_PROPOSE_SOURCE_PATCH)


def permission_problems(action: AgentAction, ctx: ActionContext) -> Tuple[str, ...]:
    """§21.1 的第二道：权限（§21.2 的表 + 三条横切规则）"""
    out = []
    allowed = _allowed_kinds(ctx)
    if action.kind not in allowed:
        why = f"路由 kind={ctx.route_kind or '(无失败驱动)'}"
        if ctx.route_kind == "performance" and action.kind in STRUCTURAL_KINDS:
            why += ("；结构升级门未放行"
                    + (f"（{'; '.join(ctx.escalation_refusals)}）"
                       if ctx.escalation_refusals else ""))
        out.append(f"{action.kind} 在{why}下不被允许 (§21.2 的动作权限); "
                   f"允许的是 {allowed}")

    # 横切 1: ABI 已破坏 ⇒ 拒绝候选，回退父代（不改候选的动作不受这条影响：
    # 「拒绝候选」说的是不许再拿一份新代码上去，而不是不许解释或请求观测）
    if ctx.abi_broken and action.kind not in SIDE_EFFECT_FREE_KINDS:
        out.append("上一轮已判「接口/调用契约不符」(必然 TypeError): 拒绝候选、回退父代, "
                   "只有不改候选的动作还允许 (§21.2 最后一行)")

    # 横切 2: 修复类必须点名它修的是哪一份候选 —— 「在坏的后代上盖一层」就是从这儿开始的
    if action.kind in REPAIR_KINDS and not str(action.parent_candidate or "").strip():
        out.append(f"{action.kind} 必须写 parent_candidate —— "
                   "不点名修的是哪一份, 就无法回滚也无法判断是否在原地打转")

    # 横切 3: 数值修复必须带上误差形态（§21.2「必须带误差形态」）
    if action.kind == KIND_REPAIR_CORRECTNESS and ctx.error_shape:
        if ctx.error_shape not in (action.rationale or ""):
            out.append(f"repair_correctness 的 rationale 必须带上误差形态 "
                       f"{ctx.error_shape!r} (§21.2) —— 形态是这次的处方方向, "
                       "丢掉它与「随便改改」在产物里长得一样")
    return tuple(out)


def budget_problems(action: AgentAction, ctx: ActionContext) -> Tuple[str, ...]:
    """§21.1 的第三道：预算。`tokens_left is None` = 没设上限 ⇒ 不拦，也不假装拦过"""
    out = []
    left = ctx.tokens_left
    if left is None:
        return ()
    if left <= 0:
        out.append(f"LLM token 预算已用尽 (剩 {left}) —— §10.2 的退化出口: "
                   "这一档只许做不花 token 的动作")
        return tuple(out)
    if isinstance(action.token_budget, int) and not isinstance(action.token_budget, bool) \
            and action.token_budget > left:
        out.append(f"动作声明要 {action.token_budget} token, 只剩 {left} —— "
                   "预算校验不过的动作控制器不接受")
    return tuple(out)


def _check_write_scope(patch: Patch, ctx: ActionContext) -> CheckOutcome:
    if not ctx.authorized_roots:
        return CheckOutcome(
            "write_scope", STATE_UNKNOWN,
            "没有声明授权根 ⇒ 判不了「在不在范围内」"
            "(不是在范围内 —— 授权根由 §22.1 的布局给出 (`candidate/` 那一格), 这里收到空元组说明调用方没有取它)")
    if _under_any(patch.path, ctx.authorized_roots):
        return CheckOutcome("write_scope", STATE_OK,
                            f"{patch.path} 落在授权根内 ({', '.join(ctx.authorized_roots)})")
    return CheckOutcome(
        "write_scope", STATE_VIOLATION,
        f"{patch.path} 不在任何授权根内 ({', '.join(ctx.authorized_roots)}) "
        "—— §21.4 第 1/2 条")


def _check_protected_paths(patch: Patch, ctx: ActionContext) -> CheckOutcome:
    parts = [p for p in _norm(patch.path).split("/") if p not in ("", ".")]
    hit = [p for p in parts if p in PROTECTED_DIR_NAMES]
    if hit:
        return CheckOutcome(
            "protected_paths", STATE_VIOLATION,
            f"{patch.path} 落在受保护目录 {hit} 里 —— §22.1: 候选不能写 "
            f"{'/'.join(PROTECTED_DIR_NAMES)} (契约哈希由 oracle.assert_hash 每轮校验, "
            "本条只管路径)")
    return CheckOutcome("protected_paths", STATE_OK,
                        f"{patch.path} 不经过任何受保护目录")


def _check_new_dependency(patch: Patch, ctx: ActionContext) -> CheckOutcome:
    lang = classify(patch.new_source, ctx.language)
    if lang.applicable is not True:
        return CheckOutcome("new_dependency", STATE_NOT_APPLICABLE,
                            f"依赖判据要 AST/正则覆盖, 而 {lang.reason}")
    # 基线的选择本身就是一条判据 (I78): 优先用调用方给的「上一次**过门**的候选」，
    # 因为它才是「已经进过流程」的那一份；patch 的父代可能是**被拒过的**候选。
    baseline = ctx.dep_baseline_source or patch.parent_source
    if not baseline:
        return CheckOutcome(
            "new_dependency", STATE_NOT_APPLICABLE,
            "没有可比基线 (「新增」是相对前一份**进过流程**的候选说的); "
            f"本次 import: {DependencyAnalyzer.extract_imports(patch.new_source)}")
    new_imports = set(DependencyAnalyzer.extract_imports(patch.new_source))
    old_imports = set(DependencyAnalyzer.extract_imports(baseline))
    # 复用现成的「哪些是外部依赖」判据: `to_pip_names` 会**剔掉内置模块**, 且有一份
    # 映射表与用例 (`tests/test_environment.py::test_builtins_excluded_from_pip`)。
    # 于是「新增依赖」= 新出现的 **pip 名**, 而不是「新增了一行 import」——
    # 后者会把 `import math` 也算成新依赖, 那是误拦截。
    added = [p for p in DependencyAnalyzer.to_pip_names(sorted(new_imports - old_imports))
             if p not in DependencyAnalyzer.to_pip_names(sorted(old_imports))]
    if added:
        return CheckOutcome(
            "new_dependency", STATE_VIOLATION,
            f"这次改动新增了外部依赖 {added} —— §21.4 第 4 条 (环境是 uv 管的固定集合, "
            "新依赖要改环境, 不能在候选里夹带)")
    return CheckOutcome("new_dependency", STATE_OK,
                        f"相对基线没有新增外部依赖 (新增 import: "
                        f"{sorted(new_imports - old_imports) or '无'})")


def _check_abi(patch: Patch, ctx: ActionContext) -> CheckOutcome:
    if not ctx.input_key_sets:
        return CheckOutcome("abi", STATE_NOT_APPLICABLE,
                            "没有输入名 (调用方这一轮会以关键字传什么) ⇒ 没有可判的对象 "
                            "(不是「签名对得上」)")
    verdicts = [interface_contract.check_kernel_interface(
                    patch.new_source, keys, language=ctx.language)
                for keys in ctx.input_key_sets]
    problem = next((v.problem for v in verdicts if v.problem), None)
    if problem:
        return CheckOutcome("abi", STATE_VIOLATION,
                            "ABI 与调用契约不符 (真跑必然 TypeError): "
                            + problem.split("\n")[0])
    if not any(v.applicable is True for v in verdicts):
        # 逐组都判不了 ⇒ 这一条**没有覆盖**（不是「签名对得上」）
        return CheckOutcome("abi", STATE_NOT_APPLICABLE,
                            "接口契约判据对这些源码都不适用/无法确立: "
                            + verdicts[0].lang.reason)
    flat = [n for keys in ctx.input_key_sets for n in keys]
    return CheckOutcome("abi", STATE_OK,
                        f"kernel 签名与调用契约一致 (输入名 {sorted(set(flat))})")


def _check_duplicate_of_parent(patch: Patch, ctx: ActionContext) -> CheckOutcome:
    if not patch.has_parent:
        return CheckOutcome("duplicate_of_parent", STATE_NOT_APPLICABLE,
                            "没有父代 ⇒ 没有可比对象")
    sig_new = novelty.signature_of(patch.new_source)
    sig_old = novelty.signature_of(patch.parent_source)
    if not sig_new or not sig_old:
        return CheckOutcome("duplicate_of_parent", STATE_NOT_APPLICABLE,
                            "结构签名读不出来 (语法错 / 没有函数定义) ⇒ 判不了重复 "
                            "(不是「不重复」)")
    if sig_new != sig_old:
        return CheckOutcome("duplicate_of_parent", STATE_OK,
                            f"结构签名与父代不同 ({sig_new} vs {sig_old})")
    # ⚠️ 边界（CHECKLOG/状态文档里必须一并读）：**只对修复类**判违约。
    # 生成类重复只观测不拒绝 —— 那是 doc 47 的实测决定（「假阳的代价是丢掉一个真的
    # 更好的核, 在没有基率之前拿它省 GPU 时间就是拿判决换一个还没量的东西」），
    # 而修复类重复是另一件事：一字未改的「修复」必然再失败一次, 那一轮 GPU 秒是纯浪费。
    if ctx.route_kind in ("", "performance") or not ctx.parent_verified:
        return CheckOutcome("duplicate_of_parent", STATE_UNKNOWN,
                            f"与父代结构相同 ({sig_new}), 但这不是修复轮/父代未验证 ⇒ "
                            "只观测不拒绝 (novelty 的固定口径: 无基率前不据此省 GPU)")
    return CheckOutcome(
        "duplicate_of_parent", STATE_VIOLATION,
        f"修复产物与父代结构完全相同 ({sig_new}) —— 一个什么都没改的修复会再失败一次; "
        "§21.4 第 6 条")


def _check_capability(action: AgentAction, ctx: ActionContext,
                      capabilities: frozenset) -> CheckOutcome:
    """§21.4 第 7 条：引用不存在的能力。**零 GPU 时间**

    两半，今天只有一半有对象：
      * **动作声明的**：预期的瓶颈有没有量具、请求的工具存不存在（这一半今天就能判死）；
      * **源码里调用的符号**：仓里没有任何语言的 API 面清单（没有 Triton 白名单、
        没有符号表），所以「这份源码调了一个不存在的 API」今天**判不了** ——
        它进 `uncovered_rules`，而不是被读成「没有幻觉 API」。
        真实代价有记录：`tl.cache` 这类不存在的 API 曾白烧一轮 Fixer（见
        `triton_opt_rules.py` 里那一行注释）。
    """
    problems = []
    if action.expected_bottleneck:
        gauged = escalation.gauged_bottlenecks()
        if action.expected_bottleneck not in gauged:
            problems.append(
                f"预期瓶颈 {action.expected_bottleneck!r} 今天没有量具 "
                f"(§10.3 条件 2 的四类里只有 {gauged} 兑现了) —— 引用不存在的能力")
    for tool in action.requested_tools or ():
        cap = TOOLS.get(tool)
        if cap is None:
            if tool in TOOLS:
                problems.append(f"请求的工具 {tool!r} 今天没有实现物 (注册表里是 None)")
            continue
        if cap not in capabilities:
            problems.append(
                f"请求的工具 {tool!r} 需要后端能力 {cap.value!r}, 而本后端的能力集合是 "
                f"{sorted(c.value for c in capabilities) or '空集'}")
    for tool in KIND_REQUIRED_TOOLS.get(action.kind, ()):
        cap = TOOLS.get(tool)
        if cap is None:
            # 「这个东西还没做」与「这台机器不支持它」是**两件事**, 压成一句「今天不可用」
            # 会让它们在后端换一台机器时长得一样 (本仓的三态纪律: 判不了/没有对象/有对象
            # 但要分开说)。措辞只在一处产生 (`ask_unavailable_reason`) —— 那也是
            # 「响应模型请求」那条路读的同一句。
            problems.append(ask_unavailable_reason(action.kind)
                            or f"{action.kind} 需要的工具 {tool!r} 今天没有实现物")
        elif cap not in capabilities:
            problems.append(
                f"{action.kind} 需要的工具 {tool!r} 需要后端能力 {cap.value!r}, 而本后端"
                f"的能力集合是 {sorted(c.value for c in capabilities) or '空集'}")
    if problems:
        return CheckOutcome("capability", STATE_VIOLATION, "; ".join(sorted(set(problems))))
    if action.patch is None:
        return CheckOutcome("capability", STATE_OK, "动作没有声明任何能力 (无需能力判据)")
    return CheckOutcome(
        "capability", STATE_NOT_APPLICABLE,
        "源码级「幻觉 API」没有判据: 仓里没有任何语言的 API 面清单 "
        "(没有 Triton 白名单/符号表) ⇒ 这一半判不了 (不是「没有幻觉 API」)")


def precompile_checks(action: AgentAction, ctx: ActionContext,
                      capabilities: frozenset = frozenset()) -> PrecompileVerdict:
    """§21.4 的独立检查。**顺序即设计原文的顺序**（报告按它读）

    只有 `Patch` 相关的那几条需要源码；`capability` 判的是动作声明本身，所以它
    在没有 patch 时也跑（`request_profiler` 这类动作没有 patch，但正是最需要
    能力判据的那一类）。

    `capabilities` 由调用方（`AgentRuntime`）传进来，**不是** `ctx` 上的一格:
    后端具备哪些能力是「一个事实、一个出处」（`Backend.capabilities`），
    在上下文里再写一份就等于允许「动作按 A 判、工具列表按 B 报」。
    """
    patch = action.patch
    lang = classify(patch.new_source, ctx.language) if patch is not None else None
    outcomes = []
    if patch is not None:
        outcomes.append(_check_write_scope(patch, ctx))
        outcomes.append(_check_protected_paths(patch, ctx))
        outcomes.append(_check_new_dependency(patch, ctx))
        outcomes.append(_check_abi(patch, ctx))
        outcomes.append(_check_duplicate_of_parent(patch, ctx))
    else:
        # 没有 patch ⇒ 前五条**没有对象**。逐条给出「为什么没有对象」，而不是干脆不出现:
        # 缺了这五行, 覆盖表会把一个「没有对象」的动作报成「五条都通过」。
        for rule in ("write_scope", "protected_paths", "new_dependency", "abi",
                     "duplicate_of_parent"):
            outcomes.append(CheckOutcome(rule, STATE_NOT_APPLICABLE,
                                         f"{action.kind} 不带 patch ⇒ 这条判据没有对象"))
    outcomes.append(_check_capability(action, ctx, capabilities))
    return PrecompileVerdict(outcomes=tuple(outcomes), language_verdict=lang)


# ─── 9. Runtime ────────────────────────────────────────────────


class AgentRuntime:
    """控制器侧的动作入口。**唯一**的公开方法语义：`admit(action, ctx) -> ActionVerdict`

    它是有状态的只有一处：`_rejected` 计数（复盘用：「这一层今天拦下过几个动作」）。
    判定本身是纯函数（`schema_problems` / `permission_problems` / `budget_problems` /
    `precompile_checks` 都是模块级函数）—— 于是判据可以脱离 Runtime 单独核，
    与整个仓「判据与调用方分开」的姿势一致。
    """

    def __init__(self, *, capabilities: Iterable = (), authorized_roots: Iterable = (),
                 tools: Optional[Mapping[str, Optional[Cap]]] = None):
        self.capabilities = frozenset(capabilities or ())
        self.authorized_roots = tuple(authorized_roots or ())
        self.tools = dict(tools) if tools is not None else dict(TOOLS)
        #: 「动作被谁拦下过」的计数（键 = 动作种类）。产物里要有这个数, 否则
        #: 「这一层今天在工作吗」只能靠重跑一遍才知道。
        self.rejected: Dict[str, int] = {}
        self.admitted = 0

    def reset_run(self) -> None:
        """把这一次运行的记账清零。**每次运行调一次**（与 `BudgetLedger` 新建同一个理由：
        跨任务累计会让第二个任务的产物里出现第一个任务拦下的动作）。判据本身无状态。"""
        self.rejected = {}
        self.admitted = 0

    # ── 工具面 ────────────────────────────────────────────────

    def available_tools(self) -> Tuple[str, ...]:
        """今天**真的能用**的工具。两档都排除：注册表里是 `None` 的，以及后端没声明能力的。

        空集是合法答案 —— 它表示「这个后端什么工具都给不了」，调用方据此不许
        在动作里点名任何工具（点名就被判「引用不存在的能力」）。
        """
        out = []
        for name, cap in self.tools.items():
            if cap is not None and cap in self.capabilities:
                out.append(name)
        return tuple(sorted(out))

    def unavailable_tools(self) -> Dict[str, str]:
        """今天**不可用**的工具 → 为什么（读者：报告与提示层，不许只说「没有」）"""
        out = {}
        for name, cap in self.tools.items():
            if cap is None:
                out[name] = "注册表里没有实现物 (设计词汇里有, 兑现它的东西今天不存在)"
            elif cap not in self.capabilities:
                out[name] = f"后端没有能力 {cap.value}"
        return out

    # ── 裁决 ──────────────────────────────────────────────────

    def admit(self, action: AgentAction, ctx: Optional[ActionContext] = None) -> ActionVerdict:
        """三道校验 + 独立检查。**三道都跑**（不是短路）—— 一次给全的理由才有修复价值

        只有 schema 不过时不跑独立检查：那些判据要拿 `patch` 的字段，而 schema
        正在说的就是「这些字段不对」。此时 `precompile=None`，它的含义在
        `ActionVerdict.precompile` 上写明是「没跑」，不是「跑过了没问题」。
        """
        ctx = ctx if ctx is not None else ActionContext()
        schema = schema_problems(action)
        permission = permission_problems(action, ctx)
        budget = budget_problems(action, ctx)
        # 「花了钱的」动作不因预算被拒 —— 见 `ActionContext.spend_committed`。
        deferred = tuple(budget) if (budget and ctx.spend_committed) else ()
        budget_refuses = () if deferred else budget
        precompile = (precompile_checks(action, ctx, self.capabilities)
                      if not schema else None)
        precompile_ok = precompile.accepted if precompile is not None else True
        accepted = not (schema or permission or budget_refuses) and precompile_ok

        fallback = bool(
            ctx.parent_verified
            and (ctx.error_shape in BOUNDARY_ERROR_SHAPES or ctx.abi_broken)
        )
        if accepted:
            self.admitted += 1
        else:
            self.rejected[action.kind] = self.rejected.get(action.kind, 0) + 1
        return ActionVerdict(
            action=action, accepted=accepted, schema_problems=schema,
            permission_problems=permission, budget_problems=budget, context=ctx,
            precompile=precompile,
            structural_licensed=bool(action.kind in STRUCTURAL_KINDS
                                     and ctx.escalation_allowed),
            budget_deferred_note=("; ".join(deferred) +
                                  " (§21.1 的预算问题已记账; 但这一次的钱已经花了 —— "
                                  "§10.2 给这一档的后果是退化成纯数值搜索, 不是拒绝一份"
                                  "已付过钱的候选)" if deferred else ""),
            fallback_available=fallback,
        )

    # ── 上下文（§21.3） ────────────────────────────────────────

    def context_for(self, *, contract_summary: str = "", patch: Optional[Patch] = None,
                    route: Any = None, first_error: str = "",
                    profiler_metrics: Optional[Mapping[str, Any]] = None,
                    budget_left: Optional[int] = None) -> ProposalContext:
        """把**这一轮手里已有的结构化事实**拼成上下文。参数全是「已知就传」的形式

        为什么由 Runtime 来拼、而不是让每个调用点自己写：§21.3 的第二句是
        「不要把完整历史日志无条件塞入上下文」—— 让调用点自由拼装，这句话就只是一条
        约定。这里唯一能进上下文的来源是：契约摘要、`patch.diff()`（现算）、
        `failure_route.Route`（结构化判定）、首个错误文本、筛过的指标、工具表、预算。
        `route` 传 `None` = 这一轮没有失败在驱动它。
        """
        kind = ""
        shape = None
        if route is not None:
            kind = getattr(route, "kind", "") or ""
            shape = getattr(route, "error_shape", None)
        return ProposalContext(
            contract_summary=contract_summary,
            parent_diff=(patch.diff() if patch is not None else ""),
            last_failure_kind=kind,
            first_error=first_error,
            error_shape=shape,
            profiler_metrics=dict(profiler_metrics or {}),
            available_tools=self.available_tools(),
            budget_left=budget_left,
            writable_paths=self.authorized_roots,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {"admitted": self.admitted, "rejected": dict(self.rejected),
                "available_tools": list(self.available_tools()),
                "unavailable_tools": self.unavailable_tools(),
                "authorized_roots": list(self.authorized_roots),
                "capabilities": sorted(c.value for c in self.capabilities)}