"""「这道静态守卫对这份源码**适用**吗？」—— 一个判据，两个守卫共用（CHECKLOG I65）

## 为什么需要这个模块

`anticheat.static_audit` 与 `interface_contract.check_kernel_interface` 都是 **Python 形状**的
守卫：前者拿 `ast.parse` 走 AST（解析不出就退回正则），后者找 `def kernel` 再比对参数名。
对一份 C++/CUDA 模板源，两者都「判不了」，而它们的**返回值形状**只有一个「有/没有问题」的
通道：

* `static_audit` 返回违规列表 —— **空列表既表示「查过了没有违规」也表示「根本没查成」**；
* `check_kernel_interface` 返回 `None` —— 同上，「签名对得上」与「看不出来」压成了同一个值。

于是「查不了」被读成「没问题」：这是 fail-open，也是这个仓的基线纪律
（**信任全部来自验证侧**）最不能接受的一种。修法不是把守卫写得更聪明（那需要真的 C++/CUDA
分析器），而是让它们**说实话**：把「适用性」变成一个有名字的三态，并且只在这**一处**判定。

## 三态

| `applicable` | 含义 | 下游该怎么读 |
|---|---|---|
| `True` | 这份源码在守卫的覆盖范围内（Python 系，含 Triton） | 「没有违规」才是没有违规 |
| `False` | **明确不在**覆盖范围内（声明了非 Python 语言） | 不许当通过；资格上按「这个数不足以支撑永久结论」处理 |
| `None` | **无法确立**：语言没声明，而源码也不像 Python | 同上，但理由不同（「没声明」与「声明了别的」要分得开） |

判定输入是**声明**（`language` / DSL 名）加**一条**推断（`ast.parse` 是否成功）——
推断只有这一处，两个守卫都只消费结论，谁也不许自己再嗅一次 `def kernel`。

## 语言词汇表

`AST_FAMILY` 是「源码是 Python」的那些名字（`REGEX_FAMILY` 是只有文本规则可用的）。
加新语言必须同时回答：它的静态守卫是谁、
它怎么编译、它怎么跑 —— 只往这个集合里加名字而不补守卫，等于把空转扩大一层。
"""

import ast
from dataclasses import dataclass
from typing import Optional, Sequence

#: 走 **AST** 的那一族: 源码就是 Python（Triton 是 Python 库, 它的核也是 Python 源码）
AST_FAMILY = ("python", "triton", "cuda_python")

#: 走 **正则** 的那一族: 源码不是 Python, 但禁用清单的正则规则就是为它写的
#: （`cuda_c` = PyTorch C++ Extension: Python 文件里嵌 C++ 源码字符串, AST 只会看到一个
#: Constant 节点 ⇒ 那一族从一开始就只有文本模式可用, 且 `_GPU_DSL_RULES` 为此存在）。
#: 这是**真实但更弱**的覆盖 —— 与「没有覆盖」必须分开记（`coverage` 那一格）。
REGEX_FAMILY = ("cuda_c",)

#: 覆盖强度: `ast` > `regex_only` > `none`（`unknown` = 无法确立）
COVERAGE_AST = "ast"
COVERAGE_REGEX = "regex_only"
COVERAGE_NONE = "none"

#: 三态的可读名（产物里记这个, 不记布尔）
APPLIES = "applies"
DOES_NOT_APPLY = "does_not_apply"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class LangVerdict:
    """一份源码的「守卫适用性」判定。`applicable` 是三态 —— `None` 不许被读成任何一边。"""

    language: str          #: 声明的语言（`""` = 没声明）
    family: str            #: `"python"` / `"non_python"` / `"unknown"`
    applicable: Optional[bool]
    reason: str
    #: 覆盖强度（`ast` / `regex_only` / `none`）。`applicable=True` 但 `regex_only` 是真的
    #: 覆盖 —— 它只看文本模式, 看不穿宏与头文件; 这一格让「弱」与「没有」不混。
    coverage: str = COVERAGE_AST

    @property
    def state(self) -> str:
        if self.applicable is True:
            return APPLIES
        if self.applicable is False:
            return DOES_NOT_APPLY
        return UNKNOWN

    def as_dict(self):
        return {"language": self.language, "family": self.family,
                "applicable": self.applicable, "state": self.state,
                "coverage": self.coverage, "reason": self.reason}


def classify(source: str, declared: str = "") -> LangVerdict:
    """判定守卫对这份源码适用吗。

    * 声明在 `AST_FAMILY` 里 ⇒ 适用（不看源码 —— 声明是权威）；
    * 声明了别的语言 ⇒ **不适用**（明确的一态，理由里带上声明的是什么）；
    * 没声明 ⇒ 只做**一条**推断：`ast.parse` 成功 ⇒ 当 Python 处理（记下「靠推断」这件事，
      因为「声明过」与「猜出来的」在后处理里必须能分开）；失败 ⇒ 无法确立。
    """
    name = (declared or "").strip().lower()
    if name in AST_FAMILY:
        return LangVerdict(language=declared, family="python", applicable=True,
                           coverage=COVERAGE_AST,
                           reason=f"声明为 Python 系 ({declared}) ⇒ AST 审计覆盖")
    if name in REGEX_FAMILY:
        return LangVerdict(language=declared, family="non_python", applicable=True,
                           coverage=COVERAGE_REGEX,
                           reason=(f"声明为 {declared} ⇒ 只有**文本模式**规则覆盖它 "
                                   f"(AST 用不上: C++ 源码嵌在字符串里)。这是真实但更弱的"
                                   f"覆盖, 看—— 不是「查过了没问题」"))
    if name:
        return LangVerdict(language=declared, family="non_python", applicable=False,
                           coverage=COVERAGE_NONE,
                           reason=(f"声明为 {declared} —— 守卫没有任何为这个语言写的规则, "
                                   f"对这份源码**判不了** (不是「查过了没问题」)"))
    try:
        ast.parse(source)
    except SyntaxError:
        return LangVerdict(language="", family="unknown", applicable=None,
                           coverage=COVERAGE_NONE,
                           reason=("语言没声明, 而源码不是 Python (解析不了) ⇒ 守卫的适用性"
                                   "**无法确立**"))
    return LangVerdict(language="", family="python", applicable=True,
                       coverage=COVERAGE_AST,
                       reason="语言没声明, 但源码就是 Python (解析得过) ⇒ AST 审计覆盖")


def combine(verdicts: Sequence[Optional[LangVerdict]]) -> Optional[LangVerdict]:
    """把多处的适用性收成一个（L1 与 L2 各判一次）。

    取**最保守**的那一档：任一处「不适用」⇒ 不适用；没有任何「不适用」但有「无法确立」
    ⇒ 无法确立；全都适用 ⇒ 适用（取第一条作代表，用于带上语言与理由）。`None` 的元素
    （那一级没跑）不参与。
    """
    live = [v for v in verdicts if v is not None]
    if not live:
        return None
    for want, state in ((False, "non_python"), (None, "unknown")):
        for v in live:
            if v.applicable is want:
                return v
    return live[0]

def flag_from_metrics(*metrics_dicts) -> Optional[bool]:
    """从各级结果**发布的 metrics** 里读出适用性, 取最保守的一档。

    读者是 Conductor: 它手里只有各级 `VerificationResult.metrics` (L1/L2 各判过一次),
    而 L3 的资格判定必须拿到**同一份源码**的那个结论。任一处 False ⇒ False (一处
    「判不了」就足以说「这两道守卫对这份源码没有覆盖」); 一处都没有 ⇒ `None` (= 没观测,
    不改判, 与缺键同义)。`None` 的元素 (那一级没跑或没记) 不参与 —— 它不是 False。
    """
    values = []
    for m in metrics_dicts:
        if not isinstance(m, dict):
            continue
        v = m.get("static_guards")
        if v is not None:
            values.append(bool(v))
    if not values:
        return None
    return all(values)
