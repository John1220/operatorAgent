"""共享内存预检：只有绑定到候选的物理存储证据才能产生资源拒绝。

BLOCK、tl.dot、累加器 dtype 和 num_stages 是源码元数据，不能证明后端将哪些
值放入 shared memory。可信 harness 可提供 shared layout 或原生编译资源回执；
缺证据时返回 unknown，交由后续原生检查。feasible=True 只表示本门未拒绝。
逻辑 tile 公式保留兼容用途，但不进入资源拒绝、候选剪枝或硬限制归因。
"""
from __future__ import annotations

import ast
import hashlib
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional

# 保守放行的倍数: 估出 1.2× 上限以内都不拦
_SLACK = 1.2

#: 同上, **公开名** —— §12 的便宜代理剪枝 (`candidates.prune_by_smem`) 用同一个常数。
#: 为什么要共用而不是各写一个: 两套阈值会长成「预检放行、剪枝拒绝」这种自相矛盾的
#: 行为。资源证据的校验与比较同样必须走同一个 assess_shared_memory。
SMEM_SLACK = _SLACK

_MAX_SOURCE_BYTES = 131072
_MAX_AST_NODES = 8192
_MAX_AST_DEPTH = 64
_MAX_BUFFERS = 256
_MAX_CONFIGURATIONS = 256
_MAX_DIMENSIONS = 32
# 物理字节/维度使用有界的无符号存储整数；超大 Python bigint 不是硬件资源回执。
_MAX_RESOURCE_INTEGER = (1 << 64) - 1
_UNKNOWN = object()

# 这是显式资源契约的存储类型词表，不是从源码猜 dtype。
_DTYPE_BYTES = {
    "bool": 1, "int8": 1, "uint8": 1, "float8_e4m3fn": 1,
    "float8_e4m3fnuz": 1, "float8_e5m2": 1, "float8_e5m2fnuz": 1,
    "int16": 2, "uint16": 2, "float16": 2, "fp16": 2, "half": 2,
    "bfloat16": 2, "bf16": 2,
    "int32": 4, "uint32": 4, "float32": 4, "fp32": 4,
    "int64": 8, "uint64": 8, "float64": 8, "fp64": 8, "complex64": 8,
    "complex128": 16,
}


@dataclass(frozen=True)
class SharedBuffer:
    """harness 声明的一个实际 shared buffer；copies 是已确立的存储份数。"""

    name: str
    shape: tuple
    dtype: str
    copies: int = 1


@dataclass(frozen=True)
class SharedMemoryEvidence:
    """可信调用方提供的资源契约，候选源码无权自报此证据。

    binding 标识资源回执覆盖的候选和配置作用域。源码入口使用完整源码 SHA256；
    模板调用方可使用模板点身份。native_compiler 的字节数不得被改写为任务 dtype，
    trusted_shared_layout 则逐缓冲区使用自己的 dtype 和 copies。
    """

    binding: str
    kind: str
    producer: str
    buffers: tuple = ()
    native_bytes: Optional[int] = None


def source_resource_binding(code: str) -> str:
    """完整源码的内容身份；不执行源码，不把结构签名当作编译资源身份。"""
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def smem_limit_bytes(caps) -> Optional[int]:
    """每 SM 的 smem 上界 (字节)。`caps` 缺这个字段 / 为 0 ⇒ **None (缺 ≠ 无上限)**

    None 与 0 必须分开: 0 会被读成「一点都放不下」(把所有点都拒掉), 而事实是
    「这个设备的 smem 上界我们不知道」。调用方 (预检 / 剪枝) 各自显式处理 None。
    """
    limit_kb = getattr(caps, "shared_mem_per_sm_kb", None)
    if (type(limit_kb) not in (int, float) or limit_kb <= 0
            or (type(limit_kb) is float and not math.isfinite(limit_kb))):
        return None
    if limit_kb > _MAX_RESOURCE_INTEGER:
        return None
    # 取整口径与重构前**逐字节一致**: int(kb * 1024), 不是 int(kb) * 1024
    # —— 小数 KB 时两者差一整 KB, 而既有用例正是钉在边界上的。
    try:
        limit = int(limit_kb * 1024)
    except (ValueError, OverflowError):
        return None
    return limit if 0 < limit <= _MAX_RESOURCE_INTEGER else None


@dataclass
class MemCaps:
    """smem 上界的**唯一载体**: 给 `smem_limit_bytes` 与剪枝用

    为什么要一个载体而不是直接传 dict/`HardwareCaps`: `smem_limit_bytes(caps)` 读的是
    **属性**, 而调用方手里有三种形状 (L3 产物那一格是 dict, `HardwareCaps` 是对象,
    搜索侧拿到的是 `perf_data`)。三种形状各写一遍 `getattr/[]` 就是三处口径。
    """

    shared_mem_per_sm_kb: int = 0


def caps_from_metrics(source: Any) -> MemCaps:
    """从 L3 产物那一格 / `HardwareCaps` 装出 `MemCaps` (**同一处口径**)

    垃圾值 / 缺键 / 0 ⇒ 记 `0` = **上界未知** (由 `smem_limit_bytes` 判成 `None`),
    不是「无上限」也不是「一点都放不下」。这里不抛: 剪枝是省钱的手段, 不该因为一个
    脏字段把整轮打断 —— 但也不能把脏值当 0 KB 用 (那就是误判成「放不下」)。
    """
    kb: Any = None
    if source is not None:
        if isinstance(source, Mapping):
            kb = source.get("shared_mem_per_sm_kb")
        else:
            kb = getattr(source, "shared_mem_per_sm_kb", None)
    # 不把 bool、数字字符串或非有限值转换成真实硬件上限。
    if (type(kb) not in (int, float) or kb <= 0
            or (type(kb) is float and not math.isfinite(kb))):
        kb = 0
    return MemCaps(shared_mem_per_sm_kb=kb)


@dataclass
class ResourceReport:
    """unknown 仍可继续原生检查；within_limit 也不证明 launch/occupancy 可行。"""

    feasible: bool = True
    reason: str = ""
    estimated_smem_bytes: Optional[int] = None
    limit_bytes: Optional[int] = None
    block_shapes: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    assessment: str = "unknown"
    evidence_kind: str = ""
    evidence_producer: str = ""
    evidence_binding: str = ""
    buffers: list = field(default_factory=list)
    configurations: list = field(default_factory=list)

    def as_dict(self) -> dict:
        """保留物理证据种类及源码元数据；内部构造的字段均为严格 JSON 值。"""
        return asdict(self)

    def as_message(self) -> str:
        if self.feasible:
            return ""
        parts = [self.reason]
        if self.estimated_smem_bytes is not None and self.limit_bytes is not None:
            parts.append(f"估算 smem {self.estimated_smem_bytes / 1024:.1f} KiB "
                         f"> 上限 {self.limit_bytes / 1024:.1f} KiB")
        if self.block_shapes:
            parts.append(f"BLOCK 字面量摘要 {self.block_shapes}")
        if self.evidence_kind:
            parts.append(f"证据 {self.evidence_kind} ({self.evidence_producer})")
        parts.append("请核对实际共享缓冲区、tile 或流水配置 (资源预检, 未动 GPU)")
        return "; ".join(parts)


def _positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


def _physical_int(value: Any, *, allow_zero: bool = False) -> bool:
    return (type(value) is int and (value >= 0 if allow_zero else value > 0)
            and value <= _MAX_RESOURCE_INTEGER)


def _buffer_record(buffer: Any) -> Optional[dict]:
    if (type(buffer) is not SharedBuffer or type(buffer.name) is not str
            or not buffer.name.strip() or len(buffer.name) > 256
            or type(buffer.shape) is not tuple or not buffer.shape
            or len(buffer.shape) > _MAX_DIMENSIONS
            or not all(_physical_int(dim) for dim in buffer.shape)
            or type(buffer.dtype) is not str or not _physical_int(buffer.copies)):
        return None
    dtype = buffer.dtype
    for prefix in ("torch.", "tl.", "np.", "numpy."):
        if dtype.startswith(prefix):
            dtype = dtype[len(prefix):]
            break
    width = _DTYPE_BYTES.get(dtype)
    if width is None:
        return None
    size = width * buffer.copies
    for dim in buffer.shape:
        size *= dim
        if size > _MAX_RESOURCE_INTEGER:
            return None
    return dict(name=buffer.name, shape=list(buffer.shape), dtype=buffer.dtype,
                copies=buffer.copies, bytes=size)


def assess_shared_memory(evidence: Any, caps, *, binding: str) -> ResourceReport:
    """共享的证据校验和拒绝判据；不接受源码里自报的资源字典。

    layout 必须描述实际 shared 存储（包含同时存活关系和复制数），native 必须是
    同一候选/配置作用域的编译回执。此函数验证数据和 binding；信任来源由 harness
    的调用边界保障。缺任一条件即 unknown，不能从逻辑 tile 补出物理字节。
    """
    rep = ResourceReport(limit_bytes=smem_limit_bytes(caps))
    if type(evidence) is not SharedMemoryEvidence:
        rep.notes.append("缺少可信 shared layout / native compiler 资源证据，交由原生检查")
        return rep
    for target, value in (("evidence_kind", evidence.kind),
                          ("evidence_producer", evidence.producer),
                          ("evidence_binding", evidence.binding)):
        if type(value) is str:
            setattr(rep, target, value)
    if (type(binding) is not str or not binding or type(evidence.binding) is not str
            or evidence.binding != binding):
        rep.notes.append("资源证据 binding 不匹配或缺失，交由原生检查")
        return rep
    if (type(evidence.kind) is not str
            or type(evidence.producer) is not str or not evidence.producer.strip()
            or len(evidence.producer) > 256):
        rep.notes.append("资源证据 kind / producer 无效，交由原生检查")
        return rep
    if type(evidence.buffers) is not tuple or len(evidence.buffers) > _MAX_BUFFERS:
        rep.notes.append("资源证据 buffers 无效，交由原生检查")
        return rep
    if evidence.kind == "native_compiler":
        if (not _physical_int(evidence.native_bytes, allow_zero=True)
                or evidence.buffers):
            rep.notes.append("原生资源字节必须为有界非负整数且不得混入声明 layout，交由原生检查")
            return rep
        rep.estimated_smem_bytes = evidence.native_bytes
    elif evidence.kind == "trusted_shared_layout":
        if evidence.native_bytes is not None or not evidence.buffers:
            rep.notes.append("声明 layout 须含共享缓冲区且不得混入原生字节，交由原生检查")
            return rep
        records = [_buffer_record(buffer) for buffer in evidence.buffers]
        if (any(record is None for record in records)
                or len({record["name"] for record in records}) != len(records)):
            rep.notes.append("共享缓冲区的 shape / dtype / copies / name 无法确立，交由原生检查")
            return rep
        total = sum(record["bytes"] for record in records)
        if not _physical_int(total):
            rep.notes.append("共享缓冲区总字节数超出资源整数边界，交由原生检查")
            return rep
        rep.buffers = records
        rep.estimated_smem_bytes = total
    else:
        rep.notes.append("未知资源证据 kind，交由原生检查")
        return rep
    if rep.limit_bytes is None:
        rep.notes.append("caps 没有有效 shared_mem_per_sm_kb，上界未知，交由原生检查")
        return rep
    # 每 SM 总量是宽松上界；不改硬件上限，不把本门放行表述成 launch fit 证明。
    if rep.estimated_smem_bytes > rep.limit_bytes * SMEM_SLACK:
        rep.feasible = False
        rep.assessment = "over_limit"
        rep.reason = "资源预检: 共享内存放不下已确立的 shared 存储"
    else:
        rep.assessment = "within_limit"
        rep.notes.append("未超过宽松资源拒绝阈值；不证明 launch 可行或 occupancy")
    return rep


def _elem_bytes_from_code(code: str) -> int:
    """旧逻辑代理的兼容提示；不得作为物理 shared 宽度或确定拒绝的依据。"""
    code = _strip_comments_and_prose(code)
    if re.search(r"float32|tl\.float32|torch\.float32|np\.float32|dtype=np\.float32", code):
        return 4
    if re.search(r"float16|tl\.float16|torch\.float16|fp16|bfloat16|tl\.bfloat16", code):
        return 2
    return 2


def _strip_comments_and_prose(code: str) -> str:
    """把注释与非键名字符串**原位抹掉** —— 判据只许读代码。

    2026-10-19 真机实测（`operator_run` 第一次真卡闭环）：Fixer 把超大的
    `BLOCK_M=256` 配置删掉之后，顺手写了一行注释
    `# 修复: 移除 BLOCK_M=256 的超大 tile …`；预检把这个 256 又读成真实配置，
    于是**正确的修复被同一个错误再拒一次**。这类假阳性比漏报更难查 ——
    它长得像「候选没改」。

    实现是**按 token 位置抹字符**，不是拼 token：拼出来的 `tl . constexpr`
    会让「`BLOCK_M: tl.constexpr = 128`」这类写法失配（第一版就这么错的）。
    """
    import io
    import tokenize
    keyish = re.compile(r"""['"]BLOCK_[A-Z0-9_]+['"]""")
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return re.sub(r"#[^\n]*", "", code)          # 兜底：至少去掉行注释

    lines = code.splitlines(keepends=True)

    def blank(sl: int, sc: int, el: int, ec: int, text: str) -> None:
        if sl == el:
            row = lines[sl - 1]
            lines[sl - 1] = row[:sc] + text + row[ec:]
            return
        # 跨行（多行 docstring）：首行留前缀、末行留后缀，中间整行清空
        lines[sl - 1] = lines[sl - 1][:sc] + text + "\n"
        for i in range(sl, el - 1):
            lines[i] = "\n"
        lines[el - 1] = lines[el - 1][ec:]

    for tok in reversed(tokens):                     # 从后往前，位置不互相干扰
        if tok.type == tokenize.COMMENT:
            blank(*tok.start, *tok.end, "")
        elif tok.type == tokenize.STRING and not keyish.fullmatch(tok.string):
            blank(*tok.start, *tok.end, "''")
    return "".join(lines)


def _extract_block_sizes(code: str) -> list[tuple[str, int]]:
    """旧 shape/机械改写接口的字面量摘要；各轴 max 绝不是一个完整配置。

    仅供展示和既有 shape 检查；不得据此推导物理资源、剪枝或硬限制。
    配置元数据使用下方 AST 解析器逐组记录。
    """
    found: dict[str, int] = {}
    if not _source_in_bounds(code):
        return []
    code = _strip_comments_and_prose(code)

    # 1) `BLOCK_M: tl.constexpr = 64` / `BLOCK_M = 64` / `BLOCK_M=64`
    for m in re.finditer(r"\b(BLOCK_[A-Z0-9_]+)\s*(?::\s*tl\.constexpr\s*)?=\s*(\d+)", code):
        name, val = m.group(1), int(m.group(2))
        found[name] = max(found.get(name, 0), val)

    # 2) autotune 的 dict 字面量: `{'BLOCK_M': 256}` / `{"BLOCK_N": 128}`
    #    兼容摘要保留 max；资源判定不读取这个合并结果。
    for m in re.finditer(r"['\"](BLOCK_[A-Z0-9_]+)['\"]\s*:\s*(\d+)", code):
        name, val = m.group(1), int(m.group(2))
        found[name] = max(found.get(name, 0), val)

    return sorted(found.items())


def _source_in_bounds(code: Any) -> bool:
    if type(code) is not str or len(code) > _MAX_SOURCE_BYTES:
        return False
    try:
        return len(code.encode("utf-8")) <= _MAX_SOURCE_BYTES
    except UnicodeError:
        return False


def _bounded_source_tree(code: Any) -> tuple:
    if not _source_in_bounds(code):
        return None, "源码类型/大小无效，配置元数据未知"
    if not code.strip():
        return None, "源码为空，资源与配置元数据未知"
    try:
        tree = ast.parse(code)
        pending, count = [(tree, 0)], 0
        while pending:
            node, depth = pending.pop()
            count += 1
            if count > _MAX_AST_NODES or depth > _MAX_AST_DEPTH:
                return None, "源码 AST 超出解析边界，配置元数据未知"
            pending.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
        return tree, ""
    except (SyntaxError, ValueError, TypeError, RecursionError, UnicodeError):
        return None, "源码无法静态解析，配置元数据未知"


def _literal_json(node: ast.AST):
    """只读取有限 JSON 字面量；不 eval、导入、调用或展开候选对象。"""
    if isinstance(node, ast.Constant):
        value = node.value
        if value is None or type(value) in (str, bool):
            return value
        if type(value) is int and -_MAX_RESOURCE_INTEGER <= value <= _MAX_RESOURCE_INTEGER:
            return value
        if type(value) is float and math.isfinite(value):
            return value
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _literal_json(node.operand)
        if type(value) in (int, float):
            return value if isinstance(node.op, ast.UAdd) else -value
    elif isinstance(node, (ast.List, ast.Tuple)):
        values = [_literal_json(item) for item in node.elts]
        if not any(value is _UNKNOWN for value in values):
            return values
    elif isinstance(node, ast.Dict):
        out = {}
        for key, value_node in zip(node.keys, node.values):
            key_value = _literal_json(key)
            value = _literal_json(value_node)
            if type(key_value) is not str or key_value in out or value is _UNKNOWN:
                return _UNKNOWN
            out[key_value] = value
        return out
    return _UNKNOWN


def _triton_call(node: ast.AST, name: str) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "triton"
            and node.func.attr == name)


def _configuration_record(node: ast.Call) -> dict:
    row = dict(assessment="unknown", source="triton.Config", line=node.lineno,
               params=None, num_stages=None, options={}, notes=[])
    params_node = node.args[0] if node.args else None
    if len(node.args) > 1:
        row["notes"].append("额外位置参数的配置语义未知")
    seen = set()
    for keyword in node.keywords:
        if keyword.arg is None or keyword.arg in seen:
            row["notes"].append("动态展开或重复配置参数，配置未知")
            continue
        seen.add(keyword.arg)
        if keyword.arg == "kwargs":
            if params_node is not None:
                row["notes"].append("重复配置字典，配置未知")
                params_node = None
            else:
                params_node = keyword.value
            continue
        value = _literal_json(keyword.value)
        if keyword.arg == "num_stages":
            if _positive_int(value):
                row["num_stages"] = value
            else:
                row["notes"].append("num_stages 非正整数字面量，流水配置未知")
        elif value is not _UNKNOWN:
            row["options"][keyword.arg] = value
        else:
            row["notes"].append(f"{keyword.arg} 非有限字面量，配置未知")
    params = _literal_json(params_node)
    if type(params) is dict:
        row["params"] = params
        if any(name.startswith("BLOCK_") and not _positive_int(value)
               for name, value in params.items()):
            row["notes"].append("BLOCK 维度非正整数字面量，tile 配置未知")
    else:
        row["notes"].append("配置参数字典非完整字面量，配置未知")
    if row["num_stages"] is None and "num_stages" not in seen:
        row["notes"].append("未声明 num_stages，不补后端默认值")
    row["notes"].append("配置字面量不证明 shared storage")
    return row


def _configuration_metadata(tree: ast.AST) -> list:
    rows = []
    nodes = sorted(ast.walk(tree), key=lambda node: (getattr(node, "lineno", 0),
                                                    getattr(node, "col_offset", 0)))
    for node in nodes:
        if _triton_call(node, "Config"):
            rows.append(_configuration_record(node))
        elif _triton_call(node, "autotune"):
            configs = next((kw.value for kw in node.keywords if kw.arg == "configs"), None)
            if configs is None and node.args:
                configs = node.args[0]
            if (not isinstance(configs, (ast.List, ast.Tuple))
                    or not configs.elts
                    or any(not _triton_call(item, "Config") for item in configs.elts)):
                rows.append(dict(assessment="unknown", source="triton.autotune",
                                 line=node.lineno, params=None, num_stages=None,
                                 options={}, notes=["autotune 配置集合动态、为空或无法完整解析"]))
        if len(rows) >= _MAX_CONFIGURATIONS:
            rows.append(dict(assessment="unknown", source="configuration_limit",
                             params=None, num_stages=None, options={},
                             notes=["配置数量超出解析边界，其余配置未知"]))
            break
    return rows


def _source_resource_scope_note(tree: ast.AST) -> str:
    """源码 hash 不区分 autotune 配置、多个 binary 或多个 launch。

    只做保守语法识别，不解析候选的运行时对象。别名 import 和动态 jit 包装也不能
    让单 binary 回执被提升为整个源码候选的资源结论。
    """
    nodes = list(ast.walk(tree))
    autotune_names, jit_names = {"autotune"}, {"jit"}
    for node in nodes:
        if (isinstance(node, ast.ImportFrom) and node.module
                and (node.module == "triton" or node.module.startswith("triton."))):
            for alias in node.names:
                if alias.name == "autotune":
                    autotune_names.add(alias.asname or alias.name)
                    return "autotune 的各配置身份未绑定；单份资源回执不能代表整个源码候选"
                if alias.name == "jit":
                    jit_names.add(alias.asname or alias.name)
    if any((isinstance(node, ast.Attribute) and node.attr == "autotune")
           or (isinstance(node, ast.Name) and node.id in autotune_names) for node in nodes):
        return "autotune 的各配置身份未绑定；单份资源回执不能代表整个源码候选"
    for node in nodes:
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "getattr" and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in ("autotune", "jit")):
            return "动态查找的 autotune / jit 作用域未知；单份资源回执不能代表整个源码候选"

    def jit_reference(node):
        return ((isinstance(node, ast.Attribute) and node.attr == "jit")
                or (isinstance(node, ast.Name) and node.id in jit_names))

    helpers = []
    dynamic_helpers = 0
    for node in nodes:
        if (isinstance(node, (ast.Assign, ast.AnnAssign))
                and jit_reference(node.value)):
            return "jit 的动态别名作用域未知；单份资源回执不能代表整个源码候选"
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if any(jit_reference(decorator.func if isinstance(decorator, ast.Call)
                                 else decorator) for decorator in node.decorator_list):
                helpers.append(node)
        elif (isinstance(node, (ast.Assign, ast.AnnAssign))
              and isinstance(node.value, ast.Call) and jit_reference(node.value.func)):
            dynamic_helpers += 1
    if len(helpers) != 1 or dynamic_helpers:
        return "无法确立唯一 jit helper 的 binary 身份；单份资源回执不能代表整个源码候选"
    helper = helpers[0]
    if helper not in tree.body or len(helper.decorator_list) != 1:
        return "jit helper 嵌套或有额外包装，单 launch 作用域未知"
    launches = [node for node in nodes
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript)]
    if len(launches) != 1:
        return "无法确立唯一 launch 的配置身份；单份资源回执不能代表整个源码候选"
    launch = launches[0]
    if (not isinstance(launch.func.value, ast.Name)
            or launch.func.value.id != helper.name):
        return "launch 未直接绑定唯一 jit helper，资源作用域未知"
    if any(isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
           and node.id in (helper.name, "kernel") for node in nodes):
        return "jit helper 或入口存在重绑定，单 launch 作用域未知"
    if sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
           and node.name == helper.name for node in nodes) != 1:
        return "jit helper 的定义不唯一，单 launch 作用域未知"

    parents = {child: parent for parent in nodes for child in ast.iter_child_nodes(parent)}
    parent, owner = parents.get(launch), None
    while parent is not None:
        if isinstance(parent, (ast.For, ast.AsyncFor, ast.While, ast.ListComp,
                               ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.Lambda)):
            return "循环、推导式或延迟调用内的 launch 可运行多个配置，资源作用域未知"
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if owner is not None:
                return "嵌套 helper 中的 launch 调用次数未知，资源作用域未知"
            owner = parent
        parent = parents.get(parent)
    if (owner is None or owner not in tree.body or owner.name != "kernel"
            or owner.decorator_list):
        return "launch 不在明确的单次 kernel 入口内，资源作用域未知"
    if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
           and node.func.id == "kernel" for node in nodes):
        return "源码自行调用 kernel 入口，launch 调用次数未知"
    return _launch_configuration_note(helper, launch)


def _launch_configuration_note(helper: ast.FunctionDef, launch: ast.Call) -> str:
    """只接受可直接读取的单次 launch 配置；不求值全局变量或展开参数。"""
    if (helper.args.vararg is not None or helper.args.kwarg is not None
            or any(isinstance(arg, ast.Starred) for arg in launch.args)
            or any(keyword.arg is None for keyword in launch.keywords)):
        return "launch 参数有动态展开或 helper 签名不固定，配置身份未知"
    keywords = {}
    for keyword in launch.keywords:
        if keyword.arg in keywords:
            return "launch 参数重复，配置身份未知"
        keywords[keyword.arg] = keyword.value
    args = [*helper.args.posonlyargs, *helper.args.args]
    all_args = [*args, *helper.args.kwonlyargs]
    resource_int_names = {"num_stages", "num_warps", "num_ctas", "maxnreg", "stages"}

    def constexpr_annotation(annotation):
        return ((isinstance(annotation, ast.Attribute) and annotation.attr == "constexpr")
                or (isinstance(annotation, ast.Name) and annotation.id == "constexpr"))

    if any(arg.annotation is not None and not constexpr_annotation(arg.annotation)
           for arg in all_args):
        return "helper 参数注解的编译期语义未解析，配置身份未知"

    def integer_resource(name):
        return name.startswith("BLOCK_") or name in resource_int_names

    constexpr = {
        arg.arg for arg in all_args
        if integer_resource(arg.arg)
        or constexpr_annotation(arg.annotation)
    }
    runtime_args = {arg.arg for arg in all_args} - constexpr
    defaults = {arg.arg: value for arg, value in
                zip(args[len(args) - len(helper.args.defaults):], helper.args.defaults)}
    defaults.update((arg.arg, value) for arg, value in
                    zip(helper.args.kwonlyargs, helper.args.kw_defaults) if value is not None)
    bound = dict(defaults)
    if len(launch.args) > len(args):
        return "launch 位置参数无法对应 helper 签名，配置身份未知"
    for arg, value in zip(args, launch.args):
        if arg.arg in keywords:
            return "launch 位置/关键字参数重复，配置身份未知"
        bound[arg.arg] = value
    bound.update(keywords)
    # Runtime tensor/scalar arguments need not be literals. Constexpr values and
    # compiler options must be literal so one call site cannot hide a config search.
    for name in constexpr | (set(keywords) - runtime_args):
        value = _literal_json(bound.get(name))
        if value is _UNKNOWN or (integer_resource(name) and not _physical_int(value)):
            return f"launch 的 {name} 非可确立的字面量资源参数，配置身份未知"
    return ""


def estimate_matmul_smem_bytes(block_sizes: dict[str, int], elem_bytes: int) -> Optional[int]:
    """兼容逻辑 tile 代理：(BM*BK + BK*BN + BM*BN) * elem。

    不是物理 shared 占用，也不是它的下界；不得用于确定拒绝或硬限制归因。
    """
    bm = block_sizes.get("BLOCK_M")
    bn = block_sizes.get("BLOCK_N")
    bk = block_sizes.get("BLOCK_K")
    if not all(_positive_int(value) for value in (bm, bn, bk, elem_bytes)):
        return None
    return (bm * bk + bk * bn + bm * bn) * elem_bytes


def estimate_generic_smem_bytes(block_sizes: dict[str, int], elem_bytes: int) -> Optional[int]:
    """兼容逻辑 BLOCK 乘积代理；不证明物理 shared 占用或其下界。"""
    nums = list(block_sizes.values())
    if not nums or not all(_positive_int(value) for value in (*nums, elem_bytes)):
        return None
    prod = 1
    for v in nums:
        prod *= v
    return prod * elem_bytes


def check_resource_feasibility(code: str, caps, kernel_type: str = "", *,
                               resource_evidence=None) -> ResourceReport:
    """源码默认只有 unknown；显式 harness 资源证据走共用的物理判据。

    kernel_type 保留调用兼容，不授权将逻辑形状视作 shared。源码解析仅提取诊断
    元数据，不执行源码，也不从 task dtype、累加器或 stage 数补造资源证据。
    """
    tree, parse_note = _bounded_source_tree(code)
    if tree is None:
        rep = ResourceReport(limit_bytes=smem_limit_bytes(caps))
        rep.notes.append(parse_note)
        return rep
    scope_note = _source_resource_scope_note(tree)
    if scope_note:
        # provenance 可以保留；这个源码 hash 没有授权评估其中某一个 binary 的字节。
        rep = ResourceReport(limit_bytes=smem_limit_bytes(caps))
        if type(resource_evidence) is SharedMemoryEvidence:
            for target, value in (("evidence_kind", resource_evidence.kind),
                                  ("evidence_producer", resource_evidence.producer),
                                  ("evidence_binding", resource_evidence.binding)):
                if type(value) is str:
                    setattr(rep, target, value)
        rep.notes.append(scope_note)
    else:
        rep = assess_shared_memory(resource_evidence, caps,
                                   binding=source_resource_binding(code))
    pairs = _extract_block_sizes(code)
    rep.block_shapes = [f"{name}={value}" for name, value in pairs]
    rep.configurations = _configuration_metadata(tree)
    if pairs:
        rep.notes.append("BLOCK 字面量摘要按轴展示，不代表同一配置或物理 shared 占用")
    return rep


def check_shape_divisibility(code: str, inputs: dict) -> ResourceReport:
    """BLOCK_* 对输入 shape 的整除性 —— 只在**硬要求**的写法下才拒。

    Triton 一般会自动 mask, 所以 `M % BM != 0` 不是错误而是性能问题;
    LeetCUDA 对 WGMMA 类硬要求才 SKIP。这里认两种硬要求:
      1. 源码里出现 `assert` / `static_assert` 含 BLOCK 与 shape;
      2. 显式 `# hard-require: BLOCK_M | M` 注释 (给 Designer 的标记)。
    认不出就放行。
    """
    rep = ResourceReport()
    if not inputs:
        return rep
    hard = bool(re.search(r"hard-require|static_assert", code))
    if not hard:
        return rep
    pairs = dict(_extract_block_sizes(code))
    if not pairs:
        return rep
    # 取第一个二维 shape (M, N) 与 BLOCK_M/N 对照; 对不上才拒
    shape = None
    for v in inputs.values():
        try:
            import numpy as np
            arr = np.asarray(v)
            if arr.ndim >= 2:
                shape = arr.shape
                break
        except Exception:  # noqa: BLE001
            continue
    if shape is None:
        return rep
    problems = []
    for dim_name, block_name in (("M", "BLOCK_M"), ("N", "BLOCK_N")):
        b = pairs.get(block_name)
        if b and shape and shape[0] % b != 0 and dim_name == "M":
            problems.append(f"{dim_name}={shape[0]} % {block_name}={b} != 0")
        if b and shape and len(shape) > 1 and shape[-1] % b != 0 and dim_name == "N":
            problems.append(f"{dim_name}={shape[-1]} % {block_name}={b} != 0")
    if problems:
        rep.feasible = False
        rep.reason = "资源预检: 形状不满足硬整除要求 (" + ", ".join(problems) + ")"
    return rep
