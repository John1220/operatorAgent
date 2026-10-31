"""候选核的结构签名 —— 「这一轮生成的东西是不是新的」

动机 (缺口 2 的读者侧 + C5): Conductor 的轮次循环是线性的, 每轮重新生成一份代码就送进
三级验证, 而 L3 是一次真机基准 (SSH + 共享卡上几秒)。以前没有任何机制能发现「第 3 轮的核
只是第 1 轮改了变量名」—— 于是整条验证的成本花在一份**已经验证过**的候选上。

两个键, 分工刻意不同:

* `structural` —— **α-等价**的 AST 哈希: 标识符按首次出现顺序重命名 (`x`→`v0`, `y`→`v1`),
  丢掉名字本身, 保留结构、常量与调用名。两份代码只有变量名不同时哈希相等; BLOCK 从 64 改成
  128 时哈希不等。**假阳 = 丢掉一个真的更好的核**, 所以只拿它做精确重复的判断。
* `descriptor` —— 粗粒度特征元组 (API 集合 / 循环数 / launch 配置 / 块尺寸集合)。它**会**
  把不同的核归到同一个桶里 (那正是它的用途: 看「配方换了但核没变」), 所以只观测、不判决。

不做的事: embedding / 相似度阈值 (与 `kernel_db` 的同一条理由, 见 docs/research/43), 以及
编译器 IR 哈希 (Triton 的 TTIR/PTX 要真编译才有, 而 L1 之前正是想省掉的那一步)。
"""
import ast
import hashlib
from typing import Optional

# 只用于描述符的 API 名: 都是本项目配方里会动的东西, 不是「常见 Triton API」全集
_WATCHED_APIS = (
    "dot", "load", "store", "sum", "max", "min", "exp", "log", "where", "softmax",
    "atomic_add", "atomic_max", "make_block_ptr", "advance", "trans", "cumsum",
    "reshape", "arange", "program_id", "num_programs", "cdiv", "next_power_of_2",
)
# launch 配置里能进入描述符的字段 (块尺寸单列, 因为配方最常改的就是它)
_CONFIG_KEYS = ("num_warps", "num_stages", "BLOCK_M", "BLOCK_N", "BLOCK_K",
                "BLOCK_SIZE", "EVEN_K", "GROUP_M")


def _alpha_normalize(tree: ast.AST) -> str:
    """把 AST 摊成一个「改名字不变、改结构必变」的字符串。

    标识符统一换成 v0/v1/… (按首次出现顺序)。关键字实参名**也**走同一张表:
    `kernel(BLOCK_M=64)` 里的 `BLOCK_M` 与被调函数形参是同一个名字, 一次合法的改名会把两处
    一起改 —— 不换它的话「只改名」的副本会被判成不同候选, 而那正是本模块要抓的唯一一类。
    属性名不换 (`tl.dot` 的 `dot` 是 API 名, 参与区分候选)。

    ⚠️ 这一版之前栽过的错: 函数节点走了「打印名字就 return」的捷径, 于是**函数体根本不在
    哈希里** —— 键只看得见「有几个函数、叫什么」。两份核函数体完全不同的代码会被判成重复,
    而「改名应当相等」那条用例照样绿 (它对错了的原因做确认)。所以现在每个特殊分支都只
    替换它要替换的那一个字段, 其余字段一律继续递归。
    """
    rename = {}

    def name_of(x: str) -> str:
        if x not in rename:
            rename[x] = f"v{len(rename)}"
        return rename[x]

    def walk(node):
        if isinstance(node, ast.Name):
            return f"Name({name_of(node.id)})"
        if isinstance(node, ast.arg):
            return f"arg({name_of(node.arg)})"
        if isinstance(node, ast.Attribute):
            return f"Attr({walk(node.value)}.{node.attr})"
        if isinstance(node, ast.keyword):
            return (f"kw({name_of(node.arg)},{walk(node.value)})" if node.arg
                    else f"kw(**,{walk(node.value)})")
        parts = [type(node).__name__]
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                parts.append(f"{field}=[" + ",".join(walk(v) for v in value) + "]")
            elif isinstance(value, ast.AST):
                parts.append(f"{field}={walk(value)}")
            elif isinstance(value, str) and field in ("name", "arg"):
                # 定义节点的 name 与形参/实参名一样是「可自由改的名字」
                parts.append(f"{field}={name_of(value)}")
            elif value is not None:
                parts.append(f"{field}={value!r}")
        return "(" + " ".join(parts) + ")"

    return walk(tree)


def _call_names(tree: ast.AST) -> set:
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute):
                out.add(f.attr)
            elif isinstance(f, ast.Name):
                out.add(f.id)
    return out


def _launch_configs(tree: ast.AST) -> dict:
    """收集 num_warps / num_stages / 块尺寸这类**编译期配置**的字面量取值。

    块尺寸的命名实在太多 (`BLOCK` / `BLOCK_SIZE` / `BLOCK_M` / `TILE_M` / `BK`), 枚举名字
    必然漏 —— 第一版就漏了最常见的裸 `BLOCK`。改判据为「全大写且长度 >1」: Triton 代码里
    这种写法基本就是 constexpr 旋钮。`num_warps` / `num_stages` 是小写, 单独列。

    只认字面量: `BLOCK=64` 进描述符, `BLOCK=next_power_of_2(N)` 不进 —— 它的值取决于运行期
    shape, 拿它当键会把「一个静态配置」与「一个自适应配置」当成同一类。
    """
    found = {}
    for n in ast.walk(tree):
        targets = []
        if isinstance(n, ast.Assign):
            targets = [t for t in n.targets if isinstance(t, ast.Name)]
            value = n.value
        elif isinstance(n, ast.keyword):
            targets = [ast.Name(id=n.arg)] if n.arg else []
            value = n.value
        else:
            continue
        for t in targets:
            if t.id in found:
                continue
            is_knob = t.id in _CONFIG_KEYS or (t.id.isupper() and len(t.id) > 1)
            if not is_knob:
                continue
            if isinstance(value, ast.Constant) and isinstance(value.value, (int, float, bool)):
                found[t.id] = value.value
    return found


def candidate_signature(code: str) -> Optional[dict]:
    """返回 {"structural": str, "descriptor": tuple, "configs": dict} ; 读不出结构返回 None。

    None 的两种情形都是**真的没有候选可言**: 语法错 (走不到 L3), 以及没有任何函数定义的
    模块 (空串、只有 import 的半个产出)。给这两种编一个键, 只会让「重复率」统计里混进
    一批把「什么都没产出」当成「产出了同一个东西」的假命中。
    """
    try:
        tree = ast.parse(code or "")
    except (SyntaxError, ValueError):
        return None
    if not any(isinstance(n, ast.FunctionDef) for n in ast.walk(tree)):
        return None

    norm = _alpha_normalize(tree)
    structural = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]

    calls = _call_names(tree)
    watched = tuple(sorted(a for a in calls if a in _WATCHED_APIS))
    loops = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.For, ast.AsyncFor, ast.While)))
    kernels = sum(1 for n in ast.walk(tree)
                  if isinstance(n, (ast.FunctionDef,)) and n.name.startswith(("kernel", "_")))
    cfg = _launch_configs(tree)
    descriptor = (watched, loops, kernels, tuple(sorted(cfg.items())))
    return {"structural": structural, "descriptor": descriptor, "configs": cfg}


def signature_of(code: str) -> str:
    """只要结构键; 读不出结构时返回空串 (调用方据此跳过比较, 不当成「与空串相同」)"""
    s = candidate_signature(code)
    return s["structural"] if s else ""
