"""§12 的**真**模板实例化器（Triton matmul）—— 把参数点变成一段**真能跑**的源码

这一片的动机（批次审查 No.2 的第二半）：Stage 3 的六片机制（参数空间 → 剪枝 → 候选队列
→ 选择器 → 停止条件）此前只在一个**假**模板后端上受力，而「模板源被执行」这件事从来没在
真机上发生过。CUTLASS/TVM 那类真后端要么没有，要么产出 C++（**今天没有任何后端能执行
非 Python 源**，见 CHECKLOG I65）。所以最划算的一步是：**用 Triton 做真模板后端** ——
它产出的源码就是 Python/Triton，现有 L1/L2/L3 立刻能吃，于是整条链可以在 H20 上端到端跑。

三个口径写死在这里：

1. **模板的身份** = `(template_id, INSTANTIATOR_VER, 参数)`（`candidates.dedup_key`）。
   改了发射器的语义**必须**升 `INSTANTIATOR_VER`：否则去重键会把两个不同的构建读成同一个
   候选（`backend_caps.Instantiation` 的 docstring 就是为这件事写的）。
2. **`stages` 与 `warps` 是启动参数**（`num_stages`/`num_warps`），不在源码文本里 ——
   这与 L1 源码正则预检拿不到流水级数是同一件事（doc 78 边界），所以它们**只影响估算与
   编译元数据**，不许被说成「源码里写着」。
3. **轴的值域来自调用方给的硬件上限**（缺上界时给一个保守的默认格），因为「这张卡上
   哪些 tile 放得下」是硬件属性，不是模板属性（§7 可迁移 vs 不可迁移）。
"""

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .backend_caps import Instantiation
from .candidates import ParamSpace

#: 模板身份的一半。**改发射语义要一起升**（见模块 docstring 第 1 条）。
TEMPLATE_ID = "triton_matmul_v1"
INSTANTIATOR_VER = "1"

#: 默认格（没有硬件上限时的保守取值）。轴序 = 枚举序 = 剪枝与截断的可见顺序。
DEFAULT_BM: Tuple[int, ...] = (64, 128)
DEFAULT_BN: Tuple[int, ...] = (64, 128)
DEFAULT_BK: Tuple[int, ...] = (32, 64)
DEFAULT_STAGES: Tuple[int, ...] = (1, 2, 3)
DEFAULT_WARPS: Tuple[int, ...] = (4, 8)

_HEAD = '''import torch
import triton
import triton.language as tl


@triton.jit
def _mm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
               stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        mask_a = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        mask_b = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        # NOTE: 默认精度 = TF32 —— fp32 容差 1e-4 下过不了 L2（doc 79 观察①），
        # 所以模板显式写 ieee：模板要么精确要么别声称自己过得了 fp32 的门。
        acc = tl.dot(a, b, acc, input_precision="ieee")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float32), mask=mask_c)
'''

_BODY = '''

def kernel(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    _mm_kernel[grid](A, B, C, M, N, K,
                     A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                     C.stride(0), C.stride(1),
                     BLOCK_M={bm}, BLOCK_N={bn}, BLOCK_K={bk},
                     num_warps={warps}, num_stages={stages})
    return C
'''


def _grain(values: Sequence[int], cap: Optional[int]) -> Tuple[int, ...]:
    """只要 <= cap 的取值；全都超了就给最小的那个（由剪枝去摘，不在这里偷偷改语义）"""
    if cap is None:
        return tuple(values)
    kept = tuple(v for v in values if v <= cap)
    return kept or (min(values),)


def param_space(smem_limit_bytes: Optional[int] = None,
                elem_bytes: int = 4) -> ParamSpace:
    """§12 参数空间；任务 dtype/tile 不证明 shared layout，不据此缩格。"""
    cap_m = cap_n = cap_k = None
    return ParamSpace(
        template_id=TEMPLATE_ID,
        axes={"BLOCK_M": _grain(DEFAULT_BM, cap_m), "BLOCK_N": _grain(DEFAULT_BN, cap_n),
              "BLOCK_K": _grain(DEFAULT_BK, cap_k), "stages": DEFAULT_STAGES,
              "warps": DEFAULT_WARPS},
        instantiator_ver=INSTANTIATOR_VER, backend="triton_template")


def instantiate(params: Mapping[str, Any]) -> Instantiation:
    """参数点 → 源码。缺轴/轴值不可用 ⇒ **抛**（不许默认成某个 tile）"""
    need = ("BLOCK_M", "BLOCK_N", "BLOCK_K", "stages", "warps")
    missing = [k for k in need if k not in params]
    if missing:
        raise ValueError(f"模板 {TEMPLATE_ID} 缺参数轴 {missing}: 不补默认值 —— "
                         f"「没说」与「说了 128」是两件事")
    for key in need:
        value = params[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"参数轴 {key}={value!r} 不是正整数 ⇒ 拒绝实例化")
    source = _HEAD + _BODY.format(bm=params["BLOCK_M"], bn=params["BLOCK_N"],
                                  bk=params["BLOCK_K"], warps=params["warps"],
                                  stages=params["stages"])
    return Instantiation(
        source=source, instantiator_ver=INSTANTIATOR_VER,
        evidence=(f"{TEMPLATE_ID}@{INSTANTIATOR_VER} "
                  f"BM={params['BLOCK_M']} BN={params['BLOCK_N']} BK={params['BLOCK_K']} "
                  f"warps={params['warps']} stages={params['stages']} (stages/warps 是"
                  f"启动参数, 不在源码文本里)"))


class TritonMatmulTemplateBackend:
    """「模板那一半」的提供者 —— `param_space()` 返空间, `instantiate()` 返源码

    刻意**不继承 `Backend`** (那是执行后端的契约, 本类四个执行方法一个都给不出):
    继承会两头堵 —— 不补抽象方法 ⇒ 组合后端在**全新进程**里构造不出来 (2026-10-02
    实测 TypeError; 全套件绿只因同文件早先用例把类污染成了具体类, CHECKLOG I71);
    补桩 ⇒ 混进 `concrete_backends()` 的登记表完备性守卫 (test_backend_caps)。
    「既是执行后端又声明空间」的对象是 `TritonTemplateSSHExecutor`, 不是本类。
    """

    def __init__(self, smem_limit_bytes: Optional[int] = None, elem_bytes: int = 4):
        self._space = param_space(smem_limit_bytes, elem_bytes)

    @property
    def label(self) -> str:
        return "triton_matmul_template"

    @property
    def capabilities(self) -> frozenset:
        return frozenset()

    def param_space(self, contract: object = None) -> ParamSpace:
        return self._space

    def instantiate(self, template_id: str, params: Mapping[str, Any],
                    contract: object = None) -> Instantiation:
        if template_id != TEMPLATE_ID:
            raise ValueError(f"未知模板 {template_id!r}: 本后端只有 {TEMPLATE_ID!r}")
        return instantiate(params)

    def as_dict(self) -> Dict[str, Any]:
        return {"backend": self.label, "template_id": TEMPLATE_ID,
                "instantiator_ver": INSTANTIATOR_VER, "size": self._space.size(),
                "axes": {k: list(v) for k, v in self._space.axes.items()}}

class TritonTemplateSSHExecutor:
    """把「真执行」与「真模板」合成一个后端：SSH 执行 + Triton 模板空间

    为什么要有这一层: `Conductor.backend` **一个对象**同时承担两件事 ——
    `Verifier(self.backend)` 要它是可执行后端, 轮首的 `param_space_of(self.backend, task)`
    要它声明参数空间 (§12 的开关)。今天生产里的真后端 (`SSHExecutor`) 是**代码型**
    (继承基类的 `param_space()` ⇒ None), 而模板模块只有空间与实例化器。两者都在真机上
    存在之后, 缺的正是这一个**显式组合**。

    刻意**不**用 `getattr/isinstance` 嗅探 (那会被 `tests/unit/test_backend_caps.py` 的 AST
    守卫拦下, 也是本项目反复禁的): 这里只有两个明确的覆写, 其余方法按名字转发给被包住的
    执行后端。`param_space`/`instantiate` 来自模板模块 —— 一个后端要么声明了空间并实现了
    实例化, 要么两者都没有, 不许只声明一半。
    """

    def __init__(self, executor: Any, smem_limit_bytes: Optional[int] = None,
                 elem_bytes: int = 4):
        self._executor = executor
        self._template = TritonMatmulTemplateBackend(smem_limit_bytes, elem_bytes)

    # ── 模板那一半 (§12) ──
    @property
    def template_backend(self) -> TritonMatmulTemplateBackend:
        return self._template

    def param_space(self, contract: object = None) -> ParamSpace:
        return self._template.param_space(contract)

    def instantiate(self, template_id: str, params: Mapping[str, Any],
                    contract: object = None) -> Instantiation:
        return self._template.instantiate(template_id, params, contract)

    # ── 执行那一半：按名转发 (`__getattr__`) ──
    def __getattr__(self, name: str) -> Any:
        # 正常属性查找**失败**才走到这里 ⇒ 上面的覆写 (param_space/instantiate/
        # label/capabilities) 不会被转发路径顶掉。转发给被包住的执行后端; 名字对
        # 不上就照旧 AttributeError (fail-closed), 不会静默返回 None —— 「少一个
        # 方法」藏不住。⚠️ 本注释曾写「不用 __getattr__」: 那是第一版草稿的残留,
        # 与代码正相反 (提交 d66c430 的说明与类 docstring 才是实现口径)。
        return getattr(self._executor, name)

    @property
    def label(self) -> str:
        return f"{self._executor.label}+{TEMPLATE_ID}@${INSTANTIATOR_VER}"

    @property
    def capabilities(self) -> frozenset:
        return self._executor.capabilities

    @property
    def fingerprint_label(self) -> str:
        """环境指纹取**执行侧**身份 (I72): 模板那一半是候选来源, 不是执行环境。

        实测教训: 不覆写时, 组合 label (`…+triton_matmul_v1@$1`) 进指纹 ⇒ 同机/同卡/
        同 runner 仅换组合壳, 首轮被判「环境漂移」degraded (can_persist=False) —— 每个
        新模板后端的第一份读数永远不可入库; 第 2 轮起 drift 比较的「上一次」也变成同
        一组合后端, 才自然恢复 valid。
        """
        return self._executor.label
