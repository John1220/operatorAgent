"""§22.1 的**一次运行的布局**：七个角色目录 + 唯一的写入口

设计原文（`docs/design/autonomous-operator-agent-design.md:1536-1550`）:

```
run_root/
├── contract/       # Agent 只读
├── harness/        # Agent 只读
├── baseline/       # Agent 不可见或只读 artifact
├── candidate/      # Agent 只写
├── build/          # 构建进程写
├── measurements/   # Oracle 写
└── report/         # Report Writer 写
```

候选**不能**写入 `contract/`、`harness/`、`baseline/`、`measurements/`、`report/`；
「这条不是『约定』，而是权限位：违反时写入必须**失败**，而不是『事后被审计发现』」。

本模块落地的是这条要求的**可机器化那一半**，并且把另一半写清楚（边界见文末）。

## 两层，各管一件事

1. **权限位（内核层）**：五个受保护目录建成 `0500`（属主可读可进、**不可写**），
   经本模块写进去的文件建成 `0400`。于是任何**裸写**——
   新建条目要靠目录的写位，覆盖已有文件要靠文件的写位——**在同一进程内也失败**
   （`PermissionError`），不依赖任何调用方自觉。
2. **唯一写入口（Python 层）**：`RunLayout.write_for(role, rel, data)` 是唯一会
   打开写窗的地方（受保护目录在写的一瞬间临时 `chmod 0700`，`finally` 复原 `0500`），
   并且它按**声明角色**判权：`ROLE_WRITERS` 一格一个写者，跨角色写直接
   `PermissionError`。

两层是**独立的**：把权限位改回去（变异表 M2）也还有角色判据；把角色判据删掉
（M1）也还有权限位。变异表 `benchmarks/mut_specs/run_layout_doors.py` 每行钉一层。

## 与 §21 判据是**同一份源**

§21.5 的「受保护路径」判据（`agent_runtime.PROTECTED_DIR_NAMES`）与这里的目录名
必须是同一个元组 —— 判据和执行读两份名单，迟早会出现「判据说 protected、
文件系统说 writable」或反过来的情形。所以那个元组定义在本模块，
`agent_runtime` 从本模块**再导出**（守卫断言两边是同一个对象）。
`candidate_path(run_root)` 与本模块的 `path("candidate/code.py")` 也必须给出同一条路径。

## ⚠️ 边界（不许读成本模块已经做到的事）

- **挡不住同 uid 的敌意候选**：候选今天就在 harness 进程里 `exec`（本地）或在远端
  以 runner 的 uid 落 `/tmp` 临时文件后 `exec_module`（`remote_scripts/run_kernel.py`），
  与 harness **同一个 uid**。属主可以自己 `chmod` 回去，也可以直接改文件。
  权限位拦的是**误写与非敌意越权**，不是敌意攻击 —— 后者要设计 §22.2 的独立 uid /
  容器 / 进程隔离，今天**没有**（`run_kernel_xpu.py` 那条厂商卡路连运行时盾与容差都没有）。
- **远端零覆盖**：本模块只描述控制端这一次运行的目录。远端 runner 侧的运行布局、
  候选落盘位置与权限位**一个字都没说**（那是 §22.2 与远端 runner 的事）。
- **它不是函数沙箱**：能执行什么、能联网、能起进程、能用多少内存，本模块一概不管
  （§22.2 清单里的编译超时 / 运行超时 / watchdog / quota / 网络关闭 / 进程树清理 /
  artifact 大小限制**都还没落地**）。
- **生产调用点只覆盖一部分驱动**：`Conductor(run_root=...)` 默认 `""`（不传就不建布局），
  生产驱动与插件无人值守入口显式提供根目录。**过门**的当前候选落进 `candidate/`，
  S1 轮次历史另经 candidate/oracle/report_writer 各自的 `write_for` 留存源码、
  完整判定和报告。被 §21 拒掉的候选只留哈希，不保存其执行源码。
  最终 ArtifactStore、kernel_db 等既有产物仍沿用原来的位置；这不声称远端或同 UID
  敌意代码已被隔离。
"""

import itertools
import os
import stat
import uuid
from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Tuple


def write_once_artifact(path, data: bytes) -> None:
    """Create ordinary tool evidence once, without opening protected write windows.

    HITL evidence directories are already reserved by their owner. O_EXCL keeps
    existing artifacts (including symlinks) intact; no chmod or replacement is
    performed. Protected RunLayout roles still require write_for().
    """
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o400)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def create_private_mutable_file_once(path) -> None:
    """Reserve an ordinary private database path exclusively, never a write window.

    Intended for mutable control databases outside protected RunLayout roles.
    The caller owns the directory; existing files and symlinks are refused.
    """
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    os.close(descriptor)


def publish_once_artifact(path, data: bytes) -> None:
    """Atomically publish a complete create-only lifecycle control record."""
    final = os.fspath(path)
    temporary = final + '.pending-' + uuid.uuid4().hex
    try:
        write_once_artifact(temporary, data)
        os.link(temporary, final)  # Fails on an existing file; never replaces it.
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

# ─── 闭合词汇 ①：七个目录格 ──────────────────────────────────
#: 一次运行的七个角色目录（顺序即设计原文里的顺序）。
DIR_NAMES: Tuple[str, ...] = (
    "contract", "harness", "baseline", "candidate", "build",
    "measurements", "report",
)

#: §22.1 点名的「候选不能写进去」的五个。**这是唯一的定义处**：
#: `agent_runtime`（§21.5 的判据）从本模块导入，判据与执行读同一份名单。
PROTECTED_DIR_NAMES: Tuple[str, ...] = (
    "contract", "harness", "baseline", "measurements", "report",
)

#: 可写目录（候选/构建进程可以在里面创建**新**文件）。它们与受保护目录的差别只有一条：
#: 权限位是 `0700`，所以裸写新文件不会失败 —— 而**已有 artifact** 仍是 `0400`。
WRITABLE_DIR_NAMES: Tuple[str, ...] = ("candidate", "build")

#: 「Agent 只写」的那一格的目录名。`agent_runtime.candidate_path` 读它（再导出），
#: 于是「候选源码写在哪一格」也只有一份定义。
CANDIDATE_DIR: str = "candidate"

#: 目录权限位。受保护目录**不可写**（属主也不行），可写目录只给属主。
PROTECTED_DIR_MODE = 0o500
WRITABLE_DIR_MODE = 0o700
#: 经本模块写出的文件一律 `0400` —— 「写一次的证据」不许被再写一次覆盖。
#: 覆盖的唯一办法是再走一次 `write_for`（它会按角色判权，然后 `os.replace`）。
ARTIFACT_MODE = 0o400

# ─── 闭合词汇 ②：角色 → 能写哪一格 ───────────────────────────
#: 角色名。每个角色名都要有**声明者**与**读者**，否则就是死词汇：
#:   * `harness`     —— 声明者/读者都是 harness 自己（契约、harness 资产、基线 artifact）；
#:   * `candidate`   —— 读者是 L1/L2/L3 与 §21 的 diff 判据（它写的是候选源码）；
#:   * `build`       —— 读者是 L1 编译步骤（它写的是构建产物/日志）；
#:   * `oracle`      —— 读者是 §10 的判决与 §14 的入库判据（它写的是测量记录）；
#:   * `report_writer` —— 读者是报告出口（它写的是 report/ 下的成稿）。
ROLE_HARNESS = "harness"
ROLE_CANDIDATE = "candidate"
ROLE_BUILD = "build"
ROLE_ORACLE = "oracle"
ROLE_REPORT_WRITER = "report_writer"

#: 一格一个写者。`write_for` 按它判权：不在名单里的角色写不进去。
#: ⚠️ 这张表**不许**放宽成「harness 什么都能写」—— 那就没有唯一的写入口了，
#: 而 §22.1 的整条主张就是「越权写入在写的那一刻失败」。
ROLE_WRITERS: Dict[str, str] = {
    "contract": ROLE_HARNESS,
    "harness": ROLE_HARNESS,
    "baseline": ROLE_HARNESS,
    "candidate": ROLE_CANDIDATE,
    "build": ROLE_BUILD,
    "measurements": ROLE_ORACLE,
    "report": ROLE_REPORT_WRITER,
}

#: 反查表（读者：`write_for` 的判权与守卫的「这张表是不是闭合的」）。
WRITABLE_BY: Dict[str, FrozenSet[str]] = {
    d: frozenset({r}) for d, r in ROLE_WRITERS.items()
}

#: 角色名的闭合列表（守卫读它核「`ROLE_WRITERS` 的值域没跑出去」）。
ROLES: FrozenSet[str] = frozenset({
    ROLE_HARNESS, ROLE_CANDIDATE, ROLE_BUILD, ROLE_ORACLE, ROLE_REPORT_WRITER,
})


class LayoutError(RuntimeError):
    """布局本身不成立（目录缺失/权限位不对/不是目录/是符号链接）"""


class CrossRoleWrite(PermissionError):
    """越权写：这个角色不许写这一格。

    继承 `PermissionError` 是刻意的 —— 调用方写 `except PermissionError` 就该同时
    接住「内核不给写」与「角色不许写」两种，它们是同一件事的两个层次。
    """


def _norm_rel(rel: str) -> str:
    """校验并规范化一个**布局内相对路径**。

    拒绝：空、绝对路径、任何 `..` 段、Windows 风格分隔符。**不是**在防敌意输入
    （本项目里写这些路径的是 harness 自己），而是让「越出 run_root」这件事在类型上
    不可表达 —— 越界路径没有别的表达方式，只能靠 `..` 或绝对路径，两条都堵掉。
    """
    if not isinstance(rel, str) or not rel.strip():
        raise LayoutError("布局内路径不能为空")
    if rel.startswith("/") or rel.startswith("\\") or ":" in rel.split("/")[0]:
        raise LayoutError(f"布局内路径必须是相对的: {rel!r}")
    parts = [p for p in rel.replace("\\", "/").split("/") if p not in ("", ".")]
    if not parts:
        raise LayoutError(f"布局内路径不能只由 ./ 组成: {rel!r}")
    if any(p == ".." for p in parts):
        raise LayoutError(f"布局内路径不许含 '..': {rel!r}")
    if parts[0] not in DIR_NAMES:
        raise LayoutError(f"布局内路径的第一段必须是七个目录之一，收到 {parts[0]!r}: {rel!r}")
    return "/".join(parts)


def layout_path(run_root: str, rel: str) -> str:
    """**纯**路径构造：布局内相对路径 → 绝对路径。不建目录、不核对权限位、无 IO。

    与 `RunLayout.path()` 是同一条规则的两个入口（后者是给已经建好布局的调用方）。
    分开的理由：`agent_runtime.candidate_path` 在 §21 的**判据**路径上被调，那里
    只该做字符串运算 —— 判据顺手建目录会把「判一下」变成「有副作用」。

    读者：`Conductor` 把 `Patch.path` 接成真路径（§22.1 落地后 §21.5 的判据判的是
    真实路径，而不是声明的角色路径）。
    """
    if not isinstance(run_root, str) or not run_root.strip():
        raise LayoutError("layout_path 需要一个非空的 run_root")
    return os.path.join(os.path.abspath(os.path.expanduser(run_root)),
                        *_norm_rel(rel).split("/"))


@dataclass(frozen=True)
class RunLayout:
    """一次运行的布局句柄。构造只经 `run_layout()` —— 它建目录并核对权限位。"""

    root: str

    # ── 只读侧 ──────────────────────────────────────────────
    def path(self, rel: str) -> str:
        """布局内相对路径 → **绝对**路径（唯一路径构造点）。

        读者：任何要**读**这些 artifact 的地方（含 §21 把 `Patch.path` 接成真路径）。
        它不做 IO、不建目录 —— 读一个还不存在的东西要报 ENOENT，而不是顺手把它造出来。
        """
        return layout_path(self.root, rel)

    def mode_of(self, dir_name: str) -> int:
        """某一格的权限位（读者：守卫与审计；也是「布局是不是原样」的读数）"""
        st = os.lstat(self.path(dir_name))
        return stat.S_IMODE(st.st_mode)

    def describe(self) -> Dict[str, object]:
        """布局的**数据形态**（读者：报告/审计/守卫，不用去 stat 文件系统猜）"""
        return {
            "root": self.root,
            "dirs": {d: {"writer": ROLE_WRITERS[d],
                         "protected": d in PROTECTED_DIR_NAMES,
                         "mode": oct(self.mode_of(d))} for d in DIR_NAMES},
            "protected_dir_names": list(PROTECTED_DIR_NAMES),
            "enforced_by": ["dir_file_modes", "single_write_entry"],
            # ⚠️ 这两个键是给读者用的**否定**声明: 它们恒为 False, 而不是「还没填」。
            "blocks_same_uid_candidate": False,
            "covers_remote_runner": False,
        }

    # ── 写侧（唯一入口）─────────────────────────────────────
    def ensure_dir(self, role: str, rel: str) -> str:
        """按角色建一个**子目录**（`write_for` 不会顺手建父目录 —— 结构要显式声明）。

        读者：artifact 的真正路径是嵌套的（`measurements/<kernel>/<hw>/<ts>/…`），
        所以「不隐式建目录」不能变成「嵌套路径不可表达」。走同一个角色判据：
        越权建目录与越权写文件是同一件事的两个前奏。
        """
        norm = _norm_rel(rel)
        parts = norm.split("/")
        self._require_role(role, parts[0])
        target = self.path(norm)
        protected = parts[0] in PROTECTED_DIR_NAMES
        # 只对**已经存在**的祖先开写窗: 这条路径上还没有的层级等 `makedirs` 建出来,
        # 建完立刻按这一格的档位收权限（`_dirs_below` 那一段）。
        window = [d for d in self._window_dirs(norm) if os.path.isdir(d)]
        if protected:
            for d in window:
                os.chmod(d, WRITABLE_DIR_MODE)
        try:
            os.makedirs(target, exist_ok=True)
            # 新子目录的档位**跟着这一格走**:
            #   * 受保护格: 子目录也 0500 —— 不能只 chmod 末级, 中间层留 0755 的话
            #     `measurements/<k>/` 就是那扇绕过写位的门;
            #   * 可写格: 子目录 0700 —— 收成 0500 会把候选/构建进程的 scratch 变成
            #     只读（那是在一个「可写格」里造出一格不可写的空间, 与本节自相矛盾）。
            want = PROTECTED_DIR_MODE if protected else WRITABLE_DIR_MODE
            for sub in self._dirs_below(norm):
                if os.path.isdir(sub):
                    os.chmod(sub, want)
        finally:
            if protected:
                for d in reversed(window):
                    os.chmod(d, PROTECTED_DIR_MODE)
        return target

    def _window_dirs(self, norm: str) -> list:
        """要临时打开写位的**整条父链**（受保护格才需要）。

        WHY 是整条链而不是只有顶层: `measurements/gemm/h20/l3.json` 这一笔要创建目录项的是
        `h20/`（0500），只把 `measurements/` 开成 0700 是不够的 —— 第一次写嵌套路径就会
        得到 `PermissionError`（实测撞到过）。反过来, 只开末级不开顶层同样不行。
        """
        parts = norm.split("/")
        if parts[0] not in PROTECTED_DIR_NAMES:
            return []
        return [self.path("/".join(parts[:i])) for i in range(1, len(parts))]

    def _dirs_below(self, norm: str) -> list:
        """`dir_name/a/b` → `[root/dir_name, root/dir_name/a, root/dir_name/a/b]`"""
        parts = norm.split("/")
        return [self.path("/".join(parts[:i])) for i in range(1, len(parts) + 1)]

    def _require_role(self, role: str, dir_name: str) -> None:
        if role not in ROLES:
            raise LayoutError(f"未声明的角色 {role!r}（闭合列表: {sorted(ROLES)}）")
        declared = ROLE_WRITERS.get(dir_name)
        if declared != role:
            raise CrossRoleWrite(
                f"角色 {role!r} 不许写 {dir_name!r}（这一格声明的写者是 {declared!r}）"
                f" —— §22.1「违反时写入必须失败」")

    def write_for(self, role: str, rel: str, data) -> str:
        """**唯一**的写入口。越权写在这里失败，不在事后审计里。

        Args:
            role: 写者角色（`ROLES` 之一）。这一格声明的写者不是它就 `CrossRoleWrite`。
            rel:  布局内相对路径（第一段必须是七个目录之一）。
            data: `bytes` 或 `str`（str 按 UTF-8 编码）。

        Returns:
            写入后的**绝对**路径。

        顺序上刻意先判**角色**再碰文件系统：越权的调用不该留下任何痕迹
        （连临时文件都不该产生 —— 否则「失败」这件事本身会污染 `measurements/`，
        而那正是要保护的东西）。
        """
        norm = _norm_rel(rel)
        parts = norm.split("/")
        dir_name, tail = parts[0], parts[1:]
        if not tail:
            raise LayoutError(f"要写的是文件，不是目录: {rel!r}")
        # 先判**角色**再碰文件系统：越权的调用不该留下任何痕迹（连临时文件都不该产生
        # —— 否则「失败」这件事本身会污染 measurements/，而那正是要保护的东西）。
        self._require_role(role, dir_name)

        payload = data.encode("utf-8") if isinstance(data, str) else bytes(data)
        target_dir = self.path(dir_name)
        # 目标目录必须是**真目录**而不是符号链接：否则 `write_for` 会成为「把写重定向到
        # 任何地方」的工具（`O_NOFOLLOW` 管不了目录这一层）。
        st_dir = os.lstat(target_dir)
        if not stat.S_ISDIR(st_dir.st_mode) or stat.S_ISLNK(st_dir.st_mode):
            raise LayoutError(f"{dir_name!r} 不是真目录: {target_dir}")

        final = os.path.join(target_dir, *tail)
        parent = os.path.dirname(final)
        if not os.path.isdir(parent) or os.path.islink(parent):
            raise LayoutError(f"父目录不存在或不是真目录: {parent}")

        # 写窗: 受保护格的**整条父链**一起开, `finally` 无条件复原（含抛错路径）。
        # 半途抛错也必须复原 —— 布局的权限位是**下一步的输入**, 留一个 0700 的
        # `measurements/` 比写失败更糟。
        window = self._window_dirs(norm)
        tmp = os.path.join(parent, f".{os.path.basename(final)}.tmp.{os.getpid()}")
        for d in window:
            os.chmod(d, WRITABLE_DIR_MODE)
        try:
            fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                         0o600)
            try:
                os.write(fd, payload)
                os.fsync(fd)
            finally:
                os.close(fd)
            os.chmod(tmp, ARTIFACT_MODE)
            os.replace(tmp, final)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
            for d in reversed(window):
                os.chmod(d, PROTECTED_DIR_MODE)
        return final


#: 「一次运行一个新根」的进程内单调序号。与「微秒时间戳 + pid」一起保证唯一性 ——
#: 时间戳的时钟粒度**不保证**连续两次调用拿到不同的值（第一版就是在这里撞的）。
_RUN_SEQ = itertools.count(1)


def new_run_root(base_dir: str, label: str = "run") -> str:
    """一次运行的布局根：`<base_dir>/<label>-<UTC 时间戳>-<pid>-<序号>`

    为什么要有这个构造函数（而不是各驱动自己拼字符串）:

    * **归属**：`base_dir` 是调用方（驱动）本来就拥有的产物目录 —— 清理/保留是它的既有
      策略，这里不引入新的「谁负责清理」问题；本函数只负责「每一次运行一个新根」
      （`run_layout` 幂等重开是允许的，但**同一根跨运行复用**会让上一轮的候选文件被这一轮
      覆盖）；
    * **唯一出处**：名字的形状只有一处，否则「哪一个目录是这一轮的根」会散在几个脚本里，
      而 §21 的 `Patch.path` 与 §22.1 的权限位都要指着它；
    * **唯一性**：微秒时间戳 + 进程内单调序号 + pid，三者合起来保证「同一进程连续两次
      调用」与「同一秒内两个进程」都不会拿到同一个根。第一版只有「秒级时间戳 + pid」——
      同一秒内连续两次调用返回同一个路径，正是本函数自己注释里要防的「上一轮候选被
      覆盖」（守卫 `tests/unit/test_run_root_wiring.py::TestTheRunRootConstructor` 当场抓到）。

    `label` 会被收成 `[A-Za-z0-9_-]`（只影响可读性，不参与判据）；取不到 base_dir 时抛
    `LayoutError`（空基目录等于把根落到 CWD —— 与 `run_layout` 拒绝空 run_root 同一条理由）。
    """
    import datetime
    import re
    if not isinstance(base_dir, str) or not base_dir.strip():
        raise LayoutError("new_run_root 需要一个非空的 base_dir（空串不是『当前目录』）")
    safe = re.sub(r"[^A-Za-z0-9_-]", "-", str(label or "run")).strip("-") or "run"
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    seq = next(_RUN_SEQ)
    return os.path.join(os.path.abspath(os.path.expanduser(base_dir)),
                        f"{safe}-{stamp}-{os.getpid()}-{seq}")


def run_layout(run_root: str) -> RunLayout:
    """建（或核对）一次运行的布局，返回句柄。

    **每一次 `execute` 调用一次** —— 布局是「这一次运行」的单位，跨运行复用一个
    `run_root` 会让上一轮的证据留在下一轮的目录里（与 §21 的记账同一条纪律）。

    幂等但**不许将就**：目录已存在且权限位不对 ⇒ `LayoutError`。刻意不「顺手改回来」——
    一个 0700 的 `measurements/` 是**篡改/上次异常**的信号，静默修好等于把这件事藏起来
    （设计 §4.3 规则 8 的三态纪律：判不了/不对时不许读成通过）。

    `run_root` 不允许空：空串意味着「没有布局」，那时调用方应该根本不建
    （而不是建一个相对当前工作目录的布局，把 `run_root` 的语义偷偷变成 CWD）。
    """
    if not isinstance(run_root, str) or not run_root.strip():
        raise LayoutError("run_layout 需要一个非空的 run_root（空串不是『默认布局』）")
    root = os.path.abspath(os.path.expanduser(run_root))
    os.makedirs(root, exist_ok=True)
    for d in DIR_NAMES:
        want = PROTECTED_DIR_MODE if d in PROTECTED_DIR_NAMES else WRITABLE_DIR_MODE
        p = os.path.join(root, d)
        if os.path.islink(p):
            raise LayoutError(f"{p} 是符号链接，不是布局里的一格")
        if not os.path.exists(p):
            os.mkdir(p, 0o700)
            os.chmod(p, want)
            continue
        if not os.path.isdir(p):
            raise LayoutError(f"{p} 已存在但不是目录")
        got = stat.S_IMODE(os.lstat(p).st_mode)
        if got != want:
            raise LayoutError(
                f"{p} 的权限位是 {oct(got)}，应该是 {oct(want)}"
                "（不静默改回来: 这是上一个写窗没复原或被改过的信号）")
    return RunLayout(root=root)
