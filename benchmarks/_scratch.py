"""共享机器上的临时落点, 和「等到真的停了」的拆除。

吸纳的一手出处 —— DeepSeek Harness 本地 vendored 树
(`~/repos/github/deepseek-harness`, HEAD `47f943859b`, 2026-08-13)
`docs/defensive-patterns.md`:

* `:27` **"Never hand untrusted output the ambient environment or predictable paths"** —
  原文: *"Temp/spill files use a private (0700) dir, random names, and exclusive
  owner-only opens (`'wx'`, `0o600`) — predictable world-readable paths invite symlink
  races and disclosure."*
* `:19` **"Dispose must reach quiescence, not just request it"** —
  原文: *"A teardown that issues kills/aborts but returns before the work stops leaves
  orphans."*

为什么这两条对本仓不是洁癖（**这条链的失效方式与别处不同**）：真机跑在那台 8 卡共享机上,
`/tmp` 是世界可写的, 而两个驱动把脚本写成**固定名**再 scp 过去**执行**
（`verify_interleaved_ab.py` 里那个叫 `_contender.py` 的、以及
`verify_util_decontamination.py` 里那个叫 `_util_probe.py` 的, 都直接落在共享的 /tmp 上）。固定名的后果分两层：DSH 说的是 symlink race 与 disclosure, 对我们
更直接的是**测量完整性** —— 那个被执行的文件路径是公开的, 任何人（或任何上一次跑剩下的
进程）都能在我们 scp 之后、执行之前换掉它 ⇒ 我们以为在跑的「干扰流」或「util 探针」
不是我们写的那份, 而这正是本仓花了几轮去消除的那个变量（doc 33/34 的邻居抢卡去污染）。

拆除那一半的失效也不是脏进程：远端那条干扰命令自带「跑到某个时间戳」的自我期限（默认
+300 秒）, 本地 `ssh` 客户端被 terminate **不保证**远端进程跟着停 ⇒ 下一次在那张卡上的
测量会在被抢卡的状态下被当成安静读数, 而我们自己就是那个邻居。

三个函数都不碰 GPU, 拆除路径靠注入的 `run_remote` 在本地标定（与「量具先标定再上真机」
是同一条纪律）。
"""

import os
import secrets
import shlex
import subprocess
import tempfile
from pathlib import Path

# 远端 scratch 文件放哪个目录。仍然是 /tmp（那台机上只有它是世界可读可写的公共区）,
# 但**名字带随机后缀**且只允许我们自己的 mode —— DSH 那条规则要的不是「不用 /tmp」,
# 是「路径不可预测 + 权限私有」。
_REMOTE_DIR = "/tmp"


def private_dir(prefix: str = "opag-") -> Path:
    """控制端的私有工作目录（0700, 名字不可预测）。"""
    return ensure_private(Path(tempfile.mkdtemp(prefix=prefix)))


def ensure_private(directory: Path, what: str = "私有工作目录") -> Path:
    """权限不是 0700 时**当场炸**, 而不是等到 scp 把可读的文件送上去。

    单独成函数是为了让它处在一条用例能走到的路径上: 写在 `private_dir` 里的那个分支
    只在 umask/文件系统异常时才执行, 删掉它没有一条用例会变红（防御分支上的变异不具
    鉴别力 —— doc 49 §八 那条陷阱）。
    """
    mode = stat_mode(directory)
    if mode & 0o077:
        raise RuntimeError(f"{what}权限不私有: {directory} mode={oct(mode)} (期望 0700)")
    return directory


def stat_mode(p: Path) -> int:
    return p.stat().st_mode & 0o777


def write_private(directory: Path, name: str, text: str) -> Path:
    """以 `'wx'`（O_CREAT|O_EXCL）+ 0600 写一份私有文件。

    用 exclusive open 而不是 `Path.write_text`: 后者会**跟随已存在的符号链接**写进去,
    而那正是 DSH 那条规则点名的失效（`invite symlink races`）。已存在就报错而不是覆盖。
    """
    path = directory / name
    ensure_private(directory, "落点目录")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(str(path), flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def remote_scratch_path(name: str) -> str:
    """远端 scratch 文件的**不可预测**路径（保留 `.py` 后缀, uv 要看扩展名）。

    后缀来自 `secrets`（不是 pid、不是时间戳: pid 在别的机器上会撞、时间戳可猜）。
    """
    stem, _, ext = name.partition(".")
    ext = ext or "tmp"
    return f"{_REMOTE_DIR}/{stem}-{secrets.token_hex(4)}.{ext}"


def unmatchable(pattern: str) -> str:
    """把路径最后一个字符包成字符类 (`x.log` → `x.lo[g]`), 让 pattern 匹配不到**自己**。

    WHY (CHECKLOG H5 的同一失效, 这里是第二处): `pkill -f <pattern>` / `pgrep -f <pattern>`
    用的是**正则**, 而执行这条命令的那个 shell 自己的 cmdline 里就写着这个 pattern 文本 ⇒
    它匹配到自己。实测后果 (2026-10-01 真机 §14-4): 干扰进程**早已死掉**, 读回却永远报
    「远端仍有 1 个进程」⇒ `quiescent` 这个判据根本立不起来 (而它正是 DSH `:19`
    「dispose must reach quiescence」要的那一步)。`pkill` 那一半同样会朝自己的 shell
    发 SIGTERM —— 只是它被推迟到整条命令跑完, 所以 `rm` 侥幸跑到了。

    包一个字符类之后, 那段 pattern **文本**不再被它自己的正则匹配到 (`lo[g]` 要求那里是个
    `g`, 而 wrapper 的 cmdline 里那里是 `[`), 而目标进程那条真正的落点照样匹配 ✓。
    """
    if not pattern:
        return pattern
    return pattern[:-1] + "[" + pattern[-1] + "]"


def stop_interferer(proc, run_remote, remote_path: str, grace_s: float = 10.0) -> dict:
    """终止本地 ssh 客户端**并确认远端那个工作也停了**, 返回可读回的证据。

    `run_remote(cmd) -> str` 由调用方注入（真机上它是一次 `ssh alias cmd`）, 这样这条
    拆除路径能在没有 GPU 的机器上被标定。

    步骤与它们各自接住的失效:
      1. `terminate()` → `wait(grace_s)`, 超时才 `kill()` 再 `wait()` —— 只 `terminate()`
         是「请求」而不是「等到」(DSH `:19`)。
      2. `pkill -f <唯一路径>` —— 本地 ssh 死了不等于远端进程死了; 而且远端那条命令写着
         「跑到某个时间戳」的自我期限 (默认 +300 秒), 不显式收它就会继续占着那张卡。
         路径带随机后缀 (见 `remote_scratch_path`) 才让 `pkill -f` 精确指得到我们那一个,
         而不是别人的同名文件 —— 这条与 ① 的两半是互相成就的。
      3. `pgrep -f <唯一路径>` **读回**剩下的进程数。没有这一步, 第 2 步失败时函数会
         和成功时返回一样的东西。
      4. `rm -f` 收掉远端那份可执行文件（留着它, 下次别人 scp 同名会撞, 而且是公开路径）。
    """
    quoted = shlex.quote(remote_path)
    evidence: dict = {"remote_path": remote_path}

    if proc is not None:
        proc.terminate()
        try:
            evidence["local_returncode"] = proc.wait(timeout=grace_s)
            evidence["local_wait"] = "terminate 后自行退出"
        except subprocess.TimeoutExpired:
            proc.kill()
            evidence["local_returncode"] = proc.wait()
            evidence["local_wait"] = "terminate 超时, 已 kill 并等到退出"

    # pattern 用「匹配不到自己」的形态 (见 `unmatchable`): 否则这两个动作都会命中执行它们
    # 的那个 shell —— pkill 朝自己发信号, pgrep 把 wrapper 数成一个残留进程。
    quoted_pattern = shlex.quote(unmatchable(remote_path))
    # sh 的 `pkill` 在没有匹配时返回 1 —— 那是**期望的结果**（没有残留）, 所以 `|| true`。
    evidence["pkill_stdout"] = (run_remote(f"pkill -f {quoted_pattern} || true") or "").strip()[:200]
    left = (run_remote(f"pgrep -f {quoted_pattern} || true") or "").strip()
    evidence["remote_still_running"] = len([x for x in left.splitlines() if x.strip()])
    run_remote(f"rm -f {quoted}")

    if evidence["remote_still_running"]:
        # 不静默: 占着卡的那个进程必须让运行的人看见, 否则下一次「安静」读数就是假的。
        evidence["warning"] = (
            f"远端仍有 {evidence['remote_still_running']} 个进程匹配 {remote_path} —— "
            f"这张卡在它们停之前不干净, 别把接下来的读数当安静值")
        print(f"  ⚠️ {evidence['warning']}")
    else:
        evidence["quiescent"] = True
    return evidence
