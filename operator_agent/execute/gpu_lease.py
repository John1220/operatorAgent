"""Leases for the physical GPU: one per task run, one per machine + GPU UUID.

两层，职责不同：

* **每 run 一份**（`gpu-lease.json`，在 run 目录里）：同一任务内 profile 与 L3 是
  两个进程，各自的账本锁无法把"查对方账本 + 预留"做成原子的，所以两边都先拿这把。
* **每台机器 × 每张物理卡一份**（`~/.cache/operator_agent/gpu-leases/`）：不同**任务
  目录**之间也要互斥。2026-11-20 之前只有第一层，于是两个任务各拿各的 lease、
  同时进同一张卡测同一件事 —— 两份读数都"有效"，但都不是独占窗口。

机器级的键取自**声明**而不是运行期观测：`OPERATOR_AGENT_REQUIRED_GPU_UUID` 说这张卡
是谁，`OPERATOR_AGENT_GPU_LEASE_HOST`（缺省 `REMOTE_SSH_HOST`）说是哪台机。声明了不同
UUID 的两个任务可以并行；**没有声明**的那一侧用 `host|*`，与任何已持有的租约冲突 ——
"不知道是哪张卡"不能当成"大概是别的卡"。
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


LEASE_FILE = 'gpu-lease.json'
LOCK_FILE = '.gpu-lease.lock'
SCHEMA = 'operator-gpu-lease-v1'

#: 机器级租约的目录。测试/多用户环境可覆盖（`/tmp/...` 或 per-run 临时目录）。
MACHINE_DIR_ENV = 'OPERATOR_AGENT_GPU_LEASE_DIR'
MACHINE_SCHEMA = 'operator-gpu-lease-machine-v1'
MACHINE_LOCK = '.machine-leases.lock'

#: 未声明卡号时用的通配。它**与任何同机租约冲突**（保守方向：宁可排队）。
UNPINNED = '*'


class GpuLeaseBusy(RuntimeError):
    def __init__(self, owner, tool, *, key=None):
        self.owner = owner
        self.tool = tool
        self.key = key
        suffix = f' on {key}' if key else ''
        super().__init__(f'GPU lease held by {tool} ({owner}){suffix}')


@contextmanager
def _locked(run):
    descriptor = os.open(Path(run)/LOCK_FILE, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _read(run):
    path = Path(run)/LEASE_FILE
    if not path.exists():
        return None
    if path.is_symlink():
        return None
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, UnicodeError):
        return None
    return value if type(value) is dict and value.get('schema') == SCHEMA else None


# ── 机器级（跨任务目录）─────────────────────────────────────────────────────

def machine_key(environ=None, *, host=None, gpu_uuid=None):
    """`host|<gpu_uuid>`；显式调用参数优先于环境声明。"""
    env = os.environ if environ is None else environ
    gpu = (gpu_uuid or env.get('OPERATOR_AGENT_REQUIRED_GPU_UUID') or '').strip()
    host = (host or env.get('OPERATOR_AGENT_GPU_LEASE_HOST') or env.get('REMOTE_SSH_HOST')
            or env.get('REMOTE_SSH_HOST_KL') or '').strip()
    if not gpu and not host:
        return None
    return f'{host or "remote"}|{gpu or UNPINNED}'


def _machine_dir():
    root = (os.environ.get(MACHINE_DIR_ENV)
            or str(Path.home()/'.cache'/'operator_agent'/'gpu-leases'))
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _machine_file(directory, key, suffix):
    return directory/(hashlib.sha256(key.encode('utf-8')).hexdigest()[:24] + suffix)


@contextmanager
def _machine_locked(directory):
    descriptor = os.open(directory/MACHINE_LOCK, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _machine_live(directory):
    live = {}
    for path in directory.glob('*.json'):
        if path.is_symlink():
            continue
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError, UnicodeError):
            continue
        if type(value) is not dict or value.get('schema') != MACHINE_SCHEMA:
            continue
        expires = value.get('deadline')
        if type(expires) in (int, float) and expires <= time.time():
            continue                        # 过期租约：不挡新人
        live[value.get('key')] = value
    return live


def _clashes(left, right):
    """两个键是否指向同一张（或无法区分的那张）卡。"""
    lhost, _, lgpu = left.partition('|')
    rhost, _, rgpu = right.partition('|')
    return lhost == rhost and (lgpu == rgpu or UNPINNED in (lgpu, rgpu))


def acquire_machine(key, *, owner, tool, deadline=None):
    """在机器级拿这张卡的独占；被占时 `GpuLeaseBusy`（带占用人）。"""
    deadline = float(deadline) if type(deadline) in (int, float) else None
    directory = _machine_dir()
    with _machine_locked(directory):
        for other_key, value in _machine_live(directory).items():
            if _clashes(key, other_key or ''):
                raise GpuLeaseBusy(value.get('owner'), value.get('tool'), key=other_key)
        value = dict(schema=MACHINE_SCHEMA, key=key, owner=str(owner), tool=str(tool),
                     pid=os.getpid(), host=os.uname().nodename,
                     started_at=time.time(), deadline=deadline, lease_id=uuid.uuid4().hex)
        path = _machine_file(directory, key, '.json')
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True),
                             encoding='utf-8')
        os.replace(temporary, path)
        return value


def release_machine(key, owner):
    directory = _machine_dir()
    with _machine_locked(directory):
        path = _machine_file(directory, key, '.json')
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError, UnicodeError):
            return False
        if type(value) is not dict or value.get('owner') != str(owner):
            return False
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return True


def machine_status(key=None):
    """谁拿着哪张卡。只读，给自检与排障用。"""
    directory = _machine_dir()
    with _machine_locked(directory):
        live = _machine_live(directory)
    if key is None:
        return live
    return {k: v for k, v in live.items() if _clashes(key, k or '')}


# ── 每 run 一层（保留原语义，外加机器级互斥）────────────────────────────────

def acquire(run, *, owner, tool, deadline=None, host=None, gpu_uuid=None):
    """Atomically take the run's GPU lease **and** the machine-level one."""
    owner = str(owner)
    deadline = float(deadline) if type(deadline) in (int, float) else None
    key = machine_key(host=host, gpu_uuid=gpu_uuid)
    with _locked(run):
        current = _read(run)
        if current is not None:
            expires = current.get('deadline')
            if type(expires) not in (int, float) or expires > time.time():
                raise GpuLeaseBusy(current.get('owner'), current.get('tool'))
        # 先拿机器级：不同 run 之间也互斥。失败时不落任何盘。
        # 没有键（不指向某台真机的卡）时只保留每 run 一层。
        if key is not None:
            acquire_machine(key, owner=owner, tool=tool, deadline=deadline)
        value = dict(schema=SCHEMA, owner=owner, tool=str(tool), machine_key=key,
                     pid=os.getpid(), started_at=time.time(),
                     deadline=deadline, lease_id=uuid.uuid4().hex)
        path = Path(run)/LEASE_FILE
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True),
                             encoding='utf-8')
        os.replace(temporary, path)
        return value


def release(run, owner):
    """Release only this owner's lease; other owners are left untouched."""
    with _locked(run):
        current = _read(run)
        if current is None or current.get('owner') != str(owner):
            return False
        try:
            (Path(run)/LEASE_FILE).unlink()
        except FileNotFoundError:
            pass
        key = current.get('machine_key') or machine_key()
        if key is not None:
            release_machine(key, owner)
        return True


def upgrade(run, owner, gpu_uuid):
    """Narrow an owned host wildcard lease to the observed physical GPU UUID."""
    owner = str(owner)
    if type(gpu_uuid) is not str or not gpu_uuid.strip():
        raise ValueError('GPU UUID is required for lease upgrade')
    with _locked(run):
        current = _read(run)
        if current is None or current.get('owner') != owner:
            return False
        old_key = current.get('machine_key')
        if not old_key or not old_key.endswith('|' + UNPINNED):
            return current
        host = old_key.partition('|')[0]
        new_key = f'{host}|{gpu_uuid.strip()}'
        directory = _machine_dir()
        with _machine_locked(directory):
            for key, value in _machine_live(directory).items():
                if value.get('owner') != owner and _clashes(new_key, key or ''):
                    raise GpuLeaseBusy(value.get('owner'), value.get('tool'), key=key)
            upgraded = dict(current, key=new_key, machine_key=new_key)
            machine_value = dict(schema=MACHINE_SCHEMA, key=new_key,
                                 owner=owner, tool=current.get('tool'),
                                 pid=current.get('pid'), host=os.uname().nodename,
                                 started_at=current.get('started_at'),
                                 deadline=current.get('deadline'),
                                 lease_id=current.get('lease_id'))
            target = _machine_file(directory, new_key, '.json')
            temporary = target.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(machine_value, ensure_ascii=False, sort_keys=True),
                                 encoding='utf-8')
            os.replace(temporary, target)
            path = Path(run)/LEASE_FILE
            run_temporary = path.with_suffix('.json.tmp')
            try:
                run_temporary.write_text(json.dumps(
                    upgraded, ensure_ascii=False, sort_keys=True), encoding='utf-8')
                os.replace(run_temporary, path)
            except BaseException:
                target.unlink(missing_ok=True)
                raise
            _machine_file(directory, old_key, '.json').unlink(missing_ok=True)
        return upgraded