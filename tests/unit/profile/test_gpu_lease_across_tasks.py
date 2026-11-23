"""v0.8.2 第三项：L3 与 NCU 要按**物理 GPU UUID** 互斥，且跨任务目录生效。

2026-11-20 之前只有"每 run 一份"的 `gpu-lease.json`：同一个任务里 profile 与 L3 会
互斥，但**两个任务目录**各拿各的锁，可以同时进同一张卡测量 —— 两份读数都"有效"，
却都不是独占窗口。机器级租约（`~/.cache/operator_agent/gpu-leases/`）补的就是这一格。

这一组不碰 GPU、不碰网络：租约是控制端的事，键取自**声明**（`REQUIRED_GPU_UUID` 与
目标主机），所以能在 prepare/reserve 阶段就判出来。
"""
import inspect
import json

import pytest

from operator_agent.app import hitl_tools as ht
from operator_agent.core.contract import freeze_from_task
from operator_agent.core.run_contract import code_hash
from operator_agent.execute import gpu_lease as gl
from operator_agent.execute import profile_budget as pb
from operator_agent.execute.host_task_budget import HostTaskBudget, HostTaskBudgetError


# v0.9 起 `operator` 必须显式（判据层不再从 request 推关键词）；
# v0.8.2 的这份夹具没有它，属于从那边搬过来时必须改造的一处。
DECL = dict(operator='relu', request='relu fp32', shapes=[[4, 8]], precision='fp32')
SEED = 'import numpy as np\ndef kernel(X):\n    return np.maximum(X, 0)\n'
BUDGET = dict(max_verifications=4, wall_clock_s=600)
HOST = 'h20'
GPU_A = 'GPU-0c16a8be-9d4a-61fc-e703-0c659f2544d4'
GPU_B = 'GPU-634bbc76-9ab6-220a-48a8-27247d7acbe2'


@pytest.fixture(autouse=True)
def leases(tmp_path, monkeypatch):
    """把机器级租约关进临时目录：不写家目录，也不跨用例串味。"""
    monkeypatch.setenv(gl.MACHINE_DIR_ENV, str(tmp_path/'leases'))
    monkeypatch.setenv('OPERATOR_AGENT_GPU_LEASE_HOST', HOST)
    monkeypatch.delenv('OPERATOR_AGENT_REQUIRED_GPU_UUID', raising=False)
    return tmp_path/'leases'



#: L3 一侧的租约集成**尚未**搬到本分支：v0.8.2 的 eb0eda5 把它接进
#: `HostTaskBudget.reserve`，而那段与 seed baseline、profile in-flight 等预算特性
#: 缠在一起，半搬会留下更糟的状态。所以这里先 skip，而且判据是**看源码**：
#: 一旦 `reserve` 里出现租约，这些用例自动开始跑 —— 不需要有人记得回来删 skip。
_LEASE_WIRED = 'gpu_lease' in inspect.getsource(HostTaskBudget.reserve)
requires_lease_integration = pytest.mark.skipif(
    not _LEASE_WIRED,
    reason='L3 侧 GPU 租约尚未移植（v0.8.2 eb0eda5）；本分支只搬了 gpu_lease 模块本身')

def task(root, name):
    run = root/name
    # 这里只需要一个已 prepare 的任务来挂租约；`seed_source` 是 v0.8.2 另一项
    # 特性（对齐种子候选），与租约语义无关，所以不在这次搬运范围内。
    ht.prepare(DECL, run, host_budget=BUDGET)
    contract = freeze_from_task(ht.build_task(DECL))
    return run, HostTaskBudget(run, contract.hash)


@requires_lease_integration
def test_two_tasks_on_the_same_card_cannot_both_enter_measurement(leases, monkeypatch):
    monkeypatch.setenv('OPERATOR_AGENT_REQUIRED_GPU_UUID', GPU_A)
    first_run, first = task(leases, 'task-a')
    second_run, second = task(leases, 'task-b')

    reservation = first.reserve(code_hash(SEED))
    with pytest.raises(HostTaskBudgetError) as error:
        second.reserve(code_hash(SEED))
    assert error.value.code == 'host_gpu_busy'
    assert GPU_A in str(error.value) or HOST in str(error.value)

    # 第一个任务结算后，第二个能进 —— 互斥不是"永久占坑"。
    first.finish(reservation['id'], outcome='failed')
    second.reserve(code_hash(SEED))


@requires_lease_integration
def test_two_tasks_on_different_cards_run_in_parallel(leases, monkeypatch):
    first_run, first = task(leases, 'task-a')
    monkeypatch.setenv('OPERATOR_AGENT_REQUIRED_GPU_UUID', GPU_A)
    first.reserve(code_hash(SEED))

    second_run, second = task(leases, 'task-b')
    monkeypatch.setenv('OPERATOR_AGENT_REQUIRED_GPU_UUID', GPU_B)
    second.reserve(code_hash(SEED))     # 不同卡：不受影响

    held = gl.machine_status()
    assert sorted(value['key'] for value in held.values()) == sorted(
        [f'{HOST}|{GPU_A}', f'{HOST}|{GPU_B}'])


@requires_lease_integration
def test_profile_and_l3_are_exclusive_across_task_directories(leases, monkeypatch):
    """NCU 与 L3 不许同卡并行 —— 跨任务目录也一样。"""
    monkeypatch.setenv('OPERATOR_AGENT_REQUIRED_GPU_UUID', GPU_A)
    profile_run, _ = task(leases, 'task-profile')
    _, other = task(leases, 'task-verify')

    owner = 'a' * 32
    pb.ProfileBudget(profile_run).reserve(timeout_s=60, owner_invocation_id=owner)
    with pytest.raises(HostTaskBudgetError) as error:
        other.reserve(code_hash(SEED))
    assert error.value.code == 'host_gpu_busy'

    pb.ProfileBudget(profile_run).settle_owner(owner, outcome='cancelled')
    other.reserve(code_hash(SEED))


@requires_lease_integration
def test_an_undeclared_card_blocks_every_card_on_that_host(leases, monkeypatch):
    """"不知道是哪张卡"不能当成"大概是别的卡"：未声明的一侧与任何同机租约冲突。"""
    _, declared = task(leases, 'task-declared')
    monkeypatch.setenv('OPERATOR_AGENT_REQUIRED_GPU_UUID', GPU_A)
    reservation = declared.reserve(code_hash(SEED))
    declared.finish(reservation['id'], outcome='failed')

    monkeypatch.delenv('OPERATOR_AGENT_REQUIRED_GPU_UUID')
    _, unpinned = task(leases, 'task-unpinned')
    unpinned.reserve(code_hash(SEED))      # 声明侧已结算，未声明侧可以进

    # 反过来：未声明的一侧先占住，声明 GPU_B 的任务要等它
    _, declared_b = task(leases, 'task-declared-b')
    monkeypatch.setenv('OPERATOR_AGENT_REQUIRED_GPU_UUID', GPU_B)
    with pytest.raises(HostTaskBudgetError) as error:
        declared_b.reserve(code_hash(SEED))
    assert error.value.code == 'host_gpu_busy'


def test_no_declaration_means_no_machine_lease(leases, monkeypatch):
    """不指向任何真机卡（mock/fake/local、单元测试）：只保留每 run 一层。"""
    monkeypatch.delenv('OPERATOR_AGENT_GPU_LEASE_HOST', raising=False)
    monkeypatch.delenv('REMOTE_SSH_HOST', raising=False)
    monkeypatch.delenv('REMOTE_SSH_HOST_KL', raising=False)
    assert gl.machine_key() is None
    _, first = task(leases, 'task-a')
    _, second = task(leases, 'task-b')
    first.reserve(code_hash(SEED))
    second.reserve(code_hash(SEED))        # 各自独立，不排队


def test_an_expired_machine_lease_does_not_block(leases, monkeypatch):
    monkeypatch.setenv('OPERATOR_AGENT_REQUIRED_GPU_UUID', GPU_A)
    key = gl.machine_key()
    gl.acquire_machine(key, owner='stale', tool='verify', deadline=1.0)   # 早已过期
    _, budget = task(leases, 'task-a')
    budget.reserve(code_hash(SEED))