"""旧版本任务的只读恢复：**能查回当时的三个事实**，且各按自己的身份对齐。

用的是一份真实产物 —— v0.8.2 的 A4（Attention，已跑完终评）。它比任何构造的夹具
都更有说服力：声明是旧形状、预算是花过的、源码是交付过的。

产物在独立证据目录里，不在仓库里。找不到就 skip —— 不假装通过，也不复制一份进来
（复制会把「真实产物」变成「我为测试编的产物」）。
"""
import json
from pathlib import Path

import pytest

from operator_agent.app import host_workflow

A4 = Path('/Users/zh/op-agent-evidence/v082-4a-step4-a4-20261123/run')

pytestmark = pytest.mark.skipif(
    not (A4 / 'run.json').is_file(),
    reason=f'A4 真实产物不在本机: {A4}')

#: 这三个值是这份产物**已经发生过**的事实，不是从今天代码推出来的。
EXPECTED_CONTRACT = 'a4c776c558c1fea9014498ac8595765de78c7c1c1d013c77ec7de5cb5bd8bb3f'
EXPECTED_SOURCE = 'f58762bcb359904b2d7e16bf2f0a93eb2f841a86e9e04102ddff1ec8610bee01'


def test_the_task_was_prepared_by_an_older_tool_version():
    """这份产物是 v14 冻结的。声明里没有 operator —— 那是后来才成为必填的。"""
    record = json.loads((A4 / 'run.json').read_text(encoding='utf-8'))
    assert record['tool_version'] == 14
    declaration = json.loads((A4 / 'declaration.json').read_text(encoding='utf-8'))
    assert 'operator' not in declaration, (
        '如果这条失败，说明夹具已经换了：本用例的前提是「旧声明没有 operator」')


def test_the_contract_is_read_back_and_its_hash_recomputed():
    """契约用 `contract_hash` 对齐 —— 不是靠重新解释声明。"""
    view = host_workflow.legacy_task_view(A4)
    assert view['contract']['recorded_hash'] == EXPECTED_CONTRACT
    assert view['contract']['recomputed_hash'] == EXPECTED_CONTRACT
    assert view['contract']['matches_run_record'] is True


def test_the_budget_ledgers_align_by_their_own_task_id():
    """**这条最容易写错**：三个 task_id 不是同一个值，要求相等会把真实产物判成损坏。

    `run.json.task_id` 是短 ID，两本预算账用的是另生成的 ID。契约按 `contract_hash`
    对齐、预算按预算 `task_id` 对齐，各自成立即可。
    """
    run_id = json.loads((A4 / 'run.json').read_text(encoding='utf-8'))['task_id']
    view = host_workflow.legacy_task_view(A4)
    ledger_id = view['verification_budget']['ledger_task_id']
    assert run_id == '56362c1f'
    assert ledger_id == '1ed2cb08e5c74b6cbe3ced1550e56b0f'
    assert run_id != ledger_id, (
        '两个 ID 不同是这份产物的**事实**，不是缺陷；夹具若断言相等就测错了')


def test_the_spent_budget_is_preserved_not_reset():
    """6 次上限里已经用掉 3 次探索 + 1 次独立终评，全部完成。恢复不许把它们清零。"""
    view = host_workflow.legacy_task_view(A4)
    budget = view['verification_budget']
    assert (budget['exploration_completed'], budget['final_completed']) == (3, 1)
    assert budget['records'] == 4


def test_the_delivered_source_is_identifiable():
    """第 3 轮推荐、第 4 轮按推荐源码终评，`override=false`。"""
    view = host_workflow.legacy_task_view(A4)
    assert view['delivery']['source_hash'] == EXPECTED_SOURCE
    assert view['delivery']['recommended_round'] == 3
    assert view['delivery']['override'] is False


def test_resume_returns_the_read_only_view_instead_of_refusing():
    """端到端：旧任务交进来，得到的是事实，不是 `explicit_declaration_required`。"""
    view = host_workflow.resume(A4)
    assert view['ok'] is True and view['read_only'] is True
    assert view['tool_version_recorded'] == 14
    assert '不要修改原 declaration.json' in view['next']


def test_the_view_does_not_write_anything():
    """只读就是只读：目录里不该多出文件。"""
    before = sorted(p.name for p in A4.iterdir())
    host_workflow.legacy_task_view(A4)
    assert sorted(p.name for p in A4.iterdir()) == before